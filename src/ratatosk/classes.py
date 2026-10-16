"""Dataclass definitions."""

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

# import numpy as np
# import numpy.typing as npt
# from loguru import logger
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

import ratatosk

DEFAULT_CONFIG = files(ratatosk) / "default_config.yaml"
DEFAULT_DATA_DICTIONARY = files(ratatosk) / "default_dictionary.yaml"


def load_default_yaml(
    yaml_file: str = "default_config.yaml", encoding: str = "utf-8"
) -> dict[str, Any]:
    """
    Load the specified YAML file.

    Parameters
    ----------
    yaml_file : str
        Filename to be loaded, should be part of the `ratatosk` package, typically under `src/ratatosk/`.
    encoding : str
        File encoding, default is `utf-8` and shouldn't need changing.

    Returns
    -------
    dict[str, Any]
        Dictionary representation of the YAML configuration.
    """
    resource = files(ratatosk).joinpath(yaml_file)
    data = yaml.safe_load(resource.read_text(encoding=encoding))
    return data or {}


class DefaultYamlSource(PydanticBaseSettingsSource):
    """
    Settings source backed by the YAML file bundled with the package.

    Parameters
    ----------
    settings_cls : type[BaseSettings]
        Inherited `settings_cls`.
    yaml_file : str
        YAML file to be loaded.
    """

    def __init__(
        self, settings_cls: type[BaseSettings], yaml_file: str = "default_config.yaml"
    ) -> None:
        """
        Initialise the class.

        Parameters
        ----------
        settings_cls : type[BaseSettings]
            Inherited `settings_cls`.
        yaml_file : str
            YAML file to be loaded.
        """
        super().__init__(settings_cls)
        self.yaml_file = yaml_file

    def get_field_value(self, field, field_name: str) -> tuple[None, str, bool]:  # noqa: ARG002
        """
        Get a field value, required due to inheritance from `PydanticBaseSettingsSource`.

        **NB** Not used directly, `__call__` supplies the complete mapping.

        Parameters
        ----------
        field : str
            Field.
        field_name : str
            Field name to extract.

        Returns
        -------
        tuple[None, str, bool]
            The `field_name` as part of a tuple.
        """
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """
        Call method for the class.

        Returns
        -------
        dict[str, Any]
            Returns the contents of a YAML file as a dictionary.
        """
        return load_default_yaml(self.yaml_file)


class Parameters(BaseSettings):
    """Parameters for analysis."""

    base_dir: Path = Field(default=Path("./"), title="Base directory")
    output_dir: Path = Field(
        default=Path("./output/"),
        title="Path to save the output to, default is './output/'.",
    )
    log_level: str = Field(default="info", title="Log level")
    cores: int = Field(default=2, title="Cores to run optimisation on in parallel")
    file_ext: str = Field(
        default="xlsx", title="File extension of battery monitoring results."
    )

    # Used to select the YAML source, not as an application setting.
    config_file: Path | None = Field(default=None, exclude=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize the sources used to populate settings.

        Parameters
        ----------
        settings_cls : type[BaseSettings]
            The settings class being configured.
        init_settings : PydanticBaseSettingsSource
            Source for values passed to the settings constructor.
        env_settings : PydanticBaseSettingsSource
            Source for environment-variable values.
        dotenv_settings : PydanticBaseSettingsSource
            Source for values loaded from a dotenv file.
        file_secret_settings : PydanticBaseSettingsSource
            Source for values loaded from secret files.

        Returns
        -------
        tuple[PydanticBaseSettingsSource, ...]
            Settings sources ordered from highest to lowest priority.
        """
        custom_file = init_settings.init_kwargs.get("config_file")
        yaml_source = (
            YamlConfigSettingsSource(settings_cls, yaml_file=custom_file)
            if custom_file is not None
            else DefaultYamlSource(settings_cls)
        )
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_source,
            file_secret_settings,
        )


class FileStructures(BaseSettings):
    """Data Dictionary for file structures."""

    neware: dict[str, Any] = Field(default=None, title="Neware data dictionary.")
    # Used to select the YAML source, not as an application setting.
    config_file: Path | None = Field(default=DEFAULT_DATA_DICTIONARY, title=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize the sources used to populate settings.

        Parameters
        ----------
        settings_cls : type[BaseSettings]
            The settings class being configured.
        init_settings : PydanticBaseSettingsSource
            Source for values passed to the settings constructor.
        env_settings : PydanticBaseSettingsSource
            Source for environment-variable values.
        dotenv_settings : PydanticBaseSettingsSource
            Source for values loaded from a dotenv file.
        file_secret_settings : PydanticBaseSettingsSource
            Source for values loaded from secret files.

        Returns
        -------
        tuple[PydanticBaseSettingsSource, ...]
            Settings sources ordered from highest to lowest priority.
        """
        custom_file = init_settings.init_kwargs.get("config_file")
        yaml_source = (
            YamlConfigSettingsSource(settings_cls, yaml_file=custom_file)
            if custom_file is not None
            else DefaultYamlSource(settings_cls, yaml_file=DEFAULT_DATA_DICTIONARY)
        )
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_source,
            file_secret_settings,
        )
