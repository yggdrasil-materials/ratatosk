"""Dataclass definitions."""

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

# import numpy as np
# import numpy.typing as npt
# from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
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


class TerminalTrim(BaseModel):
    """A class for type-checking the `terminal_trim` options."""

    model_config = ConfigDict(extra="forbid")
    run: bool = Field(default=True, title="Whether to trim half-cycles.")
    fixed_mv: float = Field(default=75.0, title="Fixed mV to trim by.")
    factor: float = Field(default=1.4, title="???")
    max_fraction: float = Field(default=0.1, title="???")
    band_mv: float = Field(default=20.0, title="???")
    interior: tuple[float, float] = Field(default=(0.2, 0.8), title="???")


class Termination(BaseModel):
    """Class for type-checking the `termination` options."""

    model_config = ConfigDict(extra="forbid")
    tol_mv: float = (Field(default=20.0, title="Tolerance for termination in mV."),)
    tol_spreads: float = (
        Field(default=5.0, title="Tolerance spread for termination."),
    )
    reference_n: int = (Field(default=10, title="Reference number for termination."),)


class DqDv(BaseModel):
    """Class for type-checking the `dqdv` options."""

    model_config = ConfigDict(extra="forbid")
    computation: str = Field(
        default="histogram",
        title="Differential method, valid values are 'histogram' or 'derivative'",
    )
    bin_mv: float | None = Field(
        default=None,
        title="Voltage bin width for the histogram path in mV. Determines the resolution of the curve.",
    )


class Figure(BaseModel):
    """Class for type-checking the `figure` options."""

    model_config = ConfigDict(extra="forbid")
    # ns-rse 2026-10-29 Duplication of theme, choose one and use it
    colour_palette: str = Field(
        default="viridis_r",
        title="Colour palette to use when plotting (default 'viridis_r') should match 'theme'.",
    )
    file_format: str = Field(
        default="png", title="Default format for figure output (default is 'png')."
    )

    height_in: float = Field(
        default=5.0,
        title="Figure height in inches applied to every figure produced.",
    )
    histogram_smooth_bins: str | float | None = Field(
        default=None, title="Histogram smoothing of bins, may be 'auto' or a float."
    )
    histogram_bin_mv: str | float | None = Field(
        default=None, title="Histogram bings in mV, may be 'auto' or a float value."
    )
    theme: str = Field(
        default="viridis_r",
        title="Theme to plot figures with.",
    )
    width_in: float = Field(
        default=6.73,
        title="Figure width in inches applied to every figure produced.",
    )


class DeadCell(BaseModel):
    """Class for type-checking the `DeadCell` options."""

    model_config = ConfigDict(extra="forbid")
    capacity_fraction: float = Field(default=0.02, title="Dead cell capacity fraction.")
    run: int = Field(default=3, title="Runs to signify a dead cell.")


class TheoreticalCapacity(BaseModel):
    """Class for type-checking the `TheoreticalCapacity` options."""

    model_config = ConfigDict(extra="forbid")
    library_formula: str | None = Field(
        default=None, title="Theoretical capacity library formula."
    )
    mah: float | None = Field(default=None, title="Theoretical capacity in mAh.")
    mah_g: float | None = Field(default=None, title="Theoretical capacity in mAh/g.")
    source: str | None = Field(default=None, title="Source of theoretical capacity.")


class TheoreticalCharge(BaseModel):
    """Class for type-checking the `TheoreticalCharge` options."""

    model_config = ConfigDict(extra="forbid")
    current_ma: float | None = Field(
        default=None, title="Theoretical charge current in mA."
    )
    rate_mah_g: float | None = Field(
        default=None, title="Theoretical charge rate in mAh/g."
    )


class Voltage(BaseModel):
    """Class for type-checking the `Voltage` options."""

    model_config = ConfigDict(extra="forbid")
    limit_lower_v_from_file: float | None = Field(
        default=None, title="Lower voltage limit from data."
    )
    limit_upper_v_from_file: float | None = Field(
        default=None, title="Upper voltage limit from data."
    )
    lower_v: float | None = Field(default=None, title="Lower voltage.")
    upper_v: float | None = Field(default=None, title="Upper voltage.")
    window_source: str = Field(default="measured from data", title="Source of voltages")


class Spike(BaseModel):
    """Class for type-checking the `Spike` options."""

    model_config = ConfigDict(extra="forbid")
    removal: str = Field(default="auto", title="Method of spike removal.")
    threshold_multiplier: str | float = Field(
        default="auto", title="Threshold multiplier for spike removal."
    )
    window_size: int = Field(default=5, title="Window size.")


class Peak(BaseModel):
    """Class for type-checking the `Peak` options."""

    model_config = ConfigDict(extra="forbid")
    min_distance_mv: str | float = Field(
        default="auto", title="Minimum distance in mV."
    )
    min_width_mvv: int = Field(default=5, title="Minimum width in mV.")
    prominence_fraction: float = Field(default=0.05, title="Prominence fraction.")


class Electrolyte(BaseModel):
    """Class for type-checking the `Electrolyte` options."""

    model_config = ConfigDict(extra="forbid")
    composition: str | None = Field(default=None, title="Electrolyte composition.")
    volume_ul: float = Field(default=100.0, title="Volume in uL.")


class Electrode(BaseModel):
    """Class for type-checking the `Electrode` options."""

    model_config = ConfigDict(extra="forbid")
    area_cm2: float | None = Field(default=None, title="Electrode area in cm2.")
    diameter_mm: float | None = Field(default=None, title="Electrode diameter in mm.")
    type: str | None = Field(
        default="Positive", title="Electrode type (either 'Positive' or 'Negative')."
    )


class CounterElectrode(BaseModel):
    """Class for type-checking the `CounterElectrode` options."""

    model_config = ConfigDict(extra="forbid")
    mass_mg: float | None = Field(default=None, title="Mass in mg.")
    measured: bool | None = Field(
        default=None, title="Whether the Counter Electrode features were measured."
    )
    metal: str | None = Field(default=None, title="Metal type of counter electrode.")
    metal_source: str | None = Field(
        default="assumed",
        title="Source of metal type of counter electrode (default is 'assumed').",
    )


class Active(BaseModel):
    """Class for type-checking the `Active` options."""

    model_config = ConfigDict(extra="forbid")
    fraction: float = Field(default=0.8, title="Active fraction")
    loading_mg_cm2: float | None = Field(
        default=None, title="Active loading in mg/cm2."
    )
    material_mass_mg: float | None = Field(
        default=None, title="Active material mass in mg."
    )


class Parameters(BaseSettings):
    """Parameters for analysis."""

    # Attributes are listed alphabetically to facilitate discovery, please maintain this when adding new attributes.
    active: Active = Field(
        default_factory=Active, title="Active configuration options."
    )

    anode_labels_swapped: bool = Field(default=False, title="Anode labels swapped.")

    base_dir: Path = Field(default=Path("./"), title="Base directory.")
    barcode: str | None = Field(
        default=None, title="Barcode (typically mirrors the filename)."
    )

    battery_chemistry: str | None = Field(default=None, title="Battery chemistry.")
    blend: list[float] = Field(default=[80.0, 10.0, 10.0], title="Blend.")
    cell_id: str | None = Field(default=None, title="Cell ID.")
    charge_current_a_from_file: float | None = Field(
        default=None, title="Charge current A from file."
    )
    charge_rate_c: float | None = Field(default=None, title="Charge rate current.")

    composition: str | None = Field(default=None, title="Composition.")
    counter_electrode: CounterElectrode = Field(
        default_factory=CounterElectrode,
        title="Counter electrode configuration options.",
    )
    cores: int = Field(default=2, title="Cores to run optimisation on in parallel.")
    cycling_verbose: bool = Field(
        default=True, title="How much of the cycling is reported."
    )
    dead_cell: DeadCell = Field(
        default_factory=DeadCell, title="Dead cell configuration options."
    )
    discharge_current_a_from_file: float | None = Field(
        default=None, title="Discharge current A from file."
    )
    dqdv: DqDv = Field(default_factory=DqDv, title="dq/dv configuration options.")
    file_ext: str = Field(
        default="xlsx", title="File extension of battery monitoring results."
    )
    edge_exclusion_mv: float | None = Field(default=50, title="Edge exclusion in mV.")

    electrode: Electrode = Field(
        default_factory=Electrode, title="Electrode attributes."
    )
    electrolyte: Electrolyte = Field(
        default_factory=Electrolyte, title="Electrolyte composition."
    )
    fit_unresolved: bool = Field(
        default=True,
        title="Fit a dataset whose reference cycle says its peaks are not resolvable.",
    )
    figure: Figure = Field(default_factory=Figure, title="Figure plotting options.")
    key_cycles: list[int] = Field(
        default=[1, 5, 10, 20, 30, 40, 50], title="Key cycles to summarise and plot."
    )
    log_level: str = Field(default="info", title="Log level")
    output_dir: Path = Field(
        default=Path("./output/"),
        title="Path to save the output to, default is './output/'.",
    )
    max_cycle: int | None = Field(
        default=None,
        title="Cap the analysis at the given cycle, default is 'None' which includes all cycles.",
    )
    peak: Peak = Field(default_factory=Peak, title="Peak fitting options.")
    polyorder: int = Field(default=3, title="Polyorder.")
    power_energy_analysis: bool = Field(
        default=True, title="Whether to run power energy analysis."
    )
    qc_cycle: int = Field(
        default=3,
        title="Which cycle the preprocessing quality control figure shows raw against.",
    )
    rebin_width_mv: str | int = Field(
        default="auto", title="Rebin width, either 'auto' or an integer."
    )
    reference_cycle: int = Field(default=2, title="Reference cycle.")
    run_peak_fitting: bool = Field(
        default=True, title="Whether to fit the dQ/dV peaks."
    )
    second_smooth_window: str | int = Field(
        default="auto",
        title="Second order smoothing window, either 'auto' or an integer.",
    )
    separator_type: str | None = Field(default=None, title="Separator type.")
    shoulder_detection: bool = Field(
        default=True, title="Whether to enable shoulder detection."
    )
    smoothing_window: str | int = Field(
        default="auto", title="Smoothing window, either 'auto' or an integer."
    )
    spike: Spike = Field(default_factory=Spike, title="Spike configuration options.")
    stop_fitting_at_death: bool = Field(
        default=True, title="Stop fitting once the cell has stopped delivering charge."
    )
    terminal_trim: TerminalTrim = Field(
        default_factory=TerminalTrim, title="Terminal trimming configuration options."
    )
    termination: Termination = Field(
        default_factory=Termination, title="Termination configuration options."
    )
    test_start_date: str | None = Field(default=None, title="Test start date")

    theoretical_capacity: TheoreticalCapacity = Field(
        default_factory=TheoreticalCapacity,
        title="Theoretical capacity configuration options.",
    )
    theoretical_charge: TheoreticalCharge = Field(
        default_factory=TheoreticalCharge,
        title="Theoretical charge configuration options.",
    )
    total_electrode_coating_mg: float | None = Field(
        default=None, title="Total electrode coating in mg"
    )
    total_loading_mg_cm2: float | None = Field(
        default=None, title="Total loading in mg/cm2"
    )

    tracking_tolerance_mv: float = Field(
        default=80.0, title="Tracking tolerance in mV."
    )
    voltage: Voltage = Field(
        default_factory=Voltage, title="Voltage configuration options."
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
