"""Functions and tools for working with configuration files."""

from argparse import Namespace
from collections.abc import MutableMapping
from datetime import UTC, datetime
from pathlib import Path
from pkgutil import get_data
from typing import Any

from loguru import logger

from ratatosk import CONFIG_DOCUMENTATION_REFERENCE
from ratatosk.io import convert_path, read_yaml


def reconcile_config_args(
    args: Namespace | None, default_config: dict[str, Any]
) -> dict:
    """
    Reconcile command line arguments with the default configuration.

    Command line arguments take precedence over the default configuration. If a partial configuration file is specified
    (with '-c' or '--config-file') the defaults are over-ridden by these values (internally the configuration
    dictionary is updated with these values). Any other command line arguments take precedence over both the default
    and those supplied in a configuration file (again the dictionary is updated).

    The final configuration is validated before processing begins.

    Parameters
    ----------
    args : Namespace
        Command line arguments passed into Ratatosk.
    default_config : dict[str, Any]
        Dictionary containing the default configuration for the package.

    Returns
    -------
    dict
        The configuration dictionary.
    """
    if args is not None:
        if args.config_file is not None:
            config = read_yaml(str(args.config_file))
            # Merge the loaded config with the default config to fill in any defaults that are missing
            # Make sure to prioritise the loaded config, so it overrides the default
            config = merge_mappings(map1=default_config, map2=config)
        else:
            # If no config file is provided, use the default config
            config = default_config
    else:
        # If no args are provided, use the default config
        config = default_config

    # Override the config with command line arguments passed in, eg --output_dir ./output/
    if args is not None:
        config = update_config(config, args)

    return config


def merge_mappings[MutableMappingType](  # pylint: disable=invalid-name
    map1: MutableMappingType, map2: MutableMappingType
) -> MutableMappingType:
    """
    Merge two mappings (dictionaries), with priority given to the second mapping.

    Note: Using a Mapping should make this robust to any mapping type, not just dictionaries. MutableMapping was needed
    as Mapping is not a mutable type, and this function needs to be able to change the dictionaries.

    Parameters
    ----------
    map1 : MutableMapping
        First mapping to merge, with secondary priority.
    map2 : MutableMapping
        Second mapping to merge, with primary priority.

    Returns
    -------
    dict
        Merged dictionary.
    """
    # Iterate over the second mapping
    for key, value in map2.items():
        # If the value is another mapping, then recurse
        if isinstance(value, MutableMapping):
            # If the key is not in the first mapping, add it as an empty dictionary before recursing
            map1[key] = merge_mappings(map1.get(key, {}), value)
        else:
            # Else simply add / override the key value pair
            map1[key] = value
    return map1


def write_config_with_comments(  # pylint: disable=too-many-branches
    args: Namespace | None = None,
    valid_config: tuple[str] = ("config", "data-dictionary"),
    valid_module: tuple[str] = ("ratatosk"),
) -> None:
    """
    Write a sample configuration with in-line comments.

    This function is not designed to be used interactively but can be, just call it without any arguments and it will
    write a configuration to './config.yaml'.

    Parameters
    ----------
    args : Namespace
        A Namespace object parsed from argparse with values for 'filename'.
    valid_config : tuple[str]
        Tuple of valid configuration options.
    valid_module : tuple[str]
        Tuple of valid modules, currently 'ratatosk'.
    """
    if args.config is not None and args.config not in valid_config:
        error_msg = f"There is no configuration for {args.config}, valid options are f{valid_config}"
        raise ValueError(error_msg)
    output_dir = Path("./") if args.output_dir is None else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger_msg = "A sample configuration has been written to"
    if args.config in ("config", None):
        try:
            config = get_data(package=args.module, resource="default_config.yaml")
            filename = "default_config.yaml" if args.filename is None else args.filename
        except AttributeError as exc:
            error_msg = f"There is no configuration for module {args.module} called 'default_config.yaml'"
            raise (FileNotFoundError) from exc
    elif args.config == "data-dictionary":
        try:
            config = get_data(package=args.module, resource="default_dictionary.yaml")
            filename = (
                "default_dictionary.yaml" if args.filename is None else args.filename
            )
        except AttributeError as exc:
            error_msg = f"There is no configuration for module {args.module} called 'default_config.yaml'"
            raise (FileNotFoundError(error_msg)) from exc
    # elif args.config == "mplstyle":
    #     try:
    #         config = get_data(package=args.module, resource="ratatosk.mplstyle")
    #         filename = "ratatosk.mplstyle" if args.filename is None else args.filename
    #     except FileNotFoundError as exc:
    #         raise (
    #             FileNotFoundError(f"There is no configuration for module {args.module} called 'ratatosk.mplstyle'")
    #         ) from exc
    else:
        error_msg = f"Invalid configuration file option ({args.config}), valid options are\n{valid_config}"
        raise ValueError(error_msg)
    # Remove `default_` if that is the filename that has been loaded, all others remain as is
    filename = (
        filename.replace("default_", "") if "default_" in str(filename) else filename
    )
    if (
        ".yaml" not in str(filename)
        and ".yml" not in str(filename)
        and ".mplstyle" not in str(filename)
    ):
        config_path = output_dir / f"{filename}.yaml"
    else:
        config_path = output_dir / filename
    with config_path.open("w", encoding="utf-8") as f:
        try:
            f.write(
                f"# Config file generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"{CONFIG_DOCUMENTATION_REFERENCE}")
            f.write(config.decode("utf-8"))
        except AttributeError as exc:
            error_msg = f"There is no configuration for module {args.module}, valid options are {valid_module}."
            raise (AttributeError(error_msg)) from exc
    logger.info(f"{logger_msg} : {config_path!s}")


def update_config(config: dict, args: dict | Namespace) -> dict:
    """
    Update the configuration with any arguments.

    Parameters
    ----------
    config : dict
        Dictionary of configuration (typically read from YAML file specified with '-c/--config <filename>').
    args : Namespace
        Command line arguments.

    Returns
    -------
    dict
        Dictionary updated with command arguments.
    """
    args = vars(args) if isinstance(args, Namespace) else args

    config_keys = config.keys()
    for arg_key, arg_value in args.items():
        if isinstance(arg_value, dict):
            update_config(config, arg_value)
        elif arg_key in config_keys and arg_value is not None:
            original_value = config[arg_key]
            config[arg_key] = arg_value
            logger.debug(
                f"Updated config config[{arg_key}] : {original_value} > {arg_value} "
            )
    if "base_dir" in config:
        config["base_dir"] = convert_path(config["base_dir"])
    if "output_dir" in config:
        config["output_dir"] = convert_path(config["output_dir"])
    return config
