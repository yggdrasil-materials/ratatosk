"""Run analysis in parallel."""

import sys
from datetime import UTC, datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from pkgutil import get_data
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse

import numpy as np
from art import tprint
from loguru import logger
from ruamel.yaml import YAML
from tqdm import tqdm

from ratatosk import (
    RATATOSK_BASE_VERSION,
    RATATOSK_COMMIT,
    classes,
    config,
    io,
    quality,
    signal,
)


def _log_setup(_config: dict[str, Any]) -> None:
    """
    Log the current configuration.

    Parameters
    ----------
    _config : dict
        Dictionary of configuration options.
    """
    logger.info(f"Base directory                      : {_config['base_dir']}")
    logger.info(f"Output directory                    : {_config['output_dir']}")
    logger.info(f"Cores                               : {_config['log_level']}")
    logger.info(f"Log-level                           : {_config['cores']}")
    logger.info(f"File Extension                      : {_config['file_ext']}")


def _set_logging(log_level: str = "info") -> None:
    """
    Set up loguru logging.

    Parameters
    ----------
    log_level : str
        Logging level.
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())


def _parse_configuration(  # pylint: disable=used-before-assignment
    args: argparse.Namespace | None = None,
) -> classes.Parameters:
    """
    Load configurations, validate and check run steps are consistent.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Arguments.

    Returns
    -------
    dict[str, Any]
        Returns the dictionary of configuration options, updated with missing values from default and command line
        options taking precedence.
    """
    # Parse command line options, load config (or default) and update with command line options
    if args.config_file is None:  # type: ignore[union-attr]
        default_config = get_data(package="ratatosk", resource="default_config.yaml")
        yaml = YAML(typ="safe")
        default_config = yaml.load(default_config)
    else:
        logger.info(f"Loading configuration from : {args.config_file!s}")  # type: ignore[union-attr]
        with args.config_file.open(encoding="utf-8") as conf:  # type: ignore[union-attr]
            yaml = YAML(typ="safe")
            default_config = yaml.load(conf)
    _config = config.reconcile_config_args(args=args, default_config=default_config)
    # Set logging level
    _set_logging(log_level=_config["log_level"].upper())
    # Create base output directory
    _config["output_dir"].mkdir(parents=True, exist_ok=True)
    _log_setup(_config=_config)
    return classes.Parameters(**_config)


def analysis(
    filename: str | Path,
    config_parameters: classes.Parameters,
    file_structures: classes.FileStructures,
) -> tuple[str, int]:
    """Analyse a single file.

    Parameters
    ----------
    filename : str | Path
        Filename of file to be processed.
    config_parameters : classes.Parameters
        `Parameters` configuration object.
    file_structures : classes.FileStructures
        `FileStructures` configuration object.

    Returns
    -------
    tuple[str, int]
        Tuple of `filename` and the value `1` (for now, will be returning actual data for aggregation in due course).
    """
    if Path(filename).exists():
        # ns-rse 2026-10-15 - sometimes get full path, need to handle this variation
        # logger.info(f"Found file : {filename.relative_to(Path().cwd())}")
        sheets = io.read_xlsx(filename)
        filename_stem = Path(filename).stem
        logger.info(f"[{filename_stem}] : Loaded all worksheets.")
    else:
        error_msg = f"Unable to load {filename}"
        raise FileExistsError(error_msg)
    # If `record` is a sheet we are likely working with Neware files
    if file_structures.neware["sheets"]["test_data"] in sheets:
        metadata = io.extract_neware_metadata(
            sheets=sheets,
            filename=filename_stem,
            sheet=file_structures.neware["sheets"]["metadata"],
            metadata=None,
        )
        logger.info(f"[{filename_stem}] : Metadata loaded.")
        dataset = io.read_neware(
            sheets=sheets,
            sheet=file_structures.neware["sheets"]["test_data"],
            path=filename,
            name=filename_stem,
            electrode_type="Positive",
            meta=metadata,
            columns=file_structures.neware["column_mapping"],
        )
        logger.info(f"[{filename_stem}] : Test data loaded.")
    else:
        logger.info(
            f"[{filename_stem}] : Missing {file_structures.neware['sheets']['test_data']}"
            " worksheet probably not a 'Neware' output file."
        )
        return (Path(filename).stem, 0)
    profile = signal.classify_dqdv_profile(df=dataset.df, filename=filename_stem)
    logger.info(f"[{filename_stem}] : dq/dv profile classified.")
    # ns-rse 2026-10-20 : Why does a function that pre-processes
    auto_params = signal.auto_preprocess_params(profile)
    logger.info(
        f"[{filename_stem}] : pre-processing parameters automatically determined."
    )
    # ns-rse 2026-10-20 : Notebook overrides auto generated parameters with user input, not doing that in batch
    # processing (yet!). What parameters should be over-written from configuration file though?
    # ns-rse 2026-10-20 : This is messy having extracted and split dataframe into dictionary indexed by (cycle, step) we
    # now have to pass the whole dataset in, which contains the original dataframe _and_ the dictionary of subsets just
    # to extract the desired cycle/step. There are better ways of doing this!
    signal_reports = {}
    for cycle_step in dataset.half_cycle_keys():
        signal_reports[cycle_step] = signal.half_cycle_report(
            dataset=dataset,
            cycle=cycle_step[0],
            step=cycle_step[1],
            params=auto_params,
            window=signal.voltage_window(dataset),
        )
    logger.info(f"[{filename_stem}] : Initial half-cycle report complete.")
    # Extract data from the dictionary of signal reports
    # ns-rse 2026-10-20 : I wonder if its better to add these as attributes to the dataset in the previous step?
    voltage_reconstruction, bin_width, empty_bins, voltage_size = [], [], 0, 0
    for cycle_step, _signal in signal_reports.items():
        # There is no voltage for this charge/cycle move on to the next
        if not _signal.voltage.size:
            continue
        if np.isfinite(_signal.bin_width_mV):
            bin_width.append(_signal.bin_width_mV)
        empty_bins += _signal.n_empty_bins
        voltage_size += _signal.voltage.size
        capacity = (
            "Charge_Capacity"
            if cycle_step[1].lower().startswith("c")
            else "Discharce_Capacity"
        )
        half_cycle_df = dataset.half_cycle(cycle=cycle_step[0], step=cycle_step[1])
        if capacity not in half_cycle_df.columns:
            continue
        voltage_reconstruction.append(
            quality.voltage_reconstruction(
                voltage=_signal.voltage,
                dqdv=_signal.dqdv,
                q_measured=half_cycle_df[capacity].to_numpy(),
                v_measured=half_cycle_df["Voltage"].to_numpy(),
            )
        )
    logger.info(
        f"[{filename_stem}] : Extracted and recontsructed voltages from half-cycles."
    )
    if config_parameters.terminal_trim:
        signal.trim_terminals(
            signal_reports,
            terminal_trim_fixed_mv=config_parameters.terminal_trim_fixed_mv,
            voltage_threshold=config_parameters.terminal_trim_band_mv,
        )
        logger.info(f"[{filename}] : Voltages trimmed.")
    # assert isinstance(config_parameters, classes.Parameters)
    # assert isinstance(dataset.df, pd.DataFrame)
    return (Path(filename).stem, 1)


def process(args: argparse.Namespace | None = None) -> None:  # pylint: disable=used-before-assignment
    """
    Process all files in a given directory.

    Parameters
    ----------
    args : argparse.Namespace | None
        Command line arguments for modifying configuration.
    """
    start_time: datetime = datetime.now(UTC)

    config_parameters = _parse_configuration(args=args)
    input_files = io.find_files(
        base_dir=config_parameters.base_dir, file_ext=config_parameters.file_ext
    )
    # Setup a partial function, we hard code the file_structures from the class, users should never have to fiddle with
    # this, if new file support is required it is a development issue.
    processing_function = partial(
        analysis,
        config_parameters=config_parameters,
        file_structures=classes.FileStructures(),
    )
    results = {}
    with (
        Pool(processes=config_parameters.cores) as pool,
        tqdm(
            total=len(input_files),
            desc=f"Processing files from {config_parameters.base_dir}, results are under {config_parameters.output_dir}",
        ) as pbar,
    ):
        # The for (...) is the list of objects returned by the processing_function which is a partial around
        # analysis() and so reflects whatever analysis() returns. If you want to aggregate more results across files
        # then make sure analysis() returns them and add something to hold those results within for (...) and
        # aggregate to a dictionary.
        for filename, counter in pool.imap_unordered(processing_function, input_files):
            results[str(filename)] = counter
            pbar.update()
            logger.info(f"Finished processing file : {filename}")
    # Write the current configuration to the output directory
    io.write_config(config_parameters)
    logger.debug(f"Files processed : {len(results)}")
    completion_message(
        config_parameters=config_parameters,
        input_files=len(input_files),
        files_processed=len(results),
        start_time=start_time,
    )


def completion_message(
    config_parameters: dict[str, Any] | classes.Parameters,
    input_files: int,
    files_processed: int,
    start_time: datetime,
) -> None:
    """
    Print a completion message summarising files processed.

    Parameters
    ----------
    config_parameters : dict[str, Any] | classes.Parameters
        Configuration parameters as either a dictionary or `Parameters` dataclass.
    input_files : int
        Number of input files identified on the `base_dir` path.
    files_processed : int
        Number of files successfully processed.
    start_time : datetime
        Start of processing.
    """
    base_dir = (
        config_parameters["base_dir"]
        if isinstance(config_parameters, dict)
        else config_parameters.base_dir
    )
    output_dir = (
        config_parameters["output_dir"]
        if isinstance(config_parameters, dict)
        else config_parameters.output_dir
    )
    file_ext = (
        config_parameters["file_ext"]
        if isinstance(config_parameters, dict)
        else config_parameters.file_ext
    )

    end_time = datetime.now(UTC)
    elapsed = end_time - start_time
    logger.info(
        "\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
    )
    tprint("Ratatosk", font="twisted")
    logger.info(
        f"\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
        f"  Ratatosk Version            : {RATATOSK_BASE_VERSION}\n"
        f"  Ratatosk Commit             : {RATATOSK_COMMIT}\n"
        f"  Base Directory              : {base_dir!s}\n"
        f"  File Extension              : {file_ext!s}\n"
        f"  Files Found                 : {input_files}\n"
        f"  Files Processed             : {files_processed} ({(files_processed * 100) / input_files}%)\n"
        f"  Time taken                  : "
        f"{datetime.fromtimestamp(elapsed.seconds, UTC).strftime('%H:%M:%S')} (HH-MM-SS)\n"
        # f"  All statistics              : {output_dir!s}/{csv_filename}\n"
        f"  Configuration               : {output_dir!s}/"
        f"config_{io.get_date_time(strftime='%Y-%m-%d-%H%M%S')}.yaml\n\n"
        f"  Email                       : ratatosk@sheffield.ac.uk\n"
        f"  Documentation               : https://yggdrasil-materials.github.io/ratatosk/\n"
        f"  Source Code                 : https://github.com/yggdrasil-materials/ratatosk/\n"
        f"  Bug Reports/Feature Request : https://github.com/yggdrasil-materials/ratatosk/issues/new/choose\n"
        f"  Citation File Format        : https://github.com/yggdrasil-materials/ratatosk/blob/main/CITATION.cff\n\n"
        f"  If you encounter bugs/issues or have feature requests please report them at the above URL\n"
        f"  or email us.\n\n"
        f"  If you have found Ratatosk useful please consider citing it. A Citation File Format is\n"
        f"  linked above and available from the Source Code page.\n"
        f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
    )
