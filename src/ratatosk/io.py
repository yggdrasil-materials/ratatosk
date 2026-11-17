"""Module for reading and writing data."""

from __future__ import annotations

import contextlib
import hashlib
import re
from argparse import Namespace
from dataclasses import replace
from datetime import datetime
from importlib.resources import files
from pathlib import Path, PosixPath
from pprint import pformat
from types import NoneType
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import RootModel
from ruamel.yaml import YAML

import ratatosk
from ratatosk import (
    CONFIG_DOCUMENTATION_REFERENCE,
    DATA_DICTIONARY_DOCUMENTATION_REFERENCE,
)
from ratatosk.classes import Dataset, FileStructures, Parameters

# Set the default configurations and dictionaries
DEFAULT_CONFIG = files(ratatosk) / "default_config.yaml"
DEFAULT_DATA_DICTIONARY = files(ratatosk) / "default_dictionary.yaml"

# Neware long names -> the short names used throughout. Anything not listed is
# carried through untouched: Current(A) in particular is needed by the
# constant-voltage detection and must survive unrenamed.
_NUMERIC = ("Cycle", "Voltage", "Charge_Capacity", "Discharge_Capacity", "dQ/dV")

_STEP_NAMES = {
    "CCCV Chg": "Charge",
    "CCCV CHG": "Charge",
    "CC Chg": "Charge",
    "CC CHG": "Charge",
    "CC DChg": "Discharge",
    "CC DCHG": "Discharge",
    "CCCV DChg": "Discharge",
}


def file_sha256(path: str | Path, block: int = 1 << 20) -> str:
    """Calculate SHA256 checksum for a given file.

    Parameters
    ----------
    path : str | Path
        Path to file.
    block : int
        Block size for reading file in chunks.

    Returns
    -------
    str
        SHA256 checksum for the file.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def read_yaml(filename: str | Path) -> Any:
    """
    Read a YAML file.

    Parameters
    ----------
    filename : Union[str, Path]
        YAML file to read.

    Returns
    -------
    Any
        Dictionary of the file.
    """
    with Path(filename).open(encoding="utf-8") as f:
        yaml = YAML(typ="safe")
        return yaml.load(f)


def convert_path(path: str | Path) -> Path:
    """
    Ensure path is Path object.

    Parameters
    ----------
    path : str | Path
        Path to be converted.

    Returns
    -------
    Path
        Pathlib object of path.
    """
    return Path().cwd() if path == "./" else Path(path).expanduser()


def dict_to_yaml(obj: Any) -> Any:
    """
    Clean a dictionary to basic data types for writing to YAML.

    Parameters
    ----------
    obj : Any
        An object for converting to basic data types.

    Returns
    -------
    Any
        `obj` as `str`, `int`, `float`, `bool`, `list` or `dict`.
    """
    # @ns-rse : we have introduced custom dataclasses to the configuration dictionary, such classes now need converting
    # to dictionaries so they can be correctly parsed
    if type(obj) not in (
        dict,
        list,
        tuple,
        set,
        str,
        int,
        float,
        bool,
        PosixPath,
        np.ndarray,
        np.generic,
        NoneType,
    ):
        logger.debug(
            f"Custom dataclass type detected ({obj=}) converting to dictionary."
        )
        obj = obj.__dict__
        logger.debug(f"Dictionary representation...\n{pformat(obj)}")

    # Recurse on dictionaries
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            key = k if isinstance(k, str) else str(k)
            new[key] = dict_to_yaml(v)
        return new
    # Recurse on lists and tuples
    if isinstance(obj, (list, tuple)):
        return [dict_to_yaml(x) for x in obj]
    # Safe types return as is
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Convert Path -> string
    if isinstance(obj, Path):
        return str(obj)
    # Convert numpy array -> nested lists
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Convert numpy scalar -> native Python scalar
    if isinstance(obj, np.generic):
        return obj.item()
    # If nothing else is matched use string representation
    return str(obj)


def write_config(args: Namespace | dict[str, Any] | Parameters | None) -> None:
    """
    Write a configuration file to YAML.

    Parameters
    ----------
    args : Namespace | dict[str, Any] | Parameters, optional
        A Namespace object parsed from argparse. If there are values for ``output_dir`` and ``filename`` these will be
        used to construct the path and filename to write the YAML file to.  If not then output files are contingent on
        how the function is being called. If it is from ``ratatosk create_config`` then ``default_config.yaml`` will be
        written. If it is at the end of processing then ``config_YY-MM-DD-hhmmss.yaml`` will be used.
    """
    # If args is `Namespace` then we are writing config with 'ratatosk create_config' subcommand
    if isinstance(args, Namespace):
        output_dir = Path("./") if args.output_dir is None else Path(args.output_dir)
        if args.type == "config":
            config = vars(Parameters(config_file=DEFAULT_CONFIG))
            filename = "default_config.yaml" if args.filename is None else args.filename
            doc_ref = CONFIG_DOCUMENTATION_REFERENCE
        elif args.type == "data-dictionary":
            config = vars(FileStructures(config_file=DEFAULT_DATA_DICTIONARY))
            filename = (
                "default_dictionary.yaml" if args.filename is None else args.filename
            )
            doc_ref = DATA_DICTIONARY_DOCUMENTATION_REFERENCE
        else:
            logger.error(
                "Invalid option for --type use either 'config' or 'data-dictionary'."
            )
    # Otherwise we are writing after 'ratatosk analysis' and config is a dictionary, this won't have a 'filename'
    # key/value pair
    elif isinstance(args, dict):
        output_dir = (
            Path("./") if args["output_dir"] is None else Path(args["output_dir"])
        )
        filename = f"config_{get_date_time(strftime='%Y-%m-%d-%H%M%S')}.yaml"
        config = args
        doc_ref = CONFIG_DOCUMENTATION_REFERENCE
    # If we have 'Parameters' then configuration is stored as a dataclass and we again don't have 'filename' key/value pair
    elif isinstance(args, Parameters):
        output_dir = Path("./") if args.output_dir is None else Path(args.output_dir)  # type: ignore[redundant-expr]
        filename = f"config_{get_date_time(strftime='%Y-%m-%d-%H%M%S')}.yaml"
        config = RootModel[Parameters](args).model_dump()
        doc_ref = CONFIG_DOCUMENTATION_REFERENCE
    # If we have 'FileStructures' then configuration is stored as a dataclass and we again don't have 'filename' key/value pair
    elif isinstance(args, FileStructures):
        output_dir = Path("./") if args.output_dir is None else Path(args.output_dir)  # type: ignore[redundant-expr]
        filename = f"config_{get_date_time(strftime='%Y-%m-%d-%H%M%S')}.yaml"
        config = RootModel[FileStructures](args).model_dump()
        doc_ref = DATA_DICTIONARY_DOCUMENTATION_REFERENCE
    else:
        msg = f"args is neither 'Namespace', 'dict' or 'Parameters' : {type(args)}"
        raise TypeError(msg)
    if ".yaml" not in str(filename) and ".yml" not in str(filename):  # pylint: disable=possibly-used-before-assignment
        config_path = output_dir / f"{filename}.yaml"
    else:
        config_path = output_dir / filename
    logger_msg = "Configuration written to"
    with config_path.open("w", encoding="utf-8") as f:
        try:
            f.write(f"# Config generated {get_date_time()}\n")
            f.write(f"{doc_ref}")  # pylint: disable=possibly-used-before-assignment
            yaml_out = YAML()
            yaml_out.indent(sequence=4, offset=2)
            yaml_out.dump(dict_to_yaml(config), f)  # pylint: disable=possibly-used-before-assignment
            logger.info(f"{logger_msg} : {config_path!s}")
        except:  # noqa: E722, pylint: disable=W0702
            logger.error(f"Failed to write config to : {config_path}")


def get_date_time(strftime: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get the current date-time as a string for the systems current timezone.

    Parameters
    ----------
    strftime : str
        String for formatting date-time, default is ``%Y-%m-%d %H:%M:%S``.

    Returns
    -------
    str
        Date-time as a string for systems current timezone.
    """
    return datetime.now(tz=datetime.now().astimezone().tzinfo).strftime(strftime)


# ---------------------------------------------------------------------------


def _tidy_df(
    df: pd.DataFrame, step_column: str = "Step Type", filter_on: str = "Rest"
) -> pd.DataFrame:
    """
    Tidys a ? dataframe.

    What is it doing?

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe.
    step_column : str
        Column name that holds the step data.
    filter_on : str
        Filter data and remove instances where the `step_column == filter_on`.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe with `step_column` renamed and rows where `step_column == filter_on` removed.
    """
    if step_column not in df.columns:
        return df
    out = df.copy()
    out[step_column] = out[step_column].astype(str).str.strip()
    out[step_column] = out[step_column].replace(_STEP_NAMES)
    return out[
        ~out[step_column].str.contains(filter_on, case=False, na=False)
    ].reset_index(drop=True)
    # return out.reset_index(drop=True)


def _apply_electrode_convention(df, electrode_type):
    """
    For a negative electrode, present the useful half-cycle as "Discharge".

    Swaps the Step labels, the capacity columns and the sign of dQ/dV, exactly
    as 1.8.x Cell 4 does, so that figures and tracked-peak tables stay
    comparable across the versions. `Dataset.step_direction` is unaffected —
    it reads the voltage.
    """
    if str(electrode_type).lower() != "negative":
        return df, False
    d = df.copy()
    if {"Charge_Capacity", "Discharge_Capacity"} <= set(d.columns):
        d = d.rename(
            columns={
                "Charge_Capacity": "_swap",
                "Discharge_Capacity": "Charge_Capacity",
            }
        )
        d = d.rename(columns={"_swap": "Discharge_Capacity"})
    for a, b in (
        ("Chg. Cap.(Ah)", "DChg. Cap.(Ah)"),
        ("Chg. Energy(Wh)", "DChg. Energy(Wh)"),
        ("Chg. Spec. Energy(mWh/g)", "DChg. Spec. Energy(mWh/g)"),
    ):
        if a in d.columns and b in d.columns:
            d = d.rename(columns={a: "_swap", b: a}).rename(columns={"_swap": b})
    if "Step" in d.columns:
        d["Step"] = (
            d["Step"]
            .map({"Charge": "Discharge", "Discharge": "Charge"})
            .fillna(d["Step"])
        )
    for c in ("dQ/dV", "dQ/dV(mAh/V)"):
        if c in d.columns:
            d[c] = -pd.to_numeric(d[c], errors="coerce")
    return d, True


def apply_electrode_convention(dataset: Dataset, electrode_type: str):
    """
    Return a copy of `dataset` with the convention for `electrode_type` applied.

    Re-applying `Negative` to an already-swapped dataset returns it unchanged; re-stamping it as `Positive` raises,
    because that would leave the `Step` labels,  capacity columns and `dQ/dV` sign inverted while the record claimed
    otherwise — silently mirrored data with nothing to notice it by.

    Parameters
    ----------
    dataset : Dataset
        Dataset to be processed.
    electrode_type : str
        Either `negative` or `positive` (case insensitive).

    Returns
    -------
    Dataset
        Dataset with electrode type switched.
    """
    if dataset.labels_swapped:
        if str(electrode_type).lower() == "negative":
            return replace(dataset, electrode_type=str(electrode_type))
        # Re-stamping a swapped dataset as Positive would leave its Step
        # labels, capacity columns and dQ/dV sign inverted while claiming
        # otherwise — silently wrong data with no way to notice.
        # ns-rse 2026-10-06 - Consider custom exception Class, see
        #    https://docs.astral.sh/ruff/rules/raise-vanilla-args/
        neg_electrode_error = (
            f"{dataset.name!r} was loaded as a negative electrode and its "
            "half-cycle labels are already swapped; it cannot be restamped "
            f"as {electrode_type!r}. Reload the file with the correct "
            "electrode type."
        )
        raise ValueError(neg_electrode_error)
    if str(electrode_type).lower() != "negative":
        return replace(dataset, electrode_type=str(electrode_type))
    df, swapped = _apply_electrode_convention(dataset.df, electrode_type)
    return replace(
        dataset,
        frame=df.reset_index(drop=True),
        electrode_type=str(electrode_type),
        labels_swapped=swapped,
    )


def truncate_cycles(dataset: Dataset, max_cycle) -> Dataset:
    """
    Filter a `Dataset.frame`, removing any values/data that exceed `max_cycle`.

    Truncation belongs HERE, to the Dataset, because the Dataset is what
    every later stage reads. 1.8.6 truncated a separate `electrochemical_data`
    dict inside the parameter collection, and 1.9.0's notebook then rebuilt
    that dict from `dataset.frame` two lines later — so the operator's answer
    to "Maximum cycle to analyse" was applied, discarded, and then asserted as
    fact by the console message, `dataset_info.txt` and the run manifest,
    while every capacity, retention and fade number was computed over the
    cycles they had asked to exclude.

    Parameters
    ----------
    dataset : Dataset
        Dataset object, must have `.frame` attribute populated.
    max_cycle : float
        Threshold for excluding values.

    Returns
    -------
    Dataset
        Dataset with updated `.frame` attribute filtered for values `<= max_cycle`.
    """
    # ns-rse 2026-10-07 simplify the conditions here.
    if not max_cycle:
        return dataset
    f = dataset.df
    if "Cycle" not in f.columns:
        return dataset
    # ns-rse 2026-10-07 use boolean indexing to filter the dataset, overly cautious using pd.numeric() as would be
    # expecting data to already be numeric
    keep = pd.to_numeric(f["Cycle"], errors="coerce") <= float(max_cycle)
    return replace(dataset, frame=f[keep].reset_index(drop=True))


def read_xlsx(path: str | Path) -> dict[str, pd.DataFrame]:
    """
    Open a `.xls[x]` spreadsheet and read all sheets.

    Raises rather than returning None. A workbook that cannot be opened at
    all is a different failure from one that has no 'test' sheet, and the
    two were indistinguishable while both were caught by the same
    `except Exception` inside the metadata reader.

    Parameters
    ----------
    path : str | Path
        Returns a class for parsing the given Excel file and the engine name.

    Returns
    -------

    dict[str, pd.DataFrame]
        Dictionary of data from a `xls[x]` file. Keys are sheet names and values are Pandas dataframes of the sheets
        contents.
    """
    try:
        with pd.ExcelFile(path, engine="calamine") as xls:
            return {sheet: pd.read_excel(xls, sheet) for sheet in xls.sheet_names}
    except OSError as e:
        error_msg = f"Could not open {path!s} as an Excel workbook."
        raise OSError(error_msg) from e


def read_neware(
    sheets: dict[str, pd.DataFrame],
    path: str | Path,
    sheet: str = "record",
    name: str | None = None,
    electrode_type: str = "Positive",
    meta: str | None = None,
    columns: dict | None = None,
) -> Dataset:
    """
    Read one Neware data from  `.xlsx` export.

    A Neware export has two sheets we want `record` and `test`. These are extracted from the dictionary and used to
    generate a `Dataset` object.

    Parameters
    ----------
    sheets : dict[str, pd.DataFrame], optional
        A dictionary of sheets from `io.open_xlsx`.
    path : str | Path,
        Path to `xlsx` file.
    sheet : str
        The sheet to extract, defaults to "record".
    name : str | None = None,
        Name of the file.
    electrode_type : str = "Positive",
        Electrode type, options are `Positive` (default) and `Negative`.
    meta : str | None = None,
        Something.
    columns : dict, optional
        Dictionary for renaming columns. If `None` a default is used.

    Returns
    -------
    Dataset
        A `Dataset` object of the required data.
    """
    name = name or Path(path).stem
    columns = (
        columns
        if columns is not None
        else {
            "Cycle Index": "Cycle",
            "Step Type": "Step",
            "Voltage(V)": "Voltage",
            "Chg. Spec. Cap.(mAh/g)": "Charge_Capacity",
            "DChg. Spec. Cap.(mAh/g)": "Discharge_Capacity",
            "dQm/dV(mAh/V.g)": "dQ/dV",
        }
    )
    sheet = sheet if sheet is not None else "record"
    logger.info(f"  Extracting {name}")
    df = _tidy_df(sheets[sheet]).rename(columns=columns)
    for column in _NUMERIC:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            logger.info(f"{path} : Missing column {column}")

    df, swapped = _apply_electrode_convention(df, electrode_type)
    n_cyc = int(df["Cycle"].nunique()) if "Cycle" in df else 0
    log_msg = (
        f"      {len(df):,} records, {n_cyc} cycles [anode: labels swapped]"
        if swapped
        else ""
    )
    logger.info(log_msg)
    return Dataset(
        name=name,
        df=df.reset_index(drop=True),
        meta=dict(meta or {}),
        electrode_type=electrode_type,
        labels_swapped=swapped,
        source_path=str(path),
        source_sha256=file_sha256(path),
    )


def _get_cell_value(df: pd.DataFrame, row: int, col: int) -> str | None:
    """
    Extract the contents of a cell.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe from which data is to be extracted.
    row : str
        Row to extract data from.
    col : str
        Column to extract data from.

    Returns
    -------
    str | None
        Returns the value of the cell, if a cell is `NaN` or out of range then `None` is returned.
    """
    try:
        value = df.iloc[row, col]
        return None if pd.isna(value) else value
    except IndexError, KeyError:
        return None


def _parse_numeric(text: str | None) -> float | None:
    """
    Extract leading numeric value from '12.72mg' (`12.7`), '4.3V' (`4.3`) and so forth.

    The regex here (`r"^([+-]?\\d*\\.?\\d+)"`) is looking for..

    1. Whether there is a -/+ at (`[+-]?`) the start of the string (`^`)  (optional)
    2. Zero or more numerical values (`\\d*`)
    3. An optional period/decimal point
    4. one or more digits (`\\d+`)

    These are grouped into a value which is then converted to a `float`.

    Parameters
    ----------
    text : str
        String from which to extract numeric values.

    Returns
    -------
    float | None
        Float if numeric value is found otherwise `None`.
    """
    if text is None:
        return None
    match = re.match(r"^([+-]?\d*\.?\d+)", str(text).strip())
    return float(match.group(1)) if match else None


def extract_neware_metadata(
    sheets: dict[str, pd.DataFrame],
    filename: str | None = None,
    sheet: str = "test",
    metadata: dict[str, int | float | None] | None = None,
) -> dict[str, int | float | None]:
    """Extract metadata from a neware worksheet.

    Parameters
    ----------
    sheets : dict[str, pd.DataFrame]
        Dictionary of Pandas dataframes, should contain `sheet`.
    filename : str, optional
        Filename being processed.
    sheet : str, optional
        Sheet to extract, default is `test` and typically won't need changing.
    metadata : dict[str, int | float | None], optional
        Metadata to extract, if `None` default is used.

    Returns
    -------
    dict[str, int | float | None]
        Dictionary of metadata, if no values are found a dictionary with empty values is returned.
    """
    metadata = FileStructures().neware["metadata"] if metadata is None else metadata
    sheet = sheet if sheet is not None else "record"
    if sheet not in sheets:
        logger.info(
            f"{filename} : Missing {sheet} worksheet, can not extract metadata."
        )
        return metadata
    df = sheets[sheet]
    n_rows = min(10, len(df))
    n_cols = min(15, len(df.columns))

    for row in range(n_rows):
        for col in range(n_cols - 2):
            label = _get_cell_value(df=df, row=row, col=col)
            if label is None:
                continue
            cell_label = str(label).lower().strip()
            # logger.debug(f"Attempting to extract {cell_label}")
            # Try cell + 2 columns to the right first (Neware merged cells), then cell + 1 column to the right
            value = _get_cell_value(
                df,
                row=row,
                col=col + 2,
            )
            # logger.debug(f"{cell_label} + 2 value : {value}")
            if value is None:
                value = _get_cell_value(df, row=row, col=col + 1)
                # logger.debug(f"{cell_label} + 1 value : {value}")
            if "active material" in cell_label:
                mass = _parse_numeric(value)
                if mass is not None and mass > 0:
                    metadata["active_material_mass_mg"] = mass
            elif "volt" in cell_label and "upper" in cell_label:
                v = _parse_numeric(value)
                if v is not None:
                    metadata["voltage_upper_V"] = v
            elif "volt" in cell_label and "lower" in cell_label:
                v = _parse_numeric(value)
                if v is not None:
                    metadata["voltage_lower_V"] = v
            elif "cycle" in cell_label and "count" in cell_label:
                with contextlib.suppress(ValueError), contextlib.suppress(TypeError):
                    metadata["cycle_count"] = int(float(str(value)))
            elif "barcode" in cell_label:
                for offset in [2, 1, 3]:
                    bv = _get_cell_value(df=df, row=row, col=col + offset)
                    if bv is not None and len(str(bv).strip()) > 3:
                        metadata["barcode"] = str(bv).strip()
                        break
            elif "start time" in cell_label:
                for offset in [2, 1]:
                    sv = _get_cell_value(df=df, row=row, col=col + offset)
                    if sv is not None:
                        metadata["test_start_date"] = str(sv).strip()
                        break

    return metadata


def find_files(
    base_dir: str | Path | None = None, file_ext: str = "xlsx"
) -> list[Path]:
    """
    Recursively scan the specified directory for images with the given file extension.

    Parameters
    ----------
    base_dir : str | Path, optional
        Directory to recursively search for files, if not specified the current directory is scanned.
    file_ext : str
        File extension to search for.

    Returns
    -------
    List
        List of files found with the extension in the given directory.
    """
    base_dir = Path("./") if base_dir is None else Path(base_dir)
    return list(base_dir.glob("**/*" + file_ext))
