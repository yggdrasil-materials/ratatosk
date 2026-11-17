"""Tests of the io module."""

import argparse
import os
import sys
from pathlib import Path
from pkgutil import get_data
from typing import Any

import pytest
from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.scanner import ScannerError

from ratatosk import io

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
CSV_DIR = RESOURCES / "csv"

GITHUB_WIN = os.getenv("GITHUB_ACTIONS") == "true" and sys.platform == "win32"

default_config = get_data(package="ratatosk", resource="default_config.yaml")
yaml = YAML(typ="safe")
DEFAULT_CONFIG = yaml.load(default_config)
CONFIG = {
    "this": "is",
    "a": "test",
    "yaml": "file",
    "numbers": 123,
    "logical": True,
    "nested": {"something": "else"},
    "a_list": [1, 2, 3],
}

# pylint: disable=protected-access


@pytest.mark.skipif(GITHUB_WIN, reason="SHA256 checksums differ under M$-Win")
@pytest.mark.parametrize(
    ("fixture_str", "expected_hash"),
    [
        pytest.param(
            "file_path_simple_text_file",
            "2d91bf37028af60bed34bae8895d99c36c6e14ca9ff3ba6c137665d9992f225b",
            id="plain text file",
        ),
        pytest.param(
            "file_path_namimno2_xlsx",
            "5e0afad0ea7d4f9ef350e39a67e964803a2e32a36d395b428b81481d07f97353",
            id="NaNiMnO2 .xlsx file",
        ),
        pytest.param(
            "file_path_mnc111_xlsx",
            "f3def1c0625658c01cd8a1188e4a8ed9ae56b897b6987cccf5b80cec0fe26a4c",
            id="MNC .xlsx file",
        ),
    ],
)
def test_file_sha256(fixture_str, expected_hash: str, request) -> None:
    """Test `io.file_sha156()` function."""
    path = request.getfixturevalue(fixture_str)
    assert io.file_sha256(str(path)) == expected_hash


def test_convert_path(tmp_path: Path) -> None:
    """Test ``convert_path()``."""
    test_dir = str(tmp_path)
    converted_path = io.convert_path(test_dir)
    assert isinstance(converted_path, Path)
    assert tmp_path == converted_path


def test_read_yaml() -> None:
    """Test reading of YAML files using ``read_yaml()``."""
    # Dummy config for testing 'read_yaml()'
    sample_config = io.read_yaml(RESOURCES / "test.yaml")
    assert sample_config == CONFIG


@pytest.mark.parametrize(
    ("filename", "expected_error"),
    [
        pytest.param(
            RESOURCES / "does_not_exist.yaml", FileNotFoundError, id="FileNotFoundError"
        ),
        pytest.param(RESOURCES / "not.yaml", ScannerError, id="ScannerError"),
        pytest.param(
            RESOURCES / "duplicate_keys.yaml",
            YAMLError,
            id="YAMLError (duplicate keys)",
        ),
        pytest.param(
            RESOURCES / "mixed_indentation.yaml",
            YAMLError,
            id="YAMLError (mixed indentation)",
        ),
    ],
)
def test_read_yaml_exceptions(filename: Path, expected_error: Any) -> None:
    """Test ``read_yaml()`` raises different exceptions."""
    with pytest.raises(expected_error):
        io.read_yaml(filename=filename)


@pytest.mark.parametrize(
    ("args"),
    [
        pytest.param(
            argparse.Namespace(filename=None, type="config"), id="config, no filename"
        ),
        pytest.param(
            argparse.Namespace(filename="another_config.yaml", type="config"),
            id="config, alternative filename",
        ),
        pytest.param(
            argparse.Namespace(filename=None, type="data-dictionary"),
            id="data-dictionary, no filename",
        ),
        pytest.param(
            argparse.Namespace(
                filename="another_data-dictionary.yaml", type="data-dictionary"
            ),
            id="data-dictionary, alternative filename",
        ),
    ],
)
def test_write_config(args: argparse.Namespace, tmp_path: Path) -> None:
    """Test writing of YAML configuration file using ``write_config()``."""
    args.output_dir = tmp_path
    io.write_config(args)
    if args.filename is None and args.type == "config":
        assert Path(tmp_path / "default_config.yaml").exists()
    elif args.filename is None and args.type == "data-dictionary":
        assert Path(tmp_path / "default_dictionary.yaml").exists()
    else:
        assert Path(tmp_path / args.filename).exists()


def test_dataset() -> None:
    """Test the `Dataset` class."""
    assert True


def test_tidy_steps() -> None:
    """Test the `io._tidy_steps()` function."""
    assert True


def test_apply_electrode_convetion_private() -> None:
    """Test the `io._apply_electrode_convention()` function."""
    assert True


def test_apply_electrode_convetion() -> None:
    """Test the `io.apply_electrode_convention()` function."""
    assert True


def test_truncate_cycles() -> None:
    """Test the `io.truncate_cycles()` function."""
    assert True


@pytest.mark.parametrize(
    ("fixture_str", "n_sheets", "sheet_names"),
    [
        pytest.param(
            "file_path_namimno2_xlsx",
            8,
            ["unit", "test", "cycle", "step", "record", "log", "idle", "curve"],
            id="NaNiMnO2 .xlsx file",
        ),
        pytest.param(
            "file_path_mnc111_xlsx",
            8,
            ["unit", "test", "cycle", "step", "record", "log", "idle", "curve"],
            id="MNC .xlsx file",
        ),
    ],
)
def test_read_xlsx(
    fixture_str: str, n_sheets: int, sheet_names: list[str], request
) -> None:
    """Test the `io.open_workbook()` function."""
    path = request.getfixturevalue(fixture_str)
    xlsx = io.read_xlsx(path)
    assert len(xlsx) == n_sheets
    assert list(xlsx.keys()) == sheet_names


@pytest.mark.skipif(GITHUB_WIN, reason="SHA256 checksums differ under M$-Win")
@pytest.mark.parametrize(
    ("fixture_str", "path", "file_hash", "shape", "colnames"),
    [
        pytest.param(
            "neware_record_df",
            CSV_DIR / "neware_record.csv",
            "3b192ee1a8ce8ed76d3b49b57f9006107ec5f416f93651952cbc64939be5d6d6",
            (3, 26),
            [
                "DataPoint",
                "Cycle",
                "Step Index",
                "Step",
                "Time",
                "Cumulative Time",
                "Current(A)",
                "Voltage",
                "Capacity(Ah)",
                "Spec. Cap.(mAh/g)",
                "Chg. Cap.(Ah)",
                "Charge_Capacity",
                "DChg. Cap.(Ah)",
                "Discharge_Capacity",
                "Energy(Wh)",
                "Spec. Energy(mWh/g)",
                "Chg. Energy(Wh)",
                "Chg. Spec. Energy(mWh/g)",
                "DChg. Energy(Wh)",
                "DChg. Spec. Energy(mWh/g)",
                "Date",
                "Power(W)",
                "dQ/dV(mAh/V)",
                "dQ/dV",
                "Contact resistance(mΩ)",
                "Module start-stop switch",
            ],
            id="neware record.csv",
        )
    ],
)
def test_read_neware(
    fixture_str: str,
    path: str | Path,
    file_hash: str,
    shape: tuple[int, int],
    colnames: list[str],
    request,
) -> None:
    """Test the `io.open_workbook()` function."""
    sheets = {"record": request.getfixturevalue(fixture_str)}
    dataset = io.read_neware(sheets=sheets, sheet="record", path=path)
    assert dataset.source_sha256 == file_hash
    assert dataset.name == "neware_record"
    assert dataset.df.shape == shape
    assert list(dataset.df.columns) == colnames


@pytest.mark.parametrize(
    ("df_fixture", "col", "row", "expected"),
    [
        pytest.param("dummy_df", 0, 0, 1, id="col 0, row 0, int"),
        pytest.param("dummy_df", 0, 1, 2.0, id="col 0, row 1, float"),
        pytest.param("dummy_df", 0, 2, "three", id="col 0, row 2, str"),
        pytest.param("dummy_df", 0, 3, None, id="col 0, row 3, NaN returned as NaN"),
        pytest.param("dummy_df", 0, 4, "", id="col 0, row 4, empty string"),
        pytest.param("dummy_df", 0, 5, None, id="col 0, row 5, None"),
        pytest.param("dummy_df", 1, 0, None, id="col 1, row 0, None"),
        pytest.param("dummy_df", 1, 1, "", id="col 1, row 1, empty string"),
        pytest.param("dummy_df", 1, 2, None, id="col 1, row 2, NaN returned as NaN"),
        pytest.param("dummy_df", 1, 3, "three", id="col 1, row 3, str"),
        pytest.param("dummy_df", 1, 4, 2.0, id="col 1, row 4, float"),
        pytest.param("dummy_df", 1, 5, 1, id="col 1, row 5, int"),
    ],
)
def test__get_cell_value(
    df_fixture: str, col: int, row: int, expected: str | float | None, request
) -> None:
    """Test for `_get_cell_value()`."""
    df = request.getfixturevalue(df_fixture)
    assert io._get_cell_value(df=df, row=row, col=col) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("0.1452", 0.1452, id="unsigned float, no text"),
        pytest.param("-0.12", -0.12, id="negative float, no text"),
        pytest.param("+10.1678", 10.1678, id="positive float, no text"),
        pytest.param("-.12", -0.12, id="negative float, no text, no leading 0"),
        pytest.param("+.1678", 0.1678, id="positive float, no text, no leading 0"),
        pytest.param("0.1452", 0.1452, id="unsigned float, text"),
        pytest.param("-0.12a", -0.12, id="negative float, text"),
        pytest.param("+10.1678a", 10.1678, id="positive float, text"),
        pytest.param("-.12a", -0.12, id="negative float, text, no leading 0"),
        pytest.param("+.1678a", 0.1678, id="positive float, text, no leading 0"),
        pytest.param("1", 1.0, id="unsigned int, no text"),
        pytest.param("-10", -10.0, id="negative int, no text"),
        pytest.param("+100", 100.0, id="positive int, no text"),
        pytest.param("1b", 1.0, id="unsigned int, text"),
        pytest.param("-10b", -10.0, id="negative int, text"),
        pytest.param("+100b", 100.0, id="positive int, text"),
    ],
)
def test__parse_numeric(text: str, expected: float) -> None:
    """Test for `_parse_numeric()`."""
    assert io._parse_numeric(text=text) == expected


@pytest.mark.parametrize(
    ("neware_metadata_fixture", "filename", "sheet", "metadata"),
    [
        pytest.param("neware_metadata_df", "testing", "test", None, id="No metadata"),
        pytest.param(
            "neware_metadata_df",
            "testing",
            "test",
            {
                "active_material_mass_mg": None,
                "voltage_upper_V": None,
                "voltage_lower_V": None,
                "cycle_count": None,
                "c_rate": None,
                "charge_current_A": None,
                "discharge_current_A": None,
                "barcode": None,
                "test_start_date": None,
            },
            id="Explicit metadata",
        ),
    ],
)
def test_extract_neware_metadata(
    neware_metadata_fixture: str,
    filename: str,
    sheet: str,
    metadata: dict[str, int | float | None],
    request,
    snapshot,
) -> None:
    """Test for `extract_neware_metadata()`."""
    sheets = {"test": request.getfixturevalue(neware_metadata_fixture)}
    assert io.extract_neware_metadata(sheets, filename, sheet, metadata) == snapshot
