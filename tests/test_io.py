"""Tests of the io module."""

from pathlib import Path

import pytest

from ratatosk import io

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


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
def test_open_xlsx(
    fixture_str: str, n_sheets: int, sheet_names: list[str], request
) -> None:
    """Test the `io.open_workbook()` function."""
    path = request.getfixturevalue(fixture_str)
    xlsx = io.open_xlsx(path)
    assert len(xlsx) == n_sheets
    assert list(xlsx.keys()) == sheet_names


@pytest.mark.parametrize(
    ("fixture_str", "path", "file_hash", "shape", "colnames"),
    [
        pytest.param(
            "record_df",
            RESOURCES / "record.csv",
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
    assert dataset.name == "record"
    assert dataset.frame.shape == shape
    assert list(dataset.frame.columns) == colnames
