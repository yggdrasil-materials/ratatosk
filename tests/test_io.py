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


def test_dataest() -> None:
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


def test_preferred_engine() -> None:
    """
    Test the `io.preferred_engine()` function.

    No test written yet, likely to remove this functions as its redundant, we will ensure that the `calamine` package is
    always available by making it a package dependency.
    """
    assert True


@pytest.mark.filterwarnings("error::ResourceWarning")
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
    fixture_str: str, n_sheets: int, sheet_names: str, request
) -> None:
    """Test the `io.open_workbook()` function."""
    path = request.getfixturevalue(fixture_str)
    xlsx = io.open_xlsx(path)
    assert len(xlsx) == n_sheets
    assert list(xlsx.keys()) == sheet_names


def test_read_neware() -> None:
    """Test the `io.open_workbook()` function."""
    assert True
