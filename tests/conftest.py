"""Fixtures for the test suite."""

from pathlib import Path

import pandas as pd
import pytest

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.fixture
def file_path_simple_text_file() -> Path:
    """Path to `tests/resources/simple_text_file.txt`."""
    return RESOURCES / "simple_text_file.txt"


@pytest.fixture
def file_path_namimno2_xlsx() -> Path:
    """Path to `tests/resources/JQ-P3-NaNiMnO2-35_-2-4.2V-0.1C-A-20062025.xlsx`"""
    return RESOURCES / "JQ-P3-NaNiMnO2-35_-2-4.2V-0.1C-A-20062025.xlsx"


@pytest.fixture
def file_path_mnc111_xlsx() -> Path:
    """Path to `tests/resources/JQ_MNC111_3-4.5V_0.1C_B_08052026.xlsx`"""
    return RESOURCES / "JQ_MNC111_3-4.5V_0.1C_B_08052026.xlsx"


@pytest.fixture
def record_df() -> pd.DataFrame:
    """Load five rows from a `record` worksheet from `.csv` and return as Pandas DataFrame."""
    return pd.read_csv(RESOURCES / "record.csv", sep=",")
