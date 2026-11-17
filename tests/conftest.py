"""Fixtures for the test suite."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ratatosk import classes

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
CSV_DIR = RESOURCES / "csv"
XLS_DIR = RESOURCES / "xls"
YAML_DIR = RESOURCES / "yaml"


@pytest.fixture
def file_path_simple_text_file() -> Path:
    """Path to `tests/resources/simple_text_file.txt`."""
    return RESOURCES / "simple_text_file.txt"


@pytest.fixture
def file_path_namimno2_xlsx() -> Path:
    """Path to `tests/resources/JQ-P3-NaNiMnO2-35_-2-4.2V-0.1C-A-20062025.xlsx`"""
    return XLS_DIR / "JQ-P3-NaNiMnO2-35_-2-4.2V-0.1C-A-20062025.xlsx"


@pytest.fixture
def file_path_mnc111_xlsx() -> Path:
    """Path to `tests/resources/JQ_MNC111_3-4.5V_0.1C_B_08052026.xlsx`"""
    return XLS_DIR / "JQ_MNC111_3-4.5V_0.1C_B_08052026.xlsx"


@pytest.fixture
def neware_metadata_df() -> pd.DataFrame:
    """Neware metadata as dataframe."""
    return pd.read_csv(CSV_DIR / "neware_metadata.csv", sep=",")


@pytest.fixture
def neware_record_df() -> pd.DataFrame:
    """Load five rows from a `record` worksheet from `.csv` and return as Pandas DataFrame."""
    return pd.read_csv(CSV_DIR / "neware_record.csv", sep=",")


@pytest.fixture
def default_parameters() -> classes.Parameters:
    """The default configuration parameters."""
    return classes.Parameters()


@pytest.fixture
def dummy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a_column": [1, 2.0, "three", np.nan, "", None],
            "b_column": [None, "", np.nan, "three", 2.0, 1],
        }
    )
