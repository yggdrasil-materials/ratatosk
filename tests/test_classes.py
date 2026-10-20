"""Tests for the classes module."""

from importlib.resources import files
from pathlib import Path

import pytest

import ratatosk
from ratatosk import classes

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
YAML = RESOURCES / "yaml"


@pytest.mark.parametrize(
    ("config_file", "base_dir", "output_dir", "log_level", "cores"),
    [
        pytest.param(
            None,
            Path("./"),
            Path("./output"),
            "info",
            2,
            id="Default configuration",
        ),
        pytest.param(
            YAML / "dummy_config.yaml",
            Path("/tmp/"),
            Path("./output"),
            "warning",
            6,
            id="Dummy configuration",
        ),
    ],
)
def test_parameters(
    config_file: Path | None,
    base_dir: Path,
    output_dir: Path,
    log_level: str,
    cores: int,
) -> None:
    """Test for the `Parameters` class."""
    if config_file is None:
        parameters = classes.Parameters()
    else:
        parameters = classes.Parameters(config_file=config_file)
    assert parameters.base_dir == base_dir
    assert parameters.output_dir == output_dir
    assert parameters.log_level == log_level
    assert parameters.cores == cores


@pytest.mark.parametrize(
    (
        "config_file",
        "neware_column_mapping",
        "neware_numeric",
        "neware_step_names",
        "neware_metadata",
        "neware_sheets",
    ),
    [
        pytest.param(
            files(ratatosk).joinpath("default_dictionary.yaml"),
            {
                "Cycle Index": "Cycle",
                "Step Type": "Step",
                "Voltage(V)": "Voltage",
                "Chg. Spec. Cap.(mAh/g)": "Charge_Capacity",
                "DChg. Spec. Cap.(mAh/g)": "Discharge_Capacity",
                "dQm/dV(mAh/V.g)": "dQ/dV",
            },
            ["Cycle", "Voltage", "Charge_Capacity", "Discharge_Capacity", "dQ/dV"],
            {
                "CCCV Chg": "Charge",
                "CCCV CHG": "Charge",
                "CC Chg": "Charge",
                "CC CHG": "Charge",
                "CC DChg": "Discharge",
                "CC DCHG": "Discharge",
                "CCCV DChg": "Discharge",
            },
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
            {
                "test_data": "record",
                "metadata": "test",
            },
            id="Default dictionary",
        ),
        pytest.param(
            YAML / "dummy_dictionary.yaml",
            {"a": "b", "c": "d"},
            ["e", "f"],
            {"g": "h", "i": "j"},
            {"k": None, "l": None},
            {"m": "n", "o": "p"},
            id="Dummy dictionary",
        ),
    ],
)
def test_data_dictionary_neware(
    config_file: Path | None,
    neware_column_mapping: dict[str, str],
    neware_numeric: list[str],
    neware_step_names: dict[str, str],
    neware_metadata: dict[str, None],
    neware_sheets: dict[str, str],
) -> None:
    """Test for the `Parameters` class."""
    data_dictionary = classes.FileStructures(config_file=config_file)
    assert data_dictionary.neware["column_mapping"] == neware_column_mapping
    assert data_dictionary.neware["numeric"] == neware_numeric
    assert data_dictionary.neware["step_names"] == neware_step_names
    assert data_dictionary.neware["metadata"] == neware_metadata
    assert data_dictionary.neware["sheets"] == neware_sheets
