"""Tests for the classes module."""

from importlib.resources import files
from pathlib import Path

import pytest

import ratatosk
from ratatosk import classes

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
YAML = RESOURCES / "yaml"

# pylint: disable=no-member


@pytest.mark.parametrize(
    (
        "base_dir",
        "output_dir",
        "log_level",
        "cores",
        "file_ext",
        "active",
        "anode_labels_swapped",
        "barcode",
        "battery_chemistry",
        "blend",
        "cell_id",
        "charge_current_a_from_file",
        "charge_rate_c",
        "composition",
        "counter_electrode",
        "cycling_verbose",
        "dead_cell",
        "discharge_current_a_from_file",
        "dqdv",
        "edge_exclusion_mv",
        "electrode",
        "electrolyte",
        "figure",
        "fit_unresolved",
        "key_cycles",
        "max_cycle",
        "peak",
        "polyorder",
        "power_energy_analysis",
        "qc_cycle",
        "rebin_width_mv",
        "reference_cycle",
        "run_peak_fitting",
        "second_smooth_window",
        "separator_type",
        "shoulder_detection",
        "smoothing_window",
        "spike",
        "stop_fitting_at_death",
        "terminal_trim",
        "termination",
        "theoretical_capacity",
        "theoretical_charge",
        "total_electrode_coating_mg",
        "total_loading_mg_cm2",
        "tracking_tolerance_mv",
        "voltage",
    ),
    [
        pytest.param(
            Path("./"),  # base_dir
            Path("./output"),  # output_dir
            "info",  # log_level
            2,  # cores
            "xlsx",  # file_ext
            {  # active
                "fraction": 0.8,
                "loading_mg_cm2": 4.173396285520811,
                "material_mass_mg": 4.72,
            },
            False,  # anode_labels_swapped
            None,  # barcode
            None,  # battery_chemistry
            [80.0, 10.0, 10.0],  # blend
            "A",  # cell_id
            8.18e-05,  # charge_current_a_from_file
            0.1741,  # charge_rate_c
            None,  # composition
            {  # counter_electrode
                "mass_mg": 97.61556693234206,
                "measured": False,
                "metal": "Na",
                "metal_source": "assumed",
            },
            True,  # cycling_verbose
            {  # dead_cell
                "capacity_fraction": 0.02,
                "run": 3,
            },
            None,  # discharge_current_a_from_file
            {"computation": "histogram", "bin_mv": None},  # dqdv
            50,  # edge_exclusion_mv
            {  # electrode:
                "area_cm2": 1.1309733552923256,
                "diameter_mm": 12.0,
                "type": "Positive",
            },
            {  # electrolyte
                "composition": None,
                "volume_ul": 100.0,
            },
            {  # figure
                "colour_palette": "viridis_r",
                "file_format": "png",
                "height_in": 5.0,
                "histogram_bin_mv": "auto",
                "histogram_smooth_bins": "auto",
                "theme": "viridis_r",
                "width_in": 6.73,
            },
            True,  # fit_unresolved
            [  # key_cycles
                1,
                5,
                10,
                20,
                30,
                40,
                50,
            ],
            None,  # max_cycle
            {  # peak
                "min_distance_mv": "auto",
                "min_width_mvv": 5,
                "prominence_fraction": 0.05,
            },
            3,  # polyorder
            True,  # power_energy_analysis
            3,  # qc_cycle
            "auto",  # rebin_width_mv
            2,  # reference_cycle
            True,  # run_peak_fitting
            "auto",  # second_smooth_window
            "SLS Select GF/6 glass microfibre (SLS4522)",  # separator_type
            True,  # shoulder_detection
            "auto",  # smoothing_window
            {  # spike:
                "removal": "auto",
                "threshold_multiplier": "auto",
                "window_size": 5,
            },
            True,  # stop_fitting_at_death
            {  # terminal_trim
                "run": True,
                "fixed_mv": 75.0,
                "factor": 1.4,
                "max_fraction": 0.1,
                "band_mv": 20.0,
                "interior": (0.2, 0.8),
            },
            {  # termination
                "tol_mv": 20.0,
                "tol_spreads": 5.0,
                "reference_n": 10,
            },
            {  # theoretical_capacity
                "library_formula": None,
                "mah": 0.4719999999999999,
                "mah_g": 100.0,
                "source": None,
            },
            {  # theoretical_charge:
                "current_ma": 0.08217519999999999,
                "rate_mah_g": 17.41,
            },
            5.8999999999999995,  # total_electrode_coating_mg
            5.216745356901013,  # total_loading_mg_cm2
            80,  # tracking_tolerance_mv
            {  # voltage
                "limit_lower_v_from_file": 1.9,
                "limit_upper_v_from_file": 4.3,
                "lower_v": 2.0,
                "upper_v": 4.2,
                "window_source": "measured from data",
            },
            id="Default configuration",
        ),
    ],
)
def test_parameters(
    base_dir: Path,
    output_dir: Path,
    log_level: str,
    cores: int,
    file_ext: str,
    active: dict[str, float],
    anode_labels_swapped: bool,
    barcode: str | None,
    battery_chemistry: str | None,
    blend: list[float],
    cell_id: str | None,
    charge_current_a_from_file: float | None,
    charge_rate_c: float | None,
    composition: str | None,
    counter_electrode: dict[str, str | float | bool],
    cycling_verbose: bool,
    dead_cell: dict[str, int | float],
    discharge_current_a_from_file: float | None,
    dqdv: dict[str, str | None],
    edge_exclusion_mv: float,
    electrode: dict[str, float | str],
    electrolyte: dict[str, float | str | None],
    figure: dict[str, float | str],
    fit_unresolved: bool,
    key_cycles: list[int],
    max_cycle: int | None,
    peak: dict[str, str | int | float],
    polyorder: int,
    power_energy_analysis: bool,
    qc_cycle: int,
    rebin_width_mv: str,
    reference_cycle: int,
    run_peak_fitting: bool,
    second_smooth_window: str,
    separator_type: str,
    shoulder_detection: bool,
    smoothing_window: str,
    spike: dict[str, str | int],
    stop_fitting_at_death: bool,
    terminal_trim: dict[str, float | list[float]],
    termination: dict[str, float | int],
    theoretical_capacity: dict[str, float | None],
    theoretical_charge: dict[str, float],
    total_electrode_coating_mg: float,
    total_loading_mg_cm2: float,
    tracking_tolerance_mv: float,
    voltage: dict[str, float | str],
) -> None:
    """Test for the `Parameters` class."""
    parameters = classes.Parameters()
    assert parameters.base_dir == base_dir
    assert parameters.output_dir == output_dir
    assert parameters.log_level == log_level
    assert parameters.cores == cores
    assert parameters.file_ext == file_ext
    assert isinstance(parameters.active, classes.Active)
    assert parameters.active.model_dump() == active
    assert parameters.anode_labels_swapped == anode_labels_swapped
    assert parameters.barcode == barcode
    assert parameters.battery_chemistry == battery_chemistry
    assert parameters.blend == blend
    assert parameters.cell_id == cell_id
    assert parameters.charge_current_a_from_file == charge_current_a_from_file
    assert parameters.charge_rate_c == charge_rate_c
    assert parameters.composition == composition
    assert isinstance(parameters.counter_electrode, classes.CounterElectrode)
    assert parameters.counter_electrode.model_dump() == counter_electrode
    assert parameters.cycling_verbose == cycling_verbose
    assert isinstance(parameters.dead_cell, classes.DeadCell)
    assert parameters.dead_cell.model_dump() == dead_cell
    assert parameters.discharge_current_a_from_file == discharge_current_a_from_file
    assert isinstance(parameters.dqdv, classes.DqDv)
    assert parameters.dqdv.model_dump() == dqdv
    assert parameters.edge_exclusion_mv == edge_exclusion_mv
    assert isinstance(parameters.electrode, classes.Electrode)
    assert parameters.electrode.model_dump() == electrode
    assert isinstance(parameters.electrolyte, classes.Electrolyte)
    assert parameters.electrolyte.model_dump() == electrolyte
    assert isinstance(parameters.figure, classes.Figure)
    assert parameters.figure.model_dump() == figure
    assert parameters.fit_unresolved == fit_unresolved
    assert parameters.key_cycles == key_cycles
    assert parameters.max_cycle == max_cycle
    assert isinstance(parameters.peak, classes.Peak)
    assert parameters.peak.model_dump() == peak
    assert parameters.polyorder == polyorder
    assert parameters.power_energy_analysis == power_energy_analysis
    assert parameters.qc_cycle == qc_cycle
    assert parameters.rebin_width_mv == rebin_width_mv
    assert parameters.reference_cycle == reference_cycle
    assert parameters.run_peak_fitting == run_peak_fitting
    assert parameters.second_smooth_window == second_smooth_window
    assert parameters.separator_type == separator_type
    assert parameters.shoulder_detection == shoulder_detection
    assert parameters.smoothing_window == smoothing_window
    assert isinstance(parameters.spike, classes.Spike)
    assert parameters.spike.model_dump() == spike
    assert parameters.stop_fitting_at_death == stop_fitting_at_death
    assert isinstance(parameters.terminal_trim, classes.TerminalTrim)
    assert parameters.terminal_trim.model_dump() == terminal_trim
    assert isinstance(parameters.termination, classes.Termination)
    assert parameters.termination.model_dump() == termination
    assert isinstance(parameters.theoretical_capacity, classes.TheoreticalCapacity)
    assert parameters.theoretical_capacity.model_dump() == theoretical_capacity
    assert isinstance(parameters.theoretical_charge, classes.TheoreticalCharge)
    assert parameters.theoretical_charge.model_dump() == theoretical_charge
    assert parameters.total_electrode_coating_mg == total_electrode_coating_mg
    assert parameters.total_loading_mg_cm2 == total_loading_mg_cm2
    assert parameters.tracking_tolerance_mv == tracking_tolerance_mv
    assert isinstance(parameters.voltage, classes.Voltage)
    assert parameters.voltage.model_dump() == voltage


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
