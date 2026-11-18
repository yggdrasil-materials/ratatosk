"""Dataclass definitions."""

from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

# import numpy.typing as npt
import pandas as pd
import yaml

# from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

import ratatosk

DEFAULT_CONFIG = files(ratatosk) / "default_config.yaml"
DEFAULT_DATA_DICTIONARY = files(ratatosk) / "default_dictionary.yaml"


def load_default_yaml(
    yaml_file: str = "default_config.yaml", encoding: str = "utf-8"
) -> dict[str, Any]:
    """
    Load the specified YAML file.

    Parameters
    ----------
    yaml_file : str
        Filename to be loaded, should be part of the `ratatosk` package, typically under `src/ratatosk/`.
    encoding : str
        File encoding, default is `utf-8` and shouldn't need changing.

    Returns
    -------
    dict[str, Any]
        Dictionary representation of the YAML configuration.
    """
    resource = files(ratatosk).joinpath(yaml_file)
    data = yaml.safe_load(resource.read_text(encoding=encoding))
    return data or {}


class DefaultYamlSource(PydanticBaseSettingsSource):
    """
    Settings source backed by the YAML file bundled with the package.

    Parameters
    ----------
    settings_cls : type[BaseSettings]
        Inherited `settings_cls`.
    yaml_file : str
        YAML file to be loaded.
    """

    def __init__(
        self, settings_cls: type[BaseSettings], yaml_file: str = "default_config.yaml"
    ) -> None:
        """
        Initialise the class.

        Parameters
        ----------
        settings_cls : type[BaseSettings]
            Inherited `settings_cls`.
        yaml_file : str
            YAML file to be loaded.
        """
        super().__init__(settings_cls)
        self.yaml_file = yaml_file

    def get_field_value(self, field, field_name: str) -> tuple[None, str, bool]:  # noqa: ARG002
        """
        Get a field value, required due to inheritance from `PydanticBaseSettingsSource`.

        **NB** Not used directly, `__call__` supplies the complete mapping.

        Parameters
        ----------
        field : str
            Field.
        field_name : str
            Field name to extract.

        Returns
        -------
        tuple[None, str, bool]
            The `field_name` as part of a tuple.
        """
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """
        Call method for the class.

        Returns
        -------
        dict[str, Any]
            Returns the contents of a YAML file as a dictionary.
        """
        return load_default_yaml(self.yaml_file)


class TerminalTrim(BaseModel):
    """A class for type-checking the `terminal_trim` options."""

    model_config = ConfigDict(extra="forbid")
    run: bool = Field(default=True, description="Whether to trim half-cycles.")
    fixed_mv: float = Field(default=75.0, description="Fixed mV to trim by.")
    factor: float = Field(default=1.4, description="???")
    max_fraction: float = Field(default=0.1, description="???")
    band_mv: float = Field(default=20.0, description="???")
    interior: tuple[float, float] = Field(default=(0.2, 0.8), description="???")


class Termination(BaseModel):
    """Class for type-checking the `termination` options."""

    model_config = ConfigDict(extra="forbid")
    tol_mv: float = Field(default=20.0, description="Tolerance for termination in mV.")
    tol_spreads: float = Field(
        default=5.0, description="Tolerance spread for termination."
    )
    reference_n: int = Field(
        default=10, description="Reference number for termination."
    )


class DqDv(BaseModel):
    """Class for type-checking the `dqdv` options."""

    model_config = ConfigDict(extra="forbid")
    computation: str = Field(
        default="histogram",
        description="Differential method, valid values are 'histogram' or 'derivative'",
    )
    bin_mv: float | None = Field(
        default=None,
        description="Voltage bin width for the histogram path in mV. Determines the resolution of the curve.",
    )


class Figure(BaseModel):
    """Class for type-checking the `figure` options."""

    model_config = ConfigDict(extra="forbid")
    # ns-rse 2026-10-29 Duplication of theme, choose one and use it
    colour_palette: str = Field(
        default="viridis_r",
        description="Colour palette to use when plotting (default 'viridis_r') should match 'theme'.",
    )
    file_format: str = Field(
        default="png",
        description="Default format for figure output (default is 'png').",
    )

    height_in: float = Field(
        default=5.0,
        description="Figure height in inches applied to every figure produced.",
    )
    histogram_smooth_bins: str | float | None = Field(
        default="auto",
        description="Histogram smoothing of bins, may be 'auto' or a float.",
    )
    histogram_bin_mv: str | float | None = Field(
        default="auto",
        description="Histogram bings in mV, may be 'auto' or a float value.",
    )
    theme: str = Field(
        default="viridis_r",
        description="Theme to plot figures with.",
    )
    width_in: float = Field(
        default=6.73,
        description="Figure width in inches applied to every figure produced.",
    )


class DeadCell(BaseModel):
    """Class for type-checking the `DeadCell` options."""

    model_config = ConfigDict(extra="forbid")
    capacity_fraction: float = Field(
        default=0.02, description="Dead cell capacity fraction."
    )
    run: int = Field(default=3, description="Runs to signify a dead cell.")


class TheoreticalCapacity(BaseModel):
    """Class for type-checking the `TheoreticalCapacity` options."""

    model_config = ConfigDict(extra="forbid")
    library_formula: str | None = Field(
        default=None, description="Theoretical capacity library formula."
    )
    mah: float | None = Field(default=None, description="Theoretical capacity in mAh.")
    mah_g: float | None = Field(
        default=None, description="Theoretical capacity in mAh/g."
    )
    source: str | None = Field(
        default=None, description="Source of theoretical capacity."
    )


class TheoreticalCharge(BaseModel):
    """Class for type-checking the `TheoreticalCharge` options."""

    model_config = ConfigDict(extra="forbid")
    current_ma: float | None = Field(
        default=None, description="Theoretical charge current in mA."
    )
    rate_mah_g: float | None = Field(
        default=None, description="Theoretical charge rate in mAh/g."
    )


class Voltage(BaseModel):
    """Class for type-checking the `Voltage` options."""

    model_config = ConfigDict(extra="forbid")
    limit_lower_v_from_file: float | None = Field(
        default=None, description="Lower voltage limit from data."
    )
    limit_upper_v_from_file: float | None = Field(
        default=None, description="Upper voltage limit from data."
    )
    lower_v: float | None = Field(default=None, description="Lower voltage.")
    upper_v: float | None = Field(default=None, description="Upper voltage.")
    window_source: str = Field(
        default="measured from data", description="Source of voltages"
    )


class Spike(BaseModel):
    """Class for type-checking the `Spike` options."""

    model_config = ConfigDict(extra="forbid")
    removal: str | None = Field(default="auto", description="Method of spike removal.")
    threshold_multiplier: str | float | None = Field(
        default="auto", description="Threshold multiplier for spike removal."
    )
    window_size: int = Field(default=5, description="Window size.")


class Peak(BaseModel):
    """Class for type-checking the `Peak` options."""

    model_config = ConfigDict(extra="forbid")
    min_distance_mv: str | float | None = Field(
        default="auto", description="Minimum distance in mV."
    )
    min_width_mvv: int = Field(default=5, description="Minimum width in mV.")
    prominence_fraction: float = Field(default=0.05, description="Prominence fraction.")


class Electrolyte(BaseModel):
    """Class for type-checking the `Electrolyte` options."""

    model_config = ConfigDict(extra="forbid")
    composition: str | None = Field(
        default=None, description="Electrolyte composition."
    )
    volume_ul: float = Field(default=100.0, description="Volume in uL.")


class Electrode(BaseModel):
    """Class for type-checking the `Electrode` options."""

    model_config = ConfigDict(extra="forbid")
    area_cm2: float | None = Field(default=None, description="Electrode area in cm2.")
    diameter_mm: float | None = Field(
        default=None, description="Electrode diameter in mm."
    )
    type: str | None = Field(
        default="Positive",
        description="Electrode type (either 'Positive' or 'Negative').",
    )


class CounterElectrode(BaseModel):
    """Class for type-checking the `CounterElectrode` options."""

    model_config = ConfigDict(extra="forbid")
    mass_mg: float | None = Field(default=None, description="Mass in mg.")
    measured: bool | None = Field(
        default=None,
        description="Whether the Counter Electrode features were measured.",
    )
    metal: str | None = Field(
        default=None, description="Metal type of counter electrode."
    )
    metal_source: str | None = Field(
        default="assumed",
        description="Source of metal type of counter electrode (default is 'assumed').",
    )


class Active(BaseModel):
    """Class for type-checking the `Active` options."""

    model_config = ConfigDict(extra="forbid")
    fraction: float = Field(default=0.8, description="Active fraction")
    loading_mg_cm2: float | None = Field(
        default=None, description="Active loading in mg/cm2."
    )
    material_mass_mg: float | None = Field(
        default=None, description="Active material mass in mg."
    )


class Parameters(BaseSettings):
    """Parameters for analysis."""

    # Attributes are listed alphabetically to facilitate discovery, please maintain this when adding new attributes.
    active: Active = Field(
        default_factory=Active, description="Active configuration options."
    )

    anode_labels_swapped: bool = Field(
        default=False, description="Anode labels swapped."
    )

    base_dir: Path = Field(default=Path("./"), description="Base directory.")
    barcode: str | None = Field(
        default=None, description="Barcode (typically mirrors the filename)."
    )

    battery_chemistry: str | None = Field(
        default=None, description="Battery chemistry."
    )
    blend: list[float] = Field(default=[80.0, 10.0, 10.0], description="Blend.")
    cell_id: str | None = Field(default=None, description="Cell ID.")
    charge_current_a_from_file: float | None = Field(
        default=None, description="Charge current A from file."
    )
    charge_rate_c: float | None = Field(
        default=None, description="Charge rate current."
    )

    composition: str | None = Field(default=None, description="Composition.")
    counter_electrode: CounterElectrode = Field(
        default_factory=CounterElectrode,
        description="Counter electrode configuration options.",
    )
    cores: int = Field(
        default=2, description="Cores to run optimisation on in parallel."
    )
    cycling_verbose: bool = Field(
        default=True, description="How much of the cycling is reported."
    )
    dead_cell: DeadCell = Field(
        default_factory=DeadCell, description="Dead cell configuration options."
    )
    discharge_current_a_from_file: float | None = Field(
        default=None, description="Discharge current A from file."
    )
    dqdv: DqDv = Field(default_factory=DqDv, description="dq/dv configuration options.")
    file_ext: str = Field(
        default="xlsx", description="File extension of battery monitoring results."
    )
    edge_exclusion_mv: float | None = Field(
        default=50, description="Edge exclusion in mV."
    )

    electrode: Electrode = Field(
        default_factory=Electrode, description="Electrode attributes."
    )
    electrolyte: Electrolyte = Field(
        default_factory=Electrolyte, description="Electrolyte composition."
    )
    fit_unresolved: bool = Field(
        default=True,
        description="Fit a dataset whose reference cycle says its peaks are not resolvable.",
    )
    figure: Figure = Field(
        default_factory=Figure, description="Figure plotting options."
    )
    key_cycles: list[int] = Field(
        default=[1, 5, 10, 20, 30, 40, 50],
        description="Key cycles to summarise and plot.",
    )
    log_level: str = Field(default="info", description="Log level")
    output_dir: Path = Field(
        default=Path("./output/"),
        description="Path to save the output to, default is './output/'.",
    )
    max_cycle: int | None = Field(
        default=None,
        description="Cap the analysis at the given cycle, default is 'None' which includes all cycles.",
    )
    peak: Peak = Field(default_factory=Peak, description="Peak fitting options.")
    polyorder: int = Field(default=3, description="Polyorder.")
    power_energy_analysis: bool = Field(
        default=True, description="Whether to run power energy analysis."
    )
    qc_cycle: int = Field(
        default=3,
        description="Which cycle the preprocessing quality control figure shows raw against.",
    )
    rebin_width_mv: str | int | None = Field(
        default="auto", description="Rebin width, either 'auto' or an integer."
    )
    reference_cycle: int = Field(default=2, description="Reference cycle.")
    run_peak_fitting: bool = Field(
        default=True, description="Whether to fit the dQ/dV peaks."
    )
    second_smooth_window: str | int | None = Field(
        default="auto",
        description="Second order smoothing window, either 'auto' or an integer.",
    )
    separator_type: str | None = Field(default=None, description="Separator type.")
    shoulder_detection: bool = Field(
        default=True, description="Whether to enable shoulder detection."
    )
    smoothing_window: str | int | None = Field(
        default="auto", description="Smoothing window, either 'auto' or an integer."
    )
    spike: Spike = Field(
        default_factory=Spike, description="Spike configuration options."
    )
    stop_fitting_at_death: bool = Field(
        default=True,
        description="Stop fitting once the cell has stopped delivering charge.",
    )
    terminal_trim: TerminalTrim = Field(
        default_factory=TerminalTrim,
        description="Terminal trimming configuration options.",
    )
    termination: Termination = Field(
        default_factory=Termination, description="Termination configuration options."
    )
    test_start_date: str | None = Field(default=None, description="Test start date")

    theoretical_capacity: TheoreticalCapacity = Field(
        default_factory=TheoreticalCapacity,
        description="Theoretical capacity configuration options.",
    )
    theoretical_charge: TheoreticalCharge = Field(
        default_factory=TheoreticalCharge,
        description="Theoretical charge configuration options.",
    )
    total_electrode_coating_mg: float | None = Field(
        default=None, description="Total electrode coating in mg"
    )
    total_loading_mg_cm2: float | None = Field(
        default=None, description="Total loading in mg/cm2"
    )

    tracking_tolerance_mv: float = Field(
        default=80.0, description="Tracking tolerance in mV."
    )
    voltage: Voltage = Field(
        default_factory=Voltage, description="Voltage configuration options."
    )

    # Used to select the YAML source, not as an application setting.
    config_file: Path | None = Field(default=None, exclude=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize the sources used to populate settings.

        Parameters
        ----------
        settings_cls : type[BaseSettings]
            The settings class being configured.
        init_settings : PydanticBaseSettingsSource
            Source for values passed to the settings constructor.
        env_settings : PydanticBaseSettingsSource
            Source for environment-variable values.
        dotenv_settings : PydanticBaseSettingsSource
            Source for values loaded from a dotenv file.
        file_secret_settings : PydanticBaseSettingsSource
            Source for values loaded from secret files.

        Returns
        -------
        tuple[PydanticBaseSettingsSource, ...]
            Settings sources ordered from highest to lowest priority.
        """
        custom_file = init_settings.init_kwargs.get("config_file")
        yaml_source = (
            YamlConfigSettingsSource(settings_cls, yaml_file=custom_file)
            if custom_file is not None
            else DefaultYamlSource(settings_cls)
        )
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_source,
            file_secret_settings,
        )


class FileStructures(BaseSettings):
    """Data Dictionary for file structures."""

    # ns-rse 2026-11-06 : Set this to be a dataclass/BaseModel Neware as is done for Parameters which makes validating
    # values easy
    neware: dict[str, Any] = Field(default=None, description="Neware data dictionary.")
    # Used to select the YAML source, not as an application setting.
    config_file: Path | None = Field(
        default=DEFAULT_DATA_DICTIONARY, description="Path to the configuration file."
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize the sources used to populate settings.

        Parameters
        ----------
        settings_cls : type[BaseSettings]
            The settings class being configured.
        init_settings : PydanticBaseSettingsSource
            Source for values passed to the settings constructor.
        env_settings : PydanticBaseSettingsSource
            Source for environment-variable values.
        dotenv_settings : PydanticBaseSettingsSource
            Source for values loaded from a dotenv file.
        file_secret_settings : PydanticBaseSettingsSource
            Source for values loaded from secret files.

        Returns
        -------
        tuple[PydanticBaseSettingsSource, ...]
            Settings sources ordered from highest to lowest priority.
        """
        custom_file = init_settings.init_kwargs.get("config_file")
        yaml_source = (
            YamlConfigSettingsSource(settings_cls, yaml_file=custom_file)
            if custom_file is not None
            else DefaultYamlSource(settings_cls, yaml_file=DEFAULT_DATA_DICTIONARY)
        )
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_source,
            file_secret_settings,
        )


class Dataset(BaseModel):
    """
    One cell's data, in acquisition order, plus what is known about it.

    `df` is never re-sorted. Acquisition order is the only order in which
    "what happened next" is answerable, and three of this project's defects
    came from code that sorted by voltage and then asked a time-ordered
    question.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )
    name: Path | str = Field(description="Filename data is derived from.")
    df: pd.DataFrame = Field(description="Dataframe of battery charge cycle data.")
    meta: dict = Field(
        default={}, description="Metadata on the battery charging cycle."
    )
    electrode_type: str = Field(
        default="Positive",
        description="Electrode type, either 'Positive' or 'Negative'.",
    )
    labels_swapped: bool = Field(
        default=False, description="Whether electrode labels have been swapped or not."
    )
    source_path: str | Path = Field(description="Path to the original file.")
    source_sha256: str = Field(
        default="", description="SHA256 checksum of the original file."
    )
    active_loading_mg_cm2: float | None = Field(
        default=None, description="Active loading in milligrams/cm^2."
    )
    barcode: str | None = Field(
        default=None, description="Barcode derived from filename."
    )
    blend: list[float] = Field(default=[80.0, 10.0, 10.0], description="Blend.")
    charge_current_a_from_file: float | None = Field(
        default=None, description="Charge current from file."
    )
    charge_rate_c: float | None = Field(
        default=None, description="Charge rate current."
    )
    composition: str | None = Field(default=None, description="Battery composition.")
    counter_electrode: CounterElectrode = Field(
        default_factory=CounterElectrode, description="Counter electrode."
    )

    # -- structure ---------------------------------------------------------

    # 1.8.7 Module 1 drops rows with a missing Voltage, Cycle, Step or dQ/dV
    # BEFORE anything else, and every half-cycle it processes comes from that
    # cleaned frame. Reproducing it matters for more than row counts: where
    # several records share a voltage, `sort_values` breaks the tie by input
    # order, so a frame containing extra rows sorts its duplicates differently
    # and the smoothed curve diverges. On the P3 cells 15 of 40 half-cycles
    # differed for exactly this reason until the dropna was matched.
    _REQUIRED = ("Voltage", "Cycle", "Step", "dQ/dV")

    # BOTH OF THESE ARE CACHED, and the cache is the difference between a
    # run you wait two minutes for and one you do not.
    #
    # `half_cycle` used to boolean-mask the WHOLE frame on every call:
    # `f[(f["Cycle"] == cycle) & (f["Step"] == step)]`, 82,337 rows scanned
    # to return 200 of them. Under pandas 3 the `Step == "Charge"` half of
    # that is a string comparison that null-checks the entire column, and it
    # dominates: 10.2 ms per call, with `_isna_string_dtype` alone taking
    # 5.4 s of a 13.4 s `termination_check`.
    #
    # Cells 4 and 5 make 3,444 such calls per dataset — 42 s, or 126 s for a
    # triplicate. ONE `groupby(["Cycle", "Step"])` produces all 440 groups in
    # 0.03 s, which is 168x faster than 440 masks, and every later call is a
    # dict lookup.
    #
    # Safe because `frame` is never re-sorted (see the class docstring) and
    # nothing mutates a half-cycle in place: `orient_dqdv` copies before
    # writing and `strip_cv_hold` only slices. `replace()` builds a NEW
    # Dataset, so a modified frame gets a fresh cache rather than a stale one.
    # The groups preserve the frame's own row order, so the stable-sort
    # tie-breaking that the port fidelity depends on is unchanged.

    @property
    def clean(self) -> pd.DataFrame:
        """
        Return a subset of the dataframe.

        Returns
        -------
        pd.DataFrame
            Subset of the required dataframes.
        """
        cached = self.__dict__.get("_clean_cache")
        if cached is None:
            # ns-rse 2026-11-06 : Switch this to use the required/numeric columns from the default_dictionary.yaml
            # (stored in FileStructures class rather than hard coded value).
            cols = [c for c in self._REQUIRED if c in self.df.columns]  # pylint: disable=no-member
            cached = self.df.dropna(subset=cols)  # pylint: disable=no-member
            self.__dict__["_clean_cache"] = cached
        return cached

    def _half_cycles(self) -> dict[tuple[int, str], pd.DataFrame]:
        """
        Build a dictionary of subsets of the `self.df` dataframe indexed by `cycle` and `step`.

        Returns
        -------
        dict[tuple[int, str], pd.DataFrame]
            Dictionary takes the form `{(cycle, step): pd.DataFrame}`.
        """
        cached = self.__dict__.get("_half_cycle_cache")
        if cached is None:
            f = self.clean
            if not {"Cycle", "Step"} <= set(f.columns):
                cached = {}
            else:
                cached = {
                    (int(c), str(s)): d
                    for (c, s), d in f.groupby(["Cycle", "Step"], sort=True)
                }
            self.__dict__["_half_cycle_cache"] = cached
        return cached

    def half_cycle_keys(self) -> list[tuple[int, str]]:
        """
        Extract keys from half cycles.

        Returns
        -------
        list[tuple[int, str]]
            `(cycle, step)` pairs, in cycle order, Charge before Discharge.
        """
        return list(self._half_cycles)

    def half_cycle(self, cycle, step) -> pd.DataFrame:
        """
        Extract a half-cycle (unique `cycle`/`step` combination).

        Parameters
        ----------
        cycle : int
            Cycle to extract.
        step : str
            Step to extract.

        Returns
        -------
        pd.DataFrame
            Dataframe corresponding to the `cycle`/`step` requested.
        """
        # An absent key returns an EMPTY FRAME WITH THE RIGHT COLUMNS, which
        # is what the boolean mask returned and what every caller's
        # `len(df) < n` and `col in df.columns` tests expect.
        return self._half_cycles.get((int(cycle), str(step)), self.clean.iloc[:0])

    # -- the part that matters ---------------------------------------------

    def step_direction(self, cycle: int, step: str) -> float:
        """
        +1 if the voltage rises through this half-cycle, -1 if it falls.

        MEASURED, never inferred from the label. Falls back to the label only
        when the net travel is too small to read (a pure constant-voltage
        step), and says so by returning the label's implication rather than
        pretending to know.

        Parameters
        ----------
        cycle : int
            Cycle to extract step-direction for.
        step : str
            Step to extract the step-direction for.

        Returns
        -------
        float
            Voltage for the given cycle/step.
        """
        v = pd.to_numeric(
            self.half_cycle(cycle, step)["Voltage"], errors="coerce"
        ).to_numpy()
        v = v[np.isfinite(v)]
        if v.size >= 2:
            net = float(v[-1] - v[0])
            if abs(net) > 1e-4:  # one voltage LSB
                return 1.0 if net > 0 else -1.0
        # The label fallback must respect the swap. On a negative electrode
        # the half-cycle LABELLED Charge is the one where the voltage falls,
        # so reading the label naively returned the wrong sign — and this
        # branch fires precisely on a pure constant-voltage hold, the one
        # place the measurement cannot correct it.
        implied = 1.0 if str(step).lower().startswith("c") else -1.0
        return -implied if self.labels_swapped else implied

    def capacity(self, cycle: int, step: str) -> float:
        """
        Charge passed in this half-cycle, from the cycler's own counter.

        The counter, not the integral of dQ/dV — those two are compared by
        `quality.integral_fidelity`, and using one to define the other would
        make that comparison meaningless.

        Parameters
        ----------
        cycle : int
            Cycle to extract step-direction for.
        step : str
            Step to extract the step-direction for.

        Returns
        -------
        float
            Charge passed in the half-cycle for the given cycle/step.
        """
        col = (
            "Charge_Capacity"
            if str(step).lower().startswith("c")
            else "Discharge_Capacity"
        )
        d = self.half_cycle(cycle, step)
        if col not in d.columns:
            return float("nan")
        q = pd.to_numeric(d[col], errors="coerce").to_numpy()
        q = q[np.isfinite(q)]
        if q.size < 2:
            return float("nan")
        return float(np.nanmax(q) - np.nanmin(q))

    def record_spacing_mv(self, step: str | None = None) -> dict[str, float]:
        """
        Median and minimum non-zero voltage step between consecutive records.

        Reported because the rebinning decision should be keyed to this and is
        not: 1.8.x keys it to the profile class, so the P3 exports get no
        rebinning at all despite 4% of their consecutive records sitting at an
        identical voltage. Measured here, acted on in 2.0.

        Parameters
        ----------
        step : str, optional
            Step to extract data for.

        Returns
        -------
        dict[str, float]
            Dictionary of the `median_mV`, `min_mV` and `duplicate_fraction` for the given step.
        """
        dv, zeros, total = [], 0, 0
        for cyc, st in self.half_cycle_keys():
            if step is not None and st != step:
                continue
            v = pd.to_numeric(
                self.half_cycle(cyc, st)["Voltage"], errors="coerce"
            ).to_numpy()
            v = v[np.isfinite(v)]
            if v.size < 50:
                continue
            d = np.abs(np.diff(v))
            zeros += int((d == 0).sum())
            total += d.size
            nz = d[d > 0]
            if nz.size:
                dv.append(nz)
        if not dv:
            return {
                "median_mV": float("nan"),
                "min_mV": float("nan"),
                "duplicate_fraction": float("nan"),
            }
        allnz = np.concatenate(dv) * 1000.0
        return {
            "median_mV": float(np.median(allnz)),
            "min_mV": float(allnz.min()),
            "duplicate_fraction": (zeros / total) if total else float("nan"),
        }
