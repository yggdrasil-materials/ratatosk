"""
Reading a cycler export, and knowing what the columns mean.

The one thing this module exists to prevent
-------------------------------------------
For a negative electrode, 1.8.x swaps the Charge/Discharge labels so that
"Discharge" means the useful half-cycle for every dataset. That is right for
presentation and it is a trap for physics: in an LTO dataset the half-cycle
labelled "Charge" is the one where the **voltage falls**.

That trap was sprung. A cycle-integrity measure inferred the expected voltage
direction from the label, and consequently reported all 64 LTO half-cycles as
anomalous at 98-99% coulombic efficiency.

So `Dataset` keeps the swap for labelling, and additionally offers
`step_direction()`, which is **measured from the recorded voltage** and cannot
be fooled. Everything in `signal` and `analyse` uses the measured direction.
The label is for humans; the measurement is for arithmetic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

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


@dataclass
class Dataset:
    """
    One cell's data, in acquisition order, plus what is known about it.

    `frame` is never re-sorted. Acquisition order is the only order in which
    "what happened next" is answerable, and three of this project's defects
    came from code that sorted by voltage and then asked a time-ordered
    question.
    """

    name: str
    frame: pd.DataFrame
    meta: dict = field(default_factory=dict)
    electrode_type: str = "Positive"
    labels_swapped: bool = False
    source_path: str = ""
    source_sha256: str = ""

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
        cached = self.__dict__.get("_clean_cache")
        if cached is None:
            cols = [c for c in self._REQUIRED if c in self.frame.columns]
            cached = self.frame.dropna(subset=cols)
            self.__dict__["_clean_cache"] = cached
        return cached

    @property
    def _half_cycles(self) -> dict:
        """`{(cycle, step): frame}` for the whole dataset, built once."""
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

    def half_cycle_keys(self):
        """(cycle, step) pairs, in cycle order, Charge before Discharge."""
        return list(self._half_cycles)

    def half_cycle(self, cycle, step) -> pd.DataFrame:
        # An absent key returns an EMPTY FRAME WITH THE RIGHT COLUMNS, which
        # is what the boolean mask returned and what every caller's
        # `len(df) < n` and `col in df.columns` tests expect.
        return self._half_cycles.get((int(cycle), str(step)), self.clean.iloc[:0])

    # -- the part that matters ---------------------------------------------

    def step_direction(self, cycle, step) -> float:
        """
        +1 if the voltage rises through this half-cycle, -1 if it falls.

        MEASURED, never inferred from the label. Falls back to the label only
        when the net travel is too small to read (a pure constant-voltage
        step), and says so by returning the label's implication rather than
        pretending to know.
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

    def capacity(self, cycle, step) -> float:
        """
        Charge passed in this half-cycle, from the cycler's own counter.

        The counter, not the integral of dQ/dV — those two are compared by
        `quality.integral_fidelity`, and using one to define the other would
        make that comparison meaningless.
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

    def record_spacing_mv(self, step=None):
        """
        Median and minimum non-zero voltage step between consecutive records.

        Reported because the rebinning decision should be keyed to this and is
        not: 1.8.x keys it to the profile class, so the P3 exports get no
        rebinning at all despite 4% of their consecutive records sitting at an
        identical voltage. Measured here, acted on in 2.0.
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


def apply_electrode_convention(dataset, electrode_type):
    """
    Return a copy of `dataset` with the convention for `electrode_type` applied.

    1.8.7 loaded in Cell 2, asked for the electrode type in Cell 3, and applied
    the convention in Cell 4 — in that order, because the operator has to be
    asked before the answer can be used. `read_neware` can apply it directly
    when the type is already known; this is for the interactive path, where it
    is not.

    Applying it twice is a no-op ONLY for the same answer. Re-applying
    "Negative" to an already-swapped dataset returns it unchanged; re-stamping
    it as "Positive" raises, because that would leave the Step labels,
    capacity columns and dQ/dV sign inverted while the record claimed
    otherwise — silently mirrored data with nothing to notice it by.
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
    df, swapped = _apply_electrode_convention(dataset.frame, electrode_type)
    return replace(
        dataset,
        frame=df.reset_index(drop=True),
        electrode_type=str(electrode_type),
        labels_swapped=swapped,
    )


def truncate_cycles(dataset, max_cycle):
    """
    Cut a dataset at `max_cycle`, returning a new one. None leaves it alone.

    Truncation belongs HERE, to the Dataset, because the Dataset is what
    every later stage reads. 1.8.6 truncated a separate `electrochemical_data`
    dict inside the parameter collection, and 1.9.0's notebook then rebuilt
    that dict from `dataset.frame` two lines later — so the operator's answer
    to "Maximum cycle to analyse" was applied, discarded, and then asserted as
    fact by the console message, `dataset_info.txt` and the run manifest,
    while every capacity, retention and fade number was computed over the
    cycles they had asked to exclude.
    """
    if not max_cycle:
        return dataset
    f = dataset.frame
    if "Cycle" not in f.columns:
        return dataset
    keep = pd.to_numeric(f["Cycle"], errors="coerce") <= float(max_cycle)
    return replace(dataset, frame=f[keep].reset_index(drop=True))


def open_xlsx(path: str | Path) -> dict[str, pd.DataFrame]:
    """
    Open a `.xls[x]` spreadsheet and read all sheets.

    Raises rather than returning None. A workbook that cannot be opened at
    all is a different failure from one that has no 'test' sheet, and the
    two were indistinguishable while both were caught by the same
    `except Exception` inside the metadata reader.

    Parameters
    ----------
    path : pd.Excelfile, str
        Returns a class for parsing the given Excel file and the engine name.

    Returns
    -------

    dict[str, pd.DataFrame]
    `(ExcelFile, engine)`. The caller is responsible for closing it;
    `Dataset` holds no reference to it, so it can be closed as soon as both
    sheets have been read.

    """
    try:
        with pd.ExcelFile(path, engine="calamine") as xls:
            return {sheet: pd.read_excel(xls, sheet) for sheet in xls.sheet_names}
    except OSError as e:
        error_msg = f"Could not open {path!s} as an Excel workbook."
        raise OSError(error_msg) from e


def read_neware(
    sheets: dict[str, pd.DataFrame] | None = None,
    sheet: str = "record",
    path: str | Path | None = None,
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
        A dictionary of sheets from `io.open_xlsx`. If `None` then
    sheet : str
        The sheet to extract, defaults to "record".
    path: str | Path,
        Path to `xlsx` file.
    name: str | None = None,
        Name of the file.
    electrode_type: str = "Positive",
        Electrode type, options are `Positive` (default) and `Negative`.
    meta: str | None = None,
        ???
    columns : dict, optionall
        Dictionary for renaming columns. If `None` a default is used.

    Returns
    """
    # name = name or os.path.splitext(os.path.basename(path))[0]
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
    for c in _NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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
        frame=df.reset_index(drop=True),
        meta=dict(meta or {}),
        electrode_type=electrode_type,
        labels_swapped=swapped,
        source_path=str(path),
        source_sha256=file_sha256(path),
    )
