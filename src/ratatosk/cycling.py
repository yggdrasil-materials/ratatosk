"""
Capacity, retention, fade, energy, power, voltage, efficiency and rate.

Everything here is 1.8.7's Cells 5 to 13, ported without editing the bodies.
Those cells were top-level scripts driving straight off `electrochemical_data`
and `user_parameters`, so the port was mechanical: each cell's statements were
wrapped in a function whose parameters are exactly the globals it read, its
imports dropped, and its constants hoisted to module scope. Nothing inside a
body was rewritten, because the figures must not move.

The notebook cell each one is called from is 1.9.0's, not 1.8.7's — the
column on the right was the 1.8.7 number and stayed put for three releases
while the console printed "Cell 9b complete" at a reader sitting in Cell 6.

    cycling_summary             6a   the per-cycle table, the headline
                                     metrics, and the incomplete-cycle flags
                                     every later plot reads
    voltage_profiles            6b   every cycle, V against capacity
    voltage_profiles_key        6b   the same overlay, key cycles only
    cycle_life                  6b   capacity and CE against cycle
    comparative_capacity        6b   every dataset on one axis
    capacity_retention          6c   retention against a reference cycle
    fade_rate                   6c   loss per cycle, and cumulative
    power_and_energy            6c   three normalisations, and the Ragone
    average_discharge_voltage   6c   mean discharge voltage against cycle
    energy_efficiency           6c   round-trip
    rate_capability             6c   capacity grouped by C-rate

`cycling_summary` comes first for a reason: it returns `all_cycle_tables`, and
every plot after it reads that to exclude incomplete cycles. Pass it through.
Passing None reproduces the not-yet-run branch exactly, including its notice.

Two constants are NOT hoisted
-----------------------------
`MARKER_SIZE` is 7 in Cells 7-9 and 6 in Cells 9b, 11 and 12; `YAXIS_MIN` is 0
in Cell 9 and 40 in Cell 12; `YAXIS_MAX` and `VISUAL_OUTLIER_THRESHOLD` differ
similarly. They stay local to the functions that define them. Hoisting any of
them would have quietly changed a figure — which is the failure mode this
whole port exists to avoid.

What this module does not do
----------------------------
It does not judge. Capacity, retention and fade are arithmetic on the cycler's
own counters; whether a given cycle's numbers mean anything is
`analyse.integrity`'s question, and the two are deliberately kept apart.
"""

from __future__ import annotations

import contextlib
import functools
import io as _io
import os
import re

import numpy as np

from .compat import trapezoid
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors            # noqa: F401  (used by ported code)
import matplotlib.ticker as mticker            # noqa: F401  (used by ported code)

try:                                            # Cell 6 only
    import seaborn as sns
except ImportError:                             # pragma: no cover
    sns = None

from .plots import (_get_display_name, _force_integer_cycles,
                    _charge_label, _discharge_label)
# Imported as a MODULE, not as values. `from .plots import
# figure_width_inches` binds the number at import time, so
# `plots.set_figure_size()` moved the dQ/dV figures and left every cycling
# figure at the old size — and the flattened notebook, having one namespace,
# did the opposite. Two builds, two different sets of figures.
from .style import (rule, heading, section, entry, verdict, bullet,
                    image_format, saved, nice_axis_limit)
from . import plots as _plots
# NOT ALIASED. `flatten.strip()` deletes every relative import, so in the flat
# notebook the name has to be the one the defining module used: an aliased
# import leaves the alias defined nowhere, and the build's free-name check
# does not catch it because the use is inside a function body. Same family as
# the late-relative-import bug the build now refuses outright.
from .detect import formation_end

__all__ = ["voltage_profiles", "cycling_summary", "voltage_profiles_key",
           "cycle_life", "comparative_capacity", "capacity_retention",
           "fade_rate", "power_and_energy", "average_discharge_voltage",
           "energy_efficiency", "rate_capability",
           "rate_protocol", "annotate_rate_protocol",
           "rate_recovery", "describe_rate", "format_crate",
           "snap_crate",
           "rate_phrase"]

# Okabe-Ito, colourblind-safe. IMPORTED, not restated — `plots` is the one
# home for the run's categorical palette, the same arrangement as the
# charge/discharge colours further down. Two copies of a colour convention is
# how the charge colour came to mean two different things in one run.
MULTI_DATASET_PALETTE = _plots.CATEGORICAL_PALETTE
MULTI_DATASET_MARKERS = _plots.CATEGORICAL_MARKERS


# =============================================================================
# PORTED FROM 1.8.7 CELLS 5-13 — do not edit; the figures are the regression
# =============================================================================

LABEL_MODE = 'auto'
COMMON_XAXIS_SCALE = True
XAXIS_PADDING_FRACTION = 0.08
AUTO_OVERLAP_THRESHOLD = 0.5
RETENTION_REFERENCE_CYCLE = 2
# THE FADE RATE IS DOCUMENTED AS EXCLUDING FORMATION AND WAS EXCLUDING CYCLE 1.
#
# `FADE_RATE_START_CYCLE = 2` is a guess about where formation ends, printed
# beside a line that says "excluding formation" — while the same run measures
# formation ending at cycles 4, 5 and 4 on the LTO triplicate
# (`detect.formation_end`, from the coulombic efficiency the same table just
# computed). The reported fade rate was **17-22% steeper** than it would be
# from the measured end, consistently across all three cells, because the
# cycles between the constant and the measurement are the steepest in the run.
#
# So the start cycle is now MEASURED per dataset and the constant below is
# only the fallback, used when the record is too short to establish formation
# at all. The cycle used and the reason are printed, because a fade rate is
# meaningless without saying what it excluded.
FADE_RATE_FROM_MEASURED_FORMATION = True
FADE_RATE_START_CYCLE = 2
OVERRIDE_KEY_CYCLES = None
# Imported from plots, not restated. These were amber-charge / blue-discharge
# here and blue-charge / vermillion-discharge there, so one run showed blue
# meaning two different things. See plots.COLOUR_CHARGE for the convention.
COLOUR_CHARGE = _plots.COLOUR_CHARGE
COLOUR_DISCHARGE = _plots.COLOUR_DISCHARGE
COLOUR_EFFICIENCY = '#000000'  # black
MARKER_CHARGE = 's'       # square
MARKER_DISCHARGE = 'o'    # circle
MARKER_EFFICIENCY = 'v'   # triangle down
COMMON_YAXIS_CAPACITY = True
# Hold the efficiency panel to a common floor and ceiling (40-105%, widened
# to fit the data) rather than letting it autoscale per cell. Read in
# `energy_efficiency`; for years it was read nowhere.
COMMON_YAXIS_EFFICIENCY = True
CAPACITY_YAXIS_PADDING = 0.08
EFFICIENCY_YMAX = 110  # CE y-axis upper limit (%)
EXCLUDE_INCOMPLETE = True
CE_START_CYCLE = 2
MARKERS = MULTI_DATASET_MARKERS
YAXIS_PADDING = 0.08
VISUAL_OUTLIER_FILTER = True
FADE_REFERENCE_CYCLE = RETENTION_REFERENCE_CYCLE  # from Cell 9; change there to affect both
ROLLING_WINDOW = 100
VOLTAGE_YMIN = None  # e.g., 2.8
VOLTAGE_YMAX = None  # e.g., 3.8
VOLTAGE_YAXIS_PADDING = 0.02  # V above/below data range
DRIFT_REFERENCE_CYCLE = 2
MARKERS_EE = 'D'   # diamond for energy efficiency
MARKERS_CE = 'o'   # circle for coulombic efficiency
COLOUR_EE = '#D55E00'   # vermillion
COLOUR_CE = '#0072B2'   # blue
EE_START_CYCLE = 2
CURRENT_GROUPING_TOLERANCE = 0.05
MIN_CYCLES_PER_RATE = 2
# How many clean cycles the reference and return blocks each need before a
# recovery is reported at all. Two is the fewest that can carry a mean.
RATE_RECOVERY_MIN_REFERENCE = 2

# THE LABEL SNAPS; THE NUMBER DOES NOT. A measured C-rate is
# `current / (mass x theoretical capacity)`, so it inherits whatever
# theoretical capacity the operator entered. Jiaqi set her currents for C/20
# against a capacity 5% away from the 168 mAh/g entered here, and the
# measured rate came out at C/19.0 — which then appeared as "C/19" on every
# figure and read as a mistake.
#
# So the label is snapped to the nearest conventional rate when the measured
# value is within RATE_LABEL_SNAP of it, and the measured value is exported
# beside it in `c_rate_measured`. Nothing downstream computes on the label.
# The conventional set is spaced far enough apart (C/20 to C/10 is a factor
# of two) that a 10% window cannot merge two rates an experimenter meant to
# distinguish.
RATE_LABEL_SNAP = 0.10
_CONVENTIONAL_RATES = (1/100., 1/50., 1/20., 1/10., 1/5., 1/4., 1/3., 1/2.,
                       1.0, 2.0, 3.0, 5.0, 10.0, 20.0)

BAR_COLOUR = '#0072B2'
RECOVERY_COLOUR = '#009E73'


# One definition of "this cycle's capacity is implausibly low for a figure".
# Six functions used to restate it as a local constant, quote that local in
# the console message, and then call the filter WITHOUT passing it — so
# editing the documented knob changed the sentence and not the data.
VISUAL_OUTLIER_FRACTION = 0.30


# =============================================================================
# THE CANONICAL PER-CYCLE QUANTITIES
# =============================================================================
# One definition each, stated here, used everywhere. Before 1.9.1 delivered
# capacity had FIVE implementations with different admission rules, coulombic
# efficiency had three (two returning 0% where the third returned NaN), and
# average discharge voltage had two names and a third hardcoded guess. Nothing
# compared them, so the number in a figure and the number in the CSV could
# differ with nothing in the run to say so.
#
#   delivered capacity   dQ across the half-cycle from the CYCLER'S OWN
#                        counter (max - min). Never the integral of dQ/dV:
#                        those two are compared by `quality.integral_fidelity`
#                        and using one to define the other would make that
#                        comparison meaningless.
#
#   coulombic efficiency Q_discharge / Q_charge for the SAME cycle, as a
#                        percentage. NaN — never 0 — when either half is
#                        absent. CE is read at the fourth decimal place
#                        (~99.96% is needed for 500 cycles), so a spurious 0%
#                        is not a rounding error: it asserts that the cell
#                        passed charge and returned none, which is a different
#                        physical claim from "we could not measure it".
#
#   average discharge V  the ENERGY-WEIGHTED mean, integral(V dQ) / dQ. This
#                        is the standard definition. A previous fallback used
#                        `voltage_upper_V * 0.85`, a hardcoded guess presented
#                        as a measurement; it is gone.
#
#   capacity retention   Q_discharge(n) / Q_discharge(n_ref), n_ref stated in
#                        the column name — see `RETENTION_REFERENCE_CYCLE`.


def _caption_window(params, df=None):
    """The window a CAPTION should quote: what the cell was cycled between.

    `df['Voltage'].min()/.max()` is the extent of the RECORD, which includes
    an open-circuit start and any excursion the cycler made — so the captions
    quoted 1.20-2.57 V for cell B and 1.20-2.68 V for the triplicate together,
    all cycled 1.2-2.5 V. Same defect as `plots._protocol_window` was written
    for, in a second module: a wrong experimental detail in text written to be
    pasted into a manuscript, and one that differed between nominally
    identical cells.

    Cell 3b measured the protocol window (the median of the per-cycle limits)
    and put it in `user_parameters`; this prefers it and falls back to the
    record only when it is absent.
    """
    lo = (params or {}).get('voltage_lower_V')
    hi = (params or {}).get('voltage_upper_V')
    try:
        lo, hi = float(lo), float(hi)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return lo, hi
    except (TypeError, ValueError):
        pass
    if df is not None and 'Voltage' in getattr(df, 'columns', ()):
        v = pd.to_numeric(df['Voltage'], errors='coerce')
        return float(v.min()), float(v.max())
    return float('nan'), float('nan')


def _caption_window_all(user_parameters, fallback=None):
    """The window for a caption spanning EVERY dataset.

    Where the cells agree — a triplicate normally does — that one window is
    quoted. Where they genuinely differ, the widest is, because the caption is
    then describing more than one protocol and should say so honestly rather
    than pick one.
    """
    los, his = [], []
    for p in (user_parameters or {}).values():
        lo, hi = _caption_window(p)
        if np.isfinite(lo) and np.isfinite(hi):
            los.append(lo); his.append(hi)
    if los:
        return min(los), max(his)
    return fallback if fallback else (float('nan'), float('nan'))


def _has_tables(all_cycle_tables):
    """Did the caller hand us `cycling_summary`'s output?

    Was `'all_cycle_tables' in dir() and isinstance(...)`, in eleven copies.
    Inside a function `dir()` lists the LOCALS, and `all_cycle_tables` is a
    parameter — so the first clause was always True and only the isinstance
    ever decided anything. It is a 1.8.7 idiom: there this was a notebook
    global and asking whether it existed was a real question.
    """
    return isinstance(all_cycle_tables, dict)



def _run_image_format(user_parameters):
    """The format for a figure that spans every dataset.

    A run-wide figure has no single dataset to ask, so the datasets have to
    agree. Where they do not, PNG — the safe one to open — and the mismatch
    is the operator's to resolve.
    """
    fmts = {image_format(p) for p in (user_parameters or {}).values()}
    return fmts.pop() if len(fmts) == 1 else 'png'


def _get_endpoints(df_charge, df_discharge, unique_cycles, key_cycles):
    """
    Discharge and charge endpoint coordinates for the key cycles.

    ONE implementation. It was defined twice, nested inside two figure
    functions — byte-identical today, which is the dangerous state: the next
    edit lands in one of them and two figures start labelling their endpoints
    by different rules with nothing in the run to say so.
    """
    endpoints = []
    for cycle in key_cycles:
        if cycle not in unique_cycles:
            continue
        dc = df_discharge[df_discharge['Cycle'] == cycle]
        if not dc.empty:
            max_row = dc.loc[dc['Discharge_Capacity'].idxmax()]
            endpoints.append({
                'cycle': cycle, 'step': 'Discharge',
                'capacity': max_row['Discharge_Capacity'],
                'voltage': max_row['Voltage']
            })
        cc = df_charge[df_charge['Cycle'] == cycle]
        if not cc.empty:
            max_row = cc.loc[cc['Charge_Capacity'].idxmax()]
            endpoints.append({
                'cycle': cycle, 'step': 'Charge',
                'capacity': max_row['Charge_Capacity'],
                'voltage': max_row['Voltage']
            })
    return endpoints


def delivered_capacity(frame):
    """
    Per-cycle delivered capacity, both half-cycles, in mAh/g.

    Returns a frame indexed 0..n with columns Cycle, Charge_mAh_g,
    Discharge_mAh_g. Rows carrying neither capacity column are dropped;
    rows carrying one are kept, because some cycler firmware fills only the
    active column.
    """
    need = ("Cycle", "Step", "Charge_Capacity", "Discharge_Capacity")
    if not all(c in frame.columns for c in need):
        return pd.DataFrame(columns=["Cycle", "Charge_mAh_g",
                                     "Discharge_mAh_g"])
    d = frame.dropna(subset=["Cycle", "Step"])
    d = d[d["Charge_Capacity"].notna() | d["Discharge_Capacity"].notna()]
    dch = d[d["Step"] == "Discharge"].groupby("Cycle")["Discharge_Capacity"]
    chg = d[d["Step"] == "Charge"].groupby("Cycle")["Charge_Capacity"]
    out = pd.DataFrame({
        "Discharge_mAh_g": dch.max() - dch.min(),
        "Charge_mAh_g": chg.max() - chg.min(),
    })
    out.index.name = "Cycle"
    out = out.reset_index()
    if len(out):
        out["Cycle"] = pd.to_numeric(out["Cycle"],
                                     errors="coerce").astype("Int64")
        out = out.dropna(subset=["Cycle"])
        out["Cycle"] = out["Cycle"].astype(int)
    return out


def coulombic_efficiency(discharge, charge):
    """Q_discharge / Q_charge as a percentage. NaN, never 0. See above."""
    d = pd.to_numeric(pd.Series(discharge), errors="coerce")
    c = pd.to_numeric(pd.Series(charge), errors="coerce")
    ok = d.notna() & c.notna() & (c > 0) & (d > 0)
    return (d / c * 100.0).where(ok)


def mean_discharge_voltage(frame):
    """
    Energy-weighted mean discharge voltage per cycle: integral(V dQ) / dQ.

    Returns Cycle, Avg_Discharge_Voltage_V. One name, one definition — the
    same quantity was previously exported as `Avg_Voltage_V` by one function
    and `Avg_V_Discharge` by another. Named `mean_...` because
    `average_discharge_voltage` is the FIGURE function further down; two
    definitions of one name in a flat namespace is how the last collision
    got shipped.
    """
    need = ("Cycle", "Step", "Voltage", "Discharge_Capacity")
    if not all(c in frame.columns for c in need):
        return pd.DataFrame(columns=["Cycle", "Avg_Discharge_Voltage_V"])
    rows = []
    d = frame[frame["Step"] == "Discharge"]
    for cyc, g in d.groupby("Cycle"):
        q = pd.to_numeric(g["Discharge_Capacity"], errors="coerce")
        v = pd.to_numeric(g["Voltage"], errors="coerce")
        m = q.notna() & v.notna()
        q, v = q[m].to_numpy(), v[m].to_numpy()
        if q.size < 3:
            continue
        o = np.argsort(q, kind="mergesort")
        q, v = q[o], v[o]
        span = float(q[-1] - q[0])
        if not np.isfinite(span) or span <= 0:
            continue
        rows.append({"Cycle": int(cyc),
                     "Avg_Discharge_Voltage_V": float(trapezoid(v, q) / span)})
    return pd.DataFrame(rows)


def _honours_verbose(fn):
    """Make `verbose=False` actually silence a cycling function.

    All eleven took the argument and none read it — 227 unconditional
    `print()` calls between them — so `cycling_summary(verbose=False)` printed
    its banner anyway and a scheduled run had no way to quieten the log.
    Rather than thread a flag through 227 call sites, the output is captured
    at the boundary, which is exactly the promise the signature was making.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kw):
        if kw.get("verbose", True):
            return fn(*args, **kw)
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            return fn(*args, **kw)
    return wrapper


def rate_block_labels(cycles, params):
    """
    Per-cycle rate-block index for a variable-rate run, or None.

    `None` for a single-rate run — there is one population and the caller
    should not partition it. On a variable-rate run each cycle gets the index
    of the block it belongs to, and a rate-TRANSITION cycle (in no block) gets
    its own label, so it is never pooled with either neighbour.
    """
    proto = ((params or {}).get("rate_protocol") or {})
    if not proto.get("is_variable"):
        return None
    blocks = proto.get("blocks") or []
    if not blocks:
        return None
    out = []
    for c in cycles:
        try:
            ci = int(c)
        except (TypeError, ValueError):
            out.append(("bad", c))
            continue
        lab = None
        for i, b in enumerate(blocks):
            if int(b["first_cycle"]) <= ci <= int(b["last_cycle"]):
                lab = i
                break
        out.append(lab if lab is not None else ("transition", ci))
    return out


def _flag_visual_outliers(caps, threshold=VISUAL_OUTLIER_FRACTION,
                          groups=None):
    """
    Flag cycles whose delivered capacity is <threshold of the running
    median of previous non-outlier, non-NaN cycles AT THE SAME RATE.

    Purpose: catch protocol anomalies (cell tripped, early cutoff,
    datalogging glitch) that delivered a small but non-zero discharge,
    so Cell 5b's protocol-level Incomplete flag doesn't catch them.

    Why 30% rather than the old 80%: Bug 1 (v1.7.1) showed that a
    running-median-based filter at 80% falsely flags healthy
    degradation as incomplete. 30% only triggers on cliff-drops — a
    cycle delivering less than a third of recent typical capacity is
    almost certainly an anomaly, not gradual fade.

    WHY `groups`. The running median was taken over every previous valid cycle
    in the record, and on a rate ladder that is a comparison across rates. A
    protocol of C/10, C/5, C/2, 1C, 2C, then 10C, then back to C/10 — the
    standard rate-capability test — reaches its 10C block with a running median
    near the low-rate capacity: at 200 mAh/g of accumulated median, the cut is
    60 mAh/g, and a perfectly healthy 10C discharge of 40 mAh/g is deleted as a
    datalogging glitch. Worse, a flagged cycle is not appended to `prev_valid`
    (the `continue` below), so the median never comes down and EVERY cycle of
    the high-rate block goes with it. Retention, fade rate, cycle life, mean
    discharge voltage, energy efficiency and the areal capacity figure are all
    drawn after this filter, so the whole fast half of a rate test could vanish
    from every one of them, announced only as "excluded N visual outlier(s)".

    A cycle is an outlier relative to cycles at ITS OWN rate. `groups` is one
    label per entry — `rate_block_labels` builds it — and the running median is
    kept per label. `groups=None` is the single-rate case and behaves exactly
    as before. A rate-transition cycle gets a label of its own and so never
    reaches the three-cycle minimum, which is the right answer: a cycle
    straddling two rates has no population to be an outlier in.

    Minimum of 3 prior valid cycles required before the filter kicks
    in (avoids flagging early formation cycles).

    Used by Cells 8, 9, 9b, 10, 11, 12 to clean plots.
    Defined once here to avoid divergent copies.
    """
    flags = [False] * len(caps)
    if groups is None:
        prev_valid = []
        for i, cap in enumerate(caps):
            if pd.isna(cap):
                flags[i] = True
                continue
            if len(prev_valid) >= 3:
                if cap < threshold * np.median(prev_valid):
                    flags[i] = True
                    continue
            prev_valid.append(cap)
        return flags

    # GROUPED: LEAVE-ONE-OUT WITHIN THE BLOCK, NOT A RUNNING MEDIAN.
    # A rate block in the standard test is five cycles long, so a running
    # median over PRECEDING cycles only reaches the three-cycle minimum at the
    # fourth — the filter would be blind to a glitch in the first three cycles
    # of every block, which is most of the record. Within one block the rate is
    # constant and the fade across five cycles is a fraction of a percent, so
    # the block's own median is a better reference than a running one and it is
    # available to every member. Each cycle is compared with the median of the
    # OTHER valid cycles in its block, so a glitch cannot raise the bar that
    # judges it. Blocks with fewer than four valid cycles are left alone: there
    # is no population there to be an outlier in, and a rate-transition cycle
    # (its own label) is always in that case.
    idx_by_group = {}
    for i, cap in enumerate(caps):
        if pd.isna(cap):
            flags[i] = True
            continue
        g = groups[i] if i < len(groups) else None
        idx_by_group.setdefault(g, []).append(i)
    for g, idxs in idx_by_group.items():
        if len(idxs) < 4:
            continue
        vals = np.array([float(caps[i]) for i in idxs])
        for j, i in enumerate(idxs):
            ref = np.median(np.delete(vals, j))
            if np.isfinite(ref) and vals[j] < threshold * ref:
                flags[i] = True
    return flags


@_honours_verbose
def voltage_profiles(electrochemical_data, user_parameters, *, save_location=None,
        all_cycle_tables=None, file_format='png', verbose=True):
    """
    Voltage vs specific capacity, every cycle.

    Ported from 1.8.7 Cell 5, wrapped in a function without editing
    the body. `all_cycle_tables` is `cycling_summary`'s output where the
    original read a notebook global; passing None reproduces the
    not-yet-run branch exactly.
    """

    def _check_overlap(endpoints, cap_tol_frac=0.03, volt_tol=0.05):
        """
        Check what fraction of endpoints would overlap.

        Returns fraction (0–1) of endpoints that are within tolerance
        of at least one other endpoint.
        """
        if len(endpoints) < 2:
            return 0.0

        caps = [e['capacity'] for e in endpoints]
        cap_range = max(caps) - min(caps) if len(caps) > 1 else max(caps)
        cap_tol = max(cap_range * cap_tol_frac, 2.0)

        n_overlapping = 0
        for i, ep in enumerate(endpoints):
            for j, other in enumerate(endpoints):
                if i == j:
                    continue
                if (abs(ep['capacity'] - other['capacity']) < cap_tol and
                    abs(ep['voltage'] - other['voltage']) < volt_tol):
                    n_overlapping += 1
                    break  # only count each endpoint once

        return n_overlapping / len(endpoints)

    def _add_annotations(ax, endpoints, unique_cycles, palette):
        """Add cycle number annotations at endpoints (no vertical offset)."""
        if not endpoints:
            return

        # Group endpoints by proximity to determine horizontal stagger
        caps = [e['capacity'] for e in endpoints]
        cap_range = max(caps) - min(caps) if len(caps) > 1 else max(caps)
        cap_tol = max(cap_range * 0.03, 2.0)
        volt_tol = 0.05

        placed = []

        # Process in reverse cycle order (later cycles get priority placement)
        sorted_endpoints = sorted(endpoints, key=lambda e: e['cycle'],
                                   reverse=True)

        for ep in sorted_endpoints:
            cycle_idx = list(unique_cycles).index(ep['cycle'])
            colour = palette[cycle_idx]

            # Determine offset
            x_offset = 10
            y_offset = 0

            # Stagger if this endpoint is close to an already-placed one
            for px, py in placed:
                if (abs(ep['capacity'] - px) < cap_tol and
                    abs(ep['voltage'] - py) < volt_tol):
                    # Shift further right
                    x_offset += 12
                    break

            # Discharge endpoints: label to the right
            # Charge endpoints: label to the left
            if ep['step'] == 'Charge':
                x_offset = -x_offset

            # Adjust vertical alignment based on position
            va = 'center'

            ax.annotate(
                str(int(ep['cycle'])),
                xy=(ep['capacity'], ep['voltage']),
                xytext=(x_offset, y_offset),
                textcoords='offset points',
                ha='center', va=va,
                fontsize=8, fontweight='bold',
                color=colour, alpha=0.9
            )
            placed.append((ep['capacity'], ep['voltage']))

    def _add_colourbar(fig, ax, unique_cycles, colour_palette):
        """Add a colourbar mapping cycle number to colour."""
        cmap = plt.get_cmap(colour_palette)
        norm = mcolors.Normalize(vmin=min(unique_cycles),
                                  vmax=max(unique_cycles))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
        cbar.set_label('Cycle number', fontsize=13)
        cbar.ax.tick_params(labelsize=11)

    # The two nested copies of `_discharge_label`/`_charge_label` that used
    # to sit here are gone. This module imports both from `.plots` at the
    # top and these SHADOWED that import for the whole function, so a fix
    # applied to the real ones reached every figure except the waterfalls —
    # which is exactly where 'Lithiation' over a sodium cell was seen. The
    # build's import-shadowing check looks for a module-level definition
    # hidden by an import, not for a function-local one hiding it.

    global_max_capacity = 0

    if COMMON_XAXIS_SCALE:
        for name, df in electrochemical_data.items():
            for col in ['Charge_Capacity', 'Discharge_Capacity']:
                if col in df.columns:
                    col_max = pd.to_numeric(df[col], errors='coerce').max()
                    if pd.notna(col_max):
                        global_max_capacity = max(global_max_capacity, col_max)

        # ROUND NUMBER, not the data's own maximum. The measured maximum
        # plus 8% padding landed the shared axis on 143 mAh/g, and a reader
        # comparing three cells across a frame that ends at 143 has to work
        # out whether the curve stopping at 136 means anything. It ends at
        # 150 now, with a gridline every 25.
        global_max_capacity, global_xaxis_step = nice_axis_limit(
            global_max_capacity)

        if global_max_capacity > 0:
            print(entry("common x-axis", f"{global_max_capacity:.0f} mAh/g",
                        f"ticks every {global_xaxis_step:g}"
                        if global_xaxis_step else ""))

    for name, df in electrochemical_data.items():
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        print(rule(composition))
        charge_rate_c = params.get('charge_rate_c', 'Unknown')
        colour_palette = params.get('colour_palette', 'viridis_r')
        key_cycles = params.get('key_cycles', [])
        is_anode = params.get('anode_labels_swapped', False)
        dch_label = _discharge_label(params)
        chg_label = _charge_label(params)

        df_work = df.copy()
        # Robust dropna (v1.8): require Voltage, Cycle, Step, and at
        # least one capacity column. Some cycler firmware versions fill
        # only the active capacity column per row.
        df_cleaned = df_work.dropna(subset=['Voltage', 'Cycle', 'Step'])
        df_cleaned = df_cleaned[
            df_cleaned['Charge_Capacity'].notna() |
            df_cleaned['Discharge_Capacity'].notna()
        ].copy()

        if df_cleaned.empty:
            print(verdict("caution", f"no valid data for {name}"))
            continue

        df_discharge = df_cleaned[df_cleaned['Step'] == 'Discharge']
        df_charge = df_cleaned[df_cleaned['Step'] == 'Charge']

        unique_cycles = sorted(df_cleaned['Cycle'].unique())
        # `voltage_profiles` accepts `all_cycle_tables` and, until 1.9.0.4,
        # never read it — so the half-cycle the export was taken during was
        # drawn as a profile that stops in mid-air, and counted in the
        # caption. Dropped here rather than dimmed: a voltage profile that
        # ends nowhere reads as a cell that failed.
        _skip = _unusable_cycles(all_cycle_tables.get(name)) \
            if isinstance(all_cycle_tables, dict) else set()
        if _skip:
            _before = len(unique_cycles)
            unique_cycles = [c for c in unique_cycles
                             if int(c) not in _skip]
            if len(unique_cycles) < _before:
                print(f"  Cycle(s) {', '.join(str(c) for c in sorted(_skip))} "
                      f"omitted: not a finished measurement")
        num_cycles = len(unique_cycles)

        palette = sns.color_palette(colour_palette, n_colors=num_cycles)

        fig, ax = plt.subplots()

        # --- Plot traces ---
        # After Cell 4b normalisation, Discharge = useful half-cycle for
        # all electrode types. Solid = discharge, dashed = charge.
        for i, cycle in enumerate(unique_cycles):
            colour = palette[i]

            dc = df_discharge[df_discharge['Cycle'] == cycle]
            if not dc.empty:
                dc_sorted = dc.sort_values('Discharge_Capacity')
                ax.plot(dc_sorted['Discharge_Capacity'],
                       dc_sorted['Voltage'],
                       color=colour, linestyle='-', linewidth=1.0)

            cc = df_charge[df_charge['Cycle'] == cycle]
            if not cc.empty:
                cc_sorted = cc.sort_values('Charge_Capacity')
                ax.plot(cc_sorted['Charge_Capacity'],
                       cc_sorted['Voltage'],
                       color=colour, linestyle='--', linewidth=1.0)

        # --- Labelling mode selection ---
        endpoints = _get_endpoints(df_charge, df_discharge,
                                    unique_cycles, key_cycles)

        if LABEL_MODE == 'auto':
            overlap_frac = _check_overlap(endpoints)
            use_colourbar = overlap_frac > AUTO_OVERLAP_THRESHOLD
            mode_used = 'colourbar' if use_colourbar else 'annotations'
            print(f"  Label mode: auto → {mode_used} "
                  f"({overlap_frac*100:.0f}% overlap, "
                  f"threshold {AUTO_OVERLAP_THRESHOLD*100:.0f}%)")
        elif LABEL_MODE == 'colourbar':
            use_colourbar = True
            mode_used = 'colourbar'
        else:
            use_colourbar = False
            mode_used = 'annotations'

        if use_colourbar:
            _add_colourbar(fig, ax, unique_cycles, colour_palette)
        else:
            _add_annotations(ax, endpoints, unique_cycles, palette)

        # --- Formatting ---
        ax.set_xlabel('Specific Capacity / mAh g$^{-1}$', fontsize=16)
        ax.set_ylabel('Voltage / V', fontsize=16)
        ax.tick_params(axis='both', labelcolor='black', labelsize=14,
                       width=1, direction='in', top=True, right=True)

        if COMMON_XAXIS_SCALE and global_max_capacity > 0:
            ax.set_xlim(left=0, right=global_max_capacity)
            # The limit is a round number, so the ticks should be the round
            # numbers under it. Matplotlib's own locator, given 0-150, picks
            # 0/20/40/.../140 and leaves the last gridline 10 short of the
            # frame; the step that produced the limit does not.
            if global_xaxis_step:
                ax.xaxis.set_major_locator(
                    mticker.MultipleLocator(global_xaxis_step))
        else:
            ax.set_xlim(left=0)
            ax.set_xlim(right=ax.get_xlim()[1] * (1 + XAXIS_PADDING_FRACTION))

        plt.tight_layout()

        # --- Save ---
        if save_location:
            file_format = params.get('file_format', 'png')
            filename = f"{name}_full_voltage_profile.{file_format}"
            filepath = os.path.join(save_location, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            saved(filepath)

        # --- Figure caption (v1.8: anode-aware) ---
        # The PROTOCOL window, not the extent of the record. See
        # `_caption_window`: the record includes the open-circuit start.
        v_min, v_max = _caption_window(params, df_work)
        caption = (
            f"Figure X. Voltage profiles for {num_cycles} galvanostatic "
            f"cycles of {composition}, cycled between {v_min:.2f} and "
            f"{v_max:.2f} V at {rate_phrase(params)}. "
            f"{dch_label} data are shown as solid lines and "
            f"{chg_label.lower()} data as dashed lines, with colour "
            f"indicating cycle number."
        )
        print(section("  Suggested caption"))
        print(bullet(caption, indent=2, label_width=2))

        plt.show()
        plt.close(fig)

    print(rule())


def _unusable(ct):
    """Cycles no figure should treat as a finished measurement.

    `Incomplete` is protocol-level and deliberately narrow — a cycle whose
    discharge is missing or zero. It cannot see the half-cycle that is
    HALFWAY THROUGH, because half a discharge is a real number, so the
    cycle an export was written during sailed past every filter in this
    module and plotted as a sudden drop at the end of ten curves. Both are
    reasons not to plot a point; they are kept as separate columns so the
    CSV still says which is which.
    """
    if ct is None or not len(ct):
        return pd.Series(dtype=bool)
    bad = pd.Series(False, index=ct.index)
    for col in ('Incomplete', 'Partial_Final'):
        if col in ct.columns:
            bad = bad | ct[col].fillna(False).astype(bool)
    return bad


def _unusable_cycles(ct):
    """The same verdict as a set of cycle numbers, for the figures that
    rebuild their y-values from the raw records and can only be told which
    cycles to drop."""
    bad = _unusable(ct)
    if ct is None or not len(bad) or 'Cycle' not in ct.columns:
        return set()
    return set(pd.to_numeric(ct.loc[bad, 'Cycle'],
                             errors='coerce').dropna().astype(int).tolist())


def _ct_with_flag(ct, value_col):
    """`ct[['Cycle', value_col, 'Incomplete']]` with the partial-final cycle
    folded into the flag, so a caller's existing `~df['Incomplete']` mask
    excludes it too."""
    out = ct[['Cycle', value_col]].copy()
    out['Incomplete'] = _unusable(ct).values
    return out


@_honours_verbose
def cycling_summary(electrochemical_data, user_parameters, *, save_location=None,
        all_cycle_tables=None, file_format='png', verbose=True,
        in_progress=None):
    """
    Per-cycle capacity, coulombic efficiency and the incomplete-cycle flags every later plot reads.

    Ported from 1.8.7 Cell 5b, wrapped in a function without editing
    the body. `all_cycle_tables` is accepted for signature symmetry with the
    other cycling functions and IGNORED — this function builds the tables and
    returns them; it does not read them.
    """
    print(rule("CYCLING PERFORMANCE"))

    def _detect_incomplete_cycles(discharge_capacities, charge_capacities=None):
        """
        Flag a cycle as incomplete if and only if the cycler did not
        complete its prescribed protocol for that cycle.

        Operational definition (v1.7.1):
          - discharge missing or zero AND charge missing or zero
                -> cycler aborted before either half-cycle -> incomplete
          - discharge missing or zero AND charge present
                -> charge completed, discharge did not -> incomplete
          - otherwise -> complete (even if degraded, low-CE, or parasitic)

        Health-based filtering (low retention, abnormal CE, parasitic
        current holds) lives in the Triplicate Analysis health pre-filter,
        NOT here. Cell 5b only reports protocol-level status.

        Parameters
        ----------
        discharge_capacities : list-like of float
            Discharge capacity (mAh/g) per cycle, in cycle order.
            After Cell 4b normalisation, this is always the "useful"
            half-cycle (delithiation for anodes, discharge for cathodes).
        charge_capacities : list-like of float, optional
            Charge capacity per cycle. If omitted, the function falls
            back to flagging only missing/zero discharges (safe default
            for Cell 5b, which always has charge data in current use).

        Returns
        -------
        list of bool
            True = incomplete; False = complete.
        """
        n = len(discharge_capacities)
        if charge_capacities is None:
            charge_capacities = [np.nan] * n
        else:
            charge_capacities = list(charge_capacities)

        flags = []
        for d, c in zip(discharge_capacities, charge_capacities):
            d_empty = pd.isna(d) or (pd.notna(d) and d <= 0.0)
            c_empty = pd.isna(c) or (pd.notna(c) and c <= 0.0)
            if d_empty:
                # Either cycler aborted entirely, or charge ran but
                # discharge was not attempted/completed.
                flags.append(True)
            else:
                # Discharge has data — the cycle completed its discharge
                # step. Charge anomalies (parasitic holds, capacity
                # spikes) are not incompleteness.
                flags.append(False)
        return flags

    # Built here and RETURNED; the parameter of the same name exists only
    # because 1.8.7's Cell 5b read a notebook global. It was documented as
    # "passing None reproduces the not-yet-run branch", which it could not
    # do, being overwritten on this line.
    all_cycle_tables = {}

    for name, df in electrochemical_data.items():
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        charge_rate_c = params.get('charge_rate_c', 'Unknown')
        is_anode = params.get('anode_labels_swapped', False)
        dch_label = _discharge_label(params)
        chg_label = _charge_label(params)

        print()
        print(heading(composition))
        if is_anode:
            print(entry("half-cell", "anode",
                        f"{dch_label.lower()} is the useful capacity"))

        if not all(col in df.columns for col in ['Cycle', 'Step',
                    'Discharge_Capacity', 'Charge_Capacity']):
            print(verdict("caution", "missing required columns, skipping."))
            continue

        # The one definition — see `delivered_capacity`.
        cycle_df = delivered_capacity(df)

        # CE = discharge / charge (correct for both electrode types
        # after Cell 4b normalisation)
        cycle_df['CE_%'] = coulombic_efficiency(
            cycle_df['Discharge_mAh_g'], cycle_df['Charge_mAh_g']).values

        # --- Detect incomplete cycles (v1.7.1: protocol-level only) ---
        incomplete_flags = _detect_incomplete_cycles(
            cycle_df['Discharge_mAh_g'].tolist(),
            cycle_df['Charge_mAh_g'].tolist()
        )
        cycle_df['Incomplete'] = incomplete_flags

        # --- Half-cycle still running when the export was written (1.9.0) ---
        # A cycler export is routinely taken while the cell is STILL
        # CYCLING, so the file's final half-cycle is usually unfinished.
        # That is a snapshot of a running experiment, not a measurement.
        #
        # `_detect_incomplete_cycles` is protocol-level and deliberately so:
        # it catches a cycle whose discharge is missing or zero. It cannot
        # catch a half-cycle that is HALFWAY THROUGH, because half a
        # discharge is a real number. On the LTO triplicate, cell B's export
        # was written during cycle 11's delithiation: 64.7 mAh/g of the
        # 123.3 it would have delivered. Every headline read that as data —
        # 52% retention, a fade rate of -3.4 mAh/g/cycle, and a CRITICAL
        # health flag predicting failure within ten cycles — for a cell
        # holding 99% of its capacity and still running.
        #
        # WHICH half-cycle is decided by `analyse.half_cycles_in_progress`
        # and handed in as `in_progress`, because the question is
        # about the raw record and not about this table: a galvanostatic
        # half-cycle terminates ON VOLTAGE, so the test is whether it
        # reached the cut-off. Judging it by capacity instead would have
        # condemned 15 perfectly good half-cycles of P3 cell B, which dies
        # at cycle 67 and is then driven to the cut-off, and back, for 153
        # more cycles. A cycle is excluded from the headline metrics here
        # when the half-cycle those metrics are computed FROM is the one
        # still running; the completed half of the same cycle is kept.
        cycle_df['Partial_Final'] = False
        _in_prog = set((in_progress or {}).get(name, ()) or ())
        _pf_note = None
        for _pf_cyc, _pf_step in sorted(_in_prog):
            _row = cycle_df['Cycle'] == _pf_cyc
            if not _row.any():
                continue
            # The headline metrics are built on the discharge half. If the
            # CHARGE half is the unfinished one, the discharge of that cycle
            # has not started, and `Incomplete` already covers it.
            _dch = float(pd.to_numeric(
                cycle_df.loc[_row, 'Discharge_mAh_g'], errors='coerce').iloc[0]
            ) if _row.any() else np.nan
            cycle_df.loc[_row, 'Partial_Final'] = True
            _pf_note = (_pf_cyc, _pf_step, _dch)
            print(verdict("caution", f"cycle {_pf_cyc} {_pf_step} was "
                                     f"still running when this file was "
                                     f"exported"))
            print(bullet("It never reached the cut-off voltage every other "
                         "half-cycle going the same way reaches. Not a "
                         "measurement yet: excluded from the headline metrics "
                         "below, still plotted, and complete in the next "
                         "export."))

        n_incomplete = sum(incomplete_flags)
        incomplete_cycles = cycle_df[cycle_df['Incomplete']]['Cycle'].tolist()
        # THE TWO REASONS, kept apart. `Incomplete` is protocol-level;
        # `Partial_Final` is the export catching a half-cycle mid-flight.
        # `complete_df` below has always excluded both, but every line that
        # TOLD the reader about it counted only the first, so cell B's run
        # excluded cycle 11 silently: full table, unmarked row, a headline
        # that claimed nothing had been dropped, and a caption that said
        # nothing at all.
        _partial_cycles = cycle_df.loc[
            cycle_df['Partial_Final'].fillna(False).astype(bool),
            'Cycle'].tolist()
        _excluded_mask = _unusable(cycle_df)
        n_excluded = int(_excluded_mask.sum())
        excluded_cycles = cycle_df.loc[_excluded_mask, 'Cycle'].tolist()
        # No print here. The verdict above already named the cycle and said
        # it was excluded from the headline metrics; this said it again in
        # weaker words, and the table below marks the row INCOMPL anyway.

        # --- Capacity retention vs reference cycle ---
        ref_row = cycle_df[
            (cycle_df['Cycle'] == RETENTION_REFERENCE_CYCLE) &
            (~cycle_df['Incomplete'])
        ]
        if not ref_row.empty:
            ref_cap = ref_row['Discharge_mAh_g'].iloc[0]
        else:
            complete = cycle_df[~cycle_df['Incomplete']]
            if not complete.empty:
                ref_cap = complete['Discharge_mAh_g'].iloc[0]
                print(f"  Note: reference cycle {RETENTION_REFERENCE_CYCLE} "
                      f"not found, using cycle "
                      f"{int(complete['Cycle'].iloc[0])}")
            else:
                ref_cap = np.nan

        if pd.notna(ref_cap) and ref_cap > 0:
            cycle_df['Retention_%'] = (
                cycle_df['Discharge_mAh_g'] / ref_cap * 100
            )
        else:
            cycle_df['Retention_%'] = np.nan

        # --- v1.8: Cell health diagnostics ---
        # Detect progressive parasitic current, voltage anomalies, and
        # definite failure signatures. Flags are stored in cycle_df so
        # every downstream cell can access them.

        # 1. Charge excess per cycle (Q_chg - Q_dchg)
        cycle_df['Charge_Excess_mAh_g'] = (
            cycle_df['Charge_mAh_g'] - cycle_df['Discharge_mAh_g'])

        # 2. Progressive parasitic current detection
        # Start from cycle 4 to avoid formation-period false positives.
        # Cycles 1-3 typically show CE evolution from SEI/CEI formation
        # that looks like a declining trend but is normal stabilisation.
        _DIAG_START_CYCLE = 4
        _diag_df = cycle_df[
            (~cycle_df['Incomplete']) & (~cycle_df['Partial_Final'])
            & (cycle_df['Cycle'] >= _DIAG_START_CYCLE)
        ].copy()

        _cell_flags = []      # list of (severity, message) tuples

        if len(_diag_df) >= 4:
            # CE trend: linear fit to CE vs cycle
            _ce_vals = _diag_df['CE_%'].dropna()
            if len(_ce_vals) >= 4:
                _ce_slope, _ce_intercept = np.polyfit(
                    _diag_df.loc[_ce_vals.index, 'Cycle'], _ce_vals, 1)

                # Check for monotonic decline over last N cycles
                _last_n = min(5, len(_ce_vals))
                _recent_ce = _ce_vals.tail(_last_n)
                _n_declining = sum(
                    _recent_ce.iloc[i] < _recent_ce.iloc[i-1]
                    for i in range(1, len(_recent_ce)))
                _monotonic = _n_declining >= _last_n - 1  # allow 1 exception

                if _ce_slope < -0.5 and _monotonic:
                    _cell_flags.append(('CRITICAL',
                        f'CE declining monotonically at '
                        f'{_ce_slope:.2f}%/cycle — progressive '
                        f'parasitic current'))
                elif _ce_slope < -0.3:
                    _cell_flags.append(('WARNING',
                        f'CE trend negative ({_ce_slope:.2f}%/cycle) — '
                        f'possible parasitic current developing'))

            # Charge excess trend: is delta_Q growing?
            _excess = _diag_df['Charge_Excess_mAh_g'].dropna()
            if len(_excess) >= 4:
                _ex_slope, _ = np.polyfit(
                    _diag_df.loc[_excess.index, 'Cycle'], _excess, 1)
                if _ex_slope > 0.3:
                    _cell_flags.append(('CRITICAL',
                        f'Charge excess growing at '
                        f'{_ex_slope:.2f} mAh/g/cycle — charge consumed '
                        f'by side reaction'))
                elif _ex_slope > 0.1:
                    _cell_flags.append(('WARNING',
                        f'Charge excess trend positive '
                        f'({_ex_slope:.2f} mAh/g/cycle) — monitor for '
                        f'parasitic activity'))

        # 3. Voltage anomaly on final data points
        # 4. Final cycle with zero or near-zero discharge
        # Skip both if the last cycle is already flagged incomplete —
        # that's a mid-experiment data pull, not a cell failure.
        #
        # `Incomplete` ALONE IS NOT THAT TEST. `Partial_Final` is the flag
        # that marks the mid-experiment data pull, set forty lines above; a
        # half-cycle caught mid-discharge has a small but non-zero delivered
        # capacity, so `Incomplete` is False and check 4 fired. A healthy
        # cell at 99% retention, exported thirty seconds into cycle 11's
        # discharge, was told "charged 123.4 mAh/g but delivered 0.4 mAh/g
        # on discharge — cell died. This cell has failed. Data after the
        # failure point is not meaningful." Both flags gate both checks.
        _last_cycle = int(cycle_df['Cycle'].max())
        _last_row = cycle_df[cycle_df['Cycle'] == _last_cycle]
        _last_incomplete = bool(_last_row['Incomplete'].iloc[0]) or bool(
            _last_row['Partial_Final'].iloc[0]
            if 'Partial_Final' in _last_row else False)

        if not _last_incomplete:
            # 3. Check if the last recorded step has voltage going the wrong way
            _last_cycle_data = df[df['Cycle'] == _last_cycle]
            if not _last_cycle_data.empty and 'Step' in _last_cycle_data.columns:
                _last_dchg = _last_cycle_data[
                    _last_cycle_data['Step'] == 'Discharge']
                if not _last_dchg.empty and len(_last_dchg) <= 3:
                    # Discharge step with <=3 data points = aborted
                    _v_start = pd.to_numeric(
                        _last_dchg['Voltage'].iloc[0], errors='coerce')
                    _v_prev_step = None
                    _last_chg = _last_cycle_data[
                        _last_cycle_data['Step'] == 'Charge']
                    if not _last_chg.empty:
                        _v_prev_step = pd.to_numeric(
                            _last_chg['Voltage'].iloc[-1], errors='coerce')

                    if (_v_prev_step is not None and pd.notna(_v_start)
                            and pd.notna(_v_prev_step)
                            and _v_start > _v_prev_step + 0.1):
                        _cell_flags.append(('FAILURE',
                            f'Cycle {_last_cycle}: voltage spiked '
                            f'{_v_start:.3f} V on discharge start '
                            f'(> {_v_prev_step:.3f} V end-of-charge) — '
                            f'open circuit / contact failure'))

            # 4. Final cycle with zero or near-zero discharge
            _last_row = cycle_df[cycle_df['Cycle'] == _last_cycle]
            if not _last_row.empty:
                _last_dchg_cap = _last_row['Discharge_mAh_g'].iloc[0]
                _last_chg_cap = _last_row['Charge_mAh_g'].iloc[0]
                if (pd.notna(_last_chg_cap) and _last_chg_cap > 0
                        and (pd.isna(_last_dchg_cap) or _last_dchg_cap < 1.0)):
                    _cell_flags.append(('FAILURE',
                        f'Cycle {_last_cycle}: charged '
                        f'{_last_chg_cap:.1f} mAh/g but delivered '
                        f'{"0" if pd.isna(_last_dchg_cap) else f"{_last_dchg_cap:.1f}"} '
                        f'mAh/g on discharge — cell died'))

        # Store flags in cycle_df metadata (accessible downstream)
        cycle_df.attrs['cell_health_flags'] = _cell_flags

        all_cycle_tables[name] = cycle_df

        # --- Print table (v1.8: anode-aware headers) ---
        # Status is 10 wide because the words are now words. The rule under
        # the header is measured from the header rather than hard-coded at
        # 56, which had not matched the columns since the CE field changed.
        _hdr = (f"  {'Cycle':>5}  {chg_label:>10}  {dch_label:>10}  "
                f"{'CE':>7}  {'Retention':>9}  {'Status':>10}")
        print("\n" + _hdr)
        print(f"  {'':>5}  {'mAh/g':>10}  {'mAh/g':>10}  "
              f"{'%':>7}  {'%':>9}  {'':>10}")
        print(f"  {'-' * (len(_hdr) - 2)}")

        for _, row in cycle_df.iterrows():
            def _f(val, fmt='.1f'):
                return f'{val:{fmt}}' if pd.notna(val) else '--'

            # Until 1.9.0.23 this read `Incomplete` alone. Cells A and C,
            # whose CHARGE half was the unfinished one, printed INCOMPL;
            # cell B, where the export was written during cycle 11's
            # discharge, printed a BLANK — and its row showed 64.7 mAh/g,
            # CE 52.4%, retention 51.8% with nothing to say those were half
            # of a measurement still being taken. The one row in the run
            # that most needed a marker was the only one without one.
            if row['Incomplete']:
                status = 'Incomplete'
            elif bool(row.get('Partial_Final', False)):
                status = 'Running'
            else:
                status = ''

            print(f"  {int(row['Cycle']):>5}  "
                  f"{_f(row['Charge_mAh_g']):>10}  "
                  f"{_f(row['Discharge_mAh_g']):>10}  "
                  f"{_f(row['CE_%']):>7}  "
                  f"{_f(row['Retention_%']):>9}  "
                  f"{status:>8}")

        # --- Headline metrics (complete cycles only) ---
        # Headline metrics only. The full `cycle_df` — partial cycle and
        # all — is what goes into every figure and the exported CSV.
        complete_df = cycle_df[~cycle_df['Incomplete']
                               & ~cycle_df['Partial_Final']]

        print("\n" + section(
            f"  Headline metrics (excluding cycle"
            f"{'s' if n_excluded != 1 else ''} "
            f"{', '.join(str(int(c)) for c in excluded_cycles)})"
            if n_excluded > 0 else "  Headline metrics"))

        # First-cycle irreversible loss (v1.8: anode-aware text)
        # After Cell 4b swap:
        #   Cathode: Charge = delithiation/desodiation, Discharge = lithiation/sodiation
        #            irrev = charge - discharge (positive = normal)
        #   Anode:   Charge = lithiation (after swap), Discharge = delithiation
        #            irrev = charge - discharge (positive = normal)
        # In both cases, charge > discharge on cycle 1 means irreversible
        # capacity consumed during first insertion.
        c1 = cycle_df[cycle_df['Cycle'] == 1]
        if not c1.empty and not c1['Incomplete'].iloc[0]:
            q_chg_1 = c1['Charge_mAh_g'].iloc[0]
            q_dchg_1 = c1['Discharge_mAh_g'].iloc[0]
            if pd.notna(q_chg_1) and pd.notna(q_dchg_1) and q_chg_1 > 0:
                irrev_loss = q_chg_1 - q_dchg_1
                irrev_pct = irrev_loss / q_chg_1 * 100
                if irrev_loss >= 0:
                    if is_anode:
                        print(f"    1st cycle irreversible loss: "
                              f"{irrev_loss:.1f} mAh/g ({irrev_pct:.1f}%)")
                        print(f"    ({chg_label}: {q_chg_1:.1f} mAh/g, "
                              f"{dch_label}: {q_dchg_1:.1f} mAh/g)")
                    else:
                        print(f"    1st cycle irreversible loss: "
                              f"{irrev_loss:.1f} mAh/g ({irrev_pct:.1f}%)")
                else:
                    print(f"    1st cycle excess {dch_label.lower()}: "
                          f"{-irrev_loss:.1f} mAh/g ({-irrev_pct:.1f}% "
                          f"more {dch_label.lower()} than "
                          f"{chg_label.lower()})")
                print(f"    1st cycle CE: "
                      f"{q_dchg_1/q_chg_1*100:.1f}%")

        # Retention: last complete cycle vs reference
        if pd.notna(ref_cap) and not complete_df.empty:
            last_complete = complete_df.iloc[-1]
            last_cap = last_complete['Discharge_mAh_g']
            last_cycle = int(last_complete['Cycle'])

            if pd.notna(last_cap):
                # RETENTION AGAINST BOTH CYCLE 1 AND THE REFERENCE.
                # Cycle 1 includes the first-cycle irreversible loss and is
                # what a reader comparing against a datasheet expects; the
                # reference (2 by default) excludes it and is what a cycling
                # study quotes. They answer different questions and a paper
                # usually needs both, so reporting only one made the other a
                # calculation the reader had to do from the table.
                for _ref in sorted({1, RETENTION_REFERENCE_CYCLE}):
                    _row = complete_df[complete_df['Cycle'] == _ref]
                    if _row.empty:
                        continue
                    _cap = _row.iloc[0]['Discharge_mAh_g']
                    if pd.isna(_cap) or not _cap:
                        continue
                    print(f"    Cycle {_ref} {dch_label.lower()}: "
                          f"{_cap:.1f} mAh/g")
                print(f"    Cycle {last_cycle} "
                      f"{dch_label.lower()}: {last_cap:.1f} mAh/g")
                for _ref in sorted({1, RETENTION_REFERENCE_CYCLE}):
                    _row = complete_df[complete_df['Cycle'] == _ref]
                    if _row.empty:
                        continue
                    _cap = _row.iloc[0]['Discharge_mAh_g']
                    if pd.isna(_cap) or not _cap:
                        continue
                    _note = ("  [includes the first-cycle irreversible loss]"
                             if _ref == 1 else "")
                    print(f"    Retention (cycle {last_cycle} vs {_ref}): "
                          f"{100 * last_cap / _cap:.1f}%{_note}")

        # WHERE DOES FORMATION END? Measured from this table's own CE, not
        # assumed. See FADE_RATE_FROM_MEASURED_FORMATION.
        _fade_start = int(FADE_RATE_START_CYCLE)
        _fade_why = (f"the {FADE_RATE_START_CYCLE} in `FADE_RATE_START_CYCLE`, "
                     f"not a measurement")
        if FADE_RATE_FROM_MEASURED_FORMATION and not complete_df.empty:
            _ce_series = {int(r['Cycle']): float(r['CE_%'])
                          for _, r in complete_df.iterrows()
                          if pd.notna(r.get('CE_%'))}
            _fe, _fe_why = formation_end(_ce_series)
            if _fe is not None:
                _fade_start, _fade_why = int(_fe), _fe_why
            else:
                _fade_why = (f"{_fe_why}; falling back to cycle "
                             f"{FADE_RATE_START_CYCLE}")

        # Average CE (complete, excluding formation)
        stable_ce = complete_df[
            complete_df['Cycle'] >= _fade_start
        ]['CE_%'].dropna()

        if not stable_ce.empty:
            last_complete_cycle = int(complete_df['Cycle'].max())
            print(f"    Average CE (cycles {_fade_start}"
                  f"--{last_complete_cycle}): "
                  f"{stable_ce.mean():.2f}% "
                  f"(+/-{stable_ce.std():.2f}%)")

        # Fade rate (complete, excluding formation)
        stable = complete_df[
            (complete_df['Cycle'] >= _fade_start) &
            (complete_df['Discharge_mAh_g'].notna())
        ]

        if len(stable) >= 3:
            slope, _ = np.polyfit(stable['Cycle'],
                                  stable['Discharge_mAh_g'], 1)
            # THE DENOMINATOR MATCHES THE RANGE. `%/cycle` divided by the
            # capacity of `RETENTION_REFERENCE_CYCLE` while the slope was
            # fitted from somewhere else entirely, so the two halves of the
            # sentence described different cycles. It is now the capacity of
            # the first cycle the slope was fitted through.
            _row0 = stable[stable['Cycle'] == stable['Cycle'].min()]
            _base = (float(_row0.iloc[0]['Discharge_mAh_g'])
                     if not _row0.empty else np.nan)
            print(f"    Formation ends at cycle {_fade_start} "
                  f"({_fade_why})")
            if pd.notna(_base) and _base > 0:
                print(f"    Fade rate (cycles {_fade_start}"
                      f"--{int(stable['Cycle'].max())}): "
                      f"{slope:.3f} mAh/g/cycle "
                      f"({slope/_base*100:.2f}%/cycle of the cycle "
                      f"{_fade_start} capacity)")
            else:
                print(f"    Fade rate (cycles {_fade_start}"
                      f"--{int(stable['Cycle'].max())}): "
                      f"{slope:.3f} mAh/g/cycle")

        # --- Suggested caption (v1.8: anode-aware) ---
        caption = (
            f"Table X. Specific {chg_label.lower()} and "
            f"{dch_label.lower()} capacities, coulombic efficiency, "
            f"and capacity retention (relative to cycle "
            f"{RETENTION_REFERENCE_CYCLE}) for {composition} cycled "
            f"at {rate_phrase(params)}."
        )
        # Both reasons, named. A caption is the one line that leaves the
        # run and goes into a paper, so "excluded from headline metrics"
        # has to say WHICH cycles and WHY, and a cycle caught mid-flight is
        # not the same claim as a cycle that never discharged.
        _cap_notes = []
        if n_incomplete > 0:
            _cap_notes.append(
                f"cycle{'s' if n_incomplete > 1 else ''} "
                f"{', '.join(str(int(c)) for c in incomplete_cycles)} "
                f"{'were' if n_incomplete > 1 else 'was'} incomplete")
        if _partial_cycles:
            _n_pf = len(_partial_cycles)
            _cap_notes.append(
                f"cycle{'s' if _n_pf > 1 else ''} "
                f"{', '.join(str(int(c)) for c in _partial_cycles)} "
                f"{'were' if _n_pf > 1 else 'was'} still in progress when "
                f"the data were exported")
        if _cap_notes:
            _joined = " and ".join(_cap_notes)
            caption += (" " + _joined[0].upper() + _joined[1:]
                        + ", and excluded from headline metrics.")
        print(section("  Suggested caption"))
        print(bullet(caption, indent=2, label_width=2))

        # --- v1.8: Print cell health diagnostics ---
        if _cell_flags:
            print(f"\n  {'!'*60}")
            print(f"  CELL HEALTH DIAGNOSTICS")
            print(f"  {'!'*60}")
            for severity, msg in _cell_flags:
                if severity == 'FAILURE':
                    print(f"  ✘ DEFINITE FAILURE: {msg}")
                elif severity == 'CRITICAL':
                    print(f"  ⚠ CRITICAL: {msg}")
                else:
                    print(f"  ● WARNING: {msg}")
            if any(s == 'FAILURE' for s, _ in _cell_flags):
                print(f"\n  This cell has failed. Data after the failure "
                      f"point is not meaningful.")
                print(f"  Check the voltage profile to confirm the "
                      f"failure mode.")
            elif any(s == 'CRITICAL' for s, _ in _cell_flags):
                print(f"\n  This cell shows signs of progressive "
                      f"degradation beyond normal fading.")
                print(f"  If this pattern continues, expect cell "
                      f"failure within the next 5-10 cycles.")

            if not any(s == 'FAILURE' for s, _ in _cell_flags):
                print(f"\n  If cycling is still in progress, the final "
                      f"partial cycle may skew")
                print(f"  these trends. Re-run on the completed dataset "
                      f"to confirm.")
            print(f"  {'!'*60}")

    if all_cycle_tables:
        combined = None
        for name, cdf in all_cycle_tables.items():
            params = user_parameters.get(name, {})
            comp = _get_display_name(name, params, user_parameters)
            dch = _discharge_label(params)
            chg = _charge_label(params)

            export_df = cdf[['Cycle']].copy()
            export_df[f'{comp}_{chg}_mAh_g'] = cdf['Charge_mAh_g']
            export_df[f'{comp}_{dch}_mAh_g'] = cdf['Discharge_mAh_g']
            export_df[f'{comp}_CE_%'] = cdf['CE_%']
            # Self-describing: a retention figure without its reference is
            # not a measurement. RETENTION_REFERENCE_CYCLE is in the name.
            export_df[f'{comp}_Retention_vs_C{RETENTION_REFERENCE_CYCLE}_%'] = \
                cdf['Retention_%']
            export_df[f'{comp}_Incomplete'] = cdf['Incomplete']
            if 'Partial_Final' in cdf.columns:
                # Two different reasons a cycle is not a measurement, kept
                # apart so a reader of the CSV can tell "the cycler aborted"
                # from "the export was taken during it".
                export_df[f'{comp}_Still_Running'] = cdf['Partial_Final']

            if combined is None:
                combined = export_df
            else:
                combined = pd.merge(combined, export_df, on='Cycle',
                                   how='outer')

        if combined is not None:
            combined = combined.sort_values('Cycle').reset_index(drop=True)

            if save_location:
                fpath = os.path.join(save_location,
                                    'cycling_performance_summary.csv')
                combined.to_csv(fpath, index=False)
                saved(fpath)

    print(rule())

    # 1.8.7 Cell 5b built `all_cycle_tables` as a NOTEBOOK GLOBAL and every
    # later cell read it from there. Wrapped in a function the name became
    # local, and returning it was the one line the port had to add: without
    # it seven downstream figures lost their incomplete-cycle filtering and
    # the START_HERE report had no capacity numbers at all.
    return all_cycle_tables


@_honours_verbose
def voltage_profiles_key(electrochemical_data, user_parameters, *, save_location=None,
        all_cycle_tables=None, file_format='png', verbose=True):
    """
    The same overlay, filtered to the key cycles.

    Ported from 1.8.7 Cell 6, wrapped in a function without editing
    the body. `all_cycle_tables` is `cycling_summary`'s output where the
    original read a notebook global; passing None reproduces the
    not-yet-run branch exactly.
    """

    def _check_overlap(endpoints, cap_tol_frac=0.03, volt_tol=0.05):
        """Check what fraction of endpoints would overlap."""
        if len(endpoints) < 2:
            return 0.0

        caps = [e['capacity'] for e in endpoints]
        cap_range = max(caps) - min(caps) if len(caps) > 1 else max(caps)
        cap_tol = max(cap_range * cap_tol_frac, 2.0)

        n_overlapping = 0
        for i, ep in enumerate(endpoints):
            for j, other in enumerate(endpoints):
                if i == j:
                    continue
                if (abs(ep['capacity'] - other['capacity']) < cap_tol and
                    abs(ep['voltage'] - other['voltage']) < volt_tol):
                    n_overlapping += 1
                    break

        return n_overlapping / len(endpoints)

    def _add_annotations(ax, endpoints, unique_cycles, palette):
        """Add cycle number annotations at endpoints with staggering."""
        if not endpoints:
            return

        caps = [e['capacity'] for e in endpoints]
        cap_range = max(caps) - min(caps) if len(caps) > 1 else max(caps)
        cap_tol = max(cap_range * 0.03, 2.0)
        volt_tol = 0.05

        placed = []
        sorted_endpoints = sorted(endpoints, key=lambda e: e['cycle'],
                                   reverse=True)

        for ep in sorted_endpoints:
            cycle_idx = list(unique_cycles).index(ep['cycle'])
            colour = palette[cycle_idx]
            is_discharge = ep['step'] == 'Discharge'

            n_nearby = sum(
                1 for pc, pv in placed
                if (abs(ep['capacity'] - pc) < cap_tol and
                    abs(ep['voltage'] - pv) < volt_tol)
            )

            x_offset = 0
            if n_nearby > 0:
                direction = 1 if n_nearby % 2 == 0 else -1
                stagger = (n_nearby + 1) // 2
                x_offset = direction * stagger * 14

            y_offset = -10 if is_discharge else 10
            va = 'top' if is_discharge else 'bottom'

            ax.annotate(
                str(int(ep['cycle'])),
                xy=(ep['capacity'], ep['voltage']),
                xytext=(x_offset, y_offset),
                textcoords='offset points',
                ha='center', va=va,
                fontsize=8, fontweight='bold',
                color=colour, alpha=0.9
            )
            placed.append((ep['capacity'], ep['voltage']))

    def _add_colourbar(fig, ax, unique_cycles, colour_palette):
        """Add a colourbar mapping cycle number to colour."""
        cmap = plt.get_cmap(colour_palette)
        norm = mcolors.Normalize(vmin=min(unique_cycles),
                                  vmax=max(unique_cycles))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
        cbar.set_label('Cycle number', fontsize=13)
        cbar.ax.tick_params(labelsize=11)

    global_max_capacity = 0

    if COMMON_XAXIS_SCALE:
        for name, df in electrochemical_data.items():
            params = user_parameters.get(name, {})
            key_cycles = (OVERRIDE_KEY_CYCLES if OVERRIDE_KEY_CYCLES
                         else params.get('key_cycles', []))

            df_work = df.copy()
            for col in ['Charge_Capacity', 'Discharge_Capacity']:
                if col in df_work.columns:
                    df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

            if key_cycles:
                df_filtered = df_work[df_work['Cycle'].isin(key_cycles)]
            else:
                df_filtered = df_work

            for col in ['Charge_Capacity', 'Discharge_Capacity']:
                if col in df_filtered.columns:
                    col_max = df_filtered[col].max()
                    if pd.notna(col_max):
                        global_max_capacity = max(global_max_capacity, col_max)

        global_max_capacity, global_xaxis_step = nice_axis_limit(
            global_max_capacity)

        if global_max_capacity > 0:
            print(entry("common x-axis", f"{global_max_capacity:.0f} mAh/g",
                        f"key cycles; ticks every {global_xaxis_step:g}"
                        if global_xaxis_step else "key cycles"))

    _have_cycle_tables = _has_tables(all_cycle_tables)

    for name, df in electrochemical_data.items():
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        print(rule(composition))
        charge_rate_c = params.get('charge_rate_c', 'Unknown')
        colour_palette = params.get('colour_palette', 'viridis_r')
        key_cycles = (OVERRIDE_KEY_CYCLES if OVERRIDE_KEY_CYCLES
                     else params.get('key_cycles', []))
        is_anode = params.get('anode_labels_swapped', False)
        dch_label = _discharge_label(params)
        chg_label = _charge_label(params)

        # Work on a copy
        df_work = df.copy()

        # Robust dropna (v1.8): require Voltage, Cycle, Step, and at
        # least one capacity column.
        df_cleaned = df_work.dropna(subset=['Voltage', 'Cycle', 'Step'])
        df_cleaned = df_cleaned[
            df_cleaned['Charge_Capacity'].notna() |
            df_cleaned['Discharge_Capacity'].notna()
        ].copy()

        if df_cleaned.empty:
            print(verdict("caution", f"no valid data for {name}"))
            continue

        # --- Detect and exclude incomplete cycles ---
        all_cycles_sorted = sorted(df_cleaned['Cycle'].unique())

        if _have_cycle_tables and name in all_cycle_tables:
            # Prefer the table's protocol-level flags (Cell 6a)
            ct = all_cycle_tables[name]
            incomplete_cycles = set(
                _unusable_cycles(ct))
        else:
            # Fallback: protocol-level only (missing/zero discharge).
            # v1.8: removed the 80%-of-median heuristic which caused
            # cascade failures for anode data where first-cycle capacity
            # is anomalously high.
            incomplete_cycles = set()
            for cyc in all_cycles_sorted:
                dc = df_cleaned[(df_cleaned['Cycle'] == cyc) &
                                (df_cleaned['Step'] == 'Discharge')]
                if dc.empty:
                    incomplete_cycles.add(cyc)
                else:
                    cap = (dc['Discharge_Capacity'].max() -
                           dc['Discharge_Capacity'].min())
                    if pd.isna(cap) or cap <= 0:
                        incomplete_cycles.add(cyc)

        # Filter key cycles: must exist in data AND not be incomplete
        available_key_cycles = [
            c for c in key_cycles
            if c in all_cycles_sorted and c not in incomplete_cycles
        ]

        if incomplete_cycles & set(key_cycles):
            removed = incomplete_cycles & set(key_cycles)
            print(entry("key cycles excluded",
                        ", ".join(str(int(c)) for c in sorted(removed)),
                        "incomplete"))

        if not available_key_cycles:
            print(verdict("caution", f"no valid key cycles found for {name}"))
            # Fall back to all available non-incomplete cycles
            available_key_cycles = [
                c for c in all_cycles_sorted
                if c not in incomplete_cycles
            ]
            if not available_key_cycles:
                continue
            print(f"  Falling back to all complete cycles: "
                  f"{', '.join(str(int(c)) for c in available_key_cycles)}")

        # Filter data
        df_key = df_cleaned[df_cleaned['Cycle'].isin(available_key_cycles)]
        df_discharge = df_key[df_key['Step'] == 'Discharge']
        df_charge = df_key[df_key['Step'] == 'Charge']

        unique_cycles = sorted(df_key['Cycle'].unique())
        num_cycles = len(unique_cycles)

        palette = sns.color_palette(colour_palette, n_colors=num_cycles)

        fig, ax = plt.subplots()

        # --- Plot traces ---
        # After Cell 4b normalisation, solid = discharge = useful
        # half-cycle for both electrode types.
        for i, cycle in enumerate(unique_cycles):
            colour = palette[i]

            dc = df_discharge[df_discharge['Cycle'] == cycle]
            if not dc.empty:
                dc_sorted = dc.sort_values('Discharge_Capacity')
                ax.plot(dc_sorted['Discharge_Capacity'],
                       dc_sorted['Voltage'],
                       color=colour, linestyle='-', linewidth=1.2)

            cc = df_charge[df_charge['Cycle'] == cycle]
            if not cc.empty:
                cc_sorted = cc.sort_values('Charge_Capacity')
                ax.plot(cc_sorted['Charge_Capacity'],
                       cc_sorted['Voltage'],
                       color=colour, linestyle='--', linewidth=1.2)

        # --- Labelling ---
        endpoints = _get_endpoints(df_charge, df_discharge,
                                    unique_cycles, available_key_cycles)

        if LABEL_MODE == 'auto':
            overlap_frac = _check_overlap(endpoints)
            use_colourbar = overlap_frac > AUTO_OVERLAP_THRESHOLD
            mode_used = 'colourbar' if use_colourbar else 'annotations'
            print(f"  Label mode: auto -> {mode_used} "
                  f"({overlap_frac*100:.0f}% overlap)")
        elif LABEL_MODE == 'colourbar':
            use_colourbar = True
        else:
            use_colourbar = False

        if use_colourbar:
            _add_colourbar(fig, ax, unique_cycles, colour_palette)
        else:
            _add_annotations(ax, endpoints, unique_cycles, palette)

        # --- Formatting ---
        ax.set_xlabel('Specific Capacity / mAh g$^{-1}$', fontsize=14)
        ax.set_ylabel('Voltage / V', fontsize=14)
        ax.tick_params(axis='both', labelcolor='black', labelsize=12,
                       width=1, direction='in', top=True, right=True)

        if COMMON_XAXIS_SCALE and global_max_capacity > 0:
            ax.set_xlim(left=0, right=global_max_capacity)
            # The limit is a round number, so the ticks should be the round
            # numbers under it. Matplotlib's own locator, given 0-150, picks
            # 0/20/40/.../140 and leaves the last gridline 10 short of the
            # frame; the step that produced the limit does not.
            if global_xaxis_step:
                ax.xaxis.set_major_locator(
                    mticker.MultipleLocator(global_xaxis_step))
        else:
            ax.set_xlim(left=0)
            ax.set_xlim(right=ax.get_xlim()[1] * (1 + XAXIS_PADDING_FRACTION))

        plt.tight_layout()

        # --- Save ---
        if save_location:
            file_format = params.get('file_format', 'png')
            filename = f"{name}_key_cycles_voltage_profile.{file_format}"
            filepath = os.path.join(save_location, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            saved(filepath)

        # --- Figure caption (v1.8: anode-aware) ---
        # The PROTOCOL window, not the extent of the record. See
        # `_caption_window`: the record includes the open-circuit start.
        v_min, v_max = _caption_window(params, df_work)
        cycles_str = ', '.join(str(int(c)) for c in unique_cycles)
        caption = (
            f"Figure X. Voltage profiles for selected galvanostatic "
            f"cycles ({cycles_str}) of {composition}, cycled between "
            f"{v_min:.2f} and {v_max:.2f} V at a rate of "
            f"{rate_phrase(params)}. {dch_label} data are shown as solid "
            f"lines and {chg_label.lower()} data as dashed lines."
        )
        print(section("  Suggested caption"))
        print(bullet(caption, indent=2, label_width=2))

        plt.show()
        plt.close(fig)

        # --- Print which cycles were plotted ---
        print(f"  Plotted cycles: "
              f"{', '.join(str(int(c)) for c in unique_cycles)}")

    print(rule())


@_honours_verbose
def cycle_life(electrochemical_data, user_parameters, *, save_location=None,
        all_cycle_tables=None, file_format='png', verbose=True):
    """
    Capacity and coulombic efficiency against cycle number.

    Ported from 1.8.7 Cell 7, wrapped in a function without editing
    the body. `all_cycle_tables` is `cycling_summary`'s output where the
    original read a notebook global; passing None reproduces the
    not-yet-run branch exactly.
    """
    MARKER_SIZE = 7

    def _calculate_cycle_capacities(df):
        """The canonical per-cycle capacity, under this function's old
        column names. One definition — see `delivered_capacity`."""
        out = delivered_capacity(df)
        return out.rename(columns={"Discharge_mAh_g": "Discharge_Capacity",
                                   "Charge_mAh_g": "Charge_Capacity"})

    def _calculate_efficiency(row):
        """CE for one row, under this function's old column names. One
        definition — see `coulombic_efficiency`."""
        return float(coulombic_efficiency(
            [row['Discharge_Capacity']], [row['Charge_Capacity']]).iloc[0])

    _have_cycle_tables = _has_tables(all_cycle_tables)

    if EXCLUDE_INCOMPLETE and not _have_cycle_tables:
        print("  Note: Cell 5b not run — using protocol-level incomplete "
              "detection (missing/zero discharge).")

    global_max_capacity = 0

    if COMMON_YAXIS_CAPACITY:
        for name, df in electrochemical_data.items():
            cap_df = _calculate_cycle_capacities(df)
            if not cap_df.empty:
                for col in ['Charge_Capacity', 'Discharge_Capacity']:
                    col_max = cap_df[col].max()
                    if pd.notna(col_max):
                        global_max_capacity = max(global_max_capacity, col_max)

        global_max_capacity, global_yaxis_step = nice_axis_limit(
            global_max_capacity)

        if global_max_capacity > 0:
            print(entry("common y-axis", f"{global_max_capacity:.0f} mAh/g",
                        f"capacity; ticks every {global_yaxis_step:g}"
                        if global_yaxis_step else "capacity"))

    for name, df in electrochemical_data.items():
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        print(rule(composition))
        charge_rate_c = params.get('charge_rate_c', 'Unknown')
        is_anode = params.get('anode_labels_swapped', False)
        dch_label = _discharge_label(params)
        chg_label = _charge_label(params)

        # Calculate capacities
        cap_df = _calculate_cycle_capacities(df)

        if cap_df.empty:
            print(verdict("caution", f"no valid capacity data for {name}"))
            continue

        # --- Detect incomplete cycles ---
        if EXCLUDE_INCOMPLETE:
            if _have_cycle_tables and name in all_cycle_tables:
                # Prefer the table's protocol-level flags (Cell 6a)
                ct = all_cycle_tables[name]
                incomplete_set = set(
                    _unusable_cycles(ct))
                cap_df['Incomplete'] = cap_df['Cycle'].isin(incomplete_set)
            else:
                # Fallback: protocol-level only (missing/zero discharge).
                # v1.8: removed 80%-of-median heuristic.
                flags = []
                for _, row in cap_df.iterrows():
                    d = row['Discharge_Capacity']
                    flags.append(pd.isna(d) or d <= 0)
                cap_df['Incomplete'] = flags

            n_incomplete = cap_df['Incomplete'].sum()
            if n_incomplete > 0:
                removed = cap_df[cap_df['Incomplete']]['Cycle'].tolist()
                print(entry("cycles excluded",
                            ", ".join(str(int(c)) for c in removed),
                            "incomplete"))

            plot_df = cap_df[~cap_df['Incomplete']].copy()
        else:
            plot_df = cap_df.copy()

        if plot_df.empty:
            print(verdict("caution", f"no complete cycles for {name}"))
            continue

        # Calculate CE (v1.8: no electrode_type needed, swap handles it)
        plot_df['CE'] = plot_df.apply(_calculate_efficiency, axis=1)

        # CE: skip formation cycle
        ce_df = plot_df[
            (plot_df['Cycle'] >= CE_START_CYCLE) &
            (plot_df['CE'].notna())
        ]

        # --- Plot ---
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # Capacity (left axis) — legend labels adapt for anodes
        ax1.plot(plot_df['Cycle'], plot_df['Charge_Capacity'],
                color=COLOUR_CHARGE, marker=MARKER_CHARGE,
                markersize=MARKER_SIZE, linestyle='',
                label=chg_label, zorder=3)

        ax1.plot(plot_df['Cycle'], plot_df['Discharge_Capacity'],
                color=COLOUR_DISCHARGE, marker=MARKER_DISCHARGE,
                markersize=MARKER_SIZE, linestyle='',
                label=dch_label, zorder=3)

        # CE (right axis)
        if not ce_df.empty:
            ax2.plot(ce_df['Cycle'], ce_df['CE'],
                    color=COLOUR_EFFICIENCY, marker=MARKER_EFFICIENCY,
                    markersize=MARKER_SIZE - 1, linestyle='',
                    label='CE', zorder=2, alpha=0.7)

        # --- Axis formatting ---
        ax1.set_xlabel('Cycle number', fontsize=14)
        _force_integer_cycles(ax1)
        ax1.set_ylabel('Specific capacity / mAh g$^{-1}$', fontsize=14)
        ax2.set_ylabel('Coulombic efficiency / %', fontsize=14)

        # Y-axis limits: capacity
        if COMMON_YAXIS_CAPACITY and global_max_capacity > 0:
            ax1.set_ylim(bottom=0, top=global_max_capacity)
            if global_yaxis_step:
                ax1.yaxis.set_major_locator(
                    mticker.MultipleLocator(global_yaxis_step))
        else:
            ax1.set_ylim(bottom=0)
            ax1.set_ylim(top=ax1.get_ylim()[1] * (1 + CAPACITY_YAXIS_PADDING))

        # Y-axis limits: efficiency
        ax2.set_ylim(bottom=0, top=EFFICIENCY_YMAX)

        # Tick formatting
        ax1.tick_params(axis='both', labelsize=12, width=1, direction='in',
                        top=True)
        ax2.tick_params(axis='y', labelsize=12, width=1, direction='in')

        # Combined legend from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   fontsize=10, framealpha=0.7, loc='best')

        # Remove grids (clean look for dual-axis)
        ax1.grid(False)
        ax2.grid(False)

        # Spines
        for ax in [ax1, ax2]:
            for sp in ax.spines.values():
                sp.set_linewidth(0.8)

        plt.tight_layout()

        # --- Save ---
        if save_location:
            file_format = params.get('file_format', 'png')
            filename = f"{name}_cycle_life.{file_format}"
            filepath = os.path.join(save_location, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            saved(filepath)

        # --- Figure caption (v1.8: anode-aware) ---
        # Protocol window; see `_caption_window`.
        v_min, v_max = _caption_window(params, df)
        n_complete = len(plot_df)

        caption = (
            f"Figure X. Specific {chg_label.lower()} and "
            f"{dch_label.lower()} capacities (left axis) and "
            f"coulombic efficiency (right axis) for {composition} "
            f"over {n_complete} galvanostatic cycles between "
            f"{v_min:.2f} and {v_max:.2f} V at a rate of "
            f"{rate_phrase(params)}. {chg_label} capacities are shown as "
            f"amber squares, {dch_label.lower()} capacities as blue "
            f"circles, and coulombic efficiency as black triangles."
        )
        print(section("  Suggested caption"))
        print(bullet(caption, indent=2, label_width=2))

        plt.show()
        plt.close(fig)

        # --- Print summary ---
        if not ce_df.empty:
            mean_ce = ce_df['CE'].mean()
            std_ce = ce_df['CE'].std()
            print(f"  Average CE (cycles {CE_START_CYCLE}–"
                  f"{int(plot_df['Cycle'].max())}): "
                  f"{mean_ce:.2f}% (+/-{std_ce:.2f}%)")

    print(rule())


@_honours_verbose
def comparative_capacity(electrochemical_data, user_parameters, *, save_location=None,
        all_cycle_tables=None, file_format='png', verbose=True):
    """
    Discharge capacity for every dataset on one axis.

    Ported from 1.8.7 Cell 8, wrapped in a function without editing
    the body. `all_cycle_tables` is `cycling_summary`'s output where the
    original read a notebook global; passing None reproduces the
    not-yet-run branch exactly.
    """
    MARKER_SIZE = 7

    # The filter default, read from the one definition rather than

    # restated: six local copies were quoted in the console message

    # while `_flag_visual_outliers` used its own default, so editing

    # one changed the sentence and not the data.

    VISUAL_OUTLIER_THRESHOLD = VISUAL_OUTLIER_FRACTION
    def _comparative_capacity_label():
        """Return the best y-axis label for a multi-dataset capacity plot."""
        labels = set()
        for name in electrochemical_data:
            params = user_parameters.get(name, {})
            labels.add(_discharge_label(params))
        if len(labels) == 1:
            return f"{labels.pop()} capacity / mAh g$^{{-1}}$"
        return 'Specific capacity / mAh g$^{-1}$'

    def _comparative_capacity_word():
        """Return the capacity noun for captions (no units)."""
        labels = set()
        for name in electrochemical_data:
            params = user_parameters.get(name, {})
            labels.add(_discharge_label(params).lower())
        if len(labels) == 1:
            return f"{labels.pop()} capacities"
        return 'specific capacities'

    print(rule("COMPARATIVE DISCHARGE CAPACITY"))

    palette = MULTI_DATASET_PALETTE

    fig, ax = plt.subplots(figsize=(_plots.figure_width_inches, _plots.figure_height_inches))

    caption_compositions = []

    caption_rates = set()

    _have_cycle_tables = _has_tables(all_cycle_tables)

    if not _have_cycle_tables:
        print("  ⚠ all_cycle_tables not found — Cell 5b has not been run. "
              "Run Cell 5b first for the cleanest output.")

    for i, (name, df) in enumerate(electrochemical_data.items()):
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        charge_rate_c = params.get('charge_rate_c', None)
        dch_label = _discharge_label(params)

        colour = palette[i % len(palette)]
        marker = MARKERS[i % len(MARKERS)]

        # Primary path: Cell 5b cycle table
        if _have_cycle_tables and name in all_cycle_tables:
            ct = all_cycle_tables[name]
            plot_data = (_ct_with_flag(ct, 'Discharge_mAh_g')
                         .rename(columns={'Discharge_mAh_g':
                                          'Discharge_Capacity'})
                         .dropna(subset=['Cycle', 'Discharge_Capacity'])
                         .copy())
        else:
            # Fallback: derive via groupby (no Python loop)
            discharge = df[df['Step'] == 'Discharge'].dropna(
                subset=['Cycle', 'Discharge_Capacity'])
            if discharge.empty:
                print(f"  {composition}: no {dch_label.lower()} data "
                      f"found, skipping.")
                continue
            grp = discharge.groupby('Cycle')['Discharge_Capacity']
            plot_data = pd.DataFrame({
                'Cycle': grp.max().index.astype(int),
                'Discharge_Capacity': (grp.max() - grp.min()).values,
                'Incomplete': False,
            })

        if plot_data.empty:
            print(f"  {composition}: no {dch_label.lower()} data "
                  f"found, skipping.")
            continue

        plot_data = plot_data.sort_values('Cycle').reset_index(drop=True)

        # Exclude incomplete cycles (protocol-level flag from Cell 5b)
        if EXCLUDE_INCOMPLETE and 'Incomplete' in plot_data.columns:
            n_incomplete = int(plot_data['Incomplete'].sum())
            if n_incomplete > 0:
                removed = plot_data.loc[plot_data['Incomplete'],
                                        'Cycle'].tolist()
                print(f"  {composition}: excluded {n_incomplete} "
                      f"protocol-incomplete cycle(s): "
                      f"{', '.join(str(int(c)) for c in removed)}")
            plot_data = plot_data[~plot_data['Incomplete']].reset_index(drop=True)

        # Visual-outlier filter
        if VISUAL_OUTLIER_FILTER and not plot_data.empty:
            outlier_flags = _flag_visual_outliers(
                plot_data['Discharge_Capacity'].tolist(), threshold=VISUAL_OUTLIER_THRESHOLD,
                groups=rate_block_labels(plot_data['Cycle'], params))
            n_outliers = sum(outlier_flags)
            if n_outliers > 0:
                plot_data['_visual_outlier'] = outlier_flags
                removed = plot_data.loc[plot_data['_visual_outlier'],
                                        'Cycle'].tolist()
                print(f"  {composition}: excluded {n_outliers} visual "
                      f"outlier(s) (<{VISUAL_OUTLIER_THRESHOLD*100:.0f}% "
                      f"of running median): "
                      f"{', '.join(str(int(c)) for c in removed)}")
                plot_data = plot_data[~plot_data['_visual_outlier']].drop(
                    columns='_visual_outlier').reset_index(drop=True)

        if plot_data.empty:
            print(f"  {composition}: no complete cycles, skipping.")
            continue

        # Plot
        ax.plot(plot_data['Cycle'], plot_data['Discharge_Capacity'],
               color=colour, marker=marker, markersize=MARKER_SIZE,
               linestyle='', label=composition, zorder=3)

        caption_compositions.append(composition)
        if charge_rate_c is not None:
            caption_rates.add(str(charge_rate_c))

        print(f"  {composition}: {len(plot_data)} cycles plotted "
              f"({plot_data['Discharge_Capacity'].iloc[0]:.1f} -> "
              f"{plot_data['Discharge_Capacity'].iloc[-1]:.1f} mAh/g)")

    ax.set_xlabel('Cycle number', fontsize=14)

    _force_integer_cycles(ax)

    ax.set_ylabel(_comparative_capacity_label(), fontsize=14)

    ax.tick_params(axis='both', labelsize=12, width=1, direction='in',
                   top=True, right=True)

    for sp in ax.spines.values():
        sp.set_linewidth(0.8)

    ax.set_xlim(left=0)

    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * (1 + YAXIS_PADDING))

    ax.legend(fontsize=10, framealpha=0.7)

    plt.tight_layout()

    if caption_compositions:
        comp_str = ', '.join(caption_compositions[:-1])
        if len(caption_compositions) > 1:
            comp_str += f' and {caption_compositions[-1]}'
        else:
            comp_str = caption_compositions[0]

        rate_str = (' and '.join(sorted(caption_rates)) + ' C'
                   if caption_rates else 'the specified C-rate')

        # EVERY dataset's protocol window, not the union of their
        # records — which read 1.20-2.68 V for a triplicate cycled
        # 1.2-2.5 V, because one cell started from open circuit.
        _vlo, _vhi = _caption_window_all(user_parameters)
        v_range = (f"{_vlo:.2f} and {_vhi:.2f} V"
                   if np.isfinite(_vlo) and np.isfinite(_vhi)
                   else "the specified voltage limits")

        cap_word = _comparative_capacity_word()
        caption = (
            f"Figure X. Comparative {cap_word} for "
            f"{comp_str}, cycled at {rate_str} between {v_range}."
        )
        print(section("  Suggested caption"))
        print(bullet(caption, indent=2, label_width=2))

    if save_location:
        filepath = os.path.join(save_location,
                               f'comparative_discharge_capacity.{_run_image_format(user_parameters)}')
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        saved(filepath)

    plt.show()

    plt.close(fig)

    if len(electrochemical_data) > 1:
        loadings = {}
        for name in electrochemical_data:
            params = user_parameters.get(name, {})
            loading = params.get('active_loading_mg_cm2')
            if loading is not None:
                comp = _get_display_name(name, params, user_parameters)
                loadings[comp] = loading

        if len(loadings) >= 2:
            lo_vals = list(loadings.values())
            lo_min, lo_max = min(lo_vals), max(lo_vals)
            if lo_min > 0:
                spread_pct = (lo_max - lo_min) / lo_min * 100
                if spread_pct > 30:
                    # Adaptive label for the warning too
                    cap_word = _comparative_capacity_word()
                    print(f"\n  ⚠ LOADING MISMATCH WARNING")
                    print(f"  Active material loadings vary by "
                          f"{spread_pct:.0f}% across datasets:")
                    for comp, val in loadings.items():
                        print(f"    {comp:<30} {val:.2f} mg/cm²")
                    print(f"  Specific capacity comparisons may "
                          f"be misleading at different")
                    print(f"  loadings — electrochemical properties often "
                          f"do not scale with mass loading")
                    print(f"  (Cao et al., Nat. Nanotechnol. 14, 200-207, "
                          f"2019).")
                    print(f"  Consider comparing areal capacity "
                          f"(mAh/cm², Cell 10) instead.")

    print(rule())


@_honours_verbose
def capacity_retention(electrochemical_data, user_parameters, *, save_location=None,
        all_cycle_tables=None, file_format='png', verbose=True):
    """
    Retention against a reference cycle.

    Ported from 1.8.7 Cell 9, wrapped in a function without editing
    the body. `all_cycle_tables` is `cycling_summary`'s output where the
    original read a notebook global; passing None reproduces the
    not-yet-run branch exactly.
    """
    MARKER_SIZE = 7

    YAXIS_MIN = 0

    YAXIS_MAX = 105  # auto-scales higher if any retention >105%

    # The filter default, read from the one definition rather than

    # restated: six local copies were quoted in the console message

    # while `_flag_visual_outliers` used its own default, so editing

    # one changed the sentence and not the data.

    VISUAL_OUTLIER_THRESHOLD = VISUAL_OUTLIER_FRACTION
    def _retention_ylabel():
        """Return the y-axis label for the retention plot."""
        labels = set()
        for name in electrochemical_data:
            params = user_parameters.get(name, {})
            labels.add(_discharge_label(params))
        if len(labels) == 1:
            lbl = labels.pop()
            return f"{lbl} capacity retention / %"
        return 'Capacity retention / %'

    def _retention_caption_word():
        """Return the capacity retention noun for captions."""
        labels = set()
        for name in electrochemical_data:
            params = user_parameters.get(name, {})
            labels.add(_discharge_label(params).lower())
        if len(labels) == 1:
            return f"{labels.pop()} capacity retention"
        return 'capacity retention'

    _cap_word = _retention_caption_word()

    print(rule(_cap_word.upper()))

    palette = MULTI_DATASET_PALETTE

    fig, ax = plt.subplots(figsize=(_plots.figure_width_inches, _plots.figure_height_inches))

    caption_compositions = []

    max_retention_seen = 0

    _have_cycle_tables = _has_tables(all_cycle_tables)

    if not _have_cycle_tables:
        print("  ⚠ all_cycle_tables not found — Cell 5b has not been run. "
              "Run Cell 5b first for the cleanest output.")

    for i, (name, df) in enumerate(electrochemical_data.items()):
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        dch_label = _discharge_label(params)

        colour = palette[i % len(palette)]
        marker = MARKERS[i % len(MARKERS)]

        # Primary path
        if _have_cycle_tables and name in all_cycle_tables:
            ct = all_cycle_tables[name]
            cap_df = (_ct_with_flag(ct, 'Discharge_mAh_g')
                      .rename(columns={'Discharge_mAh_g':
                                       'Discharge_Capacity'})
                      .dropna(subset=['Cycle', 'Discharge_Capacity'])
                      .copy())
        else:
            # Fallback: derive via groupby
            discharge = df[df['Step'] == 'Discharge'].dropna(
                subset=['Cycle', 'Discharge_Capacity'])
            if discharge.empty:
                print(f"  {composition}: no {dch_label.lower()} data, "
                      f"skipping.")
                continue
            grp = discharge.groupby('Cycle')['Discharge_Capacity']
            cap_df = pd.DataFrame({
                'Cycle': grp.max().index.astype(int),
                'Discharge_Capacity': (grp.max() - grp.min()).values,
                'Incomplete': False,
            })

        if cap_df.empty:
            print(f"  {composition}: no {dch_label.lower()} data, "
                  f"skipping.")
            continue

        cap_df = cap_df.sort_values('Cycle').reset_index(drop=True)

        # Exclude incomplete cycles (protocol-level)
        if EXCLUDE_INCOMPLETE and 'Incomplete' in cap_df.columns:
            n_incomplete = int(cap_df['Incomplete'].sum())
            if n_incomplete > 0:
                removed = cap_df.loc[cap_df['Incomplete'],
                                     'Cycle'].tolist()
                print(f"  {composition}: excluded {n_incomplete} "
                      f"protocol-incomplete cycle(s): "
                      f"{', '.join(str(int(c)) for c in removed)}")
            plot_data = cap_df[~cap_df['Incomplete']].copy().reset_index(drop=True)
        else:
            plot_data = cap_df.copy()

        # Visual-outlier filter
        if VISUAL_OUTLIER_FILTER and not plot_data.empty:
            outlier_flags = _flag_visual_outliers(
                plot_data['Discharge_Capacity'].tolist(), threshold=VISUAL_OUTLIER_THRESHOLD,
                groups=rate_block_labels(plot_data['Cycle'], params))
            n_outliers = sum(outlier_flags)
            if n_outliers > 0:
                plot_data['_visual_outlier'] = outlier_flags
                removed = plot_data.loc[plot_data['_visual_outlier'],
                                        'Cycle'].tolist()
                print(f"  {composition}: excluded {n_outliers} visual "
                      f"outlier(s) (<{VISUAL_OUTLIER_THRESHOLD*100:.0f}% "
                      f"of running median): "
                      f"{', '.join(str(int(c)) for c in removed)}")
                plot_data = plot_data[~plot_data['_visual_outlier']].drop(
                    columns='_visual_outlier').reset_index(drop=True)

        if plot_data.empty:
            print(f"  {composition}: no complete cycles, skipping.")
            continue

        # Reference capacity
        ref_row = plot_data[plot_data['Cycle'] == RETENTION_REFERENCE_CYCLE]
        if not ref_row.empty:
            ref_cap = ref_row['Discharge_Capacity'].iloc[0]
        else:
            ref_cap = plot_data['Discharge_Capacity'].iloc[0]
            ref_cycle_used = int(plot_data['Cycle'].iloc[0])
            print(f"  {composition}: reference cycle "
                  f"{RETENTION_REFERENCE_CYCLE} not found, "
                  f"using cycle {ref_cycle_used}")

        if pd.isna(ref_cap) or ref_cap <= 0:
            print(f"  {composition}: invalid reference capacity, skipping.")
            continue

        # Calculate retention
        plot_data['Retention'] = (
            plot_data['Discharge_Capacity'] / ref_cap * 100
        )

        max_retention_seen = max(max_retention_seen,
                                 plot_data['Retention'].max())

        # Plot
        ax.plot(plot_data['Cycle'], plot_data['Retention'],
               color=colour, marker=marker, markersize=MARKER_SIZE,
               linestyle='', label=composition, zorder=3)

        caption_compositions.append(composition)

        last = plot_data.iloc[-1]
        print(f"  {composition}: {len(plot_data)} cycles, "
              f"retention at cycle {int(last['Cycle'])} = "
              f"{last['Retention']:.1f}%")

    ax.set_xlabel('Cycle number', fontsize=14)

    _force_integer_cycles(ax)

    ax.set_ylabel(_retention_ylabel(), fontsize=14)

    ax.tick_params(axis='both', labelsize=12, width=1, direction='in',
                   top=True, right=True)

    for sp in ax.spines.values():
        sp.set_linewidth(0.8)

    ax.set_xlim(left=0)

    y_top = (max(YAXIS_MAX, max_retention_seen * 1.03)
             if max_retention_seen > YAXIS_MAX else YAXIS_MAX)

    ax.set_ylim(bottom=YAXIS_MIN, top=y_top)

    ax.axhline(y=100, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)

    ax.legend(fontsize=10, framealpha=0.7)

    plt.tight_layout()

    if caption_compositions:
        comp_str = ', '.join(caption_compositions[:-1])
        if len(caption_compositions) > 1:
            comp_str += f' and {caption_compositions[-1]}'
        else:
            comp_str = caption_compositions[0]

        caption = (
            f"Figure X. {_cap_word.capitalize()} "
            f"(relative to cycle {RETENTION_REFERENCE_CYCLE}) for "
            f"{comp_str}. A horizontal dotted line at 100% is shown "
            f"for reference."
        )
        print(section("  Suggested caption"))
        print(bullet(caption, indent=2, label_width=2))

    if save_location:
        filepath = os.path.join(save_location,
                               f'capacity_retention_vs_cycle.{_run_image_format(user_parameters)}')
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        saved(filepath)

    plt.show()

    plt.close(fig)

    print(rule())


@_honours_verbose
def fade_rate(electrochemical_data, user_parameters, *, save_location=None,
        all_cycle_tables=None, file_format='png', verbose=True):
    """
    Rate of capacity loss, per cycle and cumulative.

    Ported from 1.8.7 Cell 9b, wrapped in a function without editing
    the body. `all_cycle_tables` is `cycling_summary`'s output where the
    original read a notebook global; passing None reproduces the
    not-yet-run branch exactly.
    """
    # `palette` was never assigned in 1.8.7 Cell 9b. In the notebook it
    # resolved to whatever an earlier cell had left in globals — the implicit
    # shared state this package exists to remove — and in a function it is a
    # NameError. Cells 8, 9, 11 and 12 all bind it to MULTI_DATASET_PALETTE,
    # and that is the palette the published figure was drawn with, so this
    # reproduces the figure and closes the latent bug.
    palette = MULTI_DATASET_PALETTE

    MARKER_SIZE = 6

    VISUAL_OUTLIER_THRESHOLD = 0.30  # cap < 30% of running median → outlier

    print(rule("CAPACITY FADE RATE"))

    fade_data = {}

    has_rolling_data = False

    _have_cycle_tables = _has_tables(all_cycle_tables)

    if not _have_cycle_tables:
        print("  ⚠ all_cycle_tables not found — Cell 5b has not been run. "
              "Run Cell 5b first for the cleanest output.")

    for name, df in electrochemical_data.items():
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        dch_label = _discharge_label(params)

        # Primary path: read from Cell 5b
        if _have_cycle_tables and name in all_cycle_tables:
            ct = all_cycle_tables[name]
            cap_df = (_ct_with_flag(ct, 'Discharge_mAh_g')
                      .dropna(subset=['Cycle', 'Discharge_mAh_g'])
                      .copy())
        else:
            # Fallback: derive via groupby (no Python loop)
            discharge = df[df['Step'] == 'Discharge'].dropna(
                subset=['Cycle', 'Discharge_Capacity'])
            if discharge.empty:
                print(f"  {composition}: no {dch_label.lower()} data, "
                      f"skipping.")
                continue
            grp = discharge.groupby('Cycle')['Discharge_Capacity']
            cap_df = pd.DataFrame({
                'Cycle': grp.max().index.astype(int),
                'Discharge_mAh_g': (grp.max() - grp.min()).values,
                'Incomplete': False,
            })

        if cap_df.empty:
            print(f"  {composition}: no {dch_label.lower()} data, "
                  f"skipping.")
            continue

        cap_df = cap_df.sort_values('Cycle').reset_index(drop=True)

        # Exclude incomplete cycles (protocol-level flag from Cell 5b)
        if EXCLUDE_INCOMPLETE and 'Incomplete' in cap_df.columns:
            n_incomplete = int(cap_df['Incomplete'].sum())
            if n_incomplete > 0:
                removed = cap_df.loc[cap_df['Incomplete'],
                                     'Cycle'].tolist()
                print(f"  {composition}: excluded {n_incomplete} "
                      f"protocol-incomplete cycle(s): "
                      f"{', '.join(str(int(c)) for c in removed)}")
            cap_df = cap_df[~cap_df['Incomplete']].copy().reset_index(drop=True)

        # Visual-outlier filter (applied after Incomplete)
        if VISUAL_OUTLIER_FILTER and not cap_df.empty:
            outlier_flags = _flag_visual_outliers(
                cap_df['Discharge_mAh_g'].tolist(), threshold=VISUAL_OUTLIER_THRESHOLD,
                groups=rate_block_labels(cap_df['Cycle'], params))
            n_outliers = sum(outlier_flags)
            if n_outliers > 0:
                cap_df['_visual_outlier'] = outlier_flags
                removed = cap_df.loc[cap_df['_visual_outlier'],
                                     'Cycle'].tolist()
                print(f"  {composition}: excluded {n_outliers} visual "
                      f"outlier(s) (<{VISUAL_OUTLIER_THRESHOLD*100:.0f}% "
                      f"of running median): "
                      f"{', '.join(str(int(c)) for c in removed)}")
                cap_df = cap_df[~cap_df['_visual_outlier']].drop(
                    columns='_visual_outlier').reset_index(drop=True)

        if cap_df.empty:
            continue

        # Reference capacity
        ref_row = cap_df[cap_df['Cycle'] == FADE_REFERENCE_CYCLE]
        if not ref_row.empty:
            ref_cap = ref_row['Discharge_mAh_g'].iloc[0]
        else:
            ref_cap = cap_df['Discharge_mAh_g'].iloc[0]
            ref_cycle_used = int(cap_df['Cycle'].iloc[0])
            print(f"  {composition}: reference cycle "
                  f"{FADE_REFERENCE_CYCLE} not found, "
                  f"using cycle {ref_cycle_used}")

        if pd.isna(ref_cap) or ref_cap <= 0:
            print(f"  {composition}: invalid reference capacity, skipping.")
            continue

        # Cumulative capacity loss (%)
        cap_df['Capacity_Loss_%'] = (
            (ref_cap - cap_df['Discharge_mAh_g']) / ref_cap * 100
        )

        # Per-cycle change (%)
        cap_df['Per_Cycle_Change_%'] = (
            cap_df['Discharge_mAh_g'].diff() / ref_cap * 100
        )

        # Rolling-window fade
        if len(cap_df) >= ROLLING_WINDOW:
            has_rolling_data = True
            cap_df['Per_N_Cycle_Loss_%'] = (
                cap_df['Capacity_Loss_%'].rolling(
                    window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW
                ).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
            )
        else:
            cap_df['Per_N_Cycle_Loss_%'] = np.nan

        cap_df['Composition'] = composition
        fade_data[name] = cap_df

        # Summary
        last = cap_df.iloc[-1]
        print(f"  {composition}: {len(cap_df)} cycles, "
              f"total loss = {last['Capacity_Loss_%']:.1f}%")

    if fade_data:
        fig, ax = plt.subplots(figsize=(_plots.figure_width_inches, _plots.figure_height_inches))

        caption_compositions = []

        for i, (name, fdf) in enumerate(fade_data.items()):
            composition = fdf['Composition'].iloc[0]
            colour = palette[i % len(palette)]
            marker = MARKERS[i % len(MARKERS)]

            ax.plot(fdf['Cycle'], fdf['Capacity_Loss_%'],
                   color=colour, marker=marker, markersize=MARKER_SIZE,
                   linestyle='', label=composition, zorder=3)

            caption_compositions.append(composition)

        ax.set_xlabel('Cycle number', fontsize=14)
        _force_integer_cycles(ax)
        ax.set_ylabel('Cumulative capacity loss / %', fontsize=14)
        ax.tick_params(axis='both', labelsize=12, width=1, direction='in',
                       top=True, right=True)
        ax.axhline(y=0, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
        ax.legend(fontsize=10, framealpha=0.7)
        plt.tight_layout()

        # Caption
        comp_str = ', '.join(caption_compositions[:-1])
        if len(caption_compositions) > 1:
            comp_str += f' and {caption_compositions[-1]}'
        else:
            comp_str = caption_compositions[0]

        caption = (
            f"Figure X. Cumulative percentage capacity loss relative to "
            f"cycle {FADE_REFERENCE_CYCLE} for {comp_str}. "
            f"A value of 0% indicates capacity equal to the reference cycle."
        )
        print(section("  Suggested caption"))
        print(bullet(caption, indent=2, label_width=2))

        if save_location:
            fpath = os.path.join(save_location, f'capacity_fade_percentage.{_run_image_format(user_parameters)}')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            saved(fpath)

        plt.show()
        plt.close(fig)

        # =================================================================
        # PLOT 2: PER-CYCLE CAPACITY CHANGE
        # =================================================================

        fig, ax = plt.subplots(figsize=(_plots.figure_width_inches, _plots.figure_height_inches))

        for i, (name, fdf) in enumerate(fade_data.items()):
            composition = fdf['Composition'].iloc[0]
            colour = palette[i % len(palette)]
            marker = MARKERS[i % len(MARKERS)]

            # Skip the first point (NaN from diff)
            valid = fdf.dropna(subset=['Per_Cycle_Change_%'])

            ax.plot(valid['Cycle'], valid['Per_Cycle_Change_%'],
                   color=colour, marker=marker, markersize=MARKER_SIZE - 1,
                   linestyle='', label=composition, zorder=3, alpha=0.7)

        ax.set_xlabel('Cycle number', fontsize=14)
        _force_integer_cycles(ax)
        ax.set_ylabel('Capacity change per cycle / %', fontsize=14)
        ax.tick_params(axis='both', labelsize=12, width=1, direction='in',
                       top=True, right=True)
        ax.axhline(y=0, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
        ax.legend(fontsize=10, framealpha=0.7)
        plt.tight_layout()

        caption = (
            f"Figure X. Per-cycle capacity change (as percentage of "
            f"cycle {FADE_REFERENCE_CYCLE} capacity) for {comp_str}. "
            f"Negative values indicate capacity loss; positive values "
            f"indicate capacity gain. The horizontal dotted line at 0% "
            f"represents no change."
        )
        print(section("  Suggested caption"))
        print(bullet(caption, indent=2, label_width=2))

        if save_location:
            fpath = os.path.join(save_location, f'capacity_change_per_cycle.{_run_image_format(user_parameters)}')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            saved(fpath)

        plt.show()
        plt.close(fig)

        # =================================================================
        # PLOT 3: PER-ROLLING_WINDOW FADE (conditional)
        # =================================================================

        if has_rolling_data:
            fig, ax = plt.subplots(figsize=(_plots.figure_width_inches, _plots.figure_height_inches))

            plotted_any = False
            for i, (name, fdf) in enumerate(fade_data.items()):
                valid = fdf.dropna(subset=['Per_N_Cycle_Loss_%'])
                if valid.empty:
                    comp = fdf['Composition'].iloc[0]
                    print(f"  {comp}: fewer than {ROLLING_WINDOW} cycles, "
                          f"skipping per-{ROLLING_WINDOW}-cycle plot")
                    continue

                composition = fdf['Composition'].iloc[0]
                colour = palette[i % len(palette)]
                marker = MARKERS[i % len(MARKERS)]

                ax.plot(valid['Cycle'], valid['Per_N_Cycle_Loss_%'],
                       color=colour, marker=marker, markersize=MARKER_SIZE,
                       linestyle='', label=composition, zorder=3)
                plotted_any = True

            if plotted_any:
                ax.set_xlabel('Cycle number', fontsize=14)
                _force_integer_cycles(ax)
                ax.set_ylabel(f'Capacity loss per {ROLLING_WINDOW} cycles / %',
                              fontsize=14)
                ax.tick_params(axis='both', labelsize=12, width=1,
                               direction='in', top=True, right=True)
                ax.axhline(y=0, color='grey', linestyle=':',
                           linewidth=0.8, alpha=0.5)
                for sp in ax.spines.values():
                    sp.set_linewidth(0.8)
                ax.legend(fontsize=10, framealpha=0.7)
                plt.tight_layout()

                caption = (
                    f"Figure X. Capacity loss over a rolling window of "
                    f"{ROLLING_WINDOW} cycles for {comp_str}. Only "
                    f"datasets with {ROLLING_WINDOW}+ cycles are shown."
                )
                print(section("  Suggested caption"))
                print(bullet(caption, indent=2, label_width=2))

                if save_location:
                    fpath = os.path.join(save_location,
                                        f'capacity_loss_per_{ROLLING_WINDOW}_cycles.{_run_image_format(user_parameters)}')
                    fig.savefig(fpath, dpi=300, bbox_inches='tight')
                    saved(fpath)

                plt.show()
            else:
                plt.close(fig)
        else:
            print(f"\n  No datasets have {ROLLING_WINDOW}+ cycles — "
                  f"per-{ROLLING_WINDOW}-cycle plot skipped.")
    else:
        print("\nNo valid fade data to plot.")

    print(rule())


@_honours_verbose
def power_and_energy(electrochemical_data, user_parameters, *, save_location=None,
        all_cycle_tables=None, file_format='png', verbose=True):
    """
    Power and energy density, three normalisations.

    Ported from 1.8.7 Cell 10, wrapped in a function without editing
    the body. `all_cycle_tables` is `cycling_summary`'s output where the
    original read a notebook global; passing None reproduces the
    not-yet-run branch exactly.
    """
    _run_pe = any(
        user_parameters.get(name, {}).get('power_energy_analysis', True)
        for name in electrochemical_data
    )

    if not _run_pe:
        print("Power and energy density analysis skipped "
              "(disabled in Cell 3 parameters for all datasets).")
    else:
        # =====================================================================
        # USER-ADJUSTABLE DEFAULTS
        # =====================================================================
        DEFAULT_ELECTRODE_DIAMETER_MM = 12.0
        PLOT_PER_ACTIVE_MASS = True
        PLOT_PER_ELECTRODE_MASS = True
        PLOT_PER_AREA = True
        PLOT_RAGONE = True
        RAGONE_BASIS = 'active'  # 'active', 'electrode', or 'area'

        # Areal capacity plot (v1.7.2) — discharge capacity normalised to
        # electrode area. Exposes whether thick electrodes actually deliver
        # proportionally more capacity. Benchmark lines at commercial
        # cathode loadings (2-4 mAh/cm2). Ref: Cao et al., Nat. Nanotechnol.
        # 14, 200-207 (2019).
        PLOT_AREAL_CAPACITY = True

        # Areal capacity benchmarks — set per chemistry.
        # Li-ion NMC commercial: 3-5 mAh/cm2 at 15-25 mg/cm2
        #   (Xie et al., Electrochim. Acta 2023; QuantumScape 2023)
        # Na-ion layered oxide target: 2-3 mAh/cm2 at 20-25 mg/cm2
        #   (Liu et al., Nat. Energy 2025; Zheng et al., ScienceDirect 2023)
        _AREAL_BENCHMARKS = {
            'Li-ion': ([3.0, 5.0], 'Li-ion commercial'),
            'Na-ion': ([2.0, 3.0], 'Na-ion target'),
        }
        _AREAL_BENCHMARK_DEFAULT = ([2.0, 4.0], 'commercial target')

        # Exclude protocol-incomplete cycles (from Cell 5b)
        EXCLUDE_INCOMPLETE = True

        # Visual outlier filter — see Cell 9b for full rationale
        VISUAL_OUTLIER_FILTER = True
        # The filter default, read from the one definition rather than
        # restated: six local copies were quoted in the console message
        # while `_flag_visual_outliers` used its own default, so editing
        # one changed the sentence and not the data.
        VISUAL_OUTLIER_THRESHOLD = VISUAL_OUTLIER_FRACTION
        _have_cycle_tables = _has_tables(all_cycle_tables)
        if EXCLUDE_INCOMPLETE and not _have_cycle_tables:
            print("  ⚠ all_cycle_tables not found — Cell 5b has not been "
                  "run. Incomplete-cycle filtering unavailable.")

        # =====================================================================
        # HELPER FUNCTIONS
        # =====================================================================
        def _parse_active_fraction(blend_str):
            try:
                parts = [float(p) for p in blend_str.split('/')]
                return parts[0] / sum(parts) if sum(parts) > 0 else 0.8
            except (ValueError, AttributeError):
                return 0.8

        def _get_electrode_area(params):
            if 'electrode_area_cm2' in params:
                return params['electrode_area_cm2']
            diameter_mm = params.get('electrode_diameter_mm',
                                     DEFAULT_ELECTRODE_DIAMETER_MM)
            radius_cm = diameter_mm / 2 / 10
            return np.pi * radius_cm**2

        def _apply_pub_style(ax):
            ax.tick_params(axis='both', labelsize=11, width=0.8, direction='in',
                           top=True, right=True)
            for sp in ax.spines.values():
                sp.set_linewidth(0.8)

        def _set_lim_with_padding(ax, axis='y', bottom=0, padding=0.08):
            if axis == 'y':
                current_top = ax.get_ylim()[1]
                ax.set_ylim(bottom=bottom, top=current_top * (1 + padding))
            elif axis == 'x':
                current_right = ax.get_xlim()[1]
                ax.set_xlim(left=bottom, right=current_right * (1 + padding))

        def _extract_power_energy(df, params):
            """
            Per-cycle discharge energy and max power, vectorised via groupby.

            Takes the maximum of the absolute value of Spec. Energy(mWh/g)
            and Power(W) per cycle (the absolute handles sign conventions
            across different cyclers).
            """
            active_mass_mg = params.get('active_material_mass_mg')
            active_mass_g = (active_mass_mg / 1000.0
                            if active_mass_mg is not None else None)
            blend = params.get('blend', '80/10/10')
            active_fraction = params.get('active_fraction',
                                          _parse_active_fraction(blend))
            electrode_area = _get_electrode_area(params)
            total_coating_g = (active_mass_g / active_fraction
                              if active_mass_g is not None else None)

            df_discharge = df[df['Step'] == 'Discharge']
            if df_discharge.empty:
                return pd.DataFrame()

            has_spec_energy = 'Spec. Energy(mWh/g)' in df_discharge.columns
            has_power = 'Power(W)' in df_discharge.columns

            # Cell 2 has already coerced Cycle. Coerce the power/energy
            # columns here since they're not in Cell 2's coercion list.
            work = df_discharge[['Cycle']].copy()
            if has_spec_energy:
                work['_spec_energy_abs'] = pd.to_numeric(
                    df_discharge['Spec. Energy(mWh/g)'],
                    errors='coerce').abs()
            if has_power:
                work['_power_abs'] = pd.to_numeric(
                    df_discharge['Power(W)'], errors='coerce').abs()

            required = ['Cycle'] + [
                c for c in ('_spec_energy_abs', '_power_abs')
                if c in work.columns
            ]
            valid = work.dropna(subset=required)
            if valid.empty:
                return pd.DataFrame()

            # Single vectorised aggregation
            agg_dict = {}
            if has_spec_energy:
                agg_dict['_spec_energy_abs'] = 'max'
            if has_power:
                agg_dict['_power_abs'] = 'max'
            agg = valid.groupby('Cycle').agg(agg_dict).reset_index()
            agg['Cycle'] = agg['Cycle'].astype(int)

            # Derive the normalisation variants (vectorised)
            if has_spec_energy:
                agg['energy_Wh_kg_active'] = agg['_spec_energy_abs']
                if total_coating_g is not None:
                    agg['energy_Wh_kg_electrode'] = (
                        agg['_spec_energy_abs'] * active_fraction)
                if active_mass_g is not None:
                    agg['energy_mWh_cm2'] = (
                        agg['_spec_energy_abs'] * active_mass_g
                        / electrode_area)

            if has_power:
                agg['power_W'] = agg['_power_abs']
                if active_mass_g is not None:
                    agg['power_mW_g_active'] = (
                        agg['_power_abs'] / active_mass_g * 1000)
                if total_coating_g is not None:
                    agg['power_mW_g_electrode'] = (
                        agg['_power_abs'] / total_coating_g * 1000)
                agg['power_mW_cm2'] = (
                    agg['_power_abs'] / electrode_area * 1000)

            return agg.drop(columns=[c for c in ('_spec_energy_abs',
                                                 '_power_abs')
                                     if c in agg.columns])

        # =====================================================================
        # MAIN ANALYSIS
        # =====================================================================
        print(rule("POWER AND ENERGY DENSITY ANALYSIS"))

        all_pe_data = {}

        for name, df in electrochemical_data.items():
            params = user_parameters.get(name, {})
            if not params.get('power_energy_analysis', True):
                print(f"  {name}: power/energy analysis disabled, skipping.")
                continue

            composition = _get_display_name(name, params, user_parameters)
            pe_df = _extract_power_energy(df, params)

            if pe_df.empty:
                dch = _discharge_label(user_parameters.get(name, {}))
                print(f"  {name}: no {dch.lower()} power/energy data "
                      f"found, skipping.")
                continue

            # Exclude protocol-incomplete cycles (Cell 5b flag)
            if (EXCLUDE_INCOMPLETE and _have_cycle_tables
                and name in all_cycle_tables):
                ct = all_cycle_tables[name]
                incomplete_cycles = set(
                    _unusable_cycles(ct))
                if incomplete_cycles:
                    mask = pe_df['Cycle'].isin(incomplete_cycles)
                    removed = pe_df.loc[mask, 'Cycle'].tolist()
                    if removed:
                        print(f"  {composition}: excluded "
                              f"{len(removed)} protocol-incomplete "
                              f"cycle(s): "
                              f"{', '.join(str(c) for c in removed)}")
                    pe_df = pe_df[~mask].reset_index(drop=True)

            # Visual-outlier filter (using energy as the capacity proxy;
            # falls back to power if energy isn't available)
            if VISUAL_OUTLIER_FILTER and not pe_df.empty:
                proxy_col = None
                for candidate in ('energy_Wh_kg_active',
                                  'energy_mWh_cm2', 'power_W'):
                    if candidate in pe_df.columns:
                        proxy_col = candidate
                        break
                if proxy_col is not None:
                    outlier_flags = _flag_visual_outliers(
                        pe_df[proxy_col].tolist(), threshold=VISUAL_OUTLIER_THRESHOLD,
                        groups=rate_block_labels(pe_df['Cycle'], params))
                    n_outliers = sum(outlier_flags)
                    if n_outliers > 0:
                        removed = [int(c) for c, f in
                                   zip(pe_df['Cycle'], outlier_flags) if f]
                        print(f"  {composition}: excluded {n_outliers} "
                              f"visual outlier(s) "
                              f"(<{VISUAL_OUTLIER_THRESHOLD*100:.0f}% of "
                              f"running median on {proxy_col}): "
                              f"{', '.join(str(c) for c in removed)}")
                        pe_df = pe_df[
                            ~pd.Series(outlier_flags,
                                       index=pe_df.index)
                        ].reset_index(drop=True)

            if pe_df.empty:
                print(f"  {composition}: no valid cycles after filtering, "
                      "skipping.")
                continue

            all_pe_data[name] = pe_df

            active_mass = params.get('active_material_mass_mg', '?')
            blend = params.get('blend', '80/10/10')
            active_frac = params.get('active_fraction',
                                      _parse_active_fraction(blend))
            area = _get_electrode_area(params)
            loading = (float(active_mass) / area
                      if isinstance(active_mass, (int, float)) else None)

            print(heading(composition))
            print(f"    Active mass: {active_mass} mg | "
                  f"Blend: {blend} ({active_frac*100:.0f}% active)")
            print(f"    Electrode area: {area:.3f} cm2"
                  f"{f' | Loading: {loading:.2f} mg/cm2' if loading else ''}")
            print(f"    Cycles with data: {len(pe_df)}")
            if 'energy_Wh_kg_active' in pe_df.columns:
                print(f"    Energy range: "
                      f"{pe_df['energy_Wh_kg_active'].min():.1f}--"
                      f"{pe_df['energy_Wh_kg_active'].max():.1f} "
                      f"Wh/kg (active)")

        # =====================================================================
        # PLOTTING
        # =====================================================================
        if not all_pe_data:
            print("\nNo power/energy data available for any dataset.")
        else:
            # BY DATASET, so a cell keeps the same colour AND the same
            # marker shape it has in every other figure of the run. This was
            # `tab10` sampled continuously, which gave both a different
            # palette from the plots above it and, at two cells, a blue and a
            # cyan — the one pair a colourblind reader cannot separate.
            colours = _plots.categorical_colours(len(all_pe_data))
            markers = _plots.categorical_markers(len(all_pe_data))

            # v1.8: determine electrode type word for captions
            _etypes = set(
                user_parameters.get(n, {}).get('electrode_type', 'Positive')
                for n in all_pe_data)
            if _etypes == {'Negative'}:
                _mat_word = 'anode'
            elif _etypes == {'Positive'}:
                _mat_word = 'cathode'
            else:
                _mat_word = 'electrode'

            # --- Energy density vs cycle ---
            if PLOT_PER_ACTIVE_MASS:
                fig, ax = plt.subplots(figsize=(_plots.figure_width_inches, _plots.figure_height_inches))
                for i, (name, pe_df) in enumerate(all_pe_data.items()):
                    comp = _get_display_name(
                        name, user_parameters.get(name, {}), user_parameters)
                    if 'energy_Wh_kg_active' in pe_df.columns:
                        ax.plot(pe_df['Cycle'],
                               pe_df['energy_Wh_kg_active'],
                               color=colours[i], marker=markers[i], markersize=4,
                               linewidth=1.2, label=comp)
                ax.set_xlabel('Cycle number', fontsize=13)
                _force_integer_cycles(ax)
                ax.set_ylabel(
                    'Specific energy / '
                    'Wh kg$^{-1}_{\\mathrm{active}}$',
                    fontsize=13)
                ax.set_title(
                    'Specific energy vs. cycle (active material basis)',
                    fontsize=13)
                _apply_pub_style(ax)
                ax.legend(fontsize=9, framealpha=0.7)
                _set_lim_with_padding(ax, 'y', bottom=0)
                plt.tight_layout()
                if save_location:
                    fpath = os.path.join(save_location,
                        f'energy_per_active_mass_vs_cycle.{_run_image_format(user_parameters)}')
                    fig.savefig(fpath, dpi=300, bbox_inches='tight')
                    saved(fpath)
                plt.show()
                plt.close(fig)

            if PLOT_PER_ELECTRODE_MASS:
                fig, ax = plt.subplots(figsize=(_plots.figure_width_inches, _plots.figure_height_inches))
                for i, (name, pe_df) in enumerate(all_pe_data.items()):
                    comp = _get_display_name(
                        name, user_parameters.get(name, {}), user_parameters)
                    if 'energy_Wh_kg_electrode' in pe_df.columns:
                        ax.plot(pe_df['Cycle'],
                               pe_df['energy_Wh_kg_electrode'],
                               color=colours[i], marker=markers[i], markersize=4,
                               linewidth=1.2, label=comp)
                ax.set_xlabel('Cycle number', fontsize=13)
                _force_integer_cycles(ax)
                ax.set_ylabel(
                    'Specific energy / '
                    'Wh kg$^{-1}_{\\mathrm{electrode}}$',
                    fontsize=13)
                ax.set_title(
                    'Specific energy vs. cycle (electrode coating basis)',
                    fontsize=13)
                _apply_pub_style(ax)
                ax.legend(fontsize=9, framealpha=0.7)
                _set_lim_with_padding(ax, 'y', bottom=0)
                plt.tight_layout()
                if save_location:
                    fpath = os.path.join(save_location,
                        f'energy_per_electrode_mass_vs_cycle.{_run_image_format(user_parameters)}')
                    fig.savefig(fpath, dpi=300, bbox_inches='tight')
                    saved(fpath)
                plt.show()
                plt.close(fig)

            if PLOT_PER_AREA:
                fig, ax = plt.subplots(figsize=(_plots.figure_width_inches, _plots.figure_height_inches))
                for i, (name, pe_df) in enumerate(all_pe_data.items()):
                    comp = _get_display_name(
                        name, user_parameters.get(name, {}), user_parameters)
                    if 'energy_mWh_cm2' in pe_df.columns:
                        ax.plot(pe_df['Cycle'], pe_df['energy_mWh_cm2'],
                               color=colours[i], marker=markers[i], markersize=4,
                               linewidth=1.2, label=comp)
                ax.set_xlabel('Cycle number', fontsize=13)
                _force_integer_cycles(ax)
                ax.set_ylabel('Areal energy density / mWh cm$^{-2}$',
                              fontsize=13)
                ax.set_title('Areal energy density vs. cycle', fontsize=13)
                _apply_pub_style(ax)
                ax.legend(fontsize=9, framealpha=0.7)
                _set_lim_with_padding(ax, 'y', bottom=0)
                plt.tight_layout()
                if save_location:
                    fpath = os.path.join(save_location,
                        f'energy_per_area_vs_cycle.{_run_image_format(user_parameters)}')
                    fig.savefig(fpath, dpi=300, bbox_inches='tight')
                    saved(fpath)
                plt.show()
                plt.close(fig)

            # --- Power density vs cycle ---
            power_col_map = {
                'active': ('power_mW_g_active',
                           'mW g$^{-1}_{\\mathrm{active}}$',
                           'power_per_active_mass'),
                'electrode': ('power_mW_g_electrode',
                             'mW g$^{-1}_{\\mathrm{electrode}}$',
                             'power_per_electrode_mass'),
                'area': ('power_mW_cm2', 'mW cm$^{-2}$',
                         'power_per_area')
            }

            for basis, plot_flag in [('active', PLOT_PER_ACTIVE_MASS),
                                      ('electrode', PLOT_PER_ELECTRODE_MASS),
                                      ('area', PLOT_PER_AREA)]:
                if not plot_flag:
                    continue

                col, ylabel, fname = power_col_map[basis]

                fig, ax = plt.subplots(figsize=(_plots.figure_width_inches, _plots.figure_height_inches))
                for i, (name, pe_df) in enumerate(all_pe_data.items()):
                    comp = _get_display_name(
                        name, user_parameters.get(name, {}), user_parameters)
                    if col in pe_df.columns:
                        ax.plot(pe_df['Cycle'], pe_df[col],
                               color=colours[i], marker=markers[i], markersize=4,
                               linewidth=1.2, label=comp)

                ax.set_xlabel('Cycle number', fontsize=13)
                _force_integer_cycles(ax)
                ax.set_ylabel(f'Maximum power / {ylabel}', fontsize=13)
                basis_label = {'active': 'active material',
                              'electrode': 'electrode coating',
                              'area': 'electrode area'}[basis]
                ax.set_title(
                    f'Maximum power vs. cycle ({basis_label} basis)',
                    fontsize=13)
                _apply_pub_style(ax)
                ax.legend(fontsize=9, framealpha=0.7)
                _set_lim_with_padding(ax, 'y', bottom=0)
                plt.tight_layout()
                if save_location:
                    fpath = os.path.join(save_location,
                        f'{fname}_vs_cycle.{_run_image_format(user_parameters)}')
                    fig.savefig(fpath, dpi=300, bbox_inches='tight')
                    saved(fpath)
                plt.show()
                plt.close(fig)

            # --- Ragone plot ---
            if PLOT_RAGONE:
                energy_cols = {
                    'active': 'energy_Wh_kg_active',
                    'electrode': 'energy_Wh_kg_electrode',
                    'area': 'energy_mWh_cm2'
                }
                power_cols = {
                    'active': 'power_mW_g_active',
                    'electrode': 'power_mW_g_electrode',
                    'area': 'power_mW_cm2'
                }
                energy_labels = {
                    'active': 'Specific energy / '
                              'Wh kg$^{-1}_{\\mathrm{active}}$',
                    'electrode': 'Specific energy / '
                                 'Wh kg$^{-1}_{\\mathrm{electrode}}$',
                    'area': 'Areal energy / mWh cm$^{-2}$'
                }
                power_labels = {
                    'active': 'Power / '
                              'mW g$^{-1}_{\\mathrm{active}}$',
                    'electrode': 'Power / '
                                 'mW g$^{-1}_{\\mathrm{electrode}}$',
                    'area': 'Power / mW cm$^{-2}$'
                }

                e_col = energy_cols[RAGONE_BASIS]
                p_col = power_cols[RAGONE_BASIS]

                fig, ax = plt.subplots(figsize=(_plots.figure_width_inches, _plots.figure_height_inches))
                for i, (name, pe_df) in enumerate(all_pe_data.items()):
                    comp = _get_display_name(
                        name, user_parameters.get(name, {}), user_parameters)
                    key_cycles = user_parameters.get(name, {}).get(
                        'key_cycles', [])

                    if (e_col not in pe_df.columns or
                        p_col not in pe_df.columns):
                        continue

                    valid = pe_df.dropna(subset=[e_col, p_col])
                    if valid.empty:
                        continue

                    ax.scatter(valid[e_col], valid[p_col],
                              color=colours[i], s=40, alpha=0.6,
                              label=comp)

                    sorted_v = valid.sort_values('Cycle')
                    ax.plot(sorted_v[e_col], sorted_v[p_col],
                           color=colours[i], linewidth=0.8, alpha=0.4)

                    for cycle in key_cycles:
                        cd = valid[valid['Cycle'] == cycle]
                        if not cd.empty:
                            ax.annotate(
                                str(int(cycle)),
                                (cd[e_col].iloc[0], cd[p_col].iloc[0]),
                                textcoords='offset points', xytext=(6, 6),
                                fontsize=8, fontweight='bold',
                                color=colours[i], alpha=0.8
                            )

                ax.set_xlabel(energy_labels[RAGONE_BASIS], fontsize=13)
                ax.set_ylabel(power_labels[RAGONE_BASIS], fontsize=13)
                basis_note = {'active': 'active material',
                             'electrode': 'electrode coating',
                             'area': 'electrode area'}[RAGONE_BASIS]
                ax.set_title(
                    f'Ragone plot -- half-cell ({basis_note} basis)',
                    fontsize=13)
                _apply_pub_style(ax)
                ax.legend(fontsize=9, framealpha=0.7)
                _set_lim_with_padding(ax, 'x', bottom=0)
                _set_lim_with_padding(ax, 'y', bottom=0)
                plt.tight_layout()
                if save_location:
                    fpath = os.path.join(
                        save_location,
                        f'ragone_{RAGONE_BASIS}_basis.{_run_image_format(user_parameters)}')
                    fig.savefig(fpath, dpi=300, bbox_inches='tight')
                    saved(fpath)
                plt.show()
                plt.close(fig)

            # --- Areal capacity vs cycle (v1.7.2) ---
            if PLOT_AREAL_CAPACITY:
                # Calculate areal discharge capacity for each dataset
                all_areal_cap = {}
                for name, pe_df in all_pe_data.items():
                    params = user_parameters.get(name, {})
                    active_mass_mg = params.get('active_material_mass_mg')
                    area_cm2 = params.get('electrode_area_cm2')
                    if active_mass_mg is None or area_cm2 is None:
                        continue
                    active_mass_g = active_mass_mg / 1000.0

                    # Get discharge capacity per cycle from cycle tables
                    if (_have_cycle_tables and name in all_cycle_tables):
                        ct = all_cycle_tables[name]
                        ct_complete = ct[~_unusable(ct)]
                        areal_df = ct_complete[['Cycle', 'Discharge_mAh_g']].copy()
                        areal_df = areal_df.dropna(subset=['Discharge_mAh_g'])
                        # Apply visual outlier filter (same as Cells 8/9/10)
                        if VISUAL_OUTLIER_FILTER and not areal_df.empty:
                            _outlier_flags = _flag_visual_outliers(
                                areal_df['Discharge_mAh_g'].tolist(), threshold=VISUAL_OUTLIER_THRESHOLD,
                                groups=rate_block_labels(areal_df['Cycle'], params))
                            areal_df = areal_df[
                                ~pd.Series(_outlier_flags,
                                           index=areal_df.index)
                            ].reset_index(drop=True)
                        # Convert: mAh/g * g / cm2 = mAh/cm2
                        areal_df['Areal_mAh_cm2'] = (
                            areal_df['Discharge_mAh_g'] * active_mass_g
                            / area_cm2)
                        all_areal_cap[name] = areal_df
                    else:
                        # NO FALLBACK. This branch used to back-calculate
                        # capacity from specific energy divided by
                        # `voltage_upper_V * 0.85` — a hardcoded guess at the
                        # mean discharge voltage, plotted on an axis labelled
                        # mAh/cm2 with nothing to say it was not measured. On
                        # a 4.3 V layered oxide the guess is 3.66 V against a
                        # real mean near 3.8 V, a 4% error in every point.
                        # Areal capacity comes from the canonical cycle table
                        # or it is not plotted.
                        print(f"  {name}: no cycle table, so areal capacity "
                              f"is not plotted (it will not be estimated "
                              f"from energy and an assumed voltage).")

                if all_areal_cap:
                    fig, ax = plt.subplots(figsize=(_plots.figure_width_inches, _plots.figure_height_inches))
                    for i, (name, areal_df) in enumerate(
                            all_areal_cap.items()):
                        comp = _get_display_name(
                            name, user_parameters.get(name, {}),
                            user_parameters)
                        loading = user_parameters.get(name, {}).get(
                            'active_loading_mg_cm2')
                        loading_str = (f" [{loading:.1f} mg/cm²]"
                                      if loading else "")
                        ax.plot(areal_df['Cycle'],
                               areal_df['Areal_mAh_cm2'],
                               color=colours[i], marker=markers[i], markersize=4,
                               linewidth=1.2,
                               label=f"{comp}{loading_str}")

                    # Chemistry-aware benchmark lines (v1.7.2)
                    _chemistries = set(
                        user_parameters.get(n, {}).get('battery_chemistry', 'Unknown')
                        for n in all_areal_cap)
                    _chem = (_chemistries.pop() if len(_chemistries) == 1
                             else 'Unknown')
                    _bench_vals, _bench_label = _AREAL_BENCHMARKS.get(
                        _chem, _AREAL_BENCHMARK_DEFAULT)
                    _bench_annotations = []
                    for bench in _bench_vals:
                        ax.axhline(y=bench, color='grey', linestyle='--',
                                  linewidth=0.8, alpha=0.6)
                        _bench_annotations.append(bench)

                    ax.set_xlabel('Cycle number', fontsize=13)
                    _force_integer_cycles(ax)
                    ax.set_ylabel(
                        'Areal discharge capacity / mAh cm$^{-2}$',
                        fontsize=13)
                    ax.set_title(
                        'Areal discharge capacity vs. cycle',
                        fontsize=13)
                    _apply_pub_style(ax)
                    ax.legend(fontsize=9, framealpha=0.7)
                    ax.set_ylim(bottom=0)
                    _set_lim_with_padding(ax, 'y', bottom=0)
                    # Add benchmark annotations after axis limits are final
                    for bench in _bench_annotations:
                        ax.annotate(
                            f'{bench:.0f} mAh/cm² ({_bench_label})',
                            xy=(ax.get_xlim()[1] * 0.02, bench),
                            xytext=(5, 3), textcoords='offset points',
                            fontsize=8, color='grey', alpha=0.8,
                            fontstyle='italic')
                    plt.tight_layout()

                    # Caption
                    loading_range = [
                        user_parameters.get(n, {}).get(
                            'active_loading_mg_cm2')
                        for n in all_areal_cap
                    ]
                    loading_range = [l for l in loading_range
                                    if l is not None]
                    if loading_range:
                        lo_str = (f"Active material loadings: "
                                 f"{min(loading_range):.1f}--"
                                 f"{max(loading_range):.1f} mg/cm².")
                    else:
                        lo_str = ""
                    caption = (
                        f"Figure X. Areal discharge capacity for all "
                        f"datasets, normalised to electrode geometric "
                        f"area. Dashed grey lines indicate the "
                        f"{_bench_label} range "
                        f"({_bench_vals[0]:.0f}--"
                        f"{_bench_vals[-1]:.0f} "
                        f"mAh/cm²). {lo_str} "
                        f"Half-cell ({_mat_word}) data."
                    )
                    print(section("  Suggested caption"))
                    print(bullet(caption, indent=2, label_width=2))

                    if save_location:
                        fpath = os.path.join(save_location,
                            f'areal_discharge_capacity_vs_cycle.{_run_image_format(user_parameters)}')
                        fig.savefig(fpath, dpi=300, bbox_inches='tight')
                        saved(fpath)
                    plt.show()
                    plt.close(fig)

                    # Print areal capacity summary
                    print(f"\n  Areal capacity summary (cycle 2):")
                    print(f"  {'Composition':<30} {'Loading':>10} "
                          f"{'Q_areal':>10}")
                    print(f"  {'':30} {'mg/cm²':>10} "
                          f"{'mAh/cm²':>10}")
                    print(f"  {'-'*52}")
                    for name, areal_df in all_areal_cap.items():
                        comp = _get_display_name(
                            name, user_parameters.get(name, {}),
                            user_parameters)
                        loading = user_parameters.get(name, {}).get(
                            'active_loading_mg_cm2')
                        c2 = areal_df[areal_df['Cycle'] == 2]
                        if not c2.empty:
                            q_ar = c2['Areal_mAh_cm2'].iloc[0]
                            l_str = (f"{loading:.2f}"
                                    if loading else '--')
                            print(f"  {comp:<30} {l_str:>10} "
                                  f"{q_ar:>10.3f}")
                    print(f"  {'-'*52}")
                else:
                    print("  Areal capacity: insufficient data "
                          "(need active mass and electrode area).")

            # --- Summary table ---
            print()
            print(f"  Power, Energy & Areal Capacity Summary (final cycle)")
            print(f"  {'Composition':<28} {'E/Wh kg':>10} "
                  f"{'E/Wh kg':>10} {'E/mWh cm2':>11} "
                  f"{'P/mW g':>10} {'P/mW cm2':>10} {'Q_areal':>10} {'Cyc':>5}")
            print(f"  {'':28} {'(active)':>10} {'(electr.)':>10} "
                  f"{'(areal)':>11} {'(active)':>10} {'(areal)':>10} "
                  f"{'mAh/cm2':>10} {'':>5}")
            print(f"  {'-'*95}")
            for name, pe_df in all_pe_data.items():
                comp = _get_display_name(
                    name, user_parameters.get(name, {}), user_parameters)
                last = pe_df.iloc[-1]
                def _fmt(val, fmt='.1f'):
                    return f"{val:{fmt}}" if pd.notna(val) else '--'
                e_act = _fmt(last.get('energy_Wh_kg_active'))
                e_elec = _fmt(last.get('energy_Wh_kg_electrode'))
                e_area = _fmt(last.get('energy_mWh_cm2'), '.3f')
                p_act = _fmt(last.get('power_mW_g_active'))
                p_area = _fmt(last.get('power_mW_cm2'), '.3f')
                # Areal capacity from final cycle
                q_areal = '--'
                if (PLOT_AREAL_CAPACITY and 'all_areal_cap' in dir()
                    and name in all_areal_cap):
                    ac_df = all_areal_cap[name]
                    last_ac = ac_df.iloc[-1] if not ac_df.empty else None
                    if last_ac is not None:
                        q_areal = f"{last_ac['Areal_mAh_cm2']:.3f}"
                # Final cycle number
                last_cyc = str(int(last['Cycle'])) if 'Cycle' in last else '--'
                print(f"  {comp:<28} {e_act:>10} {e_elec:>10} "
                      f"{e_area:>11} {p_act:>10} {p_area:>10} "
                      f"{q_areal:>10} {last_cyc:>5}")
            print(f"  {'-'*95}")
            print(f"  Note: Half-cell data (vs metal counter electrode).")
            print(f"  Energy and power reflect {_mat_word} material only.")

            # --- Export ---
            for name, pe_df in all_pe_data.items():
                if save_location:
                    fpath = os.path.join(
                        save_location,
                        f'{name}_power_energy_data.csv')
                    pe_df.to_csv(fpath, index=False)
                    saved(fpath)

        print(rule())


@_honours_verbose
def average_discharge_voltage(electrochemical_data, user_parameters, *, save_location=None,
        all_cycle_tables=None, file_format='png', verbose=True):
    """
    Mean discharge voltage against cycle number.

    Ported from 1.8.7 Cell 11, wrapped in a function without editing
    the body. `all_cycle_tables` is `cycling_summary`'s output where the
    original read a notebook global; passing None reproduces the
    not-yet-run branch exactly.
    """
    MARKER_SIZE = 6

    # The filter default, read from the one definition rather than

    # restated: six local copies were quoted in the console message

    # while `_flag_visual_outliers` used its own default, so editing

    # one changed the sentence and not the data.

    VISUAL_OUTLIER_THRESHOLD = VISUAL_OUTLIER_FRACTION
    def _calculate_avg_voltage_per_cycle(df):
        """
        Energy-weighted average discharge voltage per cycle:
            V_avg = integral(V * dQ) / integral(dQ)

        Integration is `compat.trapezoid`, which is numpy's under whichever
        name that version of numpy uses. NOT scipy, whatever this docstring
        said until 1.9.0.15.
        """
        # Cell 2 has already coerced the numeric columns.
        discharge = df[df['Step'] == 'Discharge'].dropna(
            subset=['Cycle', 'Voltage', 'Discharge_Capacity'])

        if discharge.empty:
            return pd.DataFrame(columns=['Cycle', 'Avg_Discharge_Voltage_V',
                                          'Discharge_mAh_g'])

        # ONE DEFINITION. `mean_discharge_voltage` at the top of this module
        # was written to be the single definition of this quantity — its
        # docstring says so, and names the two divergent copies that preceded
        # it — and then nothing ever called it, so this function kept a third.
        # The two agreed, which is exactly why it survived: a duplicate that
        # matches is invisible until someone edits one of them. This body is
        # now the module-level function plus the delivered capacity, which is
        # the only thing the figure needs that the canonical one does not
        # return.
        base = mean_discharge_voltage(
            discharge.assign(Step='Discharge'))
        if base.empty:
            return pd.DataFrame(columns=['Cycle', 'Avg_Discharge_Voltage_V',
                                         'Discharge_mAh_g'])
        deliv = {}
        for cycle, cd in discharge.groupby('Cycle'):
            q = pd.to_numeric(cd['Discharge_Capacity'],
                              errors='coerce').dropna()
            if q.size < 3:
                continue
            span = float(q.max() - q.min())
            if span > 0:
                deliv[int(cycle)] = span
        base = base[base['Cycle'].isin(deliv)].copy()
        base['Discharge_mAh_g'] = base['Cycle'].map(deliv)
        return base.reset_index(drop=True)

    _avg_v_labels = set(
        _discharge_label(user_parameters.get(n, {}))
        for n in electrochemical_data)

    _avg_v_word = (_avg_v_labels.pop().lower() if len(_avg_v_labels) == 1
                   else 'discharge')

    print(rule(f"AVERAGE {_avg_v_word.upper()} VOLTAGE"))

    palette = MULTI_DATASET_PALETTE

    _have_cycle_tables = _has_tables(all_cycle_tables)

    if EXCLUDE_INCOMPLETE and not _have_cycle_tables:
        print("  ⚠ all_cycle_tables not found — Cell 5b has not been "
              "run. Protocol-incomplete filtering unavailable; "
              "visual-outlier filter still active.\n")

    all_avg_voltage = {}

    for name, df in electrochemical_data.items():
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)

        avg_v_df = _calculate_avg_voltage_per_cycle(df)

        if avg_v_df.empty:
            print(f"  {composition}: no valid {_avg_v_word} data, skipping.")
            continue

        # Exclude protocol-incomplete cycles (Cell 5b flag)
        if (EXCLUDE_INCOMPLETE and _have_cycle_tables
            and name in all_cycle_tables):
            ct = all_cycle_tables[name]
            incomplete_cycles = set(
                _unusable_cycles(ct))
            if incomplete_cycles:
                mask = avg_v_df['Cycle'].isin(incomplete_cycles)
                removed = avg_v_df.loc[mask, 'Cycle'].tolist()
                if removed:
                    print(f"  {composition}: excluded {len(removed)} "
                          f"protocol-incomplete cycle(s): "
                          f"{', '.join(str(int(c)) for c in removed)}")
                avg_v_df = avg_v_df[~mask].reset_index(drop=True)

        # Visual-outlier filter
        if VISUAL_OUTLIER_FILTER and not avg_v_df.empty:
            outlier_flags = _flag_visual_outliers(
                avg_v_df['Discharge_mAh_g'].tolist(), threshold=VISUAL_OUTLIER_THRESHOLD,
                groups=rate_block_labels(avg_v_df['Cycle'], params))
            n_outliers = sum(outlier_flags)
            if n_outliers > 0:
                removed = [int(c) for c, f in
                           zip(avg_v_df['Cycle'], outlier_flags) if f]
                print(f"  {composition}: excluded {n_outliers} visual "
                      f"outlier(s) (<{VISUAL_OUTLIER_THRESHOLD*100:.0f}% "
                      f"of running median on Discharge_mAh_g): "
                      f"{', '.join(str(c) for c in removed)}")
                avg_v_df = avg_v_df[
                    ~pd.Series(outlier_flags, index=avg_v_df.index)
                ].reset_index(drop=True)

        if avg_v_df.empty:
            continue

        all_avg_voltage[name] = avg_v_df

        # Voltage drift summary
        ref_row = avg_v_df[avg_v_df['Cycle'] == DRIFT_REFERENCE_CYCLE]
        if not ref_row.empty:
            ref_v = ref_row['Avg_Discharge_Voltage_V'].iloc[0]
            last_v = avg_v_df['Avg_Discharge_Voltage_V'].iloc[-1]
            last_cycle = int(avg_v_df['Cycle'].iloc[-1])
            drift_mV = (last_v - ref_v) * 1000
            n_cycles = last_cycle - DRIFT_REFERENCE_CYCLE
            drift_rate = drift_mV / n_cycles if n_cycles > 0 else 0

            print(f"  {composition}: V_avg = {ref_v:.4f} V (cycle "
                  f"{DRIFT_REFERENCE_CYCLE}) -> {last_v:.4f} V (cycle "
                  f"{last_cycle})")
            print(f"    Total drift: {drift_mV:+.1f} mV over {n_cycles} "
                  f"cycles ({drift_rate:+.2f} mV/cycle)")
        else:
            print(f"  {composition}: {len(avg_v_df)} cycles analysed, "
                  f"V_avg range {avg_v_df['Avg_Discharge_Voltage_V'].min():.4f}"
                  f"--{avg_v_df['Avg_Discharge_Voltage_V'].max():.4f} V")

    if all_avg_voltage:
        fig, ax = plt.subplots(figsize=(_plots.figure_width_inches, _plots.figure_height_inches))

        caption_compositions = []

        for i, (name, avdf) in enumerate(all_avg_voltage.items()):
            composition = _get_display_name(name, user_parameters.get(name, {}), user_parameters)
            colour = palette[i % len(palette)]
            marker = MARKERS[i % len(MARKERS)]

            ax.plot(avdf['Cycle'], avdf['Avg_Discharge_Voltage_V'],
                   color=colour, marker=marker, markersize=MARKER_SIZE,
                   linestyle='-', linewidth=0.8, label=composition, zorder=3)

            caption_compositions.append(composition)

        ax.set_xlabel('Cycle number', fontsize=14)
        _force_integer_cycles(ax)
        ax.set_ylabel(f'Average {_avg_v_word} voltage / V', fontsize=14)
        ax.tick_params(axis='both', labelsize=12, width=1, direction='in',
                       top=True, right=True)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)

        # Y-axis scaling
        if VOLTAGE_YMIN is not None and VOLTAGE_YMAX is not None:
            ax.set_ylim(bottom=VOLTAGE_YMIN, top=VOLTAGE_YMAX)
        else:
            y_lo, y_hi = ax.get_ylim()
            ax.set_ylim(bottom=y_lo - VOLTAGE_YAXIS_PADDING,
                        top=y_hi + VOLTAGE_YAXIS_PADDING)

        ax.set_xlim(left=0)
        ax.legend(fontsize=10, framealpha=0.7)
        plt.tight_layout()

        # Caption
        comp_str = ', '.join(caption_compositions[:-1])
        if len(caption_compositions) > 1:
            comp_str += f' and {caption_compositions[-1]}'
        else:
            comp_str = caption_compositions[0]

        charge_rate = user_parameters.get(
            list(all_avg_voltage.keys())[0], {}
        ).get('charge_rate_c', 'the specified')

        caption = (
            f"Figure X. Energy-weighted average {_avg_v_word} voltage "
            f"as a function of cycle number for {comp_str}, cycled at "
            f"{charge_rate} C. A declining average voltage indicates "
            f"increasing polarisation or structural degradation, even "
            f"if {_avg_v_word} capacity is maintained."
        )
        print(section("  Suggested caption"))
        print(bullet(caption, indent=2, label_width=2))

        # Save
        if save_location:
            filepath = os.path.join(save_location,
                                   f'average_discharge_voltage_vs_cycle.{_run_image_format(user_parameters)}')
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            saved(filepath)

        plt.show()
        plt.close(fig)

        # Export
        if save_location:
            for name, avdf in all_avg_voltage.items():
                fpath = os.path.join(save_location,
                                    f'{name}_avg_{_avg_v_word}_voltage.csv')
                avdf.to_csv(fpath, index=False)
                saved(fpath)
    else:
        print("\nNo valid data for average voltage analysis.")

    print(rule())


@_honours_verbose
def energy_efficiency(electrochemical_data, user_parameters, *, save_location=None,
        all_cycle_tables=None, file_format='png', verbose=True):
    """
    Round-trip energy efficiency against cycle number.

    Ported from 1.8.7 Cell 12, wrapped in a function without editing
    the body. `all_cycle_tables` is `cycling_summary`'s output where the
    original read a notebook global; passing None reproduces the
    not-yet-run branch exactly.
    """
    MARKER_SIZE = 6

    YAXIS_MIN = 40

    YAXIS_MAX = 105

    # The filter default, read from the one definition rather than

    # restated: six local copies were quoted in the console message

    # while `_flag_visual_outliers` used its own default, so editing

    # one changed the sentence and not the data.

    VISUAL_OUTLIER_THRESHOLD = VISUAL_OUTLIER_FRACTION
    def _calculate_energy_per_cycle(df):
        """
        Per-cycle charge and discharge ENERGY, by trapezoidal integration of
        V dQ, plus the efficiencies derived from it.

        The CAPACITIES AND CE ARE NOT COMPUTED HERE. This function used to
        derive its own, with its own admission rule (three points, no
        integrity filter), so `energy_efficiency`'s CE could and did differ
        from `cycle_life`'s for the same cycle of the same cell with nothing
        in the run to say so. They now come from `delivered_capacity` and
        `coulombic_efficiency` — the one definition — and the mean voltages
        are energy / that capacity, which is the standard energy-weighted
        definition and now shares a denominator with everything else.

        Integration is `compat.trapezoid` — numpy's, under whichever name
        that version of numpy uses. NOT scipy, whatever this docstring said
        until 1.9.0.15.
        """
        # Cell 2 has already coerced the numeric columns; we only need to
        # drop rows with missing Cycle/Voltage/Step for the integration.
        df_work = df.dropna(subset=['Cycle', 'Voltage', 'Step'])

        rows = []
        for cycle in sorted(df_work['Cycle'].unique()):
            cd = df_work[df_work['Cycle'] == cycle]
            row = {'Cycle': int(cycle)}

            # --- Discharge energy ---
            dc = cd[
                cd['Step'] == 'Discharge'
            ].dropna(subset=['Discharge_Capacity', 'Voltage'])

            if len(dc) >= 3:
                dc_sorted = dc.sort_values('Discharge_Capacity')
                Q_d = dc_sorted['Discharge_Capacity'].values
                V_d = dc_sorted['Voltage'].values

                row['Discharge_Energy'] = abs(trapezoid(V_d, Q_d))
            else:
                row['Discharge_Energy'] = np.nan

            # --- Charge energy ---
            cc = cd[
                cd['Step'] == 'Charge'
            ].dropna(subset=['Charge_Capacity', 'Voltage'])

            if len(cc) >= 3:
                cc_sorted = cc.sort_values('Charge_Capacity')
                Q_c = cc_sorted['Charge_Capacity'].values
                V_c = cc_sorted['Voltage'].values

                row['Charge_Energy'] = abs(trapezoid(V_c, Q_c))
            else:
                row['Charge_Energy'] = np.nan

            if (pd.notna(row['Charge_Energy']) and
                pd.notna(row['Discharge_Energy']) and
                row['Charge_Energy'] > 0):
                row['Energy_Efficiency_%'] = (
                    row['Discharge_Energy'] / row['Charge_Energy'] * 100
                )
            else:
                row['Energy_Efficiency_%'] = np.nan

            rows.append(row)

        out = pd.DataFrame(rows)
        if out.empty:
            return out

        # --- capacity and CE from the ONE definition ---------------------
        cap = delivered_capacity(df)
        out = out.merge(cap, on='Cycle', how='left')
        for c in ('Charge_mAh_g', 'Discharge_mAh_g'):
            if c not in out.columns:
                out[c] = np.nan
        out['CE_%'] = coulombic_efficiency(out['Discharge_mAh_g'],
                                           out['Charge_mAh_g']).values

        # --- mean voltages: energy / that same capacity ------------------
        # Voltage efficiency decomposes the energy loss: EE ~ CE * VE / 100.
        # CE tells you about material loss, VE about polarisation.
        # Ref: Cao et al., Nat. Nanotechnol. 14, 200-207 (2019)
        qc = pd.to_numeric(out['Charge_mAh_g'], errors='coerce')
        qd = pd.to_numeric(out['Discharge_mAh_g'], errors='coerce')
        out['Avg_V_Charge'] = (out['Charge_Energy'] / qc).where(qc > 0)
        out['Avg_Discharge_Voltage_V'] = (
            out['Discharge_Energy'] / qd).where(qd > 0)
        out['Voltage_Efficiency_%'] = (
            out['Avg_Discharge_Voltage_V'] / out['Avg_V_Charge'] * 100
        ).where(out['Avg_V_Charge'] > 0)

        out['CE_EE_Gap_%'] = out['CE_%'] - out['Energy_Efficiency_%']
        return out

    print(rule("ENERGY EFFICIENCY ANALYSIS"))

    palette_multi = MULTI_DATASET_PALETTE

    markers_multi = MULTI_DATASET_MARKERS

    _have_cycle_tables = _has_tables(all_cycle_tables)

    if EXCLUDE_INCOMPLETE and not _have_cycle_tables:
        print("  ⚠ all_cycle_tables not found — Cell 5b has not been "
              "run. Protocol-incomplete filtering unavailable; "
              "visual-outlier filter still active.")

    all_ee_data = {}

    for name, df in electrochemical_data.items():
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        # 1.8.7 Cell 12 read `is_anode` here (in its >100% EE notice) but
        # only ASSIGNED it further down the same loop. On the first dataset
        # it therefore fell through to whatever Cell 11 had left in globals,
        # and on every later one it carried the PREVIOUS dataset's flag.
        # Binding it at the top of the loop is what the later line does, one
        # iteration too late.
        is_anode = params.get('anode_labels_swapped', False)

        ee_df = _calculate_energy_per_cycle(df)

        if ee_df.empty:
            print(f"  {composition}: no valid data, skipping.")
            continue

        # Exclude protocol-incomplete cycles (Cell 5b flag)
        if (EXCLUDE_INCOMPLETE and _have_cycle_tables
            and name in all_cycle_tables):
            ct = all_cycle_tables[name]
            incomplete_cycles = set(
                _unusable_cycles(ct))
            if incomplete_cycles:
                mask = ee_df['Cycle'].isin(incomplete_cycles)
                removed = ee_df.loc[mask, 'Cycle'].tolist()
                if removed:
                    print(f"  {composition}: excluded {len(removed)} "
                          f"protocol-incomplete cycle(s): "
                          f"{', '.join(str(int(c)) for c in removed)}")
                ee_df = ee_df[~mask].reset_index(drop=True)

        # Visual-outlier filter
        if VISUAL_OUTLIER_FILTER and not ee_df.empty:
            outlier_flags = _flag_visual_outliers(
                ee_df['Discharge_mAh_g'].tolist(), threshold=VISUAL_OUTLIER_THRESHOLD,
                groups=rate_block_labels(ee_df['Cycle'], params))
            n_outliers = sum(outlier_flags)
            if n_outliers > 0:
                removed = [int(c) for c, f in
                           zip(ee_df['Cycle'], outlier_flags) if f]
                print(f"  {composition}: excluded {n_outliers} visual "
                      f"outlier(s) (<{VISUAL_OUTLIER_THRESHOLD*100:.0f}% "
                      f"of running median on Discharge_mAh_g): "
                      f"{', '.join(str(c) for c in removed)}")
                ee_df = ee_df[
                    ~pd.Series(outlier_flags, index=ee_df.index)
                ].reset_index(drop=True)

        if ee_df.empty:
            continue

        # Flag any EE > 100% (physically suspicious)
        over_100 = ee_df[ee_df['Energy_Efficiency_%'] > 100]
        if not over_100.empty:
            n_over = len(over_100)
            n_total = len(ee_df)
            if n_over == n_total:
                print(f"  {composition}: EE >100% in all {n_total} cycles"
                      f" (expected for anode half-cells)"
                      if is_anode else
                      f"  {composition}: EE >100% in all {n_total} cycles"
                      f" (check for measurement artefact)")
            elif n_over > 5:
                note = ("expected for anode half-cells" if is_anode
                        else "likely formation effect")
                print(f"  {composition}: EE >100% in {n_over}/{n_total} "
                      f"cycles ({note})")
            else:
                print(f"  {composition}: EE >100% in cycle(s) "
                      f"{', '.join(str(int(c)) for c in over_100['Cycle'])} "
                      f"(likely formation effect)")

        all_ee_data[name] = ee_df

        # Summary (excluding formation)
        stable = ee_df[ee_df['Cycle'] >= EE_START_CYCLE]
        if not stable.empty:
            mean_ee = stable['Energy_Efficiency_%'].mean()
            std_ee = stable['Energy_Efficiency_%'].std()
            mean_ce = stable['CE_%'].mean()
            mean_gap = stable['CE_EE_Gap_%'].mean()

            print()
            print(heading(composition))
            print(f"    Energy efficiency (cycles {EE_START_CYCLE}+): "
                  f"{mean_ee:.2f}% (+/-{std_ee:.2f}%)")
            if is_anode and mean_ee > 100:
                print(f"    Note: EE >100% is expected for anode half-cells — "
                      f"delithiation (discharge) voltage is higher than "
                      f"lithiation (charge) voltage, so E_out > E_in.")
                print(f"    This is not an error; the energy balance is closed "
                      f"by the Li/Na metal counter electrode.")
            print(f"    Coulombic efficiency (cycles {EE_START_CYCLE}+): "
                  f"{mean_ce:.2f}%")
            is_anode = params.get('anode_labels_swapped', False)
            if 'Voltage_Efficiency_%' in stable.columns:
                mean_ve = stable['Voltage_Efficiency_%'].mean()
                if is_anode:
                    abs_hysteresis = abs(100 - mean_ve)
                    print(f"    Voltage ratio (cycles {EE_START_CYCLE}+): "
                          f"{mean_ve:.2f}%")
                    print(f"    Voltage hysteresis:   "
                          f"|1 - V_ratio| = {abs_hysteresis:.2f}%")
                    print(f"    Note: For anodes, V_ratio > 100% is expected "
                          f"(the useful half-cycle occurs at higher voltage).")
                    print(f"    The deviation from 100% reflects the same "
                          f"polarisation loss as in cathodes.")
                else:
                    print(f"    Voltage efficiency (cycles {EE_START_CYCLE}+): "
                          f"{mean_ve:.2f}%")
                    print(f"    Loss decomposition:  "
                          f"CE loss = {100 - mean_ce:.2f}% (material)  |  "
                          f"VE loss = {100 - mean_ve:.2f}% (polarisation)")
            abs_gap = abs(mean_gap)
            gap_sign = '' if mean_gap >= 0 else ' (note: negative for anodes)'
            print(f"    |CE - EE| gap: {abs_gap:.2f}% "
                  f"(voltage hysteresis contribution){gap_sign}")

    if all_ee_data:
        for name, ee_df in all_ee_data.items():
            params = user_parameters.get(name, {})
            composition = _get_display_name(name, params, user_parameters)
            charge_rate_c = params.get('charge_rate_c', 'Unknown')

            # Filter to stable cycling (skip formation)
            plot_df = ee_df[ee_df['Cycle'] >= EE_START_CYCLE]

            if plot_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(_plots.figure_width_inches, _plots.figure_height_inches))

            # CE
            ax.plot(plot_df['Cycle'], plot_df['CE_%'],
                   color=COLOUR_CE, marker=MARKERS_CE,
                   markersize=MARKER_SIZE, linestyle='-',
                   linewidth=0.8, label='Coulombic efficiency', zorder=3)

            # Energy efficiency
            ax.plot(plot_df['Cycle'], plot_df['Energy_Efficiency_%'],
                   color=COLOUR_EE, marker=MARKERS_EE,
                   markersize=MARKER_SIZE, linestyle='-',
                   linewidth=0.8, label='Energy efficiency', zorder=3)

            # Voltage efficiency / ratio (v1.8: adaptive label for anodes)
            is_anode = params.get('anode_labels_swapped', False)
            if 'Voltage_Efficiency_%' in plot_df.columns:
                _ve_legend = ('Voltage ratio' if is_anode
                              else 'Voltage efficiency')
                ax.plot(plot_df['Cycle'], plot_df['Voltage_Efficiency_%'],
                       color='#009E73', marker='s',
                       markersize=MARKER_SIZE - 1, linestyle='-',
                       linewidth=0.8, label=_ve_legend,
                       zorder=3, alpha=0.8)

            # Shade the gap
            ax.fill_between(
                plot_df['Cycle'],
                plot_df['Energy_Efficiency_%'],
                plot_df['CE_%'],
                alpha=0.10, color='grey',
                label='Voltage hysteresis loss'
            )

            # 100% reference line
            ax.axhline(y=100, color='grey', linestyle=':',
                        linewidth=0.8, alpha=0.5)

            ax.set_xlabel('Cycle number', fontsize=14)
            _force_integer_cycles(ax)
            ax.set_ylabel('Efficiency / %', fontsize=14)
            ax.tick_params(axis='both', labelsize=12, width=1,
                           direction='in', top=True, right=True)
            for sp in ax.spines.values():
                sp.set_linewidth(0.8)

            # Y-axis (v1.8: extend for anode VE > 100%)
            y_lo = min(YAXIS_MIN,
                       plot_df['Energy_Efficiency_%'].min() - 2)
            y_hi = max(YAXIS_MAX,
                       plot_df['CE_%'].max() + 2)
            if ('Voltage_Efficiency_%' in plot_df.columns
                    and plot_df['Voltage_Efficiency_%'].notna().any()):
                y_hi = max(y_hi,
                           plot_df['Voltage_Efficiency_%'].max() + 5)
            # A KNOB THAT DID NOTHING. `COMMON_YAXIS_EFFICIENCY` sat beside
            # `COMMON_YAXIS_CAPACITY`, which works, and was read nowhere — so
            # a user turning it off got the fixed floor and ceiling anyway.
            # Off now means the axis follows the data, which is what its
            # neighbour's name promises.
            if COMMON_YAXIS_EFFICIENCY:
                ax.set_ylim(bottom=y_lo, top=y_hi)
            ax.set_xlim(left=0)

            ax.legend(fontsize=10, framealpha=0.7)
            plt.tight_layout()

            # Caption (v1.8: absolute gap for anodes)
            mean_gap = plot_df['CE_EE_Gap_%'].mean()
            abs_gap = abs(mean_gap)
            chg_word = _charge_label(params).lower()
            caption = (
                f"Figure X. Coulombic efficiency (blue circles) and "
                f"energy efficiency (vermillion diamonds) for "
                f"{composition} cycled at {rate_phrase(params)}. "
                f"The shaded region represents the energy lost to "
                f"voltage hysteresis, averaging {abs_gap:.1f}% of "
                f"the total {chg_word} energy. Formation cycle excluded."
            )
            print(section("  Suggested caption"))
            print(bullet(caption, indent=2, label_width=2))

            # Save
            if save_location:
                filepath = os.path.join(
                    save_location,
                    f'{name}_energy_efficiency.{image_format(params)}')
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                saved(filepath)

            plt.show()
            plt.close(fig)


        # --- Comparative plot (if multiple datasets) ---
        if len(all_ee_data) > 1:
            fig, ax = plt.subplots(figsize=(_plots.figure_width_inches, _plots.figure_height_inches))

            caption_compositions = []

            for i, (name, ee_df) in enumerate(all_ee_data.items()):
                composition = _get_display_name(
                    name, user_parameters.get(name, {}), user_parameters)
                colour = palette_multi[i % len(palette_multi)]
                marker = markers_multi[i % len(markers_multi)]

                stable = ee_df[ee_df['Cycle'] >= EE_START_CYCLE]
                if stable.empty:
                    continue

                ax.plot(stable['Cycle'], stable['Energy_Efficiency_%'],
                       color=colour, marker=marker,
                       markersize=MARKER_SIZE, linestyle='-',
                       linewidth=0.8, label=composition, zorder=3)

                caption_compositions.append(composition)

            ax.axhline(y=100, color='grey', linestyle=':',
                        linewidth=0.8, alpha=0.5)

            ax.set_xlabel('Cycle number', fontsize=14)
            _force_integer_cycles(ax)
            ax.set_ylabel('Energy efficiency / %', fontsize=14)
            ax.tick_params(axis='both', labelsize=12, width=1,
                           direction='in', top=True, right=True)
            for sp in ax.spines.values():
                sp.set_linewidth(0.8)
            ax.set_xlim(left=0)
            # Auto-extend y-axis for anode data where EE > 100%
            _all_ee_vals = []
            for _ee_df in all_ee_data.values():
                _s = _ee_df[_ee_df['Cycle'] >= EE_START_CYCLE]
                if not _s.empty:
                    _all_ee_vals.extend(_s['Energy_Efficiency_%'].dropna().tolist())
            if _all_ee_vals:
                y_hi = max(YAXIS_MAX, max(_all_ee_vals) + 5)
                y_lo = min(YAXIS_MIN, min(_all_ee_vals) - 2)
            else:
                y_hi = YAXIS_MAX
                y_lo = YAXIS_MIN
            ax.set_ylim(bottom=y_lo, top=y_hi)
            ax.legend(fontsize=10, framealpha=0.7)
            plt.tight_layout()

            comp_str = ', '.join(caption_compositions[:-1])
            if len(caption_compositions) > 1:
                comp_str += f' and {caption_compositions[-1]}'
            else:
                comp_str = caption_compositions[0]

            # Check if any dataset is an anode
            _any_anode = any(
                user_parameters.get(n, {}).get('anode_labels_swapped', False)
                for n in all_ee_data)
            _anode_note = (" Note: anode half-cell EE exceeds 100% because "
                           "the useful half-cycle occurs at higher voltage "
                           "than the insertion half-cycle."
                           if _any_anode else "")
            caption = (
                f"Figure X. Comparative energy efficiency for "
                f"{comp_str}. Formation cycle excluded.{_anode_note}"
            )
            print(section("  Suggested caption"))
            print(bullet(caption, indent=2, label_width=2))

            if save_location:
                filepath = os.path.join(save_location,
                                       f'comparative_energy_efficiency.{_run_image_format(user_parameters)}')
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                saved(filepath)

            plt.show()
            plt.close(fig)


        # --- Export ---
        if save_location:
            for name, ee_df in all_ee_data.items():
                fpath = os.path.join(save_location,
                                    f'{name}_energy_efficiency.csv')
                ee_df.to_csv(fpath, index=False)
                saved(fpath)

    else:
        print("\nNo valid data for energy efficiency analysis.")

    print(rule())


# ===========================================================================
# THE RATE PROTOCOL, DETECTED ONCE
# ===========================================================================
# A variable-rate test is not an exotic case. It is the field-standard way to
# measure rate capability: a reference block, a ramp, and a return to the
# reference rate. Until 1.9.0.64 Ratatosk read ONE C-rate from the cycler
# header — which records only the first step — and then printed it on every
# figure caption, in START_HERE and in the run page. On
# `JQ_NNM_C-rate_2-4.2V_B_29042026` that made every caption say "0.05 C" for
# a run that spans C/20 to 5C: false for 28 of its 33 cycles.
#
# The information was never missing. `rate_capability` has read a Current
# column out of the raw dataframe since 1.8.7 and grouped the cycles into
# blocks correctly. It was simply the only consumer, and it runs in Cell 6c,
# long after the captions are written.
#
# So the detection is lifted to module level and answered ONCE per dataset.
# `rate_capability` now calls these rather than keeping its own copies —
# there is no second definition to drift.
def discharge_current_per_cycle(df):
    """
    Mean absolute discharge current per cycle, in amps, from whatever the
    cycler called its current column. Empty frame when there is none.
    """
    current_col = None
    current_scale = 1.0
    for col in df.columns:
        cl = str(col).lower()
        if 'current' in cl and 'cut' not in cl:
            current_col = col
            if 'ma' in cl:
                current_scale = 0.001
            break
    if current_col is None or 'Cycle' not in df.columns:
        return pd.DataFrame(columns=['Cycle', 'Avg_Current_A'])
    work = df[['Cycle', 'Step', current_col]].copy()
    work[current_col] = pd.to_numeric(work[current_col], errors='coerce')
    dis = work[work['Step'] == 'Discharge'].dropna(subset=['Cycle',
                                                           current_col])
    if dis.empty:
        return pd.DataFrame(columns=['Cycle', 'Avg_Current_A'])
    out = (dis.groupby('Cycle')[current_col]
              .apply(lambda x: x.abs().mean() * current_scale).reset_index())
    out.columns = ['Cycle', 'Avg_Current_A']
    out['Cycle'] = out['Cycle'].astype(int)
    return out


def group_cycles_by_rate(current_per_cycle, tolerance=None):
    """
    Contiguous runs of cycles at the same discharge current.

    A cycle joins the current run when its mean current is within
    `tolerance` of the run's mean so far; otherwise it starts a new run.
    """
    tol = float(CURRENT_GROUPING_TOLERANCE if tolerance is None else tolerance)
    if current_per_cycle is None or current_per_cycle.empty:
        return []
    groups = []
    cyc = [current_per_cycle.iloc[0]['Cycle']]
    cur = [current_per_cycle.iloc[0]['Avg_Current_A']]
    for i in range(1, len(current_per_cycle)):
        row = current_per_cycle.iloc[i]
        prev = np.mean(cur)
        if (prev > 0 and abs(row['Avg_Current_A'] - prev) / prev <= tol):
            cyc.append(row['Cycle']); cur.append(row['Avg_Current_A'])
        else:
            groups.append(dict(cycles=cyc, mean_current_A=float(np.mean(cur)),
                               n_cycles=len(cyc)))
            cyc = [row['Cycle']]; cur = [row['Avg_Current_A']]
    groups.append(dict(cycles=cyc, mean_current_A=float(np.mean(cur)),
                       n_cycles=len(cyc)))
    return groups


def current_to_crate(current_A, active_mass_mg, theoretical_cap):
    """C-rate from current, active mass and theoretical capacity."""
    if (active_mass_mg is None or theoretical_cap is None
            or active_mass_mg <= 0 or theoretical_cap <= 0):
        return None
    theo_A = (active_mass_mg / 1000.0) * theoretical_cap / 1000.0
    return (current_A / theo_A) if theo_A > 0 else None


def snap_crate(crate):
    """
    The conventional rate a measured one is within RATE_LABEL_SNAP of, else
    the measured value unchanged. `None` passes through.

    Used by `format_crate` for the label and by the parameter step for the
    C-rate default, so that a measured 0.1001 C is offered as 0.1 rather
    than printed to four decimal places on every figure caption.
    """
    if crate is None or not np.isfinite(crate) or crate <= 0:
        return crate
    c = float(crate)
    near = min(_CONVENTIONAL_RATES, key=lambda r: abs(np.log(r / c)))
    return near if abs(np.log(near / c)) <= np.log(1.0 + RATE_LABEL_SNAP) else c


def format_crate(crate, snap=True):
    """`0.1 -> C/10`, `0.5 -> C/2`, `1.0 -> 1C`. See RATE_LABEL_SNAP."""
    if crate is None or not np.isfinite(crate) or crate <= 0:
        return '?C'
    c = snap_crate(float(crate)) if snap else float(crate)
    if c < 1.0:
        denom = 1.0 / c
        return f'C/{denom:.0f}' if abs(denom - round(denom)) < 0.05 \
            else f'{c:.2f}C'
    return f'{c:.0f}C' if abs(c - round(c)) < 0.05 else f'{c:.2f}C'


def rate_phrase(params):
    """
    The rate, as a caption says it: `0.05 C` for a single-rate run,
    `rates from C/20 to 5C` for a ramp. Five captions used to interpolate
    one number and append " C", which on a rate-capability run stated a rate
    that was false for most of the cycles in the figure.
    """
    rp = (params or {}).get('rate_protocol') or {}
    if rp.get('available') and rp.get('is_variable') and rp.get('label'):
        return f"rates from {rp['label']}"
    if rp.get('available') and rp.get('label'):
        return rp['label']
    c = (params or {}).get('charge_rate_c')
    return f"{c} C" if c is not None else "an unstated rate"


def rate_protocol(df, params, *, tolerance=None):
    """
    What rates this dataset was cycled at, from the data rather than the
    header.

    Returns a dict with:
      `available`   False when the file carries no current column at all
      `blocks`      [{cycles, first_cycle, last_cycle, n_cycles,
                      mean_current_A, c_rate, label}] in cycle order
      `n_blocks`    blocks holding at least MIN_CYCLES_PER_RATE cycles
      `is_variable` more than one such block
      `transition_cycles`  cycles in blocks too short to be a rate block —
                    at a rate change the cycler's last cycle of a block often
                    carries a mixed mean current and lands in a block of its
                    own. They are real cycles and they are excluded from
                    every block mean, so they are NAMED here rather than
                    silently dropped.
      `label`       'C/20' for a single rate, 'C/20 to 5C' for a ramp
      `returns_to_start`  the last block is at the first block's rate
    """
    out = dict(available=False, blocks=[], n_blocks=0, is_variable=False,
               transition_cycles=[], label=None, returns_to_start=False)
    if df is None or not hasattr(df, "columns"):
        return out
    cpc = discharge_current_per_cycle(df)
    if cpc.empty:
        return out
    mass = (params or {}).get('active_material_mass_mg')
    theo = (params or {}).get('theoretical_capacity_mAh_g')
    groups = group_cycles_by_rate(cpc, tolerance)
    if not groups:
        return out
    out['available'] = True
    blocks, short = [], []
    for g in groups:
        cr = current_to_crate(g['mean_current_A'], mass, theo)
        rec = dict(cycles=[int(c) for c in g['cycles']],
                   first_cycle=int(min(g['cycles'])),
                   last_cycle=int(max(g['cycles'])),
                   n_cycles=int(g['n_cycles']),
                   mean_current_A=float(g['mean_current_A']),
                   c_rate=cr, label=format_crate(cr))
        if g['n_cycles'] >= MIN_CYCLES_PER_RATE:
            blocks.append(rec)
        else:
            short.extend(rec['cycles'])
    out['blocks'] = blocks
    out['n_blocks'] = len(blocks)
    out['transition_cycles'] = sorted(short)
    out['is_variable'] = len(blocks) > 1
    if blocks:
        rates = [b['c_rate'] for b in blocks if b['c_rate'] is not None]
        if not out['is_variable'] or not rates:
            out['label'] = blocks[0]['label']
        else:
            out['label'] = (f"{format_crate(min(rates))} to "
                            f"{format_crate(max(rates))}")
        f, l = blocks[0]['c_rate'], blocks[-1]['c_rate']
        out['returns_to_start'] = (
            f is not None and l is not None and f > 0
            and abs(l - f) / f <= 2 * float(
                CURRENT_GROUPING_TOLERANCE if tolerance is None else tolerance))
    return out


def describe_rate(params, protocol=None):
    """
    The rate to PRINT for this dataset: the measured protocol when one was
    detected, the entered value otherwise. Never a single number for a run
    that used several.
    """
    if protocol and protocol.get('available') and protocol.get('label'):
        if protocol.get('is_variable'):
            return (f"{protocol['label']} "
                    f"({protocol['n_blocks']} rate blocks)")
        return protocol['label']
    c = (params or {}).get('charge_rate_c')
    return (f"{c} C" if c is not None else None)


# --- RATE RECOVERY: what the return to the reference rate actually says ----
# The field-standard rate schedule is a reference block, a ramp, and a return
# to the reference rate, and the number everyone quotes is
# `mean(return) / mean(reference)`. On
# `JQ_NNM_C-rate_2-4.2V_{B,C}_29042026` that is 80.4% and 80.2% — and almost
# all of it is ordinary cycling fade over the 28 cycles in between, not
# damage from the high-rate excursion.
#
# Separating the two needs a fade baseline, and the obvious one does not
# work. Projecting a line from the first reference block forward was tested
# against the CONSTANT-RATE NNM triplicate, where nothing happens between
# cycle 5 and cycle 31: a linear fit to cycles 2-5 predicts NEGATIVE capacity
# at cycle 31 against an actual 75-86 mAh/g, and a log-linear fit predicts
# about a third of it. Neither prediction interval contains the truth. Those
# cycles are formation, not fade — cell B's CE over cycles 1-6 runs 88.4,
# 76.4, 44.5, 94.0, 94.0, 96.7, and does not settle until AFTER the reference
# block has ended. On the same control, a window of cycles 6-28 lands within
# 0.3% on all three cells.
#
# So the slope is harvested from the ramp itself, where every cycle is
# post-formation. Rate sets the LEVEL and cycling sets the SLOPE:
#
#     log C(cycle) = a_block + b . cycle,     one shared b
#
# fitted on post-formation, non-transition cycles. On the two rate-tested
# cells that gives b = -0.505 %/cycle (se 0.061) and -0.517 %/cycle
# (se 0.104), residual sd 0.21% and 0.36% — the replicates agree to
# 0.012 %/cycle.
#
# THE ANCHOR IS THE PART THAT CANNOT BE FIXED HERE. Projecting that slope
# from the pre-ramp block to the return needs a level for the reference rate,
# and the only reference-rate cycles before the ramp are the formation ones.
# Every anchor drawn from them reads high, so every damage estimate drawn
# from them is biased high. Three anchors give 92.6/94.5/98.5% on cell B and
# 92.7/92.4/94.6% on cell C — a spread the same size as the effect. That is
# why this reports a RANGE and calls it an upper bound on damage, and says
# so when the reference block lies inside formation.
#
# The fix is experimental (one reference-rate cycle between blocks, or a
# reference block that outlives formation) and the protocol is field
# standard, so the tool reports the bound honestly rather than inventing a
# point estimate.
def annotate_rate_protocol(electrochemical_data, user_parameters, *,
                           verbose=True):
    """
    Detect each dataset's rate protocol and store it on its parameters.

    Called once, before any figure is drawn, so that every caption, every
    retention figure and the report itself are working from the rate the cell
    was ACTUALLY cycled at rather than the single value the cycler header
    records for its first step. Mutates and returns `user_parameters`.
    """
    any_var = False
    for name, df in (electrochemical_data or {}).items():
        params = (user_parameters or {}).get(name)
        if params is None:
            continue
        # The parameter step already answered this for the interactive path,
        # and answered it BEFORE the C-rate prompt so it could skip it. Do not
        # recompute: the declarative path has no such step, and this is where
        # it gets one.
        proto = params.get("rate_protocol") or rate_protocol(df, params)
        params["rate_protocol"] = proto
        params["rate_is_variable"] = bool(proto.get("is_variable"))
        params["rate_label"] = proto.get("label")
        if not proto.get("available") or not verbose:
            continue
        if proto.get("is_variable"):
            any_var = True
            print(f"  {name}: {proto['n_blocks']} rate blocks, "
                  f"{proto['label']}"
                  + (" , returning to the starting rate"
                     if proto.get("returns_to_start") else ""))
            print("      " + "  ".join(
                f"{b['label']}:{b['first_cycle']}-{b['last_cycle']}"
                for b in proto["blocks"]))
            if proto.get("transition_cycles"):
                print(f"      rate-transition cycle(s) not in any block: "
                      + ", ".join(str(c)
                                  for c in proto["transition_cycles"]))
        elif proto.get("label"):
            print(f"  {name}: single rate, {proto['label']}")
    if any_var and verbose:
        # THE ONE SENTENCE THAT HAS TO BE SAID OUT LOUD. Capacity against
        # cycle number in a rate test is not a fade curve; most of its shape
        # is the rate schedule. Everything downstream that says "retention"
        # now scopes itself to one rate block, and this says why.
        print("  Capacity varies with the RATE SCHEDULE in this run, not "
              "only with age — retention across rate blocks is not "
              "retention.")
    return user_parameters


def rate_recovery(cap_df, protocol, *, formation_end_cycle=None,
                  reference_cycle_min=None):
    """
    Fade rate during a rate ramp, and what the return to the reference rate
    says once that fade is accounted for.

    `cap_df` needs `Cycle` and `Discharge_mAh_g`; `protocol` is
    `rate_protocol`'s dict. Returns None when there is no ramp with a
    return to the starting rate.

    Keys: `fade_pct_per_cycle`, `fade_se`, `residual_sd_pct`, `n_fit`,
    `q_recovery_pct` (the raw, field-standard number), `anchors`
    (list of {name, cycle, capacity, predicted, ratio_pct}),
    `damage_bound_pct` (min, max over anchors), `anchor_in_formation`,
    `formation_end`, `note`.
    """
    if not protocol or not protocol.get('is_variable') \
            or not protocol.get('returns_to_start'):
        return None
    blocks = protocol.get('blocks') or []
    if len(blocks) < 3:
        return None
    first, last = blocks[0], blocks[-1]
    d = cap_df[['Cycle', 'Discharge_mAh_g']].dropna().copy()
    d['Cycle'] = d['Cycle'].astype(int)
    d = d[d['Discharge_mAh_g'] > 0]
    if d.empty:
        return None
    trans = set(protocol.get('transition_cycles') or [])

    # Post-formation, and never the transition cycles: the fit is what the
    # whole result rests on, so it gets only cycles that are measurements of
    # a settled cell at a known rate.
    f_end = (int(formation_end_cycle) if formation_end_cycle
             else int(first['last_cycle']))
    ramp = [b for b in blocks[1:-1]]
    if not ramp:
        return None
    lo = max(f_end + 1, ramp[0]['first_cycle'])
    hi = ramp[-1]['last_cycle']
    fit = d[(d['Cycle'] >= lo) & (d['Cycle'] <= hi)
            & (~d['Cycle'].isin(trans))].copy()
    labels = {}
    for b in ramp:
        for c in b['cycles']:
            labels[c] = b['label']
    fit['block'] = fit['Cycle'].map(labels)
    fit = fit.dropna(subset=['block'])
    keys = sorted(fit['block'].unique())
    if len(fit) < len(keys) + 3 or len(keys) < 2:
        return None

    y = np.log(fit['Discharge_mAh_g'].to_numpy(float))
    X = np.zeros((len(fit), len(keys) + 1))
    for i, k in enumerate(keys):
        X[:, i] = (fit['block'] == k).to_numpy(float)
    X[:, -1] = fit['Cycle'].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    b_share = float(coef[-1])
    resid = y - X @ coef
    dof = max(len(fit) - (len(keys) + 1), 1)
    s = float(np.sqrt((resid ** 2).sum() / dof))
    try:
        se_b = float(s * np.sqrt(np.linalg.inv(X.T @ X)[-1, -1]))
    except np.linalg.LinAlgError:
        se_b = float('nan')

    post = d[(d['Cycle'] >= last['first_cycle'])
             & (d['Cycle'] <= last['last_cycle'])
             & (~d['Cycle'].isin(trans))]
    pre = d[(d['Cycle'] >= first['first_cycle'])
            & (d['Cycle'] <= first['last_cycle'])
            & (~d['Cycle'].isin(trans))]
    _cmin = int(RATE_RECOVERY_MIN_REFERENCE if reference_cycle_min is None
                else reference_cycle_min)
    if len(post) < _cmin or len(pre) < _cmin:
        return None
    obs = float(post['Discharge_mAh_g'].mean())
    xt = float(post['Cycle'].mean())

    # THREE ANCHORS, NOT ONE. They differ by as much as the effect, and the
    # difference is not noise — it is how much formation is still in the
    # reference block. Reporting one of them would be choosing an answer.
    anchors = []
    _pre_body = pre[pre['Cycle'] > int(pre['Cycle'].min())] \
        if len(pre) > 1 else pre
    cands = [("block mean", float(_pre_body['Cycle'].mean()),
              float(_pre_body['Discharge_mAh_g'].mean()))]
    for _c in sorted(pre['Cycle'])[-2:]:
        cands.append((f"cycle {int(_c)}", float(_c),
                      float(pre.loc[pre['Cycle'] == _c,
                                    'Discharge_mAh_g'].iloc[0])))
    seen = set()
    for nm, x0, c0 in cands:
        if c0 <= 0 or round(x0, 3) in seen:
            continue
        seen.add(round(x0, 3))
        pred = c0 * np.exp(b_share * (xt - x0))
        anchors.append(dict(name=nm, cycle=x0, capacity=c0,
                            predicted=float(pred),
                            ratio_pct=float(100.0 * obs / pred)))
    if not anchors:
        return None
    ratios = [a['ratio_pct'] for a in anchors]

    # AN UNKNOWN FORMATION END IS NOT A CLEAN ONE. `detect.formation_end`
    # tests coulombic efficiency against the median of the later cycles, and
    # on a rate test that median is not a fixed target: CE moves WITH THE
    # RATE. On `JQ_NNM_C-rate_2-4.2V_B_29042026` the ramp sits near 97% and
    # the return block near 94%, so no window ever holds within tolerance
    # and the answer comes back None — from a reference block of five cycles
    # whose CE runs 88.4, 76.4, 44.5, 94.0, 94.0 and has plainly not settled.
    #
    # So the three states are kept apart: formation demonstrably ends after
    # the reference block (contaminated), demonstrably before it (clean), or
    # could not be established (treated as contaminated, because claiming
    # the clean case without evidence is how a bound becomes an estimate).
    if formation_end_cycle is None:
        in_formation = True
        note = (
            "Formation could not be located from coulombic efficiency: on a "
            "variable-rate run CE moves with the RATE as well as with age, "
            "so the settling test has no fixed target to hold against. The "
            f"reference block is cycles {int(first['first_cycle'])}-"
            f"{int(first['last_cycle'])}, which on this protocol is where "
            "formation happens, so every anchor for the reference level may "
            "still contain formation capacity and read high. The range below "
            "is an UPPER BOUND on the loss caused by the rate excursion, not "
            "an estimate of it.")
    elif int(first['last_cycle']) <= int(formation_end_cycle):
        in_formation = True
        note = ("The reference block ends at cycle "
            f"{int(first['last_cycle'])} and formation is not complete until "
            f"cycle {int(formation_end_cycle)}, so every anchor for the "
            "reference "
            "level still contains formation capacity and reads high. The "
            "range below is an UPPER BOUND on the loss caused by the rate "
            "excursion, not an estimate of it.")
    else:
        in_formation = False
        note = (f"Formation ends at cycle {int(formation_end_cycle)}, before "
                f"the reference block closes at cycle "
                f"{int(first['last_cycle'])}, so the anchors below are "
                f"settled reference-rate cycles and the range is an "
                f"estimate rather than a bound.")
    return dict(fade_pct_per_cycle=100.0 * (np.exp(b_share) - 1.0),
                fade_se_pct=100.0 * se_b, residual_sd_pct=100.0 * s,
                n_fit=int(len(fit)), blocks_fitted=keys,
                q_recovery_pct=float(100.0 * obs
                                     / pre['Discharge_mAh_g'].mean()),
                observed=obs, observed_cycle=xt, anchors=anchors,
                damage_bound_pct=(float(min(ratios)), float(max(ratios))),
                anchor_in_formation=in_formation,
                formation_end=formation_end_cycle,
                note=note)


@_honours_verbose
def rate_capability(electrochemical_data, user_parameters, *, save_location=None,
        all_cycle_tables=None, file_format='png', verbose=True):
    """
    Capacity grouped by C-rate.

    Ported from 1.8.7 Cell 13, wrapped in a function without editing
    the body. `all_cycle_tables` is `cycling_summary`'s output where the
    original read a notebook global; passing None reproduces the
    not-yet-run branch exactly.
    """
    # ONE DEFINITION. These were the originals; they now live at module
    # level so the parameter step and the report can ask the same question
    # and get the same answer. See `rate_protocol`.
    _get_discharge_current_per_cycle = discharge_current_per_cycle
    _group_by_rate = group_cycles_by_rate
    _current_to_crate = current_to_crate
    _format_crate = format_crate

    print(rule("RATE CAPABILITY ANALYSIS"))

    _any_multirate = False

    _have_cycle_tables = _has_tables(all_cycle_tables)

    if EXCLUDE_INCOMPLETE and not _have_cycle_tables:
        print("  ⚠ all_cycle_tables not found — Cell 5b has not been "
              "run. Protocol-incomplete filtering unavailable.\n")

    for name, df in electrochemical_data.items():
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        active_mass_mg = params.get('active_material_mass_mg')
        theoretical_cap = params.get('theoretical_capacity_mAh_g')

        # Get discharge current per cycle
        current_df = _get_discharge_current_per_cycle(df)

        if current_df.empty:
            print(f"  {composition}: no current data found, skipping.")
            continue

        # Group by rate
        rate_groups = _group_by_rate(current_df)

        # Count distinct rate levels
        unique_currents = [g['mean_current_A'] for g in rate_groups]

        # Cluster unique currents to find distinct rates
        distinct_rates = []
        for curr in unique_currents:
            matched = False
            for dr in distinct_rates:
                if abs(curr - dr) / max(dr, 1e-9) <= CURRENT_GROUPING_TOLERANCE:
                    matched = True
                    break
            if not matched:
                distinct_rates.append(curr)

        n_distinct = len(distinct_rates)

        if n_distinct < 2:
            print(f"  {composition}: single rate detected "
                  f"({_format_crate(_current_to_crate(distinct_rates[0], active_mass_mg, theoretical_cap)) if distinct_rates else '?'}), "
                  f"rate capability analysis skipped.")
            continue

        _any_multirate = True
        print(f"  {composition}: {n_distinct} distinct rates detected "
              f"across {len(rate_groups)} rate blocks")

        # Get capacity data from Cell 5b's cache, falling back to raw derivation
        if _have_cycle_tables and name in all_cycle_tables:
            ct = all_cycle_tables[name]
            cap_df = (_ct_with_flag(ct, 'Discharge_mAh_g')
                      .dropna(subset=['Cycle', 'Discharge_mAh_g'])
                      .copy())
        else:
            # Fallback: derive via groupby
            discharge = df[df['Step'] == 'Discharge'].dropna(
                subset=['Cycle', 'Discharge_Capacity'])
            if discharge.empty:
                print(f"  {composition}: no capacity data, skipping.")
                continue
            grp = discharge.groupby('Cycle')['Discharge_Capacity']
            cap_df = pd.DataFrame({
                'Cycle': grp.max().index.astype(int),
                'Discharge_mAh_g': (grp.max() - grp.min()).values,
                'Incomplete': False,
            })

        if cap_df.empty:
            print(f"  {composition}: no capacity data, skipping.")
            continue

        # Exclude protocol-incomplete cycles (from Cell 5b).
        # The heuristic-based filter that used to live here has been
        # deliberately removed: high-rate cycles in rate-capability
        # experiments genuinely deliver much less capacity than the
        # reference rate, and the heuristic falsely flagged them.
        if EXCLUDE_INCOMPLETE and 'Incomplete' in cap_df.columns:
            n_incomplete = int(cap_df['Incomplete'].sum())
            if n_incomplete > 0:
                removed = cap_df.loc[cap_df['Incomplete'],
                                     'Cycle'].tolist()
                print(f"    Excluded {n_incomplete} "
                      f"protocol-incomplete cycle(s): "
                      f"{', '.join(str(int(c)) for c in removed)}")
            cap_df = cap_df[~cap_df['Incomplete']].copy()

        # Merge current and capacity data
        merged = pd.merge(cap_df, current_df, on='Cycle', how='inner')

        if merged.empty:
            continue

        # Assign C-rate labels to each cycle
        merged['C_rate'] = merged['Avg_Current_A'].apply(
            lambda x: _current_to_crate(x, active_mass_mg, theoretical_cap)
        )
        merged['C_rate_label'] = merged['C_rate'].apply(_format_crate)

        # Assign rate group index
        group_assignments = {}
        for gi, group in enumerate(rate_groups):
            for cyc in group['cycles']:
                group_assignments[cyc] = gi
        merged['Rate_Group'] = merged['Cycle'].map(group_assignments)

        # Build summary per rate group
        rate_summary = []
        for gi, group in enumerate(rate_groups):
            group_data = merged[merged['Rate_Group'] == gi]
            if group_data.empty or len(group_data) < MIN_CYCLES_PER_RATE:
                continue

            crate = _current_to_crate(group['mean_current_A'],
                                       active_mass_mg, theoretical_cap)
            crate_label = _format_crate(crate)

            rate_summary.append({
                'group_index': gi,
                'c_rate': crate,
                'c_rate_label': crate_label,
                'mean_current_A': group['mean_current_A'],
                'mean_capacity': group_data['Discharge_mAh_g'].mean(),
                'std_capacity': group_data['Discharge_mAh_g'].std(),
                'n_cycles': len(group_data),
                'first_cycle': int(group_data['Cycle'].min()),
                'last_cycle': int(group_data['Cycle'].max())
            })

        if len(rate_summary) < 2:
            print(f"  {composition}: insufficient rate groups "
                  f"(need >= 2 with >= {MIN_CYCLES_PER_RATE} cycles each)")
            continue

        rate_df = pd.DataFrame(rate_summary)

        # Reference capacity (first rate group, usually lowest rate)
        ref_cap = rate_df.iloc[0]['mean_capacity']
        # NOT capacity retention: capacity at this RATE against the
        # reference rate group. Exporting both under one header was how a
        # reader ended up comparing two different quantities.
        rate_df['Rate_Capability_%'] = rate_df['mean_capacity'] / ref_cap * 100

        # Recovery: if final group is at the same rate as the first
        first_rate = rate_df.iloc[0]['c_rate']
        last_group = rate_df.iloc[-1]
        recovery = None
        if (first_rate is not None and last_group['c_rate'] is not None and
            abs(last_group['c_rate'] - first_rate) / max(first_rate, 1e-9)
            <= CURRENT_GROUPING_TOLERANCE * 2):
            recovery = last_group['mean_capacity'] / ref_cap * 100

        # --- Print summary ---
        print("\n" + heading(composition))
        print(f"  {'Rate':<10} {'Capacity':>10} {'Retention':>10} "
              f"{'Cycles':>8} {'Range':>12}")
        print(f"  {'':10} {'mAh/g':>10} {'%':>10} {'':>8} {'':>12}")
        print(f"  {'-'*52}")

        for _, row in rate_df.iterrows():
            cycle_range = (f"{row['first_cycle']}-{row['last_cycle']}"
                          if row['first_cycle'] != row['last_cycle']
                          else str(row['first_cycle']))
            print(f"  {row['c_rate_label']:<10} "
                  f"{row['mean_capacity']:>10.1f} "
                  f"{row['Rate_Capability_%']:>10.1f} "
                  f"{row['n_cycles']:>8} "
                  f"{cycle_range:>12}")

        if recovery is not None:
            print(f"\n  Recovery to {rate_df.iloc[0]['c_rate_label']}: "
                  f"{recovery:.1f}%")

        # --- what that recovery is made of -------------------------------
        # THE RECOVERY PERCENTAGE WAS PRINTED AND NEVER EXPORTED. It reached
        # no CSV, no run page and no START_HERE — the fifth diagnostic in
        # this pipeline computed correctly and consumed by nothing, after
        # `area_in_window`, `area_is_lower_bound`, `sig.occupancy` and
        # `height_worst`. It is now carried on `rate_df` and written out
        # with the rest, together with what it decomposes into.
        _proto = rate_protocol(df, params)
        _f_end = None
        if 'CE_%' in cap_df.columns:
            _ce = {int(r['Cycle']): float(r['CE_%'])
                   for _, r in cap_df.iterrows() if pd.notna(r.get('CE_%'))}
            _fe, _ = formation_end(_ce)
            if _fe is not None:
                _f_end = int(_fe)
        _rec = rate_recovery(cap_df, _proto, formation_end_cycle=_f_end)
        if _proto.get('transition_cycles'):
            # NAMED, NOT DROPPED. At a rate change the cycler's last cycle of
            # a block carries a mixed mean current, lands in a block of its
            # own and fails MIN_CYCLES_PER_RATE. Excluding it is right; doing
            # it silently is not — on the NNM rate test that is 7 of 33
            # cycles absent from every block mean with nothing saying so.
            print(f"    {len(_proto['transition_cycles'])} rate-transition "
                  f"cycle(s) excluded from the block means: "
                  + ", ".join(str(c)
                              for c in _proto['transition_cycles']))
        if _rec is not None:
            print(f"\n  Fade during the ramp: "
                  f"{_rec['fade_pct_per_cycle']:+.3f} %/cycle "
                  f"(se {_rec['fade_se_pct']:.3f}, residual sd "
                  f"{_rec['residual_sd_pct']:.2f}%, n={_rec['n_fit']})")
            _lo, _hi = _rec['damage_bound_pct']
            print(f"  Recovery against that fade: {_lo:.1f}-{_hi:.1f}% "
                  f"across {len(_rec['anchors'])} anchor(s) for the "
                  f"reference level")
            for _a in _rec['anchors']:
                print(f"      {_a['name']:<12} {_a['capacity']:6.1f} mAh/g "
                      f"at cycle {_a['cycle']:.1f} -> predicts "
                      f"{_a['predicted']:6.1f}, observed "
                      f"{_rec['observed']:6.1f}  ({_a['ratio_pct']:.1f}%)")
            print(f"  {_rec['note']}")
            rate_df['q_recovery_pct'] = _rec['q_recovery_pct']
            rate_df['fade_pct_per_cycle'] = _rec['fade_pct_per_cycle']
            rate_df['fade_se_pct'] = _rec['fade_se_pct']
            rate_df['fade_residual_sd_pct'] = _rec['residual_sd_pct']
            rate_df['damage_bound_low_pct'] = _rec['damage_bound_pct'][0]
            rate_df['damage_bound_high_pct'] = _rec['damage_bound_pct'][1]
            rate_df['damage_bound_is_upper_bound'] = \
                _rec['anchor_in_formation']
            rate_df['formation_end_cycle'] = _rec['formation_end']
        elif recovery is not None:
            rate_df['q_recovery_pct'] = recovery
        if _proto.get('transition_cycles'):
            rate_df['transition_cycles_excluded'] = ";".join(
                str(c) for c in _proto['transition_cycles'])


        # =================================================================
        # PLOT 1: Capacity vs C-rate bar chart
        # =================================================================

        # Deduplicate: average across groups at the same rate
        unique_rates = []
        for _, row in rate_df.iterrows():
            matched = False
            for ur in unique_rates:
                if (row['c_rate'] is not None and ur['c_rate'] is not None and
                    abs(row['c_rate'] - ur['c_rate']) / max(ur['c_rate'], 1e-9)
                    <= CURRENT_GROUPING_TOLERANCE * 2):
                    # Average with existing
                    ur['capacities'].append(row['mean_capacity'])
                    matched = True
                    break
            if not matched:
                unique_rates.append({
                    'c_rate': row['c_rate'],
                    'c_rate_label': row['c_rate_label'],
                    'capacities': [row['mean_capacity']]
                })

        bar_labels = [ur['c_rate_label'] for ur in unique_rates]
        bar_heights = [np.mean(ur['capacities']) for ur in unique_rates]
        bar_errors = [np.std(ur['capacities']) if len(ur['capacities']) > 1
                      else 0 for ur in unique_rates]

        # Colour: highlight recovery rate in green
        bar_colours = []
        for j, ur in enumerate(unique_rates):
            if (j == len(unique_rates) - 1 and recovery is not None and
                j > 0):
                bar_colours.append(RECOVERY_COLOUR)
            else:
                bar_colours.append(BAR_COLOUR)

        fig, ax = plt.subplots(figsize=(_plots.figure_width_inches, _plots.figure_height_inches))

        x_pos = np.arange(len(bar_labels))
        bars = ax.bar(x_pos, bar_heights, yerr=bar_errors,
                      color=bar_colours, edgecolor='black', linewidth=0.5,
                      capsize=4, width=0.6, zorder=3)

        # Add capacity labels on bars
        for bar, height in zip(bars, bar_heights):
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=10)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_labels, fontsize=12)
        ax.set_xlabel('C-rate', fontsize=14)
        dch_label = _discharge_label(params)
        ax.set_ylabel(f'{dch_label} capacity / mAh g$^{{-1}}$', fontsize=14)
        ax.set_ylim(bottom=0, top=max(bar_heights) * 1.15)
        ax.tick_params(axis='both', labelsize=12, width=1, direction='in',
                       top=True, right=True)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
        plt.tight_layout()

        # Caption
        rates_str = ', '.join(bar_labels[:-1])
        if len(bar_labels) > 1:
            rates_str += f' and {bar_labels[-1]}'
        else:
            rates_str = bar_labels[0]

        recovery_str = ''
        if recovery is not None:
            recovery_str = (f' The final bar (green) shows capacity '
                           f'recovery upon returning to '
                           f'{rate_df.iloc[0]["c_rate_label"]} '
                           f'({recovery:.1f}% of initial).')

        dch_word = _discharge_label(params).lower()
        caption = (
            f"Figure X. Rate capability of {composition} at "
            f"{rates_str}. Values shown are mean {dch_word} capacities "
            f"with error bars indicating standard deviation across "
            f"cycles at each rate.{recovery_str}"
        )
        print(section("  Suggested caption"))
        print(bullet(caption, indent=2, label_width=2))

        if save_location:
            fpath = os.path.join(save_location,
                                f'{name}_rate_capability.{image_format(params)}')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            saved(fpath)

        plt.show()
        plt.close(fig)


        # =================================================================
        # PLOT 2: Capacity vs cycle with rate annotations
        # =================================================================

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(merged['Cycle'], merged['Discharge_mAh_g'],
               color=BAR_COLOUR, marker='o', markersize=5,
               linestyle='-', linewidth=0.8, zorder=3)

        # Shade rate blocks and label them
        prev_right = None
        for _, row in rate_df.iterrows():
            left = row['first_cycle'] - 0.5
            right = row['last_cycle'] + 0.5

            ax.axvspan(left, right, alpha=0.06, color='grey')

            mid_cycle = (row['first_cycle'] + row['last_cycle']) / 2
            y_pos = ax.get_ylim()[1] * 0.95 if prev_right is None else ax.get_ylim()[1] * 0.95

            ax.text(mid_cycle, merged['Discharge_mAh_g'].max() * 1.05,
                   row['c_rate_label'],
                   ha='center', va='bottom', fontsize=10,
                   fontweight='bold', color='#333333')

            prev_right = right

        ax.set_xlabel('Cycle number', fontsize=14)
        _force_integer_cycles(ax)
        ax.set_ylabel(f'{dch_label} capacity / mAh g$^{{-1}}$', fontsize=14)
        ax.set_ylim(bottom=0,
                    top=merged['Discharge_mAh_g'].max() * 1.15)
        ax.set_xlim(left=0)
        ax.tick_params(axis='both', labelsize=12, width=1, direction='in',
                       top=True, right=True)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
        plt.tight_layout()

        caption = (
            f"Figure X. {dch_label} capacity vs. cycle number for "
            f"{composition} during rate capability testing. Shaded "
            f"regions indicate cycling at different C-rates as labelled."
        )
        print(section("  Suggested caption"))
        print(bullet(caption, indent=2, label_width=2))

        if save_location:
            fpath = os.path.join(save_location,
                                f'{name}_rate_capability_cycling.{image_format(params)}')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            saved(fpath)

        plt.show()


        # --- Export ---
        if save_location:
            # `c_rate_measured` beside the snapped label, because the label
            # is for reading and the number is for computing. See
            # RATE_LABEL_SNAP.
            rate_df['c_rate_measured'] = rate_df['c_rate']
            _extra = [c for c in ('c_rate_measured', 'mean_current_A',
                                  'q_recovery_pct', 'fade_pct_per_cycle',
                                  'fade_se_pct', 'fade_residual_sd_pct',
                                  'damage_bound_low_pct',
                                  'damage_bound_high_pct',
                                  'damage_bound_is_upper_bound',
                                  'formation_end_cycle',
                                  'transition_cycles_excluded')
                      if c in rate_df.columns]
            export_df = rate_df[['c_rate_label', 'mean_capacity',
                                  'std_capacity', 'Rate_Capability_%',
                                  'n_cycles', 'first_cycle', 'last_cycle']
                                 + _extra]
            fpath = os.path.join(save_location,
                                f'{name}_rate_capability_summary.csv')
            export_df.to_csv(fpath, index=False)
            saved(fpath)

    if not _any_multirate:
        print("All datasets are single-rate — "
              "rate capability analysis skipped.")
        print("(Load multi-rate data to use this analysis, e.g., "
              "C/10 -> C/5 -> C/2 -> 1C -> C/10 recovery)")

    print(rule())
