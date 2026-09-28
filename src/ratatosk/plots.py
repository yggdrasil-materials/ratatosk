"""
The presentation layer, ported unchanged.

The waterfall, the heatmaps, the key-cycle overlays and the split
charge/discharge panels are the best part of Ratatosk and their output does not
change in 1.9.0. So this module does not reimplement them: every plotting
function below was extracted from the 1.8.7 notebook programmatically and is
byte-for-byte the code that drew the figures you already have.

    from Module 2   plot_dqdv_all_cycles           plot_dqdv_waterfall
                    plot_dqdv_key_cycles_combined  plot_dqdv_heatmap
                    plot_dqdv_key_cycles_split     plot_preprocessing_qc
    from Module 3   plot_detected_peaks
    from Module 4   plot_fit_quality
    from Module 5   plot_tracked_trends
    from Module 6   plot_delta_v   plot_capacity_attribution

    shared          _get_display_name  _force_integer_cycles
                    _charge_label      _discharge_label   _apply_pub_style

How the 1.9.0 objects reach 1.8.7's plotting code
-------------------------------------------------
Those functions take the notebook's dictionaries — `processed_dqdv`,
`detected_peaks`, `fit_results`, `tracked_peaks`, `user_parameters` — and the
alternative to keeping them was rewriting fourteen tested plotting functions
against `Dataset`, `HalfCycleSignal`, `Detection` and `Tracking`, which is
precisely how figures stop being identical.

So the adapters go the other way. `as_processed_dqdv`, `as_detected_peaks`,
`as_fit_results` and `as_tracked` build the 1.8.7 shapes from the 1.9.0
objects. They are small, they are the only new code here, and they are tested.

The one thing that could not be adapted directly
------------------------------------------------
`plot_fit_quality` asks its fit for `.best_fit` and `.eval_components(x=...)` —
that is, for a live lmfit `ModelResult`. 1.9.0 deliberately does not keep one:
a `ModelResult` cannot cross a process boundary through stdlib pickle, and
carrying it would undo the single decision that made parallel fitting possible.

Rather than edit the plotting code, `as_fit_results` supplies a `_FitView`,
which answers both questions from `fitting.evaluate` — the component curves
rebuilt analytically from the stored parameters, through lmfit's own model
classes, with no refit. Verified against the data to 3e-7. The plotting
function is untouched and does not know the difference.

The one change to the fitted model, and why it is not cosmetic
-------------------------------------------------------------
1.9.0 originally prefixed its polynomial baseline `bkg_`; 1.8.7 uses `bg_`.
`plot_fit_quality` decides which component to draw as the dashed black baseline
by testing `'bg_' in comp_name`, which `'bkg_'` fails — the baseline would have
been drawn as a filled peak. The prefix is now `bg_` throughout, matching 1.8.7
everywhere including the parameter names.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors            # noqa: F401  (used by ported code)
import matplotlib.patheffects as path_effects
import matplotlib.ticker as mticker    # noqa: F401  (used by ported code)
from matplotlib import gridspec
from matplotlib.colors import Normalize

from . import fitting as _ft
from .style import (rule, heading, section, entry, verdict, bullet,
                    image_format, saved, half_cycle_labels)

__all__ = ["plot_dqdv_all_cycles", "plot_dqdv_key_cycles_combined",
           "plot_dqdv_key_cycles_split", "plot_dqdv_waterfall",
           "plot_dqdv_heatmap", "plot_preprocessing_qc",
           "plot_detected_peaks", "plot_fit_quality", "plot_tracked_trends",
           "plot_delta_v", "plot_capacity_attribution",
           "as_processed_dqdv", "as_detected_peaks", "as_fit_results",
           "as_tracked", "as_user_parameters", "set_figure_size"]


# =============================================================================
# FIGURE GEOMETRY — Cell 3's globals, as module state
# =============================================================================
# 6.73 in is the RSC double-column width. The ported functions read these as
# globals, exactly as they did in the notebook.
figure_width_inches = 6.73
figure_height_inches = 5.0


def set_figure_size(width_inches=None, height_inches=None):
    """Set the figure geometry every plot in this module inherits."""
    global figure_width_inches, figure_height_inches
    if width_inches is not None:
        figure_width_inches = float(width_inches)
    if height_inches is not None:
        figure_height_inches = float(height_inches)


# =============================================================================
# MODULE 2 DEFAULTS — unchanged
# =============================================================================

# Matplotlib resolves font weights against the weights a font actually ships.
# The DejaVu Sans that ships with matplotlib has no 'medium' face, so every
# figure title emitted 'findfont: Failed to find font weight medium, now
# using 400' and then used 400 anyway. 'normal' IS 400, so this asks for what
# it was always going to get, silently. Set 'bold' or a number if you want
# heavier titles and your font has the face.
FIGURE_TITLE_WEIGHT = 'normal'

# --- the multi-peak fit panels -------------------------------------------
# Where the curve is, for the purpose of framing the panel. The floor is a
# fraction of the panel's own maximum |dQ/dV|, so it adapts to the chemistry
# rather than to an absolute current; the pad is a fraction of the span it
# found; and the minimum width stops a 40 mV LTO peak from filling the axis
# edge to edge with no context around it. See the note in `plot_fit_quality`.
# HOW WIDE AN ERROR BAND IS ALLOWED TO BE DRAWN. Purely a display cap: a
# single standard error of half a volt on a millivolt-scale panel makes every
# other point invisible. It is not a claim about the error, and both figures
# that use it now say on the figure how many points were capped. The numbers
# themselves are unclipped in the CSVs.
# THE HEATMAP'S UNVISITED CELLS, and the envelope that explains them.
# The colour is a dark neutral rather than the colormap's own zero: close
# enough to keep the panel looking like a heat map, far enough that a region
# the cell never entered is not the same pixel as a region where dQ/dV was
# measured to be zero. The envelope is Okabe-Ito sky blue — the one cool
# colour in a warm colormap, so it reads at any density and survives
# deuteranopia, protanopia and greyscale.
# Near the zero end of the ramp, one step off it. The fill is background;
# the ENVELOPE LINE is what announces the collapse, and a large field of a
# distinctly different colour competes with it for the eye without adding
# anything the line does not already say.
HEATMAP_UNVISITED_COLOUR = '#12121A'
HEATMAP_ENVELOPE_COLOUR = '#56B4E9'
HEATMAP_ENVELOPE_MIN_BINS = 3.0
# ...AND a share of the panel. Bins alone let a 38 mV wobble on a 1.25 V axis
# draw a line hugging the edge and a caption announcing it, which is clutter
# dressed as a finding. Both gates have to clear: resolvable by the binning,
# AND large enough on the axis to be worth a reader's attention.
HEATMAP_ENVELOPE_MIN_FRACTION = 0.05

DELTA_V_BAND_CLIP_MV = 20.0
TREND_BAND_CLIP_V = 0.020

PEAK_FIT_XLIM_FLOOR = 0.02
PEAK_FIT_XLIM_PAD = 0.08
PEAK_FIT_XLIM_MIN_V = 0.20

# The fallback set of key cycles, when a params dict does not carry one.
# There were FIVE different literals for this — [1,10,20,30,40,50] in three
# places, [1,5,10,20,30,40,50] in one and [1,5,10] in another — so within a
# single run the key-cycle overlay, the split panel, the detected-peaks
# figure and the fit-quality figure each showed a different set of cycles.
# `params.default_key_cycles` is the definition; this mirrors it without
# importing params into plots (which would be a cycle).
DEFAULT_KEY_CYCLES = [1, 5, 10, 20, 30, 40, 50]

# A shaded region around a line is a BAND — that is the word journals use,
# and "±1 s.e." or "±1 standard error" is how its extent is stated. Not a
# strip, not an area (an area is what you integrate). Every band Ratatosk
# draws is ±1 standard error on ONE fitted parameter, taken from the
# covariance the least-squares fit returns, so it is the fit's own statement
# of how well that number is determined by the data — not a confidence
# interval on the measurement, and not propagated between parameters.
WATERFALL_HEIGHT_PER_CYCLE_IN = 0.35
WATERFALL_MIN_HEIGHT_IN = 4.0
WATERFALL_MAX_HEIGHT_IN = 10.0
WATERFALL_OVERLAP = 0.5
WATERFALL_DENSE_NOTE_ABOVE = 40
WATERFALL_MAX_LABELS = 12


# =============================================================================
# PORTED FROM 1.8.7 — extracted programmatically. Do not edit: the 1.9.0
# regression requires these figures to be identical.
# =============================================================================


_BAND_NOTE = ("Shaded bands span \u00b11 standard error on the fitted "
              "parameter, as returned by the least-squares fit; where a band "
              "is wider than the trend it describes, the parameter is not "
              "determined by the data.")


def _caption(text):
    """Print a figure caption in the same house style as the cycling cells."""
    print(section("  Suggested caption"))
    print(bullet(text, indent=2, label_width=2))

EXCLUDE_FORMATION_FROM_YLIM = True
SHORT_HALFCYCLE_THRESHOLD = 0.8
# ONE charge/discharge convention for the whole package. Blue means charge in
# every figure Ratatosk draws; vermillion means discharge. Both figure
# families import these rather than restating them, because until 1.9.1 they
# disagreed — blue was charge in the dQ/dV figures and DISCHARGE in the
# cycling figures, in the same run, which is indefensible whatever the
# convention.
#
# From Okabe-Ito. Blue/vermillion is the most robustly distinguishable pair
# under deuteranopia and protanopia, and the two differ in lightness so they
# survive greyscale printing. There is no universal published convention for
# which is which, so the obligation is internal consistency, and this is it.
# Okabe-Ito, colourblind-safe. The one categorical palette for the whole run;
# `cycling` imports it rather than restating it, exactly as it does the
# charge/discharge colours below.
CATEGORICAL_PALETTE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7',
                       '#D55E00', '#56B4E9', '#F0E442', '#000000']
CATEGORICAL_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']


def categorical_colours(n):
    """`n` distinguishable colours, taken BY INDEX from the shared palette.

    Four figures used to build their colours as
    `plt.colormaps['tab10'](np.linspace(0, 1, n))`, which samples a
    CATEGORICAL colormap as though it were continuous. That is not a style
    difference, it is a defect: at n = 2 it returns #1f77b4 and #17becf — a
    blue and a cyan, which is precisely the pair a colourblind reader cannot
    separate — and at n = 3 a blue, a brown and a cyan. tab10's designed
    sequence (blue, orange, green, red, purple) is never reached. Two cells or
    three peaks is the common case, so the worst behaviour was the usual one.

    Indexing the Okabe-Ito palette instead gives the colours it was designed
    with, in order, and cycles beyond eight rather than interpolating.
    """
    n = max(int(n), 1)
    return [CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
            for i in range(n)]


def categorical_markers(n):
    """The matching marker shapes, so a series is identified by SHAPE as well
    as colour — the part that survives greyscale printing."""
    n = max(int(n), 1)
    return [CATEGORICAL_MARKERS[i % len(CATEGORICAL_MARKERS)]
            for i in range(n)]


# THE THREE PEAK KINDS, from the same palette as everything else.
# `plot_detected_peaks` encoded primary/shoulder/truncated as red/green/purple
# with darkred/darkgreen/indigo annotation text. The markers differ in shape,
# which carries the distinction for the scatter — but the peak-ID numerals
# beside them are distinguished by COLOUR ALONE, and red against green is
# exactly the pair `CATEGORICAL_PALETTE` exists to keep out of this run.
# NMC111 cell A draws three shoulders against four primaries in one panel.
# Vermillion, bluish-green and reddish-purple are separable under both
# deuteranopia and protanopia and differ in lightness for greyscale.
PEAK_KIND_COLOURS = {'primary': '#D55E00',      # Okabe-Ito vermillion
                     'shoulder': '#009E73',     # Okabe-Ito bluish green
                     'truncated': '#CC79A7'}    # Okabe-Ito reddish purple
PEAK_KIND_EDGES = {'primary': '#8C3D00',
                   'shoulder': '#006146',
                   'truncated': '#8A4A6E'}

COLOUR_CHARGE = '#0072B2'        # Okabe-Ito blue
COLOUR_DISCHARGE = '#D55E00'     # Okabe-Ito vermillion

# The old names, kept so nothing outside breaks.
DEFAULT_CHARGE_COLOUR = COLOUR_CHARGE
DEFAULT_DISCHARGE_COLOUR = COLOUR_DISCHARGE

# The waterfall does not drop cycles to fit the page. The figure height is
# bounded and the vertical pitch reduced until the whole series fits, so
# nothing is hidden from the reader. WATERFALL_OVERLAP sets how much a trace
# may be overlapped by its neighbour: 0.0 is a fully separated stack, 0.9 a
# dense ridgeline.

def _protocol_window(params, fallback=None):
    """The window the cell was CYCLED between, for a caption.

    NOT `voltage_range`, which is the analysis axis: the full extent of the
    data trimmed 50 mV, so that every half-cycle of a dataset shares one axis.
    Those are different quantities and the captions were quoting the second as
    though it were the first — stating "cycled between 1.25 and 2.63 V" for a
    cell cycled 1.2-2.5 V, and giving three different cut-offs for a triplicate
    cycled identically, because one cell started from open circuit at 2.684 V.
    A wrong experimental detail in text written to be pasted into a manuscript.

    Cell 3b measured the real thing (the median of the per-cycle limits) and
    put it in `user_parameters`; this reads it back.
    """
    lo = (params or {}).get("voltage_lower_V")
    hi = (params or {}).get("voltage_upper_V")
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return fallback
    return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) and hi > lo \
        else fallback


def _get_display_name(name, params, all_params):
    """
    Build a display name that distinguishes datasets with the same
    composition. Uses composition alone if unique across loaded datasets.
    If multiple datasets share a composition (triplicates), appends a
    cell identifier extracted from the filename (e.g., 'Cell A') or
    a numeric index.

    v1.8.1 fix: when filenames contain multiple numbers (e.g.
    Sample_2_1, Sample_2_2 ... Sample_2_9), the function now finds
    the numeric token that *varies* across the triplicate set rather
    than always capturing the first or last number. This prevents all
    cells receiving the same label (e.g. all labelled "Cell 2") when
    the batch number precedes the cell number.

    Usage in all plotting cells:
        composition = _get_display_name(name, params, user_parameters)
    """
    composition = params.get('composition', name)
    # Prefer the identifier confirmed in Cell 3 over re-parsing the
    # filename. The stored value is what the operator actually agreed to.
    _cid = str(params.get('cell_id', '') or '').strip()
    same_comp = [n for n, p in all_params.items()
                 if p.get('composition') == composition]
    if len(same_comp) > 1:
        if _cid:
            return f"{composition} (Cell {_cid})"
        # --- Pattern 1: single letter between underscores (_A_, _B_) ---
        match = re.search(r'_([A-Za-z])_', name)
        if match:
            return f"{composition} (Cell {match.group(1).upper()})"

        # --- Pattern 2: find the numeric token that distinguishes files ---
        # Extract all digit groups from each filename in the triplicate set
        all_tokens = {n: re.findall(r'\d+', n) for n in same_comp}
        my_tokens = all_tokens.get(name, [])

        if my_tokens:
            # Check each token position: the distinguishing token is the
            # one whose value differs across the triplicate filenames
            n_tokens = min(len(t) for t in all_tokens.values())
            for pos in range(n_tokens):
                values_at_pos = {t[pos] for t in all_tokens.values()
                                 if len(t) > pos}
                if len(values_at_pos) > 1:
                    # This position varies — it's the cell identifier
                    return f"{composition} (Cell {my_tokens[pos]})"

            # All token positions are identical (shouldn't happen with
            # genuinely different filenames) — fall back to last number
            return f"{composition} (Cell {my_tokens[-1]})"

        # --- Fallback: positional index ---
        idx = same_comp.index(name) + 1
        return f"{composition} (Cell {idx})"
    return composition


def _force_integer_cycles(ax):
    """
    Force x-axis to show only integer tick values starting from 0.
    Prevents matplotlib from displaying decimal cycle numbers (e.g.,
    2.5, 7.5) when the data range is small. Cycle number is always
    an integer — 2.5 cycles has no physical meaning.

    v1.8: also enforces steps of 5 or 10 for datasets with >20 cycles,
    and ensures the axis starts at 0 with sensible tick placement.

    Usage in all plotting cells:
        _force_integer_cycles(ax)
    """
    ax.set_xlim(left=0)
    x_max = ax.get_xlim()[1]

    if x_max <= 10:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    elif x_max <= 25:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    elif x_max <= 55:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    elif x_max <= 110:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    elif x_max <= 275:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    elif x_max <= 550:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    else:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(100))


def _discharge_label(params):
    """The label for the 'discharge' half-cycle. See `style.working_ion`."""
    return half_cycle_labels(params)[1]


def _charge_label(params):
    """The label for the 'charge' half-cycle. See `style.working_ion`."""
    return half_cycle_labels(params)[0]


def _apply_pub_style(ax, xlabel='Voltage / V',
                      ylabel='dQ/dV / mAh V$^{-1}$ g$^{-1}$'):
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis='both', labelsize=11, width=0.8, direction='in',
                   top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _detect_short_halfcycles(data, step,
                              threshold=SHORT_HALFCYCLE_THRESHOLD):
    """
    Split the half-cycles of one step into usable and too-short.

    A half-cycle is flagged short if its voltage span is less than
    `threshold` times the median span across all half-cycles of that
    step. This is a plotting-quality filter for dQ/dV, and is unrelated
    to the protocol-level completeness check in Cell 5b, which asks
    whether the cycler finished the cycle at all. Both were named
    alike before v1.8.6; they are different tests and were separated
    so they can no longer be confused.

    Returns (complete, short): two sorted lists of cycle numbers.
    """
    if threshold <= 0.0:
        all_cycles = sorted([c for c, s in data.keys() if s == step])
        return all_cycles, []

    ranges = {}
    for (cyc, s), df in data.items():
        if s == step:
            ranges[cyc] = df['Voltage'].max() - df['Voltage'].min()

    if not ranges:
        return [], []

    median_range = np.median(list(ranges.values()))
    complete = []
    short = []
    for cyc in sorted(ranges.keys()):
        if ranges[cyc] >= threshold * median_range:
            complete.append(cyc)
        else:
            short.append(cyc)

    return complete, short


def _waterfall_label_cycles(cycles, max_labels=WATERFALL_MAX_LABELS):
    """
    Choose a readable subset of cycle numbers to annotate.

    Always includes the first and last cycle plotted. Everything else
    is evenly spaced, so the annotation density stays constant whether
    the run is 20 cycles or 500.
    """
    if not cycles:
        return set()
    stride = max(1, int(np.ceil(len(cycles) / max(1, max_labels))))
    chosen = set(cycles[::stride])
    chosen.add(cycles[0])
    chosen.add(cycles[-1])
    return chosen


def plot_dqdv_all_cycles(processed_dqdv, user_parameters,
                          exclude_formation=EXCLUDE_FORMATION_FROM_YLIM,
                          save_location=None, file_format=None):
    for name, result in processed_dqdv.items():
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        colour_palette = params.get('colour_palette', 'viridis_r')
        data = result['data']

        all_cycles = sorted(set(c for c, s in data.keys()))

        complete_c, short_c = _detect_short_halfcycles(data, 'Charge')
        complete_d, short_d = _detect_short_halfcycles(data, 'Discharge')
        short_all = set(short_c + short_d)
        plot_cycles = [c for c in all_cycles if c not in short_all]

        if short_all:
            print(f"  Note: excluding short cycle(s) "
                  f"{sorted(short_all)} from all-cycle overlay")

        if not plot_cycles:
            print(f"  WARNING: No complete cycles for {name}")
            continue

        fig, ax = plt.subplots(figsize=(figure_width_inches, figure_height_inches))

        cmap = plt.get_cmap(colour_palette)
        norm = Normalize(vmin=min(plot_cycles), vmax=max(plot_cycles))

        ylim_vals = []

        for cycle in plot_cycles:
            colour = cmap(norm(cycle))

            if (cycle, 'Charge') in data:
                df_c = data[(cycle, 'Charge')]
                ax.plot(df_c['Voltage'], df_c['dQ/dV_processed'],
                       color=colour, linewidth=0.7, alpha=0.8)
                if not (exclude_formation and cycle == 1):
                    ylim_vals.extend(df_c['dQ/dV_processed'].values)

            if (cycle, 'Discharge') in data:
                df_d = data[(cycle, 'Discharge')]
                ax.plot(df_d['Voltage'], df_d['dQ/dV_processed'],
                       color=colour, linewidth=0.7, alpha=0.8)
                if not (exclude_formation and cycle == 1):
                    ylim_vals.extend(df_d['dQ/dV_processed'].values)

        ax.axhline(y=0, color='grey', linewidth=0.5, linestyle='-')
        _apply_pub_style(ax)
        ax.set_title(f'{composition} — dQ/dV (all cycles)', fontsize=13)

        if ylim_vals:
            ymin, ymax = min(ylim_vals), max(ylim_vals)
            ypad = (ymax - ymin) * 0.10
            ax.set_ylim(ymin - ypad, ymax + ypad)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
        cbar.set_label('Cycle number', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        plt.tight_layout()

        # The cycling window for the caption; the analysis
        # axis is not what the cell was cycled between.
        v_min, v_max = _protocol_window(
            params, fallback=result['voltage_range'])
        charge_rate = params.get('charge_rate_c', 'the specified')
        # "ALL N CYCLES" IS ONLY TRUE IF NONE WERE DROPPED. Short half-cycles
        # are excluded from this overlay a hundred lines above, announced on
        # the console and nowhere else — on NNM cell A that is 93 of 220
        # cycles, the whole second half of the cell's life, under a caption
        # written for pasting into a manuscript that says "all 127
        # galvanostatic cycles". A caption is a claim about the figure.
        _excluded = len(all_cycles) - len(plot_cycles)
        _cov = (f"for all {len(plot_cycles)} galvanostatic cycles of "
                if not _excluded else
                f"for {len(plot_cycles)} of the {len(all_cycles)} "
                f"galvanostatic cycles of ")
        caption = (
            f"Figure X. Differential capacity (dQ/dV) vs. voltage "
            + _cov
            + f"{composition}, cycled between {v_min:.2f} and "
            f"{v_max:.2f} V at {charge_rate} C. Positive values "
            f"correspond to {_charge_label(params).lower()} and "
            f"negative values to {_discharge_label(params).lower()}. "
            f"Colour indicates cycle number as shown in the colourbar."
            + (f" {_excluded} cycle(s) whose charge or discharge half-cycle "
               f"did not span the analysed window are not shown; they are in "
               f"the processed-dQ/dV table."
               if _excluded else "")
        )
        # A half-cycle Cell 5 judged was not a measurement is KEPT in the
        # figure deliberately — seeing where the run has got to is the point —
        # but a caption bound for a manuscript has to say so. Cell 6's caption
        # did; this one said "for all 11 galvanostatic cycles" and left the
        # reader to find out that the eleventh was still running.
        _inc = sorted(int(c) for c in (params.get("incomplete_cycles") or ()))
        if _inc:
            caption += (
                f" Cycle{'s' if len(_inc) > 1 else ''} "
                f"{', '.join(str(c) for c in _inc)} "
                f"{'were' if len(_inc) > 1 else 'was'} incomplete at export "
                f"and {'are' if len(_inc) > 1 else 'is'} shown for "
                f"completeness only.")
        _caption(caption)

        if save_location:
            fpath = os.path.join(
                save_location,
                f'{name}_dQdV_all_cycles.{image_format(params, file_format)}')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            saved(fpath)

        plt.show()
        plt.close(fig)


def plot_dqdv_key_cycles_combined(processed_dqdv, user_parameters,
                                   exclude_formation=EXCLUDE_FORMATION_FROM_YLIM,
                                   save_location=None, file_format=None):
    for name, result in processed_dqdv.items():
        params = user_parameters.get(name, {})
        key_cycles = params.get('key_cycles', DEFAULT_KEY_CYCLES)
        composition = _get_display_name(name, params, user_parameters)
        colour_palette = params.get('colour_palette', 'viridis_r')
        data = result['data']

        available_cycles = sorted(set(c for c, s in data.keys()))
        key_cycles = [c for c in key_cycles if c in available_cycles]

        if not key_cycles:
            print(f"  WARNING: No key cycles found for {name}")
            continue

        fig, ax = plt.subplots(figsize=(figure_width_inches, figure_height_inches))

        cmap = plt.get_cmap(colour_palette)
        norm = Normalize(vmin=min(key_cycles), vmax=max(key_cycles))

        ylim_vals = []

        for cycle in key_cycles:
            colour = cmap(norm(cycle))

            if (cycle, 'Charge') in data:
                df_c = data[(cycle, 'Charge')]
                ax.plot(df_c['Voltage'], df_c['dQ/dV_processed'],
                       color=colour, linewidth=1.2, alpha=0.9,
                       linestyle='-')
                if not (exclude_formation and cycle == 1):
                    ylim_vals.extend(df_c['dQ/dV_processed'].values)

            if (cycle, 'Discharge') in data:
                df_d = data[(cycle, 'Discharge')]
                ax.plot(df_d['Voltage'], df_d['dQ/dV_processed'],
                       color=colour, linewidth=1.2, alpha=0.9,
                       linestyle='-')
                if not (exclude_formation and cycle == 1):
                    ylim_vals.extend(df_d['dQ/dV_processed'].values)

        ax.axhline(y=0, color='grey', linewidth=0.5, linestyle='-')
        _apply_pub_style(ax)
        ax.set_title(f'{composition} — dQ/dV (key cycles)', fontsize=13)

        if ylim_vals:
            ymin, ymax = min(ylim_vals), max(ylim_vals)
            ypad = (ymax - ymin) * 0.10
            ax.set_ylim(ymin - ypad, ymax + ypad)

        if len(key_cycles) > 6:
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
            cbar.set_label('Cycle number', fontsize=12)
            cbar.ax.tick_params(labelsize=10)
        else:
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], color=cmap(norm(c)), linewidth=1.5,
                       label=f'Cycle {int(c)}')
                for c in key_cycles
            ]
            ax.legend(handles=handles, fontsize=9, framealpha=0.7,
                     loc='best')

        plt.tight_layout()

        cycles_str = ', '.join(str(int(c)) for c in key_cycles)
        # The cycling window for the caption; the analysis
        # axis is not what the cell was cycled between.
        v_min, v_max = _protocol_window(
            params, fallback=result['voltage_range'])
        charge_rate = params.get('charge_rate_c', 'the specified')
        caption = (
            f"Figure X. Differential capacity (dQ/dV) vs. voltage "
            f"for selected cycles ({cycles_str}) of {composition}, "
            f"cycled between {v_min:.2f} and {v_max:.2f} V at "
            f"{charge_rate} C. Positive values correspond to "
            f"{_charge_label(params).lower()} and negative values to "
            f"{_discharge_label(params).lower()}."
        )
        # See the same note on `plot_dqdv_all_cycles`: a key-cycle figure can
        # include an incomplete cycle too, if the operator listed it.
        _inc = sorted(int(c) for c in (params.get("incomplete_cycles") or ())
                      if int(c) in set(int(x) for x in key_cycles))
        if _inc:
            caption += (
                f" Cycle{'s' if len(_inc) > 1 else ''} "
                f"{', '.join(str(c) for c in _inc)} "
                f"{'were' if len(_inc) > 1 else 'was'} incomplete at export "
                f"and {'are' if len(_inc) > 1 else 'is'} shown for "
                f"completeness only.")
        _caption(caption)

        if save_location:
            fpath = os.path.join(
                save_location,
                f'{name}_dQdV_key_cycles.{image_format(params, file_format)}')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            saved(fpath)

        plt.show()
        plt.close(fig)


def plot_dqdv_key_cycles_split(processed_dqdv, user_parameters,
                                exclude_formation=EXCLUDE_FORMATION_FROM_YLIM,
                                save_location=None, file_format=None):
    for name, result in processed_dqdv.items():
        params = user_parameters.get(name, {})
        key_cycles = params.get('key_cycles', DEFAULT_KEY_CYCLES)
        composition = _get_display_name(name, params, user_parameters)
        data = result['data']

        available_cycles = sorted(set(c for c, s in data.keys()))
        key_cycles = [c for c in key_cycles if c in available_cycles]

        if not key_cycles:
            continue

        # Two side-by-side panels: full journal width, shorter height
        fig, (ax_charge, ax_discharge) = plt.subplots(
            1, 2, figsize=(figure_width_inches, figure_height_inches * 0.75),
            sharey=False)
        fig.suptitle(f'{composition}', fontsize=14, fontweight=FIGURE_TITLE_WEIGHT)

        norm = Normalize(vmin=min(key_cycles), vmax=max(key_cycles))
        cmap_charge = plt.colormaps['Blues']
        cmap_discharge = plt.colormaps['Oranges']

        ylim_data = {'Charge': [], 'Discharge': []}

        for cycle in key_cycles:
            intensity = 0.3 + 0.7 * norm(cycle)

            if (cycle, 'Charge') in data:
                df_c = data[(cycle, 'Charge')]
                ax_charge.plot(
                    df_c['Voltage'], df_c['dQ/dV_processed'],
                    color=cmap_charge(intensity),
                    linewidth=1.2, label=f'Cycle {cycle}')
                if not (exclude_formation and cycle == 1):
                    ylim_data['Charge'].extend(
                        df_c['dQ/dV_processed'].values)

            if (cycle, 'Discharge') in data:
                df_d = data[(cycle, 'Discharge')]
                ax_discharge.plot(
                    df_d['Voltage'], df_d['dQ/dV_processed'],
                    color=cmap_discharge(intensity),
                    linewidth=1.2, label=f'Cycle {cycle}')
                if not (exclude_formation and cycle == 1):
                    ylim_data['Discharge'].extend(
                        df_d['dQ/dV_processed'].values)

        chg_t = _charge_label(params)
        dch_t = _discharge_label(params)
        for ax, title, step in [(ax_charge, chg_t, 'Charge'),
                                 (ax_discharge, dch_t, 'Discharge')]:
            _apply_pub_style(ax)
            ax.set_title(title, fontsize=12)
            ax.legend(fontsize=8, framealpha=0.7, loc='best')
            ax.axhline(y=0, color='grey', linewidth=0.5, linestyle='-')

            if ylim_data[step]:
                ymin = min(ylim_data[step])
                ymax = max(ylim_data[step])
                ypad = (ymax - ymin) * 0.10
                ax.set_ylim(ymin - ypad, ymax + ypad)

        plt.tight_layout()

        if save_location:
            fpath = os.path.join(
                save_location,
                f'{name}_dQdV_key_cycles_split.{image_format(params, file_format)}')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            saved(fpath)

        plt.show()
        plt.close(fig)


def plot_dqdv_waterfall(processed_dqdv, user_parameters, step='Charge',
                         every_n=1, offset_scale=None,
                         overlap=WATERFALL_OVERLAP,
                         max_height_in=WATERFALL_MAX_HEIGHT_IN,
                         save_location=None, file_format=None):
    """
    Stacked-offset dQ/dV traces, one per cycle.

    Every complete cycle is plotted. Nothing is thinned automatically:
    a long run produces a denser ridgeline, not a shorter subset, so a
    change confined to a handful of cycles cannot be lost to sampling.

    Density is controlled by geometry rather than by discarding data.
    The figure height is clamped between WATERFALL_MIN_HEIGHT_IN and
    max_height_in, and the vertical pitch between traces is set to
    (1 - overlap) times the median trace amplitude, so raising
    `overlap` packs more cycles into the same page.

    Pass every_n > 1 to sample deliberately; the omitted cycles are
    then listed in the console so the choice is on the record.
    """
    for name, result in processed_dqdv.items():
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        data = result['data']

        complete, short = _detect_short_halfcycles(data, step)
        if short:
            print(f"  Note: excluding short {step.lower()} cycle(s) "
                  f"{short} from waterfall")

        cycles_to_plot = complete[::every_n]
        if every_n > 1:
            omitted = [c for c in complete if c not in set(cycles_to_plot)]
            print(f"  every_n={every_n}: plotting {len(cycles_to_plot)} of "
                  f"{len(complete)} cycles. Omitted: {omitted}")

        if not cycles_to_plot:
            print(f"  WARNING: No {step} data found for {name}.")
            continue

        n_cycles = len(cycles_to_plot)

        # --- Vertical pitch between traces -------------------------
        amplitudes = []
        for cyc in cycles_to_plot:
            if (cyc, step) in data:
                vals = data[(cyc, step)]['dQ/dV_processed']
                amplitudes.append(float(vals.max() - vals.min()))
        median_amp = float(np.median(amplitudes)) if amplitudes else 1.0

        if offset_scale is not None:
            pitch = float(offset_scale)
        else:
            overlap = float(np.clip(overlap, 0.0, 0.95))
            pitch = max(median_amp * (1.0 - overlap), median_amp * 0.05)

        # --- Figure height, bounded --------------------------------
        wanted_h = n_cycles * WATERFALL_HEIGHT_PER_CYCLE_IN
        _lo = min(max(WATERFALL_MIN_HEIGHT_IN, figure_height_inches),
                  max_height_in)
        fig_h = float(np.clip(wanted_h, _lo, max_height_in))
        if wanted_h > max_height_in and n_cycles > WATERFALL_DENSE_NOTE_ABOVE:
            print(f"  {n_cycles} cycles in {fig_h:.1f} in: traces overlap "
                  f"by {overlap:.0%}. All cycles are shown. Raise "
                  f"max_height_in for a taller figure, or use the "
                  f"heatmap for a compact overview.")

        fig, ax = plt.subplots(figsize=(figure_width_inches, fig_h))

        cmap = plt.colormaps['viridis_r']
        norm = Normalize(vmin=min(cycles_to_plot),
                         vmax=max(cycles_to_plot))

        dense = n_cycles > 25
        line_w = 0.6 if dense else 0.9
        effects = ([path_effects.withStroke(linewidth=line_w + 1.1,
                                            foreground='white')]
                   if dense else None)

        label_cycles = _waterfall_label_cycles(cycles_to_plot)
        v_lo, v_hi = np.inf, -np.inf

        for i, cycle in enumerate(cycles_to_plot):
            if (cycle, step) not in data:
                continue
            df = data[(cycle, step)]
            offset = i * pitch
            colour = cmap(norm(cycle))

            ax.plot(df['Voltage'], df['dQ/dV_processed'] + offset,
                    color=colour, linewidth=line_w, alpha=1.0,
                    zorder=2 + i, path_effects=effects,
                    solid_capstyle='round')

            v_lo = min(v_lo, float(df['Voltage'].min()))
            v_hi = max(v_hi, float(df['Voltage'].max()))

            if cycle in label_cycles:
                ax.text(float(df['Voltage'].max()), offset,
                        f' {cycle}', fontsize=8, va='center', ha='left',
                        color=colour, zorder=2 + n_cycles + i,
                        clip_on=False)

        # Room on the right for the cycle labels
        if np.isfinite(v_lo) and np.isfinite(v_hi) and v_hi > v_lo:
            ax.set_xlim(v_lo, v_hi + 0.06 * (v_hi - v_lo))

        _apply_pub_style(
            ax, ylabel=f'dQ/dV (offset) / mAh V$^{{-1}}$ g$^{{-1}}$')
        step_display = (_charge_label(params) if step == 'Charge'
                        else _discharge_label(params))
        ax.set_title(
            f'{composition} — {step_display} dQ/dV waterfall '
            f'({n_cycles} cycles)', fontsize=13)
        ax.set_yticks([])

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=30, pad=0.08)
        cbar.set_label('Cycle number', fontsize=11)

        # --- HOW FAR DOES THE TALLEST TRACE REACH? ---------------------
        # The pitch is a fraction of the MEDIAN amplitude, which is what keeps
        # a hundred-cycle waterfall readable — at OVERLAP = 0.5 a median trace
        # spans two lanes BY DESIGN. So "taller than the pitch" is true of
        # nearly every trace and says nothing; the first version of this note
        # reported 17 of 17 traces and was worse than silence. What matters is
        # a trace much taller than the median the pitch came from: on LTO cell
        # C, cycle 1 is 1.54x the median and is drawn through the two rows
        # above it, which is how its doublet was read as a feature of cycles
        # 2 and 3. The heatmap does not offset and so cannot bleed — which is
        # why the same feature is absent there.
        #
        # No threshold is chosen. The note always states the worst case, so a
        # reader can calibrate the figure in front of them rather than trust a
        # number picked here; the console speaks up only when it is large.
        _amp = {}
        for cyc in cycles_to_plot:
            if (cyc, step) in data:
                _v = data[(cyc, step)]['dQ/dV_processed']
                _amp[int(cyc)] = float(_v.max() - _v.min())
        if _amp and pitch > 0 and median_amp > 0:
            _tall = max(_amp, key=_amp.get)
            _ratio = _amp[_tall] / median_amp
            _lanes = _amp[_tall] / pitch
            _over = max(0, int(np.ceil(_lanes)) - 1)
            fig.text(0.01, -0.01,
                     f"Traces are offset by {pitch / median_amp:.0%} of the "
                     f"median amplitude, so a median trace spans "
                     f"{median_amp / pitch:.1f} rows. The tallest (cycle "
                     f"{_tall}) is {_ratio:.1f}x the median and is drawn "
                     f"through the {_over} row(s) above it — a feature seen "
                     f"there may belong to a lower cycle. The heatmap does "
                     f"not offset.",
                     fontsize=7, style="italic", va="top", wrap=True)
            if _ratio > 1.5:
                _big = sorted((c for c, a in _amp.items()
                               if a / median_amp > 1.5),
                              key=lambda c: -_amp[c])
                print(f"  {step}: cycle {_tall} is {_ratio:.1f}x the median "
                      f"trace and reaches {_over} row(s) above its own"
                      + (f" ({len(_big)} trace(s) over 1.5x: "
                         + ", ".join(str(c) for c in _big[:8])
                         + (" ..." if len(_big) > 8 else "") + ")"
                         if len(_big) > 1 else "")
                      + ". Use the heatmap to place a feature exactly.")

        plt.tight_layout()

        if save_location:
            fpath = os.path.join(
                save_location,
                f'{name}_dQdV_waterfall_{step}.{image_format(params, file_format)}')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            saved(fpath)

        plt.show()
        plt.close(fig)


def plot_dqdv_heatmap(processed_dqdv, user_parameters, step='Charge',
                       v_bins=200, save_location=None, file_format=None,
                       envelope=True, suffix=''):
    """
    dQ/dV against voltage and cycle, as a heat map.

    `envelope` draws the step line at the edge of the range each cycle
    actually traversed, and the note that goes with it. It is the honest
    reading of the panel — the cells outside that line are unmeasured, not
    zero — but it is also a line across a figure, and a figure meant for a
    talk sometimes wants the shape alone. Both are produced by Cell 7b, which
    is why `suffix` exists: the two versions cannot share a filename.
    """
    for name, result in processed_dqdv.items():
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        data = result['data']
        # The cycling window for the caption; the analysis
        # axis is not what the cell was cycled between.
        v_min, v_max = _protocol_window(
            params, fallback=result['voltage_range'])

        complete, short = _detect_short_halfcycles(data, step)
        if short:
            print(f"  Note: excluding short {step.lower()} cycle(s) "
                  f"{short} from heatmap")

        cycles = complete
        if not cycles:
            continue

        # THE GRID COMES FROM THE DATA, NOT FROM THE PROTOCOL WINDOW.
        # `v_min, v_max` above are the window the cell was CYCLED between and
        # are the right thing to put in the caption; they are the wrong thing
        # to bin onto. The analysed axis is not the protocol window — LTO cell
        # A is cycled 1.2-2.5 V and analysed to 2.634 V — so a grid ending at
        # 2.5 V threw away 130 mV of real charge data, about a tenth of the
        # figure. At the other end `np.interp` does not return NaN outside the
        # data range: it CLAMPS, repeating the endpoint value, so the 51 mV
        # below the first measured point was painted as a flat band of colour
        # that no measurement supports. Both faults are silent, and both are
        # present on every dataset in every run on disk.
        _vals = [data[(c, step)]['Voltage'].values
                 for c in cycles if (c, step) in data]
        _vals = [v[np.isfinite(v)] for v in _vals]
        _vals = [v for v in _vals if v.size]
        if _vals:
            v_lo = float(min(v.min() for v in _vals))
            v_hi = float(max(v.max() for v in _vals))
        else:
            v_lo, v_hi = float(v_min), float(v_max)
        if not np.isfinite(v_lo) or not np.isfinite(v_hi) or v_hi <= v_lo:
            v_lo, v_hi = float(v_min), float(v_max)
        v_edges = np.linspace(v_lo, v_hi, v_bins + 1)
        v_centres = 0.5 * (v_edges[:-1] + v_edges[1:])

        heatmap_data = np.full((len(cycles), v_bins), np.nan)

        for i, cycle in enumerate(cycles):
            if (cycle, step) not in data:
                continue
            df = data[(cycle, step)]
            _v = df['Voltage'].values
            _y = df['dQ/dV_processed'].values
            _m = np.isfinite(_v) & np.isfinite(_y)
            if _m.sum() < 2:
                continue
            _v, _y = _v[_m], _y[_m]
            _row = np.interp(v_centres, _v, _y)
            # ...and a cell that did not reach a voltage this cycle gets NaN
            # there, drawn as the axes background, rather than a repeat of its
            # nearest measured value. A short half-cycle should look short.
            _row[(v_centres < _v.min()) | (v_centres > _v.max())] = np.nan
            heatmap_data[i, :] = _row

        fig, ax = plt.subplots(figsize=(figure_width_inches, figure_height_inches))

        plot_data = (np.abs(heatmap_data) if step == 'Discharge'
                     else heatmap_data)
        # An all-NaN panel is possible now that unvisited voltages are NaN
        # rather than a repeated endpoint, and `nanpercentile` raises on one.
        if not np.isfinite(plot_data).any():
            print(f"  {name}: no {step.lower()} data in the analysed window, "
                  f"heatmap skipped")
            plt.close(fig)
            continue
        vmax_clim = np.nanpercentile(plot_data, 98)

        # NOT-VISITED IS DARK, AND THE COLLAPSE IS A LINE.
        #
        # Two things were wrong with the original and one with the first fix.
        #
        # ORIGINAL: `np.interp` clamps outside the data range rather than
        # returning NaN, so every voltage a half-cycle never reached was
        # painted with a repeat of its nearest measured value. On this
        # colormap that renders near-black, which looks right and is the
        # reason it survived — but a fabricated zero and a measured zero are
        # the same pixel, and the region carried no information at all.
        #
        # FIRST FIX: NaN there, drawn light grey. Honest, and unusable. The
        # visited span varies enormously between cycles — the MEDIAN cycle
        # covers 56% of the grid on LTO cell A and 33% on NNM cell B — so the
        # panel became a large pale wedge with a thin bright line beside it.
        #
        # WHAT IT DOES NOW. Unvisited stays dark, close enough to the zero end
        # of the ramp that the figure reads as a heat map rather than as a
        # hole; the information the wedge was carrying is drawn as an ENVELOPE
        # instead — a step line at the highest and lowest voltage each cycle
        # actually reached. A line states the collapse more loudly than a void
        # does, because it has a shape you can read off a scale, and it costs
        # no panel area. Nothing is fabricated: the cells behind the envelope
        # are masked, not filled.
        _note_room = 0.0        # reserved only if the envelope note is drawn
        _n_missing = int(np.isnan(plot_data).sum())

        # DECIDED BEFORE THE COLOURS ARE CHOSEN, because the tint on the
        # unvisited cells is only defensible where the line and the note that
        # explain it are also drawn. On LTO cell A's delithiation the two
        # edges move 38 mV — below both gates, so no line and no note — and
        # the panel still carried a faint unexplained shade along one edge,
        # which is a mark on a figure that nothing on the figure accounts for.
        # Where the envelope is not worth drawing, the mask takes the
        # colormap's own zero and the diagnostic panel is byte-identical to
        # the plain one: nothing to explain, so nothing said.
        _bw = float(v_edges[1] - v_edges[0])
        _hi_env = np.array([np.nanmax(v) if v.size else np.nan for v in _vals])
        _lo_env = np.array([np.nanmin(v) if v.size else np.nan for v in _vals])
        _spread = max(np.nanmax(_hi_env) - np.nanmin(_hi_env),
                      np.nanmax(_lo_env) - np.nanmin(_lo_env))
        _panel = float(v_edges[-1] - v_edges[0])
        _worth = bool(envelope and _n_missing and np.isfinite(_spread)
                      and _spread > HEATMAP_ENVELOPE_MIN_BINS * _bw
                      and _panel > 0
                      and _spread / _panel > HEATMAP_ENVELOPE_MIN_FRACTION)

        _cmap = plt.get_cmap('inferno').copy()
        # `envelope=False` IS "as it was before", not "as it is now, minus the
        # line". The point of the second file is a panel with nothing on it to
        # explain — so the unvisited cells take the colormap's own zero colour
        # and the panel reads as one continuous surface, which is what the
        # clamped fill used to produce. The cells are still MASKED rather than
        # filled with a repeat of the nearest measured value: the appearance is
        # the old one, but nothing per-cycle is invented to get it, so the two
        # files differ in what they SAY and never in what was measured.
        _cmap.set_bad(HEATMAP_UNVISITED_COLOUR if _worth else _cmap(0.0))
        im = ax.pcolormesh(
            v_centres, cycles, np.ma.masked_invalid(plot_data),
            cmap=_cmap, shading='auto',
            vmin=0, vmax=vmax_clim)

        # The envelope itself. A cell whose every half-cycle covers the same
        # span has a straight line at each edge and does not need it; `_worth`
        # above is that decision, and it also chose the mask colour.
        if _worth:
            for _e in (_hi_env, _lo_env):
                if np.nanmax(_e) - np.nanmin(_e) <= _bw:
                    continue          # that edge is flat; drawing it is clutter
                ax.step(_e, cycles, where='mid',
                        color=HEATMAP_ENVELOPE_COLOUR, linewidth=1.3,
                        solid_joinstyle='miter', zorder=6)
            # BELOW the axes, and WRAPPED. Anywhere inside them collides with
            # either the peak (left) or the envelope itself (right), and a note
            # that obscures the thing it explains is worse than no note. It has
            # to be wrapped by hand: `bbox_inches='tight'` grows the CANVAS to
            # fit a figure-level text, so one long line widens the saved image
            # and squeezes the axes into the middle of it.
            # BOTH EDGES, DESCRIBED AS BOTH EDGES. Either end of the
            # traversed range can be the one that moves — LTO's charge step
            # stops progressively lower, NMC's starts progressively higher —
            # and a note that says "furthest voltage reached" is simply wrong
            # about the low edge, which is the one drawn on NMC.
            # Balanced across two lines: `bbox_inches='tight'` grows the
            # canvas to fit the widest line, so a long first line widens the
            # saved image and shrinks the axes inside it.
            _env_note = (f'Blue line: the edge of the voltage range the cell '
                         f'traversed on that cycle,\nmoving '
                         f'{1000 * _spread:.0f} mV over this record. Outside '
                         f'it the cell did not go — unmeasured, not zero.')
            fig.text(0.5, 0.105, _env_note, ha='center', va='top',
                     fontsize=8.5, style='italic', color='#444444',
                     linespacing=1.4)
            _note_room = 0.12

        cbar_label = '|dQ/dV|' if step == 'Discharge' else 'dQ/dV'
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(
            f'{cbar_label} / mAh V$^{{-1}}$ g$^{{-1}}$', fontsize=11)

        _apply_pub_style(ax, ylabel='Cycle number')
        step_display = (_charge_label(params) if step == 'Charge'
                        else _discharge_label(params))
        ax.set_title(f'{composition} — {step_display} dQ/dV evolution',
                     fontsize=13)

        fig.tight_layout(rect=(0, _note_room, 1, 1))

        if save_location:
            fpath = os.path.join(
                save_location,
                f'{name}_dQdV_heatmap_{step}{suffix}'
                f'.{image_format(params, file_format)}')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            saved(fpath)

        plt.show()
        plt.close(fig)


def plot_preprocessing_qc(processed_dqdv, user_parameters, cycle=1, *,
                          save_location=None, file_format=None):
    """Raw against processed, for one cycle.

    `save_location` is new. This was the ONLY plot function in the module
    without it: it drew the figure, called `plt.show()`, and that was the end
    of it. Seven dQ/dV figures reached the run folder and the one that
    justifies the preprocessing did not — so a run folder read a month later
    had no record of what the smoothing and binning actually did.
    """
    for name, result in processed_dqdv.items():
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        data = result['data']

        # Two side-by-side panels: full journal width, shorter height
        fig, (ax_c, ax_d) = plt.subplots(
            1, 2, figsize=(figure_width_inches, figure_height_inches * 0.75))
        fig.suptitle(
            f'{composition} — Cycle {cycle} preprocessing QC',
            fontsize=13)

        chg_t = _charge_label(params)
        dch_t = _discharge_label(params)
        for ax, step, title in [(ax_c, 'Charge', chg_t),
                                 (ax_d, 'Discharge', dch_t)]:
            if (cycle, step) not in data:
                ax.text(0.5, 0.5, f'No {step} data\nfor cycle {cycle}',
                        transform=ax.transAxes, ha='center',
                        va='center', fontsize=12, color='grey')
                ax.set_title(title, fontsize=12)
                continue

            df = data[(cycle, step)]
            ax.plot(df['Voltage'], df['dQ/dV_raw'],
                    color='grey', alpha=0.5, linewidth=0.8,
                    label='Raw')
            ax.plot(df['Voltage'], df['dQ/dV_processed'],
                    color='#0072B2', linewidth=1.3,
                    label='Processed')

            _apply_pub_style(ax)
            ax.set_title(title, fontsize=12)
            ax.legend(fontsize=9, framealpha=0.7)
            ax.axhline(y=0, color='grey', linewidth=0.4, linestyle='-')

        plt.tight_layout()
        if save_location:
            fpath = os.path.join(
                save_location,
                f'{name}_preprocessing_qc.{image_format(params, file_format)}')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            saved(fpath)
        plt.show()
        plt.close(fig)


def plot_detected_peaks(processed_dqdv, detected_peaks, user_parameters,
                         cycles_to_show=None, save_location=None,
                         file_format=None):
    """
    Annotated dQ/dV plots with detected peaks marked.
    Markers:
      ▲/▼  red   = primary peaks
      ■    green  = shoulder peaks (d² detected)
      ◆    purple = truncated peaks (retained but clipped at window edge)
    """
    import matplotlib.pyplot as plt

    for name in processed_dqdv:
        result = processed_dqdv[name]
        det    = detected_peaks[name]
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        ref_cycle   = det['reference_cycle']
        data        = result['data']

        available = sorted(set(c for c, s in data.keys()))
        # Which cycles have BOTH half-cycles present (a complete cycle).
        _steps_by_cycle = {}
        for (c, s) in data.keys():
            _steps_by_cycle.setdefault(c, set()).add(s)
        complete_cycles = sorted(
            c for c, steps in _steps_by_cycle.items()
            if {'Charge', 'Discharge'} <= steps)
        if cycles_to_show is None:
            # Show the user's key cycles (those that exist in this
            # dataset), and always add the reference cycle and the last
            # complete cycle, so the fingerprint and the aged end are
            # both visible regardless of what key_cycles contains. The
            # last cycle is often incomplete (odd half-cycle count), so
            # fall back to the last complete one rather than plotting a
            # half-cycle on its own.
            key_cycles = params.get('key_cycles',
                                    DEFAULT_KEY_CYCLES)
            wanted = [c for c in key_cycles if c in available]
            wanted.append(ref_cycle)
            if complete_cycles:
                wanted.append(complete_cycles[-1])
            elif available:
                wanted.append(available[-1])
            cycles_to_show_actual = sorted(set(wanted))
        else:
            cycles_to_show_actual = [c for c in cycles_to_show
                                      if c in available]

        n_show = len(cycles_to_show_actual)
        fig, axes = plt.subplots(n_show, 2, figsize=(13, 4 * n_show),
                                  squeeze=False)
        fig.suptitle(f'{composition} — detected peaks', fontsize=14,
                     fontweight=FIGURE_TITLE_WEIGHT)

        for row, cycle in enumerate(cycles_to_show_actual):
            for col, step in enumerate(['Charge', 'Discharge']):
                ax = axes[row, col]
                if (cycle, step) not in data:
                    ax.set_visible(False)
                    continue

                df = data[(cycle, step)]
                v, dq = df['Voltage'].values, df['dQ/dV_processed'].values
                is_ref = (cycle == ref_cycle)
                clr = COLOUR_CHARGE if step == 'Charge' else COLOUR_DISCHARGE
                ax.plot(v, dq, color=clr, linewidth=1.5 if is_ref else 1.0,
                        alpha=0.9)

                peaks = det['peaks'].get((cycle, step), pd.DataFrame())
                if not peaks.empty:
                    primary   = peaks[~peaks['is_shoulder'] & ~peaks['is_truncated']]
                    shoulders = peaks[peaks['is_shoulder']]
                    truncated = peaks[peaks['is_truncated']]
                    mk = '^' if step == 'Charge' else 'v'

                    # THE MARKER GOES ON THE CURVE THAT IS DRAWN. It used to
                    # be placed at `height_original`, which is the height on
                    # the RAW dQ/dV, while the line plotted here is
                    # `dQ/dV_processed`. Wherever processing changes the
                    # curve — a sharp formation spike is the clearest case —
                    # the marker floated off the line, and a peak found
                    # exactly at the maximum LOOKED like a miss. Detection was
                    # right and the picture said otherwise, which is the worse
                    # of the two failures.
                    #
                    # Interpolating onto the drawn curve is correct whatever
                    # the columns mean, so this cannot drift again if the
                    # processing changes.
                    def _on_curve(sub):
                        return np.interp(np.asarray(sub['voltage'], float),
                                         v, dq)

                    if not primary.empty:
                        ax.scatter(primary['voltage'], _on_curve(primary),
                                   color=PEAK_KIND_COLOURS['primary'], s=50,
                                   zorder=5, marker=mk,
                                   edgecolors=PEAK_KIND_EDGES['primary'],
                                   linewidths=0.5)
                    if not shoulders.empty:
                        ax.scatter(shoulders['voltage'], _on_curve(shoulders),
                                   color=PEAK_KIND_COLOURS['shoulder'], s=50,
                                   zorder=5, marker='s',
                                   edgecolors=PEAK_KIND_EDGES['shoulder'],
                                   linewidths=0.5)
                    if not truncated.empty:
                        ax.scatter(truncated['voltage'], _on_curve(truncated),
                                   color=PEAK_KIND_COLOURS['truncated'], s=60,
                                   zorder=5, marker='D',
                                   edgecolors=PEAK_KIND_EDGES['truncated'],
                                   linewidths=0.5)
                    for _, pk in peaks.iterrows():
                        if pk['is_truncated']:
                            clr_a = PEAK_KIND_EDGES['truncated']
                        elif pk['is_shoulder']:
                            clr_a = PEAK_KIND_EDGES['shoulder']
                        else:
                            clr_a = PEAK_KIND_EDGES['primary']
                        ax.annotate(
                            f'{int(pk["peak_id"])}',
                            (pk['voltage'],
                             float(np.interp(float(pk['voltage']), v, dq))),
                            textcoords='offset points',
                            xytext=(0, 12 if step == 'Charge' else -14),
                            ha='center', fontsize=9, fontweight='bold',
                            color=clr_a
                        )

                n_p  = len(peaks)
                n_sh = peaks['is_shoulder'].sum()  if not peaks.empty else 0
                n_tr = peaks['is_truncated'].sum() if not peaks.empty else 0
                extras = []
                if n_sh: extras.append(f'{n_sh} shoulder')
                if n_tr: extras.append(f'{n_tr} truncated')
                label = f'{n_p} peaks' + (f' ({", ".join(extras)})' if extras else '')
                suffix = ' (REFERENCE)' if is_ref else ''
                ax.set_title(f'Cycle {cycle} — {step}{suffix}', fontsize=11)
                ax.set_xlabel('Voltage / V', fontsize=11)
                ax.set_ylabel('dQ/dV / mAh V$^{-1}$ g$^{-1}$', fontsize=11)
                ax.tick_params(axis='both', labelsize=10, direction='in',
                              top=True, right=True)
                ax.axhline(y=0, color='grey', linewidth=0.4)
                ax.text(0.02, 0.95, label, transform=ax.transAxes,
                       fontsize=9, va='top', ha='left',
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white', alpha=0.8))

        plt.tight_layout()
        if save_location:
            fpath = os.path.join(save_location,
                                f'{name}_detected_peaks.{image_format(params, file_format)}')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            saved(fpath)
        plt.show()
        plt.close(fig)

        # A shoulder is not a local maximum, so `find_peaks` never measured a
        # prominence or a width for it: `detect.py` fills the prominence with
        # a placeholder of 0.1 x height and leaves the width at zero. Both are
        # printed as what they are. Until 1.9.0.33 this table showed the
        # placeholder as "18.014" and the missing width as "0.0", side by side
        # with measured values in the same columns — on a broad layered oxide
        # where most detections are shoulders, almost every number in the
        # table was fabricated and nothing said so.
        print(f"\n  Reference peak summary (Cycle {ref_cycle}):")
        _any_estimated = False
        for step in ['Charge', 'Discharge']:
            rp = det['reference_peaks'][step]
            if rp is not None and not rp.empty:
                print(f"\n  {step}:")
                print(f"  {'Peak':>4}  {'Voltage/V':>10}  {'Height':>10}  "
                      f"{'Prominence':>12}  {'FWHM/mV':>8}  {'Type':>12}")
                for _, pk in rp.iterrows():
                    if pk['is_truncated']:
                        ptype = 'truncated'
                    elif pk['is_shoulder']:
                        ptype = 'shoulder'
                    else:
                        ptype = 'primary'
                    est = bool(pk.get('prominence_estimated', False))
                    _any_estimated = _any_estimated or est
                    prom = (f"~{pk['prominence']:.3f}" if est
                            else f"{pk['prominence']:.3f}")
                    _w = float(pk['width_V']) * 1000.0
                    width = ("--" if (not np.isfinite(_w) or _w <= 0.0)
                             else f"{_w:.1f}")
                    print(f"  {int(pk['peak_id']):>4}  {pk['voltage']:>10.4f}  "
                          f"{pk['height']:>10.3f}  {prom:>12}  "
                          f"{width:>8}  {ptype:>12}")
        if _any_estimated:
            print("\n  ~ prominence NOT measured — a shoulder is not a local "
                  "maximum, so this is a\n    placeholder of 0.1 x height, and "
                  "it is what ranks the peak if the list is\n    capped. "
                  "-- width not measured, for the same reason.")


def plot_fit_quality(processed_dqdv, detected_peaks, fit_results,
                      user_parameters, cycles_to_show=None,
                      save_location=None, file_format=None):
    """
    Show fitted curves overlaid on data with individual components
    and Rietveld-style residual panels.

    Grey = data, coloured line = composite fit, shaded fills = individual
    named components. The dashed black baseline is drawn only where one
    exists: from 1.9.0.56 the model has no free background, and drawing a
    flat zero line labelled "baseline" would tell the reader something
    false about what the model contains.

    Automatically includes the worst converged cycle and one failed
    cycle (if any) for honest quality assessment.
    """
    for name in fit_results:
        data = processed_dqdv[name]['data']
        det = detected_peaks[name]
        fits = fit_results[name]['fits']
        fs = fit_results[name]['fit_summary']
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        key_cycles = params.get('key_cycles', DEFAULT_KEY_CYCLES)
        ref_cycle = det['reference_cycle']

        if cycles_to_show is None:
            available = sorted(set(c for c, s in data.keys()))
            cycles_to_show_actual = sorted(set(
                [c for c in key_cycles if c in available][:5]
            ))
            if (ref_cycle not in cycles_to_show_actual and
                ref_cycle in available):
                cycles_to_show_actual = sorted(set(
                    cycles_to_show_actual + [ref_cycle]
                ))

            # Add worst-fitting converged cycle for honest QC
            converged = fs[fs['success'] == True]
            if not converged.empty:
                worst = converged.loc[converged['r_squared'].idxmin()]
                worst_cycle = int(worst['cycle'])
                if worst_cycle not in cycles_to_show_actual:
                    cycles_to_show_actual = sorted(set(
                        cycles_to_show_actual + [worst_cycle]
                    ))
                    print(f"  Including cycle {worst_cycle} "
                          f"(lowest R² = {worst['r_squared']:.4f}, "
                          f"{worst['step']}) for QC")
                else:
                    # Worst already shown — find worst unseen
                    unseen = converged[
                        ~converged['cycle'].isin(cycles_to_show_actual)
                    ]
                    if not unseen.empty:
                        worst2 = unseen.loc[
                            unseen['r_squared'].idxmin()]
                        w2_cycle = int(worst2['cycle'])
                        cycles_to_show_actual = sorted(set(
                            cycles_to_show_actual + [w2_cycle]
                        ))
                        print(f"  Including cycle {w2_cycle} "
                              f"(lowest unseen R² = "
                              f"{worst2['r_squared']:.4f}, "
                              f"{worst2['step']}) for QC")

            # Also include one failed cycle if any exist
            failed = fs[fs['success'] == False]
            if not failed.empty:
                fail_cycle = int(failed['cycle'].iloc[0])
                fail_step = failed['step'].iloc[0]
                if fail_cycle not in cycles_to_show_actual:
                    cycles_to_show_actual = sorted(set(
                        cycles_to_show_actual + [fail_cycle]
                    ))
                    print(f"  Including cycle {fail_cycle} "
                          f"(FAILED, {fail_step}) for QC")
        else:
            cycles_to_show_actual = cycles_to_show

        n_show = len(cycles_to_show_actual)

        _drew_baseline = False
        fig = plt.figure(figsize=(13, 5.0 * n_show))

        outer_gs = gridspec.GridSpec(
            n_show, 1, figure=fig,
            hspace=0.35, top=0.95, bottom=0.04
        )

        for row, cycle in enumerate(cycles_to_show_actual):
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, 2, subplot_spec=outer_gs[row],
                height_ratios=[5, 1], hspace=0.05, wspace=0.3
            )

            for col, step in enumerate(['Charge', 'Discharge']):
                ax = fig.add_subplot(inner_gs[0, col])
                ax_res = fig.add_subplot(inner_gs[1, col], sharex=ax)

                if (cycle, step) not in data:
                    ax.set_visible(False)
                    ax_res.set_visible(False)
                    continue

                df = data[(cycle, step)]
                voltage = df['Voltage'].values
                dqdv = df['dQ/dV_processed'].values

                colour = COLOUR_CHARGE if step == 'Charge' else COLOUR_DISCHARGE
                ax.plot(voltage, dqdv, color='grey', linewidth=1.0,
                        alpha=0.6, label='Data')

                fit = fits.get((cycle, step))
                if (fit is not None and
                    fit.get('lmfit_result') is not None):
                    result = fit['lmfit_result']

                    best_fit = result.best_fit
                    if step == 'Discharge':
                        best_fit = -best_fit

                    ax.plot(voltage, best_fit, color=colour,
                            linewidth=1.5, label='Composite fit')

                    # Individual components
                    comps = result.eval_components(x=voltage)
                    for comp_name, comp_vals in comps.items():
                        if step == 'Discharge':
                            comp_vals = -comp_vals

                        if 'bg_' in comp_name:
                            # Only if there IS one. With no free background
                            # the component is identically zero, and a flat
                            # line at zero labelled "Baseline" tells the
                            # reader the model has something it has not.
                            if float(np.nanmax(np.abs(comp_vals))) <= 0.0:
                                continue
                            _drew_baseline = True
                            ax.plot(voltage, comp_vals, 'k--',
                                   linewidth=0.8, alpha=0.5,
                                   label='Baseline')
                        else:
                            ax.fill_between(voltage, comp_vals,
                                           alpha=0.15, color=colour)
                            ax.plot(voltage, comp_vals, color=colour,
                                   linewidth=0.6, alpha=0.5,
                                   linestyle='--')

                    is_ref = (cycle == ref_cycle)
                    suffix = ' (REF)' if is_ref else ''
                    ax.text(
                        0.02, 0.95,
                        f'R² = {fit["r_squared"]:.4f}{suffix}',
                        transform=ax.transAxes, fontsize=9,
                        va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.3',
                                 facecolor='white', alpha=0.8)
                    )

                    # Difference plot
                    residuals = fit['residuals']
                    if residuals is not None:
                        if step == 'Discharge':
                            residuals = -residuals
                        ax_res.plot(voltage, residuals,
                                   color=colour, linewidth=0.6,
                                   alpha=0.7)
                        ax_res.axhline(y=0, color='grey',
                                      linewidth=0.4)
                        ax_res.fill_between(voltage, residuals,
                                           alpha=0.1, color=colour)

                elif fit is not None and not fit.get('success', False):
                    ax.text(
                        0.5, 0.5, 'FIT FAILED',
                        transform=ax.transAxes, fontsize=14,
                        ha='center', va='center',
                        color='red', fontweight='bold', alpha=0.5
                    )

                # THE PANEL SHOWS THE FEATURE, NOT THE WHOLE WINDOW.
                # No x-limit was ever set here, so both panels spanned the
                # full analysis range. On a sharp profile that is unreadable:
                # LTO's single peak is about 40 mV wide inside a 1.3 V window
                # — roughly 3% of the axis — and the discharge panel renders
                # as a vertical line. A fit was misread by 4x because of it.
                #
                # The limits are taken from where the CURVE actually is: the
                # span over which |dQ/dV| stays above PEAK_FIT_XLIM_FLOOR of
                # its own maximum, padded, with a minimum width so a very
                # narrow peak still gets context around it rather than
                # filling the panel edge to edge. `sharex` carries it to the
                # residual panel. Nothing is cropped that carries signal, and
                # a curve that genuinely fills the window is unchanged.
                try:
                    _amp = np.abs(np.asarray(dqdv, float))
                    _mx = float(np.nanmax(_amp)) if _amp.size else 0.0
                    if _mx > 0:
                        _keep = np.asarray(voltage, float)[
                            _amp >= PEAK_FIT_XLIM_FLOOR * _mx]
                        if _keep.size:
                            _lo, _hi = float(_keep.min()), float(_keep.max())
                            _pad = max(PEAK_FIT_XLIM_PAD * (_hi - _lo),
                                       0.5 * (PEAK_FIT_XLIM_MIN_V - (_hi - _lo)),
                                       0.0)
                            _v0 = float(np.nanmin(voltage))
                            _v1 = float(np.nanmax(voltage))
                            _lo, _hi = max(_v0, _lo - _pad), min(_v1, _hi + _pad)
                            if _hi > _lo:
                                ax.set_xlim(_lo, _hi)
                except Exception:
                    pass

                # Fit panel formatting
                ax.set_title(f'Cycle {cycle} — {step}', fontsize=11)
                ax.set_ylabel('dQ/dV / mAh V$^{-1}$ g$^{-1}$',
                             fontsize=11)
                ax.tick_params(axis='both', labelsize=10, direction='in',
                              top=True, right=True)
                ax.axhline(y=0, color='grey', linewidth=0.4)
                ax.legend(fontsize=8, loc='best')
                plt.setp(ax.get_xticklabels(), visible=False)

                # Residual panel formatting
                ax_res.set_xlabel('Voltage / V', fontsize=11)
                ax_res.set_ylabel('Resid.', fontsize=9)
                ax_res.tick_params(axis='both', labelsize=9,
                                  direction='in', top=True, right=True)
                for sp in ax_res.spines.values():
                    sp.set_linewidth(0.6)

        fig.suptitle(f'{composition} — multi-peak fits', fontsize=14,
                     fontweight=FIGURE_TITLE_WEIGHT)

        if save_location:
            fpath = os.path.join(
                save_location, f'{name}_peak_fits.{image_format(params, file_format)}'
            )
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            saved(fpath)

        _caption(
            f"Figure X. Multi-peak pseudo-Voigt fits to the differential "
            f"capacity of {composition}, for "
            f"{len(cycles_to_show_actual)} representative cycles. Grey points "
            f"are the processed data, the solid line the summed model, "
            f"filled curves the individual named components"
            + (" and the dashed line the polynomial baseline"
               if _drew_baseline else
               "; the model carries no free background, so every part of "
               "the curve shown is charge attributed to a named component "
               "or reported as unattributed")
            + f". Each panel is annotated with its "
            f"coefficient of determination; the lower trace of each pair is "
            f"the residual, on the same voltage axis. Fitted parameters for "
            f"every cycle are tabulated in "
            f"{name}_fitted_parameters.csv.")

        plt.show()
        plt.close(fig)



# --- is a peak AREA a capacity on this dataset? ----------------------------
# `analyse.flag_low_fidelity` answers that per peak and writes the answer into
# `tracking.summary` as `area_retention_trustworthy` / `area_withheld_reason`.
# Until 1.9.0.36 no figure read it, so on a run whose START_HERE says in bold
# "Peak areas here are not capacities" the peak-trend figure was captioned
# "peak area, which is proportional to the capacity stored by that process" —
# and that caption is the line that gets pasted into a manuscript.
def _areas_are_capacities(summary, step=None):
    """False if any tracked peak (of this step) had its area caveated."""
    if not summary:
        return True, ""
    _reason = ""
    _ok = True
    for _k, _s in summary.items():
        if step is not None and _s.get("step") != step:
            continue
        if _s.get("area_retention_trustworthy") is False:
            _ok = False
            _reason = _reason or str(_s.get("area_withheld_reason") or "")
    return _ok, _reason


def plot_tracked_trends(tracked, user_parameters,
                         save_location=None, file_format=None):
    """
    Three-panel trend plots: centre voltage, peak area, FWHM vs cycle.

    Each tracked peak gets its own colour. Shoulder peaks use square
    markers; ghost peaks are omitted entirely from trend plots as their
    area values are meaningless.
    """
    for name, tr in tracked.items():
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        tdf = tr['tracked_df']
        ghost_peaks = tr.get('ghost_peaks', set())
        _summary = tr.get('summary') or {}

        for step in ['Charge', 'Discharge']:
            step_df = tdf[
                (tdf['step'] == step) & (tdf['status'] == 'tracked')
            ]
            if step_df.empty:
                continue

            # Exclude ghost peaks from trend plots
            plot_ids = sorted([
                pid for pid in step_df['tracked_peak_id'].unique()
                if (step, int(pid)) not in ghost_peaks
            ])
            if not plot_ids:
                continue
            colours = categorical_colours(len(plot_ids))

            fig, (ax1, ax2, ax3) = plt.subplots(
                3, 1, figsize=(10, 10), sharex=True)
            fig.suptitle(f'{composition} — {step} peak evolution',
                        fontsize=14, fontweight=FIGURE_TITLE_WEIGHT)
            _n_band_clipped = 0
            _split_seen = False

            for i, pid in enumerate(plot_ids):
                pk = step_df[
                    step_df['tracked_peak_id'] == pid
                ].sort_values('cycle')

                ref_v = pk['reference_voltage'].iloc[0]
                is_sh = pk['is_shoulder'].iloc[0]
                is_tr = pk.get('is_truncated', pd.Series([False])).iloc[0]
                label = f'Peak {pid} ({ref_v:.3f} V)'
                if is_sh:
                    label += ' [sh]'
                if is_tr:
                    label += ' [trunc]'
                clr    = colours[i]
                marker = 's' if is_sh else 'o'

                if 'reliable' in pk.columns:
                    pk_good = pk[pk['reliable']]
                    pk_bad  = pk[~pk['reliable']]
                else:
                    pk_good = pk
                    pk_bad  = pk.iloc[0:0]

                for ax, col, ylabel in [
                    (ax1, 'centre',         'Centre voltage / V'),
                    (ax2, 'amplitude_area', 'Peak area (∝ capacity)'),
                    (ax3, 'fwhm',           'FWHM / mV'),
                ]:
                    scale = 1000 if col == 'fwhm' else 1
                    ax.plot(pk['cycle'], pk[col] * scale,
                            color=clr, linewidth=0.5, alpha=0.3)
                    ax.plot(pk_good['cycle'], pk_good[col] * scale,
                            color=clr, marker=marker, markersize=4,
                            linewidth=0, label=label)
                    if not pk_bad.empty:
                        ax.plot(pk_bad['cycle'], pk_bad[col] * scale,
                                color=clr, marker=marker, markersize=4,
                                linewidth=0, fillstyle='none', alpha=0.35)
                    # A +/-1 sigma envelope on all three panels. Until
                    # 1.9.0.4 only the centre carried one, although the fit
                    # returns a standard error for every parameter and
                    # `track_peaks` carries them all into `tracked_df`: the
                    # area panel simply was not reading `amplitude_stderr`,
                    # and FWHM is 2 x sigma exactly (`fitting._FWHM_FACTOR`),
                    # so its error is 2 x sigma_stderr. A trend drawn without
                    # them invites the reader to believe a wobble that the
                    # fit itself calls noise.
                    _err_col, _err_scale, _clip = {
                        'centre':         ('centre_stderr',    1.0,
                                           TREND_BAND_CLIP_V),
                        'amplitude_area': ('amplitude_stderr', 1.0, None),
                        'fwhm':           ('sigma_stderr',     2.0,
                                           TREND_BAND_CLIP_V),
                    }[col]
                    if (_err_col in pk_good.columns and not pk_good.empty
                            and pk_good[_err_col].notna().any()
                            and len(pk_good) >= 2):
                        se = pk_good[_err_col].fillna(0) * _err_scale
                        if col == 'fwhm' and 'sigma_r_stderr' in pk_good.columns \
                                and pk_good['sigma_r_stderr'].notna().any():
                            _split_seen = True
                        if col == 'fwhm' and 'sigma_r_stderr' in pk_good.columns:
                            # FWHM IS NO LONGER 2 x SIGMA. With a split
                            # lineshape it is sigma + sigma_r, so its error is
                            # the two errors ADDED — not in quadrature: the
                            # widths are tied by one shared ratio
                            # (sigma_r = k * sigma), so they move together and
                            # a quadrature sum would understate a correlated
                            # pair. On LTO, where k is about 9, the old
                            # 2 x sigma_stderr drew a band five times too
                            # narrow.
                            _sr = pk_good['sigma_r_stderr']
                            if _sr.notna().any():
                                se = (pk_good['sigma_stderr'].fillna(0)
                                      + _sr.fillna(0))
                        if _clip is not None:
                            # Clipped for display: one unconverged cycle
                            # otherwise draws a triangle across the panel.
                            # COUNTED, because `_BAND_NOTE` in the caption
                            # tells the reader to look for a band wider than
                            # the trend it describes — and after a 20 mV cap
                            # no such band can be drawn on these two panels,
                            # so the instruction was unfollowable. NMC cell B
                            # has a `centre_stderr` of 14.7 V on a row flagged
                            # reliable; P3 cell C one of 153 V. Both rendered
                            # as a tidy +/-20 mV.
                            _n_band_clipped += int((se > _clip).sum())
                            se = se.clip(upper=_clip)
                        else:
                            # No natural ceiling on an area, so clip to the
                            # peak's own median area instead of a constant.
                            _med = float(pk_good['amplitude_area'].abs()
                                         .median())
                            if np.isfinite(_med) and _med > 0:
                                se = se.clip(upper=_med)
                        _lo = (pk_good[col] - se) * scale
                        if col in ('fwhm', 'amplitude_area'):
                            # A NEGATIVE WIDTH IS NOT A LOWER BOUND. On LTO
                            # cell C cycle 10 `sigma_stderr` (3.603e-3)
                            # exceeded `sigma` (3.000e-3) — a component
                            # sitting on its floor, whose error is therefore
                            # not a measurement either — and the band was
                            # drawn through zero into negative FWHM. Clipped
                            # here so the figure cannot state an impossibility;
                            # the flag that says WHY belongs in the table.
                            _lo = _lo.clip(lower=0.0)
                        ax.fill_between(
                            pk_good['cycle'], _lo,
                            (pk_good[col] + se) * scale,
                            alpha=0.15, color=clr)

            _cap_ok, _cap_why = _areas_are_capacities(_summary, step)
            for ax, ylabel in [
                (ax1, 'Centre voltage / V'),
                (ax2, 'Peak area (∝ capacity)' if _cap_ok
                 else 'Peak area / mAh V⁻¹ g⁻¹ (NOT a capacity)'),
                (ax3, 'FWHM / mV'),
            ]:
                ax.set_ylabel(ylabel, fontsize=12)
                ax.tick_params(axis='both', labelsize=10, direction='in',
                              top=True, right=True)
                ax.legend(fontsize=8, loc='best', framealpha=0.7)
                for sp in ax.spines.values():
                    sp.set_linewidth(0.8)
            ax3.set_xlabel('Cycle number', fontsize=12)
            _force_integer_cycles(ax3)

            plt.tight_layout()
            if save_location:
                fpath = os.path.join(save_location,
                    f'{name}_peak_trends_{step}.{image_format(params, file_format)}')
                fig.savefig(fpath, dpi=300, bbox_inches='tight')
                saved(fpath)
            _caption(
                f"Figure X. Evolution of the fitted {step.lower()} peaks of "
                f"{composition} with cycle number: (a) peak centre, (b) peak "
                + ("area, which is proportional to the capacity stored by "
                   "that process, " if _cap_ok else
                   "area — which on this dataset is NOT proportional to the "
                   "capacity stored by that process"
                   + (f" ({_cap_why})" if _cap_why else "")
                   + ", so the panel shows how the fitted area changes and "
                     "not how much charge the process carried, ")
                + f"and (c) full width at half maximum. Filled "
                f"markers are cycles in which the fit met the reliability "
                f"criteria; open markers are cycles in which it did not, and "
                f"are shown for completeness but excluded from the trends. "
                + f"{_BAND_NOTE} "
                + ("The width band is the standard errors on sigma and "
                   "sigma_r ADDED, since this model's FWHM is their sum and "
                   "the two are tied by one ratio, so a quadrature sum would "
                   "understate a correlated pair."
                   if _split_seen else
                   "The width band is twice the standard error on sigma, "
                   "since FWHM = 2\u03c3 exactly for the symmetric form of "
                   "this model.")
                + (f" The centre and width bands are capped at \u00b1"
                   f"{1000 * TREND_BAND_CLIP_V:.0f} mV for legibility; "
                   f"{_n_band_clipped} point(s) have a larger standard error "
                   f"than that and are drawn narrower than they are. The "
                   f"unclipped values are in the tracked-peaks table."
                   if _n_band_clipped else ""))
            plt.show()
            plt.close(fig)


def _build_ref_voltage_lookup(tracked_peaks, name):
    """
    Build a fixed peak_id → reference_voltage lookup for labelling.
    Uses the reference_peaks source (which always has ref_V) rather
    than per-cycle tracked data (which may not include all peaks in
    every cycle, causing nan labels).
    """
    lookup = {}
    ref_peaks = tracked_peaks[name]['reference_peaks']
    for step in ['Charge', 'Discharge']:
        rp = ref_peaks.get(step, pd.DataFrame())
        if rp is not None and not rp.empty:
            for _, pk in rp.iterrows():
                key = (step, int(pk['peak_id']))
                lookup[key] = pk['voltage']
    return lookup


def plot_delta_v(delta_v_results, user_parameters,
                  save_location=None, file_format=None):
    """Plot ΔV evolution with error bands and linear trend."""
    for name, dvr in delta_v_results.items():
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        dv_df = dvr['delta_v_df']
        dv_changes = dvr.get('delta_v_changes', {})

        if dv_df.empty:
            continue

        pair_labels = sorted(dv_df['pair_label'].unique())
        colours = categorical_colours(len(pair_labels))

        fig, ax = plt.subplots(figsize=(figure_width_inches, figure_height_inches))

        _n_clipped = 0
        for i, label in enumerate(pair_labels):
            pd_pair = dv_df[dv_df['pair_label'] == label].sort_values('cycle')
            rate_str = (f' (net {dv_changes[label][0]:+.0f} mV)'
                       if label in dv_changes else '')

            ax.plot(pd_pair['cycle'], pd_pair['delta_v_mV'],
                   color=colours[i], marker='o', markersize=5,
                   linewidth=1.2, label=f'{label}{rate_str}')

            if pd_pair['delta_v_stderr_mV'].notna().any():
                se = pd_pair['delta_v_stderr_mV'].fillna(0)
                # CLIPPED FOR DISPLAY, AND THE FIGURE SAYS SO. The cap keeps
                # one absurd standard error from flattening the whole panel —
                # P3 cell C carries a `delta_v_stderr_mV` of 498245 — but the
                # caption below asserts the band IS the fit's own standard
                # error, so on cells A and B (239 mV and 196 mV) a reader was
                # shown a tight +/-20 mV band under a sentence promising the
                # real one. Clipping is the right choice for the drawing;
                # silence about it is not.
                _n_clipped += int((se > DELTA_V_BAND_CLIP_MV).sum())
                se = se.clip(upper=DELTA_V_BAND_CLIP_MV)
                ax.fill_between(pd_pair['cycle'],
                    pd_pair['delta_v_mV'] - se,
                    pd_pair['delta_v_mV'] + se,
                    alpha=0.15, color=colours[i])

        ax.set_xlabel('Cycle number', fontsize=13)
        _force_integer_cycles(ax)
        ax.set_ylabel('ΔV (charge − discharge) / mV', fontsize=13)
        ax.set_title(f'{composition} — polarisation evolution (ΔV)',
                    fontsize=14)
        if _n_clipped:
            ax.text(0.99, 0.02,
                    f'shaded band capped at ±{DELTA_V_BAND_CLIP_MV:.0f} mV; '
                    f'{_n_clipped} point(s) have a larger standard error',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=8, style='italic', alpha=0.8)
        ax.tick_params(axis='both', labelsize=11, direction='in',
                      top=True, right=True)
        ax.legend(fontsize=9, framealpha=0.7)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)

        plt.tight_layout()
        if save_location:
            fpath = os.path.join(save_location,
                                f'{name}_delta_v.{image_format(params, file_format)}')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            saved(fpath)
        _caption(
            f"Figure X. Polarisation of {composition}, expressed as the "
            f"separation \u0394V between the charge and discharge peak "
            f"centres of each redox couple, against cycle number. The "
            f"legend gives each couple's net change over the cycles in which "
            f"it was tracked continuously. A widening |\u0394V| indicates "
            f"rising internal "
            f"resistance. {_BAND_NOTE} Here the band is the two centre "
            f"errors added in quadrature, since \u0394V is their difference."
            + (f" The shaded band is capped at \u00b1"
               f"{DELTA_V_BAND_CLIP_MV:.0f} mV for legibility; "
               f"{_n_clipped} point(s) have a larger standard error than "
               f"that and are drawn narrower than they are. The unclipped "
               f"values are in the delta-V table."
               if _n_clipped else ""))
        plt.show()
        plt.close(fig)


def plot_capacity_attribution(attribution, tracked_peaks,
                               user_parameters,
                               save_location=None, file_format=None):
    """
    Two-panel capacity attribution: absolute areas + stacked fractions.
    Uses fixed reference voltage lookup for consistent labels.
    """
    for name, step_dfs in attribution.items():
        params = user_parameters.get(name, {})
        composition = _get_display_name(name, params, user_parameters)
        ref_v_lookup = _build_ref_voltage_lookup(tracked_peaks, name)

        for step, adf in step_dfs.items():
            if adf.empty:
                continue

            frac_cols = sorted([c for c in adf.columns
                                if c.startswith('peak')
                                and c.endswith('_fraction')])
            if not frac_cols:
                continue

            # Build labels from reference lookup
            peak_info = []
            for col in frac_cols:
                pid_str = col.replace('_fraction', '')
                pid = int(pid_str.replace('peak', ''))
                ref_v = ref_v_lookup.get((step, pid), np.nan)
                ref_str = f'{ref_v:.3f}' if pd.notna(ref_v) else '?'
                peak_info.append((pid, pid_str, ref_str, col))

            colours = categorical_colours(len(peak_info))

            # Two vertically stacked panels: journal width, taller
            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(figure_width_inches, figure_height_inches * 1.5),
                sharex=True)
            fig.suptitle(
                f'{composition} — {step} capacity attribution',
                fontsize=14, fontweight=FIGURE_TITLE_WEIGHT)

            # Panel 1: Absolute areas
            for i, (pid, pid_str, ref_str, frac_col) in enumerate(peak_info):
                area_col = f'{pid_str}_area'
                if area_col in adf.columns:
                    valid = adf[adf[area_col].notna()]
                    if not valid.empty:
                        ax1.plot(valid['cycle'], valid[area_col],
                                color=colours[i], marker='o', markersize=4,
                                linewidth=1,
                                label=f'Peak {pid} ({ref_str} V)')

            # THE SAME GATE THE TREND FIGURE APPLIES. `plot_tracked_trends`
            # asks `_areas_are_capacities` before it writes this label, and
            # this function — which is titled "capacity attribution" and is the
            # more likely of the two to be quoted as one — did not, although
            # it is handed the same `tracked_peaks` and the verdict is on it.
            # NNM cell B has all ten peaks marked
            # `area_retention_trustworthy=False` with the reason "the dQ/dV
            # curve accounts for only 5% of this step's delivered capacity, so
            # a peak area is not a capacity", and its two figures in one folder
            # read "NOT a capacity" and "(∝ capacity)".
            _cap_ok, _cap_why = _areas_are_capacities(
                (tracked_peaks.get(name) or {}).get('summary') or {}, step)
            ax1.set_ylabel('Peak area (∝ capacity)' if _cap_ok
                           else 'Peak area / mAh V⁻¹ g⁻¹ (NOT a capacity)',
                           fontsize=12)
            ax1.legend(fontsize=8, loc='best', framealpha=0.7)

            # Panel 2: Stacked fractions
            # WITHHELD IS NOT ZERO. `capacity_attribution` writes NaN for a
            # cycle whose peak/background split is not determined by the
            # data — deliberately, "the SHARE is blank because it is not a
            # measurement". `fillna(0)` turned each of those into a bar of
            # height zero, drawn identically to a real collapse, under a
            # caption saying withheld cycles are not plotted. On a dataset
            # where withholding fires on every cycle the figure was a flat
            # row of zeros captioned "the stack is free to fall below 100%
            # as processes are lost".
            _plotted = adf.copy()
            _kept = np.zeros(len(_plotted), dtype=bool)
            for _, _, _, _fc in peak_info:
                _kept |= _plotted[_fc].notna().values
            _plotted = _plotted[_kept]
            _n_withheld = int(len(adf) - len(_plotted))
            # ...AND WITHHELD PER PEAK IS NOT ZERO EITHER. The row filter above
            # keeps a cycle when ANY peak has a fraction, and the `fillna(0)`
            # below then flattens every OTHER peak's withheld cell in that row
            # to a bar of height zero — a real collapse to the eye. On NNM cell
            # A's discharge, peak 1 is NaN on 122 of 128 cycles and peak 5 on
            # 120, and because no row was entirely NaN the withheld count came
            # out as 0 and the explanatory box never appeared: two processes
            # read as contributing exactly nothing from cycle 4 onward. The
            # geometry of a stacked bar cannot show an absent segment, so the
            # count is said instead — in the legend, per peak, where the
            # reader is already looking to identify the colour.
            _cells = int(len(_plotted) * len(peak_info))
            _cells_withheld = 0
            bottoms = np.zeros(len(_plotted))
            for i, (pid, pid_str, ref_str, frac_col) in enumerate(peak_info):
                _miss = int(_plotted[frac_col].isna().sum())
                _cells_withheld += _miss
                fracs = _plotted[frac_col].fillna(0).values * 100
                ax2.bar(_plotted['cycle'], fracs, bottom=bottoms,
                       color=colours[i], alpha=0.8, width=0.8,
                       label=(f'Peak {pid} ({ref_str} V)'
                              + (f' — withheld on {_miss}/{len(_plotted)}'
                                 if _miss else '')))
                bottoms += fracs
            if _n_withheld or _cells_withheld:
                _msg = []
                if _n_withheld:
                    _msg.append(f"{_n_withheld} of {len(adf)} cycles withheld")
                if _cells_withheld:
                    _msg.append(f"{_cells_withheld} of {_cells} peak-cycle "
                                f"shares withheld and drawn as zero")
                ax2.text(0.5, 0.5,
                         ("\n".join(_msg) + " —\nthe peak/background split is "
                          "not\ndetermined by the data"
                          if not len(_plotted) else "; ".join(_msg)),
                         transform=ax2.transAxes, ha='center',
                         va='center' if not len(_plotted) else 'top',
                         fontsize=10 if not len(_plotted) else 8,
                         color='#666666',
                         bbox=dict(boxstyle='round,pad=0.4',
                                   facecolor='white', alpha=0.85,
                                   edgecolor='#999999'))

            ax2.set_ylabel('Capacity fraction vs reference total / %',
                           fontsize=12)
            ax2.set_xlabel('Cycle number', fontsize=12)
            ax2.axhline(100, color='0.4', linewidth=0.8, linestyle='--')
            _stack_top = float(np.nanmax(bottoms)) if len(bottoms) else 100.0
            ax2.set_ylim(0, max(110, _stack_top * 1.05))
            ax2.legend(fontsize=8, loc='upper right', framealpha=0.7)

            for ax in [ax1, ax2]:
                ax.tick_params(axis='both', labelsize=10, direction='in',
                              top=True, right=True)
                for sp in ax.spines.values():
                    sp.set_linewidth(0.8)

            plt.tight_layout()
            if save_location:
                fpath = os.path.join(save_location,
                    f'{name}_capacity_attribution_{step}.{image_format(params, file_format)}')
                fig.savefig(fpath, dpi=300, bbox_inches='tight')
                saved(fpath)
            _caption(
                f"Figure X. Capacity attribution for the {step.lower()} of "
                f"{composition}: (a) the fitted area of each tracked peak "
                f"against cycle number, and (b) the same areas as a "
                f"percentage of the summed peak area at the reference cycle, "
                f"stacked. A fixed reference denominator is used throughout, "
                f"so the stack is free to fall below 100% as processes are "
                f"lost — a per-cycle denominator would redistribute lost "
                f"capacity onto the survivors. Cycles for which the "
                f"attribution is not determined by the data are withheld "
                f"rather than plotted; see the closure interval recorded in "
                f"{name}_fitted_parameters.csv."
                + (f" A withheld share for an individual peak in a cycle that "
                   f"is otherwise plotted cannot be left out of a stacked bar, "
                   f"so it is drawn at zero height; {_cells_withheld} of "
                   f"{_cells} peak-cycle shares here are withheld rather than "
                   f"measured, and the legend gives the count per peak."
                   if _cells_withheld else "")
                + ("" if _cap_ok else
                   f" The areas in panel (a) are NOT capacities on this "
                   f"dataset" + (f" — {_cap_why}" if _cap_why else "") + "."))
            plt.show()
            plt.close(fig)


# =============================================================================
# ADAPTERS — 1.9.0 objects into the shapes the ported code expects
# =============================================================================
# This is the only new code in this module. Everything above is 1.8.7's.

def as_processed_dqdv(dataset, signals, params=None, *, window=None,
                      profile=None):
    """
    `{name: {'data': {(cycle, step): DataFrame}, 'voltage_range': (lo, hi),
              'parameters_used': {...}}}` — Module 1's shape.

    The frames carry exactly the three columns the plotting code reads, in
    voltage order, which is the order `signal.half_cycle_report` already
    returns them in.
    """
    data = {}
    for (cycle, step), sig in signals.items():
        if sig.voltage.size == 0:
            continue
        data[(int(cycle), str(step))] = pd.DataFrame({
            "Voltage": sig.voltage,
            "dQ/dV_raw": sig.dqdv_raw,
            "dQ/dV_processed": sig.dqdv,
        })
    if window is None:
        allv = [d["Voltage"] for d in data.values()]
        window = ((float(min(v.min() for v in allv)),
                   float(max(v.max() for v in allv))) if allv
                  else (np.nan, np.nan))
    used = dict(params or {})
    if profile is not None:
        used.setdefault("profile_class", profile.get("class")
                        if isinstance(profile, dict) else profile)
    return {dataset.name: {"data": data, "voltage_range": tuple(window),
                           "parameters_used": used}}


def as_detected_peaks(detection):
    """`{name: {'peaks', 'reference_cycle', 'reference_peaks', ...}}` —
    Module 3's shape."""
    return {detection.name: {
        "peaks": detection.peaks,
        "reference_cycle": detection.reference_cycle,
        "reference_cycle_reason": detection.reference_reason,
        "reference_peaks": detection.reference_peaks,
        "parameters_used": (detection.spec.as_dict() if detection.spec
                            else {}),
    }}


class _FitView:
    """
    What `plot_fit_quality` thinks is an lmfit `ModelResult`.

    It answers `.best_fit` and `.eval_components(x=...)` from
    `fitting.evaluate`, which rebuilds the component curves analytically from
    the stored parameters through lmfit's own model classes. No refit, and no
    `ModelResult` anywhere near a process boundary — see the module docstring.
    """
    __slots__ = ("_result", "_voltage", "_comps", "best_fit")

    def __init__(self, result, voltage):
        self._result = result
        self._voltage = np.asarray(voltage, float)
        self._comps = _ft.evaluate(result, self._voltage)
        self.best_fit = self._comps["total"]

    def eval_components(self, x=None):
        if x is None:
            comps = self._comps
        else:
            x = np.asarray(x, float)
            comps = (self._comps
                     if (x.shape == self._voltage.shape
                         and np.array_equal(x, self._voltage))
                     else _ft.evaluate(self._result, x))
        return {k: v for k, v in comps.items() if k != "total"}


def as_fit_results(fit_results, signals, params_df=None, name=""):
    """
    `{name: {'fits': {(cycle, step): {...}}, 'parameters_df', 'fit_summary'}}`
    — Module 4's shape, with `_FitView` standing in for the ModelResult.
    """
    fits, summary = {}, []
    for fr in fit_results:
        key = fr.get("key")
        if key is None:
            continue
        cycle, step = int(key[0]), str(key[1])
        sig = signals.get((cycle, step))
        view = (_FitView(fr, sig.voltage)
                if (fr.get("success") and sig is not None
                    and sig.voltage.size) else None)
        # The `sig is not None` guard above is undone by reading sig.dqdv
        # here, and the value is discarded three lines down when view is
        # None anyway — so the branch could only ever raise.
        resid = (None if view is None
                 else np.abs(sig.dqdv) - view.best_fit)
        fits[(cycle, step)] = {
            "success": bool(fr.get("success")),
            "parameters": fr.get("components", []),
            "lmfit_result": view,
            "r_squared": fr.get("r_squared", np.nan),
            "reduced_chi_sq": fr.get("redchi", np.nan),
            # lmfit's ModelResult.residual is data - model; matched here so
            # the difference panel keeps its sign.
            "residuals": (resid if view is not None else None),
        }
        summary.append(dict(dataset=name, cycle=cycle, step=step,
                            # `success`, not `fit_success`: plot_fit_quality
                            # selects its worst converged and its failed
                            # half-cycles from this column by that name.
                            success=bool(fr.get("success")),
                            r_squared=fr.get("r_squared", np.nan),
                            reduced_chi_sq=fr.get("redchi", np.nan),
                            n_peaks=len(fr.get("components", [])),
                            seconds=fr.get("seconds", np.nan)))
    return {name: {"fits": fits,
                   "parameters_df": (params_df if params_df is not None
                                     else pd.DataFrame()),
                   "fit_summary": pd.DataFrame(summary)}}


def as_tracked(tracking):
    """`{name: {'tracked_df', 'summary', 'reference_peaks', ...}}` —
    Module 5's shape."""
    return {tracking.name: {
        "tracked_df": tracking.tracked_df,
        "summary": tracking.summary,
        "reference_peaks": tracking.reference_peaks,
        "reference_cycle": tracking.reference_cycle,
        "ghost_peaks": tracking.ghost_peaks,
        "cell_discontinuity": tracking.cell_discontinuity,
    }}


def as_user_parameters(dataset, *, composition=None, cell_id=None,
                       key_cycles=None, colour_palette="viridis_r",
                       battery_chemistry="Li-ion", charge_rate_c=None,
                       **extra):
    """
    The subset of Cell 3's parameters the plotting code actually reads:
    `composition`, `cell_id`, `key_cycles`, `colour_palette`,
    `battery_chemistry`, `charge_rate_c` and `anode_labels_swapped`.

    `anode_labels_swapped` is taken from the Dataset rather than asked for —
    it is a fact about how the file was read, not a preference.
    """
    p = dict(
        composition=composition or dataset.meta.get("composition",
                                                    dataset.name),
        cell_id=cell_id or dataset.meta.get("cell_id", ""),
        key_cycles=list(key_cycles) if key_cycles else list(DEFAULT_KEY_CYCLES),
        colour_palette=colour_palette,
        battery_chemistry=battery_chemistry,
        charge_rate_c=(charge_rate_c if charge_rate_c is not None
                       else dataset.meta.get("charge_rate_c",
                                             "the specified")),
        anode_labels_swapped=bool(getattr(dataset, "labels_swapped", False)),
        electrode_type=getattr(dataset, "electrode_type", "Positive"),
    )
    p.update(extra)
    return {dataset.name: p}