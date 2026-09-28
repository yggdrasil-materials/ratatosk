"""
From fitted components to statements about a cell — and to what must be withheld.

This module does four things, and one of them is new.

    parameters_frame     fit dicts -> the long component table, with the
                         reliability flags 1.8.7 applied in Module 4
    track_peaks          join independent per-cycle fits into peak histories
    delta_v              charge/discharge pairing and polarisation
    capacity_attribution each peak's share, WITHHELD where the closure
                         interval says the share is not determined
    coherence_audit      does each peak move like a redox feature?
    integrity            NEW: what the reversed charge in a half-cycle means

What the mechanism classification is for
----------------------------------------
1.8.x had one number, `f_wrong_sign` — the share of |dQ/dV| carried by points
of the wrong sign — and used it as a gate. It conflated three unrelated things:

    at a redox plateau, at full current, every cycle      a two-phase feature
    at the voltage limit, current tapering                a constant-voltage hold
    away from any feature, at full current, sporadic      charge going elsewhere

Only the third is a fault, and a single threshold on the first number cannot
find it: it transferred neither between cells of one triplicate nor between
chemistries. LTO's whole plateau is "wrong sign" and its coulombic efficiency
is 99%.

`signal.reversals` already inventories each contiguous backwards run — how much
charge, over what voltage band, at what current. This module asks where each
run sat on the curve:

    on a plateau     |dQ/dV| there is >= PLATEAU_DQDV_FRACTION of this
                     half-cycle's own maximum. The potential is pinned by a
                     two-phase equilibrium and wanders within the measurement;
                     the charge is real and intended.
    at the limit     current below the constant-current value, within
                     LIMIT_TOL_MV of the window edge. Protocol, not chemistry.
    parasitic        full current, |dQ/dV| small, away from the limit. Charge
                     passed with no redox process to account for it.

Every term is a property of THIS half-cycle measured against ITSELF — its own
peak |dQ/dV|, its own current, its own capacity. That is the whole reason it
transfers, and it is what `f_wrong_sign` and `f_voltage_backwards` could not do.

Measured on all six reference cells (1,215 P3 and 61 LTO half-cycles):

    CE band        n     mean parasitic charge    max parasitic fraction
    0-20%         22          6.0 mAh/g                 11.1%
    20-40%        75          2.0                       11.4%
    40-60%       118          0.5                       13.1%
    60-80%       178          0.0                        0.0%
    80-95%       550          0.0                        0.4%
    95-105%       35          0.0                        0.0%
    >105%        237          0.0                        0.0%

Twenty-one of 1,215 P3 half-cycles carry any parasitic charge at all, and every
one of them sits in a cycle whose coulombic efficiency is below 60%. Nothing
above 60% carries more than 0.4% of its capacity backwards. On LTO — a flat
two-phase plateau, the case that broke every previous rule — two half-cycles of
61 carry any, and both are the truncated final step of the record.

This is not a correlation dressed as a diagnostic. The quantity IS the charge
that went in and did not come back, localised to where on the curve it went.

Note the ordering that falls out of it: the plateau test reads |dQ/dV|, not a
peak list, so integrity does not depend on detection. The pipeline stays a
straight line, and `detect.choose_reference_cycle` can be handed the callable
`integrity_report` returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# The one definition of "this half-cycle is unfinished", shared with the
# cycling tables so the fit and the headline metrics omit the same thing.

from scipy.optimize import linear_sum_assignment
from scipy.special import erf

from .quality import (PARTITION_DETERMINED_BELOW,
                      UNATTRIBUTED_QUALIFIED_ABOVE)
from .style import (section, entry, verdict, bullet,
                    CAPACITY_COLUMN_ALIASES)

__all__ = [
    "parameters_frame", "Tracking", "track_peaks",
    "DeltaV", "delta_v", "capacity_attribution", "coherence_audit",
    "Integrity", "classify_reversals", "half_cycle_integrity", "cell_integrity_verdict",
    "integrity_report", "last_useful_cycle", "half_cycles_in_progress",
    "TRACKING_TOLERANCE_MV", "TRACK_AGAINST_PREVIOUS_CYCLE",
    "INTEGRAL_FIDELITY_CEILING",
    "PAIR_MAX_SEPARATION_MV",
    "PLATEAU_DQDV_FRACTION", "ATTRIBUTION_WITHHOLD_ABOVE",
]


# =============================================================================
# DEFAULTS — 1.8.7's values, unchanged, except where marked NEW
# =============================================================================

# --- Module 5 -----------------------------------------------------------
TRACKING_TOLERANCE_MV = 80
# How many cycles at each end a drift measurement medians over. See the
# comment where it is used: two single cycles carry sqrt(2) times the
# per-cycle centre scatter, and on a 50-cycle synthetic series that turned
# 90 mV of true drift into 79 mV with no bias anywhere in the centres.
DRIFT_ENDPOINT_CYCLES = 3
# --- the tracker's hard limit -------------------------------------------
# `_assign_components` matches every cycle against the REFERENCE CYCLE's peak
# list within +/- TRACKING_TOLERANCE_MV. A peak that drifts further than that
# over the life of the cell cannot be matched to itself any more — and it does
# not simply stop being tracked, it starts being matched to whichever
# component is now nearest, which on a crowded curve is a DIFFERENT PEAK.
#
# Measured on the synthetic ground truth. At 2 mV/cycle:
#
#     50 cycles  (~100 mV total)   centre rms  4-14 mV    tracking rate 94-100%
#    100 cycles  (~200 mV total)   centre rms 53-141 mV   tracking rate 78-99%
#                                  drift reported as -53 mV against +172 true
#
# The second row is not degradation, it is a different peak. 200 mV of drift
# over 100 cycles is unremarkable for a degrading cathode, so this is a real
# limit and not a corner case, and it is flagged rather than hidden: a peak
# whose shift ever exceeds the tolerance carries `drift_exceeds_tolerance` and
# the summary says the assignment beyond that point is not trustworthy.
#
# 1.9.0.23 DOES THIS. The expectation is now carried forward from the last
# cycle each peak was matched in, walking outward from the reference cycle in
# both directions, so `TRACKING_TOLERANCE_MV` is a PER-CYCLE step limit and
# the lifetime drift is unbounded. Measured on the same harness, 50 cycles at
# 2 mV/cycle:
#
#                      fixed reference          drifting reference
#     centre rms        4-104 mV                 see the test output
#     tracking rate     92-100%
#
# See `_assign_components`. `TRACK_AGAINST_PREVIOUS_CYCLE = False` restores
# the fixed reference, which is how the two are compared.
TRACK_AGAINST_PREVIOUS_CYCLE = True
FLAG_DRIFT_BEYOND_TOLERANCE = True
GHOST_AREA_THRESHOLD = 1e-6
DISCONTINUITY_DROPOUT_RUN = 3
DISCONTINUITY_STEP_FLOOR_MV = 80.0
DISCONTINUITY_STEP_K = 4.0
DISCONTINUITY_TERMINAL_RUN = 10
DISCONTINUITY_ESTABLISH_MIN = 15
_AREA_COLLAPSE_FRAC = 0.05

# --- Module 4's reliability guards --------------------------------------
# "Pinned at its width bound" — ONE definition, used by both the parameter
# filter and the tracking filter. They previously used 0.05 and 0.10 for the
# same idea, so a component could be reliable in one table and not the other.
SIGMA_BOUND_PROXIMITY = 0.05
SIGMA_BOUND_PROXIMITY_COMPLEMENT = 1.0 - SIGMA_BOUND_PROXIMITY
# The fallback width bound, used only when a fit result does not carry the
# bound it was actually run under (an old result, or a hand-built FitSpec).
# The live value comes from `fitting.PROFILE_SPECS` by way of
# `sigma_max_fitted`; see `_flag_reliable_peaks`.
FIT_SIGMA_MAX_MV = 200
FIT_WIDTH_FLOOR_FACTOR = 0.0     # flag only; 0 = do not enforce
FIT_MIN_AREA_FRACTION = 0.0      # 0 = disabled

# A COMPONENT THAT CARRIES NO CHARGE IS NOT A PEAK — measured against the
# largest component in ITS OWN half-cycle, not against a dataset median.
#
# `FIT_MIN_AREA_FRACTION` above compares each area to the median area over
# every reliable component in the dataset, and that test defeats itself: once
# enough components are empty the median is empty too, and nothing is flagged.
# Measured on the LTO triplicate after the detection floor came down to 5 mV,
# 54 of cell B's 78 components carried less than 1% of the charge in their own
# half-cycle and 27 of those were marked `reliable`. They are the price of a
# detector that can now see a doublet: it also seeds the odd bump in the flat
# region, and the fit correctly gives those components no area — 8.9e-13 of
# it, in one case, beside a real peak of 130.
#
# So the test is per half-cycle and scale-free. 1% is not a tuning knob: a
# component at a thousandth of its neighbour is the optimiser saying the
# feature is not there, and there is no dataset in hand where a real feature
# sits below 1% of the largest in its own half-cycle.
EMPTY_COMPONENT_FRACTION = 0.01

# HOW MANY SAMPLING INTERVALS A WIDTH NEEDS BEFORE IT IS A MEASUREMENT.
# Four is the smallest number that gives a half-maximum crossing on each
# flank with a point between: below it the FWHM is being read off three
# points or fewer, which fixes a width the way two points fix a curve.
# Flagged, never enforced — the AREA of such a component is still the best
# estimate available, and it is only the WIDTH that is unsupported.
WIDTH_MIN_SAMPLES = 4.0

# --- Is the AREA a measurement? -----------------------------------------
# The reporting audit added a gate on `converged`, and it was the wrong
# quantity: convergence is a STATUS — did the optimiser stop for its own
# reasons or run out of budget — and status is not quality. Measured on the
# NNM triplicate it pointed the wrong way outright, demoting 405 of 405 and
# 462 of 462 components whose fits had the HIGHER median R2 (0.984 / 0.987
# against 0.975 / 0.972), because "did not converge" there meant "was still
# descending when the budget ran out". With `fitting.FIT_TOLERANCE` fixed the
# budget is rarely exhausted, and the flag is now merely uninformative rather
# than inverted — on 325 components across the three chemistries, NONE of the
# 35 non-converged had a poorly-determined area while 42 of the 290 converged
# ones did. So the gate is gone and the flag is reported instead.
#
# The quantity that IS about the measurement is the one item 16 exposed on the
# FWHM panel: a parameter whose standard error is comparable to its own value
# is not a measurement of anything. Measured, `amplitude_stderr / |area|`:
#
#   LTO cellC    median 0.023   nothing above 0.20
#   NNM cellC    median 0.073   16% above 0.35
#   NMC111 cellA median 0.498   and 73 of 87 components have NO stderr at all
#
# which is the numerical statement of what the closure interval says about
# each chemistry, arrived at independently of it. A missing standard error is
# the starkest case: the covariance could not be estimated, so the area has no
# uncertainty to quote.
#
# FLAGGED, NOT ENFORCED, deliberately — the same decision `too_narrow` records
# above. Enforcing it would demote roughly a third of NNM and most of NMC, and
# that is a change to what the tool will quote, which is Nik's call to make
# with these numbers in front of him rather than mine to slip in beside an
# optimiser fix. Set ENFORCE_AREA_DETERMINACY = True to apply it.
AREA_STDERR_RATIO_MAX = 0.35
ENFORCE_AREA_DETERMINACY = False

# --- Module 6 -----------------------------------------------------------
PAIR_MAX_SEPARATION_MV = 200
EXCLUDE_FORMATION_FROM_DELTAV = True

# --- Module 7c ----------------------------------------------------------
COHERENCE_DRIFT_IMPLAUSIBLE_MV = 20.0
COHERENCE_MEDIAN_COHERENT_MV = 5.0
COHERENCE_MEDIAN_SUSPECT_MV = 15.0
COHERENCE_MIN_PAIRS = 8
COHERENCE_AREA_RATIO_MAX = 2.0
COHERENCE_BAND_TOL_MV = 70.0

# --- NEW in 1.9.0: mechanism and integrity ------------------------------
# A reversal sitting where |dQ/dV| is at least this fraction of the
# half-cycle's OWN maximum is on a plateau. Measured across three chemistries:
# at 0.05, healthy half-cycles (CE 90-110%) carry at most 0.4% of their
# capacity backwards, and every half-cycle that carries more sits in a cycle
# with CE below 60%.
PLATEAU_DQDV_FRACTION = 0.05
LIMIT_TOL_MV = 50.0              # "at the voltage limit"
# Bands. Judged against the half-cycle's OWN counter capacity, never against a
# population — the pooled-threshold mistake was made twice in 1.8.x and
# transferred neither between cells nor between chemistries.
PARASITIC_ANOMALOUS_FRACTION = 0.02
INTEGRITY_MIN_RECORDS = 20

# A half-cycle cannot pass more charge than the material can hold. Above this
# multiple of the theoretical capacity, the intended reaction does not account
# for what happened, whatever the shape of the curve says — and that is true
# independently of the plateau/parasitic split, which reads the curve.
#
# This test exists because the split alone got it wrong. On P3 cell A it
# reported 598 mAh/g of reversed charge as sitting on a two-phase plateau, and
# called that reassuring. 94% of it is on nine half-cycles delivering 1.3 to
# 4.3 times the theoretical capacity — cycle 1 charge alone puts in 681 mAh/g
# against a theoretical 160. Charge at 4.3x theoretical is something breaking,
# not a phase transition. On the half-cycles that ARE physically possible the
# plateau charge is 38 mAh/g, not 598.
#
# 1.2 rather than 1.0: the first cycle legitimately exceeds the reversible
# capacity through SEI formation and other irreversible processes, and the
# theoretical figure is itself a nominal number for the composition.
OVER_THEORETICAL_FRACTION = 1.2
# Below this a constant-voltage hold has carried nothing worth reporting.
# A cycler records capacity to about a microamp-hour; on an 11 mg electrode
# that is ~1e-4 mAh/g, so a hold under 0.01 mAh/g is the arithmetic of
# rounding rather than a protocol artefact worth a line of console.
CV_HOLD_REPORT_FLOOR = 0.01

# A cell that has stopped delivering charge. Judged against the dataset's OWN
# working capacity, like everything else here, so it transfers between a
# 50 mAh/g cathode and a 130 mAh/g anode. On P3 cell A the record runs to cycle
# 220 while the cell delivers 0.00-0.04 mAh/g from about cycle 130: ninety
# cycles of nothing, each of which the fitter would otherwise fit.
DEAD_CELL_CAPACITY_FRACTION = 0.02
DEAD_CELL_RUN = 3

# The one threshold in 1.9.0's reporting, and it sits inside a measured gap.
# Closure-interval widths: LTO 0.00-0.01, P3 0.00-0.09, NMC111 0.35-0.40,
# synthetic `degenerate` 0.81. Nothing observed lies between 0.09 and 0.35.
# Above this, the peak/background split is not determined by the data, so the
# per-peak SHARE of capacity is not a measurement and is withheld. The areas
# themselves are kept — withheld means "not reported as a fraction", not
# "deleted".
# ONE number, defined beside the measurement it thresholds. Until 1.9.0.13
# attribution withheld above 0.15 while `quality.describe_partition` printed
# its verdict against 0.05, so a dataset could be told "the partition survives
# a change of baseline" and then have every one of its cycles withheld.
ATTRIBUTION_WITHHOLD_ABOVE = PARTITION_DETERMINED_BELOW
# ...and the same rule for the charge the model could not name. Kept as its
# own name because the two grounds are different questions and a reader
# chasing one should not land on the other: the interval asks whether the
# split is decided by the data, this asks whether the components add up to
# the cell. See `quality.UNATTRIBUTED_QUALIFIED_ABOVE` for the measurement.
UNATTRIBUTED_WITHHOLD_ABOVE = UNATTRIBUTED_QUALIFIED_ABOVE

# How much of a component's analytic area has to fall OUTSIDE the fitted
# window before its area is materially a lower bound rather than a
# measurement. `fit_truncated` and `area_is_lower_bound` have been computed
# and carried to the CSV since 1.9.0.5x and read by NOTHING — the third time
# this build has found a correct diagnostic consumed by nobody, after
# `band_at_width_bound`'s attribution half and `band_width_max_fitted`.
#
# Measured on the NNM triplicate, 1.9.0.72: 507 of 3008 components (17%) are
# more than a tenth outside the window, and 214 OF THOSE ARE MARKED
# `reliable` — a median 34% of their area, up to 50%, sitting outside the
# data, with nothing on the page saying so. They are the 2.1 V and 4.1 V
# features at the ends of the voltage window.
#
# A tenth is where the number stops being a rounding matter: below it the
# enclosed-fraction correction is smaller than the area's own scatter
# between cycles, above it the quoted area understates the feature by more
# than any trend being read off it.
AREA_LOWER_BOUND_REPORTABLE = 0.10

# Should a component whose area is a LOWER BOUND be struck from `reliable`,
# or only reported as one? The same open question as
# `ENFORCE_AREA_DETERMINACY`, and answered the same way for the same reason:
# `reliable` is a statement about the FIT, and whether the window caught the
# whole feature is a statement about the MEASUREMENT. Flag, report, do not
# enforce — turn this on if you decide `reliable` should mean "the area is
# quotable as it stands". On NNM 1.9.0.72 it costs 214 of 2066 reliable
# components (68.7% -> 61.6%).
#
# THIS WAS NOT A FREE CHOICE — IT WAS MEASURED. `sigma_max` was swept
# 0.200 -> 0.350 V on the NNM triplicate to test whether it censors widths
# the way the band ceiling did. Result:
#
#   at the sigma bound      10.0% -> 1.8%
#   reliable                68.7% -> 76.3%
#   median R2             0.99566 -> 0.99567     <- IDENTICAL
#   area stderr determined  41.3% -> 49.1%
#
# and, matching the SAME components across the two runs, the 257 that were at
# the 0.200 bound moved their larger flank a median of 26 mV (0.200 ->
# 0.226) while the area sitting outside the window barely moved (39.5% ->
# 38.8%) — and 79% of them became `reliable`.
#
# 203 components flip to reliable. 200 OF THEM ARE STILL MORE THAN A TENTH
# OUTSIDE THE WINDOW, a median 39% and up to 50%, and 99% sit within one
# sigma_max of a window edge (median 55 mV from it). 6.6 of the 6.7 points
# are areas that are still lower bounds.
#
# So sigma_max is NOT the band ceiling wearing a different hat. The band
# ceiling bounded a quantity the data could measure, and freeing it improved
# R2 from 0.9916 to 0.9957. This bounds a quantity the data CANNOT measure —
# a component at the window edge has no data on one flank, so nothing but the
# bound stops it — and freeing it changes R2 in the fifth decimal place while
# moving 200 unquotable areas into the reliable column. Raising it was
# rejected. What the population needed was to be NAMED, which is what
# `area_mostly_outside_window` and the report line now do.
ENFORCE_AREA_COMPLETENESS = False


# =============================================================================
# PORTED FROM 1.8.7 — extracted programmatically, do not edit without a
# regression. `_flag_reliable_peaks` from Cell 3, the rest from Modules 5 and 6.
# =============================================================================

def _flag_reliable_peaks(params_df, sigma_max_mV=None,
                         area_floor=0.01,
                         bound_proximity=SIGMA_BOUND_PROXIMITY):
    """
    Return a boolean Series (aligned to params_df): True for peaks that are
    physically reliable, False for degenerate fits that should not appear in
    a publication parameter table.

    Degenerate = integrated area essentially zero (the fitter retained a peak
    that has effectively vanished) OR width inflated to the upper sigma bound
    (the 'peak' is acting as baseline). These mirror two of Module 7's health
    flags. The LOWER sigma bound is deliberately not used: a sharp, genuine
    peak (e.g. the LTO two-phase feature) sits near it by design, so dropping
    on that criterion would remove real data.
    """
    import pandas as pd
    if params_df is None or len(params_df) == 0:
        return pd.Series(dtype=bool)
    if 'amplitude_area' not in params_df or 'sigma' not in params_df:
        return pd.Series(True, index=params_df.index)
    # PER ROW, from the bound that fit actually used. A module-level 200 mV
    # could never be reached by a sharp profile (bound 30 mV) or a moderate
    # one (100 mV), so on every LTO and LFP dataset this test — the one the
    # docstring says catches a component acting as baseline — was dead.
    if "sigma_max_fitted" in params_df:
        bound = pd.to_numeric(params_df["sigma_max_fitted"], errors="coerce")
    else:
        bound = pd.Series(np.nan, index=params_df.index)
    if sigma_max_mV is not None:
        bound = bound.fillna(float(sigma_max_mV) / 1000.0)
    bound = bound.fillna(FIT_SIGMA_MAX_MV / 1000.0)
    sigma_high = bound * (1 - bound_proximity)
    area = params_df['amplitude_area'].abs()
    sigma = params_df['sigma']
    # A comparison against NaN is False on float dtype, not NaN, so
    # `.fillna(True)` never fired and a component with a NaN area or sigma
    # was marked RELIABLE. Unknown is not the same as sound: it is tested
    # explicitly.
    # BOTH FLANKS, 1.9.0.58. `sigma` alone was the whole width while the
    # lineshape was symmetric. With a split one, a component whose HIGH flank
    # is pinned at the ceiling — which is what a terminal feature does — was
    # passing this test and being reported as reliable. See
    # `fitting.ASYMMETRY_SHARED`.
    sigma_r = params_df["sigma_r"] if "sigma_r" in params_df else sigma
    wide = (sigma >= sigma_high) | (sigma_r >= sigma_high)
    degenerate = ((area < area_floor) | wide
                  | area.isna() | sigma.isna())
    return ~degenerate




def _coherent_net_change(cycles, values, gap_break=3, edge_n=3,
                         min_points=5):
    """
    Model-free magnitude of change over the leading coherent window.

    Makes no linearity assumption: returns the difference between the
    median of the last edge_n points and the first edge_n points within
    the coherent window (the same leading window used elsewhere). Use this
    for descriptors whose evolution is clearly not linear, e.g. ΔV.

    Returns (net_change, v_start, v_end, c_start, c_end, n_used).
    """
    s = pd.DataFrame({'c': np.asarray(cycles, float),
                      'v': np.asarray(values, float)}).dropna()
    s = s.sort_values('c')
    if len(s) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0
    c = s['c'].to_numpy(); v = s['v'].to_numpy()
    end = len(c)
    for k in range(1, len(c)):
        # MISSING cycles, as the docstring says: a cycle-number jump of 3
        # is two missing cycles, and comparing the jump itself truncated the
        # coherent window one gap earlier than documented.
        if (c[k] - c[k - 1] - 1) >= gap_break:
            end = k
            break
    c, v = c[:end], v[:end]
    n = len(c)
    if n < min_points:
        return (np.nan, np.nan, np.nan,
                c[0] if n else np.nan, c[-1] if n else np.nan, n)
    m = max(1, min(edge_n, n // 2))
    v_start = float(np.median(v[:m]))
    v_end = float(np.median(v[-m:]))
    return v_end - v_start, v_start, v_end, c[0], c[-1], n


def _discontinuity_events(pk_df):
    """
    Detect discontinuity events for one (step, peak_id).

    pk_df: all rows for this peak (any status), with 'cycle',
    'status', 'centre', and 'reliable' columns.

    An event is EITHER:
      (a) a sustained dropout — a run of >= DISCONTINUITY_DROPOUT_RUN
          consecutive cycles that are not reliably tracked, which is
          preceded AND followed by at least one reliable tracked
          cycle (vanish-and-return mid-life); OR
      (b) a single-cycle centre step between consecutive reliable
          tracked cycles exceeding max(FLOOR, K x median step).

    Returns (n_events, first_event_cycle, max_centre_step_mV, established).
    """
    d = pk_df.sort_values('cycle')
    # WHERE THE RUN STOPPED FITTING IS NOT WHERE THE FEATURE FAILED.
    # `track_peaks` writes a `no_fit` row for every cycle past the fit limit,
    # for every peak, so a peak still perfectly healthy at the limit acquired a
    # terminal run of not-good cycles beginning at limit+1 and was published as
    # "failed mid-life ... this is a feature being lost". NNM cell A said "Cell
    # stopped delivering at cycle 128. Peak fitting stopped there" and, four
    # lines above it, listed two features as failing "from cycle 129"; cell B
    # listed three "from cycle 68" against a fit limit of 67. Those rows record
    # that nothing was ASKED of the model, so they are dropped from the end of
    # the record before any dropout is measured. A `no_fit` in the middle of a
    # record is a real failure to converge and is left exactly where it is.
    if 'status' in d.columns and len(d):
        _st = d['status'].astype(str).to_numpy()
        _end = len(_st)
        while _end > 0 and _st[_end - 1] in ('no_fit', 'not_fitted'):
            _end -= 1
        if _end == 0:
            return 0, np.nan, np.nan, True
        d = d.iloc[:_end]
    cycles = d['cycle'].to_numpy()
    # "good" = reliably tracked this cycle
    if 'reliable' in d.columns:
        good = ((d['status'] == 'tracked') & (d['reliable'])).to_numpy()
    else:
        good = (d['status'] == 'tracked').to_numpy()

    if good.sum() < 3:
        return 0, np.nan, np.nan, True

    event_cycles = []

    # --- (a) sustained dropouts bracketed by good tracking ---
    first_good = np.argmax(good)            # index of first reliable track
    run_len = 0
    run_start_idx = None
    # Scan from the first reliable track to the end. A run of
    # not-good cycles that begins AFTER the peak was once reliably
    # tracked is a sustained dropout — whether or not the peak ever
    # returns cleanly. The most severe failures never return cleanly,
    # so we do NOT require a reliable cycle after the run.
    for i in range(first_good, len(good)):
        if not good[i]:
            if run_len == 0:
                run_start_idx = i
            run_len += 1
        else:
            if run_len >= DISCONTINUITY_DROPOUT_RUN:
                event_cycles.append(int(cycles[run_start_idx]))
            run_len = 0
    # close out a run that extends to the end of life. A terminal
    # run (one reaching the final cycle) is held to a longer
    # threshold: a few lost cycles at the very end is the record
    # ending, not a feature failing. A mid-life dropout that simply
    # never recovers will still be long enough to clear this.
    if run_len >= DISCONTINUITY_TERMINAL_RUN:
        event_cycles.append(int(cycles[run_start_idx]))

    # --- (b) single-cycle centre steps on consecutive good cycles ---
    gd = d[good]
    gcyc = gd['cycle'].to_numpy()
    gcen = gd['centre'].to_numpy()
    steps_mV = np.full(len(gcen), np.nan)
    adjacent = np.diff(gcyc) == 1          # only consecutive cycles
    raw_steps = np.abs(np.diff(gcen)) * 1000.0
    steps_mV[1:] = np.where(adjacent, raw_steps, np.nan)
    _valid = steps_mV[1:][np.isfinite(steps_mV[1:])]
    if _valid.size >= 3:
        med = np.median(_valid)
        # robust spread: median absolute deviation, scaled to ~sigma
        mad = np.median(np.abs(_valid - med)) * 1.4826
        thresh = max(DISCONTINUITY_STEP_FLOOR_MV,
                     med + DISCONTINUITY_STEP_K * mad)
    else:
        thresh = DISCONTINUITY_STEP_FLOOR_MV
    for j in range(1, len(steps_mV)):
        if np.isfinite(steps_mV[j]) and steps_mV[j] > thresh:
            event_cycles.append(int(gcyc[j]))

    # np.nanmax over an all-NaN array warns and returns NaN; a peak tracked
    # only on non-consecutive cycles hits that on every dataset.
    max_step = (float(np.nanmax(steps_mV))
                if np.isfinite(steps_mV).any() else np.nan)
    if not event_cycles:
        return 0, np.nan, max_step, True  # established irrelevant; no event
    first_event = int(min(event_cycles))
    # Reliable cycles accrued before the first event: was the peak ever
    # established, or did it only flicker at the start of life?
    n_good_before = int(np.sum(good & (cycles < first_event)))
    established = n_good_before >= DISCONTINUITY_ESTABLISH_MIN
    return len(event_cycles), first_event, max_step, established


def _auto_pair_peaks(charge_ref_peaks, discharge_ref_peaks,
                      max_sep_mV=PAIR_MAX_SEPARATION_MV,
                      is_anode=False):
    """
    Pair charge and discharge reference peaks by voltage proximity.

    Uses globally optimal one-to-one assignment (Hungarian algorithm),
    minimising the total charge-discharge separation across all pairs at
    once, subject to two constraints: a pairing is allowed only within
    max_sep_mV, and it must give a ΔV of the sign expected for the
    electrode. For a cathode the charge peak sits above the discharge peak
    so ΔV is non-negative; for an anode (is_anode=True) the labels are
    swapped upstream so the delithiation "Discharge" peak sits above the
    lithiation "Charge" peak, so the expected ΔV is negative and the sign
    test is flipped.

    A per-peak nearest-neighbour scheme is fragile when two couples sit
    close: one charge peak takes the other's natural partner, forcing a
    crossed assignment and a wrong-sign ΔV. Global assignment with the
    expected-sign constraint avoids this.

    All peaks participate (primary and shoulder). Returns list of
    (charge_id, discharge_id, charge_V, discharge_V, pair_is_shoulder).
    """
    if charge_ref_peaks.empty or discharge_ref_peaks.empty:
        return []

    from scipy.optimize import linear_sum_assignment

    max_sep_V = max_sep_mV / 1000.0
    neg_tol_V = 0.005   # tolerate tiny negative from reference fit noise

    charge_sorted = charge_ref_peaks.sort_values('voltage').reset_index(drop=True)
    discharge_list = discharge_ref_peaks.reset_index(drop=True)

    n_c, n_d = len(charge_sorted), len(discharge_list)
    BIG = 1.0e6
    cost = np.full((n_c, n_d), BIG)
    for i in range(n_c):
        cv = charge_sorted.loc[i, 'voltage']
        for j in range(n_d):
            dv = discharge_list.loc[j, 'voltage']
            sep = abs(cv - dv)
            # Expected-sign test, electrode-aware. For a cathode, charge
            # sits at or above discharge (ΔV = cv - dv >= 0). For an anode
            # the labels were swapped upstream so "Discharge" is the
            # delithiation peak, which sits ABOVE the lithiation "Charge"
            # peak, so the expected ΔV is negative and the test flips.
            if is_anode:
                sign_ok = cv <= dv + neg_tol_V
            else:
                sign_ok = dv <= cv + neg_tol_V
            if sep <= max_sep_V and sign_ok:
                cost[i, j] = sep

    row_idx, col_idx = linear_sum_assignment(cost)

    pairs = []
    for i, j in zip(row_idx, col_idx):
        if cost[i, j] >= BIG:
            continue   # too far apart, or would give negative ΔV
        cpk = charge_sorted.loc[i]
        dpk = discharge_list.loc[j]
        either_shoulder = (bool(cpk.get('is_shoulder', False)) or
                           bool(dpk.get('is_shoulder', False)))
        pairs.append((
            int(cpk['peak_id']), int(dpk['peak_id']),
            cpk['voltage'], dpk['voltage'],
            either_shoulder
        ))

    return pairs


# WHICH COMPONENTS COUNT AS CAPACITY.
#
# `band_at_width_bound` was half-wired: it demoted a component from `reliable`
# and its area then went on entering `total_area`, `retained_fraction` and
# every `peakN_fraction` regardless. On NMC 14 of 68 tracked components per
# cell were pinned bands being counted in shares they were too unreliable to
# appear in individually, and that is what produced the page's own warning
# about 570% area retention. A band at its width ceiling is not a composition
# window any more; it is a flat pedestal the fit is using to lift the curve,
# and a pedestal is baseline, not charge.
#
# `area_empty` joins it for the same reason from the other end: a component
# carrying a thousandth of its neighbour's charge contributes nothing but
# noise to a sum.
#
# The excluded areas are not deleted — `total_area_all` reports the sum with
# them in, and `n_excluded_from_area` says how many there were, so the
# difference between the two numbers is visible rather than silent.
# WHICH AREA IS THE CAPACITY. Switched in 1.9.0.60.
#
# Every component carries two areas and until now every capacity number used
# the wrong one. `amplitude_area` is the ANALYTIC integral of the fitted
# shape over +/- infinity; `area_in_window` is the numerical integral over the
# voltage range actually measured. They are the same thing for a peak that
# sits well inside the window and they are not the same thing at all for one
# that does not.
#
# The half-cycle level already used the in-window integral —
# `component_area_sum`, `unattributed_area` and `unattributed_fraction` in
# `fitting` are all `trapezoid` over the curve — so the per-component areas
# did not sum to the quantity the closure test checks them against. Measured
# over the current runs, sum(analytic) / sum(in-window) per half-cycle:
# LTO 1.010-1.012, NMC 1.061-1.083, NNM 1.169-1.222.
#
# WHAT CHANGES. Each component's share of its own half-cycle moves by:
#
#     LTO   median 0.00 pp   p95 0.00 pp   max 0.0 pp
#     NMC   median 0.7 pp    p95 4.1 pp    max 6.7 pp
#     NNM   median 1.8 pp    p95 8.0 pp    max 10.8 pp
#
# Nothing at all on LTO, because there the over-count is uniform (every
# component 0.99) and a uniform factor cancels out of a share. What moves is
# exactly what should move: the per-component ratio falls to 0.472 at worst,
# a component reporting more than twice the charge it accounts for inside the
# measured range, and those are the edge components whose analytic tail is an
# extrapolation rather than a measurement.
#
# The analytic value stays in the table. It is the better estimate of a
# process's TOTAL charge for a peak that is fully contained, and the two
# columns side by side are how a reader sees the difference. What it stops
# being is the number the capacity attribution, the tracked areas and the
# coherence audit's area test are computed from.
CAPACITY_AREA_COLUMN = "area_in_window"


def _area(df):
    """The area column the capacity numbers are computed from.

    Falls back to `amplitude_area` where `area_in_window` is absent, so a
    table written by an older build still reads.
    """
    if CAPACITY_AREA_COLUMN in getattr(df, "columns", ()):
        _a = pd.to_numeric(df[CAPACITY_AREA_COLUMN], errors="coerce")
        if _a.notna().any():
            return _a.abs()
    return pd.to_numeric(df["amplitude_area"], errors="coerce").abs()


ATTRIBUTION_EXCLUDES = ("band_at_width_bound", "area_empty")


def _countable_area(cd):
    """Boolean mask: components whose area may be counted as capacity."""
    keep = pd.Series(True, index=cd.index)
    for col in ATTRIBUTION_EXCLUDES:
        if col in cd.columns:
            keep &= ~cd[col].fillna(False).astype(bool)
    return keep


def _attribution_basis(step_tracked, ref_cycle):
    """
    Fixed normalisation basis for capacity attribution: the summed fitted
    area of every peak tracked at the reference cycle.

    Using the reference cycle — the same baseline Module 5 uses for drift
    and retention — keeps attribution consistent with those quantities and
    makes the basis predictable and defensible across datasets rather than
    chosen by a heuristic. By construction the stack is 100% at the
    reference cycle and falls below it as capacity is lost. A feature
    present in the reference cycle that then fades shows its band
    collapsing, which is reported, not excluded.

    Returns (basis_area, basis_cycle).
    """
    cd = step_tracked[step_tracked['cycle'] == ref_cycle]
    if cd.empty:
        # Reference cycle has no tracked peaks for this step (unusual):
        # fall back to the earliest tracked cycle so a basis still exists.
        cycles = sorted(step_tracked['cycle'].unique())
        if not cycles:
            return np.nan, None
        ref_cycle = cycles[0]
        cd = step_tracked[step_tracked['cycle'] == ref_cycle]
    basis_area = float(_area(cd.loc[_countable_area(cd)]).sum())
    return basis_area, ref_cycle


# =============================================================================
# 1. FROM FIT RESULTS TO A COMPONENT TABLE
# =============================================================================

_PARAM_COLS = ("centre", "centre_stderr", "amplitude_area", "amplitude_stderr",
               "sigma", "sigma_stderr", "fraction", "fraction_stderr",
               "fwhm", "height", "area_in_window",
               # The solid-solution band: its width, and whether the fit
               # decided this component is a band at all or a plain peak.
               # `band_at_width_bound` is the band's equivalent of a sigma
               # pinned at its ceiling — a component acting as baseline.
               "band_width", "band_width_stderr", "component_kind",
               # THE CEILING THE FLAG WAS MEASURED AGAINST, not just the flag.
               # `band_at_width_bound` was in the table from 1.9.0.5x and the
               # ceiling it compares to was not, so a reader could see that a
               # component was pinned but not what it was pinned at — and the
               # ceiling is NOT one number. A component promoted inside a
               # dataset region is capped at that region's span; one seeded
               # from the residual gets the mechanism's own value. On the NNM
               # triplicate those are 0.400 V and 0.500 V, and the missing
               # column is why the pile-up at 0.400 went unseen for nine
               # builds. See `_band_ceilings`.
               "band_width_max_fitted",
               "band_at_width_bound",
               # eta pinned at 0 or 1 — the mixing fraction's equivalent of
               # the two flags either side of it. See `fraction_at_bound` in
               # `fitting`: a pure Lorentzian's wings carry area far from the
               # centre, and on LTO that shows up directly as the model
               # over-covering the cell's charge.
               "fraction_at_bound",
               # THE SPLIT LINESHAPE. `sigma` is the low-side half-width at
               # half maximum and `sigma_r` the high side, so `fwhm` is their
               # sum and not `2 * sigma`. Both are present on every fit:
               # a symmetric one reports `sigma_r == sigma` and
               # `asymmetry == 1.0`, so the column always means the same
               # thing and nothing downstream needs to know which model ran.
               "sigma_r", "sigma_r_stderr", "asymmetry")

# Computed in `parameters_frame` rather than read from the fit result, so
# they are listed separately — but they must reach the exported table and the
# tracking table, which is what `is_inferred` failed to do for four builds.
# THIS LIST HAD DUPLICATES AND A BROKEN INDENT — three names appeared twice
# from two separate edits that each appended rather than merging. Harmless in
# effect (the consumers deduplicate) but it is the visible edge of the bug
# that HAS bitten: a flag computed here and never added to a list like this
# one silently never reaches the CSV. `is_inferred` was missing for four
# builds; `fraction_at_bound` and `mechanism_own` each went missing once.
# Sorted and unique so an addition is obvious and a duplicate is not.
_DERIVED_FLAG_COLS = ("area_determinacy",
                      "area_empty",
                      "area_poorly_determined",
                      "area_stderr_missing",
                      "area_stderr_ratio",
                      "at_sigma_floor",
                      "at_sigma_max",
                      "reliability_reason",
                      "sigma_floor_side",
                      # NEW IN 1.9.0.58, and the reason the duplicates above
                      # were noticed: which flank reached the width ceiling.
                      "sigma_max_side",
                      "width_undersampled")

# --- when is a peak truncated? -------------------------------------------
# Detection answers this from GEOMETRY, before fitting: it compares the two
# half-maximum half-widths and calls a peak truncated when the clipped side is
# short. That is an estimate of a quantity the fit measures directly, and it
# is made from the noisiest part of the curve — a half-maximum crossing on a
# flank that is partly outside the window.
#
# The fit answers it properly. A component is truncated when its own
# centre +/- TRUNCATION_SIGMAS * sigma is not wholly inside the window that
# was fitted. Two sigma covers 95% of a Gaussian, so a peak that passes has at
# most a few per cent of its area outside and its integral is a measurement;
# one that fails has an area that is a LOWER BOUND, and the column says so
# rather than the peak being dropped.
#
# The corroborating flag is free and independent: a truncated peak's centre is
# poorly determined because half the information that fixes it is missing, so
# `centre_stderr / sigma` rises. Above 0.1 the centre is uncertain at the
# 10%-of-width level, which for a 60 mV FWHM feature is +/- 3 mV.
TRUNCATION_SIGMAS = 2.0
TRUNCATION_CENTRE_STDERR_RATIO = 0.10


def _anomalous_half_cycles(P, integrity):
    """
    Per-component: was this component's half-cycle banded ANOMALOUS?

    `integrity` is `cell_integrity`'s table — one row per half-cycle, with
    `cycle`, `step` and `band`. Absent (a cycling-only run, or an older
    caller) it returns all-False rather than raising: a flag that cannot be
    evaluated must not silently condemn everything, and the column has to
    exist either way so nothing downstream has to ask whether this build
    recorded it.

    ANOMALOUS ONLY, not `suspect`. The band ladder is deliberate: `suspect`
    means something was seen and could not be explained, `ANOMALOUS` means the
    half-cycle passed more charge than the material can hold or lost charge to
    nothing. The first is a caveat; only the second says the curve is not a
    measurement of the material.
    """
    _false = pd.Series(False, index=P.index)
    if integrity is None or getattr(integrity, "empty", True):
        return _false
    if not {"cycle", "step", "band"} <= set(integrity.columns):
        return _false
    bad = {(int(c), str(s)) for c, s, b in zip(
        integrity["cycle"], integrity["step"], integrity["band"])
        if str(b) == "ANOMALOUS" and pd.notna(c)}
    if not bad:
        return _false
    return pd.Series(
        [(int(c), str(s)) in bad for c, s in zip(P["cycle"], P["step"])],
        index=P.index)


def parameters_frame(fit_results, detection, *, name=None,
                     sigma_max_mV=FIT_SIGMA_MAX_MV,
                     width_floor_factor=FIT_WIDTH_FLOOR_FACTOR,
                     min_area_fraction=FIT_MIN_AREA_FRACTION,
                     integrity=None):
    """
    The long component table, one row per fitted component.

    `fit_results` is what `fitting.fit_many` returns: a list of plain dicts,
    each carrying `key=(cycle, step)`. The seeded centres came from
    `detection.peaks[key]`, in that order, so component `peak_index` i is the
    fit of detected peak i and inherits its `is_shoulder` / `is_truncated` /
    `detected_voltage`.

    The reliability columns are 1.8.7's, in 1.8.7's order: `_flag_reliable_peaks`
    first, then the two RELATIVE floors — width and area, each measured against
    this dataset's own median reliable component rather than an absolute bound,
    because an absolute one cannot serve a sharp LTO profile and a broad
    layered-oxide profile at once. Both are recorded in their own columns so the
    exclusion is auditable rather than silent.
    """
    rows = []
    for fr in fit_results:
        key = fr.get("key")
        if key is None:
            continue
        cycle, step = key
        det_pk = detection.peaks.get((cycle, step))
        for comp in fr.get("components", []):
            i = comp["peak_index"]
            # Join on the SEED index — the component's position in the
            # detected peak list it started from. `peak_index` is renumbered
            # when a degenerate component is dropped and the half-cycle
            # refitted, so joining on it gave every surviving component the
            # detected voltage, shoulder flag and truncation flag of the one
            # below it, with nothing to say so.
            sx = comp.get("seed_index", i)
            d = {}
            if det_pk is not None and sx < len(det_pk):
                d = det_pk.iloc[sx]
            row = dict(
                dataset=name if name is not None else detection.name,
                cycle=int(cycle), step=str(step),
                fit_success=bool(fr.get("success")),
                # `success` is "the fit ran without raising". CONVERGENCE is
                # a different question and it is the one that matters here:
                # `fit_half_cycle` records `res.success` separately, and
                # until 1.9.0.36 nothing but a console tally read it, so a
                # half-cycle that ran to the 20,000-evaluation ceiling
                # reached the parameter table indistinguishable from one
                # that settled. On NMC111 that was 87% of them, and those
                # same components then fed drift, area retention and
                # capacity attribution. Carried here so the reliability
                # test can act on it.
                converged=bool(fr.get("converged", True)),
                hit_iteration_cap=bool(fr.get("hit_iteration_cap", False)),
                # DID A PARAMETER FINISH ON A BOUND, AND DID MOVING IT OFF
                # HELP? A diagnostic computed in the fitter and consumed by
                # nothing is the pattern this project has now caught five
                # times, so it travels to the table with the rest.
                # See `fitting.BOUND_ESCAPE_RESTART`.
                bound_escape_tried=int(fr.get("bound_escape_tried", 0) or 0),
                bound_escape_used=bool(fr.get("bound_escape_used", False)),
                nfev=int(fr.get("nfev", 0) or 0),
                r_squared=fr.get("r_squared", np.nan),
                reduced_chi_sq=fr.get("redchi", np.nan),
                # THE FOUR NUMBERS R2 HIDES. Per half-cycle, repeated on each
                # of its component rows exactly as `r_squared` is, so the
                # parameter table can be read without joining anything.
                height_ratio=fr.get("height_ratio", np.nan),
                overshoot=fr.get("overshoot", np.nan),
                max_residual_frac=fr.get("max_residual_frac", np.nan),
                residual_runs_z=fr.get("residual_runs_z", np.nan),
                # The half-cycle's one shared asymmetry ratio, and whether it
                # settled on a bound. There is only one of these per
                # half-cycle, so nothing else would notice it pinning.
                asymmetry_k=fr.get("asymmetry_k", np.nan),
                asymmetry_k_fitted=fr.get("asymmetry_k_fitted", np.nan),
                asymmetry_k_clipped=bool(fr.get("asymmetry_k_clipped", False)),
                asymmetry_at_bound=bool(fr.get("asymmetry_at_bound", False)),
                # A per-CYCLE detection index, not a cross-cycle identity:
                # it shifts whenever detection finds a different number of
                # peaks. `tracked_peak_id` is the one that means what
                # `reference_peak_id` invited a reader to assume.
                component_index=int(sx) + 1,
                # The width bound this half-cycle was actually fitted under,
                # so the reliability test can ask whether a component is
                # pinned at ITS OWN bound rather than at a module constant.
                sigma_max_fitted=float(fr.get("sigma_max", np.nan)),
                # ...and the LOWER bound, and the curve's own sampling
                # interval. A width sitting on its floor and a width read
                # from three points are both statements about the
                # MEASUREMENT rather than about the peak, and neither was
                # available downstream. See `at_sigma_floor` and
                # `width_undersampled`.
                sigma_min_fitted=float(fr.get("sigma_min", np.nan)),
                sample_mV=float(fr.get("sample_mV", np.nan)),
                dropped_degenerate=int(fr.get("dropped_degenerate", 0)),
                # HOW MUCH DATA THIS FIT HAD, AND HOW MUCH OF IT IT SPENT.
                # `fit_one` has computed `n_points`, `nvarys` and
                # `n_shoulders_coupled` since the histogram rewrite and not one
                # of the three reached a file, a figure or a sentence: the only
                # consumer of the per-fit dictionary is a hard-coded five-column
                # summary used to pick which half-cycles to draw. The comment
                # beside `n_shoulders_coupled` in `fitting.fit_one` says it
                # exactly — "`nvarys` alone cannot be read without them ... the
                # difference is the whole argument for believing the standard
                # errors" — and then the argument was never put. A fit of 60
                # bins with 14 varied parameters and one of 600 with the same 14
                # are not the same measurement, and every standard error, every
                # `*_poorly_determined` flag and every reliability verdict on
                # this page rests on which of the two it was.
                n_points=int(fr.get("n_points", 0) or 0),
                nvarys=int(fr.get("nvarys", 0) or 0),
                n_shoulders_coupled=int(fr.get("n_shoulders_coupled", 0) or 0),
                points_per_parameter=(
                    float(fr.get("n_points", 0) or 0) / float(fr["nvarys"])
                    if fr.get("nvarys") else np.nan),
                # WHERE THE COMPONENTS CAME FROM. A band added by the residual
                # sweep is not a detected feature, and until now the table gave
                # no way to tell one from the other at half-cycle level.
                bands_from_residual=int(fr.get("bands_from_residual", 0) or 0),
                # HOW MUCH OF THE CURVE THE BASELINE TOOK. `baseline_area` and
                # `curve_area` were both computed and neither published, so the
                # one question a reader of an attributed area always asks —
                # how much of this half-cycle did the model call background? —
                # had no answer anywhere in the run. As a fraction, because the
                # absolute is only meaningful beside a curve area nobody sees.
                baseline_area_fraction=(
                    float(fr["baseline_area"]) / float(fr["curve_area"])
                    if (np.isfinite(fr.get("baseline_area", np.nan))
                        and np.isfinite(fr.get("curve_area", np.nan))
                        and fr.get("curve_area")) else np.nan),
                unattributed_fraction_fit=float(
                    fr.get("unattributed_fraction", np.nan)),
                detected_voltage=float(d["voltage"]) if len(d) else np.nan,
                is_shoulder=bool(d["is_shoulder"]) if len(d) else False,
                # SEEDED BY RECURRENCE, not by this half-cycle's own peak
                # picker: a feature the run finds at the same potential cycle
                # after cycle, whose prominence here is below every gate.
                # Provenance matters — see `detect.recurrent_maxima`.
                is_recurrent=bool(d.get("is_recurrent", False)) if len(d)
                else False,
                # The DETECTION-STAGE geometric guess, kept because it is
                # what seeded the fit. `fit_truncated` below is the answer.
                is_truncated=bool(d["is_truncated"]) if len(d) else False,
                window_v_min=float(fr.get("v_min", np.nan)),
                window_v_max=float(fr.get("v_max", np.nan)),
                # A component seeded from a NEIGHBOURING cycle's peak list
                # because this cycle's detection found nothing there. Set by
                # `infer_missing_seeds`; the provenance matters, because a
                # component that was never detected in the cycle it is
                # reported for is a different kind of number from one that
                # was.
                is_inferred=bool(d.get("is_inferred", False)) if len(d)
                else False,
            )
            for c in _PARAM_COLS:
                row[c] = comp.get(c, np.nan)
            rows.append(row)

    P = pd.DataFrame(rows)
    if P.empty:
        return P

    # --- the truncation test, from the fit rather than from geometry ----
    # EACH FLANK MEASURED WITH ITS OWN WIDTH. `sigma` is the LOW-side
    # half-width of a split lineshape and `sigma_r` the high side; both the
    # window test and the outside-area integral used `sigma` on both sides. On
    # an LTO terminal component with sigma 2.3 mV and sigma_r 30 mV that puts
    # the modelled high flank 65 mV inside the window when it is in fact 30 mV
    # outside it: `fit_truncated` False, `area_outside_window_frac` 0.000, and
    # roughly 5% of the component's area published as a measurement rather than
    # as a lower bound. `_flag_reliable_peaks` and the sigma-bound test were
    # both corrected to read both flanks in earlier builds; this block was
    # missed. Where there is no `sigma_r` the shape is symmetric and the two
    # widths are the same number, so single-flank fits are unaffected.
    _sig_l = P["sigma"].abs()
    if "sigma_r" in P.columns:
        _sig_r = pd.to_numeric(P["sigma_r"], errors="coerce").abs()
        _sig_r = _sig_r.where(_sig_r.notna() & (_sig_r > 0), _sig_l)
    else:
        _sig_r = _sig_l
    _lo = P["centre"] - TRUNCATION_SIGMAS * _sig_l
    _hi = P["centre"] + TRUNCATION_SIGMAS * _sig_r
    P["fit_truncated"] = ((_lo < P["window_v_min"]) |
                          (_hi > P["window_v_max"])).fillna(False)
    # How much of the modelled peak fell outside the window, as a fraction of
    # its analytic area. This is the number to quote when an area is a lower
    # bound — "truncated" alone does not say whether it matters.
    #
    # For a split normal the two halves are not equally weighted: the density
    # that is continuous at the centre and integrates to one puts
    # sigma_l/(sigma_l+sigma_r) of the area below the centre. The enclosed
    # fraction between the window edges is therefore
    #
    #     2/(s_l+s_r) * [ s_l*(PHI(z_hi^-) - PHI(z_lo^-))
    #                   + s_r*(PHI(z_hi^+) - PHI(z_lo^+)) ]
    #
    # with ^- clipped at or below zero and ^+ at or above it, which reduces to
    # the symmetric expression when s_l == s_r.
    _z_lo = ((P["window_v_min"] - P["centre"]) / _sig_l).astype(float)
    _z_hi = ((P["window_v_max"] - P["centre"]) / _sig_r).astype(float)
    # The same two edges expressed in the OTHER flank's width, needed where an
    # edge falls on the far side of the centre from the flank it belongs to.
    _z_lo_r = ((P["window_v_min"] - P["centre"]) / _sig_r).astype(float)
    _z_hi_l = ((P["window_v_max"] - P["centre"]) / _sig_l).astype(float)
    _phi = lambda z: 0.5 * (1.0 + erf(z / np.sqrt(2.0)))      # noqa: E731
    _w = (_sig_l + _sig_r).replace(0.0, np.nan)
    _below = _sig_l * (_phi(np.minimum(_z_hi_l, 0.0))
                       - _phi(np.minimum(_z_lo, 0.0)))
    _above = _sig_r * (_phi(np.maximum(_z_hi, 0.0))
                       - _phi(np.maximum(_z_lo_r, 0.0)))
    _inside = (2.0 / _w) * (_below + _above)
    P["area_outside_window_frac"] = (1.0 - _inside).clip(
        lower=0.0, upper=1.0)
    P["area_is_lower_bound"] = P["fit_truncated"]
    P["centre_stderr_ratio"] = (P["centre_stderr"] / P["sigma"].abs())
    P["centre_poorly_determined"] = (
        P["centre_stderr_ratio"] > TRUNCATION_CENTRE_STDERR_RATIO).fillna(False)

    P["reliable"] = _flag_reliable_peaks(P, sigma_max_mV=sigma_max_mV).values
    # A component that was never DETECTED in the cycle it is reported for is
    # kept in the table and tracked, but is not evidence: it is excluded from
    # every headline trend by the one flag those trends already read. See
    # `detect.INFER_MISSING_SEEDS`.
    P["reliable"] = P["reliable"] & ~P["is_inferred"].fillna(False)

    # A COMPONENT OF A HALF-CYCLE THAT PASSED MORE CHARGE THAN THE MATERIAL
    # CAN HOLD IS NOT EVIDENCE EITHER.
    #
    # `cell_integrity` bands such a half-cycle ANOMALOUS and the per-cell page
    # says so loudly. Nothing downstream asked. `not_measurements` — the set
    # the fit skips — is in-progress plus early terminations; the anomalous
    # band was not in it, so those half-cycles were fitted, tracked, and their
    # areas went into the retention trends and the capacity attribution beside
    # clean ones. Measured on Jiaqi's NNM, summed fitted component area per
    # charge half-cycle: cell A clean median 96.0 against 234.9 on its nine
    # anomalous half-cycles (2.45x), cell C 3-4x.
    #
    # AND THE EXISTING GUARDS RATE THEM AS GOOD. `integral_fidelity` asks
    # whether the curve accounts for the charge the cell delivered, and it
    # does — faithfully, because the charge really was delivered: on cell A
    # the anomalous half-cycles score 0.957 against a clean median of 0.928.
    # `reliable` fired on 0.43 of their components against 0.50 on clean ones.
    # No separation, from either. The integral is right; what it is an
    # integral OF is not. Cell C cycle 28 charge carries 1040 records against
    # ~350, 102 voltage reversals against 3, and 143 mAh/g of plateau charge —
    # the cell traversed the same voltages a hundred times, and a differential
    # capacity curve built from that is not a measurement of the material
    # however well it integrates.
    #
    # NOTHING IS DELETED, exactly as for `is_inferred`: the components stay in
    # the table, stay tracked, and stay in every figure. They are marked, and
    # the one flag the headline trends read excludes them.
    P["half_cycle_anomalous"] = _anomalous_half_cycles(P, integrity)
    P["reliable"] = P["reliable"] & ~P["half_cycle_anomalous"]
    # THE CONVERGENCE GATE IS GONE. It was added here by the reporting audit
    # on an argument that was right — do not quote numbers from a fit that did
    # not work — applied to a quantity that could not carry it. `converged` is
    # lmfit's exit status, and on NNM it was exactly `not hit_iteration_cap`:
    # it demoted 405 of 405 and 462 of 462 components whose fits had the
    # HIGHER median R2, because running out of evaluations while still
    # descending is not a failure. See AREA_STDERR_RATIO_MAX above for what
    # replaced it and why that is flagged rather than enforced.
    #
    # `converged` and `hit_iteration_cap` are still measured, still in the
    # parameter table, and still printed by Cell 9. They are facts about the
    # optimiser and they are reported as such; they no longer decide what
    # counts as a measurement.

    # Is the AREA determined? Flagged, not enforced — see the constant.
    _se = pd.to_numeric(P.get("amplitude_stderr"), errors="coerce") \
        if "amplitude_stderr" in P else pd.Series(np.nan, index=P.index)
    _ar = pd.to_numeric(P.get("amplitude_area"), errors="coerce").abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        _ratio = _se / _ar.where(_ar > 0)
    P["area_stderr_ratio"] = _ratio
    # A MISSING standard error is not a small one. lmfit leaves it None when
    # the covariance could not be estimated, which is the plainest statement
    # available that the data did not determine this parameter.
    P["area_stderr_missing"] = _se.isna() & _ar.notna()
    P["area_poorly_determined"] = (
        (_ratio > AREA_STDERR_RATIO_MAX).fillna(False)
        | P["area_stderr_missing"])
    if ENFORCE_AREA_DETERMINACY:
        P["reliable"] = P["reliable"] & ~P["area_poorly_determined"]
    # BOTH PLACES OR NEITHER. `track_peaks` builds its own table and recomputes
    # `reliable` from scratch, so a demotion applied only here reaches the
    # parameter CSV and nothing else — the trap this module's own
    # SIGMA_BOUND_PROXIMITY note warns about, walked into twice already. The
    # matching clause is in `_reliable`.
    if ENFORCE_AREA_COMPLETENESS and "area_outside_window_frac" in P:
        P["reliable"] = P["reliable"] & ~(
            P["area_is_lower_bound"].fillna(False).astype(bool)
            & (pd.to_numeric(P["area_outside_window_frac"], errors="coerce")
               > AREA_LOWER_BOUND_REPORTABLE).fillna(False))
    # A band that has run to its width ceiling is a pedestal, not a
    # composition window — the same failure as a component at its sigma
    # bound, and demoted the same way.
    if "band_at_width_bound" in P:
        P["reliable"] = P["reliable"] & ~(
            P["band_at_width_bound"].fillna(False).astype(bool))
    P["too_narrow"] = False
    P["area_collapsed"] = False
    base = P["reliable"]
    if int(base.sum()) >= 20:
        typ_s = P.loc[base, "sigma"].median()
        # `width_floor_factor` is the MULTIPLE below the median width that
        # counts as narrow, so 4.0 means "narrower than a quarter of the
        # median" — the same test the fallback below always applied. The
        # previous reading inverted it (a plausible-looking 0.25 flagged
        # everything narrower than 4x the median, i.e. nearly everything)
        # AND, unlike the flag-only fallback, struck them from `reliable`.
        # Both branches now flag only; enforcement is a separate decision.
        _wf = width_floor_factor if width_floor_factor > 0 else 4.0
        if np.isfinite(typ_s):
            P["too_narrow"] = base & (P["sigma"] * _wf < typ_s)
        typ_a = P.loc[base, "amplitude_area"].median()
        if np.isfinite(typ_a) and min_area_fraction > 0:
            collapsed = P["amplitude_area"].abs() < typ_a * min_area_fraction
            P["area_collapsed"] = base & collapsed
            P["reliable"] = P["reliable"] & ~collapsed
    # --- two statements about the MEASUREMENT, not about the peak ---------
    #
    # `_flag_reliable_peaks` tests only the UPPER sigma bound, and its
    # docstring gives the reason: a genuine sharp peak sits near the lower
    # one by design, so demoting on it would delete real data. That reasoning
    # is right and it is not a reason to stay SILENT. A component exactly on
    # its width floor has a width of "one histogram bin or less" — the number
    # printed with a standard error beside it is the bound, not a fit — and
    # on LTO 43-60% of components sat there while every one of them was
    # marked reliable.
    #
    # Flagged, not enforced, like `too_narrow` and `band_at_width_bound`.
    # BOTH FLANKS. This tested `sigma` alone, which is right for a symmetric
    # fit and wrong for a split one: on LTO the low-voltage flank sits on the
    # floor in the discharge direction and the HIGH-voltage flank does on the
    # charge direction. 44 of 61 components had a flank on the floor and 26
    # were flagged; the 19 charge cases were silent. Item 32.
    P["at_sigma_floor"] = False
    if "sigma_min_fitted" in P:
        _floor = pd.to_numeric(P["sigma_min_fitted"], errors="coerce")
        _lo = (pd.to_numeric(P["sigma"], errors="coerce")
               <= _floor * (1.0 + SIGMA_BOUND_PROXIMITY)).fillna(False)
        _hi = pd.Series(False, index=P.index)
        if "sigma_r" in P:
            _hi = (pd.to_numeric(P["sigma_r"], errors="coerce")
                   <= _floor * (1.0 + SIGMA_BOUND_PROXIMITY)).fillna(False)
        P["at_sigma_floor"] = _lo | _hi
        # ...and WHICH flank, because they mean different things: the low side
        # on the floor is a leading edge steeper than the sampling, the high
        # side is a trailing edge.
        P["sigma_floor_side"] = np.where(
            _lo & _hi, "both", np.where(_lo, "low", np.where(_hi, "high", "")))
    # ITEM 36. A component at its UPPER width bound is acting as baseline and
    # `_flag_reliable_peaks` already demotes it — but nothing NAMED it, so a
    # reader saw `reliable = False` with no reason. 25 of 143 on NMC111 and
    # 48-86 per cell on NNM.
    #
    # BOTH FLANKS, 1.9.0.58 — the same correction `at_sigma_floor` above
    # received as Item 32, made on the ceiling instead of the floor, and made
    # now because per-component asymmetry is what gives the two flanks
    # different widths. Testing `sigma` alone was complete while the
    # lineshape was symmetric; with a split one it goes silent on exactly the
    # components this flag exists to catch. NNM's terminal component reaches
    # the ceiling on its HIGH flank, which is the flank that was not tested.
    P["at_sigma_max"] = False
    P["sigma_max_side"] = ""
    if "sigma_max_fitted" in P:
        _ceil = pd.to_numeric(P["sigma_max_fitted"], errors="coerce")
        _clo = (pd.to_numeric(P["sigma"], errors="coerce")
                >= _ceil * SIGMA_BOUND_PROXIMITY_COMPLEMENT).fillna(False)
        _chi = pd.Series(False, index=P.index)
        if "sigma_r" in P:
            _chi = (pd.to_numeric(P["sigma_r"], errors="coerce")
                    >= _ceil * SIGMA_BOUND_PROXIMITY_COMPLEMENT).fillna(False)
        P["at_sigma_max"] = _clo | _chi
        # WHICH flank, on the same reasoning as `sigma_floor_side`: the high
        # side at the ceiling is a trailing edge the window did not contain,
        # the low side is a leading one.
        P["sigma_max_side"] = np.where(
            _clo & _chi, "both",
            np.where(_clo, "low", np.where(_chi, "high", "")))
    # And the width that no sampling could have measured. LTO cell C's
    # discharge peak is 2.6 mV wide sampled at 1 mV — three points across the
    # full width at half maximum.
    P["width_undersampled"] = False
    if "sample_mV" in P and "fwhm" in P:
        _samp = pd.to_numeric(P["sample_mV"], errors="coerce") / 1000.0
        P["width_undersampled"] = (
            pd.to_numeric(P["fwhm"], errors="coerce")
            < _samp * WIDTH_MIN_SAMPLES).fillna(False)

    # ...and the per-half-cycle test, which runs whatever the above decided.
    # See EMPTY_COMPONENT_FRACTION.
    P["area_empty"] = False
    if EMPTY_COMPONENT_FRACTION > 0 and {"cycle", "step"} <= set(P.columns):
        _biggest = (P.groupby(["cycle", "step"])["amplitude_area"]
                    .transform(lambda a: a.abs().max()))
        _empty = (P["amplitude_area"].abs()
                  < _biggest * EMPTY_COMPONENT_FRACTION)
        _empty = _empty.fillna(False) & np.isfinite(_biggest) & (_biggest > 0)
        P["area_empty"] = _empty
        P["reliable"] = P["reliable"] & ~_empty

    # --- WHY a component is not reliable, and a separate axis for whether
    # --- its area is determined at all.
    #
    # `reliable` is one boolean carrying several different statements, and it
    # is the most-read field in the table. Measured across the three
    # chemistries, what actually sets it False is:
    #
    #                     NNM (1903 not reliable)   NMC (124)   LTO (1)
    #   band_at_width_bound        59.0%              45.2%        0%
    #   at_sigma_max               55.5%              30.6%      100%
    #   is_inferred                37.9%              46.8%        0%
    #   area_empty                  7.1%              14.5%        0%
    #
    # Those are not one statement. "This component sits on a bound" says the
    # width is a constraint rather than a fit — and, being knife-edge, is not
    # even reproducible across machines. "This component was never detected in
    # this half-cycle" says the centre is another cycle's answer. "The fit put
    # a thousandth of the charge here" says the feature is not there. A reader
    # given one `False` cannot tell which, and they call for different actions.
    #
    # AND THE DETERMINACY OF THE AREA IS A DIFFERENT AXIS AGAIN. It reaches
    # `reliable` only when ENFORCE_AREA_DETERMINACY is on, which it is not —
    # so `reliable = True` today says nothing whatever about whether the area
    # has a quotable uncertainty. On NNM 83% and on NMC 89% of components have
    # NO standard error on their area at all. A reader who takes `reliable`
    # to mean "I can quote this area with its error" is wrong about nearly
    # every component of a layered oxide, and nothing on the page said so.
    #
    # So: `reliable` keeps EXACTLY its present value — this changes what the
    # tool says, not what it does — and two new columns say what it means.
    _reason = pd.Series("ok", index=P.index, dtype=object)
    _B = lambda c: (P[c].fillna(False).astype(bool) if c in P
                    else pd.Series(False, index=P.index))
    # Named here rather than in the ladder so the threshold has one home and
    # the report can quote the same number. Flag only: this does not demote
    # anything, it says why something already demoted is not usable.
    P["area_mostly_outside_window"] = (
        _B("area_is_lower_bound")
        & (pd.to_numeric(P.get("area_outside_window_frac"), errors="coerce")
           > AREA_LOWER_BOUND_REPORTABLE).fillna(False))
    # Most specific first: a component that was never detected here is not a
    # measurement whatever else is true of it.
    for _flag, _word in (
            # FIRST: this one is a statement about the whole half-cycle, and
            # it survives whatever else is true of the component.
            ("half_cycle_anomalous",
             "this half-cycle passed more charge than the material can hold"),
            ("is_inferred", "not detected in this half-cycle"),
            ("area_empty", "the fit put almost no charge here"),
            ("area_collapsed", "area collapsed against the reference cycle"),
            ("band_at_width_bound", "band at its width ceiling"),
            # ABOVE `at_sigma_max`, DELIBERATELY. A component sitting at the
            # window edge runs BOTH flanks to the sigma bound because there
            # is no data on one side to stop it, so it earns both flags — and
            # naming the ceiling sends the reader to the width when the cause
            # is the window. On NNM 1.9.0.72, 214 of the 245 components whose
            # reason read "width at its ceiling" had more than a tenth of
            # their area outside the window; 81% of everything at the sigma
            # bound sits within one sigma_max of an edge, against 27% of the
            # dataset. Raising the bound does not help these — see
            # `quality.MECHANISM_MODEL` for why this is NOT the band-ceiling
            # problem wearing a different hat.
            ("area_mostly_outside_window",
             "most of this component lies outside the measured window"),
            ("at_sigma_max", "width at its ceiling"),
    ):
        _reason = _reason.mask((_reason == "ok") & _B(_flag) & ~P["reliable"],
                               _word)
    _reason = _reason.mask((_reason == "ok") & ~P["reliable"],
                           "width at its ceiling or area not finite")
    _reason = _reason.mask(P["reliable"], "ok")
    P["reliability_reason"] = _reason
    # Orthogonal to `reliable`, and printed beside it rather than folded in.
    P["area_determinacy"] = np.where(
        _B("area_stderr_missing"), "not estimated",
        np.where((P["area_stderr_ratio"] > AREA_STDERR_RATIO_MAX).fillna(False),
                 "imprecise", "determined"))
    return P


# =============================================================================
# 2. TRACKING  (Module 5)
# =============================================================================

@dataclass
class Tracking:
    """Peak histories for one dataset, and what became of each peak."""
    name: str
    tracked_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict = field(default_factory=dict)
    reference_peaks: dict = field(default_factory=dict)
    reference_cycle: int = 0
    ghost_peaks: set = field(default_factory=set)
    tolerance_mV: float = TRACKING_TOLERANCE_MV
    # The cell's own discharge-capacity retention, once `flag_area_growth` has
    # been given it. NaN until then, and NaN is not 100%.
    capacity_retention_pct: float = float("nan")
    # Median `quality.integral_fidelity` per step, once `flag_low_fidelity`
    # has been given it. See INTEGRAL_FIDELITY_FLOOR.
    integral_fidelity: dict = field(default_factory=dict)

    @property
    def cell_discontinuity(self):
        """Primary, non-ghost peaks that failed after being established."""
        return sum(1 for s in self.summary.values()
                   if not s["is_shoulder"] and not s["is_ghost"]
                   and s["discontinuity_is_failure"])

    @property
    def transient_losses(self):
        return sum(1 for s in self.summary.values()
                   if not s["is_shoulder"] and not s["is_ghost"]
                   and s["discontinuity_events"]
                   and not s["discontinuity_is_failure"])

    def summary_text(self):
        L = [entry("tracking against", f"cycle {self.reference_cycle}",
                   f"±{self.tolerance_mV:.0f} mV per cycle, position "
                   f"carried forward"
                   if TRACK_AGAINST_PREVIOUS_CYCLE
                   else f"±{self.tolerance_mV:.0f} mV of the reference")]
        if self.ghost_peaks:
            L.append("  ghost peaks (reference area ~ 0, retention not "
                     "reported): "
                     + ", ".join(f"{s} peak {p}"
                                 for s, p in sorted(self.ghost_peaks)))
        _far = [s for s in self.summary.values()
                if s.get("drift_exceeds_tolerance")]
        if _far:
            L.append(verdict("caution", f"{len(_far)} peak(s) jumped further "
                             f"in ONE cycle than the "
                             f"±{self.tolerance_mV:.0f} mV matching window"))
            L.append(bullet("A feature does not move that far between "
                            "consecutive cycles, so on either side of that "
                            "step the history is probably not one feature. "
                            "Total drift is not the test — the tracker "
                            "follows a peak as far as it goes — a single-"
                            "cycle jump is."))
        for _st, _f in sorted(self.integral_fidelity.items()):
            if _f < INTEGRAL_FIDELITY_FLOOR:
                L.append(f"  *** {_st}: the dQ/dV curve accounts for only "
                         f"{100 * _f:.0f}% of the delivered capacity. Peak "
                         f"AREAS on this step are not capacities and their "
                         f"trends are not retention; peak POSITIONS and "
                         f"drift are unaffected. ***")
            elif _f > INTEGRAL_FIDELITY_CEILING:
                L.append(f"  *** {_st}: the dQ/dV curve carries "
                         f"{100 * _f:.0f}% of the delivered capacity — more "
                         f"charge than the cell passed. Peak AREAS on this "
                         f"step include charge that was not measured; peak "
                         f"POSITIONS and drift are unaffected. ***")
        if np.isfinite(self.capacity_retention_pct):
            _bound = self.capacity_retention_pct + AREA_GROWTH_HEADROOM_PCT
            L.append(f"  capacity retention "
                     f"{self.capacity_retention_pct:.0f}% — a peak above "
                     f"{_bound:.0f}% area retention is flagged, not deleted")
        L.append(f"  {'step':<10}{'ref V':>8}{'tracked':>9}{'rate':>7}"
                 f"{'drift':>10}{'retention':>11}  note")
        for k in sorted(self.summary):
            s = self.summary[k]
            note = []
            if s["is_shoulder"]:
                note.append("shoulder")
            if s["is_truncated"]:
                note.append("truncated")
            if s["is_ghost"]:
                note.append("ghost")
            if s.get("drift_exceeds_tolerance"):
                note.append(f"JUMPED {s['max_centre_step_mV']:.0f} mV IN ONE "
                            f"CYCLE — BEYOND THE "
                            f"±{self.tolerance_mV:.0f} mV MATCHER")
            if s.get("area_retention_trustworthy") is False:
                note.append("AREA NOT A CAPACITY")
            if s.get("area_exceeds_capacity"):
                note.append(f"AREA > CAPACITY by "
                            f"{s['area_growth_excess_pct']:.0f} pts")
            if s["discontinuity_events"]:
                note.append(
                    f"{s['discontinuity_events']} discontinuit"
                    f"{'y' if s['discontinuity_events'] == 1 else 'ies'}"
                    + (" (FAILURE, established first)"
                       if s["discontinuity_is_failure"]
                       else " (transient, never established)"))
            drift = ("     —" if not np.isfinite(s["voltage_drift_mV"])
                     else f"{s['voltage_drift_mV']:+6.0f}")
            _pc = s.get("voltage_drift_mV_per_cycle", np.nan)
            if np.isfinite(_pc) and s.get("voltage_drift_first_cycle"):
                note.append(f"{_pc:+.2f} mV/cycle over c"
                            f"{s['voltage_drift_first_cycle']}-"
                            f"{s['voltage_drift_last_cycle']}")
            ret = ("      —" if not np.isfinite(s["area_retention_pct"])
                   else f"{s['area_retention_pct']:6.0f}%")
            L.append(f"  {s['step']:<10}{s['reference_voltage']:>8.3f}"
                     f"{s['tracked']:>9}{s['tracking_rate']:>7.0%}"
                     f"{drift:>8} mV{ret:>11}  {'; '.join(note)}")
        return "\n".join(L)


def _assign_components(params_df, ref_list, step_cycles, step, tolerance_V,
                       reference_cycle=None, chain=None):
    """
    One-to-one assignment of fitted components to reference peaks, per cycle.

    Ported from 1.8.7 Module 5. Each component is claimed once, by optimal
    assignment, with PRIMARIES matched before shoulders — so when a primary and
    its shoulder merge into one component it is the shoulder that is recorded
    as disappeared, and the primary's centre history stays continuous. The
    earlier per-peak nearest-neighbour scheme let two reference ids take the
    same component and orphan a real one.

    WHAT THE WINDOW IS MEASURED FROM (1.9.0.23)
    -------------------------------------------
    Until now every cycle was matched against the REFERENCE CYCLE's fitted
    positions, so `tolerance_V` was a LIFETIME allowance: a peak drifting
    2 mV/cycle used it up in forty cycles and was then matched to whichever
    component was nearest, which on a crowded curve is a different peak. That
    made the tolerance a ceiling on the drift the tracker could measure —
    which is the one quantity ICA is usually read for.

    The expectation now MOVES WITH THE PEAK: each reference id carries a
    running centre, initialised to its reference-cycle position and updated to
    the fitted centre every time it is matched. `tolerance_V` becomes a
    PER-CYCLE step limit — how far a feature may move between consecutive
    cycles and still be the same feature — which is what the number was always
    a sensible size for, and the lifetime drift is then unbounded.

    Two properties this has to keep, and does:

    * A peak that is MISSING for some cycles keeps its last matched position
      as the expectation rather than reverting to the reference. Reverting
      would make a gap into a jump and, on the far side of a long gap, hand
      the peak to a neighbour.
    * Matching walks OUTWARD FROM THE REFERENCE CYCLE in both directions,
      never from cycle 1. The reference is the measured end of formation, and
      the cycles before it are where the curve moves fastest; chaining
      forwards from cycle 1 would arrive at the reference having already
      wandered, and the reference cycle is the one position we are certain of.

    The cost of chaining is that an error made in one cycle is inherited by
    the next. That is bounded here by the per-cycle window and reported: a
    single-cycle step larger than the window is a discontinuity, is counted as
    one, and the summary says so.

    `chain=False` restores the fixed-reference behaviour, which is how the
    two are compared on the synthetic ground truth.
    """
    if chain is None:
        chain = TRACK_AGAINST_PREVIOUS_CYCLE
    assignment = {}

    ordered = list(step_cycles)
    if chain and reference_cycle is not None and ordered:
        # Outward from the reference, each direction starting fresh from the
        # reference-cycle positions.
        passes = [[c for c in ordered if c >= reference_cycle],
                  [c for c in ordered if c < reference_cycle][::-1]]
    else:
        passes = [ordered]

    for _pass in passes:
        # The running expectation. Fixed for the whole pass when chaining is
        # off, which reproduces the pre-1.9.0.23 assignment exactly.
        expect = {rid: rv for rid, rv, _sh in ref_list}
        for cyc in _pass:
            cps = params_df[(params_df["cycle"] == cyc) &
                            (params_df["step"] == step)]
            if cps.empty:
                continue
            centres = cps["centre"].values
            used = set()
            for shoulder_pass in (False, True):
                ids = [(rid, expect[rid])
                       for rid, _rv, sh in ref_list if sh == shoulder_pass]
                free = [ci for ci in range(len(centres)) if ci not in used]
                if not ids or not free:
                    continue
                BIG = 1e6
                cost = np.full((len(ids), len(free)), BIG)
                for i, (_rid, rv) in enumerate(ids):
                    for j, ci in enumerate(free):
                        d = abs(centres[ci] - rv)
                        if d <= tolerance_V:
                            cost[i, j] = d
                ri, cj = linear_sum_assignment(cost)
                for i, j in zip(ri, cj):
                    if cost[i, j] >= BIG:
                        continue
                    assignment[(cyc, ids[i][0])] = free[j]
                    used.add(free[j])
                    if chain:
                        # Carry the position forward. An unmatched id keeps
                        # the expectation it already had.
                        expect[ids[i][0]] = float(centres[free[j]])
    return assignment


# A tracked peak's area is a share of the charge the cell delivered. If the
# cell holds 99% of its capacity, no redox feature inside it can have grown to
# 136% — and LTO cell A reported exactly that at 1.9.0.8, from a shoulder that
# had inflated into its parent. R2 could not see it (0.999), the standard
# errors could not see it (they are conditional on the model), and nothing in
# the pipeline asked the one question thermodynamics answers for free.
#
# So ask it. The bound is capacity retention plus headroom, because the two
# are not the same quantity: capacity is the whole cell, a peak is one process
# in it, and a process can legitimately take a larger SHARE as another fades.
# The headroom is what separates that from a fitting artefact. 15 points is
# generous — a genuine redistribution of that size across ten cycles at C/10
# would itself be a result — and it still catches the 136%.
AREA_GROWTH_HEADROOM_PCT = 15.0

# --- when is a peak AREA a capacity at all? ------------------------------
# A fitted peak area is a share of the charge that appears in the dQ/dV
# CURVE, and that curve is only a measurement of the cell's charge to the
# extent that its integral recovers the cell's capacity. `integral_fidelity`
# is exactly that ratio, it has been measured on every half-cycle since
# 1.9.0, and until now it gated nothing.
#
# It should. On the LTO triplicate the CYCLER'S OWN dQ/dV column integrates
# to 43-68% of the CYCLER'S OWN delivered capacity, and the shortfall swings
# by ten points from cycle to cycle. That is not a fitting problem and no
# choice of model fixes it: on a flat two-phase plateau the true dV between
# consecutive records falls below the instrument's voltage resolution, so
# most of the charge is delivered in a voltage interval the derivative cannot
# resolve. Fit that curve as well as you like and its peak area is the area
# of the fraction that survived, which is why LTO cell A reports 141% "area
# retention" on a cell holding 98% of its capacity.
#
# So: below this floor, area-based TRENDS are marked untrustworthy and say
# why. The areas themselves are kept — they are still the best description of
# the curve that was measured — and the peak POSITIONS are unaffected, which
# matters, because voltage drift is the quantity ICA is usually read for and
# it does not depend on the integral at all.
#
# 0.80 is the floor: published ICA practice treats the differential as an
# accounting of the capacity, and a curve missing a fifth of the charge is
# not that. A cell with a constant-voltage hold sits legitimately below 1.0,
# since CV charge passes at one voltage and cannot appear in a dV integral —
# which is a reason to read the floor as "areas are not capacity here",
# not as "the data are bad".
INTEGRAL_FIDELITY_FLOOR = 0.80

# ...AND A CEILING (1.9.0.23). `quality.integral_fidelity`'s own docstring
# says "fabricated area shows up as an excess, lost features as a deficit",
# and until now only the deficit was tested. The failure it names is the more
# damning of the two: a deficit is charge the curve never saw, an excess is
# charge the cell never delivered, and there is no benign mechanism that
# invents a fifth of a cell's capacity. The 4.199 V constant-voltage artefact
# — the one the docstring holds up as what this test would have caught on
# sight — is an EXCESS, and the gate as written would have waved it through.
#
# Where the number comes from. Finite differencing on a quantised voltage
# record legitimately overshoots, because a small dV in the denominator
# inflates the quotient before the trapezoid puts it back. Measured on the
# synthetic cases at 0.1-5 mV voltage resolution, that overshoot reaches
# 1.047 and no further, so a ceiling anywhere below about 1.10 would fire on
# ordinary data. 1.20 mirrors the floor's depth and sits well clear of the
# mechanism.
INTEGRAL_FIDELITY_CEILING = 1.20

# WHAT THE BAND ACTUALLY GUARDS, WHICH DEPENDS ON HOW dQ/dV WAS COMPUTED.
# The chemistry-dependence argued above is real, but it belongs to the
# FINITE-DIFFERENCE path. Measured on the synthetic cases, differencing a
# voltage record quantised at 1 mV recovers:
#
#     two_phase       (sigma   8 mV)   0.851      5% of records lost to dV=0
#     tracking        (sigma  12 mV)   1.031      none lost
#     mixed           (sigma  28 mV)   1.007
#     solid_solution  (sigma 220 mV)   1.010
#
# — and at 2 mV the two-phase curve falls to 0.500. So fidelity there is set
# by how flat the plateau is against the instrument's voltage resolution, not
# by chemistry as such; chemistry only enters through flatness, and a
# two-phase plateau is the worst case there is. 0.80 is a defensible line for
# that path.
#
# Ratatosk's DEFAULT path is the derivative-free histogram, which bins charge
# by voltage and divides by the bin width, so its integral returns the
# delivered capacity BY CONSTRUCTION: the same sweep gives 1.000 at every
# resolution for every case, and the LTO triplicate measures 0.99-1.00 on
# every dataset. On that path a deficit no longer means "the derivative could
# not resolve the plateau" — it can only mean charge that fell OUTSIDE the
# analysed voltage window, a constant-voltage hold being the usual reason.
# That is still worth catching and the floor still catches it. It is a
# different claim from the one this comment used to make, and the verdict
# text now says which.


def cycle_column(t, wanted):
    """Find a cycling-table column by bare name, then by cell-prefixed name.

    The per-cell table uses the bare name (`Discharge_mAh_g`); the combined
    export prefixes it with the cell id. A COMBINED table carrying more than
    one cell returns None rather than the first match, because taking the
    first would put cell A's capacities in cell C's report.
    """
    if t is None or getattr(t, "empty", True):
        return None
    for w in wanted:
        if w in t.columns:
            return w
    for w in wanted:
        hit = [c for c in t.columns if c.endswith("_" + w)]
        if len(hit) == 1:
            return hit[0]
        if len(hit) > 1:
            return None
    return None


def capacity_retention_pct(cycle_table, reference_cycle=None):
    """
    Delivered discharge capacity at the end as a percentage of the reference.

    Both ends are a median of up to three cycles, matching how `track_peaks`
    measures a peak's area retention — the two numbers are only comparable if
    they are computed the same way. Incomplete cycles are excluded. Returns
    NaN when the table cannot answer, which is not the same as 100%.
    """
    t = cycle_table
    if t is None or getattr(t, "empty", True) or "Cycle" not in t.columns:
        return np.nan
    # FROM THE ION TABLE, not typed out. Three separate tuples listed Li and
    # Na only, so a third working ion would have stopped matching in silence.
    dcol = cycle_column(t, CAPACITY_COLUMN_ALIASES)
    if dcol is None:
        return np.nan
    icol = cycle_column(t, ("Incomplete",))
    good = t[t[icol] != True] if icol else t              # noqa: E712
    good = good[good["Cycle"].notna()].sort_values("Cycle")
    if reference_cycle is not None:
        good = good[good["Cycle"] >= float(reference_cycle)]
    d = pd.to_numeric(good[dcol], errors="coerce").dropna()
    d = d[d > 0]
    if len(d) < 2:
        return np.nan
    a_ref = float(d.head(3).median())
    a_end = float(d.tail(3).median())
    return (a_end / a_ref * 100.0) if a_ref > 0 else np.nan


def flag_low_fidelity(tracking, fidelity, *, floor=INTEGRAL_FIDELITY_FLOOR,
                      ceiling=INTEGRAL_FIDELITY_CEILING):
    """
    Mark a dataset's area trends untrustworthy where the dQ/dV curve does not
    account for the cell's charge. See INTEGRAL_FIDELITY_FLOOR.

    `fidelity` is `{(cycle, step): ratio}` — what `quality.integral_fidelity`
    returns per half-cycle — or a single float. The test is applied PER STEP,
    because a half-cycle with a constant-voltage hold and one without behave
    quite differently and a dataset average would hide it.

    Nothing is deleted. `area_retention_trustworthy` goes False and
    `area_withheld_reason` says why, so a reader of the summary sees the
    number and the reason for doubting it side by side.
    """
    if isinstance(fidelity, (int, float)):
        per_step = {"Charge": float(fidelity), "Discharge": float(fidelity)}
    else:
        acc = {}
        for (_cyc, step), f in (fidelity or {}).items():
            if np.isfinite(f):
                acc.setdefault(str(step), []).append(float(f))
        per_step = {k: float(np.median(v)) for k, v in acc.items() if v}
    tracking.integral_fidelity = dict(per_step)
    marked = []
    for k, sm in tracking.summary.items():
        f = per_step.get(sm.get("step"), np.nan)
        sm["integral_fidelity"] = f
        # BOTH SIDES. A curve carrying more charge than the cell delivered
        # is not a better measurement than one carrying less; it is a worse
        # one, because nothing physical produces it.
        _short = bool(np.isfinite(f) and f < float(floor))
        _over = bool(np.isfinite(f) and f > float(ceiling))
        bad = _short or _over
        sm["area_retention_trustworthy"] = not bad
        if _short:
            sm["area_withheld_reason"] = (
                f"the dQ/dV curve accounts for only {100 * f:.0f}% of this "
                f"step's delivered capacity, so a peak area is not a "
                f"capacity")
        elif _over:
            sm["area_withheld_reason"] = (
                f"the dQ/dV curve carries {100 * f:.0f}% of this step's "
                f"delivered capacity — more charge than the cell passed, so "
                f"some of this area was not measured")
        else:
            sm["area_withheld_reason"] = ""
        if bad:
            marked.append(k)
    return marked


def flag_area_growth(tracking, capacity_retention,
                     *, headroom_pct=AREA_GROWTH_HEADROOM_PCT):
    """
    Mark tracked peaks whose area grew by more than the cell's capacity allows.

    Writes `area_exceeds_capacity` and `area_growth_excess_pct` into every
    entry of `tracking.summary` and records the bound on the `Tracking`, so
    the flag travels with the object rather than being recomputed by whoever
    prints it. Returns the list of offending keys.

    Nothing is deleted and no fit is rejected: this is a WARNING that the
    decomposition in that step is not supported by the cell's own charge
    balance, and the right response is to look at the fit, not to drop it.
    """
    tracking.capacity_retention_pct = float(capacity_retention)
    flagged = []
    if not np.isfinite(capacity_retention):
        for s in tracking.summary.values():
            s["area_exceeds_capacity"] = False
            s["area_growth_excess_pct"] = np.nan
        return flagged
    bound = float(capacity_retention) + float(headroom_pct)
    for k, s in tracking.summary.items():
        a = s.get("area_retention_pct", np.nan)
        excess = (a - bound) if np.isfinite(a) else np.nan
        bad = bool(np.isfinite(excess) and excess > 0 and not s.get("is_ghost"))
        s["area_exceeds_capacity"] = bad
        s["area_growth_excess_pct"] = excess
        if bad:
            flagged.append(k)
    return flagged


def track_peaks(params_df, detection, *, tolerance_mV=TRACKING_TOLERANCE_MV,
                verbose=True):
    """
    Match fitted components to the reference peak list, cycle by cycle.

    Drift and retention are measured against the REFERENCE CYCLE, not cycle 1:
    a peak that behaves differently during formation would otherwise be given
    an artefactual retention. Peaks whose reference-cycle area is already
    ~zero — the fitter kept a collapsed component — are ghosts, and retention
    is not reported for them at all rather than reported as a ratio of two
    numbers that mean nothing.
    """
    tol_V = tolerance_mV / 1000.0
    ref_cycle = detection.reference_cycle
    P = params_df
    # Inferred components are NOT struck from the table here. They were, on
    # the reasoning that an inferred-and-unreliable component is worthless —
    # but `reliable` is now False for every inferred component by definition,
    # so that test discarded all of them and the seeding it was written to
    # guard could never show its work. They are tracked; the `reliable` flag
    # keeps them out of the drift and retention numbers, which is where the
    # exclusion belongs.

    nan_params = {c: np.nan for c in
                  ("centre", "centre_stderr", "amplitude_area",
                   "amplitude_stderr", "sigma", "sigma_stderr", "fraction",
                   "fraction_stderr", "fwhm", "height", "voltage_shift",
                   "sigma_r", "sigma_r_stderr", "asymmetry",
                   # Carried through so the reliability test below can ask
                   # whether a component is pinned at the bound ITS OWN fit
                   # used, not at a module constant.
                   "sigma_max_fitted",
                   # PROVENANCE, carried for the same reason.
                   # `parameters_frame` demotes every inferred component with
                   # `P["reliable"] &= ~P["is_inferred"]`, and the comment
                   # above claims that flag "keeps them out of the drift and
                   # retention numbers". It did not: `_reliable` below
                   # recomputes reliability from scratch on THIS table, and
                   # `is_inferred` was never copied into it. Measured on
                   # NMC111 cell A: 34 inferred components, all correctly
                   # unreliable in the parameter table, 31 of them reaching
                   # this table and 25 coming back reliable — feeding drift,
                   # area retention, the tracking rate and the discontinuity
                   # test, while detection printed "marked is_inferred, and
                   # excluded from headline trends".
                   "is_inferred",
                   # The fit's own verdict on itself. Reported, no longer a
                   # gate — see AREA_STDERR_RATIO_MAX.
                   "converged", "hit_iteration_cap",
                   # ...and the quantity that replaced it, carried for the
                   # same reason `is_inferred` had to be: a flag computed in
                   # `parameters_frame` and not copied here is a flag that
                   # reaches none of the trends.
                   "area_stderr_ratio", "area_stderr_missing",
                   "area_poorly_determined", "area_empty",
                   # WHY the boolean says what it says, and whether the area
                   # has an uncertainty at all. See the note where they are
                   # built: one boolean was carrying four statements.
                   "reliability_reason", "area_determinacy",
                   # SAME REASON AS `is_inferred` DIRECTLY ABOVE, and the same
                   # trap: `_reliable` recomputes reliability from scratch on
                   # THIS table, so a flag set only in `parameters_frame`
                   # reaches the parameter CSV and nothing else.
                   "half_cycle_anomalous",
                   # WHAT THE FIT DECIDED THIS COMPONENT IS. The mechanism
                   # classifier chooses whether bands are available at all;
                   # the fit then decides, per component, whether each one
                   # became a band or stayed a peak. That answer was in the
                   # fitted-parameter CSV and nowhere else — not in this
                   # table, not in the per-peak summary, not in either
                   # report — so a reader could not tell which shape had
                   # been used for the feature they were reading about.
                   "component_kind",
                   # THE AREA THE CAPACITY NUMBERS ARE NOW COMPUTED FROM.
                   # Carried for the same reason `is_inferred` and
                   # `component_kind` had to be: `CAPACITY_AREA_COLUMN`
                   # switched the attribution, the area trend and the
                   # coherence area test to `area_in_window`, and `_area`
                   # falls back to the analytic value when the column is
                   # absent — so leaving it out of this table would have
                   # silently reverted every one of them here while the
                   # parameter table used the new one. That is the fourth
                   # time a field computed upstream has failed to reach a
                   # consumer; the fallback would have made it the quietest.
                   "area_in_window",
                   # THE FLAG LIST IS NOW ACTUALLY USED. `_DERIVED_FLAG_COLS`
                   # was defined with a comment warning that "a flag computed
                   # here and never added to a list like this one silently
                   # never reaches the CSV" — and then read by nothing at
                   # all, while THIS list, the one that decides what reaches
                   # the tracked table, was maintained separately and drifted
                   # from it.
                   #
                   # `band_at_width_bound` is the cost. It is absent here, so
                   # `ATTRIBUTION_EXCLUDES` — which names it — could never
                   # exclude anything (`if col in cd.columns` was always
                   # False), and every capacity share, `total_area`,
                   # `retained_fraction` and `peakN_fraction` still counted
                   # pinned bands. The demotion landed in `parameters_frame`
                   # in 1.9.0.5x; the attribution half of the same fix did
                   # not. It is 59% of NNM's not-reliable components and 45%
                   # of NMC's.
                   *_DERIVED_FLAG_COLS,
                   "band_at_width_bound", "band_width_max_fitted",
                   "component_kind",
                   "area_outside_window_frac", "fit_truncated",
                   # ...and the flag `_reliable` reads when
                   # ENFORCE_AREA_COMPLETENESS is on. Carried for exactly the
                   # reason the comment above gives: a demotion that cannot
                   # see its own flag here is a demotion that does nothing.
                   "area_is_lower_bound", "centre_poorly_determined")}
    rows = []
    for step in ("Charge", "Discharge"):
        ref_peaks = detection.reference_peaks.get(step, pd.DataFrame())
        if ref_peaks is None or ref_peaks.empty:
            continue
        step_cycles = sorted({int(c) for (c, s) in detection.peaks if s == step})
        ref_list = [(int(r["peak_id"]), float(r["voltage"]),
                     bool(r.get("is_shoulder", False)))
                    for _, r in ref_peaks.iterrows()]
        assignment = _assign_components(
            P, ref_list, step_cycles, step, tol_V,
            reference_cycle=detection.reference_cycle)

        # NEVER FITTED IS NOT THE SAME AS DISAPPEARED.
        #
        # The roster is built from the DETECTED reference list while the fit
        # may have been given a SUBSET of it — `primaries_only`, set by the
        # mechanism classifier, drops the shoulder stack. Those features then
        # appeared in START_HERE as tracked peaks and were recorded as
        # `disappeared` in every cycle of the run: on NMC111, 3 of 7 discharge
        # features tracked in zero of 18 cycles. "Disappeared" says the peak
        # was there and went, which is a finding about the CELL; the truth
        # was that the model was never asked to fit it, which is a fact about
        # the RUN. The page was advertising features it had not fitted and
        # blaming the cell for their absence.
        #
        # Decided from the data rather than from a flag, so it is right
        # whatever the reason: a reference peak matched in NO cycle at all
        # was never fitted.
        _never = {rid for (rid, _v, _sh) in ref_list
                  if not any(assignment.get((c, rid)) is not None
                             for c in step_cycles)}

        for _, ref_pk in ref_peaks.iterrows():
            ref_v = float(ref_pk["voltage"])
            ref_id = int(ref_pk["peak_id"])
            base_meta = dict(is_shoulder=bool(ref_pk.get("is_shoulder", False)),
                             is_truncated=bool(ref_pk.get("is_truncated",
                                                          False)))
            for cycle in step_cycles:
                cp = P[(P["cycle"] == cycle) & (P["step"] == step)]
                base = dict(dataset=detection.name, cycle=cycle, step=step,
                            tracked_peak_id=ref_id, reference_voltage=ref_v,
                            r_squared=(cp["r_squared"].iloc[0]
                                       if not cp.empty else np.nan),
                            **base_meta)
                if ref_id in _never:
                    rows.append({**base, "status": "not_fitted", **nan_params})
                    continue
                if cp.empty:
                    rows.append({**base, "status": "no_fit", **nan_params})
                    continue
                idx = assignment.get((cycle, ref_id))
                if idx is None:
                    rows.append({**base, "status": "disappeared", **nan_params})
                    continue
                m = cp.iloc[idx]
                rows.append({**base, "status": "tracked",
                             **{c: m.get(c, np.nan) for c in nan_params
                                if c != "voltage_shift"},
                             "voltage_shift": m["centre"] - ref_v})

    T = pd.DataFrame(rows)
    tk = Tracking(name=detection.name, tracked_df=T,
                  reference_peaks=detection.reference_peaks,
                  reference_cycle=ref_cycle, tolerance_mV=tolerance_mV)
    if T.empty:
        return tk

    # --- reliability of a tracked point ---------------------------------
    # From the fit, per row, with the module constant only as a fallback —
    # see `_flag_reliable_peaks` for why a fixed 200 mV disabled this test on
    # every sharp and moderate profile.
    if "sigma_max_fitted" in T:
        _bound = pd.to_numeric(T["sigma_max_fitted"], errors="coerce")
        sigma_max_by_row = _bound.fillna(FIT_SIGMA_MAX_MV / 1000.0)
    else:
        sigma_max_by_row = pd.Series(FIT_SIGMA_MAX_MV / 1000.0, index=T.index)
    ref_areas, ghosts = {}, set()
    at_ref = T[(T["cycle"] == ref_cycle) & (T["status"] == "tracked")]
    for _, r in at_ref.iterrows():
        k = (r["step"], int(r["tracked_peak_id"]))
        ref_areas[k] = r["amplitude_area"]
        if pd.notna(r["amplitude_area"]) and \
                r["amplitude_area"] < GHOST_AREA_THRESHOLD:
            ghosts.add(k)

    def _reliable(row):
        if row["status"] != "tracked":
            return False
        # An inferred component is a seed carried in from a neighbouring
        # cycle because detection found nothing here. Its centre is another
        # cycle's answer; it is not a measurement of this one.
        #
        # NaN IS NOT FALSE HERE. `bool(np.nan)` is True, so reading these two
        # flags with a bare `bool()` marked every row of a frame that lacks
        # the column — the tracking table fills missing parameters with NaN —
        # as inferred and unreliable. Absent means "not stated", which for a
        # provenance flag is the same as False; only an explicit True demotes.
        _inf = row.get("is_inferred")
        if _inf is not None and not pd.isna(_inf) and bool(_inf):
            return False
        # ...and the half-cycle's own verdict, read the same NaN-safe way.
        # See the note in `parameters_frame`.
        _an = row.get("half_cycle_anomalous")
        if _an is not None and not pd.isna(_an) and bool(_an):
            return False
        # THE CONVERGENCE GATE IS GONE HERE TOO, and it had to go from both
        # places at once: the parameter table and this one were the pair the
        # module's own SIGMA_BOUND_PROXIMITY note warns about, where a
        # component can be reliable in one table and not the other. See
        # AREA_STDERR_RATIO_MAX in the constants for the measurements and for
        # what is flagged in its place.
        if ENFORCE_AREA_DETERMINACY:
            _ad = row.get("area_poorly_determined")
            if _ad is not None and not pd.isna(_ad) and bool(_ad):
                return False
        # The completeness half of the same open question, NaN-safe like the
        # rest. Applied here AND in `parameters_frame`; see the note there.
        if ENFORCE_AREA_COMPLETENESS:
            _lb = row.get("area_is_lower_bound")
            _of = row.get("area_outside_window_frac")
            if (_lb is not None and not pd.isna(_lb) and bool(_lb)
                    and _of is not None and not pd.isna(_of)
                    and float(_of) > AREA_LOWER_BOUND_REPORTABLE):
                return False
        # NOT behind ENFORCE_AREA_DETERMINACY. "The error on this area is too
        # large to quote" is a judgement about a measurement and is the open
        # decision; "this component carries a thousandth of the charge its
        # neighbour does" is the fit saying the feature is not there, and
        # there is nothing to decide. Struck in `parameters_frame` too, and it
        # has to be struck in BOTH — that is the pairing this module's own
        # SIGMA_BOUND_PROXIMITY note warns about.
        _ae = row.get("area_empty")
        if _ae is not None and not pd.isna(_ae) and bool(_ae):
            return False
        k = (row["step"], int(row["tracked_peak_id"]))
        if k in ghosts:
            return False
        # A BAND AT ITS WIDTH CEILING IS A PEDESTAL, and `parameters_frame`
        # has demoted one since 1.9.0.5x. This table could not, because the
        # column never travelled — so the same component was `reliable` here
        # and not there, and this is the table that feeds drift, area
        # retention and the tracking rate.
        _bw = row.get("band_at_width_bound")
        if _bw is not None and not pd.isna(_bw) and bool(_bw):
            return False
        _b = sigma_max_by_row.get(row.name, FIT_SIGMA_MAX_MV / 1000.0)
        # BOTH FLANKS. `parameters_frame` was corrected in 1.9.0.58 — a
        # component whose HIGH flank is pinned at the ceiling is what a
        # terminal feature does — and this copy of the test was not, so a
        # component could be reliable in one table and not the other. That is
        # precisely the pairing this module's own SIGMA_BOUND_PROXIMITY note
        # warns about.
        _sr = row.get("sigma_r")
        _pin = SIGMA_BOUND_PROXIMITY_COMPLEMENT * _b
        if (pd.notna(row["sigma"]) and row["sigma"] >= _pin):
            return False
        if _sr is not None and pd.notna(_sr) and float(_sr) >= _pin:
            return False
        ra = ref_areas.get(k, np.nan)
        if (pd.notna(ra) and ra > 0 and pd.notna(row["amplitude_area"])
                and row["amplitude_area"] < _AREA_COLLAPSE_FRAC * ra):
            return False
        return True

    T["reliable"] = T.apply(_reliable, axis=1)
    tk.ghost_peaks = ghosts

    # --- per-peak summary ------------------------------------------------
    def _kinds(ts):
        if ts is None or "component_kind" not in getattr(ts, "columns", []):
            return []
        return [str(k) for k in ts["component_kind"].dropna().tolist() if k]

    def _modal_kind(ts):
        ks = _kinds(ts)
        if not ks:
            return None
        return max(set(ks), key=ks.count)

    def _kind_is_mixed(ts):
        return len(set(_kinds(ts))) > 1

    summary = {}
    for step in ("Charge", "Discharge"):
        sdf = T[T["step"] == step]
        if sdf.empty:
            continue
        for tid in sorted(sdf["tracked_peak_id"].unique()):
            pk = sdf[sdf["tracked_peak_id"] == tid].copy()
            n_trk = int((pk["status"] == "tracked").sum())
            n_dis = int((pk["status"] == "disappeared").sum())
            n_nf = int((pk["status"] == "no_fit").sum())
            # Detected, rostered, and never given to the model. See the
            # `_never` note in `track_peaks`.
            n_never = int((pk["status"] == "not_fitted").sum())
            n_tot = len(pk)
            key = (step, int(tid))
            is_ghost = key in ghosts
            ts = pk[pk["status"] == "tracked"].sort_values("cycle")
            ref_row = ts[ts["cycle"] == ref_cycle]
            v_drift = a_ret = np.nan
            _vend_cycle = _vstart_cycle = None
            _span = 0
            if len(ts) >= 2 and not ref_row.empty:
                # FROM THE REFERENCE CYCLE ONWARD. `ts` is every tracked
                # cycle, so head(3) was cycles 1-3 — formation, on almost
                # every dataset — and the retention this function documents
                # as measured against the reference cycle was measured
                # against the wrong end of the run. `ref_row` existed for
                # exactly this and was reachable only as a fallback.
                rel = ts[ts["reliable"]] if "reliable" in ts else ts
                rel = rel[rel["cycle"] >= ref_cycle]
                # BOTH ENDS ARE A MEDIAN OF UP TO THREE CYCLES, not two
                # single points. A centre is recovered with a few mV of
                # scatter, so a difference of two single cycles carries
                # sqrt(2) times that scatter — on the synthetic 50-cycle
                # series, 90 mV of true drift came back as 79 mV with the
                # single-point endpoints and no bias in the centres
                # themselves. It was noise in the two chosen cycles, and the
                # same median-of-three the area retention below already uses
                # removes it. `voltage_drift_first_cycle` and
                # `voltage_drift_last_cycle` name the interval it was
                # measured over, which is what a reader needs in order to
                # turn it into mV per cycle.
                _vs = rel["voltage_shift"].dropna()
                if len(_vs) >= 2:
                    _tail = _vs.tail(DRIFT_ENDPOINT_CYCLES)
                    _head = _vs.head(DRIFT_ENDPOINT_CYCLES)
                    _vend = float(_tail.median())
                    _vstart = float(_head.median())
                    # The MIDDLE cycle of each window, because a median of
                    # three sits at the middle one. Naming the outermost
                    # cycles instead overstates the interval by two cycles,
                    # and anyone dividing by it to get mV per cycle inherits
                    # that error.
                    _vend_cycle = int(np.median(
                        rel.loc[_tail.index, "cycle"].values))
                    _vstart_cycle = int(np.median(
                        rel.loc[_head.index, "cycle"].values))
                else:
                    _vend = ts["voltage_shift"].iloc[-1]
                    _vstart = ref_row["voltage_shift"].iloc[0]
                    _vend_cycle = int(ts["cycle"].iloc[-1])
                    _vstart_cycle = int(ref_row["cycle"].iloc[0])
                v_drift = (_vend - _vstart) * 1000
                # Per cycle, which is the quantity that can be compared
                # between cells of different lengths — and the one published
                # peak-tracking work quotes.
                _span = _vend_cycle - _vstart_cycle
                if not is_ghost:
                    _ra = _area(rel)
                    b = _ra.head(3).dropna()
                    e = _ra.tail(3).dropna()
                    a_ref = (b.median() if len(b)
                             else float(_area(ref_row).iloc[0]))
                    a_last = (e.median() if len(e)
                              else float(_area(ts).iloc[-1]))
                    a_ret = (a_last / a_ref * 100
                             if (pd.notna(a_ref) and a_ref > 0) else np.nan)
            # How far this peak ever got from its reference position. Under
            # a DRIFTING reference that is no longer a warning — it is the
            # measurement, and a peak that moves 200 mV over 100 cycles is
            # now tracked rather than handed to a neighbour at 80. What still
            # breaks an identity is a single-cycle STEP wider than the
            # matching window, so that is what the flag now tests.
            _shift_mV = (pk["voltage_shift"].abs().max() * 1000
                         if "voltage_shift" in pk else np.nan)
            n_ev, first_ev, max_step, established = _discontinuity_events(pk)
            _gate_mV = (max_step if TRACK_AGAINST_PREVIOUS_CYCLE
                        else _shift_mV)
            _beyond = bool(FLAG_DRIFT_BEYOND_TOLERANCE
                           and np.isfinite(_gate_mV)
                           and _gate_mV > tolerance_mV)
            summary[f"{step}_peak{tid}"] = dict(
                step=step, peak_id=int(tid), reference_voltage=
                float(pk["reference_voltage"].iloc[0]),
                is_shoulder=bool(pk["is_shoulder"].iloc[0]),
                is_truncated=bool(pk["is_truncated"].iloc[0]),
                is_ghost=is_ghost, tracked=n_trk, disappeared=n_dis,
                no_fit=n_nf, not_fitted=n_never,
                never_fitted=bool(n_never and n_never == n_tot),
                total_cycles=n_tot,
                tracking_rate=(n_trk / n_tot if n_tot else 0.0),
                max_shift_mV=_shift_mV,
                drift_exceeds_tolerance=_beyond,
                voltage_drift_mV=v_drift,
                voltage_drift_first_cycle=_vstart_cycle,
                voltage_drift_last_cycle=_vend_cycle,
                voltage_drift_mV_per_cycle=(v_drift / _span if _span
                                            else np.nan),
                area_retention_pct=a_ret,
                discontinuity_events=n_ev,
                discontinuity_is_failure=bool(n_ev and established),
                first_discontinuity_cycle=first_ev,
                max_centre_step_mV=max_step,
                fit_conditional_continuity=(n_trk / (n_trk + n_dis)
                                            if (n_trk + n_dis) else np.nan),
                # The shape this feature was actually fitted with, over the
                # cycles where it was tracked. Reported as the majority plus a
                # count, not a single label, because a feature that is a band
                # in some cycles and a peak in others is telling you something
                # about the fit's stability that a modal label would hide.
                component_kind=_modal_kind(ts),
                component_kind_mixed=_kind_is_mixed(ts))
    tk.summary = summary
    if verbose:
        print("\n" + tk.summary_text())
    return tk


# =============================================================================
# 3. DERIVED QUANTITIES  (Module 6)
# =============================================================================

@dataclass
class DeltaV:
    """Charge/discharge pairing and the polarisation history of each couple."""
    name: str
    pairs: list = field(default_factory=list)
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    changes: dict = field(default_factory=dict)   # label -> (net, v0, v1)
    windows: dict = field(default_factory=dict)   # label -> (c0, c1, n)

    def summary_text(self):
        L = [section("  Redox pairs")]
        if not self.pairs:
            L.append("    none found within the pairing window")
        for cid, did, cv, dv, sh in self.pairs:
            # SIGNED here and a MAGNITUDE in the trend below, and both are
            # right: the sign is electrode-dependent (see this function's
            # docstring) while polarisation is |dV|, because on an anode a
            # signed trend reports growing polarisation as an improvement.
            # Printed adjacent under one name "dV", a reader saw -31 then
            # +37 for what looked like one quantity. They are now named
            # apart, and the signed value is labelled with its convention.
            L.append(f"    Charge {cid} ({cv:.3f} V) <-> Discharge {did} "
                     f"({dv:.3f} V)   V(chg)-V(dis) "
                     f"{(cv - dv) * 1000:+.0f} mV at the reference cycle"
                     + ("  [incl. shoulder]" if sh else ""))
        for label, (net, v0, v1) in self.changes.items():
            c0, c1, n = self.windows[label]
            rate = net / n if n else np.nan
            trunc = ""
            if not self.df.empty and "pair_involves_truncated" in self.df:
                sub = self.df[self.df["pair_label"] == label]
                if len(sub) and bool(sub["pair_involves_truncated"].iloc[0]):
                    trunc = "  [truncated peak — centre unreliable]"
            L.append(f"    {label}: polarisation |dV| {v0:.0f} -> "
                     f"{v1:.0f} mV (net {net:+.0f} mV = {rate:+.3f} mV/cyc "
                     f"over n={n}) cyc {c0:.0f}-{c1:.0f}{trunc}")
        return "\n".join(L)


def delta_v(tracking, *, is_anode=False, max_sep_mV=PAIR_MAX_SEPARATION_MV,
            exclude_formation=EXCLUDE_FORMATION_FROM_DELTAV, verbose=True,
            first_cycle=None):
    """
    Charge-discharge peak separation for each redox couple.

    dV = V_charge - V_discharge, and its SIGN is electrode-dependent: for a
    cathode the charge peak sits above the discharge peak, while for a negative
    electrode `io._apply_electrode_convention` has swapped the labels so the
    delithiation "Discharge" peak sits above the lithiation "Charge" peak and
    dV is negative. Polarisation is the magnitude, so a widening |dV| means
    rising internal resistance either way.

    The trend is reported as a NET CHANGE over the leading coherent window, not
    as a fitted slope: polarisation growth is sigmoidal, and a single rate would
    misstate it.
    """
    ref = tracking.reference_peaks
    pairs = _auto_pair_peaks(ref.get("Charge", pd.DataFrame()),
                             ref.get("Discharge", pd.DataFrame()),
                             max_sep_mV, is_anode=is_anode)
    T = tracking.tracked_df
    if T is None or T.empty or "step" not in T:
        return DeltaV(name=tracking.name, pairs=pairs, df=pd.DataFrame(),
                      changes={}, windows={})
    rows = []
    for cid, did, cv, dv, is_sh in pairs:
        cdat = T[(T["step"] == "Charge") & (T["tracked_peak_id"] == cid)
                 & (T["status"] == "tracked")].set_index("cycle")
        ddat = T[(T["step"] == "Discharge") & (T["tracked_peak_id"] == did)
                 & (T["status"] == "tracked")].set_index("cycle")
        # A truncated peak's centre is an extrapolation, so a pair resting on
        # one is flagged rather than silently trusted.
        pair_trunc = (bool(cdat.get("is_truncated", pd.Series(dtype=bool)).any())
                      or bool(ddat.get("is_truncated",
                                       pd.Series(dtype=bool)).any()))
        common = sorted(set(cdat.index) & set(ddat.index))
        if exclude_formation:
            # FROM THE END OF FORMATION, not from cycle 2. This was `c > 1`,
            # which drops the first cycle and nothing else — so the
            # polarisation trend was computed across cycles the pipeline had
            # already measured as unsettled, and printed beside a peak-drift
            # trend that excluded them. On the LTO triplicate that is
            # "cyc 2-10" against "over c5-9" in the same block.
            #
            # `first_cycle` is the reference cycle, which since 1.9.0.18 IS
            # the measured end of formation (detect.formation_end). Falling
            # back to 2 keeps the old behaviour where the caller does not
            # say.
            _start = (int(first_cycle) if first_cycle is not None
                      else 2)
            common = [c for c in common if c >= _start]
        for cycle in common:
            c_cen, d_cen = cdat.loc[cycle, "centre"], ddat.loc[cycle, "centre"]
            c_err = cdat.loc[cycle].get("centre_stderr", np.nan)
            d_err = ddat.loc[cycle].get("centre_stderr", np.nan)
            err = (np.sqrt(c_err ** 2 + d_err ** 2)
                   if pd.notna(c_err) and pd.notna(d_err) else np.nan)
            rows.append(dict(
                dataset=tracking.name, cycle=cycle, charge_peak_id=cid,
                discharge_peak_id=did, charge_ref_voltage=cv,
                discharge_ref_voltage=dv, pair_label=f"{cv:.2f}/{dv:.2f} V",
                charge_centre=c_cen, discharge_centre=d_cen,
                delta_v=c_cen - d_cen, delta_v_mV=(c_cen - d_cen) * 1000,
                delta_v_stderr=err,
                delta_v_stderr_mV=(err * 1000 if pd.notna(err) else np.nan),
                pair_involves_truncated=pair_trunc))

    D = pd.DataFrame(rows)
    changes, windows = {}, {}
    if not D.empty:
        for label in D["pair_label"].unique():
            p = D[D["pair_label"] == label].sort_values("cycle")
            # The MAGNITUDE, as the docstring says. Feeding the signed
            # column meant that on an anode — where dV is negative because
            # the labels are swapped — growing polarisation was reported as
            # a negative net change, which reads as an improvement.
            net, v0, v1, c0, c1, n = _coherent_net_change(
                p["cycle"], p["delta_v_mV"].abs())
            if pd.notna(net):
                changes[label] = (net, v0, v1)
                windows[label] = (c0, c1, n)

    out = DeltaV(name=tracking.name, pairs=pairs, df=D, changes=changes,
                 windows=windows)
    if verbose:
        print("\n" + out.summary_text())
    return out


def capacity_attribution(tracking, *, closure_intervals=None,
                         unattributed=None,
                         withhold_above=ATTRIBUTION_WITHHOLD_ABOVE,
                         fidelity_floor=INTEGRAL_FIDELITY_FLOOR,
                         fidelity_ceiling=INTEGRAL_FIDELITY_CEILING,
                         verbose=True):
    """
    Each peak's area as a fraction of a FIXED reference total — where that
    fraction is a measurement, and NaN where it is not.

    The basis is the summed fitted area of every peak tracked at the reference
    cycle: the same baseline tracking uses for drift and retention. A per-cycle
    denominator shrinks as peaks drop out, which redistributes lost capacity
    onto the survivors; a fixed one lets the stack fall below 100%, which is
    what actually happened.

    NEW IN 1.9.0 — attribution is withheld where the split is not determined.
    `closure_intervals` maps (cycle, step) -> (lo, hi) from
    `quality.closure_interval`. Where hi - lo exceeds `withhold_above`, a
    pseudo-Voigt and the polynomial baseline are near-degenerate over that
    window: refitting with a different baseline degree moves the "peak" share
    of the capacity by more than the number being reported. The AREA is kept —
    it is what the fit found — but the FRACTION is set to NaN and the row
    carries `attribution_withheld` and the interval that caused it. On NMC111
    that suppresses the whole dataset, which is the correct answer; on LTO and
    P3 it suppresses nothing.

    NEW IN 1.9.1 — attribution is ALSO withheld where the curve does not
    account for the cell's charge. Closure asks "do the fitted peaks account
    for the CURVE"; integral fidelity asks "does the curve account for the
    CELL". A dataset can pass the first perfectly and fail the second
    completely, and the LTO triplicate does: closure 0.98-1.00, fidelity
    0.58-0.68. Attributing capacity to peaks on a curve carrying 60% of the
    charge would be attributing a capacity that is not there. The reason is
    recorded per step in `fidelity_withheld`, so the two grounds stay
    distinguishable in the CSV.
    """
    T = tracking.tracked_df
    # `Tracking.tracked_df` defaults to a bare DataFrame, and `track_peaks`
    # returns exactly that when nothing was tracked — reachable whenever the
    # reference cycle has no detected peaks. `coherence_audit` already
    # handled it; these two raised KeyError: 'step'.
    if T is None or T.empty or "step" not in T:
        return {}
    ref_v_lookup = {}
    for step in ("Charge", "Discharge"):
        rp = tracking.reference_peaks.get(step, pd.DataFrame())
        if rp is not None and not rp.empty:
            for _, pk in rp.iterrows():
                ref_v_lookup[(step, int(pk["peak_id"]))] = float(pk["voltage"])

    # `quality.sample_half_cycles` measures the closure interval on about
    # ten half-cycles per dataset, deliberately, because stability is a
    # property of the curve's SHAPE and is consistent within a dataset and
    # step. Reading the map cycle by cycle therefore gave every unmeasured
    # cycle a NaN width and a quiet "not withheld" — the safeguard was inert
    # on ~97% of the run. The sample is propagated the same way it was
    # taken: the widest interval seen for a step stands for that step.
    _step_width = {}
    for (_c, _s), _iv in (closure_intervals or {}).items():
        if _iv is None:
            continue
        _lo, _hi = _iv[0], _iv[1]
        # A MEASURED-BUT-FAILED interval (no baseline degree converged) is
        # not evidence of stability; it is the absence of evidence.
        _w = (float(_hi - _lo) if np.isfinite(_lo) and np.isfinite(_hi)
              else np.inf)
        _step_width[_s] = max(_step_width.get(_s, -np.inf), _w)

    out = {}
    for step in ("Charge", "Discharge"):
        st = T[(T["step"] == step) & (T["status"] == "tracked")]
        if st.empty:
            continue
        _sw = _step_width.get(step, np.nan)
        # Does this step's CURVE account for the cell's charge? Set by
        # `flag_low_fidelity`; absent, this test cannot fire and does not.
        _fid = tracking.integral_fidelity.get(step, np.nan)
        # TWO-SIDED, like every other reading of this number in the package.
        # `flag_low_fidelity`, `Tracking.summary_text` ("AREA NOT A
        # CAPACITY") and `quality.describe_fidelity` ("AREAS ARE NOT
        # CAPACITIES — above 1.20") all test both ends; this gate tested only
        # the floor, so a curve carrying 155% of the charge the cell passed
        # had every per-peak share published with `fidelity_withheld=False`,
        # one page after the reader was told those areas are not capacities.
        # The excess is the more damning of the two: a deficit can be charge
        # the window did not cover, an excess is area that is not there.
        _fid_bad = bool(np.isfinite(_fid)
                        and (_fid < float(fidelity_floor)
                             or _fid > float(fidelity_ceiling)))
        pids = sorted(st["tracked_peak_id"].unique())
        basis_area, basis_cycle = _attribution_basis(st,
                                                     tracking.reference_cycle)
        basis = (basis_area if (pd.notna(basis_area) and basis_area > 0)
                 else np.nan)
        rows = []
        for cycle in sorted(st["cycle"].unique()):
            cd = st[st["cycle"] == cycle]
            iv = (closure_intervals or {}).get((int(cycle), step))
            if iv is not None:
                width = (float(iv[1] - iv[0])
                         if np.isfinite(iv[0]) and np.isfinite(iv[1])
                         else np.inf)
                measured = True
            else:
                width, measured = _sw, False
            # A THIRD GROUND, AND THE ONLY ONE LEFT WHEN THERE IS NO BASELINE.
            # The closure interval measures how much the peak/background split
            # moves with the baseline degree. From 1.9.0.56 there is no free
            # background, so that width is zero by construction and this test
            # can no longer fire — which is correct, the split IS determined.
            # What can still go wrong is that the named components do not add
            # up to the cell's charge, and a per-peak SHARE of a total that is
            # missing a fifth of the capacity is not a measurement either.
            # See `quality.UNATTRIBUTED_QUALIFIED_ABOVE`.
            _un = (unattributed or {}).get((int(cycle), step), np.nan)
            _un_bad = bool(np.isfinite(_un)
                           and abs(float(_un)) > UNATTRIBUTED_WITHHOLD_ABOVE)
            withheld = bool(width == np.inf
                            or (np.isfinite(width) and width > withhold_above)
                            or _un_bad or _fid_bad)
            # np.inf in a CSV is not a width. Report the number where there
            # is one; the withheld flag beside it carries the verdict.
            width_out = width if np.isfinite(width) else np.nan
            _keep = _countable_area(cd)
            total = float(_area(cd.loc[_keep]).sum())
            _total_all = float(_area(cd).sum())
            row = dict(cycle=int(cycle), total_area=total,
                       total_area_all=_total_all,
                       n_excluded_from_area=int((~_keep).sum()),
                       basis_area=basis_area, basis_cycle=basis_cycle,
                       closure_interval_width=width_out,
                       closure_interval_measured=measured,
                       attribution_withheld=withheld,
                       unattributed_fraction=(float(_un)
                                              if np.isfinite(_un) else np.nan),
                       unattributed_withheld=_un_bad,
                       integral_fidelity=_fid,
                       fidelity_withheld=_fid_bad,
                       retained_fraction=(np.nan if withheld or pd.isna(basis)
                                          else total / basis))
            for pid in pids:
                pk = cd[cd["tracked_peak_id"] == pid]
                area = (float(_area(pk).iloc[0]) if not pk.empty
                        else np.nan)
                # A component excluded from the SUM must not be published as
                # a SHARE of it either — that was the half-wiring.
                _counts = bool(_keep.loc[pk.index].all()) if not pk.empty else False
                frac = (np.nan if (withheld or pk.empty or pd.isna(basis)
                                   or not _counts)
                        else area / basis)
                row[f"peak{pid}_area"] = area
                row[f"peak{pid}_fraction"] = frac
                row[f"peak{pid}_ref_V"] = ref_v_lookup.get((step, pid), np.nan)
                row[f"peak{pid}_is_truncated"] = (
                    bool(pk["is_truncated"].any()) if not pk.empty else False)
            rows.append(row)
        out[step] = pd.DataFrame(rows)

    if verbose:
        print("\n" + section("  Capacity attribution "
                             "(basis = summed reference-cycle area)"))
        for step, A in out.items():
            if A.empty:
                continue
            n_w = int(A["attribution_withheld"].sum())
            last = A.iloc[-1]
            print(f"    {step} (basis cycle {int(last['basis_cycle'])}, "
                  f"{len(A)} cycles)")
            if bool(A["fidelity_withheld"].any()):
                print(f"      WITHHELD on every cycle — the dQ/dV curve "
                      f"accounts for only "
                      f"{100 * float(A['integral_fidelity'].iloc[0]):.0f}% of "
                      f"this step's delivered capacity, so there is no "
                      f"capacity here to attribute. The fitted AREAS are "
                      f"kept and so are the peak positions.")
            if n_w:
                w = A.loc[A["attribution_withheld"], "closure_interval_width"]
                print(f"      WITHHELD on {n_w}/{len(A)} cycles — closure "
                      f"interval {w.min():.2f}-{w.max():.2f} wide, above the "
                      f"{withhold_above:.2f} at which the peak/background "
                      f"split stops being determined by the data.")
                print(f"      The fitted areas are still in the table; the "
                      f"SHARE of capacity is not a measurement here.")
            rep = A[~A["attribution_withheld"]]
            if rep.empty:
                continue
            last = rep.iloc[-1]
            print(f"      cycle {int(last['cycle'])}: "
                  f"{last['retained_fraction'] * 100:.0f}% of basis retained")
            for pid in sorted({int(c.split('_')[0][4:]) for c in A.columns
                               if c.startswith("peak")
                               and c.endswith("_fraction")}):
                f = last.get(f"peak{pid}_fraction", np.nan)
                if pd.notna(f) and f > 0:
                    tr = (" [truncated — area is a lower bound]"
                          if last.get(f"peak{pid}_is_truncated", False) else "")
                    print(f"        peak {pid} "
                          f"({last[f'peak{pid}_ref_V']:.3f} V): "
                          f"{f * 100:.1f}% of basis{tr}")
    return out


# =============================================================================
# 4. COHERENCE AUDIT  (Module 7c, as a function)
# =============================================================================

def _assign_to_reference(rel, ref_voltages, tol_V):
    """One component per reference peak per cycle, chosen once.

    Reference peaks less than 2 x COHERENCE_BAND_TOL_MV apart have
    OVERLAPPING bands, and detection allows peaks 30 mV apart against a
    70 mV tolerance. Selecting each band independently let two reference
    peaks be judged from the same fitted component while another component
    was never examined at all. Assigning them jointly, cycle by cycle, on
    minimum total distance makes the mapping one-to-one.

    Returns {(cycle, ref_index): row index into `rel`}.
    """
    out = {}
    if rel.empty or not len(ref_voltages):
        return out
    rv = np.asarray(ref_voltages, float)
    for cyc, g in rel.groupby("cycle", dropna=True):
        cv = pd.to_numeric(g["centre"], errors="coerce").to_numpy()
        ok = np.isfinite(cv)
        if not ok.any():
            continue
        idx = g.index.to_numpy()[ok]
        cv = cv[ok]
        BIG = 1e6
        cost = np.abs(rv[:, None] - cv[None, :])
        cost = np.where(cost <= tol_V, cost, BIG)
        ri, cj = linear_sum_assignment(cost)
        for i, j in zip(ri, cj):
            if cost[i, j] >= BIG:
                continue
            out[(int(cyc), int(i))] = idx[j]
    return out


def coherence_audit(params_df, detection, *, profile="broad", verbose=True,
                   rate_protocol=None):
    """
    Does each peak move like a redox feature, or like a fitting artefact?

    At C/10 a redox centre moves a few mV per cycle and holds its area. On P3
    cell B the three well-behaved peaks drifted a median of 1.0, 1.3 and 4.5 mV
    per cycle; the truncated peak drifted 15.2 mV with half its consecutive
    pairs over 20 mV.

    The verdict is driven by the MEDIAN drift, not by the tail fractions. An
    earlier version used the tails and denied "coherent" to a peak drifting
    1.4 mV per cycle because 14% of its pairs crossed 20 mV — the tails are a
    noisy statistic on a few dozen pairs and the median is not. Tails are still
    reported, as corroboration.

    Returns a DataFrame, one row per reference peak. It deliberately declines
    to judge below COHERENCE_MIN_PAIRS: three consecutive pairs cannot
    establish a trend, and saying so is more useful than a verdict.
    """
    P = params_df
    if P is None or P.empty:
        return pd.DataFrame()
    rel = P[P["reliable"] == True] if "reliable" in P else P   # noqa: E712
    tol = COHERENCE_BAND_TOL_MV / 1000.0

    # A PAIR THAT CROSSES A RATE CHANGE IS NOT A PAIR. This audit compares
    # CONSECUTIVE cycles and asks whether a peak moved more than a redox
    # centre should and whether its area more than doubled. Across a rate
    # boundary both are the correct answer: on
    # `JQ_NNM_C-rate_2-4.2V_C_29042026` cycle 25 -> 26 steps from 2C to 5C,
    # where the area legitimately halves and the centre legitimately moves
    # by a hundred millivolts of overpotential. Counting those as
    # incoherence took the median drift from the 0.6-1.8 mV of the
    # constant-rate NNM triplicate to 5.0 mV, and put `f_area_over_2x` at
    # 0.2-0.5 on nearly every reference — condemning real features for
    # obeying Ohm's law.
    #
    # So cross-block pairs are DROPPED, not counted. The peak is then judged
    # only on cycle pairs measured at the same current, which is the
    # comparison the audit was always meant to make.
    _rate_of = {}
    if rate_protocol and rate_protocol.get("is_variable"):
        for _bi, _b in enumerate(rate_protocol.get("blocks") or []):
            for _c in _b.get("cycles", ()):
                _rate_of[int(_c)] = _bi
        for _c in (rate_protocol.get("transition_cycles") or []):
            _rate_of[int(_c)] = None
    _n_cross = 0

    rows = []
    for step in ("Charge", "Discharge"):
        ref = detection.reference_peaks.get(step)
        if ref is None or ref.empty:
            continue
        _rel_step = rel[rel["step"] == step]
        _assign = _assign_to_reference(
            _rel_step, [float(v) for v in ref["voltage"]], tol)
        for _ri, (_, rr) in enumerate(ref.iterrows()):
            rv = float(rr["voltage"])
            rec = dict(step=step, reference_voltage=rv,
                       is_truncated=bool(rr.get("is_truncated", False)),
                       pairs=0, median_drift_mV=np.nan,
                       f_drift_over_20mV=np.nan, f_area_over_2x=np.nan,
                       verdict="never recovered")
            _rows = {c: i for (c, k), i in _assign.items() if k == _ri}
            if not _rows:
                rows.append(rec)
                continue
            # Whole ROWS, indexed by cycle. `groupby("cycle").first()` takes
            # the first non-null value per column INDEPENDENTLY, so a single
            # "row" could be assembled from a centre in one component and an
            # area in another.
            b = _rel_step.loc[[_rows[c] for c in sorted(_rows)]].copy()
            b.index = sorted(_rows)
            cyc = list(b.index)
            dv, ar = [], []
            for a, c in zip(cyc, cyc[1:]):
                if c - a != 1:
                    continue
                if _rate_of:
                    _ra, _rc = _rate_of.get(int(a)), _rate_of.get(int(c))
                    if _ra is None or _rc is None or _ra != _rc:
                        _n_cross += 1
                        continue
                dv.append((b.loc[c, "centre"] - b.loc[a, "centre"]) * 1000)
                _ba = _area(b)
                a0 = _ba.loc[a]
                ar.append(_ba.loc[c] / a0
                          if (pd.notna(a0) and a0) else np.nan)
            if not dv:
                rec["verdict"] = "never in consecutive cycles"
                rows.append(rec)
                continue
            dv = np.asarray(dv, float)
            ar = np.asarray(ar, float)
            f_v = float((np.abs(dv) > COHERENCE_DRIFT_IMPLAUSIBLE_MV).mean())
            with np.errstate(invalid="ignore"):
                f_a = float(np.nanmean((ar > COHERENCE_AREA_RATIO_MAX)
                                       | (ar < 1 / COHERENCE_AREA_RATIO_MAX)))
            med = float(np.median(np.abs(dv)))
            if len(dv) < COHERENCE_MIN_PAIRS:
                verdict = "not assessable"
            elif med > COHERENCE_MEDIAN_SUSPECT_MV or f_v > 0.5 or f_a > 0.5:
                verdict = "NOT trend-worthy"
            elif med <= COHERENCE_MEDIAN_COHERENT_MV:
                verdict = "coherent"
            else:
                verdict = "questionable"
            rec.update(pairs=len(dv), median_drift_mV=med,
                       f_drift_over_20mV=f_v, f_area_over_2x=f_a,
                       verdict=verdict)
            rows.append(rec)

    C = pd.DataFrame(rows)
    if verbose and _n_cross:
        print(f"    {_n_cross} consecutive-cycle pair(s) crossing a rate "
              f"change excluded — a peak that moves when the current "
              f"changes is not drifting.")
    if verbose and not C.empty:
        print("\n" + section(f"  Fit coherence (profile {profile}, "
                              f"{len(P)} components, {len(rel)} reliable)"))
        print(f"    {'step':<10}{'ref V':>8}{'pairs':>7}{'median drift':>14}"
              f"{'>20 mV':>9}{'area >2x':>10}  verdict")
        for r in C.itertuples():
            med = ("      —" if not np.isfinite(r.median_drift_mV)
                   else f"{r.median_drift_mV:9.1f} mV")
            fv = ("    —" if not np.isfinite(r.f_drift_over_20mV)
                  else f"{r.f_drift_over_20mV:8.0%}")
            fa = ("     —" if not np.isfinite(r.f_area_over_2x)
                  else f"{r.f_area_over_2x:9.0%}")
            note = ("TRUNCATED; " if r.is_truncated else "") + r.verdict
            print(f"    {r.step:<10}{r.reference_voltage:>8.3f}{r.pairs:>7}"
                  f"{med:>14}{fv:>9}{fa:>10}  {note}")
        tally = C["verdict"].value_counts()
        print("    " + "   ".join(f"{int(tally.get(k, 0))} {k}" for k in
                                  ("coherent", "questionable",
                                   "NOT trend-worthy", "not assessable"))
              + f"   ({len(C)} peak(s) examined)")
    return C


# =============================================================================
# 5. MECHANISM AND INTEGRITY  —  NEW IN 1.9.0
# =============================================================================

@dataclass
class Integrity:
    """What the charge that went backwards in one half-cycle was doing."""
    cycle: int
    step: str
    capacity: float
    n_records: int
    plateau_charge: float = 0.0       # mAh/g, on a two-phase plateau
    cv_charge: float = 0.0            # mAh/g, at the limit with current tapering
    parasitic_charge: float = 0.0     # mAh/g, full current, no redox to explain it
    n_reversals: int = 0
    theoretical_mAh_g: float = float("nan")
    capacity_ratio: float = float("nan")   # capacity / theoretical
    # The threshold ACTUALLY USED, so the `over_theoretical` property and the
    # banding cannot disagree. Reading the module constant in the property
    # while the banding read the argument let a half-cycle be simultaneously
    # not-ANOMALOUS and over_theoretical=True.
    over_theoretical_fraction: float = OVER_THEORETICAL_FRACTION
    n_unknown_charge: int = 0
    outside_window_charge: float = 0.0   # mAh/g, reversed outside the window
    band: str = "unknown"
    reason: str = ""

    @property
    def over_theoretical(self):
        return (np.isfinite(self.capacity_ratio)
                and self.capacity_ratio > self.over_theoretical_fraction)

    @property
    def parasitic_fraction(self):
        if not np.isfinite(self.capacity) or self.capacity <= 0:
            return np.nan
        return self.parasitic_charge / self.capacity


def classify_reversals(sig, *, window, plateau_fraction=PLATEAU_DQDV_FRACTION,
                       limit_tol_mV=LIMIT_TOL_MV):
    """
    Say what each reversal in one `signal.HalfCycleSignal` was.

    Returns a list of dicts: the `Reversal` fields plus `mechanism` and the
    |dQ/dV| there as a fraction of the half-cycle's own maximum.

    The plateau test reads the CURVE, not a peak list — which is why integrity
    does not depend on detection, and why it transfers between a chemistry with
    one sharp two-phase peak and one with six broad solid-solution features.
    A peak-proximity test does not: with a 100 mV tolerance and six peaks across
    1.4 V, almost every voltage counts as "at a peak", and with the peaks' own
    half-maximum bands, LTO's plateau — which is many times wider than its
    peak's FWHM — counts as parasitic in half its healthy half-cycles.
    """
    if sig.voltage.size == 0 or not sig.reversals:
        return []
    a = np.abs(np.asarray(sig.dqdv, float))
    amax = float(np.nanmax(a)) if a.size else np.nan
    if not np.isfinite(amax) or amax <= 0:
        return []
    vlo, vhi = window
    tol = limit_tol_mV / 1000.0

    # How far apart consecutive samples of the TRIMMED curve are. The
    # reversals are found on the raw, untrimmed, acquisition-ordered
    # records, so a reversal inside the trimmed edge band — which is exactly
    # where a constant-voltage hold lives — has no corresponding sample, and
    # an unbounded argmin silently returned the window edge, whose |dQ/dV|
    # has nothing to do with it.
    _dv = (np.diff(np.sort(sig.voltage)) if sig.voltage.size > 1
           else np.array([np.inf]))
    _dv = _dv[_dv > 0]
    _reach = (5.0 * float(np.median(_dv)) if _dv.size else np.inf)

    out = []
    for r in sig.reversals:
        mid = 0.5 * (r.v_start + r.v_end)
        j = int(np.argmin(np.abs(sig.voltage - mid)))
        near = abs(float(sig.voltage[j]) - mid) <= _reach
        share = float(a[j] / amax) if near else np.nan
        at_limit = (mid >= vhi - tol) or (mid <= vlo + tol)
        if not r.at_full_current:
            mech = "cv_hold" if at_limit else "current_taper"
        elif at_limit:
            # Tested BEFORE the plateau share. A reversal at the cell's own
            # cut-off is a hold whether or not |dQ/dV| happens to be high
            # there; calling it a plateau misattributed the charge at the
            # one voltage where the protocol explains it.
            mech = "cv_hold"
        elif np.isfinite(share) and share >= plateau_fraction:
            mech = "plateau"
        elif not np.isfinite(share):
            # Outside the analysed window: no evidence either way.
            mech = "outside_window"
        else:
            mech = "parasitic"
        out.append(dict(i0=r.i0, i1=r.i1, charge=r.charge, v_start=r.v_start,
                        v_end=r.v_end, span_mV=r.span_mV, seconds=r.seconds,
                        at_full_current=r.at_full_current,
                        n_records=r.n_records, voltage=mid,
                        dqdv_share=share, mechanism=mech))
    return out


def half_cycle_integrity(sig, *, window,
                         plateau_fraction=PLATEAU_DQDV_FRACTION,
                         limit_tol_mV=LIMIT_TOL_MV,
                         anomalous_fraction=PARASITIC_ANOMALOUS_FRACTION,
                         min_records=INTEGRITY_MIN_RECORDS,
                         theoretical_mAh_g=None,
                         over_theoretical=OVER_THEORETICAL_FRACTION):
    """
    One half-cycle's band, from the charge it could not account for.

    Two independent tests, and either can condemn a half-cycle:

      the CURVE       where the reversed charge sat — plateau, hold, or
                      nothing that accounts for it
      the TOTAL       whether more charge passed than the material can hold

    The second needs no dQ/dV at all and catches what the first cannot: a
    half-cycle can put its charge somewhere that looks like a redox plateau
    and still be passing four times the theoretical capacity.

    clean            no charge passed backwards away from a redox process
    suspect          some did, but under `anomalous_fraction` of this
                     half-cycle's own capacity
    ANOMALOUS        at or above it
    too few records  the half-cycle is too short to read. NOT the same as
                     clean, and `detect.SOUND_BANDS` excludes it for that
                     reason: it means we could not tell.

    Every comparison is against this half-cycle itself. The two thresholds
    1.8.x tried — `f_wrong_sign` and `f_voltage_backwards` — were set on a
    pooled distribution and transferred neither between the three cells of one
    triplicate nor between chemistries. A fraction of a half-cycle's own
    capacity cannot fail that way.
    """
    g = Integrity(cycle=int(sig.cycle), step=str(sig.step),
                  capacity=float(sig.capacity),
                  n_records=int(sig.n_records_raw),
                  over_theoretical_fraction=float(over_theoretical))
    if sig.n_records_raw < min_records or sig.voltage.size == 0:
        g.band = "too few records"
        g.reason = (f"{sig.n_records_raw} records"
                    if sig.n_records_raw < min_records
                    else "no usable curve")
        return g

    events = classify_reversals(sig, window=window,
                                plateau_fraction=plateau_fraction,
                                limit_tol_mV=limit_tol_mV)
    g.n_reversals = len(events)
    n_unknown = 0
    for e in events:
        if not np.isfinite(e["charge"]):
            # `signal.reversals` sets charge=NaN whenever the capacity
            # column is absent. Folding that into a zero made
            # `parasitic_charge` 0 and the verdict "clean" — a positive
            # all-clear derived from no information at all.
            n_unknown += 1
            continue
        q = e["charge"]
        if e["mechanism"] == "plateau":
            g.plateau_charge += q
        elif e["mechanism"] in ("cv_hold", "current_taper"):
            g.cv_charge += q
        elif e["mechanism"] == "outside_window":
            g.outside_window_charge += q
        else:
            g.parasitic_charge += q
    g.n_unknown_charge = n_unknown

    # --- the total, against what the material can hold --------------------
    # `if theoretical_mAh_g` turned 0.0 into NaN, and NaN disables the
    # over-theoretical guard completely — `x > 1.2 * nan` is False for every
    # x. Cell 2 now rejects a non-positive capacity on the declarative path;
    # this closes the same hole on the interactive one, where a typed 0 could
    # still get through. An unusable value is NaN either way, but it is now
    # NaN because it was unusable, not because it was falsy.
    try:
        _theo = float(theoretical_mAh_g)
    except (TypeError, ValueError):
        _theo = np.nan
    g.theoretical_mAh_g = _theo if np.isfinite(_theo) and _theo > 0 else np.nan
    if (np.isfinite(g.theoretical_mAh_g) and g.theoretical_mAh_g > 0
            and np.isfinite(g.capacity)):
        g.capacity_ratio = g.capacity / g.theoretical_mAh_g
        if g.capacity_ratio > over_theoretical:
            g.band = "ANOMALOUS"
            g.reason = (f"{g.capacity:.0f} mAh/g passed — "
                        f"{g.capacity_ratio:.1f}x the theoretical capacity of "
                        f"{g.theoretical_mAh_g:.0f} mAh/g. More charge went in "
                        f"than the material can hold, so the intended reaction "
                        f"does not account for it")
            return g

    f = g.parasitic_fraction
    if n_unknown:
        # Excluded from detect.SOUND_BANDS for the same reason as
        # "too few records": we could not tell.
        g.band = "unknown"
        g.reason = (f"{n_unknown} of {g.n_reversals} reversal(s) carry no "
                    f"charge figure — the capacity counter is missing, so "
                    f"how much charge went backwards cannot be said")
    elif not np.isfinite(f):
        g.band = "unknown"
        g.reason = "no counter capacity to compare against"
    elif g.parasitic_charge <= 0 and np.isfinite(g.capacity_ratio) \
            and g.capacity_ratio > 1.0:
        # OVER THEORETICAL IS NOT CLEAN, even under the escalation threshold.
        # `over_theoretical` is 1.2, a tolerance for the uncertainty in the
        # active mass and in the quoted theoretical capacity — not a licence
        # to pass 1.0-1.2x and be called clean. NNM cell A cycle 9 passed
        # 1.069x with ten voltage reversals and 25.6 mAh/g going backwards
        # onto a plateau, and was reported as "no charge passed backwards
        # away from a redox process": a sentence its own `plateau_charge`
        # column contradicts.
        g.band = "suspect"
        g.reason = (f"{g.capacity:.0f} mAh/g passed — {g.capacity_ratio:.2f}x "
                    f"the theoretical {g.theoretical_mAh_g:.0f} mAh/g. No "
                    f"charge went backwards away from a redox process, but "
                    f"more went in than the material can hold"
                    + (f", and {g.plateau_charge + g.cv_charge:.1f} mAh/g of "
                       f"it passed backwards during "
                       f"{g.n_reversals} voltage reversal(s)"
                       if (g.plateau_charge + g.cv_charge) > 1.0 else "")
                    + ". Below the "
                    f"{over_theoretical:.1f}x at which this is called a fault, "
                    f"so it is flagged rather than escalated")
    elif g.parasitic_charge <= 0:
        g.band = "clean"
        # SAY WHAT WAS FOUND, not only what was not. The old sentence was
        # true and read as "nothing happened here" on half-cycles carrying
        # tens of mAh/g backwards on a plateau. Where a plateau is expected
        # physics that is fine; whether it IS expected depends on the
        # mechanism, which is decided two cells later — so this states the
        # measurement and leaves the reading to the page that knows.
        _back = g.plateau_charge + g.cv_charge
        g.reason = ("no charge passed backwards away from a redox process"
                    if _back <= 1.0 else
                    f"no charge passed backwards away from a redox process; "
                    f"{_back:.1f} mAh/g did go backwards during "
                    f"{g.n_reversals} voltage reversal(s), on a plateau or at "
                    f"the voltage limit")
    elif f >= anomalous_fraction:
        g.band = "ANOMALOUS"
        g.reason = (f"{g.parasitic_charge:.1f} mAh/g ({f:.1%} of this "
                    f"half-cycle's capacity) passed at full current where "
                    f"|dQ/dV| was below {plateau_fraction:.0%} of its own "
                    f"maximum")
    else:
        g.band = "suspect"
        g.reason = (f"{g.parasitic_charge:.2f} mAh/g ({f:.2%}) unaccounted for")
    return g


# --- is this a cell with a fault, or a cell with some bad cycles? ---------
# The run counted anomalous half-cycles and stopped there. "9 anomalous
# half-cycles of 160" reads like a handful of bad readings; what it was on NNM
# cell A is a cell that stalls on charge over and over, from the first cycle to
# the sixty-ninth, accepting up to four times the charge the material can hold
# and giving 87% of it back. Those are the same numbers and opposite readings,
# and the difference between them is RECURRENCE — which nothing aggregated.
#
# Three cases have to be told apart, and the run already holds everything
# needed to do it:
#   formation   flagged cycles all early, none after the cell settled
#   isolated    one or two, scattered, no pattern
#   recurring   enough of them, spread across the life of the cell
RECURRENCE_MIN_EVENTS = 3        # fewer than this is not a pattern
RECURRENCE_MIN_SPAN_FRACTION = 0.5   # of the cycles analysed
RECURRENCE_FORMATION_CYCLES = 5      # "early" if it is all inside this


def cell_integrity_verdict(integrity, *, reference_cycle=None,
                           n_cycles=None):
    """
    One verdict for the CELL, from the per-half-cycle integrity table.

    Returns a dict with `verdict` (one of "sound", "formation", "isolated",
    "recurring"), the numbers behind it, and `sentence` — the thing a person
    should be told. Never raises on an empty or partial table.
    """
    # "UNKNOWN" IS THE DEFAULT, NOT "SOUND". A table this cannot read must not
    # come back looking like a clean cell: the first version returned "sound"
    # for an absent or malformed table, and the page rendered nothing for
    # "sound", so a failure to assess and a cell with nothing wrong produced
    # the same silence. That is the shape of a false all-clear.
    out = dict(verdict="unknown", n_flagged=0, n_total=0, first=None,
               last=None, span=0, gaps=(), worst_ratio=np.nan,
               worst_cycle=None, steps=(),
               sentence="There is no cycle-integrity table for this dataset, "
                        "so whether its cycles are sound has not been "
                        "assessed. That is not the same as nothing being "
                        "wrong.")
    if integrity is None or getattr(integrity, "empty", True):
        return out
    d = integrity.copy()
    if "band" not in d or "cycle" not in d:
        out["sentence"] = ("the cycle-integrity table is missing the columns "
                           "this reads, so it could not be assessed")
        return out
    out["n_total"] = int(len(d))
    # FLAGGED means the run itself said something was wrong here — the bands
    # it escalates on, plus anything over theoretical, which is the physical
    # impossibility and is not always escalated (see `classify_integrity`).
    # TWO THRESHOLDS, DELIBERATELY, AND THE SENTENCE MUST SAY WHICH IT USED.
    # This tripwire is `capacity_ratio > 1.0` — any charge at all beyond what
    # the material can hold — while `Integrity.over_theoretical`, and the
    # report finding built on it, escalate only past
    # `OVER_THEORETICAL_FRACTION` (1.2), a tolerance for the uncertainty in the
    # theoretical capacity and in the electrode mass. Both are wanted, and
    # `test_cell_verdict` pins this one at 1.0. What was wrong was that one
    # page carried both lists under the same words: NNM cell A read "4
    # half-cycle(s) passed more charge than the material can hold ... Cycles:
    # 1, 13, 19, 69" three findings above "5 cycles were flagged on charge —
    # 1, 13, 19, 26, 69", and nothing reconciled them. Cycle 26 was at 1.026x:
    # over one, inside the tolerance. The counts are separated below so the
    # sentence can name the difference instead of hiding it.
    _ratio = (pd.to_numeric(d.get("capacity_ratio"), errors="coerce")
              if "capacity_ratio" in d else pd.Series(np.nan, index=d.index))
    _over = (_ratio > 1.0).fillna(False)
    _escalated = (_ratio > OVER_THEORETICAL_FRACTION).fillna(False)
    out["n_over_theoretical"] = int(_over.sum())
    out["n_over_theoretical_escalated"] = int(_escalated.sum())
    out["n_over_theoretical_within_tolerance"] = int((_over & ~_escalated).sum())
    flag = d["band"].astype(str).eq("ANOMALOUS") | _over.fillna(False)
    f = d[flag]
    out["n_flagged"] = int(len(f))
    if f.empty:
        out["verdict"] = "sound"
        out["sentence"] = (f"No half-cycle of {out['n_total']} was flagged: "
                           f"none passed more charge than the material can "
                           f"hold, and none lost charge away from a redox "
                           f"process.")
        return out
    out["verdict"] = "sound"        # provisional; refined below
    cyc = sorted(int(c) for c in pd.to_numeric(f["cycle"], errors="coerce")
                 .dropna().unique())
    out["first"], out["last"] = cyc[0], cyc[-1]
    out["span"] = cyc[-1] - cyc[0]
    out["gaps"] = tuple(np.diff(cyc).tolist())
    out["steps"] = tuple(sorted(set(f["step"].astype(str))))
    if "capacity_ratio" in f:
        _r = pd.to_numeric(f["capacity_ratio"], errors="coerce")
        if _r.notna().any():
            out["worst_ratio"] = float(_r.max())
            out["worst_cycle"] = int(f.loc[_r.idxmax(), "cycle"])
    _n = int(n_cycles or (pd.to_numeric(d["cycle"], errors="coerce").max() or 0))
    _ref = int(reference_cycle or RECURRENCE_FORMATION_CYCLES)
    _late = [c for c in cyc if c > max(_ref, RECURRENCE_FORMATION_CYCLES)]
    # TWO DECIMALS, BECAUSE THE GATE IS AT THE SECOND ONE. This clause exists
    # only where `worst_ratio > 1.0`, and at one decimal place a cycle flagged
    # at 1.0265x printed as "up to 1.0x the theoretical capacity" — a sentence
    # that says nothing exceeded theoretical, inside a sentence about a
    # half-cycle that did.
    _worst = (f" and up to {out['worst_ratio']:.2f}x the theoretical capacity"
              f" (cycle {out['worst_cycle']})"
              if np.isfinite(out["worst_ratio"]) and out["worst_ratio"] > 1.0
              else "")
    _where = (f" on {' and '.join(s.lower() for s in out['steps'])}"
              if out["steps"] else "")
    # WHY THIS LIST IS LONGER THAN THE ONE ABOVE IT. Said here rather than left
    # for the reader to work out from two sets of cycle numbers.
    _tol_n = int(out.get("n_over_theoretical_within_tolerance") or 0)
    _tol = (f" {_tol_n} of them "
            + ("is" if _tol_n == 1 else "are")
            + f" over the theoretical capacity by less than the "
              f"{OVER_THEORETICAL_FRACTION - 1.0:.0%} allowed for uncertainty "
              f"in that capacity and in the electrode mass, so "
            + ("it is" if _tol_n == 1 else "they are")
            + " flagged here and not counted as passing more charge than the "
              "material can hold."
            if _tol_n else "")

    if not _late:
        out["verdict"] = "formation"
        out["sentence"] = (
            f"{len(cyc)} cycle(s) were flagged{_where}, all of them at or "
            f"before cycle {cyc[-1]}{_worst}.{_tol} Nothing was flagged after "
            f"the cell settled, so this reads as formation rather than a "
            f"fault.")
        return out

    _long = _n and out["span"] >= RECURRENCE_MIN_SPAN_FRACTION * _n
    if len(cyc) >= RECURRENCE_MIN_EVENTS and _long:
        _gap = int(np.median(out["gaps"])) if out["gaps"] else 0
        out["verdict"] = "recurring"
        out["sentence"] = (
            f"{len(cyc)} of {_n} cycles were flagged{_where}, from cycle "
            f"{cyc[0]} to cycle {cyc[-1]} — every "
            f"{_gap} cycles or so, across "
            f"{100 * out['span'] / _n:.0f}% of the life of the cell{_worst}."
            f"{_tol} "
            f"That is not settling-in and it is not a handful of bad "
            f"readings: it is a fault that keeps happening. Capacity and "
            f"retention measured on this cell describe the fault as much as "
            f"they describe the material.")
        return out

    # THREE OR MORE EVENTS IS A PATTERN EVEN IF IT STOPPED. NNM cell B was
    # flagged at cycles 1, 12, 16, 20 and 25 on a record running to 67 —
    # "too few to call a pattern" was plainly the wrong sentence for that. A
    # run of events that ends is its own finding, and often a more interesting
    # one than a fault that simply continues.
    _list = (", ".join(str(c) for c in cyc[:8])
             + (" ..." if len(cyc) > 8 else ""))
    if len(cyc) >= RECURRENCE_MIN_EVENTS:
        _gap = int(np.median(out["gaps"])) if out["gaps"] else 0
        out["verdict"] = "clustered"
        out["sentence"] = (
            f"{len(cyc)} cycles were flagged{_where} — {_list} — every "
            f"{_gap} cycles or so{_worst}, and then nothing"
            + (f" for the remaining {_n - cyc[-1]} cycles" if _n > cyc[-1]
               else "") + "." + _tol + " A run of events that stopped is not "
            "the same as a cell that is fine: something was happening and then "
            "was not. "
            "Worth knowing what changed at cycle "
            f"{cyc[-1]}.")
        return out

    out["verdict"] = "isolated"
    out["sentence"] = (
        f"{len(cyc)} cycle(s) were flagged{_where}, at cycle(s) "
        + _list + _worst
        + "." + _tol
        + " Too few to call a pattern — read them as individual events.")
    return out


def last_useful_cycle(capacities, *, fraction=DEAD_CELL_CAPACITY_FRACTION,
                      run=DEAD_CELL_RUN):
    """
    The last cycle before the cell stopped delivering, and why.

    `capacities` is `{(cycle, step): mAh/g}` — `Dataset.capacity` for every
    half-cycle, or the `capacity` column of an integrity report.

    Returns `(cycle, reason)`. `cycle` is None when the cell never dies, which
    is the common case and must not be confused with dying at cycle 0.

    This exists to stop the FIT, not the plots. Fitting pseudo-Voigt components
    to a flat line consumes most of a run's time and every one of those fits is
    discarded downstream as unreliable — but the heatmaps and the waterfall
    should still show the whole record, because the cell going flat at cycle
    130 is exactly what the reader needs to see.
    """
    disc = {int(c): v for (c, s), v in capacities.items()
            if str(s).lower().startswith("d") and np.isfinite(v)}
    if len(disc) < run + 1:
        return None, "too few cycles to judge"
    cycles = sorted(disc)
    # The 90th percentile, NOT the median. On P3 cell B the cell dies at cycle
    # 67 and the record runs to 220, so 153 of its 220 cycles are dead and the
    # median discharge capacity is 0.04 mAh/g — a floor of 2% of that is zero
    # and the test never fires. The 90th percentile asks what this cell
    # delivered WHEN IT WAS WORKING, which is 78.6 mAh/g, and is unmoved by how
    # much dead record follows.
    ref = float(np.percentile([disc[c] for c in cycles], 90))
    if not np.isfinite(ref) or ref <= 0:
        return None, "no usable reference capacity"
    floor = fraction * ref
    dead = 0
    for i, c in enumerate(cycles):
        if disc[c] < floor:
            dead += 1
            if dead >= run:
                first = cycles[i - dead + 1]
                # A cell that is dead from its FIRST cycle has no earlier
                # cycle to fit, but returning None there means "never dies"
                # to every caller — including the notebook, which tests the
                # value for truthiness — so the 220 flat cycles this
                # function exists to skip were fitted after all. Return the
                # first cycle itself: one cycle is fitted, not all of them,
                # and the reason says what happened.
                prev = (cycles[cycles.index(first) - 1]
                        if first != cycles[0] else cycles[0])
                return prev, (f"discharge capacity fell below {floor:.2f} "
                              f"mAh/g ({fraction:.0%} of this cell's working "
                              f"capacity, {ref:.1f}) for {run} consecutive "
                              f"cycles from cycle {first}")
        else:
            dead = 0
    return None, f"never fell below {floor:.2f} mAh/g for {run} cycles running"


# How far short of the cut-off a half-cycle may end and still count as
# finished. A cycler terminates a half-cycle ON VOLTAGE, so a completed one
# lands on the limit to within the resolution of the ADC: on the LTO
# triplicate every falling half-cycle ends at 1.2000 V exactly and every
# rising one within 0.6 mV of 2.4995. The two unfinished ones stop 341 mV
# and 913 mV short. Three orders of magnitude separate the cases, so the
# threshold is not a tuned parameter.
TERMINATION_TOL_MV = 20.0
# ...or this multiple of the spread the completed half-cycles themselves
# show, whichever is larger, for a cycler with a coarser voltage record.
TERMINATION_TOL_SPREADS = 5.0
# Only the most recent same-direction half-cycles set the reference, so a
# protocol that changes its voltage window part-way through does not average
# the old limit with the new one.
TERMINATION_REFERENCE_N = 10


# --- how long did the half-cycle last? -----------------------------------
# The voltage test says a half-cycle stopped short of its cut-off. It cannot
# say WHY, and the two reasons are not the same kind of event:
#
#   still running     the export was written mid-flight. Benign. The next
#                     export has the finished half-cycle. Its duration is a
#                     plausible FRACTION of a normal one and it is the last
#                     record in the file.
#
#   terminated early  the half-cycle ran for a normal length of time and then
#                     stopped on something other than voltage — a step timer,
#                     a safety limit, an aborted schedule. That is a FINDING
#                     about the experiment and belongs in the integrity table,
#                     not in a footnote.
#
# Duration separates them, and it is free: every export carries a time column.
# Where the cycler records its own step-end reason, that column settles the
# question outright and is preferred to any inference — Neware writes one on
# some schedules, so it is looked for by name and used when found.
TERMINATION_DURATION_FRACTION = 0.90
STEP_END_REASON_COLUMNS = ("Step end reason", "End reason", "Stop reason",
                           "Cut-off reason", "Step Stop Reason")
_TIME_COLUMNS = ("Time", "Cumulative Time", "Test Time", "Total Time")


def _to_seconds(series):
    """A cycler time column as float seconds. NaN where it cannot be read."""
    s = pd.Series(series)
    if s.empty:
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    td = pd.to_timedelta(s, errors="coerce")
    if td.notna().any():
        return td.dt.total_seconds()
    # "D-HH:MM:SS", which pandas does not parse.
    def _one(x):
        try:
            t = str(x).strip()
            days = 0
            if "-" in t:
                d, t = t.split("-", 1)
                days = int(d)
            parts = [float(q) for q in t.split(":")]
            while len(parts) < 3:
                parts.insert(0, 0.0)
            return (days * 86400.0 + parts[0] * 3600.0
                    + parts[1] * 60.0 + parts[2])
        except Exception:                              # noqa: BLE001
            return np.nan
    return s.map(_one).astype(float)


def _half_cycle_seconds(frame):
    """Elapsed time of one half-cycle, in seconds. NaN if no time column."""
    for col in _TIME_COLUMNS:
        if col not in frame.columns:
            continue
        t = _to_seconds(frame[col]).dropna()
        if t.size < 2:
            continue
        # `Time` restarts at each step, so its maximum IS the duration;
        # a cumulative column needs the span. Both are covered by taking
        # the larger of the two, which for a resetting column is the max
        # and for a cumulative one is the span.
        return float(max(t.max(), t.max() - t.min()))
    return np.nan


def _step_end_reason(frame):
    """The cycler's own termination reason for this step, if it wrote one."""
    for col in STEP_END_REASON_COLUMNS:
        if col in frame.columns:
            v = frame[col].dropna()
            if len(v):
                txt = str(v.iloc[-1]).strip()
                if txt and txt.lower() not in ("nan", "none", "-"):
                    return txt
    return ""


def termination_check(dataset, *, tol_mV=None):
    """
    Which half-cycles ended before reaching their cut-off voltage, and how far short.

    Returns a DataFrame: cycle, step, direction, v_end, v_limit, short_mV,
    reached, is_last.

    A galvanostatic half-cycle ENDS ON VOLTAGE — the cycler drives until the
    cut-off and then turns round — so "did it reach the limit" is the direct
    question, asked of the quantity the instrument actually terminates on.
    Capacity is a poor proxy for it: a cell genuinely fading, or one caught
    four fifths of the way through, is indistinguishable by capacity from a
    finished half-cycle, and on this very triplicate an 80%-of-the-previous
    rule separated a truncated half-cycle from a complete one by five
    percentage points. On voltage the same two cases are 341 mV apart.

    The limit is measured from the record rather than read from the
    parameters, because the file's own `Volt. upper`/`Volt. lower` are
    channel safety limits — an LTO cell cycled 1.2-2.5 V reports 0-3.25 V.
    Half-cycles are grouped by DIRECTION OF TRAVEL, not by step label: every
    half-cycle whose voltage rises ends at the upper cut-off whatever it is
    called, which also means an anode dataset with swapped labels needs no
    special case. A constant-voltage hold still reaches the limit and still
    counts as finished.
    """
    keys = sorted(dataset.half_cycle_keys())
    rows = []
    for k in keys:
        hc = dataset.half_cycle(*k)
        v = pd.to_numeric(hc["Voltage"], errors="coerce").dropna().to_numpy()
        if v.size == 0:
            continue
        rows.append(dict(cycle=int(k[0]), step=str(k[1]),
                         direction=float(dataset.step_direction(*k)),
                         v_end=float(v[-1]), order=keys.index(k),
                         duration_s=_half_cycle_seconds(hc),
                         end_reason=_step_end_reason(hc)))
    T = pd.DataFrame(rows)
    if T.empty:
        return T

    # The LAST half-cycle in ACQUISITION order — the only one that can still
    # be running, because every earlier one finished before the next began.
    fr = dataset.frame
    last_key = (int(fr["Cycle"].iloc[-1]), str(fr["Step"].iloc[-1]))
    T["is_last"] = [(c, st) == last_key for c, st in zip(T.cycle, T.step)]

    T["v_limit"] = np.nan
    T["short_mV"] = np.nan
    T["duration_frac"] = np.nan
    T["reached"] = True
    for d in (1.0, -1.0):
        m = T["direction"] == d
        if not m.any():
            continue
        # Reference from the OTHER half-cycles going the same way, most
        # recent first, so a half-cycle is never compared with itself.
        for idx in T.index[m]:
            peers = T[m & (T.index != idx)].sort_values("order")
            # THE NEAREST PEERS, NOT THE LAST ONES IN THE FILE.
            # `peers.tail(N)` took the N most recent same-direction half-cycles
            # in the whole record for EVERY half-cycle, so an early one was
            # judged against late ones. On a protocol that widens its window
            # part-way — cycles 1-50 to 4.20 V, 51-100 to 4.40 V, which is an
            # ordinary thing to do — every one of the first fifty is 200 mV
            # short of a limit it was never asked to reach, and all fifty are
            # published as "terminated early, not on voltage". The reference
            # has to be local to be a reference at all.
            _o = float(T.at[idx, "order"])
            peers = peers.iloc[
                (peers["order"] - _o).abs().to_numpy().argsort(
                    kind="mergesort")[:TERMINATION_REFERENCE_N]]
            if len(peers) < 2:
                continue
            lim = float(peers["v_end"].median())
            # A ROBUST SPREAD. This was `max - min`, a range, multiplied by
            # five: the widest possible statistic, and one that the very
            # anomalies being detected inflate. A cell with two stalled
            # half-cycles among ten peers — ending at 4.00 and 3.90 V against
            # 4.20 — gave a 300 mV range and a 1500 mV tolerance, so each
            # stall masked the other AND every clean half-cycle in the dataset
            # inherited the same 1500 mV, after which `early_terminations`
            # returns empty and a truncated final half-cycle is fitted as
            # though it had finished. The interquartile range of the peers
            # survives up to a quarter of them being anomalous, which is the
            # case this function exists for.
            _v = pd.to_numeric(peers["v_end"], errors="coerce").dropna()
            spread = (float(_v.quantile(0.75) - _v.quantile(0.25))
                      if len(_v) >= 4 else
                      float(_v.max() - _v.min()) if len(_v) else 0.0)
            tol = max(TERMINATION_TOL_MV if tol_mV is None else float(tol_mV),
                      TERMINATION_TOL_SPREADS * spread * 1000.0)
            short = d * (lim - T.at[idx, "v_end"]) * 1000.0
            T.at[idx, "v_limit"] = lim
            T.at[idx, "short_mV"] = short
            T.at[idx, "reached"] = bool(short <= tol)
            # How long it ran, against its own peers. A half-cycle caught
            # mid-flight is SHORT; one that stopped on something other than
            # voltage ran a normal length and then stopped.
            dpeers = pd.to_numeric(peers["duration_s"],
                                   errors="coerce").dropna()
            dur = T.at[idx, "duration_s"]
            if len(dpeers) >= 2 and np.isfinite(dur) and dpeers.median() > 0:
                T.at[idx, "duration_frac"] = float(dur / dpeers.median())

    T["termination_class"] = "reached the cut-off"
    short_of = ~T["reached"].astype(bool)
    frac = pd.to_numeric(T["duration_frac"], errors="coerce")
    # SHORT and last in the file = the export caught it. Anything else that
    # stopped short of the cut-off ran a normal length of time and stopped on
    # something else, which is a finding about the experiment.
    caught = short_of & T["is_last"] & (
        frac.isna() | (frac < TERMINATION_DURATION_FRACTION))
    T.loc[short_of, "termination_class"] = "terminated early, not on voltage"
    T.loc[caught, "termination_class"] = "still running when exported"
    # The cycler's own answer wins wherever it wrote one.
    if T["end_reason"].astype(str).str.len().gt(0).any():
        known = T["end_reason"].astype(str).str.len() > 0
        T.loc[known & short_of, "termination_class"] = (
            "ended on: " + T.loc[known & short_of, "end_reason"])
    return T.drop(columns=["order"])


def half_cycles_in_progress(dataset, *, tol_mV=None, checked=None):
    """
    The half-cycle an export was written during, if there is one.

    Returns `(keys, reason)` where `keys` is a set of `(cycle, step)` — at
    most one, because only the last half-cycle in the file can still be
    running.

    A cycler export is routinely taken while the cell is CYCLING, so its
    final half-cycle is usually unfinished. That is a snapshot of a running
    experiment, not a measurement: its curve is truncated, so fitting it
    invents a peak set that changes with every re-export of the same cell,
    and its capacity is not the capacity of anything.

    The granularity is the HALF-CYCLE. On the LTO triplicate, cell B's export
    was written during cycle 11's delithiation; the lithiation half of that
    same cycle had already run to 1.2000 V and is a perfectly good
    measurement. Dropping the whole cycle would throw it away.

    Like `last_useful_cycle` this limits the FIT and not the plots — the
    partial curve still belongs in the heatmaps and the waterfall, because
    seeing where the run has got to is the point.
    """
    # `checked` is a frame this dataset's `termination_check` already
    # produced. Cell 5 asks two questions of it — what was caught in flight,
    # and what stopped early without being caught — and used to recompute the
    # whole thing for each: 10.6 s a time on P3 cell A, 21 s per dataset, for
    # one identical answer.
    T = termination_check(dataset, tol_mV=tol_mV) if checked is None else checked
    if T.empty:
        return set(), "no half-cycles to judge"
    last = T[T["is_last"]]
    if last.empty or bool(last["reached"].iloc[0]):
        return set(), ("the last half-cycle in the file reached its cut-off, "
                       "so nothing was caught mid-flight")
    r = last.iloc[0]
    if not np.isfinite(r["short_mV"]):
        return set(), "too few half-cycles to establish the cut-off voltage"
    if not str(r["termination_class"]).startswith("still running"):
        # Short of the cut-off, but it ran a NORMAL LENGTH OF TIME first. The
        # export did not catch this one; it stopped on something else, and
        # calling that "still running" would file a fault as a footnote.
        return set(), (
            f"cycle {int(r['cycle'])} {r['step']} ended "
            f"{r['short_mV']:.0f} mV short of the cut-off after "
            f"{100 * r['duration_frac']:.0f}% of a normal half-cycle's "
            f"duration — {r['termination_class']}. It is treated as a "
            f"finished half-cycle and reported in the integrity table")
    key = (int(r["cycle"]), str(r["step"]))
    _frac = ("" if not np.isfinite(r["duration_frac"])
             else f" after {100 * r['duration_frac']:.0f}% of a normal "
                  f"half-cycle's duration")
    return {key}, (
        f"cycle {key[0]} {key[1]} ended at {r['v_end']:.4f} V, "
        f"{r['short_mV']:.0f} mV short of the {r['v_limit']:.4f} V cut-off "
        f"that every other half-cycle going the same way reaches{_frac}. "
        f"It is the last half-cycle in the file, so the export was written "
        f"while it was still running. Not a measurement yet: omitted from "
        f"the fit, kept in every figure, and complete in the next export")


def early_terminations(dataset, *, tol_mV=None, checked=None):
    """
    Half-cycles that stopped short of the cut-off but were NOT caught in
    flight — a fault, not a snapshot. See TERMINATION_DURATION_FRACTION.

    Returns the subset of `termination_check`'s frame worth reporting, with
    the columns a reader needs to judge it: how far short, how long it ran
    against its peers, and the cycler's own reason where it wrote one.
    """
    T = termination_check(dataset, tol_mV=tol_mV) if checked is None else checked
    if T.empty:
        return T
    bad = T[~T["reached"].astype(bool)
            & ~T["termination_class"].astype(str).str.startswith(
                "still running")]
    cols = ["cycle", "step", "v_end", "v_limit", "short_mV", "duration_frac",
            "termination_class", "is_last"]
    return bad[[c for c in cols if c in bad.columns]].reset_index(drop=True)


def integrity_report(signals, *, window, name="", verbose=True, **kw):
    """
    Band every half-cycle, and return a lookup that `detect` can use.

    Returns (DataFrame, band) where `band(cycle, step) -> str` is exactly the
    `integrity` callable `detect.detect_all` and `detect.choose_reference_cycle`
    take. In 1.8.7 that callable was a notebook global; here it is a return
    value, which is the whole difference between a testable selector and one
    that was inert on every real run for a fortnight without anyone noticing.
    """
    recs = {}
    rows = []
    for key in sorted(signals):
        g = half_cycle_integrity(signals[key], window=window, **kw)
        recs[(int(g.cycle), str(g.step))] = g.band
        rows.append(dict(dataset=name, cycle=g.cycle, step=g.step,
                         capacity=g.capacity,
                         theoretical_mAh_g=g.theoretical_mAh_g,
                         capacity_ratio=g.capacity_ratio,
                         over_theoretical=g.over_theoretical,
                         n_records=g.n_records,
                         n_reversals=g.n_reversals,
                         plateau_charge=g.plateau_charge,
                         cv_charge=g.cv_charge,
                         parasitic_charge=g.parasitic_charge,
                         parasitic_fraction=g.parasitic_fraction,
                         band=g.band, reason=g.reason))
    R = pd.DataFrame(rows)

    def band(cycle, step):
        return recs.get((int(cycle), str(step)), "unknown")

    if verbose and not R.empty:
        counts = R["band"].value_counts()
        bad = R[R["band"] == "ANOMALOUS"].sort_values("parasitic_fraction",
                                                      ascending=False)
        over = R[R["over_theoretical"] == True]          # noqa: E712
        sane = R[R["over_theoretical"] != True]          # noqa: E712
        print(entry("half-cycles", str(len(R)),
                    ", ".join(f"{int(counts.get(k, 0))} {k}" for k in
                              ("clean", "suspect", "ANOMALOUS",
                               "too few records") if counts.get(k, 0))))
        if not bad.empty:
            tot = bad["parasitic_charge"].sum()
            print(verdict("bad", f"{tot:.0f} mAh/g unaccounted for across "
                                 f"{len(bad)} half-cycle(s)"))
            for r in bad.head(5).itertuples():
                print(bullet(f"cycle {r.cycle} {r.step}: "
                             f"{r.parasitic_charge:.0f} mAh/g of "
                             f"{r.capacity:.0f} ({r.parasitic_fraction:.0%})"))
        if len(over):
            print(verdict("bad", f"{len(over)} half-cycle(s) passed more "
                                 f"charge than the material can hold"))
            print(bullet(f"up to {over['capacity_ratio'].max():.1f}x "
                         f"theoretical. A curve cannot vouch for charge the "
                         f"material could not have held, so these are "
                         f"condemned whatever their shape."))
        if (sane["plateau_charge"] > 0).any():
            p = sane.loc[sane["plateau_charge"] > 0, "plateau_charge"]
            print(entry("plateau reversals", f"{len(p)} of {len(sane)}",
                        f"median {p.median():.1f} mAh/g"))
            print(bullet("A flat two-phase feature, not a fault — counted "
                         "over the half-cycles that stayed within the "
                         "material's capacity."))
        # `> 0` let through holds carrying a thousandth of a mAh/g, which
        # then printed as "median 0.0 mAh/g" — an alarm about zero. The floor
        # is what a cycler can resolve, and the number is printed to enough
        # figures to be read.
        if (R["cv_charge"] > CV_HOLD_REPORT_FLOOR).any():
            c = R.loc[R["cv_charge"] > CV_HOLD_REPORT_FLOOR, "cv_charge"]
            print(entry("constant-voltage hold", f"{len(c)} half-cycle(s)",
                        f"median {c.median():.3g} mAh/g — protocol, "
                        f"not chemistry"))
    return R, band
