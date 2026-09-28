"""
Where are the features, and which cycle defines them?

Detection has to happen before fitting and it cannot be undone by it. The peak
list taken from ONE cycle is fitted and tracked in every cycle of the dataset,
so a bad reference cycle is not a bad cycle — it is a bad dataset. On P3 cell C
the 1.8.6 default (cycle 2, hardcoded, checked only for existence) landed on a
half-cycle that put in 580.9 mAh/g and got 115.8 back: coulombic efficiency
19.9%, 1301 records against a median of 290, R2 0.074. Its peak list

    P3 A    3.220*, 3.301, 3.643, 3.700*
    P3 B            3.306, 3.648, 3.703*
    P3 C            3.640, 3.879, 3.901, 4.193      <- the same cell chemistry

then set the fit for all 220 cycles, missing the 3.30 V feature the other two
cells track for a hundred cycles at 2-3 mV per cycle and carrying a 4.193 V
ghost instead. Nothing in the output said so.

What is ported and what is new
------------------------------
PORTED from 1.8.7 Module 3, extracted programmatically rather than retyped:

    _voltage_to_samples   _deduplicate_peaks   _assess_truncation
    detect_peaks_single

with exactly one edit, marked in place: the charge/discharge flip now reads the
MEASURED direction rather than the step label. See the comment on that line.

NEW:

    DetectSpec            the detection parameters as a value object
    Detection             one dataset's peaks, its reference cycle, and why
    detect_all()          orchestration over `signal.HalfCycleSignal`
    choose_reference_cycle()

`choose_reference_cycle` takes the integrity check as an ARGUMENT. 1.8.7
reached into notebook globals for `integrity_band`, which meant the selector
could not be tested without a notebook and — separately — that its sibling
`ref_user_set` test had to guess at intent from a parameters dict that Cell 3
populates with defaults for every dataset. The first version of that cell was
inert on every real run for exactly this reason. Here an explicit reference
cycle is an explicit argument, and there is nothing to guess.

Serial by design
----------------
Detection is find_peaks plus one Savitzky-Golay pass: about a millisecond per
half-cycle, a few seconds for a 220-cycle dataset. That is far below
`fitting.PARALLEL_MIN_SECONDS`, and a worker pool would cost more to start than
the whole module costs to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np

from .compat import trapezoid
from .style import entry, verdict, bullet
import pandas as pd
from scipy.signal import find_peaks, peak_widths, peak_prominences, savgol_filter

# NOT `from scipy.signal import PeakPropertyWarning` — SciPy raises it but
# does not re-export it from the package namespace (checked on 1.17), so that
# import silently falls through to a stand-in class that catches nothing and
# the warnings keep printing. It lives in the private module; the try/except
# is for the day that moves, and the message filter below is the backstop
# that works whatever the class is called.
try:
    from scipy.signal._peak_finding_utils import PeakPropertyWarning
except Exception:  # pragma: no cover - layout changed
    PeakPropertyWarning = RuntimeWarning


__all__ = [
    "DetectSpec",
    "Detection",
    "detect_half_cycle",
    "detect_all",
    "choose_reference_cycle",
    "detect_peaks_single",
    "infer_missing_seeds",
    "DEFAULT_REFERENCE_CYCLE",
    "REFERENCE_CYCLE_AUTO",
    "formation_end",
    "CE_SETTLED_TOLERANCE_PP",
    "CE_SETTLED_RUN",
    "REFERENCE_CYCLE_FLOOR",
    "REFERENCE_CYCLE_FALLBACK",
    "REFERENCE_CYCLE_CEILING",
    "CE_SETTLED_MIN_LATER",
    "CE_SETTLED_MAX_PCT",
    "SOUND_BANDS",
    "detect_spec_for_profile",
    "PROFILE_MIN_DISTANCE_MV",
    "PROFILE_MIN_WIDTH_MV",
    "PROFILE_MIN_WIDTH_BY_CLASS",
    "local_noise",
    "NOISE_FLOOR",
    "NOISE_SNR_MIN",
    "NOISE_SHARE_MIN",
    "recurrent_maxima",
    "PHASE_INVARIANCE",
    "phase_stable",
    "add_recurrent_seeds",
    "RECURRENCE_SEEDS",
    "RECURRENCE_MIN_OCCUPANCY",
    "RECURRENCE_MIN_SHARE",
]


# =============================================================================
# DEFAULTS — the 1.8.7 values, unchanged. Changing one changes every peak list.
# =============================================================================

DEFAULT_REFERENCE_CYCLE = 2
REFERENCE_CYCLE_AUTO = True

# --- WHEN IS FORMATION OVER? ---------------------------------------------
# There is no consensus number, and the 2024 Energy & Environmental Science
# review of cell formation says so directly: protocols run from a single
# 1.7 h cycle to multi-day multi-cycle schedules, and "it is not possible to
# draw general conclusions and make direct comparisons between different
# studies ... cycling strategy must be adjusted to material and cell design".
# What the field DOES agree on is the criterion rather than the count:
# coulombic efficiency approaching and then holding near 100% is "a commonly
# used indicator of this stabilisation" and "useful in defining a target
# criterion for the end of formation".
#
# So this is MEASURED per cell rather than assumed. The reference cycle is
# the first one whose CE sits within CE_SETTLED_TOLERANCE_PP of the median CE
# of the cycles after it, and which is followed by CE_SETTLED_RUN - 1 more
# that also do. On the LTO triplicate that is cycle 4, 5 and 4 — where all
# three cells give one primary charge peak. At the old default of 2 it was
# cycle 2 for all three, which is the ONLY cycle in eleven where two of them
# split the plateau into two components, and where cell A's peak is
# classified a shoulder and therefore silently dropped from the replicate
# drift statistic. A triplicate reported n = 2 for that reason alone.
#
# Set REFERENCE_CYCLE_FLOOR to impose a house rule — 6 is a defensible
# conservative choice for "formation is certainly over" — and the measured
# criterion then applies at or above it.
CE_SETTLED_TOLERANCE_PP = 0.1
CE_SETTLED_RUN = 3

# TWO DIFFERENT SETTINGS, and they are easy to confuse.
#
#   REFERENCE_CYCLE_FLOOR     a hard minimum. Never use a reference below
#                             this, EVEN IF CE HAS ALREADY SETTLED. Raising
#                             it overrides a successful measurement — set it
#                             to 6 and a cell that settled at cycle 4 is
#                             still fingerprinted at 6. Left at the 1.8.7
#                             default, where it does nothing.
#
#   REFERENCE_CYCLE_FALLBACK  what to use when the measurement FAILS — when
#                             CE never holds within tolerance for long enough
#                             to say formation finished. Six is a defensible
#                             conservative choice: more conservative than
#                             most published protocols, and the point is that
#                             it only applies when nothing better is known.
#
# The distinction matters because the first silently discards a measurement
# and the second only fills a hole. If in doubt, move the fallback.
REFERENCE_CYCLE_FLOOR = DEFAULT_REFERENCE_CYCLE
REFERENCE_CYCLE_FALLBACK = 6

# FORMATION CANNOT END LATE. `formation_end` had a floor and no ceiling, and
# the P3/NNM triplicate showed why that is not a symmetric omission.
#
# The CE test asks whether CE sits within `tolerance_pp` of the median of the
# LATER cycles for `run` cycles running. Read that at the two ends of a
# record:
#
#   * early — "later" is hundreds of cycles of a fading cell, its median is
#     far from the current value, and the test is hard to pass;
#   * late  — "later" is a handful of adjacent cycles, its median is almost
#     the current value, and the test is nearly free.
#
# So the criterion gets EASIER the further in it looks, which is exactly
# backwards, and on a cell whose CE never really settles the first thing it
# passes is near the end of the file. On NNM cell C it returned **cycle 96 of
# 100**, quoting a settled CE of 80.59% for a cell that had lost 56% of its
# capacity; on cell B it returned 68, one cycle after the cell died at 67.
# Every cycle was then fitted with a peak list taken from a dead or
# exhausted cell, which is why that run inferred 208, 681 and 432 seeds and
# reported drift over a four-cycle baseline as 1.23 +/- 4.18 mV/cycle.
#
# Two guards, because there are two separate faults:
#
# CEILING — formation is a physical process that finishes in the first
# handful of cycles. A "formation end" beyond this is not a late formation,
# it is a false positive, and the honest report is that formation could not
# be measured. 15 is deliberately generous against the literature's "five or
# six" so that a genuinely slow-forming cell is not overruled.
REFERENCE_CYCLE_CEILING = 15
# MINIMUM LATER CYCLES — the median has to be a median of something. Below
# this many cycles after the candidate, the comparison is not evidence and
# the scan stops rather than accepting the easiest window in the record.
CE_SETTLED_MIN_LATER = 5
# A SETTLED CE HAS TO BE A POSSIBLE ONE. NNM cell B, which died at cycle 67,
# was reported by 1.9.0.24 as having finished formation at cycle 80 because
# its CE held steady against a later median of **150.00%** — a cell returning
# half again as much charge as it was given. That is not a settled cell, it
# is a broken measurement holding still, and nothing downstream should treat
# it as the end of formation. A little over 100 is ordinary noise on a good
# cell; well over is not.
CE_SETTLED_MAX_PCT = 105.0


def formation_end(
    efficiency,
    *,
    tolerance_pp=None,
    run=None,
    floor=None,
    ceiling=None,
    min_later=None,
    max_ce_pct=None,
):
    """
    The first cycle at which coulombic efficiency has settled, and why.

    `efficiency` is `{cycle: CE in percent}`. Returns `(cycle, reason)`, or
    `(None, reason)` where the record is too short to establish it — which is
    an answer, not a failure, and the caller falls back to the floor saying so.

    The test is against the MEDIAN OF THE LATER CYCLES, not against a fixed
    target, because a cell that settles at 99.5% has settled just as much as
    one that settles at 99.9% and neither is 100%. Half-cycles the run has
    already judged not to be measurements should be left out by the caller;
    an unfinished final cycle reads as CE 52% and would otherwise drag the
    median it is being compared against.
    """
    tolerance_pp = CE_SETTLED_TOLERANCE_PP if tolerance_pp is None else tolerance_pp
    run = CE_SETTLED_RUN if run is None else run
    floor = REFERENCE_CYCLE_FLOOR if floor is None else floor
    ceiling = REFERENCE_CYCLE_CEILING if ceiling is None else ceiling
    min_later = CE_SETTLED_MIN_LATER if min_later is None else min_later
    max_ce_pct = CE_SETTLED_MAX_PCT if max_ce_pct is None else max_ce_pct

    pairs = sorted(
        (int(c), float(v))
        for c, v in (efficiency or {}).items()
        if v is not None and np.isfinite(v)
    )
    if len(pairs) < run + 1:
        return None, (
            f"only {len(pairs)} cycles with a coulombic efficiency "
            f"— too few to say when formation ended"
        )
    cycles = [c for c, _ in pairs]
    ce = np.array([v for _, v in pairs], float)
    for i in range(len(cycles) - run + 1):
        later = ce[i + 1 :]
        # The median must be a median OF SOMETHING. Without this the test
        # gets easier the later it looks — see REFERENCE_CYCLE_CEILING — and
        # the first window it accepts is the last one in the file.
        if later.size < max(1, int(min_later)):
            break
        # Formation does not end at cycle 96. Past the ceiling the scan is
        # no longer measuring formation, so it stops and says so rather than
        # returning the first thing that happens to be flat.
        if cycles[i] > ceiling:
            # Nothing at or below the ceiling settled. Say exactly that —
            # this loop has not tested the window at `cycles[i]`, so it must
            # not claim CE settled there.
            return None, (
                f"CE never held within {tolerance_pp:g} points of the later "
                f"median for {run} cycles at or below cycle {ceiling} — "
                f"formation does not end later than that, so anything flat "
                f"beyond it is a quiet stretch of a noisy CE series and not "
                f"a measurement of formation"
            )
        median = float(np.median(later))
        if median > max_ce_pct:
            # Steady, and impossible. Keep looking rather than accepting it.
            continue
        if all(abs(ce[j] - median) <= tolerance_pp for j in range(i, i + run)):
            if cycles[i] < floor:
                return floor, (
                    f"CE settled by cycle {cycles[i]}; using the floor of {floor}"
                )
            return cycles[i], (
                f"CE within {tolerance_pp:g} points of the later median "
                f"({median:.2f}%) from here for {run} cycles running"
            )
    return None, (
        f"CE never held within {tolerance_pp:g} points of the "
        f"later median for {run} cycles — formation may not have "
        f"finished inside this record"
    )


DEFAULT_PROMINENCE_FRACTION = 0.05
DEFAULT_MIN_DISTANCE_MV = 30
# Two maxima closer than this are one feature. It is really a statement about
# peak WIDTH, so it is allowed to vary with the profile — 30 mV apart is one
# peak on a broad layered oxide whose FWHM is over 100 mV, and might not be on
# a sharp two-phase material whose FWHM is nearer 20.
#
# Every profile is nevertheless set to 30, and the reason is worth recording,
# because 1.9.0.4 briefly set "sharp" to 15 and that was WRONG.
#
# The case was LTO cell B cycle 2 charge, whose dQ/dV carries two maxima about
# 22 mV apart. They survive a six-fold change of the rebin width, and at 15 mV
# separation the fit improves from R2 0.937 to 0.999. All of that is true and
# none of it is evidence:
#
#   * Grid stability rules out the BINNING as the cause. It does not rule out
#     any other fixed stage of the pipeline, so it is necessary and nowhere
#     near sufficient.
#   * A large R2 gain from adding a free component is what overfitting looks
#     like. It is the thing to be suspicious of, not reassured by.
#
# The test that settles it is the voltage curve itself. A second dQ/dV peak
# REQUIRES a second plateau in V(Q) — the voltage must flatten, steepen and
# flatten again — and V(Q) needs no smoothing to read, because Q is a
# monotonic integral where V is a quantised measurement. Read that way:
#
#   * The "flattest point" of cell B's charge plateau sits at 1.546 +/- 1 mV
#     in cycles 3-10, and the second-flattest is 1-3 mV from it. One plateau,
#     sampled twice.
#   * Across the triplicate, local minima of |dV/dQ| appear 2-8 per half-cycle
#     at DIFFERENT voltages every cycle and every cell. A phase transition
#     does not move.
#
# And the thermodynamics agreed all along: Li4Ti5O12 -> Li7Ti5O12 is a
# first-order two-phase transition, so at fixed T and P the potential is
# invariant and there is exactly ONE peak. That is why LTO is the textbook
# flat-plateau anode.
#
# The apparent doublet is the reciprocal of a gently sloping plateau sampled
# at the cycler's 0.1 mV voltage resolution: dQ/dV = 1/(dV/dQ), and where
# dV/dQ is small its quantisation is amplified into structure.
# HOW CLOSE MAY TWO PEAKS BE. Was 30 mV on every profile, which is a
# statement about a BROAD layered oxide's features and nothing else. Swept on
# the LTO triplicate, 64 half-cycles, every value from 30 mV to 2 mV:
#
#     min_distance   detected/hc   median R2   height   fits below 0.85
#         30 mV          1.30        0.9829     0.94        10 of 64
#         12 mV          1.70        0.9850     0.94         9 of 64
#          8 mV          2.11        0.9865     0.94         8 of 64
#          5 mV          2.48        0.9867     0.95         7 of 64
#          2 mV          2.66        0.9867     0.95         7 of 64
#
# The answer stops moving at 5 mV — below it the extra detections change no
# fitted number — so that is where the floor goes rather than at the smallest
# value that still runs. It is also about one FWHM on the narrowest cell in
# hand (2.6-4.7 mV on LTO cell C) and five histogram bins at the sharp
# profile's 1 mV binning, so it is a measurement floor and not a preference.
# Costs 25 s against 5 s to fit the triplicate.
#
# Moderate and broad stay at 30 mV: no moderate dataset exists to measure on,
# and on broad the 30 mV rule is what stopped a pair 24 mV apart with a
# 30-fold height difference from collapsing into each other (see
# `_deduplicate_peaks`).
# NOT LOWERED, AND THE REASON IS THE PHASE RULE, NOT THE FIT.
#
# Lowering this to 5 mV on `sharp` lets the detector resolve the LTO cycle-1
# "doublet", and the fit then improves a great deal: on cell C cycle 1 charge
# R2 0.9161 -> 0.9951, height 0.77 -> 1.03, max residual 31% -> 7%. Swept over
# 64 half-cycles, median R2 0.9829 -> 0.9867 and the fits shorter than 0.85 of
# the data fall from 10 to 7.
#
# **That evidence is inadmissible and the change is reverted.** Appendix A7 of
# this notebook settles the same question against exactly this argument:
# Li4Ti5O12 -> Li7Ti5O12 is a first-order two-phase transition, so by the
# Gibbs phase rule the two-phase field is invariant at fixed T and P — one
# plateau, therefore ONE dQ/dV peak. V(Q), which is processing-free, shows no
# second plateau. The pair appears in one cycle of one cell and not in the
# replicates. And the literature that most threatens the claim (the 2024
# three-phase alpha/beta/gamma study) puts its third phase at 0.53-0.63 V,
# which a cell cycled 1.2-2.5 V never reaches.
#
# A7's rule is: before adding a component, ask what process it would
# correspond to; if the phase diagram does not offer one, the component is
# describing noise or a baseline, and the fit will happily accommodate it. An
# R2 gain from adding a component is what overfitting looks like. So the
# 5 mV floor stays out, and the height the fit is short by on those
# half-cycles stays in the report as a fault rather than being fitted away.
#
# The `_deduplicate_peaks` bug found while measuring this is real and IS
# fixed: for two primaries the rule was `max(shoulder_separation,
# min_distance)`, so a caller asking for anything below the 20 mV shoulder
# threshold was silently refused. At 30 mV that is a no-op (`max(20, 30) ==
# 30`), verified by re-running both broad chemistries to identical numbers —
# but it means this constant now does what it says, and a chemistry that
# genuinely has features closer than 20 mV can be given a floor that works.
PROFILE_MIN_DISTANCE_MV = {"sharp": 30, "moderate": 30, "broad": 30}
DEFAULT_MIN_WIDTH_MV = 5

# MINIMUM WIDTH, BY PROFILE — and the sharp value is the one that mattered.
#
# `DEFAULT_MIN_WIDTH_MV = 5` applied to every profile, and on LTO the measured
# FWHMs are 2.6-4.7 mV (cell C) and 12.6-22.9 mV (cell A). So on cell C the
# primary pass rejected EVERY real peak, and detection ran entirely on the
# curve-maximum rescue path — which returns exactly one component and can
# therefore never find a doublet. That is why the cycle-1 doublet Nik spotted
# on the waterfall is invisible to the pipeline.
#
# The floor a peak-finder needs is set by the MEASUREMENT: a feature narrower
# than the histogram cannot be resolved, and one at the bin width is a single
# sample. Two bins is the least that can be called a peak, and at the sharp
# profile's 1 mV bin that is 2 mV, which admits every LTO feature measured.
#
# Left at 5 mV for moderate and broad deliberately: no moderate dataset exists
# to measure on, and on broad the 5 mV floor is currently the thing standing
# between the fit and needle components (2 of 87 on NMC111 reach it and take
# the composite to 3.4x the data's height). Lowering it there is untested and
# would loosen the one guard that case has.
PROFILE_MIN_WIDTH_BY_CLASS = {"sharp": 2, "moderate": 5, "broad": 5}

# MINIMUM PEAK WIDTH, BY PROFILE. `min_distance` has been profile-dependent
# since 1.9.0 on the argument that peak separation is a statement about peak
# width; the width threshold itself was left at one global 5 mV, which on a
# sharp two-phase material is WIDER THAN THE FEATURES.
#
# What that cost, measured on LTO cell A cycle 4 delithiation: the real peak
# is 5.2 mV FWHM and 22,505 tall, and `find_peaks` rejected it for being
# narrower than the threshold. The second-derivative pass then picked up a
# shoulder at 1.5825 — a third the height, 7,662 — and that is what the run
# reported, fitted, tracked and exported. Because it was flagged a shoulder
# it also carried a FABRICATED prominence (0.1 x height, see the note in
# `detect_peaks_single`) and was excluded from every replicate statistic:
# the LTO discharge drift was reported as "—" for a triplicate where all
# three cells have a clean, tall, well-resolved delithiation peak.
#
# LOWERING THE THRESHOLD IS THE WRONG FIX and this table is not used. Measured
# on the LTO triplicate over its settled cycles: the discharge peak is only
# found as primary at 2.5 mV or below, and at 2.5 mV the CHARGE side admits
# two or three peaks per cycle where there is one. No single width serves
# both, because width is not what is actually wrong.
#
# What is wrong is that the CURVE'S OWN MAXIMUM was not in the peak list at
# all. See `_ensure_curve_maximum`.
PROFILE_MIN_WIDTH_MV = {"sharp": 2, "moderate": 4, "broad": 5}  # unused
SHOULDER_DETECTION = True
SHOULDER_MIN_SEPARATION_MV = 20
SHOULDER_HEIGHT_FRACTION = 0.10
SHOULDER_D2_PROMINENCE_FRACTION = 0.08
# THE PARENT MUST BE A FEATURE. Replaces SHOULDER_D2_ABSOLUTE_FLOOR in
# 1.9.0.57, one gate out and one gate in — the parameter count is unchanged.
#
# What the old floor was for is still right: local gates alone have a failure
# mode that is the mirror image of the global ones they replaced. On a long,
# almost-flat, slowly-rising tail — NNM below 3.0 V — the parent's local
# height is small, so a fraction of it is a very low bar and every noise
# ripple clears it. The first build of the local pass put four markers in
# that tail (2.48, 2.57, 2.67 V and friends).
#
# What was wrong was the QUANTITY chosen to say so. The old floor asked
# whether the CANDIDATE's curvature was large compared with the whole curve's
# d2 range — and the whole curve's d2 range is set by the tallest, sharpest
# feature, which may be 400 mV away and has nothing to do with the candidate.
# That is the same fault the local height gate was built to cure, left in
# place on the one test that kept a global denominator.
#
# It shows as three symptoms, all measured on the NNM triplicate (40 cycles
# per cell, 247 candidates, labelled by their PARENT: a candidate hanging off
# the 3.2/3.3 V redox peak is REAL, one hanging off the sub-2.6 V
# polarisation tail is RIPPLE):
#
#   1. The two populations do not separate on d2 share. REAL runs from 0.001
#      to 0.209 (p25 0.040); RIPPLE runs to 0.100 (p75 0.027). The threshold
#      0.04 sits inside both. It kept 76% of the real shoulders and admitted
#      8.7% of the ripples.
#
#   2. IT IS NOT EVEN ACROSS CELLS. Real shoulders kept: cell A 73.6%, cell B
#      67.2%, cell C 86.7%. Cell B's 3.15 V discharge shoulder scores
#      0.035-0.040 against a threshold of 0.040 — it is half its parent's
#      height, plainly visible in the raw trace, and rejected in 58 of 67
#      cycles, so cell B fitted one component where A and C fitted two.
#
#   3. IT FADES WITH CYCLE, which is worse than either. On discharge the old
#      gate kept 90.9% of real shoulders in cycles 1-5, then 60.0%, 60.9%,
#      50.0% through cycles 6-40. A gate that loses a feature more often as
#      the cell ages manufactures a decline that is not in the data. This is
#      the mechanism behind the P3 structural claim retracted in 1.9.0.56.
#
# The parent's own significance separates them cleanly, because that is what
# actually differs: a shoulder sits on a redox peak, a ripple sits on a
# polarisation tail. Parent local height as a share of the curve's range:
#
#     population    n     min      p25     median    p75      max
#     REAL        171   0.022    0.421    0.499    0.534    0.723
#     RIPPLE       46   0.011    0.056    0.085    0.120    0.139
#
# RIPPLE never exceeds 0.139; REAL's quartile is 0.421. 0.20 sits above the
# whole ripple population with margin and well below the real one:
#
#     rule                     REAL kept   ripple kept   per cell (A/B/C)
#     d2_frac    >= 0.04         76.0%        8.7%       73.6 / 67.2 / 86.7
#     parent_share >= 0.20       88.3%        0.0%       84.9 / 89.7 / 90.0
#
# and on discharge the new rule keeps 100% in every cycle band, 1-5 through
# 21-40 — the fade is gone. On NMC it admits two more candidates (real
# shoulders at 3.65 V on the 3.77 V charge peak, scored 0.037 and 0.038 by
# the old gate) and drops none. On LTO it changes nothing: LTO produces no
# shoulder candidates at all in 30 half-cycles.
#
# A d2 floor was tried ALONGSIDE this at 0.010, 0.015 and 0.020 and changed
# not one candidate at any of them — parent_share already excludes everything
# the noise floor existed to exclude. A parameter that does not pay for
# itself does not go in (Appendix A6), so the old gate is removed rather than
# demoted.
#
# The threshold is not a knife edge in either direction: 0.15, 0.20, 0.25 and
# 0.30 all admit zero ripples and keep 88.9%, 88.3%, 87.7%, 87.7% of the real
# shoulders.
SHOULDER_PARENT_SHARE_MIN = 0.20
# A peak's prominence must clear DEFAULT_PROMINENCE_FRACTION of the
# half-cycle's FULL RANGE. That rule has a failure mode, inherited from 1.8.x
# and visible on P3 cell A: where one feature dominates, it sets the threshold
# for everything else. On cycle 10 discharge the 3.539 V two-phase peak is
# 2075 mAh/V/g and the floor is therefore 104, so the 3.156 V feature —
# prominence 44.5, and a peak this cell tracks for a hundred cycles — is
# invisible. On the same cycle's CHARGE the range is 383, the floor is 19, and
# all four peaks are found. One half-cycle, not the other, for no reason to do
# with the chemistry.
#
# Worse, it flickers: 1.8.7 found 3.148 V on cycle 9 and 3.151 V on cycle 11
# and missed it on cycle 10. A feature that appears and disappears between
# consecutive cycles is precisely what corrupts tracking and area retention.
#
# So a peak is also kept if it is a distinct feature ON ITS OWN SCALE: its
# prominence is at least LOCAL_PROMINENCE_FRACTION of its own height (it is a
# peak, not a ripple on somebody's flank) AND its height is at least
# MIN_HEIGHT_FRACTION of the largest (it is not noise). Both are ratios, so
# neither can be dominated by an unrelated feature elsewhere in the window.
# A FLOOR REFERENCED TO THE NOISE, NOT TO THE TALLEST FEATURE.
#
# `MIN_HEIGHT_FRACTION` below is a fraction of the curve's maximum, which says
# nothing about whether a bump is above the noise. On the LTO triplicate that
# admitted 19 components that are not features at all: 2-5% of their
# half-cycle's largest, 31-87 mV away from it, and — the decisive test — NOT
# REPRODUCIBLE. Same 64 half-cycles, only the processing changed:
#
#     1.0 mV bin, unsmoothed (the run's own setting)   19 extras
#     0.5 mV bin                                        0 extras
#     2.0 mV bin                                        3 extras
#     1.0 mV bin, smoothed x3                           4 extras
#
# ...while the main peak's mean position moved by 0.2 mV across all four. The
# extras' own potentials scatter over 150 mV across cells and cycles. A process
# sits at a potential; these do not. Appendix A7's caution is that bin-width
# stability is NECESSARY BUT NOT SUFFICIENT — these fail necessity.
#
# WHY NOT KEY IT TO THE MECHANISM. The obvious alternative is a stricter floor
# on a `two_phase` run, since the Gibbs phase rule allows one peak in the
# two-phase field. Rejected, for three reasons. The phase rule constrains the
# FIELD, and `PROFILE_MIN_DISTANCE_MV = 30` already forbids a second component
# within 30 mV of the plateau — every one of these 19 is 31-87 mV away, outside
# what the phase rule speaks about. The mechanism is classified from ONE
# reference half-cycle before anything is fitted, so keying detection to it
# would silently delete the features most worth catching later — a second phase
# emerging on ageing, surface pseudocapacitance, an additive or electrolyte
# process. And it would encode a thermodynamic EXPECTATION as a measurement
# THRESHOLD, which makes the tool assert the phase rule rather than test
# against it. The expectation belongs on the page as a finding; see
# `report._phase_rule_sentence`.
#
# THE RULE IS A CONJUNCTION, because neither half separates on its own.
# Measured over 320 components and 112 half-cycles, all three chemistries:
#
#                          n     min SNR   5th pct    median
#     LTO main peaks      64        38.5      44        63
#     LTO extras          19         1.5       -         6.0  (max 10.0)
#     NMC, all           126        16.5      21.2      39.9
#     NNM, all           111        10.5      15.0     165
#
# A single SNR floor cannot work: LTO's worst artefact is 10.0 and NNM's
# weakest REAL component is 10.5. But those populations are nothing alike on
# the other axis — the artefacts are 2-5% of their half-cycle's largest, the
# low-SNR NNM components 28-41%. Substantial features on a noisy curve, not
# bumps on a clean one. So a candidate is rejected only when it is BOTH below
# the noise floor AND a small share of its own half-cycle:
#
#     k = 12, f = 0.10   ->  19 of 19 artefacts removed, 0 of 301 real
#     k = 12, f = 0.15   ->  19 of 19,                   0 of 301
#     k = 15, f = 0.10   ->  19 of 19,                   0 of 301
#     k = 20, f = 0.10   ->  19 of 19,                   1 of 301
#
# Not a knife edge — anything in k = 12-15, f = 0.10-0.20 gives the same split,
# with margin on both axes (worst artefact 10.0 against 12 and 5.0% against
# 10%; closest real component 10.5 but 27.7% of its curve).
#
# Mechanism-blind, chemistry-blind and scale-free: a statement about what the
# measurement can support, in the same family as `sigma_min` from the histogram
# bin and `WIDTH_MIN_SAMPLES` from the sampling interval.
NOISE_FLOOR = True
NOISE_SNR_MIN = 12.0  # prominence / local noise
NOISE_SHARE_MIN = 0.10  # ...unless it is this much of the largest
NOISE_WINDOW_BINS = 30  # half-width of the local noise estimate


def local_noise(signal, index=None, half=NOISE_WINDOW_BINS):
    """
    High-frequency noise near `index`, from the curve itself.

    `d = y[j] - (y[j-1] + y[j+1]) / 2` has variance 1.5 sigma^2 for
    independent noise and is blind to a smooth peak, so it measures scatter
    without being fooled by curvature.

    Two departures from the textbook estimator, both forced by the data:

    - The scale comes from `mean(|d|)` (times sqrt(pi/2)) rather than the MAD.
      A histogram dQ/dV is heavily quantised — long runs of identical values,
      including exact zeros — so the MAD is frequently exactly 0 and the
      estimate collapses.
    - It is LOCAL. dQ/dV noise scales with the signal rather than being
      constant across the window, so a whole-curve estimate is set by
      whichever region happens to be quietest.

    Returns NaN where there is too little to measure, which the caller must
    read as "no opinion" rather than as "no noise".
    """
    y = np.asarray(signal, float)
    if y.size < 7:
        return np.nan
    if index is None:
        seg = y
    else:
        lo = max(0, int(index) - int(half))
        hi = min(y.size, int(index) + int(half) + 1)
        seg = y[lo:hi]
        if seg.size < 7:
            seg = y
    d = seg[1:-1] - 0.5 * (seg[:-2] + seg[2:])
    if d.size == 0:
        return np.nan
    # E|N(0, s)| = s * sqrt(2/pi), so s = mean|d| * sqrt(pi/2); then divide
    # through by sqrt(1.5) for the Laplacian's own variance inflation.
    return float(np.mean(np.abs(d)) * 1.2533141 / np.sqrt(1.5))


LOCAL_PROMINENCE_FRACTION = 0.25
MIN_HEIGHT_FRACTION = 0.02
LOCAL_PROMINENCE = True  # set False to restore 1.8.x exactly

EDGE_EXCLUSION_MV = 50
# Minimum visible fraction of a truncated peak (estimated from half-width
# asymmetry) to retain it for fitting. 0.60 means at least 60% of the peak's
# expected full width must be visible. Below this the peak is discarded: only
# one flank is present and a fitted centre would be an extrapolation.
TRUNCATION_VISIBLE_FRACTION = 0.60
# ...but do not DISCARD on it. The visible fraction is a geometric estimate,
# made before the fit, of a quantity the fit measures directly — and it is
# measured from the half-maximum crossing on a flank that is partly outside
# the window, which is the noisiest place on the curve to measure anything.
# Discarding on it removes a component from the model, which changes the fit
# of every other component in the half-cycle; flagging it costs nothing and
# the post-fit test (`analyse.TRUNCATION_SIGMAS`) then answers the question
# properly, in the same units as everything else in the parameter table.
# Set True to restore 1.8.7's behaviour.
TRUNCATION_DROP = False

# Bands from `Module 1b` / `analyse` that a reference cycle may carry. A band
# outside this set — 'ANOMALOUS', 'unknown', 'too few records' — disqualifies
# a cycle from anchoring a dataset. 'too few records' is excluded deliberately:
# it does not mean the half-cycle was fine, it means we could not tell.
SOUND_BANDS = ("clean", "suspect")

# --- neighbour-informed seeding ------------------------------------------
# Detection is a threshold test on one half-cycle, so a peak that is genuinely
# present can fall below the prominence floor in a single cycle — noise, a
# slightly shallower feature, a rate step — and reappear in the next. Left
# alone, that cycle contributes a "disappeared" to the peak's history and a
# gap to its area trend, and the gap looks like degradation.
#
# So where a reference peak has no detected counterpart in a half-cycle that
# was otherwise fitted, a seed is placed at the position that peak occupied in
# the NEAREST cycle that did detect it. The fit then either finds something
# there or does not; either answer is informative, and neither is a gap.
#
# A component seeded this way is a DIFFERENT KIND OF NUMBER from one detected
# in the cycle it is reported for, and the parameter table says so:
# `is_inferred` is True and `reliable` is False, so it appears in the table and
# in the tracked history but never in a headline trend. An always-False
# provenance flag would be worse than none, because it reads as an assurance.
# --- a detection that finds this many peaks has not found peaks -----------
# Thirty-five components on a curve with four features is not a hard fitting
# problem, it is a detection failure — and left alone it is an EXPENSIVE one:
# every component adds four correlated parameters, and a half-cycle seeded
# that way took minutes to converge onto an answer that meant nothing. It
# happens when detection runs on an unsmoothed curve, so the guard also says
# what to check.
#
# The cap keeps the strongest peaks by prominence, which is the right ordering
# — a shoulder's placeholder prominence is 10% of its height, so genuine
# shoulders survive alongside their parents — and says how many it dropped.
MAX_PEAKS_PER_HALF_CYCLE = 12
INFER_MISSING_SEEDS = True
SEED_INFERENCE_TOLERANCE_MV = 80.0


# =============================================================================
# PORTED FROM 1.8.7 MODULE 3 — do not edit without a bit-identity regression
# =============================================================================


def _voltage_to_samples(mV, voltage_array):
    """Convert millivolts to approximate sample count."""
    if len(voltage_array) < 2:
        return 1
    avg_step = np.abs(np.diff(voltage_array)).mean()
    return max(1, int(round((mV / 1000.0) / avg_step))) if avg_step > 0 else 1


def _ensure_curve_maximum(peaks_df, voltage, signal, *, min_distance_mV):
    """
    The tallest point of a half-cycle's dQ/dV is a peak. Make sure it is in.

    `find_peaks` gates on WIDTH, and on a sharp two-phase material the real
    feature can fail that gate: LTO's delithiation peak is 22,505 tall and
    about 5 mV wide against a 5 mV minimum, so it was rejected outright. The
    second-derivative pass then picked up an inflection 3 mV away, a third
    the height, and THAT was reported as the dataset's discharge feature —
    fitted, tracked, and (being flagged a shoulder) excluded from every
    replicate statistic.

    1.9.0.18 tried to fix this by promoting any shoulder that was the tallest
    row IN THE PEAK TABLE. That was wrong twice over. A lone shoulder is
    trivially the tallest row in a one-row table, so it was always promoted;
    and it promoted the INFLECTION rather than finding the peak beside it, so
    the reported feature was still the wrong one — now mislabelled primary,
    with a prominence of 0, and admitted to the statistics it had rightly
    been excluded from. scipy said so 32 times per run
    (`PeakPropertyWarning: some peaks have a prominence of 0`) and the run
    carried on regardless.

    The rule here is the one that is true by construction: whatever else a
    half-cycle contains, the global maximum of |dQ/dV| is a feature of it.
    If no detected peak sits within `min_distance_mV` of that maximum, it is
    added, as a primary, with its prominence MEASURED. Nothing is relabelled
    and no threshold is loosened, so the charge side — where lowering the
    width gate admitted two and three peaks per cycle — is untouched.
    """
    if voltage is None or signal is None or len(signal) < 3:
        return peaks_df
    y = np.asarray(signal, float)
    if not np.isfinite(y).any():
        return peaks_df
    i = int(np.nanargmax(y))
    if i == 0 or i == len(y) - 1:
        return peaks_df  # an edge is a truncation, not a peak
    v_max = float(voltage[i])
    if peaks_df is not None and not peaks_df.empty:
        near = (peaks_df["voltage"] - v_max).abs() < min_distance_mV / 1000.0
        if bool(near.any()):
            # Already represented. If the representative is a SHOULDER but
            # the curve's maximum is a true local maximum, the label is the
            # thing that is wrong, so correct it — measured, not assumed.
            j = peaks_df.index[near][0]
            if bool(peaks_df.loc[j, "is_shoulder"]):
                from scipy.signal import peak_prominences

                prom = float(peak_prominences(y, [i])[0][0])
                if prom > 0:
                    peaks_df = peaks_df.copy()
                    peaks_df.loc[
                        j,
                        [
                            "voltage",
                            "height",
                            "prominence",
                            "is_shoulder",
                            "peak_index",
                        ],
                    ] = [v_max, float(y[i]), prom, False, i]
                    if "prominence_estimated" in peaks_df.columns:
                        peaks_df.loc[j, "prominence_estimated"] = False
            return peaks_df

    from scipy.signal import peak_prominences

    prom = float(peak_prominences(y, [i])[0][0])
    if prom <= 0:
        # Not a local maximum at all — a monotonic run or a flat top. Adding
        # it would be inventing a peak, which is the failure this whole
        # module is written against.
        return peaks_df
    row = {
        c: np.nan
        for c in (
            peaks_df.columns if peaks_df is not None and not peaks_df.empty else []
        )
    }
    row.update(
        voltage=v_max,
        height=float(y[i]),
        height_original=float(y[i]),
        prominence=prom,
        prominence_estimated=False,
        peak_index=i,
        is_shoulder=False,
        is_truncated=False,
        peaks_capped=False,
        width_V=np.nan,
        left_voltage=np.nan,
        right_voltage=np.nan,
    )
    if peaks_df is None or peaks_df.empty:
        out = pd.DataFrame([row])
    else:
        out = pd.concat([peaks_df, pd.DataFrame([row])], ignore_index=True)
    out["peak_id"] = range(1, len(out) + 1)
    return out.sort_values("voltage").reset_index(drop=True)


def _deduplicate_peaks(peaks_df, min_separation_mV=20, primary_min_separation_mV=None):
    """
    Remove peaks closer than min_separation_mV after merge.

    When both find_peaks and the second-derivative pass detect the same
    physical feature at nearly identical voltages, this keeps the detection
    with higher prominence and drops the duplicate. Generalises to any
    system — LIB, NIB, or otherwise.

    `primary_min_separation_mV` is the wider rule applied when NEITHER peak
    is a shoulder: `min_distance_mV`, the parameter that defines what counts
    as two distinct peaks in the first place. The two passes each honoured
    it internally, but a peak from the scale-free rescue pass and one from
    the primary pass were only ever compared at the 20 mV shoulder
    threshold. On P3 cell A cycle 10 Discharge that admitted a pair 24 mV
    apart whose heights differ 30-fold, and the fit could not resolve them:
    the smaller component slid onto the larger, collapsed to zero area, and
    took a quarter of the larger peak's area and 0.074 of R2 with it. A
    shoulder legitimately sits close to its parent and keeps the narrow
    threshold.
    """
    if peaks_df.empty or len(peaks_df) < 2:
        return peaks_df

    df = peaks_df.sort_values("voltage").reset_index(drop=True)
    min_sep_V = min_separation_mV / 1000.0 + 1e-9  # float tolerance
    primary_sep_V = (
        (primary_min_separation_mV / 1000.0 + 1e-9)
        if primary_min_separation_mV
        else min_sep_V
    )

    def _sep(i, j):
        # TWO PRIMARIES ARE SEPARATED BY `min_distance_mV`, FULL STOP.
        #
        # This was `max(min_sep_V, primary_sep_V)`, and the reasoning behind
        # it was sound while `min_distance_mV` was 30 mV on every profile:
        # a primary pair should not be held to the NARROW shoulder rule, so
        # take the wider of the two. The moment a profile asks for a
        # separation below the 20 mV shoulder threshold, the `max` silently
        # refuses it — two genuine primaries could not be closer than 20 mV
        # whatever the caller set.
        #
        # Measured cost, on the clearest data we own: LTO cell C cycle 1
        # charge is a visible doublet at 1.5815 V (15,200) and 1.5875 V
        # (11,183), 6 mV apart, with a valley at 4,985 between them. Both
        # passes find both peaks; `_ensure_curve_maximum` keeps both; and
        # this line then dropped the second at every `min_distance_mV` from
        # 30 mV down to 3 mV. One component was fitted across the pair, at
        # 77% of the data's height and with the asymmetry ratio driven to
        # 17.8 — a lineshape parameter standing in for a peak. That is the
        # doublet that has been invisible to the pipeline since 1.8.7.
        #
        # `min_sep_V` still governs any pair involving a shoulder, which is
        # what it was written for: a shoulder legitimately sits close to its
        # parent, and the 20 mV rule stops the second-derivative pass
        # re-reporting a feature the primary pass already has.
        if "is_shoulder" in df.columns and not (
            bool(df.loc[i, "is_shoulder"]) or bool(df.loc[j, "is_shoulder"])
        ):
            return primary_sep_V
        return min_sep_V

    keep = [True] * len(df)
    for i in range(len(df) - 1):
        if not keep[i]:
            continue
        for j in range(i + 1, len(df)):
            if not keep[j]:
                continue
            if df.loc[j, "voltage"] - df.loc[i, "voltage"] < _sep(i, j):
                # Too close — drop lower prominence
                if df.loc[i, "prominence"] >= df.loc[j, "prominence"]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
            elif df.loc[j, "voltage"] - df.loc[i, "voltage"] >= max(
                min_sep_V, primary_sep_V
            ):
                # Past the WIDEST threshold any later pair could use, so
                # nothing further can be a duplicate of i. Breaking on the
                # pair's own threshold would skip a non-shoulder pair that
                # follows a nearer shoulder.
                break

    result = df[keep].reset_index(drop=True)
    result["peak_id"] = range(1, len(result) + 1)
    return result


def _assess_truncation(
    centre_v,
    left_v,
    right_v,
    data_v_min,
    data_v_max,
    visible_fraction_threshold,
    edge_tol_V=0.002,
):
    """
    Assess whether a peak is truncated by the voltage window edge.
    Uses half-width asymmetry: for a symmetric peak, the full width would
    be 2 * (unclipped half-width). The visible fraction is estimated as
    (left_hw + right_hw) / (2 * reference_hw), where reference_hw is the
    half-width on the unclipped side.
    Parameters
    ----------
    centre_v : float
        Peak centre voltage.
    left_v : float
        Left half-maximum voltage (from peak_widths).
    right_v : float
        Right half-maximum voltage (from peak_widths).
    data_v_min, data_v_max : float
        Voltage range of the data array.
    visible_fraction_threshold : float
        Minimum visible fraction to retain the peak (e.g. 0.60).
    edge_tol_V : float
        Tolerance for deciding whether a width base is "at the edge" (V).
    Returns
    -------
    is_truncated : bool
        True if peak is retained but flagged as truncated.
    drop : bool
        True if peak should be discarded (< threshold visible).
    """
    left_hw = centre_v - left_v
    right_hw = right_v - centre_v
    near_upper = right_v >= data_v_max - edge_tol_V
    near_lower = left_v <= data_v_min + edge_tol_V
    if near_upper and not near_lower:
        # Right side clipped — use left_hw as reference
        if left_hw > 0:
            visible = (left_hw + right_hw) / (2.0 * left_hw)
        else:
            visible = 0.0
        if visible >= visible_fraction_threshold:
            return True, False  # truncated but usable
        else:
            return False, True  # discard
    elif near_lower and not near_upper:
        # Left side clipped — use right_hw as reference
        if right_hw > 0:
            visible = (left_hw + right_hw) / (2.0 * right_hw)
        else:
            visible = 0.0
        if visible >= visible_fraction_threshold:
            return True, False
        else:
            return False, True
    elif near_lower and near_upper:
        # Clipped on BOTH flanks: neither half-maximum is inside the window,
        # so the visible fraction cannot be measured at all. Certifying that
        # as "not truncated" was the one case where the most severely
        # truncated peak got the most confident label. Flag it and let the
        # fit proceed — its area is a lower bound.
        return True, False
    else:
        return False, False


def _shoulder_sigma(neg_d2, i):
    """
    A shoulder's width, measured — from the inflection points either side.

    `peak_widths` cannot help here: a shoulder is not a local maximum, so
    scipy never measured a width for it and detection left the value at zero.
    That zero was then printed in a results table beside measured widths, and
    (worse) it meant the fit had no starting width for the majority of
    components on a broad profile, where most reference peaks are shoulders.

    But the second derivative already knows. For a Gaussian of width sigma,
    d2 changes sign exactly at the inflection points, which sit at
    centre +/- sigma. `neg_d2` is positive across the top of a feature and
    goes negative outside it, so walking out from the candidate to the first
    sign change on each side measures 2*sigma directly. It is the same array
    the shoulder was found in, so nothing new is assumed.

    Returns sigma in the units of the sample index -> converted by the caller,
    or NaN if either crossing runs off the end of the curve.
    """
    n = len(neg_d2)
    if i <= 0 or i >= n - 1:
        return np.nan
    L = i
    while L > 0 and neg_d2[L] > 0:
        L -= 1
    R = i
    while R < n - 1 and neg_d2[R] > 0:
        R += 1
    if neg_d2[L] > 0 or neg_d2[R] > 0:
        return np.nan  # never crossed: the lobe runs off the window
    half = (R - L) / 2.0
    return half if half > 0 else np.nan


# THE SECOND-DERIVATIVE WINDOW, IN VOLTS RATHER THAN IN SAMPLES.
#
# `_flank_shoulders` smooths with `savgol_filter(signal, w, 3, deriv=2)` and
# `w` was `min(21, max(7, n // 5))` — a count of SAMPLES, fixed, whatever the
# bin width or the chemistry. At the bin widths this build actually produces
# that window spans:
#
#     LTO   327 bins at 2.41 mV  ->  21 samples =  51 mV, against a  10 mV FWHM
#     NNM   334 bins at 6.33 mV  ->  21 samples = 133 mV, against a  70 mV FWHM
#     NMC   205 bins at 6.82 mV  ->  21 samples = 143 mV, against a 200 mV FWHM
#
# So it is five times the feature on LTO, twice on NNM, and about right on
# NMC — which is the one chemistry it was never a problem for. A cubic fitted
# over five times a peak's width cannot see structure on that peak's flank;
# that is the whole complaint behind item 38.
#
# The scale that matters is the FEATURE's, not the curve's and not the grid's,
# and detection already knows it: `peak_widths` measured every primary before
# this function is called. The window is set from the NARROWEST primary,
# because a window wide enough to blur the sharpest feature present will blur
# everything narrower than it too, and a shoulder is by definition narrower
# than its parent.
#
# `SHOULDER_D2_WINDOW_FRACTION` is the fraction of that width the window
# spans. Half of it is the natural choice: a Savitzky-Golay cubic reproduces a
# curve faithfully over a span where the curve is close to cubic, and half a
# peak width is about where that holds. The result is flat over 0.25-1.0 on
# every chemistry, so the exact value is not doing the work — the clamp is.
#
# WHAT IT RECOVERS. NNM's 3.15 V discharge shoulder, the labelled true
# positive, detected in 30 half-cycles of 30 (cell A) and 28 of 30 (cell C)
# at every floor from 7 to 13 samples, against 18 and 19 under the old rule.
# In the pipeline that feature's coherence goes from 18 pairs and a split
# reference set to 125 pairs and a single one — see the note in
# `Ratatosk_1_9_0_61`. LTO is bit-for-bit unchanged: it produces no shoulder
# candidates at any window.
#
# WHY THE FLOOR IS 11 AND NOT 7. Differentiating over a shorter span passes
# more noise, and the synthetic ground truth is where that shows. Failures on
# `tests/test_synthetic_recovery.py`, and the time it takes:
#
#     floor        failures    total time    NNM shoulder (A / C)
#     old rule         9          75 s          18/30, 19/30
#     13              12         254 s          30/30, 28/30
#     11              12         189 s          30/30, 28/30
#      7              15         721 s          30/30, 28/30
#
# At 7 samples the 1.0%-noise scenario alone goes from 34 s and 3 failures to
# 575 s and 7, fitting nine components where the truth has four: the window is
# short enough to differentiate the noise. 11 keeps the whole real-data gain,
# costs three synthetic failures rather than six, and halves the time penalty.
# The existing noise floor cannot help here — it tests prominence against the
# noise of the SIGNAL, and what degrades is the prominence in the second
# DERIVATIVE. A d2-referenced floor is what would let this go narrower, and
# is the next thing to build rather than a reason to hold the window wide.
#
# The clamps can only ever NARROW the window relative to the old rule.
SHOULDER_D2_WINDOW_FRACTION = 0.5
SHOULDER_D2_WINDOW_MIN = 11
SHOULDER_D2_WINDOW_MAX = 21


def _d2_window(voltage, signal, peak_indices, n):
    """Odd sample count for the shoulder second derivative.

    Scaled to the narrowest primary's measured width, clamped, and odd.
    Falls back to the old sample-count rule when no width can be measured.
    """
    w_old = min(SHOULDER_D2_WINDOW_MAX, max(SHOULDER_D2_WINDOW_MIN, n // 5))
    w = w_old
    try:
        idx = np.asarray(sorted(int(i) for i in peak_indices), dtype=int)
        if idx.size:
            widths = peak_widths(np.asarray(signal, float), idx, rel_height=0.5)[0]
            widths = widths[np.isfinite(widths) & (widths > 0)]
            if widths.size:
                w = int(round(SHOULDER_D2_WINDOW_FRACTION * float(np.min(widths))))
    except Exception:
        w = w_old
    w = int(min(SHOULDER_D2_WINDOW_MAX, max(SHOULDER_D2_WINDOW_MIN, w)))
    if w % 2 == 0:
        w += 1
    return min(
        w,
        SHOULDER_D2_WINDOW_MAX
        if SHOULDER_D2_WINDOW_MAX % 2
        else SHOULDER_D2_WINDOW_MAX - 1,
    )


def _flank_shoulders(
    voltage,
    signal,
    peak_indices,
    *,
    min_separation_mV=SHOULDER_MIN_SEPARATION_MV,
    height_fraction=SHOULDER_HEIGHT_FRACTION,
    d2_prominence_fraction=SHOULDER_D2_PROMINENCE_FRACTION,
    parent_share_min=SHOULDER_PARENT_SHARE_MIN,
    v_min_edge=-np.inf,
    v_max_edge=np.inf,
):
    """
    Shoulders, found ON THE FLANKS OF PRIMARY PEAKS, judged locally.

    The primaries are found first. For each one, each flank is walked out to
    the trough separating it from its neighbour (or to the window edge), and
    the search for a shoulder happens INSIDE that span, against that span's
    own baseline and that peak's own height.

    This replaces a global pass that had three faults, all of which follow
    from asking the question of the whole curve at once:

      1. It placed shoulders IN VALLEYS. A trough is a maximum of the
         negative second derivative just as a shoulder is, so the trough
         between two peaks was detected in most cycles of the NNM
         triplicate. A trough also sits at the same voltage every cycle, so
         it scored as highly "persistent" and a persistence test could not
         catch it. Here a candidate must lie strictly between a primary and
         its trough, at least `min_separation_mV` from both, so the trough
         is not a candidate at all.

      2. BOTH GATES WERE GLOBAL: the height floor was a fraction of the
         TALLEST peak in the curve and the d2 prominence was scaled to the
         whole curve's d2 range. On a layered oxide the tallest charge peak
         is a median of 7.6x the smallest, so a clear shoulder on a small
         feature was judged against a peak seven times taller. Both gates
         are now measured against the parent peak: its height above its own
         local baseline, and the d2 range within its own flank.

      3. A REAL FEATURE FLICKERED between primary, shoulder and absent
         across cycles, because its prominence sat near two global
         thresholds and crossed them when a DIFFERENT peak grew. A local
         gate moves with its own parent, so a feature's classification no
         longer depends on what the rest of the curve is doing.

    A fourth global gate survived that change until 1.9.0.57 — a noise floor
    scaled to the whole curve's d2 range — and had all three faults over
    again: it rejected cell B's 3.15 V shoulder in 58 of 67 cycles, kept
    different fractions of the same feature in different cells of one
    triplicate, and lost the feature progressively as the cell aged. It is
    replaced here by `parent_share_min`, which asks the question the floor
    was really for — is the PARENT a feature, or a bump on a flat tail — of
    the parent rather than of the candidate. See the constant for the
    measurement.

    Returns a list of `(index, sigma_V)` pairs: the shoulder's position, and
    a WIDTH MEASURED FROM THE SAME CONSTRUCTION that found it (see
    `_shoulder_sigma`), or NaN where that could not be measured.
    """
    out = []
    n = len(signal)
    if n < 20 or len(peak_indices) == 0:
        return out
    pk = np.asarray(sorted(int(i) for i in peak_indices), dtype=int)
    min_sep = max(3, _voltage_to_samples(min_separation_mV, voltage))

    v_step = float(np.mean(np.abs(np.diff(voltage))))
    w = _d2_window(voltage, signal, peak_indices, n)
    try:
        neg_d2 = -savgol_filter(signal, w, 3, deriv=2, delta=v_step)
    except Exception:
        return out
    curve_range = float(np.nanmax(signal) - np.nanmin(signal))

    # The trough between each adjacent pair of primaries, and the window
    # ends outside the outermost pair. These are the flank boundaries.
    bounds = [0]
    for a, b in zip(pk, pk[1:]):
        bounds.append(int(a + np.argmin(signal[a : b + 1])) if b > a else int(a))
    bounds.append(n - 1)

    for i, p in enumerate(pk):
        left_edge, right_edge = bounds[i], bounds[i + 1]
        for lo, hi in ((left_edge, p), (p, right_edge)):
            if hi - lo < 2 * min_sep + 3:
                continue
            span = slice(lo, hi + 1)
            # The parent's height above the baseline of THIS flank — the
            # trough (or window edge) the flank runs down to.
            base = float(min(signal[lo], signal[hi]))
            local_height = float(signal[p]) - base
            if not np.isfinite(local_height) or local_height <= 0:
                continue
            # IS THE PARENT A FEATURE? The local gates below say whether a
            # candidate matters to its parent; this says whether the parent
            # is worth having shoulders on. Without it, a flat tail whose
            # parent is barely a rise admits every ripple — the local gates'
            # own failure mode. Asked of the parent, not of the candidate,
            # and once per flank rather than once per candidate.
            if curve_range > 0 and local_height / curve_range < parent_share_min:
                continue
            seg = neg_d2[span]
            seg_range = float(np.nanmax(seg) - np.nanmin(seg))
            if not np.isfinite(seg_range) or seg_range <= 0:
                continue
            try:
                cand, _props = find_peaks(
                    seg, prominence=d2_prominence_fraction * seg_range, distance=min_sep
                )
            except Exception:
                continue
            for c in cand:
                idx = int(lo + c)
                # ON THE FLANK, not at either end of it: away from the
                # parent, and away from the trough. This is the check whose
                # absence put markers in valleys.
                if abs(idx - p) < min_sep:
                    continue
                if min(abs(idx - left_edge), abs(idx - right_edge)) < min_sep:
                    continue
                if not (lo < idx < hi):
                    continue
                if voltage[idx] < v_min_edge or voltage[idx] > v_max_edge:
                    continue
                # LOCAL height gate: a fraction of the parent's own rise,
                # measured from this flank's own baseline.
                if (float(signal[idx]) - base) < height_fraction * local_height:
                    continue
                # And it must sit on the descending side: the curve at the
                # candidate is below the parent and above the flank's foot.
                if not (base <= signal[idx] <= signal[p]):
                    continue
                out.append((idx, _shoulder_sigma(neg_d2, idx)))
    _seen, _uniq = set(), []
    for _i, _sg in sorted(out):
        if _i not in _seen:
            _seen.add(_i)
            _uniq.append((_i, _sg))
    return _uniq


def detect_peaks_single(
    voltage,
    dqdv,
    orientation=None,
    prominence_fraction=DEFAULT_PROMINENCE_FRACTION,
    min_distance_mV=DEFAULT_MIN_DISTANCE_MV,
    min_width_mV=DEFAULT_MIN_WIDTH_MV,
    shoulder_detection=SHOULDER_DETECTION,
    shoulder_min_separation_mV=SHOULDER_MIN_SEPARATION_MV,
    shoulder_height_fraction=SHOULDER_HEIGHT_FRACTION,
    shoulder_d2_prominence_fraction=SHOULDER_D2_PROMINENCE_FRACTION,
    edge_exclusion_mV=EDGE_EXCLUSION_MV,
    truncation_visible_fraction=TRUNCATION_VISIBLE_FRACTION,
    local_prominence=LOCAL_PROMINENCE,
    local_prominence_fraction=LOCAL_PROMINENCE_FRACTION,
    min_height_fraction=MIN_HEIGHT_FRACTION,
):
    """
    Detect peaks in a single half-cycle with shoulder detection, dedup,
    and truncation handling.
    Peaks near the voltage window edge are assessed using half-width
    asymmetry. If ≥ truncation_visible_fraction of the expected peak is
    present, the peak is retained with is_truncated=True. Below this,
    it is discarded. This produces consistent detection across cycles
    for features that shift into or out of the voltage window.
    """
    if len(voltage) < 10:
        return pd.DataFrame()

    # 1.8.7 wrote `-dqdv if step == 'Discharge' else dqdv`, taking the sign
    # of the lobe FROM THE LABEL. That is the same class of assumption that
    # flagged all 64 LTO half-cycles anomalous in Module 1b.
    #
    # It is not, however, the label's DIRECTION that matters here — using
    # `Dataset.step_direction` in its place inverts every LTO half-cycle and
    # moves the 1.53 V peak by 30-40 mV, because for a negative electrode the
    # useful half-cycle is the one where the cell voltage FALLS, and
    # `io._apply_electrode_convention` has already negated dQ/dV to suit.
    # After that convention, label, direction and sign of dQ/dV no longer
    # agree, and only one of the three is a property of the curve.
    #
    # So measure the curve. The signed area under dQ/dV over the half-cycle is
    # the net charge passed: large, unambiguous, and independent of every
    # label and convention upstream. Orienting on it reproduces 1.8.7 exactly
    # wherever 1.8.7's label was right — verified on all 246 fitted P3 cell A
    # half-cycles and all 19 LTO cell A half-cycles — and is right where a
    # label would not have been. `orientation` overrides it for a caller who
    # knows better; nothing in the pipeline passes one.
    if orientation is None:
        m = np.isfinite(voltage) & np.isfinite(dqdv)
        area = (
            float(trapezoid(dqdv[m], voltage[m]))
            if m.sum() > 1
            else float(np.nansum(dqdv))
        )
        orientation = 1.0 if area >= 0 else -1.0
    signal = dqdv.copy() if orientation > 0 else -dqdv
    # NaN-safe. `dqdv` comes from a to_numeric(errors="coerce"), so one NaN
    # made signal_range NaN, `NaN <= 0` False, and every find_peaks
    # prominence comparison silently False — zero peaks from a clean curve.
    signal_range = float(np.nanmax(signal) - np.nanmin(signal))
    if not np.isfinite(signal_range) or signal_range <= 0:
        return pd.DataFrame()

    data_v_min = voltage.min()
    data_v_max = voltage.max()
    v_min_edge = data_v_min + edge_exclusion_mV / 1000.0
    v_max_edge = data_v_max - edge_exclusion_mV / 1000.0
    min_dist_s = _voltage_to_samples(min_distance_mV, voltage)
    min_width_s = _voltage_to_samples(min_width_mV, voltage)

    # === PASS 1: Standard find_peaks ===
    peak_indices, properties = find_peaks(
        signal,
        prominence=prominence_fraction * signal_range,
        distance=min_dist_s,
        width=min_width_s,
    )

    # === PASS 1b: scale-free rescue ===
    # Everything find_peaks can see, then kept on two RATIOS rather than on a
    # fraction of a range one feature may dominate. See LOCAL_PROMINENCE_FRACTION.
    if local_prominence:
        # `prominence=0` asks find_peaks for EVERY local maximum, which on a
        # histogram curve includes single-bin spikes whose width at half
        # prominence is zero — and SciPy warns once per call about it. That is
        # a warning about the sweep's own design, not about the data: the
        # rescue pass exists precisely to look at maxima the first pass
        # rejected, and it then keeps them on two ratios rather than on width.
        # Eleven identical `PeakPropertyWarning: some peaks have a width of 0`
        # lines per dataset, each carrying an ipykernel temp path, buried the
        # profile block they were printed in the middle of. Silenced HERE, at
        # the one call that provokes it, rather than globally — and the fact it
        # was reporting is not lost: `width_undersampled` in the parameter
        # table is the same statement, measured, per component.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="some peaks have a width of 0")
            warnings.filterwarnings("ignore", category=PeakPropertyWarning)
            _all_idx, _all_props = find_peaks(
                signal, prominence=0, distance=min_dist_s, width=min_width_s
            )
        if len(_all_idx):
            _hmax = float(np.nanmax(signal))
            _keep = []
            for _i, _p in zip(_all_idx, _all_props["prominences"]):
                if _i in peak_indices:
                    continue
                _h = float(signal[_i])
                if _h <= 0 or _hmax <= 0:
                    continue
                if (
                    _p >= local_prominence_fraction * _h
                    and _h >= min_height_fraction * _hmax
                ):
                    _keep.append((_i, _p))
            if _keep:
                _ki = np.array([k[0] for k in _keep])
                _kp = np.array([k[1] for k in _keep])
                _order = np.argsort(np.concatenate([peak_indices, _ki]))
                peak_indices = np.concatenate([peak_indices, _ki])[_order]
                properties = {
                    "prominences": np.concatenate([properties["prominences"], _kp])[
                        _order
                    ]
                }

    # Edge handling (primary pass): genuine local maxima near the window
    # limits are NOT dropped here. They flow into the truncation assessment
    # below, which keeps a fully or sufficiently visible peak (flagging
    # is_truncated only where a flank is actually clipped) and discards a
    # peak only when less than truncation_visible_fraction of it is visible.
    # This stops the hard edge cut from discarding real near-limit features
    # before the truncation logic can judge them. The shoulder pass below
    # keeps its edge guard, since second-derivative detection is noise-prone
    # at the scan reversal.
    # (To reinstate a small primary guard against scan-reversal noise, drop
    #  peaks here whose voltage is within a few mV of data_v_min/data_v_max.)

    # === PASS 2: Second-derivative shoulder detection ===
    #
    # KNOWN DEFECTIVE, AND LEFT ALONE UNTIL IT CAN BE FIXED PROPERLY. Three
    # faults are established by inspection of the curves (not by summary
    # statistics, which hid all three):
    #
    # 1. SHOULDERS ARE PLACED IN VALLEYS. On the NNM triplicate a marker lands
    #    in the trough BETWEEN two peaks — 3.47 V on charge, 3.37 V on
    #    discharge — in most cycles. A trough is not a shoulder of anything.
    #    Because a trough sits at the same voltage every cycle it also scores
    #    as highly "persistent", so a persistence test cannot catch it.
    #
    # 2. BOTH GATES ARE GLOBAL. The height floor is a fraction of the TALLEST
    #    peak and the d2 prominence is scaled to the whole curve's range. On a
    #    layered oxide the tallest charge peak is a median of 7.6x the
    #    smallest, so a clear shoulder on a small feature is judged against a
    #    peak seven times taller. Making the floor local (a fraction of the
    #    NEAREST primary) was tried and measurably found more candidates —
    #    but most of the extra ones were valleys, so it amplified fault 1 and
    #    was reverted.
    #
    # 3. A REAL FEATURE FLICKERS BETWEEN CLASSES. The ~3.13 V discharge
    #    feature of NNM cell C is a PRIMARY at cycle 30, ABSENT at cycle 40,
    #    and reappears as a shoulder only if the d2 threshold is dropped
    #    eightfold. Its prominence sits near two different global thresholds
    #    and crosses them as neighbouring peaks grow. That is the root cause
    #    of the cycle-to-cycle instability, and it is in the PRIMARY pass as
    #    much as here.
    #
    # A correct pass would find the primaries first, then ask of each one
    # whether its own flanks carry either a bulge or an incompletely resolved
    # second peak, judged against that peak's own prominence and its own local
    # baseline — and would require the candidate to be ON a flank, which is
    # the check whose absence causes fault 1. See the audit note.
    # Shoulders, found on the flanks of the primaries just detected, judged
    # against each primary's own height and its own flank. See
    # `_flank_shoulders` for the three faults of the global pass this
    # replaced.
    shoulder_indices = []
    shoulder_sigma_V = {}
    if shoulder_detection:
        _pairs = _flank_shoulders(
            voltage,
            signal,
            peak_indices,
            min_separation_mV=shoulder_min_separation_mV,
            height_fraction=shoulder_height_fraction,
            d2_prominence_fraction=shoulder_d2_prominence_fraction,
            v_min_edge=v_min_edge,
            v_max_edge=v_max_edge,
        )
        shoulder_indices = [i for i, _ in _pairs]
        # sigma is measured in SAMPLES by `_shoulder_sigma`; convert here,
        # where the voltage axis is in scope.
        _vstep = (
            float(np.mean(np.abs(np.diff(voltage)))) if len(voltage) > 1 else np.nan
        )
        shoulder_sigma_V = {
            i: (sg * _vstep if np.isfinite(sg) else np.nan) for i, sg in _pairs
        }
        # A shoulder that coincides with a primary is not a second feature.
        if len(peak_indices):
            shoulder_indices = [
                i
                for i in shoulder_indices
                if np.min(np.abs(i - np.asarray(peak_indices)))
                >= max(3, _voltage_to_samples(shoulder_min_separation_mV, voltage))
            ]

    # === MERGE ===
    all_indices = sorted(set(list(peak_indices) + shoulder_indices))
    if len(all_indices) == 0:
        # THE CURVE'S OWN MAXIMUM IS NOT NOTHING. `_ensure_curve_maximum`
        # further down is the safety net for a half-cycle whose tallest
        # feature the primary pass rejected, but this early return jumped
        # over it, so a half-cycle that found nothing returned an empty
        # table instead of its own biggest peak.
        #
        # It was invisible because a second accident covered it. On LTO the
        # discharge peak at 1.5795 V carries the ENTIRE signal range
        # (prominence 23,110 of a 23,110 range) and is 2.6 mV wide — below
        # `min_width_mV = 5`, which the sharp profile sets — so the primary
        # pass has always rejected it. The old GLOBAL shoulder pass then
        # found it on the second derivative, and `_ensure_curve_maximum`
        # promoted it back to a primary. Two bugs cancelling: LTO's single
        # most important feature reached the table only because a shoulder
        # pass that was looking for something else happened to trip over it,
        # and it arrived with a fabricated width of 0.0 mV.
        #
        # Replacing the shoulder pass with one that hangs off the primaries
        # removed the accident and exposed the fault. The net is now reached
        # on this path too, so the behaviour no longer depends on it.
        #
        # THE UNDERLYING FAULT IS STILL THERE and is recorded in the audit
        # note: a minimum-width filter that rejects a peak carrying the whole
        # curve is wrong, and `min_width_mV` wants measuring on the sharp
        # profile rather than adjusting on a hunch.
        _rescued = _ensure_curve_maximum(
            pd.DataFrame(), voltage, signal, min_distance_mV=min_distance_mV
        )
        if _rescued is None or len(_rescued) == 0:
            return pd.DataFrame()
        return _rescued

    all_indices = np.array(all_indices)

    # Widths
    try:
        # `import warnings` used to sit HERE, inside the function body, which
        # made `warnings` a local name for the whole of `detect_peaks_single`
        # — so the module-level import could not be used anywhere earlier in
        # it. The module imports it now; this line was the only reason it
        # could not.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wr = peak_widths(signal, all_indices, rel_height=0.5)
        left_v_arr = np.interp(wr[2], np.arange(len(voltage)), voltage)
        right_v_arr = np.interp(wr[3], np.arange(len(voltage)), voltage)
        width_V = right_v_arr - left_v_arr
    except Exception:
        width_V = np.full(len(all_indices), 0.05)
        left_v_arr = voltage[all_indices] - 0.025
        right_v_arr = voltage[all_indices] + 0.025

    # A SHOULDER'S WIDTH, FROM ITS INFLECTION POINTS.
    # `peak_widths` returns 0 for a shoulder because scipy never saw it as a
    # maximum. That zero was printed in the reference-peak table beside
    # measured widths, and it left the fit with no starting width for the
    # majority of components on a broad profile — where most reference peaks
    # ARE shoulders. `_shoulder_sigma` measures it from the second
    # derivative's own sign changes, which sit at centre +/- sigma for a
    # Gaussian; FWHM = 2*sigma for the pseudo-Voigt this pipeline fits.
    if shoulder_sigma_V:
        width_V = np.asarray(width_V, float).copy()
        for _k, _ix in enumerate(all_indices):
            _sg = shoulder_sigma_V.get(int(_ix), np.nan)
            if (
                np.isfinite(_sg)
                and _sg > 0
                and not (np.isfinite(width_V[_k]) and width_V[_k] > 0)
            ):
                width_V[_k] = 2.0 * _sg
                left_v_arr[_k] = voltage[_ix] - _sg
                right_v_arr[_k] = voltage[_ix] + _sg

    # Prominence array
    # A shoulder has no measured prominence — find_peaks never saw it as a
    # maximum. The placeholder below is what ranks it in deduplication, so
    # it is kept, but `prominence_estimated` says which values are real.
    prom_array, prom_estimated = [], []
    for p in all_indices:
        if p in peak_indices:
            prom_array.append(properties["prominences"][list(peak_indices).index(p)])
            prom_estimated.append(False)
        else:
            prom_array.append(signal[p] * 0.1)
            prom_estimated.append(True)

    is_shoulder = [p in shoulder_indices for p in all_indices]

    # === TRUNCATION ASSESSMENT ===
    # Every clipped peak is FLAGGED and kept — see TRUNCATION_DROP. A peak
    # below the visible-fraction threshold is flagged truncated too, because
    # "we could not measure how much of it is here" is a stronger reason to
    # doubt its area than "we measured, and most of it is".
    is_truncated_list = []
    drop_list = []
    for k, idx in enumerate(all_indices):
        centre_v = voltage[idx]
        truncated, drop = _assess_truncation(
            centre_v,
            left_v_arr[k],
            right_v_arr[k],
            data_v_min,
            data_v_max,
            truncation_visible_fraction,
        )
        is_truncated_list.append(bool(truncated or (drop and not TRUNCATION_DROP)))
        drop_list.append(bool(drop and TRUNCATION_DROP))

    # Filter: keep non-dropped peaks
    keep_mask = np.array([not d for d in drop_list])
    if not keep_mask.any():
        return pd.DataFrame()

    all_indices = all_indices[keep_mask]
    left_v_arr = left_v_arr[keep_mask]
    right_v_arr = right_v_arr[keep_mask]
    width_V = width_V[keep_mask]
    prom_array = [p for p, k in zip(prom_array, keep_mask) if k]
    prom_estimated = [e for e, k in zip(prom_estimated, keep_mask) if k]
    is_shoulder = [s for s, k in zip(is_shoulder, keep_mask) if k]
    is_truncated_list = [t for t, k in zip(is_truncated_list, keep_mask) if k]

    peaks_df = pd.DataFrame(
        {
            "peak_id": range(1, len(all_indices) + 1),
            "voltage": voltage[all_indices],
            "height": signal[all_indices],
            "height_original": dqdv[all_indices],
            "prominence": prom_array,
            # False = measured by find_peaks; True = a shoulder's placeholder,
            # 10% of its height, kept only so shoulders can be ranked.
            "prominence_estimated": prom_estimated,
            "width_V": width_V,
            "left_voltage": left_v_arr,
            "right_voltage": right_v_arr,
            "peak_index": all_indices,
            "is_shoulder": is_shoulder,
            "is_truncated": is_truncated_list,
        }
    )

    peaks_df = peaks_df.sort_values("voltage").reset_index(drop=True)
    peaks_df["peak_id"] = range(1, len(peaks_df) + 1)

    # A SHOULDER THAT IS THE TALLEST FEATURE HAS NO PARENT. Done before
    # deduplication, because the fabricated shoulder prominence is what ranks
    # a peak there, and a promoted peak's remeasured prominence is the number
    # that should decide.
    peaks_df = _ensure_curve_maximum(
        peaks_df, voltage, signal, min_distance_mV=min_distance_mV
    )

    # === DEDUPLICATION ===
    peaks_df = _deduplicate_peaks(
        peaks_df, shoulder_min_separation_mV, primary_min_separation_mV=min_distance_mV
    )

    # === THE NOISE FLOOR ===
    # After deduplication, so a candidate is judged once and in its final
    # form, and never applied to the curve's own maximum — `signal` has one
    # largest value and it is a feature by construction. See NOISE_FLOOR.
    if NOISE_FLOOR and len(peaks_df) > 1:
        _h = peaks_df["height"].to_numpy(float)
        _hmax = float(np.nanmax(np.abs(_h))) if _h.size else 0.0
        _tallest = int(np.argmax(np.abs(_h))) if _h.size else -1
        _keep, _why = [], []
        for _k in range(len(peaks_df)):
            if _k == _tallest or _hmax <= 0:
                _keep.append(True)
                _why.append(np.nan)
                continue
            _sd = local_noise(signal, int(peaks_df["peak_index"].iloc[_k]))
            _pr = float(peaks_df["prominence"].iloc[_k])
            _snr = (_pr / _sd) if (np.isfinite(_sd) and _sd > 0) else np.inf
            _share = abs(float(_h[_k])) / _hmax
            _keep.append(not (_snr < NOISE_SNR_MIN and _share < NOISE_SHARE_MIN))
            _why.append(_snr)
        if not all(_keep):
            peaks_df = peaks_df[np.array(_keep)].reset_index(drop=True)
            peaks_df["peak_id"] = range(1, len(peaks_df) + 1)

    if len(peaks_df) > MAX_PEAKS_PER_HALF_CYCLE:
        peaks_df = peaks_df.nlargest(MAX_PEAKS_PER_HALF_CYCLE, "prominence")
        peaks_df = peaks_df.sort_values("voltage").reset_index(drop=True)
        peaks_df["peak_id"] = range(1, len(peaks_df) + 1)
        peaks_df["peaks_capped"] = True
    else:
        peaks_df["peaks_capped"] = False

    return peaks_df


# =============================================================================
# NEW IN 1.9.0
# =============================================================================


class DetectSpec:
    """
    The detection parameters as a value object.

    Same rationale as `fitting.FitSpec`: no reference to a dataset, a notebook
    global or a file, so it can be constructed in a test, written into a run
    manifest, and compared between runs. The defaults are 1.8.7's.
    """

    __slots__ = (
        "prominence_fraction",
        "min_distance_mV",
        "min_width_mV",
        "shoulder_detection",
        "shoulder_min_separation_mV",
        "shoulder_height_fraction",
        "shoulder_d2_prominence_fraction",
        "edge_exclusion_mV",
        "truncation_visible_fraction",
    )

    def __init__(
        self,
        prominence_fraction=DEFAULT_PROMINENCE_FRACTION,
        min_distance_mV=DEFAULT_MIN_DISTANCE_MV,
        min_width_mV=DEFAULT_MIN_WIDTH_MV,
        shoulder_detection=SHOULDER_DETECTION,
        shoulder_min_separation_mV=SHOULDER_MIN_SEPARATION_MV,
        shoulder_height_fraction=SHOULDER_HEIGHT_FRACTION,
        shoulder_d2_prominence_fraction=SHOULDER_D2_PROMINENCE_FRACTION,
        edge_exclusion_mV=EDGE_EXCLUSION_MV,
        truncation_visible_fraction=TRUNCATION_VISIBLE_FRACTION,
    ):
        self.prominence_fraction = float(prominence_fraction)
        self.min_distance_mV = float(min_distance_mV)
        self.min_width_mV = float(min_width_mV)
        self.shoulder_detection = bool(shoulder_detection)
        self.shoulder_min_separation_mV = float(shoulder_min_separation_mV)
        self.shoulder_height_fraction = float(shoulder_height_fraction)
        self.shoulder_d2_prominence_fraction = float(shoulder_d2_prominence_fraction)
        self.edge_exclusion_mV = float(edge_exclusion_mV)
        self.truncation_visible_fraction = float(truncation_visible_fraction)

    def replace(self, **kw):
        d = {k: getattr(self, k) for k in self.__slots__}
        d.update(kw)
        return DetectSpec(**d)

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self):
        return (
            f"DetectSpec(prom={self.prominence_fraction}, "
            f"dist={self.min_distance_mV:.0f} mV, "
            f"shoulders={self.shoulder_detection}, "
            f"visible>={self.truncation_visible_fraction:.0%})"
        )


@dataclass
class Detection:
    """One dataset's detected peaks, and the reference cycle they anchor to."""

    name: str
    peaks: dict = field(default_factory=dict)  # (cycle, step) -> DataFrame
    reference_cycle: int = DEFAULT_REFERENCE_CYCLE
    reference_reason: str = ""
    # How bad the choice was, decided where the outcome is known rather than
    # guessed from the sentence above. See REF_OK / REF_WARN / REF_BAD.
    reference_severity: str = "ok"
    reference_peaks: dict = field(default_factory=dict)  # step -> DataFrame
    spec: DetectSpec = None
    skipped: dict = field(default_factory=dict)  # (cycle, step) -> why

    def counts(self, step):
        """Peaks detected per cycle for one step, in cycle order."""
        return np.array(
            [len(df) for (c, s), df in sorted(self.peaks.items()) if s == step],
            dtype=int,
        )

    def reference_voltages(self, step):
        df = self.reference_peaks.get(step)
        if df is None or df.empty:
            return np.empty(0)
        return df["voltage"].values

    def summary(self):
        """The console block, as a string rather than a side effect.

        Was 1.8.7's `"=" * 60` banners; now the same three-column idiom the
        rest of the notebook uses, so Cell 8 stops being an unstyled island
        between Cells 6 and 9. The CONSTANTS moved to the end and print once
        per dataset rather than being announced as though they were findings.
        """
        out = [entry("reference cycle", str(self.reference_cycle))]
        out.append(bullet(self.reference_reason))
        for step in ("Charge", "Discharge"):
            rp = self.reference_peaks.get(step)
            if rp is not None and not rp.empty:
                parts = []
                for _, pk in rp.iterrows():
                    tag = ("*" if pk["is_shoulder"] else "") + (
                        "†" if pk["is_truncated"] else ""
                    )
                    parts.append(f"{pk['voltage']:.3f}{tag}")
                notes = []
                if rp["is_shoulder"].any():
                    notes.append("* shoulder, found on the second derivative")
                if rp["is_truncated"].any():
                    notes.append("† truncated, fitted one-sided")
                out.append(
                    entry(
                        f"{step.lower()} peaks",
                        f"{', '.join(parts)} V",
                        "; ".join(notes),
                    )
                )
            c = self.counts(step)
            if c.size:
                extras = []
                nsh = sum(
                    int(df["is_shoulder"].sum())
                    for (cy, s), df in self.peaks.items()
                    if s == step and not df.empty
                )
                ntr = sum(
                    int(df["is_truncated"].sum())
                    for (cy, s), df in self.peaks.items()
                    if s == step and not df.empty
                )
                if nsh:
                    extras.append(f"{nsh} shoulders")
                if ntr:
                    extras.append(f"{ntr} truncated")
                out.append(
                    entry(
                        f"{step.lower()}, all cycles",
                        f"median {np.median(c):.0f}",
                        f"range {c.min()}-{c.max()}"
                        + ((", " + ", ".join(extras)) if extras else ""),
                    )
                )
                # A PEAK LIST THAT ONLY THIS CYCLE HAS is the failure this
                # whole selector exists to avoid: the reference's components
                # are fitted and tracked in every cycle, so one it alone
                # produces is carried through the entire dataset and reported
                # with an empty drift. Said here, where it can still be acted
                # on by setting `reference_cycle` in Cell 3b.
                rn = len(rp) if rp is not None else 0
                if rn and c.size > 2 and rn != int(np.median(c)):
                    out.append(
                        verdict(
                            "caution",
                            f"the reference cycle has {rn} {step.lower()} peak(s) "
                            f"where the median cycle has {int(np.median(c))}",
                        )
                    )
                    out.append(
                        bullet(
                            "Every cycle is fitted with the reference's list, so "
                            "a component only this cycle shows is tracked through "
                            "the whole dataset and reported with no drift."
                        )
                    )
        if self.skipped:
            out.append(entry("not detectable", f"{len(self.skipped)} half-cycle(s)"))
        if self.spec is not None:
            out.append(
                bullet(
                    f"edge exclusion ±{self.spec.edge_exclusion_mV:.0f} mV; "
                    f"truncation threshold "
                    f"{self.spec.truncation_visible_fraction:.0%} visible",
                    indent=2,
                )
            )
        return "\n".join(out)


# --- is this maximum a property of the cell, or of the bin edges? ---------
# THE PHASE TEST. A histogram dQ/dV counts records into voltage bins, and on a
# constant-current step with time-based logging the records arrive at a fixed
# CHARGE interval, so their voltage spacing is dQ_per_record / (dQ/dV): narrow
# on a peak, wide on a shelf. Where the bins are finer than that spacing they
# hold one record or none, the pre-smoothing curve is a telegraph signal
# between integer multiples of dQ_per_record / bin_width, and the kernel turns
# it into a ripple at the BEAT PERIOD between the record spacing and the bin
# width. It looks like a series of shallow, evenly-ish spaced maxima on the
# flattest part of the curve, and it passes every test we had:
#
#   - the noise floor, because it measures point-to-point scatter and the
#     ripple is correlated over several bins (measured SNR 16-30 against a
#     floor of 12);
#   - the height-share exemption, because a shelf at half the main peak's
#     height puts every wiggle on it above 10% of the curve;
#   - the recurrence pass, because a beat recurs -- that is what a beat is.
#
# Measured on NMC111 cell A, cycle 5 charge, 4.10-4.45 V: dQ per record 0.770
# mAh/g constant to 0.45%, dV per record 6.6-7.1 mV, Q(V) a straight line --
# a shelf with nothing in it. Binned at 7.2 mV that is a quantum of 107
# mAh/g/V against a shelf standing at 118, and detection found four maxima on
# it. The same records through the 1.8.7 derivative path, which has no bins to
# beat against, give 6.5% modulation and no maxima at all.
#
# The test needs no model and no threshold on any amplitude: MOVE THE BIN
# EDGES. Shifting them by a fraction of a bin changes nothing about the data,
# the width or the smoothing -- a feature that sits at a potential does not
# care, and a beat slides, because its phase is set by where the edges fall.
# On NMC111 cycles 5 and 6 charge, four phase shifts leave exactly three
# maxima standing (3.66, 3.76, 3.93 V) and move every one above 4.05 V by
# 50-120 mV.
#
# Measured cost across the three chemistries (medians per half-cycle):
#
#            candidates before -> after     persistent features lost
#   LTO           1.0  ->  1.0  (sd 0)              none
#   NNM           5.0  ->  4.0                      none
#   NMC           4.0  ->  2.0  (sd 1.89 -> 0.90)   4.08, 4.24, 4.37 V
#
# Every feature that persists across cycles in LTO and NNM survives, including
# the P3 discharge shoulder at 3.13 V that 1.9.0.55 was built to catch. What
# goes is the NMC shelf, and the count stops swinging between cycles.
#
# The TALLEST candidate is exempt, for the same reason it is exempt from the
# noise floor: the curve has one largest value and it is a feature by
# construction. Its own position can wander by more than the tolerance when
# the maximum is broad and flat, which is a statement about the centre of a
# broad feature, not about whether the feature is there.
PHASE_INVARIANCE = True
PHASE_FRACTIONS = (0.2, 0.4, 0.6, 0.8)
# How far a maximum may move between phases and still count as the same one.
# Three terms, largest wins: a floor in mV, a floor in bins, and -- the term
# that matters -- half the candidate's own width, because the argmax of a
# broad feature is not determined to the same precision as that of a narrow
# one and asking it to hold 20 mV is asking the wrong question.
PHASE_TOLERANCE_MV = 20.0
PHASE_TOLERANCE_BINS = 3.0
PHASE_TOLERANCE_WIDTH_FRACTION = 0.5


def phase_stable(sig, peaks_df, spec, *, fractions=None):
    """
    Which candidates survive moving the bin edges? See PHASE_INVARIANCE.

    Returns a boolean array aligned with `peaks_df`. All True where the test
    cannot be run -- off the histogram path, or on a curve too short to
    rebuild -- because "not measured" is not "not there".
    """
    n = len(peaks_df)
    keep = np.ones(n, dtype=bool)
    rebin = getattr(sig, "rebin_at_phase", None)
    if not PHASE_INVARIANCE or n == 0 or rebin is None:
        return keep
    fractions = tuple(fractions or PHASE_FRACTIONS)
    if not fractions:
        return keep

    binw = float(getattr(sig, "bin_width_mV", np.nan))
    if not np.isfinite(binw) or binw <= 0:
        return keep

    seen = []
    for frac in fractions:
        try:
            v2, y2 = rebin(frac)
        except Exception:
            return keep  # cannot test: do not judge
        if v2 is None or len(v2) < 10:
            return keep
        alt = type(
            "PhaseSignal",
            (),
            dict(voltage=np.asarray(v2, float), dqdv=np.abs(np.asarray(y2, float))),
        )()
        try:
            pk2 = detect_peaks_single(
                alt.voltage,
                alt.dqdv,
                prominence_fraction=spec.prominence_fraction,
                min_distance_mV=spec.min_distance_mV,
                min_width_mV=spec.min_width_mV,
                shoulder_detection=spec.shoulder_detection,
                shoulder_min_separation_mV=spec.shoulder_min_separation_mV,
                shoulder_height_fraction=spec.shoulder_height_fraction,
                shoulder_d2_prominence_fraction=(spec.shoulder_d2_prominence_fraction),
                edge_exclusion_mV=spec.edge_exclusion_mV,
                truncation_visible_fraction=spec.truncation_visible_fraction,
            )
        except Exception:
            return keep
        seen.append(
            np.asarray(pk2["voltage"], float)
            if pk2 is not None and len(pk2)
            else np.empty(0)
        )

    h = np.abs(peaks_df["height"].to_numpy(float))
    tallest = int(np.argmax(h)) if h.size else -1
    v_pk = peaks_df["voltage"].to_numpy(float)
    w_pk = (
        peaks_df["width_V"].to_numpy(float) * 1000.0
        if "width_V" in peaks_df
        else np.zeros(n)
    )
    for j in range(n):
        if j == tallest:
            continue  # a curve's own maximum is a feature
        tol_mV = max(
            PHASE_TOLERANCE_MV,
            PHASE_TOLERANCE_BINS * binw,
            PHASE_TOLERANCE_WIDTH_FRACTION * float(w_pk[j]),
        )
        tol = tol_mV / 1000.0
        for other in seen:
            if other.size == 0 or np.min(np.abs(other - v_pk[j])) > tol:
                keep[j] = False
                break
    return keep


def detect_half_cycle(sig, spec=None):
    """
    Peaks in one `signal.HalfCycleSignal`.

    Returns an empty DataFrame if the half-cycle carries no usable curve —
    a half-cycle that failed preprocessing is not an absence of peaks, and the
    caller records that separately in `Detection.skipped`.
    """
    spec = spec or DetectSpec()
    if sig.voltage.size < 10:
        return pd.DataFrame()
    peaks = detect_peaks_single(
        sig.voltage,
        sig.dqdv,
        prominence_fraction=spec.prominence_fraction,
        min_distance_mV=spec.min_distance_mV,
        min_width_mV=spec.min_width_mV,
        shoulder_detection=spec.shoulder_detection,
        shoulder_min_separation_mV=spec.shoulder_min_separation_mV,
        shoulder_height_fraction=spec.shoulder_height_fraction,
        shoulder_d2_prominence_fraction=spec.shoulder_d2_prominence_fraction,
        edge_exclusion_mV=spec.edge_exclusion_mV,
        truncation_visible_fraction=spec.truncation_visible_fraction,
    )

    if peaks is None or not len(peaks):
        return peaks
    keep = phase_stable(sig, peaks, spec)
    n_dropped = int((~keep).sum())
    if n_dropped:
        peaks = peaks[keep].reset_index(drop=True)
        peaks["peak_id"] = range(1, len(peaks) + 1)
    # ON THE FUNCTION, NOT ON `.attrs`. `DataFrame.attrs` is not propagated
    # by `pd.concat`, and both seeding passes rebuild every seeded half-cycle's
    # frame with a concat — so this count was destroyed before anything could
    # read it, and nothing read it anyway. It matters more now than it did:
    # until the grid anchor was fixed, `phase_stable` could not reject anything
    # on a half-cycle that opened above the window's foot, so a run has no
    # history to compare against and the number needs to be visible.
    detect_half_cycle.phase_unstable_dropped += n_dropped
    peaks.attrs["phase_unstable_dropped"] = n_dropped
    return peaks


detect_half_cycle.phase_unstable_dropped = 0


# --- how bad is the reference-cycle choice? --------------------------------
# `choose_reference_cycle` returns a sentence, and for a while the per-cell
# report derived its severity from that sentence with
# `"warn" if "failed" in reason else "ok"`. The mapping was exactly inverted:
# the mild outcomes are the ones that say "failed" ("cycle 5 failed the
# integrity check; 6 is the first at or above it that passes" — a sound cycle
# WAS found), and the severe ones contain no such word, so
# "NO cycle passes the integrity check — using 3 unchecked" and
# "FLAGGED ANOMALOUS by the integrity check" both rendered as a green tick.
# The severity is therefore decided HERE, where the outcome is known, and
# carried as a value. A verdict inferred from prose is not a verdict.
REF_OK = "ok"  # a cycle that passed the integrity check
REF_WARN = "warn"  # a fallback, or a cycle that could not be checked
REF_BAD = "bad"  # nothing passed, or the chosen cycle is ANOMALOUS


def choose_reference_cycle(
    available_cycles,
    exists,
    *,
    integrity=None,
    default=DEFAULT_REFERENCE_CYCLE,
    auto=REFERENCE_CYCLE_AUTO,
    exists_step=None,
    efficiency=None,
):
    """
    Pick the cycle whose peak list will anchor the whole dataset, and say why.

    Parameters
    ----------
    available_cycles : sorted sequence of int
    exists : callable(cycle) -> bool
        Whether the cycle has at least one half-cycle to detect on.
    exists_step : callable(cycle, step) -> bool, optional
        Whether that particular half-cycle exists. Used so a cycle is judged
        only on the half-cycles it has; absent, both are required, which is
        1.8.6 behaviour.
    integrity : callable(cycle, step) -> str, optional
        The integrity band. Absent (or `auto=False`) restores 1.8.6 behaviour
        exactly: the default cycle if present, else the second available one.
        Passed in rather than looked up, so this function is testable.
    default : int
    auto : bool

    Returns
    -------
    (cycle, reason)

    efficiency : {cycle: CE percent}, optional
        Where given, `formation_end` reads it and the search starts from the
        cycle at which CE settled rather than from `default`. This is the
        whole point: integrity asks "is this cycle clean?", and a cycle can
        be perfectly clean and still be the wrong fingerprint because the
        cell was still forming. Absent, behaviour is unchanged.

    The rule is LOWEST SOUND AND SETTLED, not best. A fingerprint should come
    from early in the life of the cell — moving further in means tracking
    against a peak list that has already aged — but not from inside
    formation, where the feature has not stopped moving. On the LTO
    triplicate the charge peak sits at 1.520, 1.524, then 1.548 V and stays
    there: a 24 mV jump between cycle 2 and cycle 3 against a measured drift
    of about 1 mV per cycle.

    Only if nothing at or above the start passes does it look below, and it
    takes the highest such cycle — the closest it can get.
    """
    cycles = list(available_cycles)
    if not cycles:
        return default, "no cycles available", REF_BAD
    if exists_step is None:
        exists_step = lambda c, s: True  # noqa: E731

    start, settled_why, settled_ok = default, None, False
    if auto and efficiency:
        settled, settled_why = formation_end(efficiency)
        if settled is not None:
            start, settled_ok = max(default, int(settled)), True
        else:
            # CE never settled. Fall back to the conservative cycle rather
            # than to the 1.8.7 default of 2, which is the case this whole
            # mechanism exists to avoid: a fingerprint taken from inside
            # formation. If the record is too short to reach the fallback,
            # the search below drops to the highest sound cycle it has and
            # says so.
            start = max(default, int(REFERENCE_CYCLE_FALLBACK))

    def _fallback():
        if default in cycles:
            return default, "default"
        c = cycles[1] if len(cycles) > 1 else cycles[0]
        return c, f"default cycle {default} not present"

    if not auto or integrity is None:
        c, why = _fallback()
        if why == "default":
            why = "default (no integrity check available)"
        # Unchecked is not sound. This branch is reached whenever no
        # integrity callable was supplied, and the sentence says so; the
        # severity has to say so too.
        return c, why, REF_WARN

    def _sound(c):
        # Judge only the half-cycles this cycle ACTUALLY HAS. Requiring both
        # meant a dataset with one step direction, or a truncated final
        # cycle, could never pass: the missing half returns "unknown".
        bands = [integrity(c, s) for s in ("Charge", "Discharge") if exists_step(c, s)]
        return bool(bands) and all(b in SOUND_BANDS for b in bands)

    candidates = [c for c in cycles if c >= start and exists(c) and _sound(c)]
    if candidates:
        c = candidates[0]
        if settled_why and not settled_ok:
            return (
                c,
                (
                    f"formation could not be measured — {settled_why}; "
                    f"fell back to cycle {REFERENCE_CYCLE_FALLBACK} and "
                    f"{c} is the first at or above it that passes"
                ),
                REF_WARN,
            )
        if settled_why and start > default:
            if c == start:
                return (
                    c,
                    (
                        f"formation ended here — {settled_why} — and it "
                        f"passes the integrity check"
                    ),
                    REF_OK,
                )
            return (
                c,
                (
                    f"formation ended at cycle {start} ({settled_why}), "
                    f"which failed the integrity check; {c} is the first "
                    f"after it that passes"
                ),
                REF_OK,
            )
        why_settled = f"; {settled_why}" if settled_why else ""
        if c == start:
            return (
                c,
                f"default, and it passes the integrity check{why_settled}",
                REF_OK,
            )
        return (
            c,
            (
                f"cycle {start} failed the integrity check; "
                f"{c} is the first at or above it that passes"
            ),
            REF_OK,
        )

    below = [c for c in cycles if c < start and exists(c) and _sound(c)]
    if below:
        c = below[-1]
        if settled_why and not settled_ok:
            return (
                c,
                (
                    f"formation could not be measured — {settled_why}; "
                    f"the record does not reach cycle "
                    f"{REFERENCE_CYCLE_FALLBACK}, so {c} is the latest "
                    f"sound cycle available"
                ),
                REF_WARN,
            )
        return (
            c,
            (
                f"no cycle at or above {start} passes the integrity "
                f"check; fell back to {c}"
            ),
            REF_WARN,
        )

    c, why = _fallback()
    return (
        c,
        (f"NO cycle passes the integrity check — using {c} unchecked ({why})"),
        REF_BAD,
    )


def detect_spec_for_profile(profile, **overrides):
    """A `DetectSpec` matching the profile of this dataset's own curve.

    `profile` is `signal.classify_dqdv_profile`'s dict, or its class string.
    Anything passed as a keyword overrides the profile's value, so an
    operator setting `peak_min_distance_mV` in Cell 3b still wins.
    """
    cls = profile.get("class", "broad") if isinstance(profile, dict) else str(profile)
    kw = {
        "min_distance_mV": PROFILE_MIN_DISTANCE_MV.get(cls, DEFAULT_MIN_DISTANCE_MV),
        "min_width_mV": PROFILE_MIN_WIDTH_BY_CLASS.get(cls, DEFAULT_MIN_WIDTH_MV),
    }
    kw.update({k: v for k, v in overrides.items() if v is not None})
    return DetectSpec(**kw)


# =============================================================================
# RECURRENCE — a feature is what comes back at the same potential
# =============================================================================
#
# A shoulder on a flank has, by construction, a large HEIGHT and a small
# PROMINENCE, so every prominence gate in this module is hostile to it. The
# NNM discharge shoulder at 3.15 V is the case that proved it. Measured over
# 40 half-cycles of cell A, with no gates applied at all:
#
#     prominence                     median 1.30   (real peaks: 89 and 157)
#     prominence / curve range       median 0.006  (gate wants 0.05)
#     prominence / its own height    median 0.016  (gate wants 0.25)
#     prominence / local noise       median 2.24   (the .52 floor wants 12)
#     height / curve max             median 0.449
#
# It is a genuine local maximum in 28 of those 40 half-cycles and it fails
# every gate. That is not a threshold set badly; prominence is the wrong
# quantity for this feature. At 2.24x the local noise it is, on any ONE
# half-cycle, indistinguishable from noise — and it is obviously real to the
# eye, because it comes back in the same place every cycle.
#
# So this pass asks the question the eye is actually asking. Collect every
# local maximum from every half-cycle of a step, cluster by voltage, and keep
# the clusters that RECUR. It is the same argument that rejected the LTO flank
# bumps (their potentials scattered over 150 mV) and accepted the main peak
# (stable to 0.2 mV): a redox process happens at a potential.
#
# TWO CLAUSES, because recurrence alone is not enough. On NNM cell C a band of
# noise at 2.6-2.7 V recurs in 50-65% of cycles — MORE often than the shoulder
# does — but it is 15% of the curve's height where the shoulder is 46%:
#
#     V       occupancy   share       verdict
#     2.61      0.50       0.15       rejected
#     2.69      0.65       0.15       rejected  (occupancy alone would keep it)
#     3.15      0.55       0.46       KEPT      <- the shoulder
#
# Occupancy >= 0.5 AND share >= 0.25 returns exactly four features on each NNM
# cell independently — 2.09, 3.15, 3.21, 3.53 — with the shoulder among them at
# a prominence two orders of magnitude below the real peaks. The same pair of
# clauses as the `.52` noise floor, used to ADMIT rather than to reject.
#
# THE DENSITY GATE, and why it is not a profile rule. The method needs the
# curve's local maxima to mean something:
#
#     NMC111     7 local maxima per half-cycle
#     NNM       13
#     LTO      110
#
# On LTO it saturates — 70 "recurrent features" over a 1.2 V window — because
# that curve is built from bins holding 0 or 1 raw record, so its local maxima
# are quantisation rather than structure. A noise pre-filter does not rescue it
# (110 -> 96 at 5 sigma, because the quantisation is locally self-similar) and
# it kills the NNM shoulder (lost at 2 sigma). So the pass declines where the
# density says the curve cannot support it. That is a statement about the
# MEASUREMENT, not about the material or the mechanism — the same discipline as
# the noise floor and the width floors. 7-13 against 110 leaves the gate room.
RECURRENCE_SEEDS = True
RECURRENCE_TOLERANCE_MV = 15.0  # how close two maxima must be to be "the same"
RECURRENCE_MIN_OCCUPANCY = 0.50  # in at least half the half-cycles
RECURRENCE_MIN_SHARE = 0.25  # ...and this much of the curve's own height
RECURRENCE_MAX_DENSITY = 30.0  # maxima per half-cycle above which we decline
# A seed must not merely sit NEAR the recurrent voltage; it must be a
# plausible instance of the feature. With noise there is almost always some
# local maximum within the tolerance, so position alone would place a seed on
# every half-cycle whether the feature is there or not. Half the feature's
# usual share is a wide allowance — the shoulder varies 24-42% of the curve
# across the NNM run — and still excludes a half-cycle where the feature is
# genuinely absent.
RECURRENCE_SEED_SHARE_FRACTION = 0.5
# CHECK THE SEPARATION GUARD WHERE THE SEED LANDS. Built and measured in
# 1.9.0.59, and OFF: the bug is real, the repair is one line, and the repair
# costs more than the bug does.
#
# THE BUG. `add_recurrent_seeds` tests `min_distance_mV` against `fv`, the
# dataset-level feature voltage, and then places the seed at that half-cycle's
# own nearest maximum — which its docstring says explicitly, and which can be
# a whole `tolerance_mV` away. So a seed can clear the guard and then land
# inside the peak the guard exists to protect. NNM cell A cycle 6 discharge
# had detected the shoulder at 3.1224 V; the recurrent feature sits at 3.1534,
# 31.0 mV away, clearing the 30 mV guard by one millivolt; the seed was placed
# at 3.1482, which is 25.8 mV from the shoulder already there. Cycle 6 is the
# reference cycle, so both entered the reference list and the split propagated
# to every cycle.
#
# Measured over 152 recurrent seeds on seven cells, the repair is surgical:
#
#     seeds the guard passes at `fv` but would refuse at the placement:  11
#     of those, landing on a neighbour that is ALREADY a shoulder:       11
#     their separations at the placement:                     20.5-28.4 mV
#     legitimate shoulder seeds in NNM's 3.05-3.30 V region, kept:      120
#     their separations:                     30.2-74.5 mV (median 61.6)
#
# The populations do not overlap and the boundary is the existing
# `min_distance_mV`. No new constant, no new threshold.
#
# WHAT IT DOES, AND WHY IT IS OFF. Cells B and C are bit-for-bit unchanged, as
# are LTO and NMC. On cell A it does exactly what it was built to do:
#
#     discharge references                     6 -> 5, matching B and C
#     the shoulder reference   0 pairs "never in consecutive cycles" + 19
#                              "questionable"  ->  58 pairs "coherent"
#     total coherence pairs                          494 -> 518
#     cycles with a component in the shoulder band   127/128 -> 128/128
#     cycles with TWO components in that band         41 -> 10
#
# and then loses more than that elsewhere:
#
#     cycles resolving the MAIN peak (3.185-3.32 V)  125/128 -> 102/128
#     discharge R2                    better on 37/128, median -0.0026
#
# The 26 lost cycles are 102-128, the end of the run, and they are the same
# cycle in every case: 1.9.0.58 fits 3.1375 and 3.2045 with the higher of the
# two REPORTED AS RELIABLE, and 1.9.0.59 fits one component at 3.1692 sitting
# between them, at R2 0.968 against 0.983. The R2 gap says the data supports
# the pair; detection in those late cycles finds only one maximum, and the
# second reference — a duplicate in cycle 6 — was what held them apart at the
# other end of the run.
#
# So the entry at 3.1482 is a duplicate early and load-bearing late, which is
# the same shape as the P3 structural claim retracted in 1.9.0.56: something
# that looks like an artefact in one regime is doing real work in another.
# The honest repair is not this guard but a reference that carries a feature's
# RANGE rather than one voltage, so a feature that migrates 45 mV over 128
# cycles is one entry that moves instead of two entries that do not. That is a
# design change, not a line, and it is not being improvised on the back of
# this one.
RECURRENCE_SEED_GUARD_AT_PLACEMENT = False


def recurrent_maxima(
    signals,
    *,
    tolerance_mV=RECURRENCE_TOLERANCE_MV,
    min_occupancy=RECURRENCE_MIN_OCCUPANCY,
    min_share=RECURRENCE_MIN_SHARE,
    max_density=RECURRENCE_MAX_DENSITY,
    edge_exclusion_mV=EDGE_EXCLUSION_MV,
    enabled=RECURRENCE_SEEDS,
):
    """
    Voltages at which a local maximum recurs across a dataset.

    `signals` is `{(cycle, step): HalfCycleSignal}`. Returns
    `{step: {"features": [...], "density": float, "declined": str or None}}`,
    where each feature is a dict of `voltage`, `occupancy`, `share` and
    `n_cycles`. Nothing is added to any peak table here — see
    `add_recurrent_seeds`.
    """
    out = {}
    if not enabled:
        return out
    tol = float(tolerance_mV) / 1000.0
    for step in ("Charge", "Discharge"):
        keys = [k for k in signals if k[1] == step]
        rows, n_hc = [], 0
        for k in sorted(keys):
            sig = signals[k]
            v = np.asarray(sig.voltage, float)
            y = np.abs(np.asarray(sig.dqdv, float))
            if v.size < 20:
                continue
            n_hc += 1
            idx, _ = find_peaks(y)
            if not len(idx):
                continue
            hmax = float(np.nanmax(y))
            if not np.isfinite(hmax) or hmax <= 0:
                continue
            # THE WINDOW EDGE, excluded here as it is in
            # `detect_peaks_single`. A curve that rises to its limit puts a
            # maximum at the last sample of every half-cycle, which recurs
            # perfectly and is an artefact of where the scan stopped.
            _lo = float(v[0]) + float(edge_exclusion_mV) / 1000.0
            _hi = float(v[-1]) - float(edge_exclusion_mV) / 1000.0
            for j in idx:
                if not (_lo <= v[j] <= _hi):
                    continue
                rows.append((int(k[0]), float(v[j]), float(y[j]) / hmax))
        if not n_hc or not rows:
            continue
        density = len(rows) / float(n_hc)
        entry = {
            "features": [],
            "density": density,
            "declined": None,
            "n_half_cycles": n_hc,
        }
        if density > float(max_density):
            entry["declined"] = (
                f"{density:.0f} local maxima per half-cycle — this curve's "
                f"maxima are sampling noise rather than structure, so "
                f"recurrence cannot be measured on it"
            )
            out[step] = entry
            continue
        vs = np.array([r[1] for r in rows], float)
        order = np.argsort(vs)
        vs_sorted = vs[order]
        claimed = np.zeros(vs_sorted.size, bool)
        feats = []
        for i in range(vs_sorted.size):
            if claimed[i]:
                continue
            c = float(vs_sorted[i])
            for _ in range(3):  # settle on the local mode
                m = (vs >= c - tol) & (vs <= c + tol)
                if not m.any():
                    break
                c = float(np.median(vs[m]))
            m = (vs >= c - tol) & (vs <= c + tol)
            grp = [rows[j] for j in np.flatnonzero(m)]
            if not grp:
                continue
            occ = len({g[0] for g in grp}) / float(n_hc)
            share = float(np.median([g[2] for g in grp]))
            claimed |= (vs_sorted >= c - tol) & (vs_sorted <= c + tol)
            if occ >= float(min_occupancy) and share >= float(min_share):
                feats.append(
                    dict(
                        voltage=c,
                        occupancy=occ,
                        share=share,
                        n_cycles=len({g[0] for g in grp}),
                    )
                )
        # Two clusters can settle on centres closer than the tolerance; keep
        # the better-occupied one so a feature is not seeded twice.
        feats.sort(key=lambda d: -d["occupancy"])
        keep = []
        for f in feats:
            if all(abs(f["voltage"] - g["voltage"]) > tol for g in keep):
                keep.append(f)
        entry["features"] = sorted(keep, key=lambda d: d["voltage"])
        out[step] = entry
    return out


def add_recurrent_seeds(
    peaks,
    signals,
    recurrent,
    *,
    tolerance_mV=RECURRENCE_TOLERANCE_MV,
    min_distance_mV=DEFAULT_MIN_DISTANCE_MV,
    share_tolerance=RECURRENCE_SEED_SHARE_FRACTION,
):
    """
    Seed each half-cycle with the recurrent features it did not detect itself.

    The seed is placed at the local maximum nearest the recurrent voltage IN
    THAT HALF-CYCLE, not at the run's median — so it is a measurement of this
    curve, not a value copied from another cycle. Where this half-cycle has no
    local maximum near the feature at all, nothing is added: the feature is
    genuinely absent here and inventing it would be fabricating data.

    THE SEPARATION GUARD IS CHECKED WHERE THE SEED LANDS, added 1.9.0.59.
    `min_distance_mV` was tested against `fv`, the dataset-level feature
    voltage — and the paragraph above says in as many words that the seed is
    NOT placed there. The two can differ by up to `tolerance_mV`, and that is
    enough to let a seed through the guard and then land inside the peak the
    guard was protecting.

    It is the phantom reference at 3.1224 V that has been costing the NNM
    shoulder its coherence verdict since 1.9.0.56. Cell A cycle 6 discharge
    had already detected the shoulder at 3.1224. The recurrent feature sits at
    3.1534, which is 31.0 mV away — clear of the 30 mV guard by one
    millivolt — so a seed was placed at that half-cycle's nearest maximum,
    3.1482, which is 25.8 mV from the shoulder already there. Two components
    on one shoulder. Cycle 6 is the reference cycle, so both entered the
    reference list, and from there the split propagated to every cycle: the
    real feature was left with 19 coherence pairs and a `questionable`
    verdict across 128 discharge cycles, where cell C got 30 pairs and
    `coherent` across 122 and cell B 17 and `coherent` across 67.

    Measured over 152 recurrent seeds on seven cells:

        seeds the guard passes at `fv` and would refuse at the placement: 11
        of those, landing on a neighbour that is ALREADY a shoulder:      11
        their separations at the placement:                    20.5-28.4 mV

        legitimate shoulder seeds in the NNM 3.05-3.30 V region kept:    120
        their separations:                    30.2-74.5 mV (median 61.6)
        of those, seeded beside a PRIMARY rather than a shoulder:    114/120

    The two populations do not overlap and the boundary between them is the
    existing `min_distance_mV`, which was not chosen for this. No new constant
    and no new threshold: the rule was always right, it was applied to the
    wrong voltage.

    A WIDTH-RATIO RULE WAS TRIED FIRST AND IS WRONG, recorded so it is not
    tried again. Comparing the separation to the neighbour's FWHM looks
    decisive in aggregate — NNM's seeds sit at a median 0.49 of the
    neighbour's width against NMC's 8.83 — but any threshold in that gap
    refuses 115 of NNM's 131 seeds. It has to: a shoulder legitimately sits
    at about half its parent's FWHM from the parent's centre, so "close
    relative to the width" is the definition of a shoulder, not the signature
    of a duplicate. It would have deleted the shoulder seeding wholesale.

    `peaks` is modified in place; the number of seeds added is returned.
    """
    if not recurrent:
        return 0
    tol = float(tolerance_mV) / 1000.0
    sep = float(min_distance_mV) / 1000.0
    added = 0
    for step, entry in recurrent.items():
        for feat in entry.get("features", []):
            fv = float(feat["voltage"])
            for (c, st), df in list(peaks.items()):
                if st != step or df is None or df.empty:
                    continue
                if (df["voltage"] - fv).abs().min() <= sep:
                    continue  # already has it
                if len(df) >= MAX_PEAKS_PER_HALF_CYCLE:
                    continue
                sig = signals.get((c, st))
                if sig is None:
                    continue
                v = np.asarray(sig.voltage, float)
                y = np.abs(np.asarray(sig.dqdv, float))
                if v.size < 20:
                    continue
                idx, _ = find_peaks(y)
                if not len(idx):
                    continue
                near = idx[int(np.argmin(np.abs(v[idx] - fv)))]
                if abs(float(v[near]) - fv) > tol:
                    continue  # not present on this curve
                # THE GUARD AGAIN, AT THE PLACEMENT — see
                # RECURRENCE_SEED_GUARD_AT_PLACEMENT for the measurement and
                # for why this is off. The check above was made against `fv`;
                # this seed is going to `v[near]`, which can be a whole
                # `tolerance_mV` away.
                if (
                    RECURRENCE_SEED_GUARD_AT_PLACEMENT
                    and (df["voltage"] - float(v[near])).abs().min() <= sep
                ):
                    continue  # already has it, really
                # ...and it has to LOOK like the feature, not merely sit near
                # where the feature is. With noise there is almost always some
                # maximum within the tolerance; a half-cycle where the nearest
                # one is a fraction of the feature's usual height does not have
                # the feature, and seeding it there would be inventing data.
                _hmax = float(np.nanmax(y))
                _share = (float(y[near]) / _hmax) if _hmax > 0 else 0.0
                if _share < float(share_tolerance) * float(feat["share"]):
                    continue
                prom = float(peak_prominences(y, [int(near)])[0][0])
                row = {col: np.nan for col in df.columns}
                row.update(
                    voltage=float(v[near]),
                    height=float(y[near]),
                    height_original=float(np.asarray(sig.dqdv, float)[near]),
                    prominence=prom,
                    prominence_estimated=False,
                    peak_index=int(near),
                    peak_id=int(df["peak_id"].max()) + 1,
                    is_shoulder=False,
                    is_truncated=False,
                    is_recurrent=True,
                )
                out = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                out = out.sort_values("voltage").reset_index(drop=True)
                out["peak_id"] = range(1, len(out) + 1)
                peaks[(c, step)] = out
                added += 1
    # NaN IS NOT False, AND bool(nan) IS True.
    # `row = {col: np.nan for col in df.columns}` then `row.update(...,
    # is_recurrent=True)` adds a column the existing frame does not have, so
    # `pd.concat` back-fills NaN for every peak that was genuinely detected in
    # that half-cycle. This loop only acted when the column was ABSENT, so
    # those NaNs survived to the parameter table and to
    # `*_fitted_parameters.csv` — and `analyse.parameters_frame` reads them as
    # `bool(d.get("is_recurrent", False))`, which is True for NaN. Every
    # detected peak in a seeded half-cycle was therefore published as having
    # been seeded by recurrence rather than found. `infer_missing_seeds` has
    # handled the identical situation correctly since it was written; this
    # function did not.
    for k, df in peaks.items():
        if df is None or df.empty:
            continue
        if "is_recurrent" not in df.columns:
            df["is_recurrent"] = False
        else:
            df["is_recurrent"] = df["is_recurrent"].fillna(False).astype(bool)
    return added


def infer_missing_seeds(
    peaks,
    reference_peaks,
    *,
    tolerance_mV=SEED_INFERENCE_TOLERANCE_MV,
    windows=None,
    enabled=INFER_MISSING_SEEDS,
):
    """
    Fill single-cycle detection gaps with seeds from the nearest cycle.

    `peaks` is `{(cycle, step): DataFrame}` and is modified in place; the count
    of seeds added is returned. Half-cycles with NO detected peaks at all are
    left alone: nothing was fitted there, and inventing a whole peak list for a
    half-cycle whose curve failed preprocessing would be fabricating data
    rather than bridging a gap.

    `windows` is `{(cycle, step): (v_min, v_max)}`. A SEED MUST LIE INSIDE THE
    HALF-CYCLE IT IS PLACED IN. Without this the position came from a
    NEIGHBOUR and was never checked against this half-cycle's own voltage
    range, and half-cycles do not all cover the same range — a cell that
    starts a cycle part-charged has a charge curve beginning 850 mV above the
    others. NNM cell B cycle 1 charge spans 2.897-4.200 V and was given a seed
    at 2.207 V from cycle 2. The fitter cannot refuse a seed, so it put a
    component there: sigma on the floor, height 190,936, and an analytic area
    of 2791 mAh/g on a cell that holds 110. `reliable` was False, which is the
    only thing that stood between that number and a results table.
    81 components across the NNM triplicate were centred outside their own
    window, 74 of them from this function, and three of those were `reliable`.
    LTO and NMC had none, so this guard cannot change them.

    Returns the count added; seeds refused for falling outside are counted
    separately in `infer_missing_seeds.refused` for the caller to report.

    See INFER_MISSING_SEEDS for why these rows are marked and demoted rather
    than merged in silently.
    """
    infer_missing_seeds.refused = 0
    if not enabled or not reference_peaks:
        return 0
    tol = float(tolerance_mV) / 1000.0
    added = 0
    windows = windows or {}
    for step in ("Charge", "Discharge"):
        ref = reference_peaks.get(step)
        if ref is None or getattr(ref, "empty", True):
            continue
        cycles = sorted({c for (c, s) in peaks if s == step})
        for _, rp in ref.iterrows():
            rv = float(rp["voltage"])
            # Where the peak WAS, cycle by cycle, so a seed can be placed at
            # its neighbour's position rather than at the reference position
            # it may have drifted 40 mV away from by cycle 200.
            found = {}
            for c in cycles:
                df = peaks.get((c, step))
                if df is None or df.empty:
                    continue
                d = (df["voltage"] - rv).abs()
                j = int(d.idxmin())
                if float(d.loc[j]) > tol:
                    continue
                # A DETECTED PEAK CAN ONLY STAND FOR ONE REFERENCE PEAK.
                #
                # A fixed radius alone cannot be right when two reference
                # peaks are closer together than the radius: the same detected
                # maximum then satisfies both, and the weaker of the pair is
                # recorded as present in a cycle that never found it, so no
                # seed is inferred and it is never fitted.
                #
                # Measured on NNM cell A discharge. The reference list carries
                # 3.151 V (the shoulder) and 3.207 V, 56 mV apart against an
                # 80 mV radius. In cycles 1-5 and 8-10 the only maximum near
                # either is at ~3.204, and it was answering for both — so the
                # shoulder was fitted in 2 of the first 10 cycles and inferred
                # in NONE of them, while `infer_missing_seeds` was firing 222
                # times elsewhere on the same dataset. That absence was then
                # read as the feature dying, and written up as chemistry.
                #
                # So the peak has to be NEARER to this reference peak than to
                # any other. 3.204 belongs to 3.207 (3 mV), not to 3.151
                # (53 mV), and the shoulder is correctly a gap to be filled.
                _rv_all = ref["voltage"].to_numpy(float)
                _dv = float(df.loc[j, "voltage"])
                if np.min(np.abs(_rv_all - _dv)) < abs(_dv - rv) - 1e-12:
                    continue  # that maximum is someone else's
                found[c] = _dv
            if not found:
                continue  # never seen; not a gap
            for c in cycles:
                if c in found:
                    continue
                df = peaks.get((c, step))
                if df is None or df.empty:
                    continue  # nothing fitted here at all
                near = min(found, key=lambda k: (abs(k - c), k))
                # INSIDE THIS HALF-CYCLE'S OWN WINDOW, or not at all. See the
                # docstring: the position comes from a neighbour, and a
                # neighbour's voltage range is not this one's.
                _w = windows.get((c, step))
                if _w is not None:
                    _lo, _hi = float(_w[0]), float(_w[1])
                    if not (_lo <= float(found[near]) <= _hi):
                        infer_missing_seeds.refused += 1
                        continue
                row = {col: np.nan for col in df.columns}
                row.update(
                    voltage=found[near],
                    peak_id=int(df["peak_id"].max()) + 1,
                    is_shoulder=bool(rp.get("is_shoulder", False)),
                    is_truncated=bool(rp.get("is_truncated", False)),
                    prominence_estimated=True,
                    is_inferred=True,
                    inferred_from_cycle=int(near),
                )
                out = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                out = out.sort_values("voltage").reset_index(drop=True)
                out["peak_id"] = range(1, len(out) + 1)
                peaks[(c, step)] = out
                added += 1
    # Every half-cycle now carries the column, so a reader of the table never
    # has to distinguish "not inferred" from "this build did not record it".
    for k, df in peaks.items():
        if df is None or df.empty:
            continue
        if "is_inferred" not in df.columns:
            df["is_inferred"] = False
            df["inferred_from_cycle"] = np.nan
        else:
            df["is_inferred"] = df["is_inferred"].fillna(False).astype(bool)
        # ...AND `is_recurrent` AGAIN, HERE. `add_recurrent_seeds` runs first
        # and normalises its own column, but this function then appends rows
        # built the same way — every existing column NaN — so an inferred seed
        # row picked up `is_recurrent = NaN`, which reads as True. Whichever
        # seeding pass runs last has to leave the flags clean.
        if "is_recurrent" not in df.columns:
            df["is_recurrent"] = False
        else:
            df["is_recurrent"] = df["is_recurrent"].fillna(False).astype(bool)
        for _b in ("is_shoulder", "is_truncated"):
            if _b in df.columns:
                df[_b] = df[_b].fillna(False).astype(bool)
    return added


def detect_all(
    signals,
    *,
    name="",
    spec=None,
    reference_cycle=None,
    integrity=None,
    verbose=True,
    efficiency=None,
):
    """
    Detect across a dataset and choose its reference cycle.

    Parameters
    ----------
    signals : {(cycle, step): signal.HalfCycleSignal}
    reference_cycle : int, optional
        An EXPLICIT choice. Supplying it means the user chose it, so it is
        honoured, checked and warned about — never overridden. There is no
        sentinel value and no defaults dict to disambiguate: 1.8.7 inferred
        intent from `'reference_cycle' in params`, which Cell 3 makes true for
        every dataset, and the selector was therefore inert on every real run.
    integrity : callable(cycle, step) -> str, optional
    efficiency : {cycle: CE percent}, optional
        Passed through to `choose_reference_cycle`, which uses it to find the
        end of formation. See `formation_end`.
    """
    spec = spec or DetectSpec()
    peaks, skipped = {}, {}
    detect_half_cycle.phase_unstable_dropped = 0

    for key in sorted(signals):
        sig = signals[key]
        if sig.voltage.size < 10:
            skipped[key] = sig.note or "no usable curve"
            peaks[key] = pd.DataFrame()
            continue
        peaks[key] = detect_half_cycle(sig, spec)

    # --- RECURRENCE: what comes back at the same potential -------------
    # Runs across the whole dataset, after every half-cycle has been detected
    # on its own, because that is the only level at which the question can be
    # asked. See `recurrent_maxima` for the measurement behind the two
    # thresholds and for why it declines on a quantisation-dominated curve.
    # SAID OUT LOUD. See `detect_half_cycle`: the phase-invariance test drops
    # candidates that move with the bin grid rather than with the material,
    # and how many it dropped is a statement about the binning that belonged
    # in the run all along.
    _nph = int(getattr(detect_half_cycle, "phase_unstable_dropped", 0))
    if verbose and _nph:
        print(
            f"  phase invariance: {_nph} candidate(s) dropped — they moved "
            f"with the bin grid when it was shifted, so they are the "
            f"binning and not the curve"
        )

    recurrent = recurrent_maxima(signals)
    n_seeded = add_recurrent_seeds(
        peaks, signals, recurrent, min_distance_mV=spec.min_distance_mV
    )
    if verbose and recurrent:
        for _st in ("Charge", "Discharge"):
            _e = recurrent.get(_st)
            if not _e:
                continue
            if _e.get("declined"):
                print(f"  recurrence ({_st.lower()}): not measured — {_e['declined']}")
            elif _e["features"]:
                _fv = ", ".join(
                    f"{d['voltage']:.3f} V "
                    f"({d['occupancy']:.0%} of cycles, "
                    f"{d['share']:.0%} of the curve)"
                    for d in _e["features"]
                )
                print(f"  recurrence ({_st.lower()}): {_fv}")
    if verbose and n_seeded:
        print(
            f"  {n_seeded} recurrent seed(s) added to half-cycles that did "
            f"not detect them on their own"
        )

    n_capped = sum(
        1
        for d in peaks.values()
        if not d.empty and bool(d.get("peaks_capped", pd.Series([False])).any())
    )
    if n_capped and verbose:
        print(
            f"\n  *** {n_capped} half-cycle(s) detected more than "
            f"{MAX_PEAKS_PER_HALF_CYCLE} peaks and were capped to the "
            f"strongest. ***"
        )
        print(
            "      That many features on one dQ/dV curve is a detection "
            "failure, not a rich material — check the smoothing window "
            "and the prominence fraction in Cell 4."
        )

    cycles = sorted({int(c) for c, _ in signals})

    def _exists(c):
        # "Has something to DETECT ON", not "appears in the dict". Every
        # candidate is drawn from the keys of `signals`, so the membership
        # test was tautologically true and this callable — the one the
        # module docstring introduces to make the selector testable — did
        # nothing. A half-cycle whose curve failed preprocessing has an
        # empty peak table, and a reference cycle with no peaks fixes an
        # empty peak list for the whole dataset.
        return any(
            not peaks.get((c, s), pd.DataFrame()).empty for s in ("Charge", "Discharge")
        )

    if reference_cycle is not None:
        ref = int(reference_cycle)
        reason = "set explicitly"
        # An operator-chosen cycle has not been checked unless the integrity
        # callable is present AND returns a sound band. "Set explicitly" on
        # its own is not a pass.
        severity = REF_WARN
        if not cycles:
            # The reference_cycle=None branch handles this cleanly; this one
            # raised IndexError on cycles[0].
            return Detection(
                name=name,
                peaks=peaks,
                skipped=skipped,
                spec=spec,
                reference_cycle=ref,
                reference_reason="no cycles available",
                reference_severity=REF_BAD,
                reference_peaks={},
            )
        if ref not in cycles:
            # The NEAREST available cycle, not the second one in the record.
            ref = min(cycles, key=lambda c: (abs(c - ref), c))
            reason = f"requested cycle not present; adjusted to {ref}"
        if integrity is not None:
            # ONLY THE HALF-CYCLES THIS CYCLE HAS, exactly as the auto path
            # does. Asking for both unconditionally meant a charge-only or
            # discharge-only dataset — or a cycle whose second half is missing
            # — always got "unknown" back for the absent one, "unknown" is not
            # in SOUND_BANDS, and so an explicitly chosen reference cycle
            # could NEVER be marked REF_OK: the run page carried a permanent
            # "the integrity check could not assess it" caution that no choice
            # of cycle could clear. `choose_reference_cycle` was given
            # `exists_step` for this reason and this branch was not.
            bands = [
                integrity(ref, s)
                for s in ("Charge", "Discharge")
                if (ref, s) in signals
            ]
            if not bands:
                bands = [integrity(ref, s) for s in ("Charge", "Discharge")]
            # Anything outside SOUND_BANDS is "we could not tell", which the
            # auto path already treats as disqualifying; only ANOMALOUS used
            # to be mentioned, so a cycle whose bands were 'too few records'
            # was accepted with a bare "set explicitly" and no caveat.
            if all(b in SOUND_BANDS for b in bands):
                severity = REF_OK
                reason += "; it passes the integrity check"
            elif "ANOMALOUS" not in bands:
                reason += (
                    "; the integrity check could not assess it ("
                    + ", ".join(str(b) for b in bands)
                    + ")"
                )
            if "ANOMALOUS" in bands:
                severity = REF_BAD
                reason += "; FLAGGED ANOMALOUS by the integrity check"
                if verbose:
                    print(
                        f"\n  *** WARNING: reference cycle {ref} is flagged "
                        f"ANOMALOUS. ***"
                    )
                    print(
                        "      Its peak list will be fitted and tracked in "
                        "every cycle of this dataset."
                    )
                    print("      Pass reference_cycle=None to let Ratatosk choose.")
    else:
        ref, reason, severity = choose_reference_cycle(
            cycles,
            _exists,
            integrity=integrity,
            efficiency=efficiency,
            exists_step=lambda c, st: (c, st) in signals,
        )

    ref_peaks = {
        s: peaks.get((ref, s), pd.DataFrame()) for s in ("Charge", "Discharge")
    }
    # Each half-cycle's own voltage range, so an inferred seed can be checked
    # against the data it is about to be fitted to rather than against the
    # neighbour it was copied from.
    _windows = {}
    for _k, _sig in (signals or {}).items():
        try:
            _v = np.asarray(_sig.voltage, float)
            if _v.size:
                _windows[_k] = (float(np.nanmin(_v)), float(np.nanmax(_v)))
        except Exception:
            continue
    n_inferred = infer_missing_seeds(peaks, ref_peaks, windows=_windows)
    _refused = int(getattr(infer_missing_seeds, "refused", 0))
    if verbose and _refused:
        print(
            f"\n  {_refused} inferred seed(s) REFUSED: the neighbouring "
            f"cycle's position lies outside that half-cycle's own voltage "
            f"range."
        )
        print(
            "    A seed outside the data is not a gap to bridge; the "
            "component it would have produced is not a measurement."
        )
    if verbose and n_inferred:
        print(
            f"\n  {n_inferred} seed(s) inferred from neighbouring cycles "
            f"where detection found nothing at a reference peak."
        )
        print(
            "    They are fitted and tracked, marked is_inferred, and "
            "excluded from headline trends."
        )

    det = Detection(
        name=name,
        peaks=peaks,
        reference_cycle=int(ref),
        reference_reason=reason,
        reference_severity=severity,
        spec=spec,
        skipped=skipped,
        reference_peaks=ref_peaks,
    )

    if verbose:
        print("\n" + det.summary())
    return det
