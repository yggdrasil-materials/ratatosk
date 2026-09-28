"""
Q, V, I, t  ->  a dQ/dV curve, and whether it can be trusted.

What is ported and what is new
------------------------------
PORTED VERBATIM from 1.8.7 Module 1, extracted programmatically rather than
retyped, because 1.9.0 must reproduce every fitted value bit for bit:

    classify_dqdv_profile   auto_preprocess_params   remove_spikes
    smooth_dqdv             rebin_dqdv               _strip_cv_hold

NEW, and measurement only — nothing here changes a processed curve:

    reversals()             every contiguous run where the potential moves
                            against the applied current: how much charge
                            passed, over what voltage band, at what current
    half_cycle_report()     assembles the above per half-cycle

Why reversals are inventoried rather than judged
------------------------------------------------
dQ/dV < 0 during charge means dV/dt < 0: the potential is moving backwards
while current flows. Three different things do that, and 1.8.x conflated them
under one threshold:

    at a redox peak, at full current, every cycle   a two-phase plateau
    at the voltage limit, current tapering          a constant-voltage hold
    away from any peak, at full current, sporadic   charge going elsewhere

Only the third is a fault. Telling them apart needs the peak list, which lives
downstream — so this module measures, records where and how much, and leaves
the classification to `analyse`. The numbers it records are the ones that
distinguish the three: on the P3 cells a healthy half-cycle passes 0 mAh/g
backwards at full current away from a peak, and a sick one passes 34-103.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .compat import trapezoid
from .style import entry, bullet
import pandas as pd
from scipy.signal import savgol_filter

__all__ = ["histogram_dqdv", "dqdv_sign", "DQDV_METHOD",
           "terminal_trim", "apply_terminal_trim", "TERMINAL_TRIM",
           "TERMINAL_TRIM_FIXED_MV",
           "TERMINAL_TRIM_FACTOR", "TERMINAL_TRIM_MAX_FRACTION",
           "TERMINAL_TRIM_BAND_MV",
           "histogram_bin_for_profile", "HISTOGRAM_BIN_BY_PROFILE",
           "histogram_smooth_for_profile", "HISTOGRAM_SMOOTH_BY_PROFILE",
           "DQDV_METHODS",
           "Reversal", "HalfCycleSignal", "preprocess_half_cycle",
           "voltage_window",
           "reversals", "half_cycle_report",
           "classify_dqdv_profile", "auto_preprocess_params",
           "remove_spikes", "smooth_dqdv", "rebin_dqdv", "strip_cv_hold",
           "orient_dqdv", "ORIENT_PLATEAU_RECORDS",
           "rebin_sensitivity", "plateau_structure", "REBIN_WEIGHTED",
           "EXCLUDE_CV_HOLD", "CV_CURRENT_FRACTION", "CV_DV_FRACTION",
           "CV_MIN_POINTS"]


# =============================================================================
# PORTED FROM 1.8.7 MODULE 1 — do not edit without a bit-identity regression
# =============================================================================

EXCLUDE_CV_HOLD = True
CV_CURRENT_FRACTION = 0.99   # trailing |I| below this x the step median = hold
CV_DV_FRACTION = 0.10        # fallback: trailing |dV| below this x median
CV_MIN_POINTS = 3            # ignore shorter runs; not worth trimming

# Profile classification thresholds, from 1.8.7 Module 1.
_SHARP_IQR_FRACTION = 0.08      # IQR < 8% of range  -> two-phase plateau
_MODERATE_IQR_FRACTION = 0.20   # IQR < 20% of range -> moderate


def classify_dqdv_profile(df, verbose=True):
    """
    Classify the electrochemical profile of a dataset to guide dQ/dV
    preprocessing.

    Examines the voltage profile shape across representative half-cycles
    to determine whether the material exhibits sharp two-phase behaviour
    (e.g. LTO, LFP), moderate peaks, or broad solid-solution behaviour
    (e.g. NMC, NNM layered oxides).

    The classification uses the voltage interquartile range (IQR) as a
    fraction of the total voltage window, averaged across the first few
    stable half-cycles. This approach is robust because:

    1. Two-phase materials spend most of their capacity at a nearly
       constant voltage (the plateau), so the IQR of voltage values
       within a half-cycle is small — typically <5% of the window.

    2. Solid-solution materials have continuously varying voltage, so
       the IQR spans a substantial fraction of the window — typically
       30-50%.

    3. Unlike dQ/dV statistics, this method doesn't suffer from the
       problem of the "peak" being the bulk of the data distribution.
       The voltage profile is the ground truth that dQ/dV is derived
       from.

    Parameters
    ----------
    df : pd.DataFrame
        Raw electrochemical data with 'Voltage', 'Cycle', and 'Step'
        columns.
    verbose : bool
        Print classification diagnostics.

    Returns
    -------
    profile : dict
        {
            'class': str ('sharp', 'moderate', or 'broad'),
            'iqr_fraction': float (mean IQR/range across half-cycles),
            'plateau_fraction': float (fraction of points within
                                       +/-20mV of median voltage),
            'description': str (human-readable summary)
        }
    """
    required = ['Voltage', 'Cycle', 'Step']
    if not all(c in df.columns for c in required):
        if verbose:
            print(entry("profile", "broad", "assumed — no Voltage/Cycle/Step column"))
        return {
            'class': 'broad',
            'iqr_fraction': np.nan,
            'plateau_fraction': np.nan,
            'description': 'Missing required columns; using default (broad) profile'
        }

    # --- Analyse representative half-cycles ---
    # Use cycles 2-5 (or whatever is available after cycle 1) to avoid
    # first-cycle formation effects. Examine both charge and discharge.
    cycles = sorted(df['Cycle'].dropna().unique())
    if len(cycles) < 2:
        analysis_cycles = cycles[:1]
    else:
        analysis_cycles = [c for c in cycles if c >= 2][:4]
        if not analysis_cycles:
            analysis_cycles = cycles[1:5]  # fallback

    iqr_fractions = []
    plateau_fractions = []

    for cycle in analysis_cycles:
        df_cycle = df[df['Cycle'] == cycle]
        for step in ['Charge', 'Discharge']:
            df_step = df_cycle[df_cycle['Step'] == step]
            voltages = df_step['Voltage'].dropna()

            if len(voltages) < 20:
                continue

            v_min, v_max = voltages.min(), voltages.max()
            v_range = v_max - v_min

            if v_range < 0.01:  # less than 10 mV range — skip
                continue

            # IQR as fraction of range
            q25, q75 = voltages.quantile(0.25), voltages.quantile(0.75)
            iqr = q75 - q25
            iqr_frac = iqr / v_range
            iqr_fractions.append(iqr_frac)

            # Plateau fraction: what proportion of data points sit
            # within +/-20 mV of the median voltage?
            v_median = voltages.median()
            near_median = ((voltages >= v_median - 0.020) &
                          (voltages <= v_median + 0.020))
            plat_frac = near_median.sum() / len(voltages)
            plateau_fractions.append(plat_frac)

    if not iqr_fractions:
        if verbose:
            print(entry("profile", "broad", "assumed — no half-cycle had 20+ records spanning 10 mV"))
        return {
            'class': 'broad',
            'iqr_fraction': np.nan,
            'plateau_fraction': np.nan,
            'description': 'No analysable half-cycles; using default (broad) profile'
        }

    mean_iqr_frac = np.mean(iqr_fractions)
    mean_plat_frac = np.mean(plateau_fractions)

    # --- Classify ---
    # Primary criterion: IQR fraction
    # Secondary criterion: plateau fraction (catches edge cases)
    if mean_iqr_frac < _SHARP_IQR_FRACTION or mean_plat_frac > 0.6:
        profile_class = 'sharp'
        description = (
            f'Sharp two-phase profile. '
            f'Typical of LTO, LFP, or other flat-plateau materials. '
            f'Spike removal disabled; smoothing window reduced.'
        )
    elif mean_iqr_frac < _MODERATE_IQR_FRACTION or mean_plat_frac > 0.35:
        profile_class = 'moderate'
        description = (
            f'Moderate peak sharpness. '
            f'Mixed or moderately defined phase transitions. '
            f'Spike threshold raised; smoothing window reduced.'
        )
    else:
        profile_class = 'broad'
        description = (
            f'Broad solid-solution profile. '
            f'Typical of layered oxides (NMC, NNM, etc.). '
            f'Standard preprocessing applied.'
        )

    out = {
        'class': profile_class,
        'iqr_fraction': mean_iqr_frac,
        'plateau_fraction': mean_plat_frac,
        'description': description
    }
    # `verbose` was accepted, documented, and never read. The profile class
    # is the ONLY decision this function makes and it sets the smoothing
    # window, the spike removal, the rebin width, the histogram bin width and
    # kernel, and downstream the baseline degree and the sigma bounds — so a
    # run that does not say which class it chose does not say how any of its
    # numbers were produced. It was silent for four releases.
    if verbose:
        print(entry("profile", profile_class,
                           f"IQR/range {mean_iqr_frac:.3f}, "
                           f"plateau fraction {mean_plat_frac:.2f}"))
        print(bullet(description))
    return out


def auto_preprocess_params(profile, user_params=None):
    """
    Select preprocessing parameters based on voltage profile classification.

    Returns a complete set of preprocessing parameters appropriate for the
    detected profile type. User-supplied non-'auto' values override the
    auto-selected defaults, allowing fine-tuning without losing the
    benefit of auto-detection.

    Parameter logic by profile class:

    sharp (LTO, LFP):
        - Spike removal OFF: the genuine dQ/dV peaks are so extreme
          relative to the background that any MAD-based despiking will
          treat them as outliers. With dQ/dV values of thousands at the
          plateau and near-zero elsewhere, there is no local context
          for the rolling window to use.
        - Smoothing window 5: the peaks are very narrow in voltage
          space, so a wide window attenuates the peak height and
          distorts the shape. Window 5 with polyorder 3 preserves
          the peak while removing point-to-point noise.

    moderate:
        - Spike threshold raised to 6.0 (from default 3.0): protects
          sharper peaks while still catching genuine noise spikes.
        - Smoothing window 9: compromise between peak preservation and
          noise reduction.

    broad (NMC, NNM):
        - Original defaults: window 15, spike threshold 3.0.
        - These were tuned for layered oxide cathodes and work well.

    Parameters
    ----------
    profile : dict
        Output from classify_dqdv_profile().
    user_params : dict or None
        User-supplied parameters from Cell 3. Any value that is not 'auto'
        overrides the auto-selected value.

    Returns
    -------
    params : dict
        Complete preprocessing parameter set with keys:
        smoothing_window, polyorder, spike_removal, spike_window_size,
        spike_threshold_multiplier, auto_detected (bool), profile_class (str)
    """
    if user_params is None:
        user_params = {}

    profile_class = profile.get('class', 'broad')

    # --- Auto-selected defaults by profile class ---
    if profile_class == 'sharp':
        auto = {
            'smoothing_window': 5,
            'polyorder': 3,
            'spike_removal': False,
            'spike_window_size': 5,
            'spike_threshold_multiplier': 10.0,
            'second_smooth_window': 11,
            'rebin_width_mV': 2.0,
        }
    elif profile_class == 'moderate':
        auto = {
            'smoothing_window': 9,
            'polyorder': 3,
            'spike_removal': True,
            'spike_window_size': 5,
            'spike_threshold_multiplier': 6.0,
            'second_smooth_window': None,
            'rebin_width_mV': None,
        }
    else:  # broad
        auto = {
            'smoothing_window': 15,
            'polyorder': 3,
            'spike_removal': True,
            'spike_window_size': 5,
            'spike_threshold_multiplier': 3.0,
            'second_smooth_window': None,
            'rebin_width_mV': None,
        }

    # --- Apply user overrides (non-'auto' values take priority) ---
    final = {}
    for key in auto:
        user_val = user_params.get(key, 'auto')
        if user_val == 'auto':
            final[key] = auto[key]
        else:
            final[key] = user_val

    # The histogram path's one parameter, chosen the same way everything else
    # here is: from the measured profile class. Only set if the caller has
    # not already fixed it.
    final.setdefault('histogram_bin_mV',
                     histogram_bin_for_profile(profile_class))
    final.setdefault('histogram_smooth_bins',
                     histogram_smooth_for_profile(profile_class))

    # --- Metadata ---
    final['auto_detected'] = True
    final['profile_class'] = profile_class
    final['profile_description'] = profile.get('description', '')
    final['iqr_fraction'] = profile.get('iqr_fraction', np.nan)
    final['plateau_fraction'] = profile.get('plateau_fraction', np.nan)

    return final


# --- Preprocessing functions (defined once, used everywhere) ---


def remove_spikes(series, window_size=5, threshold_multiplier=3.0):
    """
    Detect and replace spikes using rolling median and MAD.

    Uses median-based statistics rather than mean/std to prevent the spike
    from contaminating its own detection window — a known failure mode of
    mean-based rolling outlier detection.

    The MAD (median absolute deviation) is scaled by 1.4826 to give a
    consistent estimator of the standard deviation for normally-distributed
    data (Rousseeuw & Croux, 1993).

    Parameters
    ----------
    series : pd.Series
        The dQ/dV values to despike.
    window_size : int
        Rolling window size (should be odd for centred window).
    threshold_multiplier : float
        Number of scaled MADs for spike detection threshold.

    Returns
    -------
    despiked : pd.Series
        Cleaned series with spikes replaced by rolling median.
    n_spikes : int
        Number of spikes detected and replaced.
    """
    rolling_median = series.rolling(
        window=window_size, center=True, min_periods=1
    ).median()

    # MAD scaled to approximate std for normal distributions
    abs_dev = (series - rolling_median).abs()
    rolling_mad = abs_dev.rolling(
        window=window_size, center=True, min_periods=1
    ).median() * 1.4826

    # Floor to prevent division issues in constant regions
    mad_floor = series.abs().max() * 1e-6
    rolling_mad = rolling_mad.clip(lower=mad_floor)

    is_spike = abs_dev > (threshold_multiplier * rolling_mad)
    n_spikes = int(is_spike.sum())

    despiked = series.copy()
    despiked[is_spike] = rolling_median[is_spike]

    return despiked, n_spikes


def smooth_dqdv(values, window_size, polyorder=3):
    """
    Apply Savitzky-Golay smoothing to dQ/dV data.

    Handles edge cases: None/small window sizes, even windows (auto-corrected
    to odd), insufficient data points, and polyorder >= window_size.

    Parameters
    ----------
    values : np.ndarray
        The dQ/dV values to smooth.
    window_size : int or None
        Savitzky-Golay window length. If None or <= 1, no smoothing applied.
    polyorder : int
        Polynomial order for the filter.

    Returns
    -------
    smoothed : np.ndarray
        Smoothed values (or original if smoothing not applicable).
    was_smoothed : bool
        Whether smoothing was actually applied.
    """
    if window_size is None or window_size <= 1:
        return values.copy(), False

    # Ensure window_size is odd (Savitzky-Golay requirement)
    if window_size % 2 == 0:
        window_size += 1

    if len(values) <= window_size:
        return values.copy(), False

    if polyorder >= window_size:
        polyorder = window_size - 1

    try:
        return savgol_filter(values, window_size, polyorder), True
    except Exception as e:
        print(f"  Warning: Savitzky-Golay smoothing failed ({e}). "
              f"Returning unsmoothed data.")
        return values.copy(), False


# How a bin's dQ/dV is estimated from the records inside it.
#
# 1.8.7 took the plain MEAN of the per-record dQ/dV values. Each record's
# value is dQi/dVi, so an unweighted mean gives a record spanning 0.1 mV the
# same say as one spanning 3 mV — and on a plateau the 0.1 mV records are
# both the most numerous and the least trustworthy, because their dV is at
# the cycler's resolution limit. The bin's actual charge per volt is
# sum(dQi)/sum(|dVi|), which is the |dV|-WEIGHTED mean of the same values.
#
# Measured on LTO cell A cycle 2 charge, sweeping the bin width 0.5-8 mV:
#
#             integrated area      apex height      notch at the apex
#   mean       73 - 91  (24%)      3422 -> 1752     present at 2 mV
#   weighted   77 - 82  ( 5%)      3909 -> 2739     gone at 2 mV
#
# An area that moves by a quarter when the bin width changes is not a
# measurement of anything, and the notch it leaves at the apex reads as a
# split peak. Set False to restore 1.8.7's mean.
REBIN_WEIGHTED = True


def _record_dv(voltage):
    """The voltage span each record stands for, by the mid-point rule."""
    v = np.asarray(voltage, float)
    if v.size < 2:
        return np.ones_like(v)
    e = np.empty(v.size + 1)
    e[1:-1] = 0.5 * (v[1:] + v[:-1])
    e[0], e[-1] = v[0], v[-1]
    return np.abs(np.diff(e))


# =============================================================================
# THE DERIVATIVE-FREE PATH
# =============================================================================
# Ratatosk's original curve is the CYCLER's own dQ/dV column, cleaned: sorted
# by voltage, rebinned, despiked, smoothed. That column is a numerical
# derivative, and on a flat plateau a numerical derivative is in trouble —
# dV sits in the denominator and goes to zero.
#
# Measured on the LTO triplicate: the cycler's dQ/dV column, integrated over
# its own voltage axis, recovers only 43-68% of the cycler's own delivered
# capacity. The charge is in the file; the derivative loses it.
#
# The alternative needs no derivative at all. If you bin the records by
# voltage and add up the CAPACITY each record carried, the sum in a bin
# divided by the bin width IS dQ/dV over that bin:
#
#         dQ/dV  ~  (sum of dq in the bin) / (bin width)
#
# and by construction the whole curve integrates back to the delivered
# capacity, because every record's charge is counted exactly once. That is
# the point: integral fidelity is 1 by arithmetic rather than by luck.
#
# Flores and Clark (J. Electrochem. Soc. 173, 120520, 2026) give this as a
# histogram of voltage values scaled by total capacity, which is the same
# thing when capacity is sampled at uniform intervals. The dq-weighted form
# implemented here is their recommended correction for the case where it is
# not — a rate step, a sampling change, a constant-voltage hold — and it is
# exact rather than proportional, so it is used unconditionally.
#
# What this method does NOT need, and why that matters:
#
#   * no smoothing, so no smoothing parameters chosen by eye;
#   * no despiking;
#   * no `orient_dqdv`, because a histogram does not care which way the
#     voltage was moving when a record was taken — the non-monotonic plateau
#     that inverts a derivative simply puts its charge in the right bin;
#   * one parameter, the bin width, which means exactly what it says.
#
# Measured on the same LTO half-cycles: integral fidelity 0.994-0.996 at
# every bin width from 0.5 to 5 mV, against 0.62-0.71 for the derivative.

HISTOGRAM_BIN_MV = 1.0
# ...but the right bin width depends on the narrowest feature that has to be
# resolved, exactly as the rebin width already did. A bin should be a small
# fraction of a peak's width: too fine and the histogram is sparse, noisy and
# slow to fit; too coarse and it blurs shoulders together.
#
#   sharp     LTO/LFP, FWHM ~15 mV        1 mV   ~15 bins across the peak
#   moderate                              2 mV
#   broad     layered oxides, FWHM 60+    5 mV   ~12 bins across the peak
#
# Measured: on a broad P3 profile, 1 mV bins over a 2.15 V window are mostly
# empty, and fitting the resulting noise took several times longer than the
# same half-cycle at a sensible width — for no extra information, because the
# features are 30-200 mV wide.
HISTOGRAM_BIN_BY_PROFILE = {"sharp": 1.0, "moderate": 2.0, "broad": 5.0}


def histogram_bin_for_profile(profile, default=HISTOGRAM_BIN_MV):
    """Bin width in mV for a measured profile class. See the table above."""
    cls = (profile.get("class") if isinstance(profile, dict) else profile)
    return float(HISTOGRAM_BIN_BY_PROFILE.get(str(cls), default))
# Flores and Clark report good smoothness and voltage reconstruction with a
# Gaussian kernel of one — a weighted average of each bin with its immediate
# neighbours. 0 disables it. Anything larger starts reintroducing the
# parameter-chosen-by-eye problem this method exists to avoid.
HISTOGRAM_SMOOTH_KERNEL = 1
# ...and how wide it should be depends on the profile, for the same reason the
# bin width does. A histogram's bin-to-bin scatter is COUNTING noise, and peak
# DETECTION runs on that scatter unless it is damped. On a sharp profile the
# peak towers over the noise and no amount of kernel changes what is found; on
# a broad one the features are shallow and 1 mV of scatter looks like a
# shoulder.
#
# Measured on P3 cell A, half-width in bins against peaks detected per
# half-cycle (the derivative path finds 5, range 4-7):
#
#     1 -> 13     3 -> 12     4 -> 7     6 -> 6     8 -> 6
#
# 6 gives parity with the derivative path. Integral fidelity over that whole
# sweep moves from 0.9403 to 0.9379 — a quarter of a percent, and it is an
# edge effect of the zero-padded convolution, not a loss of counted charge.
# BROAD RAISED 6 -> 16 IN 1.9.0.26, measured on the NNM (P3 NaNiMnO2)
# triplicate — 100 cycles, CE 84-89%, exactly the noisy broad curve this
# entry is for. Detection was finding a different number of peaks almost
# every cycle (discharge sd 1.16 peaks, range 3-9), which is what puts a
# reference peak list out of step with the rest of the run.
#
# The test was not "does the curve look smoother" — it was whether a feature
# PERSISTS. A real redox feature appears in nearly every cycle at the same
# voltage; noise appears sporadically. Clustering every detected peak across
# 100 cycles of NNM cell C:
#
#   smoothing   discharge count sd   persistent (>=80% of cycles)
#      6 (old)        1.16            2.026, 3.153, 3.547
#     12             0.85             2.031, 3.221, 3.547
#     16 (new)       0.61             2.036, 3.221, 3.547
#     20             0.55             2.039, 3.220, 3.547
#
# The persistent set does not change. Every real feature survives to 20 bins,
# the scatter halves by 16, and on CHARGE a third feature at 3.996 V crosses
# into persistence at 16 (86% of cycles) having been below it at 6 — so the
# smoothing finds one more real feature than it loses, which is none.
# 16 rather than 20 because the gain flattens and there is no reason to
# smooth further than the evidence supports.
#
# This CANNOT affect LTO. The value is keyed to the profile class, and the
# classifier separates the two without overlap: LTO measures IQR/range
# 0.019-0.030 with a plateau fraction of 0.64-0.68 and classifies "sharp"
# (kernel 1); NNM measures 0.386-0.444 with a plateau fraction of 0.02-0.09
# and classifies "broad". Verified unchanged on the LTO triplicate.
HISTOGRAM_SMOOTH_BY_PROFILE = {"sharp": 1, "moderate": 3, "broad": 16}


def histogram_smooth_for_profile(profile, default=HISTOGRAM_SMOOTH_KERNEL):
    """Smoothing half-width in bins for a measured profile class."""
    cls = (profile.get("class") if isinstance(profile, dict) else profile)
    return int(HISTOGRAM_SMOOTH_BY_PROFILE.get(str(cls), default))
# A bin much narrower than the voltage step between records produces empty
# bins and a sparse, noisy curve. Below this multiple of the median record
# spacing the requested width is widened, and the fact is reported.
HISTOGRAM_MIN_BIN_FACTOR = 1.0

# ...and the MEDIAN is the wrong statistic to widen against, because it
# guarantees that half the curve is under-filled. Records arrive at a fixed
# charge interval (constant current, time-based logging), so their voltage
# spacing is dQ_per_record / (dQ/dV): narrow on a peak, wide on a shelf, by
# the same factor the curve itself varies. Widening to the median leaves the
# low half of the curve with fewer records than bins, the bins there alternate
# between one record and none, and the 16-bin smoothing turns that into a
# ripple at the beat period between the record spacing and the bin width.
#
# Measured, NMC111 cell A cycle 5 charge, the 4.10-4.45 V shelf:
#
#   path / bin            empty bins   mean occupancy   maxima found on shelf
#   derivative (1.8.7)         -             -                    0
#   histogram, 6.3 mV        60/223         0.87                  4
#   histogram, 8 mV          36/175         1.11                  0
#   histogram, 10 mV         24/140         1.39                  0
#   histogram, 20 mV         11/70          2.79                  0
#
# The raw Q(V) across that shelf is a straight line (dQ per record constant to
# 0.45%, dV per record 6.6-7.1 mV), so there is nothing there to resolve. The
# four maxima are the beat, and they are gone the moment the bins can be
# filled. They are also absent from the derivative path, which has no bins to
# beat against — which is how the artefact was first noticed.
#
# Mean occupancy is exactly `n_records * width / span`, so requiring a minimum
# mean occupancy is the same thing as widening to the MEAN record spacing.
# 1.0 is the floor with a name: never return a curve with fewer measurements
# in it than bins. It is not a smoothing parameter and it is not tuned to a
# chemistry -- it is the same statement as `sigma_min` from the bin width and
# `WIDTH_MIN_SAMPLES` from the sampling interval.
# THE BIN WIDTH IS SET FROM AN AVERAGE OVER A CURVE WHOSE RECORD DENSITY
# VARIES A HUNDREDFOLD, AND ON A SHARP PROFILE THE PEAK PAYS FOR THE SHELF.
# Measured in 1.9.0.60; the constant is UNCHANGED, because no value of it
# fixes this and the honest answer is to report the limit.
#
# `w_occ` below asks for one record per bin ON AVERAGE across the window. The
# paragraph under `counts` already explains why that average is not the
# quantity that matters: records arrive at a fixed charge interval, so their
# voltage spacing is dQ_per_record / (dQ/dV) and is widest exactly where the
# curve is lowest. The average is therefore set by the SHELF, and the PEAK is
# binned at the shelf's scale.
#
# LTO cell A cycle 5 charge, 373 records: 220 of them (59%) lie in the 13 mV
# across the peak's half maximum, 0.06 mV apart. The rule asks for 3.49 mV
# bins, 3.26 mV after rounding — fifty times coarser than the records there —
# and 236 of the 373 bins hold nothing at all. The peak is described by five
# bins whose occupancies are 1, 107, 47, 39, 26: the leading flank, which is
# where the asymmetry ratio comes from, is one record.
#
# SWEPT over bin width on the LTO triplicate, 64 half-cycles, same detection
# and same fitter, with this floor disabled so the request is honoured:
#
#     bin mV   FWHM/bins   fitted FWHM   sigma   sigma_r   charge k    R2
#       0.50       9.5         4.56       1.89     1.71       5.5     0.9596
#       1.00       7.0         5.57       2.34     3.35       8.3     0.9841
#       2.00       6.0         7.82       3.77     3.16      12.1     0.9940
#       3.26*      5.0        11.12       4.82     4.86       3.4     0.9972
#       5.00       5.0        14.53       6.17     6.35       1.9     0.9979
#                                                       (* what this rule picks)
#
# THE FITTED FWHM IS 2.9x THE BIN WIDTH AT EVERY WIDTH TESTED, over a factor
# of ten. It is the histogram's width, not the peak's. The asymmetry ratio has
# no stable value either — 1.9 to 12.1 on charge, and within a single width it
# ranges 0.11 to 20.0, which is the bound.
#
# AND R2 RISES MONOTONICALLY AS THE BINS COARSEN, 0.960 to 0.998, which is the
# trap: coarser bins are smoother, so a smooth model fits them better. R2 here
# measures how much the histogram has smoothed the data. It is the third time
# in this project that R2 has moved the opposite way to the answer.
#
# WHY NOT JUST BIN FINER. At 0.5 mV the peak gets 9.5 bins — but the shelf is
# then 99% empty, and empty shelf bins are what the quantisation ripple of
# 1.9.0.56 was made of. There is no width that serves both, because the two
# regions differ by two orders of magnitude in record density. Equal-occupancy
# binning is the obvious non-uniform answer and was measured in 1.9.0.55: it
# was worse, with 143 of 186 LTO bins degenerate.
#
# SO THIS IS A LIMIT OF THE MEASUREMENT, not of the code. The remedy is at the
# cycler — log on dV as well as dt, so records are spread in voltage instead of
# piling up on the plateau — and until then the widths on a sharp profile are
# not measurements and the report says so (`report._lineshape_sentence`).
#
# IT IS CONFINED TO `sharp`. Bins across the fitted FWHM, production build:
# LTO 3.7-3.8 (57-70% flagged `width_undersampled`), NNM 17.0-21.3 (0-1 of
# 3834 flagged), NMC 27.7-31.4 (none). The AREAS are unaffected — integral
# fidelity is 0.994-0.996 at every width from 0.5 to 5 mV — so what is
# withheld is the width and its ratio, not the capacity.
HISTOGRAM_MIN_OCCUPANCY = 1.0

# --- how a record's charge is assigned to a bin ---------------------------
# COUNTING (False) drops the whole of an increment's charge into the bin that
# contains the increment's MIDPOINT voltage. A bin's value is then an integer
# multiple of dq_per_record / width, and where the bins are finer than the
# records that integer is 0 or 1: the curve becomes a telegraph signal and the
# smoothing kernel turns it into a ripple at the beat period between the
# record spacing and the bin width. Measured on NMC111 cell A cycle 5 charge,
# 4.15-4.42 V, at 7.18 mV bins: 108 108 107 107 107 108 108 108 109 111 113
# 116 119 122 -- against a raw Q(V) over that span which is a straight line.
#
# EXACT (True) uses the fact that an increment is not a point. It spans
# [v_i, v_i+1], and the charge that passed inside a bin's edges is the part of
# that span the bin covers. Spreading each increment across the bins it
# actually overlaps is exact charge accounting -- it conserves the total by a
# telescoping sum, it introduces no parameter, and it reduces to the counting
# behaviour whenever an increment lies inside one bin. It is the same thing as
# interpolating Q(V) at the bin edges and differencing, which is what the
# quantity means. The same half-cycle, same bins, exact assignment:
# 110 110 111 111 112 113 113 113 113 113 112 112 112 112.
#
#   NMC111 cell A charge, identical bin width      counting     exact
#     empty bins                                     44           0
#     curvature scatter on the shelf                 0.28%        0.07%
#     maxima detected on the 4.1-4.45 V shelf        2            0
#     components detected in the half-cycle          6            3
#
# Zero-span increments -- a plateau where the recorded voltage does not move,
# which is most of an LTO half-cycle -- carry their whole charge at one
# voltage and go wholly into the bin containing it, as before. That case is
# why the histogram exists and it is not changed here.
HISTOGRAM_EXACT_CHARGE = True


def _charge_in_bins(v0, v1, dq, edges):
    """
    Charge in each bin, spreading each increment over the span it covers.

    `v0`/`v1` are the voltages either side of an increment carrying `dq`; the
    order of the pair does not matter. Returns one total per bin.

    Built as a cumulative: G(e) is the charge that passed below voltage `e`,
    so the charge in a bin is a difference of two values of G and the total
    over all bins telescopes to the charge that entered the calculation.
    """
    lo = np.minimum(v0, v1)
    hi = np.maximum(v0, v1)
    span = hi - lo
    moving = span > 0
    G = np.zeros(edges.size, dtype=float)

    if np.any(moving):
        l = lo[moving]; h = hi[moving]; w = dq[moving]
        # Chunked so a long half-cycle cannot build a huge intermediate.
        chunk = max(1, int(4_000_000 // max(edges.size, 1)))
        for a in range(0, l.size, chunk):
            b = min(a + chunk, l.size)
            frac = np.clip(
                (edges[None, :] - l[a:b, None]) / (h[a:b, None] - l[a:b, None]),
                0.0, 1.0)
            G += (w[a:b, None] * frac).sum(axis=0)

    if np.any(~moving):
        # A stationary increment is all at one voltage: it belongs to the bin
        # containing it, and to every cumulative value above that bin.
        k = np.searchsorted(edges, lo[~moving], side="right")
        k = np.clip(k, 1, edges.size - 1)
        add = np.zeros(edges.size, dtype=float)
        np.add.at(add, k, dq[~moving])
        G += np.cumsum(add)

    return np.diff(G)


def histogram_dqdv(voltage, capacity, *, bin_width_mV=None,
                   window=None, smooth_kernel=None, sign=1.0,
                   phase=0.0):
    """
    dQ/dV without differentiating anything. See the block comment above.

    `voltage` and `capacity` are one half-cycle IN ACQUISITION ORDER — the
    order matters, because each record's charge increment is the difference
    from the record before it. `sign` is applied at the end and is Ratatosk's
    display convention (+1 on the half-cycle labelled Charge, -1 on
    Discharge); the magnitude does not depend on it.

    Returns `(centres, dqdv, info)`. `info` carries the bin width actually
    used, the number of empty bins, and the charge that fell outside `window`
    — all three are how you tell a good binning from a bad one.
    """
    # READ THE MODULE CONSTANTS AT CALL TIME, not as default arguments. A
    # default is evaluated once when the function is defined, so
    # `smooth_kernel=HISTOGRAM_SMOOTH_KERNEL` freezes whatever the constant
    # was at import and silently ignores any later change — which is the same
    # import-time binding that once left every cycling figure at the wrong
    # size. Setting the constant has to work, because that is how a build
    # flag is meant to be used.
    if bin_width_mV is None:
        bin_width_mV = HISTOGRAM_BIN_MV
    if smooth_kernel is None:
        smooth_kernel = HISTOGRAM_SMOOTH_KERNEL

    v = np.asarray(voltage, float)
    q = np.asarray(capacity, float)
    m = np.isfinite(v) & np.isfinite(q)
    v, q = v[m], q[m]
    info = dict(bin_width_mV=float(bin_width_mV), n_bins=0, n_empty=0,
                charge_in_window=np.nan, charge_total=np.nan, widened=False,
                widened_by="", unvisited_mV=0.0)
    if v.size < 5:
        return np.empty(0), np.empty(0), info

    # Each record carries the charge passed since the previous one. The first
    # record carries none — it is the start of the count, not an increment.
    dq = np.abs(np.diff(q))
    v_lo_all, v_hi_all = v[:-1], v[1:]  # the span the charge was passed ACROSS
    v_mid = 0.5 * (v[1:] + v[:-1])      # the charge was passed BETWEEN them
    good = np.isfinite(dq) & np.isfinite(v_mid)
    dq, v_mid = dq[good], v_mid[good]
    v_lo_all, v_hi_all = v_lo_all[good], v_hi_all[good]
    if dq.size < 4 or not np.any(dq > 0):
        return np.empty(0), np.empty(0), info
    info["charge_total"] = float(dq.sum())

    lo, hi = (float(np.min(v_mid)), float(np.max(v_mid))) if window is None \
        else (float(window[0]), float(window[1]))
    inside = (v_mid >= lo) & (v_mid <= hi)
    info["charge_in_window"] = float(dq[inside].sum())
    if not np.any(inside) or hi <= lo:
        return np.empty(0), np.empty(0), info

    # --- BIN ONLY WHAT THE HALF-CYCLE ACTUALLY VISITED -------------------
    # The window is computed once per DATASET, deliberately (see
    # `voltage_window`): trimming each half-cycle to its own extremes removes
    # real data from every one of them. But a dataset window is a union, and
    # an individual half-cycle need not reach both ends of it -- on reversing
    # the current the cell jumps by its overpotential, carrying no charge, so
    # a charge step can begin a long way above the window's foot.
    #
    # Measured, NMC111 cell A cycle 5: the discharge before it ended at 3.000
    # V under load, the charge step opens at 3.2847 V with Q = 0.00 for its
    # first two records, and the window starts at 3.050 V. Thirty-seven bins
    # -- 233 mV -- therefore contained NO RECORDS AT ALL, the curve was flat
    # zero across them, and the cubic baseline, with nothing to constrain it,
    # climbed to 48 mAh/g/V in there and booked 5 mAh/g of capacity in a
    # region the cell never entered. That is not a fitting problem; it is a
    # model defined outside its data.
    #
    # So the GRID is built over the visited span, clipped to the window.
    # Empty bins INSIDE that span are kept -- they are real gaps in a region
    # the cell did traverse, and they carry information. This is not the
    # per-half-cycle trim that `voltage_window` warns about: nothing measured
    # is discarded, and the dataset-wide 50 mV trim still applies.
    _seen_lo = float(np.min(np.minimum(v_lo_all[inside], v_hi_all[inside])))
    _seen_hi = float(np.max(np.maximum(v_lo_all[inside], v_hi_all[inside])))
    _lo2, _hi2 = max(lo, _seen_lo), min(hi, _seen_hi)
    if _hi2 - _lo2 > 0:
        info["unvisited_mV"] = ((_lo2 - lo) + (hi - _hi2)) * 1000.0
        lo, hi = _lo2, _hi2

    # A bin narrower than the records themselves cannot be filled. Two floors,
    # and the wider one wins: the median record spacing (local scale), and the
    # width that gives at least HISTOGRAM_MIN_OCCUPANCY records per bin on
    # average across the window (global count). See both constants above.
    step = float(np.median(np.abs(np.diff(np.sort(v_mid))))) if v_mid.size > 2 \
        else 0.0
    width = float(bin_width_mV) / 1000.0
    n_rec = int(np.count_nonzero(inside))
    w_occ = (float(HISTOGRAM_MIN_OCCUPANCY) * (hi - lo) / n_rec) \
        if n_rec > 0 else 0.0
    floor = max(HISTOGRAM_MIN_BIN_FACTOR * step, w_occ)
    if floor > 0 and width < floor:
        width = floor
        info["widened"] = True
        info["widened_by"] = ("occupancy" if w_occ >= HISTOGRAM_MIN_BIN_FACTOR * step
                              else "record spacing")
        info["bin_width_mV"] = width * 1000.0

    # THE PHASE OFFSET IS APPLIED HERE, AFTER THE VISITED-SPAN CLIP.
    # `phase_stable` asks whether a detected feature survives sliding the bin
    # grid by a fraction of a bin — a wiggle that moves with the grid is the
    # grid, not the material. `_rebin_at_phase` implemented that by shifting
    # the WINDOW, and the clip above then overwrote `lo` with `_seen_lo`, the
    # lowest voltage the half-cycle actually visited, which does not depend on
    # the phase at all. Whenever a half-cycle opens above the window's foot —
    # the normal case for a charge step, and the case the clip was written
    # for — all four phases produced a byte-identical grid, every candidate
    # appeared at exactly the same voltage in each, and PHASE_INVARIANCE
    # rejected nothing. Offsetting the grid anchor itself cannot be undone by
    # a later clip.
    _ph = float(phase or 0.0)
    if _ph:
        lo = lo - (_ph - np.floor(_ph)) * width
    n_bins = max(4, int(np.ceil((hi - lo) / width)))
    edges = lo + width * np.arange(n_bins + 1)
    # `weights=dq` is the whole method: the bin's value is the CHARGE that
    # passed inside it, not the number of records that happened to land there.
    if HISTOGRAM_EXACT_CHARGE:
        totals = _charge_in_bins(v_lo_all[inside], v_hi_all[inside],
                                 dq[inside], edges)
    else:
        totals, _ = np.histogram(v_mid[inside], bins=edges, weights=dq[inside])
    centres = 0.5 * (edges[:-1] + edges[1:])
    y = totals / width                      # mAh/g per volt

    # OCCUPANCY: how many raw records actually landed in each bin.
    #
    # `weights=dq` above is what makes this a charge histogram rather than a
    # counting one, and it is right — but it means the curve carries no record
    # of how much MEASUREMENT is behind each value. That matters, because the
    # bin width is chosen from the MEDIAN record spacing over the half-cycle
    # while the actual spacing varies with dQ/dV across it: records arrive at
    # a fixed charge interval (constant current, time-based logging), so their
    # voltage spacing is dQ_per_record / (dQ/dV) and is WIDEST exactly where
    # the curve is lowest. A bin width that is comfortable on a peak is finer
    # than the records on a shelf, and the bins there alternate between one
    # record and none. Smoothing then spreads charge across the empty ones and
    # returns a curve whose fine structure is the beat between the record
    # spacing and the bin width — a ripple of order 1/occupancy, at a period
    # that moves as the record spacing does, i.e. between cycles.
    #
    # Measured on NMC111 cell A charge, 5 mV requested and widened to the
    # median: occupancy 1.06 rec/bin below 3.9 V but 0.78 above 4.1 V with 26%
    # of bins empty (cycle 5), falling to 0.57 and 42% by cycle 13 as capacity
    # fades and there are fewer records to go round. The raw Q(V) over that
    # same span is a straight line — dQ per record constant to 0.45%, dV per
    # record 6.6-7.1 mV — so there is nothing there to resolve, and the four
    # to five maxima detection finds on that shelf are the beat, not chemistry.
    #
    # So the count is carried alongside the curve and detection is entitled to
    # refuse a region that has fewer measurements than bins. Same family as
    # `sigma_min` from the bin width and `WIDTH_MIN_SAMPLES` from the sampling
    # interval: a statement about what the measurement can support.
    counts, _ = np.histogram(v_mid[inside], bins=edges)
    info["n_bins"] = int(n_bins)
    info["n_empty"] = int(np.sum(totals == 0))
    info["occupancy"] = counts.astype(float)
    info["occupancy_median"] = float(np.median(counts)) if counts.size else np.nan
    # The charge one record carries. On a constant-current step with
    # time-based logging this is a constant (measured 0.45% CV on NMC111), and
    # it is the QUANTUM of the histogram: a bin's value can only be an integer
    # multiple of `dq_per_record / width`. See `quantisation_ripple`.
    _dqi = dq[inside]
    info["dq_per_record"] = float(np.median(_dqi[_dqi > 0])) \
        if np.any(_dqi > 0) else np.nan

    # THE KERNEL IS TRIMMED TO FIT THE CURVE. `np.convolve(..., mode="same")`
    # returns an array of length max(len(signal), len(kernel)) — NOT the
    # length of the signal, which is what "same" reads as. A half-cycle short
    # enough to give fewer bins than the kernel has taps therefore came back
    # with MORE dQ/dV values than voltages, and detection died on
    # `operands could not be broadcast together with shapes (12,) (13,)`
    # rather than skipping a half-cycle that was never a curve. Two of P3
    # cell A's 440 half-cycles do this: five records each, binned at a
    # widened 185 mV into twelve bins, against a thirteen-tap kernel.
    #
    # Trimmed rather than skipped, so the behaviour degrades smoothly: a
    # curve with room for a narrower kernel gets one, and a curve with room
    # for none is left unsmoothed. Either way the array lengths agree, and
    # the binomial kernel still sums to one, so the area is still conserved.
    if smooth_kernel and smooth_kernel > 0 and y.size > 2:
        smooth_kernel = int(min(int(smooth_kernel), (y.size - 1) // 2))
    if smooth_kernel and smooth_kernel > 0 and y.size > 2:
        # A BINOMIAL kernel — the discrete Gaussian — applied to the RESULT,
        # not to the voltage record, so it cannot change which bin a record's
        # charge was counted in. It sums to exactly one, so it conserves the
        # total area exactly: the capacity accounting of this method survives
        # any amount of it. `smooth_kernel` is its half-width in bins.
        #
        # This is not the trial-and-error smoothing the histogram method
        # exists to avoid. That kind is applied to the VOLTAGE before
        # differentiating, where it changes the derivative and the integral
        # together and its parameters are unrecoverable. This is applied
        # afterwards, changes nothing that has been counted, and exists only
        # so that peak DETECTION is not run on counting noise.
        k = np.array([1.0])
        for _ in range(2 * int(smooth_kernel)):
            k = np.convolve(k, [0.5, 0.5])
        y = np.convolve(y, k / k.sum(), mode="same")

    # Belt and braces. Every return above this line is a pair built from the
    # same `edges`, and the one operation that could break that is fixed
    # above — but a curve whose two halves disagree in length is the kind of
    # thing that should never leave this function silently.
    if y.size != centres.size:                                  # pragma: no cover
        raise AssertionError(
            f"histogram_dqdv produced {centres.size} voltages and {y.size} "
            f"dQ/dV values; they are built from one set of bin edges and "
            f"cannot differ.")
    return centres, sign * y, info


def binomial_kernel(half_width):
    """The discrete Gaussian used to smooth a histogram, normalised to 1."""
    k = np.array([1.0])
    for _ in range(2 * int(half_width)):
        k = np.convolve(k, [0.5, 0.5])
    return k / k.sum()


def quantisation_ripple(centres, dqdv, *, dq_per_record, bin_width_V,
                        smooth_kernel, window_bins=None):
    """
    How much structure would this binning invent on a perfectly smooth curve?

    THE NULL MODEL. A histogram dQ/dV counts records into voltage bins, and
    each record carries the same charge, so a bin's value before smoothing can
    only be an integer multiple of `dq_per_record / bin_width`. Where the bins
    hold one record or none that quantum is the SIZE OF THE CURVE — on NMC111
    cell A cycle 5 charge, 0.77 mAh/g into a 7.2 mV bin is 107 mAh/g/V against
    a shelf standing at 118 — and the bins alternate between 0, one quantum
    and two. Smoothing damps that telegraph signal but cannot remove it: what
    comes out is a ripple at the beat period between the record spacing and
    the bin width, which moves between cycles because the record spacing does.

    So: take the delivered curve as the truth, work out where records WOULD
    have landed if it were exactly that smooth (equal charge intervals along
    its own integral), bin and smooth them the same way, and compare. The
    difference is structure this binning manufactures from a smooth input.

    Nothing here is fitted or tuned. The inputs are this half-cycle's own bin
    width, its own smoothing kernel and its own charge per record; the output
    is in the same units as the curve, so detection can ask the only question
    that matters — is this maximum bigger than what the binning invents?

    Returns a per-bin ripple amplitude (local RMS), aligned with `centres`.
    """
    y = np.abs(np.asarray(dqdv, float))
    v = np.asarray(centres, float)
    n = y.size
    if n < 8 or not np.isfinite(dq_per_record) or dq_per_record <= 0 \
            or not np.isfinite(bin_width_V) or bin_width_V <= 0:
        return np.zeros(max(n, 0))

    # Where the records would fall on a curve exactly this smooth.
    q = np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(v))))
    q_tot = float(q[-1])
    n_rec = int(round(q_tot / float(dq_per_record)))
    if n_rec < 4:
        return np.zeros(n)
    q_rec = (np.arange(1, n_rec + 1) - 0.5) * float(dq_per_record)
    q_rec = q_rec[q_rec <= q_tot]
    if q_rec.size < 4:
        return np.zeros(n)
    v_rec = np.interp(q_rec, q, v)

    edges = np.empty(n + 1)
    edges[:-1] = v - 0.5 * bin_width_V
    edges[-1] = v[-1] + 0.5 * bin_width_V
    counts, _ = np.histogram(v_rec, bins=edges)
    y_null = counts * float(dq_per_record) / float(bin_width_V)

    # Smooth BOTH the same way, so the comparison isolates the quantisation
    # and not the kernel's own low-pass or its edge droop.
    kk = int(smooth_kernel or 0)
    if kk > 0 and n > 2:
        kk = int(min(kk, (n - 1) // 2))
    if kk > 0 and n > 2:
        k = binomial_kernel(kk)
        y_null = np.convolve(y_null, k, mode="same")
        y_ref = np.convolve(y, k, mode="same")
    else:
        y_ref = y

    resid = y_null - y_ref

    # Local RMS, over a span wide enough to contain a beat period but narrow
    # enough that the amplitude still tracks the curve (it scales as
    # 1/occupancy, so it is largest where the curve is lowest).
    if window_bins is None:
        window_bins = max(9, 2 * (kk if kk > 0 else 4) + 1)
    w = int(window_bins) | 1
    pad = w // 2
    padded = np.pad(resid ** 2, pad, mode="edge")
    csum = np.concatenate(([0.0], np.cumsum(padded)))
    ms = (csum[w:] - csum[:-w]) / w
    return np.sqrt(np.maximum(ms[:n], 0.0))


def dqdv_sign(step):
    """
    Ratatosk's display convention: +1 on Charge, -1 on Discharge.

    Read off the STEP LABEL, not the direction of travel — on an anode with
    swapped labels the half-cycle called Charge is the one whose voltage
    falls, and it still carries a positive dQ/dV. Verified against both the
    LTO (labels swapped) and P3 (not swapped) exports.
    """
    return 1.0 if str(step).lower().startswith("charge") else -1.0


def rebin_dqdv(voltage, dqdv, bin_width_mV=2.0, weighted=None):
    """
    Re-bin dQ/dV onto a uniform voltage grid.

    On flat-plateau materials (LTO, LFP) the cycler records many points at
    near-identical voltages, because the voltage barely changes between
    successive capacity increments. That produces duplicate and
    near-duplicate voltages carrying wildly different dQ/dV, depending on
    whether a sub-resolution dV rounded up or down. Savitzky-Golay cannot
    repair it, because it works on point index and the voltage spacing is
    grossly uneven; binning onto a regular grid is what makes the curve a
    function of voltage again.

    The bin's value is the |dV|-weighted mean of the records in it, which is
    sum(dQ)/sum(|dV|) — the charge that bin actually passed, per volt. See
    REBIN_WEIGHTED for what the unweighted mean did instead.

    Returns `(v_binned, dqdv_binned)`; bins holding no record are dropped.
    """
    if weighted is None:
        weighted = REBIN_WEIGHTED
    bin_width_V = bin_width_mV / 1000.0
    voltage = np.asarray(voltage, float)
    dqdv = np.asarray(dqdv, float)
    v_min, v_max = voltage.min(), voltage.max()

    bin_edges = np.arange(v_min, v_max + bin_width_V, bin_width_V)
    bin_indices = np.digitize(voltage, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    w = _record_dv(voltage) if weighted else np.ones_like(voltage)

    v_binned = []
    dqdv_binned = []

    for i in range(len(bin_edges) - 1):
        mask = bin_indices == i
        if not mask.any():
            continue
        v_binned.append((bin_edges[i] + bin_edges[i + 1]) / 2.0)
        y, ww = dqdv[mask], w[mask]
        ok = np.isfinite(y) & np.isfinite(ww)
        tot = ww[ok].sum() if ok.any() else 0.0
        if ok.any() and tot > 0:
            dqdv_binned.append(float(np.sum(y[ok] * ww[ok]) / tot))
        else:
            # Every record in the bin sits at one voltage, so there is no
            # span to weight by. The mean is all that is left.
            dqdv_binned.append(float(np.nanmean(y)) if ok.any() else np.nan)

    return np.array(v_binned), np.array(dqdv_binned)


def plateau_structure(voltage, capacity, n=40):
    """
    Read the voltage curve directly: is there really a second plateau?

    A second dQ/dV peak REQUIRES a second plateau in V(Q) — the voltage must
    flatten, steepen, and flatten again. This asks that question of V(Q)
    rather than of dQ/dV, and needs no smoothing, binning or orientation to
    do it, because Q is a monotonic integral where V is a quantised
    measurement. It is therefore free of every processing choice that could
    manufacture the feature being tested.

    `voltage` and `capacity` are the RAW records of one half-cycle.

    Returns a dict with `flattest_V` (where |dV/dQ| is smallest — the peak),
    `second_flattest_V` (the flattest point at least 15 mAh/g away), their
    separation, and `n_minima`, the count of interior local minima of
    |dV/dQ|.

    How to read it, from the LTO triplicate:

      * cell B's charge plateau is flattest at 1.546 +/- 1 mV in cycles
        3-10, with the second-flattest 1-3 mV away. ONE plateau, sampled
        twice.
      * `n_minima` runs 2-8 per half-cycle at different voltages every cycle
        and every cell. A phase transition does not move, so those are the
        cycler's 0.1 mV voltage resolution showing through 1/(dV/dQ), not
        chemistry.

    Two further tests no function can do for you, and both matter more than
    anything above: does the feature recur at the SAME voltage in later
    cycles, and does it appear in the other cells of the triplicate? A
    feature that fails either is not a property of the material.
    """
    v = np.asarray(voltage, float)
    q = np.asarray(capacity, float)
    m = np.isfinite(v) & np.isfinite(q)
    v, q = v[m], q[m]
    if v.size < 20:
        return {"flattest_V": np.nan, "second_flattest_V": np.nan,
                "separation_mV": np.nan, "n_minima": 0}
    o = np.argsort(q, kind="mergesort")
    v, q = v[o], q[o]
    qg = np.linspace(q[5], q[-5], n)
    vg = np.interp(qg, q, v)
    a = np.abs(np.gradient(vg, qg))
    k = int(np.argmin(a[3:-3])) + 3
    far = [i for i in range(3, len(a) - 3) if abs(qg[i] - qg[k]) > 15]
    k2 = far[int(np.argmin(a[far]))] if far else None
    n_min = sum(1 for i in range(2, len(a) - 2)
                if a[i] < a[i - 1] and a[i] < a[i + 1]
                and a[i] < a[i - 2] and a[i] < a[i + 2])
    return {"flattest_V": float(vg[k]),
            "second_flattest_V": float(vg[k2]) if k2 else np.nan,
            "separation_mV": (abs(float(vg[k2]) - float(vg[k])) * 1000.0
                              if k2 else np.nan),
            "n_minima": int(n_min)}


def rebin_sensitivity(voltage, dqdv, widths=(1.0, 2.0, 4.0)):
    """
    Does a feature survive a change of bin width, or is it the binning?

    A peak in the material is a property of the cell and does not care what
    grid it was put on. A notch or a shoulder that appears at one bin width
    and not at another is the grid. Returns a dict with the apex voltage and
    height at each width, their spreads, and `apex_stable` / `area_stable`.

    This is NECESSARY AND NOT SUFFICIENT, and the distinction cost a
    version. It rules out the binning; it says nothing about any other fixed
    stage of the pipeline, so a feature can be perfectly grid-stable and
    still be manufactured. Use `plateau_structure` to settle it.
    """
    out = {"widths": list(widths), "apex_V": [], "apex_height": [], "area": []}
    for w in widths:
        v, y = rebin_dqdv(voltage, dqdv, bin_width_mV=w)
        if v.size < 3:
            out["apex_V"].append(np.nan)
            out["apex_height"].append(np.nan)
            out["area"].append(np.nan)
            continue
        j = int(np.nanargmax(np.abs(y)))
        out["apex_V"].append(float(v[j]))
        out["apex_height"].append(float(y[j]))
        out["area"].append(float(trapezoid(np.abs(y), v)))
    a = np.array(out["apex_V"], float)
    ar = np.array(out["area"], float)
    out["apex_spread_mV"] = float(np.nanmax(a) - np.nanmin(a)) * 1000.0
    out["area_spread_frac"] = float(
        (np.nanmax(ar) - np.nanmin(ar)) / np.nanmedian(ar)) if np.isfinite(
        np.nanmedian(ar)) and np.nanmedian(ar) else np.nan
    out["apex_stable"] = bool(out["apex_spread_mV"] <= 10.0)
    out["area_stable"] = bool(out["area_spread_frac"] <= 0.10)
    return out


# Orient the plateau records that the voltage record sends backwards.
#
# On a two-phase plateau the true dV per record falls below the cycler's
# 0.1 mV voltage resolution, so the recorded voltage wobbles: in LTO cell A
# cycle 1's charge, 113 of 401 records (28%) step BACKWARDS in voltage while
# the cell is charging steadily. The instrument still divides its positive
# dQ by that negative dV, so those records carry a dQ/dV of the wrong sign
# and, because dV is tiny, a magnitude 100-200x the honest records' — median
# |dQ/dV| of 2433 against 13.
#
# Preprocessing then SORTS BY VOLTAGE, which interleaves them with the
# forward records at the same voltages, and Savitzky-Golay averages across
# two populations of opposite sign. That is the up-and-down pattern on the
# first cycle; at the LTO plateau it is worse than cosmetic, because the
# smoothed curve crosses zero and the peak comes out INVERTED (+299 on one
# flank, -1233 at the apex).
#
# The records themselves are sound. Through every one of those 113 the
# current is constant at -0.206 mA and the capacity counter rises
# monotonically: the cell is doing exactly one thing. Only the sign of a
# sub-resolution dV is in doubt, so only the sign is corrected, and only
# where the current has NOT reversed and charge is still accumulating. A
# genuine reversal — current or charge actually going backwards — is left
# exactly as recorded, because that is a finding, and `analyse.reversals`
# reads it from the raw records before any of this runs.
#
# Set False to reproduce 1.8.7, which had no such step.
ORIENT_PLATEAU_RECORDS = True


def orient_dqdv(df_step, direction, *, capacity_col=None,
                current_col="Current(A)"):
    """
    Give every record the sign its own half-cycle's honest records carry.

    Returns `(df, n_oriented)`. Operates in ACQUISITION order, before any
    sort, because it is the sort that mixes the two populations.
    """
    if not ORIENT_PLATEAU_RECORDS or len(df_step) < 3:
        return df_step, 0
    d = df_step
    v = pd.to_numeric(d["Voltage"], errors="coerce").to_numpy()
    y = pd.to_numeric(d["dQ/dV"], errors="coerce").to_numpy()
    if not np.isfinite(y).any():
        return df_step, 0

    dv = np.diff(v, prepend=v[0])
    forward = np.sign(dv) == np.sign(direction)
    # The target sign comes from the records whose voltage moved the way the
    # half-cycle is going: those are the ones whose dQ/dV sign is not in
    # question. Fall back to the signed area if there are too few.
    ref = y[forward & np.isfinite(y)]
    target = (np.sign(np.median(ref)) if ref.size >= 3
              else np.sign(np.nansum(y)))
    if target == 0:
        return df_step, 0

    # Charge still going the same way? Both tests must pass, so a genuine
    # reversal is never quietly rectified into a peak.
    ok = ~forward & np.isfinite(y) & (np.sign(y) != target)
    if current_col in d.columns:
        i = pd.to_numeric(d[current_col], errors="coerce").to_numpy()
        i_ref = np.nanmedian(i)
        if np.isfinite(i_ref) and i_ref != 0:
            ok &= (np.sign(i) == np.sign(i_ref))
    if capacity_col and capacity_col in d.columns:
        q = pd.to_numeric(d[capacity_col], errors="coerce").to_numpy()
        dq = np.diff(q, prepend=q[0])
        ok &= ~(dq < 0)          # NaN-safe: only a real decrease disqualifies

    n = int(ok.sum())
    if not n:
        return df_step, 0
    out = d.copy()
    yy = y.copy()
    yy[ok] = np.abs(yy[ok]) * target
    out["dQ/dV"] = yy
    return out, n


def strip_cv_hold(df_step):
    """Drop the trailing constant-voltage hold. Returns (df, n_removed, how).

    Operates in ACQUISITION order (the frame's own index), because the hold is
    defined by what happened last in time, not by where it sits in voltage.
    """
    if not EXCLUDE_CV_HOLD or len(df_step) < 2 * CV_MIN_POINTS:
        return df_step, 0, 'off'
    d = df_step.sort_index()

    def _trailing(mask):
        k, n = len(mask) - 1, 0
        while k >= 0 and bool(mask[k]):
            n += 1
            k -= 1
        return n

    cur_col = next((c for c in ('Current(A)', 'Current', 'current(A)')
                    if c in d.columns), None)
    if cur_col is not None:
        i = pd.to_numeric(d[cur_col], errors='coerce').abs().values
        if np.isfinite(i).sum() >= CV_MIN_POINTS:
            med = np.nanmedian(i)
            if np.isfinite(med) and med > 0:
                n = _trailing(np.nan_to_num(i, nan=np.inf)
                              < med * CV_CURRENT_FRACTION)
                if n >= CV_MIN_POINTS:
                    return d.iloc[:len(d) - n], n, 'current decay'
                return df_step, 0, 'current decay'

    v = pd.to_numeric(d['Voltage'], errors='coerce').values
    dv = np.abs(np.diff(v))
    pos = dv[dv > 0]
    if pos.size == 0:
        return df_step, 0, 'no criterion'
    med = np.median(pos)
    n = _trailing(np.r_[False, dv < med * CV_DV_FRACTION])
    if n >= CV_MIN_POINTS:
        return d.iloc[:len(d) - n], n, 'voltage stall'
    return df_step, 0, 'voltage stall'


# =============================================================================
# NEW IN 1.9.0 — measurement only
# =============================================================================

@dataclass
class Reversal:
    """One contiguous run where the potential moved the wrong way."""
    i0: int
    i1: int                    # inclusive index of the last reversed step
    charge: float              # mAh/g passed during it
    v_start: float
    v_end: float
    span_mV: float             # widest excursion, |v_end - v_start|
    seconds: float
    at_full_current: bool      # |I| still at the step median -> not a CV taper
    n_records: int


@dataclass
class HalfCycleSignal:
    """Everything one half-cycle contributes, measured and processed."""
    cycle: int
    step: str
    direction: float                       # +1 rising, -1 falling. MEASURED.
    voltage: np.ndarray = field(default_factory=lambda: np.empty(0))
    dqdv_raw: np.ndarray = field(default_factory=lambda: np.empty(0))
    dqdv: np.ndarray = field(default_factory=lambda: np.empty(0))
    capacity: float = float("nan")         # from the cycler's counter
    n_records_raw: int = 0
    cv_points_removed: int = 0
    cv_method: str = "off"
    n_spikes: int = 0
    # Records whose dQ/dV sign was corrected because the plateau sent the
    # recorded voltage backwards while the cell kept charging. See
    # `orient_dqdv`; a large number is a flat plateau, not a fault.
    n_oriented: int = 0
    # Which path produced this curve, and — on the histogram path — the bin
    # width actually used (it is widened where the records are sparser than
    # the requested bin) and how many bins came back empty. Both are how you
    # tell a well-chosen binning from one that is too fine.
    dqdv_method: str = ""
    bin_width_mV: float = float("nan")
    n_empty_bins: int = 0
    # HOW MUCH CHARGE THE ANALYSIS WINDOW LEFT OUT, in mAh/g.
    # `preprocess_half_cycle` has computed this since the histogram rewrite —
    # `charge_total - charge_in_window` — and dropped it on the floor: no
    # field held it, nothing read the key. It is the first thing to look at
    # when integral fidelity is below one, because it separates "the curve
    # cannot represent this charge" (a flat plateau, no dV to divide by) from
    # "this charge is outside the voltage range we chose to analyse", and only
    # the second is something the operator can do anything about.
    charge_outside_window: float = float("nan")
    # Raw records per bin, aligned with `voltage`. Empty on the derivative
    # path, where there are no bins. See the occupancy note in
    # `histogram_dqdv`: this is what says whether a wiggle is a measurement.
    occupancy: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Rebuilds this half-cycle's curve with the bin EDGES moved by a fraction
    # of a bin: same records, same width, same kernel. `None` off the
    # histogram path, where there are no edges to move. See
    # `detect.PHASE_INVARIANCE` for what detection does with it.
    rebin_at_phase: object = None
    smoothed: bool = False
    reversals: list = field(default_factory=list)
    note: str = ""

    @property
    def reversed_charge_at_current(self) -> float:
        """mAh/g passed backwards while the current was still at full CC."""
        return float(sum(r.charge for r in self.reversals
                         if r.at_full_current))

    @property
    def widest_reversal_mV(self) -> float:
        return float(max((r.span_mV for r in self.reversals), default=0.0))


def reversals(df_step, direction, *, capacity_col=None, min_records=1):
    """
    Inventory every contiguous run where the voltage moves against `direction`.

    Works in ACQUISITION order. `direction` must come from
    `Dataset.step_direction` — measured — never from the step label.
    """
    d = df_step.sort_index()
    v = pd.to_numeric(d["Voltage"], errors="coerce").values.astype(float)
    n = v.size
    if n < 3:
        return []

    q = None
    if capacity_col and capacity_col in d.columns:
        q = pd.to_numeric(d[capacity_col], errors="coerce").values.astype(float)

    cur_col = next((c for c in ("Current(A)", "Current", "current(A)")
                    if c in d.columns), None)
    i_abs = (pd.to_numeric(d[cur_col], errors="coerce").abs().values
             if cur_col else None)
    med_i = (float(np.nanmedian(i_abs[np.isfinite(i_abs)]))
             if i_abs is not None and np.isfinite(i_abs).any() else np.nan)

    secs = None
    if "Time" in d.columns:
        t = pd.to_timedelta(d["Time"].astype(str), errors="coerce")
        if t.notna().any():
            secs = t.dt.total_seconds().values

    back = (np.diff(v) * direction) < 0
    out, k = [], 0
    while k < back.size:
        if not back[k]:
            k += 1
            continue
        j = k
        while j + 1 < back.size and back[j + 1]:
            j += 1
        idx = slice(k, j + 1)
        charge = (float(np.nansum(np.abs(np.diff(q[k:j + 2]))))
                  if q is not None else float("nan"))
        dt = (float(secs[j + 1] - secs[k]) if secs is not None
              and np.isfinite(secs[k]) and np.isfinite(secs[j + 1]) else float("nan"))
        full = bool(np.isfinite(med_i) and med_i > 0 and i_abs is not None
                    and np.nanmedian(i_abs[k:j + 2]) > 0.99 * med_i)
        if (j - k + 1) >= min_records:
            out.append(Reversal(
                i0=int(k), i1=int(j), charge=charge,
                v_start=float(v[k]), v_end=float(v[j + 1]),
                span_mV=float(abs(v[j + 1] - v[k]) * 1000.0),
                seconds=dt, at_full_current=full, n_records=int(j - k + 1)))
        k = j + 1
    return out



# --- the cut-off approach, and trimming it -------------------------------
# A galvanostatic half-cycle ENDS ON A VOLTAGE. As the cell approaches that
# cut-off, dV/dQ collapses and dQ/dV therefore climbs — steeply, and for
# reasons that are protocol rather than chemistry. Measured on NNM cell C,
# median curve height near the TERMINATING end relative to that half-cycle's
# own median:
#
#     within  20 mV      charge 5.22x     discharge 2.25x
#     within  50 mV      charge 3.78x     discharge 2.47x
#     within 100 mV      charge 1.95x     discharge 2.16x
#     within 300 mV      charge 1.37x     discharge 1.51x
#
# Left in, that rise is fitted as a peak, and the cubic baseline strains to
# follow what the peaks cannot — which is where the non-convergence came from
# (40% of fits reaching the 20,000-evaluation ceiling). Trimming it is the
# same argument the pipeline already makes about a constant-voltage hold:
# charge passed at the voltage limit is real, and is not a redox feature.
# OFF BY DEFAULT, and the reason is a negative result worth keeping.
#
# The mechanism is real: a half-cycle ends on a VOLTAGE, and as the cell
# reaches it dV/dQ collapses so dQ/dV climbs. Removing that rise measurably
# helps — on 30 cycles of NNM cell C an extra 50-100 mV took the quadratic
# baseline from 66% of fits converging to 87-89%, and dropped the median peak
# count from 5 to 4 by removing a component that was fitting the cut-off.
#
# What does NOT work is measuring the trim automatically. Across all 100
# cycles the rule's answer is BIMODAL: half the half-cycles want nothing and
# most of the rest saturate at the 200 mV cap, with nothing in between, and no
# pattern in cycle number —
#
#     median trim by cycle band   charge   discharge
#         1-20                     180        160
#        21-40                       0        150
#        41-60                       0          0
#        61-80                     190          0
#        81-100                      0          0
#
#     50% of charge and 51% of discharge half-cycles trim zero;
#     the 75th percentile is 200 mV, the cap.
#
# That is not a boundary being measured. It is a threshold flipping on and off
# depending on where the interior median happens to fall, and applying its
# median across a dataset would trim a physically arbitrary amount. A 30-cycle
# sample looked clean and was not: see the audit note.
#
# So the machinery stays, the default does not. Set TERMINAL_TRIM = True and
# TERMINAL_TRIM_FIXED_MV to a value you have LOOKED AT for your protocol —
# §A11 has the evidence and the procedure. Automatic detection of the cut-off
# approach is unfinished work, not a solved problem.
TERMINAL_TRIM = False
# When TERMINAL_TRIM is on, trim this fixed span off each half-cycle's
# TERMINATING end (high for charge, low for discharge). None = use the
# measured rule below, which the note above explains is not reliable.
TERMINAL_TRIM_FIXED_MV = 75.0
# How far above the INTERIOR median a 20 mV band must stand to still be the
# cut-off approach, and the most of the window that may be removed at an end.
#
# CALIBRATED ON THE CURVES THE PIPELINE ACTUALLY BUILDS — which already have
# `voltage_window`'s 50 mV off each end, so the sharpest part of the rise is
# gone before this ever sees it. Measured on those curves the elevation is
# gentle and broad (1.5-2.1x the interior, still 1.5x at 200 mV), not the
# 5x spike the raw record shows. Judging a 20 mV BAND rather than the single
# end point is what makes it measurable at all.
#
# Median trim found, NNM cell C against LTO cell A:
#
#   factor   NNM charge   NNM discharge   LTO charge   LTO discharge
#    1.1      (0, 200)      (200, 0)        (0, 0)        (0, 0)
#    1.2      (0, 200)      (200, 0)        (0, 0)        (0, 0)
#    1.3      (0, 180)      (200, 0)        (0, 0)        (0, 0)
#    1.4      (0, 120)      (150, 0)        (0, 0)        (0, 0)
#    1.5      (0, 100)      (  0, 0)        (0, 0)        (0, 0)
#
# Two things decide 1.4. Below it the rule SATURATES at the 10% cap (200 mV
# of a 2.15 V window) — that is "trim everything allowed", not a measurement.
# Above it the discharge side drops to nothing. 1.4 is the only value that
# returns a bounded answer on both steps.
#
# And at EVERY factor in that range LTO trims nothing, on both steps. The rule
# does not fire on a sharp two-phase profile at all, so this cannot reach the
# calibration dataset — that is a property of the rule, not of a tuned number.
TERMINAL_TRIM_FACTOR = 1.4
TERMINAL_TRIM_MAX_FRACTION = 0.10
TERMINAL_TRIM_BAND_MV = 20.0
# The middle of the curve, used as "the interior" the ends are judged against.
TERMINAL_TRIM_INTERIOR = (0.2, 0.8)


def terminal_trim(voltage, dqdv, *, factor=None, max_fraction=None,
                  band_mV=None):
    """How far in from each end the curve is still the cut-off approach.

    Returns `(low_V, high_V)` — the span to remove at the bottom and top of
    this half-cycle's voltage axis. Steps inward in `band_mV` bands while the
    band's MEDIAN stands above `factor` x the interior median, and stops at
    the first band that does not.

    The rule TARGETS ITSELF, which is the reason to trust it: a half-cycle has
    only one cut-off, so only the terminating end trims. On the NNM triplicate
    it returns (0, 120 mV) on CHARGE and (150 mV, 0) on DISCHARGE — the high
    end for charge, the low end for discharge — and (0, 0) on every LTO
    half-cycle, which has no such rise. Neither was put in by hand.
    """
    factor = TERMINAL_TRIM_FACTOR if factor is None else factor
    max_fraction = (TERMINAL_TRIM_MAX_FRACTION if max_fraction is None
                    else max_fraction)
    band = (TERMINAL_TRIM_BAND_MV if band_mV is None else band_mV) / 1000.0
    v = np.asarray(voltage, float)
    y = np.abs(np.asarray(dqdv, float))
    n = v.size
    if n < 40 or y.size != n or band <= 0:
        return 0.0, 0.0
    a, b = (int(TERMINAL_TRIM_INTERIOR[0] * n),
            int(TERMINAL_TRIM_INTERIOR[1] * n))
    interior = float(np.median(y[a:b])) if b > a else float("nan")
    if not np.isfinite(interior) or interior <= 0:
        return 0.0, 0.0
    span = float(v[-1] - v[0])
    if not np.isfinite(span) or span <= 0:
        return 0.0, 0.0
    cap = max_fraction * span
    thr = factor * interior

    def _walk(from_low):
        d = (v - v[0]) if from_low else (v[-1] - v)
        step, out = band, 0.0
        while step <= cap:
            m = (d >= step - band) & (d < step)
            if m.sum() < 2:
                break
            if float(np.median(y[m])) <= thr:
                break
            out = step
            step += band
        return float(out)

    return _walk(True), _walk(False)


def apply_terminal_trim(sig, low_V, high_V):
    """A copy of `sig` with the cut-off approach removed from its curve.

    `capacity` is NOT changed. It is the cycler's count of what the cell
    delivered, and the whole point of trimming is that some of that charge is
    no longer represented in the curve — which is exactly what
    `quality.integral_fidelity` then reports. Silently reducing the capacity
    to match would hide the thing this is meant to surface.
    """
    import dataclasses
    v = np.asarray(sig.voltage, float)
    if v.size == 0 or (low_V <= 0 and high_V <= 0):
        return sig
    keep = (v >= v.min() + low_V) & (v <= v.max() - high_V)
    if keep.sum() < 10:                    # nothing usable would be left
        return sig
    out = dataclasses.replace(
        sig, voltage=v[keep],
        dqdv=np.asarray(sig.dqdv, float)[keep],
        dqdv_raw=(np.asarray(sig.dqdv_raw, float)[keep]
                  if np.size(sig.dqdv_raw) == v.size else sig.dqdv_raw))
    return out

def voltage_window(dataset, voltage_trim_mV=50,
                   required=("Voltage", "Cycle", "Step", "dQ/dV")):
    """
    The trimmed voltage window, computed ONCE PER DATASET.

    This is not a detail. 1.8.7 takes the min and max over the whole cleaned
    frame and trims 50 mV from each end, then clips every half-cycle to that
    one window. Computing it per half-cycle instead — the obvious-looking
    thing — trims 50 mV off each half-cycle's own extremes and silently
    removes real data from every one of them.

    Verified against the 1.8.7 output: P3 cell A spans 1.9972-4.2517, giving a
    window of 2.0472-4.2017, and the reference processed_dqdv.csv spans
    2.0472-4.2013.
    """
    clean = dataset.clean
    v = pd.to_numeric(clean["Voltage"], errors="coerce")
    lo, hi = float(v.min()), float(v.max())
    trim = voltage_trim_mV / 1000.0
    if lo + trim >= hi - trim:            # 1.8.7 falls back to a 10 mV trim
        trim = 0.010
    return lo + trim, hi - trim


# Which dQ/dV the pipeline analyses. Both paths produce the same thing — a
# voltage axis and a dQ/dV on it — so everything downstream is unchanged.
#
#   "derivative"  the CYCLER's dQ/dV column, sorted, rebinned, despiked and
#                 smoothed. 1.8.x behaviour, and what every earlier Ratatosk
#                 result was computed with.
#   "histogram"   `histogram_dqdv`: charge binned by voltage, no derivative,
#                 no smoothing parameters, no sign correction needed.
#
# The default is chosen on measured voltage reconstruction, not on preference
# — see `quality.voltage_reconstruction` and the appendix.
DQDV_METHOD = "histogram"
DQDV_METHODS = ("derivative", "histogram")


def preprocess_half_cycle(df_step, params, *, direction, window,
                          capacity_col=None, step=None):
    """
    One half-cycle, from raw records to a curve ready for detection.

    Two paths, selected by `params["dqdv_method"]` or `DQDV_METHOD`.

    DERIVATIVE. The order of operations is 1.8.7's exactly — strip the
    constant-voltage hold in acquisition order, THEN sort by voltage, trim,
    rebin, despike, smooth — because changing it would change every fitted
    value.

    HISTOGRAM. Strip the constant-voltage hold, then bin the charge by
    voltage. No sort (a histogram is order-free), no rebin (the bin IS the
    rebin), no despike and no smoothing beyond a fixed three-point kernel,
    and no `orient_dqdv` — a record's charge lands in the bin for the voltage
    it was measured at whichever way the voltage happened to be moving.
    """
    raw_n = len(df_step)
    stripped, n_cv, how = strip_cv_hold(df_step)

    method = str(params.get("dqdv_method") or DQDV_METHOD).lower()
    if method not in DQDV_METHODS:
        raise ValueError(f"dqdv_method must be one of {DQDV_METHODS}, "
                         f"not {method!r}")

    if method == "histogram":
        qcol = capacity_col or ("Discharge_Capacity"
                               if str(step).lower().startswith("dis")
                               else "Charge_Capacity")
        if qcol not in stripped.columns:
            return None, dict(n_cv=n_cv, cv_method=how, raw_n=raw_n,
                              n_oriented=0, dqdv_method=method,
                              note=f"no {qcol} column for the histogram path")
        bw = params.get("histogram_bin_mV") or params.get("rebin_width_mV") \
            or HISTOGRAM_BIN_MV
        v_all = pd.to_numeric(stripped["Voltage"], errors="coerce").to_numpy()
        q_all = pd.to_numeric(stripped[qcol], errors="coerce").to_numpy()
        centres, y, hinfo = histogram_dqdv(
            v_all, q_all, bin_width_mV=float(bw), window=window,
            smooth_kernel=params.get("histogram_smooth_bins"),
            sign=dqdv_sign(step) if step is not None else 1.0)
        if centres.size < 5:
            return None, dict(n_cv=n_cv, cv_method=how, raw_n=raw_n,
                              n_oriented=0, dqdv_method=method,
                              note="too few filled bins for a curve")
        # A closure that rebuilds this curve with the bin EDGES shifted by a
        # fraction of a bin. Everything else is held: the same records, the
        # same width actually used, the same kernel, the same sign. Detection
        # uses it to ask whether a maximum is a property of the cell or of
        # where the edges happened to fall. Offsets are a fraction of the
        # width USED, not the width requested, because the width is widened
        # where the records are sparse and it is the used one that sets the
        # beat.
        _w_used = float(hinfo["bin_width_mV"])
        _kern = params.get("histogram_smooth_bins")
        _sign = dqdv_sign(step) if step is not None else 1.0

        def _rebin_at_phase(frac, _v=v_all, _q=q_all, _w=_w_used,
                            _win=window, _k=_kern, _s=_sign):
            # `phase=` rather than a shifted window: see the note beside the
            # grid anchor in `histogram_dqdv`. Shifting the window was undone
            # by the visited-span clip on exactly the half-cycles this test
            # matters for.
            c2, y2, _i2 = histogram_dqdv(_v, _q, bin_width_mV=_w,
                                         window=_win, smooth_kernel=_k,
                                         sign=_s, phase=float(frac))
            return c2, y2

        return (dict(voltage=centres, dqdv_raw=y, dqdv=y,
                     occupancy=hinfo.get("occupancy"),
                     rebin_at_phase=_rebin_at_phase),
                dict(n_cv=n_cv, cv_method=how, raw_n=raw_n, n_spikes=0,
                     n_oriented=0, smoothed=bool(HISTOGRAM_SMOOTH_KERNEL),
                     dqdv_method=method, note="",
                     bin_width_mV=hinfo["bin_width_mV"],
                     n_empty_bins=hinfo["n_empty"],
                     charge_outside_window=float(
                         hinfo["charge_total"] - hinfo["charge_in_window"])
                     if np.isfinite(hinfo["charge_total"]) else np.nan))

    # --- the derivative path -------------------------------------------
    # BEFORE the sort — see `orient_dqdv`. The sort is what turns a wobbling
    # plateau voltage into an inverted peak. The histogram path above needs
    # none of this, which is most of the argument for it.
    stripped, n_oriented = orient_dqdv(stripped, direction,
                                       capacity_col=capacity_col)

    # kind="mergesort" is a STABLE sort, and that is deliberate.
    #
    # 1.8.7 uses pandas' default quicksort, which is not stable, so where
    # several records share a voltage the order among them is decided by the
    # sort's internals. That is not a small population: P3 cell A cycle 1
    # charge has 483 tied voltages out of 1434, because the cycler records to
    # 0.1 mV and the cell sits on a plateau. Despiking and smoothing then run
    # over that arbitrary order, so 1.8.7's smoothed curve is not a
    # well-defined function of its input — reorder the rows and it changes.
    #
    # A stable sort ties-breaks by acquisition order, which is the order the
    # measurements were actually taken in. The result is reproducible under
    # any reordering of the input. This is the one place 1.9.0 knowingly
    # differs from 1.8.7, it is confined to half-cycles containing tied
    # voltages, and it is measured in tests/test_port_fidelity.py.
    d = stripped.sort_values("Voltage", kind="mergesort")
    lo, hi = window
    d = d[(d["Voltage"] >= lo) & (d["Voltage"] <= hi)]
    if len(d) < 5:
        return None, dict(n_cv=n_cv, cv_method=how, raw_n=raw_n,
                          n_oriented=n_oriented, dqdv_method=method,
                          note="fewer than 5 points after trimming")

    voltage = pd.to_numeric(d["Voltage"], errors="coerce").values.copy()
    dqdv_raw = pd.to_numeric(d["dQ/dV"], errors="coerce").values.copy()

    rebin = params.get("rebin_width_mV")
    if rebin:
        voltage, dqdv_raw = rebin_dqdv(voltage, dqdv_raw, bin_width_mV=rebin)

    work = pd.Series(dqdv_raw)
    n_spikes = 0
    if params.get("spike_removal"):
        work, n_spikes = remove_spikes(
            work, params.get("spike_window_size", 5),
            params.get("spike_threshold_multiplier", 3.0))

    smoothed, was = smooth_dqdv(work.values, params.get("smoothing_window"),
                                params.get("polyorder", 3))
    second = params.get("second_smooth_window")
    if second:
        smoothed, _ = smooth_dqdv(smoothed, second, params.get("polyorder", 3))

    return (dict(voltage=voltage, dqdv_raw=dqdv_raw, dqdv=smoothed),
            dict(n_cv=n_cv, cv_method=how, raw_n=raw_n, n_spikes=n_spikes,
                 n_oriented=n_oriented, smoothed=bool(was),
                 dqdv_method=method, note=""))


def half_cycle_report(dataset, cycle, step, params, *, window=None,
                      voltage_trim_mV=50):
    """
    Assemble a `HalfCycleSignal`: the curve, plus what was measured on it.

    `window` should be computed once with `voltage_window(dataset)` and
    reused for every half-cycle — see that function for why.
    """
    if window is None:
        window = voltage_window(dataset, voltage_trim_mV)
    df = dataset.half_cycle(cycle, step)
    direction = dataset.step_direction(cycle, step)
    cap_col = ("Charge_Capacity" if str(step).lower().startswith("c")
               else "Discharge_Capacity")

    sig = HalfCycleSignal(cycle=int(cycle), step=str(step),
                          direction=float(direction),
                          capacity=dataset.capacity(cycle, step),
                          n_records_raw=len(df))
    sig.reversals = reversals(df, direction, capacity_col=cap_col)

    curve, info = preprocess_half_cycle(df, params, direction=direction,
                                        window=window, capacity_col=cap_col,
                                        step=step)
    sig.cv_points_removed = info["n_cv"]
    sig.cv_method = info["cv_method"]
    sig.n_oriented = int(info.get("n_oriented", 0))
    sig.note = info.get("note", "")
    if curve is not None:
        sig.voltage = curve["voltage"]
        sig.dqdv_raw = curve["dqdv_raw"]
        sig.dqdv = curve["dqdv"]
        sig.n_spikes = info["n_spikes"]
        sig.smoothed = info["smoothed"]
        _occ = curve.get("occupancy")
        if _occ is not None and len(_occ) == len(sig.voltage):
            sig.occupancy = np.asarray(_occ, float)
        sig.rebin_at_phase = curve.get("rebin_at_phase")
    sig.dqdv_method = info.get("dqdv_method", "")
    sig.bin_width_mV = float(info.get("bin_width_mV", float("nan")))
    sig.n_empty_bins = int(info.get("n_empty_bins", 0))
    sig.charge_outside_window = float(
        info.get("charge_outside_window", float("nan")))
    return sig
