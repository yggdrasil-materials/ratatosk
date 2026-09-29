"""
Does the answer mean anything?

Two orthogonal measurements, kept separate on purpose
-----------------------------------------------------
    integral fidelity  =  integral(dQ/dV) dV  /  capacity from the counter
                          "does the derivative account for the charge?"

    capacity closure   =  sum(component areas)  /  integral(dQ/dV) dV
                          "do the fitted peaks account for the curve?"

Using the counter capacity as closure's denominator would hide a Module 1
failure inside a Module 4 number. They validate different stages and must be
read separately.

Why closure is reported as an INTERVAL
--------------------------------------
A pseudo-Voigt of sigma ~ 100 mV and a cubic across a 1.4 V window are nearly
the same function. Where that is true, the split between "peak" and
"background" is not determined by the data, and any single closure value is an
artefact of the baseline degree somebody chose.

Measured on real cells, refitting only the baseline degree:

    sharp    0.95  0.95  0.95      (deg 1, 2, 3)   R2 identical to 4 dp
    broad    0.48  0.51  0.57
    broad    0.51  0.21  0.16      R2 0.944 -> 0.983 as the peaks empty out

No threshold is placed on the closure VALUE - a reader who sees 0.16-0.51 does
not need to be told the number is unusable, and every attempt in this project
to threshold a pooled closure value has failed on the next dataset. The one
threshold is on the WIDTH of the interval (PARTITION_DETERMINED_BELOW), which
decides whether a capacity share is reported or withheld; the fitted areas are
written out either way.

The interval also carries physical meaning. A partition that survives a change
of baseline is one a polynomial cannot mimic, which is what a first-order
two-phase feature looks like; a partition that collapses is a solid-solution
envelope. That reading is consistent with operando XRD for the layered and
spinel systems it was measured on — see the project notes.
"""

from __future__ import annotations

import numpy as np

from .compat import trapezoid
from .fitting import FitSpec, fit_half_cycle, fit_many
from .style import bullet, entry, section, verdict

__all__ = [
    "CLOSURE_SAMPLE_DEFAULT",
    "DEFAULT_DEGREES",
    "MECHANISM_AT_BOUND_ESCALATES",
    "MECHANISM_AT_BOUND_FRACTION",
    "MECHANISM_MIXED",
    "MECHANISM_MODEL",
    "MECHANISM_MULTI_TRANSITION",
    "MECHANISM_SHOULDER_FRACTION",
    "MECHANISM_SOLID_SOLUTION",
    "MECHANISM_TWO_PHASE",
    "RECONSTRUCTION_APE_ABOVE",
    "RECONSTRUCTION_RMSE_WINDOW_FRACTION",
    "RESOLVABILITY_QUALIFIED",
    "RESOLVABILITY_RESOLVED",
    "RESOLVABILITY_UNRESOLVED",
    "RESOLVED_CLOSURE_FLOOR",
    "UNATTRIBUTED_QUALIFIED_ABOVE",
    "assess_closure",
    "assess_resolvability",
    "classify_mechanism",
    "closure",
    "closure_interval",
    "degrees_for",
    "describe_fidelity",
    "describe_partition",
    "describe_reconstruction",
    "describe_resolvability",
    "integral_fidelity",
    "reconcile_mechanisms",
    "sample_half_cycles",
    "voltage_reconstruction",
]

DEFAULT_DEGREES = (1, 2, 3)

# The closure interval is a DIFFERENCE between three fits, and differencing
# amplifies where each of them stopped: if every fit stops within e of its true
# minimum, their spread carries up to 2e that is not a property of the data. So
# the closure pass has a reason to want a tighter tolerance than the production
# fit, even though it fits the same curve.
#
# Whether it NEEDS one is an open question and this flag exists to settle it in
# one run. `None` means "use `fitting.FIT_TOLERANCE`" and is the default, so
# nothing changes unless it is set.
#
# What is known:
#   * On the LTO triplicate the interval widened from a median 0.0022-0.0033 to
#     0.0037-0.0058 when FIT_TOLERANCE went from the library default to 1e-4 —
#     real, and one to two orders of magnitude inside the 0.15 threshold.
#   * On the NNM triplicate it widened on cell A (+0.004) and cell C (+0.018,
#     0.083 -> 0.125), and did not move on cell B.
#   * A bench on a 40-half-cycle sample said the opposite, that the interval
#     had NARROWED. It sampled different half-cycles and used a locally
#     detected peak list rather than the propagated reference list, so it was
#     not measuring the same quantity. Closure is a full-run number; do not
#     quote it from a bench.
#
# Set this to 1e-6 and re-run Cell 10 to test whether a tighter closure pass
# buys the interval back, and what it costs in time.
CLOSURE_TOLERANCE = None

# --- the ONE threshold on the closure interval ---------------------------
# `analyse.capacity_attribution` withholds a cycle's capacity share when its
# closure interval is wider than this. It is defined HERE, next to the
# measurement, and imported there — because for a while there were two
# numbers: attribution acted on 0.15 while `describe_partition` printed a
# verdict based on 0.05. A dataset could therefore be told in one line that
# "the partition survives a change of baseline" and have every one of its
# cycles withheld two cells later.
PARTITION_DETERMINED_BELOW = 0.15

# --- WHEN THERE IS NO BASELINE, THIS IS THE QUANTITY ----------------------
# The closure interval asks how much the peak/background split moves when the
# baseline degree changes. From 1.9.0.56 there IS no background: every mAh in
# a dQ/dV passed through the cell, the polynomial has gone, and the split it
# was measuring does not exist. `closure_interval` therefore makes no sweep on
# a spec with `baseline_degree=None` and reports a width of zero, which is a
# true statement -- nothing arbitrary is left for the answer to depend on.
#
# What replaces it is the charge the model cannot NAME. That is a statement
# about the cell rather than about the fit, and it is reported rather than
# absorbed. Measured on the six cells at 1.9.0.56:
#
#     LTO   +0.5%      one component, a two-phase transition
#     NNM   +6.0%      five components, no band permitted by the mechanism
#     NMC   +0.1%      four components, one of them a 0.5 V band
#
# 0.15 is deliberately the same number as PARTITION_DETERMINED_BELOW rather
# than a new one: it is the same judgement in the same units -- above this
# much of the cell's charge going unexplained, the decomposition is not
# describing the cell, whichever way the shortfall arose.
#
# READ NNM AND NMC TOGETHER WITH CARE. A high named fraction is not by itself
# a better decomposition: NMC reaches 99.9% partly because a 0.5 V band can
# absorb a great deal, while NNM's 94.0% is five resolved transitions and an
# honest 6% left over. The report must say this rather than let a reader rank
# them on the number.
UNATTRIBUTED_QUALIFIED_ABOVE = 0.15

# --- A PEAK AT ITS WIDTH BOUND IS A BAND BEING DENIED ---------------------
# `classify_mechanism` reads the shoulder fraction of ONE reference cycle, and
# on a replicate pair that is a thin basis. Measured on the NMC111 pair at
# 1.9.0.56, cell A came back `multi_transition` and cell B `mixed` — the same
# material, the same protocol, two different models:
#
#                              cell A (no bands)   cell B (bands)
#     components at ceiling      49/97  (51%)        17/76  (22%)
#     median FWHM                399.6 mV            197.6 mV
#     bands fitted                0                  16
#     reliable                   41/97               50/76
#     above 4.05 V               20 comps, 17 at      26 comps, 14 at
#                                ceiling, 0 bands     ceiling, 9 bands
#
# Cell A's MEDIAN component sits against a 400 mV ceiling. Denied a component
# shaped like a region, the fitter reaches charge spread over one the only way
# left to it: by stretching a peak until it stops.
#
# Nothing else caught this. Both cells named ~100% of the charge, both had
# unattributed at +0.5%, and neither had anything withheld — a stack of 400 mV
# blobs covers a curve perfectly well. That is the caveat on `closure` arriving
# as a measurement: a high named fraction is not by itself a decomposition.
#
# So the fit gets a say in the mechanism it was fitted with. Where the chosen
# mechanism permits no band and the reference fits nonetheless pin this share
# of their components at the width ceiling, the call is escalated to `mixed`
# and the reason recorded. This is not the fit overriding the classifier on a
# whim: it is one specific, named failure — and the only remedy for it is the
# component the classifier declined to allow.
MECHANISM_AT_BOUND_ESCALATES = True
MECHANISM_AT_BOUND_FRACTION = 0.30

# Closure is component area over curve area. Values a little above 1 are
# ordinary — components overlap, and a baseline that dips negative lifts the
# sum. A value above this is not a wide interval, it is a FAILED FIT: one
# component has run away and taken several times the curve's own area. Those
# are counted and named separately, because a single one of them dominates
# any min/max and makes an otherwise clean dataset look catastrophic.
CLOSURE_RUNAWAY_ABOVE = 1.5

# The other end of the same idea. A closure value only says something about
# the peak/background split if the fit it came from DESCRIBES THE CURVE. On
# LTO cell C, cycle 7 charge, the cubic refit lands on one of two minima: a
# sensible one (closure 0.973, R2 0.904) and a collapsed one where the peak
# amplitude goes to zero, the cubic takes the whole curve, and R2 comes back
# NEGATIVE (-0.023 — worse than a horizontal line). `success` is True for
# both, so both were admitted, and the interval for that half-cycle was
# 0.0155 or 0.9573 depending on which minimum that run happened to find. The
# same build gave both answers across repeat runs, and the one number decided
# whether that cycle's capacity share was published or withheld.
#
# A degree whose R2 is at or below zero has not fitted anything, so it is not
# evidence about the split and is dropped from the interval — the low-side
# counterpart of CLOSURE_RUNAWAY_ABOVE. Zero is a deliberately unambiguous
# bar: it means "worse than predicting the mean", not "fits less well than
# the other degrees". A cubic that merely fits worse is still admitted, and
# still widens the interval, which is exactly what the interval is for.
CLOSURE_DEGREE_MIN_R2 = 0.0

# --- when a reconstruction stops vouching for the curve -------------------
# The cumulative integral of dQ/dV against V should give back the (Q, V)
# curve it came from. Two numbers say how well, and both now have a stated
# threshold instead of a literal buried in a notebook cell.
#
# APE is capacity: the curve's integral against the counter's own figure. The
# histogram path reaches 0.00% by construction, and the derivative path on the
# LTO plateau reached 38%. Above 5% a peak area is a share of whatever
# fraction survived, not a share of the cell.
RECONSTRUCTION_APE_ABOVE = 5.0
# RMSE is the voltage axis, and it has to be read against the window rather
# than in absolute terms: 25 mV is 1.8% of LTO's 1.38 V window and 0.6% of a
# 4 V one. Quoted as a fraction of the window for that reason. Above this the
# curve is placing charge at voltages the cell did not visit, which moves
# peak CENTRES — the one quantity that survives where areas do not.
RECONSTRUCTION_RMSE_WINDOW_FRACTION = 0.05


# ---------------------------------------------------------------------------
# Stage 1: is the derivative itself accounting for the charge?
# ---------------------------------------------------------------------------


def integral_fidelity(voltage, dqdv, measured_capacity):
    """
    integral(|dQ/dV|) dV over the half-cycle, divided by the capacity the
    cycler counted.

    A closed physical constraint, not a heuristic: fabricated area shows up as
    an excess, lost features as a deficit. On healthy half-cycles this sits at
    0.998-1.002. The 4.199 V constant-voltage artefact would have failed it on
    sight.

    Returns nan rather than raising when the capacity is unknown, because a
    missing counter column is a reason to say nothing, not to stop.
    """
    v = np.asarray(voltage, float)
    y = np.abs(np.asarray(dqdv, float))
    # The docstring promises nan rather than an exception when the capacity
    # is unknown, and `None` is exactly how "unknown" arrives — np.isfinite
    # raises on it, so it has to be coerced first.
    try:
        cap = float(measured_capacity)
    except TypeError, ValueError:
        return float("nan")
    if v.size < 2 or not np.isfinite(cap) or cap <= 0:
        return float("nan")
    measured_capacity = cap
    return float(trapezoid(y, v) / measured_capacity)


# ---------------------------------------------------------------------------
# Stage 2: do the peaks account for the curve?
# ---------------------------------------------------------------------------


def voltage_reconstruction(voltage, dqdv, q_measured, v_measured):
    """
    The objective test of a dQ/dV curve: can it give the voltage curve back?

    A derivative is reversible, so the cumulative integral of dQ/dV plotted
    against V should reproduce the (Q, V) curve it came from. Many
    combinations of smoothing and binning produce a dQ/dV that LOOKS right;
    far fewer reconstruct the voltage. That is the criterion Flores and Clark
    (J. Electrochem. Soc. 173, 120520, 2026) propose for choosing between IC
    computation methods, and it is the one used here to choose Ratatosk's
    default.

    Returns `{"ape_pct", "rmse_V", "q_reconstructed", "n"}`:

      ape_pct   absolute percentage error in the maximum reconstructed
                capacity. This is `integral_fidelity` expressed as an error:
                the capacity half of the test.
      rmse_V    root-mean-square difference between the original voltage
                curve and the reconstructed one, over the capacity interval
                they share. This is the half `integral_fidelity` cannot see —
                a curve can recover the right TOTAL charge and still put it
                at the wrong voltages.

    A perfect reconstruction is ape_pct = 0 and rmse_V = 0.
    """
    v = np.asarray(voltage, float)
    y = np.abs(np.asarray(dqdv, float))
    qm = np.asarray(q_measured, float)
    vm = np.asarray(v_measured, float)
    out = dict(
        ape_pct=float("nan"),
        rmse_V=float("nan"),
        q_reconstructed=float("nan"),
        q_measured=float("nan"),
        n=0,
    )
    ok = np.isfinite(v) & np.isfinite(y)
    v, y = v[ok], y[ok]
    okm = np.isfinite(qm) & np.isfinite(vm)
    qm, vm = qm[okm], vm[okm]
    if v.size < 3 or qm.size < 3:
        return out

    order = np.argsort(v, kind="mergesort")
    v, y = v[order], y[order]

    # COMPARE OVER THE RANGE THE CURVE DESCRIBES, not the whole half-cycle.
    # Ratatosk trims to a common voltage window, so a curve legitimately says
    # nothing about charge passed outside it. On LTO the half-cycle runs to
    # 1.200 V and the window starts at 1.250 V, putting 0.5% of the charge
    # outside — and because that charge sits on the STEEP part of the voltage
    # curve, comparing against the untrimmed measurement reported an RMSE of
    # 390 mV for a curve that reconstructs to 25 mV over its own range. The
    # test would have been measuring the trim, and would have condemned the
    # better method for it.
    keep = (vm >= v.min() - 1e-9) & (vm <= v.max() + 1e-9)
    if int(np.sum(keep)) >= 3:
        qm, vm = qm[keep], vm[keep]
    # Cumulative trapezoid: the capacity accumulated up to each voltage.
    q_rec = np.concatenate([[0.0], np.cumsum(np.diff(v) * 0.5 * (y[1:] + y[:-1]))])
    q_max = float(q_rec[-1])
    out["q_reconstructed"] = q_max

    q_span = float(np.nanmax(qm) - np.nanmin(qm))
    out["q_measured"] = q_span
    if q_span > 0:
        out["ape_pct"] = abs(q_max - q_span) / q_span * 100.0

    # Compare the two curves on the capacity interval they share, with the
    # measured curve interpolated onto the reconstructed capacity grid — the
    # two are defined on different capacity values, so a point-wise
    # difference is only meaningful after that.
    #
    # DIRECTION MATTERS. The reconstruction integrates from low voltage
    # upward, so on a RISING half-cycle its capacity axis runs the same way
    # as the measured one. On a FALLING half-cycle — every discharge, and
    # every charge on an anode with swapped labels — capacity accumulates as
    # voltage drops, so the two axes run opposite ways and must be flipped
    # before they are compared. Without this the test reports an RMSE of
    # 0.8 V on a curve that reconstructs the capacity to 0.5%, which is how
    # a good method gets mistaken for a broken one.
    qm0 = qm - np.nanmin(qm)
    om = np.argsort(qm0, kind="mergesort")
    qm0, vm_s = qm0[om], vm[om]
    if qm0.size > 2 and vm_s[-1] < vm_s[0]:
        qm0, vm_s = (qm0.max() - qm0)[::-1], vm_s[::-1]
    lo, hi = max(qm0.min(), q_rec.min()), min(qm0.max(), q_rec.max())
    inside = (q_rec >= lo) & (q_rec <= hi)
    if int(np.sum(inside)) >= 3 and hi > lo:
        vk = np.interp(q_rec[inside], qm0, vm_s)
        out["rmse_V"] = float(np.sqrt(np.mean((vk - v[inside]) ** 2)))
        out["n"] = int(np.sum(inside))
    return out


def closure(fit_result) -> float:
    """Component area over curve area, from a result `fitting` already made."""
    curve = fit_result.get("curve_area", float("nan"))
    if not np.isfinite(curve) or curve <= 0:
        return float("nan")
    return float(fit_result.get("component_area_sum", np.nan) / curve)


def degrees_for(spec, degrees=DEFAULT_DEGREES):
    """
    Which baseline degrees to refit at. `(None,)` when the model has no
    baseline, because there is then nothing arbitrary to sweep.
    """
    if getattr(spec, "baseline_degree", None) is None:
        return (None,)
    return tuple(degrees)


def closure_interval(
    voltage, dqdv, spec: FitSpec, *, degrees=DEFAULT_DEGREES, key=None
):
    """
    Refit the same data at several baseline degrees; report the spread.

    Returns a dict with `lo`, `hi`, `width`, `by_degree` and `r_squared_by
    _degree`. No verdict — see the module docstring.
    """
    # No background, nothing to sweep. See UNATTRIBUTED_QUALIFIED_ABOVE.
    degrees = degrees_for(spec, degrees)
    vals, r2s, unatt = {}, {}, {}
    for deg in degrees:
        res = fit_half_cycle(
            voltage,
            dqdv,
            spec.replace(baseline_degree=deg, tolerance=CLOSURE_TOLERANCE),
            key=key,
        )
        vals[deg] = closure(res) if res["success"] else float("nan")
        r2s[deg] = res["r_squared"] if res["success"] else float("nan")
        unatt[deg] = (
            float(res.get("unattributed_fraction", np.nan))
            if res["success"]
            else float("nan")
        )

    finite = [x for x in vals.values() if np.isfinite(x)]
    lo = float(min(finite)) if finite else float("nan")
    hi = float(max(finite)) if finite else float("nan")
    _u = [x for x in unatt.values() if np.isfinite(x)]
    return dict(
        key=key,
        lo=lo,
        hi=hi,
        width=(hi - lo) if finite else float("nan"),
        by_degree=vals,
        r_squared_by_degree=r2s,
        unattributed=(float(np.median(_u)) if _u else float("nan")),
        swept=len(degrees) > 1,
        n_degrees=len(finite),
    )


# How many half-cycles the closure interval is measured on by default.
#
# 1.9.0.9 sampled 10. 1.9.0.11 changed it to ALL of them on the argument that
# the parallel path could afford it — and on a 200-cycle P3 cell that is 600
# fits, THREE TIMES the cost of the production fit it is checking. Cell 10
# took longer than Cell 9 and looked like a hang.
#
# 40 is the compromise, and it is a statistical one rather than a budget: the
# quantity reported is a median and the fraction inside a threshold, and 40
# evenly spaced half-cycles fix both to well inside the width being measured.
# Four times the old sample, a fifth of the cost of measuring everything.
# Set CLOSURE_SAMPLE = None in the notebook to measure every half-cycle.
CLOSURE_SAMPLE_DEFAULT = 40


def sample_half_cycles(keys, n=CLOSURE_SAMPLE_DEFAULT):
    """
    Choose a spread of half-cycles for the interval measurement.

    Stability is a property of the data's shape and is consistent within a
    dataset and step — measured spreads of 0.35/0.40 (NMC111), 0.05/0.09 (P3),
    0.00/0.01 (LTO) on pairs of cycles far apart. So it does not need
    computing on all four hundred half-cycles; an evenly spaced sample of ten
    costs three fits each and answers the same question.
    """
    keys = list(keys)
    if len(keys) <= n:
        return keys
    idx = np.linspace(0, len(keys) - 1, n).round().astype(int)
    return [keys[i] for i in sorted(set(idx.tolist()))]


def assess_closure(jobs, *, degrees=DEFAULT_DEGREES, n_jobs=None, verbose=True):
    """
    Run `closure_interval` over a sample, in parallel where that pays.

    `jobs` is a list of (voltage, dqdv, spec, key). Every (job, degree) pair is
    an independent fit, so they are flattened into one batch and handed to
    `fit_many` — which decides for itself whether a worker pool is worth its
    start-up cost.
    """
    if not jobs:
        return []

    # The batch key carries the job's POSITION, not just its caller-supplied
    # key: two jobs sharing a key (the default is None) collapsed into one
    # group and every one of them got the first job's answer.
    # THE TOLERANCE TRAVELS ON THE SPEC, not in a module global: these fits
    # run in worker processes, and a value set in the parent never reaches
    # them. `CLOSURE_TOLERANCE = None` leaves each fit on `FIT_TOLERANCE`.
    flat = [
        (
            v,
            y,
            spec.replace(baseline_degree=d, tolerance=CLOSURE_TOLERANCE),
            ((i, key), d),
        )
        for i, (v, y, spec, key) in enumerate(jobs)
        for d in degrees_for(spec, degrees)
    ]
    # No count printed here: the caller has already said how many half-cycles
    # it sampled and what that costs, and saying it twice in different words
    # is how a console becomes a wall of text.
    results = fit_many(flat, n_jobs=n_jobs, verbose=verbose, label="closure")

    grouped = {}
    for res in results:
        (i, key), deg = res["key"]
        grouped.setdefault(i, {})[deg] = res

    out = []
    for i, (_, _, _, key) in enumerate(jobs):
        per = grouped.get(i, {})
        r2s = {d: r["r_squared"] for d, r in per.items()}

        def _usable(d, r):
            if not r["success"]:
                return False
            _r2 = r.get("r_squared", float("nan"))
            return bool(np.isfinite(_r2) and _r2 > CLOSURE_DEGREE_MIN_R2)

        vals = {
            d: (closure(r) if _usable(d, r) else float("nan")) for d, r in per.items()
        }
        n_collapsed = sum(
            1 for d, r in per.items() if r["success"] and not _usable(d, r)
        )
        finite = [x for x in vals.values() if np.isfinite(x)]
        lo = float(min(finite)) if finite else float("nan")
        hi = float(max(finite)) if finite else float("nan")
        _u = [
            float(r.get("unattributed_fraction", np.nan))
            for d, r in per.items()
            if _usable(d, r)
        ]
        _u = [x for x in _u if np.isfinite(x)]
        _nb = [
            (int(r.get("n_at_sigma_max", 0)), len(r.get("components", [])))
            for d, r in per.items()
            if _usable(d, r)
        ]
        _n_at = sum(a for a, _ in _nb)
        _n_tot = sum(b for _, b in _nb)
        out.append(
            dict(
                key=key,
                lo=lo,
                hi=hi,
                width=(hi - lo) if finite else float("nan"),
                by_degree=vals,
                r_squared_by_degree=r2s,
                n_at_sigma_max=_n_at,
                n_components=_n_tot,
                unattributed=(float(np.median(_u)) if _u else float("nan")),
                swept=len(per) > 1,
                n_degrees=len(finite),
                n_degrees_collapsed=n_collapsed,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Stage 0: is this curve worth decomposing at all?
# ---------------------------------------------------------------------------
# Asked BEFORE the fitting, on the reference half-cycle only, because the
# answer decides whether the next hour is worth spending.
#
# The pipeline already answers it — as `describe_partition`, in Cell 10, after
# every half-cycle has been fitted three times over. On NMC111 that verdict is
# "NOT DETERMINED — do not quote attribution", the capacity attribution is
# withheld on 65 of 69 cycles, and the coherence audit finds zero peaks that
# move like a redox feature. All true, all correct, and all arrived at after
# about an hour of fitting a model the data cannot support.
#
# It can be known in six fits. A material with no resolved peaks — a
# solid-solution layered oxide, where the dQ/dV is a smooth envelope with
# bends rather than maxima — does not become decomposable because the
# optimiser tries harder.
#
# TWO NUMBERS, one free and one cheap:
#
#   shoulder fraction   what proportion of the detected components are
#                       INFLECTIONS rather than maxima. Free: detection has
#                       already worked it out. Measured on the reference
#                       cycle of three chemistries it tracks the verdict
#                       exactly — LTO 0%, NNM 11-22%, NMC111 34-68%.
#
#   closure interval    on the reference half-cycle only, at the same three
#                       baseline degrees Cell 10 uses. Six fits, not six
#                       hundred. LTO 0.00, NNM 0.05-0.08, NMC111 0.51-0.57.
#
# The verdict is taken from the closure interval, against the threshold that
# already governs whether a capacity share is published
# (`PARTITION_DETERMINED_BELOW`), so this cell and Cell 10 cannot disagree.
# The shoulder fraction is reported beside it as corroboration and as the
# thing a reader can see for themselves on the detected-peaks figure.
#
# `RESOLVED_CLOSURE_FLOOR` is the one new number. A split can be perfectly
# determined and still account for a minority of the curve: NNM's peaks are
# real features sitting on a large genuine background (closure ~0.45, interval
# 0.06), which is a different situation from LTO (0.96) and needs saying
# differently in a paper. 0.75 separates the two cases as measured; it is a
# reporting boundary, not a gate, and nothing is withheld on it.
RESOLVED_CLOSURE_FLOOR = 0.75

RESOLVABILITY_RESOLVED = "resolved"
RESOLVABILITY_QUALIFIED = "qualified"
RESOLVABILITY_UNRESOLVED = "unresolved"


def assess_resolvability(
    jobs,
    *,
    shoulder_fraction=None,
    n_primaries=None,
    n_components=None,
    degrees=DEFAULT_DEGREES,
    verbose=True,
):
    """
    Should this dataset be decomposed into peaks at all?

    `jobs` is the reference half-cycle(s) only, in `assess_closure`'s shape:
    a list of `(voltage, dqdv, spec, key)`. One per step is the intended use.

    Returns a dict with `verdict`, the measured `width`, `closure` and
    `shoulder_fraction`, and `reason` — a sentence fit to print.
    """
    if not jobs:
        return dict(
            verdict=RESOLVABILITY_UNRESOLVED,
            width=np.nan,
            closure=np.nan,
            shoulder_fraction=shoulder_fraction,
            unattributed=np.nan,
            swept=False,
            at_bound_fraction=np.nan,
            mechanism_escalated_from=None,
            reason="no reference half-cycle to assess",
            mechanism=MECHANISM_SOLID_SOLUTION,
            mechanism_reason="",
            model=dict(MECHANISM_MODEL[MECHANISM_SOLID_SOLUTION]),
        )

    res = assess_closure(jobs, degrees=degrees, verbose=False)
    widths = np.array([r["width"] for r in res if np.isfinite(r["width"])], float)
    mids = np.array(
        [
            0.5 * (r["lo"] + r["hi"])
            for r in res
            if np.isfinite(r["lo"]) and np.isfinite(r["hi"])
        ],
        float,
    )
    width = float(np.max(widths)) if widths.size else np.nan
    closure = float(np.median(mids)) if mids.size else np.nan
    _un = np.array([r.get("unattributed", np.nan) for r in res], float)
    _un = _un[np.isfinite(_un)]
    unattributed = float(np.median(np.abs(_un))) if _un.size else np.nan
    swept = any(bool(r.get("swept")) for r in res)

    if not np.isfinite(width) and not np.isfinite(unattributed):
        verdict = RESOLVABILITY_UNRESOLVED
        reason = (
            "the reference half-cycle could not be fitted, so there is "
            "nothing to decompose"
        )
    elif not swept and np.isfinite(unattributed):
        # NO BASELINE: the split is not undetermined, because there is no free
        # background for it to be undetermined against. The question that
        # remains is how much of the cell's charge the model can name.
        # See UNATTRIBUTED_QUALIFIED_ABOVE.
        if unattributed > UNATTRIBUTED_QUALIFIED_ABOVE:
            verdict = RESOLVABILITY_QUALIFIED
            reason = (
                f"the model names {1 - unattributed:.0%} of the charge "
                f"in the reference half-cycle and leaves "
                f"{unattributed:.0%} unattributed. Every mAh in a "
                f"dQ/dV passed through the cell, so that shortfall is a "
                f"statement about the decomposition, not a residual: "
                f"areas are shares of what was named"
            )
        else:
            verdict = RESOLVABILITY_RESOLVED
            reason = (
                f"with no free background to trade against, the model "
                f"names {1 - unattributed:.0%} of the charge in the "
                f"reference half-cycle "
                f"({unattributed:+.0%} unattributed)"
            )
    elif width > PARTITION_DETERMINED_BELOW:
        verdict = RESOLVABILITY_UNRESOLVED
        reason = (
            f"on the reference cycle — the best-behaved half-cycle in "
            f"the dataset — the peak/background split moves by {width:.2f} "
            f"when only the baseline degree changes, against the "
            f"{PARTITION_DETERMINED_BELOW:.2f} at which it stops being "
            f"decided by the data. Fitting every cycle will not make "
            f"that smaller"
        )
    elif np.isfinite(closure) and closure < RESOLVED_CLOSURE_FLOOR:
        verdict = RESOLVABILITY_QUALIFIED
        reason = (
            f"the split is determined (interval {width:.2f}) but the "
            f"peaks account for {closure:.0%} of the curve: real "
            f"features on a large genuine background. Areas are "
            f"shares of the peaks, not of the cell"
        )
    else:
        verdict = RESOLVABILITY_RESOLVED
        reason = (
            f"the split holds across baseline degree (interval "
            f"{width:.2f}) and the peaks account for {closure:.0%} of "
            f"the curve"
        )

    _mech, _mech_why = classify_mechanism(
        shoulder_fraction, width, n_primaries=n_primaries, n_components=n_components
    )

    # --- the fit gets a say. See MECHANISM_AT_BOUND_ESCALATES -------------
    _n_at = sum(int(r.get("n_at_sigma_max", 0)) for r in res)
    _n_tot = sum(int(r.get("n_components", 0)) for r in res)
    _at_frac = (_n_at / _n_tot) if _n_tot else float("nan")
    _escalated_from = None
    if (
        MECHANISM_AT_BOUND_ESCALATES
        and np.isfinite(_at_frac)
        and _at_frac >= MECHANISM_AT_BOUND_FRACTION
        and not MECHANISM_MODEL[_mech].get("band_ceiling_span_fraction", 0.0)
    ):
        _escalated_from = _mech
        _mech = MECHANISM_MIXED
        _mech_why = (
            f"{_at_frac:.0%} of the reference cycle's components "
            f"({_n_at} of {_n_tot}) ended at the width ceiling under "
            f"{_escalated_from.replace('_', ' ')}, which permits no band. A "
            f"peak pinned at its width bound is not a width the data chose — "
            f"it is the only way a sum of peaks can reach charge spread over "
            f"a region. Escalated to MIXED so the model may contain the "
            f"component that shape needs. Original call: {_mech_why}"
        )
    out = dict(
        verdict=verdict,
        width=width,
        closure=closure,
        unattributed=unattributed,
        swept=swept,
        at_bound_fraction=_at_frac,
        mechanism_escalated_from=_escalated_from,
        shoulder_fraction=shoulder_fraction,
        reason=reason,
        mechanism=_mech,
        mechanism_reason=_mech_why,
        model=dict(MECHANISM_MODEL[_mech]),
    )
    if verbose:
        print(describe_resolvability(out))
    return out


def describe_resolvability(a, indent="  "):
    """The resolvability verdict, laid out to be read."""
    L = [indent + section("Are there peaks here to decompose?")]
    if a.get("shoulder_fraction") is not None and np.isfinite(a["shoulder_fraction"]):
        L.append(
            entry(
                "shoulders",
                f"{a['shoulder_fraction']:.0%}",
                "of the detected components are inflections on a flank, not maxima",
            )
        )
    if a.get("swept") and np.isfinite(a.get("width", np.nan)):
        L.append(
            entry(
                "closure interval",
                f"{a['width']:.2f}",
                f"reference half-cycle, baseline degrees "
                f"{DEFAULT_DEGREES[0]}-{DEFAULT_DEGREES[-1]}",
            )
        )
    if np.isfinite(a.get("closure", np.nan)):
        L.append(
            entry(
                "charge named",
                f"{a['closure']:.2f}",
                "of the curve accounted for by named components",
            )
        )
    if np.isfinite(a.get("unattributed", np.nan)):
        L.append(
            entry(
                "unattributed",
                f"{a['unattributed']:+.1%}",
                "of the cell's charge the model cannot name — reported, never absorbed",
            )
        )
    if not a.get("swept"):
        L.append(
            bullet(
                "There is no free background in this model, so there "
                "is no peak/background split to be undetermined and "
                "no baseline degree to sweep. A HIGH NAMED FRACTION "
                "IS NOT BY ITSELF A BETTER DECOMPOSITION: a wide band "
                "can absorb a great deal, and five resolved "
                "transitions with an honest few percent left over is "
                "the better answer. Read it beside the mechanism."
            )
        )
    v = a.get("verdict")
    if v == RESOLVABILITY_RESOLVED:
        L.append(verdict("ok", "RESOLVED — decompose"))
    elif v == RESOLVABILITY_QUALIFIED:
        L.append(
            verdict(
                "caution",
                "QUALIFIED — decompose, and say what the areas are a share OF",
            )
        )
    else:
        L.append(
            verdict(
                "bad",
                "UNRESOLVED — this curve has no peak/background split the data decides",
            )
        )
    L.append(bullet(a.get("reason", "")))
    if a.get("mechanism"):
        _m = {
            "two_phase": "TWO-PHASE — one transition at a fixed potential",
            "multi_transition": "SERIES OF TRANSITIONS — each one a peak",
            "solid_solution": "SOLID SOLUTION — a continuum, fitted as bands",
            "mixed": "MIXED — a solid solution with a transition on it: "
            "bands AND peaks",
        }.get(a["mechanism"], a["mechanism"])
        L.append(
            entry(
                "mechanism",
                a["mechanism"],
                (
                    f"escalated from {a['mechanism_escalated_from']}"
                    if a.get("mechanism_escalated_from")
                    else ""
                ),
            )
        )
        if np.isfinite(a.get("at_bound_fraction", np.nan)):
            L.append(
                entry(
                    "at the width ceiling",
                    f"{a['at_bound_fraction']:.0%}",
                    "of the reference cycle's components — a peak at "
                    "its width bound is a band being denied",
                )
            )
        L.append(
            verdict(
                "ok"
                if a["mechanism"] in ("two_phase", "multi_transition")
                else "caution",
                _m,
            )
        )
        L.append(bullet(a.get("mechanism_reason", "")))
    return "\n".join(L)


# ---------------------------------------------------------------------------
# WHICH MECHANISM? — and therefore which model
# ---------------------------------------------------------------------------
# The profile class (sharp / moderate / broad) describes the SHAPE of a dQ/dV
# curve. It does not say what produced that shape, and two materials that need
# completely different models land in the same class: NNM and NMC111 are both
# "broad", and forcing them through one model damages whichever one loses.
#
# Measured, when the flat-top band model was applied to every broad dataset:
#
#     NNM cell B   closure interval  0.101 -> 0.179   reliable 60% -> 28%
#     NNM cell C   closure interval  0.052 -> 0.140   reliable 71% -> 33%
#     NMC111       closure interval  0.34  -> 0.32/0.18, convergence 13% -> 59%
#
# The same change, opposite effect, because the curves are not the same kind
# of object. So the classification has to be about the MECHANISM:
#
#   TWO_PHASE         one transition at a fixed potential. LTO, LFP. A narrow
#                     peak on a flat background; the potential is pinned by
#                     the coexistence of two phases, so there is no window to
#                     smear over.
#
#   MULTI_TRANSITION  a SERIES of distinct transitions. P3 sodium layered
#                     oxides. Several resolved maxima, each a real feature;
#                     the answer is a series of peaks and the split between
#                     them is decided by the data.
#
#   SOLID_SOLUTION    a continuum of site energies. Charge is delivered across
#                     a WINDOW of potentials as the composition moves, so the
#                     curve is a broad flat-topped envelope and its "shoulders"
#                     are artefacts of trying to build a flat top out of
#                     Gaussians.
#
#   MIXED             a solid solution WITH a phase transition on top of it —
#                     which is what NMC111 is. It needs both: bands for the
#                     envelope and peaks for the transition, in one fit.
#
# THE TWO DISCRIMINATORS ARE ALREADY MEASURED, and they separate the three
# chemistries in hand cleanly:
#
#     dataset   shoulder fraction   reference-cycle closure interval
#     LTO             0%                    0.00 - 0.02
#     NNM            11 - 22%               0.05 - 0.08
#     NMC111         34 - 68%               0.32 - 0.36
#
# A shoulder is a bend on a flank rather than a maximum. A curve made of real
# transitions has few; a solid-solution envelope fitted with peaks has many,
# because the shoulder pass is finding the flanks of the stack. The closure
# interval then says whether the peak/background split those components imply
# is decided by the data.
#
# Both thresholds sit in the gap between the measured populations, not at a
# convenient round number, and both are stated here so the next dataset can
# move them.
MECHANISM_SHOULDER_FRACTION = 0.30  # NNM tops out at 0.22, NMC starts 0.34
MECHANISM_TWO_PHASE_MAX_COMPONENTS = 2  # one transition, per half-cycle

MECHANISM_TWO_PHASE = "two_phase"
MECHANISM_MULTI_TRANSITION = "multi_transition"
MECHANISM_SOLID_SOLUTION = "solid_solution"
MECHANISM_MIXED = "mixed"

# What each mechanism asks the fitter for. `band_ceiling_span_fraction` of 0
# means the bare pseudo-Voigt; `primaries_only` drops the shoulder stack,
# which is only ever right where the shoulders are artefacts of the wrong
# shape.
#
# THE CEILING IS A FRACTION OF THE HALF-CYCLE'S OWN VOLTAGE SPAN, NOT VOLTS.
# Until 1.9.0.72 it was 0.5 V flat, for every material and every window. That
# is 38% of LTO's 1.3 V window, 33% of NMC's 1.5 V and 24% of NNM's 2.06 V —
# one number meaning three different things, and on NNM it was far below what
# the data wanted. Measured on the NNM triplicate (3003 components):
#
#   ceiling            bands pinned at it   reliable   R2      area stderr
#                                                              determined
#   0.5 V (flat)               82%            49.9%   0.9916      13-38%
#   1.2 V                       6%            69.4%   0.9957      23-40%
#   unbounded                   5%            65.2%   0.9951         --
#
# A pinned width is not a width the data chose, and lmfit cannot estimate a
# covariance for a parameter sitting on a bound — which is why two thirds of
# NNM's components had no area error bar at all. Unbounded is WORSE than 1.2
# V, not better: 16% of bands then run past 1.2 V, 103 of them end up WIDER
# THAN THE HALF-CYCLE THEY ARE FITTED IN, and not one of those has a
# determinable area. A band wider than its own window has no flanks inside
# the data; it is a pedestal, and the ceiling is what stops it.
#
# 0.6 is where the measured distribution turns. Binned by width/span in the
# unbounded run: 0.4-0.6 is the healthiest population in the dataset (61%
# reliable, 8% truncated), 0.6-0.8 falls to 31% reliable and 47% truncated,
# and above 1.0 nothing is determinable at all. On NNM 0.6 x 2.06 V = 1.24 V,
# which is the arm that measured best.
BAND_CEILING_SPAN_FRACTION = 0.6

MECHANISM_MODEL = {
    MECHANISM_TWO_PHASE: dict(band_ceiling_span_fraction=0.0, primaries_only=False),
    MECHANISM_MULTI_TRANSITION: dict(
        band_ceiling_span_fraction=0.0, primaries_only=False
    ),
    MECHANISM_SOLID_SOLUTION: dict(
        band_ceiling_span_fraction=BAND_CEILING_SPAN_FRACTION, primaries_only=True
    ),
    # MIXED KEEPS ITS SHOULDERS. 1.9.0.56 first shipped this as
    # `primaries_only=True`, copied from `solid_solution`, and it cost a real
    # feature the same day.
    #
    # The original argument for dropping shoulders was sound FOR ITS TIME: a
    # sum of Gaussians can only build a flat top by stacking overlapping
    # components, and the shoulder pass then finds the flanks of that stack.
    # On a solid-solution envelope those shoulders really are artefacts of the
    # wrong shape, and a band should REPLACE the stack rather than join it.
    #
    # That argument depended on bands being made from the detected peak list.
    # They are not any more: from 1.9.0.56 a band is seeded from the charge
    # the peaks cannot account for, at a region established across the
    # dataset, and it has nothing to do with which components were flagged as
    # shoulders. The two are decoupled, and a real shoulder on a real peak is
    # not an artefact of anything.
    #
    # Measured, NNM cell A cycle 1 discharge. Detection finds
    # 2.081, 3.147(shoulder), 3.239, 3.551, 4.084:
    #
    #   primaries_only=True   seeds drop 3.147 -> fitted 2.089 2.636 3.207
    #                         3.549 3.869 4.082   (one component where there
    #                         are two, sitting between them)
    #   primaries_only=False  seeds keep 3.147  -> fitted 2.063 2.741 3.150
    #                         3.234 3.549 3.829 4.082   (both resolved)
    #
    # The shoulder is detected in 10 of the first 12 discharge half-cycles and
    # was being stripped from every one of them. `mixed` means bands AND
    # peaks; taking the peaks' shoulders away contradicts its own name.
    MECHANISM_MIXED: dict(
        band_ceiling_span_fraction=BAND_CEILING_SPAN_FRACTION, primaries_only=False
    ),
}


def classify_mechanism(
    shoulder_fraction, closure_interval, *, n_primaries=None, n_components=None
):
    """Which redox mechanism is this curve, and therefore which model?

    Returns `(mechanism, reason)`. Both inputs come from
    `assess_resolvability`, which measures them on the reference cycle before
    any full fitting is committed to.
    """
    sf = float(shoulder_fraction) if shoulder_fraction is not None else np.nan
    ci = float(closure_interval) if closure_interval is not None else np.nan

    # THE SHOULDER FRACTION DECIDES THE MECHANISM; THE CLOSURE INTERVAL
    # DECIDES WHETHER THE ANSWER IS QUOTABLE. Two different questions, and
    # the first version conflated them: NNM cell B has 0% shoulders — every
    # component a resolved maximum — and was called a solid solution because
    # its interval came in at 0.17, a whisker over the threshold. A curve
    # with no bends on any flank cannot be a continuum envelope whatever the
    # interval says; a wide interval there means the BASELINE is undetermined,
    # which is what `verdict` already reports, separately and correctly.
    #
    # So the interval no longer gates the mechanism. It is still the reason a
    # dataset's areas may be withheld, and nothing about that changes.
    _resolved = (not np.isfinite(sf)) or sf < MECHANISM_SHOULDER_FRACTION
    if _resolved:
        _few = (
            n_components is not None
            and n_components <= MECHANISM_TWO_PHASE_MAX_COMPONENTS
        )
        if _few:
            return (
                MECHANISM_TWO_PHASE,
                f"one resolved feature ({n_components} component(s)) and "
                f"no shoulders to speak of ({sf:.0%}): a two-phase "
                f"transition at a fixed potential",
            )
        return (
            MECHANISM_MULTI_TRANSITION,
            f"{n_components if n_components is not None else 'several'} "
            f"resolved features and only {sf:.0%} of them bends on a "
            f"flank: a series of distinct transitions, each one a peak",
        )

    # Not resolved as a set of peaks. Is there still a transition in there?
    _has_peaks = bool(n_primaries) and n_primaries >= 1
    if _has_peaks:
        return (
            MECHANISM_MIXED,
            f"{sf:.0%} of the components are bends on a flank and the "
            f"split moves by {ci:.2f} with the baseline — a solid-solution "
            f"envelope — but {n_primaries} resolved maxima remain, so a "
            f"transition sits on top of it. Bands AND peaks",
        )
    return (
        MECHANISM_SOLID_SOLUTION,
        f"{sf:.0%} of the components are bends on a flank and the split "
        f"moves by {ci:.2f} with the baseline, with no resolved maximum "
        f"left: a continuum of site energies, and no peaks to find in it",
    )


# ---------------------------------------------------------------------------
# Presentation — describes, does not judge
# ---------------------------------------------------------------------------

# --- ONE MATERIAL, ONE MODEL ---------------------------------------------
# A mechanism is classified from ONE reference cycle of ONE cell, and that is
# a thin basis on which to choose a model. On the NMC111 pair at 1.9.0.56 it
# split the replicates — cell A `multi_transition`, cell B `mixed` — and the
# run page had to warn that two cells fitted with different models cannot be
# pooled. That warning is right, and it is also an admission: the pooling is
# the point of running replicates.
#
# A material either delivers charge across a composition window or it does
# not. It is not a property that varies between nominally identical coin
# cells, so a replicate that did not detect the continuum is a DETECTION MISS,
# not evidence of absence — which is why this reconciles to the most permissive
# call in the group rather than to a majority. (A majority is undefined at
# n = 2, which is the common case here.)
#
# The disagreement is not hidden: every cell's own call is kept in
# `mechanism_own`, and the run page says which cells were reconciled and from
# what.
_MECHANISM_RANK = {
    MECHANISM_TWO_PHASE: 0,
    MECHANISM_MULTI_TRANSITION: 1,
    MECHANISM_SOLID_SOLUTION: 2,
    MECHANISM_MIXED: 3,
}


def reconcile_mechanisms(resolvability, compositions, *, verbose=True):
    """
    One model per MATERIAL, across its replicates. Mutates and returns
    `resolvability` (name -> assess_resolvability dict).

    `compositions` maps the same names to a material label. Names whose
    composition is unknown or unique are left exactly as they were.
    """
    groups = {}
    for name, a in (resolvability or {}).items():
        if not a or not a.get("mechanism"):
            continue
        comp = str((compositions or {}).get(name) or name)
        groups.setdefault(comp, []).append(name)

    changed = []
    for comp, names in groups.items():
        if len(names) < 2:
            continue
        calls = {n: resolvability[n]["mechanism"] for n in names}
        best = max(calls.values(), key=lambda m: _MECHANISM_RANK.get(m, 0))
        if len(set(calls.values())) == 1:
            continue
        for n, own in calls.items():
            a = resolvability[n]
            a.setdefault("mechanism_own", own)
            if own == best:
                continue
            a["mechanism"] = best
            a["model"] = dict(MECHANISM_MODEL[best])
            # `mechanism_reconciled_from` used to be set here, holding `own` —
            # the same value `mechanism_own` two lines above already holds,
            # under a second name, read by nothing. The run page answers "was
            # this cell reconciled, and from what" from `mechanism_own`
            # (report.py:2734). One quantity, one name.
            a["mechanism_reason"] = (
                f"reconciled to {best.replace('_', ' ')}, the most permissive "
                f"call among the {len(names)} {comp} cells: this cell's own "
                f"reference cycle read {own.replace('_', ' ')}. A material "
                f"either delivers charge across a composition window or it "
                f"does not, so a replicate that did not resolve one is a "
                f"detection miss rather than evidence of absence — and two "
                f"replicates fitted with different models cannot be pooled, "
                f"which is what replicates are for. Original: "
                + str(a.get("mechanism_reason", ""))
            )
            changed.append((n, own, best))

    if verbose and changed:
        print(section("  One material, one model"))
        for n, own, best in changed:
            print(
                entry(str(n)[:28], f"{own} -> {best}", "reconciled across replicates")
            )
    return resolvability


def describe_partition(intervals, indent="  "):
    """
    What the closure interval says about this dataset, laid out to be read.

    A titled block with aligned fields and a coloured verdict — see `style`.
    The reader is looking for one of four things (how wide, how many pass, the
    verdict, whether anything went wrong) and should land on it without
    parsing a sentence.

    The verdict is decided by the FRACTION of half-cycles inside
    `PARTITION_DETERMINED_BELOW`, the same threshold `capacity_attribution`
    acts on, so the sentence and the decision cannot disagree.

    Three things this deliberately does not do:

    * It does not read a verdict off the median alone. A median of 0.04 and
      one of 0.06 are the same measurement to within the noise on a few
      hundred half-cycles, and the previous version gave them OPPOSITE
      verdicts on either side of a hard 0.05.
    * It does not quote the min and max. One failed fit sends the maximum to
      12 and makes a dataset whose median is 0.04 look ruined.
    * It does not say the areas "may be read as capacities". Closure asks
      whether the fitted peaks account for the CURVE; whether the curve
      accounts for the CELL is `integral_fidelity`, and on a two-phase
      material the answer is no.
    """
    widths = np.array(
        [iv["width"] for iv in intervals if np.isfinite(iv["width"])], float
    )
    los = np.array([iv["lo"] for iv in intervals if np.isfinite(iv["lo"])])
    his = np.array([iv["hi"] for iv in intervals if np.isfinite(iv["hi"])])
    n_fail = sum(1 for iv in intervals if not np.isfinite(iv["width"]))

    L = [indent + section("Closure — do the fitted peaks account for the curve?")]
    if widths.size == 0:
        L.append(verdict("bad", "not measurable"))
        if n_fail:
            L.append(bullet(f"{n_fail} half-cycle(s) never converged"))
        return "\n".join(L)

    runaway = int(
        np.sum(
            [
                iv["hi"] > CLOSURE_RUNAWAY_ABOVE
                for iv in intervals
                if np.isfinite(iv["hi"])
            ]
        )
    )
    lo, hi = float(np.median(los)), float(np.median(his))
    w = float(np.median(widths))
    q1, q3 = (float(np.percentile(widths, 25)), float(np.percentile(widths, 75)))
    n_ok = int(np.sum(widths <= PARTITION_DETERMINED_BELOW))
    frac = n_ok / widths.size

    L.append(
        entry(
            "closure range",
            f"{lo:.2f} – {hi:.2f}",
            f"median across baseline degrees "
            f"{DEFAULT_DEGREES[0]}–{DEFAULT_DEGREES[-1]}",
        )
    )
    L.append(entry("interval width", f"{w:.2f}", f"middle half {q1:.2f} – {q3:.2f}"))
    L.append(
        entry(
            "determined",
            f"{n_ok} of {widths.size}",
            f"{frac:.0%} inside {PARTITION_DETERMINED_BELOW:.2f}",
        )
    )
    n_collapsed = int(sum(int(iv.get("n_degrees_collapsed") or 0) for iv in intervals))
    if n_collapsed:
        L.append(
            entry(
                "degrees dropped",
                f"{n_collapsed}",
                "fit collapsed (R2 <= 0) — not evidence about the split",
            )
        )

    if frac >= 0.95:
        L.append(verdict("ok", "DETERMINED — the split holds across baseline degree"))
        if n_ok < widths.size:
            # A dataset can sit at exactly 95% and still have a half-cycle
            # withheld. The verdict is where a reader stops, and RUN_SUMMARY
            # then says "1 cycle had their capacity attribution withheld" —
            # which reads as a contradiction unless the tick says it first.
            L.append(
                bullet(
                    f"{widths.size - n_ok} half-cycle(s) still "
                    f"exceed the threshold and have their capacity "
                    f"share withheld individually."
                )
            )
        L.append(
            bullet(
                "Whether these areas are CAPACITIES is a separate "
                "question; see integral fidelity."
            )
        )
    elif frac >= 0.5:
        L.append(verdict("caution", "MIXED — holds on most half-cycles, not all"))
        L.append(
            bullet(
                f"The {widths.size - n_ok} that fail are withheld "
                f"individually, not the whole dataset."
            )
        )
    else:
        L.append(verdict("bad", "NOT DETERMINED — do not quote attribution"))
        L.append(
            bullet(
                "The split depends on the baseline degree chosen, so "
                "the areas are an artefact of that choice."
            )
        )

    if runaway:
        L.append(
            entry(
                "failed fits",
                f"{runaway} of {widths.size}",
                f"closure > {CLOSURE_RUNAWAY_ABOVE} at some degree",
            )
        )
        L.append(
            bullet(
                "A component ran away and took several times the "
                "curve's own area — a failed fit, not a wide "
                "interval."
            )
        )
    if n_fail:
        L.append(
            entry(
                "no fit",
                f"{n_fail} of {len(intervals)}",
                "never converged at any baseline degree",
            )
        )
    return "\n".join(L)


def describe_fidelity(fidelity, indent="  ", floor=0.80, ceiling=1.20):
    """
    The other half of the quality question, laid out the same way.

    `fidelity` is `{(cycle, step): ratio}` from `integral_fidelity`. Closure
    asks whether the peaks account for the CURVE; this asks whether the curve
    accounts for the CELL. They are independent, and a dataset can pass one
    completely while failing the other — the LTO triplicate scores 0.98–1.00
    on closure and 0.58–0.68 here.
    """
    per = {}
    for key, f in (fidelity or {}).items():
        step = key[1] if isinstance(key, tuple) and len(key) > 1 else str(key)
        if np.isfinite(f):
            per.setdefault(str(step), []).append(float(f))
    L = [indent + section("Integral fidelity — does the curve account for the cell?")]
    if not per:
        L.append(verdict("caution", "not measurable"))
        return "\n".join(L)
    worst = 1.0
    most = 1.0
    for step in sorted(per):
        med = float(np.median(per[step]))
        worst = min(worst, med)
        most = max(most, med)
        L.append(
            entry(
                step.lower(), f"{med:.2f}", f"median over {len(per[step])} half-cycles"
            )
        )
    if worst < floor:
        L.append(verdict("bad", f"AREAS ARE NOT CAPACITIES — below {floor:.2f}"))
        L.append(
            bullet(
                f"Only {100 * worst:.0f}% of the delivered charge "
                f"appears in the dQ/dV curve, so a peak area is a "
                f"share of that fraction. On the derivative-free "
                f"curve this means charge delivered OUTSIDE the "
                f"analysed voltage window — a constant-voltage hold "
                f"is the usual reason."
            )
        )
        L.append(
            bullet(
                "Peak POSITIONS and drift do not use the integral and are unaffected."
            )
        )
    elif most > ceiling:
        # The other half of the docstring's promise. An excess is charge the
        # cell never passed, and nothing physical produces it.
        L.append(verdict("bad", f"AREAS ARE NOT CAPACITIES — above {ceiling:.2f}"))
        L.append(
            bullet(
                f"The dQ/dV curve carries {100 * most:.0f}% of the "
                f"charge the cycler counted. A curve cannot contain "
                f"more charge than the cell delivered, so some of "
                f"this area is an artefact — a voltage traversed "
                f"twice, or a hold spread across voltages it never "
                f"visited."
            )
        )
        L.append(
            bullet(
                "Peak POSITIONS and drift do not use the integral and are unaffected."
            )
        )
    else:
        L.append(verdict("ok", "the curve accounts for the cell's charge"))
    return "\n".join(L)


def describe_reconstruction(
    records, *, method, window, bin_widths=None, empty_fraction=None
):
    """
    Say whether the curve gives the voltage curve back, and what it cost.

    `records` are `voltage_reconstruction` results for one dataset, `window`
    is the (lo, hi) the curve was computed on, and `bin_widths` the width
    each half-cycle was ACTUALLY binned at — not the one that was asked for.
    Those differ, often: on P3 cell A, 354 of 364 half-cycles were widened
    from the requested 5 mV, to a median of 7.9 and a worst of 198, and
    nothing said so while the manifest recorded "5.0".
    """
    out = []
    if not records:
        return "\n".join(
            [
                entry(
                    "reconstruction",
                    "not measured",
                    "no half-cycle had both a curve and a capacity column",
                )
            ]
        )
    ape = float(np.nanmedian([r["ape_pct"] for r in records]))
    rmse = float(np.nanmedian([r["rmse_V"] for r in records]))
    span = float(window[1] - window[0]) if window else float("nan")
    frac = rmse / span if span and np.isfinite(span) and span > 0 else np.nan

    if bin_widths is not None and len(bin_widths):
        w = np.asarray(bin_widths, float)
        w = w[np.isfinite(w)]
    else:
        w = np.empty(0)
    if w.size:
        if w.max() - w.min() < 1e-9:
            out.append(
                entry("bin width", f"{w.min():g} mV", "the same on every half-cycle")
            )
        else:
            # The widening is not a fault: a bin narrower than the records
            # themselves cannot be filled, so it is opened up where they are
            # sparse. It is only a fault to REPORT the requested width as
            # though it had been used.
            n_wide = int((w > w.min() + 1e-9).sum())
            out.append(
                entry(
                    "bin width",
                    f"{np.median(w):.3g} mV median",
                    f"{w.min():g}-{w.max():.3g} mV; widened on "
                    f"{n_wide} of {w.size} half-cycles where the "
                    f"records were sparser than the bin",
                )
            )
    if empty_fraction is not None and np.isfinite(empty_fraction):
        out.append(entry("bins with no charge", f"{100 * empty_fraction:.0f}%"))
        if empty_fraction > 0.5:
            out.append(
                bullet(
                    "Expected on a flat plateau: the charge is concentrated in a "
                    "few mV and the rest of the window is genuinely empty. The "
                    "bins are not lost data — an empty bin carries zero and "
                    "contributes nothing to any area."
                )
            )

    out.append(
        entry(
            "reconstruction",
            f"{ape:.2f}% APE",
            f"voltage RMSE {1000 * rmse:.0f} mV"
            + (f" = {100 * frac:.1f}% of the window" if np.isfinite(frac) else "")
            + f", median over {len(records)} half-cycles",
        )
    )

    bad_ape = ape > RECONSTRUCTION_APE_ABOVE
    bad_rmse = np.isfinite(frac) and frac > RECONSTRUCTION_RMSE_WINDOW_FRACTION
    if bad_ape:
        out.append(
            verdict("caution", "the curve does not account for the cell's charge")
        )
        out.append(
            bullet(
                "A peak area on this dataset is a share of the fraction that "
                "survived, not of the cell."
                + (
                    ""
                    if method == "histogram"
                    else ' Try DQDV_COMPUTATION = "histogram" in Cell 2.'
                )
            )
        )
    elif bad_rmse:
        out.append(
            verdict(
                "caution", "the curve places charge at voltages the cell did not visit"
            )
        )
        out.append(
            bullet(
                f"Capacity is accounted for to {ape:.2f}%, but the voltage axis "
                f"is out by {100 * frac:.1f}% of the window. Peak centres are "
                f"the quantity this moves."
            )
        )
    else:
        out.append(verdict("ok", "the curve integrates back to the measured capacity"))
    return "\n".join(out)
