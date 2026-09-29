"""
Model construction and the fitting engine.

The one design decision that matters
------------------------------------
A worker returns a plain dictionary, never an lmfit `ModelResult`.

In 1.8.x, parallel fitting was blocked because a `ModelResult` cannot be
returned through stdlib pickle: `PolynomialModel` builds its function as a
closure, so only cloudpickle could carry it, which in turn meant only the
`loky` backend, which in turn meant worrying about which globals were
statically referenced from a notebook cell.

None of that is necessary. The optimiser's output that anyone downstream
actually uses is a few dozen floats. Returning those makes every backend work,
makes the payload small, and makes a fit result something that can be written
to disk and compared — which is what the 1.9.0 regression test needs.

Parallelism
-----------
`fit_many` measures before it commits: it times a few half-cycles serially,
estimates the total, and only spawns workers if the estimate clears
`PARALLEL_MIN_SECONDS`. Process start-up on Windows is expensive and a 40-second
job does not repay it. Every branch says out loud which way it went.

It measures MEMORY as well as time, and that is not a refinement — it is the
fix for a kernel that died twice in Cell 9. The old rule was `workers =
cpu_count - 1`, decided from a time estimate alone, on a machine where every
`spawn` worker re-imports numpy, scipy and lmfit and holds that footprint for
as long as it lives. Measured here: **181 MB per worker** (Linux, CPython
3.11), against a job payload of **5 KiB** — so the cost of a worker is the
cost of its imports, not of the data it is sent, and fifteen of them is two to
three gigabytes before a single fit is run.

Three things follow, and all three are implemented below:

* the pool is sized against FREE MEMORY as well as cores, using a per-worker
  cost measured on the machine actually running (`measure_worker_cost_mb`),
  not a constant;
* `PARALLEL_MAX_WORKERS` caps it regardless, because the twelfth worker buys
  little and costs another import footprint plus another Windows spawn;
* `release_workers()` exists and is called at the end of a fitting stage,
  because joblib keeps its workers alive after `Parallel` returns — measured,
  and NOT undone by exiting a `with Parallel(...)` block.

It is a ceiling, not a leak: one worker's RSS over eighty consecutive NNM
fits oscillated between 168 and 203 MB with no trend.
"""

from __future__ import annotations

import os
import time

import numpy as np
from lmfit import Model as _LMModel
from lmfit.models import PolynomialModel, PseudoVoigtModel
from scipy.special import erf

from .compat import trapezoid
from .style import bullet, entry, section, verdict

__all__ = [
    "FIT_TOLERANCE",
    "PARALLEL_MAX_WORKERS",
    "PARALLEL_MIN_SECONDS",
    "WORKER_MEMORY_HEADROOM",
    "FitSpec",
    "_describe_band_set",
    "band_regions",
    "build_model",
    "calibrate_shape",
    "evaluate",
    "fit_half_cycle",
    "fit_many",
    "fit_quality",
    "measure_worker_cost_mb",
    "memory_budget_workers",
    "plan_workers",
    "reconcile_band_sets",
    "release_workers",
    "residual_excess_regions",
    "shoulder_parents",
    "spec_for_profile",
]

# Below this estimated total, run serially. Measured on Windows/Anaconda: a
# loky pool costs ~5-8 s to start, so anything under a minute loses.
PARALLEL_MIN_SECONDS = 60.0

# --- How many workers a machine can afford --------------------------------
# A worker is a fresh interpreter that re-imports numpy, scipy and lmfit and
# then holds that footprint. Measured: 181 MB (Linux, CPython 3.11) against a
# 5 KiB job payload. `cpu_count - 1` on a 16-thread laptop is therefore ~2.7
# GB spent before the first fit, which is what killed the kernel in Cell 9 on
# a machine already at 80% RAM.

# Hard ceiling regardless of cores or memory. Past about eight workers the
# marginal fit rate is small, and each one costs another import footprint and
# another 5-8 s Windows spawn.
PARALLEL_MAX_WORKERS = 8

# Never plan to spend more than this share of the memory that is FREE RIGHT
# NOW. Half leaves room for the parent to keep loading data, for the operating
# system, and for whatever else the person has open — which on the machine
# that crashed was most of it.
WORKER_MEMORY_HEADROOM = 0.5

# The probe measures what a worker costs to START — its imports plus one
# small fit. A worker doing real broad-profile fitting settles higher, because
# lmfit and scipy allocate as they go: measured on the NNM triplicate, 150 MB
# imported against 181 MB in use, so 1.21. Rounded up, because the direction
# that is wrong here kills a kernel and the direction that is wrong the other
# way costs a worker.
WORKER_WORKING_SET_FACTOR = 1.25

# Used ONLY when the probe cannot run at all. Above the Linux measurement
# because Windows/Anaconda imports cost more, and an assumption that is too
# small is the failure being fixed.
WORKER_RSS_ASSUMED_MB = 250.0

# Sanity bounds on the probe, not a substitute for it. A measurement outside
# these is not a worker cost, it is a broken measurement.
WORKER_RSS_SANITY_MIN_MB = 80.0
WORKER_RSS_SANITY_MAX_MB = 2000.0

# Start one worker and ask it what it costs, once per session, before sizing
# the pool. Set False to skip the probe and use WORKER_RSS_FLOOR_MB instead.
WORKER_PROBE = True

# lmfit re-derives `fwhm` and `height` through asteval on every residual
# evaluation — 47.7% of fit time in 1.8.6, for quantities the optimiser never
# reads. Deleted before the fit, recomputed after (see `_derived`).
DROP_DERIVED = True

_FWHM_FACTOR = 2.0  # PseudoVoigtModel: fwhm = 2 * sigma
# sigma_gaussian = sigma / sqrt(2 ln 2), so the Gaussian and the Lorentzian
# halves of a pseudo-Voigt share one FWHM. See rect_pseudo_voigt.
_SIGMA_G_FACTOR = np.sqrt(2.0 * np.log(2.0))
_TINY = 1.0e-15


# --- NO FREE BASELINE, AND BANDS SEEDED FROM WHAT IS LEFT OVER ------------
# In a dQ/dV there is no instrumental background. Every mAh in the curve is
# charge that passed through the cell, so a "baseline" is not a thing to be
# subtracted -- it is an unnamed component that is then left out of the
# capacity attribution.
#
# It was also carrying most of the answer. Measured over 136 half-cycles from
# six cells, fraction of the cell's charge the model could NAME:
#
#                cubic baseline     no baseline
#     LTO             98.2%           100.3%      (and the cubic held 41.9%
#     NNM             62.4%            95.1%       of the area while the
#     NMC             33.1%            96.3%       components summed to 138%)
#
# On LTO the polynomial was going NEGATIVE to cancel components that had
# overshot by 38% -- a decomposition no fit statistic would ever have shown.
#
# What removing it exposes is that a band cannot be seeded from peak
# detection. A band is a REGION, not a maximum, and NMC111's 4.1-4.45 V shelf
# contains no maximum at all once the binning artefact is gone. With nothing
# to hang a band on, the fitter reaches that charge by stretching the Ni peak
# sideways: median FWHM 169 -> 329 mV.
#
# So bands are seeded from the CHARGE THE PEAKS CANNOT ACCOUNT FOR. Fit the
# peaks; find the contiguous span where the curve exceeds them; if it carries
# a material share of the step, put a band there and refit. No flatness
# criterion, no threshold tuned per chemistry, and it is the same question the
# report has to answer anyway -- which charge can we name?
#
# It lands in the same place every time: NMC111 4.094-4.448 V in 44 of 48
# half-cycles across both cells and twelve cycles, median 12.5% of the step.
# And the refit is the test that it belongs there:
#
#                      components   named    median FWHM   max residual
#     NMC peaks only       3.0       95.2%      328.6 mV       35.2%
#     NMC + band           4.0       99.8%      152.2 mV        9.2%
#
# An extra component that merely soaks up residual leaves the others alone.
# This one HALVES the peak widths, because it takes back work that was never
# theirs. That is the opposite signature to overfitting, and it is the reason
# the band is in the model rather than a fourth peak.
#
# GATED ON `band_width_max`, which only `quality.classify_mechanism` ever
# sets: two-phase and multi-transition mechanisms pass 0.0 and can never
# acquire a band this way. That gate is not decoration. Left ungated on LTO --
# one sharp two-phase peak -- residual seeding pushed the named charge to
# 103.1%, components summing to more than the cell delivered.
BAND_FROM_RESIDUAL = True
BAND_RESIDUAL_MIN_SHARE = 0.02  # of the half-cycle's charge
BAND_RESIDUAL_MAX = 2  # bands added per half-cycle
BAND_RESIDUAL_MIN_SPAN_MV = 100.0  # a band narrower than this is a peak

# --- ONE SET OF BANDS FOR THE DATASET, NOT ONE PER HALF-CYCLE -------------
# Seeding a band from THIS half-cycle's unattributed charge fixes the fit and
# breaks the tracking. The same physical region comes out as a band in one
# cycle and a peak in the next, so the component set changes identity from
# cycle to cycle and nothing downstream can follow it.
#
# Measured on NMC111 cell A charge at 1.9.0.56, twelve half-cycles:
#
#     band at 4.246 V   present in  8/12   spread 74 mV
#     band at 3.723 V   present in  4/12   spread 65 mV
#     band at 3.863 V   present in  2/12   spread 44 mV
#     band at 4.087 V   present in  1/12      —
#
# Four different peak/band patterns across those twelve (bbpp x7, ppp x2,
# bpp x2, bppp x1), five across cell B's seventeen. The coherence audit
# graded every one of cell A's features `not assessable`, and cell B's
# 4.206 V feature came back with a median drift of 106 mV per cycle — a
# tracking failure wearing the clothes of a measurement.
#
# Peaks solved this long ago: detect per half-cycle, then anchor to a
# REFERENCE LIST so a component missing from one cycle is a gap rather than a
# different model (`detect.infer_missing_seeds`). Bands get the same
# treatment. The excess regions are collected across a SAMPLE of half-cycles,
# clustered by voltage, and the recurrent ones become the dataset's band set —
# fitted in every half-cycle of that step whether or not that particular curve
# would have asked for one, because a band that comes and goes is not a
# measurement of anything.
#
# The recurrence is plainly there to find: the 4.19 V discharge band appears
# in 86% and 75% of half-cycles with a 6-9 mV spread.
BAND_REGION_SAMPLE = 12  # half-cycles used to establish the set
BAND_REGION_TOLERANCE_MV = 80.0  # two excess regions this close are the same
BAND_REGION_MIN_OCCUPANCY = 0.34  # ...and it must recur in this share of them
# --- RECORDED NEGATIVE: DO NOT POOL BAND REGIONS ACROSS REPLICATES -------
# `reconcile_mechanisms` pools the MECHANISM across the replicates of a
# material, and pooling the band regions the same way looks like the obvious
# next step. It was built, measured three ways, and it is wrong every time.
#
# The argument does not carry over. A mechanism is a PERMISSION and costs
# nothing where it is not needed; a band region is a COMPONENT, and a
# component has to earn its place in every half-cycle it is fitted to.
#
# NMC111 cell A finds two charge regions (3.712 V at 38%, 4.281 V at 88%);
# cell B's own pass finds none, because its detection already places a maximum
# on that shelf. Handing A's regions to B:
#
#                        cell B    blanket   bar at    promote the
#                        alone      union     0.67     maximum
#     components           81        100        81         69
#     at the ceiling       22         38        31         26
#     median FWHM       217.5 mV   315.3 mV  318.3 mV   268.5 mV
#     reliable           52/81      53/100     41/81      37/69
#     persistent (chg)     3          2          2          1
#
# Every refinement helped and none of them recovered cell B. The at-bound
# count rising is the model saying the imposed component does not belong
# there — the same signal that escalates a mechanism, read the other way.
#
# So the region set is established PER DATASET, from that cell's own
# half-cycles, and stops there. That is where the benefit was all along:
#
#                     sporadic components (<40% of cycles)
#                     per-half-cycle bands -> per-dataset set
#         cell A chg          1 -> 0
#         cell A dis          2 -> 0
#         cell B chg          3 -> 2
#         cell B dis          2 -> 1
#
# The code below is kept and the constant disables it, because the next person
# to have this idea should find the measurement rather than repeat it.
BAND_REGION_SHARED_OCCUPANCY = 1.01  # > 1 disables propagation entirely


def residual_excess_regions(voltage, residual, *, min_span_mV=None):
    """
    Contiguous spans where the curve exceeds the model, largest area first.

    `residual` is curve minus fit, in the curve's own units. Returns dicts of
    `lo`, `hi`, `centre`, `span_mV` and `area`; the area is the charge the
    model has not accounted for in that span.
    """
    v = np.asarray(voltage, float)
    r = np.asarray(residual, float)
    if v.size < 4 or r.size != v.size:
        return []
    min_span_mV = (
        BAND_RESIDUAL_MIN_SPAN_MV if min_span_mV is None else float(min_span_mV)
    )
    pos = np.isfinite(r) & (r > 0)
    out, i = [], 0
    while i < pos.size:
        if not pos[i]:
            i += 1
            continue
        j = i
        while j + 1 < pos.size and pos[j + 1]:
            j += 1
        span_mV = (v[j] - v[i]) * 1000.0
        if span_mV >= min_span_mV and j > i:
            out.append(
                dict(
                    lo=float(v[i]),
                    hi=float(v[j]),
                    centre=float(0.5 * (v[i] + v[j])),
                    span_mV=float(span_mV),
                    area=float(trapezoid(r[i : j + 1], v[i : j + 1])),
                )
            )
        i = j + 1
    return sorted(out, key=lambda d: -d["area"])


# A COMPONENT THAT ENDED AT ITS WIDTH BOUND IN THE PEAKS-ONLY PASS IS A
# REGION, whether or not any residual survived it. New in 1.9.0.60.
#
# `band_regions` fits the sample PEAKS ONLY and asks what the peaks cannot
# account for. That is the right question and it has a blind spot: a peak
# allowed to widen to `sigma_max` can cover a shelf entirely, and then there
# is no residual to find. The pass is blinded by the same bound it exists to
# detect, and the result is self-stabilising — no residual, so no region, so
# no band, so the peak stays at the bound.
#
# Measured on the NMC111 pair, top of charge (>4.15 V), 1.9.0.59:
#
#     cell A   18 components, ALL from detection, 14 fitted as BANDS with a
#              real width (0.31-0.48 V), several with sigma 28-60 mV
#     cell B   17 components, ALL from `add_recurrent_seeds`, ALL peaks, ALL
#              with FWHM exactly 0.400 = the ceiling, ALL `at_sigma_max`,
#              none `reliable`, carrying 37% of the charge-side area
#
# Same material, same run, two models. Cell B's chain is: the recurrent pass
# finds a maximum recurring on the shelf and seeds it AS A PEAK in every
# charge half-cycle; the peak widens to the bound to cover 300 mV of shelf;
# no residual survives; no region is found ("Charge bands none - the peaks
# account for this curve"); the band machinery never engages. Cell A escapes
# only because its own detection does not find that maximum.
#
# TWO EXPLANATIONS WERE TESTED FIRST AND BOTH ARE WRONG, recorded so they are
# not tried again:
#
#   * Cross-replicate propagation - handing cell A's regions to cell B. Built
#     three ways in 1.9.0.56 and every one made the ceiling-pinning WORSE
#     (22 at the ceiling alone, 38/31/26 with propagation). See
#     BAND_REGION_SHARED_OCCUPANCY.
#   * A local occupancy gate on detection, on the theory that the recurring
#     maximum is the quantisation beat on an under-sampled shelf. There is
#     only ONE detected charge peak above 4.10 V across both cells and 35
#     half-cycles, and its local occupancy is 2.14 - well sampled. A gate at
#     occ < 1.0 would reject 16% of NNM's real peaks and nothing on the NMC
#     shelf. Detection is not the fault.
#
# So the region is seeded from the bound itself, per dataset, from that
# cell's own half-cycles - which is where BAND_REGION_SHARED_OCCUPANCY's note
# says the benefit was all along. The span is the component's own bound
# width, which is not a tuned number: it is the widest the model was allowed
# to be, and therefore the least the region can be.
BAND_REGION_FROM_AT_BOUND = True


def band_regions(
    jobs,
    *,
    n=None,
    tolerance_mV=None,
    min_occupancy=None,
    min_share=None,
    n_jobs=None,
    verbose=True,
):
    """
    The dataset's band regions, established once. See BAND_REGION_SAMPLE.

    `jobs` is the production list of (voltage, dqdv, spec, key) with
    `key = (cycle, step)`. An evenly spaced sample is fitted PEAKS ONLY —
    whatever the spec says about bands is set aside, because the question is
    what the peaks alone cannot account for — and the contiguous excess spans
    are clustered by voltage WITHIN EACH STEP, charge and discharge being
    different curves with different regions.

    Returns `{step: [{centre, lo, hi, span_mV, occupancy, area_share}, ...]}`,
    ordered by voltage. A step with no recurrent region returns an empty list,
    which is the right answer for a two-phase cell and is what LTO gets.
    """
    n = int(BAND_REGION_SAMPLE if n is None else n)
    tol = (
        float(BAND_REGION_TOLERANCE_MV if tolerance_mV is None else tolerance_mV)
        / 1000.0
    )
    min_occ = float(
        BAND_REGION_MIN_OCCUPANCY if min_occupancy is None else min_occupancy
    )
    min_share = float(BAND_RESIDUAL_MIN_SHARE if min_share is None else min_share)
    out = {}
    if not jobs:
        return out
    # A mechanism that permits no band has no regions to look for, and the
    # sample fits would be spent finding tails it will never use. LTO's
    # residual does produce candidate spans (1.87 V charge, 1.78 V discharge)
    # and every one of them is the flank of its single two-phase peak.
    if not any(
        float(getattr(sp, "band_ceiling", 0.0) or 0.0) > 0 for _v, _y, sp, _k in jobs
    ):
        return out

    by_step = {}
    for v, y, sp, k in jobs:
        step = str(k[1]) if isinstance(k, (tuple, list)) and len(k) > 1 else ""
        by_step.setdefault(step, []).append((v, y, sp, k))

    for step, items in by_step.items():
        if len(items) > n:
            idx = np.linspace(0, len(items) - 1, n).round().astype(int)
            sample = [items[i] for i in sorted(set(idx.tolist()))]
        else:
            sample = list(items)
        # Peaks only: bands off, and no per-half-cycle seeding, so the excess
        # measured is what the PEAKS cannot account for.
        batch = [
            (
                v,
                y,
                sp.replace(
                    band_width_max=0.0,
                    band_ceiling=0.0,
                    band_from_residual=False,
                    band_seeds=(),
                ),
                k,
            )
            for v, y, sp, k in sample
        ]
        res = fit_many(batch, n_jobs=n_jobs, verbose=False, label="band regions")

        found = []
        n_ok = 0
        for (v, y, sp, k), r in zip(sample, res):
            if not r.get("success"):
                continue
            n_ok += 1
            va = np.asarray(v, float)
            ya = np.abs(np.asarray(y, float))
            fit = np.abs(evaluate(r, va)["total"])
            area = float(trapezoid(ya, va))
            if area <= 0:
                continue
            for g in residual_excess_regions(va, ya - fit):
                if g["area"] / area >= min_share:
                    g["key"] = k
                    found.append(g)
            # A PEAK AT ITS WIDTH BOUND IS A BAND BEING DENIED, and this pass
            # is the one place that rule was never applied. See
            # BAND_REGION_FROM_AT_BOUND.
            if BAND_REGION_FROM_AT_BOUND:
                _smax = float(r.get("sigma_max") or 0.0)
                for _c in r.get("components", ()):
                    _sg = abs(float(_c.get("sigma", np.nan)))
                    if not (
                        _smax > 0
                        and np.isfinite(_sg)
                        and _sg >= _smax * SIGMA_BOUND_PROXIMITY_COMPLEMENT
                    ):
                        continue
                    _a = abs(float(_c.get("amplitude_area", 0.0) or 0.0))
                    if area > 0 and _a / area < min_share:
                        continue
                    _ctr = float(_c.get("centre", np.nan))
                    if not np.isfinite(_ctr):
                        continue
                    found.append(
                        dict(
                            lo=_ctr - _smax,
                            hi=_ctr + _smax,
                            centre=_ctr,
                            span_mV=2000.0 * _smax,
                            area=_a,
                            from_bound=True,
                            key=k,
                        )
                    )
        if not n_ok or not found:
            out[step] = []
            continue

        # Cluster by centre. Single linkage at `tol`, which is generous
        # because an excess region's edges move with the fit that produced it;
        # what has to be stable is WHERE it is, not exactly how wide.
        found.sort(key=lambda g: g["centre"])
        groups, cur = [], [found[0]]
        for g in found[1:]:
            if g["centre"] - cur[-1]["centre"] <= tol:
                cur.append(g)
            else:
                groups.append(cur)
                cur = [g]
        groups.append(cur)

        regions = []
        for grp in groups:
            # HALF-CYCLES, NOT CANDIDATES. One half-cycle can now offer both a
            # residual span and an at-bound component in the same place, and
            # counting candidates made occupancy exceed 1.0 — which is not a
            # fraction of anything. See BAND_REGION_FROM_AT_BOUND.
            _seen = len({g.get("key") for g in grp}) or len(grp)
            occ = _seen / n_ok
            if occ < min_occ:
                continue
            regions.append(
                dict(
                    centre=float(np.median([g["centre"] for g in grp])),
                    lo=float(np.median([g["lo"] for g in grp])),
                    hi=float(np.median([g["hi"] for g in grp])),
                    span_mV=float(np.median([g["span_mV"] for g in grp])),
                    occupancy=float(occ),
                    n_seen=int(_seen),
                    n_sampled=int(n_ok),
                    from_bound=bool(any(g.get("from_bound") for g in grp)),
                )
            )
        out[step] = sorted(regions, key=lambda d: d["centre"])

    if verbose:
        for step in sorted(out):
            rs = out[step]
            if not rs:
                print(
                    entry(
                        f"{step} bands",
                        "none",
                        "no region recurs in enough half-cycles",
                    )
                )
                continue
            for r in rs:
                print(
                    entry(
                        f"{step} band",
                        f"{r['centre']:.3f} V",
                        f"{r['span_mV']:.0f} mV wide, in "
                        f"{r['n_seen']}/{r['n_sampled']} sampled "
                        f"half-cycles ({r['occupancy']:.0%})",
                    )
                )
    return out


def _describe_band_set(bs):
    """The dataset's band regions, laid out to be read."""
    out = []
    for step in sorted(bs or {}):
        rs = bs[step]
        if not rs:
            out.append(
                entry(f"{step} bands", "none", "the peaks account for this curve")
            )
            continue
        for r in rs:
            _n = r.get("n_cells")
            out.append(
                entry(
                    f"{step} band",
                    f"{r['centre']:.3f} V",
                    f"{r['span_mV']:.0f} mV wide, seen in "
                    f"{r['occupancy']:.0%} of sampled half-cycles"
                    + (f" across {_n} cell(s)" if _n else ""),
                )
            )
    return out


def reconcile_band_sets(band_sets, compositions, *, tolerance_mV=None, verbose=True):
    """
    One band set per MATERIAL, across its replicates. See
    `quality.reconcile_mechanisms`, which makes the same argument about the
    mechanism: a material either delivers charge across a composition window
    or it does not, and a replicate that did not resolve one has missed it
    rather than disproved it.

    It matters because the regions are found from a SAMPLE, and a sample can
    miss a region a longer look would find. Measured on the NMC111 pair: over
    every charge half-cycle both cells return 3.72-3.73 V (37%, 47%) and
    4.27-4.28 V (95%, 100%) — the same two regions — while the sampled
    pipeline pass found both on cell A and neither on cell B.

    `band_sets` maps name -> {step: [region, ...]} from `band_regions`;
    `compositions` maps the same names to a material label. Regions are
    unioned within a material and clustered by centre. Mutates and returns
    `band_sets`.
    """
    tol = (
        float(BAND_REGION_TOLERANCE_MV if tolerance_mV is None else tolerance_mV)
        / 1000.0
    )
    groups = {}
    for name in band_sets or {}:
        groups.setdefault(str((compositions or {}).get(name) or name), []).append(name)

    changed = []
    for comp, names in groups.items():
        if len(names) < 2:
            continue
        steps = set()
        for n in names:
            steps.update(band_sets[n] or {})
        for step in sorted(steps):
            # Only a WELL-ESTABLISHED region may be imposed on a replicate
            # that did not find it. See BAND_REGION_SHARED_OCCUPANCY.
            pooled = [
                r
                for n in names
                for r in (band_sets[n] or {}).get(step, [])
                if float(r.get("occupancy", 0.0)) >= BAND_REGION_SHARED_OCCUPANCY
            ]
            if not pooled:
                continue
            pooled.sort(key=lambda r: r["centre"])
            grp, cur = [], [pooled[0]]
            for r in pooled[1:]:
                if r["centre"] - cur[-1]["centre"] <= tol:
                    cur.append(r)
                else:
                    grp.append(cur)
                    cur = [r]
            grp.append(cur)
            merged = [
                dict(
                    centre=float(np.median([r["centre"] for r in g])),
                    lo=float(np.median([r["lo"] for r in g])),
                    hi=float(np.median([r["hi"] for r in g])),
                    span_mV=float(np.median([r["span_mV"] for r in g])),
                    occupancy=float(max(r["occupancy"] for r in g)),
                    n_cells=len(
                        set(
                            n
                            for n in names
                            for r2 in (band_sets[n] or {}).get(step, [])
                            if any(abs(r2["centre"] - r["centre"]) <= tol for r in g)
                        )
                    ),
                )
                for g in grp
            ]
            # A cell keeps everything it found itself, and gains only the
            # well-established regions it missed.
            for n in names:
                own = list((band_sets[n] or {}).get(step, []))
                before = len(own)
                for m in merged:
                    if not any(abs(r["centre"] - m["centre"]) <= tol for r in own):
                        own.append(dict(m, from_replicate=True))
                own.sort(key=lambda d: d["centre"])
                band_sets.setdefault(n, {})[step] = own
                if before != len(own):
                    changed.append((n, step, before, len(own)))

    if verbose and changed:
        print(section("  One material, one band set"))
        for n, step, a, b in changed:
            print(
                entry(
                    f"{str(n)[:24]} {step}",
                    f"{a} -> {b}",
                    "regions, pooled across replicates",
                )
            )
    return band_sets


def band_ceiling_volts(spec, voltage):
    """
    The widest band this half-cycle may contain, in volts.

    `spec.band_ceiling` is a FRACTION OF THE HALF-CYCLE'S OWN VOLTAGE SPAN,
    not a width — one number cannot mean the same thing on a 1.3 V window and
    a 2.1 V one. See `quality.MECHANISM_MODEL` for the measurement that
    settled the fraction, and `_band_ceilings` for the per-component array
    this feeds.

    Returns 0.0 where the mechanism permits no band, or where the half-cycle
    carries too little voltage to define a span — in both cases every
    component stays a bare pseudo-Voigt.
    """
    f = float(getattr(spec, "band_ceiling", 0.0) or 0.0)
    if f <= 0.0:
        return 0.0
    v = np.asarray(voltage, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 0.0
    return float(f * (v.max() - v.min()))


def _band_ceilings(spec):
    """Per-component band-width ceiling, as an array of len(spec.centres)."""
    n = len(spec.centres)
    bw = getattr(spec, "band_width_max", 0.0)
    if np.ndim(bw) == 0:
        return np.full(n, float(bw or 0.0))
    bw = np.asarray(bw, float)
    if bw.size == n:
        return bw
    out = np.zeros(n)
    out[: min(n, bw.size)] = bw[: min(n, bw.size)]
    return out


class FitSpec:
    """
    Everything needed to fit one half-cycle, and nothing else.

    Deliberately a value object with no reference to a dataset, a notebook
    global or a file — so it can be pickled to a worker and so a test can
    construct one by hand.
    """

    __slots__ = (
        "centres",
        "baseline_degree",
        "sigma_min",
        "sigma_max",
        "centre_tol",
        "max_nfev",
        "shoulder_of",
        "fraction",
        "sigma_ratio",
        "sigma0",
        "amp0",
        "band_width_max",
        "asymmetry",
        "asymmetry_edge_only",
        # Staged band seeding — see BAND_FROM_RESIDUAL.
        "band_from_residual",
        "band_residual_min_share",
        "band_residual_max",
        "band_residual_min_span_mV",
        "band_ceiling",
        "band_seeds",
        # None means "use the module's FIT_TOLERANCE". Carried on the
        # spec rather than read from a global because a fit may run
        # in a worker process, where a global set in the parent does
        # not exist. `quality.assess_closure` uses it.
        "tolerance",
    )

    def __init__(
        self,
        centres,
        baseline_degree=3,
        sigma_min=0.003,
        sigma_max=0.200,
        centre_tol=0.070,
        max_nfev=20000,
        shoulder_of=None,
        fraction=None,
        sigma_ratio=None,
        sigma0=None,
        amp0=None,
        band_width_max=0.0,
        asymmetry=None,
        asymmetry_edge_only=None,
        band_from_residual=None,
        band_residual_min_share=None,
        band_residual_max=None,
        band_residual_min_span_mV=None,
        band_ceiling=0.0,
        band_seeds=None,
        tolerance=None,
    ):
        # baseline_degree defaults to 3, matching 1.8.7's BASELINE_DEGREE. It
        # was 1, and that is not a small difference: on P3 cell A cycle 2
        # charge a linear baseline gives R2 0.27 against 0.54 for a cubic.
        # A broad layered-oxide dQ/dV sits on a curved background, and a
        # straight line cannot follow it. Use `spec_for_profile` rather than
        # this default wherever the profile is known.
        self.centres = np.asarray(centres, float)
        # PER-COMPONENT STARTING VALUES, when the caller has measured some.
        #
        # Without these every component of every half-cycle starts from the
        # same two numbers: sigma = min(0.02, sigma_max/2) and amplitude =
        # 5% of the curve's maximum. On a broad profile that is a 20 mV
        # start for components that fit to a median of 85 mV — four times
        # too narrow, on every peak, in every cycle — and one amplitude for
        # the tallest and the smallest alike. The optimiser then spends
        # thousands of evaluations doing nothing but growing the peaks to
        # roughly the size detection had already measured them to be.
        #
        # Detection measures a height and a width for every peak it finds.
        # Passing them costs nothing and starts the fit near the answer.
        # The widest composition window a component may claim. Zero keeps
        # the bare pseudo-Voigt and the historical behaviour exactly.
        # A CEILING PER COMPONENT, NOT ONE FOR THE HALF-CYCLE.
        #
        # A single global ceiling lets EVERY component become a 0.5 V
        # flat-topped band, and with no polynomial to compete with, the
        # optimiser takes it: measured on NMC111 cell A with the ceiling
        # global, the model named ~100% of the charge but put a component on
        # the width bound in three cycles out of four and moved every centre
        # between cycles -- 3.81/3.99, then 3.78/3.93/4.24, then 3.89/4.30.
        # Naming all the charge with a different set of blobs each cycle is
        # not a decomposition.
        #
        # So a detected PEAK gets a ceiling of zero and stays a peak, and only
        # a component seeded from unattributed charge gets a band -- bounded
        # by the SPAN OF THE REGION IT WAS FOUND IN, which is a measurement,
        # rather than by a constant nobody can defend. Scalars still work and
        # apply to every component, which is what the historical behaviour
        # and every existing test expect.
        if np.ndim(band_width_max) == 0:
            self.band_width_max = float(band_width_max or 0.0)
        else:
            self.band_width_max = np.asarray(band_width_max, float)
        # None means "take the module default", so a spec built before the
        # asymmetry existed behaves as the constant says rather than as False.
        self.asymmetry = ASYMMETRY if asymmetry is None else bool(asymmetry)
        # WHICH components may be asymmetric. See ASYMMETRY_EDGE_ONLY.
        self.asymmetry_edge_only = (
            ASYMMETRY_EDGE_ONLY
            if asymmetry_edge_only is None
            else bool(asymmetry_edge_only)
        )
        self.tolerance = None if tolerance is None else float(tolerance)
        # The widest band the MECHANISM permits, AS A FRACTION OF THE
        # HALF-CYCLE'S OWN VOLTAGE SPAN. `band_width_max` says what each
        # component currently is, in volts; this says what a component seeded
        # from unattributed charge would be allowed to become, and it is a
        # fraction because a FitSpec is built before the voltage array is in
        # hand. `band_ceiling_volts` resolves it. Zero means bands are not
        # this mechanism's model and none can be seeded.
        self.band_ceiling = float(band_ceiling or 0.0)
        # THE DATASET'S band regions, as (centre_V, span_mV) pairs, from
        # `band_regions`. When present these replace the per-half-cycle
        # discovery entirely and are fitted in EVERY half-cycle of the step,
        # whether or not this particular curve would have asked for one — a
        # band that comes and goes is not a measurement. Empty or None keeps
        # the per-half-cycle behaviour. See BAND_REGION_SAMPLE.
        self.band_seeds = (
            tuple((float(a), float(b)) for a, b in band_seeds) if band_seeds else ()
        )
        self.band_from_residual = (
            BAND_FROM_RESIDUAL
            if band_from_residual is None
            else bool(band_from_residual)
        )
        self.band_residual_min_share = float(
            BAND_RESIDUAL_MIN_SHARE
            if band_residual_min_share is None
            else band_residual_min_share
        )
        self.band_residual_max = int(
            BAND_RESIDUAL_MAX if band_residual_max is None else band_residual_max
        )
        self.band_residual_min_span_mV = float(
            BAND_RESIDUAL_MIN_SPAN_MV
            if band_residual_min_span_mV is None
            else band_residual_min_span_mV
        )
        self.sigma0 = None if sigma0 is None else np.asarray(sigma0, float)
        self.amp0 = None if amp0 is None else np.asarray(amp0, float)
        # Which components are shoulders, and of what. `shoulder_of[i]` is
        # the index of component i's parent, or None if i is a primary peak.
        # See COUPLE_SHOULDER_SIGMA.
        self.shoulder_of = (
            tuple(shoulder_of)
            if shoulder_of is not None
            else (None,) * len(self.centres)
        )
        # The two SHAPE parameters, calibrated once per dataset and then
        # held. None means "refine it here", which is what `calibrate_shape`
        # does on its sample and what a bare FitSpec still does. See
        # REFINE_FRACTION_PER_HALF_CYCLE.
        self.fraction = None if fraction is None else float(fraction)
        self.sigma_ratio = None if sigma_ratio is None else float(sigma_ratio)
        self.baseline_degree = baseline_degree
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.centre_tol = float(centre_tol)
        self.max_nfev = int(max_nfev)

    def replace(self, **kw):
        d = {k: getattr(self, k) for k in self.__slots__}
        d.update(kw)
        return FitSpec(**d)

    def __repr__(self):
        n_sh = sum(1 for x in self.shoulder_of if x is not None)
        return (
            f"FitSpec(n={len(self.centres)}, deg={self.baseline_degree}, "
            f"sigma={self.sigma_min}-{self.sigma_max}"
            + (f", {n_sh} shoulder(s) coupled" if n_sh else "")
            + (
                f", eta={self.fraction:.2f}"
                if self.fraction is not None
                else ", eta free"
            )
            + (f", k={self.sigma_ratio:.2f}" if self.sigma_ratio is not None else "")
            + ")"
        )


# 1.8.7 Module 4 chose these from the dataset's profile class, and the choice
# matters more than any other fitting parameter. A sharp two-phase profile
# (LTO, LFP) has a narrow peak on a nearly flat background: a wide sigma bound
# lets a component inflate until it IS the background, so the bound is tight
# and the baseline is linear. A broad layered-oxide profile has overlapping
# features on a curved background, and needs the opposite.
# --- constraints that make the decomposition identifiable -----------------
#
# A pseudo-Voigt has four free parameters and they are strongly correlated:
# amplitude and sigma trade off almost exactly at fixed area, and the mixing
# fraction trades against sigma in the wings. Add a shoulder 20-40 mV from a
# parent whose FWHM is 60 mV and eight correlated parameters are being asked
# of a region that can support perhaps three. The optimiser always finds a
# lower residual — that is what free parameters do — and the standard errors
# it returns are CONDITIONAL ON THE MODEL BEING RIGHT, so they do not widen
# to warn you. R2 cannot see this; area retention can, and did: LTO cell A
# reported 136% area retention on a cell holding 99% of its capacity.
#
# The remedy is the crystallographic one. Nobody refines an independent
# profile width for every reflection in a Rietveld fit; widths are tied to a
# few physically motivated parameters and released only when the data demand
# it. Constrain, don't fix, and don't free.
#
# 1. A SHOULDER'S WIDTH IS TIED TO ITS PARENT'S by a bounded ratio, with one
#    ratio shared by every shoulder in the half-cycle rather than one each.
#    A shoulder is a distinct redox process on the same electrode: it has no
#    reason to be wider than its parent and every reason to be comparable.
#    1.8.7 tied them rigidly; a bounded ratio is the weaker, better
#    constraint.
COUPLE_SHOULDER_SIGMA = True
SHOULDER_SIGMA_RATIO_MIN = 0.30
SHOULDER_SIGMA_RATIO_MAX = 1.00
SHOULDER_SIGMA_RATIO_INIT = 0.70
#
# 2. ONE MIXING FRACTION PER DATASET, not one per component and not one per
#    half-cycle. It is the least identifiable parameter in the set and the
#    one that buys least: the Gaussian/Lorentzian balance is a property of
#    the MEASUREMENT — the cycler's voltage resolution, the rebin width, the
#    smoothing — not of the redox process, and none of those change from
#    cycle to cycle.
#
#    This is not a stylistic preference. On LTO cell A, one peak per
#    half-cycle, with eta refined per cycle: sigma rose 6.9 -> 8.7 mV and
#    eta wandered 0.65 -> 0.36 -> 0.52 across ten cycles, and the fitted
#    analytic area rose from 61 to 85 — 37% growth — on a cell that had lost
#    1.7% of its capacity. R2 was 0.995-0.998 throughout. There is no
#    shoulder in that fit to blame; the free shape parameter alone did it.
#    Every cycle's fit was excellent and the trend they formed was fiction.
#
#    So eta is MEASURED ONCE on a sample of the dataset's own half-cycles
#    (`calibrate_shape`) and then held fixed for the run. The same discipline
#    applies to the shoulder ratio k. Set these True to refine per half-cycle
#    and reproduce the behaviour above.
SHARED_FRACTION = True
REFINE_FRACTION_PER_HALF_CYCLE = False
REFINE_SIGMA_RATIO_PER_HALF_CYCLE = False
# How many half-cycles the calibration fits. Evenly spaced through the run,
# so a shape that genuinely changes with age shows up as a wide spread rather
# than being read off the first cycle.
CALIBRATION_SAMPLE = 8
# ...but only if the sample AGREES. A 10-90 spread wider than this means the
# shape genuinely varies through the run, and pinning it to the median would
# be asserting a constant the data has just denied. Above it, `calibrate_shape`
# declines and every half-cycle refines its own — with the spread printed, so
# the decision is visible rather than silent.
FRACTION_SPREAD_MAX = 0.25
SIGMA_RATIO_SPREAD_MAX = 0.30
# ...and above THIS, the shape is not merely variable but undetermined. eta is
# bounded [0, 1], so a 10-90 spread approaching 1 means the sampled fits
# landed everywhere the parameter is allowed to go — the curve does not
# constrain it. Reported as a different verdict, because "varies" and "was
# never measured" are different things to tell a reader about an area.
FRACTION_UNDETERMINED = 0.90

# ONE ASYMMETRY FOR THE HALF-CYCLE, NOT ONE PER COMPONENT.
#
# `asym_rect_pseudo_voigt` gives each component an independent width either
# side of its centre, which on its own doubles the shape parameters of every
# component — and NNM fits five or six of them per half-cycle. That is the
# objection to a split lineshape and it is a good one.
#
# It is also unnecessary. The asymmetry measured on LTO is a property of the
# HALF-CYCLE, not of the individual peak: the sign is the same on 82% of
# charge and 100% of discharge half-cycles, broad leading flank and sharp
# trailing flank in both directions. So the ratio sigma_r / sigma is ONE
# shared parameter, tied by `expr` exactly as the mixing fraction `eta` and
# the shoulder ratio `sh_k` are. A half-cycle with six components pays one
# parameter for the asymmetry, not six.
#
# Bounds: the measured HWHM ratios are 0.66 (charge) and 2.49 (discharge), so
# [0.25, 4.0] contains them with room and still forbids a component that is
# a peak on one side and a baseline on the other. Init 1.0 — the symmetric
# shape, which is interior to the box, not on it. See _SEED_BOUND_MARGIN.
# Bounds measured, not guessed. The raw half-widths of the LTO curve either
# side of its own maximum give a high/low ratio of 2.0 on charge and 0.5 on
# discharge; the FITTED ratio, which sees the whole flank rather than one
# slice at half maximum, wants 7-10 and 0.10-0.15. Swept: at [0.25, 4] the
# ratio is pinned on 18 of 20 LTO half-cycles, at [0.10, 10] on 3 of 20, and
# at [0.02, 50] on none, with R2 and the fitted ratio identical between the
# last two. So the box below contains the answer with room, and a ratio that
# reaches it is a fault to report rather than a bound doing the fitting.
ASYMMETRY = False
# How close to a bound counts as on it, as a fraction of the bound.
_BOUND_PROXIMITY = 0.01
ASYM_RATIO_MIN = 0.05
ASYM_RATIO_MAX = 20.0
ASYM_RATIO_INIT = 1.00
# Per-component widths instead of one shared ratio. Costs n parameters rather
# than 1 and is not recommended; kept because the shared ratio is an assertion
# about the half-cycle that a dataset could contradict.
# ONE RATIO PER COMPONENT, NOT ONE PER HALF-CYCLE. Flipped in 1.9.0.58.
#
# The argument above for sharing is sound and its evidence does not reach.
# "The asymmetry is a property of the HALF-CYCLE, not of the individual peak"
# was concluded on LTO — which fits exactly ONE component per half-cycle. A
# dataset with one peak cannot distinguish a property of the half-cycle from
# a property of the peak, so that measurement was silent on the question it
# was taken to settle. It is the same fault as the shoulder gate's threshold
# in 1.9.0.57: the sample was chosen where the distinction cannot appear.
#
# On a chemistry that fits five to seven components the two hypotheses come
# apart immediately, because the components are not alike. NNM discharge
# carries a near-symmetric redox peak at 3.54 V and, 1.4 V below it, a
# terminal feature whose flanks differ by a factor of five. One shared ratio
# has to compromise between them, and compromising is what it did — which is
# why asymmetry measured as "a coin toss" on the broad chemistries and was
# left off.
#
# Measured on 278 half-cycles (NNM x3, NMC x2, LTO x3), same data, same
# detection, four models, scored on BIC so the extra parameters are paid for:
#
#     model                              LTO        NMC        NNM
#     A  current, symmetric            2774.599    718.98    1126.26
#     B  symmetric, sigma_max 0.5      2774.600    863.16    1119.03
#     C  shared asymmetry              2774.599    570.84    1051.29
#     D  per-component asymmetry       2774.596    380.83     782.20
#
#     D beats A on   LTO 39/60 (dBIC -0.0)   NMC 62/68 (-251)   NNM 140/150 (-326)
#     D beats C on   LTO 39/60 (dBIC -0.0)   NMC 59/68 (-158)   NNM 136/150 (-213)
#
# D costs six more free parameters on NNM (nvarys 20 -> 26) and wins anyway,
# on 93% and 91% of half-cycles. LTO is unchanged to three decimal places in
# BIC, as it must be: with one component, shared and free are the same model.
#
# WHAT IT DOES TO THE TERMINAL COMPONENT, which is why this was opened at
# all. NNM components centred within 300 mV of a window edge:
#
#     sigma 195.4 -> 138.5 mV   (the ceiling is 200)
#     fraction of the modelled area falling OUTSIDE the window  0.264 -> 0.129
#
# The 400 mV width was never a statement about the feature. NNM's terminal
# peak is resolved — it has a visible maximum 25-40 mV inside the window and
# both flanks present — but its low flank is ~25 mV and its high flank ~120.
# Fitted symmetrically, the only way to cover the high flank is a sigma large
# enough to spill a third of the component out through the low edge. The
# ceiling was holding that spill down, not describing a peak.
#
# WHICH IS ALSO WHY RAISING THE CEILING IS THE WRONG MOVE, and model B is in
# the table to record it: at sigma_max 0.5 the component widens further, the
# outside-window share rises (NMC 0.015 -> 0.087), and BIC gets WORSE on NMC
# (+114, better on only 19 of 68 half-cycles). The obvious fix was measured
# and is a regression.
ASYMMETRY_SHARED = True
# ASYMMETRY FOR THE COMPONENTS AT THE ENDS OF THE WINDOW, AND ONLY THOSE.
#
# WHAT THE TERMINAL COMPONENT ACTUALLY IS. It was reported as a polarisation
# tail running off the edge of the data. It is not. Read off the raw curve,
# NNM's bottom-of-discharge feature has a resolved maximum 25-40 mV INSIDE
# the window with both flanks present — |dQ/dV| climbs from 34 at the cutoff
# to 50 at 2.085 V and then falls away — and its flanks are wildly unequal:
# about 25 mV on the low side against about 120 mV on the high side. NMC's
# is the opposite case: a monotone ramp into the cutoff with no maximum at
# all, and no detected primary within 300 mV of either edge in 54 half-cycles.
#
# Fitted SYMMETRICALLY, the only way to cover a 120 mV high flank is a sigma
# large enough to spill a third of the component out through the low edge.
# That is where the 400 mV FWHM and the 26-40% extrapolated area came from:
# not from a wide feature, but from a narrow asymmetric one being described
# by a shape that cannot be narrow on one side.
#
# WHY RAISING THE CEILING IS THE WRONG MOVE, measured so it stays refuted: at
# sigma_max 0.5 the component widens further, the outside-window share rises
# (NMC 0.015 -> 0.087), and BIC gets WORSE on NMC (+114, better on only 19 of
# 68 half-cycles). The obvious fix is a regression.
#
# WHY NOT ASYMMETRY EVERYWHERE. Per-component asymmetry on every component
# wins decisively on a SINGLE fit — 278 half-cycles, scored on BIC:
#
#     model                            LTO        NMC        NNM
#     current, symmetric             2774.599    718.98    1126.26
#     symmetric, sigma_max 0.5       2774.600    863.16    1119.03
#     shared asymmetry               2774.599    570.84    1051.29
#     per-component asymmetry        2774.596    380.83     782.20
#
# and is a serious regression in the PIPELINE, which is not the same test:
#
#     NNM cell A   R2 0.9886 -> 0.9011   coherence pairs 504 -> 326
#     NNM cell C   R2 0.9876 -> 0.9724   the 3.15 V shoulder pair resolved in
#                                        119/122 discharge cycles -> 67/122
#     NNM cell B   extrapolated area share 24.4% -> 85.4%
#     NMC          R2 0.9908 -> 0.9991   coherence pairs 53 -> 37, 58 -> 32
#
# The single-fit test is fair to the FITTER and blind to the LOOP around it.
# In the pipeline an asymmetric flank competes with the band machinery for
# the same residual: a flank that absorbs the continuum is a band that never
# gets seeded, and on NMC — which has a real solid-solution continuum — half
# the components ended with a flank at the width ceiling, carrying 51-58% of
# the area, spread across 3.2-4.4 V rather than at the edges. R2 rose and the
# decomposition got worse. That is the cubic baseline's failure exactly, and
# BIC did not catch it because BIC charges per parameter and asymmetry buys a
# very flexible shape for one.
#
# So the freedom goes ONLY where the diagnosis says it belongs: a component
# seeded within `sigma_max` of a window edge — one whose model necessarily
# extends past the measured range — fits its two flanks separately. Every
# other component stays exactly symmetric, tied by `expr`, costing nothing
# and unable to impersonate a continuum. The threshold is not a tuned number:
# it is the width the model is already allowed.
# AND EDGE-ONLY ASYMMETRY WAS MEASURED THROUGH THE PIPELINE TOO, AND LOST.
# It does what the diagnosis says it should — where a terminal component was
# pinned at the ceiling it comes off it, and the extrapolated area falls:
#
#     cell   R2                 terminal FWHM      extrapolated area
#     A      0.9886 -> 0.9787   200 -> 249 mV      6.7% -> 5.8%
#     B      0.9883 -> 0.9904   399 -> 293 mV     24.4% -> 24.3%
#     C      0.9876 -> 0.9896   400 -> 269 mV     11.4% ->  9.2%
#
# and the coherence pairs fall in all three cells (504 -> 477, 311 -> 306,
# 568 -> 553), the component count falls on cell A (6.38 -> 5.88), and cell A
# — the cell whose terminal components were ALREADY mostly inside the window,
# 28% at the ceiling against 69% and 78% — gets worse on every measure. A
# change that helps the cells that were wrong and harms the one that was
# right is not a model improvement; it is a change of where the error sits.
# Two of three cells better is not a result. NMC is a wash on the same test
# (cell A 0.9908 -> 0.9937 and pairs 53 -> 55; cell B unchanged and pairs
# 58 -> 56), which is what the diagnosis predicts: NMC's terminal region has
# no resolved peak for an asymmetric lineshape to describe.
#
# DEFAULT OFF, TURNED ON BY `broad` ALONE. This started as True — the policy —
# and LTO's R2 fell from 0.9966 to 0.9562 on all three cells in the first
# pipeline run, because `sharp` does not pass the field and so inherited it
# from here. LTO's single peak sits 320 mV from the nearest edge against a
# 30 mV sigma_max, so "edge only" made it symmetric and threw away the
# asymmetry that class was measured to need. A default that silently reaches a
# profile which never asked for it is the same fault as a global gate; the
# profile chooses the model, so the profile says so explicitly.
ASYMMETRY_EDGE_ONLY = False

# How far inside its own bounds a seeded parameter is held. See the note at
# the seeding site: a start exactly on a bound can stall the optimiser.
_SEED_BOUND_MARGIN = 0.25

# --- the bound stall (item 39) -------------------------------------------
# `_SEED_BOUND_MARGIN` above solves this problem for the START of a fit. These
# solve it for the END: a parameter that WALKS onto a bound during refinement
# has no outward gradient, so the minimiser declares convergence at a wall.
# See the long note at the restart site in `fit_one` for the measurements.
#
#   PROXIMITY  how close to a bound counts as on it, as a fraction of the
#              parameter's own range. 1% of the box, so it scales with the
#              parameter rather than assuming millivolts.
#   STEP       how far inside to restart. The same 25% as the seed margin —
#              far enough that the first step is not back onto the wall,
#              near enough that a genuinely binding bound is found again.
#   GAIN       how much better the restart must be before its answer is
#              taken, as a fraction of chi-square. A restart that lands on
#              the same minimum must not churn the result, so the bar is a
#              real improvement rather than a rounding one.
BOUND_ESCAPE_RESTART = True
BOUND_ESCAPE_PROXIMITY = 0.01
BOUND_ESCAPE_STEP = 0.25
BOUND_ESCAPE_GAIN = 1e-4

# HOW HARD TO ASK. scipy's `leastsq` defaults to ftol = xtol = 1.5e-8: stop
# when a step changes the sum of squares, or the solution vector, by less than
# one part in 70 million. That is a demand for eight significant figures from a
# dQ/dV curve that carries three, and it is the whole of items 9 and 13 on the
# master list — the optimiser was not failing, it was being asked a question
# the data cannot answer, and it kept stepping until it hit the 20,000
# evaluation ceiling and was then recorded as "not converged".
#
# Measured on 16 half-cycles of NNM cell C and 16 of NMC111 cell A, one-shot:
#
#   tolerance    NNM total nfev / capped / median R2    NMC total / capped / R2
#   1.5e-8            20,000 / 11 / 0.9802                  20,000 / 11 / 0.9764
#   1e-5               1,498 /  3 / 0.9802                  20,000 /  9 / 0.9764
#   1e-4                 274 /  0 / 0.9786                   2,519 /  5 / 0.9760
#
# 1e-4 is the choice: the WORST fit is unchanged on both chemistries
# (0.9682 and 0.9340), nothing runs to the ceiling on NNM, and it costs 0.0016
# of median R2 on NNM — an order of magnitude inside the closure interval that
# same dataset reports. LTO is indifferent: identical R2 to four decimal places
# at every tolerance tested, because a 6-parameter fit on a sharp peak
# converges long before the tolerance is the binding constraint.
FIT_TOLERANCE = 1.0e-4

# RETIRED, and the reason is worth keeping.
#
# The idea was Nik's and it is a good one: in a Rietveld refinement you get the
# positions right with the shape held, then release the shape. Implemented
# here it made things worse, and the measurements say why. Stage 1 was given
# the FULL evaluation budget at the 1.5e-8 tolerance above, so it did not
# "hold the shape for a moment" — it solved the WRONG SHAPE EXACTLY, driving
# the fit into a deep minimum of a model with the seeded widths frozen in.
# Releasing the widths there finds a gradient near zero, and stage 2 stops in
# 24-33 evaluations. The signature is a see-saw visible on both chemistries:
# wherever stage 1 ran long, stage 2 stopped at once.
#
# Tested against one-shot at four stage-1 budgets and two tolerances. Staging
# lost every time:
#
#              NNM median R2 / worst      NMC median R2 / worst
#   one-shot      0.9786 / 0.9682            0.9760 / 0.9340
#   stage1 100    0.9777 / 0.9681            0.9504 / 0.8780
#   stage1 300    0.9778 / 0.9681            0.9464 / 0.8783
#   stage1 1000   0.9759 / 0.9681            0.9506 / 0.8827
#
# On NMC it costs 0.025 of R2 — and it is FASTER while doing it, because
# stalling early is cheap. That is the shape of the trap.
#
# It is a flag, not a deletion: the code path is intact and the manifest
# records the choice, so re-testing it after the seeding or the bounds change
# again costs one edit.
#
# What the staging work DID produce, and what is kept: detection-seeded
# starting values and measured shoulder widths. Those were bundled with it in
# 1.9.0.40's "convergence 13% -> 59%" and they are the half of that result
# which survives.
STAGED_REFINEMENT = False


# Below this, `width` is not a band: the component is reported as a peak and
# the analytic w = 0 branch is used. 1 mV is below the cyclers' own voltage
# resolution, so a narrower band is not a claim the data can support.
_BAND_WIDTH_NEGLIGIBLE = 0.001

# How close to its ceiling a band width counts as pinned. Mirrors
# `analyse.SIGMA_BOUND_PROXIMITY`, and means the same thing: a parameter that
# has run to its limit is not a measurement of anything.
SIGMA_BOUND_PROXIMITY_COMPLEMENT = 0.95


# ---------------------------------------------------------------------------
# One component that can be a PEAK or a BAND, and lets the data decide
# ---------------------------------------------------------------------------
# A solid solution does not deliver its charge at one potential. It delivers it
# across a WINDOW of potentials, as the composition moves — which is why a
# layered-oxide dQ/dV is a broad flat-topped envelope rather than a peak, and
# why fitting it with a sum of pseudo-Voigts goes wrong. Measured on NMC111
# cell A cycle 6 discharge, seven pseudo-Voigts reach R2 0.999 with component
# areas summing to 3.65x the curve's own area: not a decomposition, a massive
# cancellation against a baseline driven negative to accommodate it.
#
# It is the same shape, and the same reason, as a carboxylic acid O-H band in
# the infrared: hydrogen bonding gives a distribution of environments, so the
# band is broad and flat-topped instead of sharp.
#
# The right basis function is therefore a rectangle of width `w` convolved
# with the line shape. Both halves of a pseudo-Voigt convolve analytically:
#
#     rect(w) (x) Gaussian(s)    = [erf(a) - erf(b)] / 2w
#     rect(w) (x) Lorentzian(s)  = [atan(a') - atan(b')] / pi.w
#
# so the mixture is closed-form and costs nothing. THE POINT IS THAT IT NESTS:
# at w = 0 this is EXACTLY the pseudo-Voigt the pipeline already fits. There
# is no model selection, no information criterion and no second fit. One extra
# parameter per component, and `w` answers "is this a peak or a band?" by
# refining to zero or not.
#
# Verified numerically before it was written in: unit area across w, sigma and
# eta (the only departures are Lorentzian tails genuinely leaving the window —
# at sigma = 100 mV over a 2 V window the analytic fraction inside is
# (2/pi).atan(10) = 0.9365, which is what came back to four figures); w -> 0
# reproduces the bare pseudo-Voigt to 3e-7 relative; w = 600 mV with
# sigma = 20 mV gives a 534 mV plateau; and the shape is smooth through the
# w = 0 branch, so the optimiser sees no step.
def rect_pseudo_voigt(
    x, amplitude=1.0, center=0.0, width=0.0, sigma=0.01, fraction=0.5
):
    """Unit-area pseudo-Voigt smeared over a rectangle of width `width`."""
    sigma = max(float(sigma), 1e-9)
    w = float(width)
    xc = np.asarray(x, float) - float(center)
    # THE GAUSSIAN'S OWN WIDTH. A pseudo-Voigt is a Lorentzian and a Gaussian
    # of the SAME full width at half maximum, which for the Gaussian means
    # sigma_g = sigma / sqrt(2 ln 2) — that is what lmfit's PseudoVoigtModel
    # uses and what makes `fwhm = 2 * sigma` true.
    #
    # This function used `sigma` directly in the Gaussian until 1.9.0.51, so
    # its w -> 0 limit was NOT the pseudo-Voigt: 15% different in shape at
    # eta = 0 and 17.7% wider at half maximum, while `_derived` went on
    # reporting `fwhm = 2 * sigma`. Every width and height published from a
    # band-model fit — that is, every NNM and NMC component — was computed
    # for a curve of a different width from the one that was drawn, and
    # `evaluate` compounded it by rebuilding narrow components through
    # lmfit's model and wide ones through this one. Both are now the same
    # shape, verified against PseudoVoigtModel to machine precision at w = 0.
    sg = sigma / _SIGMA_G_FACTOR
    if w <= _BAND_WIDTH_NEGLIGIBLE:
        g = np.exp(-(xc**2) / (2 * sg**2)) / (sg * np.sqrt(2 * np.pi))
        l = sigma / np.pi / (xc**2 + sigma**2)
    else:
        a = (xc + w / 2.0) / (sg * np.sqrt(2))
        b = (xc - w / 2.0) / (sg * np.sqrt(2))
        g = (erf(a) - erf(b)) / (2.0 * w)
        l = (np.arctan((xc + w / 2.0) / sigma) - np.arctan((xc - w / 2.0) / sigma)) / (
            np.pi * w
        )
    return float(amplitude) * ((1.0 - float(fraction)) * g + float(fraction) * l)


def _rpv_area1(xc, width, sigma, fraction):
    """`rect_pseudo_voigt` at unit amplitude, centred — the shape alone."""
    sigma = max(float(sigma), 1e-9)
    w = float(width)
    xc = np.asarray(xc, float)
    sg = sigma / _SIGMA_G_FACTOR
    if w <= _BAND_WIDTH_NEGLIGIBLE:
        g = np.exp(-(xc**2) / (2 * sg**2)) / (sg * np.sqrt(2 * np.pi))
        l = sigma / np.pi / (xc**2 + sigma**2)
    else:
        a = (xc + w / 2.0) / (sg * np.sqrt(2))
        b = (xc - w / 2.0) / (sg * np.sqrt(2))
        g = (erf(a) - erf(b)) / (2.0 * w)
        l = (np.arctan((xc + w / 2.0) / sigma) - np.arctan((xc - w / 2.0) / sigma)) / (
            np.pi * w
        )
    return (1.0 - float(fraction)) * g + float(fraction) * l


# THE PEAK IS ASYMMETRIC AND UNTIL 1.9.0.51 THE MODEL COULD NOT BE.
#
# Measured on LTO without fitting anything — the half-widths of the raw curve
# either side of its own maximum: charge high/low HWHM ratio 0.66, discharge
# 2.49, the sign consistent on 82% and 100% of half-cycles. In time order that
# is a broad leading flank and a sharp trailing flank in both directions, which
# is what a distribution of particle sizes or a kinetic tail looks like and is
# not something a symmetric lineshape can absorb. It is not the baseline: the
# ratios are identical at baseline degree 1, 2 and 3, the sign is preserved on
# 42 of 42 half-cycles, and the flexible baselines fit WORSE on BIC.
#
# This is the generalisation of `rect_pseudo_voigt`, not a second model: each
# side of the centre gets its own width, the two halves are scaled to the same
# peak height so the join is continuous, and the whole is renormalised so
# `amplitude` is still the area. With `sigma_r == sigma` it reduces to
# `rect_pseudo_voigt` exactly (verified to machine precision across w, sigma
# and eta), so a symmetric fit is unchanged and the band model still composes
# with it.
#
# One consequence worth stating: `sigma` is now the LEFT half-width at half
# maximum and `sigma_r` the right, so the full width is `sigma + sigma_r` and
# not `2 * sigma`. Everything downstream reads `fwhm`, which is computed from
# both.
def asym_rect_pseudo_voigt(
    x, amplitude=1.0, center=0.0, width=0.0, sigma=0.01, sigma_r=0.01, fraction=0.5
):
    """Unit-area rect-smeared pseudo-Voigt with an independent width each side."""
    sl = max(float(sigma), 1e-9)
    sr = max(float(sigma_r), 1e-9)
    if abs(sr - sl) <= _TINY:
        return rect_pseudo_voigt(
            x,
            amplitude=amplitude,
            center=center,
            width=width,
            sigma=sl,
            fraction=fraction,
        )
    xc = np.asarray(x, float) - float(center)
    hl = float(_rpv_area1(0.0, width, sl, fraction))
    hr = float(_rpv_area1(0.0, width, sr, fraction))
    if not (hl > 0 and hr > 0):
        return np.zeros_like(xc)
    shape = np.where(
        xc < 0.0,
        _rpv_area1(xc, width, sl, fraction) / hl,
        _rpv_area1(xc, width, sr, fraction) / hr,
    )
    # Each side contributes half of its own unit-area shape's area, which at
    # unit HEIGHT is 1/h. The composite therefore has area (1/hl + 1/hr)/2.
    return float(amplitude) * shape / (0.5 * (1.0 / hl + 1.0 / hr))


PROFILE_SPECS = {
    # The values are 1.8.7 Module 4's: FIT_CENTRE_TOLERANCE_MV = 70,
    # FIT_SIGMA_MIN_MV = 5, FIT_SIGMA_MAX_MV = 200, BASELINE_DEGREE = 3, with the
    # sharp-profile overrides that cell applies. Read off the source, not recalled.
    #                centre_tol  sigma_min  sigma_max  baseline_degree
    # band_width_max = 0 means "no band, use the bare pseudo-Voigt", and the
    # fit is then bit-identical to what it has always been. A two-phase
    # reaction happens AT a potential, so a sharp profile has no composition
    # window to smear over and giving it one would only add a free parameter
    # that must refine to zero.
    # SHARP sigma_min DROPPED 3.0 -> 0.5 mV IN 1.9.0.51.
    #
    # FWHM = 2*sigma for this model, so 3 mV forced every fitted peak to be at
    # least 6 mV wide — against LTO peaks measured at 2.6-4.7 mV on cell C.
    # The model could not be as narrow as the material, so it conserved area
    # by being wide and SHORT: measured with `fit_quality`, the fitted peak
    # reached only 0.68 of the data's height on discharge, with a residual of
    # 40% of the amplitude. R2 0.78 was the visible symptom and it read as
    # "a bit noisy"; the fit was missing the peak height by a third.
    #
    # 0.5 mV is not a guess. A peak cannot be narrower than the histogram that
    # sampled it, and the sharp profile bins at 1 mV
    # (`signal.HISTOGRAM_BIN_BY_PROFILE`), so sigma >= bin/2 means FWHM >= one
    # bin — the model may be exactly as narrow as the measurement resolves and
    # no narrower. Swept beforehand: at 0.5 mV median R2 went 0.9664 -> 0.9791
    # and NOTHING ran to the new floor, so there is no collapse risk; the
    # floor stops being a constraint rather than becoming a looser one.
    #
    # Moderate and broad are unchanged: no moderate dataset exists to measure
    # on, and on broad the 5 mV floor is what stands between the fit and
    # needle components.
    # ASYMMETRY IS ON FOR SHARP AND OFF FOR THE OTHER TWO, because that is
    # what three chemistries say. Same seeds, same bounds, one extra shared
    # parameter, judged on BIC so the parameter has to pay for itself:
    #
    #                    R2 sym -> asym   max residual    BIC better
    #   LTO   (sharp)    0.826 -> 0.958     27% -> 18%      20 of 20
    #                    0.909 -> 0.994     24% ->  8%
    #   NNM   (broad)    0.984 -> 0.984     12% ->  8%      14 of 20
    #   NMC   (broad)    0.994 -> 0.998     16% ->  7%       8 of 18
    #                    0.986 -> 0.971     11% -> 13%
    #
    # On LTO it is not close: every half-cycle, both cells, both directions,
    # median dBIC -1424 (charge) and -2546 (discharge). It also beats the
    # honest alternative — one more symmetric COMPONENT, which costs three
    # parameters instead of one — on 51 of 64 half-cycles, and on charge the
    # extra component is worse than not adding it at all (dBIC +19.5). So the
    # asymmetry is a lineshape, not an unresolved peak.
    #
    # On the two broad chemistries it is a coin toss that costs 1.2-4.8x the
    # fitting time, and on NMC discharge it is worse. A parameter that does
    # not pay for itself does not go in. This is the same rule the rest of
    # the module follows: the profile chooses the model.
    # BASELINE REMOVED IN 1.9.0.56 — `baseline_degree=None` on every class.
    # The degrees below (1, 2, 2) were each measured and each right for the
    # question being asked at the time, which was "which polynomial fits the
    # background best". That question had no answer because there is no
    # background: see BAND_FROM_RESIDUAL for the measurement, and for what the
    # polynomial was doing instead. `quality.DEFAULT_DEGREES` no longer has a
    # sweep to make; `unattributed_fraction` is the quantity that replaces it.
    "sharp": dict(
        centre_tol=0.030,
        sigma_min=0.0005,
        sigma_max=0.030,
        asymmetry=True,
        baseline_degree=None,
        band_width_max=0.0,
    ),
    # MODERATE DROPPED 3 -> 2 IN 1.9.0.36. See the note below the table.
    # Moderate: OFF until measured, on the same principle as the baseline
    # degree above it. Whoever brings a moderate dataset should try it.
    "moderate": dict(
        centre_tol=0.070,
        sigma_min=0.005,
        sigma_max=0.100,
        asymmetry=False,
        baseline_degree=None,
        band_width_max=0.0,
    ),
    # BROAD DROPPED 3 -> 2 IN 1.9.0.28. The cubic was 1.8.7's and it is the
    # single cause of the fitting not converging.
    #
    # Measured on NNM cell C, 30 cycles, same peaks, same starts, only the
    # baseline degree changed:
    #
    #     degree   converged   median nfev   median R2
    #       1         80%          368         0.9677
    #       2         72%          412         0.9678
    #       3 (old)   40%       20,000         0.9716
    #
    # The cubic buys 0.004 of R2 and costs 54x the evaluations and half the
    # convergence. That is §A6's argument arriving as a measurement: adding
    # free parameters always lowers the residual, and R2 rose while the answer
    # got worse — non-converged fits, an undetermined peak/background split,
    # and 126 cycles with their capacity attribution withheld.
    #
    # WHY A CUBIC SPECIFICALLY. The original reasoning was that a broad peak
    # sits on a curved background a straight line cannot follow, and that is
    # still right — which is why this is 2 and not 1. But a cubic can carry an
    # INFLECTION, and an inflection is exactly what lets a baseline impersonate
    # a peak. That is the degenerate direction the optimiser then crawls along
    # for 20,000 evaluations without finding a minimum, because there is no
    # unique one to find. A quadratic curves without being able to do that.
    #
    # Convergence by peak count makes the same point: at 5 peaks a cubic
    # converged on 8% of half-cycles against 50% for a linear baseline. The
    # more components there are to trade with, the worse the cubic is.
    #
    # `quality.DEFAULT_DEGREES` still probes (1, 2, 3), so the closure interval
    # measures the same sensitivity as before and this change does not hide it.
    # NO CLASS TURNS THE BAND ON. The band belongs to a MECHANISM — a solid
    # solution delivering charge across a composition window — and the shape
    # class cannot see mechanisms: NNM and NMC111 are both "broad", and one
    # of them is a series of real transitions. Giving every broad dataset
    # bands cost NNM its determined split (interval 0.052 -> 0.140 and
    # 0.101 -> 0.179, reliable components more than halved).
    #
    # `quality.classify_mechanism` decides, before the fitting, and the
    # caller passes `band_width_max` and `primaries_only` from its answer.
    # 0.5 V is the ceiling it passes when it does turn them on: a generous
    # composition window, still well inside the analysis range, bounding the
    # parameter without deciding the answer.
    # ASYMMETRY ON IN 1.9.0.58. It was off on a measurement taken while the
    # cubic baseline still existed and while one ratio was shared across
    # every component; both premises are gone. See ASYMMETRY_SHARED for the
    # four-model comparison that reopened it. `moderate` stays off: there is
    # still no moderate dataset, and this is not evidence about one.
    # ASYMMETRY STAYS OFF FOR `broad`, MEASURED AGAIN IN 1.9.0.58 AND STILL OFF.
    # `asymmetry_edge_only` is set here so that turning `asymmetry` on is a
    # one-word change for whoever revisits this — the machinery is built and
    # tested, the verdict is what failed. Three candidate fixes for the
    # terminal component were measured through the whole pipeline and all
    # three are refuted; see ASYMMETRY_EDGE_ONLY for the numbers, including
    # this one, which was the closest and still lost a cell.
    "broad": dict(
        centre_tol=0.070,
        sigma_min=0.005,
        sigma_max=0.200,
        asymmetry=False,
        asymmetry_edge_only=True,
        baseline_degree=None,
        band_width_max=0.0,
    ),
}
# MODERATE, MEASURED IN 1.9.0.36. There is still no real moderate dataset in
# hand, so this was measured on SYNTHETIC half-cycles from `synthetic.py`
# ("mixed", sigma 28-40 mV over a 2.2 V window, which is squarely inside the
# moderate band), 30 seeds per condition. That is weaker evidence than the
# NNM measurement above and is labelled as such — but the effect is not
# subtle, and the condition that exposes it is the realistic one.
#
# Fitted from the TRUE centres with no noise, all three degrees converge 30/30
# and this test says nothing. That is why the cubic survived so long. Give the
# fit what detection actually hands it — centres up to 20 mV off, one spurious
# extra seed, and real noise — and the degrees separate:
#
#   noise   deg   converged   median nfev   median R2
#    0.01    1       30/30         351        0.9974
#    0.01    2       30/30         387        0.9975
#    0.01    3       24/30         828        0.9974
#    0.05    1       29/30         425        0.9543
#    0.05    2       29/30         611        0.9544
#    0.05    3       13/30      20,000        0.9518
#    0.10    1       29/30         442        0.8589
#    0.10    2       27/30       1,288        0.8586
#    0.10    3       15/30      19,844        0.8589
#
# At 5% noise the cubic converges on 43% of half-cycles against 97% for the
# quadratic, sits at the 20,000-evaluation ceiling at the MEDIAN, and fits
# very slightly WORSE. It buys nothing and costs everything the broad
# measurement said it would — the same degeneracy, the same inflection, the
# same crawl along a direction with no minimum in it.
#
# So moderate inherits the answer after all. It was right not to assume it:
# the assumption would have been correct and the reason for believing it
# would not have been. Anyone who brings a real moderate dataset should still
# run the comparison on it — this is a synthetic result, and the honest thing
# is for the next measurement to be able to overturn it.


def shoulder_parents(centres, is_shoulder):
    """
    Assign each shoulder to its NEAREST PRIMARY peak in voltage.

    Returns a tuple the length of `centres`: the parent's index for a
    shoulder, None for a primary. If nothing was flagged primary the whole
    tuple is None — a half-cycle of shoulders has no parent to tie to, and
    tying them to each other would be arbitrary.
    """
    c = np.asarray(centres, float)
    sh = np.asarray(is_shoulder, bool)
    if sh.size != c.size:
        return (None,) * c.size
    primaries = np.flatnonzero(~sh)
    if primaries.size == 0:
        return (None,) * c.size
    out = []
    for i in range(c.size):
        if not sh[i]:
            out.append(None)
        else:
            out.append(int(primaries[np.argmin(np.abs(c[primaries] - c[i]))]))
    return tuple(out)


def spec_for_profile(
    peaks,
    profile,
    *,
    fraction=None,
    sigma_ratio=None,
    primaries_only=False,
    **overrides,
):
    """
    A `FitSpec` matching 1.8.7's profile-dependent settings.

    `peaks` is either the detected-peaks DataFrame (columns `voltage` and,
    optionally, `is_shoulder`) or a bare sequence of centres. Passing the
    DataFrame is preferred: it is the only way `shoulder_of` gets filled in,
    and without that the sigma coupling that makes the decomposition
    identifiable cannot be applied. A bare sequence still works, and simply
    fits every component free — which is what 1.9.0.8 did.

    `profile` is `signal.classify_dqdv_profile`'s dict, or its class string.
    Anything passed as a keyword overrides the profile's value.

    Use this rather than a bare `FitSpec` wherever the profile is known. The
    first 1.9.0 run on a real cell used the bare defaults and fitted a broad
    P3 dQ/dV with a LINEAR baseline, which is what a visibly bad fit looks
    like: R2 0.27 where a cubic gives 0.54.
    """
    shoulder_of = None
    sigma0 = amp0 = None
    # THE BAND MODEL REPLACES COMPONENTS, IT DOES NOT JOIN THEM.
    #
    # A sum of Gaussians can only build a flat top by stacking overlapping
    # components, and the flanks of that stack are what the shoulder pass
    # finds. On a solid-solution envelope those shoulders are therefore
    # ARTEFACTS OF THE WRONG SHAPE, not features. Give a component a flat top
    # of its own and they should not be separate components at all.
    #
    # Measured on NMC111 cell A cycle 6 discharge — 7 detected components, 5
    # of them shoulders, so 2 primaries:
    #
    #     components   model          closure       interval   R2
    #     all 7        pseudo-Voigt   0.17-0.51       0.34     0.946
    #     all 7        bands          0.22-1.00       0.78     0.920
    #     2 primaries  pseudo-Voigt   0.22-0.50       0.28     0.952
    #     2 primaries  bands          0.34-0.47       0.12     0.978
    #
    # Bands ON TOP OF the full list are WORSE than what we had: seven extra
    # free widths and no reduction in anything. Bands INSTEAD OF the stack
    # bring the interval to 0.12 — under the 0.15 at which the split counts
    # as determined — with the best R2 of the four. That is the difference
    # between "NMC is not decomposable" and "NMC needed the right model".
    # PRIMARIES ONLY IS NOT A PROPERTY OF "BROAD". Deciding it from the
    # profile class applied it to NNM, where the shoulders are real
    # transitions, and the closure interval went 0.052 -> 0.140 and
    # 0.101 -> 0.179 while reliable components more than halved. The caller
    # passes it, from `quality.classify_mechanism`, which asks what produced
    # the curve rather than what shape it is.
    # A BAND CEILING FROM THE MECHANISM IS A PERMISSION, NOT AN INSTRUCTION.
    # `quality.classify_mechanism` says whether this curve's model may contain
    # bands. It does not say that every detected maximum is one -- a detected
    # maximum is a peak, and a band belongs to a region found from the charge
    # the peaks cannot account for. So the value is moved onto `band_ceiling`
    # and the components start as peaks. See BAND_FROM_RESIDUAL.
    # `band_ceiling` HOLDS A FRACTION OF THE HALF-CYCLE'S SPAN, resolved to
    # volts by `band_ceiling_volts` at fit time, where the voltage array is
    # in hand. A FitSpec is built before any of that is known — see
    # `quality.MECHANISM_MODEL` for why a flat volts ceiling was wrong.
    _ceiling = float(overrides.pop("band_ceiling_span_fraction", 0.0) or 0.0)
    if _ceiling > 0:
        overrides["band_ceiling"] = _ceiling
        overrides["band_width_max"] = 0.0

    if (
        primaries_only
        and hasattr(peaks, "columns")
        and "is_shoulder" in getattr(peaks, "columns", [])
    ):
        _prim = peaks[~peaks["is_shoulder"].fillna(False)]
        if len(_prim) >= 1:
            peaks = _prim.reset_index(drop=True)
    if hasattr(peaks, "columns"):
        centres = np.asarray(peaks["voltage"].values, float)
        if "is_shoulder" in peaks.columns:
            shoulder_of = shoulder_parents(
                centres, peaks["is_shoulder"].fillna(False).values
            )

        # START THE FIT WHERE DETECTION ALREADY IS.
        #
        # Detection measures a height and a width for every peak it finds and
        # nothing used them. Every component of every half-cycle therefore
        # started from the same two numbers — sigma = min(0.02, sigma_max/2),
        # amplitude = 5% of the curve's maximum — so on a broad profile every
        # peak began 20 mV wide and fitted to a median of 85 mV, and the
        # tallest and the smallest began identical. The optimiser's first few
        # thousand evaluations were spent doing nothing but growing the
        # components to the size detection had already measured.
        #
        # Measured on 14 NMC111 discharge half-cycles, same peak list, only
        # the starting values changed:
        #
        #     generic start   2 of 14 converged   median 20,000 nfev   87 s
        #     seeded          11 of 14 converged  median    170 nfev   24 s
        #
        # A hundredfold fewer evaluations. FWHM = 2*sigma exactly for this
        # model, and the area of a peak is about height x FWHM; neither has
        # to be accurate, only close enough that the optimiser starts inside
        # the right basin instead of walking to it.
        #
        # WARM-STARTING FROM THE PREVIOUS CYCLE WAS TRIED AND IS WORSE: 0 of
        # 14 converged. A fitted shape can be degenerate — a component sitting
        # at the sigma bound, acting as baseline — and carrying it forward
        # propagates that into every later cycle, where re-reading detection
        # each time is self-correcting. At 170 evaluations there is nothing
        # left for a warm start to buy anyway.
        def _num(col):
            # numpy only: `fitting` does not import pandas, and the
            # module-isolation build guard rightly refused the version of
            # this that did.
            try:
                return np.asarray(peaks[col].values, dtype=float)
            except KeyError, TypeError, ValueError:
                return None

        _w = _num("width_V") if "width_V" in peaks.columns else None
        if _w is not None:
            sigma0 = np.where(np.isfinite(_w) & (_w > 0), _w / 2.0, np.nan)
            # A SHOULDER HAS NO MEASURED WIDTH — it is not a local maximum, so
            # `peak_widths` never measured one and detection leaves it at
            # zero. On a broad layered oxide most reference peaks ARE
            # shoulders (five of seven on NMC111 cell A discharge), so
            # without this the seeding fired on almost nothing and every
            # shoulder fell back to the generic 20 mV that caused the problem.
            # The half-cycle's own measured widths are the best available
            # statement of how wide a feature on THIS curve is, so an
            # unmeasured component starts at their median.
            _ok = np.isfinite(sigma0)
            if _ok.any() and not _ok.all():
                sigma0 = np.where(_ok, sigma0, float(np.median(sigma0[_ok])))
            _h = _num("height") if "height" in peaks.columns else None
            if _h is not None:
                amp0 = np.where(
                    np.isfinite(_h) & np.isfinite(sigma0),
                    np.abs(_h) * 2.0 * sigma0,
                    np.nan,
                )
    else:
        centres = np.asarray(peaks, float)

    cls = profile.get("class", "broad") if isinstance(profile, dict) else profile
    kw = dict(PROFILE_SPECS.get(str(cls), PROFILE_SPECS["broad"]))
    if shoulder_of is not None:
        kw["shoulder_of"] = shoulder_of
    if fraction is not None:
        kw["fraction"] = fraction
    if sigma_ratio is not None:
        kw["sigma_ratio"] = sigma_ratio
    if sigma0 is not None:
        kw.setdefault("sigma0", sigma0)
    if amp0 is not None:
        kw.setdefault("amp0", amp0)
    kw.update(overrides)
    return FitSpec(centres, **kw)


def _resolved_shoulders(spec: FitSpec):
    """
    `{child: root_parent}` for every coupled shoulder in `spec`.

    Chains are collapsed to a primary component (a shoulder of a shoulder is
    tied to the primary at the head of the chain) and self-references,
    out-of-range indices and cycles are dropped rather than raised: a bad
    `shoulder_of` should cost the coupling, not the fit.
    """
    n = len(spec.centres)
    raw = getattr(spec, "shoulder_of", None) or ((None,) * n)
    out = {}
    for i in range(n):
        p = raw[i] if i < len(raw) else None
        seen = {i}
        while p is not None:
            p = int(p)
            if p < 0 or p >= n or p in seen:
                p = None
                break
            seen.add(p)
            nxt = raw[p] if p < len(raw) else None
            if nxt is None:
                break
            p = nxt
        if p is not None:
            out[i] = p
    return out


def calibrate_shape(jobs, *, n=CALIBRATION_SAMPLE, n_jobs=None, verbose=True):
    """
    Measure this dataset's peak SHAPE once, so no cycle has to re-decide it.

    `jobs` is the production list of (voltage, dqdv, spec, key). An evenly
    spaced sample is refitted with the mixing fraction and the shoulder ratio
    FREE, and the median of each is returned:

        {"fraction": eta, "sigma_ratio": k, "fraction_spread": ...,
         "sigma_ratio_spread": ..., "n": ...}

    Pass those back through `spec_for_profile(..., fraction=eta,
    sigma_ratio=k)` and every production fit is one or two parameters lighter,
    with the shape held at what the data said it was rather than at whatever
    each cycle's optimiser preferred.

    The SPREAD is returned as well as the median, and it is the number to
    look at: a wide one means the shape genuinely varies through the run, and
    fixing it is then a modelling decision that should be made deliberately
    rather than a tidy-up.

    THE FIGURE THAT USED TO BE HERE WAS STALE. This said "on the LTO
    triplicate it is 0.1-0.2"; re-measured at 1.9.0.56 it is 0.63-0.72, so
    every LTO run refines eta per half-cycle and says so. Measured with the
    cubic baseline restored it is 0.68-0.79 -- WIDER, not narrower, so this is
    not something removing the baseline caused. eta on a sharp two-phase peak
    is simply not well determined by these data, and the calibration is right
    to decline to assert it.

    What that costs is visible in the capacity accounting. On LTO cell A
    charge the unattributed charge tracks eta at r = -0.935 across ten cycles:
    the higher the Lorentzian fraction, the further the wings reach and the
    more the single component over-covers, from +1.0% of the cell's charge at
    eta = 0.41 to -6.0% at eta = 1.00. Before 1.9.0.56 a polynomial absorbed
    that and nobody could see it.

    Returns None for either quantity the sample could not measure — no
    shoulders in the dataset, or nothing converged.
    """
    if not jobs:
        return dict(
            fraction=None,
            sigma_ratio=None,
            n=0,
            fraction_spread=float("nan"),
            sigma_ratio_spread=float("nan"),
        )
    idx = np.linspace(0, len(jobs) - 1, min(int(n), len(jobs)))
    sample = [jobs[i] for i in sorted(set(idx.round().astype(int).tolist()))]
    # Free BOTH, whatever the module defaults say: this is the measurement
    # those defaults are supposed to be based on.
    free = [
        (v, y, spec.replace(fraction=None, sigma_ratio=None), k)
        for (v, y, spec, k) in sample
    ]
    if verbose:
        print(
            entry(
                "sample",
                f"{len(free)} of {len(jobs)}",
                "half-cycles, refitted with the shape free",
            )
        )
    res = fit_many(free, n_jobs=n_jobs, verbose=False, label="calibration")

    etas, ks = [], []
    for r in res:
        if not r.get("success"):
            continue
        e = r.get("shared_fraction", np.nan)
        if not np.isfinite(e):
            # One component: no shared eta was created, so read the
            # component's own.
            comps = r.get("components", [])
            e = comps[0]["fraction"] if len(comps) == 1 else np.nan
        if np.isfinite(e):
            etas.append(float(e))
        kk = r.get("sigma_ratio_k", np.nan)
        if np.isfinite(kk):
            ks.append(float(kk))

    def _med(xs):
        return (
            (float(np.median(xs)), float(np.percentile(xs, 90) - np.percentile(xs, 10)))
            if xs
            else (None, float("nan"))
        )

    eta, eta_sp = _med(etas)
    k, k_sp = _med(ks)
    eta_ok = eta is not None and eta_sp <= FRACTION_SPREAD_MAX
    k_ok = k is not None and k_sp <= SIGMA_RATIO_SPREAD_MAX
    if verbose:
        if eta is None:
            print(verdict("caution", "mixing fraction not measurable"))
        else:
            _pin = "  (pinned at a bound)" if eta <= 0.02 or eta >= 0.98 else ""
            print(
                entry(
                    "mixing fraction eta",
                    f"{eta:.3f}",
                    f"10-90 spread {eta_sp:.3f} over {len(etas)} half-cycles{_pin}",
                )
            )
            if eta_ok:
                print(verdict("ok", "held fixed for every fit in this dataset"))
            elif eta_sp >= FRACTION_UNDETERMINED:
                # A SPREAD OF ~1.0 IS NOT A WIDE MEASUREMENT, it is no
                # measurement: eta is bounded [0, 1], so a 10-90 range that
                # spans the whole interval says the data does not constrain
                # the peak shape at all. That is a different statement from
                # "it varies", and it used to print the same verdict as a
                # spread of 0.375.
                print(
                    verdict(
                        "bad",
                        f"spread {eta_sp:.2f} spans the whole "
                        f"allowed range — the peak shape is not "
                        f"determined by this data",
                    )
                )
                print(
                    bullet(
                        "Every half-cycle refines its own eta, which is "
                        "the only honest option, but the areas that "
                        "result carry a shape parameter the curve never "
                        "pinned down. Treat area TRENDS from this "
                        "dataset as indicative; peak positions are "
                        "unaffected."
                    )
                )
            else:
                print(
                    verdict(
                        "caution",
                        f"spread {eta_sp:.2f} > "
                        f"{FRACTION_SPREAD_MAX} — refined "
                        f"per half-cycle",
                    )
                )
                print(
                    bullet(
                        "The peak shape is not constant through this "
                        "run, so it is not asserted to be. Area trends "
                        "from this dataset carry that free parameter."
                    )
                )
        if k is not None:
            _pin = "  (pinned at a bound)" if k >= 0.98 else ""
            print(
                entry(
                    "shoulder ratio k",
                    f"{k:.3f}",
                    f"10-90 spread {k_sp:.3f} over {len(ks)} half-cycles{_pin}",
                )
            )
            print(
                verdict("ok", "held fixed")
                if k_ok
                else verdict(
                    "caution",
                    f"spread > {SIGMA_RATIO_SPREAD_MAX} — refined per half-cycle",
                )
            )
    return dict(
        fraction=(eta if eta_ok and not REFINE_FRACTION_PER_HALF_CYCLE else None),
        sigma_ratio=(k if k_ok and not REFINE_SIGMA_RATIO_PER_HALF_CYCLE else None),
        fraction_median=eta,
        sigma_ratio_median=k,
        fraction_spread=eta_sp,
        sigma_ratio_spread=k_sp,
        n=len(res),
    )


def build_model(voltage, dqdv, spec: FitSpec):
    """
    Compose the pseudo-Voigt sum and its baseline. Pure; no fitting.

    Parameters are created from the COMPOSITE model, not merged from the
    components. That is not a style choice: lmfit validates the supplied
    Parameters against the composite's `param_names`, and a Parameters object
    assembled component-by-component fails that check once the derived
    `fwhm`/`height` entries are removed. Building from the composite and then
    deleting works, because the composite is the thing being validated.
    """
    v = np.asarray(voltage, float)
    y = np.abs(np.asarray(dqdv, float))

    _use_band = np.any(np.asarray(_band_ceilings(spec), float) > 0.0)
    _use_asym = bool(getattr(spec, "asymmetry", False))
    model = None
    for i in range(len(spec.centres)):
        if _use_asym:
            # Subsumes the band: `asym_rect_pseudo_voigt` carries `width` too,
            # and `width` is simply held at zero when bands are off.
            m = _LMModel(asym_rect_pseudo_voigt, prefix=f"p{i}_")
        elif _use_band:
            m = _LMModel(rect_pseudo_voigt, prefix=f"p{i}_")
        else:
            m = PseudoVoigtModel(prefix=f"p{i}_")
        model = m if model is None else model + m

    baseline = None
    if spec.baseline_degree is not None:
        baseline = PolynomialModel(degree=spec.baseline_degree, prefix="bg_")
        model = model + baseline

    params = model.make_params()

    amp0 = max(y.max() * 0.05, _TINY)
    sigma0 = min(0.02, max(spec.sigma_max / 2.0, spec.sigma_min))
    # A COMPONENT MAY MOVE, BUT NOT PAST ITS NEIGHBOUR.
    #
    # Every centre used to get the same +/- centre_tol, which on a broad
    # profile is 70 mV. A shoulder sits as little as 20 mV from its parent
    # (`SHOULDER_MIN_SEPARATION_MV`), so the two boxes overlapped almost
    # completely and the pair could simply trade places — a flat direction
    # in the residual that the optimiser walks up and down until it hits the
    # 20,000-evaluation ceiling. It is the same degeneracy the cubic
    # baseline causes, in the centres rather than the background.
    #
    # It was latent while shoulders were being placed in VALLEYS, far from
    # any parent. Rebuilding the picker in 1.9.0.37 put them where they
    # belong — on their parent's flank — and the fitting time on NMC111
    # cell A went from 251 s to 523 s, with nearly every discharge fit
    # running to the ceiling. A better peak list made a worse fit, because
    # the bound was wrong.
    #
    # So each centre is now free within HALF the distance to its nearest
    # neighbouring seed, capped by centre_tol. Two seeds 20 mV apart get
    # 10 mV each and cannot cross; an isolated peak is unaffected. This is
    # a constraint on identity, not on movement: a peak that needs to travel
    # further than half way to its neighbour has stopped being that peak.
    _c = np.asarray(spec.centres, float)
    _tol = np.full(_c.size, float(spec.centre_tol))
    if _c.size > 1:
        _order = np.argsort(_c)
        _sorted = _c[_order]
        _gaps = np.diff(_sorted)
        _half = np.empty(_c.size)
        _half[0] = _gaps[0] / 2.0
        _half[-1] = _gaps[-1] / 2.0
        if _c.size > 2:
            _half[1:-1] = np.minimum(_gaps[:-1], _gaps[1:]) / 2.0
        _tol[_order] = np.minimum(_tol[_order], _half)
    # A CENTRE CANNOT LEAVE THE DATA. The tolerance above is a constraint on
    # identity, and it is applied around the SEED — so a seed placed near an
    # edge could still be fitted to a centre outside the window, where there
    # is nothing to fit it to. `detect.infer_missing_seeds` now refuses to
    # place a seed outside the half-cycle it belongs to, which removes the
    # source; this bounds the destination as well, so no future seeder can
    # reintroduce it. A seed already outside the window is pulled to the edge
    # rather than dropped, because dropping a component silently changes the
    # model and this is a fitter, not a selector.
    _wlo, _whi = float(v.min()), float(v.max())
    for i, c in enumerate(spec.centres):
        pre = f"p{i}_"
        _c0 = float(np.clip(float(c), _wlo, _whi))
        _cmin = max(_wlo, _c0 - float(_tol[i]))
        _cmax = min(_whi, _c0 + float(_tol[i]))
        if _cmin >= _cmax:  # degenerate window: keep a box
            _cmin, _cmax = _wlo, _whi
        params[pre + "center"].set(
            value=float(np.clip(_c0, _cmin, _cmax)), min=_cmin, max=_cmax
        )
        _s0 = sigma0
        if spec.sigma0 is not None and i < spec.sigma0.size:
            _v = float(spec.sigma0[i])
            if np.isfinite(_v) and _v > 0:
                # NOT ON THE BOUNDARY. A parameter started exactly at its own
                # limit has a one-sided derivative there, and the optimiser
                # can read that as a minimum and stop: seeding LTO's
                # discharge peak at its measured 2.6 mV width clipped to
                # sigma_min = 3 mV, the fit "converged" in 15 evaluations,
                # and R2 fell from 0.905 to 0.808 with a third of the area
                # missing. A better starting point made a worse fit, purely
                # because of where the bound is. The seed is held a little
                # inside the box; the fit is still free to go to the bound
                # if that is where the answer is, and on LTO it does.
                _lo = spec.sigma_min * (1.0 + _SEED_BOUND_MARGIN)
                _hi = spec.sigma_max * (1.0 - _SEED_BOUND_MARGIN)
                if _lo < _hi:
                    _s0 = float(np.clip(_v, _lo, _hi))
                else:
                    _s0 = float(np.clip(_v, spec.sigma_min, spec.sigma_max))
        _a0 = amp0
        if spec.amp0 is not None and i < spec.amp0.size:
            _v = float(spec.amp0[i])
            if np.isfinite(_v) and _v > 0:
                _a0 = _v
        params[pre + "sigma"].set(value=_s0, min=spec.sigma_min, max=spec.sigma_max)
        params[pre + "amplitude"].set(value=_a0, min=0)
        params[pre + "fraction"].set(value=0.5, min=0, max=1)
        if _use_asym:
            # Tied below when the ratio is shared; given its own bounds here
            # so the per-component mode has a box to work in, and so the
            # value is right if the tie is never made (a single component
            # with ASYMMETRY_SHARED off).
            params[pre + "sigma_r"].set(
                value=_s0, min=spec.sigma_min, max=spec.sigma_max
            )
        if _use_asym and not _use_band:
            # The asymmetric model carries a band width whether or not the
            # profile wants one. Held at zero, not varied: this is the bare
            # split pseudo-Voigt.
            params[pre + "width"].set(value=0.0, vary=False)
        if _use_band:
            # It has to EARN its width from the residual — but it cannot earn
            # anything if it starts on its own floor. Set to exactly 0.0 and
            # bounded below at 0.0, every component came back a peak with
            # w = 0.000 on every half-cycle: the derivative at a bound is
            # one-sided and the optimiser never moved it. That is the same
            # stall that made the seeded sigma "converge" in 15 evaluations
            # at R2 0.808 on LTO, made twice in one afternoon, which is why
            # the margin below is now applied to any seeded-and-bounded
            # parameter rather than to sigma alone.
            #
            # The start is a nudge, not an assertion: 2% of the ceiling, far
            # narrower than any real composition window, and free to fall
            # back to zero. CONVERGING to a bound is ordinary; STARTING on
            # one is the fault.
            _cap = float(_band_ceilings(spec)[i])
            if _cap <= 0.0:
                # This component is a peak: hold the band width at zero.
                params[pre + "width"].set(value=0.0, vary=False)
            else:
                _w0 = max(2.0 * _BAND_WIDTH_NEGLIGIBLE, 0.02 * _cap)
                params[pre + "width"].set(value=min(_w0, _cap), min=0.0, max=_cap)

    # --- tie the shoulders, share the mixing fraction --------------------
    # Both are lmfit `expr` constraints, so the tied parameters stop being
    # varied and `nvarys` falls: the reduction in free parameters is real and
    # is what makes the standard errors mean something. See the constants
    # block above for why these two and not others.
    shoulders = _resolved_shoulders(spec) if COUPLE_SHOULDER_SIGMA else {}
    if shoulders:
        # ONE ratio for the whole half-cycle, not one per shoulder. Two
        # shoulders sharing a ratio cost one parameter between them; a ratio
        # each costs exactly what tying them was meant to save.
        _k = spec.sigma_ratio
        params.add(
            "sh_k",
            value=(SHOULDER_SIGMA_RATIO_INIT if _k is None else float(_k)),
            min=SHOULDER_SIGMA_RATIO_MIN,
            max=SHOULDER_SIGMA_RATIO_MAX,
            vary=(_k is None),
        )
        for child, parent in shoulders.items():
            params[f"p{child}_sigma"].set(expr=f"sh_k * p{parent}_sigma")
            # BOTH FLANKS, ADDED IN 1.9.0.58 with per-component asymmetry.
            # Tying `sigma` alone was complete while the lineshape was
            # symmetric. Once each component carries its own `sigma_r`, a
            # shoulder whose low flank is tied and whose high flank is free
            # has regained an independent width parameter — which is the
            # exact ill-determination the coupling exists to prevent, half
            # restored. A shoulder is a scaled copy of its parent's SHAPE,
            # so the same ratio governs both sides and this costs nothing.
            if f"p{child}_sigma_r" in params and f"p{parent}_sigma_r" in params:
                params[f"p{child}_sigma_r"].set(expr=f"sh_k * p{parent}_sigma_r")

    # --- which components are allowed two flanks -------------------------
    # A component seeded within `sigma_max` of a window edge is one whose
    # model necessarily extends past the measured range; that is the one the
    # split lineshape is for. Everything else is tied exactly symmetric by
    # `expr`, so it costs no parameter and cannot use a runaway flank to
    # impersonate a continuum the band machinery is there to carry. See
    # ASYMMETRY_EDGE_ONLY.
    if _use_asym and getattr(spec, "asymmetry_edge_only", False):
        _reach = float(spec.sigma_max)
        for i, c in enumerate(spec.centres):
            _near_edge = (float(c) - v.min() <= _reach) or (
                v.max() - float(c) <= _reach
            )
            if not _near_edge:
                params[f"p{i}_sigma_r"].set(expr=f"p{i}_sigma")
    # --- one asymmetry ratio for the half-cycle --------------------------
    elif _use_asym and ASYMMETRY_SHARED and len(spec.centres) >= 1:
        params.add(
            "asym_k",
            value=ASYM_RATIO_INIT,
            min=ASYM_RATIO_MIN,
            max=ASYM_RATIO_MAX,
            vary=True,
        )
        for i in range(len(spec.centres)):
            params[f"p{i}_sigma_r"].set(expr=f"asym_k * p{i}_sigma")

    if spec.fraction is not None:
        # CALIBRATED AND HELD. The shape of the instrument's response is not
        # something each cycle gets to re-decide.
        for i in range(len(spec.centres)):
            params[f"p{i}_fraction"].set(value=float(spec.fraction), vary=False)
    elif SHARED_FRACTION and len(spec.centres) > 1:
        params.add("eta", value=0.5, min=0, max=1, vary=True)
        for i in range(len(spec.centres)):
            params[f"p{i}_fraction"].set(expr="eta")

    if baseline is not None:
        params.update(baseline.guess(y, x=v))

    if DROP_DERIVED:
        for i in range(len(spec.centres)):
            for suffix in ("fwhm", "height"):
                key = f"p{i}_{suffix}"
                if key in params:
                    del params[key]

    return model, params


def _derived(sigma, amplitude, fraction, sigma_r=None):
    """Recompute what we refused to let asteval compute during the fit.

    `sigma` is the half-width at half maximum on the LOW side and `sigma_r`
    on the high side; passing `sigma_r` None (the symmetric case) reproduces
    the old `fwhm = 2 * sigma` exactly. The height of the split shape is the
    area divided by the mean of the two sides' unit-height areas, which is
    what `asym_rect_pseudo_voigt` normalises by.
    """
    sig = max(sigma, _TINY)
    if sigma_r is None or not np.isfinite(sigma_r) or abs(sigma_r - sigma) <= _TINY:
        fwhm = _FWHM_FACTOR * sigma
        height = (amplitude / max(sig * np.sqrt(2 * np.pi), _TINY)) * (1 - fraction) + (
            amplitude / max(np.pi * sig, _TINY)
        ) * fraction
        return float(fwhm), float(height)
    sgr = max(float(sigma_r), _TINY)
    fwhm = sig + sgr
    hl = float(_rpv_area1(0.0, 0.0, sig, fraction))
    hr = float(_rpv_area1(0.0, 0.0, sgr, fraction))
    if not (hl > 0 and hr > 0):
        return float(fwhm), float("nan")
    height = float(amplitude) / (0.5 * (1.0 / hl + 1.0 / hr))
    return float(fwhm), float(height)


# Two fitted components have COLLAPSED ONTO ONE FEATURE when their centres
# lie within this multiple of the narrower one's sigma and one of them holds
# less than DEGENERATE_AREA_FRACTION of the total fitted area. Detection is
# supposed to prevent it — two peaks closer than `min_distance_mV` are one
# peak — but a shoulder seeded next to its parent can still converge onto it.
# The cost is not cosmetic: on P3 cell A cycle 10 Discharge such a pair took
# a quarter of the surviving peak's area with it.
DEGENERATE_CENTRE_SIGMAS = 0.5
DEGENERATE_AREA_FRACTION = 0.01
DROP_DEGENERATE = True


def _degenerate_component(rows):
    """Index of one collapsed component, or None. See DROP_DEGENERATE."""
    if len(rows) < 2:
        return None
    tot = sum(abs(r["amplitude_area"]) for r in rows)
    if not np.isfinite(tot) or tot <= 0:
        return None
    order = sorted(range(len(rows)), key=lambda i: abs(rows[i]["amplitude_area"]))
    for i in order:
        if abs(rows[i]["amplitude_area"]) / tot >= DEGENERATE_AREA_FRACTION:
            break
        for j in range(len(rows)):
            if j == i:
                continue
            sep = abs(rows[i]["centre"] - rows[j]["centre"])
            ref = min(abs(rows[i]["sigma"]), abs(rows[j]["sigma"]))
            if np.isfinite(ref) and sep <= DEGENERATE_CENTRE_SIGMAS * ref:
                return rows[i]["peak_index"]
    return None


def _drop_shoulder_index(spec: FitSpec, drop):
    """
    `shoulder_of` for `spec` with component `drop` removed.

    Every index above the dropped one shifts down by one; a shoulder whose
    PARENT was dropped is released rather than re-parented, because the peak
    it was a shoulder OF no longer exists and inventing a new parent would be
    inventing physics.
    """
    n = len(spec.centres)
    raw = getattr(spec, "shoulder_of", None) or ((None,) * n)
    out = []
    for i in range(n):
        if i == drop:
            continue
        p = raw[i] if i < len(raw) else None
        if p is None or int(p) == drop:
            out.append(None)
        else:
            p = int(p)
            out.append(p - 1 if p > drop else p)
    return tuple(out)


def fit_half_cycle(
    voltage,
    dqdv,
    spec: FitSpec,
    *,
    key=None,
    _refit_depth=0,
    _seed_index=None,
    _bands_added=0,
) -> dict:
    """
    Fit one half-cycle. Returns plain data — see the module docstring.

    Never raises: a failure is a result with success=False and a reason, so
    that one bad half-cycle cannot take down a run of four hundred.

    If two components converge onto the same feature, the smaller is dropped
    and the half-cycle refitted once — see `_degenerate_component`. The
    result records `dropped_degenerate` so the count is visible rather than
    quietly better, and each component keeps `seed_index`: its position in
    the DETECTED peak list it started from. `peak_index` is renumbered by a
    refit, so joining the detected peaks on it labelled every surviving
    component with the properties of the one below it.
    """
    seed_index = (
        list(range(len(spec.centres))) if _seed_index is None else list(_seed_index)
    )
    v = np.asarray(voltage, float)
    y = np.abs(np.asarray(dqdv, float))
    out = {
        "key": key,
        "success": False,
        "reason": "",
        "n_points": int(v.size),
        "components": [],
        "r_squared": np.nan,
        "redchi": np.nan,
        "nvarys": 0,
        "baseline_degree": spec.baseline_degree,
        # The BOUNDS THIS FIT ACTUALLY USED. Downstream has to know them:
        # `analyse._flag_reliable_peaks` asks "is this component pinned at
        # its width bound, acting as baseline?", and it was asking against
        # a module constant of 200 mV while a sharp profile's real bound is
        # 30 mV — so on every LTO and LFP dataset the guard could not fire.
        "sigma_max": float(spec.sigma_max),
        "sigma_min": float(spec.sigma_min),
        # THE SAMPLING INTERVAL OF THIS CURVE, so a width can be judged
        # against what the measurement can resolve rather than against a
        # constant. LTO cell C's discharge peak is 2.6 mV wide sampled at
        # 1 mV: three points across the full width at half maximum, and a
        # width read from three points is not a measurement.
        "sample_mV": float(np.median(np.diff(np.asarray(voltage, float))) * 1000.0)
        if np.asarray(voltage).size > 2
        else float("nan"),
        "centre_tol": float(spec.centre_tol),
        "dropped_degenerate": 0,
        # How much of the parameter count the constraints bought back.
        # `nvarys` alone cannot be read without them: a five-component fit
        # with two tied shoulders and a shared fraction varies 15
        # parameters, not 20, and the difference is the whole argument for
        # believing the standard errors.
        "n_shoulders_coupled": 0,
        "sigma_ratio_k": np.nan,
        "shared_fraction": np.nan,
        "fraction_fixed": False,
        # The ANALYSED WINDOW. The truncation test downstream asks whether
        # a fitted peak's centre +/- 2 sigma lies inside it, which is a
        # question about this fit and cannot be answered from the detected
        # peak list.
        "v_min": float(np.min(v)) if v.size else np.nan,
        "v_max": float(np.max(v)) if v.size else np.nan,
        "curve_area": float(trapezoid(y, v)) if v.size > 1 else np.nan,
        "baseline_coeffs": {},
        "component_area_sum": 0.0,
        "baseline_area": np.nan,
        "unattributed_area": np.nan,
        "unattributed_fraction": np.nan,
        "bands_from_residual": 0,
        "n_at_sigma_max": 0,
        "seconds": 0.0,
    }

    if v.size < 5 or len(spec.centres) == 0:
        out["reason"] = "too few points or no peaks proposed"
        return out

    t0 = time.perf_counter()
    try:
        model, params = build_model(v, y, spec)
        # STAGED REFINEMENT — POSITIONS FIRST, THEN SHAPE.
        #
        # The same discipline a Rietveld refinement uses, and for the same
        # reason: centre, width and amplitude are strongly correlated, and
        # releasing all three at once on a multi-component curve lets the
        # optimiser trade one against another along a nearly flat direction
        # instead of descending. Get the positions right while the shape is
        # held, and the shape that follows is a short, well-conditioned step.
        #
        # Stage 1 fixes sigma and fraction at their seeded values and refines
        # centres and amplitudes (and the baseline). Stage 2 releases
        # everything, starting from stage 1's answer.
        #
        # This is also what makes the seeding worth having: stage 1 is only
        # meaningful if the widths it holds fixed are close, which is exactly
        # what detection now measures for every component including shoulders.
        # The tolerance the DATA supports, not the library default. See
        # FIT_TOLERANCE: 1.5e-8 is what made a 5-component fit spend 20,000
        # evaluations and then be recorded as a failure.
        _tol = getattr(spec, "tolerance", None)
        if _tol is None:
            _tol = FIT_TOLERANCE
        _kws = None if not _tol else dict(ftol=float(_tol), xtol=float(_tol))
        if STAGED_REFINEMENT and len(spec.centres) > 1:
            _frozen = []
            for _pn, _p in params.items():
                if (
                    (_pn.endswith("sigma") or _pn.endswith("fraction"))
                    and _p.vary
                    and _p.expr is None
                ):
                    _p.set(vary=False)
                    _frozen.append(_pn)
            if _frozen:
                _stage1 = model.fit(
                    y, params, x=v, max_nfev=spec.max_nfev, fit_kws=_kws
                )
                params = _stage1.params.copy()
                for _pn in _frozen:
                    params[_pn].set(vary=True)
        res = model.fit(y, params, x=v, max_nfev=spec.max_nfev, fit_kws=_kws)

        # --- THE BOUND STALL (item 39) ---------------------------------
        # A least-squares minimiser reports convergence when it can no
        # longer reduce the cost. On a parameter sitting AT its bound it
        # cannot move outward at all, so the projected gradient is zero and
        # the fit is declared converged — whether or not the bound is where
        # the minimum is. `_SEED_BOUND_MARGIN` starts every width a quarter
        # of the way inside its box for exactly this reason; nothing has
        # ever protected a fit that WALKS onto a bound during refinement.
        #
        # The measured consequence: fits stopping under 100 evaluations have
        # median R2 0.79 and height ratio 0.57, against 0.98 and 0.98 for
        # fits over 1000. And on NMC111 cell B's top of charge, 13 of 15
        # charge components finish pinned at the sigma ceiling with 18.9% of
        # their area outside the fitting window, carrying about a third of
        # the charge-side area — after 1.9.0.61 gave them bands, which moved
        # them from one bound to another rather than freeing the width.
        #
        # THE TEST IS NOT "IS THE BOUND WRONG", IT IS "DID THE OPTIMISER
        # STOP FOR A REAL REASON". A wider ceiling was measured in .58 and
        # was worse; this does not touch the bounds. It restarts the pinned
        # parameters a defined distance INSIDE their box and minimises
        # again, then keeps whichever answer has the lower cost. If the fit
        # walks back to the bound, the bound is genuinely binding and the
        # result is unchanged and still flagged. If it finds an interior
        # minimum, the previous answer was the optimiser stopping at a wall.
        #
        # Cost: one extra fit, only on half-cycles that finished on a bound.
        if BOUND_ESCAPE_RESTART and res is not None:
            _p2 = res.params.copy()
            _moved = []
            for _pn, _p in _p2.items():
                if not _p.vary or _p.expr is not None:
                    continue
                _lo, _hi = _p.min, _p.max
                if not (np.isfinite(_lo) and np.isfinite(_hi)) or _hi <= _lo:
                    continue
                _span = _hi - _lo
                _eps = BOUND_ESCAPE_PROXIMITY * _span
                if _p.value >= _hi - _eps:
                    _p.set(value=_hi - BOUND_ESCAPE_STEP * _span)
                    _moved.append(_pn)
                elif _p.value <= _lo + _eps:
                    _p.set(value=_lo + BOUND_ESCAPE_STEP * _span)
                    _moved.append(_pn)
            if _moved:
                out["bound_escape_tried"] = len(_moved)
                try:
                    _res2 = model.fit(y, _p2, x=v, max_nfev=spec.max_nfev, fit_kws=_kws)
                except Exception:  # noqa: BLE001
                    _res2 = None
                if _res2 is not None:
                    _c1 = float(getattr(res, "chisqr", np.inf) or np.inf)
                    _c2 = float(getattr(_res2, "chisqr", np.inf) or np.inf)
                    # STRICTLY better, by more than rounding. A restart that
                    # lands on the same minimum must not churn the answer.
                    if np.isfinite(_c2) and _c2 < _c1 * (1.0 - BOUND_ESCAPE_GAIN):
                        res = _res2
                        out["bound_escape_used"] = True
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"{type(exc).__name__}: {exc}"
        out["seconds"] = time.perf_counter() - t0
        return out

    # DID THE OPTIMISER ACTUALLY CONVERGE? Until 1.9.0.26 `success` was set
    # to True a few lines below for every fit that did not raise, and
    # `res.success` — lmfit's own verdict — was never read. On the NNM
    # triplicate that hid a great deal: 57% of half-cycle fits ran to the
    # 20,000-evaluation ceiling, gave up, and returned whatever parameters
    # they were holding, and every one of them was reported with an R2 and
    # standard errors as though it had converged. A 21-30 parameter
    # pseudo-Voigt-plus-polynomial on a noisy broad curve frequently does not
    # converge; LTO, at 6 parameters, uses a median of 183 evaluations and
    # never comes close. Recorded, not deleted — the parameters are still the
    # best description the optimiser reached — but a reader must be told.
    out["converged"] = bool(getattr(res, "success", True))
    out["nfev"] = int(getattr(res, "nfev", 0) or 0)
    out["max_nfev"] = int(spec.max_nfev)
    out["hit_iteration_cap"] = bool(out["nfev"] >= int(spec.max_nfev))

    comps = res.eval_components(x=v)
    peak_total = sum(val for name, val in comps.items() if not name.startswith("bg_"))
    baseline = comps.get("bg_", np.zeros_like(v))

    rows = []
    for i in range(len(spec.centres)):
        pre = f"p{i}_"
        try:
            centre = float(res.params[pre + "center"].value)
            sigma = float(res.params[pre + "sigma"].value)
            amp = float(res.params[pre + "amplitude"].value)
            frac = float(res.params[pre + "fraction"].value)
        except KeyError:
            continue

        def _err(pname):
            e = res.params[pre + pname].stderr
            return float(e) if e is not None else np.nan

        _sigma_r = sigma
        _sigma_r_err = np.nan
        if (pre + "sigma_r") in res.params:
            _sigma_r = float(res.params[pre + "sigma_r"].value)
            _sigma_r_err = _err("sigma_r")
        fwhm, height = _derived(sigma, amp, frac, _sigma_r)
        # PEAK OR BAND — the component's own answer, not a setting.
        # `width` is the composition window it claims. It starts at zero and
        # is bounded at zero, so it can only be non-zero if the residual paid
        # for it; and it is called a band only when it is BOTH wider than the
        # cycler's voltage resolution AND separated from zero by its own
        # standard error. Anything else is a peak, which is what the w = 0
        # branch of the model reduces to exactly.
        _bw = 0.0
        _bw_err = np.nan
        if (pre + "width") in res.params:
            _bw = float(res.params[pre + "width"].value)
            _bw_err = _err("width")
        _is_band = bool(
            _bw > _BAND_WIDTH_NEGLIGIBLE
            and (not np.isfinite(_bw_err) or _bw > 2.0 * _bw_err)
        )
        # A BAND AT ITS WIDTH CEILING IS ACTING AS BASELINE, exactly as a
        # component at its sigma bound is. It is not a composition window
        # any more; it is a flat pedestal the fit is using to lift the
        # curve. Flagged here rather than discovered later, and carried out
        # to the parameter table so `reliable` can act on it.
        _bw_max = float(_band_ceilings(spec)[i])
        _bw_pinned = bool(
            _bw_max > 0 and _bw >= _bw_max * SIGMA_BOUND_PROXIMITY_COMPLEMENT
        )
        rows.append(
            dict(
                peak_index=i,
                seed_index=(seed_index[i] if i < len(seed_index) else i),
                centre=centre,
                centre_stderr=_err("center"),
                sigma=sigma,
                sigma_stderr=_err("sigma"),
                # The HIGH-side half-width and the ratio between the two. Equal to
                # `sigma` and 1.0 exactly when the lineshape is symmetric, so the
                # columns are always present and always mean the same thing.
                sigma_r=_sigma_r,
                sigma_r_stderr=_sigma_r_err,
                asymmetry=(float(_sigma_r / sigma) if sigma > 0 else np.nan),
                amplitude_area=amp,
                amplitude_stderr=_err("amplitude"),
                fraction=frac,
                fraction_stderr=_err("fraction"),
                band_width=_bw,
                band_width_stderr=_bw_err,
                band_width_max_fitted=_bw_max,
                band_at_width_bound=_bw_pinned,
                # ETA AT ITS BOUND, for parity with `at_sigma_max` and
                # `asymmetry_at_bound`. eta = 1 is a pure Lorentzian, whose wings
                # carry area far from the centre; on a two-phase transition, whose
                # tails are set by kinetics and inhomogeneity rather than by
                # lifetime broadening, that is a shape the data has been allowed
                # to reach rather than one anything argues for. Measured on LTO
                # cell A cycle 1: eta = 1.000 and the component over-covers the
                # cell's charge by 6%.
                fraction_at_bound=bool(
                    frac >= 1.0 - _BOUND_PROXIMITY or frac <= _BOUND_PROXIMITY
                ),
                component_kind=("band" if _is_band else "peak"),
                fwhm=fwhm,
                height=height,
                # Area actually inside the fitted window, which is what any
                # capacity statement must use — the analytic amplitude includes
                # tails that were never measured.
                area_in_window=float(trapezoid(comps[pre], v)),
            )
        )

    bkg = {k: float(v.value) for k, v in res.params.items() if k.startswith("bg_")}

    # --- did two components land on one feature? -------------------------
    if DROP_DEGENERATE and _refit_depth < len(spec.centres):
        bad = _degenerate_component(rows)
        if bad is not None:
            keep = [c for i, c in enumerate(spec.centres) if i != bad]
            keep_seeds = [sx for i, sx in enumerate(seed_index) if i != bad]
            keep_sh = _drop_shoulder_index(spec, bad)
            if keep:
                out2 = fit_half_cycle(
                    voltage,
                    dqdv,
                    spec.replace(centres=tuple(keep), shoulder_of=keep_sh),
                    key=key,
                    _refit_depth=_refit_depth + 1,
                    _seed_index=keep_seeds,
                )
                if out2.get("success"):
                    out2["dropped_degenerate"] = out2.get("dropped_degenerate", 0) + 1
                    out2["seconds"] = (
                        out2.get("seconds", 0.0) + time.perf_counter() - t0
                    )
                    return out2

    # --- BANDS SEEDED FROM THE CHARGE THE PEAKS CANNOT ACCOUNT FOR ------
    # See BAND_FROM_RESIDUAL. Gated on `band_width_max`, which only the
    # mechanism router sets, so a two-phase cell can never acquire one.
    # Runs once per half-cycle (`_bands_added` guards the recursion) and is
    # KEPT ONLY IF IT NAMES MORE CHARGE than the peaks alone did -- an extra
    # component that does not is not evidence of anything.
    _bands_added = int(_bands_added)
    if (
        getattr(spec, "band_from_residual", False)
        and float(getattr(spec, "band_ceiling", 0.0) or 0.0) > 0
        and _bands_added == 0
        and v.size > 8
    ):
        _curve_area = float(trapezoid(y, v))
        _named0 = float(trapezoid(peak_total, v))
        if _curve_area > 0:
            if spec.band_seeds:
                # THE DATASET DECIDED. Every half-cycle of this step gets the
                # same regions, so the component set has a stable identity and
                # tracking can follow it. A region this curve would not have
                # produced on its own is still fitted; whether there is
                # anything there is then a measurement rather than a change of
                # model.
                #
                # A REGION AND A MAXIMUM INSIDE IT ARE THE SAME FEATURE. Where
                # detection already placed a component in the region, that
                # component becomes the band — it is not joined by a second
                # one. Adding alongside was measured and is worse: NMC111 cell
                # B, whose detection does find a maximum on the 4.28 V shelf,
                # went from 18 components at the ceiling to 31 and from 57 to
                # 41 reliable when a band was appended next to it. Two
                # components on one feature is the degeneracy this whole pass
                # exists to remove, not a way of removing it.
                _existing = np.asarray(spec.centres, float)
                _caps0 = list(_band_ceilings(spec))
                _ceil_V = band_ceiling_volts(spec, v)
                _promoted = False
                _seed = []
                for c, w in spec.band_seeds:
                    c = float(c)
                    if not (v[0] <= c <= v[-1]):
                        continue
                    _lo, _hi = c - float(w) / 2000.0, c + float(w) / 2000.0
                    _inside = np.nonzero((_existing >= _lo) & (_existing <= _hi))[0]
                    if _inside.size:
                        # Promote the component nearest the region's centre.
                        _j = int(_inside[np.argmin(np.abs(_existing[_inside] - c))])
                        # THE REGION SIZES THE BAND, THE MECHANISM BOUNDS
                        # IT — the same rule the appended-seed branch below
                        # states and follows, and which this branch never
                        # got. Capping a PROMOTED component at its region's
                        # own span is the policy that was measured and
                        # rejected there ("capping at the span exactly pinned
                        # the fitted width to the bound on 49 of 206 NMC
                        # components"). It survived here because the two
                        # branches were written eleven builds apart.
                        #
                        # An excess region is where the peaks fell SHORT, so
                        # its span UNDERSTATES the feature. On the NNM
                        # triplicate every from-bound region is exactly
                        # 2*sigma_max = 400 mV wide by construction, so this
                        # line set a 400 mV ceiling on 616 components and
                        # 616 of them ended pinned at it: the fitted width
                        # was the bound, not a measurement.
                        _caps0[_j] = max(_caps0[_j], _ceil_V)
                        _promoted = True
                        continue
                    _seed.append(
                        dict(
                            centre=c,
                            span_mV=float(w),
                            lo=_lo,
                            hi=_hi,
                            area=float("nan"),
                            from_reference=True,
                        )
                    )
                if _promoted and not _seed:
                    # Nothing to add, but a component has become a band.
                    out2 = fit_half_cycle(
                        voltage,
                        dqdv,
                        spec.replace(band_width_max=np.asarray(_caps0, float)),
                        key=key,
                        _refit_depth=_refit_depth,
                        _seed_index=_seed_index,
                        _bands_added=1,
                    )
                    if out2.get("success"):
                        out2["seconds"] = (
                            out2.get("seconds", 0.0) + time.perf_counter() - t0
                        )
                        return out2
            else:
                _regs = residual_excess_regions(
                    v,
                    y - (peak_total + baseline),
                    min_span_mV=spec.band_residual_min_span_mV,
                )
                _seed = [
                    g
                    for g in _regs
                    if g["area"] / _curve_area >= spec.band_residual_min_share
                ][: max(0, int(spec.band_residual_max))]
            if _seed:
                _c = list(np.asarray(spec.centres, float)) + [
                    g["centre"] for g in _seed
                ]
                _sh = list(
                    getattr(spec, "shoulder_of", None) or ((None,) * len(spec.centres))
                )
                _sh = _sh[: len(spec.centres)] + [None] * len(_seed)
                _s0 = (
                    list(spec.sigma0)
                    if spec.sigma0 is not None
                    else [np.nan] * len(spec.centres)
                )
                _a0 = (
                    list(spec.amp0)
                    if spec.amp0 is not None
                    else [np.nan] * len(spec.centres)
                )
                for g in _seed:
                    # Start the band at the size of the region it belongs to:
                    # half its span as sigma, and its own unattributed charge
                    # as amplitude where that was measured. A REFERENCE region
                    # has no area of its own in this half-cycle, so it starts
                    # from what the curve is actually carrying there.
                    _s0.append(max(spec.sigma_min, g["span_mV"] / 2000.0))
                    _a = g.get("area", np.nan)
                    if not np.isfinite(_a):
                        _m = (v >= g["lo"]) & (v <= g["hi"])
                        _a = float(trapezoid(y[_m], v[_m])) if _m.sum() > 1 else 0.0
                    _a0.append(max(float(_a), 0.0))
                _si = list(_seed_index or range(len(spec.centres)))
                _si = _si[: len(spec.centres)] + [
                    int(np.searchsorted(v, g["centre"])) for g in _seed
                ]
                # Detected peaks keep whatever ceiling they had (zero, so
                # they stay peaks); each seeded component is allowed a band
                # no wider than the region it was found in.
                # THE REGION SIZES THE BAND, THE MECHANISM BOUNDS IT.
                # The excess region is where the peaks fell SHORT, so its
                # span understates the band's true extent -- capping at the
                # span exactly pinned the fitted width to the bound on 49 of
                # 206 NMC components. The region is therefore the STARTING
                # SIZE (`sigma0`/`amp0` above) and the mechanism's ceiling is
                # the bound. What keeps this from becoming the 0.5 V
                # free-for-all that made the decomposition incoherent is that
                # only a residual-seeded component may be a band at all;
                # every detected maximum stays a peak with a ceiling of zero.
                _caps = list(_caps0) if spec.band_seeds else list(_band_ceilings(spec))
                _ceil = band_ceiling_volts(spec, v)
                for g in _seed:
                    _caps.append(_ceil)
                out2 = fit_half_cycle(
                    voltage,
                    dqdv,
                    spec.replace(
                        centres=tuple(_c),
                        shoulder_of=tuple(_sh),
                        sigma0=np.asarray(_s0, float),
                        amp0=np.asarray(_a0, float),
                        band_width_max=np.asarray(_caps, float),
                    ),
                    key=key,
                    _refit_depth=_refit_depth,
                    _seed_index=_si,
                    _bands_added=len(_seed),
                )
                if out2.get("success"):
                    _named2 = float(out2.get("component_area_sum", np.nan))
                    # A region the DATASET established is fitted whether or
                    # not it happens to help this half-cycle — that constancy
                    # is the point. A region discovered here still has to earn
                    # its place by naming more charge.
                    _keep2 = bool(spec.band_seeds) or (
                        np.isfinite(_named2) and _named2 > _named0
                    )
                    if _keep2:
                        out2["seconds"] = (
                            out2.get("seconds", 0.0) + time.perf_counter() - t0
                        )
                        out2["band_seed_regions"] = _seed
                        return out2

    def _val(pname):
        pr = res.params.get(pname)
        return float(pr.value) if pr is not None else np.nan

    _eta = _val("eta")
    if not np.isfinite(_eta) and rows:
        # No shared `eta` parameter exists when the fraction was FIXED, or
        # when there is a single component. Either way the value that was
        # used is on the components.
        _eta = float(rows[0]["fraction"])
    _ak = _val("asym_k")
    if not np.isfinite(_ak) and rows:
        _r0 = rows[0]
        _ak = float(_r0["sigma_r"] / _r0["sigma"]) if _r0["sigma"] > 0 else np.nan
    # THE TIE CAN BE BROKEN WITHOUT SAYING SO. `sigma_r` carries
    # `expr = asym_k * sigma` AND `min = sigma_min`, and lmfit applies bounds
    # to a constrained parameter AFTER evaluating its expression. So whenever
    # `asym_k * sigma` falls below the width floor, `sigma_r` is clipped and
    # the effective ratio is no longer `asym_k`. Measured on LTO: 19 of 61
    # components, disagreeing by up to 1.94x — and `asym_k` is the number the
    # page was quoting. The EFFECTIVE ratio is the one the curve was drawn
    # with, so that is what is reported; `asymmetry_k_fitted` keeps the
    # optimiser's value and `asymmetry_k_clipped` says they parted company.
    _ak_eff = _ak
    if rows:
        _eff = [
            float(r["sigma_r"] / r["sigma"])
            for r in rows
            if r["sigma"] > 0 and np.isfinite(r["sigma_r"])
        ]
        if _eff:
            _ak_eff = float(np.median(_eff))
    _ak_clipped = bool(
        np.isfinite(_ak)
        and np.isfinite(_ak_eff)
        and _ak > 0
        and abs(_ak_eff - _ak) / _ak > 0.01
    )
    # A SHARED PARAMETER ON ITS BOUND IS NOT A MEASUREMENT — the same fault
    # as a sigma pinned on its floor, one level up, and it would be silent
    # because there is only one of these per half-cycle. Reported, not
    # clamped: the fit is still the best available description, but a reader
    # must not read a bound as an answer.
    # Against the FITTED value: that is the parameter that carries the bounds.
    _ak_pinned = bool(
        np.isfinite(_ak)
        and (
            _ak <= ASYM_RATIO_MIN * (1.0 + _BOUND_PROXIMITY)
            or _ak >= ASYM_RATIO_MAX * (1.0 - _BOUND_PROXIMITY)
        )
    )
    out.update(
        n_shoulders_coupled=len(_resolved_shoulders(spec))
        if COUPLE_SHOULDER_SIGMA
        else 0,
        sigma_ratio_k=_val("sh_k"),
        shared_fraction=_eta,
        fraction_fixed=spec.fraction is not None,
        asymmetry_k=_ak_eff,
        asymmetry_k_fitted=_ak,
        asymmetry_k_clipped=_ak_clipped,
        asymmetry_at_bound=_ak_pinned,
        asymmetry_fitted=bool(getattr(spec, "asymmetry", False)),
    )

    ss_res = float(np.sum(res.residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    # THE FOUR NUMBERS R2 HIDES. Computed here because this is the only place
    # the composite and the baseline both exist; costs one pass each.
    _q = (
        fit_quality(v, y, peak_total + baseline, baseline)
        if FIT_QUALITY
        else dict(
            height_ratio=np.nan,
            overshoot=np.nan,
            max_residual_frac=np.nan,
            residual_runs_z=np.nan,
        )
    )
    out.update(_q)
    out.update(
        success=True,
        r_squared=(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        redchi=float(res.redchi),
        nvarys=int(res.nvarys),
        components=rows,
        # Kept because a lineshape that adds a parameter has to EARN it, and
        # R2 cannot say whether it did. lmfit computes both from the same
        # residual the fit minimised.
        bic=float(getattr(res, "bic", np.nan)),
        aic=float(getattr(res, "aic", np.nan)),
        component_area_sum=float(trapezoid(peak_total, v)),
        baseline_area=float(trapezoid(np.abs(baseline), v)),
        # THE CHARGE THE MODEL CANNOT NAME. Every mAh in a dQ/dV passed
        # through the cell, so this is a number about the cell and not about
        # the fit: it is reported, never absorbed. Positive means components
        # fall short of the curve; negative means they sum to more than the
        # cell delivered, which is a decomposition fault however good the
        # residual looks.
        unattributed_area=float(trapezoid(y - (peak_total + baseline), v)),
        unattributed_fraction=(
            float(trapezoid(y - (peak_total + baseline), v) / trapezoid(y, v))
            if v.size > 1 and trapezoid(y, v) > 0
            else np.nan
        ),
        bands_from_residual=int(_bands_added),
        # HOW MANY COMPONENTS ENDED AT THE WIDTH CEILING. A peak pinned at
        # `sigma_max` is not a peak the data chose the width of; on a curve
        # with a continuum in it, it is a BAND BEING DENIED -- the fitter's
        # only way to reach charge spread over a region when the mechanism
        # has not permitted a component shaped like one. Counted here so the
        # mechanism decision can act on it: see
        # `quality.MECHANISM_AT_BOUND_ESCALATES`.
        n_at_sigma_max=int(
            sum(
                1
                for _c in rows
                if float(_c["sigma"])
                >= float(spec.sigma_max) * SIGMA_BOUND_PROXIMITY_COMPLEMENT
            )
        ),
        # The baseline coefficients, so the fitted curve can be rebuilt from
        # this dict alone. `evaluate` needs them; keeping them costs four
        # floats and saves shipping a ModelResult, which is the one thing this
        # module refuses to do.
        baseline_coeffs=bkg,
        seconds=time.perf_counter() - t0,
    )
    return out


# ---------------------------------------------------------------------------
# What R2 does not see
# ---------------------------------------------------------------------------
# R2 is dominated by the BULK of a curve, so a composite can be half again
# taller than the data, spike to 3.4x its height, or miss a peak by a third,
# and still score 0.97. Measured on the runs in hand:
#
#                     R2      height  max resid  overshoot  runs z
#   LTO discharge   0.784      0.68       40%        0%       -18
#   NNM discharge   0.977      0.98       12%        0%       -14
#   NMC discharge   0.968      1.42       41%       31%       -13
#
# Three different faults — a fit too short, a fit too tall and outside the
# data, and a wrong lineshape — and R2 announces none of them. These four do,
# they cost one pass over an array each, and they are what the eye was seeing.
FIT_QUALITY = True


def _runs_z(resid):
    """Runs test on the SIGN of the residual.

    Noise changes sign about half the time; a residual whose SHAPE is wrong
    does not — it sits above the data for a stretch and then below. Large
    negative z means long same-sign runs, which is systematic misfit rather
    than scatter, and it is the one of these four that fires even where the
    height is right (NNM: z = -14 at R2 0.98).
    """
    s = np.sign(np.asarray(resid, float))
    s = s[s != 0]
    if s.size < 10:
        return np.nan
    n1 = int((s > 0).sum())
    n2 = int((s < 0).sum())
    if n1 == 0 or n2 == 0:
        return np.nan
    runs = 1 + int((s[1:] != s[:-1]).sum())
    n = n1 + n2
    mu = 2.0 * n1 * n2 / n + 1.0
    var = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n)) / (n**2 * (n - 1))
    return float((runs - mu) / np.sqrt(var)) if var > 0 else np.nan


def fit_quality(voltage, dqdv, total, baseline):
    """
    Four numbers about the fit that R2 cannot carry.

    `total` is the composite (peaks + baseline) and `baseline` the baseline
    alone, both evaluated on `voltage`. Returns a dict; every entry is NaN
    rather than absent when it cannot be computed, so the columns are always
    the same shape.
    """
    v = np.asarray(voltage, float)
    y = np.asarray(dqdv, float)
    t = np.asarray(total, float)
    b = np.asarray(baseline, float)
    out = dict(
        height_ratio=np.nan,
        overshoot=np.nan,
        max_residual_frac=np.nan,
        residual_runs_z=np.nan,
    )
    if y.size < 3 or t.size != y.size:
        return out
    amp = float(y.max() - y.min())
    # HEIGHT, measured from the fitted baseline on both sides. The fit being
    # SHORT is what a width floor looks like (LTO); the fit being TALL is what
    # an undetermined peak/background split looks like, because the baseline
    # is drawn high and the peaks are deepened to reach the data (NMC).
    _hd = float(np.max(np.abs(y - b)))
    _hf = float(np.max(np.abs(t - b)))
    if _hd > 0:
        out["height_ratio"] = _hf / _hd
    if amp > 0:
        # How far the composite goes OUTSIDE the data's own range. A fit has
        # no business being anywhere the curve never went.
        out["overshoot"] = float(max(t.max() - y.max(), y.min() - t.min(), 0.0) / amp)
        out["max_residual_frac"] = float(np.max(np.abs(y - t)) / amp)
    out["residual_runs_z"] = _runs_z(y - t)
    return out


def evaluate(fit_result, voltage):
    """
    Rebuild the fitted curves from a result dict — no refit, no `ModelResult`.

    Returns {"p0_": ndarray, ..., "bg_": ndarray, "total": ndarray}, matching
    what `ModelResult.eval_components` would have returned plus the sum.

    This exists because `fit_half_cycle` deliberately returns plain data: a
    `ModelResult` cannot cross a process boundary through stdlib pickle, and
    keeping one would undo the decision that made parallel fitting possible.
    The component shapes are evaluated through lmfit's own model classes rather
    than reimplemented here, so the curve drawn is the curve that was fitted.
    """
    v = np.asarray(voltage, float)
    out, total = {}, np.zeros_like(v)
    for comp in fit_result.get("components", []):
        pre = f"p{comp['peak_index']}_"
        # THE SHAPE THAT WAS FITTED, NOT A BARE PSEUDO-VOIGT.
        #
        # Until 1.9.0.51 this rebuilt every component as a `PseudoVoigtModel`
        # regardless of what the fit had used, so a component carrying a band
        # width or a split width was redrawn as a symmetric peak of the same
        # area — a much taller, much narrower curve. `plots._FitView` takes
        # `best_fit` AND the residual panel from here, so the peak-fits figure
        # was not showing the fit. Measured on the runs in hand: NMC111 cell A
        # cycle 5 charge fits to a maximum residual of 12% of the data's
        # amplitude and was DRAWN at 354%, with the composite 3.75x the height
        # of the curve; NNM cell A cycle 5 discharge, 2% fitted, 220% drawn.
        # That is what "the fits aren't great" was looking at.
        _w = float(comp.get("band_width", 0.0) or 0.0)
        _sr = comp.get("sigma_r", None)
        _sr = (
            float(_sr) if _sr is not None and np.isfinite(_sr) else float(comp["sigma"])
        )
        _split = abs(_sr - float(comp["sigma"])) > _TINY
        if _w > 0.0 or _split:
            m = _LMModel(asym_rect_pseudo_voigt, prefix=pre)
            _extra = (("width", _w), ("sigma_r", _sr))
        else:
            m = PseudoVoigtModel(prefix=pre)
            _extra = ()
        # Values are SET on the Parameters object rather than passed to
        # make_params: lmfit silently ignores a prefixed keyword there, which
        # leaves `fraction` at its 0.5 default and draws the wrong curve.
        par = m.make_params()
        for suffix, val in (
            ("center", comp["centre"]),
            ("sigma", comp["sigma"]),
            ("amplitude", comp["amplitude_area"]),
            ("fraction", comp["fraction"]),
        ) + _extra:
            if (pre + suffix) in par:
                par[pre + suffix].set(value=float(val))
        y = m.eval(par, x=v)
        out[pre] = y
        total = total + y
    bkg = fit_result.get("baseline_coeffs") or {}
    if bkg:
        deg = max(int(k.split("c")[-1]) for k in bkg)
        m = PolynomialModel(degree=deg, prefix="bg_")
        par = m.make_params()
        for k, val in bkg.items():
            if k in par:
                par[k].set(value=float(val))
        y = m.eval(par, x=v)
        out["bg_"] = y
        total = total + y
    out["total"] = total
    return out


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


def _cpu_budget(n_jobs):
    if n_jobs is not None:
        return max(1, int(n_jobs))
    return max(1, (os.cpu_count() or 2) - 1)


# Measured once per session, because it cannot change within one: it is the
# cost of importing this module into a fresh interpreter.
_WORKER_RSS_MB = None


def _worker_rss_probe():
    """Run inside a worker: report what this process costs to exist, in MB.

    Unpickling this function is what makes the worker import the fitting
    machinery, so the number it returns is the real cost of a worker for THIS
    build on THIS machine. It is correct for both shapes the code ships in:
    imported from the package, loky pickles it by reference and the worker
    imports `ratatosk.fitting`; flattened into the notebook, cloudpickle sends
    it by value and the worker imports the same modules it references. Either
    way the probe pays exactly what a real job pays.

    IT RUNS A FIT, and that is not decoration. Importing the module measured
    148 MB here; the same worker after real fitting held 181 MB, because lmfit
    and scipy pull more in on first use and the fit allocates. A probe that
    only imports under-reports the thing it exists to report by a fifth, so it
    fits a small synthetic peak first — milliseconds, once per session — and
    measures afterwards.
    """
    import psutil

    x = np.linspace(0.0, 1.0, 201)
    y = 0.05 * np.exp(-0.5 * ((x - 0.5) / 0.05) ** 2)
    try:
        fit_half_cycle(
            x,
            y,
            FitSpec(
                [0.5], baseline_degree=1, sigma_min=0.005, sigma_max=0.20, max_nfev=200
            ),
            key="probe",
        )
    except Exception:
        # A probe that cannot fit is still a probe: the imports it paid for
        # are the point, and the caller clamps whatever comes back.
        pass
    return psutil.Process().memory_info().rss / (1024.0**2)


def measure_worker_cost_mb(*, force=False):
    """What one worker costs on this machine, in MB. None if unmeasurable.

    Starts a single worker in a PRIVATE executor, asks it, and shuts that down.

    Private matters. 1.9.0.43 used `get_reusable_executor`, which is joblib's
    process-wide SINGLETON: the probe created it, shut it down, and joblib then
    had to build a new one on top of a torn-down instance. Shutting a loky
    executor down and immediately recreating it is the classic way to leave its
    resource tracker or call queue in a state the next pool waits on forever,
    and 1.9.0.44's first long run froze in Cell 10. That is a hypothesis, not a
    proof — it did not reproduce on Linux — but the churn bought nothing, so it
    is gone. A private `ProcessPoolExecutor` cannot touch the singleton at all.

    One extra process spawn per session, ~5-8 s on Windows, against a pass
    that has already been estimated at a minute or more. Cheap, and it is a
    measurement rather than a guess about someone else's laptop.
    """
    global _WORKER_RSS_MB
    if _WORKER_RSS_MB is not None and not force:
        return _WORKER_RSS_MB
    if not WORKER_PROBE:
        return None
    ex = None
    try:
        # NOT `get_reusable_executor` — see the docstring. This is a private
        # instance of the same class, so it uses the same cloudpickle path
        # (which is what lets the flattened notebook send a `__main__`
        # function to a worker) without going near joblib's singleton.
        from joblib.externals.loky import ProcessPoolExecutor

        ex = ProcessPoolExecutor(max_workers=1)
        mb = float(ex.submit(_worker_rss_probe).result(timeout=180))
        ex.shutdown(kill_workers=True)
        ex = None
    except Exception:
        # A FAILED PROBE MUST COST NOTHING BUT THE MEASUREMENT. Its own
        # executor is torn down on the way out; joblib's singleton was never
        # touched, so the fitting that follows is unaffected either way, and
        # the caller falls back to the assumed cost.
        try:
            if ex is not None:
                ex.shutdown(kill_workers=True)
        except Exception:
            pass
        return None
    if not np.isfinite(mb) or mb <= 0:
        return None
    _WORKER_RSS_MB = mb
    return mb


def _available_memory_mb():
    """Free memory right now, in MB, or None if it cannot be read."""
    try:
        import psutil

        return psutil.virtual_memory().available / (1024.0**2)
    except Exception:
        return None


def memory_budget_workers():
    """
    How many workers this machine can afford right now, and why.

    Returns (n_workers, reason). `n_workers` is None when memory could not be
    read at all — which is NOT the same as "as many as you like": the caller
    still applies `PARALLEL_MAX_WORKERS`. Zero means not even one worker fits,
    and the caller should run serially.
    """
    avail = _available_memory_mb()
    if avail is None:
        return None, (
            "free memory could not be read (psutil unavailable) — sizing on cores alone"
        )
    raw = measure_worker_cost_mb()
    # Say which number was actually used. A line reading "(measured)" beside a
    # figure a clamp had already replaced is the reporting fault this project
    # spent a week removing from everywhere else.
    if raw is None:
        cost, how = WORKER_RSS_ASSUMED_MB, "assumed — could not be measured"
    else:
        want = float(raw) * WORKER_WORKING_SET_FACTOR
        cost = min(max(want, WORKER_RSS_SANITY_MIN_MB), WORKER_RSS_SANITY_MAX_MB)
        how = (
            f"measured {raw:.0f} MB to start, ×{WORKER_WORKING_SET_FACTOR:g} in use"
            if abs(cost - want) < 0.5
            else f"measured {raw:.0f} MB, outside the sanity bounds"
        )
    n = int((avail * WORKER_MEMORY_HEADROOM) // cost)
    return max(0, n), (
        f"{avail / 1024.0:.1f} GB free, "
        f"{WORKER_MEMORY_HEADROOM:.0%} of it usable, "
        f"{cost:.0f} MB per worker ({how})"
    )


def release_workers(*, verbose=False):
    """
    Shut the worker pool down and give its memory back. True if it did.

    joblib keeps its workers alive after `Parallel` returns, and they hold
    their full import footprint until the kernel exits. Measured: 412 MB still
    resident after a two-worker pass had finished, and **exiting a `with
    Parallel(...)` block does not release it** — only shutting the executor
    down does. Cell 9 and Cell 10 therefore call this when their stage is
    finished, so the rest of the notebook, and the next run in the same
    kernel, do not start against a pool that is no longer doing anything.

    NEVER CREATES A POOL IN ORDER TO CLOSE ONE. `get_reusable_executor()` is
    the obvious call and it is the wrong one: on a singleton that has already
    been shut down it starts a fresh pool, so "release the workers" would
    spawn `cpu_count` of them. The existing object is shut down directly, and
    if there is no object there is nothing to do.
    """
    try:
        from joblib.externals.loky import reusable_executor as _re

        ex = getattr(_re, "_executor", None)
        if ex is None:
            return False
        # ALREADY SHUT DOWN IS NOT "RELEASED". loky leaves the object in place
        # after a shutdown, so a bare `is None` check reported True every time
        # it was called again and claimed to have freed memory it had freed
        # once. Small, and exactly the kind of claim this project keeps
        # finding and removing.
        if getattr(getattr(ex, "_flags", None), "shutdown", False):
            return False
        ex.shutdown(kill_workers=True)
    except Exception:
        return False
    if verbose:
        print(entry("worker pool", "released", "memory returned"))
    return True


def plan_workers(jobs, *, n_jobs=None, calibrate=3, verbose=True):
    """
    Decide serially-or-parallel by measuring, and say which and why.

    Returns (n_workers, calibration_results, estimate_seconds). The calibration
    fits are real results and are reused, never thrown away.
    """
    budget = _cpu_budget(n_jobs)
    if len(jobs) <= calibrate or budget == 1:
        if verbose:
            print(
                f"  fitting {len(jobs)} half-cycle(s) serially "
                f"({'too few to parallelise' if budget > 1 else 'one core'})"
            )
        return 1, [], 0.0

    done, elapsed = [], 0.0
    for v, y, spec, key in jobs[:calibrate]:
        t = time.perf_counter()
        done.append(fit_half_cycle(v, y, spec, key=key))
        elapsed += time.perf_counter() - t

    per = elapsed / max(len(done), 1)
    # Weight by peak count: a 6-peak fit is not a 2-peak fit.
    n_cal = np.mean([len(j[2].centres) for j in jobs[:calibrate]]) or 1
    n_all = np.mean([len(j[2].centres) for j in jobs]) or 1
    estimate = per * len(jobs) * (n_all / n_cal)

    if estimate < PARALLEL_MIN_SECONDS:
        if verbose:
            print(
                f"  fitting {len(jobs)} half-cycle(s) serially — estimated "
                f"{estimate:.0f} s, below the {PARALLEL_MIN_SECONDS:.0f} s "
                f"floor where a worker pool repays its start-up"
            )
        return 1, done, estimate

    workers = min(budget, len(jobs) - len(done))

    # --- and now the part that stops the kernel dying ----------------------
    # Everything above is a time argument. A worker also costs its imports —
    # measured at 181 MB, and more on Windows — and the old code spent that
    # `cpu_count - 1` times without ever asking whether the machine had it.
    capped_by = None
    if workers > PARALLEL_MAX_WORKERS:
        workers, capped_by = PARALLEL_MAX_WORKERS, "the ceiling"
    mem_n, why = memory_budget_workers()
    if mem_n is not None and mem_n < workers:
        if mem_n < 1:
            if verbose:
                print(
                    f"  fitting {len(jobs)} half-cycle(s) serially — "
                    f"{why}, which is not enough for one worker. This will "
                    f"be slow; closing something would make it faster."
                )
            return 1, done, estimate
        workers, capped_by = mem_n, "free memory"

    if verbose:
        print(
            f"  fitting {len(jobs)} half-cycle(s) on {workers} worker(s) — "
            f"estimated {estimate:.0f} s serial, {estimate / workers:.0f} s "
            f"parallel"
        )
        if capped_by == "free memory":
            print(f"      held to {workers} by memory, not cores: {why}")
        elif capped_by == "the ceiling":
            print(
                f"      held to the {PARALLEL_MAX_WORKERS}-worker ceiling "
                f"({budget} cores available); {why}"
            )
        elif mem_n is not None:
            print(f"      {why} — room for {mem_n}")
    return workers, done, estimate


# A long fit is a long silence. On a 200-cycle P3 cell the closure-interval
# pass is 600 fits and can run for ten minutes with nothing on screen, which
# is indistinguishable from a hang. Progress is printed at most this often, so
# a fast run stays quiet and a slow one keeps saying it is alive.
PROGRESS_EVERY_SECONDS = 15.0
PROGRESS_MIN_JOBS = 40


def fit_many(jobs, *, n_jobs=None, verbose=True, label=""):
    """
    Fit a list of (voltage, dqdv, spec, key) tuples.

    Order of the returned list matches `jobs`. `label` names the pass in the
    progress line, because a run makes three different ones — calibration,
    the production fit, and the closure interval — and "fitting 600" with no
    label does not say which.
    """
    if not jobs:
        return []

    workers, done, _ = plan_workers(jobs, n_jobs=n_jobs, verbose=verbose)
    total = len(jobs)
    show = verbose and total >= PROGRESS_MIN_JOBS
    tag = f"{label} " if label else ""
    t0 = time.perf_counter()
    state = {"n": len(done), "t": t0}

    def _tick(n):
        """Print at most every PROGRESS_EVERY_SECONDS, with an ETA."""
        now = time.perf_counter()
        if n < total and now - state["t"] < PROGRESS_EVERY_SECONDS:
            return
        state["t"] = now
        el = now - t0
        eta = (el / n * (total - n)) if n else float("nan")
        print(
            f"      {tag}{n}/{total} fitted   {el:.0f} s elapsed"
            + (f", ~{eta:.0f} s left" if np.isfinite(eta) and n < total else ""),
            flush=True,
        )

    if workers == 1:
        rest = []
        for v, y, sp, k in jobs[len(done) :]:
            rest.append(fit_half_cycle(v, y, sp, key=k))
            if show:
                _tick(len(done) + len(rest))
        return done + rest

    from joblib import Parallel, delayed

    # `return_as="generator_unordered"` lets results be counted as they land
    # rather than all at once at the end, which is the whole point of a
    # progress line. Older joblib does not have it; there the pass runs as it
    # always did and simply reports when it finishes.
    try:
        gen = Parallel(n_jobs=workers, return_as="generator_unordered")(
            delayed(fit_half_cycle)(v, y, sp, key=k)
            for v, y, sp, k in jobs[len(done) :]
        )
        got = []
        for r in gen:
            got.append(r)
            if show:
                _tick(len(done) + len(got))
        # generator_unordered gives no ordering guarantee, so the caller's
        # order is restored from the keys it asked for. `fit_many` promises
        # order and several callers zip its output against `jobs`.
        by_key = {}
        for r in got:
            by_key.setdefault(_hashable(r.get("key")), []).append(r)
        rest = []
        for v, y, sp, k in jobs[len(done) :]:
            bucket = by_key.get(_hashable(k))
            rest.append(
                bucket.pop(0)
                if bucket
                else {
                    "key": k,
                    "success": False,
                    "reason": "result lost in the worker pool",
                    "components": [],
                }
            )
        return done + rest
    except TypeError:
        rest = Parallel(n_jobs=workers)(
            delayed(fit_half_cycle)(v, y, sp, key=k)
            for v, y, sp, k in jobs[len(done) :]
        )
        return done + list(rest)


def _hashable(k):
    """A dict key for a job key that may itself be a list or an ndarray."""
    try:
        hash(k)
        return k
    except TypeError:
        return repr(k)
