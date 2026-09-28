"""
Synthetic dQ/dV data with a known right answer.

Why this module exists, and why it is first
-------------------------------------------
Every conclusion drawn from Ratatosk so far has been inference from cells whose
true answer nobody knows. That is how a polynomial baseline came to absorb 84%
of a half-cycle's capacity at R2 = 0.98 without anything noticing for the life
of the tool.

This module emits curves whose peak areas, background and noise are set by us.
It is the only non-circular way to test whether a change to the fitting or the
metrics is an improvement.

Design rule: the ground truth is defined by INTEGRATION, not by the fitting
model. `true_areas` is the analytic integral of each component, so a test that
recovers it is not merely reproducing its own assumptions.

Cases provided
--------------
    two_phase        one sharp peak on a flat background   (an LTO analogue)
    solid_solution   a broad envelope, no discrete peaks   (an NMC111 analogue)
    mixed            discrete peaks plus a continuum       (a P3 analogue)
    degenerate       a broad peak the baseline can mimic — MUST be detected
                     as indeterminate, not fitted confidently

The degenerate case is the important one. A metric that cannot flag it is not
doing its job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .compat import trapezoid

__all__ = ["SyntheticHalfCycle", "make_case", "make_series", "CASES",
           "pseudo_voigt"]


# ---------------------------------------------------------------------------
# The peak shape, written out rather than imported, so that a test using it is
# not implicitly testing lmfit's parameterisation as well.
# ---------------------------------------------------------------------------

def pseudo_voigt(v, centre, sigma, fraction, area):
    """
    Pseudo-Voigt normalised so that its integral over (-inf, inf) is `area`.

    fraction = 0 -> pure Gaussian, 1 -> pure Lorentzian, matching lmfit's
    convention so the recovered parameters are directly comparable.
    """
    sigma_g = sigma / np.sqrt(2 * np.log(2))
    gauss = np.exp(-((v - centre) ** 2) / (2 * sigma_g ** 2)) / (
        sigma_g * np.sqrt(2 * np.pi))
    lorentz = (sigma / np.pi) / ((v - centre) ** 2 + sigma ** 2)
    return area * ((1 - fraction) * gauss + fraction * lorentz)


@dataclass
class SyntheticHalfCycle:
    """A generated curve together with everything that made it."""
    voltage: np.ndarray
    dqdv: np.ndarray                 # what the "instrument" reports
    dqdv_clean: np.ndarray           # before quantisation and noise
    capacity: float                  # analytic: sum of areas + background area
    true_areas: np.ndarray           # analytic integral of each component
    true_centres: np.ndarray
    true_sigmas: np.ndarray
    true_fractions: np.ndarray
    background_area: float
    case: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def true_closure(self) -> float:
        """The fraction of the curve's area carried by discrete components."""
        return float(self.true_areas.sum() / (
            self.true_areas.sum() + self.background_area))


# ---------------------------------------------------------------------------

def _background(v, kind, scale):
    """
    Backgrounds are described by shape, not by polynomial degree, so that a
    test is not rigged in favour of the polynomial the fitter happens to use.
    """
    x = (v - v.min()) / (v.max() - v.min())
    if kind == "flat":
        return np.full_like(v, scale)
    if kind == "linear":
        return scale * (0.5 + x)
    if kind == "sigmoid":                    # solid-solution-like
        return scale / (1 + np.exp(-12 * (x - 0.45)))
    if kind == "dome":                       # broad, polynomial-mimicable
        return scale * np.exp(-((x - 0.5) ** 2) / (2 * 0.28 ** 2))
    raise ValueError(f"unknown background {kind!r}")


def make_case(case="two_phase", *, n_points=400, v_range=None,
              lsb_mV=0.1, noise_frac=0.01, seed=0,
              quantise=True) -> SyntheticHalfCycle:
    """
    Build one synthetic half-cycle.

    lsb_mV       voltage least-significant bit. The P3 and LTO exports both
                 record to 0.1 mV, which is what produces the +/-dQ/dV comb;
                 set to 0 to disable quantisation and isolate its effect.
    noise_frac   Gaussian noise as a fraction of the curve's maximum.
    """
    spec = CASES[case]
    lo, hi = v_range or spec["v_range"]
    rng = np.random.default_rng(seed)

    v = np.linspace(lo, hi, n_points)
    if quantise and lsb_mV > 0:
        v = np.round(v / (lsb_mV / 1000.0)) * (lsb_mV / 1000.0)

    centres = np.asarray(spec["centres"], float)
    sigmas = np.asarray(spec["sigmas"], float)
    fracs = np.asarray(spec["fractions"], float)
    areas = np.asarray(spec["areas"], float)

    clean = np.zeros_like(v)
    for c, s, f, a in zip(centres, sigmas, fracs, areas):
        clean += pseudo_voigt(v, c, s, f, a)

    bg = _background(v, spec["background"], spec["background_scale"])
    clean = clean + bg

    y = clean.copy()
    if noise_frac > 0:
        y = y + rng.normal(0.0, noise_frac * np.abs(clean).max(), size=v.size)

    # Background area by integration over the window, so the reported capacity
    # is the area actually present rather than an analytic idealisation.
    bg_area = float(trapezoid(bg, v))
    # Component areas are the analytic integrals, minus the tails that fall
    # outside the window — otherwise `true_closure` would be unreachable.
    inwin = np.array([float(trapezoid(pseudo_voigt(v, c, s, f, a), v))
                      for c, s, f, a in zip(centres, sigmas, fracs, areas)])

    return SyntheticHalfCycle(
        voltage=v, dqdv=y, dqdv_clean=clean,
        capacity=float(inwin.sum() + bg_area),
        true_areas=inwin, true_centres=centres, true_sigmas=sigmas,
        true_fractions=fracs, background_area=bg_area, case=case,
        meta=dict(lsb_mV=lsb_mV, noise_frac=noise_frac, seed=seed,
                  n_points=n_points, background=spec["background"],
                  note=spec["note"]),
    )


# ---------------------------------------------------------------------------
# The cases. Voltages and widths are chosen to resemble the real systems so
# that a failure here is informative about a failure there.
# ---------------------------------------------------------------------------

CASES = {
    "two_phase": dict(
        v_range=(1.35, 1.75), centres=[1.529], sigmas=[0.008],
        fractions=[0.5], areas=[120.0],
        background="flat", background_scale=6.0,
        note="LTO analogue: one sharp two-phase peak. Closure should be high "
             "and insensitive to baseline degree.",
    ),
    "solid_solution": dict(
        v_range=(3.0, 4.5), centres=[3.80], sigmas=[0.22],
        fractions=[0.3], areas=[60.0],
        background="sigmoid", background_scale=70.0,
        note="NMC111 analogue: a broad envelope on a smooth rising "
             "background. Closure should be LOW and, because the two are "
             "nearly the same function, poorly determined.",
    ),
    "mixed": dict(
        v_range=(2.0, 4.2), centres=[3.22, 3.30, 3.64, 3.70],
        sigmas=[0.030, 0.028, 0.035, 0.040], fractions=[0.4, 0.4, 0.5, 0.5],
        areas=[18.0, 26.0, 22.0, 14.0],
        background="linear", background_scale=22.0,
        note="P3 analogue: several discrete peaks over a continuum. Closure "
             "should be intermediate and reasonably well determined.",
    ),
    "tracking": dict(
        v_range=(3.05, 4.00), centres=[3.20, 3.42, 3.44, 3.85],
        sigmas=[0.030, 0.035, 0.012, 0.030], fractions=[0.4, 0.4, 0.4, 0.4],
        areas=[30.0, 34.0, 9.0, 22.0],
        background="linear", background_scale=14.0,
        note="The TRACKING ground truth: three well-separated peaks plus one "
             "genuine shoulder 20 mV from its parent and a third its width. "
             "Deliberately RESOLVABLE, unlike `mixed`, whose 3.64/3.70 pair "
             "sit 60 mV apart with sigma 35-40 mV and therefore form one "
             "visible feature that no detector can or should split. A "
             "tracking tolerance quoted on an unresolvable pair would be a "
             "tolerance on a coin toss. The window is kept tight around the "
             "features for the same reason: 400 mV of pure background at "
             "each end is 400 mV in which a detector can only find noise, "
             "and every spurious component it seeds there costs four "
             "correlated parameters and seconds of optimiser time without "
             "telling anyone anything.",
    ),
    "degenerate": dict(
        v_range=(3.0, 4.5), centres=[3.75], sigmas=[0.30],
        fractions=[0.0], areas=[100.0],
        background="dome", background_scale=40.0,
        note="A broad Gaussian on a dome. A polynomial can absorb either. Any "
             "honest metric MUST report this as indeterminate rather than "
             "quoting a closure.",
    ),
}


# ---------------------------------------------------------------------------
# A SERIES with a known trajectory
# ---------------------------------------------------------------------------
# Bit-identity against 1.8.7 is the wrong regression target for tracking, delta
# V and attribution: it would enshrine 1.8.7's non-stable sort, its inert
# reference-cycle selector and its uncoupled shoulders as the definition of
# correct. The right target is a series whose answer we set — peaks that drift
# by a known number of millivolts per cycle and fade by a known percentage —
# and a stated tolerance on recovering it.
#
# The claim the test makes is a scientific one and should read like one:
#
#   "On synthetic data with 2% noise, tracked peak centres are recovered to
#    within +/-X mV and areas to within +/-Y% over N cycles of D mV/cycle
#    drift."
#
# Note what is NOT varied: the mixing fraction is held at its true value
# across the series, because a shape that wanders cycle to cycle is exactly
# the artefact `fitting.calibrate_shape` exists to prevent, and a generator
# that reproduced it would be testing the artefact rather than the pipeline.

def make_series(case="mixed", *, n_cycles=50, drift_mV_per_cycle=2.0,
                fade_pct_per_cycle=0.2, broadening_pct_per_cycle=0.0,
                noise_frac=0.02, seed=0, **kw):
    """
    A run of half-cycles with a KNOWN trajectory, and that trajectory.

    Returns `(cycles, truth)`:

        cycles : {cycle_number: SyntheticHalfCycle}, numbered from 1
        truth  : DataFrame-ready dict of lists with one row per
                 (cycle, component): cycle, component, centre, sigma, area

    `drift_mV_per_cycle` moves every centre up by that much per cycle (the
    sign convention of a polarising cell). `fade_pct_per_cycle` is compound,
    so the area after n cycles is `area * (1 - fade/100) ** (n - 1)` — fade is
    multiplicative in every published capacity model and a linear ramp would
    make the test easier than reality.
    """
    spec = CASES[case]
    base_c = np.asarray(spec["centres"], float)
    base_s = np.asarray(spec["sigmas"], float)
    base_a = np.asarray(spec["areas"], float)

    cycles, rows = {}, []
    for n in range(1, int(n_cycles) + 1):
        k = n - 1
        c = base_c + k * drift_mV_per_cycle / 1000.0
        a = base_a * (1.0 - fade_pct_per_cycle / 100.0) ** k
        sg_ = base_s * (1.0 + broadening_pct_per_cycle / 100.0) ** k
        one = dict(spec)
        one["centres"], one["sigmas"], one["areas"] = (list(c), list(sg_),
                                                       list(a))
        CASES[f"_series_{case}"] = one
        try:
            # A DIFFERENT NOISE DRAW PER CYCLE. One draw reused would let a
            # tracker lock onto the noise and report a tolerance nobody can
            # reproduce on real data.
            hc = make_case(f"_series_{case}", noise_frac=noise_frac,
                           seed=seed + n, **kw)
        finally:
            CASES.pop(f"_series_{case}", None)
        hc.case = case
        hc.meta["cycle"] = n
        cycles[n] = hc
        for i in range(len(c)):
            rows.append(dict(cycle=n, component=i, centre=float(c[i]),
                             sigma=float(sg_[i]),
                             area=float(hc.true_areas[i])))
    truth = {k: [r[k] for r in rows] for k in rows[0]}
    return cycles, truth
