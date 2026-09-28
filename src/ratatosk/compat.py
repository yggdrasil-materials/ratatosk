"""
Where the scientific stack changed under us, handled in one place.

Why this module exists
----------------------
`np.trapz` was REMOVED in numpy 2.0 and `np.trapezoid` does not exist before
it. The codebase had all three responses to that at once: nine direct calls to
`np.trapezoid` (numpy 2 only), a `trapz` alias imported unconditionally from
numpy (crashes on numpy 1), and a careful `getattr` fallback in two places —
one of which sat twenty-five lines AFTER the crashing import, so it could
never run.

The result was an undeclared hard dependency on numpy >= 2. On a numpy 1.26
Anaconda environment, which is still common, Cell 1 died with
`ImportError: cannot import name 'trapezoid'` and nothing in the notebook
explained why.

One name, defined once, used everywhere. Nothing else in the package should
reach for `np.trapz` or `np.trapezoid` directly.

The environment check
---------------------
`check_environment()` reports what is installed and warns where a version is
below what this pipeline was developed and tested against. It WARNS rather
than raising: a floor is a statement about what has been tested, not a
certainty that older will fail, and refusing to run on an untested version
helps nobody at 6pm. What it will not do is let a version problem surface
three cells later as an unrelated error.
"""

from __future__ import annotations

import os
import sys

import numpy as np

__all__ = ["trapezoid", "check_environment", "TESTED_AGAINST",
           "PYTHON_MINIMUM"]

# `np.trapezoid` on numpy >= 2, `np.trapz` before it. Identical signature and
# identical results; only the name changed.
trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")

# What this pipeline has actually been run against. Not a hard requirement —
# see the module docstring — but the numbers a bug report should quote.
TESTED_AGAINST = {
    "numpy": "1.24",
    "scipy": "1.10",
    "pandas": "2.0",
    "matplotlib": "3.7",
    "lmfit": "1.2",
}
PYTHON_MINIMUM = (3, 9)


def _tuple(version):
    """'2.4.4rc1' -> (2, 4, 4). Anything unparseable sorts as newest."""
    out = []
    for part in str(version).split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        out.append(int(digits))
    return tuple(out) if out else (999,)


def installed_versions():
    """`{package: version string}` for everything the pipeline imports."""
    import importlib
    found = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "lmfit",
                 "seaborn", "joblib", "openpyxl", "python_calamine"):
        try:
            mod = importlib.import_module(name)
            found[name] = getattr(mod, "__version__", "present")
        except Exception:                              # noqa: BLE001
            found[name] = None
    return found


def numerical_backend():
    """
    The platform and the linear-algebra library the fits were solved with.

    WHY THIS IS PROVENANCE AND NOT TRIVIA. Two runs of the SAME build on the
    SAME file give the same answer to about six figures and not beyond it: the
    least-squares solver reaches a different local minimum depending on the
    order floating-point reductions happen in, which is set by the BLAS and by
    how many threads it uses. Measured on NMC111 — twelve to twenty-four free
    parameters per half-cycle, and a documented non-identifiability between
    overlapping components — one machine and another agreed on every cycling
    number, every reference voltage and every coherence verdict, and disagreed
    on `reliable` for nine of 164 components and on one drift rate by 12 mV
    per cycle. Everything the coherence audit certifies matched; everything
    that moved was already quarantined by it.

    That is the right outcome, but it is only checkable if the run says what
    it ran on. The manifest recorded Python and package versions, which are
    identical across those two machines and therefore explain nothing.
    """
    import platform as _pf
    out = dict(system=_pf.system(), release=_pf.release(),
               machine=_pf.machine(), processor=_pf.processor() or "",
               python_implementation=_pf.python_implementation())
    # The BLAS numpy is linked against. `__config__` has changed shape more
    # than once, so every access is guarded and a failure records that it
    # could not be read rather than dropping the field.
    try:
        import numpy as _np
        _cfg = getattr(_np, "__config__", None)
        _show = getattr(_cfg, "show", None)
        _blas = None
        if _show is not None:
            try:                                  # numpy >= 1.25
                _d = _show(mode="dicts")
                _b = (((_d or {}).get("Build Dependencies") or {})
                      .get("blas") or {})
                _blas = {k: _b.get(k) for k in ("name", "version")
                         if _b.get(k)} or None
            except Exception:                     # noqa: BLE001
                _blas = None
        if _blas is None:                         # older layouts
            for _attr in ("blas_opt_info", "blas_ilp64_opt_info"):
                _i = getattr(_cfg, _attr, None)
                if isinstance(_i, dict) and _i.get("libraries"):
                    _blas = {"name": ",".join(_i["libraries"])}
                    break
        out["blas"] = _blas or "not reported by numpy"
    except Exception:                             # noqa: BLE001
        out["blas"] = "could not be read"
    # Thread counts change the reduction ORDER even on one machine, so they
    # belong beside the library name.
    out["threads"] = {v: os.environ[v] for v in
                      ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                       "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
                      if v in os.environ} or "not set (library default)"
    return out


def check_environment(verbose=True):
    """
    Say what is installed, and warn about anything below what was tested.

    Returns a dict the manifest can record verbatim, so a run's provenance
    carries the environment it was produced in rather than leaving it to be
    reconstructed later.
    """
    found = installed_versions()
    old = {n: v for n, v in found.items()
           if v and n in TESTED_AGAINST
           and _tuple(v) < _tuple(TESTED_AGAINST[n])}
    missing = [n for n in ("numpy", "scipy", "pandas", "matplotlib", "lmfit")
               if not found.get(n)]
    py = tuple(sys.version_info[:2])

    state = dict(python=".".join(str(x) for x in sys.version_info[:3]),
                 packages=found, below_tested=old, missing=missing,
                 python_below_minimum=py < PYTHON_MINIMUM,
                 # See `numerical_backend`. Package versions are identical
                 # across two machines that disagree in the sixth figure;
                 # this is the field that says why.
                 numerical_backend=numerical_backend(),
                 # Recorded so a run says whether warnings were being hidden
                 # while it ran. A suppressed FutureWarning is exactly the
                 # kind that changes a number between pandas releases.
                 future_warnings_suppressed=True)

    if verbose:
        print(f"  Python {state['python']}   "
              + "   ".join(f"{n} {found[n]}" for n in
                           ("numpy", "pandas", "scipy", "lmfit")
                           if found.get(n)))
        # SAID ON EVERY RUN, because it is the field that explains why two
        # machines running the same build on the same file agree to six
        # figures and not beyond. See `numerical_backend`.
        _nb = state["numerical_backend"]
        _bl = _nb.get("blas")
        _bl = (f"{_bl.get('name')} {_bl.get('version') or ''}".strip()
               if isinstance(_bl, dict) else str(_bl))
        print(f"  {_nb['system']} {_nb['machine']}   linear algebra: {_bl}")
        if found.get("python_calamine"):
            print("  calamine present — Excel reading is 5-7x faster")
        else:
            print("  calamine NOT present — `pip install python-calamine` "
                  "makes loading 5-7x faster")
        if not found.get("seaborn"):
            print("  seaborn NOT present — the key-cycle voltage profile "
                  "figure will fail; everything else is unaffected")
        if py < PYTHON_MINIMUM:
            print(f"  *** Python {state['python']} is below the "
                  f"{'.'.join(str(x) for x in PYTHON_MINIMUM)} this was "
                  f"written for. ***")
        for n in missing:
            print(f"  *** {n} is not installed — the run will fail. ***")
        for n, v in sorted(old.items()):
            print(f"  *** {n} {v} is older than the {TESTED_AGAINST[n]} this "
                  f"was tested against; results are not guaranteed. ***")
    return state
