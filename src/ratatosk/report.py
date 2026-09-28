"""
The first thing a newcomer should open.

A finished run is forty-six files across seven folders. That is the right way
to STORE the output and the wrong way to hand it to somebody: "here are 45
well-organised files, good luck" is not a briefing. This module writes two more
files at the top of the run folder, above the numbered directories:

    START_HERE.html   self-contained, images embedded, safe to email
    START_HERE.md     the same content as text, to paste into a report

Both say the same four things, in this order:

    1. What this cell is, and what it did
    2. What went wrong, if anything, and on which cycles
    3. What Ratatosk declined to tell you, and why
    4. Four figures, in order, with what to look for in each

Everything else stays exactly where it is. This is a way in, not a replacement.

Why the third section exists
----------------------------
The withheld numbers are the ones a newcomer is most likely to misread, because
absence looks like an oversight rather than a decision. A capacity share left
blank because the closure interval was 0.46 wide is a finding; found as a gap
in a spreadsheet three weeks later it is a bug report.
"""

from __future__ import annotations

import base64
import os
import html
import re

import numpy as np
import pandas as pd

from .analyse import (
    cycle_column,
    cell_integrity_verdict,
    UNATTRIBUTED_WITHHOLD_ABOVE,
    AREA_GROWTH_HEADROOM_PCT,
    AREA_LOWER_BOUND_REPORTABLE,
)
from .style import CAPACITY_COLUMN_ALIASES


__all__ = [
    "build_report",
    "write_report",
    "build_run_summary",
    "write_run_summary",
    "FIGURE_GUIDE",
]


# Which four, and what to say about each. Ordered as a reader should meet them:
# does the cell work, is it dying, what is the mechanism, is the mechanism
# moving. Each entry is (folder, filename suffix, title, what to look for).
FIGURE_GUIDE = [
    (
        "1_cycling",
        "_cycle_life.png",
        "Does the cell work?",
        "Capacity and coulombic efficiency against cycle number. CE should sit "
        "at or just under 100% from a few cycles in. A first cycle well below "
        "that is formation; a later dip is charge going somewhere it should not, "
        "and section 2 above says where.",
    ),
    # RUN-WIDE. `capacity_retention` draws ONE figure for the whole run, so
    # there is no per-cell file to find; `organise_run` mirrors it into each
    # cell folder as `comparative_capacity_retention_vs_cycle.png`. `_find`
    # skips `comparative_` files on purpose — a group figure must not stand in
    # where a cell's own was asked for — so this tile resolved to nothing and
    # every START_HERE report in every run has said "(figure not found)"
    # against question 2 of 4.
    (
        "1_cycling",
        "capacity_retention_vs_cycle.png",
        "Is it dying, and how fast?",
        "Discharge capacity as a percentage of the reference cycle. A straight "
        "decline is ordinary ageing. A cliff is an event — cross-reference the "
        "cycle number against section 2.",
        True,
    ),
    (
        "4_dqdv",
        "_dQdV_waterfall_Discharge.png",
        "What is the mechanism?",
        "Every cycle's differential capacity, stacked. Each peak is a redox "
        "process. A peak that shifts is polarising; one that broadens is losing "
        "kinetics; one that shrinks is losing the material behind it. This is "
        "the figure that says WHY the capacity curve does what it does.",
    ),
    (
        "6_descriptors",
        "_peak_trends_Discharge.png",
        "Is the mechanism moving?",
        "Each tracked peak's centre and area against cycle number. A redox peak "
        "at C/10 should drift a few mV per cycle. Tens of mV per cycle is a "
        "fitting artefact, not chemistry — the coherence table in "
        "5_peak_fitting says which peaks passed that test.",
    ),
]

_CSS = """
:root{--ink:#1a1a1a;--soft:#5a5a5a;--rule:#dcdcdc;--bg:#ffffff;
      --ok:#0a7d3c;--warn:#b06000;--bad:#a4262c;--tint:#f5f7f9;--accent:#0072B2}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:940px;margin:0 auto;padding:48px 28px 96px}
h1{font-size:30px;line-height:1.2;margin:0 0 4px;letter-spacing:-.02em}
.sub{color:var(--soft);font-size:15px;margin:0 0 32px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.09em;
   color:var(--soft);font-weight:700;margin:44px 0 14px;
   padding-bottom:8px;border-bottom:1px solid var(--rule)}
h3{font-size:18px;margin:0 0 6px;letter-spacing:-.01em}
p{margin:0 0 12px}
.lede{font-size:18px;line-height:1.55}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
      gap:14px;margin:18px 0 8px}
.stat{background:var(--tint);border-radius:10px;padding:14px 16px}
.stat .k{font-size:12px;color:var(--soft);text-transform:uppercase;
         letter-spacing:.06em}
.stat .v{font-size:24px;font-weight:650;letter-spacing:-.02em;margin-top:2px}
.stat .n{font-size:12px;color:var(--soft);margin-top:2px}
.flag{border-left:3px solid var(--rule);padding:12px 16px;margin:12px 0;
      background:var(--tint);border-radius:0 8px 8px 0}
.flag.ok{border-color:var(--ok)} .flag.warn{border-color:var(--warn)}
.flag.bad{border-color:var(--bad)}
.flag b{display:block;margin-bottom:3px}
.fig{margin:30px 0 38px}
.fig img{width:100%;height:auto;border:1px solid var(--rule);border-radius:8px;
         background:#fff}
.fig .cap{color:var(--soft);font-size:14.5px;margin-top:10px}
.n{color:var(--soft);font-size:14px}
table{border-collapse:collapse;width:100%;font-size:14.5px;margin:8px 0 4px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--rule)}
th{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--soft)}
td.num{text-align:right;font-variant-numeric:tabular-nums}
code{background:var(--tint);padding:1px 5px;border-radius:4px;font-size:13.5px}
footer{margin-top:56px;padding-top:16px;border-top:1px solid var(--rule);
       color:var(--soft);font-size:13px}
@media (prefers-color-scheme:dark){
  :root{--ink:#e8e8e8;--soft:#a0a0a0;--rule:#333;--bg:#141414;--tint:#1e1e1e;
        --ok:#4ec97e;--warn:#e0a04a;--bad:#e07a80}
  .fig img{background:#fff}
}
"""


# THE FINDINGS ARE WRITTEN IN MARKDOWN. Both renderers share them, and the
# HTML one only escaped — so the page the docstring calls "safe to email"
# printed literal `**mixed**` and backticks, a dozen times per cell. Applied
# AFTER `_esc`, so the input is already HTML-safe and this only re-introduces
# the two tags the findings actually use.
def _md_inline(t):
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", str(t), flags=re.S)
    return re.sub(r"`([^`]+?)`", r"<code>\1</code>", t)


_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _embed(path):
    """Inline a figure as a data URI, with the MIME type its extension says."""
    mime = _MIME.get(os.path.splitext(path)[1].lower())
    if mime is None:
        # TIFF and PDF exports cannot be shown in a browser. Say so in the
        # report rather than emitting a data URI a browser will not render.
        return None
    try:
        with open(path, "rb") as fh:
            return f"data:{mime};base64," + base64.b64encode(fh.read()).decode("ascii")
    except OSError:
        return None


def _esc(x):
    """Every field interpolated into the HTML goes through this.

    Compositions legitimately contain characters that are markup: `Na0.67
    [Ni0.25Mn0.75]O2` is fine, `<Ni>` and `A&B` are not, and an apostrophe
    inside an alt attribute closes it.
    """
    return html.escape("" if x is None else str(x), quote=True)


def _md_cell(x):
    """A value safe to drop into a markdown TABLE cell."""
    return ("" if x is None else str(x)).replace("|", "\\|").replace("\n", " ")


def _rel(path, start):
    """A relative path a markdown viewer will follow.

    `os.path.relpath` returns backslashes on Windows, and a backslash in a
    markdown link is an escape character, so every embedded figure link was
    dead on the platform this is actually run on.
    """
    return os.path.relpath(path, start).replace(os.sep, "/")


def _trailing_partial(t, dcol, in_progress):
    """The cycle whose export caught a half-cycle mid-flight, from `in_progress`.

    `analyse.half_cycles_in_progress` is the authority and the only one:
    it asks whether the half-cycle reached the cut-off VOLTAGE the cycler
    terminates on. There is no capacity fallback, because capacity cannot
    answer that question — on P3 cell B an 80%-of-the-previous rule fires
    15 times on a cell that is simply dying, and misses a half-cycle caught
    four fifths of the way through.

    Returns (cycle, step, previous_cycle, capacity_so_far) or None.
    """
    if not in_progress:
        return None
    cyc, step = sorted(in_progress)[0]
    prev = val = None
    if dcol is not None and "Cycle" in t:
        d = t.dropna(subset=["Cycle"])
        earlier = d[d["Cycle"] < cyc].dropna(subset=[dcol])
        if len(earlier):
            prev = int(earlier["Cycle"].iloc[-1])
        here = pd.to_numeric(d.loc[d["Cycle"] == cyc, dcol], errors="coerce")
        if len(here) and np.isfinite(here.iloc[0]):
            val = float(here.iloc[0])
    return cyc, step, prev, val


def _find(folder, suffix, name, run_wide=False):
    """The figure for this dataset, whether or not its filename is prefixed.

    `run_wide` says this guide entry names a figure that is drawn ONCE for
    the whole run rather than per cell — capacity retention is the only one.
    Without it the comparative-skip below refuses the only candidate there
    is, and the tile renders "(figure not found)".
    """
    if not os.path.isdir(folder):
        return None
    exact = os.path.join(folder, f"{name}{suffix}")
    if os.path.isfile(exact):
        return exact
    plain = os.path.join(folder, suffix)
    if os.path.isfile(plain):
        return plain
    if run_wide:
        mirrored = os.path.join(folder, f"comparative_{suffix}")
        if os.path.isfile(mirrored):
            return mirrored
    for f in sorted(os.listdir(folder)):
        # A run-wide figure is mirrored into every cell folder with a
        # 'comparative_' prefix. Matching one here would show the group
        # figure where the cell's own was asked for — unless the guide has
        # said this entry IS the group figure.
        if f.startswith("comparative_") and not run_wide:
            continue
        if f.endswith(suffix.lstrip("_")) or f.endswith(suffix):
            return os.path.join(folder, f)
    return None


def _facts(
    name,
    params,
    integrity,
    cycle_table,
    fit_limit,
    detection,
    tracking,
    attribution,
    closure,
    profile,
    in_progress=None,
    coherence=None,
    resolvability=None,
    parameters=None,
):
    """Everything both renderers need, computed once."""
    f = {"name": name, "params": params or {}}
    # THE MECHANISM VERDICT. `quality.assess_resolvability` decides, before a
    # single production fit is run, what produced this curve and therefore
    # which model to fit it with — and until 1.9.0.47 it said so only in the
    # console. It is the single most important thing this build does
    # differently from 1.8.7 and neither document mentioned it, which is the
    # exact defect class the reporting audit was about: a verdict computed and
    # not carried to where a person reads.
    f["resolvability"] = dict(resolvability or {})
    # THE FIT-QUALITY NUMBERS, one row per half-cycle. R2 is dominated by the
    # bulk of a curve; these are the three ways a fit goes wrong that it
    # cannot carry. See `fitting.fit_quality`.
    f["fit_quality"] = {}
    if parameters is not None and not getattr(parameters, "empty", True):
        _pp = parameters
        _cols = [
            "height_ratio",
            "overshoot",
            "max_residual_frac",
            "residual_runs_z",
            "r_squared",
        ]
        if all(c in getattr(_pp, "columns", []) for c in _cols):
            _hc = _pp.groupby(["step", "cycle"])[_cols].first().reset_index()
            for _st, _g in _hc.groupby("step"):
                f["fit_quality"][str(_st)] = {c: float(_g[c].median()) for c in _cols}
                f["fit_quality"][str(_st)]["n"] = int(len(_g))
                f["fit_quality"][str(_st)]["height_worst"] = float(
                    _g["height_ratio"].min()
                )
                f["fit_quality"][str(_st)]["height_tallest"] = float(
                    _g["height_ratio"].max()
                )
                # ...and WHERE, because "one half-cycle is at 0.57" is only
                # actionable with a cycle number beside it.
                _hh = _g.dropna(subset=["height_ratio"])
                if not _hh.empty:
                    f["fit_quality"][str(_st)]["height_worst_cycle"] = int(
                        _hh.loc[_hh["height_ratio"].idxmin(), "cycle"]
                    )
                    f["fit_quality"][str(_st)]["height_tallest_cycle"] = int(
                        _hh.loc[_hh["height_ratio"].idxmax(), "cycle"]
                    )
                    f["fit_quality"][str(_st)]["n_short"] = int(
                        (_hh["height_ratio"] < 0.85).sum()
                    )
                    f["fit_quality"][str(_st)]["n_tall"] = int(
                        (_hh["height_ratio"] > 1.15).sum()
                    )
                _rr = _g.dropna(subset=["r_squared"])
                if not _rr.empty:
                    f["fit_quality"][str(_st)]["r2_worst"] = float(
                        _rr["r_squared"].min()
                    )
                    f["fit_quality"][str(_st)]["r2_worst_cycle"] = int(
                        _rr.loc[_rr["r_squared"].idxmin(), "cycle"]
                    )
    # THE LINESHAPE. A sharp two-phase peak is asymmetric and until 1.9.0.51
    # the model could not be; where the split shape was fitted, the ratio it
    # settled on is a measurement and belongs on the page — including when it
    # settled on a bound, which is not one.
    f["lineshape"] = {}
    if parameters is not None and not getattr(parameters, "empty", True):
        _pp = parameters
        if all(
            c in getattr(_pp, "columns", [])
            for c in ("asymmetry_k", "asymmetry_at_bound")
        ):
            _hc = (
                _pp.groupby(["step", "cycle"])[["asymmetry_k", "asymmetry_at_bound"]]
                .first()
                .reset_index()
            )
            for _st, _g in _hc.groupby("step"):
                _k = _g["asymmetry_k"].dropna()
                if _k.empty:
                    continue
                # ITEM 35. A median ratio of "19.3x" was quoted over a range
                # of 0.05-3.12 — which crosses 1.0, so one half-cycle leans
                # the other way and the median states a direction the data
                # does not hold. Count how many agree with the median's sign.
                _same = int(((_k > 1.0) == (_k.median() > 1.0)).sum())
                f["lineshape"][str(_st)] = dict(
                    n=int(len(_g)),
                    k=float(_k.median()),
                    k_lo=float(_k.min()),
                    k_hi=float(_k.max()),
                    same_sign=_same,
                    n_k=int(len(_k)),
                    at_bound=int(_g["asymmetry_at_bound"].sum()),
                    clipped=int(_g["asymmetry_k_clipped"].sum())
                    if "asymmetry_k_clipped" in _g
                    else 0,
                    split=bool((_k - 1.0).abs().median() > 0.02),
                )
            # ...and how often the NARROW side of the LARGEST component in a
            # half-cycle sits on the width floor. A split lineshape can put
            # one flank below what the histogram samples, and on LTO
            # discharge it does so on the component carrying all of the area.
        # HOW MANY BINS THE WIDTH IS MEASURED ACROSS. A ratio of two widths
        # is only as good as the widths, and on a sharp profile they are read
        # off three or four bins. See `signal.HISTOGRAM_MIN_OCCUPANCY` for the
        # sweep: on LTO the fitted FWHM tracks the bin width almost linearly
        # over a factor of ten, so it is the histogram's width, not the peak's.
        if {"fwhm", "sample_mV"} <= set(_pp.columns):
            _b = (
                pd.to_numeric(_pp["fwhm"], errors="coerce")
                * 1000.0
                / pd.to_numeric(_pp["sample_mV"], errors="coerce")
            )
            for _st, _gb in _pp.assign(_bins=_b).groupby("step"):
                if str(_st) not in f["lineshape"]:
                    continue
                _v = _gb["_bins"].replace([np.inf, -np.inf], np.nan).dropna()
                if _v.empty:
                    continue
                f["lineshape"][str(_st)].update(
                    fwhm_bins=float(_v.median()),
                    n_comp=int(len(_gb)),
                    undersampled=int(_gb["width_undersampled"].fillna(False).sum())
                    if "width_undersampled" in _gb
                    else 0,
                )
        if {"at_sigma_floor", "amplitude_area"} <= set(_pp.columns):
            _dom = (
                _pp.assign(_a=_pp["amplitude_area"].abs())
                .sort_values("_a")
                .groupby(["step", "cycle"])
                .last()
                .reset_index()
            )
            for _st, _g2 in _dom.groupby("step"):
                if str(_st) in f["lineshape"]:
                    f["lineshape"][str(_st)]["dominant_at_floor"] = int(
                        _g2["at_sigma_floor"].fillna(False).sum()
                    )
    # HOW MANY COMPONENTS DID EACH HALF-CYCLE GET, AND HOW MANY DID THE
    # REFERENCE CYCLE GET? Components per half-cycle swing 1-5 on NMC111 with
    # no pattern, and the reference cycle got the THINNEST fit in the dataset
    # (2) with a roster of 7 hung off it — every tracked feature, every drift
    # rate and the whole attribution basis derived from the one half-cycle
    # the model described least well. The reference cycle is chosen on the
    # cell's INTEGRITY, which is the right criterion and should not be traded
    # for a fatter fit; but a reader is entitled to know when the basis is
    # thin, and nothing said so.
    # SET IT BEFORE IT IS READ. `f["reference_cycle"]` was assigned ~270
    # lines below this block, so `_ref` here was ALWAYS None and the whole
    # warn branch — the one the paragraph above exists to explain, saying the
    # basis is the thinnest fit in the dataset — could never fire on any
    # dataset. The heading was permanently the benign one.
    f["reference_cycle"] = getattr(detection, "reference_cycle", None)
    f["component_census"] = {}
    if parameters is not None and not getattr(parameters, "empty", True):
        _pp = parameters
        if {"cycle", "step"} <= set(getattr(_pp, "columns", [])):
            _n = _pp.groupby(["step", "cycle"]).size().rename("n").reset_index()
            _ref = f.get("reference_cycle")
            for _st, _g in _n.groupby("step"):
                _med = float(_g["n"].median())
                _refrow = _g[_g["cycle"] == _ref] if _ref is not None else _g.iloc[0:0]
                f["component_census"][str(_st)] = dict(
                    n_half_cycles=int(len(_g)),
                    lo=int(_g["n"].min()),
                    hi=int(_g["n"].max()),
                    median=_med,
                    reference_cycle=(int(_ref) if _ref is not None else None),
                    reference_n=(
                        int(_refrow.iloc[0]["n"]) if not _refrow.empty else None
                    ),
                )
    # ITEM 37. A HALF-CYCLE WITH NO PRIMARY PEAK AT ALL. `primaries_only`
    # drops shoulders, but guards with "keep them if there are no primaries" —
    # so when the picker finds no resolved maximum the whole half-cycle is
    # described by features it called shoulders. 19 of 60 on NMC111, every one
    # a discharge, from cycle 4 onward as the curve flattens, and the two
    # worst fits in that run were among them. Nothing said so.
    f["shoulder_only"] = {}
    if parameters is not None and not getattr(parameters, "empty", True):
        _pp = parameters
        if {"cycle", "step", "is_shoulder"} <= set(getattr(_pp, "columns", [])):
            _so = (
                _pp.groupby(["step", "cycle"])["is_shoulder"]
                .all()
                .reset_index(name="all_shoulder")
            )
            for _st, _g in _so.groupby("step"):
                _bad = _g[_g["all_shoulder"]]
                if len(_bad):
                    f["shoulder_only"][str(_st)] = dict(
                        n=int(len(_bad)),
                        total=int(len(_g)),
                        first=int(_bad["cycle"].min()),
                        last=int(_bad["cycle"].max()),
                    )
    # HOW MUCH DATA EACH FIT HAD. Reduced to a handful of numbers here rather
    # than carrying the whole parameter table into the facts dict, which both
    # renderers copy.
    f["identifiability"] = {}
    if parameters is not None and not getattr(parameters, "empty", True):
        _pp = parameters
        _need = ("n_points", "nvarys", "points_per_parameter")
        _key = [c for c in ("cycle", "step") if c in getattr(_pp, "columns", [])]
        if _key and all(c in _pp.columns for c in _need):
            _H = _pp.drop_duplicates(subset=_key)
            _ppp = pd.to_numeric(_H["points_per_parameter"], errors="coerce").dropna()
            if not _ppp.empty:
                _npt = pd.to_numeric(_H["n_points"], errors="coerce").dropna()
                _nvy = pd.to_numeric(_H["nvarys"], errors="coerce").dropna()
                _cpl = pd.to_numeric(
                    _H.get("n_shoulders_coupled"), errors="coerce"
                ).dropna()
                f["identifiability"] = dict(
                    n_half_cycles=int(len(_H)),
                    median=float(_ppp.median()),
                    worst=float(_ppp.min()),
                    n_thin=int((_ppp < IDENTIFIABILITY_POINTS_PER_PARAM).sum()),
                    median_points=(float(_npt.median()) if len(_npt) else float("nan")),
                    median_varied=(float(_nvy.median()) if len(_nvy) else float("nan")),
                    coupled=(int(_cpl.sum()) if len(_cpl) else 0),
                )

    # WHAT `reliable` MEANS ON THIS CELL, broken out. One boolean was
    # carrying four different statements — see the note in
    # `analyse.parameters_frame` — and the page never printed any of them.
    # How many components the anomalous-half-cycle gate actually struck, so
    # the over-theoretical finding can say whether it did anything here.
    f["n_anomalous_components"] = int(
        parameters["half_cycle_anomalous"].fillna(False).astype(bool).sum()
        if (
            parameters is not None
            and not getattr(parameters, "empty", True)
            and "half_cycle_anomalous" in getattr(parameters, "columns", [])
        )
        else 0
    )

    f["reliability"] = {}
    if parameters is not None and not getattr(parameters, "empty", True):
        _pp = parameters
        if "reliable" in getattr(_pp, "columns", []):
            _rel = _pp["reliable"].fillna(False).astype(bool)
            _r = dict(n=int(len(_pp)), ok=int(_rel.sum()))
            if "reliability_reason" in _pp.columns:
                _r["reasons"] = {
                    str(k): int(v)
                    for k, v in _pp.loc[~_rel, "reliability_reason"]
                    .value_counts()
                    .items()
                }
            if "area_determinacy" in _pp.columns:
                _r["determinacy"] = {
                    str(k): int(v)
                    for k, v in _pp["area_determinacy"].value_counts().items()
                }
            # AREAS THAT ARE LOWER BOUNDS. `area_is_lower_bound` has been in
            # the parameter table for several builds and read by nothing.
            # The count that matters is not how many components are
            # truncated — it is how many are truncated AND marked reliable,
            # because those are the ones whose areas a reader will quote.
            _out = pd.to_numeric(_pp.get("area_outside_window_frac"), errors="coerce")
            if _out is not None and _out.notna().any():
                _lb = (
                    _pp["area_is_lower_bound"].fillna(False).astype(bool)
                    if "area_is_lower_bound" in _pp.columns
                    else pd.Series(False, index=_pp.index)
                )
                _mat = _lb & (_out > AREA_LOWER_BOUND_REPORTABLE).fillna(False)
                _r["lower_bound"] = int(_mat.sum())
                _r["lower_bound_reliable"] = int((_mat & _rel).sum())
                _r["lower_bound_worst"] = (
                    float(_out[_mat].max()) if int(_mat.sum()) else 0.0
                )
                _r["lower_bound_median"] = (
                    float(_out[_mat & _rel].median())
                    if int((_mat & _rel).sum())
                    else 0.0
                )
            f["reliability"] = _r

    # THE CELL-LEVEL VERDICT. The page counted anomalous half-cycles and
    # stopped; what it never said is whether they RECUR. See
    # `analyse.cell_integrity_verdict`.
    # IMPORTED AT MODULE SCOPE, not here. A relative import inside a function
    # body survives verbatim into the flattened notebook, where there is no
    # package for it to resolve against — and the `try/except` that used to
    # wrap it turned that ImportError into silence. The verdict vanished from
    # every notebook run and nothing said so. The flattener rewrites top-level
    # imports; `build_inline` now refuses a relative import anywhere else.
    f["cell_integrity"] = cell_integrity_verdict(getattr(integrity, "table", integrity))
    p = f["params"]
    # `or name`, not `get(..., name)`: a composition recorded as "" — which
    # the run summary already handles explicitly — gave a title of "— cell A".
    f["title"] = f"{p.get('composition') or name}" + (
        f" — cell {p['cell_id']}" if p.get("cell_id") else ""
    )

    # --- capacity and retention -----------------------------------------
    # Reported at the last cycle the cell was still WORKING, not the last row
    # of the table. A record that runs 90 cycles past the cell's death ends at
    # 0 mAh/g and 0% retention, and reporting that as "the final capacity" is
    # true of the file and false about the cell.
    disc = ret = ce = None
    f["n_cycles"] = f["last_good_cycle"] = None
    f["partial_final"] = None
    f["in_progress"] = []
    if cycle_table is not None and not cycle_table.empty:
        t = cycle_table.copy()
        t["Cycle"] = pd.to_numeric(t["Cycle"], errors="coerce")
        # `cycling_summary` returns a PER-CELL table whose columns are bare
        # ('Discharge_mAh_g'), while its combined export prefixes them with
        # the cell label ('LTO (Cell A)_Delithiation_mAh_g'). Matching only
        # the suffixed spelling found neither: "Discharge_mAh_g" does not
        # end with "_Discharge_mAh_g", so every capacity number was silently
        # dropped from START_HERE. On an anode the useful half-cycle is
        # labelled Delithiation or Desodiation, so that is looked for too.
        dcol = cycle_column(t, CAPACITY_COLUMN_ALIASES)
        # The per-cell table still uses the bare name; the combined export
        # carries the reference cycle in the column, e.g. Retention_vs_C2_%.
        rcol = cycle_column(t, ("Retention_%", "Retention_vs_C2_%"))
        # ...and read the reference OUT of that column name, so the page can
        # say "99% of cycle 2" instead of "of the reference cycle" three
        # lines under a column headed "% of first".
        _m = re.search(r"_vs_C(\d+)_", str(rcol or ""))
        f["retention_reference"] = int(_m.group(1)) if _m else None
        ccol = cycle_column(t, ("CE_%",))
        icol = cycle_column(t, ("Incomplete",))
        _cmax = t["Cycle"].max()
        f["n_cycles"] = int(_cmax) if pd.notna(_cmax) else None
        good = t[t[icol] != True] if icol else t  # noqa: E712
        # A half-cycle the export caught mid-flight is not a measurement.
        f["in_progress"] = sorted(in_progress or ())
        f["partial_final"] = _trailing_partial(good, dcol, f["in_progress"])
        if f["partial_final"]:
            good = good[good["Cycle"] < f["partial_final"][0]]
        if dcol:
            d = pd.to_numeric(good[dcol], errors="coerce")
            # "Still working" uses the SAME rule as `analyse.last_useful_cycle`
            # — 2% of the 90th-percentile discharge capacity — applied to the
            # table, rather than relying on a limit passed in. A `d > 0` test
            # is not enough: the dead cycles here deliver 0.01 mAh/g, which is
            # greater than zero and is not a working cell.
            good = good[good["Cycle"].notna()]
            d = pd.to_numeric(good[dcol], errors="coerce")
            live = good[d.notna()]
            if len(live):
                dv = pd.to_numeric(live[dcol], errors="coerce")
                floor = 0.02 * float(np.percentile(dv.dropna(), 90))
                live = live[dv > floor]
            if fit_limit and len(live):
                live = live[live["Cycle"] <= fit_limit]
            if len(live):
                _lastc = int(live["Cycle"].iloc[-1])
                # A DEATH is a sustained collapse, not the end of the file.
                # Testing `_lastc < n_cycles` called the last cycle of every
                # record a death whenever that cycle was incomplete or
                # partial — which it usually is. The same run length as
                # `analyse.DEAD_CELL_RUN`: three consecutive dead cycles.
                _tail = t[(t["Cycle"] > _lastc) & t["Cycle"].notna()]
                _n_tail = len(_tail)
                _dead_tail = 0
                if _n_tail and dcol is not None:
                    _tv = pd.to_numeric(_tail[dcol], errors="coerce")
                    _dead_tail = int((_tv.isna() | (_tv <= floor)).sum())
                f["death_cycle"] = _lastc if _dead_tail >= 3 else None
                dv = pd.to_numeric(live[dcol], errors="coerce")
                disc = (float(dv.iloc[0]), float(dv.iloc[-1]))
                f["last_good_cycle"] = _lastc
                if rcol:
                    rv = pd.to_numeric(live[rcol], errors="coerce").dropna()
                    ret = float(rv.iloc[-1]) if len(rv) else None
                if ccol:
                    cv = pd.to_numeric(live[ccol], errors="coerce").dropna()
                    ce = float(cv[cv.between(1, 150)].median()) if len(cv) else None
    # Discharge capacity at the key cycles — the numbers that go in a paper.
    # First, the reference cycle, and whichever key cycles the cell reached.
    f["key_capacities"] = []
    if cycle_table is not None and not cycle_table.empty and disc:
        t2 = cycle_table.copy()
        t2["Cycle"] = pd.to_numeric(t2["Cycle"], errors="coerce")
        # Same lookup as above; the second copy of the old suffix-only
        # match survived the first fix and raised KeyError: None.
        dcol2 = cycle_column(t2, CAPACITY_COLUMN_ALIASES)
        if dcol2 is None:
            t2 = t2.iloc[0:0]
        want = [1] + list((params or {}).get("key_cycles") or [])
        if f.get("last_good_cycle"):
            want.append(f["last_good_cycle"])
        seen, base = set(), disc[0]
        for cyc in sorted({int(c) for c in want}):
            row = t2[t2["Cycle"] == cyc]
            if row.empty or cyc in seen:
                continue
            v = pd.to_numeric(row[dcol2], errors="coerce").iloc[0]
            if not np.isfinite(v) or v <= 0:
                continue
            if f.get("last_good_cycle") and cyc > f["last_good_cycle"]:
                continue
            seen.add(cyc)
            f["key_capacities"].append(
                (cyc, float(v), 100.0 * v / base if base else None)
            )

    # IS THIS DATASET A MEASUREMENT AT ALL?
    # `JQ_NNM_C-rate_2-4.2V_A_29042026` is a 485 kB export against 3.4 MB for
    # its two siblings: two cycles, median CE 41.3%, two cycles passing more
    # charge than the material can hold. It received a full analysis, a full
    # report and a place in the replicate mean, and nothing anywhere said the
    # file was short or the efficiency impossible. It was caught by eye.
    #
    # The two tests below are DEFINITIONAL rather than tuned:
    #
    #   too few cycles   retention is measured against
    #                    `cycling.RETENTION_REFERENCE_CYCLE`; with fewer
    #                    cycles than that reference plus one there is nothing
    #                    to measure it against, so the figure cannot exist.
    #   CE below 50%     more charge is lost every cycle than comes back. That
    #                    is not a working cell, whatever else the file says.
    #
    # A dataset failing either still gets its own page — it is data, and the
    # page is where you look to find out what went wrong. What it does not get
    # is a vote in the replicate mean.
    _unusable = []
    if f["n_cycles"] is not None and f["n_cycles"] <= UNUSABLE_MIN_CYCLES:
        _unusable.append(
            f"only {f['n_cycles']} complete cycle(s) — capacity retention is "
            f"measured against cycle {UNUSABLE_MIN_CYCLES}, so there is no "
            f"reference for it here"
        )
    if ce is not None and ce < UNUSABLE_MIN_MEDIAN_CE:
        _unusable.append(
            f"median coulombic efficiency {ce:.1f}% — more charge is lost each "
            f"cycle than returns, which is not a working cell"
        )
    f["unusable"] = _unusable

    f.setdefault("death_cycle", None)
    f["first_last_discharge"] = disc
    # WITHHELD AT SOURCE, so nothing downstream can quote it. A variable-rate
    # run's `Q(n)/Q(ref)` is dominated by the rate schedule, and the run page
    # and the replicate table both pool this field across cells. Keeping the
    # arithmetic under its own key means the number is not lost, only barred
    # from being called retention.
    f["rate_is_variable"] = bool(
        ((params or {}).get("rate_protocol") or {}).get("is_variable")
    )
    f["retention_uncorrected"] = ret
    f["retention"] = None if f["rate_is_variable"] else ret
    f["median_ce"] = ce
    f["fit_limit"] = fit_limit

    # --- integrity --------------------------------------------------------
    f["bands"], f["anomalous"] = {}, pd.DataFrame()
    if integrity is not None and not integrity.empty:
        f["bands"] = integrity["band"].value_counts().to_dict()
        # ANOMALOUS for the PARASITIC reason only. A half-cycle condemned for
        # passing more charge than the material can hold is reported under
        # that heading instead; listing it twice reads as two faults.
        _an = integrity[integrity["band"] == "ANOMALOUS"]
        if "over_theoretical" in _an:
            _an = _an[_an["over_theoretical"] != True]  # noqa: E712
        f["anomalous"] = _an.sort_values("parasitic_fraction", ascending=False)
        f["parasitic_total"] = float(integrity["parasitic_charge"].sum())
        f["plateau_total"] = float(integrity["plateau_charge"].sum())
        if "over_theoretical" in integrity:
            ot = integrity["over_theoretical"] == True  # noqa: E712
            f["over_theoretical"] = integrity[ot].sort_values(
                "capacity_ratio", ascending=False
            )
            _sane_pl = integrity.loc[~ot, "plateau_charge"]
            f["plateau_sane"] = float(_sane_pl.sum())
        else:
            f["over_theoretical"] = None
            f["plateau_sane"] = f["plateau_total"]
            _sane_pl = integrity["plateau_charge"]
        # The scale of ONE half-cycle, so the total can be read against a
        # capacity without being mistaken for one. See the plateau finding.
        _sane_pl = pd.to_numeric(_sane_pl, errors="coerce")
        _nz = _sane_pl[_sane_pl.notna() & (_sane_pl > 0)]
        f["plateau_n_half_cycles"] = int(len(_nz))
        f["plateau_worst_mAh_g"] = float(_nz.max()) if len(_nz) else float("nan")

    # --- what was withheld ------------------------------------------------
    withheld = 0
    total = 0
    widths = []
    for step, A in (attribution or {}).items():
        if A is None or A.empty:
            continue
        withheld += int(A["attribution_withheld"].sum())
        total += len(A)
        w = pd.to_numeric(A["closure_interval_width"], errors="coerce").dropna()
        widths.extend(w.tolist())
    f["withheld"] = (withheld, total)
    f["closure_widths"] = (min(widths), max(widths)) if widths else None
    # The charge the model could not name, and how often that alone was the
    # reason a share was withheld. From 1.9.0.56 there is no free background,
    # so the closure interval is zero by construction and THIS is the number
    # a reader should be looking at.
    _un, _un_with = [], 0
    for step, A in (attribution or {}).items():
        if A is None or A.empty or "unattributed_fraction" not in A:
            continue
        _u = pd.to_numeric(A["unattributed_fraction"], errors="coerce").dropna()
        _un.extend(_u.tolist())
        if "unattributed_withheld" in A:
            _un_with += int(A["unattributed_withheld"].sum())
    f["unattributed"] = float(np.median(_un)) if _un else None
    f["unattributed_range"] = (min(_un), max(_un)) if _un else None
    f["unattributed_withheld"] = _un_with
    # True when nothing was swept: every measured width is zero (one degree
    # only), or none could be measured at all.
    _w = [x for x in widths if np.isfinite(x)]
    f["no_baseline"] = bool(attribution) and (not _w or max(_w) == 0.0)

    # --- peaks ------------------------------------------------------------
    f["reference_reason"] = getattr(detection, "reference_reason", "")
    f["reference_severity"] = getattr(detection, "reference_severity", "ok")
    f["peaks"] = (
        {
            s: list(
                np.round(getattr(detection, "reference_voltages", lambda _: [])(s), 3)
            )
            for s in ("Charge", "Discharge")
        }
        if detection
        else {}
    )
    f["tracked"] = []
    if tracking is not None and tracking.summary:
        for k in sorted(tracking.summary):
            s = tracking.summary[k]
            f["tracked"].append(s)
    # THE NUMBER THE AREA-GROWTH FLAG WAS ACTUALLY MADE AGAINST.
    # `analyse.flag_area_growth` compares each peak's area retention with
    # `capacity_retention_pct(cycle_table, reference_cycle)` — median-of-three
    # at each end, measured from the REFERENCE cycle — and stores it on the
    # Tracking. The finding that reports the flag was printing `f["retention"]`
    # instead, which is the last working cycle's `Retention_vs_C2_%`: a
    # different basis, a different reference, and on a cell that died mid-record
    # a different number by tens of points. NNM cell A published "22% area
    # retention ... against the cell's 34%" from a guard whose bound implied
    # ~0%. One flag, one basis, printed from where the flag put it.
    f["area_growth_basis_pct"] = float(
        getattr(tracking, "capacity_retention_pct", np.nan)
        if tracking is not None
        else np.nan
    )
    # THE COHERENCE VERDICT, carried to the run summary.
    # `coherence_audit` decides, per reference peak, whether that peak moves
    # like a redox feature or like a fitting artefact. Until 1.9.0.34 the
    # verdict lived only in the console: RUN_SUMMARY's drift table published a
    # rate for every primary peak, captioned as "the quantity that survives
    # where the areas do not", while the audit two cells earlier had called
    # the same peak NOT trend-worthy. On NMC111 that meant a table of drift
    # rates on a run where NOT ONE peak in either cell was coherent.
    f["coherence"] = []
    if coherence is not None and getattr(coherence, "empty", True) is False:
        for _r in coherence.itertuples():
            f["coherence"].append(
                dict(
                    step=str(_r.step),
                    reference_voltage=float(_r.reference_voltage),
                    verdict=str(_r.verdict),
                    pairs=int(_r.pairs),
                )
            )
    # Does the dQ/dV curve account for the cell's charge? Set by
    # `analyse.flag_low_fidelity`. This belongs in START_HERE and not only in
    # the console: it decides whether the peak AREAS on this page mean
    # anything, and START_HERE is where somebody reads them.
    f["integral_fidelity"] = dict(getattr(tracking, "integral_fidelity", {}) or {})
    f["profile"] = (
        (profile or {}).get("class") if isinstance(profile, dict) else profile
    )
    return f


# What each mechanism means, and what it implies about the model — in the
# words a reader needs rather than the enum name. `quality.MECHANISM_*`.
_MECHANISM_TEXT = {
    "two_phase": (
        "one transition at a fixed potential",
        "The potential is pinned by two phases coexisting, so there is no "
        "composition window to smear over and every component was fitted as "
        "a narrow peak on a straight baseline.",
    ),
    "multi_transition": (
        "a series of distinct transitions",
        "Each feature is a real transition in its own right, so the curve "
        "was fitted as a series of peaks — no flat-topped bands, which would "
        "give a resolved feature a width it does not have.",
    ),
    "solid_solution": (
        "a continuum of site energies",
        "Charge is delivered across a window of composition rather than at "
        "one potential, so the components were fitted as flat-topped bands. "
        "The shoulders a peak model finds on an envelope like this are "
        "artefacts of the wrong shape, so only the primary features were "
        "seeded.",
    ),
    "mixed": (
        "a solid solution with a transition on top of it",
        "Both shapes were used in the same fit: bands for the envelope, "
        "peaks for the transition riding on it, and the data decided which "
        "each component became.",
    ),
}


def _mechanism_sentence(f):
    """The verdict that chose the model, in the document that reports it.

    Until 1.9.0.47 this existed only in the console. A reader of START_HERE
    could not tell whether their data had been fitted with pseudo-Voigts,
    with bands, or with both — which is the one thing this build decides
    differently for every dataset.
    """
    r = f.get("resolvability") or {}
    mech = r.get("mechanism")
    if not mech:
        return []
    what, implies = _MECHANISM_TEXT.get(str(mech), (str(mech).replace("_", " "), ""))
    why = (r.get("mechanism_reason") or "").strip().rstrip(".")
    sf = r.get("shoulder_fraction")
    bits = [
        f"The run classified this dataset as **{str(mech).replace('_', ' ')}"
        f"** — {what}."
    ]
    if why:
        bits.append(why[0].upper() + why[1:] + ".")
    elif sf is not None and np.isfinite(sf):
        bits.append(
            f"{sf:.0%} of the detected components are shoulders "
            f"rather than resolved maxima."
        )
    if implies:
        bits.append(implies)
    bits.append(
        "The classification is made on the reference half-cycle "
        "before anything is fitted, and it is what chose the model "
        "below; `component_kind` in the fitted-parameter table says "
        "what each component actually became."
    )
    return [("ok", "How this curve was modelled", " ".join(bits))]


_KIND_WORD = {"band": "flat-topped band", "peak": "peak"}


def _kind_by_ref(f):
    """(step, reference voltage) -> the shape that feature was fitted with."""
    out = {}
    for t in f.get("tracked") or []:
        k = t.get("component_kind")
        if not k:
            continue
        out[(t.get("step"), round(float(t.get("reference_voltage", np.nan)), 3))] = (
            _KIND_WORD.get(str(k), str(k))
            + (
                " (band in some cycles, peak in others)"
                if t.get("component_kind_mixed")
                else ""
            )
        )
    return out


_QUALITY_NOTE = (
    "R² is dominated by the bulk of a curve, so a composite can be half again "
    "taller than the data, or miss a peak height by a third, and still score "
    "0.97. These say what it cannot."
)


def _quality_sentence(f):
    """How well does the model actually draw this curve?

    Three faults, none of which R² carries: a fit too SHORT (a width floor —
    the model cannot be as narrow as the peak, so it conserves area by being
    wide and low), a fit too TALL and outside the data (the peak/background
    split is not determined, so the baseline is drawn high and the peaks are
    deepened to reach the curve), and a residual with STRUCTURE rather than
    scatter (the lineshape itself is wrong).
    """
    q = f.get("fit_quality") or {}
    if not q:
        return []
    bits, worst = [], "ok"
    for step in ("Charge", "Discharge"):
        s = q.get(step)
        if not s:
            continue
        h, o = s.get("height_ratio", np.nan), s.get("overshoot", np.nan)
        mr, z = s.get("max_residual_frac", np.nan), s.get("residual_runs_z", np.nan)
        _p = [
            f"**{step}** (median of {s.get('n', 0)} half-cycles): the fitted "
            f"peak is {h:.2f}x the height of the data"
        ]
        if np.isfinite(h) and h < 0.85:
            _p.append(
                "— the model is drawing the peak too SHORT, which is "
                "what a width floor looks like: it cannot be as narrow "
                "as the feature, so it conserves area by being wide and "
                "low"
            )
            worst = "bad"
        elif np.isfinite(h) and h > 1.15:
            _p.append(
                "— the model is drawing the peak TALLER than the curve, "
                "which is what an undetermined peak/background split "
                "looks like: the baseline sits high and the peaks are "
                "deepened to reach the data"
            )
            worst = "bad"
        if np.isfinite(o) and o > 0.05:
            _p.append(
                f", and the composite goes {o:.0%} of the amplitude "
                f"outside the range the data ever reached"
            )
            worst = "bad"
        if np.isfinite(mr):
            _p.append(f". The largest residual is {mr:.0%} of the amplitude")
            if mr > 0.20 and worst == "ok":
                worst = "warn"
        if np.isfinite(z) and z < -4:
            # ITEM 34. This used to read "the LINESHAPE is wrong, not just the
            # noise level", and on LTO that is not supported: turning the
            # smoothing off makes z WORSE, binning finer makes it worse still
            # (-46.5), and z does not correlate with how sparsely the peak is
            # sampled. A runs test needs a RANDOM component to test against —
            # a residual carrying a systematic term at half the noise
            # amplitude scores only -2 to -5 even at n = 5000. Reaching -25 to
            # -34 means the residual is almost entirely deterministic, which
            # is what a curve built from bins holding 0 or 1 raw record is.
            # So it says the residual has structure; it does not say what put
            # it there.
            _p.append(
                f", and the residual runs the same sign far longer than "
                f"noise would (z = {z:.0f}) — there is structure left in "
                f"it. That can be the lineshape, the baseline, or a "
                f"curve with too little independent noise to test "
                f"against; z alone does not say which"
            )
            if worst == "ok":
                worst = "warn"
        # ITEM 31. THE MEDIAN HID THE FAULT. Every number above is a median,
        # and on NMC111 that reported "0.98x the height of the data" while one
        # half-cycle was at 0.57x and two on the replicate at 0.40x and 0.35x.
        # `height_worst` was computed for exactly this and read by nothing —
        # the same defect class as items 1, 5 and 22, in the paragraph whose
        # own closing line warns that a fit can "miss a peak height by a third
        # and still score 0.97".
        _hw, _hwc = s.get("height_worst", np.nan), s.get("height_worst_cycle")
        _ht, _htc = s.get("height_tallest", np.nan), s.get("height_tallest_cycle")
        _ns, _nt = s.get("n_short", 0), s.get("n_tall", 0)
        _r2w, _r2c = s.get("r2_worst", np.nan), s.get("r2_worst_cycle")
        _ex = []
        if np.isfinite(_hw) and _hw < 0.85 and _hwc is not None:
            _ex.append(
                f"the WORST is cycle {_hwc} at {_hw:.2f}x"
                + (
                    f", and {_ns} of {s.get('n', 0)} half-cycles are below 0.85x"
                    if _ns > 1
                    else ""
                )
            )
        if np.isfinite(_ht) and _ht > 1.15 and _htc is not None:
            _ex.append(
                f"the TALLEST is cycle {_htc} at {_ht:.2f}x"
                + (f", and {_nt} of {s.get('n', 0)} are above 1.15x" if _nt > 1 else "")
            )
        if _ex:
            _p.append(". A median hides a single bad half-cycle, so: " + "; ".join(_ex))
            worst = "bad"
        elif np.isfinite(_r2w) and _r2w < 0.90 and _r2c is not None:
            _p.append(
                f". The median is not the whole story — the worst "
                f"half-cycle is cycle {_r2c} at R2 {_r2w:.3f}"
            )
            if worst == "ok":
                worst = "warn"
        bits.append(" ".join(_p).replace(" ,", ",").replace(" .", ".") + ".")
    if not bits:
        return []
    head = (
        "The model does not draw this curve well"
        if worst == "bad"
        else "The fit has structure left in it"
        if worst == "warn"
        else "The model draws this curve well"
    )
    return [(worst, head, " ".join(bits) + " " + _QUALITY_NOTE)]


def _voltage_envelope_sentence(f):
    """
    How far the cell actually got, cycle by cycle.

    The dQ/dV heatmap draws this as a step line at the edge of the measured
    region; nothing stated it. It is a different quantity from capacity: a
    cell can deliver the same charge every cycle while the voltage window it
    traverses to do so collapses, and that is polarisation rather than loss of
    active material. LTO cell A moves 622 mV over ten cycles at 99 % capacity
    retention.
    """
    env = (f.get("params") or {}).get("voltage_envelope") or {}
    if not env:
        return []
    bits, worst = [], "ok"
    for step in ("Charge", "Discharge"):
        e = env.get(step)
        if not e:
            continue
        _sp = max(float(e.get("hi_spread_mV", 0.0)), float(e.get("lo_spread_mV", 0.0)))
        # `drawn` IS THE GATE. It is decided once, beside the envelope
        # measurement, using the plotting module's own thresholds, so this
        # sentence describes exactly the steps on which the heat map drew a
        # line. See the note there for what went wrong with two thresholds.
        if not e.get("drawn") or not np.isfinite(_sp):
            continue
        _hi_moves = float(e.get("hi_spread_mV", 0.0)) >= float(
            e.get("lo_spread_mV", 0.0)
        )
        _a = e["hi_first"] if _hi_moves else e["lo_first"]
        _b = e["hi_last"] if _hi_moves else e["lo_last"]
        _edge = "furthest" if _hi_moves else "nearest"
        bits.append(
            f"**{step}**: the {_edge} voltage this step reached went from "
            f"{_a:.3f} V to {_b:.3f} V over {int(e.get('n', 0))} half-cycles, "
            f"a span of {_sp:.0f} mV"
        )
        if _sp >= ENVELOPE_SEVERE_MV:
            worst = "warn"
    if not bits:
        return []
    return [
        (
            worst,
            "The voltage window the cell traverses is not constant",
            "; ".join(bits)
            + ". This is not a capacity statement — a cell can deliver the "
            "same charge every cycle while the window it needs to do so "
            "moves, and that is polarisation, or a cut-off being reached "
            "sooner, rather than active material lost. Read it beside the "
            "retention above: the two moving together is fade, the window "
            "moving alone is resistance. The blue step line on the dQ/dV "
            "heatmap is this quantity, and the region beyond it is "
            "unmeasured rather than zero.",
        )
    ]


def _identifiability_sentence(f):
    """
    How much data each fit had, against how many parameters it varied.

    Every standard error on the page, every `*_poorly_determined` flag and the
    `reliable` boolean itself are statements about a least-squares fit, and a
    least-squares fit of 60 points with 14 varied parameters says something
    very different from one of 600 with the same 14. `fitting.fit_one` has
    recorded both numbers since the histogram rewrite; nothing read them.
    """
    q = f.get("identifiability") or {}
    if not q:
        return []
    _med, _min = q["median"], q["worst"]
    _thin, _n = q["n_thin"], q["n_half_cycles"]
    _coup = q.get("coupled", 0)
    _mp, _mv = q.get("median_points"), q.get("median_varied")
    _lead = (
        f"Median {_med:.0f} points per varied parameter across "
        f"{_n} half-cycles"
        + (
            f" ({_mp:.0f} bins, {_mv:.0f} parameters)"
            if _mp is not None
            and np.isfinite(_mp)
            and _mv is not None
            and np.isfinite(_mv)
            else ""
        )
        + f"; the thinnest fit had {_min:.1f}."
        + (
            f" {_coup} shoulder parameter(s) were tied to a parent rather "
            f"than varied freely, which is why the parameter count is "
            f"lower than the component count implies."
            if _coup
            else ""
        )
    )
    if _thin:
        return [
            (
                "warn",
                "Some fits had little data per parameter",
                _lead + f" {_thin} of {_n} half-cycles fell below "
                f"{IDENTIFIABILITY_POINTS_PER_PARAM:.0f} points per "
                f"parameter. Below that the standard errors, and every flag "
                f"derived from them — `centre_poorly_determined`, "
                f"`area_poorly_determined`, `reliable` — describe a fit with "
                f"barely more information than it has unknowns. "
                f"`points_per_parameter` is in the fitted-parameters table "
                f"per half-cycle.",
            )
        ]
    return [
        (
            "ok",
            "The fits had enough data for their parameters",
            _lead + " Nothing here is close to the point where a "
            "least-squares standard error stops meaning anything. "
            "`points_per_parameter` is in the fitted-parameters table.",
        )
    ]


def _reliability_sentence(f):
    """
    What `reliable` says on this cell, and what it does NOT say.

    The boolean was the most-read field in the parameter table and the page
    printed neither its count nor its reasons. Worse, a reader who takes it
    to mean "I can quote this area with its uncertainty" is wrong about
    nearly every component of a layered oxide: on NNM 83% and on NMC 89% of
    components have no standard error on their area at all, and that never
    reaches `reliable` because `ENFORCE_AREA_DETERMINACY` is off. The two
    are different axes and are now printed as two.
    """
    r = f.get("reliability") or {}
    n, ok = r.get("n"), r.get("ok")
    if not n:
        return []
    bits = [f"{ok} of {n} fitted components are marked reliable."]
    reasons = r.get("reasons") or {}
    if reasons:
        _ordered = sorted(reasons.items(), key=lambda kv: -kv[1])
        # ONLY GLOSS THE REASONS THAT ACTUALLY FIRED. Explaining what an
        # inferred component means on a cell that has none is noise, and the
        # LTO page carried three sentences to cover a single pinned width.
        _gloss = {
            "band at its width ceiling": "a component at a bound has a width that is a constraint "
            "rather than a fit — and being knife-edge, one that need not "
            "reproduce on another machine",
            "width at its ceiling": "a component at a bound has a width that is a constraint "
            "rather than a fit — and being knife-edge, one that need not "
            "reproduce on another machine",
            "width at its ceiling or area not finite": "a component at a bound has a width that is a constraint "
            "rather than a fit",
            "not detected in this half-cycle": "a component not detected in its own half-cycle carries a "
            "centre borrowed from a neighbouring one",
            "this half-cycle passed more charge than the material can hold": "a half-cycle that passed more charge than the material can "
            "hold did not produce a differential capacity curve of the "
            "material, however faithfully that curve integrates — its "
            "components are fitted, tracked and drawn, and left out of "
            "the trends",
            "the fit put almost no charge here": "a component carrying almost no charge is the fit saying the "
            "feature is not there",
            "most of this component lies outside the measured window": "a component sitting at an end of the voltage window has "
            "part of its profile outside the data, and runs both flanks "
            "to the width bound because nothing on that side stops "
            "them — the window is the reason, not the width",
            "area collapsed against the reference cycle": "a component that has collapsed against the reference cycle "
            "is no longer the feature it was tracked as",
        }
        _seen, _say = set(), []
        for _k, _ in _ordered:
            _g = _gloss.get(_k)
            if _g and _g not in _seen:
                _seen.add(_g)
                _say.append(_g)
        _lead = (
            "The rest are not, for reasons that are not interchangeable: "
            if len(_ordered) > 1
            else "The rest are not: "
        )
        _tail = ""
        if _say:
            _tail = (
                " — "
                + "; ".join(_say)
                + ". "
                + (
                    "None of these is the same as a bad measurement, and "
                    "none is the same as the others."
                    if len(_say) > 2
                    else "Neither is the same as a bad measurement, and "
                    "neither is the same as the other."
                    if len(_say) == 2
                    else "That is not the same as a bad measurement."
                )
            )
        bits.append(_lead + "; ".join(f"{v} {k}" for k, v in _ordered) + _tail)
    det = r.get("determinacy") or {}
    if det:
        _no = int(det.get("not estimated", 0))
        _im = int(det.get("imprecise", 0))
        _de = int(det.get("determined", 0))
        _s = (
            f"Separately, and NOT part of that flag: {_de} of {n} "
            f"components have a determined area uncertainty, {_im} have one "
            f"too large to quote, and {_no} have none at all"
            + (
                " — the covariance could not be estimated, so the area has "
                "no error bar to put beside it."
                if _no
                else ", so every area here can be quoted with one."
            )
        )
        if _no > 0.5 * n:
            _s += (
                " That is most of this cell. `reliable` says nothing "
                "about it, so an area from here should be quoted without "
                "an uncertainty rather than with an implied one."
            )
        bits.append(_s)
    # AN AREA THAT IS A LOWER BOUND, said out loud. A component centred near
    # an end of the voltage window has part of its profile outside the data;
    # the analytic area covers the whole profile, so the number is what the
    # feature would carry IF the window reached far enough, and the measured
    # part is less. `reliable` is deliberately silent on this — it is the
    # same separation as the determinacy line above — so the page has to say
    # it instead.
    _lb_rel = int(r.get("lower_bound_reliable", 0) or 0)
    _lb = int(r.get("lower_bound", 0) or 0)
    if _lb:
        _md = float(r.get("lower_bound_median", 0.0) or 0.0)
        _wo = float(r.get("lower_bound_worst", 0.0) or 0.0)
        _s2 = (
            f"Also separately: {_lb} of {n} components have more than "
            f"{AREA_LOWER_BOUND_REPORTABLE:.0%} of their fitted profile "
            f"outside the measured voltage window, so their areas are "
            f"LOWER BOUNDS rather than measurements"
        )
        if _lb_rel:
            _s2 += (
                f" — and {_lb_rel} of those are marked reliable, with a "
                f"median {_md:.0%} outside (worst {_wo:.0%}). `reliable` "
                f"is a statement about the fit, not about how much of "
                f"the feature the window caught. Widen the window, or "
                f"quote these as “at least”."
            )
        else:
            _s2 += (
                f", worst {_wo:.0%} outside. None of them is marked "
                f"reliable, so nothing quotable is affected."
            )
        bits.append(_s2)
    _lvl = "warn" if (reasons and ok < 0.5 * n) else "ok"
    if _lb_rel:
        _lvl = "warn"
    return [(_lvl, "What \u201creliable\u201d means here", " ".join(bits))]


# How many bins a width must span before the report will call it a width.
# The same number as `analyse.WIDTH_MIN_SAMPLES`, named here so the sentence
# and the flag cannot drift apart.
# Below this, plateau charge is not worth a finding of its own. The tail
# clause of that same finding has always used 1 mAh/g; the gate on the
# finding used 0, so a third of a percent of one half-cycle raised a red mark.
PLATEAU_REPORTABLE_MAH_G = 1.0


def _unattributed_clause(u, rng=None):
    """
    " The model names 99.8% of the charge, leaving +0.2% unattributed."

    At 0 dp this printed "names 100% of the charge, leaving +0% unattributed"
    on all three chemistries — and on LTO, whose median is -0.0012, the
    literal string "-0%". A signed zero is not a measurement of anything, and
    the RANGE was computed for this and rendered only in the other branch.
    """
    if u is None or not np.isfinite(u):
        return ""
    _r = ""
    if rng and all(v is not None and np.isfinite(v) for v in rng):
        _r = f" (over the run, {rng[0]:+.1%} to {rng[1]:+.1%})"
    if abs(u) < 0.005:
        return (
            f" The model names essentially all of the charge — the median "
            f"unattributed fraction is under 0.5%{_r}."
        )
    return (
        f" The model names {1 - u:.1%} of the charge, leaving "
        f"{u:+.1%} unattributed{_r}."
    )


def _sentence_case(t):
    """
    Upper-case the first letter and LEAVE THE REST ALONE.

    `str.capitalize()` lower-cases everything after the first character, so
    `detect.formation_end`'s "CE never held within 0.1 points of the later
    median" reached every page in this project as "ce never held...".
    """
    t = str(t or "")
    return t[:1].upper() + t[1:]


WIDTH_BINS_MEASURABLE = 4.0

# POINTS PER VARIED PARAMETER, below which a least-squares standard error is
# not worth quoting. Not a significance threshold — there is no such thing here
# — but the point at which the fit has so little information per unknown that
# the covariance matrix it reports is dominated by the model rather than by the
# data. Ten is the conventional rule of thumb for non-linear least squares and
# is used here as a flag, never as a rejection: nothing is dropped on it.
IDENTIFIABILITY_POINTS_PER_PARAM = 10.0

# WHEN THE REACHABLE-VOLTAGE ENVELOPE IS BAD NEWS RATHER THAN A FACT.
# Whether it is worth mentioning at all is not decided here — it is decided
# once, where the envelope is measured, by the same test the heat map uses to
# decide whether to draw the line, so the sentence and the figure cannot
# disagree. This is only the severity: above it, the dQ/dV curves being
# compared cycle to cycle no longer cover the same voltages, which changes
# what a peak area means.
ENVELOPE_SEVERE_MV = 300.0

# WHEN IS A DATASET NOT A MEASUREMENT? Both of these are definitional rather
# than tuned, which is why they are two numbers and not a scoring function.
#
# `UNUSABLE_MIN_CYCLES` is `cycling.RETENTION_REFERENCE_CYCLE`: retention is
# quoted against that cycle, so a record no longer than it has no reference to
# be measured against and the figure cannot exist. Stated here rather than
# imported to keep `report` free of a cycling import; the two are checked
# against each other by `test_reference_cycle_agreement`.
#
# `UNUSABLE_MIN_MEDIAN_CE` is 50%: below it more charge is lost every cycle
# than comes back, which is not a working cell whatever else the file says.
# It is deliberately far below the ~95% at which a cell is merely UNHEALTHY —
# this flag is for "this is not a measurement", not for "this is a poor cell".
UNUSABLE_MIN_CYCLES = 2
UNUSABLE_MIN_MEDIAN_CE = 50.0


def _lineshape_sentence(f):
    """Was the peak fitted as asymmetric, and by how much?

    Reported because it is a measurement the reader would otherwise have to
    take from a CSV, and because a ratio that settled ON its bound is not a
    measurement at all — the same fault as a width pinned on its floor, one
    level up, and invisible without this because there is one ratio per
    half-cycle rather than one per component.
    """
    q = f.get("lineshape") or {}
    if not q:
        return []
    bits, worst = [], "ok"
    for step in ("Charge", "Discharge"):
        s = q.get(step)
        if not s or not s.get("split"):
            continue
        k = s["k"]
        _side = "high-voltage" if k > 1 else "low-voltage"
        _ratio = k if k > 1 else (1.0 / k if k > 0 else float("nan"))
        # THE RANGE IN THE SAME ORIENTATION AS THE HEADLINE. `_ratio` is
        # flipped to 1/k when the LOW flank is the wide one, so that the
        # sentence can always say "N times the width of the other" with N > 1.
        # `k_lo`/`k_hi` were printed un-flipped, so LTO cell B read "the
        # low-voltage flank is 3.9x the width of the other (ratio 0.09-0.33)" —
        # a headline outside its own quoted range, which reads as scatter in
        # the data rather than as a unit flip. Inverting a range also reverses
        # its ends, so k_hi becomes the lower bound.
        _klo, _khi = float(s["k_lo"]), float(s["k_hi"])
        if not (k > 1):
            _klo, _khi = (
                (1.0 / _khi if _khi > 0 else float("nan")),
                (1.0 / _klo if _klo > 0 else float("nan")),
            )
        _ss, _nk = s.get("same_sign"), s.get("n_k", s["n"])
        _p = [
            f"**{step}** ({s['n']} half-cycles): the {_side} flank is "
            f"{_ratio:.1f}x the width of the other "
            f"(ratio {_klo:.2f}-{_khi:.2f}"
            + (
                f", and {_ss} of {_nk} lean that way"
                if _ss is not None and _ss < _nk
                else ""
            )
            + ")"
        ]
        if _ss is not None and _nk and _ss < 0.8 * _nk:
            _p.append(
                f". The sign is NOT consistent — {_nk - _ss} half-cycles "
                f"lean the other way, so the median above describes a "
                f"direction this dataset does not hold"
            )
            worst = "warn"
        if s.get("clipped"):
            _p.append(
                f". On {s['clipped']} of {_nk}, one flank was held at "
                f"the width floor, so the ratio quoted is the one the "
                f"curve was drawn with rather than the one the "
                f"optimiser reached (`asymmetry_k_fitted` in the table)"
            )
        # THE WIDTHS THEMSELVES, BEFORE THEIR RATIO. Said first, because it
        # governs everything after it: a ratio of two numbers that are both
        # the bin width is 1.0 plus noise, whatever the fit reports.
        _fb, _us, _nc = (
            s.get("fwhm_bins"),
            int(s.get("undersampled", 0) or 0),
            int(s.get("n_comp", 0) or 0),
        )
        if _fb is not None and np.isfinite(_fb) and _fb < WIDTH_BINS_MEASURABLE:
            _p.append(
                f". The width this is a ratio OF is not a measurement: "
                f"the fitted FWHM spans a median {_fb:.1f} bins"
                + (
                    f" and {_us} of {_nc} components are flagged `width_undersampled`"
                    if _nc
                    else ""
                )
                + f", against the {WIDTH_BINS_MEASURABLE:.0f} a width "
                f"needs. Swept over bin widths from 0.5 to 5 mV on "
                f"this chemistry the fitted FWHM tracks the BIN, not "
                f"the peak, so neither the widths nor this ratio "
                f"should be quoted as a property of the material"
            )
            worst = "warn"
        _daf = int(s.get("dominant_at_floor", 0) or 0)
        if _daf:
            _p.append(
                f". On {_daf} of {s['n']} half-cycles the NARROW side "
                f"of the largest component sits exactly on the width "
                f"floor — the leading edge is steeper than the "
                f"histogram samples, so that half of the width is the "
                f"floor and not a measurement, and `fwhm` carries it"
            )
            if worst == "ok":
                worst = "warn"
        if s["at_bound"]:
            _p.append(
                f". {s['at_bound']} of {s['n']} settled ON the ratio "
                f"bound, where the number is the limit and not an "
                f"answer — read those widths as a floor"
            )
            worst = "warn"
        bits.append(" ".join(_p).replace(" .", ".") + ".")
    if not bits:
        return []
    head = (
        "The peak is asymmetric and was fitted that way"
        if worst == "ok"
        else "The peak is asymmetric, and one flank is not resolved"
    )
    # WHAT THE TWO DIRECTIONS ACTUALLY DID. This closed with "The two
    # directions lean OPPOSITE ways" unconditionally — appended to whatever
    # `bits` held, including a single fitted step, where "the two directions"
    # has no referent at all, and including two steps leaning the SAME way,
    # where the sentence states the opposite of the data above it. It happened
    # to be true of the LTO cells on disk, which is how it survived. The
    # asymmetry itself is the finding; which way the two steps lean is a
    # separate observation and is only worth making when both were measured.
    _ks = [
        float(q[st]["k"])
        for st in ("Charge", "Discharge")
        if q.get(st)
        and q[st].get("split")
        and np.isfinite(q[st].get("k", np.nan))
        and q[st]["k"] > 0
    ]
    if len(_ks) < 2:
        _lean = (
            " Only one direction was fitted with a split lineshape, so "
            "there is nothing here to compare it with."
        )
    elif (max(_ks) > 1.0) and (min(_ks) < 1.0):
        _lean = (
            " The two directions lean OPPOSITE ways, which is what a "
            "kinetic tail or a spread of particle sizes gives and is not "
            "something a symmetric peak can absorb."
        )
    else:
        _lean = (
            " Both directions lean the SAME way — towards "
            + ("high" if min(_ks) > 1.0 else "low")
            + " voltage. A "
            "kinetic tail reverses with the sweep, so a bias that does "
            "not reverse is more likely to be in the model or the "
            "binning than in the material, and is worth checking against "
            "the fits before it is read as a property of the electrode."
        )
    return [
        (
            worst,
            head,
            " ".join(bits)
            + _lean
            + " Read which flank is the trailing one from the direction of "
            "the sweep, since a negative-electrode cell has its step "
            "labels the other way round. The width columns are `sigma` "
            "(low voltage) and `sigma_r` (high voltage); `fwhm` is their "
            "sum, not twice either one.",
        )
    ]


def _phase_rule_sentence(f):
    """What the thermodynamics expects, beside what was fitted.

    On a `two_phase` run the Gibbs phase rule is a real constraint and it is
    the argument that settles most "is this a doublet?" questions — see
    Appendix A7. It is stated here rather than enforced in the detector,
    deliberately: the phase rule bounds the two-phase FIELD, not the voltage
    window, and a detector that applied it would be asserting the expectation
    instead of testing against it. The reader gets the expectation at the
    moment they are looking at the fit, and makes the call.
    """
    r = f.get("resolvability") or {}
    if str(r.get("mechanism") or "") != "two_phase":
        return []
    q = f.get("component_census") or {}
    if not q:
        return []
    bits = []
    for step in ("Charge", "Discharge"):
        c = q.get(step)
        if not c:
            continue
        bits.append(f"**{step}**: {c['lo']}-{c['hi']} fitted, median {c['median']:.0f}")
    if not bits:
        return []
    extra = any((q.get(k) or {}).get("hi", 1) > 1 for k in ("Charge", "Discharge"))
    return [
        (
            "warn" if extra else "ok",
            "What the phase rule expects here",
            "This dataset was classified **two phase**, and a two-phase field in "
            "a binary system at fixed temperature and pressure is invariant — the "
            "potential is held while both phases coexist, so the expectation is "
            "**one plateau and therefore one dQ/dV peak per transition**. "
            + "; ".join(bits)
            + ". "
            + (
                "Where more than one component was fitted, ask what process the "
                "extra one would correspond to before believing it: an R² gain "
                "from adding a component is what overfitting looks like, and "
                "stability against a change of bin width is necessary but not "
                "sufficient. The processing-free test is V(Q) — Q is a monotonic "
                "integral and V a directly quantised measurement, so a second "
                "plateau shows there if it is real. "
                if extra
                else ""
            )
            + "Components closer together than the profile's own separation floor "
            "cannot be admitted at all, which is what enforces the constraint "
            "inside the plateau; components further out are outside the field the "
            "phase rule speaks about, and are judged against the noise instead.",
        )
    ]


def _shoulder_only_sentence(f):
    """Half-cycles the model fitted with no primary peak at all."""
    q = f.get("shoulder_only") or {}
    if not q:
        return []
    bits = []
    for step in ("Charge", "Discharge"):
        s = q.get(step)
        if not s:
            continue
        bits.append(
            f"**{step}**: {s['n']} of {s['total']} half-cycles, "
            f"cycles {s['first']}-{s['last']}"
        )
    if not bits:
        return []
    return [
        (
            "bad",
            "Some half-cycles were fitted with no primary peak",
            "; ".join(bits) + ". The peak picker found no resolved maximum "
            "on these, so the model was given the second-derivative "
            "SHOULDERS instead and the whole half-cycle is described by "
            "features the picker itself called shoulders of something. That "
            "is not a fit to be read as a decomposition — check those "
            "half-cycles against the dQ/dV figure before quoting anything "
            "from them.",
        )
    ]


def _census_sentence(f):
    """Is the reference cycle a thin fit, and do the fits vary in size?"""
    q = f.get("component_census") or {}
    if not q:
        return []
    bits, worst = [], "ok"
    for step in ("Charge", "Discharge"):
        s = q.get(step)
        if not s:
            continue
        _p = [
            f"**{step}**: {s['lo']}-{s['hi']} components per half-cycle "
            f"over {s['n_half_cycles']}, median {s['median']:.0f}"
        ]
        rn, rc = s.get("reference_n"), s.get("reference_cycle")
        if rn is not None and rc is not None:
            _p.append(f"; the reference cycle ({rc}) got {rn}")
            if rn <= s["lo"] and s["hi"] > s["lo"]:
                _p.append(
                    " — **the thinnest fit in the dataset**, and every "
                    "tracked feature, every drift rate and the whole "
                    "attribution basis is derived from it"
                )
                worst = "warn"
            elif rn < s["median"]:
                _p.append(", below the median for this dataset")
                if worst == "ok":
                    worst = "warn"
        bits.append(" ".join(_p).replace(" ;", ";") + ".")
    if not bits:
        return []
    head = (
        "The attribution basis is a thin fit"
        if worst == "warn"
        else "How many components each half-cycle was fitted with"
    )
    tail = (
        " The reference cycle is chosen on the cell's integrity, not on "
        "how well the model fitted it, and that is the right criterion — "
        "but a basis taken from the thinnest fit in the run is worth "
        "knowing about before the shares below are read as capacities."
        if worst == "warn"
        else " The attribution basis and every tracked feature come from the "
        "reference cycle, so this is how much of the curve that basis "
        "was built from."
    )
    return [(worst, head, " ".join(bits) + tail)]


def _peak_sentences(f):
    """Findings about the peaks: are their areas capacities, and how fast do
    they move? Separate from `_sentences` only because it needs the tracking,
    which a cycling-only run does not have."""
    out = (
        _quality_sentence(f)
        + _voltage_envelope_sentence(f)
        + _identifiability_sentence(f)
        + _reliability_sentence(f)
        + _lineshape_sentence(f)
        + _shoulder_only_sentence(f)
        + _census_sentence(f)
        + _mechanism_sentence(f)
        + _phase_rule_sentence(f)
    )
    fid = f.get("integral_fidelity") or {}
    low = _fidelity_outside_band(fid)
    if low:
        # The value furthest from 1, whichever side of the band it is on.
        worst = max(low.values(), key=lambda v: abs(v - 1.0))
        _short = worst < 1.0
        # BOTH EXPLANATIONS WHERE BOTH FAULTS ARE PRESENT. A curve can be
        # short on one step and over on the other, and those are opposite
        # faults with opposite causes; picking the explanation from the single
        # worst value attached the under-count reason to an over-count.
        _under = (
            "Charge delivered outside the analysed voltage window "
            "cannot appear in the integral — a constant-voltage hold "
            "is the usual reason, and on a finite-difference curve a "
            "flat two-phase plateau is another: the voltage change "
            "between records falls below the instrument's resolution "
            "and the charge delivered there has no dV to be divided "
            "by. No choice of fitting model recovers it."
        )
        _over = (
            "A curve cannot contain more charge than the cell "
            "delivered, so the excess is an artefact — a voltage "
            "traversed twice, or a hold spread across voltages the "
            "cell never visited. No choice of fitting model removes "
            "it."
        )
        _lo_steps = sorted(k for k, v in low.items() if v < 1.0)
        _hi_steps = sorted(k for k, v in low.items() if v > 1.0)
        if _lo_steps and _hi_steps:
            _why = (
                f"These are opposite faults. On "
                f"{' and '.join(s.lower() for s in _lo_steps)}: "
                + _under
                + f" On {' and '.join(s.lower() for s in _hi_steps)}: "
                + _over
            )
        else:
            _why = _under if _lo_steps else _over
        # PER STEP, NOT THE WORST OF THE TWO APPLIED TO BOTH. `worst` is the
        # single value furthest from 1, and the closing clause used it to
        # describe every area on the page: NNM cell B was told "accounts for
        # only 5% of the charge, 56% of the discharge capacity ... Every area
        # and area-retention figure on this page is a share of that 5%", an
        # order of magnitude wrong for half its own figures. Where the two
        # steps differ, name both. The leading verb has the same fault: it is
        # picked from `worst` alone, so a curve carrying 145% of one step and
        # 56% of the other read "accounts for only 145%".
        _lo_all = all(v < 1.0 for v in low.values())
        _hi_all = all(v > 1.0 for v in low.values())
        _verb = (
            "accounts for only" if _lo_all else "carries" if _hi_all else "accounts for"
        )
        if len(low) > 1 and (max(low.values()) - min(low.values())) >= 0.05:
            _share_clause = (
                " Every area and area-retention figure on this page is a share "
                "of its own step's figure — "
                + ", ".join(
                    f"{100 * v:.0f}% for {k.lower()}" for k, v in sorted(low.items())
                )
                + " — and the two are not interchangeable. Read them as "
                "descriptions of the curve, not of the cell. Peak POSITIONS "
                "and their drift do not depend on the integral and are "
                "unaffected."
            )
        else:
            _share_clause = (
                f" Every area and area-retention figure on this page is a "
                f"share of that {100 * worst:.0f}% — read them as "
                f"descriptions of the curve, not of the cell. Peak POSITIONS "
                f"and their drift do not depend on the integral and are "
                f"unaffected."
            )
        out.append(
            (
                "bad",
                "Peak areas here are not capacities",
                f"The dQ/dV curve {_verb} "
                + ", ".join(
                    f"{100 * v:.0f}% of the {k.lower()}" for k, v in sorted(low.items())
                )
                + " capacity this cell delivered. "
                + _why
                + _share_clause,
            )
        )
    # THE COHERENCE VERDICT REACHES THIS PAGE TOO. `f["coherence"]` was
    # assembled for the run summary and rendered nowhere here, so on a
    # dataset where the audit graded ZERO peaks coherent this section still
    # printed every drift rate as a green finding — while the run-level page
    # one folder up said the same peaks were excluded because none of them
    # moved like a redox feature. Two Ratatosk pages, opposite claims.
    _verd = {
        (c["step"], round(c["reference_voltage"], 3)): c["verdict"]
        for c in (f.get("coherence") or [])
    }

    def _grade(t):
        return _verd.get(
            (t.get("step"), round(float(t.get("reference_voltage", np.nan)), 3))
        )

    moved = [
        t
        for t in (f.get("tracked") or [])
        if np.isfinite(t.get("voltage_drift_mV_per_cycle", np.nan))
    ]
    _sound = [t for t in moved if _grade(t) in (None, "coherent", "questionable")]
    _rejected = [t for t in moved if t not in _sound]
    if _sound:
        bits = []
        for t in _sound:
            _g = _grade(t)
            bits.append(
                f"{t['step'].lower()} {t['reference_voltage']:.3f} V "
                f"{t['voltage_drift_mV_per_cycle']:+.2f} mV/cycle"
                + (" (questionable)" if _g == "questionable" else "")
            )
        out.append(
            (
                "ok" if any(_grade(t) == "coherent" for t in _sound) else "warn",
                "How fast the features move",
                "; ".join(bits) + ". Measured between medians of three "
                "cycles at each end, from the reference cycle onward — a "
                "difference of two single cycles carries the per-cycle "
                "centre scatter and can be out by 15 mV."
                + (
                    f" {len(_rejected)} further peak(s) moved too, and are "
                    f"not quoted: the coherence audit found they do not "
                    f"move like a redox feature."
                    if _rejected
                    else ""
                ),
            )
        )
    elif _rejected:
        out.append(
            (
                "warn",
                "No drift rate is quotable",
                f"All {len(_rejected)} tracked peak(s) with a measurable "
                f"drift were rejected by the coherence audit — none of "
                f"them moves like a redox feature. The rates are in "
                f"the fit-coherence table with their verdicts; they are "
                f"not reported here because a rate read from a peak the "
                f"fit could not hold on to is a property of the fit.",
            )
        )
    # DETECTED, ROSTERED, AND NEVER GIVEN TO THE MODEL.
    #
    # The tracked-peak roster is built from the DETECTED reference list while
    # the fit may be given a subset of it (`primaries_only`, chosen by the
    # mechanism classifier). Those features were recorded as `disappeared` in
    # every cycle — 3 of 7 discharge features on NMC111, tracked in zero of
    # 18 cycles — and this page advertised them as tracked peaks that had
    # gone. "Gone" is a finding about the cell; "never fitted" is a fact
    # about the run, and only one of them is true.
    _never = [t for t in (f.get("tracked") or []) if t.get("never_fitted")]
    if _never:
        out.append(
            (
                "warn",
                "Some detected features were never fitted",
                ", ".join(
                    f"{t['step'].lower()} {t['reference_voltage']:.3f} V"
                    for t in _never
                )
                + f" — {len(_never)} feature(s) that detection found and the "
                "model was never asked to fit, because the mechanism this run "
                "classified fits primary peaks only. They are absent from the "
                "fit, not from the cell, and they carry no area, no drift and "
                "no retention here. If they matter, the mechanism verdict is "
                "the thing to look at.",
            )
        )
    # NOT ON AREAS THE RUN HAS ALREADY WITHHELD. `flag_area_growth` compares
    # area retention with capacity retention; `flag_untrustworthy_areas` decides
    # separately whether an area retention is a capacity at all, and writes
    # `area_retention_trustworthy=False` with the reason. NNM cell B had all ten
    # peaks withheld — "the dQ/dV curve accounts for only 5% of this step's
    # delivered capacity, so a peak area is not a capacity" — and this finding
    # still printed "129% area retention, excess 114 points" and concluded the
    # decomposition was wrong, four paragraphs below the page's own statement
    # that these are not capacities. A comparison between two numbers is only as
    # good as the weaker of them.
    grown = [
        t
        for t in (f.get("tracked") or [])
        if t.get("area_exceeds_capacity")
        and t.get("area_retention_trustworthy") is not False
    ]
    _grown_withheld = [
        t
        for t in (f.get("tracked") or [])
        if t.get("area_exceeds_capacity")
        and t.get("area_retention_trustworthy") is False
    ]
    if not grown and _grown_withheld:
        _wr = next(
            (
                str(t.get("area_withheld_reason") or "")
                for t in _grown_withheld
                if t.get("area_withheld_reason")
            ),
            "",
        )
        out.append(
            (
                "ok",
                "Area growth could not be assessed",
                f"{len(_grown_withheld)} peak(s) have an area retention above what "
                f"the cell's capacity allows, but every one of them is a peak whose "
                f"area this run has already withheld as not being a capacity"
                + (f" — {_wr}" if _wr else "")
                + ". A ratio of two areas that are not capacities cannot be "
                "compared with a capacity retention, so no verdict is offered "
                "here. The excesses are in `*_tracked_peaks_summary.csv` "
                "(`area_growth_excess_pct`) for anyone who wants to look.",
            )
        )
    if grown:
        # NOT "GREW". `area_exceeds_capacity` means the peak's area retention
        # is HIGHER THAN THE CELL'S capacity retention plus headroom — not
        # that the area rose. On NNM cell A, whose capacity retention is 34%,
        # this headed a list of peaks at 49% and 22% area retention with "A
        # peak grew while the cell shrank"; every one of them had shrunk.
        # And the 34% the comparison is against was never printed, which is
        # the only number that makes the list mean anything.
        _cap = f.get("area_growth_basis_pct")
        # AND THE HEADROOM, because the excess is measured from the SUM.
        # "214% area retention, excess 124 points ... against the cell's 75%"
        # invites the reader to check 214 - 75 = 139 and conclude one of the
        # three numbers is wrong. The bound is 75 + 15.
        _against = (
            f" against the cell's {_cap:.0f}% capacity retention plus "
            f"{AREA_GROWTH_HEADROOM_PCT:.0f} points of headroom, so "
            f"the excesses above are measured from {_cap + AREA_GROWTH_HEADROOM_PCT:.0f}%"
            if _cap is not None and np.isfinite(_cap)
            else ""
        )
        out.append(
            (
                "warn",
                "Peak areas held up better than the cell's capacity",
                ", ".join(
                    f"{t['step'].lower()} {t['reference_voltage']:.3f} V "
                    f"({t['area_retention_pct']:.0f}% area retention"
                    + (
                        f", excess {t['area_growth_excess_pct']:.0f} points"
                        if t.get("area_growth_excess_pct") is not None
                        and np.isfinite(t.get("area_growth_excess_pct"))
                        else ""
                    )
                    + ")"
                    for t in grown
                )
                + _against
                + ". A redox feature cannot keep more charge than the cell "
                "delivers, so where this appears the decomposition is wrong — "
                "not the cell. Nothing has been deleted; look at the fit.",
            )
        )
    return out


def _sentences(f):
    """
    The findings, as prose, shared by both renderers.

    Returns a list of (level, heading, text). Level is 'ok', 'warn' or 'bad'
    and is the only styling either renderer applies to a finding.
    """
    out = []
    n = f.get("n_cycles")
    dl = f.get("first_last_discharge")
    ret, ce = f.get("retention"), f.get("median_ce")

    # FIRST, BEFORE ANYTHING ELSE ON THE PAGE. A reader who opens this file
    # must not have to reach the coulombic-efficiency paragraph to discover
    # that the export is not a measurement. See UNUSABLE_MIN_CYCLES.
    if f.get("unusable"):
        out.append(
            (
                "bad",
                "This dataset is not a usable measurement",
                _sentence_case("; ".join(f["unusable"]))
                + ". Everything below is still computed from what the "
                "file contains, because that is how you find out what "
                "went wrong — but none of it is a result, and this cell "
                "is left out of the replicate mean rather than "
                "averaged into it. Check the export: a file much "
                "shorter than its replicates is usually a run that was "
                "stopped or a save that did not finish.",
            )
        )

    if dl and n:
        first, last = dl
        lg = f.get("last_good_cycle")
        died = f.get("fit_limit") or f.get("death_cycle")
        # "Runs to cycle N" is wrong for an export taken mid-run: the file
        # ends at cycle N because that is when it was written, not because
        # the experiment did.
        s = (
            (
                f"The export covers {n} cycles, the last of them still running. "
                if f.get("partial_final")
                else f"The record runs to cycle {n}. "
            )
            + f"The cell delivered "
            f"{first:.0f} mAh/g on its first complete discharge and "
            f"{last:.0f} mAh/g"
        )
        s += (
            f" at cycle {lg}, the last one it was still working"
            if died and lg
            else (f" at cycle {lg}" if lg else " on the last")
        )
        # NAME THE CYCLE. "99% of the reference cycle" sat three lines under
        # a table column headed "% of first", which is a DIFFERENT base — and
        # the reference is read from a column literally called
        # `Retention_vs_C2_%`, so the number was available all along.
        s += (
            f" — {ret:.0f}% of cycle {f.get('retention_reference') or 2}."
            if ret is not None and not _rate_variable(f)
            else "."
        )
        if died and n and died < n:
            s += (
                f" The {n - died} cycles after that are the cycler still "
                f"running against a cell that had stopped."
            )
        out.append(("ok", "What it did", s))
    pf = f.get("partial_final")
    if pf:
        cyc, step, prev, val = pf
        got = (
            f"It had delivered {val:.0f} mAh/g of cycle {prev}'s "
            f"{dl[1]:.0f} when the file was written. "
            if (val is not None and prev and dl)
            else ""
        )
        out.append(
            (
                "ok",
                "The cell was still cycling when this was exported",
                f"Cycle {cyc} {step} never reached the cut-off voltage "
                f"that every other half-cycle going the same way "
                f"reaches, and it is the last half-cycle in the file — "
                f"so the export was taken while it was still running. "
                f"{got}A half-cycle in progress is not a measurement "
                f"yet, so it is not fitted and is left out of the "
                f"capacities and the retention above; it is still drawn "
                f"in every figure, and the completed half of the same "
                f"cycle is kept. Re-export once it finishes and it "
                f"counts like any other. Everything above describes "
                f"cycles 1 to {cyc - 1}.",
            )
        )
    if ce is not None:
        lvl = "ok" if 95 <= ce <= 101 else "warn"
        out.append(
            (
                lvl,
                "Coulombic efficiency",
                f"Median {ce:.1f}%. Below about 95% means charge is going "
                f"somewhere other than the intended reaction; well above "
                f"100% usually means the counter electrode, not the "
                f"material under test.",
            )
        )

    bands = f.get("bands", {})
    anom = f.get("anomalous")
    if anom is not None and len(anom):
        worst = anom.iloc[0]
        cycles = ", ".join(
            str(c) for c in sorted(int(c) for c in anom["cycle"].head(8))
        )
        out.append(
            (
                "bad",
                f"{len(anom)} half-cycle(s) lost charge",
                f"Charge passed at full current where |dQ/dV| was small "
                f"and away from the voltage limit — no redox process "
                f"accounts for it. Worst: cycle {int(worst['cycle'])} "
                f"{worst['step']}, {worst['parasitic_charge']:.0f} mAh/g "
                f"of {worst['capacity']:.0f} "
                f"({worst['parasitic_fraction']:.0%}). Cycles: {cycles}.",
            )
        )
    elif bands and not (
        f.get("over_theoretical") is not None and len(f["over_theoretical"])
    ):
        # SAID FIRST, AND FOR EITHER BRANCH. The "not every half-cycle is
        # clean" caveat lived inside the `else` only, so a cell whose mechanism
        # was anything but two-phase — NMC111 cell A, integrity table
        # {clean: 36, suspect: 1} — was told "Every half-cycle's reversed
        # charge sat either on a flat region or at the voltage limit" with a
        # suspect row sitting in the table whose `reason` column says exactly
        # what was seen. The guard is about the table, not about the mechanism,
        # so it cannot be conditioned on the mechanism.
        _grey = {
            k: int(v)
            for k, v in (bands or {}).items()
            if str(k) in ("suspect", "unknown") and v
        }
        if _grey:
            out.append(
                (
                    "warn",
                    "Not every half-cycle is clean",
                    "; ".join(f"{v} {k}" for k, v in _grey.items())
                    + ". None passed more charge than the material "
                    "can hold, so none is anomalous — but they are "
                    "not clean either, and the `reason` column of "
                    "`*_cycle_integrity.csv` says what was seen on "
                    "each.",
                )
            )
        _m2 = str((f.get("resolvability") or {}).get("mechanism") or "")
        if _m2 and _m2 not in ("two_phase", "multi_transition"):
            # Same fault, one paragraph up: "it sat on a plateau" is only an
            # innocent explanation where a plateau is expected.
            out.append(
                (
                    "warn",
                    "Reversed charge, on a plateau this mechanism should not have",
                    f"Every half-cycle's reversed charge sat either on a "
                    f"flat region or at the voltage limit with the current "
                    f"tapering. The second is expected. The first is not, "
                    f"on a dataset classified "
                    f"**{_m2.replace('_', ' ')}** — see below.",
                )
            )
        else:
            # The suspect/unknown caveat is now raised once, above, for
            # either mechanism branch. What is left here is the all-clear,
            # and it is only an all-clear when the table is in fact clean.
            if not _grey:
                out.append(
                    (
                        "ok",
                        "No unaccounted charge",
                        "Every half-cycle's reversed charge sat either on "
                        "a two-phase plateau or at the voltage limit with "
                        "the current tapering. Both are expected.",
                    )
                )
            else:
                out.append(
                    (
                        "ok",
                        "The rest is accounted for",
                        "Setting aside the half-cycles named above, every "
                        "half-cycle's reversed charge sat either on a "
                        "two-phase plateau or at the voltage limit with "
                        "the current tapering. Both are expected.",
                    )
                )
    if bands.get("too few records"):
        out.append(
            (
                "warn",
                f"{bands['too few records']} half-cycle(s) too short to judge",
                "Usually the tail of the record after the cell stopped "
                "delivering. Not the same as clean: it means we could not "
                "tell.",
            )
        )
    # EVERY VERDICT IS RENDERED, including the good ones. While only the bad
    # ones printed, an unreadable integrity table and a healthy cell produced
    # the same page — nothing — which is a silence that reads as reassurance.
    _civ = f.get("cell_integrity") or {}
    if _civ.get("verdict") == "unknown":
        out.append(
            ("warn", "Cycle integrity could not be assessed", _civ.get("sentence", ""))
        )
    elif _civ.get("verdict") == "sound":
        out.append(("ok", "No cycle was flagged", _civ.get("sentence", "")))
    elif _civ.get("verdict") == "recurring":
        out.append(("bad", "This cell has a recurring fault", _civ.get("sentence", "")))
    elif _civ.get("verdict") == "formation":
        out.append(("ok", "The flagged cycles are all early", _civ.get("sentence", "")))
    elif _civ.get("verdict") == "clustered":
        out.append(
            (
                "warn",
                "A run of flagged cycles, which then stopped",
                _civ.get("sentence", ""),
            )
        )
    elif _civ.get("verdict") == "isolated":
        out.append(
            ("warn", "Flagged cycles, with no pattern", _civ.get("sentence", ""))
        )
    over = f.get("over_theoretical")
    if over is not None and len(over):
        w = over.iloc[0]
        # `over` is sorted by ratio, so `head(8)` is the EIGHT WORST and not
        # the first eight — and the sentence opens by counting all of them.
        # On NNM cell A that printed "9 half-cycle(s) passed more charge than
        # the material can hold ... Cycles: 1, 13, 19, 26, 34, 44, 54, 69":
        # nine claimed, eight listed, cycle 6 (the mildest, 1.23x) silently
        # dropped. A count that does not match its own list is the same fault
        # as a caption that does not match its figure.
        _cyc = sorted(int(c) for c in over["cycle"].head(8))
        cy = ", ".join(str(c) for c in _cyc)
        if len(over) > len(_cyc):
            cy += f" (the {len(_cyc)} worst of {len(over)} by ratio)"
        out.append(
            (
                "bad",
                f"{len(over)} half-cycle(s) passed more charge than "
                f"the material can hold",
                f"Worst: cycle {int(w['cycle'])} {w['step']}, "
                f"{w['capacity']:.0f} mAh/g — {w['capacity_ratio']:.1f}× "
                f"the theoretical {w['theoretical_mAh_g']:.0f} mAh/g. Over "
                f"theoretical is not a phase transition and not a "
                f"measurement artefact; it is charge going into something "
                f"other than the material, and it usually means something "
                f"has broken. Cycles: {cy}."
                + (
                    " These half-cycles are still fitted, tracked and "
                    "drawn — the figure is where you see what happened — "
                    "but their components carry "
                    "`half_cycle_anomalous` and are left out of the drift "
                    "rates, the area retention and the coherence audit. "
                    "Integral fidelity does NOT catch them: the curve "
                    "accounts for the charge faithfully, because the "
                    "charge really was delivered."
                    if f.get("n_anomalous_components")
                    else ""
                ),
            )
        )
    # THE SAME THRESHOLD ITS OWN TAIL CLAUSE USES. This gate was `> 0`
    # while the "a further N mAh/g" clause twenty lines below is `> 1`, so
    # NMC111 cell A — 0.79 mAh/g of plateau charge out of a 277 mAh/g
    # half-cycle, 0.3% of one half-cycle — earned the loudest mark on the
    # page, with `:.0f` rounding 0.79 up to a printed "1".
    if f.get("plateau_sane", 0) >= PLATEAU_REPORTABLE_MAH_G:
        # IS A PLATEAU EXPECTED PHYSICS HERE, OR IS IT THE FAULT?
        #
        # This paragraph called every flat region "a first-order phase
        # transition holding the potential, not a fault" — a correct reading
        # for LTO and the WRONG one for a solid solution, whose potential must
        # slope with composition because there is no two-phase coexistence to
        # pin it. A flat region in a solid solution's charge curve is
        # therefore evidence OF a fault, not evidence against one.
        #
        # The run already knows which case it is in. `assess_resolvability`
        # classifies the mechanism on the reference cycle before anything is
        # fitted, and this check — one module away — was not consulting it. It
        # printed the two-phase sentence for NNM (a P3 solid solution) and for
        # NMC111 cells the same run had just classified `mixed`.
        _mech = str((f.get("resolvability") or {}).get("mechanism") or "")
        _pinned = _mech in ("two_phase", "multi_transition")
        _tail = (
            f" A further {f['plateau_total'] - f['plateau_sane']:.0f} "
            f"mAh/g sat on a plateau in half-cycles that were over "
            f"theoretical, and is not counted here — the shape of a "
            f"curve cannot vouch for charge the material could not "
            f"have held."
            if f.get("plateau_total", 0) - f.get("plateau_sane", 0) > 1
            else ""
        )
        # A TOTAL, SAID TO BE A TOTAL. `plateau_sane` is
        # `integrity["plateau_charge"].sum()` over every half-cycle in the
        # record, and it was printed bare in mAh/g on a page whose header
        # gives the theoretical capacity in the same unit: LTO cell A read
        # "222.2 mAh/g passed backwards" against a 175 mAh/g theoretical and a
        # cell delivering 130, which reads as a physical impossibility rather
        # than as 21 half-cycles summed. The per-half-cycle worst case is the
        # number that can be compared with a capacity.
        _pn = int(f.get("plateau_n_half_cycles") or 0)
        _pw = f.get("plateau_worst_mAh_g")
        _lead = (
            f"{f['plateau_sane']:.1f} mAh/g in total"
            + (f" across {_pn} half-cycles" if _pn else "")
            + " passed backwards while sitting on a redox plateau, on "
            "half-cycles whose total charge is physically possible"
            + (
                f" — at most {_pw:.1f} mAh/g in any one of them"
                if _pw is not None and np.isfinite(_pw)
                else ""
            )
            + "."
        )
        if not _mech:
            out.append(
                (
                    "ok",
                    "Plateau detected",
                    _lead + " On a material whose potential is pinned by "
                    "two phases coexisting that is expected physics; on a "
                    "solid solution it is not. This run did not classify "
                    "the mechanism, so which of the two this is has not "
                    "been established here." + _tail,
                )
            )
        elif _pinned:
            out.append(
                (
                    "ok",
                    "Two-phase plateau detected",
                    _lead + f" This dataset is classified "
                    f"**{_mech.replace('_', ' ')}**, so a region where the "
                    f"potential holds still is the transition itself: a "
                    f"first-order phase change pinning the potential, not "
                    f"a fault." + _tail,
                )
            )
        else:
            out.append(
                (
                    "bad",
                    "A plateau here is not expected physics",
                    _lead + f" This dataset is classified "
                    f"**{_mech.replace('_', ' ')}**. A solid solution "
                    f"delivers its charge across a window of composition, "
                    f"and its potential must move as that composition "
                    f"changes — there is no two-phase coexistence to pin "
                    f"it. A flat region in this cell's curve is therefore "
                    f"evidence OF something, not evidence against it: a "
                    f"voltage stall, a hold, or charge going somewhere "
                    f"other than the intended reaction. Earlier builds "
                    f"reported this as a first-order phase transition; "
                    f"that reading belongs to the other mechanism." + _tail,
                )
            )
    if f.get("fit_limit") or f.get("death_cycle"):
        out.append(
            (
                "warn",
                f"Cell stopped delivering at cycle "
                f"{f.get('fit_limit') or f.get('death_cycle')}",
                f"Peak fitting stopped there. The plots still show all "
                f"{n} cycles, because a cell going flat is exactly what "
                f"the heatmap is for.",
            )
        )

    w, tot = f.get("withheld", (0, 0))
    cw = f.get("closure_widths")
    if tot:
        if w:
            # WHICH GROUND. Attribution is withheld either because the
            # peak/background split is undetermined (closure) or because the
            # curve does not account for the cell's charge (fidelity), and
            # naming the wrong one sends the reader to look at the baseline
            # when the problem is the instrument's voltage resolution.
            _fid = _fidelity_outside_band(f.get("integral_fidelity"))
            if _fid:
                # SHORT, and pointing at the finding that explains it. The
                # full argument is one item below under "Peak areas here are
                # not capacities"; repeating it here made the page read as
                # two separate problems when it is one.
                why = (
                    "There is no capacity here to attribute — see *Peak "
                    "areas here are not capacities* below. The fitted "
                    "areas are still in the CSV."
                )
                # NOT WHEN THERE IS NO FREE BACKGROUND. `no_baseline` means
                # every measured closure width is zero BY CONSTRUCTION, so
                # this printed "closure interval only 0.00–0.00 wide: the
                # fits are good" — reassurance drawn from a measurement that
                # could not have come out any other way, and in direct
                # contradiction of the branch below, which says the split
                # "cannot be undetermined" because there is nothing to
                # determine.
                if cw and not f.get("no_baseline"):
                    why += (
                        f" Note that the peak/background split itself is "
                        f"well determined, closure interval only "
                        f"{cw[0]:.2f}–{cw[1]:.2f} wide: the fits are "
                        f"good, and that is a different question from "
                        f"the one this fails."
                    )
            elif f.get("unattributed_withheld"):
                _u = f.get("unattributed")
                _ur = f.get("unattributed_range")
                why = (
                    "The named components do not add up to the cell's "
                    "charge. Every mAh in a dQ/dV passed through the "
                    "cell, so a share OF a total that is missing part of "
                    "the capacity is not a measurement — the fitted areas "
                    "are still in the CSV, and the SHARE is blank."
                    + (
                        f" Unattributed {_u:+.0%} of the charge"
                        + (f" (range {_ur[0]:+.0%} to {_ur[1]:+.0%})" if _ur else "")
                        + "."
                        if _u is not None
                        else ""
                    )
                )
            elif f.get("no_baseline"):
                why = (
                    "This model has no free background, so the "
                    "peak/background split cannot be undetermined. These "
                    "cycles were withheld because the check fit did not "
                    "converge on them: there is no measured share to "
                    "report. The fitted areas are still in the CSV."
                )
            else:
                why = (
                    "Where a broad peak and the polynomial baseline are "
                    "nearly the same function, the split between them is "
                    "not decided by the data: refitting with a different "
                    "baseline moves the peak's share of the capacity by "
                    "more than the number would be reporting. The fitted "
                    "areas are still in the CSV; the SHARE is blank "
                    "because it is not a measurement."
                    + (
                        f" Closure interval {cw[0]:.2f}–{cw[1]:.2f} wide."
                        if cw and not f.get("no_baseline")
                        else ""
                    )
                )
            # HALF-CYCLES. `w` and `tot` are summed over both steps, so a
            # 67-cycle cell was told "134 of 134 cycles" — a count of twice
            # what it ran, in the one unit this page otherwise says
            # carefully everywhere else.
            out.append(
                (
                    "warn",
                    f"Capacity attribution withheld on {w} of {tot} half-cycles",
                    why,
                )
            )
        elif f.get("no_baseline"):
            _u = f.get("unattributed")
            out.append(
                (
                    "ok",
                    "Capacity attribution is reportable",
                    "There is no free background in this model, so there "
                    "is no peak/background split to be undetermined — "
                    "every component is named and its area is a share of "
                    "the cell rather than of a polynomial."
                    + _unattributed_clause(_u, f.get("unattributed_range"))
                    + " A HIGH NAMED FRACTION IS NOT BY ITSELF A BETTER "
                    "DECOMPOSITION: a wide band can absorb a great "
                    "deal, and a few resolved transitions with an "
                    "honest few percent left over is the better "
                    "answer. Read it beside the mechanism.",
                )
            )
        else:
            out.append(
                (
                    "ok",
                    "Capacity attribution is reportable",
                    "Refitting with a different baseline degree barely "
                    "moved the peak/background split"
                    + (
                        f" (closure interval {cw[0]:.2f}–{cw[1]:.2f} wide)"
                        if cw
                        else ""
                    )
                    + ", so the per-peak shares mean "
                    "what they say.",
                )
            )

    if f.get("reference_cycle"):
        # The severity is now a VALUE carried from `detect.detect_all`, not a
        # substring search on the sentence. The search read
        # `"warn" if "failed" in reason else "ok"` and was exactly inverted:
        # "NO cycle passes the integrity check — using 3 unchecked" and
        # "FLAGGED ANOMALOUS by the integrity check" contain no "failed" and
        # rendered green, while the benign "cycle 5 failed the integrity
        # check; 6 is the first that passes" rendered as a warning.
        lvl = f.get("reference_severity") or "ok"
        out.append(
            (
                lvl,
                f"Peaks are tracked against cycle {f['reference_cycle']}",
                _sentence_case(f.get("reference_reason", ""))
                + ". That cycle's peak list "
                f"is fitted in every other cycle, so it decides what the "
                f"whole dataset is measured against.",
            )
        )
    bad_peaks = [s for s in f.get("tracked", []) if s.get("discontinuity_is_failure")]
    if bad_peaks:
        out.append(
            (
                "warn",
                f"{len(bad_peaks)} tracked peak(s) failed mid-life",
                ", ".join(
                    f"{s['step']} {s['reference_voltage']:.3f} V "
                    f"from cycle "
                    f"{int(s['first_discontinuity_cycle'])}"
                    for s in bad_peaks
                )
                + ". Each was reliably tracked for at least 15 cycles "
                "before it went, so this is a feature being lost, not a "
                "feature that never established.",
            )
        )
    return out


def _experiment(f):
    """
    What was done: the cell, not its performance.

    First, because a capacity means nothing until you know the mass it is per,
    the rate it was measured at and the window it was measured in.
    """
    p = f["params"]
    src = " (from file)" if p.get("barcode") else ""
    out = []
    if p.get("active_material_mass_mg"):
        out.append(
            (
                "Active mass",
                f"{p['active_material_mass_mg']:.2f} mg",
                (p.get("blend") or "") + src,
            )
        )
    if p.get("theoretical_capacity_mAh_g"):
        out.append(("Theoretical", f"{p['theoretical_capacity_mAh_g']:.0f}", "mAh/g"))
    # THE RATE THE CELL WAS CYCLED AT, not the one in the cycler header. A
    # rate-capability run reports its range and its block count; a
    # single-rate run reads exactly as it did before. `rate_protocol` is
    # attached to the parameters in Cell 6a. See `cycling.describe_rate`.
    _rp = p.get("rate_protocol")
    if _rp and _rp.get("available") and _rp.get("is_variable"):
        out.append(
            (
                "Rate",
                str(_rp.get("label")),
                f"{_rp.get('n_blocks')} rate blocks — variable-rate run",
            )
        )
    elif p.get("charge_rate_c"):
        # When the current was recorded, say what it MEASURED as well as what
        # was entered — the two disagreeing is worth seeing, and the entered
        # value comes from a header that only describes the first step.
        _m = (_rp or {}).get("label")
        out.append(("Rate", f"{p['charge_rate_c']} C", f"measured {_m}" if _m else ""))
    if p.get("voltage_lower_V") and p.get("voltage_upper_V"):
        out.append(
            (
                "Window",
                f"{p['voltage_lower_V']}–{p['voltage_upper_V']} V",
                "from the cycler file"
                if p.get("voltage_window_source") == "file"
                else (
                    "at least this wide — short record"
                    if str(p.get("voltage_window_source", "")).startswith(
                        "observed range"
                    )
                    else "as set"
                ),
            )
        )
    if p.get("active_loading_mg_cm2"):
        out.append(
            (
                "Loading",
                f"{p['active_loading_mg_cm2']:.2f}",
                f"mg/cm² on {p.get('electrode_diameter_mm', 0):.0f} mm",
            )
        )
    if p.get("counter_electrode_metal"):
        # WHERE THE METAL CAME FROM, not just that the mass was estimated. A
        # counter metal that was ASSUMED sets the density used in every areal
        # and gravimetric energy figure (Li 0.534 against Na 0.97 g/cm3), and
        # reading "estimated" gave no clue that the element itself was a guess.
        _src = {
            "stated": "as stated",
            "chemistry": "from the electrolyte / formula",
            "library": "from the composition library — confirm it",
            "assumed": "ASSUMED — chemistry could not be inferred",
        }.get(p.get("counter_electrode_metal_source"), "")
        _mass = "measured" if p.get("counter_electrode_measured") else "mass estimated"
        out.append(
            (
                "Counter",
                p["counter_electrode_metal"],
                f"{_mass}{(', ' + _src) if _src else ''}",
            )
        )
    if f.get("n_cycles"):
        _died = f.get("fit_limit") or f.get("death_cycle")
        if f.get("partial_final"):
            # The count of COMPLETE cycles, which is what the numbers above
            # this tile are drawn from. The record length goes in the note.
            out.append(
                (
                    "Cycles complete",
                    str(f["partial_final"][0] - 1),
                    f"cycle {f['partial_final'][0]} still running",
                )
            )
        else:
            out.append(
                (
                    "Cycles run",
                    str(f["n_cycles"]),
                    f"working to {_died}" if _died else "",
                )
            )
    return out


# RETENTION ACROSS RATE BLOCKS IS NOT RETENTION. On a rate-capability run
# the capacity falls because the current rose and comes back when it falls
# again, and quoting `Q(n)/Q(ref)` across that reads as a cell fading and
# recovering. On `JQ_NNM_C-rate_2-4.2V_B_29042026` it produced 78.5% and a
# key-capacity column of 75%, 64%, 80% — every one of which a reader would
# take for degradation. The figure is not wrong arithmetic; it is the wrong
# question, so it is withheld and the reason is given.
def _rate_variable(f):
    return bool(((f.get("params") or {}).get("rate_protocol") or {}).get("is_variable"))


def _rate_caveat(f):
    rp = (f.get("params") or {}).get("rate_protocol") or {}
    if not rp.get("is_variable"):
        return None
    return (
        f"This cell was cycled at {rp.get('n_blocks')} different rates "
        f"({rp.get('label')}). Capacity against cycle number here is "
        f"mostly the rate schedule, not age: it falls because the "
        f"current rose and returns when the current falls again. "
        f"Retention and percent-of-first are therefore withheld — they "
        f"would read as fading and recovery. The rate-capability summary "
        f"compares each rate against the reference rate, and the "
        f"recovery to the starting rate is reported there."
    )


def _stats(f):
    """The results tiles: what the cell delivered."""
    p = f["params"]
    out = []
    # Discharge capacity leads, because that is what a paper reports.
    _var = _rate_variable(f)
    # NOT `[:4]`. The tiles stopped at four, so a run with key cycles
    # 1, 5, 10, 20, 30, 40, 50, 128 showed the first four and the page then
    # said "Retention 34% at cycle 128" with no capacity for cycle 128
    # anywhere on it. The Markdown table always printed them all.
    for cyc, d, pct in f.get("key_capacities", []):
        out.append(
            (
                f"Cycle {cyc}",
                f"{d:.0f}",
                "mAh/g" + (f"  ·  {pct:.0f}%" if pct is not None and not _var else ""),
            )
        )
    if _var:
        out.append(("Retention", "—", "withheld: several rates in this run"))
    elif f.get("retention") is not None and f.get("last_good_cycle"):
        out.append(
            (
                "Retention",
                f"{f['retention']:.0f}%",
                f"at cycle {f['last_good_cycle']}, "
                f"vs cycle {f.get('retention_reference') or 2}",
            )
        )
    if f.get("median_ce") is not None:
        out.append(("Median CE", f"{f['median_ce']:.1f}%", ""))
    return out


def build_report(
    name,
    *,
    dataset=None,
    params=None,
    integrity=None,
    detection=None,
    tracking=None,
    attribution=None,
    cycle_table=None,
    fit_limit=None,
    profile=None,
    dataset_dir=None,
    run_id="",
    file_format="png",
    in_progress=None,
    coherence=None,
    resolvability=None,
    parameters=None,
):
    """
    The two documents, as strings.

    `dataset_dir` is the per-dataset folder inside the run — the one holding
    `1_cycling`, `4_dqdv` and the rest. Figures are looked up there; if it is
    None or the figures are missing, both documents are still written and say
    so rather than failing.
    """
    f = _facts(
        name,
        params,
        integrity,
        cycle_table,
        fit_limit,
        detection,
        tracking,
        attribution,
        None,
        profile,
        in_progress=in_progress,
        coherence=coherence,
        resolvability=resolvability,
        parameters=parameters,
    )
    # SEVERITY FIRST, and stably so. `_peak_sentences` came second by
    # construction, so every `bad` finding about the peaks sat below every
    # routine `ok` one about the cycling — on NNM cell B the paragraph that
    # invalidates every area on the page was twelfth, under "Capacity
    # attribution is reportable", and a `warn` twelve lines above it
    # forward-referenced it.
    _SEVERITY = {"bad": 0, "warn": 1, "ok": 2}
    findings = sorted(
        _sentences(f) + _peak_sentences(f), key=lambda t: _SEVERITY.get(t[0], 3)
    )
    stats = _stats(f)
    experiment = _experiment(f)
    p = f["params"]

    # THE MECHANISM BELONGS IN THE HEADER, next to the profile class it is
    # constantly confused with. `profile` describes the SHAPE of the curve;
    # `mechanism` says what produced it, and it is the mechanism that chose
    # the model. NNM and NMC111 are both "broad" and one of them is a series
    # of real transitions.
    _mech = (f.get("resolvability") or {}).get("mechanism")
    subtitle = " · ".join(
        x
        for x in [
            p.get("electrode_type", "") and f"{p['electrode_type'].lower()} electrode",
            p.get("battery_chemistry", ""),
            f"profile {f['profile']}" if f.get("profile") else "",
            f"mechanism {str(_mech).replace('_', ' ')}" if _mech else "",
            run_id,
        ]
        if x
    )

    # FIGURE_GUIDE names its files with a .png suffix, but a run exported as
    # tiff writes *_cycle_life.tiff and every figure went missing from the
    # report of a run that had produced them.
    _ext = "." + str(file_format or "png").lstrip(".").lower()
    figs = []
    for _entry in FIGURE_GUIDE:
        folder, suffix, title, blurb = _entry[:4]
        run_wide = bool(_entry[4]) if len(_entry) > 4 else False
        suf = suffix[:-4] + _ext if suffix.endswith(".png") else suffix
        path = (
            _find(os.path.join(dataset_dir, folder), suf, name, run_wide)
            if dataset_dir
            else None
        )
        if path is None and dataset_dir and suf != suffix:
            path = _find(os.path.join(dataset_dir, folder), suffix, name, run_wide)
        figs.append((title, blurb, path, _rel(path, dataset_dir) if path else None))

    # ---------------------------------------------------------------- HTML
    h = [
        "<!doctype html><html lang='en-GB'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{_esc(f['title'])} — Ratatosk</title><style>{_CSS}</style>",
        "</head><body><div class='wrap'>",
        f"<h1>{_esc(f['title'])}</h1>",
        f"<p class='sub'>{_esc(subtitle)}</p>",
    ]

    def _tiles(rows):
        out = ["<div class='grid'>"]
        for k, v, n2 in rows:
            out.append(
                f"<div class='stat'><div class='k'>{_esc(k)}</div>"
                f"<div class='v'>{_esc(v)}</div>"
                f"<div class='n'>{_esc(n2)}</div></div>"
            )
        out.append("</div>")
        return out

    if experiment:
        h.append("<h2>The experiment</h2>")
        h += _tiles(experiment)
    if stats:
        h.append("<h2>Discharge capacity</h2>")
        h += _tiles(stats)
        _cav = _rate_caveat(f)
        if _cav:
            h.append(f"<p class='n'>{_esc(_cav)}</p>")
        else:
            h.append(
                "<p class='n'>Percentages are of the first complete discharge.</p>"
            )

    h.append("<h2>What the run found</h2>")
    for lvl, head, text in findings:
        # The Markdown renderer writes "**Head.** text"; the HTML ran the two
        # straight together with no stop and no space.
        h.append(
            f"<div class='flag {lvl}'><b>{_esc(head)}.</b> "
            f"{_md_inline(_esc(text))}</div>"
        )

    if f["peaks"].get("Charge") or f["peaks"].get("Discharge"):
        h.append("<h2>Redox features being tracked</h2>")
        _kinds = _kind_by_ref(f)
        h.append(
            "<table><tr><th>Step</th><th>Reference peak / V</th><th>Fitted as</th></tr>"
        )
        for step in ("Charge", "Discharge"):
            for x in f["peaks"].get(step) or []:
                h.append(
                    f"<tr><td>{step}</td><td>{x:.3f}</td><td>"
                    f"{_esc(_kinds.get((step, round(float(x), 3)), '—'))}"
                    f"</td></tr>"
                )
        h.append("</table>")
        if not _kinds:
            h.append(
                "<p class='n'>Nothing here was fitted — these are the "
                "detected reference features only.</p>"
            )

    h.append("<h2>Four figures, in this order</h2>")
    for i, (title, blurb, path, rel) in enumerate(figs, 1):
        h.append(
            f"<div class='fig'><h3>{i}. {_esc(title)}</h3>"
            f"<p class='cap'>{_esc(blurb)}</p>"
        )
        src = _embed(path) if path else None
        if src:
            h.append(f"<img alt='{_esc(title)}' src='{src}'>")
        elif path:
            h.append(
                f"<p class='n'>(figure written as "
                f"<code>{_esc(os.path.splitext(path)[1])}</code>, which "
                f"a browser cannot show — open it from the folder)</p>"
            )
        else:
            h.append("<p class='n'>(figure not found in this run)</p>")
        if path:
            h.append(f"<p class='n'><code>{_esc(_rel(path, dataset_dir))}</code></p>")
        h.append("</div>")

    h.append(
        "<h2>Where everything else is</h2><table>"
        "<tr><th>Folder</th><th>What is in it</th></tr>"
    )
    for folder, what in _FOLDERS:
        h.append(f"<tr><td><code>{folder}</code></td><td>{what}</td></tr>")
    h.append("</table>")
    h.append(
        f"<footer>Ratatosk {run_id or ''} — images are embedded, so this "
        f"file can be moved or emailed on its own. Every number here "
        f"comes from the CSVs in the folders above.</footer>"
    )
    h.append("</div></body></html>")

    # ------------------------------------------------------------ Markdown
    m = [f"# {f['title']}", ""]
    if subtitle:
        m += [f"*{subtitle}*", ""]
    if experiment:
        # THREE COLUMNS. The note is sometimes a unit ("mAh/g"), sometimes
        # a provenance ("measured C/10"), sometimes both with a dash already
        # in it — so no single separator can join it to the value without
        # reading wrongly in one of those cases. It gets its own column.
        m += ["## The experiment", "", "| | | |", "|---|---|---|"]
        # A SEPARATOR. "| Cycles complete | 10 cycle 11 still running |"
        # and "| Rate | 0.1 C measured C/10 |" ran the value straight into
        # its note, so both read as one garbled value.
        m += [
            f"| {_md_cell(k)} | {_md_cell(v)} | {_md_cell(n2 or '')} |"
            for k, v, n2 in experiment
        ]
        m += [""]
    if f.get("key_capacities"):
        _var = _rate_variable(f)
        m += [
            "## Discharge capacity",
            "",
            ("| Cycle | mAh/g |" if _var else "| Cycle | mAh/g | % of first |"),
            ("|---|---|" if _var else "|---|---|---|"),
        ]
        m += [
            (
                f"| {c} | {d:.0f} |"
                if _var
                else f"| {c} | {d:.0f} | "
                f"{('%.0f%%' % pct) if pct is not None else '—'} |"
            )
            for c, d, pct in f["key_capacities"]
        ]
        extra = [(k, v, n2) for k, v, n2 in stats if not k.startswith("Cycle ")]
        if extra:
            m += [""] + [f"**{k}** {v} {n2}".strip() for k, v, n2 in extra]
        _cav = _rate_caveat(f)
        if _cav:
            m += ["", _cav]
        m += [""]
    m += ["## What the run found", ""]
    MARK = {"ok": "", "warn": "⚠ ", "bad": "✘ "}
    for lvl, head, text in findings:
        m += [f"**{MARK.get(lvl, '')}{head}.** {text}", ""]
    if f["peaks"].get("Charge") or f["peaks"].get("Discharge"):
        _kinds = _kind_by_ref(f)
        m += [
            "## Redox features being tracked",
            "",
            "| Step | Reference peak / V | Fitted as |",
            "|---|---|---|",
        ]
        for step in ("Charge", "Discharge"):
            for x in f["peaks"].get(step) or []:
                m.append(
                    f"| {step} | {x:.3f} | "
                    f"{_kinds.get((step, round(float(x), 3)), '—')} |"
                )
        if not _kinds:
            m += [
                "",
                "Nothing here was fitted — these are the detected "
                "reference features only.",
            ]
        m += [""]
    m += ["## Four figures, in this order", ""]
    for i, (title, blurb, path, rel) in enumerate(figs, 1):
        m += [f"### {i}. {title}", "", blurb, ""]
        # ANGLE BRACKETS ROUND THE DESTINATION. Every dataset name in this
        # project contains spaces and parentheses —
        # `JQ_LTO_2.5-1.2V_0.1C_A_08052026 (2)` — and CommonMark cannot parse
        # a bare destination containing either, so EVERY figure in EVERY
        # START_HERE.md rendered as literal text instead of an image. The
        # HTML page embeds its images as data URIs and was unaffected, which
        # is why this survived.
        m += [f"![{title}](<{rel}>)" if rel else "*(figure not found)*", ""]
    m += ["## Where everything else is", "", "| Folder | What is in it |", "|---|---|"]
    m += [f"| `{folder}` | {what} |" for folder, what in _FOLDERS]
    m += [
        "",
        f"---",
        "",
        "Generated by Ratatosk. Every number above comes from the CSVs in "
        "the folders listed.",
    ]
    return "\n".join(h), "\n".join(m), f


_FOLDERS = [
    (
        "1_cycling",
        "capacity, coulombic efficiency, retention, fade, "
        "average voltage, energy efficiency — and the cycle integrity table",
    ),
    ("2_voltage_profiles", "voltage against capacity, all cycles and key cycles"),
    ("3_energy_power", "energy and power density on three bases, and the Ragone plot"),
    (
        "4_dqdv",
        "differential capacity: overlays, waterfalls, heatmaps, and "
        "the processed curves as CSV",
    ),
    (
        "5_peak_fitting",
        "detected peaks, the fits themselves, every fitted "
        "parameter, and the coherence audit",
    ),
    ("6_descriptors", "tracked peaks, polarisation, capacity attribution"),
    ("7_summary", "the per-cycle summary table"),
]


def write_report(dataset_dir, name, **kwargs):
    """
    Write both files at the top of the dataset folder.

    Returns `(html_path, md_path, facts)` — the facts go to
    `write_run_summary`, so the run-level page is assembled from exactly the
    numbers each per-cell page reported rather than recomputed.
    """
    html_doc, md, facts = build_report(name, dataset_dir=dataset_dir, **kwargs)
    os.makedirs(dataset_dir, exist_ok=True)
    ph = os.path.join(dataset_dir, "START_HERE.html")
    pm = os.path.join(dataset_dir, "START_HERE.md")
    with open(ph, "w", encoding="utf-8") as fh:
        fh.write(html_doc)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(md)
    return ph, pm, facts


# =============================================================================
# The run summary — one page for the whole run
# =============================================================================
# `START_HERE` answers "what did this cell do". A run is usually a triplicate,
# and the question a triplicate exists to answer is different: do the cells
# agree, and if not, which one is lying?
#
# 1.8.x answered that with MAD-proximity scoring — pick the cell closest to the
# group median on every metric. That rewards a cell for being average, which is
# exactly wrong when the failure mode is all three cells being wrong in the
# same way, and it says nothing about whether the spread is small enough for
# the median to mean anything. So this reports the SPREAD first, names the
# outlier only when there is one, and declines to nominate a representative
# cell when the cells do not agree.

REPRESENTATIVE_MAX_SPREAD = 0.15  # relative spread above which "typical"
# is not a thing this run has

# --- what a triplicate REPORTS ------------------------------------------
# Published reproducibility practice for replicate cells is mean +/- standard
# deviation across the replicates, with the individual cells shown. Nominating
# one cell as representative is a SELECTION STEP, and selection steps are
# where irreproducibility enters — it is the same move as choosing the
# prettiest diffraction pattern.
#
# So: the headline for a group of replicates is mean +/- SD with n stated, the
# representative cell is nominated for FIGURES ONLY and labelled as such, and
# the representative cell's number is never the reported number. A caption
# reads "cell B shown; n = 3, retention 98.6 +/- 0.4%", which is a stronger
# claim than any single cell can make and is what a reviewer would ask for.
#
# The SD is the sample standard deviation (ddof=1) — an estimate of the
# population spread from a sample, which is what three cells are. numpy's
# default ddof=0 is the population SD and understates a triplicate's spread
# by 18%.
# Mirrors `analyse.INTEGRAL_FIDELITY_FLOOR` and `INTEGRAL_FIDELITY_CEILING`.
# Restated rather than imported so `report` keeps its one-way dependency,
# exactly as DEAD_CELL_RUN is; if one moves, move both. The band is TWO-SIDED
# since 1.9.0.23: a curve carrying more charge than the cell delivered is as
# disqualifying as one carrying less, and only the deficit used to be caught.
INTEGRAL_FIDELITY_FLOOR = 0.80
INTEGRAL_FIDELITY_CEILING = 1.20


def _fidelity_outside_band(fid):
    """The half-cycles whose integral fidelity is outside the band, either
    way, as `{key: value}`."""
    return {
        k: v
        for k, v in (fid or {}).items()
        if np.isfinite(v)
        and not (INTEGRAL_FIDELITY_FLOOR <= v <= INTEGRAL_FIDELITY_CEILING)
    }


REPLICATE_SD_DDOF = 1
# READ, not decorative. This was defined and the guard below used a literal
# 2, so changing the documented knob did nothing.
REPLICATE_MIN_N = 2


def _spread(values):
    """
    Relative spread of a group: (max - min) / median. NaN if not usable.

    Coerced through pandas rather than filtered with `np.isfinite`, because
    these values come from report facts where a missing number is `None`, and
    `np.isfinite(None)` raises rather than returning False.
    """
    v = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().values
    if v.size < 2:
        return np.nan
    med = float(np.median(v))
    if not np.isfinite(med) or med == 0:
        return np.nan
    return float((v.max() - v.min()) / abs(med))


# Two tracked peaks are the same redox feature if their reference voltages sit
# closer than this. Mirrors `analyse.TRACKING_TOLERANCE_MV`, which is the
# pipeline's existing answer to the same question, restated rather than
# imported to keep `report`'s one-way dependency.
FEATURE_GROUP_TOLERANCE_MV = 80.0


def _group_drift(points):
    """Group (reference_voltage, cell_name, drift) into redox features.

    Sorted by voltage and split wherever consecutive peaks are further apart
    than `FEATURE_GROUP_TOLERANCE_MV`. Returns one dict per feature with the
    mean voltage, and mean/sd/n of the drift where **n counts CELLS** — a
    cell contributing two peaks to one feature contributes their median, so
    it is never counted twice and never inflates the spread.
    """
    if not points:
        return []
    pts = sorted(points, key=lambda t: t[0])
    groups, cur = [], [pts[0]]
    for pt in pts[1:]:
        if (pt[0] - cur[-1][0]) * 1000.0 > FEATURE_GROUP_TOLERANCE_MV:
            groups.append(cur)
            cur = [pt]
        else:
            cur.append(pt)
    groups.append(cur)

    out = []
    for grp in groups:
        per_cell = {}
        for _pt in grp:
            per_cell.setdefault(_pt[1], []).append(_pt[2])
        vals = [float(np.median(v)) for v in per_cell.values()]
        mean, sd, n = _mean_sd(vals)
        # A point may carry the coherence verdict as a fourth element. A
        # feature backed only by "questionable" peaks is published — the
        # audit did not reject it — but marked, because "questionable" and
        # "coherent" are not the same evidence and a table that prints them
        # identically is asserting they are.
        _verds = {_pt[3] for _pt in grp if len(_pt) > 3 and _pt[3]}
        # NO VERDICT IS NOT A PASS. A peak with no row in the coherence table
        # — the audit never reached it, or peak fitting ran without the audit
        # — was let through the filter above and then printed in the same
        # typeface as an audited one, under a caption asserting that only
        # trend-worthy features are in the table. Unaudited is a third state
        # and gets its own mark.
        _unaudited = all(len(_pt) < 4 or not _pt[3] for _pt in grp)
        out.append(
            dict(
                voltage=float(np.mean([p[0] for p in grp])),
                mean=mean,
                sd=sd,
                n=n,
                n_peaks=len(grp),
                unaudited=bool(_unaudited),
                questionable_only=bool(_verds) and _verds <= {"questionable"},
            )
        )
    return out


def _mean_sd(values, *, ddof=REPLICATE_SD_DDOF):
    """
    (mean, sd, n) across replicates. See REPLICATE_SD_DDOF.

    `sd` is NaN for a single cell — one measurement has no spread, and
    reporting 0 would claim a precision nothing established.
    """
    v = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().values
    if v.size == 0:
        return (np.nan, np.nan, 0)
    if v.size < 2:
        return (float(v[0]), np.nan, 1)
    return (float(v.mean()), float(v.std(ddof=ddof)), int(v.size))


def _pm(mean, sd, n, unit="", dp=1, show_n=False):
    """`98.6 +/- 0.4% (n = 3)`, or the bare value when there is no spread.

    `show_n` keeps the count on a single measurement too. A table whose caption
    says "n is the number of CELLS in which that feature was tracked" must
    print n on every row, or a one-cell row is indistinguishable from a
    two-cell row whose replicates happened to agree exactly — which is the one
    thing that caption exists to prevent.
    """
    if not np.isfinite(mean):
        return "—"
    if not np.isfinite(sd) or n < 2:
        return (
            f"{mean:.{dp}f}{unit} (n = {n})" if show_n and n else f"{mean:.{dp}f}{unit}"
        )
    return f"{mean:.{dp}f} ± {sd:.{dp}f}{unit} (n = {n})"


def _pct(x):
    """A relative spread as a percentage that never rounds away to nothing.

    `{:.0%}` turned a 0.34% spread into "0%", which reads as "the cells are
    identical" one line below a table quoting +/- 0.2. A spread small enough
    to round to zero is small, not absent, and the page should say so.
    """
    if not np.isfinite(x):
        return "—"
    if x <= 0:
        return "0%"
    if x < 0.005:
        return "<1%"
    if x < 0.095:
        return f"{100 * x:.1f}%"
    return f"{100 * x:.0f}%"


def _last_cell(last, last_cycle):
    """Last working capacity, with its cycle number when there is one."""
    if not np.isfinite(last):
        return "—"
    if not np.isfinite(last_cycle):
        return f"{last:.0f}"
    return f"{last:.0f} (c{int(last_cycle)})"


def build_run_summary(facts_by_name, *, run_id="", n_files=0):
    """
    One page for the whole run: the cells side by side, and whether they agree.

    `facts_by_name` is `{name: dict}` as `build_report` assembles — pass the
    `facts` each per-dataset report returned.
    """
    names = list(facts_by_name)
    rows = []
    for n in names:
        f = facts_by_name[n]
        p = f.get("params", {})
        dl = f.get("first_last_discharge") or (np.nan, np.nan)
        anom = f.get("anomalous")
        over = f.get("over_theoretical")
        rows.append(
            dict(
                name=n,
                composition=str(p.get("composition", n)),
                label=(
                    f"{p.get('composition', n)}"
                    + (f" {p['cell_id']}" if p.get("cell_id") else "")
                ),
                mass=p.get("active_material_mass_mg", np.nan),
                cycles=f.get("n_cycles", np.nan),
                first=dl[0],
                last=dl[1],
                last_cycle=f.get("last_good_cycle", np.nan),
                retention=f.get("retention", np.nan),
                unusable="; ".join(f.get("unusable") or []) or None,
                # The arithmetic that was barred from being called retention on a
                # variable-rate run. Kept under its own name so it is not LOST —
                # which it was, because nothing read it.
                retention_uncorrected=f.get("retention_uncorrected", np.nan),
                rate_is_variable=bool(f.get("rate_is_variable")),
                rate_label=(
                    ((f.get("params") or {}).get("rate_protocol") or {}).get("label")
                ),
                ce=f.get("median_ce", np.nan),
                died=f.get("fit_limit") or f.get("death_cycle"),
                n_anom=(len(anom) if anom is not None else 0),
                n_over=(len(over) if over is not None else 0),
                withheld=f.get("withheld", (0, 0))[0],
                unattributed=f.get("unattributed"),
                unattributed_withheld=f.get("unattributed_withheld", 0),
                no_baseline=f.get("no_baseline", False),
                mechanism_own=str(
                    (f.get("resolvability") or {}).get("mechanism_own") or ""
                )
                or None,
                # WHICH MODEL EACH CELL WAS FITTED WITH. The run page exists to
                # say whether the cells agree, and two replicates classified
                # differently — one `multi_transition`, one `mixed` — are not
                # being compared like with like. That is a difference this page
                # must show, not one a reader should have to open three folders
                # to find.
                mechanism=str((f.get("resolvability") or {}).get("mechanism") or ""),
                # The WORST step fidelity for this cell, so the run summary can
                # say which ground the withholding rested on and can carry the
                # caveat itself. A reader who opens only this page must not be
                # left thinking the areas are capacities.
                # The WORST is the one FURTHEST FROM 1, not the smallest. The
                # band is two-sided (0.80-1.20); `min` recorded 0.95 for a cell
                # whose charge step carried 1.45, so the run page saw a healthy
                # number and the excess never appeared on it.
                fidelity=max(
                    [
                        v
                        for v in (f.get("integral_fidelity") or {}).values()
                        if np.isfinite(v)
                    ],
                    key=lambda v: abs(v - 1.0),
                    default=np.nan,
                ),
                partial=bool(f.get("partial_final")),
            )
        )
    _COLS = [
        "name",
        "label",
        "composition",
        "mass",
        "cycles",
        "first",
        "last",
        "last_cycle",
        "retention",
        "ce",
        "died",
        "n_anom",
        "n_over",
        "withheld",
        "fidelity",
        "partial",
        # 1.9.0.56: the charge the model could not name, and whether
        # that alone withheld a share. Added here as well as to `rows`
        # for the reason the note below gives.
        "unattributed",
        "unattributed_withheld",
        "no_baseline",
        "mechanism_own",
        "rate_is_variable",
        "rate_label",
        "unusable",
        "retention_uncorrected",
        # ADDING A FIELD TO `rows` IS NOT ENOUGH. The list below is
        # explicit, so a key that is not in it is silently dropped and
        # every `r.<name>` after it raises AttributeError. That is what
        # took RUN_SUMMARY off the 1.9.0.47 NMC run: the per-cell pages
        # were written, the run page was not, and nothing said why.
        "mechanism",
    ]
    # An explicit column list so a run where every dataset failed produces a
    # summary saying so, rather than a KeyError on the first coercion.
    T = pd.DataFrame(rows, columns=_COLS)

    # Every one of these is rendered with np.isfinite, and every one of them
    # can arrive as None or a string from a report that had nothing to put
    # there. Coerce once, at the boundary, rather than guarding each use:
    # the LTO run died on `int(r.cycles)` because one dataset carried None.
    for _c in (
        "mass",
        "cycles",
        "first",
        "last",
        "last_cycle",
        "retention",
        "ce",
        "died",
        "n_anom",
        "n_over",
        "withheld",
        "fidelity",
    ):
        T[_c] = pd.to_numeric(T[_c], errors="coerce")
    for _c in ("n_anom", "n_over", "withheld", "unattributed_withheld"):
        T[_c] = pd.to_numeric(T[_c], errors="coerce").fillna(0)
    T["unattributed"] = pd.to_numeric(T["unattributed"], errors="coerce")
    T["no_baseline"] = T["no_baseline"].fillna(False).astype(bool)
    T["rate_is_variable"] = T["rate_is_variable"].fillna(False).astype(bool)

    # THE TEXT COLUMNS NEED COERCING AT THE BOUNDARY TOO, and for a reason
    # the numeric ones do not have. pandas 3 infers a StringDtype whose
    # missing value is `float('nan')`, not None. nan is TRUTHY, so it passes
    # the `if v` guard on `mechanism_own` below and then raises on
    # `v.replace`, taking RUN_SUMMARY.md off the run with an AttributeError
    # after every per-cell page has already been written — the same silent
    # shape as the 1.9.0.47 missing-column failure two notes up.
    #
    # The column is only MIXED, and therefore only inferred as StringDtype,
    # when some cells went through reconciliation and others did not:
    # `reconcile_mechanisms` setdefaults `mechanism_own` on EVERY cell of a
    # group it changed, so a lone triplicate is all-string or all-None and
    # comes back as object either way. Two materials in one run, or one cell
    # left out of a reconciled group, is the mix that breaks it — which is a
    # normal thing to ask for, three datasets at a time out of a folder of
    # eight.
    #
    # On pandas 2 the column is object and None survives, which is why this
    # has never shown on Linux. Nik runs pandas 3.0.5.
    # `dtype=object` is not decoration: assigning a plain list back into a
    # StringDtype column keeps the dtype and turns the Nones straight back
    # into nan, which is the bug this loop exists to remove.
    for _c in ("mechanism_own", "mechanism", "composition", "rate_label", "unusable"):
        T[_c] = pd.Series(
            [
                None
                if (v is None or (isinstance(v, float) and not np.isfinite(v)))
                else str(v)
                for v in T[_c]
            ],
            index=T.index,
            dtype=object,
        )

    # --- do they agree? ---------------------------------------------------
    verdicts = []
    # Group on the composition ITSELF. Reverse-engineering it from the
    # label by stripping the last word merged unrelated compositions whose
    # names differ only in the final token, and mangled any composition
    # recorded without a cell id.
    # A DATASET THAT IS NOT A MEASUREMENT DOES NOT GET A VOTE. Averaging a
    # two-cycle export at 41% coulombic efficiency into a triplicate mean does
    # not make the mean more robust; it makes it wrong, and hides the failure
    # inside a standard deviation. Excluded cells keep their own page and are
    # named on this one.
    _unusable_cells = [r.label for r in T.itertuples() if getattr(r, "unusable", None)]
    T_ok = T[[not bool(getattr(r, "unusable", None)) for r in T.itertuples()]]
    grouped = T_ok.groupby("composition")
    for comp, g in grouped:
        if len(g) < REPLICATE_MIN_N:
            continue
        s_ret = _spread(g["retention"])
        s_first = _spread(g["first"])
        entry = dict(
            composition=comp, n=len(g), spread_retention=s_ret, spread_first=s_first
        )
        # THE HEADLINE. Computed for every group whatever the spread, because
        # a wide spread is a reason to report the SD loudly, not a reason to
        # stop reporting it.
        for _k, _col, _dp in (
            ("retention", "retention", 1),
            ("first", "first", 1),
            ("last", "last", 1),
            ("ce", "ce", 2),
        ):
            entry[f"{_k}_mean"], entry[f"{_k}_sd"], entry[f"{_k}_n"] = _mean_sd(g[_col])
        # PEAK DRIFT, per step, across the replicates. On a dataset where the
        # areas are withheld this is the only quantity the run actually
        # measured, and it had appeared nowhere on this page.
        # ONE ROW PER REDOX FEATURE, not one row per step. Until 1.9.0.25
        # every primary peak from every cell went into a single list, so on
        # LTO — one charge peak per cell — n was the cell count and the
        # number read as a replicate statistic. On a layered oxide with
        # three charge features it silently became n = 9: the mean of three
        # DIFFERENT redox processes across three cells, whose spread is
        # dominated by the differences between the processes rather than
        # between the cells. The NNM run reported 0.80 +/- 2.31 mV/cycle
        # that way, which is not a quantity anyone can use.
        #
        # Peaks are grouped by reference voltage — the pipeline's own notion
        # of "the same feature" is TRACKING_TOLERANCE_MV, so the same number
        # separates features here — and each group reports mean +/- SD with
        # **n = the number of CELLS** contributing. A cell offering two peaks
        # to one group contributes their median, so no cell is counted twice.
        #
        # AND THE COHERENCE AUDIT DECIDES WHICH PEAKS GET IN. A drift rate is
        # only "the quantity that survives where the areas do not" if the peak
        # it belongs to moves like a redox feature. `coherence_audit` already
        # judges that, peak by peak; before 1.9.0.34 its verdict never reached
        # this page, so every primary peak was published here whatever the
        # audit had said about it. On the NMC111 pair that produced a table of
        # five drift rates on a run where the audit found ZERO coherent peaks
        # in either cell — including one rate read from four consecutive
        # pairs and marked "not assessable" two cells earlier.
        _n_excluded = 0
        for _step in ("Charge", "Discharge"):
            _pts = []
            for _n in g["name"]:
                _verd = {
                    (c["step"], round(c["reference_voltage"], 3)): c["verdict"]
                    for c in (facts_by_name[_n].get("coherence") or [])
                }
                for _t in facts_by_name[_n].get("tracked") or []:
                    if not (
                        _t.get("step") == _step
                        and not _t.get("is_shoulder")
                        and np.isfinite(_t.get("voltage_drift_mV_per_cycle", np.nan))
                        and np.isfinite(_t.get("reference_voltage", np.nan))
                    ):
                        continue
                    _rv = float(_t["reference_voltage"])
                    # No audit at all (peak fitting off, or an older facts
                    # dict) leaves the old behaviour: publish it. A verdict
                    # that exists and is not trend-worthy excludes the point.
                    _v = _verd.get((_step, round(_rv, 3)))
                    if _v is not None and _v not in ("coherent", "questionable"):
                        _n_excluded += 1
                        continue
                    _pts.append((_rv, _n, float(_t["voltage_drift_mV_per_cycle"]), _v))
            entry[f"drift_{_step.lower()}_features"] = _group_drift(_pts)
        entry["drift_excluded_incoherent"] = _n_excluded
        # The odd one out, if there is one: the cell furthest from the median
        # on retention, but only worth naming when the spread is wide enough
        # that "furthest" means something.
        # "Furthest from the median" needs at least three cells that HAVE a
        # retention figure. With two, the median sits between them and both
        # are equally far — on a run where one cell died and one had no
        # figure at all, that named the healthy cell as the outlier.
        # DO THE CELLS' NUMBERS COME FROM THE SAME CYCLE? "Last working" and
        # the retention beside it are read at each cell's own last good
        # cycle, so a group containing a cell that died early is being
        # averaged across DIFFERENT endpoints. On the NNM triplicate that
        # meant 46% at cycle 100, 58% at cycle 67 and 44% at cycle 100
        # pooled into "49.5 +/- 7.8 (n = 3)", which reads as retention after
        # 100 cycles for three cells and is not that — and the spread is
        # inflated by exactly the mismatch. The per-cell table above already
        # prints "58 (c67)"; the headline threw the qualifier away.
        _lc = pd.to_numeric(g["last_cycle"], errors="coerce").dropna()
        entry["last_cycle_min"] = float(_lc.min()) if len(_lc) else np.nan
        entry["last_cycle_max"] = float(_lc.max()) if len(_lc) else np.nan
        entry["last_cycle_mixed"] = bool(len(_lc) > 1 and _lc.nunique() > 1)
        gf = g[g["retention"].notna()]
        entry["n_compared"] = len(gf)
        if np.isfinite(s_ret) and len(gf) >= 3:
            dist = (gf["retention"] - gf["retention"].median()).abs()
            if s_ret > REPRESENTATIVE_MAX_SPREAD:
                entry["outlier"] = gf.loc[dist.idxmax(), "label"]
                entry["representative"] = None
            else:
                entry["outlier"] = None
                entry["representative"] = gf.loc[dist.idxmin(), "label"]
        verdicts.append(entry)

    # --- markdown ---------------------------------------------------------
    m = ["# Run summary", ""]
    m += [
        f"*{run_id}* — {len(names)} dataset(s)"
        + (f" of {n_files} in the folder" if n_files else ""),
        "",
    ]

    # THE MECHANISM ROW, above the table rather than in it: it is one word
    # per cell and it decides how everything below was measured, so it reads
    # as a premise rather than as another column.
    # KEYED ON THE DATASET, NOT ITS LABEL. `label` is composition + cell id
    # and the cell id is optional, so three replicates with no cell id shared
    # one label, `len(_mechs)` was 1, and the page told a three-row table
    # "There is one dataset in this run, so there was nothing to reconcile it
    # against."
    _label_of = {r.name: r.label for r in T.itertuples()}
    _mechs = {r.name: r.mechanism for r in T.itertuples() if r.mechanism}
    # What each cell's OWN reference cycle said, before reconciliation.
    _own = (
        {
            r.name: getattr(r, "mechanism_own", None)
            for r in T.itertuples()
            if r.mechanism
        }
        if "mechanism_own" in T
        else {}
    )

    # WHICH CELLS COULD BE RECONCILED AT ALL, and if not, why not.
    # `quality.reconcile_mechanisms` groups by composition with the dataset's
    # own name as the fallback, and skips any group of fewer than two. So a
    # cell is left with its own reference cycle's call whenever its
    # composition is blank (the fallback name is unique) OR it is the only
    # cell of its composition in the run — and until 1.9.0.62 the run page
    # said neither.
    #
    # Measured, NNM 1.9.0.61: cell A's own call is `multi_transition`, whose
    # model permits no band at all. Reconciled with cells B and C it is
    # fitted `mixed` and gets 215 bands; in a group of one it gets NONE and
    # R2 falls 0.9930 -> 0.9314 over 160 half-cycles. That happened twice on
    # the same day — once from a blank composition box, once from re-running
    # the cell on its own to check the first — and both runs read as normal.
    _group_of = {}
    for r in T.itertuples():
        if not r.mechanism:
            continue
        _c = str(getattr(r, "composition", "") or "").strip()
        _group_of[r.name] = (_c or f"\x00{r.name}", bool(_c))
    _sizes = {}
    for _g, _ in _group_of.values():
        _sizes[_g] = _sizes.get(_g, 0) + 1
    _alone = {
        lab: named for lab, (g, named) in _group_of.items() if _sizes.get(g, 0) < 2
    }

    def _display(label, name):
        """A cell with no composition has a label of just its cell id."""
        return str(label).strip() or str(name)

    _CONSEQUENCE = (
        "The same cell analysed alongside its replicates can be reconciled "
        "to a more permissive model and fitted differently, so a result from "
        "a group of one is not interchangeable with one from a group run."
    )

    def _alone_sentence(*, why=True):
        """Why a cell kept its own call, named cell by cell."""
        if not _alone:
            return []
        _lead = (
            "A mechanism is chosen from ONE reference cycle of ONE cell, "
            "and it is reconciled across a material's replicates because "
            "a material either delivers charge across a composition "
            "window or it does not. "
        )
        if not why:
            return [
                _lead + "There was no group to do that with here, so "
                "this cell kept whatever its own reference cycle "
                "said and was fitted with that model. " + _CONSEQUENCE,
                "",
            ]
        _bits = []
        for _nm0, named in _alone.items():
            _row = next((r for r in T.itertuples() if r.name == _nm0), None)
            _shown = _display(_label_of.get(_nm0, _nm0), _nm0)
            if not named:
                _bits.append(
                    f"**{_shown}** has no composition recorded, so "
                    f"it could not be grouped with anything"
                )
            else:
                _comp = str(getattr(_row, "composition", "") or "its material")
                _bits.append(f"**{_shown}** is the only {_comp} cell in this run")
        _tail = (
            " — so each kept whatever its own reference cycle said, and "
            "was fitted with that model. "
            if len(_bits) > 1
            else " — so it kept whatever its own reference cycle said, and "
            "was fitted with that model. "
        )
        return [_lead + "; ".join(_bits) + _tail + _CONSEQUENCE, ""]

    if _mechs:
        _uniq = sorted(set(_mechs.values()))
        m += ["## How these cells were modelled", ""]
        if len(_mechs) == 1:
            # ONE CELL IS NOT AN AGREEMENT. "All 1 cells classified X, so the
            # same model was fitted to each and the numbers below are
            # comparable" reads as reassurance about a comparison that does
            # not exist, and hides that this cell was never reconciled.
            _nm, _mv = next(iter(_mechs.items()))
            m += [
                f"**{_display(_label_of.get(_nm, _nm), _nm)}** was classified "
                f"**{_mv.replace('_', ' ')}** and fitted with that model. "
                f"There is one dataset in this run, so there was nothing "
                f"to reconcile it against.",
                "",
            ]
            m += _alone_sentence(why=False)
        elif len(_uniq) == 1:
            m += [
                f"All {len(_mechs)} cells classified "
                f"**{_uniq[0].replace('_', ' ')}**, so the same model was "
                f"fitted to each and the numbers below are comparable.",
                "",
            ]
            # ...but say if that agreement was ARRIVED AT rather than found.
            _rec = {k: v for k, v in (_own or {}).items() if v and v != _mechs.get(k)}
            if _rec:
                m += [
                    "Not every cell's own reference cycle said so. "
                    + "; ".join(
                        f"**{_display(_label_of.get(k, k), k)}** read "
                        f"{v.replace('_', ' ')}"
                        for k, v in _rec.items()
                    )
                    + (
                        " and were reconciled to "
                        if len(_rec) > 1
                        else " and was reconciled to "
                    )
                    + f"**{_uniq[0].replace('_', ' ')}**, the most "
                    f"permissive call in the group. A material either "
                    f"delivers charge across a composition window or it "
                    f"does not, so a replicate that did not resolve one "
                    f"is a detection miss rather than evidence of "
                    f"absence — and replicates fitted with different "
                    f"models cannot be pooled, which is what replicates "
                    f"are for.",
                    "",
                ]
            # ...and say which cells had no group to be reconciled with,
            # AFTER the reconciliation note, because it is the exception
            # to it.
            m += _alone_sentence()
        else:
            m += [
                "**The cells were not all fitted with the same model.**",
                "",
                "| Cell | Mechanism |",
                "|---|---|",
            ]
            m += [
                f"| {_display(_label_of.get(k, k), k)} | {v.replace('_', ' ')} |"
                for k, v in _mechs.items()
            ]
            m += [
                "",
                "A mechanism is chosen per dataset from its own "
                "reference cycle, so replicates can differ — but two "
                "cells fitted with different models are not being "
                "compared like with like, and areas in particular "
                "should not be pooled across them.",
                "",
            ]
            m += _alone_sentence()
    if _unusable_cells:
        m += ["## Not every dataset here is a measurement", ""]
        for r in T.itertuples():
            if getattr(r, "unusable", None):
                m += [f"- **{r.label}** — {r.unusable}"]
        m += [
            "",
            "These are excluded from the replicate mean and standard "
            "deviation below, and from the comparison across cells. "
            "They keep their own page, which is where to look for what "
            "went wrong.",
            "",
        ]
    _var = [r.label for r in T.itertuples() if getattr(r, "rate_is_variable", False)]
    if _var:
        m += [
            "## These cells were cycled at more than one rate",
            "",
            "**Retention is withheld for "
            + ("every cell" if len(_var) == len(T) else ", ".join(_var))
            + ".** Capacity against cycle number in a rate-capability run "
            "is mostly the rate schedule: it falls because the current "
            "rose and returns when the current falls again, so a "
            "retention percentage across it reads as fading and "
            "recovery. Each cell's rate-capability summary compares "
            "every rate against the reference rate and reports the "
            "recovery to the starting rate, the fade measured during "
            "the ramp, and what that leaves for the ramp itself.",
            "",
        ]
    m += [
        "## The cells side by side",
        "",
        "| Cell | Mass / mg | Cycles | 1st discharge | Last working | "
        "Retention | Median CE | Flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in T.itertuples():
        flags = []
        # `died` goes through the same DataFrame as everything else, so a
        # None arrives as NaN, which is truthy.
        if np.isfinite(r.died):
            flags.append(f"died c{int(r.died)}")
        if r.n_over:
            flags.append(f"{r.n_over} over Qtheo")
        if r.n_anom:
            flags.append(f"{r.n_anom} unaccounted")
        if r.withheld:
            flags.append(f"{r.withheld} withheld")
        if r.partial:
            flags.append("still cycling")
        fmt = lambda v, d=0: "—" if not np.isfinite(v) else f"{v:.{d}f}"
        m.append(
            f"| {r.label} | {fmt(r.mass, 2)} | "
            f"{'—' if not np.isfinite(r.cycles) else int(r.cycles)} | "
            f"{fmt(r.first)} | "
            f"{_last_cell(r.last, r.last_cycle)} | "
            f"{fmt(r.retention) + '%' if np.isfinite(r.retention) else '—'} | "
            f"{fmt(r.ce, 1) + '%' if np.isfinite(r.ce) else '—'} | "
            f"{', '.join(flags) if flags else 'none'} |"
        )
    m += [""]

    # The replicate result as a table anyone can read back — the headline
    # number for the group, machine-readable, beside the per-cell CSV.
    R = (
        pd.DataFrame(verdicts)
        if verdicts
        else pd.DataFrame(
            columns=[
                "composition",
                "n",
                "retention_mean",
                "retention_sd",
                "retention_n",
            ]
        )
    )

    if verdicts:
        # --- THE HEADLINE: mean +/- SD across replicates ------------------
        m += [
            "## The replicate result",
            "",
            "The number to quote for a group of cells is the mean and "
            "standard deviation across them, with n stated. A single cell "
            "nominated as representative is a selection step, and the "
            "figures below say which cell is *shown*, never which number "
            "is *reported*.",
            "",
            "| Material | n | 1st discharge / mAh g⁻¹ | Last working / "
            "mAh g⁻¹ | Retention / % | Median CE / % |",
            "|---|---|---|---|---|---|",
        ]
        for v in verdicts:
            m.append(
                f"| {v['composition']} | {v['retention_n']} | "
                + _pm(v["first_mean"], v["first_sd"], v["first_n"])
                + " | "
                + _pm(v["last_mean"], v["last_sd"], v["last_n"])
                + " | "
                + _pm(v["retention_mean"], v["retention_sd"], v["retention_n"])
                + " | "
                + _pm(v["ce_mean"], v["ce_sd"], v["ce_n"], dp=2)
                + " |"
            )
        m += [
            "",
            "*Standard deviation across cells, ddof = 1. A single cell "
            "shows its value with no ± : one measurement has no spread, and "
            "quoting 0 would claim a precision nothing established.*",
            "",
        ]
        # Said plainly, next to the number it qualifies.
        for v in verdicts:
            if v.get("last_cycle_mixed"):
                m += [
                    f"> **These cells did not all reach the same cycle.** "
                    f"For {v['composition']} the last working capacity and "
                    f"the retention beside it are read between cycle "
                    f"{v['last_cycle_min']:.0f} and cycle "
                    f"{v['last_cycle_max']:.0f} depending on the cell, so "
                    f"the mean above is **not** retention at a common "
                    f"cycle and its spread carries that mismatch as well "
                    f"as any real difference between the cells. The "
                    f"per-cell table shows each cell's own cycle. Quote "
                    f"these cells separately, or re-run capped at cycle "
                    f"{v['last_cycle_min']:.0f} so every cell is compared "
                    f"over the same life.",
                    "",
                ]

        # Peak drift, where there is any. This is the quantity that survives
        # a low integral fidelity, so on a two-phase material it is the
        # replicate result and belongs beside the capacities, not buried in a
        # per-cell page.
        _has_drift = [
            v
            for v in verdicts
            if v.get("drift_charge_features") or v.get("drift_discharge_features")
        ]
        _no_drift = [
            v
            for v in verdicts
            if v not in _has_drift and v.get("drift_excluded_incoherent")
        ]
        for v in _no_drift:
            m += [
                "### How fast the redox features move",
                "",
                f"**Not reported for {v['composition']}.** Every primary "
                f"peak that could have gone in this table "
                f"({int(v['drift_excluded_incoherent'])} across "
                f"{int(v['n'])} cell(s)) was excluded by the coherence "
                f"audit: none of them moved like a redox feature. A drift "
                f"rate read from a peak the fit could not hold on to is a "
                f"property of the fit. The per-cell coherence tables say "
                f"which peaks and why.",
                "",
            ]
        if _has_drift:
            m += [
                "### How fast the redox features move",
                "",
                "| Material | Step | Feature / V | Drift / mV cycle⁻¹ |",
                "|---|---|---|---|",
            ]
            for v in _has_drift:
                for _step in ("Charge", "Discharge"):
                    for _f in v.get(f"drift_{_step.lower()}_features", []):
                        if not np.isfinite(_f["mean"]):
                            continue
                        _mark = (
                            " ‡"
                            if _f.get("unaudited")
                            else " †"
                            if _f.get("questionable_only")
                            else ""
                        )
                        m.append(
                            f"| {v['composition']} | {_step} | "
                            f"{_f['voltage']:.3f} | "
                            + _pm(_f["mean"], _f["sd"], _f["n"], dp=2, show_n=True)
                            + _mark
                            + " |"
                        )
            m += [
                "",
                "*One row per redox feature, primary peaks only, "
                "measured between medians of three cycles at each end from "
                "the reference cycle onward. **n is the number of CELLS** "
                "in which that feature was tracked — averaging different "
                "features together would give a spread that measures the "
                "difference between the processes rather than between the "
                "cells. Peak positions do not depend on the dQ/dV integral, "
                "so this is the quantity that survives where the areas do "
                "not — but only for a peak the coherence audit found "
                "trend-worthy; peaks it called NOT trend-worthy or not "
                "assessable are left out.*",
                "",
            ]
            if any(
                _f.get("questionable_only")
                for v in _has_drift
                for _step in ("charge", "discharge")
                for _f in v.get(f"drift_{_step}_features", [])
            ):
                m += [
                    "*† every peak behind this row was graded "
                    "*questionable* by the coherence audit, not *coherent*: "
                    "its median drift cleared the implausible threshold but "
                    "not the clean one. The rate is reported; the evidence "
                    "for it is weaker than for an unmarked row.*",
                    "",
                ]
            if any(
                _f.get("unaudited")
                for v in _has_drift
                for _step in ("charge", "discharge")
                for _f in v.get(f"drift_{_step}_features", [])
            ):
                m += [
                    "*‡ no peak behind this row has a coherence verdict at "
                    "all — the audit did not reach it, or peak fitting ran "
                    "without it. The rate is what the tracking measured; "
                    "nothing has tested whether the feature moves like a "
                    "redox process or like a fitting artefact.*",
                    "",
                ]
            _exc = sum(int(v.get("drift_excluded_incoherent") or 0) for v in _has_drift)
            if _exc:
                m += [
                    f"*{_exc} further primary peak(s) were excluded by the "
                    f"coherence audit and are not in the table above.*",
                    "",
                ]

        m += ["## Do the cells agree?", ""]
        for v in verdicts:
            s = v["spread_retention"]
            # THE VERDICT INHERITS THE CAVEAT. `spread_retention` is computed
            # from the same per-cell retention figures the note above warns
            # are read at different cycles, and this is the most
            # conclusion-shaped sentence in the report — so it is the one
            # place the qualifier most needs to reach.
            #
            # Measured on NNM cells B and C, same cells, same data, only the
            # cycle cap changed:
            #
            #   capped at 100   B 58% (c67), C 44% (c100)  spread 29%
            #                   -> "the 2 cells do NOT agree"
            #   capped at  80   B 58% (c67), C 53% (c80)   spread 9.0%
            #                   -> "2 cells agree"
            #
            # Opposite conclusions from where the operator stopped, and in
            # BOTH the two numbers come from different cycles. Neither
            # verdict is wrong about the arithmetic; both are silent about
            # what they compared.
            _mixed = bool(v.get("last_cycle_mixed"))
            _mixed_note = ""
            if _mixed:
                _mixed_note = (
                    f" **Read this with the endpoint warning above:** these "
                    f"retentions come from different cycles — between "
                    f"{v['last_cycle_min']:.0f} and "
                    f"{v['last_cycle_max']:.0f} — so this verdict is partly "
                    f"about where each cell stopped, not only about how they "
                    f"differ. Cap the run at cycle "
                    f"{v['last_cycle_min']:.0f} to settle it."
                )
            if not np.isfinite(s):
                m += [
                    f"**{v['composition']}** — {v['n']} cell(s), of which "
                    f"{v.get('n_compared', 0)} have a retention figure; "
                    f"nothing to compare.",
                    "",
                ]
                continue
            if s <= REPRESENTATIVE_MAX_SPREAD:
                _sf = v["spread_first"]
                line = (
                    f"**{v['composition']}** — {v['n']} cells agree. "
                    f"Retention spreads {_pct(s)} of the median"
                    + (
                        f", and first discharge {_pct(_sf)}."
                        if np.isfinite(_sf)
                        else "."
                    )
                )
                if v.get("representative"):
                    line += (
                        f" For a FIGURE, show **{v['representative']}** "
                        f"— it sits closest to the group — and caption "
                        f'it as such: "{v["representative"]} shown; '
                        f"n = {v['retention_n']}, retention "
                        + _pm(
                            v["retention_mean"], v["retention_sd"], v["retention_n"]
                        ).replace(f" (n = {v['retention_n']})", "")
                        + '%". The reported number stays the mean '
                        "above; the representative cell is a choice of "
                        "illustration, not of result."
                    )
                m += [line + _mixed_note, ""]
            else:
                line = (
                    f"**{v['composition']}** — the {v['n']} cells do NOT "
                    f"agree. Retention spreads {_pct(s)} of the median"
                )
                if v.get("outlier"):
                    line += f", with **{v['outlier']}** furthest out"
                elif v.get("n_compared", 0) < 3:
                    line += (
                        f" across the {v['n_compared']} cell(s) that have "
                        f"one; too few to say which is the odd one"
                    )
                line += (
                    ". No representative cell is nominated: picking the "
                    "one nearest the median would be choosing a number, "
                    "not measuring one. Find out why they differ first."
                )
                m += [line + _mixed_note, ""]

    # --- what the run withheld or flagged ---------------------------------
    tot_over = int(T["n_over"].sum())
    tot_anom = int(T["n_anom"].sum())
    tot_with = int(T["withheld"].sum())
    # A cell whose curve carries the wrong amount of charge has something to
    # declare even when nothing was withheld. `capacity_attribution` withheld
    # only below the floor until 1.9.0.36, so a run with fidelity 1.45 had
    # `tot_with == 0` and this page printed "Nothing was withheld. Every
    # half-cycle's charge is accounted for" directly opposite a per-cell page
    # saying the areas are not capacities.
    _n_outband = int(
        (
            (T["fidelity"] < INTEGRAL_FIDELITY_FLOOR)
            | (T["fidelity"] > INTEGRAL_FIDELITY_CEILING)
        ).sum()
    )
    _un_col = T["unattributed"] if "unattributed" in T else None
    _un_vals = (
        [x for x in _un_col.tolist() if x is not None and np.isfinite(x)]
        if _un_col is not None
        else []
    )
    _un_med = float(np.median(_un_vals)) if _un_vals else None
    _un_any = bool(
        int(T["unattributed_withheld"].sum()) if "unattributed_withheld" in T else 0
    )
    _no_baseline = bool(
        T["no_baseline"].all() if "no_baseline" in T and len(T) else False
    )
    m += ["## What the run would not tell you", ""]
    if not (tot_over or tot_anom or tot_with or _n_outband):
        m += [
            "Nothing was withheld. Every half-cycle's charge is accounted "
            "for"
            + (
                ", and every capacity share is a share of named "
                "components rather than of a free background."
                if _no_baseline
                else ", and every capacity share survived a change of baseline degree."
            ),
            "",
        ]
    else:
        if tot_over:
            m += [
                f"- **{tot_over} half-cycle(s) passed more charge than the "
                f"material can hold.** Over theoretical is not a phase "
                f"transition; it is charge going into something else.",
                "",
            ]
        if tot_anom:
            m += [
                f"- **{tot_anom} half-cycle(s) carry charge no redox process "
                f"accounts for** — at full current, away from the voltage "
                f"limit, where |dQ/dV| is small.",
                "",
            ]
        if tot_with:
            # WHICH GROUND. Closure (the peak/background split is undetermined)
            # and integral fidelity (the curve does not account for the cell's
            # charge) are different failures with different remedies, and this
            # page said "baseline degree" for both — sending a reader to look
            # at the fit when the limit is the instrument's voltage
            # resolution.
            # Two-sided, like `_fidelity_outside_band` above and like every
            # other reading of this number. Testing only the floor meant an
            # over-ceiling cell was never named here.
            _lowfid = T[
                (T["fidelity"] < INTEGRAL_FIDELITY_FLOOR)
                | (T["fidelity"] > INTEGRAL_FIDELITY_CEILING)
            ]
            if len(_lowfid):
                _lo = 100 * float(_lowfid["fidelity"].min())
                _hi = 100 * float(_lowfid["fidelity"].max())
                _rng = (
                    f"{_lo:.0f}%" if abs(_hi - _lo) < 0.5 else f"{_lo:.0f}-{_hi:.0f}%"
                )
                m += [
                    f"- **{tot_with} half-cycle(s) had their capacity "
                    f"attribution withheld — and on "
                    f"{len(_lowfid)} of {len(T)} cell(s) the reason is not "
                    f"the fit.** The dQ/dV curve "
                    + (
                        "accounts for only "
                        if _hi <= 100.0
                        else "carries "
                        if _lo >= 100.0
                        else "accounts for "
                    )
                    + f"{_rng} "
                    f"of the capacity those cells delivered — per cell and "
                    f"per step; where a cell's two steps differ, its own "
                    f"page gives both. On a flat "
                    f"two-phase plateau the voltage change between records "
                    f"falls below the instrument's resolution, so charge "
                    f"delivered there cannot appear in a dV integral and no "
                    f"model recovers it. **A fitted peak area is not a "
                    f"capacity on these cells**, and neither is its "
                    f"retention. Peak POSITIONS and their drift do not "
                    f"depend on the integral and are unaffected — they are "
                    f"the result to quote here. The fitted areas are still "
                    f"in the CSVs.",
                    "",
                ]
            elif _un_any:
                # THE THIRD GROUND. From 1.9.0.56 there is no free background,
                # so the peak/background split cannot be undetermined and this
                # page must not say it was. What can still fail is the model
                # not adding up to the cell.
                m += [
                    f"- **{tot_with} cycle(s) had their capacity "
                    f"attribution withheld.** The named components do not "
                    f"add up to the charge the cell delivered"
                    # THE MEDIAN IS THE WRONG NUMBER TO PUT HERE. Quoting
                    # the run's typical unattributed fraction beside a
                    # withholding reads as though 3% had caused it, when the
                    # cycles that were withheld are the ones above the
                    # limit. Say what the limit is and let the CSV carry the
                    # per-cycle values.
                    + (
                        f" on those cycles — more than the "
                        f"{UNATTRIBUTED_WITHHOLD_ABOVE:.0%} of the cell's "
                        f"charge above which a share stops being a "
                        f"measurement (the run's median is {_un_med:+.0%})"
                        if _un_med is not None
                        else ""
                    )
                    + ". Every mAh in a dQ/dV passed through the cell, so a "
                    "share OF a total that is missing part of the "
                    "capacity is not a measurement. The fitted areas are "
                    "still in the CSVs.",
                    "",
                ]
            elif _no_baseline:
                # No free background: the split CANNOT be undetermined, so the
                # only remaining reason a share was withheld is that the
                # check fit did not converge on those half-cycles. Saying
                # "the baseline degree changed" here would send a reader to
                # look for a background this model does not have.
                m += [
                    f"- **{tot_with} cycle(s) had their capacity "
                    f"attribution withheld.** This model has no free "
                    f"background, so the peak/background split cannot be "
                    f"undetermined; these cycles were withheld because the "
                    f"check fit did not converge on them and there is "
                    f"therefore no measured share to report. The fitted "
                    f"areas are still in the CSVs.",
                    "",
                ]
            else:
                m += [
                    f"- **{tot_with} cycle(s) had their capacity "
                    f"attribution withheld.** The peak/background split "
                    f"moved more than the number would have reported when "
                    f"the baseline degree changed. The fitted areas are "
                    f"still in the CSVs.",
                    "",
                ]

    m += [
        "## Where to look",
        "",
        "Each cell has a `START_HERE.html` and `START_HERE.md` in its own "
        "folder, with four figures and what to look for in each. Start "
        "there; come back here to compare.",
        "",
    ]
    return "\n".join(m), T, R


def write_run_summary(run_dir, facts_by_name, *, run_id="", n_files=0):
    """Write `RUN_SUMMARY.md` at the top of the run folder. Returns its path."""
    md, T, R = build_run_summary(facts_by_name, run_id=run_id, n_files=n_files)
    path = os.path.join(run_dir, "RUN_SUMMARY.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    T.to_csv(os.path.join(run_dir, "RUN_SUMMARY.csv"), index=False)
    if not R.empty:
        # The REPORTED number for each material — mean, SD and n across its
        # replicates. Separate from RUN_SUMMARY.csv, which is per cell, so
        # that nobody has to re-derive a group statistic from the cells and
        # get ddof wrong.
        R.to_csv(os.path.join(run_dir, "REPLICATE_SUMMARY.csv"), index=False)
    return path
