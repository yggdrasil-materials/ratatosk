"""
What the operator is asked, and what the cycler already knows.

This is 1.8.6's Cell 2 and Cell 3, restored. It was dropped when 1.9.0 replaced
them with a hardcoded `DATASETS` list, and that was a mistake: the prompts are
not a convenience layer, they are where a researcher's knowledge of the cell
enters the analysis, and the defaults behind them are read out of the file
rather than invented.

    extract_neware_metadata   read the workbook's 'test' sheet: active mass,
                              voltage limits, C-rate, currents, cycle count
    collect_parameters        ask for the rest, with those values pre-filled,
                              and record which came from the file and which
                              from a person

Why it prompts rather than takes a dict
---------------------------------------
Because the default shown is the answer in most cases, and seeing it is the
point. `_input_with_default` prints "[from file: 12.72]" against a mass read
out of the workbook and "[default: 12.0]" against one that is only a
convention, so the operator knows which numbers they are actually responsible
for. A dict hides that distinction, and a value nobody checked looks exactly
like a value somebody confirmed.

For a headless or scheduled run, pass `answers=` to `collect_parameters` — a
dict keyed by dataset name whose entries pre-answer any prompt. Anything not
answered there falls back to the metadata default without asking. Nothing is
ever silently invented: the manifest records the source of every value.

Estimated quantities are marked as estimates
--------------------------------------------
The counter-electrode mass is estimated from geometry and metal density when
the operator has not weighed it, and the record carries both the mass and the
metal, with `counter_electrode_measured` saying which it is.

NOT implemented, though earlier versions of this docstring said it was:
separator areal density and electrolyte density are neither looked up nor
recorded, and no `*_source` field exists for them. Their lookup tables were
written and never called, so they have been removed rather than left to read
as a feature. The energy and power densities in `cycling.power_and_energy`
use only the active mass, the blend fraction and the electrode area, so no
number currently depends on either quantity.
"""

from __future__ import annotations

import builtins as _builtins
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The final summary block prints display names. `_get_display_name` lives in
# `plots` because every figure needs it; importing it here rather than keeping
# a second copy is what stops the two drifting apart. `plots` does not import
# `params`, so there is no cycle.
from .cycling import rate_protocol, snap_crate
from .plots import _get_display_name

from .style import _S, entry, half_cycle_labels

__all__ = ["extract_neware_metadata", "collect_parameters", "describe_defaults",
           "recommended_palettes", "default_key_cycles",
           "choose_files", "select_files", "pick_files_dialog",
           "detection_overrides", "analysis_overrides", "infer_chemistry",
           "USE_FILE_DIALOG", "validate_run_settings",
           "find_data_dir", "DATA_DIR_ENV",
           "parameters_from_dict", "preprocess_overrides",
           "SPECIFIC_CAPACITY_LIBRARY", "lookup_specific_capacity",
           "FARADAY_mAh_PER_MOL"]


# =============================================================================
# THE ANALYSIS KNOBS, DEFINED ONCE
# =============================================================================
# Both entry points — the interactive `collect_parameters` and the declarative
# `parameters_from_dict` — build the same record, and for a while they built it
# from two hand-written literal blocks. They diverged twice before anyone
# noticed: `anode_labels_swapped` existed on one path only, and the centre
# tolerance was 50 on one and 70 on the other, so the same dataset fitted
# different peaks depending on which door you came in by. Two independent
# derivations of one record will diverge; the only question is when.
#
# 'auto' means "let the measured profile decide" and is dropped by the
# override readers below. A number is an instruction and is kept.
ANALYSIS_DEFAULTS = {
    "smoothing_window": "auto",
    "polyorder": 3,
    "spike_removal": "auto",
    "spike_window_size": 5,
    "spike_threshold_multiplier": "auto",
    "rebin_width_mV": "auto",
    "second_smooth_window": "auto",
    # THE HISTOGRAM PATH'S OWN KNOBS. Everything above this line belongs to
    # the DERIVATIVE path, and since 1.9.0.14 the default is the histogram —
    # so every preprocessing override an operator could set was silently
    # inert, while the console still echoed "operator overrides: {...}" back
    # at them. These two are what the histogram actually reads.
    # 'auto' = take it from the measured profile, as everything else does.
    "histogram_bin_mV": "auto",
    "histogram_smooth_bins": "auto",
    "peak_prominence_fraction": 0.05,
    # 'auto' = let the profile decide, exactly as the smoothing and rebin
    # fields do. Two peaks 30 mV apart are one feature on a broad layered
    # oxide and two on a sharp two-phase material, so a fixed 30 here
    # silently overrode the profile's own answer. Put a number to force it.
    "peak_min_distance_mV": "auto",
    "peak_min_width_mV": 5,
    "reference_cycle": 2,
    "shoulder_detection": True,
    "edge_exclusion_mV": 50,
    "tracking_tolerance_mV": 80,
}

# Which reader claims which knob. Every key of ANALYSIS_DEFAULTS appears in
# exactly one of these, checked at import — a knob nobody reads is a knob that
# does nothing, and that is how the four deleted fit fields survived.
PREPROCESS_KEYS = ("smoothing_window", "polyorder", "spike_removal",
                   "spike_window_size", "spike_threshold_multiplier",
                   "rebin_width_mV", "second_smooth_window",
                   "histogram_bin_mV", "histogram_smooth_bins")

# Which dQ/dV path each preprocessing knob reaches. A knob set for the path
# that is not running does nothing, and Cell 4 says so rather than echoing it
# back as though it had been applied. The assert below proves every key is
# claimed by a reader; this says WHICH reader, which is the part that was
# actually wrong.
KEYS_BY_DQDV_METHOD = {
    "derivative": ("smoothing_window", "polyorder", "spike_removal",
                   "spike_window_size", "spike_threshold_multiplier",
                   "rebin_width_mV", "second_smooth_window"),
    "histogram": ("histogram_bin_mV", "histogram_smooth_bins"),
}
assert set(sum(KEYS_BY_DQDV_METHOD.values(), ())) == set(PREPROCESS_KEYS), \
    "a preprocessing knob belongs to no dQ/dV path"
DETECTION_KEYS = ("peak_min_distance_mV", "peak_prominence_fraction",
                  "peak_min_width_mV", "edge_exclusion_mV",
                  "shoulder_detection")
ANALYSIS_KEYS = ("reference_cycle", "tracking_tolerance_mV")
assert (set(PREPROCESS_KEYS) | set(DETECTION_KEYS) | set(ANALYSIS_KEYS)
        == set(ANALYSIS_DEFAULTS)), "an analysis knob has no reader"


def _explicit(params, keys):
    """The subset of `keys` the operator actually set. See ANALYSIS_DEFAULTS."""
    out = {}
    for k in keys:
        default = ANALYSIS_DEFAULTS[k]
        v = params.get(k, default)
        if v is None or str(v).lower() == "auto" or v == default:
            continue
        out[k] = v
    return out


def extract_neware_metadata(source):
    """
    Extract metadata from the 'test' sheet of a Neware xlsx file.

    Accepts either a filepath string or an already-opened
    pd.ExcelFile object. Passing an ExcelFile avoids re-parsing the
    workbook when the caller has already opened it for the 'record'
    sheet.

    Handles the Neware merged-cell layout where labels are in columns
    0, 3, 6 and values are at label_col + 2 (with an empty column
    between label and value due to merged cells in Excel).

    Falls back to label_col + 1 if +2 is empty, so also works with
    non-merged layouts from other cycler brands.
    """
    metadata = {
        'active_material_mass_mg': None,
        'voltage_upper_V': None,
        'voltage_lower_V': None,
        'cycle_count': None,
        'c_rate': None,
        'charge_current_A': None,
        'discharge_current_A': None,
        'barcode': None,
        'test_start_date': None,
    }

    # An absent 'test' sheet and an unreadable workbook were both swallowed
    # by one `except Exception`, so a corrupt file looked exactly like a
    # cycler that had not written its metadata: no prompt was pre-filled and
    # nothing said why. They are separated here, and neither is silent.
    if isinstance(source, pd.ExcelFile):
        if 'test' not in source.sheet_names:
            print(f"      no 'test' sheet in this workbook "
                  f"(sheets: {', '.join(map(str, source.sheet_names))}) "
                  f"— nothing to pre-fill from")
            return metadata
    try:
        # These exports carry no default cell style, and openpyxl warns about
        # it once per file. It is not a problem with the data and it buries
        # the loading messages, so it is silenced here rather than left to
        # scroll past eight times.
        import warnings as _w
        with _w.catch_warnings():
            _w.filterwarnings("ignore", message=".*no default style.*")
            df_test = pd.read_excel(source, sheet_name='test', header=None)
    except ValueError:
        # pandas raises ValueError for a sheet that is not in the workbook.
        print("      no 'test' sheet in this workbook — nothing to pre-fill")
        return metadata
    except Exception as exc:                                # noqa: BLE001
        print(f"      *** the 'test' sheet could not be read ({exc}); "
              f"every parameter will have to be typed ***")
        return metadata

    def _get(row, col):
        """Safely get a cell value, returning None for NaN/missing."""
        try:
            val = df_test.iloc[row, col]
            return None if pd.isna(val) else val
        except (IndexError, KeyError):
            return None

    def _parse_numeric(text):
        """Extract leading numeric value from '12.72mg', '4.3V', etc."""
        if text is None:
            return None
        match = re.match(r'^([+-]?\d*\.?\d+)', str(text).strip())
        return float(match.group(1)) if match else None

    n_rows = min(10, len(df_test))
    n_cols = min(15, len(df_test.columns))

    # --- Header section ---
    # Neware layout: label at col X, value at col X+2 (merged cell gap)
    # Fallback: try X+1 for non-Neware formats
    for row in range(n_rows):
        for label_col in range(n_cols - 2):
            label = _get(row, label_col)
            if label is None:
                continue
            ll = str(label).lower().strip()

            # Try +2 first (Neware merged cells), then +1
            value = _get(row, label_col + 2)
            if value is None:
                value = _get(row, label_col + 1)

            if 'active material' in ll:
                mass = _parse_numeric(value)
                if mass is not None and mass > 0:
                    metadata['active_material_mass_mg'] = mass

            elif 'volt' in ll and 'upper' in ll:
                v = _parse_numeric(value)
                if v is not None:
                    metadata['voltage_upper_V'] = v

            elif 'volt' in ll and 'lower' in ll:
                v = _parse_numeric(value)
                if v is not None:
                    metadata['voltage_lower_V'] = v

            elif 'cycle' in ll and 'count' in ll:
                try:
                    metadata['cycle_count'] = int(float(str(value)))
                except (ValueError, TypeError):
                    pass

            elif 'barcode' in ll:
                for offset in [2, 1, 3]:
                    bv = _get(row, label_col + offset)
                    if bv is not None and len(str(bv).strip()) > 3:
                        metadata['barcode'] = str(bv).strip()
                        break

            elif 'start time' in ll:
                for offset in [2, 1]:
                    sv = _get(row, label_col + offset)
                    if sv is not None:
                        metadata['test_start_date'] = str(sv).strip()
                        break

    # --- Step plan section ---
    # Find header row containing "Step Index"
    step_header_row = None
    for row in range(len(df_test)):
        for col in range(min(3, n_cols)):
            val = _get(row, col)
            if (val is not None and
                'step' in str(val).lower() and
                'index' in str(val).lower()):
                step_header_row = row
                break
        if step_header_row is not None:
            break

    if step_header_row is not None:
        # Read column headers from the step plan row
        headers = [str(_get(step_header_row, c) or '')
                   for c in range(n_cols)]

        # Find column indices — exclude "cut-off" variants
        crate_idx = None
        current_idx = None
        step_name_idx = None

        for i, h in enumerate(headers):
            hl = h.lower()
            if ('c-rate' in hl or 'c_rate' in hl) and 'cut' not in hl:
                crate_idx = i
            elif 'current' in hl and 'cut' not in hl:
                current_idx = i
            elif 'step name' in hl or 'step_name' in hl:
                step_name_idx = i

        if step_name_idx is not None:
            for row in range(step_header_row + 1, len(df_test)):
                sn = str(_get(row, step_name_idx) or '').lower()

                if any(t in sn for t in ['chg', 'charge']):
                    if crate_idx is not None:
                        try:
                            cr = float(_get(row, crate_idx))
                            if not np.isnan(cr):
                                metadata['c_rate'] = cr
                        except (ValueError, TypeError):
                            pass
                    if current_idx is not None:
                        try:
                            cu = float(_get(row, current_idx))
                            if not np.isnan(cu):
                                metadata['charge_current_A'] = cu
                        except (ValueError, TypeError):
                            pass

                elif any(t in sn for t in ['dchg', 'discharge']):
                    if crate_idx is not None and metadata['c_rate'] is None:
                        try:
                            cr = float(_get(row, crate_idx))
                            if not np.isnan(cr):
                                metadata['c_rate'] = cr
                        except (ValueError, TypeError):
                            pass
                    if current_idx is not None:
                        try:
                            cu = float(_get(row, current_idx))
                            if not np.isnan(cu):
                                metadata['discharge_current_A'] = cu
                        except (ValueError, TypeError):
                            pass

    return metadata


recommended_palettes = [
    "viridis_r",    # perceptually uniform, colourblind-safe (recommended)
    "Greys",        # greyscale light->dark
    "Greys_r",      # greyscale dark->light
    "plasma",       # perceptually uniform, vibrant
    "inferno",      # perceptually uniform, high contrast
    "magma",        # perceptually uniform, warm
    "cividis",      # perceptually uniform, colourblind-safe
    "Reds",         # sequential red
    "YlOrRd",       # yellow->orange->red
    "OrRd",         # orange->red
    "Blues",        # sequential blue
    "Paired",       # categorical pairs
    "Accent",       # categorical accent colours
    "rocket",       # seaborn sequential (dark->light)
    "mako",         # seaborn sequential (cool)
    "flare",        # seaborn sequential (warm)
    "crest",        # seaborn sequential (teal)
    "vlag",         # seaborn diverging (blue/red)
    "icefire",      # seaborn diverging (cool/warm)
    "PRGn",         # diverging purple/green
    "Spectral",     # diverging rainbow
]


default_palette = "viridis_r"


default_dqdv_charge_colour = '#0072B2'


default_dqdv_discharge_colour = '#D55E00'


default_dqdv_charge_linestyle = '-'


default_dqdv_discharge_linestyle = '-'


default_key_cycles = [1, 5, 10, 20, 30, 40, 50]
# `plots.DEFAULT_KEY_CYCLES` mirrors this (importing params into plots would
# be a cycle), so the two are checked against each other rather than trusted.
_plots_key_cycles = None
try:
    from .plots import DEFAULT_KEY_CYCLES as _plots_key_cycles
except Exception:                                    # pragma: no cover
    pass
if _plots_key_cycles is not None and list(_plots_key_cycles) != default_key_cycles:
    raise RuntimeError(
        "params.default_key_cycles and plots.DEFAULT_KEY_CYCLES disagree: "
        f"{default_key_cycles} vs {list(_plots_key_cycles)}")


default_electrode_diameter_mm = 12.0


_VALID_CMAPS = set(plt.colormaps())








_COUNTER_METAL_DENSITY = {
    'Li': 0.534,
    'Na': 0.971,
}


_DEFAULT_COUNTER_THICKNESS_MM = 0.5   # typical foil thickness


_DEFAULT_COUNTER_DIAMETER_MM = 16.0   # same as separator punch


def _validate_palette(name):
    """
    Return True if name is a valid matplotlib colormap.
    All downstream cells use plt.get_cmap(), so only matplotlib
    colormaps are valid — seaborn-only names will crash at runtime.
    """
    return name in _VALID_CMAPS


def _suggest_cell_id(name, all_names):
    """
    Best guess at which physical cell a dataset came from.

    Looks for a single letter delimited by an underscore OR a hyphen, then
    for the numeric token that varies across the loaded set. Returns '' if it
    cannot tell, which is honest rather than guessing wrongly.

    1.8.6 matched only `_A_`, so a hyphenated name like
    `JQ-P3-NaNiMnO2-35%-2-4.2V-0.1C-A-20062025` fell through to the numeric
    branch and suggested "3" — the varying digit of "35%" — for a cell called
    A. Both delimiters are now accepted, and a lone letter is preferred over
    any digit, because a cell identifier is a letter far more often than not.

    THE LAST DELIMITED LETTER, NOT THE FIRST. `JQ_NNM_C-rate_2-4.2V_A_29042026`
    has two: the C of "C-rate" and the A that names the cell. Taking the first
    labelled cell A as "cell C" — in a run of three cells that is a figure
    legend naming the wrong cell. A cell identifier sits at the end of these
    names, before the date, and everything earlier is protocol.

    An explicit `CellB` or `cell_B` wins outright, because that is somebody
    saying it rather than a convention being inferred. Without it
    `Nik_LTO-DD2_CellB` matched nothing at all and the dataset was labelled
    with its composition alone, so two single-cell runs of two different
    cells produced two identically named folders.
    """
    m = re.search(r'[Cc]ell[\s_-]?([A-Za-z0-9])(?![A-Za-z0-9])', name)
    if m:
        return m.group(1).upper()
    m = re.findall(r'[_-]([A-Za-z])[_-]', name)
    if m:
        return m[-1].upper()
    toks = {n: re.findall(r'\d+', n) for n in all_names}
    mine = toks.get(name, [])
    if mine and len(all_names) > 1:
        n_tok = min((len(t) for t in toks.values()), default=0)
        for pos in range(n_tok):
            if len({t[pos] for t in toks.values() if len(t) > pos}) > 1:
                return mine[pos]
    return ''


def _input_with_default(prompt, default_value, cast_type=str,
                         min_val=None, source=None):
    """Prompt for input with a pre-filled default shown in brackets."""
    # input() does not render ANSI codes — build a plain-text prompt.
    # Styling is applied only to print() calls (errors, confirmations).
    if default_value is not None:
        if source == 'file':
            src_tag = "[from file: "
        elif source == 'library':
            src_tag = "[from library: "
        else:
            src_tag = "[default: "
        full_prompt = f"{prompt} {src_tag}{default_value}]: "
    else:
        full_prompt = f"{prompt}: "
    while True:
        raw = _builtins.input(full_prompt).strip()
        if not raw:
            if default_value is not None:
                return cast_type(default_value) if cast_type != str else default_value
            else:
                print(f"  {_S.WARN}No default available — please enter a value.{_S.RESET}")
                continue
        try:
            value = cast_type(raw)
            if min_val is not None and value < min_val:
                print(f"  {_S.WARN}Value must be >= {min_val}.{_S.RESET}")
                continue
            return value
        except (ValueError, TypeError):
            print(f"  {_S.WARN}Invalid input. Please enter a {cast_type.__name__}.{_S.RESET}")


def _skip_module(module_name,
                 reason="RUN_PEAK_FITTING=False (set in Cell 3)"):
    """
    Print a clear skip message. Does NOT raise — the caller is
    expected to guard the module body with an if/else block so
    execution simply falls through to the next cell.

    Typical usage at the top of a module cell:

        if not RUN_PEAK_FITTING:
            _skip_module("Module 3")
        else:
            # existing module body, indented +4 spaces
    """
    print(f"{_S.WARN}--- {module_name} skipped ({reason}). ---{_S.RESET}")
    print(f"    To run this analysis now, set "
          f"RUN_PEAK_FITTING = True in Cell 3, re-run Cell 3,")
    print(f"    then run this module's cell manually. "
          f"No other cells need to re-run.")






# ---------------------------------------------------------------------------
# Specific capacity library
# ---------------------------------------------------------------------------
# A THEORETICAL CAPACITY IS ARITHMETIC, NOT AN OPINION. It is
# `n * F / M` — electrons per formula unit, Faraday's constant, molar mass —
# and typing it by hand at a prompt, once per cell, is a transcription step
# with nothing checking it. On a triplicate that is three chances to enter a
# different number for the same material, and the capacity divides every
# normalised figure the run produces.
#
# So the library stores `n` and `M` and DERIVES the capacity. The number
# cannot be a typo; it can only be wrong if the stoichiometry is wrong, and
# the stoichiometry is printed at the prompt so a wrong one is visible at the
# moment it is accepted rather than buried in a manifest.
#
# It is a DEFAULT, not a constraint. Every value can be overridden by typing
# one, exactly as before, and a composition that is not in the library gets
# no default at all — which is the current behaviour and the right one. A
# library that guessed would be worse than no library.
#
# F = 96485.33212 C/mol / 3600 s/h = 26801.48 mAh/mol.
FARADAY_mAh_PER_MOL = 96485.33212 / 3.6

# Molar masses computed from IUPAC 2021 conventional atomic weights:
#   Li 6.94  Na 22.98977  Ti 47.867  Mn 54.93804  Ni 58.6934
#   Co 58.93319  O 15.999
#
#   LiCoO2               M =  97.871   n = 1     -> 273.8
#   LiMn2O4              M = 180.812   n = 1     -> 148.2
#   LiNi1/3Mn1/3Co1/3O2  M =  96.460   n = 1     -> 277.9
#   LiNi0.8Mn0.1Co0.1O2  M =  97.280   n = 1     -> 275.5
#   LiCoMnO4             M = 184.807   n = 1     -> 145.0
#   Li4Ti5O12            M = 459.083   n = 3     -> 175.1
#   Na0.67Ni0.33Mn0.67O2 M = 103.578   n = 0.67  -> 173.4
#
# `n` is the FULL theoretical extraction, which for several of these is not
# the number a cell delivers in its working window: LiCoO2 is cycled over
# about half its lithium, and Li4Ti5O12 inserts three lithium into the
# 8a/16c sites without touching the rest. That is the definition the field
# quotes and the one a % -of-theoretical figure is read against, so it is the
# one stored — but it is why the note is printed with it.
#
# NNM IS THE UNDOPED PARENT, AND THAT IS THE POINT OF PRINTING THE FORMULA.
# The library entry is Na0.67Ni0.33Mn0.67O2 -> 173.4, which is the nominal P3
# composition. The cells this project actually cycles are DOPED, and their
# capacity is 168.0 — a property of a specific substituted composition that
# no library can derive and none should try to. So the default is the parent
# arithmetic, the prompt says which formula it came from, and an operator
# working on a doped sample types their own number over it, which the
# manifest then records as an override rather than as a library value.
#
# (Na2/3Ni1/3Mn2/3O2 is a DIFFERENT entry, M = 103.514 and 172.6 mAh/g, and
# is deliberately not aliased to this one. The 0.67 spelling is the one used
# here because it is the one written on the samples.)
SPECIFIC_CAPACITY_LIBRARY = {
    "LiCoO2": dict(
        n=1.0, M=97.871, ion="Li",
        note="1 Li per formula unit (full extraction to CoO2)",
        aliases=("lco", "licoo2", "lithiumcobaltoxide", "lithiumcobaltate")),
    "LiMn2O4": dict(
        n=1.0, M=180.812, ion="Li",
        note="1 Li per formula unit (spinel, 4 V plateau)",
        aliases=("lmo", "limn2o4", "lithiummanganeseoxide",
                 "lithiummanganatespinel")),
    "LiNi1/3Mn1/3Co1/3O2": dict(
        n=1.0, M=96.460, ion="Li",
        note="1 Li per formula unit (full delithiation)",
        aliases=("nmc111", "ncm111", "mnc111", "lini13mn13co13o2",
                 "lini033mn033co033o2", "lini0333mn0333co0333o2")),
    "LiNi0.8Mn0.1Co0.1O2": dict(
        n=1.0, M=97.280, ion="Li",
        note="1 Li per formula unit (full delithiation)",
        aliases=("nmc811", "ncm811", "mnc811", "lini08mn01co01o2")),
    "LiCoMnO4": dict(
        n=1.0, M=184.807, ion="Li",
        note="1 Li per formula unit (5 V spinel)",
        aliases=("licomno4", "lcmo", "limncoo4")),
    "Li4Ti5O12": dict(
        n=3.0, M=459.083, ion="Li",
        note="3 Li per formula unit (8a -> 16c, Ti4+/Ti3+)",
        aliases=("lto", "li4ti5o12", "lithiumtitanate",
                 "lithiumtitaniumoxide")),
    "Na0.67Ni0.33Mn0.67O2": dict(
        n=0.67, M=103.578, ion="Na",
        note="0.67 Na per formula unit, UNDOPED parent P3 composition — "
             "a doped sample is not this number",
        # NOT `Na2/3Ni1/3Mn2/3O2`: that stoichiometry is M = 103.514 and
        # 172.6 mAh/g, and answering it with 173.4 under a formula the
        # operator did not write is exactly the near-match this library
        # refuses everywhere else. It gets no default until it is entered
        # here with its own arithmetic.
        aliases=("nnm", "na067ni033mn067o2")),
}


def _capacity_key(text):
    """Normalise a composition for library lookup.

    Case, spaces, hyphens, brackets and the separators people put in formulae
    are not chemistry. `NMC-111`, `nmc 111` and `NMC111` are one material;
    so are `Li4Ti5O12` and `li4ti5o12`.
    """
    return re.sub(r"[\s\-_,.()\[\]/·]", "", str(text or "")).lower()


def lookup_specific_capacity(composition):
    """
    `(capacity_mAh_g, canonical_formula, note)` for a known material, else
    `(None, None, None)`.

    The capacity is DERIVED from the stored `n` and `M` and rounded to one
    decimal place. Matching is on the normalised composition against the
    canonical formula and its aliases — nothing is inferred from a partial
    match, because "NMC" with no digits is not a stoichiometry and a library
    that guessed 111 would put a number on a cell nobody chose it for.
    """
    key = _capacity_key(composition)
    if not key:
        return (None, None, None)
    for formula, rec in SPECIFIC_CAPACITY_LIBRARY.items():
        if key == _capacity_key(formula) or key in rec["aliases"]:
            cap = round(rec["n"] * FARADAY_mAh_PER_MOL / rec["M"], 1)
            return (cap, formula, rec["note"])
    return (None, None, None)


def infer_chemistry(composition, electrolyte=""):
    """
    Li-ion, Na-ion or Unknown, from the electrolyte salt first.

    The ELECTROLYTE decides it, because that is where the working ion
    actually comes from; the composition is only a fallback. A sodium
    half-cell with a hard-carbon or LTO working electrode has no Na in its
    formula at all, and the declarative path — which tested
    `"Na" in composition` and nothing else — called every one of them
    Li-ion. That picks the counter-electrode metal and its density (Li 0.534
    vs Na 0.97 g/cm3, a 1.8x mass error) and the areal-capacity benchmark
    lines, so the two entry points produced different energy densities from
    the same cell.
    """
    e = str(electrolyte or "").lower()
    c = str(composition or "").lower()
    if "napf6" in e or "naclo4" in e or "natfsi" in e:
        return "Na-ion"
    if "lipf6" in e or "litfsi" in e or "libf4" in e or "liclo4" in e:
        return "Li-ion"
    # The composition fallback matches the ELEMENT SYMBOL, not a prefix: Na
    # or Li followed by anything that is not a lower-case letter, so NaNiMnO2
    # and Li4Ti5O12 match while Nb2O5 and a stray "lithiated" do not. A
    # prefix test missed every formula where the alkali is not written first,
    # and `"Na" in composition` matched things like "NaN".
    comp = str(composition or "")
    if re.search(r"Na(?![a-z])", comp):
        return "Na-ion"
    if re.search(r"Li(?![a-z])", comp):
        return "Li-ion"
    # An abbreviation such as LTO or NMC names no element, so say so rather
    # than guess: the answer sets the counter-electrode metal.
    return "Unknown"


# WHICH METAL IS THE COUNTER ELECTRODE MADE OF?
# `infer_chemistry` deliberately answers `Unknown` for a composition that names
# no element, because an abbreviation is not a formula and the electrolyte is
# what decides the working ion. That is right. What was wrong is what happened
# next: the estimator fell through to `metal = 'Na'` for every Unknown, with
# nothing said.
#
# Every abbreviation this project actually uses is Unknown — LTO, NMC, NNM —
# so a lithium half-cell cycled against lithium metal was reported as
# "~98 mg (Na metal, estimated)" on all four LTO cells today. Li is
# 0.534 g/cm3 and Na is 0.97: a 1.8x mass error, and the counter mass sets the
# areal and gravimetric energy benchmarks.
#
# So the metal is now decided in this order, and the ORDER IS RECORDED:
#   stated      the operator said so
#   chemistry   inferred from the electrolyte salt, or from the formula
#   library     the composition is in SPECIFIC_CAPACITY_LIBRARY, whose entries
#               carry the working ion (LTO -> Li). A weaker claim than the
#               electrolyte, because a lithium-containing material can be
#               cycled in a sodium cell — but far better than a blanket guess
#   assumed     none of the above. The interactive path asks rather than
#               reaching here; the declarative path lands here and the value
#               is labelled `assumed` everywhere it is printed
COUNTER_METAL_DEFAULT = "Na"


def counter_electrode_metal(params_or_composition, *, stated=None,
                            battery_chemistry=None, electrolyte=""):
    """`(metal_symbol, source)` — see COUNTER_METAL_DEFAULT."""
    if stated:
        return str(stated).strip().title()[:2], "stated"
    chem = battery_chemistry or infer_chemistry(
        params_or_composition if isinstance(params_or_composition, str)
        else (params_or_composition or {}).get("composition"), electrolyte)
    if chem == "Li-ion":
        return "Li", "chemistry"
    if chem == "Na-ion":
        return "Na", "chemistry"
    comp = (params_or_composition if isinstance(params_or_composition, str)
            else (params_or_composition or {}).get("composition"))
    key = _capacity_key(comp)
    for formula, rec in SPECIFIC_CAPACITY_LIBRARY.items():
        if key and (key == _capacity_key(formula) or key in rec["aliases"]):
            return rec.get("ion", COUNTER_METAL_DEFAULT), "library"
    return COUNTER_METAL_DEFAULT, "assumed"


def _estimate_counter_electrode_mass_mg(diameter_mm=None,
                                         thickness_mm=None,
                                         battery_chemistry=None,
                                         metal=None):
    """
    Estimate counter electrode mass from geometry and metal density.
    Returns (mass_mg, metal_symbol). `metal` overrides the inference.
    """
    if diameter_mm is None:
        diameter_mm = _DEFAULT_COUNTER_DIAMETER_MM
    if thickness_mm is None:
        thickness_mm = _DEFAULT_COUNTER_THICKNESS_MM

    # Determine counter electrode metal
    if not metal:
        if battery_chemistry == 'Li-ion':
            metal = 'Li'
        elif battery_chemistry == 'Na-ion':
            metal = 'Na'
        else:
            metal = COUNTER_METAL_DEFAULT

    density = _COUNTER_METAL_DENSITY.get(metal, 0.534)
    radius_cm = diameter_mm / 2 / 10
    area_cm2 = np.pi * radius_cm**2
    volume_cm3 = area_cm2 * (thickness_mm / 10)
    mass_mg = volume_cm3 * density * 1000
    return mass_mg, metal




def collect_parameters(electrochemical_data, cycler_metadata=None,
                       answers=None, headless=False, verbose=True):
    """
    Ask for each dataset's parameters, pre-filled from the cycler's own file.

    Returns `(user_parameters, run_peak_fitting)`. Ported from 1.8.6 Cell 3
    without editing the prompt sequence; the only change is that its
    notebook-global state (`_last_composition`, `_POWER_ENERGY_GLOBAL` and the
    rest) are locals here, which is what they always should have been.

    `answers` pre-answers prompts, matched on the start of the prompt text:
    `{"NNM_cellA": {"Theoretical capacity": 168}}`. `headless=True` accepts
    every default without asking. Both exist for scheduled runs and tests; the
    interactive path is the one a researcher should use.
    """
    cycler_metadata = cycler_metadata or {}
    _ANSWERS = dict(answers or {})
    _CURRENT = {"name": None}
    _last_electrode_type = None
    _asked = {}

    # `input` is shadowed UNCONDITIONALLY. Defining it inside `if headless:`
    # makes the name local to this whole function anyway, so on the interactive
    # path every raw `input()` call would raise UnboundLocalError before
    # reaching a person.
    def input(prompt=""):                       # noqa: A001
        text = str(prompt).lower().strip()
        pre = _ANSWERS.get(_CURRENT["name"], {})
        for k, v in pre.items():
            if text.startswith(str(k).lower().strip()):
                if verbose:
                    print(f"{prompt}{v}   [answered]")
                return str(v)
        if not headless:
            return _builtins.input(prompt)
        # Only the confirmation prompt is auto-confirmed. Answering "y" to
        # every (yes/no) question made a headless run silently opt IN to
        # full peak fitting, whose own prompt says "leave blank for no";
        # a blank lets each prompt's own default decide, which is what the
        # docstring promises.
        reply = "y" if "correct?" in text else ""
        _asked[text] = _asked.get(text, 0) + 1
        if _asked[text] > 3:
            raise ValueError(
                f"headless run is stuck on {_CURRENT['name']!r}: {prompt!r} "
                f"rejects a blank answer and has no default. Supply it in "
                f"`answers`.")
        if verbose:
            print(f"{prompt}{reply}   [auto]")
        return reply

    def _ask(prompt, default_value, cast_type=str, min_val=None, source=None):
        """`_input_with_default`, answerable for a headless run."""
        # PREFIX FIRST, THEN CONTAINMENT — longest key wins.
        # The match was `prompt.startswith(key)` alone, so the obvious key for
        # a headless run, `"Composition"`, silently matched NOTHING: the
        # prompt is "Active material composition (e.g., LiCoO2)". While the
        # composition question used a bare `input()` that merely hung; since
        # 1.9.0.62 it goes through here, and a scripted run died on it with
        # "has no default from the cycler file" — which is true and is not
        # the problem. Containment is what a caller means by a key, and
        # taking the LONGEST match keeps "Theoretical capacity" from being
        # captured by a bare "capacity".
        pre = _ANSWERS.get(_CURRENT["name"], {})
        _pl = str(prompt).lower().strip()
        _cands = [(k, v) for k, v in pre.items()
                  if _pl.startswith(str(k).lower().strip())]
        if not _cands:
            _cands = [(k, v) for k, v in pre.items()
                      if str(k).lower().strip() in _pl]
        hit = (max(_cands, key=lambda kv: len(str(kv[0])))[1]
               if _cands else None)
        if hit is not None:
            if verbose:
                print(f"  {prompt}: {hit}   [answered]")
            return cast_type(hit) if cast_type is not str else hit
        if headless:
            if default_value is None:
                raise ValueError(
                    f"headless run needs an answer for {_CURRENT['name']!r}: "
                    f"{prompt!r} has no default from the cycler file.")
            if verbose:
                print(f"  {prompt}: {default_value}   [default accepted]")
            return (cast_type(default_value) if cast_type is not str
                    else default_value)
        return _input_with_default(prompt, default_value, cast_type,
                                   min_val, source)

    user_parameters = {}
    _last_composition = None  # carries forward between datasets
    _last_theoretical_capacity = None  # carries forward for same composition

    # Pipeline-level flag for Cells 8-12 (peak detection, multi-peak
    # fitting, tracking, derived quantities, diagnostics, export). Asked
    # once during the first dataset's inputs, immediately after the
    # power-analysis question. None = not yet asked; True/False = user
    # answered. Cells 8-12 each guard their body with
    # `if not RUN_PEAK_FITTING: _skip_module("Module N")`.
    RUN_PEAK_FITTING = None
    _POWER_ENERGY_GLOBAL = None
    _FILE_FORMAT_GLOBAL = None
    _ELECTROLYTE_COMP_GLOBAL = None
    # The cycle cut-off carries forward, like the electrolyte above it.
    # Datasets analysed together are usually replicates of one experiment, so a
    # cap set on the first is almost always wanted on the rest — and when it is
    # NOT applied consistently the comparative figures quietly compare
    # different cycle ranges, which is how the NNM triplicate came to be
    # analysed to cycle 80 in one build and 75 in the next. None means nothing
    # is being carried, and then the prompt is exactly as it was.
    _MAX_CYCLE_CARRIED = None

    for name, df in electrochemical_data.items():
        _CURRENT["name"] = name
        # 1.8.7 guarded this with `if 'cycler_metadata' in globals()` because
        # Cell 2 might not have run. Here it is a PARAMETER, so the guard is
        # always False and every metadata default silently vanishes — which
        # leaves prompts with no default sitting in a `while True` that
        # rejects a blank answer.
        meta = cycler_metadata.get(name, {}) or {}
        print(f"\n{_S.HEADER}{'='*60}")
        print(f"  Setting parameters for: {name}")
        if any(v is not None for v in meta.values()):
            print(f"  (Values marked [from file] were read from the cycler)")
        print(f"{'='*60}{_S.RESET}")
        confirmed = False
        while not confirmed:
            # ----- MATERIAL -----
            print(f"\n{_S.SECTION}--- Material ---{_S.RESET}")
            # A BLANK COMPOSITION IS NOT A MISSING LABEL, IT CHANGES THE
            # MODEL. `quality.reconcile_mechanisms` groups cells by
            # composition and reconciles a group to its most permissive
            # mechanism, because a material either delivers charge across a
            # composition window or it does not. Its fallback for an unknown
            # composition is the dataset's own name, which is unique, so the
            # cell forms a group of one and keeps whatever its own reference
            # cycle called it.
            #
            # Measured, NNM 1.9.0.61, 2026-09-09: cell A's own call is
            # `multi_transition`, whose model has `band_width_max = 0.0`.
            # Reconciled with cells B and C it is fitted as `mixed` and gets
            # 215 bands; left alone it gets NONE, and R2 falls 0.9930 ->
            # 0.9314 across 160 half-cycles. The only difference between the
            # two runs was an empty answer here.
            #
            # So this prompt no longer accepts one. `_ask` already refuses a
            # blank when there is no default, carries the previous cell's
            # answer when there is one, and is answerable for a headless run.
            composition = str(_ask(
                "Active material composition (e.g., LiCoO2)",
                _last_composition, cast_type=str,
                source='previous cell' if _last_composition else None
            )).strip()
            # ----- CELL IDENTIFIER -----
            # Which physical cell this is. Previously inferred from the
            # filename at display time, only when two datasets shared a
            # composition, and never stored. Load one cell alone and it had no
            # identity at all, which is how one label ends up on two different
            # cells in different figures. Asked when several cells are loaded;
            # otherwise the parsed value is stored silently so a record exists.
            _cid_suggest = _suggest_cell_id(name, list(electrochemical_data))
            if len(electrochemical_data) > 1:
                while True:
                    _cid_in = input(
                        f"Cell identifier (A, B, 1, 2 ...) "
                        f"[{_cid_suggest or 'none'}]: ").strip().upper()
                    cell_id = _cid_in or _cid_suggest
                    if not cell_id:
                        print(f"  {_S.WARN}Enter an identifier. Without one you "
                              f"cannot tell which cell a figure came from."
                              f"{_S.RESET}")
                        continue
                    _clash = [n for n, p in user_parameters.items()
                              if str(p.get('cell_id', '')).upper() == cell_id]
                    if _clash:
                        print(f"  {_S.WARN}'{cell_id}' is already used by "
                              f"{_clash[0]}.{_S.RESET}")
                        print(f"  {_S.WARN}Two datasets labelled as the same "
                              f"physical cell is exactly the ambiguity this "
                              f"field prevents.{_S.RESET}")
                        continue
                    break
            else:
                cell_id = _cid_suggest
            while True:
                blend_input = input(
                    "Electrode blend ratio "
                    "(e.g., 80/10/10, leave blank for 80/10/10): "
                )
                blend = blend_input if blend_input else "80/10/10"
                try:
                    blend_values = [float(p) for p in blend.split('/')]
                    if (len(blend_values) >= 2
                            and all(v >= 0 for v in blend_values)
                            and sum(blend_values) > 0):
                        break
                    else:
                        print(f"  {_S.WARN}Need at least 2 components, all non-negative.{_S.RESET}")
                except ValueError:
                    print(f"  {_S.WARN}Invalid format. Use numbers separated by '/'.{_S.RESET}")
            active_material_mass_mg = _ask(
                "Mass of active material (mg)",
                meta.get('active_material_mass_mg'),
                cast_type=float, min_val=0.001,
                source='file' if meta.get('active_material_mass_mg') else None
            )
            # Carry forward theoretical capacity if composition unchanged.
            # The previous cell's answer wins over the library, because it is
            # what THIS OPERATOR chose for THIS material in THIS run — an
            # override typed once should not have to be typed again.
            _theo_default = (_last_theoretical_capacity
                             if (_last_theoretical_capacity is not None
                                 and _last_composition is not None
                                 and composition == _last_composition)
                             else None)
            _theo_source = 'previous cell' if _theo_default else None
            _lib_formula = None
            if _theo_default is None:
                _lib_cap, _lib_formula, _lib_note = \
                    lookup_specific_capacity(composition)
                if _lib_cap is not None:
                    # PRINT THE STOICHIOMETRY THE NUMBER CAME FROM. A default
                    # is accepted by pressing return, so the assumption behind
                    # it has to be visible at that moment and not in a
                    # manifest read afterwards.
                    print(f"  {_S.VALUE}Library: {_lib_formula} -> "
                          f"{_lib_cap} mAh/g{_S.RESET}  ({_lib_note})")
                    _theo_default = _lib_cap
                    _theo_source = 'library'
            theoretical_capacity_mAh_g = _ask(
                "Theoretical capacity (mAh/g)",
                _theo_default,
                cast_type=float, min_val=0.001,
                source=_theo_source
            )
            # Provenance, not decoration: a number that came from the library
            # and a number somebody chose are different claims, and only one
            # of them is evidence about this electrode.
            if (_theo_source == 'library' and _theo_default is not None
                    and float(theoretical_capacity_mAh_g)
                    != float(_theo_default)):
                _theo_source = 'entered (library offered '\
                               f'{_theo_default})'
            elif _theo_source is None:
                _theo_source = 'entered'
            _theo_provenance = _theo_source
            _theo_library_formula = _lib_formula
            # ----- ELECTRODE GEOMETRY -----
            print(f"\n{_S.SECTION}--- Electrode geometry ---{_S.RESET}")
            electrode_diameter_mm = _ask(
                "Electrode punch diameter (mm)",
                default_electrode_diameter_mm,
                cast_type=float, min_val=0.1,
                source=None
            )
            # v1.8: auto-detect electrode type from Neware protocol.
            # After Cell 2 processing, Rest rows are removed and Step is
            # renamed to 'Charge'/'Discharge'. Anode protocols start with
            # Discharge (drive voltage down to insert Li/Na). Cathode
            # protocols start with Charge (extract Li/Na from as-made
            # cathode). Also carries forward from previous dataset.
            # (1.8.6 guarded this with `'_last_electrode_type' not in
            # globals()`. It is a LOCAL, assigned above and below, so the
            # guard was always true and reset the carry-forward on every
            # dataset. Removed; the initialisation at the top of the
            # function is the one that counts.)

            # Detect polarity from the protocol every time. Carry-forward is
            # only a fallback when detection is inconclusive, otherwise the
            # first dataset in a mixed load sets the default for all of them.
            _detected_electrode = None
            _first_step = None
            try:
                if 'Step' in df.columns and not df.empty:
                    _first_step = str(df['Step'].iloc[0]).strip()
                    if _first_step == 'Discharge':
                        _detected_electrode = "Negative"
                    elif _first_step == 'Charge':
                        _detected_electrode = "Positive"
            except Exception:
                pass

            _suggested_electrode = (_detected_electrode
                                    or _last_electrode_type
                                    or "Positive")

            if _detected_electrode is not None:
                print(f"  Protocol starts with {_first_step} — "
                      f"suggesting {_detected_electrode} electrode")

            _suggest_str = _suggested_electrode
            while True:
                electrode_type_input = input(
                    f"Positive (cathode) or negative (anode) electrode? "
                    f"(p/n, leave blank for {_suggest_str}): "
                ).lower()
                if not electrode_type_input:
                    electrode_type = _suggested_electrode
                    break
                elif electrode_type_input in ["positive", "pos", "p"]:
                    electrode_type = "Positive"
                    break
                elif electrode_type_input in ["negative", "neg", "n"]:
                    electrode_type = "Negative"
                    break
                else:
                    print(f"  {_S.WARN}Enter 'positive'/'p' or 'negative'/'n'.{_S.RESET}")

            _last_electrode_type = electrode_type
            # ----- PIPELINE-LEVEL: POWER/ENERGY (asked once) -----
            if _POWER_ENERGY_GLOBAL is None:
                pe_input = input(
                    "Run power and energy density analysis (Cell 6c)? "
                    "(yes/no, leave blank for yes): "
                ).lower()
                _POWER_ENERGY_GLOBAL = pe_input not in ["no", "n"]
            else:
                _pe_state = (f"{_S.CONFIRM}ON{_S.RESET}" if _POWER_ENERGY_GLOBAL
                             else f"{_S.WARN}SKIP{_S.RESET}")
                print(f"Power/energy (Cell 6c): {_pe_state} "
                      f"{_S.FILE}(set on first dataset){_S.RESET}")
            power_energy_analysis = _POWER_ENERGY_GLOBAL

            # ----- PIPELINE-LEVEL: dQ/dV PEAK FITTING (asked once) -----
            # Cell 7 (dQ/dV curve plotting) runs regardless and is fast.
            # Cells 8-12 (peak detection, multi-peak fitting, tracking,
            # derived quantities, fit diagnostics, single-cycle inspector,
            # export) are slow and only needed when Cell 7 reveals
            # features worth quantifying. Asked once on the first dataset;
            # the answer applies to the whole notebook run. Can be changed
            # by re-running Cell 3.
            if RUN_PEAK_FITTING is None:
                # Blank means YES. dQ/dV peak fitting is what this tool is
                # FOR — everything from Cell 9 to Cell 13 depends on it, and
                # a blank default of "no" meant a full run could complete
                # with no fitted peaks, no tracking, no polarisation and no
                # capacity attribution, having asked once, quietly, twenty
                # prompts earlier. Answer 'no' for a quick cycling-only pass.
                while True:
                    _pk_input = input(
                        "Run full dQ/dV peak fitting (Cells 8-12)? "
                        "Slower, and the point of the tool. "
                        "(yes/no, blank = YES): "
                    ).strip().lower()
                    if _pk_input in ('n', 'no'):
                        RUN_PEAK_FITTING = False
                        break
                    if _pk_input in ('', 'y', 'yes'):
                        RUN_PEAK_FITTING = True
                        break
                    print(f"  {_S.WARN}Enter 'yes' or 'no'.{_S.RESET}")
            else:
                # Second and subsequent datasets: show the existing choice
                # for reassurance, without asking again.
                _pk_state = (f"{_S.CONFIRM}ON{_S.RESET}" if RUN_PEAK_FITTING
                             else f"{_S.WARN}SKIP{_S.RESET}")
                print(f"dQ/dV peak fitting (Cells 8-12): {_pk_state} "
                      f"{_S.FILE}(set on first dataset){_S.RESET}")

            # ----- CYCLING CONDITIONS -----
            print(f"\n{_S.SECTION}--- Cycling conditions ---{_S.RESET}")
            # ASKING WHAT THE C-RATE WAS MAKES NO SENSE WHEN THE FILE RECORDS
            # THE CURRENT. The cycler's header states one rate — its FIRST
            # step — and on a rate-capability run that is a value the
            # operator is being asked to confirm for a protocol it does not
            # describe. `cycling.rate_protocol` reads the recorded current
            # and answers the question from the data, so:
            #
            #   several rates   the prompt is SKIPPED. There is no single
            #                   answer to give, and inviting one produced
            #                   "0.05 C" on every caption of a run that
            #                   ramped C/20 to 5C. The reference block's rate
            #                   is kept for the theoretical-current line and
            #                   the protocol travels on the parameters.
            #   one rate        the MEASURED rate is the default, in place of
            #                   the header value, which describes only the
            #                   first step even when every step matches.
            #   no current      exactly as before.
            #
            # Answered here rather than in Cell 6a because every caption
            # between the two would otherwise still be quoting the header.
            _proto = rate_protocol(df, dict(
                active_material_mass_mg=active_material_mass_mg,
                theoretical_capacity_mAh_g=theoretical_capacity_mAh_g))
            _rate_meas = None
            if _proto.get('available') and _proto.get('blocks'):
                _rate_meas = _proto['blocks'][0].get('c_rate')
            if _proto.get('is_variable'):
                print(f"  {_S.VALUE}This cell was cycled at "
                      f"{_proto['n_blocks']} different rates "
                      f"({_proto['label']}){_S.RESET} — read from the "
                      f"recorded current, so you are not asked for one.")
                for _b in _proto['blocks']:
                    print(f"      {_b['label']:>6}  cycles "
                          f"{_b['first_cycle']}-{_b['last_cycle']}")
                if _proto.get('transition_cycles'):
                    print(f"      rate-transition cycle(s) in no block: "
                          + ", ".join(str(c)
                                      for c in _proto['transition_cycles']))
                if _proto.get('returns_to_start'):
                    print(f"      returns to {_proto['blocks'][0]['label']} "
                          f"at the end — recovery will be reported")
                charge_rate_c = float(snap_crate(_rate_meas)) if _rate_meas else (
                    meta.get('c_rate') or 0.1)
            else:
                charge_rate_c = _ask(
                    "C-rate",
                    # `extract_neware_metadata` always returns the key, set to
                    # None when the file does not state a rate, so `.get`'s
                    # default was unreachable and the prompt had no default at
                    # all — which a headless run cannot answer.
                    (round(float(snap_crate(_rate_meas)), 4)
                     if _rate_meas else (meta.get('c_rate') or 0.1)),
                    cast_type=float, min_val=0.001,
                    source=('measured' if _rate_meas else
                            ('file' if meta.get('c_rate') else None))
                )
            # ----- VOLTAGE WINDOW -----
            # The file's 'Volt. upper' and 'Volt. lower' are the channel safety
            # limits, not the protocol. An LTO cell cycled 1.2-2.5 V reports
            # 0-3.25 V, which is the range the cycler was permitted to use.
            #
            # Derive the tested window from the data instead: the median of the
            # per-cycle minimum and maximum. The first cycle usually starts from
            # open circuit and reads high, and an aborted final cycle reads
            # short; a median across cycles ignores both without needing to
            # identify either. Rest rows were already dropped in Cell 2.
            # Percentiles were tried and are too aggressive, because the cell
            # spends most of its time on the plateau.
            #
            # A SHORT RECORD IS NOT A REASON TO QUOTE THE CHANNEL LIMITS. The
            # median needs three cycles to ignore the first (which starts from
            # open circuit) and the last (which may be cut short), and below
            # that the fallback was the file's safety limits — which on
            # `JQ_NNM_C-rate_2-4.2V_A_29042026`, a two-cycle export of a cell
            # cycled 2-4.2 V, put "1.0--4.6 V" on the run and every caption in
            # it. The filename says otherwise and so does the data.
            #
            # With one or two cycles the OBSERVED RANGE is still a fact: the
            # cell demonstrably reached those voltages, so the protocol window
            # is AT LEAST that wide. It is reported as exactly that — a lower
            # bound, not the protocol — which beats a pair of numbers that are
            # certainly wrong in the other direction.
            _v_meas_lo = _v_meas_hi = None
            _v_source = None
            try:
                if {'Cycle', 'Voltage'} <= set(df.columns) and not df.empty:
                    _pc = df.groupby('Cycle')['Voltage'].agg(['min', 'max'])
                    if len(_pc) >= 3:
                        _v_meas_lo = round(float(_pc['min'].median()), 2)
                        _v_meas_hi = round(float(_pc['max'].median()), 2)
                        _v_source = 'measured from data'
                    elif len(_pc) >= 1:
                        _v_meas_lo = round(float(_pc['min'].min()), 2)
                        _v_meas_hi = round(float(_pc['max'].max()), 2)
                        _v_source = 'observed range (short record)'
            except Exception:
                pass

            _v_file_lo = meta.get('voltage_lower_V')
            _v_file_hi = meta.get('voltage_upper_V')

            if _v_meas_lo is not None:
                voltage_lower_V, voltage_upper_V = _v_meas_lo, _v_meas_hi
                voltage_window_source = _v_source
                if _v_source == 'measured from data':
                    print(f"  {_S.FILE}Voltage window measured from data: "
                          f"{voltage_lower_V}--{voltage_upper_V} V{_S.RESET}")
                else:
                    print(f"  {_S.WARN}Only "
                          f"{len(_pc)} cycle(s) — too few to measure the "
                          f"protocol window from the per-cycle medians. Using "
                          f"the range the cell was OBSERVED to reach, "
                          f"{voltage_lower_V}--{voltage_upper_V} V: the "
                          f"protocol window is at least this wide.{_S.RESET}")
                if (_v_file_lo is not None and _v_file_hi is not None
                        and (abs(float(_v_file_lo) - voltage_lower_V) > 0.05
                             or abs(float(_v_file_hi) - voltage_upper_V) > 0.05)):
                    print(f"  {_S.FILE}(file reports "
                          f"{_v_file_lo}--{_v_file_hi} V, which are the channel "
                          f"limits rather than the protocol){_S.RESET}")
            else:
                voltage_lower_V, voltage_upper_V = _v_file_lo, _v_file_hi
                voltage_window_source = 'channel limits from file'
                if voltage_lower_V is not None:
                    print(f"  {_S.WARN}Voltage window taken from the file's "
                          f"channel limits ({voltage_lower_V}--"
                          f"{voltage_upper_V} V). Too few cycles to measure the "
                          f"tested window, so this may be wider than the "
                          f"protocol.{_S.RESET}")
            while True:
                key_cycles_str = input(
                    f"Key cycles to plot "
                    f"(comma-separated, leave blank for "
                    f"{', '.join(map(str, default_key_cycles))}): "
                )
                if not key_cycles_str:
                    # A copy. Storing the module-level list itself made every
                    # dataset share one object, so an in-place edit anywhere
                    # rewrote the default for the whole session.
                    key_cycles = list(default_key_cycles)
                    break
                try:
                    key_cycles = [int(c.strip())
                                  for c in key_cycles_str.split(",")]
                    break
                except ValueError:
                    print(f"  {_S.WARN}Enter comma-separated integers.{_S.RESET}")
                
            # ----- OPTIONAL CYCLE CUT-OFF (v1.8.5) -----
            # Cap analysis at a chosen cycle for this dataset. Applied to the
            # loaded data table at the end of Cell 3, so every downstream cell
            # sees only cycles 1..max_cycle with no other code changes. Blank
            # keeps all cycles. Use this to trim post-failure 'bad' data from a
            # faulty cell once its reliable range is known from the full run.
            _cyc_hint = (f" (file reports {meta['cycle_count']} cycles)"
                         if meta.get('cycle_count') else "")
            # CARRIED FORWARD, AND OVERRIDABLE — the same shape as the
            # electrolyte question below. Blank keeps what the previous dataset
            # used; a number changes it; 'all' removes the cap for this dataset
            # only. Blank had to keep meaning something, and "keep the cap" is
            # the answer wanted far more often than "drop it", so 'all' is the
            # word that drops it and it is named in the prompt rather than
            # left to be guessed.
            if _MAX_CYCLE_CARRIED is not None:
                print(f"  Current: {_S.VALUE}cycles 1-{_MAX_CYCLE_CARRIED}"
                      f"{_S.RESET} {_S.FILE}(carried from the previous "
                      f"dataset){_S.RESET}")
                _maxcyc_prompt = (
                    f"Maximum cycle to analyse for this dataset{_cyc_hint} "
                    f"— blank keeps {_MAX_CYCLE_CARRIED}, a number changes "
                    f"it, 'all' removes the cap: ")
            else:
                _maxcyc_prompt = (
                    f"Maximum cycle to analyse for this dataset{_cyc_hint} "
                    f"(leave blank for all cycles): ")
            while True:
                _maxcyc_str = input(_maxcyc_prompt).strip()
                if not _maxcyc_str:
                    # None on the first dataset, which is the old behaviour.
                    max_cycle = _MAX_CYCLE_CARRIED
                    break
                if _maxcyc_str.lower() in ("all", "none"):
                    max_cycle = None
                    break
                try:
                    max_cycle = int(_maxcyc_str)
                    if max_cycle >= 1:
                        break
                    print(f"  {_S.WARN}Enter a cycle number >= 1, "
                          f"'all' for no cap, or leave blank.{_S.RESET}")
                except ValueError:
                    print(f"  {_S.WARN}Enter a whole number, 'all' for no "
                          f"cap, or leave blank.{_S.RESET}")
            # Carry whatever this dataset settled on, including 'all': if the
            # cap is dropped here it stays dropped for the next one, and the
            # prompt goes back to its original form.
            _MAX_CYCLE_CARRIED = max_cycle
            # ----- CELL ASSEMBLY -----
            print(f"\n{_S.SECTION}--- Cell assembly (for methods section) ---{_S.RESET}")
            separator_type = input(
                "Separator type (leave blank for GF/6): "
            ) or "SLS Select GF/6 glass microfibre (SLS4522)"
            while True:
                elyte_vol_input = input(
                    "Electrolyte volume in uL (leave blank for 100): "
                )
                if not elyte_vol_input:
                    electrolyte_volume_uL = 100.0
                    break
                try:
                    electrolyte_volume_uL = float(elyte_vol_input)
                    if electrolyte_volume_uL >= 0:
                        break
                    print(f"  {_S.WARN}Volume must be non-negative.{_S.RESET}")
                except ValueError:
                    print(f"  {_S.WARN}Please enter a number.{_S.RESET}")
            print(f"\n{_S.SECTION}Electrolyte composition:{_S.RESET}")
            if _ELECTROLYTE_COMP_GLOBAL is not None:
                print(f"  Current: {_S.VALUE}{_ELECTROLYTE_COMP_GLOBAL}{_S.RESET} "
                      f"{_S.FILE}(set on first dataset){_S.RESET}")
                _elyte_override = input(
                    "Press Enter to keep, or enter 1-4 to change: "
                ).strip()
                if not _elyte_override:
                    electrolyte_composition = _ELECTROLYTE_COMP_GLOBAL
                else:
                    elyte_choice = _elyte_override
                    if elyte_choice == '1':
                        electrolyte_composition = "1M LiPF6 in EC:DMC 1:1 v/v"
                    elif elyte_choice == '2':
                        electrolyte_composition = "1M NaPF6 in EC:DMC 1:1 v/v"
                    elif elyte_choice == '3':
                        electrolyte_composition = "1M NaPF6 in EC:PC 1:1 v/v"
                    elif elyte_choice == '4':
                        electrolyte_composition = input(
                            "Enter electrolyte composition: "
                        ) or "Not specified"
                    else:
                        print(f"  {_S.WARN}Invalid choice, keeping previous.{_S.RESET}")
                        electrolyte_composition = _ELECTROLYTE_COMP_GLOBAL
                    _ELECTROLYTE_COMP_GLOBAL = electrolyte_composition
            else:
                print(f"  1. 1M LiPF6 in EC:DMC 1:1 v/v")
                print(f"  2. 1M NaPF6 in EC:DMC 1:1 v/v")
                print(f"  3. 1M NaPF6 in EC:PC 1:1 v/v")
                print(f"  4. Other")
                print(f"  {_S.FILE}(leave blank to skip){_S.RESET}")
                while True:
                    elyte_choice = input("Enter 1, 2, 3, 4, or blank: ").strip()
                    if not elyte_choice:
                        electrolyte_composition = "Not specified"
                        break
                    elif elyte_choice == '1':
                        electrolyte_composition = "1M LiPF6 in EC:DMC 1:1 v/v"
                        break
                    elif elyte_choice == '2':
                        electrolyte_composition = "1M NaPF6 in EC:DMC 1:1 v/v"
                        break
                    elif elyte_choice == '3':
                        electrolyte_composition = "1M NaPF6 in EC:PC 1:1 v/v"
                        break
                    elif elyte_choice == '4':
                        electrolyte_composition = input(
                            "Enter electrolyte composition: "
                        ) or "Not specified"
                        break
                    else:
                        print(f"  {_S.WARN}Enter 1, 2, 3, 4, or leave blank.{_S.RESET}")
                _ELECTROLYTE_COMP_GLOBAL = electrolyte_composition
            # ----- COUNTER ELECTRODE -----
            # ASK RATHER THAN ASSUME. `infer_chemistry` returns Unknown for an
            # abbreviation, and the estimate then used to fall through to
            # sodium in silence. One question, only when nothing better is
            # available, and the library's working ion is the default.
            # `battery_chemistry` is not assigned until the parameters are
            # built, well below this; `counter_electrode_metal` does the same
            # inference itself from the composition and the electrolyte, both
            # of which are known by here.
            _ce_metal, _ce_metal_src = counter_electrode_metal(
                composition, electrolyte=electrolyte_composition)
            if _ce_metal_src in ("library", "assumed"):
                _why = ("from the composition library — a lithium material can "
                        "still be cycled in a sodium cell, so confirm it"
                        if _ce_metal_src == "library" else
                        "a guess: neither the electrolyte nor the composition "
                        "names a working ion")
                print(f"  {_S.WARN}Counter electrode metal not established "
                      f"({_why}).{_S.RESET}")
                while True:
                    _m = input(f"Counter electrode metal (Li or Na) "
                               f"[{_ce_metal}]: ").strip().title()
                    if not _m:
                        _ce_metal_src = ("library" if _ce_metal_src == "library"
                                         else "assumed")
                        break
                    if _m in _COUNTER_METAL_DENSITY:
                        _ce_metal, _ce_metal_src = _m, "stated"
                        break
                    print(f"  {_S.WARN}Enter Li or Na, or leave blank for "
                          f"{_ce_metal}.{_S.RESET}")
            while True:
                ce_input = input(
                    "Counter electrode mass in mg "
                    "(leave blank for estimate from geometry): "
                )
                if not ce_input:
                    counter_electrode_mass_mg = None
                    counter_electrode_measured = False
                    break
                try:
                    counter_electrode_mass_mg = float(ce_input)
                    if counter_electrode_mass_mg > 0:
                        counter_electrode_measured = True
                        break
                    print(f"  {_S.WARN}Mass must be positive.{_S.RESET}")
                except ValueError:
                    print(f"  {_S.WARN}Please enter a number.{_S.RESET}")
            # ----- OUTPUT PREFERENCES -----
            print(f"\n{_S.SECTION}--- Output preferences ---{_S.RESET}")
            print(f"Recommended colour palettes "
                  f"{_S.FILE}(all are valid matplotlib colormaps){_S.RESET}:")
            for i, p in enumerate(recommended_palettes):
                print(f"  {i+1:>2}. {p}")
            print(f"\n  {_S.WARN}Note: seaborn-only names (bright, deep, dark, hls, husl,")
            print(f"  Greys_d, etc.) are not valid here — they will crash at Cell 5.{_S.RESET}")
            while True:
                palette_choice = input(
                    f"\nEnter number or colormap name "
                    f"(leave blank for '{default_palette}'): "
                ).strip()
                if not palette_choice:
                    colour_palette = default_palette
                    print(f"  {_S.CONFIRM}Colour palette: {colour_palette} ✓{_S.RESET}")
                    break
                try:
                    idx = int(palette_choice) - 1
                    if 0 <= idx < len(recommended_palettes):
                        colour_palette = recommended_palettes[idx]
                        print(f"  {_S.CONFIRM}Colour palette: {colour_palette} ✓{_S.RESET}")
                        break
                    print(f"  {_S.WARN}Enter a number between 1 and "
                          f"{len(recommended_palettes)}.{_S.RESET}")
                    continue
                except ValueError:
                    pass
                if _validate_palette(palette_choice):
                    colour_palette = palette_choice
                    print(f"  {_S.CONFIRM}Colour palette: {colour_palette} ✓{_S.RESET}")
                    break
                else:
                    _suggestions = sorted([
                        c for c in _VALID_CMAPS
                        if palette_choice.lower().replace('_d', '').replace('_r', '')
                        in c.lower()
                    ])[:6]
                    print(f"  {_S.WARN}✗ '{palette_choice}' is not a valid matplotlib colormap.{_S.RESET}")
                    if _suggestions:
                        print(f"    Did you mean: {_S.VALUE}{', '.join(_suggestions)}{_S.RESET}?")
                    else:
                        print(f"    Check spelling, or choose a number from the list above.")
            if _FILE_FORMAT_GLOBAL is None:
                while True:
                    file_format = input(
                        "Plot file format (png or tiff, leave blank for png): "
                    ).lower()
                    if file_format in ["png", "tiff", ""]:
                        file_format = file_format or "png"
                        _FILE_FORMAT_GLOBAL = file_format
                        break
                    print(f"  {_S.WARN}Enter 'png' or 'tiff'.{_S.RESET}")
            else:
                file_format = _FILE_FORMAT_GLOBAL
                print(f"File format: {_S.VALUE}{file_format}{_S.RESET} "
                      f"{_S.FILE}(set on first dataset){_S.RESET}")
            # =====================================================================
            # C. COMPUTED VALUES
            # =====================================================================
            # blend_values already validated during input above
            active_fraction = blend_values[0] / sum(blend_values)
            electrode_radius_cm = electrode_diameter_mm / 2 / 10
            electrode_area_cm2 = np.pi * electrode_radius_cm**2
            active_material_mass_g = active_material_mass_mg / 1000
            total_electrode_coating_mg = active_material_mass_mg / active_fraction
            active_loading_mg_cm2 = active_material_mass_mg / electrode_area_cm2
            total_loading_mg_cm2 = total_electrode_coating_mg / electrode_area_cm2
            theoretical_capacity_mAh = (active_material_mass_g *
                                        theoretical_capacity_mAh_g)
            theoretical_charge_current_mA = (charge_rate_c *
                                             theoretical_capacity_mAh)
            theoretical_charge_rate_mAh_g = (theoretical_charge_current_mA /
                                             active_material_mass_g)

            # Infer battery chemistry from electrolyte and composition (v1.7.2)
            # Used by Cell 6c for areal capacity benchmarks and Cell 14 NAMCC
            # for counter electrode metal selection.
            battery_chemistry = infer_chemistry(composition,
                                                electrolyte_composition)
            # =====================================================================
            # BUILD PARAMETERS DICTIONARY
            # =====================================================================
            # The metal, whether or not the counter electrode was weighed.
            # "97.6 mg of sodium" is a different claim from "97.6 mg", and
            # the declarative path already recorded it.
            _ce_est_mg, _ce_metal = _estimate_counter_electrode_mass_mg(
                battery_chemistry=battery_chemistry, metal=_ce_metal)

            sample_parameters = {
                "composition": composition,
                "cell_id": cell_id,
                "blend": blend,
                "active_material_mass_mg": active_material_mass_mg,
                "theoretical_capacity_mAh_g": theoretical_capacity_mAh_g,
                "theoretical_capacity_source": _theo_provenance,
                "theoretical_capacity_library_formula": _theo_library_formula,
                "electrode_type": electrode_type,
                # Read by cycling.py and plots.py to relabel the half-cycles.
                # Its ABSENCE here — the declarative path has always set it —
                # meant an operator who answered "negative" still got figures
                # and CSV columns labelled Charge/Discharge.
                "anode_labels_swapped": (str(electrode_type).lower()
                                         == "negative"),
                "electrode_diameter_mm": electrode_diameter_mm,
                "electrode_area_cm2": electrode_area_cm2,
                "active_fraction": active_fraction,
                "total_electrode_coating_mg": total_electrode_coating_mg,
                "active_loading_mg_cm2": active_loading_mg_cm2,
                "total_loading_mg_cm2": total_loading_mg_cm2,
                "power_energy_analysis": power_energy_analysis,
                "charge_rate_c": charge_rate_c,
                "key_cycles": key_cycles,
                "max_cycle": max_cycle,
                "theoretical_capacity_mAh": theoretical_capacity_mAh,
                "theoretical_charge_current_mA": theoretical_charge_current_mA,
                "theoretical_charge_rate_mAh_g": theoretical_charge_rate_mAh_g,
                "voltage_upper_V": voltage_upper_V,
                "voltage_lower_V": voltage_lower_V,
                "voltage_window_source": voltage_window_source,
                "voltage_limit_upper_V_from_file": _v_file_hi,
                "voltage_limit_lower_V_from_file": _v_file_lo,
                "separator_type": separator_type,
                "electrolyte_volume_uL": electrolyte_volume_uL,
                "electrolyte_composition": electrolyte_composition,
                "counter_electrode_mass_mg": (counter_electrode_mass_mg
                                              if counter_electrode_measured
                                              else _ce_est_mg),
                "counter_electrode_metal": _ce_metal,
                "counter_electrode_metal_source": _ce_metal_src,
                "counter_electrode_measured": counter_electrode_measured,
                "battery_chemistry": battery_chemistry,
                "barcode": meta.get('barcode'),
                "test_start_date": meta.get('test_start_date'),
                "charge_current_A_from_file": meta.get('charge_current_A'),
                "discharge_current_A_from_file": meta.get('discharge_current_A'),
                "colour_palette": colour_palette,
                "file_format": file_format,
                # The analysis knobs, from the one table. See
                # ANALYSIS_DEFAULTS for why this is not a literal block.
                **ANALYSIS_DEFAULTS,
            }
            # =====================================================================
            # REVIEW
            # =====================================================================
            mass_src = f" {_S.FILE}[from file]{_S.RESET}" if meta.get('active_material_mass_mg') else ""
            # SAY WHERE THE NUMBER CAME FROM, not where a number was
            # available. Marking a measured rate "[from file]" credits the
            # header for an answer the current gave.
            if _proto.get('is_variable'):
                rate_src = (f" {_S.FILE}[reference block of "
                            f"{_proto['n_blocks']}; this run used "
                            f"{_proto['label']}]{_S.RESET}")
            elif _rate_meas:
                rate_src = f" {_S.FILE}[measured from the current]{_S.RESET}"
            elif meta.get('c_rate'):
                rate_src = f" {_S.FILE}[from file]{_S.RESET}"
            else:
                rate_src = ""
            print(f"\n{_S.HEADER}{'='*60}")
            print(f"  Review parameters for: {name}")
            print(f"{'='*60}{_S.RESET}")
            print(f"\n{_S.SECTION}  Material:{_S.RESET}")
            print(f"    Composition:          {_S.VALUE}{composition}{_S.RESET}")
            print(f"    Cell identifier:      {_S.VALUE}"
                  f"{cell_id or '(not recorded)'}{_S.RESET}")
            print(f"    Battery chemistry:    {_S.VALUE}{battery_chemistry}{_S.RESET}")
            # WHAT THE FIGURES WILL CALL THE HALF-CYCLES, said here where the
            # answer can still be changed. A negative electrode names the
            # process rather than the direction, and until 1.9.0.75 an
            # unestablished ion was silently printed as lithium — an NTO
            # anode in a sodium half-cell was captioned "Delithiation" on
            # every waterfall. Neutral labels are no longer a silent
            # outcome: if they are what this dataset will get, it says so and
            # says what to do about it.
            if str(electrode_type).lower().startswith("neg"):
                _ci, _co = half_cycle_labels(
                    dict(anode_labels_swapped=True,
                         counter_electrode_metal=_ce_metal,
                         counter_electrode_metal_source=_ce_metal_src,
                         battery_chemistry=battery_chemistry))
                if _ci == "Charge":
                    print(f"    Half-cycle labels:    {_S.WARN}Charge / "
                          f"Discharge{_S.RESET}")
                    print(f"      {_S.WARN}The working ion is not "
                          f"established, so the figures will not name the "
                          f"process. Give the electrolyte, or a formula "
                          f"rather than an abbreviation, or state the "
                          f"counter electrode metal.{_S.RESET}")
                else:
                    print(f"    Half-cycle labels:    {_S.VALUE}{_ci} / "
                          f"{_co}{_S.RESET} (working ion "
                          f"{_ce_metal}, {_ce_metal_src})")
            print(f"    Theoretical capacity: {_S.VALUE}{theoretical_capacity_mAh_g:.1f} mAh/g{_S.RESET}")
            print(f"    Electrode type:       {_S.VALUE}{electrode_type}{_S.RESET}")
            print(f"\n{_S.SECTION}  Electrode:{_S.RESET}")
            print(f"    Active mass:          {_S.VALUE}{active_material_mass_mg:.2f} mg{_S.RESET}{mass_src}")
            print(f"    Blend:                {_S.VALUE}{blend} ({active_fraction*100:.1f}% active){_S.RESET}")
            print(f"    Punch diameter:       {_S.VALUE}{electrode_diameter_mm:.1f} mm "
                  f"(area = {electrode_area_cm2:.3f} cm2){_S.RESET}")
            print(f"    Active loading:       {_S.VALUE}{active_loading_mg_cm2:.2f} mg/cm2{_S.RESET}")
            print(f"    Total loading:        {_S.VALUE}{total_loading_mg_cm2:.2f} mg/cm2{_S.RESET}")
            print(f"\n{_S.SECTION}  Cycling:{_S.RESET}")
            print(f"    C-rate:               {_S.VALUE}{charge_rate_c:.2f} C{_S.RESET}{rate_src}")
            print(f"    Theoretical current:  {_S.VALUE}{theoretical_charge_current_mA:.3f} mA{_S.RESET}")
            # The window BEING STORED, not the file's. The file's figures
            # are channel safety limits — an LTO cell cycled 1.2-2.5 V
            # reports 0-3.25 V — and the code above deliberately prefers the
            # measured window. Showing the file's under "are these correct?"
            # asked the operator to confirm a number that was not being used.
            if voltage_lower_V is not None and voltage_upper_V is not None:
                print(f"    Voltage window:       {_S.VALUE}"
                      f"{voltage_lower_V:.3f}--{voltage_upper_V:.3f} V"
                      f"{_S.RESET} {_S.FILE}[{voltage_window_source}]"
                      f"{_S.RESET}")
            print(f"    Key cycles:           {_S.VALUE}{key_cycles}{_S.RESET}")
            if max_cycle is not None:
                print(f"    Cycle cut-off:        {_S.VALUE}cycles 1-{max_cycle} "
                      f"(later cycles removed){_S.RESET}")
            else:
                print(f"    Cycle cut-off:        {_S.VALUE}none "
                      f"(all cycles){_S.RESET}")
            if meta.get('charge_current_A') is not None:
                file_current_mA = meta['charge_current_A'] * 1000
                calc_current_mA = theoretical_charge_current_mA
                pct_diff = (abs(file_current_mA - calc_current_mA) /
                            max(file_current_mA, 1e-9) * 100)
                if pct_diff > 5:
                    print(f"    {_S.WARN}WARNING: Current mismatch: calculated "
                          f"{calc_current_mA:.4f} mA vs file "
                          f"{file_current_mA:.4f} mA "
                          f"({pct_diff:.0f}% difference)")
                    print(f"      Check active mass and theoretical capacity.{_S.RESET}")
                else:
                    print(f"    {_S.CONFIRM}Current matches file: "
                          f"{file_current_mA:.4f} mA ✓{_S.RESET}")
            print(f"\n{_S.SECTION}  Cell assembly:{_S.RESET}")
            print(f"    Separator:            {_S.VALUE}{separator_type}{_S.RESET}")
            if electrolyte_volume_uL is not None:
                print(f"    Electrolyte:          {_S.VALUE}{electrolyte_composition} "
                      f"({electrolyte_volume_uL:.0f} uL){_S.RESET}")
            else:
                print(f"    Electrolyte:          {_S.VALUE}{electrolyte_composition}{_S.RESET}")
            if counter_electrode_measured:
                print(f"    Counter electrode:    {_S.VALUE}"
                      f"{counter_electrode_mass_mg:.1f} mg "
                      f"(weighed){_S.RESET}")
            else:
                # THE SAME METAL THE PARAMETERS RECORD. This line used to
                # recompute the estimate without it, so a cell whose counter
                # was resolved to lithium was still reviewed as "~98 mg (Na
                # metal)" — the sodium figure — while the stored parameter
                # said Li. Two call sites, one of them not told the answer.
                _est_ce, _est_metal = _estimate_counter_electrode_mass_mg(
                    battery_chemistry=battery_chemistry, metal=_ce_metal)
                _src_word = {"stated": "you said so",
                             "chemistry": "from the electrolyte / formula",
                             "library": "from the composition library",
                             "assumed": "ASSUMED"}.get(_ce_metal_src, "")
                print(f"    Counter electrode:    {_S.VALUE}"
                      f"~{_est_ce:.0f} mg ({_est_metal} metal, "
                      f"{_DEFAULT_COUNTER_DIAMETER_MM:.0f} mm x "
                      f"{_DEFAULT_COUNTER_THICKNESS_MM:.1f} mm, "
                      f"estimated){_S.RESET}"
                      + (f" {_S.FILE}[{_src_word}]{_S.RESET}"
                         if _src_word else ""))
            print(f"\n{_S.SECTION}  Analysis:{_S.RESET}")
            pe_str = f"{_S.CONFIRM}ON{_S.RESET}" if power_energy_analysis else f"{_S.WARN}SKIP{_S.RESET}"
            pk_str = f"{_S.CONFIRM}ON{_S.RESET}" if RUN_PEAK_FITTING else f"{_S.WARN}SKIP{_S.RESET}"
            print(f"    Power/energy (Cell 6c):           {pe_str}")
            print(f"    dQ/dV curves (Cell 7):            {_S.CONFIRM}ON{_S.RESET}")
            print(f"    dQ/dV peak fitting (Cells 8-12):  {pk_str}")
            # dQ/dV preprocessing summary
            _sw = sample_parameters['smoothing_window']
            _sr = sample_parameters['spike_removal']
            _st = sample_parameters['spike_threshold_multiplier']
            _auto_count = sum(1 for v in [_sw, _sr, _st] if v == 'auto')
            if _auto_count == 3:
                print(f"    dQ/dV preprocessing:               "
                      f"{_S.CONFIRM}AUTO{_S.RESET} "
                      f"(adapts to material type)")
            elif _auto_count > 0:
                print(f"    dQ/dV preprocessing:               "
                      f"{_S.VALUE}MIXED{_S.RESET} "
                      f"({_auto_count} auto, {3-_auto_count} manual)")
            else:
                print(f"    dQ/dV preprocessing:               "
                      f"{_S.VALUE}MANUAL{_S.RESET} "
                      f"(window={_sw}, spike={_sr}, "
                      f"threshold={_st})")
            print(f"\n{_S.SECTION}  Output:{_S.RESET}")
            print(f"    Palette:              {_S.VALUE}{colour_palette}{_S.RESET}")
            print(f"    File format:          {_S.VALUE}{file_format}{_S.RESET}")
            print(f"\n{_S.HEADER}{'='*60}{_S.RESET}")
            while True:
                confirmation = input(
                    "\nAre these parameters correct? (yes/no): "
                ).lower()
                if confirmation in ["yes", "y"]:
                    confirmed = True
                    _last_composition = composition
                    _last_theoretical_capacity = theoretical_capacity_mAh_g
                    print(f"{_S.CONFIRM}  ✓ Parameters confirmed.{_S.RESET}")
                    break
                elif confirmation in ["no", "n"]:
                    print(f"\n{_S.WARN}  Re-entering parameters...{_S.RESET}\n")
                    break
                else:
                    print(f"  {_S.WARN}Enter 'yes' or 'no'.{_S.RESET}")
        user_parameters[name] = sample_parameters

    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================
    print(f"\n{_S.HEADER}{'='*60}")
    print(f"  Parameters set for {len(user_parameters)} dataset(s)")
    print(f"{'='*60}{_S.RESET}")
    for name, params in user_parameters.items():
        display = _get_display_name(name, params, user_parameters)
        mass    = params['active_material_mass_mg']
        loading = params['active_loading_mg_cm2']
        rate    = params['charge_rate_c']
        pe_flag = (f"{_S.CONFIRM}ON{_S.RESET}"
                   if params.get('power_energy_analysis', True)
                   else f"{_S.WARN}SKIP{_S.RESET}")
        print(f"  {_S.VALUE}{display}{_S.RESET}")
        print(f"    {mass:.2f} mg active | "
              f"{loading:.2f} mg/cm2 | "
              f"{rate:.2f} C | "
              f"P/E: {pe_flag}")

    print(f"\n{_S.SECTION}  Pipeline-level:{_S.RESET}")
    pk_flag = (f"{_S.CONFIRM}ON{_S.RESET}" if RUN_PEAK_FITTING
               else f"{_S.WARN}SKIP{_S.RESET}")
    print(f"    dQ/dV peak fitting (Cells 8-12):  {pk_flag}")
    if not RUN_PEAK_FITTING:
        print(f"    {_S.FILE}(Cell 7 still runs. To enable fitting later,")
        print(f"     set RUN_PEAK_FITTING = True above and re-run Cell 3.){_S.RESET}")

    print(f"{_S.HEADER}{'='*60}{_S.RESET}")
    print(f"\n{_S.SECTION}  Helper functions available for all downstream cells:{_S.RESET}")
    print(f"    _get_display_name(name, params, user_parameters)")
    print(f"    _force_integer_cycles(ax)")
    print(f"    _skip_module(module_name)  {_S.FILE}— used by Modules 3-8 guards{_S.RESET}")

    # =============================================================================
    # CYCLE TRUNCATION  (v1.8.5)
    # =============================================================================
    # If a maximum analysis cycle was set for any dataset above, trim the loaded
    # data table now, once, here, so every downstream cell sees only the
    # retained cycles. Only Cell 2 reads the raw files; every other cell works
    # off electrochemical_data, so trimming it here propagates everywhere.
    # Destructive: to restore full data, re-run the data-loading cell (Cell 2).
    # =============================================================================
    # The cut-off is RECORDED here and applied by the caller to the Dataset
    # itself (`io.truncate_cycles`). This function used to truncate a copy of
    # `electrochemical_data`, which the notebook then rebuilt from the
    # untruncated Dataset two lines later — so the answer was collected,
    # applied, discarded, and reported as done.
    _all_cuts = {n: user_parameters.get(n, {}).get('max_cycle')
                 for n in electrochemical_data}
    _cuts = {n: c for n, c in _all_cuts.items() if c}
    if _cuts:
        print(f"\n{_S.HEADER}{'='*60}")
        print(f"  Cycle cut-offs recorded")
        print(f"{'='*60}{_S.RESET}")
        for n, c in _cuts.items():
            print(f"  {_S.VALUE}"
                  f"{_get_display_name(n, user_parameters[n], user_parameters)}"
                  f"{_S.RESET}: cycles above {c} will be excluded")
        # DO THEY AGREE? Datasets analysed together are compared with each
        # other — the comparative capacity, retention and fade figures put
        # them on one axis. Different cut-offs there mean those figures span
        # different cycle ranges, which is a real difference in what is being
        # compared and is invisible once the figure is drawn. Said out loud
        # here rather than left to be noticed.
        if len(_all_cuts) > 1 and len(set(_all_cuts.values())) > 1:
            _uncapped = [n for n, c in _all_cuts.items() if not c]
            print(f"\n  {_S.WARN}The cut-offs are not the same for every "
                  f"dataset.{_S.RESET}")
            if _uncapped:
                print(f"  {_S.FILE}No cap on: "
                      f"{', '.join(_get_display_name(n, user_parameters[n], user_parameters) for n in _uncapped)}"
                      f"{_S.RESET}")
            print(f"  Cells 6b and 6c draw these on shared axes, so those "
                  f"figures will compare different cycle ranges. Deliberate "
                  f"for a cell that died early; otherwise re-run this cell.")
        print(f"{_S.HEADER}{'='*60}{_S.RESET}")
    return user_parameters, RUN_PEAK_FITTING


# =============================================================================
# The declarative path
# =============================================================================
# `collect_parameters` is the interactive cell restored, and it is the one a
# researcher should use: seeing "[from file: 4.72]" against a mass is how you
# know which numbers you are responsible for. But a scheduled run, a test, and
# an application front end all need the same record built without a keyboard,
# and faking keystrokes into a 630-line prompt sequence is not a way to get it.
# So the declarative path is its own function, producing the same 53-field
# record, with the same derivations, from a dict.

# What a dict-mode run must state for itself. `composition` and
# `theoretical_capacity_mAh_g` are properties of the material and the cycler
# file cannot know them. `electrode_type` decides the SIGN of dQ/dV and which
# half-cycle is called "Discharge"; defaulting it to "Positive" meant a run on
# an anode silently produced mirror-image figures that looked entirely
# plausible, so it is asked for rather than assumed.
_REQUIRED = ("composition", "theoretical_capacity_mAh_g", "electrode_type")

_ELECTRODE_TYPES = ("Positive", "Negative")


def parameters_from_dict(specs, cycler_metadata=None, verbose=True):
    """
    Build the same parameter record as `collect_parameters`, without prompting.

    `specs` is `{dataset_name: {field: value}}`. Anything omitted falls back to
    the cycler metadata, then to the documented default. Only `composition` and
    `theoretical_capacity_mAh_g` have no defensible default and must be given —
    a theoretical capacity is a property of the material, not of the file.

    Every derived quantity is computed exactly as the interactive path computes
    it: electrode area from the diameter, active fraction from the blend,
    loadings from the mass and area, and the theoretical current from the
    capacity and the C-rate.
    """
    cycler_metadata = cycler_metadata or {}
    out = {}
    for name, spec in specs.items():
        spec = dict(spec or {})
        meta = dict(cycler_metadata.get(name) or {})
        # THE LIBRARY IS A DEFAULT HERE TOO. A declarative run that names a
        # material the library knows should not have to restate its
        # theoretical capacity, and should not be able to restate it WRONG
        # while the interactive path gets it right. `composition` is still
        # required — it is what the lookup is made from.
        _lib_formula = None
        if spec.get("theoretical_capacity_mAh_g") in (None, ""):
            _lib_cap, _lib_formula, _lib_note = \
                lookup_specific_capacity(spec.get("composition"))
            if _lib_cap is not None:
                spec["theoretical_capacity_mAh_g"] = _lib_cap
                spec.setdefault("theoretical_capacity_source", "library")
                if verbose:
                    print(f"  {name}: theoretical capacity {_lib_cap} mAh/g "
                          f"from the library ({_lib_formula}, {_lib_note})")
        missing = [k for k in _REQUIRED if spec.get(k) in (None, "")]
        if missing:
            raise ValueError(f"{name}: {', '.join(missing)} must be given — "
                             f"the cycler file cannot supply "
                             f"{'them' if len(missing) > 1 else 'it'}.")
        # A theoretical capacity of NaN is not a missing value the guards can
        # see: `x > 1.2 * nan` is False, so the over-theoretical check simply
        # stopped firing and said nothing. Rejected here instead.
        _tc = spec.get("theoretical_capacity_mAh_g")
        try:
            _tc = float(_tc)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: theoretical_capacity_mAh_g must be a "
                             f"number, not {spec.get('theoretical_capacity_mAh_g')!r}")
        if not np.isfinite(_tc) or _tc <= 0:
            raise ValueError(f"{name}: theoretical_capacity_mAh_g is {_tc} — "
                             f"it must be a positive number, because the "
                             f"over-theoretical capacity check compares "
                             f"against it and a NaN silently disables it.")
        spec["theoretical_capacity_mAh_g"] = _tc
        if str(spec["electrode_type"]).strip().title() not in _ELECTRODE_TYPES:
            raise ValueError(
                f"{name}: electrode_type is {spec['electrode_type']!r}; it "
                f"must be one of {', '.join(_ELECTRODE_TYPES)}.")
        spec["electrode_type"] = str(spec["electrode_type"]).strip().title()

        def g(key, default=None, meta_key=None):
            v = spec.get(key)
            if v not in (None, ""):
                return v
            if meta_key and meta.get(meta_key) not in (None, ""):
                return meta[meta_key]
            return default

        mass = float(g("active_material_mass_mg", 0.0,
                       "active_material_mass_mg") or 0.0)
        diameter = float(g("electrode_diameter_mm", default_electrode_diameter_mm))
        area = np.pi * (diameter / 20.0) ** 2          # mm -> cm, then pi r^2
        blend = str(g("blend", "80/10/10"))
        try:
            parts = [float(x) for x in blend.split("/")]
            frac = parts[0] / sum(parts) if sum(parts) else 0.8
        except (ValueError, ZeroDivisionError):
            frac = 0.8
        coating = mass / frac if frac else np.nan
        theo_g = float(spec["theoretical_capacity_mAh_g"])
        theo_mAh = theo_g * mass / 1000.0
        c_rate = g("charge_rate_c", None, "c_rate")
        try:
            c_rate_f = float(c_rate)
        except (TypeError, ValueError):
            c_rate_f = np.nan

        _chem = g("battery_chemistry",
                  infer_chemistry(spec.get("composition"),
                                  g("electrolyte_composition", "")))
        # Same order as the interactive path, minus the question it cannot
        # ask. `assumed` is recorded and printed as such rather than passing
        # for an inference.
        _ce_metal, _ce_metal_src = counter_electrode_metal(
            spec.get("composition"), stated=g("counter_electrode_metal", None),
            battery_chemistry=_chem, electrolyte=g("electrolyte_composition", ""))
        _counter = _estimate_counter_electrode_mass_mg(
            battery_chemistry=_chem, metal=_ce_metal)
        _pal = str(g("colour_palette", default_palette))
        if not isinstance(_counter, tuple):
            _counter = (_counter, None)

        rec = {
            "composition": spec["composition"],
            "cell_id": g("cell_id", _suggest_cell_id(name, list(specs))),
            "blend": blend,
            "active_material_mass_mg": mass,
            "theoretical_capacity_mAh_g": theo_g,
            "theoretical_capacity_source": g("theoretical_capacity_source",
                                             "given"),
            "theoretical_capacity_library_formula": _lib_formula,
            "electrode_type": g("electrode_type", "Positive"),
            "electrode_diameter_mm": diameter,
            "electrode_area_cm2": area,
            "active_fraction": frac,
            "total_electrode_coating_mg": coating,
            "active_loading_mg_cm2": mass / area if area else np.nan,
            "total_loading_mg_cm2": coating / area if area else np.nan,
            "power_energy_analysis": bool(g("power_energy_analysis", True)),
            # The COERCED rate. Storing the raw answer let a string or None
            # through, which printed as "None C" in the report and the
            # dataset_info file while the derived currents used the float.
            "charge_rate_c": (float(c_rate_f)
                              if np.isfinite(c_rate_f) else None),
            "key_cycles": list(g("key_cycles", default_key_cycles)),
            "max_cycle": g("max_cycle"),
            "theoretical_capacity_mAh": theo_mAh,
            "theoretical_charge_current_mA": theo_mAh * c_rate_f,
            "theoretical_charge_rate_mAh_g": theo_g * c_rate_f,
            "voltage_upper_V": g("voltage_upper_V", None, "voltage_upper_V"),
            "voltage_lower_V": g("voltage_lower_V", None, "voltage_lower_V"),
            "voltage_window_source": ("file" if meta.get("voltage_upper_V")
                                      and "voltage_upper_V" not in spec
                                      else "operator"),
            "voltage_limit_upper_V_from_file": meta.get("voltage_upper_V"),
            "voltage_limit_lower_V_from_file": meta.get("voltage_lower_V"),
            "separator_type": g("separator_type", "glass fibre"),
            "electrolyte_volume_uL": g("electrolyte_volume_uL", 100.0),
            "electrolyte_composition": g("electrolyte_composition", ""),
            # The estimator returns (mass, metal); 1.8.6 unpacked it and kept
            # both, because "97.6 mg of sodium" is a very different claim from
            # "97.6 mg" on its own.
            "counter_electrode_mass_mg": g("counter_electrode_mass_mg",
                                           _counter[0]),
            "counter_electrode_metal": _counter[1],
            "counter_electrode_metal_source": _ce_metal_src,
            "counter_electrode_measured": "counter_electrode_mass_mg" in spec,
            "battery_chemistry": _chem,
            "barcode": meta.get("barcode"),
            "test_start_date": meta.get("test_start_date"),
            "charge_current_A_from_file": meta.get("charge_current_A"),
            "discharge_current_A_from_file": meta.get("discharge_current_A"),
            # `_validate_palette` returns a BOOLEAN — it answers "is this a
            # real matplotlib colormap", not "give me one". Storing its result
            # put True in the palette field and seaborn tried to iterate it.
            "colour_palette": (_pal if _validate_palette(_pal)
                               else default_palette),
            "file_format": g("file_format", "png"),
            # The analysis knobs, from the one table — the SAME defaults the
            # interactive path uses, taken from the same place so they cannot
            # drift apart again. 'auto' hands the choice to the code that
            # reads this dataset's own curve; give a number to override it.
            **{k: g(k, v) for k, v in ANALYSIS_DEFAULTS.items()},
            "anode_labels_swapped": str(g("electrode_type", "Positive")
                                        ).lower() == "negative",
        }
        out[name] = rec
        if verbose:
            src = "file" if meta.get("active_material_mass_mg") else "given"
            print(f"  {name}: {rec['composition']} cell {rec['cell_id']}, "
                  f"{mass:.2f} mg [{src}], {rec['electrode_type'].lower()} "
                  f"electrode, {rec['active_loading_mg_cm2']:.2f} mg/cm2")
    return out


def detection_overrides(params):
    """
    The DETECTION fields a user set explicitly, to merge over the profile's.

    'auto' means "let the profile decide" and is dropped; a value is an
    instruction and is kept. The defaults come from ANALYSIS_DEFAULTS — the
    same table the record was built from — so a value equal to one of them is
    not an operator choice.
    """
    return _explicit(params, DETECTION_KEYS)


def analysis_overrides(params):
    """
    The ANALYSIS knobs a user set explicitly: reference cycle and tracking
    tolerance. Same contract as `detection_overrides`.

    These two are kept, and the four that used to sit beside them
    (`fit_centre_tolerance_mV`, `fit_sigma_min_mV`, `fit_sigma_max_mV`,
    `baseline_degree`) are GONE, because they were of a different kind.
    A sigma bound cannot serve LTO and a layered oxide at once, so it is
    derived from the measured profile class — and the record was asserting
    200 mV while the sharp profile fitted at 30. A record field that
    disagrees with what ran is worse than an absent one: it reads as
    provenance. The manifest records what ran, never what was offered.

    The reference cycle and the tracking tolerance are the opposite case:
    genuine judgements the operator can make better than the code. 80 mV
    varies with peak spacing, and published ICA practice is unambiguous that
    the reference should be a POST-FORMATION cycle, which only the operator
    knows the length of.
    """
    return _explicit(params, ANALYSIS_KEYS)


def preprocess_overrides(params):
    """
    The dQ/dV preprocessing fields a user set explicitly, as a dict to merge
    over `signal.auto_preprocess_params`.

    'auto' means "let the profile decide" and is dropped; anything else is an
    instruction and is kept. This is how the smoothing, spike and rebin
    controls from 1.8.6's Cell 3 reach the preprocessing again.
    """
    # Compared against ANALYSIS_DEFAULTS — the values BOTH record builders
    # write when nobody asked for anything — so an override means the OPERATOR
    # chose it, not that a default happened to be a number rather than 'auto'.
    return _explicit(params, PREPROCESS_KEYS)


# 1.8.7 opened a native file dialog. It is the right interface on the machine
# this actually runs on — a Windows desktop with the data on a mapped drive —
# because the operator can see the folder, sort by date, and multi-select the
# three files of a triplicate without typing a path. Set False to skip the
# dialog and always use the typed list below.
USE_FILE_DIALOG = True


def pick_files_dialog(initialdir=None, title="Select one or more cycler files"):
    """
    1.8.7's Tkinter file picker. Returns a list of paths, or None.

    Returns None — not an empty list — when no dialog could be shown at all
    (no display, no tkinter, a headless kernel), so the caller can tell
    "there is no dialog here" from "the operator pressed Cancel".

    The `-topmost` / `update()` pair is not decoration: without it the dialog
    opens BEHIND the Jupyter window on Windows and the run appears to hang.
    """
    if not USE_FILE_DIALOG:
        return None
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        paths = filedialog.askopenfilenames(
            title=title,
            initialdir=initialdir if initialdir and os.path.isdir(initialdir)
            else None,
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")])
    except Exception:
        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
    return [os.path.abspath(p) for p in paths]


def select_files(candidates, *, initialdir=None, verbose=True, headless=False,
                 answer=None, use_dialog=None):
    """
    Ask which files to load: a file dialog where there is one, a list where not.

    `candidates` is what the folder holds, used for the typed fallback and as
    the answer when neither route can ask. The dialog is not restricted to
    them — the whole point of a file browser is to go somewhere else.
    """
    want_dialog = USE_FILE_DIALOG if use_dialog is None else use_dialog
    if want_dialog and not headless and answer is None:
        picked = pick_files_dialog(initialdir)
        if picked is not None:
            if picked:
                if verbose:
                    print(f"  {len(picked)} file(s) chosen from the dialog:")
                    for p in picked:
                        print(f"    {os.path.basename(p)}")
                return picked
            # An explicit Cancel. Fall through to the typed list rather than
            # loading everything, which is what Cancel plainly does not mean.
            if verbose:
                print("  File dialog cancelled — choose from the folder "
                      "listing instead.")
    return choose_files(candidates, verbose=verbose, headless=headless,
                        answer=answer)


def choose_files(paths, verbose=True, headless=False, answer=None):
    """
    Show what is in the folder and ask which files to load.

    Ratatosk used to take every match, which is wrong the moment a folder holds
    more than one experiment: eight files in one directory can be three
    chemistries, and loading all of them puts an LTO anode and an NMC cathode
    on the same axes.

    Accepts a comma-separated list, ranges (`1-3`), `all`, or a blank line
    meaning all. Groups the listing by the leading token of each filename, so
    a triplicate reads as a triplicate.
    """
    if not paths:
        return []
    groups = {}
    for i, p in enumerate(paths, 1):
        base = os.path.splitext(os.path.basename(p))[0]
        # The chemistry is usually the second underscore- or hyphen-delimited
        # token; falling back to the first keeps unusual names together.
        toks = re.split(r"[_-]", base)
        key = toks[1] if len(toks) > 1 else toks[0]
        groups.setdefault(key, []).append((i, base))

    if verbose:
        print(f"{len(paths)} file(s) found:\n")
        for key, items in groups.items():
            print(f"  {key}")
            for i, base in items:
                print(f"    {i:>2}. {base}")
        print()

    if headless or answer is not None:
        raw = "" if answer is None else str(answer)
    else:
        raw = _builtins.input(
            "Which files? (numbers, ranges like 1-3, or blank for all): "
        ).strip()

    if not raw or raw.lower() in ("all", "a", "*"):
        if verbose:
            print(f"  loading all {len(paths)}")
        return list(paths)

    picked = []
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            try:
                lo, hi = (int(x) for x in part.split("-", 1))
            except ValueError:
                continue
            picked.extend(range(lo, hi + 1))
        else:
            try:
                picked.append(int(part))
            except ValueError:
                continue
    out = [paths[i - 1] for i in sorted(set(picked))
           if 1 <= i <= len(paths)]
    if not out:
        if verbose:
            print("  nothing recognised in that answer — loading all")
        return list(paths)
    if verbose:
        print(f"  loading {len(out)} of {len(paths)}:")
        for p in out:
            print(f"    {os.path.splitext(os.path.basename(p))[0]}")
    return out


def describe_defaults(metadata):
    """One line per dataset saying what the file already told us."""
    out = []
    for name, meta in metadata.items():
        bits = []
        if meta.get("active_material_mass_mg"):
            bits.append(f"{meta['active_material_mass_mg']:.2f} mg")
        if meta.get("c_rate"):
            bits.append(f"{meta['c_rate']} C")
        if meta.get("voltage_lower_V") and meta.get("voltage_upper_V"):
            bits.append(f"{meta['voltage_lower_V']}-{meta['voltage_upper_V']} V")
        if meta.get("cycle_count"):
            bits.append(f"{meta['cycle_count']} cycles planned")
        out.append(f"  {name}: " + (", ".join(bits) if bits
                                    else "no 'test' sheet — nothing pre-filled"))
    return "\n".join(out)


# =============================================================================
# CELL 2'S ANSWERS, CHECKED BEFORE ANYTHING ACTS ON THEM
# =============================================================================
# =============================================================================
# WHERE THE DATA IS, WITHOUT HARD-CODING ONE PERSON'S DRIVE
# =============================================================================
# A notebook that names `G:\My Drive\...` runs on exactly one machine. It was
# doing that until 1.9.0.74 and it failed on the author's SECOND computer, let
# alone anybody else's — which for a tool about to be handed to somebody else
# is not an inconvenience, it is a defect.
#
# The search is ordered most-specific-first and every step is printed, so a
# reader can see which folder was chosen and why rather than wondering.
#
# `RATATOSK_DATA_DIR` comes FIRST and unconditionally. That is the handle for
# a scheduled run, a CI job, a container, or anyone who simply wants to say
# where the data is without editing a notebook cell.
DATA_DIR_ENV = "RATATOSK_DATA_DIR"

# Folder names to look for under each root, most specific first.
DATA_DIR_LEAVES = (
    os.path.join("Python Scripting", "Example Battery Data"),
    os.path.join("Python Scripting", "Battery Data"),
    "Example Battery Data",
    "Battery Data",
    "",
)


def _candidate_roots():
    """Places a data folder plausibly lives, most likely first."""
    roots, home = [], os.path.expanduser("~")

    def add(p):
        if p and p not in roots:
            roots.append(p)

    # Cloud-drive mounts, however this machine spells them.
    if os.name == "nt":
        # Only drives that actually exist — probing a missing letter on
        # Windows can block for seconds on a disconnected network mapping.
        for _L in "GHIJKLMNOPQRSTUVWXYZABCDEF":
            _d = f"{_L}:\\"
            try:
                if not os.path.exists(_d):
                    continue
            except OSError:
                continue
            add(os.path.join(_d, "My Drive"))
            add(os.path.join(_d, "Shared drives"))
            add(_d)
    for _n in ("Google Drive", "GoogleDrive", "OneDrive", "Dropbox"):
        add(os.path.join(home, _n, "My Drive"))
        add(os.path.join(home, _n))
    add(os.path.join(home, "My Drive"))
    add(os.path.join(home, "Documents"))
    add(home)
    # Beside the notebook, and wherever the kernel happens to be running.
    try:
        add(os.getcwd())
    except OSError:
        pass
    return roots


def find_data_dir(pattern="*.xlsx", extra=(), verbose=True):
    """
    The folder holding the cycler exports, discovered rather than assumed.

    Order: `$RATATOSK_DATA_DIR`, then anything in `extra`, then every
    combination of `_candidate_roots()` and `DATA_DIR_LEAVES`.

    A FOLDER THAT EXISTS BUT HOLDS NO MATCHING FILES IS NOT THE DATA FOLDER.
    Existence alone would stop the search at `~/Documents` on almost every
    machine, so a candidate has to contain at least one file matching
    `pattern` to win. A folder that exists and is empty is remembered as a
    fallback and reported as such, because "I found it but it is empty" and
    "I could not find it" are different problems with different fixes.

    Returns the path, or "" if nothing was found. Everything tried is left on
    `find_data_dir.searched` so the caller's error message can list it.
    """
    import glob as _glob

    tried, empty = [], []
    env = os.environ.get(DATA_DIR_ENV, "").strip()
    order = ([env] if env else []) + [str(p) for p in (extra or [])]
    # LEAF-MAJOR, NOT ROOT-MAJOR. Root-major tries `G:\My Drive` itself — the
    # bare-root leaf — before it tries `~/Documents/Example Battery Data`, so
    # one stray spreadsheet at the top of a cloud drive beats a folder that
    # is actually named after the data. Sweeping the most specific leaf
    # across EVERY root first means a folder called "Example Battery Data"
    # wins wherever it lives, and the bare roots are only reached when no
    # named folder exists anywhere.
    _roots = _candidate_roots()
    for leaf in DATA_DIR_LEAVES:
        for root in _roots:
            order.append(os.path.join(root, leaf) if leaf else root)

    found = ""
    for cand in order:
        if not cand or cand in tried:
            continue
        tried.append(cand)
        try:
            if not os.path.isdir(cand):
                continue
            if _glob.glob(os.path.join(cand, pattern)):
                found = cand
                break
            empty.append(cand)
        except OSError:
            continue                      # unreadable mount, keep looking

    find_data_dir.searched = tried
    find_data_dir.empty = empty
    if verbose:
        if found:
            _why = ("$" + DATA_DIR_ENV if env and found == env
                    else f"{len(_glob.glob(os.path.join(found, pattern)))} "
                         f"file(s) matching {pattern}")
            print(entry("data folder", "found", _why))
            print(entry("", found[:70]))
        elif empty:
            print(entry("data folder", "none with data",
                        f"{len(empty)} folder(s) exist but hold no {pattern}"))
        else:
            print(entry("data folder", "not found",
                        f"searched {len(tried)} location(s)"))
    return found


def validate_run_settings(*, data_dir, output_base, files, pattern,
                          parameter_mode, parameters, max_cycle,
                          dqdv_computation, dqdv_bin_mV, closure_sample,
                          run_peak_fitting, verbose=True):
    """
    Read Cell 2 back and refuse the run if it says something impossible.

    Every one of these used to fail LATER, and most of them failed quietly.
    `PARAMETER_MODE = "Ask"` fell through the `== "ask"` test into the
    declarative branch and ran the whole pipeline on a NaN theoretical
    capacity; `DQDV_COMPUTATION = "hist"` fell through to the derivative path
    and produced a perfectly presentable set of figures computed the way the
    cell said it did not want; a mistyped `DATA_DIR` surfaced as an empty
    glob three cells later. A settings cell that is wrong should say so
    where it was written, not somewhere downstream.

    Returns the settings as a dict, which the manifest records verbatim.
    """
    modes = ("ask", "dict")
    methods = ("histogram", "derivative")

    if not isinstance(data_dir, str) or not data_dir.strip():
        # AN EMPTY DATA_DIR MEANS THE SEARCH FOUND NOTHING, not that the user
        # forgot to type something — Cell 2 leaves it blank on purpose and
        # `find_data_dir` fills it in. Say what was looked for and give the
        # two ways out, rather than "set it to the folder", which is what
        # they were trying to avoid having to do per machine.
        _tried = list(getattr(find_data_dir, "searched", ()) or ())
        _empty = list(getattr(find_data_dir, "empty", ()) or ())
        _msg = ["DATA_DIR is empty and no data folder was found."]
        if _empty:
            _msg.append(
                "These folders exist but hold no file matching the pattern:")
            _msg += [f"    {p}" for p in _empty[:5]]
        elif _tried:
            _msg.append(f"{len(_tried)} location(s) were searched, including:")
            _msg += [f"    {p}" for p in _tried[:5]]
        _msg += [
            "",
            "Either set DATA_DIR in Cell 2 to the folder holding the cycler",
            f"exports, or set the {DATA_DIR_ENV} environment variable to it",
            "before starting Jupyter — which is the one to use if you move",
            "between machines.",
        ]
        raise ValueError("\n".join(_msg))
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(
            f"DATA_DIR does not exist:\n    {data_dir}\n"
            f"Check the drive letter and that any network or Drive folder "
            f"is mounted. To stop naming one machine's drive, clear DATA_DIR "
            f"in Cell 2 and it will be searched for, or set the "
            f"{DATA_DIR_ENV} environment variable.")

    if not isinstance(output_base, str) or not output_base.strip():
        raise ValueError("OUTPUT_BASE is empty — set it to the folder the "
                         "run folders should be created in.")
    # The folder itself is created by `start_run`; what cannot be created is
    # a missing drive, and that is worth saying before an hour of fitting.
    anchor = os.path.splitdrive(os.path.abspath(output_base))[0]
    if anchor and not os.path.exists(anchor + os.sep):
        raise NotADirectoryError(
            f"OUTPUT_BASE is on {anchor} which is not available:\n"
            f"    {output_base}")

    if isinstance(files, str):
        raise TypeError("FILES must be a list of file names, not a single "
                        f"string. Write FILES = [{files!r}].")
    files = list(files or [])
    for f in files:
        if not isinstance(f, str):
            raise TypeError(f"FILES contains {f!r}, which is not a file name.")
    missing = [f for f in files
               if not os.path.isfile(os.path.join(data_dir, f))]
    if missing:
        raise FileNotFoundError(
            "FILES names files that are not in DATA_DIR:\n"
            + "\n".join(f"    {m}" for m in missing)
            + f"\n  looked in {data_dir}")
    if not files and (not isinstance(pattern, str) or not pattern.strip()):
        raise ValueError("PATTERN is empty and FILES is empty, so nothing "
                         "would be listed. Use \"*.xlsx\".")

    mode = str(parameter_mode).strip().lower()
    if mode not in modes:
        raise ValueError(f"PARAMETER_MODE is {parameter_mode!r}; it must be "
                         f"one of {', '.join(repr(m) for m in modes)}.")
    if mode == "dict" and not parameters:
        raise ValueError(
            "PARAMETER_MODE is \"dict\" but PARAMETERS is empty. Dict mode "
            "takes every parameter from PARAMETERS and there is nothing to "
            "take — fill it in, or set PARAMETER_MODE = \"ask\".")

    method = str(dqdv_computation).strip().lower()
    if method not in methods:
        raise ValueError(f"DQDV_COMPUTATION is {dqdv_computation!r}; it must "
                         f"be one of {', '.join(repr(m) for m in methods)}.")

    if max_cycle is not None:
        if not isinstance(max_cycle, (int, np.integer)) or max_cycle < 1:
            raise ValueError(f"MAX_CYCLE is {max_cycle!r}; it must be None or "
                             f"a whole number of cycles, 1 or more.")
    if dqdv_bin_mV is not None:
        if not isinstance(dqdv_bin_mV, (int, float, np.number)) \
                or not np.isfinite(dqdv_bin_mV) or not 0.05 <= dqdv_bin_mV <= 50:
            raise ValueError(
                f"DQDV_BIN_MV is {dqdv_bin_mV!r}; it must be None (let the "
                f"measured profile choose) or a bin width in mV between "
                f"0.05 and 50.")
    if closure_sample is not None:
        if not isinstance(closure_sample, (int, np.integer)) or closure_sample < 3:
            raise ValueError(
                f"CLOSURE_SAMPLE is {closure_sample!r}; it must be None "
                f"(every half-cycle) or at least 3 — a median and a pass "
                f"fraction over fewer than three fits is not a measurement.")
    if not isinstance(run_peak_fitting, (bool, np.bool_)):
        raise TypeError(f"RUN_PEAK_FITTING is {run_peak_fitting!r}; it must "
                        f"be True or False.")

    settings = dict(
        data_dir=data_dir, output_base=output_base, files=files,
        pattern=pattern, parameter_mode=mode, parameter_count=len(parameters or {}),
        max_cycle=max_cycle, dqdv_computation=method, dqdv_bin_mV=dqdv_bin_mV,
        closure_sample=closure_sample, run_peak_fitting=bool(run_peak_fitting))

    if verbose:
        print(entry("data", os.path.basename(data_dir.rstrip("\\/")) or data_dir,
                    data_dir, value_width=34))
        print(entry("output", os.path.basename(output_base.rstrip("\\/")),
                    output_base, value_width=34))
        print(entry("files",
                    f"{len(files)} named" if files else "chosen at Cell 3a",
                    "" if files else f"pattern {pattern}", value_width=34))
        print(entry("parameters", mode,
                    "prompted, pre-filled from the file" if mode == "ask"
                    else f"{len(parameters or {})} taken from PARAMETERS",
                    value_width=34))
        print(entry("dQ/dV", method,
                    f"{dqdv_bin_mV} mV bins" if dqdv_bin_mV
                    else "bin width from the measured profile",
                    value_width=34))
        print(entry("peak fitting", "yes" if run_peak_fitting else "no",
                    (f"closure on {closure_sample} half-cycles"
                     if closure_sample else "closure on every half-cycle")
                    if run_peak_fitting else "cycling analysis only",
                    value_width=34))
        print(entry("cycles", f"to {max_cycle}" if max_cycle else "all",
                    "" if max_cycle else "no cap set", value_width=34))
    return settings
