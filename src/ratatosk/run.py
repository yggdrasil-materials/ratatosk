"""
Where a run's outputs go, and what it records about itself.

Ported from 1.8.7 Cell 3b: every run gets its own timestamped folder under
`runs/`, a `run_manifest.json` recording who ran what against which files with
which package versions, and — at the end — the outputs sorted into numbered
subfolders and the manifest closed with a checksum of everything written.

    start_run(base, version, input_paths)  -> (run_dir, manifest)
    write_manifest(run_dir, manifest)
    finalise_run(run_dir, manifest)         organise, checksum, close
    freeze_run(run_dir)                     copy to published/, never overwritten

The one change: `pipeline_flags`
--------------------------------
1.8.7 captured the module-level switches that change what a run produces by
reading `globals()` for a hardcoded list of names. In a package there is no
single namespace holding them all, so `start_run` and `finalise_run` take a
`flags` dict instead, and `default_flags()` assembles it by asking each module
for its own constants. That makes the manifest complete rather than dependent
on which cells happened to have run — which was 1.8.7's own complaint about it
in the docstring the port preserves.

Why it matters: with the version deliberately held across a series of
behaviour changes, these flags are the only thing distinguishing one run from
another.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import re
import shutil
import sys

from .style import WORKING_ION_WORDS
from datetime import datetime, timezone

__all__ = ["start_run", "write_manifest", "finalise_run", "freeze_run",
           "dataset_folder", "dataset_folders",
           "organise_run", "write_dataset_info", "default_flags",
           "subfolders"]

FREEZE_ROOT_NAME = 'published'   # frozen runs live here, never overwritten
HASH_CHUNK = 1 << 20
_MISSING = object()      # "this value is not a recordable flag"

# Where each output lands. Rules are ordered and the first match wins:
# 'capacity_attribution' must be caught as a descriptor before the generic
# 'capacity' rule claims it for cycling.
FOLDER_RULES = [
    ('7_summary',           ['analysis_summary', 'results_at_a_glance',
                             'ratatosk_summary', 'summary.pdf',
                             'cycling_performance_summary']),
    ('6_descriptors',       ['tracked_peaks', 'delta_v', 'peak_trends',
                             'capacity_attribution', 'trend_shapes']),
    # 'fit_coherence' and 'cycle_integrity' are new in 1.9.0; without a rule
    # they land in '9_other'.
    ('5_peak_fitting',      ['detected_peaks', 'peak_fits',
                             'fit_diagnostics', 'fitted_parameters',
                             'fit_coherence']),
    # 'preprocessing_qc' is the raw-against-processed panel for one cycle —
    # a dQ/dV figure whose filename does not contain the string, so it landed
    # in '9_other', on its own, with nothing in the run linking to it.
    ('4_dqdv',              ['dqdv', 'preprocessing_qc']),
    ('2_voltage_profiles',  ['voltage_profile']),
    ('3_energy_power',      ['ragone', 'power_per', 'energy_per',
                             'power_energy']),
    # The ion-named voltage tokens are GENERATED from `WORKING_ION_WORDS`,
    # not typed: one output filename is built from the anode-aware half-cycle
    # label, so an anode dataset writes _avg_desodiation_voltage.csv where a
    # cathode writes _avg_discharge_voltage.csv. It is the only filename in
    # Ratatosk that varies with electrode polarity, and a hand-written list
    # of Li and Na tokens would quietly stop filing the file the day a third
    # ion existed — it would land in the run root with no error.
    ('1_cycling',           ['cycle_integrity',
                             'capacity', 'cycle_life', 'retention', 'fade',
                             'efficiency', 'coulombic',
                             'discharge_voltage']
                            + [f'{_w.lower()}_voltage'
                               for _p in WORKING_ION_WORDS.values()
                               for _w in _p]),
]
COMPARATIVE_DIR = '_comparative'   # outputs not tied to one cell
COPY_COMPARATIVE_TO_CELLS = True
COMPARATIVE_COPY_PREFIX = 'comparative_'


def subfolders():
    """The output folders a run creates, in reading order."""
    return [f for f, _ in sorted(FOLDER_RULES)] + [COMPARATIVE_DIR]


def default_flags():
    """
    Every module-level switch that changes what a run produces.

    Collected by asking each module for its own constants, so the manifest is
    complete whatever order the notebook was run in. 1.8.7 read a hardcoded
    name list out of `globals()`, and noted in its own docstring that a flag
    defined in Module 4 could not be seen from Cell 3b.

    Works in BOTH builds. The package has modules to ask; the inline notebook
    is one flat namespace with no modules at all, so the relative import
    below raised `ImportError` there and — since `start_run` does not catch
    it — the flags were never recorded in the notebook anyone actually runs.
    In that build the constants are plain globals and are read directly.
    """
    def _scalar(v):
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, (list, tuple)) and all(
                isinstance(x, (bool, int, float, str)) for x in v):
            return list(v)
        if isinstance(v, dict) and all(
                isinstance(x, (bool, int, float, str)) for x in v.values()):
            # A per-profile table such as PROFILE_MIN_DISTANCE_MV decides how
            # many peaks a run finds; it is as much a switch as a scalar is.
            return dict(v)
        return _MISSING

    out = {}
    try:
        from . import (signal, detect, fitting, quality, analyse, plots,
                       cycling, io, params, report)  # inline-ok: the notebook
        # has no package, raises ImportError here, and the `except` below
        # reads the same constants from the flat namespace instead. Deliberate
        # and handled — see the docstring.
        mods = [signal, detect, fitting, quality, analyse, plots,
                cycling, io, params, report]
    except ImportError:
        mods = []

    if mods:
        for mod in mods:
            short = mod.__name__.rsplit('.', 1)[-1]
            for k in dir(mod):
                if not re.fullmatch(r"[A-Z][A-Z0-9_]*", k):
                    continue
                v = _scalar(getattr(mod, k))
                if v is not _MISSING:
                    out[f"{short}.{k}"] = v
        return out

    # Flattened notebook: one namespace, no modules. The names are the same;
    # only the module prefix is unavailable, so they are recorded bare.
    g = dict(globals())
    for k, raw in list(g.items()):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", k):
            continue
        v = _scalar(raw)
        if v is not _MISSING:
            out[k] = v
    return out

def _sha256(path, chunk=HASH_CHUNK):
    """Checksum a file without loading it into memory."""
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(chunk), b''):
                h.update(block)
        return h.hexdigest()
    except Exception as exc:
        return f'unavailable: {exc}'


def _jsonable(obj):
    """
    Coerce anything the parameter dictionaries might hold into JSON.

    numpy scalars, numpy arrays, pandas types and Path objects all appear in
    user_parameters depending on how a value was entered, and a manifest that
    fails to write is worse than no manifest at all.
    """
    try:
        import numpy as _np
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _package_versions():
    """
    Versions of the libraries whose behaviour a result depends on.

    Called twice, like _pipeline_flags, and for a related reason: on
    2026-08-29 a run recorded lmfit as 'not loaded' because lmfit was not
    installed when Cell 3b ran — it was pip-installed after Module 4 raised,
    and Module 4 was re-run. The manifest therefore did not record the version
    of the one library that does the fitting. finalise_run re-reads this.

    'not installed' and 'import failed' are now distinguished, because
    'not loaded' read as "we did not check" when it meant "it was not there".
    """
    import importlib, importlib.util
    out = {}
    for mod in ('numpy', 'scipy', 'pandas', 'matplotlib', 'lmfit',
                'seaborn', 'python_calamine'):
        try:
            if importlib.util.find_spec(mod) is None:
                out[mod] = 'not installed'
                continue
            out[mod] = getattr(importlib.import_module(mod),
                               '__version__', 'unknown version')
        except Exception as _e:
            out[mod] = f'import failed: {type(_e).__name__}'
    return out




def start_run(base_location, version, input_paths=None, note='',
              flags=None):
    """
    Create a run folder and write the opening manifest.

    Returns (run_dir, manifest). The caller rebinds save_location to run_dir.
    """
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    run_id = f'{stamp}_v{version}'
    # normpath so the manifest does not record a path with mixed
    # separators. Cell 2 builds save_location from a forward-slash string
    # via expanduser, which on Windows yields C:\Users\me/Documents/...
    run_dir = os.path.normpath(
        os.path.join(base_location, 'runs', run_id))
    os.makedirs(run_dir, exist_ok=True)

    inputs = []
    for p in (input_paths or []):
        try:
            st = os.stat(p)
            inputs.append({
                'path': os.path.abspath(p),
                'name': os.path.basename(p),
                'bytes': st.st_size,
                'modified': datetime.fromtimestamp(
                    st.st_mtime, timezone.utc).isoformat(),
                'sha256': _sha256(p),
            })
        except Exception as exc:
            inputs.append({'path': str(p), 'error': str(exc)})

    manifest = {
        'run_id': run_id,
        'ratatosk_version': version,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'note': note,
        'operator': getpass.getuser(),
        'machine': platform.node(),
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'packages': _package_versions(),
        # Passed in rather than scraped from globals() — see the module
        # docstring. `default_flags()` builds the complete set.
        'pipeline_flags': (flags if flags is not None
                           else default_flags()),
        'inputs': inputs,
        'datasets': {},
        'outputs': [],
        'status': 'running',
    }
    return run_dir, manifest


def write_manifest(run_dir, manifest):
    path = os.path.join(run_dir, 'run_manifest.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, default=_jsonable)
    return path


_WIN_RESERVED = {'CON', 'PRN', 'AUX', 'NUL'} | {
    f'{p}{i}' for p in ('COM', 'LPT') for i in range(1, 10)}


def _safe_dirname(name):
    out = str(name)
    for ch in '<>:"/\\|?*':
        out = out.replace(ch, '_')
    out = out.strip().rstrip('. ') or 'dataset'
    # Windows cannot create a directory named CON, PRN, AUX, NUL, COM1-9 or
    # LPT1-9, with or without an extension, and rejects a trailing space.
    if out.split('.')[0].upper() in _WIN_RESERVED:
        out += '_'
    return out


def _dataset_folder(name, params, used=None):
    """
    Reader-legible folder name: composition and cell, nothing more.

    Kept short deliberately. Windows caps a full path at 260 characters and
    Ratatosk's own filenames already run to 70, so the descriptive detail
    goes in dataset_info.txt inside the folder rather than in its name.
    """
    # Truncate FIRST, then sanitise: slicing after the strip could leave a
    # trailing space or dot, which Windows will not accept in a path segment.
    comp = _safe_dirname(str(params.get('composition', '') or name)[:34])
    # _safe_dirname substitutes 'dataset' for an empty string, so guard the
    # empty case rather than feeding it through and getting '_celldatase'.
    raw_cid = str(params.get('cell_id', '') or '').strip()
    cid = _safe_dirname(raw_cid[:6]) if raw_cid else ''
    base = f"{comp}_cell{cid}" if cid else comp
    if used is not None:
        stem, n = base, 2
        while base in used:
            base = f"{stem}_{n}"
            n += 1
        used.add(base)
    return base


def write_dataset_info(folder, name, params, manifest=None, decided=None):
    """A plain-text record of the cell, written beside its results.

    `params` is what the operator ASKED for. `decided` is what the pipeline
    MEASURED and then used — the profile class it read off the curve, the
    reference cycle it chose and why, the peak shape it calibrated, the
    integral fidelity it found. Those are the numbers that determined the
    result, and until 1.9.0.11 this file recorded only the request. It is the
    same rule as the manifest's: record what ran, never what was offered.

    `decided` is a plain dict of label -> value, built by the notebook from
    the objects that did the work; anything missing is simply omitted, so an
    older caller still produces a valid record.
    """
    src = ''
    for inp in (manifest or {}).get('inputs', []):
        if inp.get('name', '').startswith(name):
            src = f"{inp.get('name')}  (sha256 {inp.get('sha256', '')[:16]})"
            break
    rows = [
        ('Composition', params.get('composition')),
        ('Cell identifier', params.get('cell_id') or '(not recorded)'),
        ('Dataset key', name),
        ('Source file', src or '(not captured)'),
        # The version, beside the source file, so this folder says what made
        # it without anyone having to open the manifest.
        ('Ratatosk version',
         (manifest or {}).get('ratatosk_version') or '(not captured)'),
        ('Run', (manifest or {}).get('run_id') or '(not captured)'),
        ('Electrode', params.get('electrode_type')),
        ('Active mass (mg)', params.get('active_material_mass_mg')),
        ('Areal loading (mg/cm2)', params.get('active_loading_mg_cm2')),
        ('Blend', params.get('blend')),
        ('Rate (C)', params.get('charge_rate_c')),
        ('Voltage window (V)', f"{params.get('voltage_lower_V')} - "
                               f"{params.get('voltage_upper_V')}"),
        ('Electrolyte', params.get('electrolyte_composition')),
        ('Electrolyte volume (uL)', params.get('electrolyte_volume_uL')),
        ('Separator', params.get('separator_type')),
        ('Theoretical capacity (mAh/g)',
         params.get('theoretical_capacity_mAh_g')),
        ('Test start date', params.get('test_start_date')),
        # 'all cycles', not an empty field. A blank against a named row
        # reads as something the pipeline failed to record, when it means
        # the opposite: no cap was asked for and none was applied.
        ('Max cycle analysed', params.get('max_cycle') or 'all cycles'),
    ]
    decided = dict(decided or {})
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'dataset_info.txt'), 'w',
              encoding='utf-8') as f:
        f.write("Ratatosk dataset record\n")
        f.write("=" * 58 + "\n")
        for k, v in rows:
            if isinstance(v, float):
                v = f"{v:.4g}"
            f.write(f"{k:<30} {'' if v is None else v}\n")
        f.write("=" * 58 + "\n")
        if decided:
            f.write("\nWhat the pipeline measured and then used\n")
            f.write("-" * 58 + "\n")
            for k, v in decided.items():
                if isinstance(v, float):
                    v = f"{v:.4g}"
                f.write(f"{k:<30} {'' if v is None else v}\n")
            f.write("-" * 58 + "\n")
        f.write("Full parameters and checksums: ../run_manifest.json\n")


def _classify(filename):
    low = filename.lower()
    for folder, keys in FOLDER_RULES:
        if any(k in low for k in keys):
            return folder
    return '9_other'


def organise_run(run_dir=None, dataset_names=None, user_parameters=None,
                 verbose=True, manifest=None, decided=None):
    """
    Sort a flat run folder into per-dataset, per-stage subfolders.

    Idempotent: only files sitting at the top level are moved, so running
    it twice is harmless and running it mid-analysis then again at the end
    simply picks up whatever arrived in between.
    """
    if not run_dir or not os.path.isdir(run_dir):
        print("  No run folder to organise.")
        return {}
    # 1.8.7 read `user_parameters` out of the notebook's globals. In a package
    # there is no such namespace, so it is a parameter — and without it every
    # output lands in `_comparative/` because no filename matches a dataset.
    all_params = dict(user_parameters or {})
    names = list(dataset_names or all_params.keys())
    names.sort(key=len, reverse=True)   # longest prefix wins
    used = set()
    folder_for = {nm: _dataset_folder(nm, all_params.get(nm, {}), used)
                  for nm in sorted(names)}

    moved, longest = {}, 0
    for fn in sorted(os.listdir(run_dir)):
        src = os.path.join(run_dir, fn)
        if not os.path.isfile(src) or fn == 'run_manifest.json':
            continue
        owner, stem = COMPARATIVE_DIR, fn
        for nm in names:
            if fn.startswith(nm):
                owner, stem = folder_for[nm], fn[len(nm):].lstrip('_')
                break
        # One cell loaded means the run-wide figures are that cell's own.
        if owner == COMPARATIVE_DIR and len(names) == 1:
            owner = folder_for[names[0]]
        dest_dir = os.path.join(run_dir, owner, _classify(stem))
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, fn)
        if os.path.abspath(src) == os.path.abspath(dest):
            continue
        if os.path.exists(dest):
            os.remove(dest)
        shutil.move(src, dest)
        longest = max(longest, len(os.path.abspath(dest)))
        moved.setdefault(os.path.join(owner, _classify(stem)), []).append(fn)

    # Mirror the run-wide figures into each cell folder.
    comp_root = os.path.join(run_dir, COMPARATIVE_DIR)
    if COPY_COMPARATIVE_TO_CELLS and len(names) > 1 \
            and os.path.isdir(comp_root):
        n_copied = 0
        for stage in sorted(os.listdir(comp_root)):
            stage_dir = os.path.join(comp_root, stage)
            if not os.path.isdir(stage_dir):
                continue
            for fn in sorted(os.listdir(stage_dir)):
                src_f = os.path.join(stage_dir, fn)
                if not os.path.isfile(src_f):
                    continue
                for nm in names:
                    tgt = os.path.join(run_dir, folder_for[nm], stage)
                    os.makedirs(tgt, exist_ok=True)
                    # Some run-wide figures are already called
                    # comparative_*.png, and blindly prefixing gave
                    # comparative_comparative_discharge_capacity.png.
                    stem = (fn if fn.startswith(COMPARATIVE_COPY_PREFIX)
                            else COMPARATIVE_COPY_PREFIX + fn)
                    out = os.path.join(tgt, stem)
                    if not os.path.exists(out):
                        shutil.copy2(src_f, out)
                        n_copied += 1
                        # These are the LONGEST paths a run writes — a
                        # nested cell folder plus a 12-character prefix —
                        # and they were the ones the 260-character warning
                        # never measured.
                        longest = max(longest, len(os.path.abspath(out)))
        if verbose and n_copied:
            print(f"      copied {n_copied} run-wide figure(s) into the "
                  f"{len(names)} cell folder(s), prefixed "
                  f"'{COMPARATIVE_COPY_PREFIX}'")

    for nm in names:
        d = os.path.join(run_dir, folder_for[nm])
        if os.path.isdir(d):
            # The manifest is passed in. `globals().get('RUN_MANIFEST')`
            # was a leftover of the notebook-globals pattern this module
            # exists to remove: run.py's own globals never hold it, so every
            # dataset_info.txt said "Source file: (not captured)".
            write_dataset_info(d, nm, all_params.get(nm, {}), manifest,
                               decided=(decided or {}).get(nm))

    if verbose and moved:
        print(f"\n  Sorted {sum(len(v) for v in moved.values())} file(s):")
        for folder in sorted(moved):
            print(f"      {folder:<44} {len(moved[folder])} file(s)")
    if longest > 240 and verbose:
        print(f"      WARNING: longest output path is {longest} "
              f"characters. Windows fails above 260; shorten the "
              f"composition or move the output folder nearer the "
              f"drive root.")
    return moved


def finalise_run(run_dir=None, manifest=None, verbose=True, organise=True,
                 flags=None, user_parameters=None, decided=None,
                 detection=None):
    """
    Close the manifest: inventory every file written, with checksums.

    Run this after the last analysis cell. Safe to run more than once; it
    simply refreshes the inventory. An empty file is recorded as such rather
    than merely present, because a zero-byte CSV passes an existence check
    while containing nothing.
    """
    run_dir = run_dir or globals().get('RUN_DIR')
    manifest = manifest if manifest is not None else globals().get('RUN_MANIFEST')
    if not run_dir or manifest is None:
        print("  No active run to finalise.")
        return None

    if organise:
        organise_run(run_dir, user_parameters=user_parameters,
                     verbose=verbose, manifest=manifest, decided=decided)

    files = []
    for root, _dirs, fns in os.walk(run_dir):
        for fn in sorted(fns):
            if fn == 'run_manifest.json':
                continue
            fp = os.path.join(root, fn)
            size = os.path.getsize(fp)
            files.append({
                'name': fn,
                'path': os.path.relpath(fp, run_dir).replace(os.sep, '/'),
                'bytes': size, 'empty': size == 0, 'sha256': _sha256(fp)})

    manifest['outputs'] = files
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['status'] = 'complete'

    # Re-read the flags at the end. In 1.8.7 this mattered because start_run
    # ran before Modules 1, 1b, 3, 3b and 4 existed and could not see their
    # constants. In 1.9.0 `default_flags()` reads them from the modules, so
    # both readings are complete — but a value the notebook CHANGED mid-run
    # still differs, and the one that was in force at the end is the one that
    # produced the outputs. Anything changed since start_run is kept alongside
    # under the name it had.
    _late = (flags if flags is not None else default_flags())
    _early = manifest.get('pipeline_flags', {}) or {}
    for k, v in _early.items():
        if k in _late and _late[k] != v:
            _late[f'{k}__at_start'] = v
    manifest['pipeline_flags'] = _late

    # Same again for package versions, and for the same reason: a library
    # installed or upgraded mid-run is invisible to the probe at start_run.
    _pkg_late = _package_versions()
    _pkg_early = manifest.get('packages', {}) or {}
    for k, v in _pkg_early.items():
        if k in _pkg_late and _pkg_late[k] != v:
            _pkg_late[f'{k}__at_start'] = v
    manifest['packages'] = _pkg_late

    if 'datasets' in manifest and not manifest['datasets'] and user_parameters:
        # From the ARGUMENT. Reading it from globals() recorded
        # "datasets": {} in every run_manifest.json ever written.
        manifest['datasets'] = dict(user_parameters)
    # The reference cycle is a per-dataset choice with dataset-wide reach, so
    # record which one was used and why. Read-only; nothing is mutated.
    try:
        # FROM THE ARGUMENT — the same correction the comment eleven lines
        # above records for `datasets`, written again immediately below it.
        # `detected_peaks` is never a module global here, so `_det` was
        # always {} and `reference_cycle_used` reached no manifest ever
        # written. The reference cycle decides what every tracked feature is
        # measured against; it belongs in the record of the run.
        _det = detection or {}
        for _n, _d in _det.items():
            _entry = manifest.setdefault('datasets', {}).setdefault(_n, {})
            if not isinstance(_entry, dict):
                continue
            _rc = (_d.get('reference_cycle') if isinstance(_d, dict)
                   else getattr(_d, 'reference_cycle', None))
            _rr = (_d.get('reference_cycle_reason') if isinstance(_d, dict)
                   else getattr(_d, 'reference_cycle_reason', None))
            if _rc is not None:
                _entry['reference_cycle_used'] = _rc
            if _rr:
                _entry['reference_cycle_reason'] = _rr
    except Exception:
        pass
    path = write_manifest(run_dir, manifest)

    if verbose:
        empties = [f['name'] for f in files if f['empty']]
        print(f"\n  Run finalised: {len(files)} output file(s) recorded")
        print(f"  Manifest: {path}")
        if empties:
            print(f"  WARNING: {len(empties)} file(s) written empty: "
                  f"{', '.join(empties[:5])}"
                  f"{' ...' if len(empties) > 5 else ''}")
    return path


def freeze_run(run_dir=None, label='', base_location=None):
    """
    Copy a run into published/ so it can never be overwritten or re-run over.

    Use this the moment a figure goes into a thesis or a paper. The frozen
    copy keeps its manifest, so the provenance travels with it.
    """
    run_dir = run_dir or globals().get('RUN_DIR')
    base = base_location or globals().get('BASE_SAVE_LOCATION')
    if not run_dir or not base:
        print("  Nothing to freeze.")
        return None
    tag = f"{os.path.basename(run_dir)}{('_' + label) if label else ''}"
    dest = os.path.join(base, FREEZE_ROOT_NAME, tag)
    if os.path.exists(dest):
        print(f"  Already frozen: {dest}")
        return dest
    shutil.copytree(run_dir, dest)
    with open(os.path.join(dest, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Frozen Ratatosk run\n"
                f"Label: {label or '(none)'}\n"
                f"Source run: {run_dir}\n"
                f"Frozen: {datetime.now().isoformat(timespec='seconds')}\n\n"
                f"Do not overwrite or re-run into this folder. Provenance is "
                f"in run_manifest.json.\n")
    print(f"  Frozen to: {dest}")
    return dest


def dataset_folders(all_params, dataset_names=None):
    """
    The folder `organise_run` gives EVERY dataset, as `{name: folder}`.

    Ask for all of them at once, because the answer for one depends on the
    others: two cells sharing a composition and a cell id collide, and
    `_dataset_folder` resolves that by appending _2, _3 — but only when it is
    told what is already taken. The old single-dataset wrapper never passed
    that set, so `organise_run` created X and X_2 while the notebook wrote
    both START_HERE reports into X, and the second silently replaced the
    first beside the other cell's figures.
    """
    # SORTED, BECAUSE `organise_run` SORTS.
    # `_dataset_folder` appends _2 to whichever colliding name it sees SECOND,
    # so the suffix depends entirely on iteration order. `organise_run` builds
    # its map over `sorted(names)`; this one built it over the caller's list
    # order, and the notebook passes `list(datasets)` — load order. Two cells
    # sharing a composition with no cell_id therefore got OPPOSITE folder
    # names from the two functions, and each cell's report was written beside
    # the other cell's figures. That is the exact failure this function's
    # docstring says it exists to prevent, reintroduced by the one line that
    # differs between the two.
    names = sorted(dataset_names if dataset_names is not None else all_params)
    used = set()
    return {n: _dataset_folder(n, (all_params or {}).get(n, {}) or {}, used)
            for n in names}


def dataset_folder(name, params=None, *, used=None):
    """One dataset's folder. Prefer `dataset_folders` — see why there."""
    return _dataset_folder(name, params or {}, used)
