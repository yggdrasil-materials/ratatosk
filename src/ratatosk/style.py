"""
Console presentation: colour, rules, and aligned fields.

Why this is a module and not a class in each file
-------------------------------------------------
1.8.7 styled exactly one cell — the parameter review — with a `_S` class local
to it, and the other twenty printed flat text. So a run looked designed for
thirty seconds and like a log file for the other ten minutes, and there was
nowhere to change that from. The palette now lives here, every module draws on
it, and the flat notebook emits this cell first so the bare names resolve.

The vocabulary is deliberately small and SEMANTIC. `HEADER`, `SECTION`,
`VALUE`, `WARN` say what a thing is; nothing anywhere else in the codebase
writes an escape code. That is what makes it possible to change the whole
run's appearance — or turn it off — in one place.

Turning it off
--------------
`set_colour(False)`, or the environment variable `NO_COLOR` (the de facto
standard), or any non-terminal destination that is not a notebook. Colour is
an aid to reading, never a carrier of meaning: every line says in words what
its colour says in hue, so a log file with the codes stripped loses nothing.
"""

from __future__ import annotations

import math
import os
import sys

__all__ = [
    "ENTRY_LINE_WIDTH",
    "RULE_WIDTH",
    "S",
    "Style",
    "bullet",
    "entry",
    "heading",
    "image_format",
    "nice_axis_limit",
    "rule",
    "saved",
    "section",
    "set_colour",
    "supports_colour",
    "verdict",
]

RULE_WIDTH = 76


def _enable_windows_vt():
    """Ask a Windows console to interpret ANSI. Harmless everywhere else.

    Anaconda Prompt on Windows 10+ can render escape codes but does not by
    default; Jupyter and Windows Terminal already do. Without this the run
    prints its own escape codes as literal noise, which is worse than no
    colour at all — so a failure here turns colour OFF rather than hoping.
    """
    if os.name != "nt":
        return True
    try:
        import ctypes

        k = ctypes.windll.kernel32
        for handle in (-11, -12):  # stdout, stderr
            h = k.GetStdHandle(handle)
            mode = ctypes.c_uint32()
            if not k.GetConsoleMode(h, ctypes.byref(mode)):
                continue
            k.SetConsoleMode(h, mode.value | 0x0004)  # VIRTUAL_TERMINAL
        return True
    except Exception:  # noqa: BLE001
        return False


def supports_colour():
    """Whether escape codes will be rendered rather than printed."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    # A notebook's stdout is not a tty but does render ANSI.
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            return _enable_windows_vt()
    except Exception:  # noqa: BLE001
        pass
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    return _enable_windows_vt()


class Style:
    """The palette. Semantic names only — nothing here is called 'red'."""

    # Sheffield purple banner, from 1.8.7's parameter review.
    HEADER = "\033[1;48;5;54;97m"
    SECTION = "\033[1;96m"  # bold cyan   — section labels
    VALUE = "\033[92m"  # green       — confirmed values
    FILE = "\033[38;5;117m"  # light blue  — values read from the file
    WARN = "\033[1;91m"  # bold red    — warnings and errors
    CONFIRM = "\033[1;92m"  # bold green  — confirmed / accepted
    OK = "\033[92m"  # green       — a measurement that passed
    CAUTION = "\033[93m"  # amber       — passed, with a caveat
    BAD = "\033[91m"  # red         — did not pass
    DIM = "\033[2m"  # de-emphasis — units, notes, provenance
    BOLD = "\033[1m"
    RULE = "\033[38;5;98m"  # muted purple — separators
    RESET = "\033[0m"


class _NoStyle:
    """Every attribute is an empty string, so styled code needs no branches."""

    def __getattr__(self, _name):
        return ""


_COLOUR = supports_colour()
S = Style() if _COLOUR else _NoStyle()
# 1.8.7's name, kept so the parameter review reads unchanged.
_S = S


def set_colour(on):
    """Force colour on or off for the rest of the run."""
    global S, _S, _COLOUR
    _COLOUR = bool(on)
    S = Style() if _COLOUR else _NoStyle()
    _S = S
    return _COLOUR


# ---------------------------------------------------------------------------
# Layout. Every one of these returns a string; nothing here prints, so a
# caller can put a block in a report as easily as on the console.
# ---------------------------------------------------------------------------

# =============================================================================
# WHAT THE WORKING ION IS CALLED
# =============================================================================
# For a NEGATIVE electrode the step labels are swapped (see
# `io.apply_electrode_convention`) and the figures name the process rather
# than the direction: charge is the working ion going IN, discharge is it
# coming out. Naming it requires knowing which ion it is.
#
# Until 1.9.0.75 the two label helpers read `battery_chemistry` and fell
# through to lithium for anything that was not exactly 'Na-ion' — so an NTO
# anode cycled in a sodium half-cell was labelled "Lithiation" and
# "Delithiation" on every waterfall. `infer_chemistry` answers Unknown for an
# abbreviation ("NTO" is not a formula, and it is genuinely ambiguous between
# sodium titanate and a niobium titanium oxide), which is the right answer,
# and the label then guessed the commonest case in silence.
#
# The pipeline ALREADY KNOWS. `counter_electrode_metal` resolves the working
# ion in a recorded order — stated by the operator, inferred from the
# electrolyte salt, taken from the composition library, or assumed — and the
# interactive path ASKS when it is not established. That answer sits in
# `counter_electrode_metal` with its provenance in
# `counter_electrode_metal_source`, and the labels were reading a weaker
# field two seats along. One quantity, two places, the two disagreeing: the
# same shape of defect this build has now found five times.
WORKING_ION_WORDS = {
    "Li": ("Lithiation", "Delithiation"),
    "Na": ("Sodiation", "Desodiation"),
    "K": ("Potassiation", "Depotassiation"),
    "Mg": ("Magnesiation", "Demagnesiation"),
    "Ca": ("Calciation", "Decalciation"),
    "Zn": ("Zincation", "Dezincation"),
    "Al": ("Aluminiation", "Dealuminiation"),
}

# Every capacity-column name the labels below can produce, for the readers
# that have to recognise one written by an earlier run. Generated from the
# table rather than typed out, because three separate hard-coded tuples
# (`analyse.cycle_column`, two in `report`) already listed Li and Na only and
# would have silently stopped matching the moment a third ion existed.
CAPACITY_COLUMN_ALIASES = tuple(
    ["Discharge_mAh_g"] + [f"{_d}_mAh_g" for _c, _d in WORKING_ION_WORDS.values()]
)


def working_ion(params):
    """
    The working ion's element symbol, or "" when it is not established.

    `counter_electrode_metal` FIRST: it is the resolved answer and it carries
    its own provenance, and an "assumed" one is a guess that must not reach a
    figure caption. `battery_chemistry` is the fallback for a params dict
    written before that field existed.
    """
    params = params or {}
    _src = str(params.get("counter_electrode_metal_source") or "").lower()
    _m = str(params.get("counter_electrode_metal") or "").strip().title()
    if _m in WORKING_ION_WORDS and _src and _src != "assumed":
        return _m
    _chem = str(params.get("battery_chemistry") or "").strip().lower()
    if _chem.startswith("na"):
        return "Na"
    if _chem.startswith("li"):
        return "Li"
    # A metal that is only ASSUMED is better than nothing for a mass estimate,
    # where the alternative is no number at all. It is not better than nothing
    # for a word printed on an axis, where the alternative is a neutral word
    # that is true whatever the ion. So it stops here.
    return ""


def half_cycle_labels(params):
    """
    `(charge_label, discharge_label)` for this dataset.

    Neutral "Charge"/"Discharge" for a positive electrode, and ALSO for a
    negative one whose working ion is not established — a caption that says
    "Charge" is merely unspecific, whereas one that says "Lithiation" over a
    sodium cell is wrong.
    """
    if not (params or {}).get("anode_labels_swapped", False):
        return "Charge", "Discharge"
    _ion = working_ion(params)
    if not _ion:
        return "Charge", "Discharge"
    return WORKING_ION_WORDS[_ion]


def rule(title="", width=RULE_WIDTH, char="─"):
    """A horizontal separator, optionally naming the section it opens."""
    if not title:
        return f"{S.RULE}{char * width}{S.RESET}"
    left = char * 2
    right = char * max(2, width - len(title) - 4)
    return f"{S.RULE}{left}{S.RESET} {S.BOLD}{title}{S.RESET} {S.RULE}{right}{S.RESET}"


def heading(text, width=RULE_WIDTH):
    """A banner for the thing being worked on — a dataset, a stage."""
    pad = f" {text} ".ljust(width)
    return f"{S.HEADER}{pad}{S.RESET}"


def section(text):
    """A labelled question or sub-stage inside a block."""
    return f"{S.SECTION}{text}{S.RESET}"


# How wide a three-column `entry` line may run before its note drops to a
# wrapped block underneath. Wider than RULE_WIDTH on purpose: a separator that
# ran this long would look like a fault, but a line of text at this width is
# comfortable in a notebook cell and in any terminal worth using.
ENTRY_LINE_WIDTH = 96


def entry(label, value, note="", label_width=22, value_width=16, indent=2):
    """One aligned `label   value   note` line.

    NOT called `field`: seven modules do `from dataclasses import field`, and
    in the flat notebook — one namespace — whichever ran last would win. The
    build guard now catches that class of collision, but a name this generic
    should not be spent on a formatting helper in the first place.

    The three columns are the whole point: a reader scanning for one number
    should never have to read a sentence to find it.

    THE NOTE COLUMN WRAPS. It used to run on to whatever length it was given,
    which put 130-character lines into a notebook cell that shows about 78 —
    so the end of the sentence was off the side of the screen and read by
    nobody, which is the same defect `bullet` was written to avoid. Overflow
    continues on the next line, aligned under the note, so the value column
    stays scannable.
    """
    import textwrap

    pad = " " * indent
    # A LABEL LONGER THAN ITS COLUMN must not run into the value. `f"{x:<22}"`
    # pads a short label and does nothing at all to a long one, so
    # "discharge across all cycles" and "median 1" printed as
    # "discharge across all cyclesmedian 1". Two spaces is the minimum
    # separation, and the value column simply starts later on that row.
    lab = f"{label:<{label_width}}"
    if len(label) >= label_width:
        lab = label + "  "
    val = f"{value:<{value_width}}" if note else str(value)
    out = f"{pad}{S.DIM}{lab}{S.RESET}{val}"
    if not note:
        return out.rstrip()
    left = indent + label_width + max(len(str(value)), value_width) + 2
    # Against ENTRY_LINE_WIDTH, not RULE_WIDTH. The rule is a deliberate 76
    # because a full-width separator any wider looks like a fault; a line of
    # text may run past it, and forcing three columns into 76 dropped notes
    # of 38 characters onto their own line for no reason.
    room = ENTRY_LINE_WIDTH - left
    if len(str(note)) <= room:
        return (out + f"  {S.DIM}{note}{S.RESET}").rstrip()
    # TOO LONG FOR THE THIRD COLUMN. Wrapping it there would leave a 34-column
    # gutter and break a sentence every four words, so it drops WHOLE onto
    # continuation lines under the label instead — the same block `bullet`
    # produces, and the value column stays scannable above it.
    block = "\n".join(
        f"{' ' * (indent + label_width)}{S.DIM}{ln}{S.RESET}"
        for ln in textwrap.wrap(
            str(note), width=max(20, RULE_WIDTH - indent - label_width)
        )
    )
    return out.rstrip() + "\n" + block


_MARK = {"ok": ("✓", "OK"), "caution": ("!", "CAUTION"), "bad": ("✗", "BAD")}


def verdict(level, text, label="verdict", label_width=22, indent=2):
    """The line a reader looks for first: did this pass, and what does it mean.

    `level` is 'ok', 'caution' or 'bad'. The glyph and the colour agree, and
    the text says it in words as well — so the line survives being pasted
    into an email, a log, or a colourblind reader's terminal.
    """
    import textwrap

    glyph, _ = _MARK.get(level, ("", ""))
    col = {"ok": S.OK, "caution": S.CAUTION, "bad": S.BAD}.get(level, "")
    pad = " " * indent
    lab = f"{label:<{label_width}}"
    # WRAPPED, like `entry`'s note. A verdict is the line a reader looks for
    # first and it was the one line that could run off the side of the cell.
    body = f"{glyph} {text}"
    room = max(20, ENTRY_LINE_WIDTH - indent - label_width)
    lines = textwrap.wrap(body, width=room) or [body]
    out = f"{pad}{S.DIM}{lab}{S.RESET}{col}{lines[0]}{S.RESET}"
    cont = " " * (indent + label_width)
    for extra in lines[1:]:
        out += f"\n{cont}{col}{extra}{S.RESET}"
    return out


def bullet(text, indent=4, label_width=22, width=RULE_WIDTH):
    """Continuation lines under a field or verdict, aligned to its value.

    WRAPPED at the rule width. An unwrapped explanation runs off the side of
    a notebook cell and is then read by nobody, which defeats the point of
    having written it.
    """
    import textwrap

    pad = " " * (indent + label_width - 2)
    lines = textwrap.wrap(str(text), width=max(20, width - len(pad))) or [""]
    return "\n".join(f"{pad}{S.DIM}{ln}{S.RESET}" for ln in lines)


def image_format(params, requested=None):
    """The image format the OPERATOR asked for, in Cell 3b.

    Lives here because BOTH figure modules need it and, defined in each, the
    flat notebook kept only whichever cell ran last — which is what the
    duplicate-definition build guard exists to catch, and did.

    Every plotting function took a `file_format='png'` argument and used it,
    but no notebook cell ever passed one and only three of `cycling.py`'s
    functions fell back to the per-dataset answer. So 42 of a run's figures
    came out PNG whatever was chosen. An explicit argument still wins.
    """
    if requested:
        return str(requested)
    return str((params or {}).get("file_format", "png") or "png")


def saved(path):
    """One 'saved' line naming the FILE, not reciting the path.

    Thirty-five of these across the two figure modules printed the absolute
    Windows path in full, once per figure. The run folder is printed once, by
    Cell 3b, and nothing else about the path changes between figures.
    """
    import os as _os

    print(entry("saved", _os.path.basename(str(path))))


def nice_axis_limit(vmax, target_ticks=6):
    """Round an axis maximum up to a round number, with a tick step to match.

    A common axis computed straight from the data lands on 143 mAh/g, and the
    reader is then left deciding whether a curve ending at 136 is meaningfully
    short of the frame or just short of an arbitrary number. Rounding to 150
    with ticks every 25 answers that by inspection: the gridline the curve
    stops near is a number a person would have chosen.

    Steps come from the 1 / 2 / 2.5 / 5 ladder, which is what produces tick
    labels like 25, 50, 75 rather than 23.8, 47.7. Returns
    `(limit, step)`, or `(vmax, None)` if there is nothing sensible to round.

    The limit is always strictly greater than `vmax`, so the largest data
    point never sits on the frame; a maximum that is already an exact
    multiple of the step is given one more step rather than being touched.
    """
    try:
        vmax = float(vmax)
    except TypeError, ValueError:
        return vmax, None
    if not math.isfinite(vmax) or vmax <= 0:
        return vmax, None
    target_ticks = max(2, int(target_ticks))
    raw = vmax / target_ticks
    mag = 10.0 ** math.floor(math.log10(raw))
    step = 10.0 * mag
    for m in (1.0, 2.0, 2.5, 5.0):
        if raw <= m * mag:
            step = m * mag
            break
    limit = math.ceil(vmax / step) * step
    if limit <= vmax * (1.0 + 1e-9):
        limit += step
    return limit, step
