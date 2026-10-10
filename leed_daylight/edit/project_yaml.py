# leed_daylight/edit/project_yaml.py
"""Write project.yaml as the template lays it out (wizard spec 7.2), and the
pre-filled standard values (wizard spec 6.1, W3, W10).

``render(raw)`` writes a loaded project.yaml mapping back in the template's
form: the template's header comments; each template section in template
order; each template key on its template line with only the value replaced,
its comment kept at the template's column (or one space after a longer
value). A key missing from ``raw`` is written ``FILL_ME``, never the
template's own value, except a role line of ``reflectance:``: the template
says to delete one to leave that role as modelled, so a role the section
leaves out stays out. Keys the template holds only as comments, or does not
list, follow their section's template lines in the order the file had them.
The optional sections ``views:``, ``report:`` and ``atlas:`` follow when
present, then any top-level section the loader ignores, in file order.
Comments a person added by hand are not kept; ``hand_comments`` finds them
so the page can warn first.

The pre-filled values are the thirteen of spec 6.1, each with the source the
template's own comment states; ``tests/test_wizard_project_yaml.py`` holds
the table against the template's comments. Nothing else is ever pre-filled.
"""
from __future__ import annotations

import copy
import dataclasses
import datetime
import functools
import json
import math
import pathlib
import re
from collections.abc import Collection, Mapping

from leed_daylight.constants import FILL_ME

from ..drivers.init import PROJECT_TEMPLATE

#: The template ``init`` writes; the writer lays every project.yaml out as it.
TEMPLATE: pathlib.Path = PROJECT_TEMPLATE

#: The optional top-level sections, in the order they are written.
OPTIONAL_SECTIONS = ("views", "report", "atlas")


@dataclasses.dataclass(frozen=True)
class TemplateKey:
    """One key line of the template: its dotted path, its comment (the text
    after ``#``) and its value as the template writes it."""
    path: str
    comment: str
    value: str


@dataclasses.dataclass(frozen=True)
class Prefill:
    """One pre-filled standard value (spec 6.1): the key, the value, the
    source the page shows, and the group one Confirm covers."""
    key: str
    value: object
    source: str
    group: str


PREFILLS: tuple[Prefill, ...] = (
    Prefill("grid.spacing_m", 0.6, "LEED v4.1: at most 0.60 m", "grid"),
    Prefill("grid.work_plane_m", 0.762, "LEED v4.1: 30 in", "grid"),
    Prefill("analysis.aggregation", "floor_area", "LEED v4.1", "analysis.aggregation"),
    Prefill("option2.equinox_dates", ("03-21", "09-21"), "LEED v4.1 Option 2", "option2"),
    Prefill("option2.clearest_day_window_days", 15, "LEED v4.1 Option 2", "option2"),
    Prefill("option2.times", ("09:00", "15:00"), "LEED v4.1 Option 2", "option2"),
    Prefill("option2.lux_min", 300, "LEED v4.1 Option 2", "option2"),
    Prefill("option2.lux_max", 3000, "LEED v4.1 Option 2", "option2"),
    Prefill("option2.sky", "weather_file", "tool method (Option 2 spec)", "option2"),
    Prefill("option2.blinds", "exclude", "LEED v4.1 Option 2", "option2"),
    Prefill("option2.time_basis", "weather_file_local", "decision 40", "option2"),
    Prefill("sky.basis", "Reinhart", "tool method", "sky"),
    Prefill("sky.MF", 4, "tool method", "sky"),
)

#: The groups a person confirms, one Confirm each (spec 6.1).
GROUPS = ("grid", "analysis.aggregation", "option2", "sky")


def group_keys(group: str) -> tuple[str, ...]:
    """The keys of one group, in table order."""
    if group not in GROUPS:
        raise ValueError(f"unknown pre-filled group {group!r}; the groups are {', '.join(GROUPS)}")
    return tuple(p.key for p in PREFILLS if p.group == group)


_MISSING = object()

#: Template keys a section may leave out, by section: a ``reflectance``
#: role line deleted leaves that role as modelled (the template's own
#: comment), so it is not written back as ``FILL_ME``. ``policy`` is
#: always written.
_MAY_LEAVE_OUT = {"reflectance": frozenset({"ceiling", "wall", "floor", "context", "ground"})}


def _is_fill_me(value) -> bool:
    return isinstance(value, str) and value.strip() == FILL_ME


def _holds_fill_me(value) -> bool:
    if isinstance(value, Mapping):
        return any(_holds_fill_me(v) for v in value.values())
    if isinstance(value, list):
        return any(_holds_fill_me(v) for v in value)
    return _is_fill_me(value)


def get_path(raw: Mapping, path: str) -> object:
    """The value at a dotted path, or ``FILL_ME`` when any part is missing."""
    value: object = raw
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return FILL_ME
        value = value[part]
    return value


def set_path(raw: dict, path: str, value: object) -> None:
    """Set the value at a dotted path, making (or replacing with) a mapping
    at each part on the way that is not one."""
    parts = path.split(".")
    target = raw
    for part in parts[:-1]:
        if not isinstance(target.get(part), dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def group_confirmed(raw: Mapping, group: str) -> bool:
    """A group counts as confirmed when every one of its keys already holds
    a real value (spec 6.1): saving then never turns it back into FILL_ME."""
    return all(not _holds_fill_me(get_path(raw, key)) for key in group_keys(group))


def apply_prefills(raw: Mapping, confirmed: Collection[str]) -> dict:
    """A deep copy of ``raw`` in which each key of each confirmed group that
    is still FILL_ME (or missing) holds its pre-filled value. Every other
    key, a real value in a confirmed group included, is left as it is."""
    out = copy.deepcopy(dict(raw))
    for group in confirmed:
        for key in group_keys(group):
            current = get_path(out, key)
            if _is_fill_me(current):
                value = next(p.value for p in PREFILLS if p.key == key)
                set_path(out, key, list(value) if isinstance(value, tuple) else value)
    return out


# --- Comments ------------------------------------------------------------

#: Characters after which a quote opens a quoted scalar.
_QUOTE_OPENERS = " \t[{,:-"


def _comment_start(line: str) -> int:
    """The column of the ``#`` that starts the line's comment, or -1. A
    ``#`` starts a comment outside quotes, at the line start or after a
    space."""
    quote = ""
    i = 0
    while i < len(line):
        char = line[i]
        if quote:
            if char == "\\" and quote == '"':
                i += 2
                continue
            if char == quote:
                quote = ""
        elif char in "\"'" and (i == 0 or line[i - 1] in _QUOTE_OPENERS):
            quote = char
        elif char == "#" and (i == 0 or line[i - 1] in " \t"):
            return i
        i += 1
    return -1


def comments(text: str) -> tuple[str, ...]:
    """The comment of each line that has one (from its ``#``, stripped), in order."""
    found = []
    for line in text.splitlines():
        col = _comment_start(line)
        if col >= 0:
            found.append(line[col:].strip())
    return tuple(found)


def hand_comments(text: str) -> tuple[str, ...]:
    """The comments of ``text`` that the template does not have, in order:
    what a wizard save would lose."""
    template = set(comments(_template_text()))
    return tuple(c for c in comments(text) if c not in template)


# --- The template's lines ------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _Line:
    text: str
    indent: int
    key: str | None      # None for a comment-only or blank line
    value: str           # the template's value text; "" opens a mapping
    comment_col: int     # -1 when the line has no comment


_KEY_LINE = re.compile(r"^( *)([A-Za-z_][A-Za-z0-9_]*):(?: +(.*?))? *$")


@functools.lru_cache(maxsize=1)
def _template_text() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def _template_lines() -> tuple[_Line, ...]:
    lines = []
    for text in _template_text().splitlines():
        indent = len(text) - len(text.lstrip(" "))
        if not text.strip() or text.strip().startswith("#"):
            lines.append(_Line(text, indent, None, "", -1))
            continue
        col = _comment_start(text)
        match = _KEY_LINE.match(text if col < 0 else text[:col])
        lines.append(_Line(text, indent, match.group(2), match.group(3) or "", col))
    return tuple(lines)


def template_keys() -> tuple[TemplateKey, ...]:
    """Every key line of the template that holds a value, in template order."""
    found = []
    stack: list[tuple[int, str]] = []
    for line in _template_lines():
        if line.key is None:
            continue
        while stack and stack[-1][0] >= line.indent:
            stack.pop()
        path = ".".join([name for _, name in stack] + [line.key])
        if line.value:
            comment = line.text[line.comment_col + 1:].strip() if line.comment_col >= 0 else ""
            found.append(TemplateKey(path, comment, line.value))
        else:
            stack.append((line.indent, line.key))
    return tuple(found)


# --- Values --------------------------------------------------------------

_BARE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: Words YAML 1.1 reads as a boolean or null: never written as a bare key.
_YAML_WORDS = frozenset({"yes", "no", "true", "false", "on", "off", "null"})


def _key(key) -> str:
    if isinstance(key, str):
        if _BARE_KEY.match(key) and key.lower() not in _YAML_WORDS:
            return key
        return json.dumps(key, ensure_ascii=False)
    return _flow(key)


def _float(value: float) -> str:
    if math.isnan(value):
        return ".nan"
    if math.isinf(value):
        return ".inf" if value > 0 else "-.inf"
    text = repr(value)
    mantissa, e, exponent = text.partition("e")
    if e and "." not in mantissa:
        # YAML 1.1 reads a number with an exponent but no dot (1e-05) as text.
        text = f"{mantissa}.0e{exponent}"
    return text


def _flow(value) -> str:
    """One value in YAML flow style."""
    if isinstance(value, str):
        return FILL_ME if _is_fill_me(value) else json.dumps(value, ensure_ascii=False)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _float(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_flow(item) for item in value) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{_key(k)}: {_flow(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    raise TypeError(f"cannot write a {type(value).__name__} in project.yaml: {value!r}")


def _keyed(line: _Line, value_text: str) -> str:
    """A template key line with only its value replaced; its comment stays
    at the template's column, or one space after a longer value."""
    head = f"{' ' * line.indent}{line.key}: {value_text}"
    if line.comment_col < 0:
        return head
    comment = line.text[line.comment_col:]
    if len(head) < line.comment_col:
        return head.ljust(line.comment_col) + comment
    return f"{head} {comment}"


# --- Rendering -----------------------------------------------------------

def _block_end(lines: tuple[_Line, ...], i: int, hi: int) -> int:
    """The index after key line ``i``'s own lines: the next key line at its
    indent or less, or ``hi``."""
    j = i + 1
    while j < hi and not (lines[j].key is not None and lines[j].indent <= lines[i].indent):
        j += 1
    return j


def _trim_blanks(lines: tuple[_Line, ...], lo: int, hi: int) -> int:
    while hi > lo and not lines[hi - 1].text.strip():
        hi -= 1
    return hi


def _render_mapping(lines: tuple[_Line, ...], lo: int, hi: int, values, indent: int,
                    out: list[str], *, extras: bool = True, leave_out: frozenset[str] = frozenset()) -> None:
    """Write template lines ``lo:hi`` (one mapping's body, keys at
    ``indent``) with the values of ``values``, leaving out a key of
    ``leave_out`` that ``values`` lacks; then, with ``extras``, the keys of
    ``values`` the template does not have, in their loaded order."""
    values = values if isinstance(values, Mapping) else {}
    written = set()
    i = lo
    while i < hi:
        line = lines[i]
        if line.key is None:
            out.append(line.text)
            i += 1
            continue
        j = _block_end(lines, i, hi)
        written.add(line.key)
        value = values.get(line.key, _MISSING)
        if value is _MISSING and line.key in leave_out:
            i = j
            continue
        if not line.value and (value is _MISSING or isinstance(value, Mapping)):
            out.append(line.text)
            body_hi = _trim_blanks(lines, i + 1, j)
            omissible = frozenset()
            if indent == 0 and value is not _MISSING:
                omissible = _MAY_LEAVE_OUT.get(line.key, frozenset())
            _render_mapping(lines, i + 1, body_hi, {} if value is _MISSING else value, indent + 2, out,
                            leave_out=omissible)
            out.extend(line.text for line in lines[body_hi:j])
        else:
            out.append(_keyed(line, FILL_ME if value is _MISSING else _flow(value)))
            if line.value:
                out.extend(line.text for line in lines[i + 1:j])
            else:
                # A template mapping the file holds as one value: its
                # template lines would not fit under it.
                out.extend(line.text for line in lines[_trim_blanks(lines, i + 1, j):j])
        i = j
    if extras:
        for key, value in values.items():
            if key not in written:
                out.append(f"{' ' * indent}{_key(key)}: {_flow(value)}")


def _section(name, value, out: list[str]) -> None:
    """An optional or unknown top-level section, in block style, after a
    blank line; ``atlas.drawings`` as a block list of block mappings."""
    out.append("")
    if not isinstance(value, Mapping) or not value:
        out.append(f"{_key(name)}: {_flow(value)}")
        return
    out.append(f"{_key(name)}:")
    for key, item in value.items():
        entries = item if isinstance(item, list) else None
        if name == "atlas" and key == "drawings" and entries and all(
            isinstance(e, Mapping) and e for e in entries
        ):
            out.append("  drawings:")
            for entry in entries:
                for n, (entry_key, entry_value) in enumerate(entry.items()):
                    lead = "    - " if n == 0 else "      "
                    out.append(f"{lead}{_key(entry_key)}: {_flow(entry_value)}")
        else:
            out.append(f"  {_key(key)}: {_flow(item)}")


def render(raw: Mapping) -> str:
    """project.yaml's text for the loaded mapping ``raw``, in the template's
    form (spec 7.2)."""
    lines = _template_lines()
    out: list[str] = []
    _render_mapping(lines, 0, len(lines), raw, 0, out, extras=False)
    sections = {line.key for line in lines if line.key is not None and line.indent == 0}
    for name in OPTIONAL_SECTIONS:
        if name in raw:
            _section(name, raw[name], out)
    for name, value in raw.items():
        if name not in sections and name not in OPTIONAL_SECTIONS:
            _section(name, value, out)
    return "\n".join(out) + "\n"
