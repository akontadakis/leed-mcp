# leed_mcp/adapters/input_read.py
"""What ``inputs(action="read")`` returns for each file (spec 6.3).

Each reader takes the bytes already read (``input_files.read_bytes``), so
the digest, the content and the problems all describe the same bytes. A
file that cannot be parsed still answers: its content is None and the
parse error is a problem, so the model can still replace it.

Every CSV cell is returned as text, a name column (``room``, ``modifier``,
``floor``) included, even when it reads as a number.
"""
from __future__ import annotations

import csv
import io
import pathlib
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import yaml

from leed_daylight.edit.project_yaml import PREFILLS, hand_comments
from leed_daylight.project.materials import load_materials
from leed_daylight.project.spaces import load_spaces
from leed_daylight.project.splits import read_splits
from leed_daylight.scene.input_file import InputFileError
from leed_mcp.adapters.input_files import as_file, digest, jsonable, load_yaml, read_bytes, read_project_raw
from leed_mcp.adapters.narrative_file import narrative_content
from leed_mcp.adapters.project_state import fill_me_keys
from leed_mcp.envelope import refusal
from leed_mcp.errors import Refused, one_line

__all__ = [
    "FILES", "NAMES", "READ_ROWS_MAX", "Content", "content_of", "credit_rules_name", "project_content", "read",
    "splits_content", "table", "table_content",
]

#: The files ``inputs`` reads, by the name the tool takes (spec 6.3).
FILES = ("project", "materials", "spaces", "splits", "credit_rules", "glare_narrative")
#: The file each name stands for; ``credit_rules`` is project.yaml's
#: ``standard.credit_rules``.
NAMES = {"project": "project.yaml", "materials": "materials.csv", "spaces": "spaces.csv",
         "splits": "splits.csv", "glare_narrative": "glare_narrative.md"}
#: The most CSV rows (split lines for splits.csv) a read returns. Chosen.
READ_ROWS_MAX = 5000
#: Cells that tick a case column (``project/spaces.py`` ``_TRUE``).
_TICKED = frozenset({"1", "yes", "true"})


class Content(NamedTuple):
    """One file's answer: its content (None when absent or unreadable), the
    problems found reading it, and the file's extra fields."""

    content: Any
    problems: list[str]
    extra: dict[str, Any]


def table(data: bytes) -> tuple[list[str], list[list[str]]]:
    """A CSV file's header (each name stripped) and its rows, every cell as
    text; blank lines left out, short rows padded. ``ValueError`` when it
    is not UTF-8 or has no header."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"not UTF-8 text (byte 0x{data[exc.start]:02x}); save it again as CSV UTF-8") from None
    lines = [row for row in csv.reader(io.StringIO(text, newline="")) if row]
    if not lines:
        raise ValueError("empty file; expected a header row")
    header = [name.strip() for name in lines[0]]
    return header, [row + [""] * (len(header) - len(row)) for row in lines[1:]]


def table_content(folder: pathlib.Path, name: str, data: bytes,
                  check: Callable[[pathlib.Path], object]) -> Content:
    """A CSV file as ``{"columns", "rows"}`` (rows as ``{column: text}``),
    and what ``check`` (the pipeline's loader) refuses in it."""
    try:
        header, rows = table(data)
    except (ValueError, csv.Error) as exc:
        return Content(None, [f"{name}: {one_line(exc)}"], {})
    problems = []
    try:
        with as_file(folder, name, data) as path:
            check(path)
    except (OSError, ValueError) as exc:
        problems.append(one_line(exc))
    return Content({"columns": header, "rows": [dict(zip(header, row)) for row in rows]}, problems, {})


def _yaml(data: bytes) -> tuple[Any, list[str]]:
    try:
        return jsonable(load_yaml(data.decode("utf-8"))), []
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        return None, [f"not valid YAML: {one_line(exc)}"]


def project_content(data: bytes) -> Content:
    """project.yaml as its mapping, the keys still ``FILL_ME``, the comments
    a write would lose (``hand_comments``) and the 13 standard values of
    ``PREFILLS`` as suggestions, each with its source (spec 6.3)."""
    raw, problems = _yaml(data)
    if raw is not None and not isinstance(raw, dict):
        raw, problems = None, ["project.yaml must be a mapping at the top level"]
    elif raw is None and not problems:
        problems = ["project.yaml is empty; it must be a mapping at the top level"]
    suggestions = [{"key": p.key, "value": jsonable(p.value), "source": p.source} for p in PREFILLS]
    try:
        hand = list(hand_comments(data.decode("utf-8")))
    except UnicodeDecodeError:
        hand = []
    return Content(raw, problems, {"fill_me": fill_me_keys(raw) if raw is not None else None,
                                   "hand_comments": hand, "suggestions": suggestions})


def splits_content(folder: pathlib.Path, data: bytes) -> Content:
    """splits.csv as ``{"lines": [{"level_z", "points", "note"}]}``, read by
    the pipeline's ``read_splits``."""
    try:
        with as_file(folder, "splits.csv", data) as path:
            lines = read_splits(path)
    except (OSError, ValueError) as exc:
        return Content(None, [one_line(exc)], {})
    return Content({"lines": [{"level_z": line.level_z, "points": line.points.tolist(), "note": line.note}
                              for line in lines]}, [], {})


def credit_rules_name(folder: pathlib.Path) -> str:
    """The credit rules file project.yaml names (``standard.credit_rules``),
    relative to the folder; ``credit_rules.yaml`` while it is not set.
    ``PATH_OUTSIDE_ROOTS`` when it leads outside the folder (spec 10)."""
    try:
        _, raw = read_project_raw(folder)
    except InputFileError:
        raise  # not a plain file, or too large: PATH_UNSAFE (errors.MAPPING)
    except (ValueError, yaml.YAMLError):
        raw = None
    standard = raw.get("standard") if isinstance(raw, dict) else None
    value = standard.get("credit_rules") if isinstance(standard, dict) else None
    if not isinstance(value, str) or not value.strip() or value.strip() == "FILL_ME":
        return "credit_rules.yaml"
    resolved = (folder / value).resolve()
    if folder not in resolved.parents:
        raise Refused(refusal("PATH_OUTSIDE_ROOTS", f"standard.credit_rules ({resolved}) is outside the "
                              f"project folder ({folder}); it is not read.", field="file"))
    return value


def _cases(columns: Sequence[str], case: str) -> list[str]:
    names = [f"include_{case}", f"views_{case}"]
    present = [n for n in names if n in columns]
    if not present:
        cases = sorted({c.split("_", 1)[1] for c in columns
                        if c.startswith(("include_", "views_")) and c != "views_reason"})
        raise Refused(refusal("CASE_UNKNOWN", f"spaces.csv has no include_{case} or views_{case} column; "
                              f"its cases are {', '.join(cases) or 'none'}.", field="case"))
    return present


def content_of(folder: pathlib.Path, file: str, data: bytes) -> Content:
    """The answer for ``file`` from its bytes ``data``."""
    if file == "project":
        return project_content(data)
    if file in ("materials", "spaces"):
        return table_content(folder, NAMES[file], data, load_materials if file == "materials" else load_spaces)
    if file == "splits":
        return splits_content(folder, data)
    if file == "credit_rules":
        content, problems = _yaml(data)
        return Content(content, problems, {})
    try:
        content, problems, left = narrative_content(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        return Content(None, [f"glare_narrative.md: not UTF-8 text ({one_line(exc)})"], {})
    return Content(content, problems, {"marks_left": left})


def read(folder: pathlib.Path, file: str, case: str | None = None) -> dict[str, Any]:
    """One input file as ``inputs(action="read")`` returns it: ``name``,
    ``sha256``, ``content``, ``problems``, the file's extra fields, and
    ``truncated``: None, or the sentence saying the CSV rows (split lines)
    were cut at ``READ_ROWS_MAX``.
    With ``case`` (spaces only), the rows ticked in that case's
    ``include_`` or ``views_`` column."""
    name = credit_rules_name(folder) if file == "credit_rules" else NAMES[file]
    data = read_bytes(folder, name)
    if data is None:
        return {"name": name, "sha256": None, "content": None, "problems": [], "extra": {}, "truncated": None}
    found = content_of(folder, file, data)
    if case is not None and found.content is not None:
        columns = _cases(found.content["columns"], case)
        rows = [r for r in found.content["rows"] if any(r[c].strip().lower() in _TICKED for c in columns)]
        found = found._replace(content={**found.content, "rows": rows})
    truncated = None
    key = "lines" if file == "splits" else "rows"
    if isinstance(found.content, dict) and len(found.content.get(key) or ()) > READ_ROWS_MAX:
        total = len(found.content[key])
        what = "rows" if case is None else f"rows ticked in case {case}"
        what = "split lines" if key == "lines" else what
        truncated = f"{name} has {total} {what}; only the first {READ_ROWS_MAX} are shown."
        found = found._replace(content={**found.content, key: found.content[key][:READ_ROWS_MAX]})
    return {"name": name, "sha256": digest(data), "content": found.content, "problems": found.problems,
            "extra": found.extra, "truncated": truncated}
