# leed_mcp/adapters/narrative_file.py
"""glare_narrative.md for the ``inputs`` tool (spec 6.3).

``deliver/narrative.py`` imports ``python-docx`` when it is loaded, and the
``report`` extra is optional (spec 3), so this server must not import it.
The few text rules it needs are ported here (PORTED.md lists each with its
source lines); ``tests/mcp/test_mcp_inputs_read.py`` holds them equal to
the pipeline's when the extra is installed.
"""
from __future__ import annotations

import pathlib
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from leed_mcp.adapters.input_files import Written, write_file
from leed_mcp.envelope import refusal
from leed_mcp.errors import Refused

__all__ = ["AUTO_BLOCKS", "MARK", "NAME", "block_state", "body_span", "count_marks", "edit", "is_space_row",
           "marks", "measures", "narrative_content", "split_row", "update_narrative"]

#: The file, one per project folder, shared by every case.
NAME = "glare_narrative.md"
#: The generated blocks of glare_narrative.md.
AUTO_BLOCKS = ("results", "spaces")
#: An open item a person fills in (``narrative._MARK``).
MARK = re.compile(r"\[(?:ADD|CONFIRM)\b[^\]]*\]")
#: A cell boundary: a ``|`` not escaped (``narrative._PIPE``).
_PIPE = re.compile(r"(?<!\\)\|")
#: A prose key: a mark index in ASCII digits.
_INDEX = re.compile(r"[0-9]+")


def count_marks(text: str) -> int:
    """Every ``[ADD ...]`` and ``[CONFIRM ...]`` mark (``narrative.count_marks``)."""
    return len(MARK.findall(text))


def split_row(line: str, maxsplit: int = 0) -> list[str]:
    """The cells of one markdown table row, split on unescaped ``|`` only
    (``narrative._split_row``)."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    return [c.strip() for c in _PIPE.split(line, maxsplit)]


def is_space_row(cells: list[str]) -> bool:
    """A row of the spaces table, not its head or rule (``narrative._measures``)."""
    return len(cells) == 6 and cells[1] not in ("Space", "---") and not set(cells[1]) <= {"-", ":"}


def measures(body: str) -> dict[str, str]:
    """Space name -> its "Glare measures" cell, in the ``auto:spaces`` body
    (``narrative._measures``)."""
    kept = {}
    for line in body.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = split_row(line, 5)
        if is_space_row(cells):
            kept[cells[1].replace("\\|", "|")] = cells[5]
    return kept


def block_state(text: str, name: str) -> str:
    """``missing``, ``well_formed`` or ``malformed``, by counting the block's
    markers (``narrative._block_state``)."""
    open_tag, close_tag = f"<!-- auto:{name} -->", f"<!-- /auto:{name} -->"
    opens, closes = text.count(open_tag), text.count(close_tag)
    if opens == 0 and closes == 0:
        return "missing"
    if opens == 1 and closes == 1 and text.index(open_tag) < text.index(close_tag):
        return "well_formed"
    return "malformed"


def body_span(text: str, name: str) -> tuple[int, int, str]:
    """``(start, end, lead)`` of a well-formed block's body
    (``narrative._body_span``)."""
    open_tag, close_tag = f"<!-- auto:{name} -->", f"<!-- /auto:{name} -->"
    after_open = text.index(open_tag) + len(open_tag)
    end = text.index(close_tag, after_open)
    line_end = text.find("\n", after_open, end)
    if line_end == -1:
        return after_open, end, "\n"
    return line_end + 1, end, ""


def _auto_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` of each well-formed auto block, markers included."""
    spans = []
    for name in AUTO_BLOCKS:
        if block_state(text, name) == "well_formed":
            start = text.index(f"<!-- auto:{name} -->")
            close = f"<!-- /auto:{name} -->"
            spans.append((start, text.index(close, start) + len(close)))
    return spans


def marks(text: str) -> list[dict[str, Any]]:
    """The marks outside the auto blocks, in file order: ``index`` (from 1),
    the mark, the line it is on, and its ``start`` and ``end`` in ``text``."""
    spans = _auto_spans(text)
    found = []
    for match in MARK.finditer(text):
        if any(start <= match.start() < end for start, end in spans):
            continue
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        found.append({"index": len(found) + 1, "mark": match.group(0),
                      "line": text[line_start:line_end if line_end >= 0 else len(text)],
                      "start": match.start(), "end": match.end()})
    return found


def narrative_content(text: str) -> tuple[dict[str, Any], list[str], int]:
    """``(content, problems, marks_left)``: the text, the "Glare measures"
    cell of each space in ``auto:spaces``, and the marks outside the auto
    blocks; ``marks_left`` counts every mark, as ``deliver report`` does."""
    problems, spaces = [], []
    state = block_state(text, "spaces")
    if state == "well_formed":
        start, end, _ = body_span(text, "spaces")
        spaces = [{"space": space, "measures": cell} for space, cell in measures(text[start:end]).items()]
    else:
        problems.append(f"glare_narrative.md: the auto:spaces markers are {state}; "
                        "deliver(action='report') writes them")
    found = [{key: m[key] for key in ("index", "mark", "line")} for m in marks(text)]
    return {"text": text, "spaces": spaces, "marks": found}, problems, count_marks(text)


def _breaks_line(ch: str) -> bool:
    """Whether ``ch`` ends a line for ``str.splitlines`` (``\\n``, ``\\r``,
    ``\\x0b``, ``\\x0c``, ``\\x1c``-``\\x1e``, ``\\x85``, U+2028, U+2029)."""
    return len(("x" + ch + "x").splitlines()) > 1


def _text(value: str, where: str, *, one_line: bool) -> str:
    """Text to put in the file: no auto-block marker and no control
    character but a tab; a table cell (``one_line``) also no line break of
    any kind, prose may hold ``\\n``."""
    field = where.split("[")[0]
    if "<!-- auto:" in value or "<!-- /auto:" in value:
        raise Refused(refusal("INPUT_INVALID", f"{where}: the text holds an auto-block marker; write it without "
                              "'<!-- auto:' or '<!-- /auto:'.", field=field))
    if one_line:
        bad = [ch for ch in value if ch != "\t" and (_breaks_line(ch) or unicodedata.category(ch) == "Cc")]
        if bad:
            raise Refused(refusal("INPUT_INVALID", f"{where}: a Glare measures cell is one line; write it without "
                                  f"line breaks or control characters (found U+{ord(bad[0]):04X}).", field=field))
        return value.strip()
    bad = [ch for ch in value if ch not in "\t\n" and unicodedata.category(ch) == "Cc"]
    if bad:
        raise Refused(refusal("INPUT_INVALID", f"{where}: the text holds a control character (U+{ord(bad[0]):04X}); "
                              "write it without one (a tab and a line break are kept).", field=field))
    return value


def _edit_measures(text: str, given: Mapping[str, str]) -> str:
    """Replace the sixth cell of each named space's row in ``auto:spaces``,
    escaping ``|`` as ``\\|``, the cell ``deliver report`` carries forward."""
    if block_state(text, "spaces") != "well_formed":
        raise Refused(refusal("INPUT_INVALID", "glare_narrative.md: the auto:spaces block is missing or its "
                              "markers are malformed; deliver(action='report') writes it.", field="measures"))
    start, end, _ = body_span(text, "spaces")
    body = text[start:end]
    known = measures(body)
    unknown = sorted(set(given) - set(known))
    if unknown:
        raise Refused(refusal("INPUT_INVALID", f"measures: {', '.join(map(repr, unknown))} not in the auto:spaces "
                              f"table; its spaces are {', '.join(known) or 'none'}.", field="measures"))
    cells = {space: _text(value, f"measures[{space}]", one_line=True).replace("|", "\\|")
             for space, value in given.items()}
    out = []
    for line in body.splitlines(keepends=True):
        row = line.rstrip("\n")
        parts = split_row(row, 5) if row.strip().startswith("|") else []
        space = parts[1].replace("\\|", "|") if is_space_row(parts) else None
        if space in cells:
            end_of_line = "\n" if line.endswith("\n") else ""
            line = "| " + " | ".join(parts[:5]) + f" | {cells[space]} |" + end_of_line
        out.append(line)
    return text[:start] + "".join(out) + text[end:]


def _edit_prose(text: str, given: Mapping[str, str]) -> str:
    """Replace each mark ``given`` names by its index (from ``marks``)."""
    found = marks(text)
    edits = []
    seen: dict[int, str] = {}
    for key, value in given.items():
        if not _INDEX.fullmatch(key) or not 1 <= int(key) <= len(found):
            raise Refused(refusal("INPUT_INVALID", f"prose: {key!r} is not a mark index; the marks outside the "
                                  f"auto blocks are 1 to {len(found)} (inputs action='read').", field="prose"))
        index = int(key)
        if index in seen:
            raise Refused(refusal("INPUT_INVALID", f"prose: {seen[index]!r} and {key!r} name the same mark "
                                  f"({index}); give each mark once.", field="prose"))
        seen[index] = key
        edits.append((found[index - 1], _text(value, f"prose[{key}]", one_line=False)))
    for mark, value in sorted(edits, key=lambda e: e[0]["start"], reverse=True):
        text = text[:mark["start"]] + value + text[mark["end"]:]
    return text


def edit(text: str, measures_: Mapping[str, str] | None, prose: Mapping[str, str] | None) -> str:
    """``text`` with the measures cells and the prose marks replaced;
    nothing else in it changes."""
    if measures_:
        text = _edit_measures(text, measures_)
    if prose:
        text = _edit_prose(text, prose)
    return text


def update_narrative(folder: pathlib.Path, expected: str | None, measures_: Mapping[str, str] | None,
                     prose: Mapping[str, str] | None) -> tuple[Written, bytes]:
    """Edit glare_narrative.md; the ``.part`` file must keep every auto
    block as it was (well formed, malformed or missing)."""
    held: dict[str, Any] = {}

    def build(old: bytes | None, raw: Any) -> bytes:
        if old is None:
            raise Refused(refusal("FILE_MISSING", f"{folder}: no glare_narrative.md; deliver(action='report') "
                                  "writes it.", field="file"))
        text = old.decode("utf-8")
        held["states"] = [block_state(text, name) for name in AUTO_BLOCKS]
        held["data"] = edit(text, measures_, prose).encode("utf-8")
        return held["data"]

    def check(part: pathlib.Path) -> None:
        text = part.read_text(encoding="utf-8")
        if [block_state(text, name) for name in AUTO_BLOCKS] != held["states"]:
            raise ValueError("glare_narrative.md: the edit would change the auto blocks; nothing was written")

    written = write_file(folder, NAME, expected, build, check)
    return written, held["data"]
