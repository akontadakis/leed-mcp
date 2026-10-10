# leed_mcp/adapters/spaces_file.py
"""``inputs`` on spaces.csv (spec 6.3): ``create`` and ``update``.

``create`` makes one row per room that detection finds now, with the
current splits.csv (never from a stale ``_run/spaces_template.csv``),
labelled as ``scene(action="rooms")`` labels them. The geometry columns
(``floor``, ``area_m2``, the centroid, ``level_z``) come from detection.
The template's fixed placeholders are never written: each room's other
columns come from ``rows`` (keyed by label) or, when the file is replaced,
from the old row that ``matches`` pairs with the room.

Every write is checked as the verbs check spaces.csv: ``load_spaces`` (the
merge checks among them), then ``match_to_rooms`` and ``lay_out``.

``matches`` and ``table_bytes`` are ported from the wizard's
``steps/rooms.py`` (PORTED.md).
"""
from __future__ import annotations

import csv
import io
import math
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

from shapely.geometry import Polygon

from leed_daylight.build.prepare import split_hint
from leed_daylight.drivers.rooms import SPACES_TEMPLATE_HEADER
from leed_daylight.project.config import ProjectError
from leed_daylight.project.layout import lay_out
from leed_daylight.project.spaces import (
    CENTROID_TOLERANCE_M, LEVEL_TOLERANCE_M, load_spaces, match_to_rooms,
)
from leed_daylight.scene.storeys import Room
from leed_mcp.adapters import scene
from leed_mcp.adapters.input_files import Written, check_cell_text, write_file
from leed_mcp.adapters.input_read import table
from leed_mcp.arguments import check_case
from leed_mcp.envelope import refusal, refuse_missing
from leed_mcp.errors import Refused, one_line

__all__ = ["GEOMETRY", "NAME", "create_spaces", "matches", "old_table", "table_bytes", "update_spaces"]

NAME = "spaces.csv"
#: Columns detection fills (spec 6.3).
GEOMETRY = ("floor", "area_m2", "centroid_x", "centroid_y", "level_z")
#: Columns a row may leave out; they are written empty.
_OPTIONAL = ("merge_into", "reason", "views_reason")
#: Columns written ``yes`` / ``no`` from a JSON boolean.
_YES_NO = ("regularly_occupied", "view_preserving_shades")


def table_bytes(header: Sequence[str], rows: Sequence[Sequence[str]]) -> bytes:
    """spaces.csv: CSV with ``\\n`` line ends (wizard ``table_bytes``)."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def _centroid(room: Room) -> tuple[float, float]:
    point = Polygon(room.outline).centroid
    return point.x, point.y


def _float(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def matches(header: Sequence[str], rows: Sequence[Sequence[str]], rooms: Sequence[Room]) -> dict[int, int]:
    """Row index to room index: the one room within the centroid and level
    tolerances of ``match_to_rooms``, taken by no earlier row (wizard
    ``matches``)."""
    x, y, z = header.index("centroid_x"), header.index("centroid_y"), header.index("level_z")
    centres = [_centroid(room) for room in rooms]
    found: dict[int, int] = {}
    taken: set[int] = set()
    for i, row in enumerate(rows):
        cx, cy, level = _float(row[x]), _float(row[y]), _float(row[z])
        if None in (cx, cy, level):
            continue
        near = [j for j, room in enumerate(rooms)
                if abs(room.level_z - level) <= LEVEL_TOLERANCE_M
                and math.hypot(cx - centres[j][0], cy - centres[j][1]) <= CENTROID_TOLERANCE_M]
        if len(near) == 1 and near[0] not in taken:
            found[i] = near[0]
            taken.add(near[0])
    return found


def is_case_column(column: str) -> bool:
    return column.startswith(("include_", "views_")) and column != "views_reason"


def cell(column: str, value: Any, where: str) -> str:
    """A JSON value as the text of one spaces.csv cell: a case column takes
    a boolean (``1``/``0``), ``regularly_occupied`` and
    ``view_preserving_shades`` a boolean (``yes``/``no``), every other
    column text (``input_files.check_cell_text``). A name stays text:
    ``101`` is given as ``"101"``."""
    if is_case_column(column) or column in _YES_NO:
        if not isinstance(value, bool):
            raise Refused(refusal("INPUT_INVALID", f"{where}: {column} takes true or false.", field="rows"))
        return ("1" if value else "0") if is_case_column(column) else ("yes" if value else "no")
    if not isinstance(value, str):
        raise Refused(refusal("INPUT_INVALID", f"{where}: {column} takes text (a name such as 101 is given as "
                              '"101").', field="rows"))
    return check_cell_text(value, where, column).strip()


def user_cells(header: Sequence[str], row: Mapping[str, Any], where: str, skip: Sequence[str] = ()) -> dict[str, str]:
    """The cells a row gives, as text; refuses a geometry column and a
    column the header does not have. Keys in ``skip`` are not cells."""
    out = {}
    for key, value in row.items():
        if key in skip:
            continue
        if key in GEOMETRY:
            raise Refused(refusal("INPUT_INVALID", f"{where}: {key} comes from the scene's rooms and is not "
                                  "given.", field="rows"))
        if key not in header:
            raise Refused(refusal("INPUT_INVALID", f"{where}: spaces.csv has no column {key!r}; its columns "
                                  f"are {', '.join(header)}.", field="rows"))
        out[key] = cell(key, value, where)
    return out


def check(detection) -> Any:
    """The verbs' check of a spaces.csv file against ``detection``. A row
    that matches no room also names this server's way to rebuild them."""
    def run(part: pathlib.Path) -> None:
        spaces = load_spaces(part)
        try:
            matched = match_to_rooms(spaces, detection.rooms, hint=split_hint(detection.rooms))
        except ProjectError as exc:
            raise ProjectError(f"{exc}. inputs(action='create', file='spaces') rebuilds the rows from the "
                               "rooms and keeps the values of each row that still matches one.") from None
        lay_out(spaces, matched)
    return run


def old_table(old: bytes | None) -> tuple[list[str], list[list[str]]] | None:
    """The old file's header and rows, when it has the columns a match needs."""
    if old is None:
        return None
    try:
        header, rows = table(old)
    except (ValueError, csv.Error):
        return None
    need = {"room", "centroid_x", "centroid_y", "level_z", *GEOMETRY}
    return (header, rows) if need <= set(header) else None


def _unmatchable(old: bytes) -> tuple[list[str], list[str]]:
    """For an old file ``old_table`` cannot match: ``(rooms, problems)``.
    Every non-blank room cell is unmatched when the file parses and has a
    ``room`` column; otherwise one problem says nothing was carried."""
    try:
        header, rows = table(old)
    except (ValueError, csv.Error) as exc:
        why = one_line(exc)
    else:
        if "room" in header:
            at = header.index("room")
            return [row[at].strip() for row in rows if row[at].strip()], []
        why = "it has no room column"
    return [], [f"spaces.csv: the old file could not be matched ({why}); its values were not carried; "
                "the backup holds them"]


def create_spaces(folder: pathlib.Path, expected: str | None,
                  rows: Sequence[Any]) -> tuple[Written, bytes, list[str], list[str]]:
    """Write spaces.csv from the rooms detection finds now; returns the
    file, its bytes, the old rows that match no room, and the problems of
    an old file that could not be matched at all."""
    _, _, detection = scene.detected(folder)
    named = scene.labelled(detection.rooms)
    labels = [label for label, _ in named]
    given: dict[str, Mapping[str, Any]] = {}
    for n, row in enumerate(rows, start=1):
        label = row.get("label") if isinstance(row, Mapping) else None
        if label not in labels:
            raise Refused(refusal("INPUT_INVALID", f"rows[{n}]: label {label!r} is not a room of the scene; the "
                                  f"rooms are {', '.join(labels)} (scene action='rooms').", field="rows"))
        if label in given:
            raise Refused(refusal("INPUT_INVALID", f"rows[{n}]: {label} is given twice.", field="rows"))
        given[label] = row
    held: dict[str, Any] = {}

    def build(old: bytes | None, raw: Any) -> bytes:
        found = old_table(old)
        header = list(found[0]) if found else list(SPACES_TEMPLATE_HEADER)
        carried: dict[str, dict[str, str]] = {}
        unmatched: list[str] = []
        problems: list[str] = []
        if old is not None and found is None:
            unmatched, problems = _unmatchable(old)
        if found:
            paired = matches(*found, [room for _, room in named])
            carried = {labels[j]: dict(zip(header, found[1][i])) for i, j in paired.items()}
            unmatched = [r[header.index("room")] for i, r in enumerate(found[1]) if i not in paired]
        required = [c for c in header if c not in GEOMETRY and c not in _OPTIONAL]
        out, missing = [], []
        for label, room in named:
            cells = {k: v for k, v in carried.get(label, {}).items() if k not in GEOMETRY}
            if label in given:
                cells.update(user_cells(header, given[label], f"rows ({label})", skip=("label",)))
            absent = [c for c in required if c not in cells]
            if absent:
                missing.append(f"{label} ({', '.join(absent)})")
                continue
            cx, cy = _centroid(room)
            cells.update(floor=f"L{room.storey}", area_m2=f"{room.area_m2:.2f}", centroid_x=f"{cx:.2f}",
                         centroid_y=f"{cy:.2f}", level_z=f"{room.level_z:.2f}")
            out.append([cells.get(c, "") for c in header])
        if missing:
            raise Refused(refuse_missing(
                "inputs(action='create', file='spaces')", rows=None,
                hint=f"These rooms have no row to take their values from: {'; '.join(missing)}. Ask the user "
                     "for each, by label (scene action='rooms' shows where they are)."))
        held.update(data=table_bytes(header, out), unmatched=unmatched, problems=problems)
        return held["data"]

    written = write_file(folder, NAME, expected, build, check(detection))
    return written, held["data"], held["unmatched"], held["problems"]


def _add_case(header: list[str], body: list[list[str]], add_case: Mapping[str, Any]) -> None:
    """Insert ``<kind>_<name>`` after the last column of its kind (a first
    ``views_`` column after the ``include_`` ones), ``value`` in every row."""
    column = f"{add_case['kind']}_{check_case(add_case['name'], field='add_case')}"
    if column in header:
        raise Refused(refusal("INPUT_INVALID", f"spaces.csv already has the column {column}.", field="add_case"))
    kinds = [add_case["kind"]] + (["include"] if add_case["kind"] == "views" else [])
    for kind in kinds:
        last = [i for i, c in enumerate(header) if is_case_column(c) and c.startswith(f"{kind}_")]
        if last:
            at = last[-1] + 1
            break
    else:
        raise Refused(refusal("INPUT_INVALID", "spaces.csv has no include_<case> column to add a case after.",
                              field="add_case"))
    header.insert(at, column)
    for row in body:
        row.insert(at, "1" if add_case["value"] else "0")


def _remove_case(header: list[str], body: list[list[str]], column: str) -> None:
    if not is_case_column(column) or column not in header:
        cases = ", ".join(c for c in header if is_case_column(c))
        raise Refused(refusal("INPUT_INVALID", f"remove_case {column!r} is not a case column of spaces.csv; its "
                              f"case columns are {cases}.", field="remove_case"))
    at = header.index(column)
    del header[at]
    for row in body:
        del row[at]


def _edit_rows(header: list[str], body: list[list[str]], rows: Sequence[Any]) -> None:
    """Apply each row: found by ``floor`` and ``room``, its columns set,
    ``rename_to`` renaming it and every ``merge_into`` that names it."""
    floor, room, merge = header.index("floor"), header.index("room"), header.index("merge_into")
    for n, row in enumerate(rows, start=1):
        where = f"rows[{n}]"
        if not isinstance(row, Mapping) or not all(isinstance(row.get(k), str) for k in ("floor", "room")):
            raise Refused(refusal("INPUT_INVALID", f"{where} must name its row by floor and room, both text.",
                                  field="rows"))
        found = [r for r in body if r[floor].strip() == row["floor"].strip() and r[room].strip() == row["room"].strip()]
        if not found:
            raise Refused(refusal("INPUT_INVALID", f"{where}: spaces.csv has no row on floor {row['floor']!r} named "
                                  f"{row['room']!r}.", field="rows"))
        target = found[0]
        for column, value in user_cells(header, row, where, skip=("floor", "room", "rename_to")).items():
            target[header.index(column)] = value
        new = row.get("rename_to")
        if new is not None:
            if isinstance(new, str):
                check_cell_text(new, where, "rename_to")
            new = cell("room", new, where)
            if not new:
                raise Refused(refusal("INPUT_INVALID", f"{where}: rename_to is blank.", field="rows"))
            old = target[room].strip()
            target[room] = new
            for other in body:
                if other[merge].strip() == old:
                    other[merge] = new


def update_spaces(folder: pathlib.Path, expected: str | None, rows: Sequence[Any] | None,
                  add_case: Mapping[str, Any] | None, remove_case: str | None) -> tuple[Written, bytes]:
    """Change spaces.csv: ``add_case`` first, then ``rows``, then
    ``remove_case``. Rows are not added or removed here; after a split,
    ``create`` rebuilds them and keeps their values."""
    _, _, detection = scene.detected(folder)
    held: dict[str, bytes] = {}

    def build(old: bytes | None, raw: Any) -> bytes:
        if old is None:
            raise Refused(refusal("FILE_MISSING", f"{folder}: no spaces.csv; write it with "
                                  "inputs(action='create', file='spaces').", field="file"))
        header, body = table(old)
        if not {"floor", "room", "merge_into"} <= set(header):
            raise Refused(refusal("INPUT_INVALID", f"spaces.csv: the header {','.join(header)} lacks floor, room "
                                  "or merge_into; write it again with action='create'."))
        if add_case is not None:
            _add_case(header, body, add_case)
        _edit_rows(header, body, rows or [])
        if remove_case is not None:
            _remove_case(header, body, remove_case)
        held["data"] = table_bytes(header, body)
        return held["data"]

    written = write_file(folder, NAME, expected, build, check(detection))
    return written, held["data"]
