# leed_mcp/adapters/materials_file.py
"""``inputs`` on materials.csv (spec 6.3): ``create`` and ``update``.

The modifiers and their primitive types come from the scene (as in
``scene(action="materials_template")``); the person gives each one its
role, and may give a ``view`` and a ``note``. ``create`` needs a role for
every modifier, because ``load_materials`` refuses an empty one. The file
is written as the wizard wrote it (``table_bytes``, ported): the ``view``
column only when a row has a view, ``\\n`` line ends; ``update`` keeps
the file's BOM, line ends and columns. The ``.part`` file
is checked with ``load_materials`` and ``declared_types``.
"""
from __future__ import annotations

import codecs
import csv
import io
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

from leed_daylight.drivers.materials import TEMPLATE_HEADER
from leed_daylight.project.materials import declared_types, load_materials
from leed_daylight.scene.inventory import inventory
from leed_mcp.adapters.input_files import Written, check_cell_text, write_file
from leed_mcp.adapters.input_read import table
from leed_mcp.adapters.scene import checked_modifiers, load_scene
from leed_mcp.envelope import refusal, refuse_missing
from leed_mcp.errors import Refused

__all__ = ["COLUMNS", "create_materials", "table_bytes", "update_materials"]

NAME = "materials.csv"
#: materials.csv's columns with the two optional last ones, ``view`` and ``vlt``.
COLUMNS = (*TEMPLATE_HEADER, "view", "vlt")
_KEYS = frozenset({"modifier", "role", "view", "note", "vlt"})


def table_bytes(rows: Sequence[Sequence[str]], *, keep: Sequence[str] = (), bom: bool = False,
                newline: str = "\n") -> bytes:
    """materials.csv for ``rows`` of six cells (``view``, ``vlt`` last): each
    optional column when a row has a value in it or ``keep`` names it;
    ``newline`` line ends, a UTF-8 BOM first when ``bom``."""
    with_view = "view" in keep or any(row[4] for row in rows)
    with_vlt = "vlt" in keep or any(row[5] for row in rows)
    columns = [*range(4), *([4] if with_view else []), *([5] if with_vlt else [])]
    out = io.StringIO()
    writer = csv.writer(out, lineterminator=newline)
    writer.writerow([COLUMNS[i] for i in columns])
    for row in rows:
        writer.writerow([row[i] for i in columns])
    return (codecs.BOM_UTF8 if bom else b"") + out.getvalue().encode("utf-8")


def _scene_types(folder: pathlib.Path) -> dict[str, str]:
    """Modifier -> primitive type ("" when the scene never defines it), in
    the inventory's order: largest area first."""
    _, prims = load_scene(folder)
    found = inventory(prims)
    checked_modifiers(found)
    return {info.name: info.ptype for info in found}


def _rows(rows: Sequence[Any], scene: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """The given rows by modifier, each a mapping of text cells; refuses a
    row that is not one, an unknown key, ``primitive_type``, a modifier the
    scene lacks, a modifier given twice and bad text in role, view or note
    (``input_files.check_cell_text``)."""
    found: dict[str, dict[str, str]] = {}
    for n, row in enumerate(rows, start=1):
        if isinstance(row, Mapping):
            row = dict(row)
            if isinstance(row.get("vlt"), (int, float)) and not isinstance(row.get("vlt"), bool):
                row["vlt"] = f"{row['vlt']:g}"
        if not isinstance(row, Mapping) or not isinstance(row.get("modifier"), str):
            raise Refused(refusal("INPUT_INVALID", f"rows[{n}] must be an object with a modifier.", field="rows"))
        name = row["modifier"]
        if "primitive_type" in row:
            raise Refused(refusal("INPUT_INVALID", f"rows[{n}] ({name}): primitive_type comes from the scene and "
                                  "cannot be changed.", field="rows"))
        extra = sorted(set(row) - _KEYS)
        bad = [k for k in sorted(_KEYS & set(row)) if not isinstance(row[k], str)]
        if extra or bad:
            what = f"unknown key(s) {', '.join(extra)}" if extra else f"{', '.join(bad)} must be text"
            raise Refused(refusal("INPUT_INVALID", f"rows[{n}] ({name}): {what}; a row has modifier, role, "
                                  "view, note and vlt.", field="rows"))
        if name not in scene:
            raise Refused(refusal("INPUT_INVALID", f"rows[{n}]: {name!r} is not a modifier of the scene; its "
                                  f"modifiers are {', '.join(scene)}.", field="rows"))
        if name in found:
            raise Refused(refusal("INPUT_INVALID", f"rows[{n}]: {name!r} is given twice.", field="rows"))
        for key in sorted(_KEYS & set(row) - {"modifier"}):
            check_cell_text(row[key], f"rows[{n}] ({name})", key)
        found[name] = {k: row[k].strip() for k in _KEYS & set(row)}
    return found


def _check(part: pathlib.Path) -> None:
    load_materials(part)
    declared_types(part)


def create_materials(folder: pathlib.Path, expected: str | None, rows: Sequence[Any]) -> tuple[Written, bytes]:
    """Write materials.csv with one row per scene modifier, each with the
    role ``rows`` gives it."""
    scene = _scene_types(folder)
    given = _rows(rows, scene)
    missing = [name for name in scene if not given.get(name, {}).get("role")]
    if missing:
        raise Refused(refuse_missing(
            "inputs(action='create', file='materials')", rows=None,
            hint=f"Every modifier needs a role; these have none: {', '.join(missing)}."))
    lines = [[name, ptype, given[name]["role"], given[name].get("note", ""), given[name].get("view", ""),
             given[name].get("vlt", "")] for name, ptype in scene.items()]
    data = table_bytes(lines)
    return write_file(folder, NAME, expected, lambda old, raw: data, _check), data


def update_materials(folder: pathlib.Path, expected: str | None, rows: Sequence[Any]) -> tuple[Written, bytes]:
    """Change the role, view, note or vlt of the rows ``rows`` names
    (matched on modifier). A scene modifier the file lacks is added. Every
    other row, the BOM, the line ends and the file's columns are kept."""
    scene = _scene_types(folder)
    given = _rows(rows, scene)
    held: dict[str, bytes] = {}

    def build(old: bytes | None, raw: Any) -> bytes:
        if old is None:
            raise Refused(refusal("FILE_MISSING", f"{folder}: no materials.csv; write it with "
                                  "inputs(action='create', file='materials').", field="file"))
        header, body = table(old)
        names = [h.strip() for h in header]
        if names[:4] != list(TEMPLATE_HEADER) or any(n not in COLUMNS for n in names):
            raise Refused(refusal("INPUT_INVALID", f"materials.csv: the header is {','.join(header)}; expected "
                                  f"{','.join(TEMPLATE_HEADER)}[,view][,vlt]. Write it again with action='create'."))
        lines = [[dict(zip(names, row)).get(c, "") for c in COLUMNS] for row in body]
        index = {row[0].strip(): row for row in lines}
        for name, cells in given.items():
            if name not in index:
                index[name] = [name, scene[name], "", "", "", ""]
                lines.append(index[name])
            for column, value in cells.items():
                index[name][COLUMNS.index(column)] = value
        # The file's own BOM, line ends and optional columns stay, so a VLT
        # added here leaves materials_digest as it was.
        newline = "\r\n" if b"\r\n" in old else "\n"
        held["data"] = table_bytes(lines, keep=names[4:], bom=old.startswith(codecs.BOM_UTF8), newline=newline)
        return held["data"]

    written = write_file(folder, NAME, expected, build, _check)
    return written, held["data"]
