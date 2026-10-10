# leed_mcp/adapters/splits_file.py
"""``inputs(action="update", file="splits")`` (spec 6.3).

``lines`` is the full new list of split lines. The file is written as the
wizard wrote it (``splits_bytes``, ported) and the ``.part`` file is
checked as the verbs check it: ``read_splits``, then ``apply_splits`` on
the rooms detection finds (one room crossed, ends on the outline or
outside, halves of at least 1 m2, ends between panes). A split changes the
rooms, so the reply names the spaces.csv rows that no longer match one.
"""
from __future__ import annotations

import csv
import io
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

from leed_daylight.project.layout import apply_splits
from leed_daylight.project.splits import SPLITS_FILE, SPLITS_HEADER, read_splits
from leed_daylight.scene.storeys import detect_rooms
from leed_mcp.adapters import scene, spaces_file
from leed_mcp.adapters.input_files import Written, check_cell_text, read_bytes, write_file

__all__ = ["splits_bytes", "update_splits"]


def _number(value: float) -> str:
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def splits_bytes(lines: Sequence[Mapping[str, Any]]) -> bytes:
    """splits.csv for ``lines``: ``level_z,points,note``, ``\\n`` line ends
    (wizard ``splits_bytes``)."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(SPLITS_HEADER)
    for line in lines:
        points = "; ".join(f"{_number(x)} {_number(y)}" for x, y in line["points"])
        writer.writerow([_number(line["level_z"]), points, line.get("note", "")])
    return out.getvalue().encode("utf-8")


def unmatched_rows(folder: pathlib.Path, rooms) -> list[str]:
    """The spaces.csv rows that match none of ``rooms``; none when there is
    no readable spaces.csv."""
    found = spaces_file.old_table(read_bytes(folder, spaces_file.NAME))
    if found is None:
        return []
    header, rows = found
    matched = spaces_file.matches(header, rows, rooms)
    return [row[header.index("room")] for i, row in enumerate(rows) if i not in matched]


def update_splits(folder: pathlib.Path, expected: str | None,
                  lines: Sequence[Mapping[str, Any]]) -> tuple[Written, bytes, list[str]]:
    """Replace splits.csv with ``lines``; returns the file, its bytes and the
    spaces.csv rows that match no room after the change."""
    for n, line in enumerate(lines):
        check_cell_text(line.get("note", ""), f"lines[{n}]", "note")
    prims, roles = scene.scene_and_roles(folder)
    data = splits_bytes(lines)
    held: dict[str, Any] = {}

    def check(part: pathlib.Path) -> None:
        detection = detect_rooms(prims, roles)
        read = read_splits(part)
        held["rooms"] = apply_splits(detection, read, prims, roles).rooms if read else detection.rooms

    written = write_file(folder, SPLITS_FILE, expected, lambda old, raw: data, check)
    return written, data, unmatched_rows(folder, held["rooms"])
