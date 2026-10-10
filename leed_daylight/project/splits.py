# leed_daylight/project/splits.py
"""Read splits.csv: the lines that cut a detected room in two (merge-split spec 3.1).

Room detection gives one room per connected floor patch, so an open-plan
floor drawn as one slab arrives as one room. A person who needs it as
several LEED spaces draws split lines in this optional file::

    level_z,points,note
    3.50,"12.0 4.0; 12.0 9.5; 18.0 9.5",open office east edge

- ``level_z``: the storey's floor level in metres, whatever ``scene.units``
  (like ``level_z`` and the centroids in spaces.csv);
- ``points``: two or more ``x y`` pairs in metres, separated by ``;``; a line
  may bend. No two consecutive points closer than ``SPLIT_TOL_M``, and the
  line must be simple: no segment crosses or overlaps another;
- ``note``: free text, never read by the tool.

The file is read with the shared ``read_csv_text`` rules (a BOM is
stripped; a cp1252 file and a ``;`` header are refused by name). Row
numbers are physical file lines, the header being line 1, so a blank line
still counts. This module only reads and checks each line on its own;
``project/layout.py`` checks it against the rooms and cuts them.
"""
from __future__ import annotations

import csv
import io
import math
import pathlib
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString

from .config import ProjectError
from .materials import read_csv_text

SPLITS_FILE = "splits.csv"
#: The split geometry tolerance, metres: the smallest gap between two
#: consecutive points, and (``project/layout.py``) the snapping distance of
#: an end point and the shortest length that counts as crossing a room.
SPLIT_TOL_M: float = 0.01
SPLITS_HEADER = ("level_z", "points", "note")


class SplitError(ProjectError):
    """A split line cannot be read, or cannot cut the rooms as written."""


@dataclass(frozen=True, eq=False)
class SplitLine:
    #: The physical line of splits.csv this row starts on (header = line 1).
    row: int
    #: The storey's floor level, metres.
    level_z: float
    #: ``(k, 2)`` float64, metres, ``k >= 2``, in file order.
    points: np.ndarray
    note: str


def _fail(detail: str) -> None:
    raise SplitError(f"{SPLITS_FILE}: {detail}") from None


def _fail_at(line_no: int, detail: str) -> None:
    raise SplitError(f"{SPLITS_FILE} line {line_no}: {detail}") from None


def _finite(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _parse_points(cell: str, line_no: int) -> np.ndarray:
    points = []
    for n, text in enumerate(cell.split(";"), start=1):
        text = text.strip()
        parts = text.split()
        values = [_finite(part) for part in parts]
        if len(values) != 2 or None in values:
            _fail_at(
                line_no,
                f"point {n} {text!r} is not two finite numbers; write each point as "
                "'x y' in metres, points separated by ';'",
            )
        points.append(values)
    if len(points) < 2:
        _fail_at(line_no, f"the line has {len(points)} point; a split line needs two or more points")
    array = np.array(points, dtype=np.float64)
    for n in range(1, len(array)):
        gap = float(np.linalg.norm(array[n] - array[n - 1]))
        if gap < SPLIT_TOL_M:
            _fail_at(
                line_no,
                f"points {n} and {n + 1} are {gap:.3f} m apart; consecutive points must be "
                f"at least {SPLIT_TOL_M:g} m apart: remove one of them",
            )
    if not LineString(array).is_simple:
        _fail_at(
            line_no,
            "the line crosses or overlaps itself; a split line must be simple: "
            "remove the loop, or draw two lines",
        )
    return array


def read_splits(path: str | pathlib.Path) -> tuple[SplitLine, ...]:
    """Every split line of ``path``, in file order; ``()`` for a file with a
    header and no row. Refuses with ``SplitError`` naming the line."""
    path = pathlib.Path(path)
    f = io.StringIO(read_csv_text(path, _fail), newline="")
    lines: list[SplitLine] = []
    with f:
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration:
            _fail(f"empty file; expected the header {','.join(SPLITS_HEADER)}")
        header = tuple(name.strip() for name in raw_header)
        if header != SPLITS_HEADER:
            _fail(f"the header must be exactly {','.join(SPLITS_HEADER)}, got {','.join(header)}")
        # Physical FILE line numbers, from csv.reader's own line_num (as
        # materials.csv and spaces.csv count them).
        last_line = reader.line_num
        while True:
            try:
                raw_row = next(reader)
            except StopIteration:
                break
            line_no = last_line + 1
            last_line = reader.line_num
            if raw_row == []:
                continue  # a blank line: not a row
            if len(raw_row) != len(SPLITS_HEADER):
                _fail_at(
                    line_no,
                    f"row has {len(raw_row)} field(s), expected {len(SPLITS_HEADER)} "
                    f"({','.join(SPLITS_HEADER)}); put the points in double quotes, "
                    f"for example \"0.0 0.0; 4.0 0.0\": {raw_row!r}",
                )
            level_text, points_text, note = (cell.strip() for cell in raw_row)
            level_z = _finite(level_text)
            if level_z is None:
                _fail_at(
                    line_no,
                    f"level_z {level_text!r} is not a finite number; give the storey's "
                    "floor level in metres",
                )
            lines.append(SplitLine(row=line_no, level_z=level_z,
                                   points=_parse_points(points_text, line_no), note=note))
    return tuple(lines)


def load_splits(root: str | pathlib.Path) -> tuple[SplitLine, ...] | None:
    """The split lines of ``<root>/splits.csv``, or None when the file does
    not exist (a project that splits no room)."""
    path = pathlib.Path(root) / SPLITS_FILE
    if not path.exists():
        return None
    return read_splits(path)
