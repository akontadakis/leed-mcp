# leed_daylight/project/points.py
"""Read the optional sensor points file (``grid.points_file``).

A project normally gets the tool's own lattice in every space
(``build/grid.py``). A project that must use a given set of sensor points
(the points of an earlier study, to compare like with like) names a file
of them in ``project.yaml``::

    grid:
      points_file: points.csv

The file holds one point per row::

    space,x,y,z
    office a,1.00,1.00,0.80

- ``space``: the name of the point's space, as the ``room`` column of
  spaces.csv writes it (a space row, never a piece row);
- ``x``, ``y``, ``z``: the point's position in metres, whatever
  ``scene.units`` (like the centroids in spaces.csv); ``z`` is the work
  plane, the floor level plus ``grid.work_plane_m``.

The file is read with the shared ``read_csv_text`` rules (a BOM is
stripped; a cp1252 file and a ``;`` header are refused by name). Row
numbers are physical file lines, the header being line 1, so a blank line
still counts. This module only reads and types the rows;
``build/grid.py`` checks each point against its space.

A refusal never repeats the file's text beyond a short excerpt
(``excerpt``): the file may be any file the project names, so its first
line is never quoted (a ``;`` header shows only the count of ``;``), and
any field shown is escaped and cut short.
"""
from __future__ import annotations

import csv
import io
import math
import pathlib
from dataclasses import dataclass
from typing import NoReturn

import numpy as np

from .config import ProjectError
from .materials import read_csv_text

POINTS_HEADER = ("space", "x", "y", "z")
#: The most characters of a field a refusal shows (``excerpt``).
EXCERPT_MAX_CHARS = 40
#: The most points (data rows) a points file may hold; more is refused.
POINTS_MAX_ROWS = 1_000_000


class PointsError(ProjectError):
    """The sensor points file cannot be read as written."""


@dataclass(frozen=True, eq=False)
class SpacePoints:
    space: str
    #: ``(n, 3)`` float64, metres, in file order.
    points: np.ndarray
    #: The physical line of the file each point is on (header = line 1).
    lines: tuple[int, ...]


def _fail(path: pathlib.Path, detail: str) -> NoReturn:
    raise PointsError(f"{path.name}: {detail}") from None


def _fail_at(path: pathlib.Path, line_no: int, detail: str) -> NoReturn:
    raise PointsError(f"{path.name} line {line_no}: {detail}") from None


def excerpt(text: str) -> str:
    """``text`` as a refusal shows it: quoted and escaped as ``repr`` does
    (a control character shows as ``\\x1b``, never raw), and cut to at most
    ``EXCERPT_MAX_CHARS`` characters, ending in ``...`` when cut."""
    shown = repr(text)
    if len(shown) <= EXCERPT_MAX_CHARS:
        return shown
    return shown[: EXCERPT_MAX_CHARS - 3] + "..."


def _count_semicolons(header_line: str) -> str:
    """What a ``;``-header refusal shows of the first line: only the count
    of ``;`` on it, never its text (even a short line may be a secret)."""
    return f"{header_line.count(';')} ';' on line 1"


def _finite(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def read_points(path: str | pathlib.Path) -> dict[str, SpacePoints]:
    """Every point of ``path``, grouped by space: spaces in the order they
    first appear, each space's points in file order; ``{}`` for a file with
    a header and no row. Refuses with ``PointsError`` naming the line,
    among others a row with the same space, x and y as an earlier row
    (naming that row's line too) and a file of more than
    ``POINTS_MAX_ROWS`` points."""
    path = pathlib.Path(path)
    text = read_csv_text(path, lambda detail: _fail(path, detail), show=_count_semicolons)
    f = io.StringIO(text, newline="")
    rows: dict[str, list[tuple[int, float, float, float]]] = {}
    first_line: dict[tuple[str, float, float], int] = {}
    with f:
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration:
            _fail(path, f"empty file; expected the header {','.join(POINTS_HEADER)}")
        except csv.Error as exc:
            _fail_at(path, 1, f"cannot be read as CSV: {exc}")
        header = tuple(name.strip() for name in raw_header)
        if header != POINTS_HEADER:
            _fail(path, f"the header must be exactly {','.join(POINTS_HEADER)}; "
                        f"the first line has {len(header)} field(s)")
        last_line = reader.line_num
        while True:
            line_no = last_line + 1
            try:
                raw_row = next(reader)
            except StopIteration:
                break
            except csv.Error as exc:
                _fail_at(path, line_no, f"cannot be read as CSV: {exc}")
            last_line = reader.line_num
            if raw_row == []:
                continue  # a blank line: not a row
            if len(first_line) >= POINTS_MAX_ROWS:
                _fail_at(path, line_no, f"more than {POINTS_MAX_ROWS} points; "
                                        f"a points file holds at most {POINTS_MAX_ROWS}")
            if len(raw_row) != len(POINTS_HEADER):
                _fail_at(path, line_no, f"row has {len(raw_row)} field(s), expected {len(POINTS_HEADER)} "
                                        f"({','.join(POINTS_HEADER)}): {excerpt(','.join(raw_row))}")
            space, *cells = (cell.strip() for cell in raw_row)
            if not space:
                _fail_at(path, line_no, "the space is empty; give the name of the point's space row")
            values = []
            for name, text in zip(POINTS_HEADER[1:], cells, strict=True):
                value = _finite(text)
                if value is None:
                    _fail_at(path, line_no, f"{name} {excerpt(text)} is not a finite number; give metres")
                values.append(value)
            key = (space, values[0], values[1])
            if key in first_line:
                _fail_at(path, line_no, f"the point x {values[0]}, y {values[1]} of space {excerpt(space)} "
                                        f"is already on line {first_line[key]}")
            first_line[key] = line_no
            rows.setdefault(space, []).append((line_no, *values))
    return {
        space: SpacePoints(
            space=space,
            points=np.array([row[1:] for row in found], dtype=np.float64),
            lines=tuple(row[0] for row in found),
        )
        for space, found in rows.items()
    }
