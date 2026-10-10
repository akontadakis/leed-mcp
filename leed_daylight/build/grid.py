# leed_daylight/build/grid.py
"""Build the sensor grid for each included, regularly occupied room (spec 6.10).

sDA and ASE are computed at a grid of point-in-time illuminance values, not
over the whole floor plate, so where those points sit decides the result.
Two rules make the grid reproducible and honest:

**The anchor.** On each axis the lattice is centred in the room's
clearance band: with ``c = min_clearance_m`` and the room's own
bounding box ``[lo, hi]``, the band is ``[lo + c, hi - c]``, the first
candidate sits at ``lo + c + ((hi - lo - 2c) mod spacing) / 2`` and the
rest follow every ``spacing`` up to ``hi - c``. The first and last rows
are therefore each between ``c`` and ``c + spacing`` from the walls, and
the layout is symmetric. (A lattice started half a cell from the box, the
earlier rule, put its outer ring 0.30 m from the walls -- inside a
0.305 m clearance -- and dropped every point of it.) A point less than
``c`` from the box on some axis is always less than ``c`` from the
outline too (a ray from it toward that side of the box must cross the
outline first), so restricting candidates to the band loses nothing.
Anchoring to the room's own geometry means the grid a room gets does not
depend on where the building happens to sit in the scene, and moving the
whole building leaves every room's grid identical up to that same
translation (``test_the_grid_does_not_move_when_the_building_moves``);
a ``1e-9`` m tolerance in the step count keeps float noise from a large
offset from adding or losing a row.

**The refusal.** A room that yields zero points would silently contribute
zero to sDA -- the worst possible failure mode for a regularly occupied
space, indistinguishable from a room that legitimately fails the credit.
``build_grids`` refuses it instead, naming the room and its area.
``sensor_points`` itself does not know the room's name (spec interface: it
takes an outline, not a ``Room``), so it returns an ordinary empty
``(0, 3)`` array when a room's own geometry yields no point; only
``build_grids``, which does have the room's name, turns that into a named
refusal.

**Defence in depth.** ``project.config`` already refuses a spacing above
0.60 m when a project loads from YAML, but a hand-built ``Grid`` (as any
caller, including a test, can construct) skips that check entirely.
``sensor_points`` repeats it here, quoting the same LEED v4 EQ Daylight
limit, so the refusal exists on the object that actually walks the
lattice, not only on the loader (controller ruling for this task).

A candidate is kept when it is strictly inside the outline (never on its
boundary) AND at least ``grid.min_clearance_m`` from the boundary,
following the polygon toward the outline itself (with a ``1e-9`` m
tolerance, so float noise cannot drop a point that sits exactly at the
clearance). Points are returned in
row-major order -- by y, then by x -- so two candidates at the same
spacing and clearance always come out in the same order (needed for
``test_the_grid_is_reproducible``, and generally: a metrics stage that
zips point order against a schedule must see the same order every run).

**Spaces of several pieces** (merge-split spec 7). A space is gridded
piece by piece (``piece_points``) and its grid is its pieces' points
concatenated in piece order. A split half is not gridded over its own
outline: the original detected room's lattice is laid, and the half keeps
the points the split line assigns to it (``project.layout.half_mask``), so
the two halves together hold that lattice point for point and neither gets
a clearance band along the line. Every piece of a regularly occupied space
must yield a point (LM-83-12 2.2.6: the grid extends across the entire
space); an empty piece is refused by name.

**A given set of points** (``grid.points_file``, ``project/points.py``).
When the project names a points file, ``build_grids`` takes each space's
points from it instead of laying a lattice: in file order, with no
clearance band and no spacing rule, since the file is the authority (a
study compared like with like keeps its own points). Each point must lie
strictly inside one of its space's pieces (``Room.footprint``: not on the
outline, not in a hole) and sit on the work plane of a piece that holds
it (any of them, where pieces overlap in plan on different levels), its
floor level plus ``grid.work_plane_m`` within ``POINT_Z_TOLERANCE_M``. A point
naming no space row of spaces.csv is refused (a piece row by its space
row's name); points of a space the case leaves out are not used, so one
file serves every case.

Every numeric literal here besides the lattice formula's own structure is
invented and round (phase3-global.md #5): the spacing limit, the edge and
area tolerances are plain, round numbers, not read off a real project.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, Polygon

from ..project.config import Grid
from ..project.layout import LaidOutSpace, SpaceLayout, half_mask, lay_out
from ..project.points import SpacePoints, excerpt
from ..project.spaces import Space, included, space_of, space_rows
from ..scene.storeys import Room

#: LEED v4 EQ Daylight: a sensor grid may be no coarser than 2 ft (600 mm)
#: on a side. Checked here too, not only at config load (see module
#: docstring): a hand-built ``Grid`` can skip config's own check.
_MAX_SPACING_M: float = 0.60

#: An outline edge shorter than this (a duplicated consecutive vertex)
#: makes the outline degenerate along that side -- same tolerance and
#: reasoning as ``build.blinds``/``build.reveals``.
_EDGE_LENGTH_TOLERANCE_M: float = 1e-9

#: An outline with less area than this is degenerate (collinear or
#: coincident vertices), not a real room footprint.
_DEGENERATE_AREA_M2: float = 1e-9

#: A points-file point's z may differ from its work plane by this much:
#: the file is written to the millimetre.
POINT_Z_TOLERANCE_M: float = 0.001

#: Sentinel distinguishing "field missing entirely" from "field present
#: but not a usable number", so a malformed ``Grid`` gets one named
#: refusal either way, never a raw ``AttributeError``.
_MISSING = object()


class GridError(ValueError):
    """A sensor grid cannot be built from these inputs."""


@dataclass(frozen=True)
class RoomGrid:
    #: The space name (``project.spaces.Space.room``), never
    #: ``Room.room_id`` -- the same convention as ``build.groups.WindowGroup``.
    room: str
    #: (n, 3) metres, world coordinates, row-major (by y, then by x).
    points: np.ndarray
    spacing_m: float
    #: ``spacing_m ** 2``.
    cell_area_m2: float
    #: The room's own area (``Room.area_m2``), never the grid's covered
    #: area -- spec 6.10: "The floor area used is the room area from the
    #: scene, not the grid area."
    room_area_m2: float
    #: From the matched ``Room`` (controller ruling P26, additive): so
    #: phase 4 can place and file a grid without reopening ``scene.storeys``.
    storey: int
    level_z: float


def _as_finite_float(value, field: str) -> float:
    if value is _MISSING:
        raise GridError(f"{field}: expected a value, none found")
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise GridError(
            f"{field}: expected a real number, got {value!r}: {exc}"
        ) from None
    if not math.isfinite(v):
        raise GridError(f"{field}: expected a finite number, got {value!r}")
    return v


def _validated_spacing(grid: Grid) -> float:
    spacing = _as_finite_float(getattr(grid, "spacing_m", _MISSING), "grid.spacing_m")
    if not (0.0 < spacing <= _MAX_SPACING_M):
        raise GridError(
            "grid.spacing_m must be greater than zero and no more than "
            f"{_MAX_SPACING_M:g} m (LEED v4 EQ Daylight's 2 ft / 600 mm "
            f"limit): 0 < spacing_m <= {_MAX_SPACING_M:g}, got {spacing!r}"
        )
    return spacing


def _validated_outline(outline_xy) -> np.ndarray:
    try:
        pts = np.asarray(outline_xy, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GridError(f"outline_xy: not a valid array of numbers: {exc}") from None
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise GridError(f"outline_xy: expected shape (n, 2), got shape {pts.shape}")
    if pts.shape[0] < 3:
        raise GridError(
            f"outline_xy: a polygon needs at least 3 vertices, got {pts.shape[0]}"
        )
    if not np.all(np.isfinite(pts)):
        raise GridError("outline_xy: outline has non-finite coordinates")

    count = len(pts)
    for i in range(count):
        j = (i + 1) % count
        edge_length = float(np.hypot(*(pts[j] - pts[i])))
        if edge_length < _EDGE_LENGTH_TOLERANCE_M:
            raise GridError(
                f"outline_xy: edge {i} (vertex {i} to vertex {j}) has "
                f"length {edge_length:.3g} m, shorter than the "
                f"{_EDGE_LENGTH_TOLERANCE_M:g} m tolerance; a duplicated "
                "vertex makes the outline degenerate"
            )
    return pts


#: Float noise allowed in the lattice step count and the clearance test.
_TOLERANCE_M: float = 1e-9


def _lattice(lo: float, hi: float, spacing: float, clearance: float) -> list[float]:
    """Candidate coordinates centred in the clearance band ``[lo + c, hi - c]``.

    The first sits at ``lo + c + ((hi - lo - 2c) mod spacing) / 2``, the
    rest every ``spacing`` after it, up to ``hi - c``. An empty band (the
    room narrower than twice the clearance) gives no candidate.
    """
    band = hi - lo - 2.0 * clearance
    if band < -_TOLERANCE_M:
        return []
    band = max(band, 0.0)
    steps = math.floor(band / spacing + _TOLERANCE_M)
    first = lo + clearance + (band - steps * spacing) / 2.0
    return [first + spacing * i for i in range(steps + 1)]


def sensor_points(outline_xy, level_z: float, grid: Grid, holes=()) -> np.ndarray:
    """The (n, 3) sensor lattice for one room outline, or an empty (0, 3).

    ``outline_xy`` is the room's own 2-D outline (``Room.outline``); the
    lattice is centred in its clearance band on each axis (module
    docstring), never anchored to the world origin. A candidate is kept
    when it is strictly inside the outline and at least
    ``grid.min_clearance_m`` from the boundary. ``holes`` are the room's
    interior rings (``Room.holes``: an atrium or shaft the floor surrounds);
    a hole edge is boundary like any other, so no point sits in a hole or
    within the clearance of its edge. Every kept point's z is
    ``level_z + grid.work_plane_m``. Points come back row-major: by y,
    then by x.

    A room whose geometry admits no valid candidate returns an ordinary
    empty array, shape (0, 3) -- this function does not know the room's
    name, so it cannot make that a named refusal; ``build_grids`` does.

    A malformed ``outline_xy`` (wrong shape, too few vertices, a
    non-numeric or non-finite coordinate, a zero-length edge, or a
    degenerate zero-area polygon), a non-finite ``level_z``, a malformed
    ``grid`` (a missing or non-finite field), or ``grid.spacing_m``
    outside ``(0, 0.60]`` is a :class:`GridError` naming the offending
    value.
    """
    pts = _validated_outline(outline_xy)
    z = _as_finite_float(level_z, "level_z")
    spacing = _validated_spacing(grid)
    clearance = _as_finite_float(
        getattr(grid, "min_clearance_m", _MISSING), "grid.min_clearance_m"
    )
    work_plane = _as_finite_float(
        getattr(grid, "work_plane_m", _MISSING), "grid.work_plane_m"
    )

    rings = []
    for k, hole in enumerate(holes):
        try:
            rings.append(_validated_outline(hole))
        except GridError as exc:
            raise GridError(f"hole {k}: {exc}") from None
    poly = Polygon(pts, rings)
    if poly.area < _DEGENERATE_AREA_M2:
        raise GridError(
            f"outline_xy: outline is degenerate (area {poly.area:.3g} m2), "
            "not a usable room footprint"
        )

    x_min, y_min, x_max, y_max = poly.bounds
    xs = _lattice(x_min, x_max, spacing, clearance)
    ys = _lattice(y_min, y_max, spacing, clearance)

    kept: list[tuple[float, float, float]] = []
    for y in ys:
        for x in xs:
            point = Point(x, y)
            if not poly.contains(point):
                continue
            if poly.boundary.distance(point) < clearance - _TOLERANCE_M:
                continue
            kept.append((x, y, z + work_plane))

    if not kept:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array(kept, dtype=np.float64)


def piece_points(space: LaidOutSpace, grid: Grid) -> tuple[np.ndarray, ...]:
    """One ``(n_i, 3)`` array of sensor points per piece of ``space``, in
    piece order. A split half gets the lattice of its original room
    (``space.grid_source``), keeping, in order, the points assigned to it;
    any other piece gets its own lattice. A malformed outline is a
    :class:`GridError` naming the space and, for a space of several pieces,
    the piece's spaces.csv row."""
    arrays = []
    for piece, source, row in zip(space.pieces, space.grid_source, space.rows, strict=True):
        where = f"room {space.name!r}" if len(space.pieces) == 1 else f"room {space.name!r}, piece row {row!r}"
        try:
            if source is None:
                points = sensor_points(piece.outline, piece.level_z, grid, holes=piece.holes)
            else:
                outline, holes = source
                points = sensor_points(outline, piece.level_z, grid, holes=holes)
                points = points[half_mask(piece, points)]
        except GridError as exc:
            raise GridError(f"{where}: {exc}") from None
        arrays.append(points)
    return tuple(arrays)


def file_points(space: LaidOutSpace, given: SpacePoints, grid: Grid) -> np.ndarray:
    """The ``(n, 3)`` sensor points of ``space`` from the points file, in
    file order, after checking each one (module docstring): strictly inside
    one of the space's pieces and on the work plane of a piece that contains
    it (pieces may overlap in plan on different levels). A point that
    fails is a :class:`GridError` naming its file line, the point and the
    space."""
    work_plane = _as_finite_float(getattr(grid, "work_plane_m", _MISSING), "grid.work_plane_m")
    footprints = [piece.footprint for piece in space.pieces]
    for (x, y, z), line in zip(given.points, given.lines, strict=True):
        where = (
            f"grid.points_file line {line}: point ({x:.3f}, {y:.3f}, {z:.3f}) of space {space.name!r}"
        )
        point = Point(x, y)
        inside = [piece for piece, footprint in zip(space.pieces, footprints) if footprint.contains(point)]
        if not inside:
            raise GridError(
                f"{where} lies outside the space's floor polygon; every point must lie inside "
                "its space, not on its outline or in a hole"
            )
        if any(abs(z - (piece.level_z + work_plane)) <= POINT_Z_TOLERANCE_M for piece in inside):
            continue
        level = inside[0].level_z
        expected = level + work_plane
        raise GridError(
            f"{where} is not on the work plane: z must be {expected:.3f} m (floor "
            f"{level:.3f} m + grid.work_plane_m {work_plane:g} m)"
        )
    return np.array(given.points, dtype=np.float64).reshape(-1, 3)


def _check_point_names(spaces: Sequence[Space], points: Mapping[str, SpacePoints]) -> None:
    """Refuse a points-file space that is not a space row of spaces.csv."""
    rows = {s.room for s in space_rows(spaces)}
    owner = space_of(spaces)
    for name, given in points.items():
        if name in rows:
            continue
        line = given.lines[0]
        if name in owner:
            raise GridError(
                f"grid.points_file line {line}: {excerpt(name)} is a piece row of space "
                f"{owner[name]!r}; give its points the space row's name {owner[name]!r}"
            )
        raise GridError(
            f"grid.points_file line {line}: space {excerpt(name)} is not in spaces.csv; give each point "
            "the name of its space row, as the room column writes it"
        )


def _room_grid(room: str, laid: LaidOutSpace, points: np.ndarray, grid: Grid) -> RoomGrid:
    """The :class:`RoomGrid` of space ``room`` (laid out as ``laid``) on ``points``
    (the file's or the lattice's): one cell of ``grid.spacing_m`` squared
    per point, the space's own area, storey and level."""
    return RoomGrid(
        room=room,
        points=points,
        spacing_m=grid.spacing_m,
        cell_area_m2=grid.spacing_m ** 2,
        room_area_m2=laid.area_m2,
        storey=laid.storey,
        level_z=laid.level_z,
    )


def build_grids(
    spaces: Sequence[Space],
    matched: Mapping[str, Room],
    grid: Grid,
    case: str,
    *,
    layout: SpaceLayout | None = None,
    points: Mapping[str, SpacePoints] | None = None,
) -> list[RoomGrid]:
    """One :class:`RoomGrid` per included, regularly occupied space.

    Only spaces in ``included(spaces, case)`` (space rows) that are also
    ``regularly_occupied`` get a grid; the rest are skipped, not refused.
    A selected space missing from ``matched`` is a :class:`GridError`
    naming it. A selected space whose geometry yields zero sensor points,
    or a space of several pieces with a piece that yields none, is a
    :class:`GridError` naming the room (and the piece) and its area
    (module docstring) -- the one refusal this whole module exists to
    raise.

    ``layout`` is the space layout (``project.layout.lay_out``); without
    one it is laid out here from ``spaces`` and ``matched``. Returned in
    the order of ``spaces`` (``included`` already preserves that order).
    ``RoomGrid.room_area_m2`` is the space's area (the sum of its pieces'
    ``Room.area_m2``), never a sum over the grid's own cells (spec 6.10).

    ``points`` (``project.points.read_points`` of ``grid.points_file``)
    replaces the lattice: each selected space gets its file points
    (``file_points``); a selected space with none is a :class:`GridError`,
    as is a file space that is no space row (module docstring).
    """
    if not isinstance(matched, Mapping):
        raise GridError(
            f"matched: expected a mapping of space name to Room, got "
            f"{type(matched).__name__}"
        )

    selected = [space for space in included(spaces, case) if space.regularly_occupied]
    for space in selected:
        if matched.get(space.room) is None:
            raise GridError(
                f"space {space.room!r} is included in case {case!r} and "
                "regularly occupied, but has no matched room"
            )
    if layout is None:
        layout = lay_out(spaces, matched)
    if points is not None:
        _check_point_names(spaces, points)

    grids: list[RoomGrid] = []
    for space in selected:
        laid = layout.space(space.room)
        if points is not None:
            if space.room not in points:
                raise GridError(
                    f"grid.points_file has no point for space {space.room!r}, which case {case!r} "
                    "includes as regularly occupied; every such space needs its points"
                )
            grids.append(_room_grid(space.room, laid, file_points(laid, points[space.room], grid), grid))
            continue
        pieces = piece_points(laid, grid)
        if len(laid.pieces) == 1 and pieces[0].shape[0] == 0:
            raise GridError(
                f"room {space.room!r} (area {laid.area_m2:.2f} m2) yields "
                f"no sensor point at {grid.spacing_m:g} m spacing with "
                f"{grid.min_clearance_m:g} m clearance; a regularly "
                "occupied space needs at least one sensor point"
            )
        for piece, row, piece_pts in zip(laid.pieces, laid.rows, pieces, strict=True):
            if piece_pts.shape[0] == 0:
                raise GridError(
                    f"room {space.room!r}: piece {piece.room_id} (row {row!r}, area "
                    f"{piece.area_m2:.2f} m2) yields no sensor point at {grid.spacing_m:g} m "
                    f"spacing with {grid.min_clearance_m:g} m clearance; every piece of a "
                    "regularly occupied space needs at least one sensor point (LM-83-12 "
                    "2.2.6: the grid extends across the entire space); clear its merge_into and "
                    "untick it or mark it not regularly occupied, or check the room in the model"
                )
        lattice = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        grids.append(_room_grid(space.room, laid, lattice, grid))
    return grids
