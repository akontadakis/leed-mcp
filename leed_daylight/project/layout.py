# leed_daylight/project/layout.py
"""Split rooms by the lines of splits.csv and lay the spaces out over the
rooms (merge-split spec 4 steps 2 and 4, 6, 7).

Room detection gives one room per connected patch of floor. ``apply_splits``
replaces each room a split line cuts by its two halves, in file order, so
every later step (``match_to_rooms``, the grid, the window groups) sees the
halves as ordinary rooms.

**A half is a room.** Each half is a ``SplitHalf``: every ``Room`` field,
recomputed for the half (outline and holes, plan area, ceiling height, the
floors it overlaps, the id ``<id>a`` or ``<id>b``), plus the rings of the
original detected room (``source_outline``, ``source_holes``) and the cuts
that made it (``cuts``, in file order). A later line may cut a half again:
``S0_R0a`` becomes ``S0_R0aa`` and ``S0_R0ab``.

**Left and right.** A line runs from its first point to its last. Half
``a`` is the one on its left, found by one probe point ``SPLIT_TOL_M / 2``
to the left of the middle of the longest segment of the part of the line
inside the room.

**The cut.** A line crosses a room when more than ``SPLIT_TOL_M`` of it
lies inside the room's footprint (running along an edge or touching a
corner is not crossing). It must cross exactly one room of its storey. An
end point inside the room within ``SPLIT_TOL_M`` of the outline is moved
onto the outline; an end point outside is kept. Each end on the outline is
then extended ``SPLIT_TOL_M`` outward along its end segment, so shapely's
``split`` sees a crossing, and the cut must give exactly two polygons, each
at least ``MIN_HALF_AREA_M2``.

**Glazing.** A split must end between panes, so that each pane lights one
half. Which pane belongs to which room is decided in ``build/groups.py``,
which this layer may not import, so the same plan rules are restated
here. A *facade* pane (no plan area) belongs to the room when its centroid
is in the room's storey band and its plan segment lies within
``PANE_REACH_M`` of the footprint; the line, each end extended
``PANE_REACH_M``, must not touch that segment away from its two ends, nor
run along it (so set-back glass is caught). Any other pane is a *roof*
pane; it is over the room when, going down from the storey band that holds
its centroid, the first storey with a room covering its plan centroid is
the room's own storey and that room is this one. The line must not run
through its plan (running along a mullion between two panes is allowed).

**Points.** The grid of a half is the original room's grid, each point
assigned by ``left_of`` (spec 7): a point within ``ON_LINE_TOL_M`` of the
line goes to half ``a``; a point strictly inside a half goes to it; any
other point goes to the side of the line's nearest segment. The two halves
therefore hold every point of the original room exactly once, each in its
original order (``half_mask``).

``scene_rooms`` runs detection and then the lines of ``splits.csv``, when
that file exists: every command that finds rooms calls it.

**The space layout.** ``lay_out`` gives each space row of spaces.csv one
``LaidOutSpace``: its pieces are the space row's own matched room, then the
rooms of its piece rows (``merge_into``), in file order. A space with no
split and no merge has one piece, its matched room object itself, so every
later step sees what it saw before.

This module is layer 2: it never imports ``build/``.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, nearest_points
from shapely.ops import split as split_polygon

from .. import geometry
from ..geometry import drop_redundant_vertices
from ..scene.primitives import Primitive
from ..scene.storeys import (
    FLOOR_ROLES,
    Room,
    RoomDetection,
    ceiling_height,
    ceiling_surfaces,
    detect_rooms,
    plan_polygon,
)
from .config import Project, ProjectError
from .materials import EXTERIOR_GLAZING_ROLES
from .spaces import LEVEL_TOLERANCE_M, Space, space_of, space_rows
from .splits import SPLIT_TOL_M, SPLITS_FILE, SplitError, SplitLine, load_splits

#: A half smaller than this, in square metres, is refused (spec 5).
MIN_HALF_AREA_M2: float = 1.0
#: A grid point this close to a split line, in metres, goes to half ``a`` (spec 7).
ON_LINE_TOL_M: float = 1e-9
#: How far a pane may sit from a room's outline and still be that room's,
#: metres. Equals ``build.groups.ROOM_MATCH_TOLERANCE_M``, restated here
#: because the project layer cannot import ``build/``.
PANE_REACH_M: float = 0.6
#: A laid-out space's origin, for the reports and the manifest (spec 6).
ORIGINS = ("detected", "split", "merged", "split+merged")
#: A pane whose plan area is at most this, in square metres, is vertical:
#: a facade pane, seen in plan as a segment.
_FACADE_PLAN_AREA_M2: float = 1e-6


class LayoutError(ProjectError):
    """The spaces cannot be laid out over the rooms."""


@dataclasses.dataclass(frozen=True, eq=False)
class Cut:
    #: The splits.csv line (header = line 1).
    row: int
    #: The ``(k, 2)`` line as cut, its end points moved onto the outline.
    line: np.ndarray
    #: The ``(j, 2)`` part of the line inside the room it cut: the edge the
    #: two halves share, oriented like the line (drawn on the plans).
    drawn: np.ndarray
    #: ``"a"`` or ``"b"``: the half this piece took at this cut.
    side: str
    #: The footprints of the two halves at this cut; ``left`` is half ``a``.
    left: Polygon
    right: Polygon


@dataclasses.dataclass(frozen=True)
class SplitHalf(Room):
    #: The original detected room's outline and holes: its grid is laid
    #: over them and filtered by ``half_mask`` (spec 7).
    source_outline: np.ndarray | None = None
    source_holes: tuple[np.ndarray, ...] = ()
    #: Every cut that made this half, in file order.
    cuts: tuple[Cut, ...] = ()


@dataclasses.dataclass(frozen=True, eq=False)
class _Pane:
    name: str
    #: The area centroid, metres.
    centroid: np.ndarray
    #: True for a facade pane (``plan`` a segment), False for a roof pane
    #: (``plan`` its plan polygon).
    facade: bool
    plan: object


def _panes(prims: Sequence[Primitive], roles: Mapping[str, str]) -> list[_Pane]:
    """Every readable exterior glazing polygon, in scene order."""
    panes = []
    for prim in prims:
        if roles.get(prim.modifier) not in EXTERIOR_GLAZING_ROLES or prim.ptype != "polygon":
            continue
        try:
            points = prim.points
            centroid = geometry.centroid(points)
        except ValueError:
            continue  # scene validation names a malformed or degenerate pane
        xy = points[:, :2]
        if abs(Polygon(xy).area) <= _FACADE_PLAN_AREA_M2:
            gaps = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
            i, j = np.unravel_index(int(np.argmax(gaps)), gaps.shape)
            panes.append(_Pane(prim.name, centroid, True, LineString([xy[i], xy[j]])))
        else:
            panes.append(_Pane(prim.name, centroid, False, plan_polygon(prim)))
    return panes


def _bands(rooms: Sequence[Room]) -> dict[int, tuple[float, float]]:
    """Each storey's band ``[level_z, next level_z)``, open above the top one."""
    levels = sorted({(room.storey, room.level_z) for room in rooms})
    return {
        storey: (z, levels[i + 1][1] if i + 1 < len(levels) else math.inf)
        for i, (storey, z) in enumerate(levels)
    }


def _stretched(points: np.ndarray, reach: float) -> np.ndarray:
    """The line with both ends pushed ``reach`` further out along their end segments."""
    out = points.copy()
    for end, inner in ((0, 1), (-1, -2)):
        direction = points[end] - points[inner]
        out[end] = points[end] + direction / np.linalg.norm(direction) * reach
    return out


def _meets_facade(points: np.ndarray, plan: LineString) -> bool:
    """Whether the line, stretched ``PANE_REACH_M`` at both ends, runs along
    the pane's plan segment or touches it away from its two ends."""
    hit = LineString(_stretched(points, PANE_REACH_M)).intersection(plan)
    if hit.length > SPLIT_TOL_M:
        return True
    ends = [Point(c) for c in plan.coords]
    for part in getattr(hit, "geoms", [hit]):
        for coord in (part.coords if not part.is_empty else ()):
            if all(Point(coord).distance(end) > SPLIT_TOL_M for end in ends):
                return True
    return False


def _roof_room(pane: _Pane, rooms: Sequence[Room], bands) -> list[Room]:
    """The rooms under a roof pane (``build.groups._roof_search``'s rule):
    those covering its plan centroid on the first storey, going down from
    the band that holds its centroid, where any room does."""
    start = [storey for storey, (lo, hi) in bands.items() if lo <= pane.centroid[2] < hi]
    if not start:
        return []
    point = Point(float(pane.centroid[0]), float(pane.centroid[1]))
    for storey in sorted((s for s in bands if s <= start[0]), reverse=True):
        covering = [room for room in rooms if room.storey == storey and room.footprint.covers(point)]
        if covering:
            return covering
    return []


def _check_glazing(line: SplitLine, points: np.ndarray, room: Room, rooms: Sequence[Room], panes) -> None:
    """Refuse a line that meets a facade pane of ``room`` or a roof pane over it."""
    bands = _bands(rooms)
    low, high = bands[room.storey]
    footprint = room.footprint
    geom = LineString(points)
    for pane in panes:
        if pane.facade:
            met = (
                low <= pane.centroid[2] < high
                and pane.plan.distance(footprint) <= PANE_REACH_M
                and _meets_facade(points, pane.plan)
            )
        else:
            met = (
                pane.centroid[2] > room.level_z
                and any(r is room for r in _roof_room(pane, rooms, bands))
                and _inside_length(geom, pane.plan) > SPLIT_TOL_M
            )
        if met:
            raise SplitError(
                f"{SPLITS_FILE} line {line.row}: the line meets pane {pane.name} of {_where(room)}; "
                "a split must end between panes, so that each pane lights one half"
            )


def _inside_length(geom, poly) -> float:
    """The length of ``geom`` inside ``poly``, not counting any part that
    runs along ``poly``'s boundary."""
    return float(geom.intersection(poly).length - geom.intersection(poly.boundary).length)


def _where(room: Room) -> str:
    centre = Polygon(room.outline).centroid
    return f"room {room.room_id} (centroid {centre.x:.2f}, {centre.y:.2f})"


def _storey(line: SplitLine, rooms: Sequence[Room]) -> int:
    levels = sorted({(room.storey, room.level_z) for room in rooms})
    near = [(abs(z - line.level_z), storey) for storey, z in levels
            if abs(z - line.level_z) <= LEVEL_TOLERANCE_M]
    if not near:
        if levels:
            known = "storeys at " + ", ".join(f"{z:.2f} m" for _, z in levels)
        else:
            known = "the scene has no floor"
        raise SplitError(
            f"{SPLITS_FILE} line {line.row}: level_z {line.level_z:.2f} m matches no storey "
            f"({known}); level_z and points are metres, whatever scene.units"
        )
    return min(near)[1]


def _crossed_room(line: SplitLine, rooms: Sequence[Room], storey: int) -> int:
    """The index in ``rooms`` of the one room of ``storey`` the line crosses."""
    geom = LineString(line.points)
    crossed = [
        i for i, room in enumerate(rooms)
        if room.storey == storey and _inside_length(geom, room.footprint) > SPLIT_TOL_M
    ]
    if len(crossed) == 1:
        return crossed[0]
    level = next(room.level_z for room in rooms if room.storey == storey)
    if not crossed:
        raise SplitError(
            f"{SPLITS_FILE} line {line.row}: the line crosses no room on storey {storey} "
            f"(level {level:.2f} m); a split line must cross exactly one room: check its "
            "points (metres, whatever scene.units)"
        )
    names = ", ".join(_where(rooms[i]).removeprefix("room ") for i in crossed)
    raise SplitError(
        f"{SPLITS_FILE} line {line.row}: the line crosses {len(crossed)} rooms on storey "
        f"{storey}: {names}; a split line must cross exactly one room: shorten it so that "
        "it stays inside one room"
    )


def _snapped(line: SplitLine, room: Room) -> tuple[np.ndarray, list[bool]]:
    """The line with each end point inside ``room`` within ``SPLIT_TOL_M``
    of its outline moved onto the outline, and which ends lie on it."""
    footprint = room.footprint
    boundary = footprint.boundary
    points = line.points.copy()
    on_outline = []
    for index, which in ((0, "first"), (-1, "last")):
        end = Point(points[index])
        if footprint.contains(end):
            depth = boundary.distance(end)
            if depth > SPLIT_TOL_M:
                raise SplitError(
                    f"{SPLITS_FILE} line {line.row}: its {which} point ({end.x:.2f}, {end.y:.2f}) "
                    f"is {depth:.2f} m inside {_where(room)}; a split line must end on the "
                    "room's outline or outside it: move that point onto the outline or past it"
                )
            nearest = nearest_points(boundary, end)[0]
            points[index] = (nearest.x, nearest.y)
            on_outline.append(True)
        else:
            on_outline.append(boundary.distance(end) <= ON_LINE_TOL_M)
    return points, on_outline


def _extended(points: np.ndarray, on_outline: list[bool]) -> np.ndarray:
    """The line with each end on the outline pushed ``SPLIT_TOL_M`` further
    out along its end segment."""
    out = points.copy()
    if on_outline[0]:
        direction = points[0] - points[1]
        out[0] = points[0] + direction / np.linalg.norm(direction) * SPLIT_TOL_M
    if on_outline[1]:
        direction = points[-1] - points[-2]
        out[-1] = points[-1] + direction / np.linalg.norm(direction) * SPLIT_TOL_M
    return out


def _clean(poly: Polygon) -> Polygon:
    outline = drop_redundant_vertices(np.asarray(poly.exterior.coords[:-1], dtype=np.float64))
    holes = [drop_redundant_vertices(np.asarray(r.coords[:-1], dtype=np.float64)) for r in poly.interiors]
    return Polygon(outline, holes)


def _drawn(first: Polygon, second: Polygon, line: np.ndarray) -> np.ndarray:
    """The edge the two pieces share, as one ``(j, 2)`` run oriented like the line."""
    shared = first.boundary.intersection(second.boundary)
    parts = [g for g in getattr(shared, "geoms", [shared]) if isinstance(g, LineString) and g.length > 0]
    merged = linemerge(parts) if len(parts) > 1 else (parts[0] if parts else LineString())
    if merged.is_empty:
        return np.zeros((0, 2))
    if merged.geom_type != "LineString":
        merged = max(merged.geoms, key=lambda g: g.length)
    coords = np.asarray(merged.coords, dtype=np.float64)
    along = LineString(line)
    if along.project(Point(coords[0])) > along.project(Point(coords[-1])):
        coords = coords[::-1]
    return coords


def _left_piece(drawn: np.ndarray, pieces: Sequence[Polygon]) -> int | None:
    """The index of the piece on the line's left, or None when the probe
    point lies in neither."""
    if len(drawn) < 2:
        return None
    lengths = np.linalg.norm(np.diff(drawn, axis=0), axis=1)
    i = int(np.argmax(lengths))
    a, b = drawn[i], drawn[i + 1]
    direction = (b - a) / lengths[i]
    probe = Point(*((a + b) / 2.0 + np.array([-direction[1], direction[0]]) * (SPLIT_TOL_M / 2.0)))
    inside = [n for n, piece in enumerate(pieces) if piece.contains(probe)]
    return inside[0] if len(inside) == 1 else None


def _floor_plans(prims: Sequence[Primitive], roles: Mapping[str, str]) -> list[tuple[str, float, Polygon]]:
    """Every readable floor-role polygon: (name, mean z, plan), in scene order."""
    plans = []
    for prim in prims:
        if roles.get(prim.modifier) not in FLOOR_ROLES or prim.ptype != "polygon":
            continue
        try:
            points = prim.points
        except ValueError:
            continue
        if points is None or len(points) < 3:
            continue
        plans.append((prim.name, float(points[:, 2].mean()), plan_polygon(prim)))
    return plans


def _half(room: Room, poly: Polygon, side: str, cut: Cut, ceilings, floors) -> SplitHalf:
    outline = np.asarray(poly.exterior.coords[:-1], dtype=np.float64)
    holes = tuple(np.asarray(r.coords[:-1], dtype=np.float64) for r in poly.interiors)
    names = set(room.floor_names)
    floor_names = tuple(
        name for name, z, plan in floors
        if name in names and abs(z - room.level_z) <= LEVEL_TOLERANCE_M and plan.intersection(poly).area > 0.0
    )
    changes = {
        "room_id": room.room_id + side,
        "outline": outline,
        "holes": holes,
        "area_m2": float(poly.area),
        "ceiling_height_m": ceiling_height(poly, room.level_z, ceilings)[0],
        "floor_names": floor_names,
    }
    fields = {f.name: getattr(room, f.name) for f in dataclasses.fields(Room)} | changes
    if isinstance(room, SplitHalf):
        source_outline, source_holes, cuts = room.source_outline, room.source_holes, room.cuts
    else:
        source_outline, source_holes, cuts = room.outline, room.holes, ()
    return SplitHalf(**fields, source_outline=source_outline, source_holes=source_holes, cuts=(*cuts, cut))


def _cut(
    line: SplitLine, rooms: Sequence[Room], index: int, ceilings, floors, panes,
) -> tuple[SplitHalf, SplitHalf]:
    room = rooms[index]
    points, on_outline = _snapped(line, room)
    result = split_polygon(room.footprint, LineString(_extended(points, on_outline)))
    pieces = [g for g in result.geoms if isinstance(g, Polygon) and not g.is_empty]
    if len(pieces) != 2:
        raise SplitError(
            f"{SPLITS_FILE} line {line.row}: the line cuts {_where(room)} into {len(pieces)} "
            "piece(s); a split line must cut its room into exactly two: draw it so that it "
            "enters and leaves the room once, or use two lines"
        )
    drawn = _drawn(pieces[0], pieces[1], points)
    left = _left_piece(drawn, pieces)
    if left is None:
        raise SplitError(
            f"{SPLITS_FILE} line {line.row}: cannot tell the two sides of the line in "
            f"{_where(room)}; draw the line so that it runs through the room clearly"
        )
    first, second = _clean(pieces[left]), _clean(pieces[1 - left])
    small = min(first.area, second.area)
    if small < MIN_HALF_AREA_M2:
        raise SplitError(
            f"{SPLITS_FILE} line {line.row}: the line cuts {_where(room)} into halves of "
            f"{first.area:.2f} m2 and {second.area:.2f} m2; each half must be at least "
            f"{MIN_HALF_AREA_M2:g} m2: move the line"
        )
    _check_glazing(line, points, room, rooms, panes)
    halves = []
    for side, poly in (("a", first), ("b", second)):
        cut = Cut(row=line.row, line=points, drawn=drawn, side=side, left=first, right=second)
        halves.append(_half(room, poly, side, cut, ceilings, floors))
    return halves[0], halves[1]


def apply_splits(
    detection: RoomDetection,
    splits: Sequence[SplitLine],
    prims: Sequence[Primitive],
    roles: Mapping[str, str],
) -> RoomDetection:
    """``detection`` with every room a line cuts replaced, at its place in
    ``rooms``, by its halves ``a`` then ``b``; lines in file order. With no
    line it returns ``detection`` itself. ``prims`` and ``roles`` give a
    half its ceiling height and floor names, and the panes a line must not
    meet. Refuses with ``SplitError`` naming the splits.csv line."""
    if not splits:
        return detection
    rooms = list(detection.rooms)
    ceilings = ceiling_surfaces(prims, roles)
    floors = _floor_plans(prims, roles)
    panes = _panes(prims, roles)
    for line in splits:
        storey = _storey(line, rooms)
        index = _crossed_room(line, rooms, storey)
        rooms[index:index + 1] = _cut(line, rooms, index, ceilings, floors, panes)
    return RoomDetection(tuple(rooms), detection.warnings)


def scene_rooms(project: Project, prims: Sequence[Primitive], roles: Mapping[str, str]) -> RoomDetection:
    """``detect_rooms``, then the lines of ``<project.root>/splits.csv``
    when that file exists. Without the file it returns ``detect_rooms``'
    own result object, so every command sees exactly what it saw before."""
    detection = detect_rooms(prims, roles)
    splits = load_splits(project.root)
    if splits is None:
        return detection
    return apply_splits(detection, splits, prims, roles)


def left_of(points_xy: np.ndarray, cut: Cut) -> np.ndarray:
    """``(n,)`` bool, True where a point goes to half ``a`` of ``cut``.

    ``points_xy`` is ``(n, 2)`` or ``(n, 3)`` (z ignored). Per point: within
    ``ON_LINE_TOL_M`` of the line: True; else strictly inside ``cut.left``:
    True; else strictly inside ``cut.right``: False; else the side of the
    line's nearest segment (the lowest index on a tie; left when the cross
    product is zero or more)."""
    xy = np.asarray(points_xy, dtype=np.float64)[:, :2]
    start = cut.line[:-1]
    along = cut.line[1:] - start  # (m, 2): one row per segment
    rel = xy[:, None, :] - start[None, :, :]  # (n, m, 2)
    t = np.clip((rel * along).sum(axis=2) / (along * along).sum(axis=1), 0.0, 1.0)
    distance = np.linalg.norm(rel - t[..., None] * along, axis=2)  # (n, m)
    nearest = np.argmin(distance, axis=1)  # the first on a tie
    rows = np.arange(len(xy))
    seg, off = along[nearest], rel[rows, nearest]
    cross = seg[:, 0] * off[:, 1] - seg[:, 1] * off[:, 0]
    on_line = distance[rows, nearest] <= ON_LINE_TOL_M
    in_left = shapely.contains_xy(cut.left, xy[:, 0], xy[:, 1])
    in_right = shapely.contains_xy(cut.right, xy[:, 0], xy[:, 1])
    return on_line | in_left | (~in_right & (cross >= 0.0))


def half_mask(piece: Room, points_xy: np.ndarray) -> np.ndarray:
    """``(n,)`` bool: which of ``points_xy`` (a grid laid over the piece's
    original room) belong to ``piece``. All True for a plain ``Room``; for a
    ``SplitHalf``, its cuts applied in order."""
    n = len(points_xy)
    mask = np.ones(n, dtype=bool)
    if not isinstance(piece, SplitHalf):
        return mask
    for cut in piece.cuts:
        left = left_of(points_xy, cut)
        mask &= left if cut.side == "a" else ~left
    return mask


@dataclasses.dataclass(frozen=True, eq=False)
class LaidOutSpace:
    """One space: a space row of spaces.csv and its pieces (spec 6)."""

    name: str
    storey: int
    level_z: float
    #: The space row's room first, then its piece rows' rooms in file order.
    pieces: tuple[Room, ...]
    #: The sum of the pieces' ``area_m2``.
    area_m2: float
    #: One of ``ORIGINS``.
    origin: str
    #: Per piece: ``(source_outline, source_holes)`` for a ``SplitHalf``
    #: (its grid is laid over the original room), else ``None``.
    grid_source: tuple[tuple[np.ndarray, tuple[np.ndarray, ...]] | None, ...]
    #: The spaces.csv row name of each piece, aligned with ``pieces``.
    rows: tuple[str, ...]


@dataclasses.dataclass(frozen=True, eq=False)
class SpaceLayout:
    """Every space, in the spaces.csv order of the space rows."""

    spaces: tuple[LaidOutSpace, ...]

    def space(self, name: str) -> LaidOutSpace:
        for space in self.spaces:
            if space.name == name:
                return space
        known = ", ".join(repr(s.name) for s in self.spaces)
        raise LayoutError(f"no space {name!r} in the layout; the spaces are {known}")

    def pieces_by_space(self) -> dict[str, tuple[Room, ...]]:
        return {space.name: space.pieces for space in self.spaces}


def _origin(pieces: Sequence[Room]) -> str:
    split = any(isinstance(piece, SplitHalf) for piece in pieces)
    if len(pieces) == 1:
        return "split" if split else "detected"
    return "split+merged" if split else "merged"


def lay_out(spaces: Sequence[Space], matched: Mapping[str, Room]) -> SpaceLayout:
    """One ``LaidOutSpace`` per space row (spec 6). Refuses a row with no
    room in ``matched`` and a space whose pieces lie on two storeys."""
    for row in spaces:
        if row.room not in matched:
            raise LayoutError(f"space {row.room!r} has no matched room")
    owner = space_of(spaces)
    laid_out = []
    for head in space_rows(spaces):
        rows = (head.room, *(s.room for s in spaces if s.merge_into is not None and owner[s.room] == head.room))
        pieces = tuple(matched[name] for name in rows)
        first = pieces[0]
        for name, piece in zip(rows[1:], pieces[1:]):
            if piece.storey != first.storey:
                raise LayoutError(
                    f"space {name!r} is a piece of {head.room!r} (merge_into), but its room is on "
                    f"storey {piece.storey} (level {piece.level_z:.2f} m) and the space row's on "
                    f"storey {first.storey} (level {first.level_z:.2f} m); the pieces of a space "
                    f"must be on one storey: clear merge_into for {name!r}"
                )
        laid_out.append(LaidOutSpace(
            name=head.room,
            storey=first.storey,
            level_z=first.level_z,
            pieces=pieces,
            area_m2=sum(piece.area_m2 for piece in pieces),
            origin=_origin(pieces),
            grid_source=tuple(
                (piece.source_outline, piece.source_holes) if isinstance(piece, SplitHalf) else None
                for piece in pieces
            ),
            rows=rows,
        ))
    return SpaceLayout(tuple(laid_out))
