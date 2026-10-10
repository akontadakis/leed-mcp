# leed_daylight/build/groups.py
"""Match exterior glazing to a room and form window groups (spec 6.2).

A Radiance scene has no idea which opening belongs to which room, or which
openings share a facade. This module rebuilds both: every exterior glazing
polygon is matched to exactly one named room, oriented so its normal points
away from that room, binned into one of sixteen 22.5-degree facade
directions, and grouped by the tuple (room, storey, kind, bin). Task
3.10 offsets a blind along a group's ``outward_normals``; task 3.12 extrudes
a reveal the same way; phase 4 names a blind file and a daylight-coefficient
matrix after a group's ``group_id``. A renumbered id would silently reuse a
stale matrix, so the id is derived from the room's own name, never from a
counter.

``glazing_interior`` never reaches this module: only the roles in
``EXTERIOR_GLAZING_ROLES`` (``glazing``, ``glazing_curtain_wall``) make a
real opening in the envelope. ``glazing_diffusing`` is exterior too; it is
grouped by kind (``<room>_dif``), whatever its tilt.

Roof glass (skylights, sloped roof glass, roof monitors). Every pane is
first given a tilt class (``build.glazing_class``). Vertical and sloped
panes try the window rule above; a pane it matches keeps today's room and
orientation (position ``perimeter``), except in the one case today's rule
refused as "cannot tell which side": both plan steps inside the outline and
the nearer exit more than ``_SETBACK_LIMIT_M`` away. That pane, every
skylight-class pane, and every vertical or sloped pane with no room within
the window rule's tolerance on its storey, is roof glass (position
``roof``). A vertical or sloped pane whose window-rule room is not named in
``spaces.csv`` is unmatched facade glass, never roof glass: it is refused
with the other unmatched glazing and never reaches the roof search, so a
forgotten ``spaces.csv`` row cannot hand a facade window to a named room on
a storey below. A roof pane with no room yet
is found by the roof search: from the storey whose band holds its centroid
downward, the first storey with a room whose footprint covers the plan
centroid decides. Its outside is its upper side, except for vertical roof
glass (a monitor), whose outside is found by looking straight down from a
point 0.05 m to each side: the inside side reaches one of the room's own
floor polygons.

Rooms arrive as a mapping of space name to ``Room`` -- exactly what
``project.spaces.match_to_rooms`` returns -- because ``Room.room_id`` is a
sort-order index that renumbers whenever a room is added or resized, and a
window group's id must survive that. A room's z range for matching is its
storey BAND: ``[level_z, next storey's level_z)``, open above for the
highest storey in the mapping. ``Room.ceiling_height_m`` is never used for
this: it is derived from a mean z without a flatness check (a known defect
in ``scene.storeys``), and the storey band does not depend on it.

Every opening the scene has that carries an exterior glazing role but
cannot be matched to a named room is collected and refused together, by
name (spec 9's risk row: "the tool lists unmatched glazing and refuses to
run"). A malformed opening -- not a polygon, too few vertices, degenerate,
or non-finite -- is a
separate, immediate refusal naming that one opening, because it is a
different kind of problem: nothing about matching it to a room would fix
it.

A space may have several pieces (merge-split spec 7: the rooms of its
``merge_into`` rows, or split halves). Panes are matched to pieces with the
rules above, and groups are formed per piece: the bucket is (space, piece,
storey, kind, bin). A space of one piece keeps today's group ids; a space
of several adds ``_p<n>`` after the slug (``n`` the piece index from 1), so
each group's windows serve one piece. A pane that ties between two pieces
of the same space goes to the earlier piece; a tie between different
spaces is refused as before.

A group under ``MIN_GROUP_AREA_M2`` is kept, never dropped and never folded
into a different bin -- a group already is the finest grouping there is for
one room, storey and bin -- but a warning names it, since a run of daylight
coefficients on a sub-0.2 m2 patch of glass is easy to overlook.

Every numeric literal here is invented and round (phase3-global.md #5):
the bin width, the tiny-group threshold and the room-matching tolerance are
each a plain, round number, not read off a real project.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .. import geometry
from ..project.materials import EXTERIOR_GLAZING_ROLES
from ..scene.primitives import Primitive
from ..scene.storeys import Room, storey_levels
from ._names import slug as _slug
from .glazing_class import PaneClass, classify, tilt_class_of, tilt_deg
from .lookdown import first_hit_below

#: Spec 6.2 item 2: sixteen 22.5-degree compass bins.
AZIMUTH_BIN_DEG: float = 22.5
#: Spec 6.2 item 3: a group under this area is kept and warned about, never
#: merged or dropped (controller ruling R-P8).
MIN_GROUP_AREA_M2: float = 0.2
#: Spec 6.2 item 1: a glazing centroid may sit this far outside a room's
#: outline and still match it.
ROOM_MATCH_TOLERANCE_M: float = 0.6

#: Two room-match candidates this close in distance are an unresolved tie
#: (controller ruling: "never guess").
_TIE_TOLERANCE_M: float = 1e-9

#: Fix I2: how far to step from an opening's centroid, in plan, along its
#: own (candidate) normal when deciding which side faces the room's
#: interior. Short enough to stay local to the opening, long enough not to
#: be swallowed by floating-point noise -- a plain, round number, not read
#: off a real project.
_ORIENT_STEP_M: float = 0.05

#: Glass drawn mid-wall sits inside a room outline traced to the EXTERIOR
#: wall face. Such glass is oriented by which way it leaves the outline
#: sooner, but only if that exit lies within this distance -- a wall's
#: depth, the same round figure as the room-matching tolerance. Glass
#: deeper inside than this is not a set-back facade opening: it is roof
#: glass (spec 5.2), raised as ``_RoofBranch`` rather than refused.
_SETBACK_LIMIT_M: float = ROOM_MATCH_TOLERANCE_M

#: Monitor glass: how far to step from its centroid, along its horizontal
#: normal, before looking straight down (spec 5.3). A plain, round number.
_LOOKDOWN_STEP_M: float = 0.05

#: The roles the look-down ray can stop at (spec 5.3). Every other role --
#: glazing, shading, eaves, furniture, context -- is passed through.
#: Public: Task 8 imports this same set for its own look-down gathering.
LOOKDOWN_ROLES: frozenset[str] = frozenset({"wall", "ceiling", "floor"})

#: Kinds that have no facade azimuth: one group per room and storey.
_UNBINNED_KINDS: frozenset[str] = frozenset({"skylight", "diffusing"})


class GroupError(ValueError):
    """A window group cannot be formed from these inputs."""


class _RoofBranch(GroupError):
    """``_orient_outward``'s one refusal that now means roof glass: both
    plan steps inside the outline and the nearer exit farther than
    ``_SETBACK_LIMIT_M``. Same message as the refusal it replaces; the
    caller catches it, so it never leaves this module."""


@dataclass(frozen=True)
class WindowGroup:
    #: Stable across runs: ``f"{slug(room)}_{bin:03.0f}"`` for a window
    #: (R-P21); ``_s<bin>``, ``_m<bin>``, ``_sky`` and ``_dif`` for the
    #: other kinds (spec 5.7).
    group_id: str
    #: The space name from the ``rooms`` mapping, never ``Room.room_id``.
    room: str
    storey: int
    #: ``None`` for the skylight and diffusing kinds.
    azimuth_bin_deg: float | None
    openings: tuple[Primitive, ...]
    #: One outward unit normal per opening, aligned with ``openings``.
    #: Oriented by which side of the opening actually lands inside the
    #: matched room's own outline (I2), not trusted from the polygon's own
    #: winding and not simply flipped toward the room's 2-D centroid --
    #: that centroid can sit on the wrong side of a re-entrant wall for an
    #: L- or U-shaped room.
    outward_normals: tuple[np.ndarray, ...]
    #: Sum of the full opening areas, before any frame.
    area_m2: float
    #: ``window``, ``sloped``, ``monitor``, ``skylight`` or ``diffusing``.
    kind: str = "window"
    #: Area-weighted mean of the panes' tilts (90 = vertical), 6 decimals.
    tilt_deg: float = 90.0
    #: One class per opening, aligned with ``openings``.
    pane_classes: tuple[PaneClass, ...] = ()
    #: The 0-based index of the piece of the space ``room`` whose panes
    #: these are (merge-split spec 7); 0 for a space of one piece.
    piece: int = 0


def facade_azimuth_deg(outward_normal) -> float:
    """Degrees clockwise from +Y, in [0, 360); +X -> 90, -Y -> 180.

    The scene's own frame -- the north rotation is applied in the engine,
    not here. A normal with no horizontal component (vertical, a skylight)
    has no azimuth and is refused, as is anything that is not a finite
    3-vector.
    """
    try:
        n = np.asarray(outward_normal, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GroupError(
            f"outward_normal: not a valid vector of numbers: {exc}"
        ) from None
    if n.shape != (3,):
        raise GroupError(
            f"outward_normal: expected a 3-vector, got {outward_normal!r} "
            f"with shape {n.shape}"
        )
    if not np.all(np.isfinite(n)):
        raise GroupError(
            f"outward_normal: expected finite components, got {outward_normal!r}"
        )
    horiz = math.hypot(float(n[0]), float(n[1]))
    if horiz < geometry.DEGENERATE:
        raise GroupError(
            f"outward_normal {outward_normal!r} has a zero horizontal "
            "component; azimuth is undefined for a vertical normal"
        )
    return math.degrees(math.atan2(n[0], n[1])) % 360.0


def _bin_of(azimuth_deg: float) -> float:
    """Centred bin: 359 and 1 degree both fall in the bin at 0."""
    index = round(azimuth_deg / AZIMUTH_BIN_DEG) % 16
    return float(index) * AZIMUTH_BIN_DEG


def _validated_opening(p: Primitive, role: str):
    """Return ``(pts, normal, centroid, area)`` for one glazing opening, or
    raise ``GroupError`` naming ``p`` for anything that makes it unusable.
    """
    if p.ptype != "polygon":
        raise GroupError(
            f"{p.name}: modifier {p.modifier!r} has glazing role {role!r} "
            f"but the primitive is a {p.ptype!r}, not a polygon"
        )
    try:
        pts = p.points
    except ValueError as exc:
        raise GroupError(f"{p.name}: {exc}") from None
    n_vertices = 0 if pts is None else pts.shape[0]
    if pts is None or n_vertices < 3:
        raise GroupError(
            f"{p.name}: glazing polygon needs at least 3 vertices, got {n_vertices}"
        )
    if not np.all(np.isfinite(pts)):
        raise GroupError(f"{p.name}: glazing polygon has non-finite coordinates")

    try:
        normal_raw = geometry.normal(pts)
        centroid = geometry.centroid(pts)
        area = geometry.area(pts)
    except ValueError as exc:
        raise GroupError(f"{p.name}: {exc}") from None
    return pts, normal_raw, centroid, area


def _storey_bands(rooms: Sequence[Room]) -> dict[int, tuple[float, float]]:
    """This mapping's own storey bands: ``[level_z, next level_z)``, open
    above for the highest storey present. Never a global scene notion --
    only the storeys the caller's rooms actually cover."""
    storeys = sorted({r.storey for r in rooms})
    if not storeys:
        return {}
    levels = storey_levels(list(rooms))
    if len(levels) != len(storeys):
        raise GroupError(
            f"rooms: {len(storeys)} storey(s) but {len(levels)} distinct "
            "level(s) of level_z; every room on the same storey must share "
            "one level_z"
        )
    bands: dict[int, tuple[float, float]] = {}
    for i, storey in enumerate(storeys):
        lo = levels[i]
        hi = levels[i + 1] if i + 1 < len(levels) else math.inf
        bands[storey] = (lo, hi)
    return bands


def _in_band(bands: dict[int, tuple[float, float]], room: Room, z: float) -> bool:
    band = bands.get(room.storey)
    if band is None:
        raise GroupError(
            f"room on storey {room.storey}: no storey band was built for "
            "it; all_rooms must include every storey the named rooms sit on"
        )
    lo, hi = band
    return lo <= z < hi


def _orient_outward(
    normal_raw: np.ndarray, opening_centroid: np.ndarray, room: Room, name: str
) -> np.ndarray:
    """Flip ``normal_raw`` if needed so it points away from the room (I2).

    A room's 2-D centroid can sit on the WRONG side of a wall for an
    L- or U-shaped room: the old rule (flip toward/away from the room's
    own centroid) got the re-entrant walls of a concave room backwards.
    Instead, step ``_ORIENT_STEP_M`` from the opening's centroid, in plan,
    along ``normal_raw`` and along its negation; whichever side actually
    lands INSIDE the room's own outline is the inward side, so the other
    direction is outward. If both steps land inside -- glass drawn mid-wall
    inside an outline traced to the exterior wall face -- cast a ray each
    way in plan: the side that leaves the outline sooner is outward,
    provided that exit is within ``_SETBACK_LIMIT_M`` and the two distances
    differ; otherwise refuse, naming the opening. If neither step lands
    inside, fall back to whichever side sits nearer the room, by distance
    to the room polygon, and refuse only if that still ties.
    """
    poly = room.footprint
    xy = opening_centroid[:2]
    horiz = np.array([float(normal_raw[0]), float(normal_raw[1])])
    length = float(np.hypot(horiz[0], horiz[1]))
    if length < geometry.DEGENERATE:
        raise GroupError(
            f"{name}: outward_normal {normal_raw!r} has a zero horizontal "
            "component; its inward/outward side cannot be determined"
        )
    direction = horiz / length

    plus = Point(xy[0] + _ORIENT_STEP_M * direction[0], xy[1] + _ORIENT_STEP_M * direction[1])
    minus = Point(xy[0] - _ORIENT_STEP_M * direction[0], xy[1] - _ORIENT_STEP_M * direction[1])
    plus_inside = poly.contains(plus)
    minus_inside = poly.contains(minus)

    if plus_inside and not minus_inside:
        return -normal_raw
    if minus_inside and not plus_inside:
        return normal_raw

    if plus_inside and minus_inside:
        # Glass set back inside an outline drawn to the exterior wall face:
        # the outward side is the one that leaves the outline sooner.
        plus_exit = _exit_distance(poly, xy, direction)
        minus_exit = _exit_distance(poly, xy, -direction)
        nearer = min(plus_exit, minus_exit)
        if abs(plus_exit - minus_exit) > _TIE_TOLERANCE_M and nearer <= _SETBACK_LIMIT_M:
            return normal_raw if plus_exit < minus_exit else -normal_raw
        # Deeper than a wall: roof glass (spec 5.2), same message as before.
        refusal = _RoofBranch if nearer > _SETBACK_LIMIT_M else GroupError
        raise refusal(
            f"{name}: cannot tell which side of this opening faces the room's "
            f"interior; it sits inside the room's outline, {plus_exit:.3f} m "
            f"from its edge along one side of its normal and {minus_exit:.3f} m "
            f"along the other, and set-back glass must be within "
            f"{_SETBACK_LIMIT_M:g} m of the outline on exactly one side"
        )

    plus_dist = poly.distance(plus)
    minus_dist = poly.distance(minus)
    if plus_dist < minus_dist:
        return -normal_raw
    if minus_dist < plus_dist:
        return normal_raw

    raise GroupError(
        f"{name}: cannot tell which side of this opening faces the room's "
        f"interior; stepping {_ORIENT_STEP_M:g} m along its normal lands "
        "equally (in or out of) the room on both sides"
    )


def _exit_distance(poly: Polygon, xy: np.ndarray, direction: np.ndarray) -> float:
    """Plan distance from ``xy`` along ``direction`` to the first crossing of
    ``poly``'s boundary (inf if the ray never meets it)."""
    minx, miny, maxx, maxy = poly.bounds
    reach = 2.0 * math.hypot(maxx - minx, maxy - miny) + 2.0 * float(
        np.max(np.abs(xy))
    ) + 1.0
    ray = LineString([tuple(xy), tuple(xy + reach * direction)])
    hit = ray.intersection(poly.boundary)
    if hit.is_empty:
        return math.inf
    origin = Point(tuple(xy))
    parts = getattr(hit, "geoms", [hit])
    best = math.inf
    for part in parts:
        for coord in part.coords:
            d = origin.distance(Point(coord))
            if d > geometry.DEGENERATE:
                best = min(best, d)
    return best


def _describe_room(room: Room, named_by_id: dict[int, tuple[str, int]]) -> str:
    """A label for an error message: the space name if ``room`` is named,
    else something built from the room's own identity (never by value, and
    never a crash if ``room_id`` is odd)."""
    named = named_by_id.get(id(room))
    if named is not None:
        return named[0]
    return f"room {room.room_id!r} (not named in spaces.csv)"


def _one_space(rooms: Sequence[Room], named_by_id: dict[int, tuple[str, int]]) -> Room | None:
    """The earliest piece when every room of a tie is a piece of one
    space (merge-split spec 7), else ``None``."""
    named = [named_by_id.get(id(room)) for room in rooms]
    if any(n is None for n in named) or len({n[0] for n in named}) != 1:
        return None
    return min(zip(named, rooms), key=lambda t: t[0][1])[1]


def _match_room(
    name: str,
    xy: np.ndarray,
    z: float,
    candidate_rooms: Sequence[Room],
    bands: dict[int, tuple[float, float]],
    tolerance_m: float,
    named_by_id: dict[int, tuple[str, int]],
) -> Room | None:
    on_level = [room for room in candidate_rooms if _in_band(bands, room, z)]
    if not on_level:
        return None

    scored = sorted(
        ((room.footprint.distance(Point(xy)), room) for room in on_level),
        key=lambda t: t[0],
    )
    within = [t for t in scored if t[0] <= tolerance_m]
    if not within:
        return None

    best_d = within[0][0]
    tied = [t for t in within if abs(t[0] - best_d) <= _TIE_TOLERANCE_M]
    if len(tied) > 1:
        earliest = _one_space([room for _, room in tied], named_by_id)
        if earliest is not None:
            return earliest
        labels = ", ".join(sorted(_describe_room(room, named_by_id) for _, room in tied))
        raise GroupError(
            f"{name}: matches more than one room within {tolerance_m:g} m: {labels}"
        )
    _, room = within[0]
    return room


def _roof_search(
    name: str,
    xy: np.ndarray,
    z: float,
    candidate_rooms: Sequence[Room],
    bands: dict[int, tuple[float, float]],
    named_by_id: dict[int, tuple[str, int]],
) -> Room | None:
    """Spec 5.2 step 2: the room under a roof pane, or ``None``.

    Start at the storey whose band holds ``z`` and go down; the first
    storey with a room (named or not) whose footprint, holes respected,
    covers the plan centroid decides. Rooms on one storey are disjoint, so
    two covering rooms means the centroid lies on their shared boundary:
    refused, never guessed -- unless they are all pieces of one space,
    when the earliest piece takes the pane.
    """
    start = [storey for storey, (lo, hi) in bands.items() if lo <= z < hi]
    if not start:
        return None
    point = Point(float(xy[0]), float(xy[1]))
    for storey in sorted((s for s in bands if s <= start[0]), reverse=True):
        covering = [
            room for room in candidate_rooms
            if room.storey == storey and room.footprint.covers(point)
        ]
        if len(covering) == 1:
            return covering[0]
        if len(covering) > 1:
            earliest = _one_space(covering, named_by_id)
            if earliest is not None:
                return earliest
            labels = ", ".join(sorted(_describe_room(room, named_by_id) for room in covering))
            raise GroupError(
                f"{name}: roof glass lies on the boundary of more than one "
                f"room: {labels}"
            )
    return None


def structure_polygons(prims, roles: Mapping[str, str]) -> list[tuple[str, np.ndarray]]:
    """The ``wall``, ``ceiling`` and ``floor`` polygons the look-down ray
    can stop at. ``prepare`` validates every primitive before grouping, so
    a malformed one here is skipped rather than refused a second time.

    Public: Task 8 imports this same gathering for its own look-down use.
    """
    polygons: list[tuple[str, np.ndarray]] = []
    for p in prims:
        if roles.get(p.modifier) not in LOOKDOWN_ROLES or p.ptype != "polygon":
            continue
        try:
            pts = p.points
        except ValueError:
            continue
        if pts is None or len(pts) < 3 or not np.all(np.isfinite(pts)):
            continue
        polygons.append((p.name, pts))
    return polygons


def _reaches_floor(origin: np.ndarray, room: Room, structure) -> bool:
    hit = first_hit_below(origin, structure)
    if hit is None:
        return False
    hit_name, hit_point = hit
    return hit_name in room.floor_names and room.footprint.covers(
        Point(float(hit_point[0]), float(hit_point[1]))
    )


def _monitor_outward(
    normal_raw: np.ndarray, centroid: np.ndarray, room: Room, structure, name: str
) -> np.ndarray:
    """Spec 5.3: vertical roof glass. Step ``_LOOKDOWN_STEP_M`` to each
    side along the horizontal normal and look straight down; the side
    whose first hit is one of ``room``'s own floor polygons, inside its
    footprint, is inside. Both or neither: refused by name."""
    horiz = np.array([float(normal_raw[0]), float(normal_raw[1])])
    direction = horiz / float(np.hypot(horiz[0], horiz[1]))
    step = np.array([direction[0], direction[1], 0.0]) * _LOOKDOWN_STEP_M
    plus_inside = _reaches_floor(centroid + step, room, structure)
    minus_inside = _reaches_floor(centroid - step, room, structure)
    if plus_inside and not minus_inside:
        return -normal_raw
    if minus_inside and not plus_inside:
        return normal_raw
    side = "both sides" if plus_inside else "neither side"
    raise GroupError(
        f"{name}: monitor glass reaches its room's floor on {side}; looking "
        f"straight down from {_LOOKDOWN_STEP_M:g} m either side of it, exactly "
        "one side must reach a floor of the room it stands over"
    )


def _group_id(slug: str, kind: str, bin_deg: float | None, piece: int, pieces: int) -> str:
    if pieces > 1:
        slug = f"{slug}_p{piece + 1}"
    if kind == "skylight":
        return f"{slug}_sky"
    if kind == "diffusing":
        return f"{slug}_dif"
    if kind == "sloped":
        return f"{slug}_s{bin_deg:03.0f}"
    if kind == "monitor":
        return f"{slug}_m{bin_deg:03.0f}"
    return f"{slug}_{bin_deg:03.0f}"


def _build_groups(
    matched: list[tuple[str, int, Room, Primitive, np.ndarray, float, PaneClass]],
    piece_counts: Mapping[str, int],
) -> tuple[list[WindowGroup], tuple[str, ...]]:
    buckets: dict[
        tuple[str, int, int, str, float | None],
        list[tuple[Primitive, np.ndarray, float, PaneClass]],
    ] = {}
    for room_name, piece, room, p, outward, area, pane_class in matched:
        if pane_class.kind in _UNBINNED_KINDS:
            bin_deg = None
        else:
            bin_deg = _bin_of(facade_azimuth_deg(outward))
        key = (room_name, piece, room.storey, pane_class.kind, bin_deg)
        buckets.setdefault(key, []).append((p, outward, area, pane_class))

    slug_owner: dict[str, str] = {}
    id_owner: dict[str, str] = {}
    groups: list[WindowGroup] = []
    warnings: list[str] = []
    for (room_name, piece, storey, kind, bin_deg), entries in buckets.items():
        slug = _slug(room_name)
        owner = slug_owner.setdefault(slug, room_name)
        if owner != room_name:
            raise GroupError(
                f"room {room_name!r} and room {owner!r} both slug to "
                f"{slug!r}; rename one so window-group ids stay distinct"
            )
        group_id = _group_id(slug, kind, bin_deg, piece, piece_counts[room_name])
        other = id_owner.setdefault(group_id, room_name)
        if other != room_name:
            raise GroupError(
                f"window group id {group_id!r} comes from space {other!r} and from "
                f"space {room_name!r}; rename one so window-group ids stay distinct"
            )
        openings = tuple(p for p, _, _, _ in entries)
        outward_normals = tuple(n for _, n, _, _ in entries)
        total_area = float(sum(a for _, _, a, _ in entries))
        pane_classes = tuple(c for _, _, _, c in entries)
        mean_tilt = round(
            sum(a * c.tilt_deg for _, _, a, c in entries) / total_area, 6
        )
        if total_area < MIN_GROUP_AREA_M2:
            warnings.append(
                f"{group_id}: total glazing area {total_area:.3f} m2 is "
                f"below the {MIN_GROUP_AREA_M2:g} m2 minimum; kept as its "
                "own group"
            )
        groups.append(
            WindowGroup(
                group_id=group_id,
                room=room_name,
                storey=storey,
                azimuth_bin_deg=bin_deg,
                openings=openings,
                outward_normals=outward_normals,
                area_m2=total_area,
                kind=kind,
                tilt_deg=mean_tilt,
                pane_classes=pane_classes,
                piece=piece,
            )
        )

    groups.sort(key=lambda g: g.group_id)
    return groups, tuple(warnings)


def window_groups(
    prims,
    roles: Mapping[str, str],
    rooms: Mapping[str, Room | Sequence[Room]],
    *,
    all_rooms: Sequence[Room],
    tolerance_m: float = ROOM_MATCH_TOLERANCE_M,
) -> tuple[list[WindowGroup], tuple[str, ...]]:
    """Match every exterior glazing opening to a room and group it.

    Every glazing opening is matched, by distance, storey band and tie
    rule, against ALL of ``all_rooms`` -- every room ``detect_rooms`` found
    on the scene, named or not (I1, P28) -- never against ``rooms`` alone.
    Storey bands likewise come from this full set, so a storey with no
    named room still bounds its neighbours' bands correctly instead of
    leaving them open-ended. ``rooms`` (space name -> Room) says which of
    those matched rooms are actually usable: if the nearest room to an
    opening is not one of ``rooms``' values, the opening is unmatched, even
    though a room did exist within tolerance -- a narrow, un-named room
    must never silently steal glazing that only looks close to a named
    neighbour. Every ``Room`` in ``rooms`` must also appear in
    ``all_rooms`` (by identity, since ``Room`` is unhashable-by-value); one
    that does not is a caller error, refused immediately by naming the
    space.

    A value of ``rooms`` may be one ``Room`` or a space's pieces, in order
    (``project.layout.SpaceLayout.pieces_by_space``): each piece is matched
    like a room and grouped on its own (module docstring).

    Roof glass (module docstring, spec 5.2 and 5.3) is matched by the roof
    search and oriented by its upper side or, for a monitor, by the
    look-down test against the ``wall``, ``ceiling`` and ``floor``
    polygons of ``prims``.

    Returns ``(groups, warnings)``. ``groups`` is sorted by ``group_id``;
    openings within a group keep scene order. Every opening that matches
    no named room is refused, once, after every opening has been checked:
    facade glass under today's "unmatched exterior glazing" wording, roof
    glass as "roof glass over no named room".
    """
    all_rooms_list = list(all_rooms)
    all_room_ids = {id(r) for r in all_rooms_list}
    named_by_id: dict[int, tuple[str, int]] = {}
    piece_counts: dict[str, int] = {}
    for space_name, value in rooms.items():
        pieces = (value,) if isinstance(value, Room) else tuple(value)
        piece_counts[space_name] = len(pieces)
        for piece, room in enumerate(pieces):
            if id(room) not in all_room_ids:
                raise GroupError(
                    f"room {space_name!r}: this named room is not present in "
                    "all_rooms; every named room must also be one of the "
                    "rooms detect_rooms found on the scene"
                )
            named_by_id[id(room)] = (space_name, piece)

    bands = _storey_bands(all_rooms_list)

    openings: list[tuple[Primitive, np.ndarray, np.ndarray, float, str]] = []
    for p in prims:
        role = roles.get(p.modifier)
        if role not in EXTERIOR_GLAZING_ROLES:
            continue
        _, normal_raw, centroid, area = _validated_opening(p, role)
        openings.append((p, normal_raw, centroid, area, role))

    if not openings:
        return [], ()

    structure: list[tuple[str, np.ndarray]] | None = None
    unmatched: list[str] = []
    unmatched_roof: list[str] = []
    matched: list[tuple[str, int, Room, Primitive, np.ndarray, float, PaneClass]] = []

    for p, normal_raw, centroid, area, role in openings:
        z = float(centroid[2])
        xy = centroid[:2]
        tilt_class = tilt_class_of(tilt_deg(normal_raw))
        room: Room | None = None
        outward: np.ndarray | None = None
        position = "roof"

        if tilt_class != "skylight":
            found = _match_room(p.name, xy, z, all_rooms_list, bands, tolerance_m, named_by_id)
            if found is not None and id(found) not in named_by_id:
                # The window rule found this pane's room, and nobody named
                # it: facade glass of an unnamed room, refused as unmatched.
                # The roof search must not re-home it on a named room below.
                unmatched.append(p.name)
                continue
            if found is not None:
                room = found
                try:
                    outward = _orient_outward(normal_raw, centroid, found, p.name)
                    position = "perimeter"
                except _RoofBranch:
                    pass

        if room is None:
            found = _roof_search(p.name, xy, z, all_rooms_list, bands, named_by_id)
            if found is None or id(found) not in named_by_id:
                if tilt_class == "skylight" or found is not None:
                    unmatched_roof.append(p.name)
                else:
                    unmatched.append(p.name)
                continue
            room = found

        if outward is None:
            if tilt_class == "vertical":
                if structure is None:
                    structure = structure_polygons(prims, roles)
                outward = _monitor_outward(normal_raw, centroid, room, structure, p.name)
            else:
                outward = normal_raw if float(normal_raw[2]) > 0.0 else -normal_raw

        pane_class = classify(normal_raw, position, role == "glazing_diffusing")
        matched.append((*named_by_id[id(room)], room, p, outward, area, pane_class))

    if unmatched or unmatched_roof:
        parts = []
        if unmatched:
            parts.append(f"unmatched exterior glazing: {', '.join(sorted(set(unmatched)))}")
        parts.extend(
            f"{name}: roof glass over no named room" for name in sorted(set(unmatched_roof))
        )
        raise GroupError(
            "; ".join(parts)
            + "; every room with exterior glazing must be named in spaces.csv"
        )

    return _build_groups(matched, piece_counts)
