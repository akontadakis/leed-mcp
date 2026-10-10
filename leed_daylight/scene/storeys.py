"""Storeys and rooms, read from the floor polygons of the scene.

A Radiance scene has no idea what a room is.  The only structure it reliably
carries is its floors: horizontal, upward-facing, at a small number of levels.
So the levels become storeys and each connected patch of floor becomes a
candidate room, which a person then names, merges or splits.

An open-plan storey modelled as one slab arrives here as one room.  That is a
real limit of the method and the wizard states it rather than hiding it.

A floor polygon that is tilted -- even within the normal-angle tolerance --
is excluded rather than assigned a level from its mean z.  A slab like that
is not flat, so it is not a storey floor and contributes no room.  That is a
second limit of the method, alongside the open-plan one.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .. import geometry
from .primitives import Primitive
from .validate import crossing_edges

FLOOR_ROLES = frozenset({"floor"})
CEILING_ROLES = frozenset({"ceiling"})
#: Roles that are never floors, listed so the intent is visible to a reader.
EXCLUDED_ROLES = frozenset({"ground", "context", "ignore"})
assert not (FLOOR_ROLES & EXCLUDED_ROLES)

#: A storey-to-storey or floor-to-ceiling height outside this range, in
#: metres, is warned about: it is far more often a wrong `scene.units`
#: (cm read as mm, inches as feet) than a real building.  A warning, not
#: an error -- a mezzanine or a tall atrium can be real.
PLAUSIBLE_HEIGHT_M = (2.0, 10.0)


@dataclass(frozen=True)
class RoomDetection:
    """`detect_rooms`'s result: the rooms it found, plus what it excluded.

    Mirrors `Report` in `validate.py` -- results, plus what the caller
    should know.  A floor-role primitive dropped from the floor set takes
    its area out of the LEED area-weighted denominator; naming it here,
    rather than dropping it silently, is this project's error policy: an
    exclusion names the object.
    """
    rooms: tuple[Room, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class Room:
    #: A sort-order index ("S{storey}_R{n}"), not a stable identity: it
    #: renumbers whenever a room is added.  Spec 6.1 persists user edits by
    #: matching the room centroid within 0.5 m instead.  Never key a user
    #: annotation off room_id.
    room_id: str
    storey: int
    level_z: float
    outline: np.ndarray
    #: Shapely's 2D horizontal projection of the outline, not the 3D
    #: surface area.  That is the correct choice for LEED floor-area
    #: weighting, which uses occupiable plan area, not slab surface area.
    area_m2: float
    ceiling_height_m: float | None
    #: Interior rings of the floor patch, each (m, 2) and not closed: an
    #: atrium or a shaft that the merged floor pieces surround.  Part of
    #: the room's footprint only as a void -- no sensor, no floor area.
    holes: tuple[np.ndarray, ...] = ()
    #: Names of the floor primitives merged into this room, in scene order.
    #: Roof-monitor glass is oriented by which side looks down onto one of
    #: these (skylights spec 5.3). Empty for a room built by hand.
    floor_names: tuple[str, ...] = ()

    @property
    def footprint(self) -> Polygon:
        """The room's plan polygon: `outline` less every one of `holes`."""
        return Polygon(self.outline, [h for h in self.holes])


def _not_a_polygon(prim: Primitive) -> str | None:
    """Why `prim` cannot be read as a polygon, or None when it can.

    A floor- or ceiling-role primitive that is not a polygon (a sphere,
    say) or is a malformed one (a real count that is not a multiple of
    three) is named for what it is, not for failing the horizontal test.
    """
    if prim.ptype != "polygon":
        return f"a {prim.ptype} is not a polygon"
    try:
        pts = prim.points
    except ValueError as exc:
        return f"malformed polygon ({exc})"
    if len(pts) < 3:
        return "a polygon with fewer than three vertices"
    return None


def _horizontal(prim: Primitive, want_up: bool, tol_deg: float) -> bool:
    pts = prim.points
    if pts is None or len(pts) < 3:
        return False
    try:
        n = geometry.normal(pts)
    except ValueError:
        return False
    limit = np.cos(np.radians(tol_deg))
    return (n[2] >= limit) if want_up else (n[2] <= -limit)


def _flat(prim: Primitive, z_tolerance: float) -> bool:
    """True only if the polygon's own z range is within `z_tolerance`.

    A polygon can be near-horizontal -- its normal within the angle
    tolerance -- while still tilted enough that its corners span well
    more than z_tolerance.  `levels` assigns a floor's level from the
    MEAN z of its vertices, so a tilted-but-near-horizontal slab like
    that would be given a level that does not match its physical storey.
    Requiring the polygon's own z range (not its mean) to fit within
    z_tolerance is what keeps a slab like that from being treated as a
    storey floor at all.
    """
    pts = prim.points
    if pts is None or len(pts) < 3:
        return False
    z = pts[:, 2]
    return float(z.max() - z.min()) <= z_tolerance


#: A floor whose plan area changes by more than this, in square metres,
#: when its outline is normalised (`_plan_polygon`) is named in a warning.
_NORMALISE_AREA_TOLERANCE_M2 = 1e-6


def _plan_polygon(prim: Primitive, warnings: list[str] | None = None):
    """The primitive's plan polygon, normalised so a zero-width seam is not
    boundary.

    Radiance writes a polygon with a hole as a keyhole: the outer loop, a
    seam in, the hole the other way round, the same seam back out. Shapely
    reads that as one self-touching ring whose seam is boundary, so the
    hole is never an interior ring and sensor points near the seam are
    dropped. An invalid outline is rebuilt with ``buffer(0)``, which turns
    the seam into a proper interior ring (exterior and hole orientation as
    shapely's own union gives for the same floor in pieces) and keeps the
    area. A valid outline is returned as it is. If normalising changes the
    area (an outline that overlaps itself), the floor is named in
    ``warnings``.
    """
    poly = Polygon(prim.points[:, :2])
    if poly.is_valid:
        return poly
    fixed = poly.buffer(0)
    if warnings is not None and abs(fixed.area - abs(poly.area)) > _NORMALISE_AREA_TOLERANCE_M2:
        warnings.append(
            f"{prim.name}: its outline overlaps itself; its plan area is "
            f"taken as {fixed.area:.2f} m2, not the {abs(poly.area):.2f} m2 "
            "its vertex loop encloses"
        )
    return fixed


def plan_polygon(prim: Primitive) -> Polygon:
    """The primitive's plan polygon, normalised as `detect_rooms` normalises
    a floor (see `_plan_polygon`), without warnings: ``project/layout.py``
    uses it to find which floors a split half overlaps."""
    return _plan_polygon(prim)


def _cluster(values: Sequence[float], tolerance: float) -> list[list[int]]:
    """Group sorted values so every cluster spans at most `tolerance`.

    Each candidate is compared to its cluster's SEED -- the first (lowest)
    value in the cluster -- never to the last value added.  Comparing to
    the last value added lets a cluster drift without bound: a run of
    values each a small step above the one before, every step inside
    tolerance, can walk arbitrarily far from where the cluster started.
    Comparing to the seed is what keeps every cluster's own span bounded
    by `tolerance`.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    clusters: list[list[int]] = []
    for i in order:
        if clusters and abs(values[i] - values[clusters[-1][0]]) <= tolerance:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def detect_rooms(
    prims: Sequence[Primitive],
    roles: dict[str, str],
    z_tolerance: float = 0.05,
    normal_tolerance_deg: float = 5.0,
) -> RoomDetection:
    """Candidate rooms, one per connected patch of floor on each storey.

    Every primitive whose role is `floor` is a floor candidate.  One that
    is not a polygon, or is a malformed one, is warned about as such and
    skipped (a ceiling-role one likewise, rather than raising).  A
    candidate that fails `_horizontal` (not an upward-facing horizontal
    surface within `normal_tolerance_deg`) or `_flat` (its own z range
    exceeds `z_tolerance`, so it is not a storey floor even though its
    normal is inside the angle tolerance) is excluded from `floors` and
    warned about by name, rather than dropped silently -- a dropped floor
    would otherwise take its area out of the LEED area-weighted
    denominator with nothing looking wrong.  `_horizontal` is checked
    before `_flat`, so a candidate failing both warns once, for failing
    `_horizontal`: two warnings are never produced for the same root
    cause.  A storey-to-storey or ceiling height outside
    `PLAUSIBLE_HEIGHT_M`, and a room whose ceiling is not flat, are warned
    about too (the first two usually mean a wrong `scene.units`).  A
    candidate whose edges cross (a bow-tie) is excluded and
    warned about too: its plan area is wrong, and merged with a neighbour
    it would make shapely raise.  Only floor-role primitives are checked; a wall or a ceiling is
    never a candidate and is never warned about.
    """
    floor_candidates = [p for p in prims if roles.get(p.modifier) in FLOOR_ROLES]
    floors: list[Primitive] = []
    warnings: list[str] = []
    for p in floor_candidates:
        problem = _not_a_polygon(p)
        if problem is not None:
            warnings.append(f"{p.name}: {problem}, so it is not a storey floor")
            continue
        if not _horizontal(p, True, normal_tolerance_deg):
            warnings.append(f"{p.name}: not an upward-facing horizontal surface")
            continue
        if not _flat(p, z_tolerance):
            warnings.append(
                f"{p.name}: tilted; its own z range exceeds the z tolerance, "
                "so it is not a storey floor"
            )
            continue
        if crossing_edges(p.points) is not None:
            warnings.append(
                f"{p.name}: polygon crosses itself, so its plan area is wrong "
                "and it is not a storey floor"
            )
            continue
        floors.append(p)

    ceilings: list[Primitive] = []
    for p in prims:
        if roles.get(p.modifier) not in CEILING_ROLES:
            continue
        problem = _not_a_polygon(p)
        if problem is not None:
            warnings.append(
                f"{p.name}: {problem}, so it is not used as a ceiling"
            )
        elif _horizontal(p, False, normal_tolerance_deg):
            ceilings.append(p)
    if not floors:
        return RoomDetection((), tuple(warnings))

    levels = [float(p.points[:, 2].mean()) for p in floors]
    rooms: list[Room] = []
    lo, hi = PLAUSIBLE_HEIGHT_M
    previous_z = None
    for storey, indices in enumerate(_cluster(levels, z_tolerance)):
        level_z = float(np.mean([levels[i] for i in indices]))
        if previous_z is not None and not lo <= level_z - previous_z <= hi:
            warnings.append(
                f"storey {storey}: {level_z - previous_z:.2f} m above storey "
                f"{storey - 1}, outside the plausible {lo:g}-{hi:g} m; the "
                "declared scene.units may be wrong (cm vs mm, in vs ft)"
            )
        previous_z = level_z
        plans = {i: _plan_polygon(floors[i], warnings) for i in indices}
        merged = unary_union([plans[i] for i in indices])
        pieces = list(getattr(merged, "geoms", [merged]))
        for n, piece in enumerate(sorted(pieces, key=lambda g: (-g.area, g.bounds))):
            outline = np.asarray(piece.exterior.coords[:-1], dtype=np.float64)
            holes = tuple(
                np.asarray(ring.coords[:-1], dtype=np.float64)
                for ring in piece.interiors
            )
            room_id = f"S{storey}_R{n}"
            # Pieces of one union overlap no other piece, so a floor with
            # area inside this piece belongs to it; `floors` is in scene order.
            floor_names = tuple(
                floors[i].name for i in sorted(indices)
                if piece.intersection(plans[i]).area > 0.0
            )
            height, ceiling = ceiling_height(piece, level_z, ceilings)
            if height is not None and not lo <= height <= hi:
                warnings.append(
                    f"room {room_id}: ceiling height {height:.2f} m is outside "
                    f"the plausible {lo:g}-{hi:g} m; the declared scene.units "
                    "may be wrong (cm vs mm, in vs ft)"
                )
            if ceiling is not None and not _flat(ceiling, z_tolerance):
                warnings.append(
                    f"room {room_id}: its ceiling {ceiling.name} is not flat, so "
                    f"its ceiling height ({height:.2f} m) is from the mean z of a "
                    "sloped surface"
                )
            rooms.append(
                Room(
                    room_id=room_id,
                    storey=storey,
                    level_z=level_z,
                    outline=outline,
                    area_m2=float(piece.area),
                    ceiling_height_m=height,
                    holes=holes,
                    floor_names=floor_names,
                )
            )
    return RoomDetection(tuple(rooms), tuple(warnings))


def ceiling_height(
    piece, level_z: float, ceilings: Sequence[Primitive]
) -> tuple[float | None, Primitive | None]:
    """Height to the lowest downward-facing surface overlapping this room,
    and that surface (so a sloped one can be named); ``(None, None)`` when
    no ceiling overlaps. ``detect_rooms`` measures each room with it, and
    ``project/layout.py`` each half of a split room (``ceilings`` from
    `ceiling_surfaces`).

    "Overlapping" means positive shared area with the room's own footprint
    -- `piece.intersection(poly).area > 0` -- not `.intersects()`, which is
    also true for a ceiling that only touches the room along a shared edge
    or corner and supplies no real height for it.  `.overlaps()` is not the
    fix either: shapely's `.overlaps()` returns False whenever one polygon
    fully contains the other, and a ceiling exactly covering its room -- the
    normal case, since ceilings are usually modelled per room -- is exactly
    that shape.
    """
    best, best_prim = None, None
    for prim in ceilings:
        z = float(prim.points[:, 2].mean())
        if z <= level_z:
            continue
        poly = _plan_polygon(prim)
        if piece.intersection(poly).area <= 0:
            continue
        height = z - level_z
        if best is None or height < best:
            best, best_prim = height, prim
    return best, best_prim


def ceiling_surfaces(
    prims: Sequence[Primitive],
    roles: Mapping[str, str],
    normal_tolerance_deg: float = 5.0,
) -> tuple[Primitive, ...]:
    """The ceiling-role polygons `detect_rooms` measures ceiling heights
    against -- readable polygons facing down within
    ``normal_tolerance_deg`` -- in scene order. No warnings: `detect_rooms`
    already names every ceiling it cannot use."""
    return tuple(
        p for p in prims
        if roles.get(p.modifier) in CEILING_ROLES
        and _not_a_polygon(p) is None
        and _horizontal(p, False, normal_tolerance_deg)
    )


def storey_levels(rooms: Sequence[Room]) -> list[float]:
    """The z of each storey, in order."""
    return [z for _, z in sorted({(r.storey, r.level_z) for r in rooms})]
