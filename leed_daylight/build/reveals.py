# leed_daylight/build/reveals.py
"""Build the window reveal side walls (spec 6.4, LM-83-12 clause 2.2.9 item 4).

A reveal is needed whenever the glass is set back from the wall face: the
side walls of that setback are extruded from the ORIGINAL opening outline
(not the inset glass outline a frame may have produced), inward and
outward, along the opening's own outward normal. Only exterior glazing
ever gets one; deciding which glazing that is is task 3.12's job -- this
module only builds the quads for an opening it is told to.

Two edges get no reveal. A sill at the floor or a head at the ceiling
(floor-to-ceiling glazing) would be a quad coplanar with that floor or
ceiling but with the reveal's reflectance -- two coincident surfaces
Radiance cannot tell apart. And the joint between two panes of one
curtain wall is not a setback at all: a reveal there would turn every
mullion line into a deep fin (``shared_edges`` finds such joints).

``reveal_depth_m: 0`` disables reveals entirely (spec 5.1): no default
depth exists anywhere in this module, and a depth of exactly zero returns
an empty tuple rather than a set of zero-area quads.

Roof glass (skylights design, section 5.6). Sloped glass on a roof keeps
only the inner half of its reveal (``inner_only``), so no curb stands
above the roof. A skylight gets a well instead (``skylight_well_walls``):
one vertical quad per edge of its rough opening, straight down to one
level plane ``depth_m`` under its lowest vertex. The joints between the
panes of one skylight -- a ridge, a hip -- get no well wall; unlike a
curtain wall's mullions they need not be coplanar, so
``well_shared_edges`` finds them without the coplanarity test.
"""
from __future__ import annotations

import math
from collections.abc import Collection, Sequence

import numpy as np

from .. import geometry

#: Controller ruling: an outward_normal more than this many degrees off
#: the opening plane's true normal (either sign) is refused rather than
#: silently used, since it would produce skewed or zero-area quads.
_NORMAL_TOLERANCE_DEG = 1.0

#: Fix round 1: an opening edge shorter than this (a duplicated consecutive
#: vertex) is refused rather than silently producing a zero-area quad.
_EDGE_LENGTH_TOLERANCE_M = 1e-9


#: A sill/head edge this close to the floor/ceiling level lies on it; two
#: pane edges this close to each other (or a pane this close to another's
#: plane) are one joint in one wall. A plain, round centimetre.
SURFACE_TOLERANCE_M = 0.01

#: Two panes are coplanar only if their planes are within this angle.
_COPLANAR_TOLERANCE_DEG = 1.0


class RevealError(ValueError):
    """A reveal cannot be built for this opening with these inputs."""


def _validated_points(opening_pts) -> np.ndarray:
    try:
        pts = np.asarray(opening_pts, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise RevealError(
            f"opening_pts: not a valid (n, 3) array of numbers: {exc}"
        ) from None
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise RevealError(
            f"opening_pts: expected an (n, 3) polygon with n >= 3, got shape {pts.shape}"
        )
    _validated_edges(pts)
    return pts


def _validated_edges(pts: np.ndarray) -> None:
    """Refuse any opening edge shorter than the tolerance.

    Newell's method (``geometry.normal``) does not notice a duplicated
    consecutive vertex -- the polygon's overall area and normal can still
    come out fine -- but the reveal quad built on that edge would have
    zero area, which this module's invariants forbid (fix round 1).
    """
    count = len(pts)
    for i in range(count):
        j = (i + 1) % count
        edge_length = float(np.linalg.norm(pts[j] - pts[i]))
        if edge_length < _EDGE_LENGTH_TOLERANCE_M:
            raise RevealError(
                f"opening_pts: edge {i} (vertex {i} to vertex {j}) has length "
                f"{edge_length:.3g} m, shorter than the {_EDGE_LENGTH_TOLERANCE_M:g} m "
                "tolerance; a reveal quad on it would have zero area"
            )


def _validated_depth(depth_m) -> float:
    try:
        depth = float(depth_m)
    except (TypeError, ValueError) as exc:
        raise RevealError(f"depth_m: expected a real number, got {depth_m!r}") from exc
    if not math.isfinite(depth):
        raise RevealError(f"depth_m: expected a finite number, got {depth_m!r}")
    if depth < 0.0:
        raise RevealError(f"depth_m: expected depth_m >= 0, got {depth_m!r}")
    return depth


def _validated_normal(outward_normal, pts: np.ndarray) -> np.ndarray:
    try:
        n = np.asarray(outward_normal, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise RevealError(
            f"outward_normal: not a valid 3-vector of numbers, got "
            f"{outward_normal!r}: {exc}"
        ) from None
    if n.shape != (3,):
        raise RevealError(
            f"outward_normal: expected a 3-vector, got {outward_normal!r} "
            f"with shape {n.shape}"
        )
    if not np.all(np.isfinite(n)):
        raise RevealError(
            f"outward_normal: expected finite components, got {outward_normal!r}"
        )
    length = float(np.linalg.norm(n))
    if length < geometry.DEGENERATE:
        raise RevealError(
            f"outward_normal: expected a nonzero vector along the opening's "
            f"outward normal, got {outward_normal!r} (zero length)"
        )
    unit = n / length

    try:
        plane_normal = geometry.normal(pts)
    except ValueError as exc:
        raise RevealError(f"opening_pts: {exc}") from None

    cos_angle = min(1.0, abs(float(np.dot(unit, plane_normal))))
    angle_deg = math.degrees(math.acos(cos_angle))
    if angle_deg > _NORMAL_TOLERANCE_DEG:
        raise RevealError(
            f"outward_normal {outward_normal!r} is {angle_deg:.3f} degrees from "
            f"the opening plane's normal {plane_normal!r} (tolerance "
            f"{_NORMAL_TOLERANCE_DEG:g} degrees, either sign); such a normal "
            "would make zero-area or skewed reveal quads"
        )
    return unit


def reveal_walls(
    opening_pts: np.ndarray,
    outward_normal: np.ndarray,
    depth_m: float,
    *,
    floor_z: float | None = None,
    ceiling_z: float | None = None,
    skip_edges: Collection[int] = frozenset(),
    inner_only: bool = False,
) -> tuple[np.ndarray, ...]:
    """One (4, 3) quad per opening edge, for the inward and outward extrusion.

    For opening edge ``(p_i, p_{i+1})`` and unit outward normal ``n``:

    * the inward quad is ``(p_i, p_{i+1}, p_{i+1} - d*n, p_i - d*n)``;
    * the outward quad is ``(p_i, p_{i+1}, p_{i+1} + d*n, p_i + d*n)``.

    Returns all inward quads first, then all outward quads (2n quads for
    an n-sided opening, fewer when edges are skipped, below); this order is deterministic and
    is the only thing this function promises about ordering.

    ``depth_m == 0`` returns an empty tuple without even checking
    ``outward_normal`` -- a disabled reveal has nothing to validate.
    ``depth_m`` must be finite and >= 0; anything else is a
    :class:`RevealError` (a :class:`ValueError`). ``outward_normal`` is
    normalised inside; a non-numeric input, a non-finite component (NaN
    or Inf), a zero-length normal, or one more than 1 degree away from
    the opening plane's own normal (either sign), is also a
    :class:`RevealError`. ``opening_pts`` is refused the same way when
    any consecutive pair of vertices is closer together than the edge
    tolerance (fix round 1): such a duplicated vertex would otherwise
    produce a zero-area quad.

    An edge gets neither quad when: both its ends are within
    ``SURFACE_TOLERANCE_M`` of ``floor_z`` (a sill on the floor) or of
    ``ceiling_z`` (a head at the ceiling) -- ``None`` checks nothing --
    or its index (edge ``i`` runs from vertex ``i`` to ``i + 1``) is in
    ``skip_edges``, typically from :func:`shared_edges`.

    ``inner_only`` returns the inward quads alone (sloped roof glass,
    spec 5.6): exactly the first half of the full result.
    """
    pts = _validated_points(opening_pts)
    depth = _validated_depth(depth_m)
    if depth == 0.0:
        return ()

    n = _validated_normal(outward_normal, pts)
    offset = depth * n

    count = len(pts)
    inward_quads = []
    outward_quads = []
    skip = set(skip_edges)
    for i in range(count):
        j = (i + 1) % count
        p_i, p_j = pts[i], pts[j]
        if i in skip or _on_level(p_i, p_j, floor_z) or _on_level(p_i, p_j, ceiling_z):
            continue
        inward_quads.append(np.stack([p_i, p_j, p_j - offset, p_i - offset]))
        outward_quads.append(np.stack([p_i, p_j, p_j + offset, p_i + offset]))
    if inner_only:
        return tuple(inward_quads)
    return tuple(inward_quads) + tuple(outward_quads)


def skylight_well_walls(
    opening_pts: np.ndarray,
    depth_m: float,
    *,
    skip_edges: Collection[int] = frozenset(),
) -> tuple[np.ndarray, ...]:
    """One vertical (4, 3) quad per edge of a skylight's rough opening.

    Edge ``(p_i, p_{i+1})`` gives ``(p_i, p_{i+1}, b_{i+1}, b_i)``, where
    ``b`` is the same point moved straight down (along -z) to the level
    plane ``z_low - depth_m``, and ``z_low`` is the opening's lowest
    vertex. So every wall is vertical and every bottom edge lies on one
    level plane, whatever the pane's tilt (spec 5.6). Edges whose index is
    in ``skip_edges`` (from :func:`well_shared_edges`) get no wall; the
    order is edge order. ``depth_m`` must be a finite number above zero;
    an edge with no length in plan (it would make a wall of no area) is
    refused, as is a malformed outline.
    """
    pts = _validated_points(opening_pts)
    depth = _validated_depth(depth_m)
    if depth == 0.0:
        raise RevealError(f"depth_m: a skylight well needs a depth above 0, got {depth_m!r}")
    bottom_z = float(pts[:, 2].min()) - depth
    skip = set(skip_edges)
    count = len(pts)
    walls = []
    for i in range(count):
        if i in skip:
            continue
        j = (i + 1) % count
        p_i, p_j = pts[i], pts[j]
        if float(np.hypot(*(p_j[:2] - p_i[:2]))) < _EDGE_LENGTH_TOLERANCE_M:
            raise RevealError(
                f"opening_pts: edge {i} (vertex {i} to vertex {j}) is vertical; "
                "a skylight well wall on it would have zero area"
            )
        b_i = np.array([p_i[0], p_i[1], bottom_z])
        b_j = np.array([p_j[0], p_j[1], bottom_z])
        walls.append(np.stack([p_i, p_j, b_j, b_i]))
    return tuple(walls)


def _on_level(p_i: np.ndarray, p_j: np.ndarray, z: float | None) -> bool:
    if z is None:
        return False
    return (
        abs(float(p_i[2]) - z) <= SURFACE_TOLERANCE_M
        and abs(float(p_j[2]) - z) <= SURFACE_TOLERANCE_M
    )


def shared_edges(
    outlines: Sequence[np.ndarray], tolerance_m: float = SURFACE_TOLERANCE_M
) -> tuple[frozenset[int], ...]:
    """For each outline, the indices of its edges that are joints with
    other, coplanar outlines -- one per outline, aligned with ``outlines``.

    An edge counts when the collinear edges of coplanar outlines (each end
    within ``tolerance_m`` of its line) together cover its whole length to
    within ``tolerance_m``: a shared mullion between two panes, or the top
    of a wide pane under two narrow ones (a T-joint). An edge only partly
    covered keeps its reveal. Outlines are coplanar when their planes are
    within 1 degree (either sign) and every vertex of one is within
    ``tolerance_m`` of the other's plane; panes a wall's depth apart, or
    meeting at a corner, are not.
    """
    pts_list = [np.asarray(o, dtype=np.float64) for o in outlines]
    normals = []
    for pts in pts_list:
        try:
            normals.append(geometry.normal(pts))
        except ValueError as exc:
            raise RevealError(f"opening_pts: {exc}") from None
    cos_limit = math.cos(math.radians(_COPLANAR_TOLERANCE_DEG))

    def coplanar(a: int, b: int) -> bool:
        if abs(float(np.dot(normals[a], normals[b]))) < cos_limit:
            return False
        offsets = (pts_list[b] - pts_list[a][0]) @ normals[a]
        return bool(np.all(np.abs(offsets) <= tolerance_m))

    return tuple(_joined_edges(pts_list, coplanar, tolerance_m))


def well_shared_edges(
    outlines: Sequence[np.ndarray], tolerance_m: float = SURFACE_TOLERANCE_M
) -> list[frozenset[int]]:
    """For each skylight outline, the indices of its edges shared with
    another outline, coplanar or not -- one per outline, aligned with
    ``outlines``.

    The same covering rule as :func:`shared_edges`, without its
    coplanarity test: the ridge of a two-pane skylight and the hips of a
    pyramid are joints, and a well wall there would stand inside the
    skylight. The caller passes the skylight-class panes of ONE room.
    """
    pts_list = [np.asarray(o, dtype=np.float64) for o in outlines]
    return _joined_edges(pts_list, lambda a, b: True, tolerance_m)


def _joined_edges(pts_list, partnered, tolerance_m: float) -> list[frozenset[int]]:
    """Per outline, the edges the collinear edges of its partner outlines
    (``partnered(a, b)``) cover end to end, within ``tolerance_m``."""
    result = []
    for a, pts in enumerate(pts_list):
        partners = [b for b in range(len(pts_list)) if b != a and partnered(a, b)]
        found = set()
        count = len(pts)
        for i in range(count):
            start, end = pts[i], pts[(i + 1) % count]
            length = float(np.linalg.norm(end - start))
            if length <= tolerance_m:
                continue
            direction = (end - start) / length
            spans = []
            for b in partners:
                other = pts_list[b]
                for k in range(len(other)):
                    q0, q1 = other[k], other[(k + 1) % len(other)]
                    if _off_line(q0, start, direction) > tolerance_m:
                        continue
                    if _off_line(q1, start, direction) > tolerance_m:
                        continue
                    t0 = float(np.dot(q0 - start, direction))
                    t1 = float(np.dot(q1 - start, direction))
                    spans.append((min(t0, t1), max(t0, t1)))
            if _covers(spans, length, tolerance_m):
                found.add(i)
        result.append(frozenset(found))
    return result


def _off_line(point: np.ndarray, start: np.ndarray, direction: np.ndarray) -> float:
    rel = point - start
    return float(np.linalg.norm(rel - np.dot(rel, direction) * direction))


def _covers(spans: list[tuple[float, float]], length: float, tolerance_m: float) -> bool:
    """Whether the union of ``spans`` covers ``[0, length]`` within tolerance."""
    reached = 0.0
    for lo, hi in sorted(spans):
        if lo > reached + tolerance_m:
            return False
        reached = max(reached, hi)
        if reached >= length - tolerance_m:
            return True
    return False
