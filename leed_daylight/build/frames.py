# leed_daylight/build/frames.py
"""Solve the constant-width window-frame inset (spec 6.3).

For a glazing polygon of area ``A`` and frame share ``s`` the glass that
remains must be ``(1 - s) * A``. Two styles produce that inset:

* ``constant_width`` (the default): every edge is offset inward by the
  same width ``w``, which is what a real frame profile looks like. The
  offset is a mitred negative buffer done with shapely, in the polygon's
  own plane (``geometry.plane_basis`` / ``to_plane`` / ``from_plane``), and
  ``w`` is found by bisection because the offset area is monotonically
  decreasing in ``w``. Shapely already implements the mitred (straight
  skeleton) offset correctly for non-convex outlines, so this module does
  not reimplement it.
* ``uniform_scale`` (the fallback): the outline is scaled by
  ``sqrt(1 - s)`` about its area centroid -- or, for a concave outline
  whose centroid cannot see every edge, about the centroid of its kernel;
  a concave outline with no kernel is refused (use ``constant_width``). Closed form, no solver error, but the
  border width grows with the opening size.

Repeated vertices and vertices on a straight edge (a panel joint) are
dropped before either style runs, so the glass may have fewer vertices
than the opening it came from.

The frame is the ring of quads between the original outline and the inset:
one quad per edge, because a Radiance polygon cannot have a hole.

No style has a built-in default share, VT or reflectance anywhere in this
module (phase3-global.md #7): a missing input is the caller's problem, not
this module's to paper over.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from shapely.geometry import Point, Polygon

from .. import geometry
from ..geometry import drop_redundant_vertices

#: Spec 6.3 item 3: the achieved glass area must be within 0.5 % of target.
AREA_TOLERANCE = 0.005

#: Bisection stop conditions (controller ruling): converge to 1e-9 relative
#: area error, or give up after this many halvings -- either way the
#: AREA_TOLERANCE check afterwards is what actually gates acceptance.
_BISECT_REL_TOL = 1e-9
_BISECT_MAX_ITER = 200
_BISECT_MAX_DOUBLINGS = 200

_STYLES = ("constant_width", "uniform_scale")


class FrameError(ValueError):
    """A frame cannot be built for this opening at this share."""


@dataclasses.dataclass(frozen=True)
class Frame:
    opening: str  # the primitive's name, for every message
    glass: np.ndarray  # (n, 3) the inset outline, in metres
    ring: tuple  # one (4, 3) quad per edge pair
    target_glass_area_m2: float
    achieved_glass_area_m2: float
    width_m: float | None  # the solved w; None for uniform_scale


def _validated_points(opening_pts, name: str) -> np.ndarray:
    try:
        pts = np.asarray(opening_pts, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise FrameError(
            f"{name}: opening is not a valid (n, 3) array of numbers: {exc}"
        ) from None
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise FrameError(
            f"{name}: expected an opening as an (n, 3) polygon with n >= 3, "
            f"got shape {pts.shape}"
        )
    return pts


def _kernel(poly2: Polygon) -> Polygon:
    """The polygon's kernel: every point from which the whole outline is
    visible (the intersection of the inner half-planes of its edges)."""
    coords = np.asarray(poly2.exterior.coords[:-1], dtype=np.float64)
    sign = 1.0 if _signed_area_2d(coords) > 0.0 else -1.0
    minx, miny, maxx, maxy = poly2.bounds
    reach = 4.0 * (maxx - minx + maxy - miny) + 1.0
    kernel = poly2
    n = len(coords)
    for i in range(n):
        a, b = coords[i], coords[(i + 1) % n]
        d = b - a
        d = d / np.linalg.norm(d)
        inward = sign * np.array([-d[1], d[0]])
        half = Polygon(
            [a - reach * d, b + reach * d, b + reach * d + reach * inward, a - reach * d + reach * inward]
        )
        kernel = kernel.intersection(half)
        if kernel.is_empty:
            return kernel
    return kernel


def _scale_centre(pts: np.ndarray, name: str) -> np.ndarray:
    """The point ``uniform_scale`` scales about.

    Scaling an outline about a point keeps it inside itself exactly when
    that point sees the whole outline -- lies in its kernel. For a convex
    opening the area centroid always does and is kept. For a concave one
    (an L) the centroid can sit outside the kernel and push glass past the
    re-entrant corner, so the kernel's own centroid is used instead. An
    opening with no kernel of positive area (a U, say) cannot be scaled
    inside itself about any point and is refused.
    """
    centre = geometry.centroid(pts)
    basis = geometry.plane_basis(pts)
    poly2 = Polygon(geometry.to_plane(pts, basis))
    if not poly2.is_valid or poly2.area <= geometry.DEGENERATE:
        raise FrameError(
            f"{name}: opening is not a valid simple polygon in its own plane"
        )
    kernel = _kernel(poly2)
    if kernel.is_empty or kernel.area <= geometry.DEGENERATE * max(1.0, poly2.area):
        raise FrameError(
            f"{name}: uniform_scale cannot frame this concave opening -- no "
            "point sees its whole outline, so scaling about any point pushes "
            "glass outside it; use frame_style: constant_width"
        )
    centre2 = geometry.to_plane(centre[None, :], basis)[0]
    if kernel.buffer(-1e-9).contains(Point(centre2)):
        return centre
    k = kernel.centroid
    return geometry.from_plane(np.array([k.x, k.y]), basis)


def _within_tolerance(achieved: float, target: float) -> bool:
    if target <= geometry.DEGENERATE:
        return abs(achieved) <= AREA_TOLERANCE
    return abs(achieved - target) <= AREA_TOLERANCE * target


def _relative_close(actual: float, expected: float, rel: float) -> bool:
    if abs(expected) <= geometry.DEGENERATE:
        return abs(actual) <= rel
    return abs(actual - expected) <= rel * abs(expected)


def _erosion_area(poly: Polygon, w: float) -> float:
    """Total area of the mitred negative buffer at width `w` (0 for empty)."""
    if w <= 0.0:
        return poly.area
    eroded = poly.buffer(-w, join_style="mitre")
    return 0.0 if eroded.is_empty else eroded.area


def _solve_width(poly: Polygon, target: float, name: str) -> float:
    """Bisect the width whose mitred erosion area equals `target`.

    The erosion area is monotonically non-increasing in `w` (erosion is
    inclusion-monotone), so this is an ordinary bisection once a bracket
    `[lo, hi]` with `area(lo) >= target >= area(hi)` is found.
    """
    lo, hi = 0.0, max(1e-6, (poly.area**0.5) * 0.01)
    hi_area = _erosion_area(poly, hi)
    doublings = 0
    while hi_area > target and doublings < _BISECT_MAX_DOUBLINGS:
        hi *= 2.0
        hi_area = _erosion_area(poly, hi)
        doublings += 1
    if hi_area > target:
        raise FrameError(
            f"{name}: could not bracket a width that reaches the target glass "
            f"area {target:.6f} m2; the opening looks degenerate"
        )

    w = hi
    for _ in range(_BISECT_MAX_ITER):
        w = (lo + hi) / 2.0
        area_w = _erosion_area(poly, w)
        if abs(area_w - target) <= _BISECT_REL_TOL * target:
            break
        if area_w > target:
            lo = w
        else:
            hi = w
    return w


def _signed_area_2d(pts2: np.ndarray) -> float:
    x, y = pts2[:, 0], pts2[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _miter_offsets(orig2d: np.ndarray, w: float, name: str) -> np.ndarray:
    """The analytic mitred inset point for every original vertex.

    Vertex `i`'s point is `orig[i] + w * (n_prev + n_next) / (1 + n_prev.n_next)`,
    where `n_prev`/`n_next` are the unit INWARD normals of the two edges
    meeting at that vertex. This is the same closed form as "the unit
    bisector of the two inward edge normals, scaled by 1 / cos(half the
    turn)" -- projecting the bisector-form vector onto either normal and
    requiring it equal `w` gives exactly this expression. It only LABELS
    the solved inset; shapely's mitred buffer remains the actual source of
    the geometry (spec 6.3's method).
    """
    n = len(orig2d)
    # `to_plane` need not hand back a CCW polygon for every basis choice,
    # so the interior side (which rotation of an edge direction points
    # inward) is read off the polygon's own signed area rather than
    # assumed.
    sign = 1.0 if _signed_area_2d(orig2d) > 0.0 else -1.0
    offsets = np.empty_like(orig2d)
    for i in range(n):
        prev_pt, cur_pt, next_pt = orig2d[i - 1], orig2d[i], orig2d[(i + 1) % n]
        d_prev, d_next = cur_pt - prev_pt, next_pt - cur_pt
        len_prev, len_next = np.linalg.norm(d_prev), np.linalg.norm(d_next)
        if len_prev <= geometry.DEGENERATE or len_next <= geometry.DEGENERATE:
            raise FrameError(f"{name}: opening has a zero-length edge at vertex {i}")
        d_prev, d_next = d_prev / len_prev, d_next / len_next
        n_prev = sign * np.array([-d_prev[1], d_prev[0]])
        n_next = sign * np.array([-d_next[1], d_next[0]])
        denom = 1.0 + np.dot(n_prev, n_next)
        if denom <= 1e-9:
            raise FrameError(
                f"{name}: vertex {i} turns back on itself (near 180 degrees); "
                "the mitred offset is undefined there"
            )
        offsets[i] = cur_pt + w * (n_prev + n_next) / denom
    return offsets


def _correspond(orig2d: np.ndarray, inset2d: np.ndarray, w: float, name: str) -> np.ndarray:
    """Label shapely's inset ring by construction, not by raw proximity.

    Shapely does not preserve the starting vertex or winding of a negative
    buffer. Matching each shapely vertex to its nearest ORIGINAL vertex
    (the previous approach) silently mislabels a concave outline whenever
    two non-adjacent original vertices sit closer together than either
    does to its own inset point -- the match stays one-to-one, so nothing
    flags it, but the returned "glass" swaps two vertices. Instead, the
    analytic mitred inset point for every original vertex is computed
    directly (`_miter_offsets`) and matched to whichever shapely vertex
    actually coincides with it; a miss, or a match that is not
    one-to-one, is refused rather than guessed at.
    """
    n = len(orig2d)
    analytic = _miter_offsets(orig2d, w, name)
    scale = abs(_signed_area_2d(orig2d)) ** 0.5
    tol = 1e-6 * scale
    dists = np.linalg.norm(analytic[:, None, :] - inset2d[None, :, :], axis=-1)
    nearest_inset_for_orig = dists.argmin(axis=1)
    min_dists = dists[np.arange(n), nearest_inset_for_orig]
    if np.any(min_dists > tol):
        bad = int(np.argmax(min_dists))
        raise FrameError(
            f"{name}: the solved inset does not coincide with the analytic "
            f"mitred offset at vertex {bad} (off by {min_dists[bad]:.6g} m, "
            f"tolerance {tol:.6g} m); the mitred offset came out malformed"
        )
    if len(set(nearest_inset_for_orig.tolist())) != n:
        raise FrameError(
            f"{name}: the inset vertices do not correspond one-to-one with "
            "the opening's vertices; the mitred offset came out malformed"
        )
    return inset2d[nearest_inset_for_orig]


def _build_ring(
    orig_pts: np.ndarray, inset_pts: np.ndarray, opening_normal: np.ndarray, name: str
) -> tuple:
    """One quad per edge, `(orig[i], orig[i+1], inset[i+1], inset[i])`.

    Every quad's normal is forced to match the opening's (dot product > 0,
    controller ruling); shapely's inset ring can come back either winding,
    so the quad's own vertex order is reversed when needed rather than
    trusted.
    """
    n = len(orig_pts)
    quads = []
    for i in range(n):
        j = (i + 1) % n
        quad = np.stack([orig_pts[i], orig_pts[j], inset_pts[j], inset_pts[i]])
        try:
            quad_normal = geometry.normal(quad)
        except ValueError as exc:
            raise FrameError(f"{name}: ring quad {i} is degenerate: {exc}") from None
        if np.dot(quad_normal, opening_normal) < 0.0:
            quad = quad[::-1].copy()
        quads.append(quad)
    return tuple(quads)


def build_frame(opening_pts: np.ndarray, name: str, share: float, style: str) -> Frame:
    pts = _validated_points(opening_pts, name)
    if not (0.0 <= share < 1.0):
        raise FrameError(
            f"{name}: frame share must satisfy 0 <= share < 1, got {share!r}"
        )
    if style not in _STYLES:
        raise FrameError(f"{name}: unknown frame style {style!r}, expected one of {_STYLES}")

    try:
        total_area = geometry.area(pts)
        opening_normal = geometry.normal(pts)
    except ValueError as exc:
        raise FrameError(f"{name}: {exc}") from None
    if total_area <= geometry.DEGENERATE:
        raise FrameError(f"{name}: opening has zero area")

    if share == 0.0:
        return Frame(
            opening=name,
            glass=pts.copy(),
            ring=(),
            target_glass_area_m2=total_area,
            achieved_glass_area_m2=total_area,
            width_m=0.0 if style == "constant_width" else None,
        )

    target = (1.0 - share) * total_area
    pts = drop_redundant_vertices(pts)

    if style == "uniform_scale":
        centre = _scale_centre(pts, name)
        scale = (1.0 - share) ** 0.5
        inset_pts = centre + scale * (pts - centre)
        achieved = total_area * scale * scale
        width_m = None
    else:
        basis = geometry.plane_basis(pts)
        pts2 = geometry.to_plane(pts, basis)
        poly = Polygon(pts2)
        if not poly.is_valid or poly.area <= geometry.DEGENERATE:
            raise FrameError(
                f"{name}: opening is not a valid simple polygon in its own plane"
            )

        w = _solve_width(poly, target, name)
        eroded = poly.buffer(-w, join_style="mitre")
        if eroded.is_empty:
            raise FrameError(
                f"{name}: the constant_width inset at width {w:.6f} m is empty; "
                f"the target glass area {target:.6f} m2 (share={share!r}) is not "
                "reachable with this style"
            )
        if not isinstance(eroded, Polygon):
            n_parts = len(eroded.geoms) if hasattr(eroded, "geoms") else "several"
            raise FrameError(
                f"{name}: the mitred inset split into {n_parts} pieces before "
                f"reaching the target glass area {target:.6f} m2 (share={share!r}); "
                "constant_width cannot build this frame"
            )

        coords2 = np.asarray(eroded.exterior.coords[:-1], dtype=np.float64)
        if len(coords2) != len(pts):
            raise FrameError(
                f"{name}: the solved inset has {len(coords2)} vertices, expected "
                f"{len(pts)} (a corner collapsed or was bevelled by the mitre join)"
            )

        matched2 = _correspond(pts2, coords2, w, name)
        inset_pts = geometry.from_plane(matched2, basis)
        achieved = float(eroded.area)
        width_m = w

    if not _within_tolerance(achieved, target):
        raise FrameError(
            f"{name}: achieved glass area {achieved:.6f} m2 is not within "
            f"{AREA_TOLERANCE:.1%} of the target {target:.6f} m2 (share={share!r})"
        )

    ring = _build_ring(pts, inset_pts, opening_normal, name)

    # Two post-conditions on what is actually returned, not on the
    # intermediate maths that got here: a correspondence bug (or any
    # future one) that assembles `glass`/`ring` from mismatched vertices
    # still produces a one-to-one, in-tolerance-looking construction, so
    # these check the assembled outputs directly rather than trusting the
    # steps that built them.
    returned_glass_area = geometry.area(inset_pts)
    if not _relative_close(returned_glass_area, achieved, 1e-9):
        raise FrameError(
            f"{name}: the returned glass area {returned_glass_area:.9f} m2 "
            f"does not match the reported achieved area {achieved:.9f} m2; "
            "the inset vertices are not what they were built from"
        )
    ring_area_total = sum(geometry.area(quad) for quad in ring) if ring else 0.0
    expected_ring_area = total_area - achieved
    if not _within_tolerance(ring_area_total, expected_ring_area):
        raise FrameError(
            f"{name}: ring area {ring_area_total:.6f} m2 does not match the "
            f"opening area minus the glass area {expected_ring_area:.6f} m2 "
            f"within {AREA_TOLERANCE:.1%}; the ring does not tile the border"
        )

    return Frame(
        opening=name,
        glass=inset_pts,
        ring=ring,
        target_glass_area_m2=target,
        achieved_glass_area_m2=achieved,
        width_m=width_m,
    )
