# leed_daylight/build/lookdown.py
"""A vertical ray cast straight down onto polygons (skylights spec 5.3, 5.6).

Two questions about roof glass are answered by looking straight down from
a point: which side of a roof monitor's glass is the room (the side whose
downward ray lands on the room's own floor), and whether a ceiling
surface hides a skylight from the room below it. Both need only the first
polygon a downward ray meets, so that is all this module computes.

Pure: numpy only, no Radiance, no subprocess, no scene or room knowledge.
The caller chooses which polygons to pass (by role) and what a hit means.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: A hit must lie at least this far below the ray's origin (metres), so a
#: point that already sits on a surface never "hits" that same surface.
_BELOW_M = 1e-9
#: A polygon whose unit normal has a z component smaller than this is
#: vertical to the ray: the ray runs along it and never crosses it.
_VERTICAL_NZ = 1e-12
#: Plan distance (metres) within which a point counts as on a polygon's
#: edge; the edge belongs to the polygon, so a ray down a shared edge hits.
_ON_EDGE_M = 1e-9


def _newell_normal(pts: np.ndarray) -> np.ndarray:
    nxt = np.roll(pts, -1, axis=0)
    return np.array([
        np.sum((pts[:, 1] - nxt[:, 1]) * (pts[:, 2] + nxt[:, 2])),
        np.sum((pts[:, 2] - nxt[:, 2]) * (pts[:, 0] + nxt[:, 0])),
        np.sum((pts[:, 0] - nxt[:, 0]) * (pts[:, 1] + nxt[:, 1])),
    ])


def _covers_xy(pts: np.ndarray, x: float, y: float) -> bool:
    """True when (x, y) lies inside the polygon's plan outline or on its edge."""
    xs, ys = pts[:, 0], pts[:, 1]
    xn, yn = np.roll(xs, -1), np.roll(ys, -1)
    # On an edge: the point's distance to the segment is (almost) zero.
    dx, dy = xn - xs, yn - ys
    length2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(length2 > 0.0, ((x - xs) * dx + (y - ys) * dy) / length2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    gap = np.hypot(xs + t * dx - x, ys + t * dy - y)
    if np.any(gap <= _ON_EDGE_M):
        return True
    # Inside: even-odd rule on a ray towards +x.
    crosses = (ys > y) != (yn > y)
    with np.errstate(invalid="ignore", divide="ignore"):
        x_at = xs + (y - ys) * (xn - xs) / (yn - ys)
    return bool(np.count_nonzero(crosses & (x < x_at)) % 2)


def first_hit_below(
    origin, polygons: Sequence[tuple[str, np.ndarray]]
) -> tuple[str, np.ndarray] | None:
    """The first polygon a ray from ``origin`` straight down (toward -z) meets.

    ``polygons`` are ``(name, (n, 3) points)`` pairs, each planar. Returns
    ``(name, hit_point)`` for the hit nearest below ``origin`` -- the hit's
    z must be more than 1e-9 m below the origin's -- or ``None`` when the
    ray meets nothing. A hit on a polygon's edge counts. A vertical
    polygon is never hit. Among hits at the same height the first in
    ``polygons`` wins. A malformed origin or polygon is a ``ValueError``.
    """
    o = np.asarray(origin, dtype=np.float64)
    if o.shape != (3,) or not np.all(np.isfinite(o)):
        raise ValueError(f"origin: expected a finite 3-vector, got {origin!r}")
    ox, oy, oz = float(o[0]), float(o[1]), float(o[2])

    best: tuple[str, np.ndarray] | None = None
    best_z = -np.inf
    for name, points in polygons:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3 or not np.all(np.isfinite(pts)):
            raise ValueError(f"{name}: expected a finite (n, 3) polygon with n >= 3")
        normal = _newell_normal(pts)
        length = float(np.linalg.norm(normal))
        if length == 0.0 or abs(float(normal[2])) / length < _VERTICAL_NZ:
            continue
        p0 = pts.mean(axis=0)
        z = float(p0[2] - (normal[0] * (ox - p0[0]) + normal[1] * (oy - p0[1])) / normal[2])
        if not z < oz - _BELOW_M or not z > best_z:
            continue
        if _covers_xy(pts, ox, oy):
            best, best_z = (name, np.array([ox, oy, z])), z
    return best
