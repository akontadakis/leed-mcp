# leed_daylight/geometry.py
"""Polygon maths in three dimensions.

Every length here is metres.  Newell's method is used for the normal because it
is correct for non-planar and concave polygons, which a real exported scene is
full of.  The centroid is the true area centroid, not the mean of the vertices:
a frame inset about the vertex mean of an L-shaped opening lands off centre.
"""
from __future__ import annotations

import numpy as np

DEGENERATE = 1e-12


def _as_points(pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected points of shape (n, 3), got {pts.shape}")
    if len(pts) < 3:
        raise ValueError(f"a polygon needs at least three vertices, got {len(pts)}")
    return pts


def _newell(pts: np.ndarray) -> np.ndarray:
    rolled = np.roll(pts, -1, axis=0)
    return np.cross(pts, rolled).sum(axis=0) / 2.0


def normal(pts) -> np.ndarray:
    """Unit normal.  Winding sets its direction."""
    vector = _newell(_as_points(pts))
    length = np.linalg.norm(vector)
    if length < DEGENERATE:
        raise ValueError("degenerate polygon: zero area, cannot take a normal")
    return vector / length


def area(pts) -> float:
    """Area, independent of winding, in whatever unit `pts` is expressed in.

    In this package that unit is metres: `read_scene` converts every length
    once, at intake, so every polygon this function ever sees is already
    metres.

    This function measures; it does not gate.  It returns 0.0 for a collinear
    (degenerate) polygon on purpose, rather than raising, so `validate_scene`
    can name the offending primitive and keep collecting every other finding.
    Making it raise would break the validator's never-raises contract.
    (`inventory()` already crashes on a malformed polygon by a different
    route -- an unguarded `prim.points` call -- so that particular crash
    does not depend on this function raising.)
    """
    return float(np.linalg.norm(_newell(_as_points(pts))))


def plane_basis(pts):
    """An orthonormal in-plane basis: (origin, u, v) with u x v along the normal."""
    pts = _as_points(pts)
    n = normal(pts)
    origin = pts[0]
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(seed, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return origin, u, v


def to_plane(pts, basis) -> np.ndarray:
    origin, u, v = basis
    rel = np.asarray(pts, dtype=np.float64) - origin
    return np.stack([rel @ u, rel @ v], axis=-1)


def from_plane(pts2, basis) -> np.ndarray:
    origin, u, v = basis
    pts2 = np.asarray(pts2, dtype=np.float64)
    return origin + pts2[..., 0:1] * u + pts2[..., 1:2] * v


def centroid(pts) -> np.ndarray:
    """The area centroid, computed in the polygon's own plane."""
    pts = _as_points(pts)
    basis = plane_basis(pts)
    p = to_plane(pts, basis)
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    signed = cross.sum() / 2.0
    if abs(signed) < DEGENERATE:
        raise ValueError("degenerate polygon: zero area, cannot take a centroid")
    cx = ((p[:, 0] + q[:, 0]) * cross).sum() / (6.0 * signed)
    cy = ((p[:, 1] + q[:, 1]) * cross).sum() / (6.0 * signed)
    return from_plane(np.array([cx, cy]), basis)


def planarity_error(pts) -> float:
    """Largest distance of any vertex from the best-fit plane, in metres."""
    pts = _as_points(pts)
    n = normal(pts)
    return float(np.abs((pts - pts.mean(axis=0)) @ n).max())


def point_in_polygon_2d(point, poly2) -> bool:
    """Ray casting.  A point exactly on an edge may fall either way."""
    point = np.asarray(point, dtype=np.float64)
    poly = np.asarray(poly2, dtype=np.float64)
    x, y = point
    inside = False
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y):
            t = (y - y1) / (y2 - y1)
            if x < x1 + t * (x2 - x1):
                inside = not inside
    return inside


#: A vertex this close (metres) to the straight line through its two
#: neighbours -- or to its predecessor -- adds no corner and is dropped by
#: `drop_redundant_vertices`. Well below any real frame width or room size.
COLLINEAR_TOL_M: float = 1e-6


def drop_redundant_vertices(pts: np.ndarray) -> np.ndarray:
    """Remove repeated vertices and vertices lying on the straight edge
    between their neighbours. Works on ``(n, 2)`` plan outlines and on
    ``(n, 3)`` polygons alike.

    Shapely's mitred buffer drops such points, so a solved window-frame
    inset would have fewer vertices than the outline and could not be
    matched to it (``build/frames.py``); a split half's outline is cleaned
    the same way (``project/layout.py``). A panel joint on a curtain-wall
    edge adds no corner, so dropping it changes neither the area nor the
    frame. A vertex that doubles back (a spike) is not "between" its
    neighbours and is left for the later checks to refuse.
    """
    out = [p for p in pts]
    changed = True
    while changed and len(out) > 3:
        changed = False
        for i in range(len(out)):
            prev_pt, cur_pt, next_pt = out[i - 1], out[i], out[(i + 1) % len(out)]
            if np.linalg.norm(cur_pt - prev_pt) <= COLLINEAR_TOL_M:
                del out[i]
                changed = True
                break
            chord = next_pt - prev_pt
            length = float(np.linalg.norm(chord))
            if length <= COLLINEAR_TOL_M:
                continue
            t = float(np.dot(cur_pt - prev_pt, chord)) / (length * length)
            off = float(np.linalg.norm(cur_pt - (prev_pt + t * chord)))
            if 0.0 < t < 1.0 and off <= COLLINEAR_TOL_M:
                del out[i]
                changed = True
                break
    return np.array(out, dtype=np.float64)
