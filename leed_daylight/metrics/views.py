"""LEED v4.1 EQ Credit Quality Views: per ray, per point, per space and per
project (layer 5, pure: arrays in, numbers out).

The engine traces each view ray through the model one segment at a time.
This module reads those segments. For every ray it decides whether the ray
reaches exterior vision glass (the line of sight of Type 3) and what it
sees beyond that glass (the view content of Type 2). It then rolls rays up
into points, points into spaces and spaces into the project score.

Quality View = Type 2 and Type 3 (LEED v4.1, the ClimateStudio reading).
Comments cite sections of the Quality Views design notes as "QV spec 6.x";
the notes are not part of this repository.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from leed_daylight.constants import UNVERIFIED

from .aggregate import aggregate_counts
from .scoring import Score, score

#: What one traced segment hit. MISS and UNUSED are also the engine's own
#: modifier-index sentinels (engine.view_trace.MISS / UNUSED); a contract
#: test keeps the two in step.
KIND_UNUSED = -2
KIND_MISS = -1
KIND_EXTERIOR_GLASS = 1
KIND_INTERIOR_GLASS = 2
KIND_SEE_THROUGH = 3
KIND_BLOCKS = 4
KIND_TAG = 5

#: QV spec 6.4: two glass hits this close count as one layer (a partition
#: or a pane modelled as two faces). Same value as engine.view_trace.
#: Measured perpendicular to the glass when surface normals are known
#: (see same_layer), so a grazing ray does not split one partition in two.
LAYER_MERGE_M = 0.10
#: Two faces are one layer only if their normals are this close to
#: parallel (|cos| of the angle between them, about 2.6 deg). Same value
#: as engine.view_trace.
LAYER_PARALLEL_COS = 0.999

_TAG_CLASSES = frozenset({"nature", "art", "landmark"})
_CLEAR_KIND = {
    "exterior": KIND_EXTERIOR_GLASS,
    "interior": KIND_INTERIOR_GLASS,
    "none": KIND_SEE_THROUGH,
}


def kind_of(view_class: str, glazing: str) -> int:
    """The segment kind for a modifier's view class and its glazing family.

    ``glazing`` is "exterior" (role glazing or glazing_curtain_wall),
    "interior" (glazing_interior) or "none" (any other role).
    """
    if glazing not in _CLEAR_KIND:
        raise ValueError(f"glazing must be one of {sorted(_CLEAR_KIND)}, got {glazing!r}")
    if view_class in _TAG_CLASSES:
        return KIND_TAG
    if view_class == "blocks":
        return KIND_BLOCKS
    if view_class == "clear":
        return _CLEAR_KIND[glazing]
    raise ValueError(
        f"view class must be clear, blocks, nature, art or landmark, got {view_class!r}"
    )


def segment_kinds(modifier_index: np.ndarray, kind_by_modifier: Sequence[int]) -> np.ndarray:
    """Map the engine's per-segment modifier indices onto kinds.

    Negative indices (MISS, UNUSED) pass through unchanged.
    """
    index = np.asarray(modifier_index, dtype=np.int64)
    lookup = np.asarray(list(kind_by_modifier), dtype=np.int64)
    kinds = index.copy()
    known = index >= 0
    if known.any():
        if int(index[known].max()) >= lookup.size:
            raise ValueError(
                f"modifier index {int(index[known].max())} has no kind; "
                f"kind_by_modifier has {lookup.size} entries"
            )
        kinds[known] = lookup[index[known]]
    return kinds


@dataclass(frozen=True)
class RayOutcomes:
    """Per ray, shape (n,). Float fields are NaN where ``reach`` is False."""

    reach: np.ndarray
    plan_distance_m: np.ndarray
    head_height_m: np.ndarray
    outside_m: np.ndarray
    tagged: np.ndarray
    limit_hit: np.ndarray


def _open_distance(start: np.ndarray, directions: np.ndarray, ground_z: float) -> np.ndarray:
    """QV spec 6.4: a ray that leaves the scene. Upward or level: infinite.
    Downward: the distance to the ground plane z = ground_z."""
    out = np.full(start.shape[0], np.inf)
    dz = directions[:, 2]
    down = dz < 0
    out[down] = np.maximum((start[down, 2] - ground_z) / (-dz[down]), 0.0)
    return out


def same_layer(prev_point, prev_normal, point, normal) -> np.ndarray:
    """QV spec 6.4, per ray: is ``point`` on the same glass layer as ``prev_point``?

    Yes when the two hits are within LAYER_MERGE_M along the ray, or when
    their surfaces are parallel (LAYER_PARALLEL_COS) and the gap measured
    perpendicular to them -- the distance along the ray times the cosine
    of the ray's angle to the normal -- is within LAYER_MERGE_M. Without
    normals (``normal`` None) only the distance along the ray counts.
    NaN anywhere gives False.
    """
    gap = np.asarray(point, dtype=float) - np.asarray(prev_point, dtype=float)
    with np.errstate(invalid="ignore"):
        close = np.linalg.norm(gap, axis=1) < LAYER_MERGE_M
        if normal is None or prev_normal is None:
            return close
        n0 = _unit(prev_normal)
        n1 = _unit(normal)
        parallel = np.abs(np.sum(n0 * n1, axis=1)) >= LAYER_PARALLEL_COS
        perpendicular = np.abs(np.sum(gap * n1, axis=1))
        return close | (parallel & (perpendicular < LAYER_MERGE_M))


def _unit(vectors) -> np.ndarray:
    v = np.asarray(vectors, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def ray_outcomes(
    origins, directions, kinds, surfaces, hits, pane_top_z, *,
    level_z: float, max_interior_glass: int, skip_within_m: float, ground_z: float,
    normals=None,
) -> RayOutcomes:
    """QV spec 6.4-6.6 for every ray.

    ``normals`` (n, S, 3) are the surface normals at the hits, or None;
    they let two glass hits be judged one layer by their gap perpendicular
    to the glass (``same_layer``).

    Before the exterior glass: interior glass adds a layer (hits on the
    same layer as the previous one, ``same_layer``, do not), see-through
    surfaces are passed, anything else stops the ray. The first exterior
    glass hit is the exit. After the exit: any clear hit (a see-through
    surface, an interior glass surface such as a terrace balustrade, or
    the same glazing again) is looked through, like the spec's "a hit on a
    surface whose view class is clear restarts" rule; the first hit that
    is neither clear nor within ``skip_within_m`` of the exit point is
    what the ray sees; a miss is sky or ground.
    """
    origins = np.asarray(origins, dtype=float)
    directions = np.asarray(directions, dtype=float)
    kinds = np.asarray(kinds)
    surfaces = np.asarray(surfaces)
    hits = np.asarray(hits, dtype=float)
    pane_top = np.asarray(pane_top_z, dtype=float)
    if kinds.ndim != 2:
        raise ValueError(f"kinds must have shape (n, S), got {kinds.shape}")
    n, segments = kinds.shape
    if origins.shape != (n, 3) or directions.shape != (n, 3):
        raise ValueError(
            f"origins and directions must have shape ({n}, 3), got {origins.shape} and {directions.shape}"
        )
    if surfaces.shape != (n, segments) or hits.shape != (n, segments, 3):
        raise ValueError(
            f"surfaces and hits must have shapes ({n}, {segments}) and ({n}, {segments}, 3), "
            f"got {surfaces.shape} and {hits.shape}"
        )
    if normals is not None:
        normals = np.asarray(normals, dtype=float)
        if normals.shape != (n, segments, 3):
            raise ValueError(f"normals must have shape ({n}, {segments}, 3), got {normals.shape}")

    exited = np.zeros(n, dtype=bool)
    stopped = np.zeros(n, dtype=bool)
    found = np.zeros(n, dtype=bool)
    layers = np.zeros(n, dtype=np.int64)
    has_last = np.zeros(n, dtype=bool)
    last_interior = np.full((n, 3), np.nan)
    last_interior_normal = np.full((n, 3), np.nan)
    exit_point = np.full((n, 3), np.nan)
    exit_normal = np.full((n, 3), np.nan)
    exit_surface = np.full(n, -1, dtype=np.int64)
    outside = np.full(n, np.nan)
    tagged = np.zeros(n, dtype=bool)

    with np.errstate(invalid="ignore"):
        for s in range(segments):
            k = kinds[:, s]
            p = hits[:, s, :]
            nrm = None if normals is None else normals[:, s, :]

            after = exited & ~found
            if after.any():
                d = np.linalg.norm(p - exit_point, axis=1)
                miss = after & (k == KIND_MISS)
                real = after & (k >= KIND_EXTERIOR_GLASS)
                skip = real & (
                    (d < skip_within_m)
                    | (k == KIND_SEE_THROUGH)
                    | (k == KIND_INTERIOR_GLASS)
                    | ((k == KIND_EXTERIOR_GLASS)
                       & same_layer(exit_point, None if normals is None else exit_normal, p, nrm))
                )
                count = real & ~skip
                outside[count] = d[count]
                tagged[count] = k[count] == KIND_TAG
                if miss.any():
                    outside[miss] = _open_distance(exit_point[miss], directions[miss], ground_z)
                found |= count | miss

            before = ~exited & ~stopped
            if before.any():
                is_interior = before & (k == KIND_INTERIOR_GLASS)
                if is_interior.any():
                    merged = has_last & same_layer(
                        last_interior, None if normals is None else last_interior_normal, p, nrm,
                    )
                    layers += (is_interior & ~merged).astype(np.int64)
                    last_interior[is_interior] = p[is_interior]
                    if normals is not None:
                        last_interior_normal[is_interior] = nrm[is_interior]
                    has_last |= is_interior
                is_exit = before & (k == KIND_EXTERIOR_GLASS)
                exit_point[is_exit] = p[is_exit]
                if normals is not None:
                    exit_normal[is_exit] = nrm[is_exit]
                exit_surface[is_exit] = surfaces[is_exit, s]
                exited |= is_exit
                stopped |= before & ~(is_interior | is_exit | (k == KIND_SEE_THROUGH))

    ran_out_after = exited & ~found
    outside[ran_out_after] = 0.0
    limit_hit = ran_out_after | (~exited & ~stopped)
    reach = exited & (layers <= max_interior_glass)

    plan = np.hypot(exit_point[:, 0] - origins[:, 0], exit_point[:, 1] - origins[:, 1])
    top = np.full(n, np.nan)
    on_pane = exit_surface >= 0
    if on_pane.any():
        if int(exit_surface[on_pane].max()) >= pane_top.size:
            raise ValueError(
                f"surface index {int(exit_surface[on_pane].max())} has no pane top; "
                f"pane_top_z has {pane_top.size} entries"
            )
        top[on_pane] = pane_top[exit_surface[on_pane]]
    head = top - level_z

    nan = np.nan
    return RayOutcomes(
        reach=reach,
        plan_distance_m=np.where(reach, plan, nan),
        head_height_m=np.where(reach, head, nan),
        outside_m=np.where(reach, outside, nan),
        tagged=tagged & reach,
        limit_hit=limit_hit,
    )


def weighted_median(values, weights) -> float:
    """The weighted median (QV spec 6.6): the smallest value at which the
    running weight reaches half the total. NaN for an empty set."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.shape != w.shape:
        raise ValueError(f"values and weights differ in shape: {v.shape} and {w.shape}")
    if v.size == 0:
        return math.nan
    if np.isnan(v).any():
        raise ValueError("values must not be NaN")
    if (w < 0).any() or not w.sum() > 0:
        raise ValueError(f"weights must be >= 0 with a positive total, got total {w.sum()!r}")
    order = np.argsort(v, kind="stable")
    running = np.cumsum(w[order])
    index = int(np.searchsorted(running, 0.5 * running[-1], side="left"))
    return float(v[order][index])


@dataclass(frozen=True)
class PointViews:
    """Per point, shape (p,)."""

    type2: np.ndarray
    type3: np.ndarray
    median_outside_m: np.ndarray
    reaching_rays: np.ndarray
    limit_rays: np.ndarray

    @property
    def quality(self) -> np.ndarray:
        return self.type2 & self.type3

    @classmethod
    def concatenate(cls, parts: Sequence["PointViews"]) -> "PointViews":
        if not parts:
            raise ValueError("need at least one PointViews to concatenate")
        return cls(
            type2=np.concatenate([p.type2 for p in parts]),
            type3=np.concatenate([p.type3 for p in parts]),
            median_outside_m=np.concatenate([p.median_outside_m for p in parts]),
            reaching_rays=np.concatenate([p.reaching_rays for p in parts]),
            limit_rays=np.concatenate([p.limit_rays for p in parts]),
        )


def point_views(
    outcomes: RayOutcomes, weights, *, head_height_multiple: float, object_distance_m: float,
) -> PointViews:
    """QV spec 6.5 and 6.6. ``len(weights)`` rays per point, points in order."""
    w = np.asarray(weights, dtype=float)
    n = outcomes.reach.size
    m = w.size
    if m == 0 or n % m:
        raise ValueError(f"{n} rays do not split into points of {m} rays each")
    p = n // m
    reach = outcomes.reach.reshape(p, m)
    with np.errstate(invalid="ignore"):
        within = reach & (
            outcomes.plan_distance_m.reshape(p, m)
            <= head_height_multiple * outcomes.head_height_m.reshape(p, m)
        )
        tag = (reach & outcomes.tagged.reshape(p, m)).any(axis=1)
        outside = outcomes.outside_m.reshape(p, m)
        median = np.array([weighted_median(outside[i, reach[i]], w[reach[i]]) for i in range(p)])
        far = median >= object_distance_m
    return PointViews(
        type2=tag | far,
        type3=within.any(axis=1),
        median_outside_m=median,
        reaching_rays=reach.sum(axis=1),
        limit_rays=outcomes.limit_hit.reshape(p, m).sum(axis=1),
    )


@dataclass(frozen=True)
class SpaceViews:
    room: str
    area_m2: float
    points: int
    type2_points: int
    type3_points: int
    quality_points: int

    @property
    def type2_pct(self) -> float:
        return 100.0 * self.type2_points / self.points

    @property
    def type3_pct(self) -> float:
        return 100.0 * self.type3_points / self.points

    @property
    def quality_pct(self) -> float:
        return 100.0 * self.quality_points / self.points

    @property
    def quality_area_m2(self) -> float:
        return self.area_m2 * self.quality_points / self.points


def space_views(room: str, area_m2: float, points: PointViews) -> SpaceViews:
    """QV spec 6.7: the share of a space's points passing each type."""
    count = int(points.type2.size)
    if count == 0:
        raise ValueError(f"space {room!r} has no points")
    if not area_m2 > 0:
        raise ValueError(f"space {room!r}: area must be > 0, got {area_m2!r}")
    return SpaceViews(
        room=room,
        area_m2=float(area_m2),
        points=count,
        type2_points=int(points.type2.sum()),
        type3_points=int(points.type3.sum()),
        quality_points=int(points.quality.sum()),
    )


@dataclass(frozen=True)
class ProjectViews:
    quality_area_pct: float
    quality_point_pct: float
    type2_area_pct: float
    type3_area_pct: float
    total_area_m2: float
    quality_area_m2: float
    score: Score


def project_views(
    spaces: Sequence[SpaceViews], *, threshold_pct: float | str, points: int,
) -> ProjectViews:
    """QV spec 6.7: area-weighted shares, the point-weighted figure beside
    them, and the score. ``threshold_pct`` UNVERIFIED gives no point."""
    if not spaces:
        raise ValueError("need at least one space")
    areas = [s.area_m2 for s in spaces]
    totals = [s.points for s in spaces]
    # Exact integer counts, as Option 1 does, so a share on a threshold stays on it.
    quality = aggregate_counts(areas, [s.quality_points for s in spaces], totals)
    type2 = aggregate_counts(areas, [s.type2_points for s in spaces], totals)
    type3 = aggregate_counts(areas, [s.type3_points for s in spaces], totals)
    if threshold_pct == UNVERIFIED:
        bands = UNVERIFIED
    else:
        bands = [{"min_pct": float(threshold_pct), "points": int(points)}]
    headline = min(max(quality["area_pct"], 0.0), 100.0)
    return ProjectViews(
        quality_area_pct=quality["area_pct"],
        quality_point_pct=quality["point_pct"],
        type2_area_pct=type2["area_pct"],
        type3_area_pct=type3["area_pct"],
        total_area_m2=float(sum(areas)),
        quality_area_m2=float(sum(s.quality_area_m2 for s in spaces)),
        score=score(headline, bands),
    )
