"""metrics.views, per point, per space and per project. Invented numbers."""
from __future__ import annotations

import math

import numpy as np
import pytest

from leed_daylight.constants import UNVERIFIED
from leed_daylight.metrics.views import (
    PointViews,
    RayOutcomes,
    SpaceViews,
    point_views,
    project_views,
    space_views,
    weighted_median,
)

INF = math.inf


def _outcomes(reach, plan, head, outside, tagged=None):
    reach = np.asarray(reach, dtype=bool)
    n = reach.size
    return RayOutcomes(
        reach=reach,
        plan_distance_m=np.where(reach, np.asarray(plan, dtype=float), np.nan),
        head_height_m=np.where(reach, np.asarray(head, dtype=float), np.nan),
        outside_m=np.where(reach, np.asarray(outside, dtype=float), np.nan),
        tagged=np.zeros(n, bool) if tagged is None else np.asarray(tagged, dtype=bool),
        limit_hit=np.zeros(n, bool),
    )


def _points(outcomes, rays_per_point):
    return point_views(
        outcomes, np.ones(rays_per_point), head_height_multiple=3.0, object_distance_m=7.5
    )


def test_weighted_median():
    assert weighted_median([1.0, 2.0, INF], [1, 1, 1]) == 2.0
    assert weighted_median([1.0, 2.0, INF], [1, 1, 3]) == INF
    assert weighted_median([5.0, 10.0], [1, 1]) == 5.0
    assert weighted_median([10.0, 5.0], [1, 3]) == 5.0
    assert math.isnan(weighted_median([], []))
    with pytest.raises(ValueError, match="NaN"):
        weighted_median([1.0, float("nan")], [1, 1])
    with pytest.raises(ValueError, match="weights"):
        weighted_median([1.0], [0.0])


def test_type3_passes_within_three_head_heights():
    # A room 12 m deep, one window with its head 2.0 m above the floor:
    # points at 0.5, 1.5, ..., 11.5 m, one reaching ray each.
    distances = np.arange(12) + 0.5
    points = _points(_outcomes(np.ones(12), distances, np.full(12, 2.0), np.full(12, INF)), 1)

    assert points.type3.tolist() == [d <= 6.0 for d in distances]
    assert points.type3.mean() == 0.5


def test_a_column_that_hides_some_rays_does_not_fail_the_point():
    partly = _points(_outcomes([False, True, True], [1, 1, 1], [2, 2, 2], [INF, INF, INF]), 3)
    fully = _points(_outcomes([False, False, False], [1, 1, 1], [2, 2, 2], [INF, INF, INF]), 3)

    assert partly.type3.tolist() == [True]
    assert fully.type3.tolist() == [False]


def test_type2_uses_the_median_outside_distance():
    far = _points(_outcomes([True] * 3, [1] * 3, [2] * 3, [3.0, 9.0, INF]), 3)
    near = _points(_outcomes([True] * 3, [1] * 3, [2] * 3, [3.0, 3.0, 9.0]), 3)

    assert far.median_outside_m.tolist() == [9.0]
    assert far.type2.tolist() == [True]
    assert near.type2.tolist() == [False]


def test_type2_passes_on_a_tagged_surface():
    tagged = _points(_outcomes([True] * 3, [1] * 3, [2] * 3, [3.0, 3.0, 3.0], tagged=[False, True, False]), 3)

    assert tagged.type2.tolist() == [True]


def test_a_point_with_no_reaching_ray_fails_both_types():
    none = _points(_outcomes([False] * 3, [0] * 3, [0] * 3, [0] * 3), 3)

    assert none.type2.tolist() == [False]
    assert none.type3.tolist() == [False]
    assert math.isnan(none.median_outside_m[0])
    assert none.reaching_rays.tolist() == [0]


def test_quality_is_type2_and_type3():
    points = PointViews(
        type2=np.array([True, True, False]), type3=np.array([True, False, True]),
        median_outside_m=np.array([INF, INF, 1.0]), reaching_rays=np.array([1, 1, 1]),
        limit_rays=np.array([0, 0, 0]),
    )

    assert points.quality.tolist() == [True, False, False]


def test_point_views_concatenate():
    a = _points(_outcomes([True], [1], [2], [INF]), 1)
    b = _points(_outcomes([False], [0], [0], [0]), 1)

    both = PointViews.concatenate([a, b])

    assert both.type3.tolist() == [True, False]
    assert both.reaching_rays.tolist() == [1, 0]


def test_the_rays_must_split_evenly_into_points():
    with pytest.raises(ValueError, match="5 rays"):
        _points(_outcomes([True] * 5, [1] * 5, [2] * 5, [INF] * 5), 2)


def test_space_views():
    points = PointViews(
        type2=np.array([True, True, True, False]), type3=np.array([True, True, False, False]),
        median_outside_m=np.full(4, INF), reaching_rays=np.ones(4, int), limit_rays=np.zeros(4, int),
    )

    space = space_views("Room A", 20.0, points)

    assert (space.points, space.type2_points, space.type3_points, space.quality_points) == (4, 3, 2, 2)
    assert space.quality_pct == 50.0
    assert space.quality_area_m2 == 10.0


def test_project_views_weights_by_area_and_prints_points_beside_it():
    a = SpaceViews("A", 10.0, points=4, type2_points=4, type3_points=4, quality_points=4)
    b = SpaceViews("B", 30.0, points=100, type2_points=0, type3_points=100, quality_points=0)

    result = project_views([a, b], threshold_pct=75.0, points=1)

    assert result.quality_area_pct == pytest.approx(25.0)
    assert result.quality_point_pct == pytest.approx(400 / 104)
    assert result.type3_area_pct == pytest.approx(100.0)
    assert result.total_area_m2 == 40.0
    assert result.quality_area_m2 == 10.0
    assert result.score.points == 0


def test_the_score():
    full = SpaceViews("A", 10.0, points=4, type2_points=4, type3_points=4, quality_points=4)
    three_quarters = SpaceViews("B", 10.0, points=4, type2_points=3, type3_points=3, quality_points=3)

    assert project_views([full], threshold_pct=75.0, points=1).score.points == 1
    assert project_views([three_quarters], threshold_pct=75.0, points=1).score.points == 1
    assert project_views([full], threshold_pct=UNVERIFIED, points=1).score.points is None
