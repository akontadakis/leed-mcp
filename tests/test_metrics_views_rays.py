"""metrics.views, per ray: what a traced ray reaches and what it sees.

Every ray is written out as its segments, one (kind, surface, hit point)
triple each. Surface 0 is an exterior pane whose top is 2.0 m above z = 0.
All geometry is invented and round.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from leed_daylight.metrics.views import (
    KIND_BLOCKS,
    KIND_EXTERIOR_GLASS,
    KIND_INTERIOR_GLASS,
    KIND_MISS,
    KIND_SEE_THROUGH,
    KIND_TAG,
    KIND_UNUSED,
    kind_of,
    ray_outcomes,
    segment_kinds,
)

EYE = (0.0, 0.0, 1.1)
PANE_TOP = np.array([2.0])


def _at(x, z=1.1):
    return (x, 0.0, z)


def _rays(*rays, direction=(1.0, 0.0, 0.0)):
    n = len(rays)
    s = max(len(r) for r in rays)
    kinds = np.full((n, s), KIND_UNUSED)
    surfaces = np.full((n, s), -1)
    hits = np.full((n, s, 3), np.nan)
    for i, ray in enumerate(rays):
        for j, (kind, surface, point) in enumerate(ray):
            kinds[i, j] = kind
            surfaces[i, j] = surface
            if point is not None:
                hits[i, j] = point
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return np.tile(EYE, (n, 1)), np.tile(d, (n, 1)), kinds, surfaces, hits


def _outcomes(rays, *, level_z=0.0, max_interior_glass=1, skip_within_m=2.0, ground_z=0.0, pane_top=PANE_TOP):
    origins, directions, kinds, surfaces, hits = rays
    return ray_outcomes(
        origins, directions, kinds, surfaces, hits, pane_top,
        level_z=level_z, max_interior_glass=max_interior_glass,
        skip_within_m=skip_within_m, ground_z=ground_z,
    )


GLASS = (KIND_EXTERIOR_GLASS, 0, _at(3.0))
SKY = (KIND_MISS, -1, None)


def test_kind_of():
    assert kind_of("clear", "exterior") == KIND_EXTERIOR_GLASS
    assert kind_of("clear", "interior") == KIND_INTERIOR_GLASS
    assert kind_of("clear", "none") == KIND_SEE_THROUGH
    assert kind_of("blocks", "exterior") == KIND_BLOCKS
    for tag in ("nature", "art", "landmark"):
        assert kind_of(tag, "none") == KIND_TAG
    with pytest.raises(ValueError, match="'opaque'"):
        kind_of("opaque", "none")
    with pytest.raises(ValueError, match="'outside'"):
        kind_of("clear", "outside")


def test_segment_kinds_keeps_miss_and_unused():
    index = np.array([[0, 1, -1, -2]])

    assert segment_kinds(index, [KIND_BLOCKS, KIND_TAG]).tolist() == [[KIND_BLOCKS, KIND_TAG, KIND_MISS, KIND_UNUSED]]


def test_a_ray_through_exterior_glass_to_open_sky():
    out = _outcomes(_rays([GLASS, SKY]))

    assert out.reach.tolist() == [True]
    assert out.plan_distance_m[0] == pytest.approx(3.0)
    assert out.head_height_m[0] == pytest.approx(2.0)
    assert math.isinf(out.outside_m[0])
    assert out.tagged.tolist() == [False]
    assert out.limit_hit.tolist() == [False]


def test_the_outside_distance_is_measured_from_the_glass():
    out = _outcomes(_rays([GLASS, (KIND_BLOCKS, -1, _at(8.0))]))

    assert out.outside_m[0] == pytest.approx(5.0)


def test_a_tagged_surface_is_seen():
    out = _outcomes(_rays([GLASS, (KIND_TAG, -1, _at(8.0))]))

    assert out.tagged.tolist() == [True]
    assert out.outside_m[0] == pytest.approx(5.0)


def test_an_object_inside_the_cut_off_is_skipped_or_counted():
    rays = _rays([GLASS, (KIND_BLOCKS, -1, _at(4.0)), SKY])

    assert math.isinf(_outcomes(rays, skip_within_m=2.0).outside_m[0])
    assert _outcomes(rays, skip_within_m=0.0).outside_m[0] == pytest.approx(1.0)


def test_a_see_through_surface_outside_is_looked_through():
    out = _outcomes(_rays([GLASS, (KIND_SEE_THROUGH, -1, _at(6.0)), SKY]), skip_within_m=0.0)

    assert math.isinf(out.outside_m[0])


def test_a_clear_interior_glass_surface_outside_is_looked_through():
    out = _outcomes(_rays([GLASS, (KIND_INTERIOR_GLASS, -1, _at(6.0)), SKY]))

    assert math.isinf(out.outside_m[0])


def test_a_second_pane_of_the_same_glazing_is_one_layer():
    out = _outcomes(_rays([GLASS, (KIND_EXTERIOR_GLASS, 0, _at(3.05)), SKY]), skip_within_m=0.0)

    assert math.isinf(out.outside_m[0])


def test_the_interior_glass_limit():
    rays = _rays([(KIND_INTERIOR_GLASS, -1, _at(1.0)), GLASS, SKY])

    assert _outcomes(rays, max_interior_glass=1).reach.tolist() == [True]
    assert _outcomes(rays, max_interior_glass=0).reach.tolist() == [False]


def test_a_partition_modelled_as_two_faces_counts_once():
    rays = _rays([(KIND_INTERIOR_GLASS, -1, _at(1.0)), (KIND_INTERIOR_GLASS, -1, _at(1.02)), GLASS, SKY])

    assert _outcomes(rays, max_interior_glass=1).reach.tolist() == [True]


def test_two_partitions_count_twice():
    rays = _rays([(KIND_INTERIOR_GLASS, -1, _at(1.0)), (KIND_INTERIOR_GLASS, -1, _at(2.0)), GLASS, SKY])

    assert _outcomes(rays, max_interior_glass=1).reach.tolist() == [False]


def test_a_see_through_screen_inside_is_not_a_glass_layer():
    rays = _rays([(KIND_SEE_THROUGH, -1, _at(1.0)), GLASS, SKY])

    assert _outcomes(rays, max_interior_glass=0).reach.tolist() == [True]


def test_a_blocked_ray_does_not_reach_and_is_not_a_limit_ray():
    out = _outcomes(_rays([(KIND_BLOCKS, -1, _at(1.0))]))

    assert out.reach.tolist() == [False]
    assert out.limit_hit.tolist() == [False]
    assert math.isnan(out.plan_distance_m[0])


def test_a_ray_that_leaves_without_passing_glass_does_not_reach():
    assert _outcomes(_rays([SKY])).reach.tolist() == [False]


def test_a_downward_ray_to_open_ground_meets_the_ground_plane():
    direction = (1.0, 0.0, -0.5)
    out = _outcomes(_rays([GLASS, SKY], direction=direction), ground_z=0.0)
    sin_down = 0.5 / math.hypot(1.0, 0.5)

    assert out.outside_m[0] == pytest.approx(1.1 / sin_down)


def test_head_height_is_measured_from_the_storey_floor():
    out = _outcomes(_rays([GLASS, SKY]), level_z=3.0, pane_top=np.array([5.0]))

    assert out.head_height_m[0] == pytest.approx(2.0)


def test_a_ray_that_runs_out_of_segments_is_flagged():
    out = _outcomes(_rays([GLASS, (KIND_BLOCKS, -1, _at(3.5))]), skip_within_m=2.0)

    assert out.limit_hit.tolist() == [True]
    assert out.outside_m[0] == 0.0


def test_a_downward_ray_to_the_ground_is_measured_from_the_glass_not_the_eye():
    glass = (KIND_EXTERIOR_GLASS, 0, _at(3.0, z=1.5))
    direction = (1.0, 0.0, -0.5)
    out = _outcomes(_rays([glass, SKY], direction=direction), ground_z=0.0)
    sin_down = 0.5 / math.hypot(1.0, 0.5)

    assert out.outside_m[0] == pytest.approx(1.5 / sin_down)


# A near-grazing ray: 84 deg off the normal of planes x = c.
GRAZING = np.array([1.0, 10.0, 0.0]) / math.sqrt(101.0)
X_NORMAL = (1.0, 0.0, 0.0)


def _grazing(planes):
    """One grazing ray from EYE through planes (kind, x, normal); then sky."""
    n_seg = len(planes) + 1
    kinds = np.full((1, n_seg), KIND_UNUSED)
    surfaces = np.full((1, n_seg), -1)
    hits = np.full((1, n_seg, 3), np.nan)
    normals = np.full((1, n_seg, 3), np.nan)
    for j, (kind, x, normal) in enumerate(planes):
        t = (x - EYE[0]) / GRAZING[0]
        kinds[0, j] = kind
        surfaces[0, j] = 0 if kind == KIND_EXTERIOR_GLASS else -1
        hits[0, j] = np.asarray(EYE) + t * GRAZING
        normals[0, j] = normal
    kinds[0, -1] = KIND_MISS
    return np.array([EYE]), GRAZING[None, :], kinds, surfaces, hits, normals


def _grazing_outcomes(planes, *, normals=True, **kw):
    origins, directions, kinds, surfaces, hits, n = _grazing(planes)
    settings = dict(level_z=0.0, max_interior_glass=1, skip_within_m=2.0, ground_z=0.0)
    settings.update(kw)
    return ray_outcomes(origins, directions, kinds, surfaces, hits, PANE_TOP,
                        normals=n if normals else None, **settings)


def test_a_two_face_partition_counts_once_for_a_near_grazing_ray():
    # The faces are 0.02 m apart, about 0.2 m apart along this ray.
    planes = [(KIND_INTERIOR_GLASS, 1.0, X_NORMAL), (KIND_INTERIOR_GLASS, 1.02, X_NORMAL),
              (KIND_EXTERIOR_GLASS, 3.0, X_NORMAL)]

    assert _grazing_outcomes(planes).reach.tolist() == [True]
    # Without normals the along-ray gap is all there is: two layers.
    assert _grazing_outcomes(planes, normals=False).reach.tolist() == [False]


def test_two_parallel_partitions_apart_still_count_twice_for_a_grazing_ray():
    planes = [(KIND_INTERIOR_GLASS, 1.0, X_NORMAL), (KIND_INTERIOR_GLASS, 1.5, X_NORMAL),
              (KIND_EXTERIOR_GLASS, 3.0, X_NORMAL)]

    assert _grazing_outcomes(planes).reach.tolist() == [False]


def test_a_crossing_partition_the_ray_grazes_is_another_layer():
    # The second glass is horizontal and the ray runs level: no gap at all
    # along that glass's own normal, yet it is a different pane.
    planes = [(KIND_INTERIOR_GLASS, 1.0, X_NORMAL), (KIND_INTERIOR_GLASS, 1.4, (0.0, 0.0, 1.0)),
              (KIND_EXTERIOR_GLASS, 3.0, X_NORMAL)]

    assert _grazing_outcomes(planes).reach.tolist() == [False]


def test_the_second_face_of_the_exterior_pane_is_one_layer_for_a_grazing_ray():
    planes = [(KIND_EXTERIOR_GLASS, 3.0, X_NORMAL), (KIND_EXTERIOR_GLASS, 3.02, X_NORMAL)]

    out = _grazing_outcomes(planes, skip_within_m=0.0)

    assert math.isinf(out.outside_m[0])
