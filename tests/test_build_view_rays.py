"""The panorama of view directions (QV spec 6.3)."""
from __future__ import annotations

import numpy as np
import pytest

from leed_daylight.build.view_rays import ViewRayError, panorama, rays_for_points


def test_the_default_panorama():
    directions, weights = panorama(2.0, 60.0)

    assert directions.shape == (180 * 31, 3)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert weights.sum() == pytest.approx(1.0)
    elevations = np.degrees(np.arcsin(directions[:, 2]))
    assert elevations.min() == pytest.approx(-30.0)
    assert elevations.max() == pytest.approx(30.0)
    assert weights.argmax() in np.flatnonzero(np.isclose(elevations, 0.0))


def test_weights_are_symmetric_about_the_horizon():
    directions, weights = panorama(5.0, 60.0)
    elevations = np.round(np.degrees(np.arcsin(directions[:, 2])), 6)

    for e in (5.0, 15.0, 30.0):
        assert weights[elevations == e].sum() == pytest.approx(weights[elevations == -e].sum())


def test_a_step_that_does_not_divide_360_is_refused():
    with pytest.raises(ViewRayError, match=r"0\.7.*divide 360"):
        panorama(0.7, 60.0)


def test_a_step_that_does_not_divide_the_half_height_is_refused():
    # 4 deg divides 360 but not 30: the band would stop at +/-28 deg.
    with pytest.raises(ViewRayError, match=r"4\.0 deg must divide.*30"):
        panorama(4.0, 60.0)


@pytest.mark.parametrize("step", [0.5, 1.0, 2.0, 3.0, 5.0])
def test_a_step_dividing_both_reaches_the_band_edges_exactly(step):
    directions, _ = panorama(step, 60.0)

    elevations = np.degrees(np.arcsin(directions[:, 2]))
    assert elevations.max() == pytest.approx(30.0)
    assert elevations.min() == pytest.approx(-30.0)


@pytest.mark.parametrize("step, vertical, match", [
    (0.0, 60.0, "step_deg"),
    (2.0, 0.0, "vertical_deg"),
    (2.0, 200.0, "vertical_deg"),
])
def test_bad_panorama_arguments_are_refused(step, vertical, match):
    with pytest.raises(ViewRayError, match=match):
        panorama(step, vertical)


def test_rays_for_points_groups_rays_by_point():
    directions, _ = panorama(5.0, 60.0)
    points = np.array([[1.0, 2.0, 1.1], [3.0, 4.0, 1.1]])

    origins, dirs = rays_for_points(points, directions)

    m = directions.shape[0]
    assert origins.shape == dirs.shape == (2 * m, 3)
    assert np.all(origins[:m] == points[0])
    assert np.all(origins[m:] == points[1])
    assert np.array_equal(dirs[m:], directions)
