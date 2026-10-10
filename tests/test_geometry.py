# tests/test_geometry.py
import numpy as np
import pytest

from leed_daylight import geometry as g

UNIT_SQUARE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
L_SHAPE = np.array(
    [[0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]], dtype=float
)


def test_normal_of_a_counter_clockwise_square_points_up():
    assert np.allclose(g.normal(UNIT_SQUARE), [0, 0, 1])


def test_normal_flips_with_winding():
    assert np.allclose(g.normal(UNIT_SQUARE[::-1]), [0, 0, -1])


def test_normal_is_a_unit_vector():
    tilted = np.array([[0, 0, 0], [2, 0, 1], [2, 3, 1], [0, 3, 0]], dtype=float)
    assert np.linalg.norm(g.normal(tilted)) == pytest.approx(1.0)


def test_area_of_the_unit_square():
    assert g.area(UNIT_SQUARE) == pytest.approx(1.0)


def test_area_of_an_l_shape():
    assert g.area(L_SHAPE) == pytest.approx(3.0)


def test_area_is_independent_of_winding():
    assert g.area(UNIT_SQUARE[::-1]) == pytest.approx(1.0)


def test_centroid_of_the_unit_square():
    assert np.allclose(g.centroid(UNIT_SQUARE), [0.5, 0.5, 0.0])


def test_centroid_of_an_l_shape_is_not_the_vertex_mean():
    c = g.centroid(L_SHAPE)
    assert not np.allclose(c, L_SHAPE.mean(axis=0))
    assert np.allclose(c[:2], [5 / 6 * 1.0, 5 / 6 * 1.0], atol=1e-9)


def test_a_flat_polygon_has_no_planarity_error():
    assert g.planarity_error(UNIT_SQUARE) == pytest.approx(0.0, abs=1e-12)


def test_planarity_error_finds_a_lifted_vertex():
    # The normal follows the best-fit plane, so a 3 mm lift shows as 0.75 mm.
    bent = UNIT_SQUARE.copy()
    bent[2, 2] = 0.003
    # source: derived from the test's own vertex offsets
    assert g.planarity_error(bent) == pytest.approx(0.00075, abs=1e-6)


def test_plane_round_trip_preserves_points():
    tilted = np.array([[0, 0, 0], [2, 0, 1], [2, 3, 1], [0, 3, 0]], dtype=float)
    basis = g.plane_basis(tilted)
    assert np.allclose(g.from_plane(g.to_plane(tilted, basis), basis), tilted)


def test_point_in_polygon_inside_and_outside():
    square2 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert g.point_in_polygon_2d(np.array([0.5, 0.5]), square2)
    assert not g.point_in_polygon_2d(np.array([1.5, 0.5]), square2)


def test_point_in_polygon_handles_a_concave_notch():
    l2 = L_SHAPE[:, :2]
    assert g.point_in_polygon_2d(np.array([0.5, 0.5]), l2)
    assert not g.point_in_polygon_2d(np.array([1.5, 1.5]), l2)


def test_fewer_than_three_points_is_refused():
    with pytest.raises(ValueError, match="three vertices"):
        g.area(UNIT_SQUARE[:2])


def test_a_degenerate_polygon_is_refused():
    line = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    with pytest.raises(ValueError, match="degenerate"):
        g.normal(line)


# -- drop_redundant_vertices, moved from build/frames.py (merge-split spec 6.1) --


def test_drop_redundant_vertices_lives_in_geometry_and_frames_uses_it():
    from leed_daylight.build import frames

    # build/frames.py and build/prepare.py use the one routine in geometry.py;
    # tests/test_golden_prepare.py is the byte check that nothing changed.
    assert frames.drop_redundant_vertices is g.drop_redundant_vertices
    assert not hasattr(frames, "_COLLINEAR_TOL_M")
    assert g.COLLINEAR_TOL_M == 1e-6


def test_drop_redundant_vertices_drops_repeats_and_straight_edge_points_in_plan_and_in_3d():
    # A 4 x 2 rectangle with a repeated corner and a point in the middle of an edge.
    plan = np.array([[0, 0], [2, 0], [4, 0], [4, 2], [4, 2], [0, 2]], dtype=float)

    kept = g.drop_redundant_vertices(plan)

    assert kept.dtype == np.float64
    assert kept.tolist() == [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]]
    lifted = np.column_stack([plan, np.full(len(plan), 3.0)])
    assert g.drop_redundant_vertices(lifted).tolist() == [
        [0.0, 0.0, 3.0], [4.0, 0.0, 3.0], [4.0, 2.0, 3.0], [0.0, 2.0, 3.0],
    ]
