# tests/test_build_frames.py
"""Tests for leed_daylight.build.frames (T3.7): the constant-width inset.

Every numeric literal below is invented and round, per the confidentiality
rule (phase3-global.md #5): fictional openings with fictional shares, never
a value the plan lists as client-derived.
"""
from __future__ import annotations

import numpy as np
import pytest

from leed_daylight import geometry
from leed_daylight.build.frames import FrameError, build_frame

# -- fixtures -------------------------------------------------------------

SQUARE_2x2 = np.array(
    [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
)

SQUARE_2x2_VERTICAL = np.array(
    # the same 2x2 square, in the x-z plane rather than x-y (R-P7 settled
    # detail): the plane basis, not the world axes, must drive the maths.
    [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, 2.0), (0.0, 0.0, 2.0)]
)

RECT_3x1 = np.array(
    [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
)

# The L outline settled by the controller ruling: area 5.0.
L_SHAPE = np.array(
    [
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (3.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 3.0, 0.0),
        (0.0, 3.0, 0.0),
    ]
)

# The dumbbell outline settled by R-P7: two 1x1 squares joined by a
# 1.0 x 0.10 m neck, total area 2.10 m2. A share of 0.50 (target 1.05 m2)
# needs a multi-part inset once the neck closes at w = 0.05.
DUMBBELL = np.array(
    [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.45, 0.0),
        (2.0, 0.45, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (3.0, 1.0, 0.0),
        (2.0, 1.0, 0.0),
        (2.0, 0.55, 0.0),
        (1.0, 0.55, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
)

# Fix round 1 (review): a concave hexagon with two non-adjacent reflex
# vertices (indices 0 and 3) that sit only ~1.4 m apart while every other
# vertex is far out. At share=0.47 the old nearest-ORIGINAL-vertex
# correspondence in `_correspond` mapped shapely's solved inset vertex 3
# to original vertex 0 and vice versa (both stay a valid one-to-one
# match, so nothing refused it) -- a silent swap between non-adjacent
# vertices. All coordinates are round integers; area is 47.0 m2.
CONCAVE_HEXAGON = np.array(
    [
        (1.0, 1.0, 0.0),
        (-1.0, 8.0, 0.0),
        (-8.0, 3.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, -5.0, 0.0),
        (4.0, 0.0, 0.0),
    ]
)


def _dist_point_to_line(point, line_a, line_b) -> float:
    """Perpendicular distance from `point` to the infinite line a->b."""
    d = line_b - line_a
    d = d / np.linalg.norm(d)
    v = point - line_a
    proj = v - np.dot(v, d) * d
    return float(np.linalg.norm(proj))


def _assert_every_inset_vertex_is_width_from_its_edges(opening, frame) -> None:
    """Every inset vertex must sit exactly `width_m` from both its edges.

    This is the width-per-edge check from the brief's test table, factored
    out so it runs on more than one outline (fix round 1 review: "run the
    width-per-edge check on the L-shape, not only the square" -- a
    vertex-correspondence bug can hit an area check while still misplacing
    individual vertices, which only a per-vertex check like this catches).
    """
    n = len(opening)
    for i in range(n):
        prev_a, prev_b = opening[i - 1], opening[i]
        next_a, next_b = opening[i], opening[(i + 1) % n]
        d_prev = _dist_point_to_line(frame.glass[i], prev_a, prev_b)
        d_next = _dist_point_to_line(frame.glass[i], next_a, next_b)
        assert d_prev == pytest.approx(frame.width_m, abs=1e-9)
        assert d_next == pytest.approx(frame.width_m, abs=1e-9)


# -- constant_width ---------------------------------------------------


def test_a_square_hits_its_target_area():
    frame = build_frame(SQUARE_2x2, "opening_a", 0.34, "constant_width")
    assert frame.achieved_glass_area_m2 == pytest.approx(2.64, rel=0.005)
    assert frame.target_glass_area_m2 == pytest.approx(2.64, rel=1e-9)


def test_the_solved_width_is_the_same_on_every_edge():
    frame = build_frame(SQUARE_2x2, "opening_a", 0.34, "constant_width")
    _assert_every_inset_vertex_is_width_from_its_edges(SQUARE_2x2, frame)


def test_the_solved_width_is_the_same_on_every_edge_of_a_non_convex_opening():
    # Fix round 1 review: the width-per-edge check must not run on the
    # square alone.
    frame = build_frame(L_SHAPE, "opening_c", 0.34, "constant_width")
    _assert_every_inset_vertex_is_width_from_its_edges(L_SHAPE, frame)


def test_a_long_rectangle_hits_its_target_area():
    frame = build_frame(RECT_3x1, "opening_b", 0.34, "constant_width")
    assert frame.achieved_glass_area_m2 == pytest.approx(1.98, rel=0.005)


def test_uniform_scale_hits_the_target_exactly():
    frame = build_frame(RECT_3x1, "opening_b", 0.34, "uniform_scale")
    assert frame.achieved_glass_area_m2 == pytest.approx(1.98, rel=1e-12)
    assert frame.width_m is None


def test_constant_width_and_uniform_scale_differ_on_a_long_opening():
    a = build_frame(RECT_3x1, "opening_b", 0.34, "constant_width")
    b = build_frame(RECT_3x1, "opening_b", 0.34, "uniform_scale")
    assert np.abs(a.glass - b.glass).max() > 0.01


def test_a_non_convex_opening_hits_its_target_area():
    frame = build_frame(L_SHAPE, "opening_c", 0.34, "constant_width")
    assert frame.achieved_glass_area_m2 == pytest.approx(0.66 * 5.0, rel=0.005)


def test_a_share_that_closes_the_opening_is_refused():
    # R-P7: a square is always reachable; the dumbbell's neck forces a
    # multi-part inset before the target area is reached.
    with pytest.raises(FrameError, match="opening_d"):
        build_frame(DUMBBELL, "opening_d", 0.50, "constant_width")


def test_a_share_of_one_is_refused():
    with pytest.raises(FrameError, match="opening_e"):
        build_frame(SQUARE_2x2, "opening_e", 1.0, "constant_width")
    with pytest.raises(FrameError, match="opening_e"):
        build_frame(SQUARE_2x2, "opening_e", -0.10, "constant_width")


def test_a_concave_opening_does_not_swap_non_adjacent_vertices():
    # Fix round 1 (review): a nearest-ORIGINAL-vertex correspondence can
    # stay one-to-one while still mislabelling a concave outline, because
    # two non-adjacent original vertices (0 and 3 here) sit closer to each
    # other than either does to its own inset point. The fixed
    # correspondence must be established by construction (the analytic
    # miter point for each vertex), so every inset vertex still sits
    # exactly `width_m` from both of its own edges, not someone else's.
    frame = build_frame(CONCAVE_HEXAGON, "opening_k", 0.47, "constant_width")
    assert frame.achieved_glass_area_m2 == pytest.approx(0.53 * 47.0, rel=0.005)
    _assert_every_inset_vertex_is_width_from_its_edges(CONCAVE_HEXAGON, frame)


def test_a_malformed_opening_is_refused_by_name():
    # Fix round 1 (review): a ragged or non-numeric opening must raise
    # FrameError naming the opening, not numpy's bare ValueError.
    with pytest.raises(FrameError, match="opening_l"):
        build_frame([[1, 2, 3], [1, 2]], "opening_l", 0.34, "constant_width")
    with pytest.raises(FrameError, match="opening_l"):
        build_frame([["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]], "opening_l", 0.34, "constant_width")


def test_zero_share_returns_the_opening_unchanged():
    frame = build_frame(SQUARE_2x2, "opening_f", 0.0, "constant_width")
    assert np.allclose(frame.glass, SQUARE_2x2)
    assert frame.ring == ()
    assert frame.achieved_glass_area_m2 == pytest.approx(4.0, rel=1e-9)
    assert frame.target_glass_area_m2 == pytest.approx(4.0, rel=1e-9)
    assert frame.width_m == 0.0

    scaled = build_frame(SQUARE_2x2, "opening_f", 0.0, "uniform_scale")
    assert scaled.width_m is None


def test_the_ring_tiles_the_border():
    frame = build_frame(SQUARE_2x2, "opening_g", 0.34, "constant_width")
    assert len(frame.ring) == 4
    ring_area = sum(geometry.area(q) for q in frame.ring)
    assert ring_area == pytest.approx(4.0 - 2.64, rel=0.005)


def test_a_vertical_window_works_the_same_as_a_horizontal_one():
    flat = build_frame(SQUARE_2x2, "opening_h", 0.34, "constant_width")
    vertical = build_frame(SQUARE_2x2_VERTICAL, "opening_h", 0.34, "constant_width")
    assert vertical.achieved_glass_area_m2 == pytest.approx(
        flat.achieved_glass_area_m2, rel=1e-6
    )
    assert vertical.target_glass_area_m2 == pytest.approx(
        flat.target_glass_area_m2, rel=1e-9
    )


def test_the_frame_is_planar():
    cases = [
        build_frame(SQUARE_2x2, "opening_i", 0.34, "constant_width"),
        build_frame(RECT_3x1, "opening_i", 0.34, "uniform_scale"),
        build_frame(L_SHAPE, "opening_i", 0.34, "constant_width"),
        build_frame(SQUARE_2x2_VERTICAL, "opening_i", 0.34, "constant_width"),
    ]
    for frame in cases:
        assert geometry.planarity_error(frame.glass) < 1e-9
        for quad in frame.ring:
            assert geometry.planarity_error(quad) < 1e-9


def test_the_frame_faces_the_same_way_as_the_opening():
    frame = build_frame(SQUARE_2x2, "opening_j", 0.34, "constant_width")
    opening_normal = geometry.normal(SQUARE_2x2)
    for quad in frame.ring:
        quad_normal = geometry.normal(quad)
        assert np.dot(quad_normal, opening_normal) > 0.0


# -- collinear and duplicate vertices (M1) ------------------------------------

# A 2 x 1 m rectangle with an extra vertex at the middle of its bottom edge.
RECT_2x1_MIDPOINT = np.array(
    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
)

# A 4 x 3 m curtain-wall-like opening in the x-z plane: panel joints left
# several collinear vertices on every edge, plus one repeated vertex.
CURTAIN_WALL_4x3 = np.array(
    [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0), (4.0, 0.0, 1.0), (4.0, 0.0, 2.0), (4.0, 0.0, 3.0),
        (2.0, 0.0, 3.0), (2.0, 0.0, 3.0), (0.0, 0.0, 3.0), (0.0, 0.0, 1.5),
    ]
)


@pytest.mark.parametrize("style", ["constant_width", "uniform_scale"])
def test_an_extra_collinear_vertex_is_dropped_before_solving(style):
    frame = build_frame(RECT_2x1_MIDPOINT, "opening_m", 0.20, style)
    assert frame.achieved_glass_area_m2 == pytest.approx(1.6, rel=0.005)
    assert geometry.area(frame.glass) == pytest.approx(1.6, rel=0.005)
    assert len(frame.glass) == 4
    assert len(frame.ring) == 4


@pytest.mark.parametrize("style", ["constant_width", "uniform_scale"])
def test_a_curtain_wall_outline_with_collinear_points_is_framed(style):
    frame = build_frame(CURTAIN_WALL_4x3, "opening_cw", 0.10, style)
    assert geometry.area(frame.glass) == pytest.approx(10.8, rel=0.005)
    assert len(frame.glass) == 4
    ring_area = sum(geometry.area(q) for q in frame.ring)
    assert ring_area == pytest.approx(1.2, rel=0.005)
    if style == "constant_width":
        _assert_every_inset_vertex_is_width_from_its_edges(
            np.array([(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 0.0, 3.0), (0.0, 0.0, 3.0)]),
            frame,
        )


# -- uniform_scale on concave openings -------------------------------------


def _inside_or_on(outer_pts, inner_pts) -> bool:
    from shapely.geometry import Polygon

    basis = geometry.plane_basis(outer_pts)
    outer = Polygon(geometry.to_plane(outer_pts, basis)).buffer(1e-9)
    inner = Polygon(geometry.to_plane(inner_pts, basis))
    return outer.contains(inner)


def test_uniform_scale_frames_an_l_shaped_opening_inside_itself():
    # The L's area centroid sits outside its kernel, so scaling about it
    # pushes glass past the re-entrant corner; the scale centre must be a
    # point the whole outline can be seen from.
    frame = build_frame(L_SHAPE, "opening_l", 0.20, "uniform_scale")
    assert frame.achieved_glass_area_m2 == pytest.approx(4.0)
    assert geometry.area(frame.glass) == pytest.approx(4.0)
    assert _inside_or_on(L_SHAPE, frame.glass)
    assert sum(geometry.area(q) for q in frame.ring) == pytest.approx(1.0)


def test_uniform_scale_keeps_the_area_centroid_for_a_convex_opening():
    frame = build_frame(RECT_3x1, "opening_r", 0.36, "uniform_scale")
    centre = np.array([1.5, 0.5, 0.0])
    np.testing.assert_allclose(frame.glass, centre + 0.8 * (RECT_3x1 - centre), atol=1e-12)


# A U: a 1 m wide and a 0.2 m wide upright on a 0.2 m base. No single
# point sees the top of both uprights, so no scale centre keeps the glass
# inside the opening.
NOT_STAR_SHAPED = np.array(
    [
        (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 4.0, 0.0), (3.0, 4.0, 0.0),
        (3.0, 0.2, 0.0), (0.2, 0.2, 0.0), (0.2, 3.0, 0.0), (0.0, 3.0, 0.0),
    ]
)


def test_uniform_scale_refuses_an_opening_no_point_can_scale_inside():
    with pytest.raises(FrameError, match="opening_u.*constant_width"):
        build_frame(NOT_STAR_SHAPED, "opening_u", 0.20, "uniform_scale")
