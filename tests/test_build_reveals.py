# tests/test_build_reveals.py
"""Tests for leed_daylight.build.reveals (T3.8): the window reveal side walls.

Every numeric literal below is invented and round, per the confidentiality
rule (phase3-global.md #5): a fictional 1 x 1 m opening with a fictional
reveal depth, never a value the plan names as client-derived.
"""
from __future__ import annotations

import numpy as np
import pytest

from leed_daylight import geometry
from leed_daylight.build.blinds import BLIND_OFFSET_M, blind_polygons
from leed_daylight.build.glazing_class import PaneClass
from leed_daylight.build.groups import WindowGroup
from leed_daylight.build.reveals import (
    RevealError,
    reveal_walls,
    shared_edges,
    skylight_well_walls,
    well_shared_edges,
)
from leed_daylight.scene.primitives import Primitive

# -- fixtures -------------------------------------------------------------

# A 1 x 1 m opening in the x-y plane, wound so its outward normal is +z.
SQUARE_1x1 = np.array(
    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
)
SQUARE_1x1_NORMAL = np.array([0.0, 0.0, 1.0])

# The same opening, in the x-z plane, wound so its outward normal is -y
# (settled by the brief's controller rulings for this test).
OPENING_XZ = np.array(
    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)]
)
OPENING_XZ_NORMAL = np.array([0.0, -1.0, 0.0])

# The same opening with vertex 1 duplicated: a zero-length edge between
# vertex 1 and vertex 2 (fix round 1).
SQUARE_WITH_DUPLICATE_VERTEX = np.array(
    [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
)

REVEAL_DEPTH_M = 0.15


def test_a_square_opening_gets_eight_quads():
    quads = reveal_walls(SQUARE_1x1, SQUARE_1x1_NORMAL, REVEAL_DEPTH_M)
    assert len(quads) == 8
    for quad in quads:
        assert quad.shape == (4, 3)
        # Invariant: every quad is planar and has positive area (the
        # construction makes them rectangles, so this also guards against
        # a degenerate, zero-area result slipping through).
        assert geometry.area(quad) > 0.0
        assert geometry.planarity_error(quad) == pytest.approx(0.0, abs=1e-9)


def test_each_quad_has_the_reveal_depth():
    quads = reveal_walls(SQUARE_1x1, SQUARE_1x1_NORMAL, REVEAL_DEPTH_M)
    for quad in quads:
        # The opening lies at z=0; every quad has two vertices on the
        # opening (z=0) and two extruded along the normal by the depth,
        # so its extent along z (peak-to-peak) is exactly the depth.
        extent = float(quad[:, 2].max() - quad[:, 2].min())
        assert extent == pytest.approx(REVEAL_DEPTH_M)


def test_zero_depth_makes_no_reveal():
    quads = reveal_walls(SQUARE_1x1, SQUARE_1x1_NORMAL, 0.0)
    assert quads == ()


def test_the_quads_share_the_opening_s_edges():
    quads = reveal_walls(SQUARE_1x1, SQUARE_1x1_NORMAL, REVEAL_DEPTH_M)
    n = len(SQUARE_1x1)
    for i in range(n):
        p_i, p_j = SQUARE_1x1[i], SQUARE_1x1[(i + 1) % n]
        count = sum(
            1
            for quad in quads
            if np.allclose(quad[0], p_i) and np.allclose(quad[1], p_j)
        )
        assert count == 2, f"opening edge {i} appears in {count} quads, expected 2"


def test_the_extrusion_follows_the_opening_normal():
    quads = reveal_walls(OPENING_XZ, OPENING_XZ_NORMAL, REVEAL_DEPTH_M)
    for quad in quads:
        # quad[0]/quad[3] and quad[1]/quad[2] are the near/far pairs
        # (controller ruling's quad construction). The far point must
        # differ from the near point only along y (the opening's own
        # normal), never along x.
        for near, far in ((quad[0], quad[3]), (quad[1], quad[2])):
            offset = far - near
            assert offset[0] == pytest.approx(0.0, abs=1e-9)
            assert offset[2] == pytest.approx(0.0, abs=1e-9)
            assert abs(offset[1]) == pytest.approx(REVEAL_DEPTH_M)


def test_a_negative_depth_is_refused():
    with pytest.raises(ValueError, match="depth_m"):
        reveal_walls(SQUARE_1x1, SQUARE_1x1_NORMAL, -0.1)


# -- fix round 1: outward-normal and edge validation -----------------------


def test_a_zero_length_normal_is_refused():
    # Brief-mandated case (controller rulings), previously untested.
    with pytest.raises(RevealError, match="outward_normal"):
        reveal_walls(SQUARE_1x1, (0.0, 0.0, 0.0), REVEAL_DEPTH_M)


def test_a_normal_more_than_one_degree_off_is_refused():
    # Brief-mandated case (controller rulings), previously untested: 45
    # degrees off the opening plane's +z normal, well past the 1 degree
    # tolerance either sign allows.
    with pytest.raises(RevealError, match="outward_normal"):
        reveal_walls(SQUARE_1x1, (1.0, 0.0, 1.0), REVEAL_DEPTH_M)


@pytest.mark.parametrize(
    "bad_normal",
    [
        ["a", "b", "c"],
        [1.0, 2.0, [3.0]],
    ],
)
def test_a_non_numeric_normal_is_refused(bad_normal):
    # Review finding 1: an unguarded np.asarray let numpy's bare
    # ValueError (never naming outward_normal) leak straight out.
    with pytest.raises(RevealError, match="outward_normal"):
        reveal_walls(SQUARE_1x1, bad_normal, REVEAL_DEPTH_M)


@pytest.mark.parametrize("bad_component", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_normal_component_is_refused(bad_component):
    # Review finding 2: min(1.0, abs(nan)) == 1.0, so the angle check
    # passed and 8 NaN-filled quads came back silently.
    with pytest.raises(RevealError, match="outward_normal"):
        reveal_walls(SQUARE_1x1, [0.0, 0.0, bad_component], REVEAL_DEPTH_M)


def test_a_zero_length_opening_edge_is_refused():
    # Review finding 3: a duplicated consecutive vertex passes Newell's
    # check but yields a zero-area quad on that edge.
    with pytest.raises(RevealError, match="opening_pts"):
        reveal_walls(SQUARE_WITH_DUPLICATE_VERTEX, SQUARE_1x1_NORMAL, REVEAL_DEPTH_M)


# -- floor-to-ceiling glazing and curtain-wall joints -------------------------


def _pane(x0, x1, z0, z1):
    """A vertical pane in the x-z plane (y = 0), outward normal -y."""
    return np.array([(x0, 0.0, z0), (x1, 0.0, z0), (x1, 0.0, z1), (x0, 0.0, z1)])


MINUS_Y = np.array([0.0, -1.0, 0.0])


def _horizontal_quads_at(quads, z):
    return [q for q in quads if np.allclose(q[:, 2], z)]


def test_a_floor_to_ceiling_opening_gets_no_sill_or_head_reveal():
    quads = reveal_walls(_pane(0.0, 2.0, 0.0, 3.0), MINUS_Y, 0.2, floor_z=0.0, ceiling_z=3.0)
    assert len(quads) == 4  # the two jambs, inward and outward
    assert _horizontal_quads_at(quads, 0.0) == []
    assert _horizontal_quads_at(quads, 3.0) == []


def test_a_sill_within_a_centimetre_of_the_floor_is_skipped():
    quads = reveal_walls(_pane(0.0, 2.0, 0.005, 2.0), MINUS_Y, 0.2, floor_z=0.0, ceiling_z=3.0)
    assert len(quads) == 6
    assert _horizontal_quads_at(quads, 2.0) != []


def test_a_raised_sill_and_dropped_head_keep_their_reveals():
    quads = reveal_walls(_pane(0.0, 2.0, 0.9, 2.1), MINUS_Y, 0.2, floor_z=0.0, ceiling_z=3.0)
    assert len(quads) == 8


def test_without_floor_or_ceiling_levels_every_edge_gets_a_reveal():
    assert len(reveal_walls(_pane(0.0, 2.0, 0.0, 3.0), MINUS_Y, 0.2)) == 8


def test_skip_edges_drops_both_quads_of_that_edge():
    quads = reveal_walls(_pane(0.0, 1.0, 0.0, 3.0), MINUS_Y, 0.2, skip_edges={1})
    assert len(quads) == 6
    assert not [q for q in quads if np.allclose(q[:, 0], 1.0)]


def test_a_shared_edge_between_two_coplanar_panes_is_found():
    left, right = _pane(0.0, 1.0, 0.0, 3.0), _pane(1.0, 2.0, 0.0, 3.0)
    assert shared_edges([left, right]) == (frozenset({1}), frozenset({3}))


def test_a_t_joint_is_found_on_both_sides():
    bottom = _pane(0.0, 2.0, 0.0, 1.0)
    upper_left, upper_right = _pane(0.0, 1.0, 1.0, 3.0), _pane(1.0, 2.0, 1.0, 3.0)
    found = shared_edges([bottom, upper_left, upper_right])
    assert found[0] == frozenset({2})  # its top edge, covered by the two panes
    assert 0 in found[1] and 0 in found[2]  # their bottom edges
    assert 1 in found[1] and 3 in found[2]  # and the joint between them


def test_an_edge_only_partly_covered_is_kept():
    wide = _pane(0.0, 3.0, 0.0, 1.0)
    narrow = _pane(0.0, 1.0, 1.0, 2.0)
    found = shared_edges([wide, narrow])
    assert 2 not in found[0]
    assert 0 in found[1]


def test_panes_meeting_at_a_corner_are_not_coplanar():
    south = _pane(0.0, 1.0, 0.0, 3.0)
    east = np.array([(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 3.0), (1.0, 0.0, 3.0)])
    assert shared_edges([south, east]) == (frozenset(), frozenset())


def test_panes_a_wall_thickness_apart_are_not_coplanar():
    front = _pane(0.0, 1.0, 0.0, 3.0)
    back = _pane(1.0, 2.0, 0.0, 3.0) + np.array([0.0, 0.2, 0.0])
    assert shared_edges([front, back]) == (frozenset(), frozenset())


# -- roof glass: inner-half reveal and the skylight well (spec 5.6) ----------

WELL_DEPTH_M = 0.4

#: A flat 2 x 1 m skylight at z = 3.5, wound to face up.
SKYLIGHT_2x1 = np.array(
    [(4.0, 4.0, 3.5), (6.0, 4.0, 3.5), (6.0, 5.0, 3.5), (4.0, 5.0, 3.5)]
)

#: The same plan, tilted 15 degrees: it rises along +y from z = 3.5.
_RISE_15 = float(np.tan(np.radians(15.0)))
SKYLIGHT_15_DEG = np.array(
    [(4.0, 4.0, 3.5), (6.0, 4.0, 3.5), (6.0, 5.0, 3.5 + _RISE_15), (4.0, 5.0, 3.5 + _RISE_15)]
)


def _plan_normal_z(quad):
    return float(geometry.normal(quad)[2])


def test_an_inner_only_reveal_is_the_inward_half():
    full = reveal_walls(SQUARE_1x1, SQUARE_1x1_NORMAL, REVEAL_DEPTH_M)
    inner = reveal_walls(SQUARE_1x1, SQUARE_1x1_NORMAL, REVEAL_DEPTH_M, inner_only=True)
    assert len(inner) == 4
    for got, want in zip(inner, full[:4]):
        np.testing.assert_allclose(got, want)
    for quad in inner:  # every quad runs from the glass into the room, -z
        assert float(quad[:, 2].min()) == pytest.approx(-REVEAL_DEPTH_M)
        assert float(quad[:, 2].max()) == pytest.approx(0.0)


def test_an_inner_only_reveal_still_skips_joints_and_levels():
    quads = reveal_walls(
        _pane(0.0, 1.0, 0.0, 3.0), MINUS_Y, 0.2, floor_z=0.0, skip_edges={1}, inner_only=True
    )
    assert len(quads) == 2  # the head and the west jamb; sill on the floor, edge 1 skipped


def test_a_skylight_well_has_one_vertical_wall_per_edge():
    walls = skylight_well_walls(SKYLIGHT_2x1, WELL_DEPTH_M)
    assert len(walls) == 4
    for i, wall in enumerate(walls):
        assert wall.shape == (4, 3)
        np.testing.assert_allclose(wall[0], SKYLIGHT_2x1[i])
        np.testing.assert_allclose(wall[1], SKYLIGHT_2x1[(i + 1) % 4])
        assert _plan_normal_z(wall) == pytest.approx(0.0, abs=1e-12)
        assert geometry.area(wall) > 0.0
        np.testing.assert_allclose(wall[2:, 2], [3.1, 3.1])


def test_a_well_under_a_15_degree_pane_has_a_level_bottom():
    walls = skylight_well_walls(SKYLIGHT_15_DEG, WELL_DEPTH_M)
    assert len(walls) == 4
    for wall in walls:
        np.testing.assert_allclose(wall[2:, 2], [3.1, 3.1])  # z_low 3.5 less 0.4
        np.testing.assert_allclose(wall[[2, 3], :2], wall[[1, 0], :2])  # straight down
        assert _plan_normal_z(wall) == pytest.approx(0.0, abs=1e-12)
    heights = sorted(float(w[:, 2].max() - w[:, 2].min()) for w in walls)
    assert heights[0] == pytest.approx(WELL_DEPTH_M)
    assert heights[-1] == pytest.approx(WELL_DEPTH_M + _RISE_15)


def test_skip_edges_drops_that_well_wall():
    walls = skylight_well_walls(SKYLIGHT_2x1, WELL_DEPTH_M, skip_edges={1})
    assert len(walls) == 3
    assert not [w for w in walls if np.allclose(w[:, 0], 6.0)]


@pytest.mark.parametrize("bad_depth", [0.0, -0.4, float("nan"), "deep"])
def test_a_well_depth_that_is_not_above_zero_is_refused(bad_depth):
    with pytest.raises(RevealError, match="depth_m"):
        skylight_well_walls(SKYLIGHT_2x1, bad_depth)


def test_a_well_on_a_duplicated_vertex_is_refused():
    with pytest.raises(RevealError, match="opening_pts"):
        skylight_well_walls(SQUARE_WITH_DUPLICATE_VERTEX, WELL_DEPTH_M)


def test_the_ridge_of_a_two_pane_skylight_gets_no_well_wall():
    # Two 2 x 1 m panes pitched 15 degrees, meeting at a ridge along y = 5.
    south = SKYLIGHT_15_DEG
    north = np.array(
        [(4.0, 5.0, 3.5 + _RISE_15), (6.0, 5.0, 3.5 + _RISE_15), (6.0, 6.0, 3.5), (4.0, 6.0, 3.5)]
    )
    joints = well_shared_edges([south, north])
    assert joints == [frozenset({2}), frozenset({0})]
    assert shared_edges([south, north]) == (frozenset(), frozenset())  # not coplanar
    walls = skylight_well_walls(south, WELL_DEPTH_M, skip_edges=joints[0])
    assert len(walls) == 3
    assert not [w for w in walls if np.allclose(w[:2, 1], 5.0)]


def test_the_hips_of_a_pyramid_skylight_get_no_well_wall():
    apex = (5.0, 5.0, 3.5 + 0.5)
    corners = [(4.0, 4.0, 3.5), (6.0, 4.0, 3.5), (6.0, 6.0, 3.5), (4.0, 6.0, 3.5)]
    faces = [np.array([corners[k], corners[(k + 1) % 4], apex]) for k in range(4)]
    joints = well_shared_edges(faces)
    assert joints == [frozenset({1, 2})] * 4
    for face, skip in zip(faces, joints):
        (wall,) = skylight_well_walls(face, WELL_DEPTH_M, skip_edges=skip)
        np.testing.assert_allclose(wall[2:, 2], [3.1, 3.1])


def test_separate_skylights_share_no_well_edge():
    other = SKYLIGHT_2x1 + np.array([4.0, 0.0, 0.0])
    assert well_shared_edges([SKYLIGHT_2x1, other]) == [frozenset(), frozenset()]


def test_an_l_shaped_skylight_gets_one_well_wall_per_edge():
    # A flat L: a 2 x 2 m square at z = 3.5 less its 1 x 1 m north-east corner.
    l_outline = np.array(
        [(4.0, 4.0, 3.5), (6.0, 4.0, 3.5), (6.0, 5.0, 3.5),
         (5.0, 5.0, 3.5), (5.0, 6.0, 3.5), (4.0, 6.0, 3.5)]
    )
    walls = skylight_well_walls(l_outline, WELL_DEPTH_M)
    assert len(walls) == 6  # the two re-entrant edges included
    for i, wall in enumerate(walls):
        np.testing.assert_allclose(wall[:2], [l_outline[i], l_outline[(i + 1) % 6]])
        np.testing.assert_allclose(wall[2:, 2], [3.1, 3.1])
        assert _plan_normal_z(wall) == pytest.approx(0.0, abs=1e-12)

    # Its blind is the same outline moved straight down by the blind offset.
    flat = tuple(v for pt in l_outline for v in pt)
    pane = Primitive("glass", "polygon", "l_sky", (), (), flat, "test")
    group = WindowGroup(
        group_id="hall_sky", room="hall", storey=0, azimuth_bin_deg=None,
        openings=(pane,), outward_normals=(np.array([0.0, 0.0, 1.0]),), area_m2=3.0,
        kind="skylight", tilt_deg=0.0,
        pane_classes=(PaneClass(0.0, "skylight", "roof", False, "skylight"),),
    )
    (blind,) = blind_polygons(group)
    np.testing.assert_allclose(blind, l_outline - np.array([0.0, 0.0, BLIND_OFFSET_M]))
