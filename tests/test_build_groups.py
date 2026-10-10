# tests/test_build_groups.py
"""Tests for leed_daylight.build.groups (T3.9): window groups.

Every fixture is a fictional building with round numbers (phase3-global.md
#5): a 4 x 3 m room called "room_a", 1 m and 0.1 m openings, a 0.6 m
matching tolerance -- never a value read off a real project.

Most openings here are synthetic vertical rectangles built directly from
their intended outward direction (``tilted_opening``) or plane normal
(``plane_opening``), the same style ``tests/test_build_reveals.py`` uses:
what matters for this module is the maths (matching, orientation, binning),
not that the polygon looks like a real wall.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest
from shapely.geometry import Polygon

from leed_daylight.build.glazing_class import PaneClass
from leed_daylight.build.groups import (
    AZIMUTH_BIN_DEG,
    MIN_GROUP_AREA_M2,
    ROOM_MATCH_TOLERANCE_M,
    GroupError,
    WindowGroup,
    facade_azimuth_deg,
    window_groups,
)
from leed_daylight.scene.primitives import Primitive
from leed_daylight.scene.storeys import Room

from golden_prepare import FIXTURE_NAMES, GOLDEN_DIR, prepare_golden

ROLES = {
    "glass": "glazing",
    "cw_glass": "glazing_curtain_wall",
    "int_glass": "glazing_interior",
}

ROOM_A_OUTLINE = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]  # 4 x 3 m, area 12 m2


def make_room(storey, level_z, outline, room_id="S0_R0", ceiling_height_m=None,
              floor_names=()):
    return Room(
        room_id=room_id,
        storey=storey,
        level_z=level_z,
        outline=np.array(outline, dtype=np.float64),
        area_m2=float(Polygon(outline).area),
        ceiling_height_m=ceiling_height_m,
        floor_names=tuple(floor_names),
    )


def glazing(name, points, modifier="glass"):
    flat = tuple(v for pt in points for v in pt)
    return Primitive(modifier, "polygon", name, (), (), flat, "test")


def tilted_opening(name, theta_deg, cx, cy, z0, z1, half_width=0.5, modifier="glass"):
    """A vertical rectangle whose outward normal, once oriented away from
    the room the caller places it near, is ``(sin(theta), cos(theta), 0)``
    -- i.e. ``facade_azimuth_deg`` of the result is ``theta_deg`` (mod 360).

    ``(cx, cy)`` is placed ON the room boundary itself (never nudged off
    it): the orientation step must tell inward from outward by actually
    crossing the room's own outline, not by being handed a pre-nudged
    position that already encodes the answer.
    """
    theta = math.radians(theta_deg)
    d = np.array([math.sin(theta), math.cos(theta)])
    u = np.array([math.cos(theta), -math.sin(theta)])
    centre_xy = np.array([cx, cy])
    p0 = centre_xy - half_width * u
    p1 = centre_xy + half_width * u
    pts = [
        (p0[0], p0[1], z0),
        (p1[0], p1[1], z0),
        (p1[0], p1[1], z1),
        (p0[0], p0[1], z1),
    ]
    return glazing(name, pts, modifier)


def plane_opening(name, normal_dir, center, half_u=0.5, half_v=0.5, modifier="glass"):
    """A rectangle whose plane has (approximately) the given normal direction."""
    n = np.array(normal_dir, dtype=np.float64)
    n = n / np.linalg.norm(n)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(seed, n)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    c = np.array(center, dtype=np.float64)
    corners = [c - half_u * u - half_v * v, c + half_u * u - half_v * v,
               c + half_u * u + half_v * v, c - half_u * u + half_v * v]
    return glazing(name, [tuple(p) for p in corners], modifier)


ROOM_A = {"room_a": make_room(0, 0.0, ROOM_A_OUTLINE)}


# -- the brief's nine cases -------------------------------------------------


def test_interior_glazing_forms_no_group():
    prims = [
        tilted_opening("ext", 0.0, 2.0, 3.0, 1.0, 2.0),
        tilted_opening("int", 0.0, 2.2, 3.0, 1.0, 2.0, modifier="int_glass"),
    ]
    groups, warnings = window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))
    assert len(groups) == 1
    assert warnings == ()
    assert [o.name for o in groups[0].openings] == ["ext"]


def test_two_windows_on_the_same_facade_of_one_room_form_one_group():
    prims = [
        tilted_opening("w1", 0.0, 1.5, 3.0, 1.0, 2.0, half_width=0.5),  # 1 m2
        tilted_opening("w2", 0.0, 2.5, 3.0, 1.0, 2.0, half_width=0.5),  # 1 m2
    ]
    groups, warnings = window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))
    assert len(groups) == 1
    assert groups[0].area_m2 == pytest.approx(2.0)
    assert {o.name for o in groups[0].openings} == {"w1", "w2"}
    assert warnings == ()


def test_two_facades_of_one_room_form_two_groups():
    prims = [
        tilted_opening("north", 0.0, 2.0, 3.0, 1.0, 2.0),
        tilted_opening("east", 90.0, 4.0, 1.5, 1.0, 2.0),
    ]
    groups, _ = window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))
    assert len(groups) == 2
    assert {g.azimuth_bin_deg for g in groups} == {0.0, 90.0}


def test_normals_inside_one_bin_group_together():
    prims = [
        tilted_opening("a", 175.0, 2.0, 0.0, 1.0, 2.0),
        tilted_opening("b", 185.0, 2.0, 0.0, 1.0, 2.0),
    ]
    groups, _ = window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))
    assert len(groups) == 1
    assert groups[0].azimuth_bin_deg == pytest.approx(180.0)


def test_normals_across_a_bin_boundary_do_not_group():
    prims = [
        tilted_opening("a", 170.0, 2.0, 0.0, 1.0, 2.0),
        tilted_opening("b", 200.0, 2.0, 0.0, 1.0, 2.0),
    ]
    groups, _ = window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))
    assert len(groups) == 2
    assert {g.azimuth_bin_deg for g in groups} == {180.0, 202.5}


def test_a_tiny_group_merges_into_its_neighbour():
    prims = [
        tilted_opening("big", 0.0, 1.5, 3.0, 1.0, 2.0, half_width=1.0),   # 2.0 m2
        tilted_opening("small", 0.0, 2.5, 3.0, 1.0, 2.0, half_width=0.05),  # 0.1 m2
    ]
    groups, warnings = window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))
    assert len(groups) == 1
    assert groups[0].area_m2 == pytest.approx(2.1)
    assert warnings == ()


def test_unmatched_glazing_is_refused_by_name():
    prims = [tilted_opening("stray", 0.0, 2.0, 5.0, 1.0, 2.0)]  # 2 m past the y=3 edge
    with pytest.raises(GroupError, match="stray"):
        window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))


def test_glazing_on_the_wrong_storey_is_not_matched():
    rooms = {
        "room_a": make_room(0, 0.0, ROOM_A_OUTLINE, room_id="S0_R0"),
        "room_b": make_room(1, 3.0, ROOM_A_OUTLINE, room_id="S1_R0"),
    }
    prims = [tilted_opening("upstairs", 0.0, 2.0, 3.0, 3.2, 3.8)]
    groups, _ = window_groups(prims, ROLES, rooms, all_rooms=list(rooms.values()))
    assert len(groups) == 1
    assert groups[0].room == "room_b"
    assert groups[0].storey == 1


def test_group_ids_are_stable_when_a_room_is_added():
    prims = [tilted_opening("w", 0.0, 2.0, 3.0, 1.0, 2.0)]
    groups_before, _ = window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))
    group_id_before = groups_before[0].group_id

    rooms_after = dict(ROOM_A)
    # A room with a room_id that would have renumbered "room_a" under the
    # old room_id-keyed scheme; irrelevant now since group_id is name-based.
    rooms_after["room_new"] = make_room(
        0, 0.0, [(20.0, 20.0), (24.0, 20.0), (24.0, 23.0), (20.0, 23.0)], room_id="S0_R0"
    )
    groups_after, _ = window_groups(prims, ROLES, rooms_after, all_rooms=list(rooms_after.values()))
    assert len(groups_after) == 1
    assert groups_after[0].group_id == group_id_before


# -- controller-mandated additions ------------------------------------------


def test_a_skylight_is_grouped_as_its_room_s_skylight_group():
    prims = [plane_opening("roof_light", (0.0, 0.0, 1.0), (2.0, 1.5, 5.0))]
    groups, warnings = window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))
    assert warnings == ()
    assert [g.group_id for g in groups] == ["room_a_sky"]
    assert groups[0].kind == "skylight"
    assert groups[0].azimuth_bin_deg is None
    assert groups[0].tilt_deg == 0.0
    np.testing.assert_allclose(groups[0].outward_normals[0], [0.0, 0.0, 1.0], atol=1e-12)


def test_a_lone_tiny_group_is_kept_and_warned():
    prims = [tilted_opening("tiny", 0.0, 2.0, 3.0, 1.0, 2.0, half_width=0.025)]  # 0.05 m2
    groups, warnings = window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))
    assert len(groups) == 1
    assert groups[0].area_m2 == pytest.approx(0.05)
    assert len(warnings) == 1
    assert groups[0].group_id in warnings[0]


def test_room_ceiling_height_m_is_not_used_for_the_storey_band():
    rooms = {"room_a": make_room(0, 0.0, ROOM_A_OUTLINE, ceiling_height_m=0.1)}
    prims = [tilted_opening("high", 0.0, 2.0, 3.0, 4.5, 5.0)]
    groups, _ = window_groups(prims, ROLES, rooms, all_rooms=list(rooms.values()))
    assert len(groups) == 1
    assert groups[0].room == "room_a"


# -- I1: storey bands come from every detected room, not just named ones --


def test_glazing_on_an_unnamed_storey_is_not_folded_into_a_lower_storey_room():
    # Reproduction: storey 0 has a named room; storey 1 exists in the scene
    # (detect_rooms found it) but nobody named it in spaces.csv. Before the
    # fix, storey bands were built only from the NAMED rooms, so storey 0's
    # band came out open-ended above (nothing bounded it) and a window on
    # the unnamed storey silently fell into storey 0's room instead of
    # being refused as unmatched.
    named = {"room_a": make_room(0, 0.0, ROOM_A_OUTLINE, room_id="S0_R0")}
    all_rooms = [named["room_a"], make_room(1, 3.0, ROOM_A_OUTLINE, room_id="S1_R0")]
    prims = [tilted_opening("upstairs", 0.0, 2.0, 3.0, 3.2, 3.8)]
    # The window rule finds the unnamed storey-1 room: unmatched, with no
    # roof search, so it is master's refusal, not a roof-glass one.
    with pytest.raises(GroupError, match="unmatched exterior glazing: upstairs") as excinfo:
        window_groups(prims, ROLES, named, all_rooms=all_rooms)
    assert "roof glass" not in str(excinfo.value)
    assert str(excinfo.value).endswith("must be named in spaces.csv")


# -- I2: outward orientation on a concave (L-shaped) room -------------------

#: An L-shaped room: a 4 x 4 square with its top-right 2 x 2 corner removed
#: (the same notch shape tests/test_build_grid.py uses), so the room's own
#: 2-D centroid sits well away from the re-entrant wall's midpoint.
L_ROOM_OUTLINE = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 2.0), (2.0, 4.0), (0.0, 4.0)]
L_ROOM = {"room_l": make_room(0, 0.0, L_ROOM_OUTLINE)}


def test_a_re_entrant_wall_is_oriented_correctly_on_an_l_shaped_room():
    # The wall from (2, 2) to (4, 2) borders the notch: for x in [2, 4],
    # the room only exists for y < 2, so this wall's true outward normal is
    # +Y (into the notch), even though the room's overall centroid sits on
    # the window's +X, +Y side. A centroid-vector test flips this and bins
    # it at 180 instead of 0, putting the blind at y = 2.05, outside the
    # room.
    window_pts = [
        (2.5, 2.0, 1.0), (3.5, 2.0, 1.0), (3.5, 2.0, 2.0), (2.5, 2.0, 2.0),
    ]
    prims = [glazing("notch_window", window_pts)]
    groups, _ = window_groups(prims, ROLES, L_ROOM, all_rooms=list(L_ROOM.values()))
    assert len(groups) == 1
    assert groups[0].azimuth_bin_deg == pytest.approx(0.0)
    assert groups[0].outward_normals[0][1] > 0.0  # +Y, into the notch


def test_an_opening_with_no_boundary_to_cross_is_monitor_glass_with_no_floor():
    # An opening sitting well inside a large, convex room: stepping a short
    # distance either way along its own normal lands inside the room both
    # times, 5 m from the outline each way. That is roof glass now (a
    # monitor, spec 5.2); with no floor polygon in the scene, neither side
    # looks down onto the room's floor, so it is still refused, by name.
    room = {"room_big": make_room(0, 0.0, [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])}
    pts = [(4.5, 5.0, 1.0), (5.5, 5.0, 1.0), (5.5, 5.0, 2.0), (4.5, 5.0, 2.0)]
    prims = [glazing("stray_interior", pts)]
    with pytest.raises(
        GroupError, match="stray_interior: monitor glass reaches its room's floor on neither side"
    ):
        window_groups(prims, ROLES, room, all_rooms=list(room.values()))


#: A 6 x 5 m room whose outline is drawn to the EXTERIOR wall face, so a
#: mid-wall glass line sits a little inside the outline.
SETBACK_ROOM = {"room_s": make_room(0, 0.0, [(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (0.0, 5.0)])}


@pytest.mark.parametrize("y", [0.0, 0.15, 0.30])
@pytest.mark.parametrize("flip", [False, True])
def test_glass_set_back_inside_the_outline_is_oriented_outward(y, flip):
    pts = [(2.5, y, 1.0), (3.5, y, 1.0), (3.5, y, 2.0), (2.5, y, 2.0)]
    if flip:
        pts = pts[::-1]
    prims = [glazing("south_window", pts)]
    groups, _ = window_groups(
        prims, ROLES, SETBACK_ROOM, all_rooms=list(SETBACK_ROOM.values())
    )
    assert len(groups) == 1
    assert groups[0].azimuth_bin_deg == pytest.approx(180.0)
    assert groups[0].outward_normals[0][1] < 0.0


def test_glass_set_back_on_the_far_wall_is_oriented_outward():
    pts = [(2.5, 4.7, 1.0), (3.5, 4.7, 1.0), (3.5, 4.7, 2.0), (2.5, 4.7, 2.0)]
    prims = [glazing("north_window", pts)]
    groups, _ = window_groups(
        prims, ROLES, SETBACK_ROOM, all_rooms=list(SETBACK_ROOM.values())
    )
    assert groups[0].azimuth_bin_deg == pytest.approx(0.0)


def test_glass_set_back_deeper_than_a_wall_is_refused():
    # 1.5 m inside the outline is not a mid-wall glass line: it is roof
    # glass (a monitor, spec 5.2), and with no floor polygon to look down
    # onto it is refused by the monitor rule.
    pts = [(2.5, 1.5, 1.0), (3.5, 1.5, 1.0), (3.5, 1.5, 2.0), (2.5, 1.5, 2.0)]
    prims = [glazing("deep_glass", pts)]
    with pytest.raises(
        GroupError, match="deep_glass: monitor glass reaches its room's floor on neither side"
    ):
        window_groups(prims, ROLES, SETBACK_ROOM, all_rooms=list(SETBACK_ROOM.values()))


# -- facade_azimuth_deg -------------------------------------------------


def test_facade_azimuth_deg_matches_known_directions():
    assert facade_azimuth_deg((1.0, 0.0, 0.0)) == pytest.approx(90.0)
    assert facade_azimuth_deg((0.0, -1.0, 0.0)) == pytest.approx(180.0)
    assert facade_azimuth_deg((0.0, 1.0, 0.0)) == pytest.approx(0.0)


def test_facade_azimuth_deg_rejects_a_vertical_normal():
    with pytest.raises(GroupError, match="vertical"):
        facade_azimuth_deg((0.0, 0.0, 1.0))


def test_facade_azimuth_deg_rejects_wrong_shape():
    with pytest.raises(GroupError, match="3-vector"):
        facade_azimuth_deg((1.0, 2.0))


def test_facade_azimuth_deg_rejects_non_finite_components():
    with pytest.raises(GroupError, match="finite"):
        facade_azimuth_deg((float("nan"), 1.0, 0.0))
    with pytest.raises(GroupError, match="finite"):
        facade_azimuth_deg((float("inf"), 1.0, 0.0))


def test_facade_azimuth_deg_rejects_a_ragged_input():
    with pytest.raises(GroupError):
        facade_azimuth_deg([[1.0, 2.0], [3.0]])


def test_facade_azimuth_deg_rejects_non_numeric_input():
    with pytest.raises(GroupError):
        facade_azimuth_deg(("a", "b", "c"))


# -- hostile-input pass on window_groups ------------------------------------


def test_a_non_polygon_glazing_surface_is_refused_by_name():
    prims = [Primitive("glass", "sphere", "odd_shape", (), (), (0.0, 0.0, 0.0, 1.0), "test")]
    with pytest.raises(GroupError, match="odd_shape"):
        window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))


def test_a_degenerate_glazing_polygon_is_refused():
    collinear = (0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0)
    prims = [Primitive("glass", "polygon", "flat_sliver", (), (), collinear, "test")]
    with pytest.raises(GroupError, match="flat_sliver"):
        window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))


def test_a_glazing_polygon_with_nonfinite_coordinates_is_refused():
    bad = (0.0, 0.0, 1.0, 1.0, 0.0, 1.0, float("nan"), 0.0, 2.0)
    prims = [Primitive("glass", "polygon", "nan_opening", (), (), bad, "test")]
    with pytest.raises(GroupError, match="nan_opening"):
        window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))


def test_a_glazing_polygon_with_too_few_vertices_is_refused():
    two_points = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    prims = [Primitive("glass", "polygon", "line_not_polygon", (), (), two_points, "test")]
    with pytest.raises(GroupError, match="line_not_polygon"):
        window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))


def test_a_glazing_primitive_with_a_bad_fargs_count_is_refused():
    seven_reals = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0)
    prims = [Primitive("glass", "polygon", "malformed", (), (), seven_reals, "test")]
    with pytest.raises(GroupError, match="malformed"):
        window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))


def test_two_rooms_with_colliding_slugs_are_refused():
    rooms = {
        "Room A": make_room(0, 0.0, ROOM_A_OUTLINE, room_id="S0_R0"),
        "Room#A": make_room(
            0, 0.0, [(10.0, 0.0), (14.0, 0.0), (14.0, 3.0), (10.0, 3.0)], room_id="S0_R1"
        ),
    }
    prims = [
        tilted_opening("w1", 0.0, 2.0, 3.0, 1.0, 2.0),
        tilted_opening("w2", 0.0, 12.0, 3.0, 1.0, 2.0),
    ]
    with pytest.raises(GroupError, match=re.escape("Room A")) as excinfo:
        window_groups(prims, ROLES, rooms, all_rooms=list(rooms.values()))
    assert "Room#A" in str(excinfo.value)


def test_a_room_match_tie_is_refused_naming_both_rooms():
    rooms = {
        "left": make_room(0, 0.0, [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)], room_id="S0_R0"),
        "right": make_room(0, 0.0, [(3.0, 0.0), (5.0, 0.0), (5.0, 2.0), (3.0, 2.0)], room_id="S0_R1"),
    }
    # Centred exactly midway (0.5 m from each edge, both within the 0.6 m tolerance).
    midpoint = [(2.5, 0.9, 1.0), (2.5, 1.1, 1.0), (2.5, 1.1, 2.0), (2.5, 0.9, 2.0)]
    prims = [glazing("between", midpoint)]
    with pytest.raises(GroupError, match="left") as excinfo:
        window_groups(prims, ROLES, rooms, all_rooms=list(rooms.values()))
    assert "right" in str(excinfo.value)
    assert "between" in str(excinfo.value)


def test_an_opening_exactly_at_the_room_match_tolerance_matches():
    # ROOM_A's east edge is at x=4.0; y=1.0 is well inside its [0, 3] span,
    # so the nearest boundary point is (4.0, 1.0) and the distance is exact.
    x = 4.0 + ROOM_MATCH_TOLERANCE_M
    edge_opening = [(x, 0.9, 1.0), (x, 1.1, 1.0), (x, 1.1, 2.0), (x, 0.9, 2.0)]
    prims = [glazing("edge", edge_opening)]
    groups, _ = window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))
    assert len(groups) == 1


def test_a_group_area_exactly_at_the_minimum_is_not_warned():
    prims = [tilted_opening("just_enough", 0.0, 2.0, 3.0, 1.0, 2.0, half_width=MIN_GROUP_AREA_M2 / 2.0)]
    groups, warnings = window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))
    assert len(groups) == 1
    assert groups[0].area_m2 == pytest.approx(MIN_GROUP_AREA_M2)
    assert warnings == ()


def test_empty_scene_returns_no_groups_and_no_warnings():
    assert window_groups([], ROLES, {}, all_rooms=()) == ([], ())


def test_no_glazing_in_the_scene_returns_no_groups():
    prims = [glazing("wall", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0)], modifier="opaque_wall")]
    groups, warnings = window_groups(
        prims, {"opaque_wall": "wall"}, ROOM_A, all_rooms=list(ROOM_A.values())
    )
    assert groups == []
    assert warnings == ()


# -- P28: glazing must be matched against every detected room, not just ---
# -- named ones -------------------------------------------------------------

#: Two rooms, same storey and level, 0.3 m apart: A is named, B is not.
_P28_ROOM_A_OUTLINE = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
_P28_ROOM_B_OUTLINE = [(4.3, 0.0), (8.0, 0.0), (8.0, 3.0), (4.3, 3.0)]


def _p28_rooms():
    room_a = make_room(0, 0.0, _P28_ROOM_A_OUTLINE, room_id="S0_R0")
    room_b = make_room(0, 0.0, _P28_ROOM_B_OUTLINE, room_id="S0_R1")
    named = {"office_a": room_a}
    all_rooms = [room_a, room_b]
    return named, all_rooms


def test_glazing_on_an_unnamed_room_near_a_named_room_is_refused():
    named, all_rooms = _p28_rooms()
    # Centroid at x=4.45: distance 0.45 m to room A's east-of-y3 edge
    # (nearest point (4.0, 3.0)), but distance 0.0 m to room B's own
    # boundary (4.45 lies ON B's y=3 edge, which spans x in [4.3, 8]). B is
    # nearer and un-named, so this must be refused, not folded into A.
    window_pts = [
        (4.35, 3.0, 1.0), (4.55, 3.0, 1.0), (4.55, 3.0, 2.0), (4.35, 3.0, 2.0),
    ]
    # The window rule's nearest room is un-named: the pane is unmatched
    # (master's refusal), with no roof search.
    prims = [glazing("unnamed_neighbour", window_pts)]
    with pytest.raises(GroupError, match="unmatched exterior glazing: unnamed_neighbour") as excinfo:
        window_groups(prims, ROLES, named, all_rooms=all_rooms)
    assert "roof glass" not in str(excinfo.value)
    assert "spaces.csv" in str(excinfo.value)


def test_glazing_nearer_the_named_room_still_groups_there():
    named, all_rooms = _p28_rooms()
    window_pts = [
        (1.0, 3.0, 1.0), (2.0, 3.0, 1.0), (2.0, 3.0, 2.0), (1.0, 3.0, 2.0),
    ]
    prims = [glazing("on_a", window_pts)]
    groups, _ = window_groups(prims, ROLES, named, all_rooms=all_rooms)
    assert len(groups) == 1
    assert groups[0].room == "office_a"


def test_a_named_room_missing_from_all_rooms_is_refused():
    orphan = make_room(0, 0.0, _P28_ROOM_A_OUTLINE, room_id="S0_R0")
    named = {"office_a": orphan}
    with pytest.raises(GroupError, match="office_a"):
        window_groups([], ROLES, named, all_rooms=[])


def test_a_tie_between_a_named_and_an_unnamed_room_is_refused():
    named, all_rooms = _p28_rooms()
    # Centred at x=4.15, exactly midway in the 0.3 m gap: 0.15 m from each
    # room's nearest edge point, so this is an unresolved tie between a
    # named room and an un-named one.
    window_pts = [
        (4.05, 3.0, 1.0), (4.25, 3.0, 1.0), (4.25, 3.0, 2.0), (4.05, 3.0, 2.0),
    ]
    prims = [glazing("between_a_and_b", window_pts)]
    with pytest.raises(GroupError, match="between_a_and_b") as excinfo:
        window_groups(prims, ROLES, named, all_rooms=all_rooms)
    assert "office_a" in str(excinfo.value)


def test_window_group_is_a_frozen_dataclass_with_outward_normals():
    prims = [tilted_opening("w", 0.0, 2.0, 3.0, 1.0, 2.0)]
    groups, _ = window_groups(prims, ROLES, ROOM_A, all_rooms=list(ROOM_A.values()))
    group = groups[0]
    assert isinstance(group, WindowGroup)
    assert len(group.outward_normals) == len(group.openings) == 1
    # The window sits on room_a's north wall (y=3), so the oriented normal
    # must point away from the room's interior, toward +Y.
    assert group.outward_normals[0][1] > 0.0
    with pytest.raises(Exception):
        group.area_m2 = 999.0  # frozen: assignment must fail


def test_a_window_on_an_atrium_wall_faces_the_atrium():
    # A 8 x 8 m room round a 2 x 2 m atrium (3..5): a window on the atrium's
    # south wall (y = 3) looks north into the atrium, so its outward
    # normal points +y. Without the hole both sides of it read as inside.
    room = Room(
        room_id="S0_R0", storey=0, level_z=0.0,
        outline=np.array([(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)]),
        area_m2=60.0, ceiling_height_m=None,
        holes=(np.array([(3.0, 3.0), (5.0, 3.0), (5.0, 5.0), (3.0, 5.0)]),),
    )
    rooms = {"room_a": room}
    prims = [tilted_opening("atrium", 0.0, 4.0, 3.0, 1.0, 2.0)]
    groups, _ = window_groups(prims, ROLES, rooms, all_rooms=[room])
    assert len(groups) == 1
    assert groups[0].azimuth_bin_deg == 0.0
    assert groups[0].outward_normals[0][1] > 0


# -- roof glass (skylights design, sections 5.2, 5.3 and 5.7) ---------------
#
# A 10 x 10 m hall on one storey (its band open upward), round numbers
# only. Roof glass sits at z = 3.0 to 4.0; the hall's floor polygon is
# "hall_floor" at z = 0.

HALL_OUTLINE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def flat_rect(name, x0, y0, x1, y1, z, modifier="glass", up=True):
    """A horizontal rectangle, wound counter-clockwise from above (+z)
    when ``up``, clockwise (-z) otherwise."""
    pts = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    return glazing(name, pts if up else pts[::-1], modifier)


def hall_rooms(floor_names=("hall_floor",)):
    hall = make_room(0, 0.0, HALL_OUTLINE, floor_names=floor_names)
    return {"hall": hall}, [hall]


def test_a_courtyard_window_stays_perimeter_with_today_s_id():
    room = Room(
        room_id="S0_R0", storey=0, level_z=0.0,
        outline=np.array([(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)]),
        area_m2=60.0, ceiling_height_m=None,
        holes=(np.array([(3.0, 3.0), (5.0, 3.0), (5.0, 5.0), (3.0, 5.0)]),),
    )
    prims = [tilted_opening("court", 0.0, 4.0, 3.0, 1.0, 2.0)]
    groups, _ = window_groups(prims, ROLES, {"room_a": room}, all_rooms=[room])
    assert [g.group_id for g in groups] == ["room_a_000"]
    assert groups[0].kind == "window"
    assert groups[0].tilt_deg == 90.0
    assert groups[0].pane_classes == (PaneClass(90.0, "vertical", "perimeter", False, "window"),)


def test_a_set_back_upper_storey_keeps_its_facade_glass():
    # Storey 1 covers only x in [0, 6]; the lower roof runs from x = 6 to 10.
    lower = make_room(0, 0.0, HALL_OUTLINE, room_id="S0_R0")
    upper = make_room(1, 3.5, [(0.0, 0.0), (6.0, 0.0), (6.0, 10.0), (0.0, 10.0)], room_id="S1_R0")
    rooms = {"lower": lower, "upper": upper}
    pts = [(6.0, 4.0, 4.5), (6.0, 6.0, 4.5), (6.0, 6.0, 5.5), (6.0, 4.0, 5.5)]
    groups, _ = window_groups([glazing("upper_east", pts)], ROLES, rooms, all_rooms=[lower, upper])
    assert [g.group_id for g in groups] == ["upper_090"]
    assert groups[0].room == "upper"
    assert groups[0].pane_classes[0].position == "perimeter"


def test_a_skylight_over_a_double_height_room_is_found_on_the_storey_below():
    # Storey 1 covers only x in [0, 5]; over x in [5, 10] the hall is double height.
    hall = make_room(0, 0.0, HALL_OUTLINE, room_id="S0_R0")
    gallery = make_room(1, 3.5, [(0.0, 0.0), (5.0, 0.0), (5.0, 10.0), (0.0, 10.0)], room_id="S1_R0")
    rooms = {"hall": hall, "gallery": gallery}
    prims = [flat_rect("sky", 7.0, 4.0, 8.0, 6.0, 7.0)]
    groups, _ = window_groups(prims, ROLES, rooms, all_rooms=[hall, gallery])
    assert [g.group_id for g in groups] == ["hall_sky"]
    assert groups[0].storey == 0


def test_sloped_glass_with_its_sill_at_floor_level_is_matched_to_that_storey():
    ground = make_room(0, 0.0, HALL_OUTLINE, room_id="S0_R0")
    first = make_room(1, 3.0, HALL_OUTLINE, room_id="S1_R0")
    rooms = {"ground": ground, "first": first}
    # Sill on storey 1's floor level (z = 3.0), leaning 0.5 m into the room.
    pts = [(4.0, 0.0, 3.0), (6.0, 0.0, 3.0), (6.0, 0.5, 4.0), (4.0, 0.5, 4.0)]
    groups, _ = window_groups([glazing("sloped_sill", pts)], ROLES, rooms, all_rooms=[ground, first])
    assert [g.group_id for g in groups] == ["first_s180"]
    assert groups[0].kind == "sloped"
    assert groups[0].pane_classes[0].position == "perimeter"
    assert groups[0].outward_normals[0][1] < 0.0


def test_a_skylight_over_no_named_room_is_refused():
    named = {"office_a": make_room(0, 0.0, [(20.0, 0.0), (24.0, 0.0), (24.0, 3.0), (20.0, 3.0)], room_id="S0_R1")}
    unnamed = make_room(0, 0.0, HALL_OUTLINE, room_id="S0_R0")
    prims = [
        flat_rect("sky_unnamed", 4.0, 4.0, 6.0, 5.0, 3.5),
        flat_rect("sky_nowhere", 40.0, 40.0, 42.0, 41.0, 3.5),
    ]
    with pytest.raises(GroupError) as excinfo:
        window_groups(prims, ROLES, named, all_rooms=[named["office_a"], unnamed])
    message = str(excinfo.value)
    assert "sky_unnamed: roof glass over no named room" in message
    assert "sky_nowhere: roof glass over no named room" in message


def test_a_facade_window_of_an_unnamed_upper_room_is_not_given_to_the_room_below():
    # A setback: an unnamed upper room (x 0-5) over a larger named lower
    # room (x 0-10), with no roof polygon between them. The upper room's
    # east window is drawn 0.04 m outside its outline. The window rule
    # finds the unnamed upper room, so the pane is unmatched; it must not
    # reach the roof search and land on the lower room as a monitor.
    # Both storeys have floors, so a look-down from the window would find
    # the lower room's floor on its east side (the silent re-homing).
    low = make_room(0, 0.0, HALL_OUTLINE, room_id="S0_R0", floor_names=("low_floor",))
    up = make_room(1, 3.5, [(0.0, 0.0), (5.0, 0.0), (5.0, 10.0), (0.0, 10.0)], room_id="S1_R0",
                   floor_names=("up_floor",))
    named = {"low": low}
    pts = [(5.04, 4.0, 4.5), (5.04, 6.0, 4.5), (5.04, 6.0, 5.5), (5.04, 4.0, 5.5)]
    prims = [
        flat_rect("low_floor", 0.0, 0.0, 10.0, 10.0, 0.0, modifier="floor_mat"),
        flat_rect("up_floor", 0.0, 0.0, 5.0, 10.0, 3.5, modifier="floor_mat"),
        glazing("upper_window", pts),
    ]
    with pytest.raises(GroupError) as excinfo:
        window_groups(prims, STRUCTURE_ROLES, named, all_rooms=[low, up])
    message = str(excinfo.value)
    assert message == (
        "unmatched exterior glazing: upper_window; every room with exterior "
        "glazing must be named in spaces.csv"
    )


def test_a_skylight_on_the_boundary_of_two_rooms_is_refused_as_a_tie():
    left = make_room(0, 0.0, [(0.0, 0.0), (5.0, 0.0), (5.0, 10.0), (0.0, 10.0)], room_id="S0_R0")
    right = make_room(0, 0.0, [(5.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.0, 10.0)], room_id="S0_R1")
    rooms = {"left": left, "right": right}
    prims = [flat_rect("on_the_line", 4.0, 4.0, 6.0, 5.0, 3.5)]
    with pytest.raises(GroupError, match="on_the_line: roof glass lies on the boundary") as excinfo:
        window_groups(prims, ROLES, rooms, all_rooms=[left, right])
    assert "left" in str(excinfo.value)
    assert "right" in str(excinfo.value)


def test_a_pane_0_3_m_in_from_the_outline_stays_perimeter():
    rooms, all_rooms = hall_rooms()
    pts = [(4.0, 0.3, 1.0), (6.0, 0.3, 1.0), (6.0, 0.3, 2.0), (4.0, 0.3, 2.0)]
    groups, _ = window_groups([glazing("near_edge", pts)], ROLES, rooms, all_rooms=all_rooms)
    assert [g.group_id for g in groups] == ["hall_180"]
    assert groups[0].pane_classes[0] == PaneClass(90.0, "vertical", "perimeter", False, "window")


def _sawtooth(flip):
    # 60 degrees from flat, rising to the south, so its glass faces north and up.
    pts = [(4.0, 5.0, 3.0), (6.0, 5.0, 3.0), (6.0, 4.5, 3.0 + math.sqrt(3.0) / 2.0),
           (4.0, 4.5, 3.0 + math.sqrt(3.0) / 2.0)]
    return glazing("sawtooth", pts[::-1] if flip else pts)


@pytest.mark.parametrize("flip", [False, True])
def test_a_mid_roof_sawtooth_pane_is_sloped_roof_glass_facing_up(flip):
    rooms, all_rooms = hall_rooms()
    groups, _ = window_groups([_sawtooth(flip)], ROLES, rooms, all_rooms=all_rooms)
    assert [g.group_id for g in groups] == ["hall_s000"]
    (pane_class,) = groups[0].pane_classes
    assert pane_class.position == "roof"
    assert pane_class.kind == "sloped"
    assert pane_class.tilt_deg == pytest.approx(60.0)
    assert groups[0].outward_normals[0][2] > 0.0


def test_a_pyramid_skylight_makes_four_sloped_roof_groups():
    # Four triangles at 30 degrees over a 2 x 2 m base at z = 3.0, apex over (5, 5).
    apex = (5.0, 5.0, 3.0 + math.tan(math.radians(30.0)))
    corners = [(4.0, 4.0, 3.0), (6.0, 4.0, 3.0), (6.0, 6.0, 3.0), (4.0, 6.0, 3.0)]
    prims = [
        glazing(f"pyramid_{k}", [corners[k], corners[(k + 1) % 4], apex]) for k in range(4)
    ]
    rooms, all_rooms = hall_rooms()
    groups, _ = window_groups(prims, ROLES, rooms, all_rooms=all_rooms)
    assert [g.group_id for g in groups] == ["hall_s000", "hall_s090", "hall_s180", "hall_s270"]
    for group in groups:
        assert group.room == "hall"
        assert group.kind == "sloped"
        assert group.tilt_deg == pytest.approx(30.0)
        assert group.pane_classes[0].position == "roof"
        assert group.outward_normals[0][2] > 0.0


@pytest.mark.parametrize("up", [True, False])
def test_a_skylight_drawn_up_or_down_faces_up(up):
    rooms, all_rooms = hall_rooms()
    prims = [flat_rect("sky", 4.0, 4.0, 6.0, 5.0, 3.5, up=up)]
    groups, _ = window_groups(prims, ROLES, rooms, all_rooms=all_rooms)
    assert [g.group_id for g in groups] == ["hall_sky"]
    assert groups[0].azimuth_bin_deg is None
    assert groups[0].kind == "skylight"
    np.testing.assert_allclose(groups[0].outward_normals[0], [0.0, 0.0, 1.0], atol=1e-12)
    assert groups[0].pane_classes == (PaneClass(0.0, "skylight", "roof", False, "skylight"),)


def test_an_outward_leaning_sloped_facade_faces_out_and_down():
    rooms, all_rooms = hall_rooms()
    # Sill on the south outline at z = 1.0; head 0.5 m further out at z = 2.0.
    pts = [(4.0, 0.0, 1.0), (6.0, 0.0, 1.0), (6.0, -0.5, 2.0), (4.0, -0.5, 2.0)]
    groups, _ = window_groups([glazing("leaning_out", pts)], ROLES, rooms, all_rooms=all_rooms)
    assert [g.group_id for g in groups] == ["hall_s180"]
    assert groups[0].pane_classes[0].position == "perimeter"
    outward = groups[0].outward_normals[0]
    assert outward[1] < 0.0
    assert outward[2] < 0.0


# -- monitors: vertical roof glass, oriented by looking down ----------------
#
# The roof of the hall is at z = 3.0 with a 1 m open strip at x in [2, 3];
# the monitor glass stands on the strip's west edge (x = 2), 2 m in from
# the outline, z = 3.0 to 4.0. Its east side looks down through the strip
# to the hall floor; its west side looks down onto the roof.

STRUCTURE_ROLES = dict(
    ROLES, roof_mat="ceiling", floor_mat="floor", wall_mat="wall",
    desk="furniture", eave_mat="fixed_shading", diff_glass="glazing_diffusing",
)


def _monitor(flip=False):
    pts = [(2.0, 4.0, 3.0), (2.0, 6.0, 3.0), (2.0, 6.0, 4.0), (2.0, 4.0, 4.0)]
    return glazing("monitor", pts[::-1] if flip else pts)


def _hall_structure(roof=("west", "east")):
    prims = [flat_rect("hall_floor", 0.0, 0.0, 10.0, 10.0, 0.0, modifier="floor_mat")]
    if "west" in roof:
        prims.append(flat_rect("roof_w", 0.0, 0.0, 2.0, 10.0, 3.0, modifier="roof_mat", up=False))
    if "east" in roof:
        prims.append(flat_rect("roof_e", 3.0, 0.0, 10.0, 10.0, 3.0, modifier="roof_mat", up=False))
    if "all" in roof:
        prims.append(flat_rect("roof_all", 0.0, 0.0, 10.0, 10.0, 3.0, modifier="roof_mat", up=False))
    return prims


@pytest.mark.parametrize("flip", [False, True])
def test_a_top_storey_monitor_two_metres_in_is_roof_glass(flip):
    rooms, all_rooms = hall_rooms()
    prims = _hall_structure() + [_monitor(flip)]
    groups, _ = window_groups(prims, STRUCTURE_ROLES, rooms, all_rooms=all_rooms)
    assert [g.group_id for g in groups] == ["hall_m270"]
    assert groups[0].kind == "monitor"
    assert groups[0].pane_classes[0].position == "roof"
    assert groups[0].outward_normals[0][0] < 0.0  # out over the roof, west


def test_a_monitor_under_an_eave_overhanging_both_sides_is_still_oriented():
    rooms, all_rooms = hall_rooms()
    eave = flat_rect("eave", 1.5, 3.5, 3.5, 6.5, 3.8, modifier="eave_mat", up=False)
    prims = _hall_structure() + [eave, _monitor()]
    groups, _ = window_groups(prims, STRUCTURE_ROLES, rooms, all_rooms=all_rooms)
    assert [g.group_id for g in groups] == ["hall_m270"]


def test_furniture_under_a_monitor_does_not_block_the_look_down():
    rooms, all_rooms = hall_rooms()
    desk = flat_rect("desk_top", 2.0, 4.0, 3.0, 6.0, 0.75, modifier="desk")
    prims = _hall_structure() + [desk, _monitor()]
    groups, _ = window_groups(prims, STRUCTURE_ROLES, rooms, all_rooms=all_rooms)
    assert [g.group_id for g in groups] == ["hall_m270"]
    assert groups[0].outward_normals[0][0] < 0.0


def test_a_monitor_reaching_the_floor_on_both_sides_is_refused():
    rooms, all_rooms = hall_rooms()
    prims = _hall_structure(roof=()) + [_monitor()]
    with pytest.raises(GroupError, match="monitor: monitor glass reaches its room's floor on both sides"):
        window_groups(prims, STRUCTURE_ROLES, rooms, all_rooms=all_rooms)


def test_a_monitor_reaching_the_floor_on_neither_side_is_refused():
    rooms, all_rooms = hall_rooms()
    prims = _hall_structure(roof=("all",)) + [_monitor()]
    with pytest.raises(GroupError, match="monitor: monitor glass reaches its room's floor on neither side"):
        window_groups(prims, STRUCTURE_ROLES, rooms, all_rooms=all_rooms)


def test_a_floor_of_another_room_is_not_the_monitor_s_inside():
    # The hall's own floor list does not name "hall_floor": neither side counts.
    rooms, all_rooms = hall_rooms(floor_names=("some_other_floor",))
    prims = _hall_structure() + [_monitor()]
    with pytest.raises(GroupError, match="on neither side"):
        window_groups(prims, STRUCTURE_ROLES, rooms, all_rooms=all_rooms)


# -- group ids, kinds and tilt per kind (spec 5.7) --------------------------


def test_each_kind_gets_its_own_group_id_and_the_window_id_is_unchanged():
    rooms, all_rooms = hall_rooms()
    prims = [
        glazing("north", [(4.0, 10.0, 1.0), (6.0, 10.0, 1.0), (6.0, 10.0, 2.0), (4.0, 10.0, 2.0)]),
        flat_rect("sky", 2.0, 2.0, 4.0, 3.0, 3.5),
        flat_rect("diffuser", 6.0, 6.0, 8.0, 7.0, 3.5, modifier="diff_glass"),
        glazing("diffuser_east", [(10.0, 4.0, 1.0), (10.0, 6.0, 1.0), (10.0, 6.0, 2.0), (10.0, 4.0, 2.0)],
                modifier="diff_glass"),
    ]
    groups, warnings = window_groups(prims, STRUCTURE_ROLES, rooms, all_rooms=all_rooms)
    assert warnings == ()
    by_id = {g.group_id: g for g in groups}
    assert sorted(by_id) == ["hall_000", "hall_dif", "hall_sky"]
    assert by_id["hall_000"].kind == "window"
    assert by_id["hall_000"].azimuth_bin_deg == 0.0
    assert by_id["hall_sky"].azimuth_bin_deg is None
    assert by_id["hall_dif"].azimuth_bin_deg is None
    assert by_id["hall_dif"].kind == "diffusing"
    assert [o.name for o in by_id["hall_dif"].openings] == ["diffuser", "diffuser_east"]
    assert [c.position for c in by_id["hall_dif"].pane_classes] == ["roof", "perimeter"]
    assert all(c.diffusing for c in by_id["hall_dif"].pane_classes)
    assert by_id["hall_dif"].tilt_deg == pytest.approx(45.0)  # 2 m2 at 0, 2 m2 at 90


def test_a_group_s_tilt_is_the_area_weighted_mean_of_its_panes():
    rooms, all_rooms = hall_rooms()
    ten = math.radians(10.0)
    prims = [
        flat_rect("sky_flat", 2.0, 2.0, 4.0, 3.0, 3.5),  # 2 m2 at 0 degrees
        plane_opening("sky_tilted", (0.0, math.sin(ten), math.cos(ten)), (7.0, 7.0, 3.5)),  # 1 m2 at 10
    ]
    groups, _ = window_groups(prims, ROLES, rooms, all_rooms=all_rooms)
    assert [g.group_id for g in groups] == ["hall_sky"]
    assert groups[0].tilt_deg == pytest.approx(10.0 / 3.0, abs=1e-6)
    assert [c.tilt_deg for c in groups[0].pane_classes] == pytest.approx([0.0, 10.0])


def test_a_tiny_skylight_group_is_kept_and_warned():
    rooms, all_rooms = hall_rooms()
    prims = [flat_rect("tiny_sky", 5.0, 5.0, 5.3, 5.3, 3.5)]  # 0.09 m2
    groups, warnings = window_groups(prims, ROLES, rooms, all_rooms=all_rooms)
    assert [g.group_id for g in groups] == ["hall_sky"]
    assert len(warnings) == 1
    assert warnings[0].startswith("hall_sky: total glazing area 0.090 m2")


def test_glass_midway_across_a_narrow_room_keeps_today_s_refusal():
    # Both plan steps inside, both exits 0.5 m: within a wall's depth but
    # equal, so this stays today's "cannot tell which side" refusal.
    narrow = {"corridor": make_room(0, 0.0, [(0.0, 0.0), (1.0, 0.0), (1.0, 10.0), (0.0, 10.0)])}
    pts = [(0.5, 4.0, 1.0), (0.5, 6.0, 1.0), (0.5, 6.0, 2.0), (0.5, 4.0, 2.0)]
    with pytest.raises(GroupError, match="midway: cannot tell which side of this opening"):
        window_groups([glazing("midway", pts)], ROLES, narrow, all_rooms=list(narrow.values()))


# -- controller ruling: every golden fixture is still an all-vertical -------
# -- building, and every one of its panes must still group as a perimeter --
# -- window, under the SAME group ids master's frozen manifest records ------

@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_golden_fixture_groups_as_perimeter_windows_with_masters_ids(tmp_path, name):
    # Master's own groups, frozen in tests/fixtures/golden_prepare/<name>/
    # manifest.json on the commit before roof glass existed: ids, room,
    # storey, azimuth bin and opening names. Comparing against that frozen
    # record (not a value this test invents) is what makes this a
    # regression guard. Built through prepare_golden -- the same path
    # golden_prepare.py and test_golden_prepare.py use -- rather than a
    # hand copy of detect_rooms/match_to_rooms/window_groups, so this
    # exercises window_groups exactly as prepare() calls it (against the
    # rebuilt materials, not the raw fixture primitives).
    prepared = prepare_golden(name, tmp_path / name)
    master_groups = json.loads((GOLDEN_DIR / name / "manifest.json").read_text())["groups"]

    assert sorted(g.group_id for g in prepared.groups) == sorted(master_groups)
    for group in prepared.groups:
        assert group.kind == "window", f"{name}: {group.group_id} is not kind 'window'"
        for pane_class in group.pane_classes:
            assert pane_class.position == "perimeter", (
                f"{name}: {group.group_id} has a non-perimeter pane"
            )
        want = master_groups[group.group_id]
        assert group.room == want["room"], f"{name}: {group.group_id} room"
        assert group.storey == want["storey"], f"{name}: {group.group_id} storey"
        assert group.azimuth_bin_deg == pytest.approx(want["azimuth_bin_deg"]), (
            f"{name}: {group.group_id} azimuth_bin_deg"
        )
        assert [o.name for o in group.openings] == want["openings"], (
            f"{name}: {group.group_id} openings"
        )
