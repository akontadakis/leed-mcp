# tests/test_build_blinds.py
"""Tests for leed_daylight.build.blinds (T3.10): operable blinds.

Every fixture is a fictional building with round, invented numbers
(phase3-global.md #5): a 4 x 3 m room called "room_a", 1 x 1 m openings, a
0.05 m blind offset -- and, for the material tests, invented blind product
values, never a real LM-83-12 clause 2.2.7 substitution or anything read
off a client's product data.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from leed_daylight import geometry
from leed_daylight.build.blinds import (
    BLIND_OFFSET_M,
    BlindError,
    blind_material,
    blind_polygons,
)
from leed_daylight.build.glazing_class import PaneClass
from leed_daylight.build.groups import WindowGroup, window_groups
from leed_daylight.project.config import Blinds
from leed_daylight.scene.primitives import Primitive
from leed_daylight.scene.storeys import Room

# -- fixtures ---------------------------------------------------------------


def opening(name, points, modifier="glass"):
    flat = tuple(v for pt in points for v in pt)
    return Primitive(modifier, "polygon", name, (), (), flat, "test")


# A 1 x 1 m opening in the x-y plane at y=0, wound so its outward normal is
# +y (a north-facing window as seen from outside; the room is to the south,
# i.e. at smaller y -- so "inward" for this opening is -y).
OPENING_PLUS_Y = np.array(
    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)]
)
NORMAL_PLUS_Y = np.array([0.0, 1.0, 0.0])

# An opening on a south wall: outward normal -y, room to the north (+y), so
# "inward" is +y -- the direction the brief's offset test checks.
OPENING_SOUTH = np.array(
    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)]
)
NORMAL_SOUTH = np.array([0.0, -1.0, 0.0])

# The same opening with vertex 1 duplicated: a zero-length edge.
OPENING_WITH_DUPLICATE_VERTEX = np.array(
    [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    ]
)

# Three collinear points: zero area, no normal, degenerate.
OPENING_DEGENERATE = np.array(
    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
)


def make_group(openings, normals, group_id="room_a_000", room="room_a", storey=0,
                azimuth_bin_deg=0.0, area_m2=1.0):
    return WindowGroup(
        group_id=group_id,
        room=room,
        storey=storey,
        azimuth_bin_deg=azimuth_bin_deg,
        openings=tuple(openings),
        outward_normals=tuple(normals),
        area_m2=area_m2,
    )


# -- blind_polygons: the brief's table --------------------------------------


def test_one_blind_per_opening():
    op1 = opening("w1", OPENING_PLUS_Y)
    op2 = opening("w2", OPENING_PLUS_Y + np.array([2.0, 0.0, 0.0]))
    group = make_group([op1, op2], [NORMAL_PLUS_Y, NORMAL_PLUS_Y])
    polys = blind_polygons(group)
    assert len(polys) == len(group.openings) == 2


def test_the_blind_is_offset_inward():
    op = opening("south_window", OPENING_SOUTH)
    group = make_group([op], [NORMAL_SOUTH])
    polys = blind_polygons(group)
    assert len(polys) == 1
    blind = polys[0]
    for i in range(len(OPENING_SOUTH)):
        offset = blind[i] - OPENING_SOUTH[i]
        assert offset[0] == pytest.approx(0.0, abs=1e-12)
        assert offset[1] == pytest.approx(BLIND_OFFSET_M)  # north, i.e. +y
        assert offset[2] == pytest.approx(0.0, abs=1e-12)


def test_the_blind_matches_its_opening_s_area():
    op = opening("w1", OPENING_PLUS_Y)
    group = make_group([op], [NORMAL_PLUS_Y])
    blind = blind_polygons(group)[0]
    assert geometry.area(blind) == pytest.approx(geometry.area(OPENING_PLUS_Y), abs=1e-9)


def test_the_blind_material_carries_the_project_values():
    # Blinds(transmittance_diffuse, transmittance_specular, room_side_reflectance)
    blinds = Blinds(0.045, 0.0, 0.56)
    prim = blind_material(blinds)
    assert prim.ptype == "trans"
    assert prim.modifier == "void"
    assert prim.name == "leed_blind"
    assert len(prim.fargs) == 7
    red, green, blue, spec, rough, a6, a7 = prim.fargs
    assert red == green == blue
    c = red
    assert spec == 0.0
    assert rough == 0.0
    # the inverse recovers the project's own values (controller ruling)
    rd = c * (1.0 - a6)
    td = c * a6 * (1.0 - a7)
    ts = c * a6 * a7
    assert rd == pytest.approx(blinds.room_side_reflectance, abs=1e-12)
    assert td == pytest.approx(blinds.transmittance_diffuse, abs=1e-12)
    assert ts == pytest.approx(blinds.transmittance_specular, abs=1e-12)


def test_the_blind_material_carries_the_project_values_with_specular_transmittance():
    # A second case so A7 (the specular share of transmittance) is exercised.
    blinds = Blinds(0.04, 0.02, 0.50)
    prim = blind_material(blinds)
    red, green, blue, spec, rough, a6, a7 = prim.fargs
    c = red
    assert a7 > 0.0
    rd = c * (1.0 - a6)
    td = c * a6 * (1.0 - a7)
    ts = c * a6 * a7
    assert rd == pytest.approx(blinds.room_side_reflectance, abs=1e-12)
    assert td == pytest.approx(blinds.transmittance_diffuse, abs=1e-12)
    assert ts == pytest.approx(blinds.transmittance_specular, abs=1e-12)


def test_the_blind_material_has_no_default():
    with pytest.raises(TypeError):
        blind_material()


def test_interior_glazing_never_gets_a_blind():
    # The T3.9 mixed scene: an exterior and an interior opening, coplanar
    # (spec 7's integration test, second half).
    room = {"room_a": Room(
        room_id="S0_R0", storey=0, level_z=0.0,
        area_m2=12.0, ceiling_height_m=3.0,
        outline=np.array([(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]),
    )}
    roles = {"glass": "glazing", "int_glass": "glazing_interior"}
    ext = opening("ext", OPENING_PLUS_Y + np.array([1.5, 0.0, 1.0]), modifier="glass")
    interior = opening(
        "int", OPENING_PLUS_Y + np.array([1.7, 0.0, 1.0]), modifier="int_glass"
    )
    groups, warnings = window_groups([ext, interior], roles, room, all_rooms=list(room.values()))
    assert warnings == ()
    interior_pts = interior.points
    interior_normal = geometry.normal(interior_pts)
    interior_origin = interior_pts[0]

    def is_coplanar(poly_pts):
        return bool(
            np.all(
                np.abs((poly_pts - interior_origin) @ interior_normal) < 1e-9
            )
        )

    for group in groups:
        assert "int" not in {o.name for o in group.openings}
        for blind in blind_polygons(group):
            assert not is_coplanar(blind)


def test_fixed_shading_is_untouched():
    room = {"room_a": Room(
        room_id="S0_R0", storey=0, level_z=0.0,
        area_m2=12.0, ceiling_height_m=3.0,
        outline=np.array([(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]),
    )}
    roles = {"glass": "glazing", "shade_mod": "fixed_shading"}
    ext = opening("ext", OPENING_PLUS_Y + np.array([1.5, 0.0, 1.0]), modifier="glass")
    shade = opening(
        "overhang",
        [(0.0, -0.5, 2.0), (1.0, -0.5, 2.0), (1.0, 0.5, 2.0), (0.0, 0.5, 2.0)],
        modifier="shade_mod",
    )
    prims = [ext, shade]
    snapshot = copy.deepcopy(prims)

    groups, _ = window_groups(prims, roles, room, all_rooms=list(room.values()))
    for group in groups:
        blind_polygons(group)

    assert prims == snapshot
    assert prims[1] is shade


# -- hostile-input pass -------------------------------------------------


def test_blind_polygons_on_an_empty_group_returns_empty():
    group = make_group([], [])
    assert blind_polygons(group) == ()


def test_blind_polygons_rejects_mismatched_openings_and_normals():
    op = opening("w1", OPENING_PLUS_Y)
    group = make_group([op], [NORMAL_PLUS_Y, NORMAL_PLUS_Y])
    with pytest.raises(BlindError, match="outward_normals"):
        blind_polygons(group)


def test_blind_polygons_rejects_a_zero_length_edge():
    op = opening("w1", OPENING_WITH_DUPLICATE_VERTEX)
    group = make_group([op], [NORMAL_PLUS_Y])
    with pytest.raises(BlindError, match="w1"):
        blind_polygons(group)


def test_blind_polygons_rejects_a_degenerate_opening():
    op = opening("w1", OPENING_DEGENERATE)
    group = make_group([op], [NORMAL_PLUS_Y])
    with pytest.raises(BlindError, match="w1"):
        blind_polygons(group)


def test_blind_polygons_rejects_non_finite_opening_coordinates():
    pts = OPENING_PLUS_Y.copy()
    pts[0, 0] = float("nan")
    op = opening("w1", pts)
    group = make_group([op], [NORMAL_PLUS_Y])
    with pytest.raises(BlindError, match="w1"):
        blind_polygons(group)


def test_blind_polygons_rejects_too_few_vertices():
    op = opening("w1", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    group = make_group([op], [NORMAL_PLUS_Y])
    with pytest.raises(BlindError, match="w1"):
        blind_polygons(group)


def test_blind_polygons_rejects_a_zero_length_normal():
    op = opening("w1", OPENING_PLUS_Y)
    group = make_group([op], [(0.0, 0.0, 0.0)])
    with pytest.raises(BlindError, match="outward_normals"):
        blind_polygons(group)


def test_blind_polygons_rejects_a_non_finite_normal():
    op = opening("w1", OPENING_PLUS_Y)
    group = make_group([op], [(0.0, float("inf"), 0.0)])
    with pytest.raises(BlindError, match="outward_normals"):
        blind_polygons(group)


def test_blind_polygons_rejects_a_non_numeric_normal():
    op = opening("w1", OPENING_PLUS_Y)
    group = make_group([op], [("a", "b", "c")])
    with pytest.raises(BlindError, match="outward_normals"):
        blind_polygons(group)


def test_blind_polygons_rejects_a_wrong_shape_normal():
    op = opening("w1", OPENING_PLUS_Y)
    group = make_group([op], [(0.0, 1.0)])
    with pytest.raises(BlindError, match="outward_normals"):
        blind_polygons(group)


def test_blind_polygons_rejects_a_wrong_shape_opening():
    class _Fake:
        name = "w1"
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]  # (n, 2), not (n, 3)

    group = make_group([_Fake()], [NORMAL_PLUS_Y])
    with pytest.raises(BlindError, match="w1"):
        blind_polygons(group)


def test_blind_polygons_rejects_an_empty_opening():
    class _Fake:
        name = "w1"
        points = []

    group = make_group([_Fake()], [NORMAL_PLUS_Y])
    with pytest.raises(BlindError, match="w1"):
        blind_polygons(group)


def test_blind_polygons_rejects_a_ragged_opening():
    class _Fake:
        name = "w1"
        points = [(0.0, 0.0, 0.0), (1.0, 0.0), (1.0, 0.0, 1.0)]

    group = make_group([_Fake()], [NORMAL_PLUS_Y])
    with pytest.raises(BlindError, match="w1"):
        blind_polygons(group)


def test_blind_polygons_accepts_an_edge_exactly_at_the_tolerance():
    # 1e-9 is the module's own edge-length tolerance; an edge exactly at
    # that length is not a duplicate and must be accepted (strict "<").
    pts = OPENING_PLUS_Y.copy()
    pts[1] = pts[0] + np.array([1e-9, 0.0, 0.0])
    op = opening("w1", pts)
    group = make_group([op], [NORMAL_PLUS_Y])
    # Not a refusal for the edge length; the polygon is still valid (a
    # near-degenerate sliver, but not below the tolerance).
    blind_polygons(group)


def test_blind_material_rejects_zero_sum():
    with pytest.raises(BlindError, match="room_side_reflectance"):
        blind_material(Blinds(0.0, 0.0, 0.0))


def test_blind_material_rejects_sum_above_one():
    with pytest.raises(BlindError, match="room_side_reflectance"):
        blind_material(Blinds(0.6, 0.0, 0.6))


def test_blind_material_accepts_sum_exactly_one():
    prim = blind_material(Blinds(0.5, 0.0, 0.5))
    assert prim.fargs[0] == pytest.approx(1.0)


def test_blind_material_rejects_nan():
    with pytest.raises(BlindError, match="transmittance_diffuse"):
        blind_material(Blinds(float("nan"), 0.0, 0.5))


def test_blind_material_rejects_inf():
    with pytest.raises(BlindError, match="room_side_reflectance"):
        blind_material(Blinds(0.1, 0.0, float("inf")))


def test_blind_material_rejects_non_numeric():
    with pytest.raises(BlindError, match="transmittance_specular"):
        blind_material(Blinds(0.1, "a lot", 0.5))


def test_blind_material_rejects_a_malformed_object():
    with pytest.raises(BlindError, match="room_side_reflectance"):
        blind_material(object())


# -- roof glass: the skylight blind moves straight down (spec 5.8) ----------

#: A 2 x 1 m pane tilted 15 degrees, rising along +y from z = 3.5, and its
#: upward unit normal.
_RISE_15 = math.tan(math.radians(15.0))
SKYLIGHT_15_DEG = np.array(
    [(4.0, 4.0, 3.5), (6.0, 4.0, 3.5), (6.0, 5.0, 3.5 + _RISE_15), (4.0, 5.0, 3.5 + _RISE_15)]
)
NORMAL_15_DEG = np.array([0.0, -math.sin(math.radians(15.0)), math.cos(math.radians(15.0))])


def _classed_group(pts, normal, pane_class, group_id="hall_sky"):
    return WindowGroup(
        group_id=group_id, room="hall", storey=0, azimuth_bin_deg=None,
        openings=(opening("pane", pts),), outward_normals=(normal,), area_m2=2.0,
        kind=pane_class.kind, tilt_deg=pane_class.tilt_deg, pane_classes=(pane_class,),
    )


def test_a_skylight_blind_moves_straight_down_not_along_its_tilted_normal():
    group = _classed_group(
        SKYLIGHT_15_DEG, NORMAL_15_DEG, PaneClass(15.0, "skylight", "roof", False, "skylight")
    )
    (blind,) = blind_polygons(group)
    np.testing.assert_allclose(blind, SKYLIGHT_15_DEG - np.array([0.0, 0.0, BLIND_OFFSET_M]))


def test_a_sloped_pane_blind_still_moves_along_its_normal():
    tilt = math.radians(30.0)
    rise = math.tan(tilt)
    pts = np.array([(4.0, 4.0, 3.5), (6.0, 4.0, 3.5), (6.0, 5.0, 3.5 + rise), (4.0, 5.0, 3.5 + rise)])
    normal = np.array([0.0, -math.sin(tilt), math.cos(tilt)])
    group = _classed_group(
        pts, normal, PaneClass(30.0, "sloped", "roof", False, "sloped"), group_id="hall_s180"
    )
    (blind,) = blind_polygons(group)
    np.testing.assert_allclose(blind, pts - BLIND_OFFSET_M * normal)


def test_a_group_without_pane_classes_keeps_today_s_blind():
    group = make_group([opening("w1", OPENING_SOUTH)], [NORMAL_SOUTH])
    assert group.pane_classes == ()
    (blind,) = blind_polygons(group)
    np.testing.assert_allclose(blind, OPENING_SOUTH - BLIND_OFFSET_M * NORMAL_SOUTH)


def test_pane_classes_not_aligned_with_openings_are_refused():
    group = WindowGroup(
        group_id="hall_sky", room="hall", storey=0, azimuth_bin_deg=None,
        openings=(opening("a", SKYLIGHT_15_DEG), opening("b", SKYLIGHT_15_DEG + 3.0)),
        outward_normals=(NORMAL_15_DEG, NORMAL_15_DEG), area_m2=4.0, kind="skylight",
        pane_classes=(PaneClass(15.0, "skylight", "roof", False, "skylight"),),
    )
    with pytest.raises(BlindError, match="pane_classes"):
        blind_polygons(group)


def test_a_grouped_flat_skylight_gets_its_blind_0_05_m_below():
    room = Room(
        room_id="S0_R0", storey=0, level_z=0.0,
        outline=np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]),
        area_m2=100.0, ceiling_height_m=None,
    )
    sky = opening("sky", [(4.0, 4.0, 3.5), (6.0, 4.0, 3.5), (6.0, 5.0, 3.5), (4.0, 5.0, 3.5)])
    groups, _ = window_groups([sky], {"glass": "glazing"}, {"hall": room}, all_rooms=[room])
    (blind,) = blind_polygons(groups[0])
    np.testing.assert_allclose(blind[:, 2], [3.45] * 4)
