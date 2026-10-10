import numpy as np
import pytest

from leed_daylight.scene.primitives import Primitive
from leed_daylight.scene.storeys import _flat, _horizontal, detect_rooms, storey_levels

ROLES = {"slab": "floor", "soffit": "ceiling", "glass": "glazing"}


def slab(name, x0, y0, x1, y1, z, modifier="slab", up=True):
    corners = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    if not up:
        corners = corners[::-1]
    return Primitive(modifier, "polygon", name, (), (),
                     tuple(v for c in corners for v in c), "test")


def test_one_slab_is_one_room():
    rooms = detect_rooms([slab("f", 0, 0, 4, 3, 0.0)], ROLES).rooms
    assert len(rooms) == 1
    assert rooms[0].area_m2 == pytest.approx(12.0)
    assert rooms[0].storey == 0


def test_two_separate_slabs_on_one_level_are_two_rooms():
    prims = [slab("a", 0, 0, 2, 2, 0.0), slab("b", 5, 0, 7, 2, 0.0)]
    assert len(detect_rooms(prims, ROLES).rooms) == 2


def test_two_touching_slabs_merge_into_one_room():
    prims = [slab("a", 0, 0, 2, 2, 0.0), slab("b", 2, 0, 4, 2, 0.0)]
    rooms = detect_rooms(prims, ROLES).rooms
    assert len(rooms) == 1
    assert rooms[0].area_m2 == pytest.approx(8.0)


def test_levels_within_the_tolerance_are_one_storey():
    prims = [slab("a", 0, 0, 2, 2, 0.0), slab("b", 5, 0, 7, 2, 0.02)]
    assert {r.storey for r in detect_rooms(prims, ROLES).rooms} == {0}


def test_levels_beyond_the_tolerance_are_separate_storeys():
    prims = [slab("a", 0, 0, 2, 2, 0.0), slab("b", 0, 0, 2, 2, 3.5)]
    rooms = detect_rooms(prims, ROLES).rooms
    assert {r.storey for r in rooms} == {0, 1}
    assert storey_levels(rooms) == pytest.approx([0.0, 3.5])


def test_downward_facing_polygons_are_not_floors():
    prims = [slab("a", 0, 0, 2, 2, 0.0), slab("c", 0, 0, 2, 2, 3.0, "soffit", up=False)]
    rooms = detect_rooms(prims, ROLES).rooms
    assert len(rooms) == 1
    assert rooms[0].level_z == pytest.approx(0.0)


def test_ceiling_height_comes_from_the_soffit_above():
    prims = [slab("a", 0, 0, 2, 2, 0.0), slab("c", 0, 0, 2, 2, 2.7, "soffit", up=False)]
    assert detect_rooms(prims, ROLES).rooms[0].ceiling_height_m == pytest.approx(2.7)


def test_a_room_with_no_ceiling_reports_none():
    assert detect_rooms([slab("a", 0, 0, 2, 2, 0.0)], ROLES).rooms[0].ceiling_height_m is None


def test_ground_and_context_are_excluded():
    roles = dict(ROLES, terrain="ground", neighbour="context")
    prims = [
        slab("a", 0, 0, 2, 2, 0.0),
        slab("g", -50, -50, 50, 50, -0.1, "terrain"),
        slab("n", 20, 20, 30, 30, 0.0, "neighbour"),
    ]
    assert len(detect_rooms(prims, roles).rooms) == 1


def test_a_tilted_surface_is_not_a_floor():
    tilted = Primitive("slab", "polygon", "ramp", (), (),
                       (0, 0, 0, 2, 0, 0, 2, 2, 1.0, 0, 2, 1.0), "test")
    assert detect_rooms([tilted], ROLES).rooms == ()


def test_room_ids_are_stable_and_unique():
    prims = [slab("a", 0, 0, 2, 2, 0.0), slab("b", 5, 0, 7, 2, 0.0)]
    ids = [r.room_id for r in detect_rooms(prims, ROLES).rooms]
    assert len(set(ids)) == 2
    assert ids == [r.room_id for r in detect_rooms(prims, ROLES).rooms]


def test_outline_is_two_dimensional_and_closed_without_repeating():
    room = detect_rooms([slab("a", 0, 0, 4, 3, 0.0)], ROLES).rooms[0]
    assert room.outline.shape[1] == 2
    assert not np.allclose(room.outline[0], room.outline[-1])


def test_a_staircase_of_slabs_does_not_collapse_into_one_storey():
    # Reproduces the unbounded-drift defect at its smallest scale: three
    # slabs, each gap inside the tolerance of the one before it, but the
    # first and last more than z_tolerance apart. A cluster that compares
    # to the last value added (instead of the cluster's seed) walks this
    # into one storey.
    z_tolerance = 0.05
    steps = [0.00, 0.04, 0.08]
    prims = [slab(f"f{i}", i * 3, 0, i * 3 + 2, 2, z) for i, z in enumerate(steps)]
    rooms = detect_rooms(prims, ROLES, z_tolerance=z_tolerance).rooms
    assert len(rooms) == 3

    by_storey: dict[int, list[float]] = {}
    for r in rooms:
        i = round(r.outline[:, 0].min() / 3)
        by_storey.setdefault(r.storey, []).append(steps[i])

    assert len(by_storey) == 2
    for members in by_storey.values():
        assert max(members) - min(members) <= z_tolerance


def test_no_storey_spans_more_than_the_tolerance():
    # The ledger's full reproduction: thirty slabs, each 0.04 m above the
    # last, every single gap inside the 0.05 m tolerance. The old code
    # collapsed all thirty into one storey spanning 1.16 m. Each floor is
    # given its own non-touching footprint so it survives detect_rooms as
    # its own room; the true, known z of each floor is then grouped by the
    # storey detect_rooms actually assigned it, and the span within every
    # storey is checked directly -- not just the storey count.
    z_tolerance = 0.05
    step = 0.04
    n = 30
    prims = [slab(f"f{i}", i * 3, 0, i * 3 + 2, 2, i * step) for i in range(n)]
    rooms = detect_rooms(prims, ROLES, z_tolerance=z_tolerance).rooms
    assert len(rooms) == n

    by_storey: dict[int, list[float]] = {}
    for r in rooms:
        i = round(r.outline[:, 0].min() / 3)
        by_storey.setdefault(r.storey, []).append(i * step)

    assert len(by_storey) > 1
    for members in by_storey.values():
        assert max(members) - min(members) <= z_tolerance


def test_a_ceiling_touching_only_along_an_edge_is_not_this_room_s_ceiling():
    # The .intersects() defect, reproduced: a ceiling slab sharing only the
    # line x=2 with the room's floor has zero-area overlap with it.
    prims = [
        slab("floor", 0, 0, 2, 2, 0.0),
        slab("ceiling", 2, 0, 4, 2, 2.4, "soffit", up=False),
    ]
    rooms = detect_rooms(prims, ROLES).rooms
    assert len(rooms) == 1
    assert rooms[0].ceiling_height_m is None


def test_a_ceiling_that_exactly_covers_the_room_still_counts():
    # Proves .overlaps() was not used as the repair: shapely's .overlaps()
    # returns False when one polygon exactly covers (or contains) the
    # other, so this case would wrongly report no ceiling under that rule.
    prims = [
        slab("floor", 0, 0, 2, 2, 0.0),
        slab("ceiling", 0, 0, 2, 2, 3.0, "soffit", up=False),
    ]
    rooms = detect_rooms(prims, ROLES).rooms
    assert rooms[0].ceiling_height_m == pytest.approx(3.0)


def test_a_ceiling_that_partly_overlaps_counts():
    # Positive shared area, not full containment, is the rule.
    prims = [
        slab("floor", 0, 0, 4, 2, 0.0),
        slab("ceiling", 0, 0, 2, 2, 2.5, "soffit", up=False),
    ]
    rooms = detect_rooms(prims, ROLES).rooms
    assert rooms[0].ceiling_height_m == pytest.approx(2.5)


def test_a_tilted_slab_is_not_a_storey_floor():
    # A 10 x 10 m slab tilted 4.9 degrees -- inside the 5-degree normal
    # tolerance -- rises from z=0 at one edge to z=h at the other. The
    # mean-z defect would assign it a level around h / 2; the fix is that
    # its own z range exceeds z_tolerance, so it is not a storey floor at
    # all.
    h = 0.857  # 10 m * tan(4.9 degrees), rounded to the test's own tilt
    tilted = Primitive(
        "slab", "polygon", "tilted", (), (),
        (0, 0, 0, 10, 0, 0, 10, 10, h, 0, 10, h), "test",
    )
    assert detect_rooms([tilted], ROLES).rooms == ()


def test_a_flat_floor_beside_a_tilted_one_still_makes_its_room():
    # The exclusion must be surgical: a genuine flat floor elsewhere in the
    # scene must not lose its room because an unrelated slab is tilted.
    h = 0.857  # 10 m * tan(4.9 degrees), rounded to the test's own tilt
    tilted = Primitive(
        "slab", "polygon", "tilted", (), (),
        (20, 0, 0, 30, 0, 0, 30, 10, h, 20, 10, h), "test",
    )
    flat = slab("flat", 0, 0, 4, 3, 0.0)
    detection = detect_rooms([tilted, flat], ROLES)
    assert len(detection.rooms) == 1
    assert detection.rooms[0].level_z == pytest.approx(0.0)
    assert detection.rooms[0].area_m2 == pytest.approx(12.0)
    assert len(detection.warnings) == 1
    assert detection.warnings[0].startswith("tilted:")


# -- detect_rooms reports the floors it excludes -----------------------------
#
# A floor-role primitive that fails _horizontal or _flat used to be dropped
# and named nowhere. Its area silently left the LEED area-weighted
# denominator. These tests cover the warning each exclusion now produces.


def test_a_tilted_floor_is_named_in_the_warnings():
    # Same tilted slab as test_a_tilted_slab_is_not_a_storey_floor.
    h = 0.857  # 10 m * tan(4.9 degrees), rounded to the test's own tilt
    tilted = Primitive(
        "slab", "polygon", "tilted", (), (),
        (0, 0, 0, 10, 0, 0, 10, 10, h, 0, 10, h), "test",
    )
    detection = detect_rooms([tilted], ROLES)
    assert detection.rooms == ()
    assert detection.warnings == (
        "tilted: tilted; its own z range exceeds the z tolerance, "
        "so it is not a storey floor",
    )


def test_a_non_horizontal_floor_role_surface_is_named():
    # A floor-role surface standing vertically -- its normal has no upward
    # component at all, well outside the angle tolerance.
    vertical = Primitive(
        "slab", "polygon", "wall_shaped_floor", (), (),
        (0, 0, 0, 2, 0, 0, 2, 0, 3, 0, 0, 3), "test",
    )
    assert not _horizontal(vertical, True, 5.0)
    detection = detect_rooms([vertical], ROLES)
    assert detection.rooms == ()
    assert detection.warnings == (
        "wall_shaped_floor: not an upward-facing horizontal surface",
    )


def test_a_primitive_failing_both_tests_warns_once():
    # The same vertical surface fails both _horizontal (checked above) and
    # _flat -- its own z range (3 m) is far beyond z_tolerance -- but only
    # one warning is ever produced for it.
    both_fail = Primitive(
        "slab", "polygon", "both_fail", (), (),
        (0, 0, 0, 2, 0, 0, 2, 0, 3, 0, 0, 3), "test",
    )
    assert not _horizontal(both_fail, True, 5.0)
    assert not _flat(both_fail, 0.05)
    detection = detect_rooms([both_fail], ROLES)
    assert len(detection.warnings) == 1
    assert detection.warnings[0] == "both_fail: not an upward-facing horizontal surface"


def test_a_wall_is_never_warned_about():
    prims = [
        slab("floor", 0, 0, 2, 2, 0.0),
        Primitive(
            "wallmod", "polygon", "sidewall", (), (),
            (0, 0, 0, 2, 0, 0, 2, 0, 3, 0, 0, 3), "test",
        ),
    ]
    roles = dict(ROLES, wallmod="wall")
    detection = detect_rooms(prims, roles)
    assert len(detection.rooms) == 1
    assert not any("sidewall" in w for w in detection.warnings)


def test_a_clean_scene_warns_about_nothing():
    detection = detect_rooms([slab("f", 0, 0, 4, 3, 0.0)], ROLES)
    assert detection.warnings == ()


def _ring_of_slabs():
    # Four slabs round a 2 x 2 m atrium in the middle of a 6 x 6 m plate.
    return [
        slab("s", 0, 0, 6, 2, 0.0),
        slab("n", 0, 4, 6, 6, 0.0),
        slab("w", 0, 2, 2, 4, 0.0),
        slab("e", 4, 2, 6, 4, 0.0),
    ]


def test_a_room_round_an_atrium_keeps_the_atrium_as_a_hole():
    rooms = detect_rooms(_ring_of_slabs(), ROLES).rooms
    assert len(rooms) == 1
    room = rooms[0]
    assert room.area_m2 == pytest.approx(32.0)
    assert len(room.holes) == 1
    hole = room.holes[0]
    assert hole.shape[1] == 2
    assert hole.min(axis=0) == pytest.approx([2.0, 2.0])
    assert hole.max(axis=0) == pytest.approx([4.0, 4.0])
    assert room.footprint.area == pytest.approx(32.0)


def test_a_room_with_no_hole_has_an_empty_hole_tuple():
    room = detect_rooms([slab("a", 0, 0, 4, 3, 0.0)], ROLES).rooms[0]
    assert room.holes == ()


def test_a_bow_tie_floor_is_warned_and_skipped_not_a_traceback():
    bow = Primitive("slab", "polygon", "bow", (), (),
                    (0, 0, 0, 3, 0, 0, 0, 2, 0, 1, 3, 0), "test")
    detection = detect_rooms([bow, slab("a", 0, 0, 2, 2, 0.0)], ROLES)
    assert len(detection.rooms) == 1
    assert detection.rooms[0].area_m2 == pytest.approx(4.0)
    assert any("bow" in w and "crosses itself" in w for w in detection.warnings)


def test_a_floor_role_sphere_is_warned_as_not_a_polygon():
    ball = Primitive("slab", "sphere", "ball", (), (), (0, 0, 0, 1), "test")
    detection = detect_rooms([ball, slab("a", 0, 0, 2, 2, 0.0)], ROLES)
    assert len(detection.rooms) == 1
    [warning] = detection.warnings
    assert "ball" in warning and "sphere" in warning and "not a polygon" in warning
    assert "upward-facing" not in warning


def test_a_malformed_floor_polygon_is_warned_not_raised():
    bad = Primitive("slab", "polygon", "bad", (), (), (0, 0, 0, 1, 0, 0, 1, 1), "test")
    detection = detect_rooms([bad, slab("a", 0, 0, 2, 2, 0.0)], ROLES)
    assert len(detection.rooms) == 1
    assert any("bad" in w and "multiple of 3" in w for w in detection.warnings)


def test_a_malformed_ceiling_polygon_is_warned_not_raised():
    bad = Primitive("soffit", "polygon", "badc", (), (), (0, 0, 3, 1, 0, 3, 1, 1), "test")
    detection = detect_rooms([bad, slab("a", 0, 0, 2, 2, 0.0)], ROLES)
    assert len(detection.rooms) == 1
    assert any("badc" in w and "ceiling" in w for w in detection.warnings)


def test_plausible_storey_and_ceiling_heights_give_no_unit_warning():
    prims = [
        slab("a", 0, 0, 2, 2, 0.0), slab("b", 0, 0, 2, 2, 3.5),
        slab("c", 0, 0, 2, 2, 3.0, "soffit", up=False),
    ]
    assert detect_rooms(prims, ROLES).warnings == ()


def test_an_implausible_storey_height_warns_about_the_units():
    # A 3.5 m storey in a centimetre model declared as millimetres reads
    # as 0.35 m.
    prims = [slab("a", 0, 0, 2, 2, 0.0), slab("b", 0, 0, 2, 2, 0.35)]
    warnings = detect_rooms(prims, ROLES).warnings
    assert any("storey 1" in w and "0.35" in w and "scene.units" in w for w in warnings), warnings


def test_an_implausible_ceiling_height_warns_about_the_units():
    prims = [slab("a", 0, 0, 2, 2, 0.0), slab("c", 0, 0, 2, 2, 12.0, "soffit", up=False)]
    detection = detect_rooms(prims, ROLES)
    assert detection.rooms[0].ceiling_height_m == pytest.approx(12.0)
    assert any("S0_R0" in w and "12.00" in w and "scene.units" in w
               for w in detection.warnings), detection.warnings


def test_a_sloped_ceiling_is_warned_about():
    sloped = Primitive("soffit", "polygon", "slope", (), (),
                       (0, 0, 2.95, 0, 2, 2.95, 2, 2, 3.05, 2, 0, 3.05), "test")
    detection = detect_rooms([slab("a", 0, 0, 2, 2, 0.0), sloped], ROLES)
    assert detection.rooms[0].ceiling_height_m == pytest.approx(3.0)
    assert any("S0_R0" in w and "slope" in w and "not flat" in w
               for w in detection.warnings), detection.warnings


def _keyhole_floor():
    """A 10 x 10 m floor round a 4 x 4 m hole, written as ONE Radiance
    polygon: the outer loop, a zero-width seam in, the hole the other way
    round, and the same seam back out."""
    ring = [
        (0, 0), (10, 0), (10, 10), (0, 10), (0, 0),
        (3, 3), (3, 7), (7, 7), (7, 3), (3, 3),
    ]
    return Primitive("slab", "polygon", "key", (), (),
                     tuple(v for x, y in ring for v in (x, y, 0.0)), "test")


def _merged_floor():
    """The same floor as four touching slabs."""
    return [
        slab("s", 0, 0, 10, 3, 0.0),
        slab("n", 0, 7, 10, 10, 0.0),
        slab("w", 0, 3, 3, 7, 0.0),
        slab("e", 7, 3, 10, 7, 0.0),
    ]


def test_a_keyhole_floor_gives_the_same_room_as_the_merged_pieces():
    from shapely.geometry import Polygon

    from leed_daylight.build.grid import sensor_points
    from leed_daylight.project.config import Grid

    [key] = detect_rooms([_keyhole_floor()], ROLES).rooms
    [merged] = detect_rooms(_merged_floor(), ROLES).rooms

    assert key.area_m2 == pytest.approx(84.0)
    assert key.area_m2 == pytest.approx(merged.area_m2)
    assert len(key.holes) == len(merged.holes) == 1
    assert Polygon(key.holes[0]).equals(Polygon(merged.holes[0]))
    assert key.footprint.equals(merged.footprint)
    # the same orientation: shapely's exterior and hole rings, both ways
    assert Polygon(key.outline).exterior.is_ccw == Polygon(merged.outline).exterior.is_ccw
    assert Polygon(key.holes[0]).exterior.is_ccw == Polygon(merged.holes[0]).exterior.is_ccw

    grid = Grid(spacing_m=0.5, work_plane_m=0.8, min_clearance_m=0.3)
    key_points = sensor_points(key.outline, key.level_z, grid, holes=key.holes)
    merged_points = sensor_points(merged.outline, merged.level_z, grid, holes=merged.holes)
    assert key_points.shape == merged_points.shape
    assert key_points == pytest.approx(merged_points)


def test_a_keyhole_whose_hole_runs_the_wrong_way_is_named():
    # The hole loop wound the same way as the outer loop: the vertex loop
    # encloses 100 + 16 m2, the plan area is 100 m2. Named, not silent.
    ring = [
        (0, 0), (10, 0), (10, 10), (0, 10), (0, 0),
        (3, 3), (7, 3), (7, 7), (3, 7), (3, 3),
    ]
    prim = Primitive("slab", "polygon", "odd_key", (), (),
                     tuple(v for x, y in ring for v in (x, y, 0.0)), "test")
    detection = detect_rooms([prim], ROLES)
    assert detection.rooms[0].area_m2 == pytest.approx(100.0)
    assert any("odd_key" in w and "overlaps itself" in w for w in detection.warnings)


# -- Room.floor_names (skylights spec 5.3) ------------------------------------


def test_a_room_names_its_floor():
    room = detect_rooms([slab("f", 0, 0, 4, 3, 0.0)], ROLES).rooms[0]
    assert room.floor_names == ("f",)


def test_merged_slabs_are_named_in_scene_order():
    prims = [slab("b", 2, 0, 4, 2, 0.0), slab("a", 0, 0, 2, 2, 0.0)]
    (room,) = detect_rooms(prims, ROLES).rooms
    assert room.floor_names == ("b", "a")


def test_each_separate_room_names_only_its_own_floor():
    prims = [slab("a", 0, 0, 2, 2, 0.0), slab("b", 5, 0, 7, 2, 0.0), slab("c", 0, 0, 2, 2, 3.5)]
    names = {(r.storey, r.floor_names) for r in detect_rooms(prims, ROLES).rooms}
    assert names == {(0, ("a",)), (0, ("b",)), (1, ("c",))}


def test_a_room_round_an_atrium_names_all_four_slabs():
    (room,) = detect_rooms(_ring_of_slabs(), ROLES).rooms
    assert room.floor_names == ("s", "n", "w", "e")


def test_an_excluded_floor_is_not_named():
    tilted = Primitive("slab", "polygon", "ramp", (), (),
                       (2, 0, 0, 4, 0, 0.5, 4, 2, 0.5, 2, 2, 0), "test")
    (room,) = detect_rooms([slab("a", 0, 0, 2, 2, 0.0), tilted], ROLES).rooms
    assert room.floor_names == ("a",)


def test_a_hand_built_room_has_no_floor_names():
    from leed_daylight.scene.storeys import Room

    room = Room(room_id="S0_R0", storey=0, level_z=0.0, outline=np.zeros((3, 2)),
                area_m2=0.0, ceiling_height_m=None)
    assert room.floor_names == ()


# -- public helpers for split halves (merge-split spec 6.1) --------------------


def test_ceiling_height_is_public_and_gives_the_detected_height():
    from leed_daylight.scene import storeys
    from leed_daylight.scene.storeys import ceiling_height, ceiling_surfaces

    soffit = slab("c", 0, 0, 4, 3, 2.7, "soffit", up=False)
    prims = [slab("f", 0, 0, 4, 3, 0.0), soffit]
    (room,) = detect_rooms(prims, ROLES).rooms

    height, surface = ceiling_height(room.footprint, room.level_z, ceiling_surfaces(prims, ROLES))

    assert height == pytest.approx(room.ceiling_height_m) == pytest.approx(2.7)
    assert surface is soffit
    assert not hasattr(storeys, "_ceiling_height")


def test_ceiling_surfaces_are_the_downward_ceiling_polygons():
    from leed_daylight.scene.storeys import ceiling_surfaces

    high = slab("high", 0, 0, 2, 2, 3.0, "soffit", up=False)
    low = slab("low", 2, 0, 4, 2, 2.5, "soffit", up=False)
    upward = slab("upward", 0, 0, 2, 2, 3.0, "soffit", up=True)
    sphere = Primitive("soffit", "sphere", "ball", (), (), (1, 1, 2, 0.5), "test")
    floor = slab("floor", 0, 0, 4, 2, 0.0)
    wall = Primitive("wall", "polygon", "wall", (), (), (0, 0, 0, 4, 0, 0, 4, 0, 3, 0, 0, 3), "test")
    roles = dict(ROLES, wall="wall")

    assert ceiling_surfaces([floor, high, upward, sphere, wall, low], roles) == (high, low)
    assert ceiling_surfaces([floor], roles) == ()


def test_plan_polygon_is_the_normalised_plan_of_a_floor():
    from shapely.geometry import Polygon

    from leed_daylight.scene.storeys import plan_polygon

    keyhole = plan_polygon(_keyhole_floor())
    assert isinstance(keyhole, Polygon)
    assert keyhole.is_valid
    assert keyhole.area == pytest.approx(84.0)
    assert len(keyhole.interiors) == 1
    assert plan_polygon(slab("f", 0, 0, 4, 3, 0.0)).equals(Polygon([(0, 0), (4, 0), (4, 3), (0, 3)]))
