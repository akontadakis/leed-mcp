# tests/test_project_layout_split.py
"""Cutting a detected room by a split line (merge-split spec 4 step 2, 6.1, 7).

Every room here is an invented rectangle drawn as floor polygons and found by
``detect_rooms``; every coordinate is round. The refusals are in
``tests/test_project_layout_refusals.py``.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from shapely.geometry import Polygon

from leed_daylight.project.config import Grid, ProjectError
from leed_daylight.project.layout import (
    MIN_HALF_AREA_M2,
    ON_LINE_TOL_M,
    ORIGINS,
    PANE_REACH_M,
    Cut,
    LayoutError,
    SplitHalf,
    apply_splits,
    half_mask,
    left_of,
)
from leed_daylight.project.splits import SplitLine
from leed_daylight.scene.primitives import Primitive
from leed_daylight.scene.storeys import Room, detect_rooms

ROLES = {"slab": "floor", "soffit": "ceiling"}
#: A grid whose lattice over the 6 x 4 m room runs x = 0.25, 0.75, ... 5.75.
GRID = Grid(spacing_m=0.5, work_plane_m=0.75, min_clearance_m=0.1)


def slab(name, x0, y0, x1, y1, z, modifier="slab", up=True):
    corners = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    if not up:
        corners = corners[::-1]
    return Primitive(modifier, "polygon", name, (), (), tuple(float(v) for c in corners for v in c), "test")


def line(*points, row=2, level_z=0.0):
    return SplitLine(row=row, level_z=level_z, points=np.array(points, dtype=np.float64), note="")


def room_6_by_4():
    """One 6 x 4 m room at z = 0 under a ceiling at z = 3."""
    return [slab("floor_a", 0, 0, 6, 4, 0.0), slab("ceiling_a", 0, 0, 6, 4, 3.0, "soffit", up=False)]


def split(prims, *lines):
    return apply_splits(detect_rooms(prims, ROLES), lines, prims, ROLES)


def test_the_constants_of_the_contract():
    assert MIN_HALF_AREA_M2 == 1.0
    assert ON_LINE_TOL_M == 1e-9
    assert PANE_REACH_M == 0.6
    assert ORIGINS == ("detected", "split", "merged", "split+merged")
    assert issubclass(LayoutError, ProjectError)
    assert issubclass(SplitHalf, Room)


def test_a_straight_line_cuts_a_rectangle_into_two_halves():
    prims = room_6_by_4()
    before = detect_rooms(prims, ROLES)
    (room,) = before.rooms

    detection = apply_splits(before, (line((2, 0), (2, 4)),), prims, ROLES)

    a, b = detection.rooms
    assert [type(a), type(b)] == [SplitHalf, SplitHalf]
    assert (a.room_id, b.room_id) == ("S0_R0a", "S0_R0b")
    # The line runs north, so its left is west: half a is x < 2.
    assert (a.area_m2, b.area_m2) == (8.0, 16.0)
    assert a.footprint.equals(Polygon([(0, 0), (2, 0), (2, 4), (0, 4)]))
    assert b.footprint.equals(Polygon([(2, 0), (6, 0), (6, 4), (2, 4)]))
    assert len(a.outline) == len(b.outline) == 4
    assert a.holes == b.holes == ()
    assert (a.storey, a.level_z, b.storey, b.level_z) == (0, 0.0, 0, 0.0)
    assert a.ceiling_height_m == b.ceiling_height_m == 3.0
    assert a.floor_names == b.floor_names == ("floor_a",)
    assert a.source_outline is room.outline and b.source_outline is room.outline
    assert a.source_holes == b.source_holes == ()
    (cut_a,), (cut_b,) = a.cuts, b.cuts
    assert isinstance(cut_a, Cut)
    assert (cut_a.row, cut_a.side, cut_b.side) == (2, "a", "b")
    assert cut_a.line.tolist() == [[2.0, 0.0], [2.0, 4.0]]
    assert cut_a.drawn.tolist() == [[2.0, 0.0], [2.0, 4.0]]
    assert cut_a.left.equals(a.footprint) and cut_a.right.equals(b.footprint)
    assert detection.warnings == ()


def test_a_bent_line_cuts_a_rectangle():
    prims = room_6_by_4()

    # East along y = 2, then north along x = 3: the left of the longest
    # segment (the first, 3 m) is north, so half a is the north-west corner.
    a, b = split(prims, line((0, 2), (3, 2), (3, 4))).rooms

    assert (a.area_m2, b.area_m2) == (6.0, 18.0)
    assert a.footprint.equals(Polygon([(0, 2), (3, 2), (3, 4), (0, 4)]))
    assert b.footprint.equals(Polygon([(0, 0), (6, 0), (6, 4), (3, 4), (3, 2), (0, 2)]))
    assert a.cuts[0].drawn.tolist() == [[0.0, 2.0], [3.0, 2.0], [3.0, 4.0]]


def test_the_halves_points_are_the_unsplit_grid_point_for_point():
    from leed_daylight.build.grid import sensor_points

    prims = room_6_by_4()
    (room,) = detect_rooms(prims, ROLES).rooms
    whole = sensor_points(room.outline, room.level_z, GRID, holes=room.holes)
    # A bent line through lattice points: x = 2.75 is a lattice column.
    a, b = split(prims, line((2.75, 0), (2.75, 2.25), (6, 2.25))).rooms

    in_a, in_b = half_mask(a, whole), half_mask(b, whole)

    assert in_a.dtype == bool and in_a.shape == (len(whole),)
    assert not np.any(in_a & in_b)
    assert np.all(in_a | in_b)
    # Points on the line go left: west of the north run, north of the east run.
    on_line = (np.isclose(whole[:, 0], 2.75) & (whole[:, 1] <= 2.25)) | (
        np.isclose(whole[:, 1], 2.25) & (whole[:, 0] >= 2.75))
    assert on_line.sum() == 5 + 7 - 1  # (2.75, 2.25) is on both segments
    assert np.all(in_a[on_line])
    assert np.all(in_a[whole[:, 0] < 2.75])
    assert not np.any(in_a[(whole[:, 0] > 2.75) & (whole[:, 1] < 2.25)])
    assert in_a.sum() + in_b.sum() == len(whole) == 96


def test_a_plain_room_keeps_every_point():
    prims = room_6_by_4()
    (room,) = detect_rooms(prims, ROLES).rooms

    mask = half_mask(room, np.array([[1.0, 1.0, 0.75], [5.0, 3.0, 0.75]]))

    assert mask.tolist() == [True, True]


def test_a_point_on_the_line_goes_left():
    a, b = split(room_6_by_4(), line((3, 0), (3, 4))).rooms
    (cut,) = a.cuts
    points = np.array([[3.0, 2.0], [3.0 + 1e-10, 1.0], [2.0, 2.0], [4.0, 2.0], [3.0 + 1e-6, 3.0]])

    assert left_of(points, cut).tolist() == [True, True, True, False, False]
    assert half_mask(a, points).tolist() == [True, True, True, False, False]
    assert half_mask(b, points).tolist() == [False, False, False, True, True]


def test_a_point_at_a_bend_vertex_goes_left():
    a, _ = split(room_6_by_4(), line((0, 2), (3, 2), (3, 4))).rooms
    (cut,) = a.cuts

    # The bend (3, 2) is on the line; (3.5, 1.5) is inside half b, (1, 3) inside half a.
    assert left_of(np.array([[3.0, 2.0], [3.5, 1.5], [1.0, 3.0]]), cut).tolist() == [True, False, True]


def test_a_point_outside_both_halves_takes_the_side_of_the_nearest_segment():
    a, _ = split(room_6_by_4(), line((0, 2), (3, 2), (3, 4))).rooms
    (cut,) = a.cuts

    # Outside the room: west of the building (nearest the first segment,
    # left of it when north of y = 2) and north of it (nearest the second).
    points = np.array([[-1.0, 3.0], [-1.0, 1.0], [2.0, 5.0], [4.0, 5.0]])

    assert left_of(points, cut).tolist() == [True, False, True, False]
    assert left_of(np.zeros((0, 3)), cut).tolist() == []


def test_an_end_point_1e_12_inside_is_snapped_not_refused():
    a, b = split(room_6_by_4(), line((3, 1e-12), (3, 4))).rooms

    assert a.cuts[0].line.tolist() == [[3.0, 0.0], [3.0, 4.0]]
    assert (a.area_m2, b.area_m2) == (12.0, 12.0)


def test_an_end_point_outside_the_room_is_kept():
    a, b = split(room_6_by_4(), line((3, -1), (3, 5))).rooms

    assert a.cuts[0].line.tolist() == [[3.0, -1.0], [3.0, 5.0]]
    assert a.cuts[0].drawn.tolist() == [[3.0, 0.0], [3.0, 4.0]]
    assert (a.area_m2, b.area_m2) == (12.0, 12.0)


def test_chained_splits_give_ids_a_b_and_ab():
    prims = room_6_by_4()
    before = detect_rooms(prims, ROLES)
    (room,) = before.rooms

    # Line 2 cuts the room at x = 3 (a is west); line 3 cuts S0_R0a along
    # y = 2, running east, so S0_R0aa is its north part.
    detection = apply_splits(before, (line((3, 0), (3, 4), row=2), line((0, 2), (3, 2), row=3)), prims, ROLES)

    aa, ab, b = detection.rooms
    assert [r.room_id for r in detection.rooms] == ["S0_R0aa", "S0_R0ab", "S0_R0b"]
    assert (aa.area_m2, ab.area_m2, b.area_m2) == (6.0, 6.0, 12.0)
    assert aa.footprint.equals(Polygon([(0, 2), (3, 2), (3, 4), (0, 4)]))
    assert [(c.row, c.side) for c in aa.cuts] == [(2, "a"), (3, "a")]
    assert [(c.row, c.side) for c in ab.cuts] == [(2, "a"), (3, "b")]
    assert [(c.row, c.side) for c in b.cuts] == [(2, "b")]
    assert aa.source_outline is room.outline and ab.source_outline is room.outline
    assert aa.cuts[1].line.tolist() == [[0.0, 2.0], [3.0, 2.0]]


def test_chained_halves_share_the_unsplit_grid():
    from leed_daylight.build.grid import sensor_points

    prims = room_6_by_4()
    (room,) = detect_rooms(prims, ROLES).rooms
    whole = sensor_points(room.outline, room.level_z, GRID, holes=room.holes)
    pieces = split(prims, line((3, 0), (3, 4), row=2), line((0, 2), (3, 2), row=3)).rooms

    masks = np.array([half_mask(piece, whole) for piece in pieces])

    assert masks.sum(axis=0).tolist() == [1] * len(whole)
    assert masks.sum(axis=1).tolist() == [24, 24, 48]


def test_a_half_gets_its_own_ceiling_height():
    prims = [
        slab("floor_a", 0, 0, 6, 4, 0.0),
        slab("ceiling_west", 0, 0, 3, 4, 3.0, "soffit", up=False),
        slab("ceiling_east", 3, 0, 6, 4, 2.5, "soffit", up=False),
    ]
    (room,) = detect_rooms(prims, ROLES).rooms
    assert room.ceiling_height_m == 2.5

    a, b = split(prims, line((3, 0), (3, 4))).rooms

    # The east ceiling only touches the west half along x = 3: no overlap.
    assert (a.ceiling_height_m, b.ceiling_height_m) == (3.0, 2.5)


def test_a_half_with_no_ceiling_over_it_has_none():
    prims = [slab("floor_a", 0, 0, 6, 4, 0.0), slab("ceiling_east", 3, 0, 6, 4, 3.0, "soffit", up=False)]

    a, b = split(prims, line((3, 0), (3, 4))).rooms

    assert (a.ceiling_height_m, b.ceiling_height_m) == (None, 3.0)


def test_a_half_keeps_the_floor_names_it_overlaps():
    # Two touching slabs make one room; the line at x = 4 leaves the west
    # slab wholly in half a and cuts the east slab.
    prims = [slab("floor_west", 0, 0, 3, 4, 0.0), slab("floor_east", 3, 0, 6, 4, 0.0)]
    (room,) = detect_rooms(prims, ROLES).rooms
    assert room.floor_names == ("floor_west", "floor_east")

    a, b = split(prims, line((4, 0), (4, 4))).rooms

    assert a.floor_names == ("floor_west", "floor_east")
    assert b.floor_names == ("floor_east",)


def test_a_floor_touching_a_half_only_along_the_line_is_not_its_floor():
    prims = [slab("floor_west", 0, 0, 3, 4, 0.0), slab("floor_east", 3, 0, 6, 4, 0.0)]

    a, b = split(prims, line((3, 0), (3, 4))).rooms

    assert (a.floor_names, b.floor_names) == (("floor_west",), ("floor_east",))


def test_a_half_takes_the_place_of_its_room_and_other_rooms_are_the_same_objects():
    prims = [
        slab("floor_a", 0, 0, 6, 4, 0.0), slab("floor_b", 10, 0, 14, 4, 0.0), slab("floor_c", 0, 0, 6, 4, 3.5),
    ]
    detection = detect_rooms(prims, ROLES)
    assert [r.room_id for r in detection.rooms] == ["S0_R0", "S0_R1", "S1_R0"]

    after = apply_splits(detection, (line((12, 0), (12, 4)),), prims, ROLES)

    assert [r.room_id for r in after.rooms] == ["S0_R0", "S0_R1a", "S0_R1b", "S1_R0"]
    assert after.rooms[0] is detection.rooms[0] and after.rooms[3] is detection.rooms[2]
    assert after.warnings is detection.warnings


def test_a_line_picks_its_storey_by_level():
    prims = [slab("floor_low", 0, 0, 6, 4, 0.0), slab("floor_high", 0, 0, 6, 4, 3.5)]

    after = split(prims, line((3, 0), (3, 4), level_z=3.52))

    assert [r.room_id for r in after.rooms] == ["S0_R0", "S1_R0a", "S1_R0b"]
    assert after.rooms[1].level_z == 3.5


def test_a_half_is_a_full_room_built_from_every_room_field():
    a, _ = split(room_6_by_4(), line((2, 0), (2, 4))).rooms

    names = [f.name for f in dataclasses.fields(SplitHalf)]
    assert names[: len(dataclasses.fields(Room))] == [f.name for f in dataclasses.fields(Room)]
    assert names[len(dataclasses.fields(Room)):] == ["source_outline", "source_holes", "cuts"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.room_id = "other"


def test_no_split_returns_the_same_detection_object():
    prims = room_6_by_4()
    detection = detect_rooms(prims, ROLES)

    assert apply_splits(detection, (), prims, ROLES) is detection
    assert apply_splits(detection, [], prims, ROLES) is detection
