# tests/test_build_grid.py
"""Tests for leed_daylight.build.grid (T3.11): the sensor grid.

Every fixture is a fictional building with round, invented numbers
(phase3-global.md #5): a 4 x 3 m room called "room_a", an L-shaped room
called "room_l", a 0.4 x 0.4 m room too small for a single sensor point.
The spacing (0.60 m), clearance (0.305 m) and work-plane height (0.762 m)
in the four-by-three fixture are the brief's own derived test values, not
anything read off a real project.
"""
from __future__ import annotations

import math
import types

import numpy as np
import pytest

from leed_daylight.build.grid import GridError, RoomGrid, build_grids, sensor_points
from leed_daylight.project.config import Grid
from leed_daylight.project.spaces import Space
from leed_daylight.scene.storeys import Room

# -- fixtures ---------------------------------------------------------------

# A 4 x 3 m room, spacing 0.60, clearance 0.305, work plane 0.762. The
# lattice is centred in the clearance band on each axis: x has 4 - 0.61 =
# 3.39 m of band, 5 whole steps (3.00 m) plus 0.39 m left over, so the
# first column sits at 0.305 + 0.195 = 0.50 and there are 6 columns (0.50
# .. 3.50); y has 2.39 m of band, 3 whole steps plus 0.59 m, so the first
# row sits at 0.305 + 0.295 = 0.60 and there are 4 rows (0.60 .. 2.40).
# 6 x 4 = 24 points.
FOUR_BY_THREE = np.array([(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)])
GRID_STANDARD = Grid(spacing_m=0.60, work_plane_m=0.762, min_clearance_m=0.305)

# An L-shaped room: a 4 x 4 square with its top-right 2 x 2 corner removed,
# so the notch is the square x > 2, y > 2.
L_SHAPE = np.array(
    [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 2.0), (2.0, 4.0), (0.0, 4.0)]
)
GRID_FOR_L = Grid(spacing_m=0.50, work_plane_m=0.762, min_clearance_m=0.10)

# A 0.4 x 0.4 m room: too small to hold a single point at the standard
# spacing and clearance.
TINY_ROOM = np.array([(0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.0, 0.4)])


def _room(outline, level_z=0.0, area_m2=None, room_id="S0_R0", storey=0) -> Room:
    if area_m2 is None:
        # Shoelace area of a simple polygon -- good enough for a fixture.
        x = outline[:, 0]
        y = outline[:, 1]
        area_m2 = float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)
    return Room(
        room_id=room_id,
        storey=storey,
        level_z=level_z,
        outline=outline,
        area_m2=area_m2,
        ceiling_height_m=None,
    )


def _space(room, regularly_occupied=True, include=None, area_m2=10.0, level_z=0.0) -> Space:
    if include is None:
        include = {"base": True}
    return Space(
        floor="L0",
        room=room,
        space_type="office",
        regularly_occupied=regularly_occupied,
        area_m2=area_m2,
        centroid_x=0.0,
        centroid_y=0.0,
        level_z=level_z,
        view_preserving_shades=False,
        merge_into=None,
        include=include,
        reason="",
    )


# -- required shape (brief table) --------------------------------------------


def test_a_four_by_three_room_yields_twenty_four_points():
    points = sensor_points(FOUR_BY_THREE, 0.0, GRID_STANDARD)
    assert points.shape == (24, 3)
    assert sorted(set(np.round(points[:, 0], 9))) == pytest.approx(
        [0.5, 1.1, 1.7, 2.3, 2.9, 3.5]
    )
    assert sorted(set(np.round(points[:, 1], 9))) == pytest.approx([0.6, 1.2, 1.8, 2.4])


def test_a_six_by_six_room_keeps_its_outer_ring():
    # 6 - 0.61 = 5.39 m of band: 8 whole steps plus 0.59 m, so the first
    # row sits at 0.305 + 0.295 = 0.60 and there are 9 rows (0.60 .. 5.40).
    # The old half-cell rule put its first candidate at 0.30, inside the
    # 0.305 clearance, and so dropped the whole outer ring (64 points).
    outline = np.array([(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)])
    points = sensor_points(outline, 0.0, GRID_STANDARD)
    assert points.shape == (81, 3)


@pytest.mark.parametrize("outline", [FOUR_BY_THREE, L_SHAPE])
def test_first_and_last_rows_sit_within_one_step_of_the_clearance(outline):
    c, s = GRID_STANDARD.min_clearance_m, GRID_STANDARD.spacing_m
    points = sensor_points(outline, 0.0, GRID_STANDARD)
    lo = outline.min(axis=0)
    hi = outline.max(axis=0)
    for axis in (0, 1):
        first = points[:, axis].min() - lo[axis]
        last = hi[axis] - points[:, axis].max()
        assert c - 1e-9 <= first < c + s
        assert c - 1e-9 <= last < c + s


def test_the_lattice_is_symmetric_in_a_rectangle():
    points = sensor_points(FOUR_BY_THREE, 0.0, GRID_STANDARD)
    assert points[:, 0].min() == pytest.approx(4.0 - points[:, 0].max())
    assert points[:, 1].min() == pytest.approx(3.0 - points[:, 1].max())


def test_the_point_count_does_not_change_with_an_awkward_offset():
    offset = np.array([1000.1, 2000.3])
    for outline in (FOUR_BY_THREE, L_SHAPE):
        original = sensor_points(outline, 0.0, GRID_STANDARD)
        moved = sensor_points(outline + offset, 0.0, GRID_STANDARD)
        assert moved.shape == original.shape
        assert moved[:, :2] == pytest.approx(original[:, :2] + offset)


def test_every_point_clears_the_boundary():
    from shapely.geometry import Point, Polygon

    poly = Polygon(FOUR_BY_THREE)
    points = sensor_points(FOUR_BY_THREE, 0.0, GRID_STANDARD)
    for x, y, _ in points:
        assert poly.exterior.distance(Point(x, y)) >= GRID_STANDARD.min_clearance_m - 1e-9


def test_the_pitch_is_exactly_the_spacing():
    points = sensor_points(FOUR_BY_THREE, 0.0, GRID_STANDARD)
    xs = sorted(set(round(float(x), 9) for x in points[:, 0]))
    diffs = [b - a for a, b in zip(xs, xs[1:])]
    assert diffs
    for d in diffs:
        assert d == pytest.approx(0.60)


def test_the_work_plane_height_is_applied():
    points = sensor_points(FOUR_BY_THREE, 3.5, GRID_STANDARD)
    assert points.shape[0] > 0
    for z in points[:, 2]:
        assert z == pytest.approx(4.262)


def test_the_grid_is_reproducible():
    first = sensor_points(FOUR_BY_THREE, 0.0, GRID_STANDARD)
    second = sensor_points(FOUR_BY_THREE, 0.0, GRID_STANDARD)
    assert np.array_equal(first, second)


def test_the_grid_does_not_move_when_the_building_moves():
    offset = np.array([1000.0, 1000.0])
    moved_outline = FOUR_BY_THREE + offset
    original = sensor_points(FOUR_BY_THREE, 0.0, GRID_STANDARD)
    moved = sensor_points(moved_outline, 0.0, GRID_STANDARD)
    assert moved.shape == original.shape
    assert moved[:, :2] == pytest.approx(original[:, :2] + offset)
    assert moved[:, 2] == pytest.approx(original[:, 2])


def test_a_room_too_small_for_a_point_is_refused():
    room = _room(TINY_ROOM, area_m2=0.16)
    space = _space("room_a")
    matched = {"room_a": room}
    with pytest.raises(GridError, match="room_a") as exc_info:
        build_grids([space], matched, GRID_STANDARD, "base")
    assert "0.16" in str(exc_info.value)


def test_an_l_shaped_room_has_no_point_in_its_notch():
    points = sensor_points(L_SHAPE, 0.0, GRID_FOR_L)
    assert points.shape[0] > 0
    for x, y, _ in points:
        assert not (x > 2.0 and y > 2.0)


def test_only_included_regularly_occupied_spaces_get_a_grid():
    included_room = _room(FOUR_BY_THREE)
    excluded_room = _room(FOUR_BY_THREE + np.array([100.0, 0.0]))
    not_occupied_room = _room(FOUR_BY_THREE + np.array([200.0, 0.0]))

    spaces = [
        _space("room_included", include={"base": True}),
        _space("room_excluded", include={"base": False}),
        _space("room_not_occupied", regularly_occupied=False, include={"base": True}),
    ]
    matched = {
        "room_included": included_room,
        "room_excluded": excluded_room,
        "room_not_occupied": not_occupied_room,
    }
    grids = build_grids(spaces, matched, GRID_STANDARD, "base")
    assert [g.room for g in grids] == ["room_included"]


def test_the_room_area_is_the_room_s_not_the_grid_s():
    room = _room(FOUR_BY_THREE, area_m2=12.0)
    space = _space("room_a")
    matched = {"room_a": room}
    [grid_result] = build_grids([space], matched, GRID_STANDARD, "base")
    assert grid_result.room_area_m2 == pytest.approx(12.0)
    assert grid_result.room_area_m2 != pytest.approx(15 * 0.36)
    assert grid_result.cell_area_m2 == pytest.approx(0.36)


# -- controller rulings: defence in depth, ordering, RoomGrid shape --------


def test_spacing_above_0_60_is_refused_here_too():
    bad_grid = Grid(spacing_m=0.61, work_plane_m=0.762, min_clearance_m=0.305)
    with pytest.raises(GridError, match="600 mm"):
        sensor_points(FOUR_BY_THREE, 0.0, bad_grid)


def test_the_room_grid_carries_the_matched_room_s_storey_and_level_z():
    # I3 (controller ruling P26, additive): so phase 4 can place and file a
    # grid without reopening scene.storeys for the matched Room again.
    room = _room(FOUR_BY_THREE, storey=2, level_z=6.0)
    space = _space("room_a", level_z=6.0)
    matched = {"room_a": room}
    [grid_result] = build_grids([space], matched, GRID_STANDARD, "base")
    assert grid_result.storey == 2
    assert grid_result.level_z == pytest.approx(6.0)


def test_spacing_zero_is_refused():
    bad_grid = Grid(spacing_m=0.0, work_plane_m=0.762, min_clearance_m=0.305)
    with pytest.raises(GridError, match="spacing_m"):
        sensor_points(FOUR_BY_THREE, 0.0, bad_grid)


def test_spacing_negative_is_refused():
    bad_grid = Grid(spacing_m=-0.1, work_plane_m=0.762, min_clearance_m=0.305)
    with pytest.raises(GridError, match="spacing_m"):
        sensor_points(FOUR_BY_THREE, 0.0, bad_grid)


def test_points_are_ordered_row_major_by_y_then_x():
    points = sensor_points(FOUR_BY_THREE, 0.0, GRID_STANDARD)
    ys = [round(float(y), 9) for y in points[:, 1]]
    # y is non-decreasing across the whole array.
    assert ys == sorted(ys)
    # within a y-row, x is strictly increasing.
    xs = points[:, 0]
    row_starts = [i for i in range(len(ys)) if i == 0 or ys[i] != ys[i - 1]]
    for start, end in zip(row_starts, row_starts[1:] + [len(ys)]):
        row_xs = list(xs[start:end])
        assert row_xs == sorted(row_xs)


def test_a_space_missing_from_matched_is_refused():
    space = _space("room_a")
    with pytest.raises(GridError, match="room_a"):
        build_grids([space], {}, GRID_STANDARD, "base")


def test_the_room_grid_dataclass_fields():
    room = _room(FOUR_BY_THREE, area_m2=12.0)
    space = _space("room_a")
    [result] = build_grids([space], {"room_a": room}, GRID_STANDARD, "base")
    assert isinstance(result, RoomGrid)
    assert result.spacing_m == pytest.approx(0.60)


# -- hostile-input pass -------------------------------------------------


def test_outline_ragged_is_refused():
    with pytest.raises(GridError, match="outline_xy"):
        sensor_points([[0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0]], 0.0, GRID_STANDARD)


def test_outline_non_numeric_is_refused():
    with pytest.raises(GridError, match="outline_xy"):
        sensor_points([["a", "b"], ["c", "d"], ["e", "f"]], 0.0, GRID_STANDARD)


def test_outline_with_nan_is_refused():
    outline = np.array([(0.0, 0.0), (4.0, 0.0), (4.0, math.nan), (0.0, 3.0)])
    with pytest.raises(GridError, match="outline_xy"):
        sensor_points(outline, 0.0, GRID_STANDARD)


def test_outline_with_inf_is_refused():
    outline = np.array([(0.0, 0.0), (4.0, 0.0), (4.0, math.inf), (0.0, 3.0)])
    with pytest.raises(GridError, match="outline_xy"):
        sensor_points(outline, 0.0, GRID_STANDARD)


def test_outline_wrong_shape_is_refused():
    # (n, 3) instead of (n, 2): a caller passing a 3-D primitive by mistake.
    outline = np.array([(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 3.0, 0.0)])
    with pytest.raises(GridError, match="outline_xy"):
        sensor_points(outline, 0.0, GRID_STANDARD)


def test_outline_empty_is_refused():
    with pytest.raises(GridError, match="outline_xy"):
        sensor_points(np.zeros((0, 2)), 0.0, GRID_STANDARD)


def test_outline_too_few_vertices_is_refused():
    with pytest.raises(GridError, match="outline_xy"):
        sensor_points(np.array([(0.0, 0.0), (1.0, 0.0)]), 0.0, GRID_STANDARD)


def test_outline_with_zero_length_edge_is_refused():
    outline = np.array(
        [(0.0, 0.0), (0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    )
    with pytest.raises(GridError, match="outline_xy"):
        sensor_points(outline, 0.0, GRID_STANDARD)


def test_outline_degenerate_zero_area_is_refused():
    outline = np.array([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)])
    with pytest.raises(GridError, match="degenerate"):
        sensor_points(outline, 0.0, GRID_STANDARD)


def test_level_z_nan_is_refused():
    with pytest.raises(GridError, match="level_z"):
        sensor_points(FOUR_BY_THREE, math.nan, GRID_STANDARD)


def test_level_z_inf_is_refused():
    with pytest.raises(GridError, match="level_z"):
        sensor_points(FOUR_BY_THREE, math.inf, GRID_STANDARD)


def test_grid_missing_field_is_refused():
    bad_grid = types.SimpleNamespace(spacing_m=0.60, work_plane_m=0.762)
    with pytest.raises(GridError, match="min_clearance_m"):
        sensor_points(FOUR_BY_THREE, 0.0, bad_grid)


def test_grid_field_non_finite_is_refused():
    bad_grid = types.SimpleNamespace(
        spacing_m=0.60, work_plane_m=math.nan, min_clearance_m=0.305
    )
    with pytest.raises(GridError, match="work_plane_m"):
        sensor_points(FOUR_BY_THREE, 0.0, bad_grid)


def test_clearance_boundary_point_is_kept_when_exactly_equal():
    # A 0.6 x 0.6 m room: its clearance band is empty (0.6 - 2 x 0.3 = 0),
    # so the one candidate is its centre (0.3, 0.3), exactly 0.3 m from
    # every side. min_clearance_m == 0.3 exactly: kept, since the rule is
    # "at least" the clearance (inclusive).
    outline = np.array([(0.0, 0.0), (0.6, 0.0), (0.6, 0.6), (0.0, 0.6)])
    grid = Grid(spacing_m=0.60, work_plane_m=0.0, min_clearance_m=0.30)
    points = sensor_points(outline, 0.0, grid)
    assert points.shape == (1, 3)


def test_matched_not_a_mapping_is_refused():
    space = _space("room_a")
    room = _room(FOUR_BY_THREE)
    with pytest.raises(GridError, match="matched"):
        build_grids([space], [("room_a", room)], GRID_STANDARD, "base")


# A 6 x 6 m room round a 2 x 2 m atrium in its middle.
RING_OUTER = np.array([(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)])
RING_HOLE = np.array([(2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)])


def test_no_sensor_sits_in_or_near_an_atrium_hole():
    from shapely.geometry import Point, Polygon

    points = sensor_points(RING_OUTER, 0.0, GRID_STANDARD, holes=(RING_HOLE,))
    hole = Polygon(RING_HOLE)
    assert points.shape[0] > 0
    for x, y, _ in points:
        assert not hole.contains(Point(x, y))
        assert hole.exterior.distance(Point(x, y)) >= GRID_STANDARD.min_clearance_m - 1e-9
    solid = sensor_points(RING_OUTER, 0.0, GRID_STANDARD)
    assert points.shape[0] < solid.shape[0]


def test_build_grids_uses_the_room_s_holes():
    room = Room(
        room_id="S0_R0", storey=0, level_z=0.0, outline=RING_OUTER,
        area_m2=32.0, ceiling_height_m=None, holes=(RING_HOLE,),
    )
    [grid] = build_grids([_space("room_a")], {"room_a": room}, GRID_STANDARD, "base")
    expected = sensor_points(RING_OUTER, 0.0, GRID_STANDARD, holes=(RING_HOLE,))
    assert np.array_equal(grid.points, expected)


def test_a_malformed_hole_is_refused():
    with pytest.raises(GridError, match="hole"):
        sensor_points(RING_OUTER, 0.0, GRID_STANDARD, holes=(np.zeros((2, 2)),))
