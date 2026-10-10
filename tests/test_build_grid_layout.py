# tests/test_build_grid_layout.py
"""Grids per space (merge-split spec 7): a merged space gets one grid over
all its pieces; a split half gets the original room's lattice, filtered by
the split line; every piece of a regularly occupied space needs a point.

Rooms are invented rectangles; the split half comes from ``apply_splits``
on an invented 6 x 4 m floor. Every number is round.
"""
from __future__ import annotations

import numpy as np
import pytest

from leed_daylight.build.grid import GridError, build_grids, piece_points, sensor_points
from leed_daylight.build.view_model import ViewModelError, view_grids
from leed_daylight.project.config import Grid
from leed_daylight.project.layout import apply_splits, half_mask, lay_out
from leed_daylight.project.spaces import Space
from leed_daylight.project.splits import SplitLine
from leed_daylight.scene.primitives import Primitive
from leed_daylight.scene.storeys import Room, detect_rooms

#: Over the 6 x 4 m floor this lattice runs x = 0.25, 0.75, ... 5.75.
GRID = Grid(spacing_m=0.5, work_plane_m=0.75, min_clearance_m=0.1)


def _room(x0, y0, x1, y1, room_id="S0_R0", outline=None):
    if outline is None:
        outline = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    return Room(
        room_id=room_id, storey=0, level_z=0.0, outline=outline,
        area_m2=float((x1 - x0) * (y1 - y0)), ceiling_height_m=3.0,
    )


def _space(name, merge_into=None, occupied=True):
    return Space(
        floor="L0", room=name, space_type="office", regularly_occupied=occupied, area_m2=10.0,
        centroid_x=0.0, centroid_y=0.0, level_z=0.0, view_preserving_shades=False,
        merge_into=merge_into, include={"base": True}, reason="", views={"base": True},
    )


def _halves(x):
    """The two halves of a 6 x 4 m floor cut by the line x = ``x``, north."""
    corners = [(0, 0, 0), (6, 0, 0), (6, 4, 0), (0, 4, 0)]
    prims = [Primitive("slab", "polygon", "floor_a", (), (), tuple(float(v) for c in corners for v in c), "test")]
    line = SplitLine(row=2, level_z=0.0, points=np.array([[x, -1.0], [x, 5.0]]), note="")
    return apply_splits(detect_rooms(prims, {"slab": "floor"}), (line,), prims, {"slab": "floor"}).rooms


def test_a_merged_space_gets_one_grid_of_its_pieces_points_in_piece_order():
    west, east = _room(0, 0, 4, 3, "S0_R1"), _room(10, 0, 12, 3, "S0_R0")
    spaces = [_space("Office West", "Office"), _space("Office")]
    matched = {"Office West": west, "Office": east}

    (grid,) = build_grids(spaces, matched, GRID, "base")

    expected = np.vstack([
        sensor_points(east.outline, 0.0, GRID),
        sensor_points(west.outline, 0.0, GRID),
    ])
    assert grid.room == "Office"
    assert np.array_equal(grid.points, expected)
    assert grid.room_area_m2 == 18.0
    assert (grid.storey, grid.level_z) == (0, 0.0)


def test_a_given_layout_is_used():
    west, east = _room(0, 0, 4, 3, "S0_R1"), _room(10, 0, 12, 3, "S0_R0")
    spaces = [_space("Office West", "Office"), _space("Office")]
    matched = {"Office West": west, "Office": east}
    layout = lay_out(spaces, matched)

    (grid,) = build_grids(spaces, matched, GRID, "base", layout=layout)

    assert np.array_equal(grid.points, np.vstack(piece_points(layout.space("Office"), GRID)))


def test_a_split_half_gets_the_original_rooms_lattice_filtered_by_the_line():
    a, b = _halves(2.1)
    layout = lay_out([_space("West"), _space("East")], {"West": a, "East": b})

    (west,) = piece_points(layout.space("West"), GRID)
    (east,) = piece_points(layout.space("East"), GRID)

    whole = sensor_points(a.source_outline, 0.0, GRID)
    assert np.array_equal(west, whole[half_mask(a, whole)])
    assert np.array_equal(east, whole[half_mask(b, whole)])
    assert len(west) + len(east) == len(whole) == 96
    # Not a lattice over the half: no clearance band at the split line, and
    # the columns keep the original room's x (0.25 ... 1.75, then 2.25 ...).
    assert sorted(set(np.round(west[:, 0], 9))) == [0.25, 0.75, 1.25, 1.75]
    assert sorted(set(np.round(east[:, 0], 9)))[0] == 2.25
    assert sorted(set(np.round(sensor_points(a.outline, 0.0, GRID)[:, 0], 9))) == [0.3, 0.8, 1.3, 1.8]


def test_a_split_space_grid_through_build_grids():
    a, b = _halves(2.1)
    spaces = [_space("West"), _space("East")]

    west, east = build_grids(spaces, {"West": a, "East": b}, GRID, "base")

    assert west.room_area_m2 == pytest.approx(8.4)
    assert len(west.points) == 32 and len(east.points) == 64


def test_view_grids_at_eye_height_on_a_merged_and_a_split_space():
    a, b = _halves(2.1)
    other = _room(10, 0, 12, 3, "S0_R1")
    spaces = [_space("East"), _space("West", "Other"), _space("Other")]
    matched = {"East": b, "West": a, "Other": other}

    east, merged = view_grids(spaces, matched, GRID, 1.2, "base")

    assert (east.room, merged.room) == ("East", "Other")
    assert np.allclose(merged.points[:, 2], 1.2)
    assert len(merged.points) == len(sensor_points(other.outline, 0.0, GRID)) + 32
    assert merged.room_area_m2 == pytest.approx(6.0 + 8.4)
    assert len(east.points) == 64


def test_a_piece_with_no_grid_point_is_refused_naming_it():
    narrow = _room(0, 0, 4, 0.15, "S0_R1")
    spaces = [_space("Office Strip", "Office"), _space("Office")]
    matched = {"Office Strip": narrow, "Office": _room(10, 0, 14, 3, "S0_R0")}

    with pytest.raises(GridError) as excinfo:
        build_grids(spaces, matched, GRID, "base")

    assert str(excinfo.value) == (
        "room 'Office': piece S0_R1 (row 'Office Strip', area 0.60 m2) yields no sensor point "
        "at 0.5 m spacing with 0.1 m clearance; every piece of a regularly occupied space "
        "needs at least one sensor point (LM-83-12 2.2.6: the grid extends across the entire space); "
        "clear its merge_into and untick it or mark it not regularly occupied, or check the room in the model"
    )


def test_a_piece_with_no_view_point_is_refused_naming_it():
    narrow = _room(0, 0, 4, 0.15, "S0_R1")
    spaces = [_space("Office Strip", "Office"), _space("Office")]
    matched = {"Office Strip": narrow, "Office": _room(10, 0, 14, 3, "S0_R0")}

    with pytest.raises(ViewModelError) as excinfo:
        view_grids(spaces, matched, GRID, 1.2, "base")

    assert str(excinfo.value) == (
        "room 'Office': piece S0_R1 (row 'Office Strip', area 0.60 m2) yields no view point "
        "at 0.5 m spacing; every piece of a space needs at least one view point; clear its merge_into "
        "and untick it, or check the room in the model"
    )


def test_a_one_piece_space_with_no_point_keeps_todays_message():
    spaces = [_space("Office")]

    with pytest.raises(GridError) as excinfo:
        build_grids(spaces, {"Office": _room(0, 0, 4, 0.15)}, GRID, "base")

    assert str(excinfo.value) == (
        "room 'Office' (area 0.60 m2) yields no sensor point at 0.5 m spacing with 0.1 m "
        "clearance; a regularly occupied space needs at least one sensor point"
    )


def test_a_bad_piece_outline_is_refused_naming_the_space_and_the_row():
    bad = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    spaces = [_space("Office West", "Office"), _space("Office")]
    matched = {"Office West": _room(0, 0, 4, 3, "S0_R1", outline=bad), "Office": _room(10, 0, 14, 3)}

    with pytest.raises(GridError, match=r"^room 'Office', piece row 'Office West': outline_xy: edge 1"):
        build_grids(spaces, matched, GRID, "base")


def test_a_space_not_regularly_occupied_gets_no_grid_whatever_its_pieces_say():
    spaces = [_space("Office West", "Office", occupied=True), _space("Office", occupied=False)]
    matched = {"Office West": _room(0, 0, 4, 3, "S0_R1"), "Office": _room(10, 0, 14, 3)}

    assert build_grids(spaces, matched, GRID, "base") == []


def test_a_piece_marked_not_regularly_occupied_still_contributes_grid_points():
    """Controller ruling (pre-flight F5b): the space row's
    ``regularly_occupied`` decides for every piece, including a piece row
    whose own row says not occupied (Task 4 warns but does not refuse it).
    That piece must still contribute grid points to the merged space's grid,
    not merely fail to break the merge."""
    west, east = _room(0, 0, 4, 3, "S0_R1"), _room(10, 0, 12, 3, "S0_R0")
    spaces = [_space("Office West", "Office", occupied=False), _space("Office")]
    matched = {"Office West": west, "Office": east}

    (grid,) = build_grids(spaces, matched, GRID, "base")

    west_points = sensor_points(west.outline, 0.0, GRID)
    assert len(west_points) > 0
    assert len(grid.points) == len(sensor_points(east.outline, 0.0, GRID)) + len(west_points)
