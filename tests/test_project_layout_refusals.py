# tests/test_project_layout_refusals.py
"""Every split refusal of merge-split spec 5, seen end to end through
``scene_rooms`` with a splits.csv in the project folder, and the lines that
must be accepted (a grazed neighbour, a joint between panes, a mullion).

Every building here is invented: rooms drawn as round rectangles of floor,
glass as round rectangles. Messages are matched whole, so each one is seen
to name the splits.csv line, the room, the reason and what to change.
"""
from __future__ import annotations

import re

import pytest

from golden_prepare import golden_project
from leed_daylight.project import layout
from leed_daylight.project.layout import SplitHalf, scene_rooms
from leed_daylight.project.splits import SplitError
from leed_daylight.scene.primitives import Primitive
from leed_daylight.scene.storeys import detect_rooms

ROLES = {"slab": "floor", "soffit": "ceiling", "glass": "glazing", "wall": "wall"}
HEADER = "level_z,points,note\n"


def slab(name, x0, y0, x1, y1, z, modifier="slab", up=True):
    corners = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    if not up:
        corners = corners[::-1]
    return Primitive(modifier, "polygon", name, (), (), tuple(float(v) for c in corners for v in c), "test")


def pane_y(name, y, x0, x1, z0, z1):
    """A vertical pane in the plane y = const."""
    corners = [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
    return Primitive("glass", "polygon", name, (), (), tuple(float(v) for c in corners for v in c), "test")


def roof_pane(name, x0, y0, x1, y1, z):
    return slab(name, x0, y0, x1, y1, z, modifier="glass")


def room_6_by_4(z=0.0, tag="a"):
    """A 6 x 4 m room: floor at ``z``, ceiling 3 m above."""
    return [
        slab(f"floor_{tag}", 0, 0, 6, 4, z),
        slab(f"ceiling_{tag}", 0, 0, 6, 4, z + 3.0, "soffit", up=False),
    ]


def rooms_with(tmp_path, prims, rows: str):
    """``scene_rooms`` on ``prims`` with a splits.csv holding ``rows``."""
    (tmp_path / "splits.csv").write_text(HEADER + rows, encoding="utf-8", newline="\n")
    return scene_rooms(golden_project(tmp_path), prims, ROLES)


def refused(tmp_path, prims, rows: str, message: str) -> None:
    with pytest.raises(SplitError, match=f"^{re.escape(message)}$"):
        rooms_with(tmp_path, prims, rows)


# -- scene_rooms -----------------------------------------------------------------------


def test_scene_rooms_without_splits_csv_is_detect_rooms(tmp_path, monkeypatch):
    seen = []

    def detect(prims, roles):
        seen.append(detect_rooms(prims, roles))
        return seen[-1]

    monkeypatch.setattr(layout, "detect_rooms", detect)
    prims = room_6_by_4()

    detection = scene_rooms(golden_project(tmp_path), prims, ROLES)

    assert detection is seen[0]
    assert [r.room_id for r in detection.rooms] == ["S0_R0"]


def test_scene_rooms_applies_splits_csv(tmp_path):
    detection = rooms_with(tmp_path, room_6_by_4(), '0,"3 0; 3 4",an invented split\n')

    assert [r.room_id for r in detection.rooms] == ["S0_R0a", "S0_R0b"]
    assert all(isinstance(r, SplitHalf) for r in detection.rooms)


def test_a_header_only_splits_csv_changes_nothing(tmp_path):
    detection = rooms_with(tmp_path, room_6_by_4(), "")

    assert [(type(r).__name__, r.room_id) for r in detection.rooms] == [("Room", "S0_R0")]


# -- the line and its storey ---------------------------------------------------------------


def test_a_split_line_in_millimetres_is_refused_saying_metres(tmp_path):
    prims = [*room_6_by_4(0.0, "low"), *room_6_by_4(3.5, "high")]

    refused(tmp_path, prims, '3500,"3000 0; 3000 4000",copied from a drawing\n',
            "splits.csv line 2: level_z 3500.00 m matches no storey (storeys at 0.00 m, 3.50 m); "
            "level_z and points are metres, whatever scene.units")


def test_a_level_between_storeys_is_refused(tmp_path):
    prims = [*room_6_by_4(0.0, "low"), *room_6_by_4(3.5, "high")]

    refused(tmp_path, prims, '1.0,"3 0; 3 4",\n',
            "splits.csv line 2: level_z 1.00 m matches no storey (storeys at 0.00 m, 3.50 m); "
            "level_z and points are metres, whatever scene.units")


def test_a_line_in_a_scene_with_no_floor_is_refused(tmp_path):
    refused(tmp_path, [slab("ceiling_a", 0, 0, 6, 4, 3.0, "soffit", up=False)], '0,"3 0; 3 4",\n',
            "splits.csv line 2: level_z 0.00 m matches no storey (the scene has no floor); "
            "level_z and points are metres, whatever scene.units")


def test_a_self_crossing_line_is_refused(tmp_path):
    refused(tmp_path, room_6_by_4(), '0,"1 1; 5 3; 5 1; 1 3",\n',
            "splits.csv line 2: the line crosses or overlaps itself; a split line must be simple: "
            "remove the loop, or draw two lines")


def test_a_line_that_doubles_back_over_itself_is_refused(tmp_path):
    refused(tmp_path, room_6_by_4(), '0,"3 0; 3 4; 3 2",\n',
            "splits.csv line 2: the line crosses or overlaps itself; a split line must be simple: "
            "remove the loop, or draw two lines")


# -- crossing ----------------------------------------------------------------------------


def test_a_line_crossing_no_room_is_refused(tmp_path):
    refused(tmp_path, room_6_by_4(), '0,"10 0; 10 4",\n',
            "splits.csv line 2: the line crosses no room on storey 0 (level 0.00 m); a split line "
            "must cross exactly one room: check its points (metres, whatever scene.units)")


def test_a_line_crossing_two_rooms_is_refused(tmp_path):
    prims = [slab("floor_west", 0, 0, 6, 4, 0.0), slab("floor_east", 8, 0, 12, 4, 0.0)]

    refused(tmp_path, prims, '0,"1 2; 11 2",\n',
            "splits.csv line 2: the line crosses 2 rooms on storey 0: S0_R0 (centroid 3.00, 2.00), "
            "S0_R1 (centroid 10.00, 2.00); a split line must cross exactly one room: shorten it so "
            "that it stays inside one room")


def test_a_line_grazing_a_neighbours_corner_is_not_crossing_it(tmp_path):
    # The line cuts the south room at x = 4, touches the north room's
    # south-west corner (4, 5) and runs 1 m along its west edge.
    prims = [slab("floor_south", 0, 0, 6, 4, 0.0), slab("floor_north", 4, 5, 8, 9, 0.0)]
    north = detect_rooms(prims, ROLES).rooms[1]

    detection = rooms_with(tmp_path, prims, '0,"4 -1; 4 6",\n')

    assert [r.room_id for r in detection.rooms] == ["S0_R0a", "S0_R0b", "S0_R1"]
    assert (detection.rooms[0].area_m2, detection.rooms[1].area_m2) == (16.0, 8.0)
    assert detection.rooms[2].footprint.equals(north.footprint)


# -- end points, pieces, sides, size ---------------------------------------------------------


def test_an_end_point_inside_the_room_is_refused(tmp_path):
    refused(tmp_path, room_6_by_4(), '0,"3 0.02; 3 4",\n',
            "splits.csv line 2: its first point (3.00, 0.02) is 0.02 m inside room S0_R0 "
            "(centroid 3.00, 2.00); a split line must end on the room's outline or outside it: "
            "move that point onto the outline or past it")


def test_a_last_point_inside_the_room_is_refused(tmp_path):
    refused(tmp_path, room_6_by_4(), '0,"3 -1; 3 2",\n',
            "splits.csv line 2: its last point (3.00, 2.00) is 2.00 m inside room S0_R0 "
            "(centroid 3.00, 2.00); a split line must end on the room's outline or outside it: "
            "move that point onto the outline or past it")


def test_an_end_point_just_inside_is_snapped(tmp_path):
    detection = rooms_with(tmp_path, room_6_by_4(), '0,"3 0.005; 3 4",\n')

    assert detection.rooms[0].cuts[0].line.tolist() == [[3.0, 0.0], [3.0, 4.0]]


def test_a_cut_giving_three_pieces_is_refused(tmp_path):
    # A U-shaped room open to the north: a line across both legs makes
    # three pieces.
    prims = [
        slab("floor_leg_west", 0, 0, 2, 6, 0.0),
        slab("floor_base", 2, 0, 6, 2, 0.0),
        slab("floor_leg_east", 6, 0, 8, 6, 0.0),
    ]

    refused(tmp_path, prims, '0,"-1 4; 9 4",\n',
            "splits.csv line 2: the line cuts room S0_R0 (centroid 4.00, 2.50) into 3 piece(s); "
            "a split line must cut its room into exactly two: draw it so that it enters and leaves "
            "the room once, or use two lines")


def test_a_half_smaller_than_one_square_metre_is_refused(tmp_path):
    refused(tmp_path, room_6_by_4(), '0,"0.125 0; 0.125 4",\n',
            "splits.csv line 2: the line cuts room S0_R0 (centroid 3.00, 2.00) into halves of "
            "0.50 m2 and 23.50 m2; each half must be at least 1 m2: move the line")


def test_a_later_line_is_refused_naming_its_own_row(tmp_path):
    refused(tmp_path, room_6_by_4(), '0,"3 0; 3 4",\n\n0,"1 1; 1 4",\n',
            "splits.csv line 4: its first point (1.00, 1.00) is 1.00 m inside room S0_R0a "
            "(centroid 1.50, 2.00); a split line must end on the room's outline or outside it: "
            "move that point onto the outline or past it")


# -- glazing --------------------------------------------------------------------------------


def test_a_line_ending_in_the_middle_of_a_facade_pane_is_refused(tmp_path):
    prims = [*room_6_by_4(), pane_y("window_south", 0, 2, 4, 1, 2)]

    refused(tmp_path, prims, '0,"3 0; 3 4",\n',
            "splits.csv line 2: the line meets pane window_south of room S0_R0 (centroid 3.00, 2.00); "
            "a split must end between panes, so that each pane lights one half")


def test_a_line_ending_on_a_joint_between_two_facade_panes_is_accepted(tmp_path):
    prims = [*room_6_by_4(), pane_y("window_west", 0, 1, 3, 1, 2), pane_y("window_east", 0, 3, 5, 1, 2)]

    detection = rooms_with(tmp_path, prims, '0,"3 0; 3 4",\n')

    assert [r.room_id for r in detection.rooms] == ["S0_R0a", "S0_R0b"]


def test_a_line_ending_at_the_end_of_a_facade_pane_is_accepted(tmp_path):
    prims = [*room_6_by_4(), pane_y("window_south", 0, 1, 3, 1, 2)]

    assert len(rooms_with(tmp_path, prims, '0,"3 -1; 3 5",\n').rooms) == 2


def test_a_line_running_along_a_facade_pane_is_refused(tmp_path):
    prims = [*room_6_by_4(), pane_y("window_south", 0, 2, 4, 1, 2)]

    # The line enters along the facade (y = 0) from outside, then turns north.
    refused(tmp_path, prims, '0,"1 0; 3 0; 3 4",\n',
            "splits.csv line 2: the line meets pane window_south of room S0_R0 (centroid 3.00, 2.00); "
            "a split must end between panes, so that each pane lights one half")


@pytest.mark.parametrize("y", [-0.3, 0.3], ids=["outside_the_outline", "inside_the_outline"])
def test_set_back_facade_glass_is_met_by_the_extended_line(tmp_path, y):
    # Glass in the wall's thickness, 0.3 m off the outline: the line stops
    # on the outline, and is extended 0.6 m past it when checked.
    prims = [*room_6_by_4(), pane_y("window_south", y, 2, 4, 1, 2)]

    refused(tmp_path, prims, '0,"3 0; 3 4",\n',
            "splits.csv line 2: the line meets pane window_south of room S0_R0 (centroid 3.00, 2.00); "
            "a split must end between panes, so that each pane lights one half")


def test_facade_glass_beyond_the_reach_does_not_stop_a_line(tmp_path):
    prims = [*room_6_by_4(), pane_y("window_far", -1.0, 2, 4, 1, 2)]

    assert len(rooms_with(tmp_path, prims, '0,"3 0; 3 4",\n').rooms) == 2


def test_a_pane_on_another_storey_does_not_stop_a_line(tmp_path):
    prims = [*room_6_by_4(0.0, "low"), *room_6_by_4(3.5, "high"), pane_y("window_high", 0, 2, 4, 4.5, 5.5)]

    detection = rooms_with(tmp_path, prims, '0,"3 0; 3 4",\n')

    assert [r.room_id for r in detection.rooms] == ["S0_R0a", "S0_R0b", "S1_R0"]
    with pytest.raises(SplitError, match=r"^splits\.csv line 2: the line meets pane window_high of room S1_R0 "):
        rooms_with(tmp_path, prims, '3.5,"3 0; 3 4",\n')


def test_a_line_through_a_roof_pane_is_refused(tmp_path):
    prims = [*room_6_by_4(), roof_pane("skylight_a", 1, 1.5, 2, 2.5, 3.4)]

    refused(tmp_path, prims, '0,"1.5 0; 1.5 4",\n',
            "splits.csv line 2: the line meets pane skylight_a of room S0_R0 (centroid 3.00, 2.00); "
            "a split must end between panes, so that each pane lights one half")


def test_a_line_along_a_mullion_between_two_roof_panes_is_accepted(tmp_path):
    prims = [*room_6_by_4(), roof_pane("skylight_west", 1, 1.5, 2, 2.5, 3.4),
             roof_pane("skylight_east", 2, 1.5, 3, 2.5, 3.4)]

    detection = rooms_with(tmp_path, prims, '0,"2 0; 2 4",\n')

    assert [r.area_m2 for r in detection.rooms] == [8.0, 16.0]


def test_a_roof_pane_over_the_room_above_does_not_stop_a_line(tmp_path):
    # The skylight is over the upper room; the lower room's line passes
    # under it.
    prims = [
        *room_6_by_4(0.0, "low"), *room_6_by_4(3.5, "high"),
        roof_pane("skylight_a", 1, 1.5, 2, 2.5, 6.9),
    ]

    detection = rooms_with(tmp_path, prims, '0,"1.5 0; 1.5 4",\n')

    assert [r.room_id for r in detection.rooms] == ["S0_R0a", "S0_R0b", "S1_R0"]
    with pytest.raises(SplitError, match=r"^splits\.csv line 2: the line meets pane skylight_a of room S1_R0 "):
        rooms_with(tmp_path, prims, '3.5,"1.5 0; 1.5 4",\n')
