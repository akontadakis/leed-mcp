# tests/test_build_groups_pieces.py
"""Window groups per piece (merge-split spec 7): a space of several pieces
gets one group per piece, facing and kind, with ``_p<n>`` ids; a tie
inside one space goes to the earlier piece.

Every room is an invented rectangle and every pane an invented rectangle
of glass; every number is round.
"""
from __future__ import annotations

import numpy as np
import pytest

from leed_daylight.build.groups import GroupError, WindowGroup, window_groups
from leed_daylight.project.layout import apply_splits
from leed_daylight.project.splits import SplitLine
from leed_daylight.scene.primitives import Primitive
from leed_daylight.scene.storeys import Room, detect_rooms

ROLES = {"glass": "glazing", "slab": "floor"}


def room(x0, y0, x1, y1, room_id):
    outline = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    return Room(
        room_id=room_id, storey=0, level_z=0.0, outline=outline,
        area_m2=float((x1 - x0) * (y1 - y0)), ceiling_height_m=3.0,
    )


def pane(name, corners):
    return Primitive("glass", "polygon", name, (), (), tuple(float(v) for c in corners for v in c), "test")


def south_window(name, x0, x1, y=0.0):
    """A 1 m tall window in the wall y = ``y``, between x0 and x1."""
    return pane(name, [(x0, y, 1.0), (x1, y, 1.0), (x1, y, 2.0), (x0, y, 2.0)])


def skylight(name, x0, y0, x1, y1, z=3.0):
    return pane(name, [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)])


def test_two_pieces_of_one_space_get_a_group_each_with_p_ids():
    west, east = room(0, 0, 4, 3, "S0_R1"), room(5, 0, 9, 3, "S0_R0")
    prims = [south_window("w_west", 1, 3), south_window("w_east", 6, 8)]

    groups, warnings = window_groups(
        prims, ROLES, {"Office 1": (east, west)}, all_rooms=[east, west],
    )

    assert warnings == ()
    assert [(g.group_id, g.room, g.piece) for g in groups] == [
        ("Office_1_p1_180", "Office 1", 0),
        ("Office_1_p2_180", "Office 1", 1),
    ]
    assert [[o.name for o in g.openings] for g in groups] == [["w_east"], ["w_west"]]


@pytest.mark.parametrize("value", ["room", "tuple"])
def test_a_one_piece_space_keeps_todays_id(value):
    only = room(0, 0, 4, 3, "S0_R0")
    rooms = {"Office 1": only if value == "room" else (only,)}

    (group,), _ = window_groups([south_window("w", 1, 3)], ROLES, rooms, all_rooms=[only])

    assert (group.group_id, group.piece) == ("Office_1_180", 0)
    assert isinstance(group, WindowGroup)


def test_a_pane_equidistant_from_two_pieces_of_one_space_goes_to_the_first_piece():
    left, right = room(0, 0, 2, 2, "S0_R0"), room(3, 0, 5, 2, "S0_R1")
    # In the 1 m gap between the two rooms, 0.5 m from each.
    between = pane("between", [(2.5, 0.9, 1.0), (2.5, 1.1, 1.0), (2.5, 1.1, 2.0), (2.5, 0.9, 2.0)])

    (group,), _ = window_groups([between], ROLES, {"Office": (right, left)}, all_rooms=[left, right])

    assert group.piece == 0
    assert group.group_id.startswith("Office_p1_")


def test_the_same_tie_between_two_spaces_is_refused_as_today():
    left, right = room(0, 0, 2, 2, "S0_R0"), room(3, 0, 5, 2, "S0_R1")
    between = pane("between", [(2.5, 0.9, 1.0), (2.5, 1.1, 1.0), (2.5, 1.1, 2.0), (2.5, 0.9, 2.0)])

    with pytest.raises(GroupError) as excinfo:
        window_groups([between], ROLES, {"Left": left, "Right": right}, all_rooms=[left, right])

    assert str(excinfo.value) == "between: matches more than one room within 0.6 m: Left, Right"


def _halves():
    """The two halves of a 10 x 10 m floor cut along x = 5: a (west), b."""
    corners = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]
    prims = [Primitive("slab", "polygon", "floor_a", (), (), tuple(float(v) for c in corners for v in c), "test")]
    line = SplitLine(row=2, level_z=0.0, points=np.array([[5.0, -1.0], [5.0, 11.0]]), note="")
    return apply_splits(detect_rooms(prims, ROLES), (line,), prims, ROLES).rooms


def test_a_roof_pane_on_the_shared_edge_of_two_pieces_of_one_space_goes_to_the_first():
    a, b = _halves()
    # Its plan centroid (5, 4.5) lies on the edge the two halves share.
    prims = [skylight("on_the_line", 4.0, 4.0, 6.0, 5.0)]

    (group,), _ = window_groups(prims, ROLES, {"Hall": (b, a)}, all_rooms=[a, b])

    assert (group.group_id, group.piece) == ("Hall_p1_sky", 0)


def test_a_roof_pane_on_the_shared_edge_of_two_spaces_is_refused_as_today():
    a, b = _halves()
    prims = [skylight("on_the_line", 4.0, 4.0, 6.0, 5.0)]

    with pytest.raises(GroupError, match=r"on_the_line: roof glass lies on the boundary of more than one room: East, West"):
        window_groups(prims, ROLES, {"West": a, "East": b}, all_rooms=[a, b])


def test_a_p_suffix_id_that_collides_with_another_space_is_refused():
    west, east, other = room(0, 0, 4, 3, "S0_R0"), room(5, 0, 9, 3, "S0_R1"), room(10, 0, 14, 3, "S0_R2")
    prims = [south_window("w_west", 1, 3), south_window("w_east", 6, 8), south_window("w_other", 11, 13)]

    with pytest.raises(GroupError) as excinfo:
        window_groups(
            prims, ROLES, {"Office": (west, east), "Office p1": other}, all_rooms=[west, east, other],
        )

    assert str(excinfo.value) == (
        "window group id 'Office_p1_180' comes from space 'Office' and from space 'Office p1'; "
        "rename one so window-group ids stay distinct"
    )


def test_the_halves_of_a_split_room_match_their_facade_panes_by_the_window_rule():
    a, b = _halves()
    prims = [south_window("w_west", 1, 3), south_window("w_east", 7, 9), south_window("w_far_east", 9.5, 10)]

    groups, _ = window_groups(prims, ROLES, {"West": a, "East": b}, all_rooms=[a, b])

    assert [(g.group_id, [o.name for o in g.openings]) for g in groups] == [
        ("East_180", ["w_east", "w_far_east"]),
        ("West_180", ["w_west"]),
    ]
