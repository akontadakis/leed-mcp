# tests/test_project_layout.py
"""The space layout (merge-split spec 6): one ``LaidOutSpace`` per space
row, its pieces in spaces.csv order, its area and origin.

Rooms are invented rectangles; a split half comes from ``apply_splits`` on
an invented 6 x 4 m floor. Every number is round.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from leed_daylight.project.layout import (
    LaidOutSpace,
    LayoutError,
    SpaceLayout,
    SplitHalf,
    apply_splits,
    lay_out,
)
from leed_daylight.project.spaces import Space
from leed_daylight.project.splits import SplitLine
from leed_daylight.scene.primitives import Primitive
from leed_daylight.scene.storeys import Room, detect_rooms

ROLES = {"slab": "floor"}


def _room(x0, y0, x1, y1, storey=0, level_z=0.0, room_id="S0_R0"):
    outline = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    return Room(
        room_id=room_id, storey=storey, level_z=level_z, outline=outline,
        area_m2=float((x1 - x0) * (y1 - y0)), ceiling_height_m=3.0,
    )


def _space(name, merge_into=None, level_z=0.0):
    return Space(
        floor="L0", room=name, space_type="office", regularly_occupied=True, area_m2=10.0,
        centroid_x=0.0, centroid_y=0.0, level_z=level_z, view_preserving_shades=False,
        merge_into=merge_into, include={"base": True}, reason="",
    )


def _halves():
    """The two halves of a 6 x 4 m floor cut along x = 2: areas 8 and 16."""
    corners = [(0, 0, 0), (6, 0, 0), (6, 4, 0), (0, 4, 0)]
    prims = [Primitive("slab", "polygon", "floor_a", (), (), tuple(float(v) for c in corners for v in c), "test")]
    line = SplitLine(row=2, level_z=0.0, points=np.array([[2.0, 0.0], [2.0, 4.0]]), note="")
    detection = apply_splits(detect_rooms(prims, ROLES), (line,), prims, ROLES)
    return detection.rooms


def test_a_space_with_no_merge_has_its_own_room_as_its_one_piece():
    room = _room(0, 0, 4, 3)
    layout = lay_out([_space("Room North")], {"Room North": room})

    assert isinstance(layout, SpaceLayout)
    (space,) = layout.spaces
    assert isinstance(space, LaidOutSpace)
    assert space.name == "Room North"
    assert space.pieces[0] is room and len(space.pieces) == 1
    assert space.origin == "detected"
    assert space.area_m2 == 12.0
    assert (space.storey, space.level_z) == (0, 0.0)
    assert space.grid_source == (None,)
    assert space.rows == ("Room North",)
    assert layout.space("Room North") is space
    assert layout.pieces_by_space() == {"Room North": (room,)}


def test_pieces_follow_the_space_row_then_spaces_csv_order():
    rooms = {name: _room(10 * i, 0, 10 * i + 4, 2, room_id=f"S0_R{i}") for i, name in enumerate("ABCD")}
    spaces = [_space("A", "C"), _space("B"), _space("C"), _space("D", "C")]

    layout = lay_out(spaces, rooms)

    assert [s.name for s in layout.spaces] == ["B", "C"]
    merged = layout.space("C")
    assert [p.room_id for p in merged.pieces] == ["S0_R2", "S0_R0", "S0_R3"]
    assert merged.rows == ("C", "A", "D")
    assert merged.area_m2 == 24.0
    assert merged.origin == "merged"
    assert merged.grid_source == (None, None, None)


def test_a_chain_puts_every_row_on_the_last():
    rooms = {name: _room(10 * i, 0, 10 * i + 4, 2, room_id=f"S0_R{i}") for i, name in enumerate("ABC")}
    spaces = [_space("A", "B"), _space("B", "C"), _space("C")]

    (space,) = lay_out(spaces, rooms).spaces

    assert space.rows == ("C", "A", "B")


def test_a_split_half_is_a_split_space_with_its_grid_source():
    a, b = _halves()
    layout = lay_out([_space("West"), _space("East")], {"West": a, "East": b})

    west, east = layout.spaces
    assert (west.origin, east.origin) == ("split", "split")
    assert west.pieces[0] is a and west.area_m2 == 8.0
    outline, holes = west.grid_source[0]
    assert outline is a.source_outline and holes == a.source_holes
    assert isinstance(a, SplitHalf)


def test_a_split_half_merged_with_a_room_is_split_plus_merged():
    a, b = _halves()
    other = _room(10, 0, 14, 4, room_id="S0_R1")
    layout = lay_out(
        [_space("East"), _space("West", "Other"), _space("Other")],
        {"East": b, "West": a, "Other": other},
    )

    merged = layout.space("Other")
    assert merged.origin == "split+merged"
    assert [p is q for p, q in zip(merged.pieces, (other, a), strict=True)] == [True, True]
    assert merged.rows == ("Other", "West")
    assert merged.grid_source[0] is None
    assert merged.grid_source[1][0] is a.source_outline
    assert merged.area_m2 == 24.0


def test_a_row_missing_from_matched_is_refused():
    with pytest.raises(LayoutError) as excinfo:
        lay_out([_space("A", "B"), _space("B")], {"B": _room(0, 0, 4, 2)})

    assert str(excinfo.value) == "space 'A' has no matched room"


def test_pieces_on_two_storeys_are_refused():
    """Review Focus 4: merging across storeys is out of scope, and the
    floor labels alone cannot catch it."""
    rooms = {"A": _room(0, 0, 4, 2, storey=1, level_z=3.5), "C": _room(0, 0, 4, 2)}

    with pytest.raises(LayoutError) as excinfo:
        lay_out([_space("A", "C", level_z=3.5), _space("C")], rooms)

    assert str(excinfo.value) == (
        "space 'A' is a piece of 'C' (merge_into), but its room is on storey 1 (level 3.50 m) "
        "and the space row's on storey 0 (level 0.00 m); the pieces of a space must be on "
        "one storey: clear merge_into for 'A'"
    )


def test_an_unknown_space_name_is_refused():
    layout = lay_out([_space("A")], {"A": _room(0, 0, 4, 2)})

    with pytest.raises(LayoutError, match=r"no space 'B' in the layout; the spaces are 'A'"):
        layout.space("B")


def test_the_layout_classes_are_frozen():
    layout = lay_out([_space("A")], {"A": _room(0, 0, 4, 2)})

    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.spaces[0].name = "B"
