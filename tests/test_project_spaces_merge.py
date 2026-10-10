# tests/test_project_spaces_merge.py
"""merge_into in spaces.csv (merge-split spec 3.2, 4): piece rows and space
rows, the tick and floor rules, the piece-row warnings, linked zones, and
the refusal hint of match_to_rooms.

Every room name, area and coordinate here is invented and round.
"""
from __future__ import annotations

import numpy as np
import pytest

from leed_daylight.build.grid import build_grids
from leed_daylight.project.config import Grid, ProjectError
from leed_daylight.project.spaces import (
    check_linked_zones,
    included,
    load_spaces,
    match_to_rooms,
    piece_row_warnings,
    space_of,
    space_rows,
    views_included,
)
from leed_daylight.scene.storeys import Room

HEADER = (
    "floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,"
    "view_preserving_shades,merge_into,include_base,include_variant,views_base,reason"
)
GRID = Grid(spacing_m=0.5, work_plane_m=0.75, min_clearance_m=0.1)


def _write(tmp_path, *rows):
    path = tmp_path / "spaces.csv"
    path.write_text(HEADER + "\n" + "".join(r + "\n" for r in rows), encoding="utf-8", newline="\n")
    return path


def _room(x0, y0, x1, y1, storey=0, level_z=0.0):
    outline = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    return Room(
        room_id=f"S{storey}_R{int(x0)}", storey=storey, level_z=level_z, outline=outline,
        area_m2=float((x1 - x0) * (y1 - y0)), ceiling_height_m=3.0,
    )


#: Office 1 is one space of two pieces: its own row and Office 1 West,
#: which merges into it. Meeting is an ordinary space.
MERGED = (
    "L0,Office 1 West,office,yes,8,2.0,2.0,0.0,no,Office 1,1,1,1,",
    "L0,Meeting,meeting,yes,12,20.0,2.0,0.0,no,,1,0,0,",
    "L0,Office 1,office,yes,12,7.0,2.0,0.0,no,,1,1,1,",
)


def test_included_returns_space_rows_only(tmp_path):
    spaces = load_spaces(_write(tmp_path, *MERGED))

    assert [s.room for s in included(spaces, "base")] == ["Meeting", "Office 1"]
    assert [s.room for s in included(spaces, "variant")] == ["Office 1"]


def test_views_included_returns_space_rows_only(tmp_path):
    spaces = load_spaces(_write(tmp_path, *MERGED))

    assert [s.room for s in views_included(spaces, "base")] == ["Office 1"]


def test_space_rows_and_space_of(tmp_path):
    spaces = load_spaces(_write(tmp_path, *MERGED))

    assert [s.room for s in space_rows(spaces)] == ["Meeting", "Office 1"]
    assert space_of(spaces) == {"Office 1 West": "Office 1", "Meeting": "Meeting", "Office 1": "Office 1"}


def test_a_merge_chain_makes_both_rows_pieces_of_the_last(tmp_path):
    spaces = load_spaces(_write(
        tmp_path,
        "L0,Room A,office,yes,8,2.0,2.0,0.0,no,Room B,1,0,0,",
        "L0,Room B,office,yes,8,7.0,2.0,0.0,no,Room C,1,0,0,",
        "L0,Room C,office,yes,8,12.0,2.0,0.0,no,,1,0,0,",
    ))

    assert space_of(spaces) == {"Room A": "Room C", "Room B": "Room C", "Room C": "Room C"}
    assert [s.room for s in space_rows(spaces)] == ["Room C"]
    assert [s.room for s in included(spaces, "base")] == ["Room C"]


def test_a_piece_row_with_a_different_tick_is_refused_naming_both_rows_and_the_column(tmp_path):
    path = _write(
        tmp_path,
        "L0,Room A,office,yes,8,2.0,2.0,0.0,no,Room C,1,1,0,",
        "L0,Room B,office,yes,8,20.0,2.0,0.0,no,,1,0,0,",
        "L0,Room C,office,yes,8,12.0,2.0,0.0,no,,1,0,0,",
    )

    with pytest.raises(ProjectError) as excinfo:
        load_spaces(path)

    assert str(excinfo.value) == (
        "spaces.csv: space 'Room A' (line 2) is a piece of 'Room C' (line 4) but its "
        "include_variant differs (1, not 0); a piece row must have the same floor and "
        "include_/views_ ticks as its space row"
    )


def test_a_piece_row_with_a_different_views_tick_is_refused(tmp_path):
    path = _write(
        tmp_path,
        "L0,Room A,office,yes,8,2.0,2.0,0.0,no,Room C,1,0,0,",
        "L0,Room C,office,yes,8,12.0,2.0,0.0,no,,1,0,1,",
    )

    with pytest.raises(ProjectError, match=r"'Room A' \(line 2\).*'Room C' \(line 3\).*views_base differs \(0, not 1\)"):
        load_spaces(path)


def test_a_piece_row_with_a_different_floor_label_is_refused(tmp_path):
    path = _write(
        tmp_path,
        "L1,Room A,office,yes,8,2.0,2.0,0.0,no,Room C,1,1,0,",
        "L0,Room C,office,yes,8,12.0,2.0,0.0,no,,1,0,0,",
    )

    with pytest.raises(ProjectError) as excinfo:
        load_spaces(path)

    # The floor is checked first, before the ticks (which differ too).
    assert str(excinfo.value) == (
        "spaces.csv: space 'Room A' (line 2) is a piece of 'Room C' (line 3) but its "
        "floor differs ('L1', not 'L0'); a piece row must have the same floor and "
        "include_/views_ ticks as its space row"
    )


def test_a_piece_row_value_that_differs_is_a_warning(tmp_path):
    spaces = load_spaces(_write(
        tmp_path,
        "L0,Room A,store,yes,8,2.0,2.0,0.0,yes,Room C,1,0,0,",
        "L0,Room C,office,yes,8,12.0,2.0,0.0,no,,1,0,0,",
    ))

    assert piece_row_warnings(spaces) == (
        "space 'Room A' is a piece of 'Room C': its space_type 'store' differs from the "
        "space row's 'office'; the space row's value is used",
        "space 'Room A' is a piece of 'Room C': its view_preserving_shades 'yes' differs "
        "from the space row's 'no'; the space row's value is used",
    )


def test_no_merge_into_gives_no_warning(tmp_path):
    spaces = load_spaces(_write(tmp_path, "L0,Room C,office,yes,8,12.0,2.0,0.0,no,,1,0,0,"))

    assert piece_row_warnings(spaces) == ()


def test_a_piece_row_marked_not_occupied_warns_and_is_not_refused(tmp_path):
    """Review Focus 3: the per-row views rule would refuse a piece row that
    is ticked in views_base (its space row's tick) while marked not
    regularly occupied; the piece row only warns, and the space row's value
    decides."""
    spaces = load_spaces(_write(
        tmp_path,
        "L0,Room A,office,no,8,2.0,2.0,0.0,no,Room C,1,0,1,",
        "L0,Room C,office,yes,8,12.0,2.0,0.0,no,,1,0,1,",
    ))

    assert piece_row_warnings(spaces) == (
        "space 'Room A' is a piece of 'Room C': its regularly_occupied 'no' differs from "
        "the space row's 'yes'; the space row's value is used",
    )
    assert [s.room for s in views_included(spaces, "base")] == ["Room C"]
    matched = {"Room A": _room(0, 0, 4, 4), "Room C": _room(10, 0, 14, 4)}
    (grid,) = build_grids(spaces, matched, GRID, "base")
    assert grid.room == "Room C"


def test_a_space_row_marked_not_occupied_and_ticked_in_views_is_still_refused(tmp_path):
    path = _write(
        tmp_path,
        "L0,Room A,office,yes,8,2.0,2.0,0.0,no,Room C,1,0,1,",
        "L0,Room C,office,no,8,12.0,2.0,0.0,no,,1,0,1,",
    )

    with pytest.raises(ProjectError, match=r"line 3: space 'Room C': ticked in 'views_base'"):
        load_spaces(path)


def test_a_linked_zone_naming_a_piece_row_is_refused(tmp_path):
    spaces = load_spaces(_write(tmp_path, *MERGED))

    with pytest.raises(ProjectError) as excinfo:
        check_linked_zones(spaces, {1: ("Meeting", "Office 1 West")})

    assert str(excinfo.value) == (
        "performance.linked_zones storey 1: 'Office 1 West' is a piece of space 'Office 1' "
        "(merge_into); name the space row 'Office 1'"
    )


def test_a_linked_zone_naming_space_rows_or_unknown_names_passes(tmp_path):
    spaces = load_spaces(_write(tmp_path, *MERGED))

    # An unknown name is left to engine.coefficients.sensors_in_scope, as today.
    check_linked_zones(spaces, {"1": ("Meeting", "Office 1", "Elsewhere")})
    check_linked_zones(spaces, {})


def test_match_to_rooms_appends_the_hint_to_its_refusal(tmp_path):
    spaces = load_spaces(_write(tmp_path, "L0,Room C,office,yes,8,12.0,2.0,0.0,no,,1,0,0,"))
    rooms = [_room(0, 0, 4, 4)]

    with pytest.raises(ProjectError) as excinfo:
        match_to_rooms(spaces, rooms, hint="; look again")

    assert str(excinfo.value) == (
        "spaces.csv: space 'Room C': nearest room on its level is 10.00 m away, outside "
        "the 0.50 m tolerance; look again"
    )
    with pytest.raises(ProjectError) as excinfo:
        match_to_rooms(spaces, rooms)
    assert str(excinfo.value).endswith("outside the 0.50 m tolerance")
