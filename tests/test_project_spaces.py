# tests/test_project_spaces.py
"""Task 3.4: spaces.csv -- rooms, cases, and matching by centroid and level.

Matching is by centroid AND level, never by a detected room's own sort-order
identifier (it renumbers whenever a room is added -- see
`tests/test_storeys.py`'s `test_room_ids_are_stable_and_unique` and the
carry-forward note in the plan). `test_the_module_never_reads_room_ids` pins
that down by reading the module's own source text.
"""
from __future__ import annotations

import csv
import inspect
import pathlib

import numpy as np
import pytest

from leed_daylight.project.config import ProjectError
from leed_daylight.project.spaces import (
    CENTROID_TOLERANCE_M,
    LEVEL_TOLERANCE_M,
    Space,
    case_names,
    included,
    load_spaces,
    match_to_rooms,
)
from leed_daylight.scene.primitives import Primitive
from leed_daylight.scene.storeys import Room, detect_rooms

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "spaces_example.csv"

HEADER = (
    "floor", "room", "space_type", "regularly_occupied", "area_m2",
    "centroid_x", "centroid_y", "level_z", "view_preserving_shades",
    "merge_into", "include_base", "include_variant", "reason",
)

SIMPLE_HEADER = (
    "floor", "room", "space_type", "regularly_occupied", "area_m2",
    "centroid_x", "centroid_y", "level_z", "view_preserving_shades",
    "merge_into", "include_base", "reason",
)


def _write(tmp_path: pathlib.Path, data_rows, header=SIMPLE_HEADER, name="spaces.csv") -> pathlib.Path:
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)
    return path


def _space(room, cx, cy, level_z, area=10.0) -> Space:
    """A hand-built Space, for tests that exercise matching, not parsing."""
    return Space(
        floor="L0",
        room=room,
        space_type="office",
        regularly_occupied=True,
        area_m2=area,
        centroid_x=cx,
        centroid_y=cy,
        level_z=level_z,
        view_preserving_shades=False,
        merge_into=None,
        include={"base": True},
        reason="",
    )


def _room(room_id, storey, level_z, cx, cy, half=1.0) -> Room:
    """A hand-built square Room, so its outline centroid is exactly (cx, cy)."""
    outline = np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ])
    return Room(
        room_id=room_id,
        storey=storey,
        level_z=level_z,
        outline=outline,
        area_m2=float((2 * half) ** 2),
        ceiling_height_m=None,
    )


def _slab(name, x0, y0, x1, y1, z) -> Primitive:
    corners = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    return Primitive("slab", "polygon", name, (), (),
                     tuple(v for c in corners for v in c), "test")


# ---------------------------------------------------------------------------
# Happy path: loading the fixture
# ---------------------------------------------------------------------------

def test_the_example_file_loads():
    spaces = load_spaces(FIXTURE)

    assert len(spaces) == 5
    assert case_names(spaces) == ("base", "variant")


# ---------------------------------------------------------------------------
# Matching by centroid and level
# ---------------------------------------------------------------------------

def test_a_space_matches_its_room_by_centroid():
    space = _space("Room North", 2.0, 1.5, 0.0)
    room = _room("S0_R0", 0, 0.0, 2.2, 1.6)

    matched = match_to_rooms([space], [room])

    assert matched["Room North"] is room


def test_a_space_just_outside_the_tolerance_is_refused():
    assert CENTROID_TOLERANCE_M == 0.5
    space = _space("Room North", 0.0, 0.0, 0.0)
    room = _room("S0_R0", 0, 0.0, 0.6, 0.0)

    with pytest.raises(ProjectError, match=r"Room North.*0\.60"):
        match_to_rooms([space], [room])


def test_two_candidate_rooms_are_refused_not_guessed():
    space = _space("Room North", 0.0, 0.0, 0.0)
    room_a = _room("S0_R0", 0, 0.0, 0.1, 0.0)
    room_b = _room("S0_R1", 0, 0.0, -0.2, 0.1)

    with pytest.raises(ProjectError) as excinfo:
        match_to_rooms([space], [room_a, room_b])
    message = str(excinfo.value)
    assert "Room North" in message
    assert "0.10" in message
    assert "-0.20" in message


def test_renumbered_room_ids_do_not_break_matching():
    # Two separate floors: a 3x3 room and a 2x2 room, same storey.
    prims_before = [
        _slab("big", 5, 0, 8, 3, 0.0),
        _slab("small", 0, 0, 2, 2, 0.0),
    ]
    rooms_before = detect_rooms(prims_before, {"slab": "floor"}).rooms

    # A third, even larger floor is added: every sort-order id shifts.
    prims_after = prims_before + [_slab("biggest", 20, 0, 26, 7, 0.0)]
    rooms_after = detect_rooms(prims_after, {"slab": "floor"}).rooms

    ids_before = {r.room_id for r in rooms_before}
    ids_after = {r.room_id for r in rooms_after}
    assert ids_before != ids_after, "the test must actually exercise a renumbering"

    space_big = _space("Room Big", 6.5, 1.5, 0.0)
    space_small = _space("Room Small", 1.0, 1.0, 0.0)

    matched_before = match_to_rooms([space_big, space_small], rooms_before)
    matched_after = match_to_rooms([space_big, space_small], rooms_after)

    assert matched_before["Room Big"].area_m2 == pytest.approx(9.0)
    assert matched_after["Room Big"].area_m2 == pytest.approx(9.0)
    assert matched_before["Room Small"].area_m2 == pytest.approx(4.0)
    assert matched_after["Room Small"].area_m2 == pytest.approx(4.0)
    assert matched_before["Room Big"].room_id != matched_after["Room Big"].room_id


def test_stacked_rooms_match_by_level():
    lower = _room("S0_R0", 0, 0.0, 2.0, 1.5)
    upper = _room("S1_R0", 1, 3.5, 2.0, 1.5)
    space_lower = _space("Room Ground", 2.0, 1.5, 0.0)
    space_upper = _space("Room Upper", 2.0, 1.5, 3.5)

    matched = match_to_rooms([space_lower, space_upper], [lower, upper])

    assert matched["Room Ground"] is lower
    assert matched["Room Upper"] is upper


def test_a_space_on_no_storey_is_refused():
    assert LEVEL_TOLERANCE_M == 0.05
    space = _space("Room North", 2.0, 1.5, 7.0)
    room = _room("S0_R0", 0, 0.0, 2.0, 1.5)

    with pytest.raises(ProjectError, match=r"Room North.*7\.00"):
        match_to_rooms([space], [room])


def test_two_spaces_claiming_one_room_are_refused():
    # Both spaces are within tolerance of the SAME single room, and no
    # other room is nearby: the mapping must stay injective.
    room = _room("S0_R0", 0, 0.0, 2.2, 1.5)
    space_a = _space("Room North", 2.0, 1.5, 0.0)
    space_b = _space("Room South", 2.1, 1.6, 0.0)

    with pytest.raises(ProjectError) as excinfo:
        match_to_rooms([space_a, space_b], [room])
    message = str(excinfo.value)
    assert "Room North" in message
    assert "Room South" in message


def test_the_module_never_reads_room_ids():
    from leed_daylight.project import spaces as spaces_module
    source = inspect.getsource(spaces_module)
    assert "room_id" not in source


# ---------------------------------------------------------------------------
# case_names / included
# ---------------------------------------------------------------------------

def test_included_filters_by_case():
    spaces = load_spaces(FIXTURE)

    base_rooms = {s.room for s in included(spaces, "base")}
    variant_rooms = {s.room for s in included(spaces, "variant")}

    assert base_rooms == {"Room North", "Room South", "Open Office", "Meeting 1", "Store"}
    assert variant_rooms == {"Room North", "Room South", "Meeting 1", "Store"}
    assert variant_rooms < base_rooms


def test_an_unknown_case_is_refused():
    spaces = load_spaces(FIXTURE)

    with pytest.raises(ProjectError, match=r"nope.*base.*variant"):
        included(spaces, "nope")


# ---------------------------------------------------------------------------
# view_preserving_shades
# ---------------------------------------------------------------------------

def test_the_view_preserving_shades_column_loads(tmp_path):
    path = _write(tmp_path, [
        ("L0", "Room North", "office", "yes", "10", "1.0", "1.0", "0.0", "yes", "", "1", ""),
        ("L0", "Room South", "office", "yes", "10", "5.0", "1.0", "0.0", "no", "", "1", ""),
    ])

    spaces = load_spaces(path)

    by_room = {s.room: s.view_preserving_shades for s in spaces}
    assert by_room == {"Room North": True, "Room South": False}


# The missing-column column under test, plus the row cell to drop for it, so
# a single parametrised test covers both required columns the brief calls
# out (the plan's "eleven" tests undercounted these two).
_REQUIRED_COLUMN_ROWS = {
    "view_preserving_shades": ("L0", "Room North", "office", "yes", "10", "1.0", "1.0", "0.0", "", "1", ""),
    "level_z": ("L0", "Room North", "office", "yes", "10", "1.0", "1.0", "no", "", "1", ""),
}


@pytest.mark.parametrize("column", ["view_preserving_shades", "level_z"])
def test_a_missing_required_column_is_refused(tmp_path, column):
    header = tuple(c for c in SIMPLE_HEADER if c != column)
    path = _write(tmp_path, [_REQUIRED_COLUMN_ROWS[column]], header=header)

    with pytest.raises(ProjectError, match=column):
        load_spaces(path)


# ---------------------------------------------------------------------------
# Refusals settled by the loading rules
# ---------------------------------------------------------------------------

def test_a_scenario_must_be_a_subset_of_the_base_case(tmp_path):
    path = _write(tmp_path, [
        ("L0", "Room North", "office", "yes", "10", "1.0", "1.0", "0.0", "no", "", "0", "1", ""),
    ], header=HEADER)

    with pytest.raises(ProjectError, match=r"include_variant.*include_base"):
        load_spaces(path)


def test_a_zero_area_space_is_refused(tmp_path):
    path = _write(tmp_path, [
        ("L0", "Room North", "office", "yes", "0", "1.0", "1.0", "0.0", "no", "", "1", ""),
    ])

    with pytest.raises(ProjectError, match="Room North"):
        load_spaces(path)


def test_a_merge_target_that_does_not_exist_is_refused(tmp_path):
    path = _write(tmp_path, [
        ("L0", "Room North", "office", "yes", "10", "1.0", "1.0", "0.0", "no", "Nowhere", "1", ""),
    ])

    with pytest.raises(ProjectError, match="Nowhere"):
        load_spaces(path)


def test_a_merge_cycle_is_refused(tmp_path):
    path = _write(tmp_path, [
        ("L0", "Room A", "office", "yes", "10", "1.0", "1.0", "0.0", "no", "Room B", "1", ""),
        ("L0", "Room B", "office", "yes", "10", "5.0", "1.0", "0.0", "no", "Room A", "1", ""),
    ])

    with pytest.raises(ProjectError, match=r"Room A.*Room B|Room B.*Room A"):
        load_spaces(path)


# ---------------------------------------------------------------------------
# Excel-saved files: cp1252 bytes and ';' as the delimiter
# ---------------------------------------------------------------------------


def test_a_non_utf8_spaces_file_is_refused_by_name(tmp_path):
    text = FIXTURE.read_text().replace("Room North", "Caf\xe9")
    path = tmp_path / "spaces.csv"
    path.write_bytes(text.encode("cp1252"))
    with pytest.raises(ProjectError) as excinfo:
        load_spaces(path)
    message = str(excinfo.value)
    assert "spaces.csv" in message and "UTF-8" in message and "line 2" in message


def test_a_semicolon_delimited_spaces_file_is_refused_as_such(tmp_path):
    path = tmp_path / "spaces.csv"
    path.write_text(FIXTURE.read_text().replace(",", ";"), newline="\n")
    with pytest.raises(ProjectError) as excinfo:
        load_spaces(path)
    message = str(excinfo.value)
    assert "';'" in message and "comma" in message
    assert "missing column" not in message



# ---------------------------------------------------------------------------
# Case names (security review H2): a case name becomes a folder name
# ---------------------------------------------------------------------------

_ONE_ROW = ("L0", "Room North", "office", "yes", "10", "1.0", "1.0", "0.0", "no", "", "1", "")


@pytest.mark.parametrize("column", [
    "include_../../../../victim_dir", "include_a/b", "include_a\\b", "include_.", "include_..",
    "include_C:", "include_a\x01b",
])
def test_a_case_name_that_is_not_a_safe_folder_name_is_refused(tmp_path, column):
    header = tuple(column if c == "include_base" else c for c in SIMPLE_HEADER)
    path = _write(tmp_path, [_ONE_ROW], header=header)

    with pytest.raises(ProjectError, match=r"is not a safe case name; it becomes a folder name"):
        load_spaces(path)


def test_a_views_case_name_that_is_not_a_safe_folder_name_is_refused(tmp_path):
    header = SIMPLE_HEADER[:-1] + ("views_../outside",) + SIMPLE_HEADER[-1:]
    path = _write(tmp_path, [_ONE_ROW[:-1] + ("1",) + _ONE_ROW[-1:]], header=header)

    with pytest.raises(ProjectError, match=r"views case '\.\./outside' is not a safe case name"):
        load_spaces(path)


@pytest.mark.parametrize("name", ["Base_2-a", "base case", "café", ".hidden", "a.b"])
def test_every_other_case_name_loads_as_it_is(tmp_path, name):
    header = tuple(f"include_{name}" if c == "include_base" else c for c in SIMPLE_HEADER)
    path = _write(tmp_path, [_ONE_ROW], header=header)

    assert case_names(load_spaces(path)) == (name,)


def test_a_row_of_empty_cells_is_skipped_like_a_blank_line(tmp_path):
    # Excel exports a cleared row as ",,,,,"; it is not a data row.
    path = _write(tmp_path, [
        ("L0", "Room A", "office", "yes", "10", "1", "1", "0", "no", "", "1", ""),
        ("",) * len(SIMPLE_HEADER),
        ("L0", "Room B", "office", "yes", "10", "5", "1", "0", "no", "", "1", ""),
    ])

    assert [s.room for s in load_spaces(path)] == ["Room A", "Room B"]


def test_a_row_with_data_but_no_room_is_still_refused(tmp_path):
    path = _write(tmp_path, [("L0", "", "office", "yes", "10", "1", "1", "0", "no", "", "1", "")])

    with pytest.raises(ProjectError, match="room is blank"):
        load_spaces(path)
