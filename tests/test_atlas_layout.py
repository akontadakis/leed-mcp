# tests/test_atlas_layout.py
"""The floor atlas on split and merged spaces (atlas spec 8, merge-split
spec 7): split lines dashed, a space of several pieces filled piece by piece
and labelled once, on its largest piece; the bundle's rooms are the space
rows; the room table shows the rows' summed area; the storey panel lists the
layout sentence, with the blind clause only for a space the atlas shows
Option 1 results of.
A project with neither gives the same storey panels as before.

The projects are those of ``tests/merge_split_folder.py`` (runs on fake
engines) and ``tests/deliver_folder.py``; every number is invented and round.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import re

import numpy as np
import pytest

import leed_daylight.drivers.run as run_module
from atlas_folder import atlas_inputs
from deliver_folder import N_OCCUPIED, Option1Fake, build_two_storey_project
from leed_daylight.deliver.atlas import page
from leed_daylight.deliver.atlas.colours import ROOM_FILL_RGB
from leed_daylight.deliver.figures.spec import rgb_hex
from merge_split_folder import (
    CASE, HEADER, build_combined_project, build_merge_project, build_split_project, prepared_model,
)

MERGED_NOTE = (
    "Room North is two detected rooms treated as one space; each keeps its own grid, and blinds "
    "respond to the whole space."
)
MERGED_NOTE_NO_BLINDS = "Room North is two detected rooms treated as one space; each keeps its own grid."
#: Controller ruling, pre-flight F2b: a merged space excluded from the case
#: (so it has no grid) states neither the grid clause nor the blind clause.
MERGED_NOTE_NO_GRID = "Room North is two detected rooms treated as one space."
SPLIT_NOTES = (
    "Room West was split from a detected room by line 2 of splits.csv.",
    "Room Strip was split from a detected room by line 2 of splits.csv.",
)

#: The merge project with its space rows the other way round: Room East
#: (16 m2) is the space row and Room North (24 m2) its piece, so the label
#: leaves the space row's centroid (9, 2) for the larger piece's (3, 2).
EAST_FIRST_SPACES_CSV = HEADER + (
    "L0,Room East,office,yes,16,9.0,2.0,0.0,no,,1,1,\n"
    "L0,Room North,office,yes,24,3.0,2.0,0.0,no,Room East,1,1,\n"
)

#: The merge project with Room North excluded from the case (F2b): still
#: laid out and drawn (grey and hatched), but with no grid. A piece row must
#: share its space row's include/views ticks, so both are excluded.
EXCLUDED_MERGE_SPACES_CSV = HEADER + (
    "L0,Room North,office,yes,24,3.0,2.0,0.0,no,,0,1,invented: not part of this case\n"
    "L0,Room East,office,yes,16,9.0,2.0,0.0,no,Room North,0,1,invented: not part of this case\n"
)

#: The combined project with the split+merged space Room East (and its piece
#: Room Strip) ticked in the case but not regularly occupied (review I1):
#: included, so the room table says "included", but with no grid.
UNOCCUPIED_MERGE_SPACES_CSV = HEADER + (
    "L0,Room West,office,yes,18,2.25,2.0,0.0,no,,1,1,\n"
    "L0,Room East,office,no,16,9.0,2.0,0.0,no,,1,0,invented: not regularly occupied\n"
    "L0,Room Strip,office,no,6,5.25,2.0,0.0,no,Room East,1,0,invented: not regularly occupied\n"
)
UNOCCUPIED_MERGE_NOTE = (
    "Room East is two rooms treated as one space, one or more of them split from a detected room by "
    "line 2 of splits.csv."
)

#: sha256 of the storey panels (every plan SVG and the room table of each
#: storey) of the invented two-storey project, all three options run on the
#: fake engines, taken before this change (merge-split plan Task 11).
PANELS_BEFORE = "433bead7fc416f3d3e39455f5c70224c115ecd9c4c8faa455b7bacd354f822fe"


def _texts(svg: str) -> list[str]:
    return re.findall(r"<text[^>]*>([^<]*)</text>", svg)


def _panels(case, bundle) -> str:
    svgs = page.plan_svgs(case, bundle, {})
    titles = {"model": "Model"} | {layer.key: layer.title for layer, _ in bundle.layers}
    return "".join(page._storey_panel(case, bundle, si, label, svgs[label], titles)
                   for si, label in enumerate(bundle.storeys))


def _page(tmp_path, case, bundle) -> str:
    path = tmp_path / "atlas.html"
    page.write_page(path, case, bundle, {})
    return path.read_text(encoding="utf-8")


def _run_option1(folder) -> None:
    run_module.run_folder(folder, CASE, engine=Option1Fake(prepared_model(folder)), out=io.StringIO())


class _GridlessRoomFake(Option1Fake):
    """``Option1Fake``, and a window group of a room with no grid (included
    but not regularly occupied) reaches no sensor point."""

    def _of(self, group):
        return self._by_room.get(group.room, ((), np.zeros((0, N_OCCUPIED))))

    def direct(self, group, hours):
        index, sky = self._of(group)
        return index, np.zeros_like(sky)

    def sky(self, group, hours, open_schedule):
        index, sky = self._of(group)
        return index, sky, None


# -- plans -------------------------------------------------------------------


def test_a_split_space_plan_draws_the_split_line_dashed_and_each_name_once(tmp_path, write_example, write_wea):
    folder = build_split_project(tmp_path, write_example, write_wea)
    _run_option1(folder)
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    svgs = page.plan_svgs(case, bundle, {})["L0"]

    assert list(svgs) == ["model", "option1.da_pct", "option1.sda_pass", "option1.ase_hours", "option1.ase_exceeds"]
    for key, svg in svgs.items():
        # One line, x = 4.5 m from y = 0 to 4 m, shared by the two halves: drawn once.
        assert svg.count("stroke-dasharray") == 1, key
        assert _texts(svg).count("Room West") == 1, key
        assert _texts(svg).count("Room Strip") == 1, key


def test_a_merged_space_fills_both_pieces_and_is_labelled_once(tmp_path, write_example, write_wea):
    folder = build_merge_project(tmp_path, write_example, write_wea)
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    svg = page.plan_svgs(case, bundle, {})["L0"]["model"]

    assert svg.count(f'fill="{rgb_hex(ROOM_FILL_RGB)}"') == 2
    assert _texts(svg).count("Room North") == 1
    assert "Room East" not in _texts(svg)
    assert "stroke-dasharray" not in svg


# -- the bundle --------------------------------------------------------------


def test_the_bundle_has_no_piece_row_and_labels_a_space_on_its_largest_piece(tmp_path, write_example, write_wea):
    folder = build_merge_project(tmp_path, write_example, write_wea)
    (folder / "spaces.csv").write_text(EAST_FIRST_SPACES_CSV)
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    assert [(r.name, r.storey, r.x, r.y, r.z, r.included) for r in bundle.rooms] == [
        ("Room East", 0, 3.0, 2.0, 0.0, True),
    ]
    assert set(bundle.grid.room) == {"Room East"}
    assert page.payload(bundle)["rooms"] == ["Room East"]
    svg = page.plan_svgs(case, bundle, {})["L0"]["model"]
    assert _texts(svg).count("Room East") == 1
    assert "Room North" not in _texts(svg)


# -- the room table and the storey panel --------------------------------------


def test_the_room_table_shows_the_rows_summed_area(tmp_path, write_example, write_wea):
    folder = build_merge_project(tmp_path, write_example, write_wea)
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    table = page._room_table(case, bundle, "L0")

    rows = re.findall(r"<tr><td>([^<]*)</td><td class=\"num\">([^<]*)</td>", table)
    assert rows == [("Room North", "40.0")]  # 24 + 16 m2


def test_the_storey_panel_lists_the_sentence_with_blinds_when_option1_is_shown(tmp_path, write_example, write_wea):
    folder = build_merge_project(tmp_path, write_example, write_wea)
    _run_option1(folder)
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")
    assert list(case.options) == ["option1"]

    html = _page(tmp_path, case, bundle)

    assert f'</table><ul class="layout-notes"><li>{MERGED_NOTE}</li></ul></section>' in html


def test_without_option1_the_sentence_says_nothing_of_blinds(tmp_path, write_example, write_wea):
    folder = build_merge_project(tmp_path, write_example, write_wea)
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")
    assert dict(case.options) == {}

    html = _page(tmp_path, case, bundle)

    assert f'<ul class="layout-notes"><li>{MERGED_NOTE_NO_BLINDS}</li></ul>' in html
    assert "blinds respond" not in html


def test_a_space_option1_did_not_analyse_gets_no_blind_clause(tmp_path, write_example, write_wea):
    folder = build_merge_project(tmp_path, write_example, write_wea)
    _run_option1(folder)
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")
    option1 = case.options["option1"]
    # As if Option 1 had analysed no space of this storey (the space left out of the case).
    unanalysed = dataclasses.replace(option1, results={**option1.results, "spaces": []})

    notes = page._layout_notes(dataclasses.replace(case, options={"option1": unanalysed}), bundle, "L0")

    assert notes == f'<ul class="layout-notes"><li>{MERGED_NOTE_NO_BLINDS}</li></ul>'


def test_the_storey_panel_lists_each_split_space_in_table_order(tmp_path, write_example, write_wea):
    folder = build_split_project(tmp_path, write_example, write_wea)
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    html = _page(tmp_path, case, bundle)

    items = "".join(f"<li>{note}</li>" for note in SPLIT_NOTES)
    assert f'<ul class="layout-notes">{items}</ul>' in html


# -- controller ruling (pre-flight F2b): no grid, no grid or blind clause ----


def test_a_merged_space_excluded_from_the_case_gets_no_grid_or_blind_clause(tmp_path, write_example, write_wea):
    """A merged space that is not analysed in this case (excluded, so it
    has no grid -- shown "excluded" in the room table) gets the bare
    sentence, with neither the grid nor the blind clause, whether or not
    Option 1 is shown."""
    folder = build_merge_project(tmp_path, write_example, write_wea)
    (folder / "spaces.csv").write_text(EXCLUDED_MERGE_SPACES_CSV)
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    assert [r.included for r in bundle.rooms if r.name == "Room North"] == [False]
    assert dict(case.layout_notes_no_grid) == {"Room North": MERGED_NOTE_NO_GRID}

    notes = page._layout_notes(case, bundle, "L0")

    assert notes == f'<ul class="layout-notes"><li>{MERGED_NOTE_NO_GRID}</li></ul>'


@pytest.mark.parametrize("with_option1", [False, True])
def test_a_merged_space_included_but_not_occupied_gets_no_grid_or_blind_clause(
    tmp_path, write_example, write_wea, with_option1,
):
    """Review I1: a merged space ticked in the case but not regularly
    occupied is "included" in the room table yet has no grid point; its
    sentence states neither the grid nor the blind clause, whether or not
    Option 1 is shown. The gridded split space keeps its sentence."""
    folder = build_combined_project(tmp_path, write_example, write_wea)
    (folder / "spaces.csv").write_text(UNOCCUPIED_MERGE_SPACES_CSV)
    if with_option1:
        engine = _GridlessRoomFake(prepared_model(folder))
        run_module.run_folder(folder, CASE, engine=engine, out=io.StringIO())
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")
    assert list(case.options) == (["option1"] if with_option1 else [])

    assert [r.included for r in bundle.rooms if r.name == "Room East"] == [True]
    assert "Room East" not in set(bundle.grid.room)
    assert "Room West" in set(bundle.grid.room)
    table = page._room_table(case, bundle, "L0")
    assert "<td>Room East</td><td class=\"num\">22.0</td><td class=\"num\">0</td><td>included</td>" in table

    html = _page(tmp_path, case, bundle)

    items = (
        "<li>Room West was split from a detected room by line 2 of splits.csv.</li>"
        f"<li>{UNOCCUPIED_MERGE_NOTE}</li>"
    )
    assert f'<ul class="layout-notes">{items}</ul>' in html
    assert "keeps its own grid" not in html
    assert "blinds respond" not in html


# -- a project with neither ----------------------------------------------------


def test_a_project_without_split_or_merge_gives_the_same_panels_as_before(tmp_path, write_example, write_wea):
    folder = build_two_storey_project(tmp_path, write_example, write_wea)
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    panels = _panels(case, bundle)

    assert [(r.name, r.storey, r.x, r.y, r.z, r.included) for r in bundle.rooms] == [
        ("Room North", 0, 3.0, 2.0, 0.0, True),
        ("Store East", 0, 10.0, 2.0, 0.0, False),
        ("Room Upper", 1, 3.0, 2.0, 4.0, True),
    ]
    assert "layout-notes" not in panels
    assert hashlib.sha256(panels.encode("utf-8")).hexdigest() == PANELS_BEFORE
