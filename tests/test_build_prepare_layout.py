# tests/test_build_prepare_layout.py
"""``prepare`` with split and merged spaces (merge-split spec 4, 7, 8): the
layout, reveals per piece, the manifest ``layout`` key, the piece-row
warnings, the linked-zone refusal, well skips across a split, and the
matching hint.

The scenes are ``tests/fixtures/mixed_glazing.rad`` (a fictional two-room
building) and small variants of it; every added number is invented and
round.
"""
from __future__ import annotations

import dataclasses
from types import MappingProxyType

import numpy as np
import pytest

from leed_daylight.build.prepare import prepare
from leed_daylight.project.config import Performance, ProjectError
from leed_daylight.project.layout import apply_splits
from leed_daylight.project.spaces import Space
from leed_daylight.project.splits import SplitLine
from leed_daylight.scene.primitives import Primitive
from leed_daylight.scene.reader import read_scene
from leed_daylight.scene.storeys import detect_rooms

from test_build_prepare import (
    MIXED_GLAZING,
    ROLES,
    ROOF_ROLES,
    SKY_KW,
    _readback,
    make_project,
    roof_scene,
)


def space(room, cx, cy, *, merge_into=None, space_type="office", include=None):
    return Space(
        floor="L0", room=room, space_type=space_type, regularly_occupied=True, area_m2=10.0,
        centroid_x=cx, centroid_y=cy, level_z=0.0, view_preserving_shades=False,
        merge_into=merge_into, include=include or {"base_case": True, "variant": True}, reason="",
    )


def poly(modifier, name, corners):
    return Primitive(modifier, "polygon", name, (), (), tuple(float(v) for c in corners for v in c), "test")


def split_line(x, row=2):
    """A split line running north along x = ``x``, past both long walls."""
    return SplitLine(row=row, level_z=0.0, points=np.array([[x, -1.0], [x, 5.0]]), note="")


def split_rooms(prims, roles, *lines):
    return apply_splits(detect_rooms(prims, roles), lines, prims, roles)


def run(tmp_path, prims, roles, rooms, spaces, case="base_case", project=None, **project_kwargs):
    out_dir = tmp_path / "prepared"
    if project is None:
        project = make_project(out_dir, **project_kwargs)
    return prepare(project, prims, roles, rooms, spaces, case, out_dir)


def test_a_reveal_is_clipped_at_its_own_pieces_ceiling(tmp_path):
    # Two rooms of one space: room a's ceiling at 3.0 m, room b's at 2.5 m,
    # each with a south window whose head meets its own room's ceiling.
    prims = [
        poly("floor_carpet", "floor_a", [(0, 0, 0), (6, 0, 0), (6, 4, 0), (0, 4, 0)]),
        poly("floor_carpet", "floor_b", [(7, 0, 0), (11, 0, 0), (11, 4, 0), (7, 4, 0)]),
        poly("ceiling_tile", "ceiling_a", [(0, 0, 3), (0, 4, 3), (6, 4, 3), (6, 0, 3)]),
        poly("ceiling_tile", "ceiling_b", [(7, 0, 2.5), (7, 4, 2.5), (11, 4, 2.5), (11, 0, 2.5)]),
        poly("glass_clear", "window_a", [(2, 0, 1), (4, 0, 1), (4, 0, 3), (2, 0, 3)]),
        poly("glass_clear", "window_b", [(8, 0, 1), (10, 0, 1), (10, 0, 2.5), (8, 0, 2.5)]),
    ]
    prims = [p for p in read_scene([MIXED_GLAZING], "m") if not p.is_surface] + prims
    rooms = detect_rooms(prims, ROLES)
    spaces = [space("Office", 3.0, 2.0), space("Office East", 9.0, 2.0, merge_into="Office")]

    result = run(tmp_path, prims, ROLES, rooms, spaces)

    assert [g.group_id for g in result.groups] == ["Office_p1_180", "Office_p2_180"]
    reveals = _readback(result)
    for window, head in (("window_a", 3.0), ("window_b", 2.5)):
        names = [n for n in reveals if n.startswith(f"{window}_reveal_")]
        # Four edges, the head at its own piece's ceiling: three edges, two
        # quads each (clipped at piece 0's 3.0 m, window_b would get eight).
        assert len(names) == 6, window
        assert not any(np.allclose(reveals[n].points[:, 2], head) for n in names)


def test_the_manifest_layout_key_for_a_split_and_a_split_plus_merged_space(tmp_path):
    prims = read_scene([MIXED_GLAZING], "m")
    rooms = split_rooms(prims, ROLES, split_line(5.0, row=3))
    assert [r.room_id for r in rooms.rooms] == ["S0_R0a", "S0_R0b", "S0_R1"]
    spaces = [
        space("West", 2.5, 2.0),
        space("East", 5.5, 2.0, merge_into="office_b"),
        space("office_b", 8.2, 2.0),
    ]

    result = run(tmp_path, prims, ROLES, rooms, spaces)

    assert result.manifest["layout"] == {
        "West": {"origin": "split", "pieces": [{"room_id": "S0_R0a", "area_m2": 20.0, "splits_rows": [3]}]},
        "office_b": {"origin": "split+merged", "pieces": [
            {"room_id": "S0_R1", "area_m2": pytest.approx(16.0)},
            {"room_id": "S0_R0b", "area_m2": 4.0, "splits_rows": [3]},
        ]},
    }
    assert [g.group_id for g in result.groups] == ["West_180"]
    assert [g.room for g in result.grids] == ["West", "office_b"]


def _three_rooms():
    """The fixture plus office_c, a 4 x 3 m room north of office_a with no glass."""
    prims = read_scene([MIXED_GLAZING], "m") + [
        poly("floor_carpet", "floor_office_c", [(0, 5, 0), (4, 5, 0), (4, 8, 0), (0, 8, 0)]),
    ]
    return prims, detect_rooms(prims, ROLES)


def _three_spaces(space_type="office"):
    return [
        space("office_a", 3.0, 2.0),
        space("office_b", 8.2, 2.0, include={"base_case": True, "variant": False}),
        space("office_c", 2.0, 6.5, merge_into="office_b", space_type=space_type,
              include={"base_case": True, "variant": False}),
    ]


def test_no_layout_key_without_a_split_or_a_merge(tmp_path):
    prims = read_scene([MIXED_GLAZING], "m")
    spaces = [space("office_a", 3.0, 2.0), space("office_b", 8.2, 2.0)]

    result = run(tmp_path, prims, ROLES, detect_rooms(prims, ROLES), spaces)

    assert "layout" not in result.manifest
    assert result.layout.space("office_a").origin == "detected"


def test_no_layout_key_when_the_only_merged_space_is_not_in_the_case(tmp_path):
    prims, rooms = _three_rooms()

    variant = run(tmp_path / "variant", prims, ROLES, rooms, _three_spaces(), case="variant")
    base = run(tmp_path / "base", prims, ROLES, rooms, _three_spaces())

    assert "layout" not in variant.manifest
    assert list(base.manifest["layout"]) == ["office_b"]
    assert base.manifest["layout"]["office_b"]["origin"] == "merged"


def test_piece_row_warnings_reach_the_manifest(tmp_path):
    prims, rooms = _three_rooms()

    result = run(tmp_path, prims, ROLES, rooms, _three_spaces(space_type="store"))

    warning = (
        "space 'office_c' is a piece of 'office_b': its space_type 'store' differs from the "
        "space row's 'office'; the space row's value is used"
    )
    assert warning in result.manifest["warnings"]


def test_a_linked_zone_naming_a_piece_row_is_refused_by_prepare(tmp_path):
    prims, rooms = _three_rooms()
    out_dir = tmp_path / "prepared"
    project = dataclasses.replace(
        make_project(out_dir),
        performance=Performance(trace_scope="storey", linked_zones=MappingProxyType({0: ("office_c",)})),
    )

    with pytest.raises(ProjectError, match=r"'office_c' is a piece of space 'office_b' \(merge_into\)"):
        run(tmp_path, prims, ROLES, rooms, _three_spaces(), project=project)
    assert not out_dir.exists()


def test_halves_of_one_detected_room_share_well_skips(tmp_path):
    # Two skylight panes of office_a meet at a ridge at x = 3; the split line
    # runs along the ridge, so each half is its own space with one pane.
    rise = np.tan(np.radians(10.0))
    top = 3.5 + rise
    west = (2.0, 1.5, 3.5, 3.0, 1.5, top, 3.0, 2.5, top, 2.0, 2.5, 3.5)
    east = (3.0, 1.5, top, 4.0, 1.5, 3.5, 4.0, 2.5, 3.5, 3.0, 2.5, top)
    prims, _ = roof_scene([("glass_clear", "sky_west", west), ("glass_clear", "sky_east", east)])
    prims = [p for p in prims if p.name != "window_ext"]
    rooms = split_rooms(prims, ROOF_ROLES, split_line(3.0))
    spaces = [space("West", 1.5, 2.0), space("East", 4.5, 2.0), space("office_b", 8.2, 2.0)]

    result = run(tmp_path, prims, ROOF_ROLES, rooms, spaces, **SKY_KW)

    assert [g.group_id for g in result.groups] == ["East_sky", "West_sky"]
    readback = _readback(result)
    for pane in ("sky_west", "sky_east"):
        wells = sorted(n for n in readback if n.startswith(f"{pane}_well_"))
        # Four edges, one of them the ridge the halves share: three walls.
        assert wells == [f"{pane}_well_{i}" for i in range(3)]


def test_a_row_for_a_room_now_split_is_refused_with_the_rooms_hint(tmp_path):
    """Review Focus 1: spaces.csv still carries the whole room's centroid."""
    prims = read_scene([MIXED_GLAZING], "m")
    rooms = split_rooms(prims, ROLES, split_line(4.5))
    spaces = [space("office_a", 3.0, 2.0), space("office_b", 8.2, 2.0)]

    with pytest.raises(ProjectError) as excinfo:
        run(tmp_path, prims, ROLES, rooms, spaces)

    assert str(excinfo.value) == (
        "spaces.csv: space 'office_a': nearest room on its level is 0.75 m away, outside the "
        "0.50 m tolerance; splits.csv changed the rooms: run the rooms command again and copy "
        "each half's centroid into spaces.csv"
    )
    assert not (tmp_path / "prepared").exists()


def test_two_rows_on_one_half_are_refused_with_the_rooms_hint(tmp_path):
    prims = read_scene([MIXED_GLAZING], "m")
    rooms = split_rooms(prims, ROLES, split_line(4.5))
    spaces = [space("office_a", 2.5, 2.0), space("office_a_old", 2.0, 2.0), space("office_b", 8.2, 2.0)]

    with pytest.raises(ProjectError, match=r"both match .*; splits\.csv changed the rooms: run the rooms command again"):
        run(tmp_path, prims, ROLES, rooms, spaces)
