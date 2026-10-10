# tests/test_build_grid_points.py
"""The sensor grid built from ``grid.points_file`` instead of the lattice.

With a points file, every included, regularly occupied space gets exactly
the file's points for it, in file order, after two checks per point: it
lies strictly inside one of the space's floor pieces, and its z is that
piece's floor level plus ``grid.work_plane_m`` (within a millimetre).
Points of a space the case leaves out are not used; a point naming no
space row is refused. Every name and coordinate is invented and round.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from test_build_prepare import ROLES, load_mixed_glazing, make_project, make_spaces

from leed_daylight.build.grid import POINT_Z_TOLERANCE_M, GridError, build_grids
from leed_daylight.build.prepare import prepare
from leed_daylight.project.config import Grid
from leed_daylight.project.points import PointsError, SpacePoints
from leed_daylight.project.spaces import Space
from leed_daylight.scene.storeys import Room

GRID = Grid(spacing_m=0.60, work_plane_m=0.75, min_clearance_m=0.30)
SQUARE = np.array([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
ANNEX = np.array([(5.0, 0.0), (7.0, 0.0), (7.0, 2.0), (5.0, 2.0)])


def _room(outline, room_id="S0_R0", level_z=0.0, holes=()) -> Room:
    return Room(room_id=room_id, storey=0, level_z=level_z, outline=outline, area_m2=16.0,
                ceiling_height_m=None, holes=holes)


def _space(name, *, merge_into=None, occupied=True, include=True) -> Space:
    return Space(floor="L0", room=name, space_type="office", regularly_occupied=occupied, area_m2=16.0,
                 centroid_x=2.0, centroid_y=2.0, level_z=0.0, view_preserving_shades=False,
                 merge_into=merge_into, include={"base": include}, reason="")


def _points(space, *rows, first_line=2):
    return SpacePoints(space=space, points=np.array(rows, dtype=np.float64),
                       lines=tuple(range(first_line, first_line + len(rows))))


def test_the_grid_is_the_file_s_points_in_file_order():
    given = {"office a": _points("office a", (3.0, 1.0, 0.75), (1.0, 1.0, 0.75), (1.0, 3.0, 0.75))}

    [grid] = build_grids([_space("office a")], {"office a": _room(SQUARE)}, GRID, "base", points=given)

    np.testing.assert_array_equal(grid.points, given["office a"].points)
    assert grid.room_area_m2 == 16.0
    assert grid.cell_area_m2 == pytest.approx(0.36)


def test_the_points_need_no_clearance_and_no_lattice_spacing():
    given = {"office a": _points("office a", (0.1, 0.1, 0.75), (0.15, 0.1, 0.75))}

    [grid] = build_grids([_space("office a")], {"office a": _room(SQUARE)}, GRID, "base", points=given)

    assert grid.points.shape == (2, 3)


def test_without_points_the_lattice_is_unchanged():
    matched = {"office a": _room(SQUARE)}

    [plain] = build_grids([_space("office a")], matched, GRID, "base")
    [none] = build_grids([_space("office a")], matched, GRID, "base", points=None)

    np.testing.assert_array_equal(plain.points, none.points)
    assert plain.points.shape[0] > 0


def test_a_point_outside_its_space_is_refused_naming_the_point():
    given = {"office a": _points("office a", (1.0, 1.0, 0.75), (5.0, 1.0, 0.75))}

    with pytest.raises(GridError, match=(
        r"grid\.points_file line 3: point \(5\.000, 1\.000, 0\.750\) of space 'office a' "
        r"lies outside the space's floor polygon"
    )):
        build_grids([_space("office a")], {"office a": _room(SQUARE)}, GRID, "base", points=given)


def test_a_point_on_the_outline_is_outside():
    given = {"office a": _points("office a", (0.0, 2.0, 0.75))}

    with pytest.raises(GridError, match="lies outside the space's floor polygon"):
        build_grids([_space("office a")], {"office a": _room(SQUARE)}, GRID, "base", points=given)


def test_a_point_in_a_hole_is_outside():
    hole = np.array([(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])
    given = {"office a": _points("office a", (2.0, 2.0, 0.75))}

    with pytest.raises(GridError, match="lies outside the space's floor polygon"):
        build_grids([_space("office a")], {"office a": _room(SQUARE, holes=(hole,))}, GRID, "base",
                    points=given)


def test_a_point_off_the_work_plane_is_refused_naming_the_height():
    given = {"office a": _points("office a", (1.0, 1.0, 0.80))}

    with pytest.raises(GridError, match=(
        r"grid\.points_file line 2: point \(1\.000, 1\.000, 0\.800\) of space 'office a' is not on "
        r"the work plane: z must be 3\.750 m \(floor 3\.000 m \+ grid\.work_plane_m 0\.75 m\)"
    )):
        build_grids([_space("office a")], {"office a": _room(SQUARE, level_z=3.0)}, GRID, "base",
                    points=given)


def test_a_millimetre_off_the_work_plane_is_kept():
    given = {"office a": _points("office a", (1.0, 1.0, 0.75 + POINT_Z_TOLERANCE_M * 0.9))}

    [grid] = build_grids([_space("office a")], {"office a": _room(SQUARE)}, GRID, "base", points=given)

    assert grid.points.shape == (1, 3)
    assert POINT_Z_TOLERANCE_M == 0.001


def test_a_space_with_no_point_in_the_file_is_refused():
    given = {"office a": _points("office a", (1.0, 1.0, 0.75))}
    spaces = [_space("office a"), _space("office b")]
    matched = {"office a": _room(SQUARE), "office b": _room(ANNEX, room_id="S0_R1")}

    with pytest.raises(GridError, match=(
        r"grid\.points_file has no point for space 'office b', which case 'base' includes as "
        r"regularly occupied"
    )):
        build_grids(spaces, matched, GRID, "base", points=given)


def test_points_of_a_space_the_case_leaves_out_are_not_used():
    given = {"office a": _points("office a", (1.0, 1.0, 0.75)),
             "store": _points("store", (99.0, 99.0, 0.75), first_line=3),
             "spare": _points("spare", (99.0, 99.0, 0.75), first_line=4)}
    spaces = [_space("office a"), _space("store", occupied=False), _space("spare", include=False)]
    matched = {"office a": _room(SQUARE), "store": _room(ANNEX, room_id="S0_R1"),
               "spare": _room(ANNEX + 3.0, room_id="S0_R2")}

    grids = build_grids(spaces, matched, GRID, "base", points=given)

    assert [g.room for g in grids] == ["office a"]


def test_a_point_naming_no_row_of_spaces_csv_is_refused():
    given = {"office a": _points("office a", (1.0, 1.0, 0.75)),
             "offce a": _points("offce a", (1.0, 1.0, 0.75), first_line=3)}

    with pytest.raises(GridError, match=r"grid\.points_file line 3: space 'offce a' is not in spaces\.csv"):
        build_grids([_space("office a")], {"office a": _room(SQUARE)}, GRID, "base", points=given)


def test_an_unknown_space_name_is_shown_as_a_short_escaped_excerpt():
    name = "\x1b]0;" + "invented" * 10
    given = {"office a": _points("office a", (1.0, 1.0, 0.75)), name: _points(name, (1.0, 1.0, 0.75), first_line=3)}

    refusal = r"grid\.points_file line 3: space '\\x1b\]0;.*\.\.\. is not in spaces\.csv"
    with pytest.raises(GridError, match=refusal) as caught:
        build_grids([_space("office a")], {"office a": _room(SQUARE)}, GRID, "base", points=given)

    shown = str(caught.value).split("space ", 1)[1].split(" is not", 1)[0]
    assert len(shown) <= 40
    assert "\x1b" not in str(caught.value)


def test_a_point_naming_a_piece_row_is_refused_naming_its_space_row():
    given = {"office a": _points("office a", (1.0, 1.0, 0.75)),
             "annex": _points("annex", (6.0, 1.0, 0.75), first_line=3)}
    spaces = [_space("office a"), _space("annex", merge_into="office a")]
    matched = {"office a": _room(SQUARE), "annex": _room(ANNEX, room_id="S0_R1")}

    with pytest.raises(GridError, match=(
        r"grid\.points_file line 3: 'annex' is a piece row of space 'office a'; give its points "
        r"the space row's name 'office a'"
    )):
        build_grids(spaces, matched, GRID, "base", points=given)


def test_a_merged_space_takes_points_in_any_of_its_pieces_in_file_order():
    given = {"office a": _points("office a", (6.0, 1.0, 0.75), (1.0, 1.0, 0.75))}
    spaces = [_space("office a"), _space("annex", merge_into="office a")]
    matched = {"office a": _room(SQUARE), "annex": _room(ANNEX, room_id="S0_R1")}

    [grid] = build_grids(spaces, matched, GRID, "base", points=given)

    np.testing.assert_array_equal(grid.points[:, 0], [6.0, 1.0])
    assert grid.room_area_m2 == 32.0


def test_a_merged_piece_on_another_level_checks_its_own_work_plane():
    spaces = [_space("office a"), _space("annex", merge_into="office a")]
    matched = {"office a": _room(SQUARE), "annex": _room(ANNEX, room_id="S0_R1", level_z=0.5)}
    given = {"office a": _points("office a", (6.0, 1.0, 1.25), (1.0, 1.0, 0.75))}

    [grid] = build_grids(spaces, matched, GRID, "base", points=given)

    assert grid.points.shape == (2, 3)
    bad = {"office a": dataclasses.replace(given["office a"], points=np.array([[6.0, 1.0, 0.75]]), lines=(2,))}
    with pytest.raises(GridError, match="z must be 1.250 m"):
        build_grids(spaces, matched, GRID, "base", points=bad)


def test_a_point_on_the_work_plane_of_any_containing_piece_is_kept():
    spaces = [_space("office a"), _space("upper", merge_into="office a")]
    upper = np.array([(1.0, 0.0), (3.0, 0.0), (3.0, 2.0), (1.0, 2.0)])
    matched = {"office a": _room(SQUARE), "upper": _room(upper, room_id="S1_R0", level_z=3.0)}
    given = {"office a": _points("office a", (2.0, 1.0, 3.75), (2.0, 1.0, 0.75))}

    [grid] = build_grids(spaces, matched, GRID, "base", points=given)

    np.testing.assert_array_equal(grid.points[:, 2], [3.75, 0.75])
    bad = {"office a": dataclasses.replace(given["office a"], points=np.array([[2.0, 1.0, 2.0]]), lines=(2,))}
    with pytest.raises(GridError, match=r"line 2: .* is not on the work plane: z must be"):
        build_grids(spaces, matched, GRID, "base", points=bad)


# -- through prepare --------------------------------------------------------------


def _prepare_with(tmp_path, text):
    prims, rooms = load_mixed_glazing()
    out_dir = tmp_path / "prepared"
    points_file = tmp_path / "points.csv"
    points_file.write_text(text, encoding="utf-8", newline="\n")
    grid = Grid(spacing_m=0.5, work_plane_m=0.75, min_clearance_m=0.1, points_file=points_file)
    project = make_project(out_dir, grid=grid)
    return prepare(project, prims, ROLES, rooms, make_spaces(), "base_case", out_dir)


def test_prepare_writes_the_file_s_points_as_the_grid(tmp_path):
    text = ("space,x,y,z\n"
            "office_a,1.0,1.0,0.75\noffice_b,7.0,1.0,0.75\noffice_a,2.0,1.0,0.75\n")

    result = _prepare_with(tmp_path, text)

    assert result.grid_files["office_a"].read_text().splitlines() == [
        "1.0 1.0 0.75 0 0 1", "2.0 1.0 0.75 0 0 1",
    ]
    assert result.manifest["rooms"]["office_a"]["point_count"] == 2
    assert result.manifest["totals"]["grid_point_count"] == 3
    assert result.manifest["grid_source"] == "points_file"


def test_prepare_without_a_points_file_writes_no_grid_source(tmp_path):
    prims, rooms = load_mixed_glazing()
    out_dir = tmp_path / "prepared"

    result = prepare(make_project(out_dir), prims, ROLES, rooms, make_spaces(), "base_case", out_dir)

    assert "grid_source" not in result.manifest


def test_prepare_refuses_a_bad_points_file_and_writes_nothing(tmp_path):
    with pytest.raises(PointsError, match=r"points\.csv line 2: x 'east' is not a finite number"):
        _prepare_with(tmp_path, "space,x,y,z\noffice_a,east,1.0,0.75\n")
    assert not (tmp_path / "prepared").exists()


def test_prepare_refuses_a_point_outside_its_space_and_writes_nothing(tmp_path):
    with pytest.raises(GridError, match="point \\(8.000, 1.000, 0.750\\) of space 'office_a' lies outside"):
        _prepare_with(tmp_path, "space,x,y,z\noffice_a,8.0,1.0,0.75\noffice_b,7.0,1.0,0.75\n")
    assert not (tmp_path / "prepared").exists()
