# tests/test_scan_command.py
"""Tests for leed_daylight.drivers.scan: the ``scan`` verb (CLI spec 4.2).

A fictional 6 m x 4 m x 3 m room with one window, invented and round
throughout, in a folder built fresh in ``tmp_path`` by ``write_example``.
``scan`` reads and checks; it never writes into the folder.
"""
from __future__ import annotations

import pathlib

import leed_daylight.drivers.scan as scan_module

SCENE_RAD = """\
void plastic wall_paint
0
0
5  0.6 0.6 0.6  0 0

void plastic floor_carpet
0
0
5  0.3 0.3 0.3  0 0

void glass glass_clear
0
0
3  0.6 0.6 0.6

floor_carpet polygon floor_room_north
0
0
12  0 0 0   6 0 0   6 4 0   0 4 0

wall_paint polygon wall_room_north_south
0
0
12  0 0 0   6 0 0   6 0 3   0 0 3

glass_clear polygon window_ext
0
0
12  2 0 1   4 0 1   4 0 2   2 0 2
"""

#: The same room plus a light source, which the scene checks refuse.
SCENE_WITH_SOURCE = SCENE_RAD + """
void light sun_glow
0
0
3  1 1 1

sun_glow source extra_sun
0
0
4  0 0 1  0.5
"""

MATERIALS_CSV = """\
modifier,primitive_type,role,note
wall_paint,plastic,wall,
floor_carpet,plastic,floor,
glass_clear,glass,glazing,
"""


def _folder(tmp_path, write_example, scene=SCENE_RAD, materials=None) -> pathlib.Path:
    folder = write_example(tmp_path).parent
    (folder / "scene.rad").write_text(scene, newline="\n")
    if materials is not None:
        (folder / "materials.csv").write_text(materials, newline="\n")
    return folder


def _listing(folder: pathlib.Path) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*"))


def test_scan_without_materials_lists_the_materials(tmp_path, write_example, capsys):
    folder = _folder(tmp_path, write_example)
    assert scan_module.main([str(folder)]) == 0
    out = capsys.readouterr().out
    assert "surfaces: 3" in out
    assert "size: x 6.00 m, y 4.00 m, z 3.00 m" in out
    assert "floor_carpet" in out and "24.00" in out
    assert "(no materials.csv)" in out
    assert "storey" not in out


def test_scan_with_materials_adds_roles_and_storeys(tmp_path, write_example, capsys):
    folder = _folder(tmp_path, write_example, materials=MATERIALS_CSV)
    assert scan_module.main([str(folder)]) == 0
    out = capsys.readouterr().out
    assert "glazing" in out
    assert "(no materials.csv)" not in out
    assert "storey 0: level 0.00 m, 1 room(s), 24.00 m2" in out


def test_scan_writes_nothing(tmp_path, write_example):
    folder = _folder(tmp_path, write_example, materials=MATERIALS_CSV)
    before = _listing(folder)
    assert scan_module.main([str(folder)]) == 0
    assert _listing(folder) == before


def test_a_scene_problem_is_listed_and_returns_1(tmp_path, write_example, capsys):
    folder = _folder(tmp_path, write_example, scene=SCENE_WITH_SOURCE, materials=MATERIALS_CSV)
    assert scan_module.main([str(folder)]) == 1
    captured = capsys.readouterr()
    assert "problems: 1" in captured.out
    assert "extra_sun" in captured.out
    assert captured.err.strip() == "error: the scene has 1 problem(s), listed above"
    assert "storey" not in captured.out


def test_a_missing_project_yaml_is_refused(tmp_path, capsys):
    folder = tmp_path / "empty"
    folder.mkdir()
    assert scan_module.main([str(folder)]) == 1
    assert "project.yaml" in capsys.readouterr().err


def test_a_filled_in_template_value_is_refused_by_name(tmp_path, write_example, capsys):
    def mutate(raw):
        raw["site"]["north_rotation_source"] = "FILL_ME"
        return raw

    folder = write_example(tmp_path, mutate).parent
    assert scan_module.main([str(folder)]) == 1
    assert "site.north_rotation_source" in capsys.readouterr().err


def test_a_broken_polygon_is_a_problem_and_the_size_comes_from_the_rest(tmp_path, write_example, capsys):
    scene = SCENE_RAD + """
floor_carpet polygon floor_broken
0
0
6  0 0 0   1 1 1
"""
    folder = _folder(tmp_path, write_example, scene=scene, materials=MATERIALS_CSV)
    assert scan_module.main([str(folder)]) == 1
    out = capsys.readouterr().out
    assert "size: x 6.00 m, y 4.00 m, z 3.00 m" in out
    assert "floor_broken: polygon has fewer than three vertices" in out


def test_a_surface_with_an_undefined_modifier_is_a_problem(tmp_path, write_example, capsys):
    scene = SCENE_RAD + """
paint_missing polygon wall_room_north_east
0
0
12  6 0 0   6 4 0   6 4 3   6 0 3
"""
    folder = _folder(tmp_path, write_example, scene=scene)
    assert scan_module.main([str(folder)]) == 1
    assert "modifier 'paint_missing' is never defined" in capsys.readouterr().out


# =============================================================================
# merge-split spec 9: scan reports the split lines, only with splits.csv
# =============================================================================


def _splits(folder: pathlib.Path, x: float) -> None:
    """One split line along ``x``, run north past both long walls."""
    (folder / "splits.csv").write_text(f'level_z,points,note\n0.0,"{x:g} -1; {x:g} 5",invented\n', newline="\n")


def test_scan_prints_the_split_lines_and_the_storeys_after_splits(tmp_path, write_example, capsys):
    folder = _folder(tmp_path, write_example, materials=MATERIALS_CSV)
    _splits(folder, 4.5)

    assert scan_module.main([str(folder)]) == 0

    lines = capsys.readouterr().out.splitlines()
    at = lines.index("split lines: 1")
    assert lines[at + 1] == "storey 0: level 0.00 m, 2 room(s), 24.00 m2"


def test_scan_reports_a_bad_split_line_as_a_problem(tmp_path, write_example, capsys):
    folder = _folder(tmp_path, write_example, materials=MATERIALS_CSV)
    _splits(folder, 3.0)  # through the middle of window_ext

    assert scan_module.main([str(folder)]) == 1

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    at = lines.index("split problems: 1")
    assert lines[at + 1].startswith("  - splits.csv line 2: the line meets pane window_ext of room S0_R0")
    assert "storey" not in captured.out
    assert captured.err.strip() == "error: the scene has 1 problem(s), listed above"


def test_scan_without_materials_does_not_read_splits_csv(tmp_path, write_example, capsys):
    folder = _folder(tmp_path, write_example)
    (folder / "splits.csv").write_text("not,a,splits,file\n", newline="\n")

    assert scan_module.main([str(folder)]) == 0

    out = capsys.readouterr().out
    assert "split lines" not in out and "split problems" not in out
