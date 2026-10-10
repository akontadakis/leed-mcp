# tests/test_materials_command.py
"""Tests for leed_daylight.drivers.materials: the ``materials`` verb (CLI
spec 4.3).

A fictional one-room scene, invented and round throughout. The template
has an empty role in every row, so ``load_materials`` refuses it until a
person has chosen each role.
"""
from __future__ import annotations

import csv
import pathlib

import pytest

import leed_daylight.drivers.materials as materials_module
from leed_daylight.project.config import ProjectError
from leed_daylight.project.materials import load_materials

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

void light lamp_glow
0
0
3  1 1 1

floor_carpet polygon floor_room_north
0
0
12  0 0 0   6 0 0   6 4 0   0 4 0

wall_paint polygon wall_room_north_south
0
0
12  0 0 0   6 0 0   6 0 3   0 0 3

wall_paint polygon wall_room_north_west
0
0
12  0 0 0   0 0 3   0 4 3   0 4 0

glass_clear polygon window_ext
0
0
12  2 0 1   4 0 1   4 0 2   2 0 2

lamp_glow polygon lamp_panel
0
0
12  2 2 2.9   3 2 2.9   3 3 2.9   2 3 2.9
"""


def _folder(tmp_path, write_example) -> pathlib.Path:
    folder = write_example(tmp_path).parent
    (folder / "scene.rad").write_text(SCENE_RAD)
    return folder


def _rows(path: pathlib.Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_template_header_order_and_empty_roles(tmp_path, write_example, capsys):
    folder = _folder(tmp_path, write_example)
    assert materials_module.main([str(folder)]) == 0
    template = folder / "_run" / "materials_template.csv"
    assert template.read_text().splitlines()[0] == "modifier,primitive_type,role,note"
    rows = _rows(template)
    # Largest area first: wall 30 m2, floor 24 m2, glass 2 m2, lamp 1 m2.
    assert [r["modifier"] for r in rows] == ["wall_paint", "floor_carpet", "glass_clear", "lamp_glow"]
    assert [r["primitive_type"] for r in rows] == ["plastic", "plastic", "glass", "light"]
    assert all(r["role"] == "" for r in rows)
    assert rows[0]["note"] == "guess: wall; 2 surfaces, 30.00 m2"
    assert rows[2]["note"] == "guess: glazing; 1 surfaces, 2.00 m2"
    assert rows[3]["note"] == "guess: none; 1 surfaces, 1.00 m2"
    out = capsys.readouterr().out
    assert f"template written to {template}" in out
    assert "copy it to materials.csv and fill every role" in out


def test_load_materials_refuses_the_template_as_written(tmp_path, write_example):
    folder = _folder(tmp_path, write_example)
    materials_module.main([str(folder)])
    with pytest.raises(ProjectError):
        load_materials(folder / "_run" / "materials_template.csv")


def test_filled_template_loads(tmp_path, write_example):
    folder = _folder(tmp_path, write_example)
    materials_module.main([str(folder)])
    template = folder / "_run" / "materials_template.csv"
    roles = {"wall_paint": "wall", "floor_carpet": "floor", "glass_clear": "glazing", "lamp_glow": "ignore"}
    rows = _rows(template)
    for row in rows:
        row["role"] = roles[row["modifier"]]
    target = folder / "materials.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["modifier", "primitive_type", "role", "note"])
        writer.writeheader()
        writer.writerows(rows)
    assert load_materials(target) == roles


def test_an_existing_materials_csv_is_untouched(tmp_path, write_example):
    folder = _folder(tmp_path, write_example)
    (folder / "materials.csv").write_text("mine\n")
    assert materials_module.main([str(folder)]) == 0
    assert (folder / "materials.csv").read_text() == "mine\n"


def test_a_second_run_overwrites_the_template(tmp_path, write_example):
    folder = _folder(tmp_path, write_example)
    template = folder / "_run" / "materials_template.csv"
    template.parent.mkdir()
    template.write_text("old\n")
    assert materials_module.main([str(folder)]) == 0
    assert template.read_text().startswith("modifier,")


def test_a_missing_project_yaml_is_refused(tmp_path, capsys):
    folder = tmp_path / "empty"
    folder.mkdir()
    assert materials_module.main([str(folder)]) == 1
    assert "project.yaml" in capsys.readouterr().err


def test_a_modifier_the_scene_never_defines_is_named_in_the_note(tmp_path, write_example):
    folder = _folder(tmp_path, write_example)
    (folder / "scene.rad").write_text(SCENE_RAD + """
paint_missing polygon wall_room_north_east
0
0
12  6 0 0   6 4 0   6 4 3   6 0 3
""")
    assert materials_module.main([str(folder)]) == 0
    rows = {r["modifier"]: r for r in _rows(folder / "_run" / "materials_template.csv")}
    assert rows["paint_missing"]["primitive_type"] == ""
    assert rows["paint_missing"]["note"] == "guess: none; 1 surfaces, 12.00 m2; not defined in the scene"
