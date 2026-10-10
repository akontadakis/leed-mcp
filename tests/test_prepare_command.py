# tests/test_prepare_command.py
"""Tests for leed_daylight.drivers.prepare: the ``prepare`` verb (CLI spec
4.4).

A fictional 6 m x 4 m x 3 m room with one exterior window, invented and
round throughout. ``prepare`` builds the model ``run`` would build and
never looks for, or runs, a Radiance binary.
"""
from __future__ import annotations

import io
import json
import pathlib

import leed_daylight.drivers.prepare as prepare_module
import leed_daylight.engine.radiance as radiance_module
from leed_daylight.build.groups import WindowGroup
from leed_daylight.scene.primitives import Primitive

SCENE_RAD = """\
void plastic wall_paint
0
0
5  0.6 0.6 0.6  0 0

void plastic ceiling_tile
0
0
5  0.8 0.8 0.8  0 0

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

ceiling_tile polygon ceiling_room_north
0
0
12  0 0 3   0 4 3   6 4 3   6 0 3

wall_paint polygon wall_room_north_south
0
0
12  0 0 0   6 0 0   6 0 3   0 0 3

glass_clear polygon window_ext
0
0
12  2 0 1   4 0 1   4 0 2   2 0 2
"""

MATERIALS_CSV = """\
modifier,primitive_type,role,note
wall_paint,plastic,wall,
ceiling_tile,plastic,ceiling,
floor_carpet,plastic,floor,
glass_clear,glass,glazing,
"""

SPACES_CSV = """\
floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,view_preserving_shades,merge_into,include_base_case,include_second_case,reason
L0,Room North,office,yes,24,3.0,2.0,0.0,no,,1,1,
"""


def _folder(tmp_path, write_example) -> pathlib.Path:
    folder = write_example(tmp_path).parent
    (folder / "scene.rad").write_text(SCENE_RAD)
    (folder / "materials.csv").write_text(MATERIALS_CSV)
    (folder / "spaces.csv").write_text(SPACES_CSV)
    return folder


def _no_radiance(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("prepare must not look for Radiance")

    monkeypatch.setattr(radiance_module, "find_binaries", refuse)
    monkeypatch.setattr(radiance_module, "subprocess_runner", refuse)


def test_prepare_writes_the_model_and_prints_the_summary(tmp_path, write_example, monkeypatch, capsys):
    _no_radiance(monkeypatch)
    folder = _folder(tmp_path, write_example)
    assert prepare_module.main([str(folder)]) == 0
    manifest = json.loads((folder / "_run" / "manifest.json").read_text())
    assert manifest["case"] == "base_case"
    points = manifest["totals"]["grid_point_count"]
    assert points > 0
    (group_id,) = manifest["groups"]
    out = capsys.readouterr().out
    assert "case: base_case" in out
    assert f"rooms: 1, grid points: {points}, regularly occupied area: 24.00 m2, window groups: 1" in out
    assert f"  Room North: 24.00 m2, {points} points, window groups: {group_id}" in out
    assert f"prepared model written to {folder / '_run'}" in out


def test_prepare_takes_the_case(tmp_path, write_example, monkeypatch, capsys):
    _no_radiance(monkeypatch)
    folder = _folder(tmp_path, write_example)
    assert prepare_module.main([str(folder), "--case", "second_case"]) == 0
    assert json.loads((folder / "_run" / "manifest.json").read_text())["case"] == "second_case"


def test_an_unknown_case_is_refused(tmp_path, write_example, capsys):
    folder = _folder(tmp_path, write_example)
    assert prepare_module.main([str(folder), "--case", "nope"]) == 1
    assert capsys.readouterr().err.startswith("error: unknown case 'nope'")


def test_a_missing_spaces_csv_is_refused(tmp_path, write_example, capsys):
    folder = _folder(tmp_path, write_example)
    (folder / "spaces.csv").unlink()
    assert prepare_module.main([str(folder)]) == 1
    assert "spaces.csv" in capsys.readouterr().err


def test_the_module_does_not_import_the_radiance_engine():
    assert not hasattr(prepare_module, "find_binaries")
    assert not hasattr(prepare_module, "Radiance")


def test_a_materials_csv_type_that_differs_from_the_scene_is_refused(tmp_path, write_example, capsys):
    folder = _folder(tmp_path, write_example)
    # The row says glass; the scene defines wall_paint as plastic.
    (folder / "materials.csv").write_text(MATERIALS_CSV.replace("wall_paint,plastic,", "wall_paint,glass,"))
    assert prepare_module.main([str(folder)]) == 1
    err = capsys.readouterr().err
    assert "wall_paint" in err and "materials.csv says primitive type 'glass'" in err
    assert not (folder / "_run" / "manifest.json").exists()


def test_every_command_that_validates_the_scene_passes_the_declared_types():
    # Seven entry points load a scene the same way; each must pass the
    # materials.csv types, or that command silently skips the check.
    root = pathlib.Path(prepare_module.__file__).parents[1]
    calls = []
    for path in sorted((root / "drivers").glob("*.py")) + sorted((root / "deliver").glob("*.py")):
        text = path.read_text()
        start = 0
        while (i := text.find("validate_scene(", start)) != -1:
            start = i + 1
            if text[max(0, i - 4):i] == "def " or text[i - 1] == "`":
                continue
            call = text[i:text.find(")", i) + 1]
            calls.append((path.name, call))
    assert len(calls) >= 7
    missing = [name for name, call in calls if "declared_types" not in call]
    assert missing == []


def test_a_spaces_csv_area_far_from_the_scene_room_area_warns(tmp_path, write_example, monkeypatch):
    _no_radiance(monkeypatch)
    folder = _folder(tmp_path, write_example)
    # The scene room is 6 m x 4 m = 24 m2; the row says an invented 30 m2.
    (folder / "spaces.csv").write_text(SPACES_CSV.replace(",office,yes,24,", ",office,yes,30,"))
    assert prepare_module.main([str(folder)]) == 0
    warnings = json.loads((folder / "_run" / "manifest.json").read_text())["warnings"]
    matches = [w for w in warnings if "Room North" in w and "area_m2" in w]
    assert len(matches) == 1
    assert "30.00" in matches[0] and "24.00" in matches[0]


def test_a_spaces_csv_area_within_five_percent_does_not_warn(tmp_path, write_example, monkeypatch):
    _no_radiance(monkeypatch)
    folder = _folder(tmp_path, write_example)
    # An invented 25 m2 against the scene's 24 m2: about 4 %, inside the band.
    (folder / "spaces.csv").write_text(SPACES_CSV.replace(",office,yes,24,", ",office,yes,25,"))
    assert prepare_module.main([str(folder)]) == 0
    warnings = json.loads((folder / "_run" / "manifest.json").read_text())["warnings"]
    assert not any("area_m2" in w for w in warnings)


# =============================================================================
# roof-glass design section 7: the summary counts groups per kind
# =============================================================================


def _pane(name):
    return Primitive(
        "glass_clear", "polygon", name, (), (), (0.0, 0.0, 3.5, 1.0, 0.0, 3.5, 1.0, 1.0, 3.5), "test",
    )


def _summary(groups):
    manifest = {
        "case": "base_case",
        "totals": {"grid_point_count": 100, "regularly_occupied_area_m2": 100.0,
                   "group_count": len(groups)},
        "rooms": {"Office A": {"area_m2": 100.0, "point_count": 100}},
        "groups": {g.group_id: {"room": "Office A"} for g in groups},
        "warnings": [],
    }
    out = io.StringIO()
    prepare_module._print_summary(manifest, pathlib.Path("_run"), out=out, groups=tuple(groups))
    return out.getvalue().splitlines()


def _window(group_id):
    return WindowGroup(
        group_id=group_id, room="Office A", storey=0, azimuth_bin_deg=float(group_id[-3:]),
        openings=(_pane(f"{group_id}_pane"),), outward_normals=(), area_m2=2.0,
    )


def test_the_summary_counts_groups_per_kind_and_lists_each_roof_group():
    groups = [
        _window("office_a_000"), _window("office_a_090"), _window("office_a_180"),
        WindowGroup(
            group_id="office_a_dif", room="Office A", storey=0, azimuth_bin_deg=None,
            openings=(_pane("dif_1"),), outward_normals=(), area_m2=3.0,
            kind="diffusing", tilt_deg=0.0,
        ),
        WindowGroup(
            group_id="office_a_sky", room="Office A", storey=0, azimuth_bin_deg=None,
            openings=(_pane("sky_1"), _pane("sky_2")), outward_normals=(), area_m2=2.0,
            kind="skylight", tilt_deg=0.0,
        ),
    ]

    lines = _summary(groups)

    assert lines[1] == (
        "rooms: 1, grid points: 100, regularly occupied area: 100.00 m2, "
        "window groups: 5 (window 3, skylight 1, diffusing 1)"
    )
    assert lines[2:4] == [
        "  office_a_dif: 1 pane, 3.0 m2, tilt 0.0, no blind (diffusing)",
        "  office_a_sky: 2 panes, 2.0 m2, tilt 0.0, blind yes",
    ]


def test_an_all_window_summary_is_unchanged():
    lines = _summary([_window("office_a_000"), _window("office_a_180")])

    assert lines[1] == "rooms: 1, grid points: 100, regularly occupied area: 100.00 m2, window groups: 2"
    assert lines[2].startswith("  Office A: 100.00 m2, 100 points, window groups: ")


# =============================================================================
# merge-split spec 9: the summary lists the pieces of a split or merged space
# =============================================================================


def test_the_summary_lists_the_pieces_of_a_merged_and_a_split_space():
    manifest = {
        "case": "base_case",
        "totals": {"grid_point_count": 300, "regularly_occupied_area_m2": 80.0, "group_count": 0},
        "rooms": {
            "Office 1": {"area_m2": 20.0, "point_count": 100},
            "Open Office East": {"area_m2": 40.0, "point_count": 150},
            "Office 2": {"area_m2": 20.0, "point_count": 50},
        },
        "groups": {},
        "layout": {
            "Office 1": {"origin": "merged", "pieces": [
                {"room_id": "S0_R2", "area_m2": 12.0}, {"room_id": "S0_R3", "area_m2": 8.0},
            ]},
            "Open Office East": {"origin": "split", "pieces": [
                {"room_id": "S1_R0ab", "area_m2": 40.0, "splits_rows": [2, 3]},
            ]},
        },
        "warnings": [],
    }
    out = io.StringIO()

    prepare_module._print_summary(manifest, pathlib.Path("_run"), out=out)

    assert out.getvalue().splitlines()[2:7] == [
        "  Office 1: 20.00 m2, 100 points, window groups: none",
        "    pieces (merged): S0_R2 12.00 m2, S0_R3 8.00 m2",
        "  Open Office East: 40.00 m2, 150 points, window groups: none",
        "    pieces (split): S1_R0ab 40.00 m2 (splits.csv lines 2, 3)",
        "  Office 2: 20.00 m2, 50 points, window groups: none",
    ]


def test_prepare_prints_the_pieces_of_a_split_space(tmp_path, write_example, monkeypatch, capsys):
    _no_radiance(monkeypatch)
    folder = _folder(tmp_path, write_example)
    (folder / "splits.csv").write_text('level_z,points,note\n0.0,"4.5 -1; 4.5 5",invented\n')
    (folder / "spaces.csv").write_text(
        SPACES_CSV.splitlines()[0] + "\n"
        "L0,Room West,office,yes,18,2.25,2.0,0.0,no,,1,1,\n"
        "L0,Room Strip,office,yes,6,5.25,2.0,0.0,no,Room West,1,1,\n"
    )

    assert prepare_module.main([str(folder)]) == 0

    out = capsys.readouterr().out
    assert "    pieces (split+merged): S0_R0a 18.00 m2 (splits.csv line 2), S0_R0b 6.00 m2 (splits.csv line 2)" in out
