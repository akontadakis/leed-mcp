# tests/test_rooms_command.py
"""Tests for leed_daylight.drivers.rooms (task ROOMS): the `rooms` verb.

Every test drives ``rooms_folder``/``main`` against a small fictional
project folder built fresh in ``tmp_path`` -- two disjoint 12 m2 floor
patches on two storeys, invented and round throughout (phase4-global.md
#5). No Radiance is involved: ``rooms_folder`` never reaches past
``detect_rooms``, so no engine seam is needed at all.

``_build_project`` writes project.yaml (from ``tests/fixtures/
project_example.yaml`` via the shared ``write_example`` fixture,
unmutated: every file it names already gets created there), then adds the
scene and materials.csv this task's own chain needs. Unlike
``tests/test_run_command.py``'s own ``_build_project``, ``spaces.csv`` is
never written here -- the whole point of this command is to work without
one.
"""
from __future__ import annotations

import csv
import io
import pathlib

import pytest

from leed_daylight.project.config import load_project
from leed_daylight.project.materials import load_materials
from leed_daylight.project.spaces import load_spaces, match_to_rooms
from leed_daylight.scene.reader import read_scene
from leed_daylight.scene.storeys import detect_rooms
from leed_daylight.scene.validate import validate_scene

import leed_daylight.drivers.rooms as rooms_module

#: Two disjoint 4 m x 3 m (12 m2) floor patches, one at storey 0
#: (z = 0.0) and one at storey 1 (z = 3.0), far enough apart in plan that
#: they never merge into one room even before the z split -- a fictional
#: two-storey shell, invented and round throughout.
TWO_ROOM_SCENE = """\
void plastic floor_mat
0
0
5  0.3 0.3 0.3  0 0

floor_mat polygon floor_room_a
0
0
12  0 0 0   4 0 0   4 3 0   0 3 0

floor_mat polygon floor_room_b
0
0
12  10 0 3   14 0 3   14 3 3   10 3 3
"""

#: The same storey-0 room as ``TWO_ROOM_SCENE``, plus one floor-role patch
#: tilted about its long edge (3 m deep, rising 0.20 m -- well inside the
#: default 5-degree normal tolerance but far outside the default 0.05 m z
#: tolerance), so ``detect_rooms`` excludes it and warns by name instead of
#: treating it as a second storey.
ONE_ROOM_WITH_TILTED_SCENE = """\
void plastic floor_mat
0
0
5  0.3 0.3 0.3  0 0

floor_mat polygon floor_room_a
0
0
12  0 0 0   4 0 0   4 3 0   0 3 0

floor_mat polygon floor_tilted
0
0
12  20 0 0   24 0 0   24 3 0.20   20 3 0.20
"""

#: A scene with a wall only -- no floor-role primitive at all, so
#: ``detect_rooms`` finds zero rooms.
NO_FLOOR_SCENE = """\
void plastic wall_paint
0
0
5  0.6 0.6 0.6  0 0

wall_paint polygon wall_only
0
0
12  0 0 0   4 0 0   4 0 3   0 0 3
"""

MATERIALS_CSV = """\
modifier,primitive_type,role,note
floor_mat,plastic,floor,
"""

WALL_ONLY_MATERIALS_CSV = """\
modifier,primitive_type,role,note
wall_paint,plastic,wall,
"""


def _build_project(tmp_path, write_example, scene_rad: str, materials_csv: str) -> pathlib.Path:
    """A fresh project folder: project.yaml (+ its referenced files, via
    write_example) plus this task's own scene.rad and materials.csv. No
    spaces.csv -- the rooms command must work without one."""
    project_path = write_example(tmp_path)
    project_dir = project_path.parent
    (project_dir / "scene.rad").write_text(scene_rad)
    (project_dir / "materials.csv").write_text(materials_csv)
    return project_dir


def _detected_rooms(project_dir: pathlib.Path, materials_csv: str = MATERIALS_CSV):
    project = load_project(project_dir / "project.yaml")
    prims = read_scene(project.scene.files, project.scene.units)
    roles = load_materials(project_dir / "materials.csv")
    validate_scene(prims, project.scene.units, roles).raise_if_errors()
    return detect_rooms(prims, roles)


# =============================================================================
# the brief's six rows
# =============================================================================


def test_every_detected_room_is_listed(tmp_path, write_example):
    project_dir = _build_project(tmp_path, write_example, TWO_ROOM_SCENE, MATERIALS_CSV)
    out = io.StringIO()

    rooms_module.rooms_folder(project_dir, out=out)

    printed = out.getvalue()
    lines = [line for line in printed.splitlines() if line.startswith("R")]
    assert len(lines) == 2
    # Room A: storey 0, 12 m2, centroid (2.00, 1.50), level_z 0.00.
    assert "storey 0" in lines[0]
    assert "12.00 m2" in lines[0]
    assert "2.00, 1.50" in lines[0]
    assert "0.00" in lines[0]
    # Room B: storey 1, 12 m2, centroid (12.00, 1.50), level_z 3.00.
    assert "storey 1" in lines[1]
    assert "12.00 m2" in lines[1]
    assert "12.00, 1.50" in lines[1]
    assert "3.00" in lines[1]


def test_the_template_round_trips_through_load_spaces_and_matching(tmp_path, write_example):
    project_dir = _build_project(tmp_path, write_example, TWO_ROOM_SCENE, MATERIALS_CSV)

    template_path = rooms_module.rooms_folder(project_dir, out=io.StringIO())

    assert template_path == project_dir / "_run" / "spaces_template.csv"
    spaces = load_spaces(template_path)
    assert len(spaces) == 2

    detection = _detected_rooms(project_dir)
    matched = match_to_rooms(spaces, detection.rooms)
    assert len(matched) == 2
    assert set(matched) == {s.room for s in spaces}


def test_an_existing_spaces_csv_is_never_touched(tmp_path, write_example):
    project_dir = _build_project(tmp_path, write_example, TWO_ROOM_SCENE, MATERIALS_CSV)
    existing = (
        "floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,"
        "level_z,view_preserving_shades,merge_into,include_base,reason\n"
        "L0,Kept As Is,office,yes,12.00,2.00,1.50,0.00,no,,1,\n"
    )
    (project_dir / "spaces.csv").write_text(existing)
    before = (project_dir / "spaces.csv").read_bytes()

    rooms_module.rooms_folder(project_dir, out=io.StringIO())

    after = (project_dir / "spaces.csv").read_bytes()
    assert after == before


def test_a_missing_materials_csv_is_named(tmp_path, write_example, capsys):
    project_dir = _build_project(tmp_path, write_example, TWO_ROOM_SCENE, MATERIALS_CSV)
    (project_dir / "materials.csv").unlink()

    exit_code = rooms_module.main([str(project_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("error: ")
    assert "materials.csv" in captured.err


def test_a_folder_inside_the_repository_is_refused(tmp_path, write_example, monkeypatch, capsys):
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    monkeypatch.setattr(rooms_module, "REPO_ROOT", fake_repo)
    inside = fake_repo / "some_client_model"
    inside.mkdir()

    exit_code = rooms_module.main([str(inside)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("error: ")
    assert str(inside) in captured.err


def test_detect_rooms_warnings_are_printed(tmp_path, write_example):
    project_dir = _build_project(
        tmp_path, write_example, ONE_ROOM_WITH_TILTED_SCENE, MATERIALS_CSV,
    )
    out = io.StringIO()

    rooms_module.rooms_folder(project_dir, out=out)

    printed = out.getvalue()
    assert "floor_tilted" in printed
    assert "tilted" in printed


# =============================================================================
# hostile-input pass: refusals this driver defines itself (mirrors
# tests/test_run_command.py's own coverage of the same shared checks)
# =============================================================================


def test_a_non_path_folder_argument_is_refused_by_name():
    with pytest.raises(rooms_module.RoomsError, match="NoneType"):
        rooms_module.rooms_folder(None)


def test_a_nonexistent_folder_is_refused(tmp_path, capsys):
    missing = tmp_path / "does_not_exist"

    exit_code = rooms_module.main([str(missing)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("error: ")
    assert str(missing) in captured.err


def test_a_file_given_as_the_folder_is_refused(tmp_path, capsys):
    not_a_dir = tmp_path / "im_a_file.txt"
    not_a_dir.write_text("not a folder")

    exit_code = rooms_module.main([str(not_a_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("error: ")


def test_a_missing_project_yaml_is_named(tmp_path, write_example, capsys):
    project_dir = _build_project(tmp_path, write_example, TWO_ROOM_SCENE, MATERIALS_CSV)
    (project_dir / "project.yaml").unlink()

    exit_code = rooms_module.main([str(project_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "project.yaml" in captured.err


def test_a_chain_refusal_prints_error_and_exit_1_with_no_traceback(tmp_path, write_example, capsys):
    """SceneError (a ValueError subclass) from a broken scene reaches
    main() as a named, one-line refusal, not a traceback."""
    project_dir = _build_project(tmp_path, write_example, TWO_ROOM_SCENE, MATERIALS_CSV)
    # Degenerate: a materials.csv gap (a role validate_scene rejects).
    (project_dir / "materials.csv").write_text(
        "modifier,primitive_type,role,note\n"
        # floor_mat's role is missing entirely.
    )

    exit_code = rooms_module.main([str(project_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("error: ")
    assert "Traceback" not in captured.err


def test_no_rooms_detected_writes_a_header_only_template(tmp_path, write_example):
    project_dir = _build_project(
        tmp_path, write_example, NO_FLOOR_SCENE, WALL_ONLY_MATERIALS_CSV,
    )
    out = io.StringIO()

    template_path = rooms_module.rooms_folder(project_dir, out=out)

    with template_path.open(newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1  # header only, no rooms
    assert rows[0] == list(rooms_module.SPACES_TEMPLATE_HEADER)
    assert "warnings: 0" in out.getvalue()


# ---------------------------------------------------------------------------
# Every failure is a one-line named refusal, exit 1 -- never a traceback
# ---------------------------------------------------------------------------


def _assert_one_line_refusal(exit_code, captured, *words):
    assert exit_code == 1
    assert captured.err.startswith("error: ")
    assert captured.err.count("\n") == 1
    assert "Traceback" not in captured.err
    for word in words:
        assert word in captured.err


def test_invalid_yaml_is_a_one_line_refusal(tmp_path, write_example, capsys):
    project_dir = _build_project(tmp_path, write_example, TWO_ROOM_SCENE, MATERIALS_CSV)
    (project_dir / "project.yaml").write_text("project: [unclosed\n  name: x\n")

    exit_code = rooms_module.main([str(project_dir)])

    _assert_one_line_refusal(exit_code, capsys.readouterr(), "project.yaml", "YAML")


def test_a_geometry_failure_is_a_one_line_refusal(tmp_path, write_example, monkeypatch, capsys):
    from shapely.errors import GEOSException

    project_dir = _build_project(tmp_path, write_example, TWO_ROOM_SCENE, MATERIALS_CSV)

    def broken(*args, **kwargs):
        raise GEOSException("TopologyException: side location conflict")

    monkeypatch.setattr(rooms_module, "scene_rooms", broken)
    exit_code = rooms_module.main([str(project_dir)])

    _assert_one_line_refusal(exit_code, capsys.readouterr(), "geometry", "side location conflict")


def test_an_os_error_is_a_one_line_refusal(tmp_path, write_example, monkeypatch, capsys):
    project_dir = _build_project(tmp_path, write_example, TWO_ROOM_SCENE, MATERIALS_CSV)

    def unwritable(*args, **kwargs):
        raise PermissionError(13, "Permission denied", "spaces_template.csv")

    monkeypatch.setattr(rooms_module, "_write_template", unwritable)
    exit_code = rooms_module.main([str(project_dir)])

    _assert_one_line_refusal(exit_code, capsys.readouterr(), "Permission denied")


# =============================================================================
# merge-split spec 9: rooms applies splits.csv before writing the template
# =============================================================================

#: Room A of ``TWO_ROOM_SCENE`` cut along x = 2 m, run north: two 6 m2 halves.
SPLITS_CSV = 'level_z,points,note\n0.0,"2 -1; 2 4",invented\n'


def test_rooms_writes_a_row_for_each_half(tmp_path, write_example):
    project_dir = _build_project(tmp_path, write_example, TWO_ROOM_SCENE, MATERIALS_CSV)
    (project_dir / "splits.csv").write_text(SPLITS_CSV)

    template_path = rooms_module.rooms_folder(project_dir, out=io.StringIO())

    with template_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["room"], r["floor"], r["area_m2"], r["centroid_x"], r["centroid_y"]) for r in rows] == [
        ("R1", "L0", "6.00", "1.00", "1.50"),
        ("R2", "L0", "6.00", "3.00", "1.50"),
        ("R3", "L1", "12.00", "12.00", "1.50"),
    ]


def test_rooms_prints_the_split_count_only_with_splits_csv(tmp_path, write_example):
    project_dir = _build_project(tmp_path, write_example, TWO_ROOM_SCENE, MATERIALS_CSV)
    before = io.StringIO()
    rooms_module.rooms_folder(project_dir, out=before)
    (project_dir / "splits.csv").write_text(SPLITS_CSV)
    after = io.StringIO()

    rooms_module.rooms_folder(project_dir, out=after)

    assert before.getvalue().splitlines()[0].startswith("R1: storey 0, area 12.00 m2")
    assert "split lines" not in before.getvalue()
    lines = after.getvalue().splitlines()
    assert lines[0] == "split lines applied: 1 (splits.csv)"
    assert lines[1] == "R1: storey 0, area 6.00 m2, centroid (1.00, 1.50), level_z 0.00"


def test_a_bad_split_line_is_a_one_line_refusal(tmp_path, write_example, capsys):
    project_dir = _build_project(tmp_path, write_example, TWO_ROOM_SCENE, MATERIALS_CSV)
    (project_dir / "splits.csv").write_text('level_z,points,note\n0.0,"20 -1; 20 4",invented\n')

    exit_code = rooms_module.main([str(project_dir)])

    _assert_one_line_refusal(exit_code, capsys.readouterr(), "splits.csv line 2", "crosses no room")
    assert not (project_dir / "_run" / "spaces_template.csv").exists()
