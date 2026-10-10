# tests/test_run_command.py
"""Tests for leed_daylight.drivers.run (task RUN): the `run` verb.

Every test drives ``run_folder``/``main`` against a small fictional
one-room project folder built fresh in ``tmp_path`` -- a 6 m x 4 m x 3 m
room ("Room North") with one exterior window, invented and round
throughout (phase4-global.md #5). A FAKE engine (the same
``Option1Engine`` shape as ``tests/test_run_option1.py``'s own
``FakeEngine``) is injected with ``rad=None`` in every test: no test here
builds the real Radiance engine or executes a Radiance binary.

``_build_project`` writes ``project.yaml`` (from ``tests/fixtures/
project_example.yaml`` via the shared ``write_example`` fixture, unmutated:
every file it names already gets created there), then adds the scene,
materials.csv and spaces.csv this task's own chain also needs, which
``write_example`` does not create.  ``_prepared_shape`` calls the real
``prepare()`` once, directly, only to learn the real group id and the real
grid point count for the one room -- so the FAKE engine below can be built
with arrays of the right shape without hand-deriving the sensor-grid
lattice formula. ``run_folder`` then calls the real ``prepare()`` again
internally; both calls are pure functions of the same inputs and produce
byte-identical output (proven by ``tests/test_build_prepare.py``'s own
"preparing twice is byte-identical" case), so the two group ids and point
counts always agree.
"""
from __future__ import annotations

import dataclasses
import io
import json
import pathlib

import numpy as np
import pytest

from leed_daylight.build.prepare import prepare
from leed_daylight.build.groups import WindowGroup
from leed_daylight.constants import UNVERIFIED
from leed_daylight.project.config import load_project
from leed_daylight.project.materials import load_materials
from leed_daylight.project.spaces import load_spaces
from leed_daylight.scene.reader import read_scene
from leed_daylight.scene.storeys import detect_rooms
from leed_daylight.scene.validate import validate_scene

import leed_daylight.drivers.run as run_module

ROOM = "Room North"

#: A fictional 6 m x 4 m x 3 m room with one exterior window in its south
#: wall -- invented and round throughout, the same geometry style as
#: tests/fixtures/mixed_glazing.rad's own office_a.
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

#: Two case columns (in file order) so the default-case test can tell
#: "picks the first case" apart from "there was only one case to pick".
SPACES_CSV = """\
floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,view_preserving_shades,merge_into,include_base_case,include_second_case,reason
L0,Room North,office,yes,24,3.0,2.0,0.0,no,,1,1,
"""

#: LEED thresholds this test file uses to build a FAKE engine's canned lux
#: (from tests/conftest.py's own MINIMAL_CREDIT_RULES, written by
#: write_example): sDA lux 300, ASE lux 1000/250 hours. A round margin
#: above/below, never the threshold's own value.
SDA_LUX = 300.0
ASE_LUX = 1000.0


def _build_project(tmp_path, write_example) -> pathlib.Path:
    """A fresh project folder: project.yaml (+ its referenced files, via
    write_example) plus this task's own scene.rad, materials.csv and
    spaces.csv."""
    project_path = write_example(tmp_path)
    project_dir = project_path.parent
    (project_dir / "scene.rad").write_text(SCENE_RAD, newline="\n")
    (project_dir / "materials.csv").write_text(MATERIALS_CSV, newline="\n")
    (project_dir / "spaces.csv").write_text(SPACES_CSV, newline="\n")
    return project_dir


def _prepared_shape(project_dir: pathlib.Path):
    """Run the real chain through ``prepare()`` once, directly, to learn
    the real group id and grid point count for ``ROOM`` -- never executed
    a second time by the test itself; ``run_folder`` does its own,
    independent (but deterministic, hence identical) ``prepare()`` call."""
    project = load_project(project_dir / "project.yaml")
    prims = read_scene(project.scene.files, project.scene.units)
    roles = load_materials(project_dir / "materials.csv")
    validate_scene(prims, project.scene.units, roles).raise_if_errors()
    rooms = detect_rooms(prims, roles)
    spaces = load_spaces(project_dir / "spaces.csv")
    prepared = prepare(project, prims, roles, rooms, spaces, "base_case", project_dir / "_shape")
    (group,) = prepared.groups
    (grid,) = prepared.grids
    return group.group_id, int(grid.points.shape[0])


class FakeEngine:
    """The ``Option1Engine`` seam: one group, canned lux, always passing
    sDA (well above ``SDA_LUX``) and never triggering ASE (zero direct
    lux). Never touches a Radiance binary."""

    def __init__(self, n_hours: int, group_id: str, n_points: int, n_occupied: int):
        self._n_hours = n_hours
        idx = (ROOM,) * n_points
        self._group_id = group_id
        self._direct = {group_id: (idx, np.zeros((n_points, n_occupied)))}
        self._sky = {group_id: (idx, np.full((n_points, n_occupied), SDA_LUX + 100.0), None)}

    @property
    def n_hours(self):
        return self._n_hours

    def direct(self, group, hours):
        return self._direct[group.group_id]

    def sky(self, group, hours, open_schedule):
        return self._sky[group.group_id]

    def residual(self, hours):
        idx, direct = self._direct[self._group_id]
        return [(idx, np.zeros_like(direct), np.zeros_like(direct))]


class HalfPassEngine:
    """The ``Option1Engine`` seam (F10): a 50/50 split of sensor points
    passing and failing sDA, plus a few points exceeding ASE and the rest
    not -- enough variation that ``points.csv``'s own share of "yes" rows
    can be checked against results.json's point-weighted aggregates,
    unlike ``FakeEngine`` above, which is all-pass/never-exceeds.

    ``ASE_LIT_POINTS`` (3) stays within this test project's own room
    (24 m2, under 500 sf -> LM-83-12 2.2.6 allows 3 lit points, see
    ``metrics.blinds.AREA_BANDS``) so the blind schedule never closes and
    the engine is never asked for blinds-closed sky lux.
    """

    ASE_LIT_POINTS = 3

    def __init__(self, n_hours: int, group_id: str, n_points: int, n_occupied: int):
        self._n_hours = n_hours
        self._group_id = group_id
        idx = (ROOM,) * n_points
        half = n_points // 2

        sky_lux = np.zeros((n_points, n_occupied))
        sky_lux[:half] = SDA_LUX + 100.0
        self._sky = {group_id: (idx, sky_lux, None)}

        direct_lux = np.zeros((n_points, n_occupied))
        direct_lux[: self.ASE_LIT_POINTS] = ASE_LUX + 500.0
        self._direct = {group_id: (idx, direct_lux)}

    @property
    def n_hours(self):
        return self._n_hours

    def direct(self, group, hours):
        return self._direct[group.group_id]

    def sky(self, group, hours, open_schedule):
        return self._sky[group.group_id]

    def residual(self, hours):
        idx, direct = self._direct[self._group_id]
        return [(idx, np.zeros_like(direct), np.zeros_like(direct))]


#: The occupied-hour count for project_example.yaml's own analysis
#: (08:00-18:00, US daylight saving): 365 days x 10 occupied hours a day,
#: whatever the DST rule (spec 6.9) -- the same value
#: tests/test_run_option1.py uses throughout without overriding hours.
N_OCCUPIED = 3650
#: leed_daylight.metrics.occupancy.HOURS_IN_YEAR -- the mask's own length.
FULL_MASK_LEN = 8760


def _engine_for(project_dir: pathlib.Path) -> FakeEngine:
    group_id, n_points = _prepared_shape(project_dir)
    return FakeEngine(FULL_MASK_LEN, group_id, n_points, N_OCCUPIED)


# =============================================================================
# the brief's eight rows
# =============================================================================


def test_results_json_is_written_with_every_field(tmp_path, write_example):
    project_dir = _build_project(tmp_path, write_example)
    engine = _engine_for(project_dir)

    results_path = run_module.run_folder(
        project_dir, "base_case", engine=engine, rad=None, out=io.StringIO(),
    )

    assert results_path == project_dir / "_run" / "option1" / "base_case" / "results.json"
    payload = json.loads(results_path.read_text())

    expected_keys = {
        "tool_version", "radiance_version", "case", "occupied_hours", "headline",
        "sda_area_pct", "sda_point_pct", "ase_area_pct", "ase_point_pct", "score",
        "narrative_spaces", "spaces", "blinds_closed_hours", "warnings", "settings",
        "inputs", "groups",
    }
    assert set(payload) == expected_keys
    assert payload["settings"]["daylight_saving_year"] == 2025

    assert payload["case"] == "base_case"
    assert payload["occupied_hours"] == N_OCCUPIED
    assert payload["radiance_version"] is None  # an engine was injected, no rad given
    assert payload["sda_area_pct"] == pytest.approx(100.0)
    assert payload["sda_point_pct"] == pytest.approx(100.0)
    assert payload["ase_area_pct"] == pytest.approx(0.0)
    assert payload["ase_point_pct"] == pytest.approx(0.0)
    assert payload["score"]["points"] == 1  # MINIMAL_CREDIT_RULES: one band, 40 % -> 1 point
    assert payload["narrative_spaces"] == []
    (space,) = payload["spaces"]
    assert space["room"] == ROOM
    assert space["area_m2"] == pytest.approx(24.0)
    assert space["sda_pct"] == pytest.approx(100.0)
    assert space["ase_pct"] == pytest.approx(0.0)
    assert space["needs_glare_narrative"] is False
    assert space["residual_sky_pct"] == 0.0  # the fake residual pass is dark
    assert set(payload["blinds_closed_hours"]) == {engine._group_id}
    assert payload["blinds_closed_hours"][engine._group_id] == 0  # sky lux is always open_lux
    assert isinstance(payload["warnings"], list)


def test_the_summary_prints_the_headline_and_every_space(tmp_path, write_example):
    project_dir = _build_project(tmp_path, write_example)
    engine = _engine_for(project_dir)
    out = io.StringIO()

    run_module.run_folder(project_dir, "base_case", engine=engine, rad=None, out=out)

    printed = out.getvalue()
    assert "base_case" in printed
    assert "sDA" in printed
    assert "ASE" in printed
    assert ROOM in printed


def test_the_default_case_is_the_first_in_spaces_csv(tmp_path, write_example):
    project_dir = _build_project(tmp_path, write_example)
    engine = _engine_for(project_dir)

    results_path = run_module.run_folder(
        project_dir, engine=engine, rad=None, out=io.StringIO(),
    )

    payload = json.loads(results_path.read_text())
    assert payload["case"] == "base_case"


def test_an_unknown_case_is_refused(tmp_path, write_example, capsys):
    project_dir = _build_project(tmp_path, write_example)

    exit_code = run_module.main([str(project_dir), "--case", "no_such_case"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("error: ")
    assert "no_such_case" in captured.err
    assert "base_case" in captured.err
    assert "second_case" in captured.err
    assert captured.out == ""


def test_a_missing_input_file_is_named(tmp_path, write_example, capsys):
    project_dir = _build_project(tmp_path, write_example)
    (project_dir / "spaces.csv").unlink()

    exit_code = run_module.main([str(project_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("error: ")
    assert "spaces.csv" in captured.err


def test_a_folder_inside_the_repository_is_refused(tmp_path, write_example, monkeypatch, capsys):
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    monkeypatch.setattr(run_module, "REPO_ROOT", fake_repo)
    inside = fake_repo / "some_client_model"
    inside.mkdir()

    exit_code = run_module.main([str(inside)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("error: ")
    assert str(inside) in captured.err


def test_unverified_bands_print_not_awarded(tmp_path, write_example):
    project_dir = _build_project(tmp_path, write_example)
    (project_dir / "credit_rules.yaml").write_text(
        'version: "Example rules v0"\n'
        "max_points: 3\n"
        "option1:\n"
        "  sda:\n"
        "    lux: 300\n"
        "    fraction: 0.50\n"
        f"    bands: {UNVERIFIED}\n"
        "  ase:\n"
        "    lux: 1000\n"
        "    hours: 250\n"
        "    per_space_narrative_trigger_pct: 10.0\n"
        "option2:\n"
        f"  bands: {UNVERIFIED}\n"
        "  lux_min: 300\n"
        "  lux_max: 3000\n"
        "glare_control_required: true\n"
    , newline="\n")
    engine = _engine_for(project_dir)
    out = io.StringIO()

    results_path = run_module.run_folder(
        project_dir, "base_case", engine=engine, rad=None, out=out,
    )

    payload = json.loads(results_path.read_text())
    assert payload["score"]["points"] is None
    assert "not awarded" in out.getvalue()


def test_results_json_is_stable(tmp_path, write_example):
    project_dir = _build_project(tmp_path, write_example)
    engine1 = _engine_for(project_dir)

    path1 = run_module.run_folder(
        project_dir, "base_case", engine=engine1, rad=None, out=io.StringIO(),
    )
    bytes1 = path1.read_bytes()

    engine2 = _engine_for(project_dir)
    path2 = run_module.run_folder(
        project_dir, "base_case", engine=engine2, rad=None, out=io.StringIO(),
    )
    bytes2 = path2.read_bytes()

    assert bytes1 == bytes2


# =============================================================================
# hostile-input pass: refusals this driver defines itself
# =============================================================================


def test_a_non_path_folder_argument_is_refused_by_name():
    with pytest.raises(run_module.RunError, match="NoneType"):
        run_module.run_folder(None)


def test_a_nonexistent_folder_is_refused(tmp_path, capsys):
    missing = tmp_path / "does_not_exist"

    exit_code = run_module.main([str(missing)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("error: ")
    assert str(missing) in captured.err


def test_a_file_given_as_the_folder_is_refused(tmp_path, capsys):
    not_a_dir = tmp_path / "im_a_file.txt"
    not_a_dir.write_text("not a folder", newline="\n")

    exit_code = run_module.main([str(not_a_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("error: ")


def test_a_missing_project_yaml_is_named(tmp_path, write_example, capsys):
    project_dir = _build_project(tmp_path, write_example)
    (project_dir / "project.yaml").unlink()

    exit_code = run_module.main([str(project_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "project.yaml" in captured.err


def test_a_missing_materials_csv_is_named(tmp_path, write_example, capsys):
    project_dir = _build_project(tmp_path, write_example)
    (project_dir / "materials.csv").unlink()

    exit_code = run_module.main([str(project_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "materials.csv" in captured.err


def test_a_chain_refusal_prints_error_and_exit_1_with_no_traceback(tmp_path, write_example, capsys):
    """SceneError (a ValueError subclass) from a broken scene reaches
    main() as a named, one-line refusal, not a traceback (spec: 'any
    refusal the chain raises')."""
    project_dir = _build_project(tmp_path, write_example)
    # Degenerate: a materials.csv gap (a role validate_scene rejects).
    (project_dir / "materials.csv").write_text(
        "modifier,primitive_type,role,note\n"
        "wall_paint,plastic,wall,\n"
        "ceiling_tile,plastic,ceiling,\n"
        "floor_carpet,plastic,floor,\n"
        # glass_clear's role is missing entirely.
    , newline="\n")

    exit_code = run_module.main([str(project_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("error: ")
    assert "Traceback" not in captured.err


def test_the_real_runner_searches_the_scene_and_model_folders(tmp_path, write_example, monkeypatch):
    """Radiance commands run in the process's own working directory, so a
    scene's relative .cal/.xml reference resolves only through RAYPATH: the
    real runner gets the scene files' folder(s) and the prepared model
    folder as extra search directories."""
    project_dir = _build_project(tmp_path, write_example)
    seen = {}

    class _Stop(Exception):
        pass

    def fake_subprocess_runner(search_dirs=()):
        seen["search_dirs"] = tuple(search_dirs)
        raise _Stop

    def fake_find_binaries(search_path=None, names=None):
        # the real signature, so a wrong positional call is seen
        seen["search_path"] = search_path
        return {}

    monkeypatch.setattr(run_module, "find_binaries", fake_find_binaries)
    monkeypatch.setattr(run_module, "subprocess_runner", fake_subprocess_runner)

    with pytest.raises(_Stop):
        run_module.run_folder(project_dir, "base_case", out=io.StringIO())

    assert seen["search_path"] is None
    assert seen["search_dirs"] == (project_dir.resolve(), (project_dir / "_run").resolve())


def test_processes_is_passed_through_to_the_run(tmp_path, write_example, monkeypatch):
    project_dir = _build_project(tmp_path, write_example)
    seen = {}

    class _Stop(Exception):
        pass

    def fake_run_option1(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(run_module, "run_option1", fake_run_option1)

    with pytest.raises(_Stop):
        run_module.run_folder(
            project_dir, "base_case", engine=_engine_for(project_dir), out=io.StringIO(),
            processes=6,
        )
    assert seen["processes"] == 6

    seen.clear()
    monkeypatch.setattr(run_module, "run_folder", lambda folder, case, **kw: seen.update(kw))
    assert run_module.main([str(project_dir), "--processes", "3"]) == 0
    assert seen["processes"] == 3
    assert run_module.main([str(project_dir)]) == 0
    assert seen["processes"] == 1


@pytest.mark.parametrize("bad", ["0", "-2", "two", "1.5"])
def test_a_bad_processes_value_is_refused_by_the_parser(tmp_path, bad, capsys):
    with pytest.raises(SystemExit) as exc_info:
        run_module.main([str(tmp_path), "--processes", bad])
    assert exc_info.value.code == 2
    assert "--processes" in capsys.readouterr().err


@pytest.mark.parametrize("make_error", [
    lambda: __import__("yaml").YAMLError("mapping values are not allowed here\n  line 3"),
    lambda: __import__("shapely.errors", fromlist=["GEOSException"]).GEOSException(
        "TopologyException: side location conflict\nat 1 2"),
    lambda: OSError(28, "No space left on device"),
])
def test_main_prints_yaml_geometry_and_os_errors_as_one_line(tmp_path, capsys, monkeypatch, make_error):
    error = make_error()

    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(run_module, "run_folder", boom)

    assert run_module.main([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert err.count("\n") == 1


def _summary_of(score):
    from leed_daylight.drivers.run_option1 import Option1Result

    result = Option1Result(
        case="base_case", occupied_hours=3650, spaces=(),
        sda_area_pct=20.0, sda_point_pct=20.0, ase_area_pct=0.0, ase_point_pct=0.0,
        headline="floor_area", score=score, narrative_spaces=(),
        blind_schedule={}, warnings=(),
    )
    out = io.StringIO()
    run_module._print_summary(result, pathlib.Path("results.json"), out=out)
    return out.getvalue()


def test_a_score_below_every_band_prints_the_reason_not_none():
    from leed_daylight.metrics.scoring import Option1Score

    printed = _summary_of(Option1Score(
        points=0, band=None, additional_point_awarded=False,
        reason="result is 20.0 %, below the lowest band",
    ))

    assert "None" not in printed
    assert "score: 0 points (below the lowest band)" in printed


def test_a_score_in_a_band_prints_the_band():
    from leed_daylight.metrics.scoring import Option1Score

    printed = _summary_of(Option1Score(
        points=2, band="55.0 % or more", additional_point_awarded=False,
        reason="result is 60.0 %, which reaches the 55 % band",
    ))

    assert "score: 2 points (55.0 % or more)" in printed


def test_points_csv_is_written_beside_the_results(tmp_path, write_example):
    project_dir = _build_project(tmp_path, write_example)
    engine = _engine_for(project_dir)
    _, n_points = _prepared_shape(project_dir)

    results_path = run_module.run_folder(
        project_dir, "base_case", engine=engine, rad=None, out=io.StringIO(),
    )

    lines = (results_path.parent / "points.csv").read_text().splitlines()
    assert lines[0] == "room,storey,x,y,z,da_pct,sda_pass,ase_hours,ase_exceeds"
    assert len(lines) - 1 == n_points
    first = lines[1].split(",")
    assert first[0] == ROOM
    assert first[1] == "0"
    assert float(first[5]) == pytest.approx(100.0)  # the fake sky lux always passes
    assert first[6] == "yes"
    assert first[7] == "0"
    assert first[8] == "no"


def test_a_second_case_does_not_overwrite_the_first(tmp_path, write_example):
    project_dir = _build_project(tmp_path, write_example)
    engine = _engine_for(project_dir)

    first = run_module.run_folder(project_dir, "base_case", engine=engine, rad=None, out=io.StringIO())
    second = run_module.run_folder(project_dir, "second_case", engine=engine, rad=None, out=io.StringIO())

    assert first != second
    assert first.is_file() and second.is_file()
    assert json.loads(first.read_text())["case"] == "base_case"
    assert json.loads(second.read_text())["case"] == "second_case"


def test_points_csv_agrees_with_results_json_aggregates(tmp_path, write_example):
    """F10: a half-pass fake engine -- some points pass sDA and some fail
    it, a few points exceed ASE and the rest do not -- so points.csv's own
    share of "yes" rows can be checked against results.json's
    point-weighted aggregates, not just that both files exist."""
    project_dir = _build_project(tmp_path, write_example)
    group_id, n_points = _prepared_shape(project_dir)
    engine = HalfPassEngine(FULL_MASK_LEN, group_id, n_points, N_OCCUPIED)

    results_path = run_module.run_folder(
        project_dir, "base_case", engine=engine, rad=None, out=io.StringIO(),
    )
    payload = json.loads(results_path.read_text())

    rows = [
        line.split(",")
        for line in (results_path.parent / "points.csv").read_text().splitlines()[1:]
    ]
    assert len(rows) == n_points

    sda_pct_from_csv = 100.0 * sum(row[6] == "yes" for row in rows) / n_points
    ase_pct_from_csv = 100.0 * sum(row[8] == "yes" for row in rows) / n_points

    # Sanity: a real half-pass split, not all-or-nothing like FakeEngine's.
    assert 0.0 < sda_pct_from_csv < 100.0
    assert 0.0 < ase_pct_from_csv < 100.0

    assert sda_pct_from_csv == pytest.approx(payload["sda_point_pct"])
    assert ase_pct_from_csv == pytest.approx(payload["ase_point_pct"])


def test_results_json_records_the_input_fingerprints(tmp_path, write_example):
    from leed_daylight.project.inputs import input_fingerprints

    project_dir = _build_project(tmp_path, write_example)
    results_path = run_module.run_folder(
        project_dir, "base_case", engine=_engine_for(project_dir), rad=None, out=io.StringIO(),
    )

    payload = json.loads(results_path.read_text())
    project = load_project(project_dir / "project.yaml")
    assert payload["inputs"] == input_fingerprints(project, project_dir)


# =============================================================================
# roof-glass design 5.9: results.json lists every window group
# =============================================================================


def test_results_json_lists_every_window_group(tmp_path, write_example):
    project_dir = _build_project(tmp_path, write_example)
    engine = _engine_for(project_dir)

    results_path = run_module.run_folder(
        project_dir, "base_case", engine=engine, rad=None, out=io.StringIO(),
    )

    payload = json.loads(results_path.read_text())
    # the one south window: azimuth bin 180, vertical, with a blind
    assert payload["groups"] == [{
        "group_id": engine._group_id, "room": ROOM, "kind": "window",
        "tilt_deg": 90.0, "azimuth_bin_deg": 180.0, "blind": True,
    }]


def test_a_diffusing_group_is_written_with_no_blind_and_no_closed_hours(
    tmp_path, write_example, monkeypatch,
):
    project_dir = _build_project(tmp_path, write_example)
    engine = _engine_for(project_dir)
    diffusing = WindowGroup(
        group_id="room_north_dif", room=ROOM, storey=0, azimuth_bin_deg=None,
        openings=(), outward_normals=(), area_m2=3.0, kind="diffusing", tilt_deg=0.0,
    )
    real_prepare = run_module.prepare

    def prepare_with_a_diffuser(*args, **kwargs):
        prepared = real_prepare(*args, **kwargs)
        return dataclasses.replace(prepared, groups=prepared.groups + (diffusing,))

    monkeypatch.setattr(run_module, "prepare", prepare_with_a_diffuser)
    idx, direct = engine._direct[engine._group_id]
    engine._direct["room_north_dif"] = (idx, np.zeros_like(direct))
    engine._sky["room_north_dif"] = (idx, np.zeros_like(direct), None)

    results_path = run_module.run_folder(
        project_dir, "base_case", engine=engine, rad=None, out=io.StringIO(),
    )

    payload = json.loads(results_path.read_text())
    by_id = {g["group_id"]: g for g in payload["groups"]}
    assert list(by_id) == [engine._group_id, "room_north_dif"]
    assert by_id["room_north_dif"] == {
        "group_id": "room_north_dif", "room": ROOM, "kind": "diffusing",
        "tilt_deg": 0.0, "azimuth_bin_deg": None, "blind": False,
    }
    assert payload["blinds_closed_hours"]["room_north_dif"] == 0
    assert '"azimuth_bin_deg": null' in results_path.read_text()
