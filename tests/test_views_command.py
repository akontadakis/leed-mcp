"""The `views` command (QV spec 1.3, 6.7-6.9).

A fictional 6 m x 4 m x 3 m room, one window in its south wall (x 2-4 m,
z 1-2 m), eye height 1.1 m. A FAKE tracer answers analytically: a ray
heading south that meets the window opening passes the glass and then
leaves the scene; any other ray hits a wall. No Radiance binary runs.
Every point sees the window within 3 x head height, and more than half of
each point's reaching rays go up or level (open sky), so every point has a
Quality View.
"""
from __future__ import annotations

import io
import json
import math

import numpy as np
import pytest
import yaml

import leed_daylight.drivers.views as views_module
from conftest import symlink_or_skip
from leed_daylight.engine.view_trace import MISS, UNUSED, RaySegments

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
floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,view_preserving_shades,merge_into,include_base_case,views_base,reason
L0,Room North,office,yes,24,3.0,2.0,0.0,no,,1,1,
"""

RULES_YAML = """\
version: "Example rules v0"
max_points: 3
option1:
  sda: {lux: 300, fraction: 0.50, bands: [{min_pct: 40.0, points: 1}]}
  ase: {lux: 1000, hours: 250, per_space_narrative_trigger_pct: 10.0}
option2:
  bands: UNVERIFIED
  lux_min: 300
  lux_max: 3000
glare_control_required: true
views:
  version: "Example views rules v0"
  quality_definition: "type2_and_type3"
  threshold_pct: 75.0
  points: 1
  vlt_min_exclusive: 0.40
  head_height_multiple: 3.0
  object_distance_m: 7.5
  obstruction_exception_m: 0.30
  method: {distance_statistic: "median", min_outside_distance_m: 2.0, panorama_vertical_deg: 60.0}
"""

VIEWS = {"eye_height_m": 1.1, "eye_height_source": "invented for a test", "angular_step_deg": 5.0}

#: A second, disconnected room (x 10-16 m), with no window at all, for the
#: no-reaching-ray test below. It needs no walls: validate_scene does not
#: require them, and detect_rooms finds a room from its floor alone.
SCENE_RAD_SECOND_ROOM = """\
floor_carpet polygon floor_room_core
0
0
12  10 0 0   16 0 0   16 4 0   10 4 0

ceiling_tile polygon ceiling_room_core
0
0
12  10 0 3   10 4 3   16 4 3   16 0 3
"""

SPACES_CSV_SECOND_ROOM = "L0,Room Core,office,yes,24,13.0,2.0,0.0,no,,1,1,\n"


class FakeTracer:
    radiance_version = None

    def __init__(self):
        self.trace_calls = 0

    def octree(self, scene, materials, work_dir):
        work_dir.mkdir(parents=True, exist_ok=True)
        path = work_dir / "views.oct"
        path.write_bytes(scene.read_bytes() + materials.read_bytes())
        return path

    def trace(self, octree, origins, directions, *, pass_through, exterior, skip_within_m, max_segments):
        self.trace_calls += 1
        assert exterior == frozenset({"glass_clear"})
        n = origins.shape[0]
        modifier = np.full((n, max_segments), UNUSED, dtype=np.int32)
        surface = np.full((n, max_segments), -1, dtype=np.int32)
        hit = np.full((n, max_segments, 3), np.nan)
        south = directions[:, 1] < 0
        t = np.where(south, -origins[:, 1] / np.where(south, directions[:, 1], 1.0), 1.0)
        at = origins + t[:, None] * directions
        window = south & (at[:, 0] >= 2) & (at[:, 0] <= 4) & (at[:, 2] >= 1) & (at[:, 2] <= 2)
        modifier[:, 0] = np.where(window, 0, 1)
        surface[:, 0] = np.where(window, 0, 1)
        hit[:, 0] = at
        modifier[window, 1] = MISS
        return RaySegments(("glass_clear", "wall_paint"), ("window_ext", "wall_room_north_south"),
                           modifier, surface, hit)


@pytest.fixture
def folder(tmp_path, write_example):
    def mutate(d):
        d["views"] = dict(VIEWS)
        return d
    project_dir = write_example(tmp_path, mutate).parent
    (project_dir / "scene.rad").write_text(SCENE_RAD, newline="\n")
    (project_dir / "materials.csv").write_text(MATERIALS_CSV, newline="\n")
    (project_dir / "spaces.csv").write_text(SPACES_CSV, newline="\n")
    (project_dir / "credit_rules.yaml").write_text(RULES_YAML, newline="\n")
    return project_dir


@pytest.fixture
def two_room_folder(tmp_path, write_example):
    def mutate(d):
        d["views"] = dict(VIEWS)
        return d
    project_dir = write_example(tmp_path, mutate).parent
    (project_dir / "scene.rad").write_text(SCENE_RAD + SCENE_RAD_SECOND_ROOM, newline="\n")
    (project_dir / "materials.csv").write_text(MATERIALS_CSV, newline="\n")
    (project_dir / "spaces.csv").write_text(SPACES_CSV + SPACES_CSV_SECOND_ROOM, newline="\n")
    (project_dir / "credit_rules.yaml").write_text(RULES_YAML, newline="\n")
    return project_dir


class TwoRoomTracer(FakeTracer):
    """Like FakeTracer, but only a ray whose origin is in Room North (x <
    6 m) can ever meet the window; a ray from Room Core (no window) always
    hits a wall, whatever direction it takes."""

    def trace(self, octree, origins, directions, *, pass_through, exterior, skip_within_m, max_segments):
        self.trace_calls += 1
        assert exterior == frozenset({"glass_clear"})
        n = origins.shape[0]
        modifier = np.full((n, max_segments), UNUSED, dtype=np.int32)
        surface = np.full((n, max_segments), -1, dtype=np.int32)
        hit = np.full((n, max_segments, 3), np.nan)
        south = directions[:, 1] < 0
        t = np.where(south, -origins[:, 1] / np.where(south, directions[:, 1], 1.0), 1.0)
        at = origins + t[:, None] * directions
        in_room_north = origins[:, 0] < 6.0
        window = in_room_north & south & (at[:, 0] >= 2) & (at[:, 0] <= 4) & (at[:, 2] >= 1) & (at[:, 2] <= 2)
        modifier[:, 0] = np.where(window, 0, 1)
        surface[:, 0] = np.where(window, 0, 1)
        hit[:, 0] = at
        modifier[window, 1] = MISS
        return RaySegments(("glass_clear", "wall_paint"), ("window_ext", "wall_room_north_south"),
                           modifier, surface, hit)


def _run(folder, tracer=None, case=None):
    out = io.StringIO()
    path = views_module.run_views(folder, case, tracer=tracer or FakeTracer(), out=out)
    return path, out.getvalue()


def test_the_command_writes_results_points_and_plans(folder):
    path, printed = _run(folder)

    assert path == folder / "_run" / "views" / "base" / "views_results.json"
    result = json.loads(path.read_text())
    (space,) = result["spaces"]
    assert space["room"] == "Room North"
    assert space["quality_pct"] == 100.0
    assert result["quality_area_pct"] == 100.0
    assert result["score"]["points"] == 1
    assert result["settings"]["eye_height_m"] == 1.1
    assert result["settings"]["max_interior_glass"] == 1
    assert any("atria" in w for w in result["warnings"])
    points_csv = (path.parent / "points.csv").read_text().splitlines()
    assert points_csv[0] == "room,storey,x,y,z,type2,type3,quality,median_outside_m,reaching_rays,limit_rays"
    assert len(points_csv) - 1 == space["points"]
    for layer in ("type2", "type3", "quality"):
        assert (path.parent / f"plan_0_{layer}.svg").is_file()
    assert "Quality Views" in printed and "100.0" in printed


def test_results_json_has_no_nan(folder):
    path, _ = _run(folder)

    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text


def test_a_second_run_reuses_the_cached_rooms(folder):
    _run(folder)
    tracer = FakeTracer()

    _run(folder, tracer)

    assert tracer.trace_calls == 0


def test_changing_a_setting_recomputes_the_cached_rooms(folder):
    _run(folder)
    raw = yaml.safe_load((folder / "project.yaml").read_text())
    raw["views"]["max_interior_glass"] = 0
    (folder / "project.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), newline="\n")
    tracer = FakeTracer()

    _run(folder, tracer)

    assert tracer.trace_calls > 0


def test_a_different_radiance_version_recomputes_the_cached_rooms(folder):
    _run(folder)
    tracer = FakeTracer()
    tracer.radiance_version = "RADIANCE 6.2 fake"

    _run(folder, tracer)

    assert tracer.trace_calls > 0


def test_a_project_without_a_views_section_is_refused(folder):
    raw = yaml.safe_load((folder / "project.yaml").read_text())
    del raw["views"]
    (folder / "project.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), newline="\n")

    with pytest.raises(views_module.ViewsError, match="no views: section"):
        _run(folder)


def test_spaces_without_views_columns_are_refused(folder):
    (folder / "spaces.csv").write_text(SPACES_CSV.replace(",views_base", "").replace(",1,1,\n", ",1,\n"), newline="\n")

    with pytest.raises(ValueError, match=r"no views_<case> column"):
        _run(folder)


def test_main_prints_a_one_line_error_and_returns_1(folder, capsys, monkeypatch):
    raw = yaml.safe_load((folder / "project.yaml").read_text())
    del raw["views"]
    (folder / "project.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), newline="\n")

    assert views_module.main([str(folder)]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_a_room_with_no_reaching_ray_gives_zero_percent_and_valid_json(two_room_folder):
    path, _ = _run(two_room_folder, tracer=TwoRoomTracer())

    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    result = json.loads(text)
    core = next(s for s in result["spaces"] if s["room"] == "Room Core")
    assert core["quality_pct"] == 0.0
    points_csv = (path.parent / "points.csv").read_text().splitlines()[1:]
    core_rows = [line.split(",") for line in points_csv if line.split(",")[0] == "Room Core"]
    assert core_rows
    for row in core_rows:
        assert row[8] == ""


def test_a_folder_inside_the_repository_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(views_module, "REPO_ROOT", tmp_path)
    inside = tmp_path / "project"
    inside.mkdir()

    with pytest.raises(views_module.ViewsError, match="inside the repository"):
        views_module.run_views(inside, tracer=FakeTracer(), out=io.StringIO())


def test_a_plan_draws_a_room_s_atrium_hole(tmp_path):
    import types

    from leed_daylight.project.layout import LaidOutSpace, SpaceLayout
    from leed_daylight.scene.storeys import Room

    room = Room(
        room_id="S0_R0", storey=0, level_z=0.0,
        outline=np.array([(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)]),
        area_m2=32.0, ceiling_height_m=None,
        holes=(np.array([(2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)]),),
    )
    grid = types.SimpleNamespace(room="a", storey=0, spacing_m=0.6, points=np.array([[1.0, 1.0, 1.2]]))
    flags = np.array([True])
    result = types.SimpleNamespace(type2=flags, type3=flags, quality=flags)
    space = types.SimpleNamespace(area_m2=32.0, type2_pct=100.0, type3_pct=100.0, quality_pct=100.0)
    layout = SpaceLayout((LaidOutSpace(
        name="a", storey=0, level_z=0.0, pieces=(room,), area_m2=32.0, origin="detected",
        grid_source=(None,), rows=("a",),
    ),))
    views_module._write_plans(tmp_path, [grid], [result], [space], layout, 0.0)
    svg = (tmp_path / "plan_0_type2.svg").read_text()
    assert svg.count('class="room"') == 2


def test_without_a_ground_surface_ground_is_the_lowest_floor_and_warned(folder):
    path, printed = _run(folder)

    result = json.loads(path.read_text())
    assert "no ground surface: ground taken at the lowest floor level, z = 0.00" in result["warnings"]
    assert "no ground surface" in printed


def test_a_datum_height_model_measures_downward_rays_to_its_lowest_floor(folder):
    """Floors at +100 m and no ground: the ground plane is z = 100, not 0,
    so a downward ray's outside distance is not inflated by a hundred metres."""
    lifted = []
    for line in SCENE_RAD.splitlines():
        parts = line.split()
        if len(parts) == 13 and parts[0] == "12":
            values = [float(v) for v in parts[1:]]
            values[2::3] = [z + 100.0 for z in values[2::3]]
            line = "12  " + "  ".join(" ".join(f"{v:g}" for v in values[i:i + 3]) for i in range(0, 12, 3))
        lifted.append(line)
    (folder / "scene.rad").write_text("\n".join(lifted) + "\n", newline="\n")
    (folder / "spaces.csv").write_text(SPACES_CSV.replace(",0.0,no,", ",100.0,no,"), newline="\n")

    class LiftedTracer(FakeTracer):
        def trace(self, octree, origins, directions, **kw):
            shifted = origins - np.array([0.0, 0.0, 100.0])
            segs = super().trace(octree, shifted, directions, **kw)
            return RaySegments(segs.modifiers, segs.surfaces, segs.modifier, segs.surface,
                               segs.hit + np.array([0.0, 0.0, 100.0]))

    path, _ = _run(folder, LiftedTracer())

    result = json.loads(path.read_text())
    assert result["settings"]["ground_z_m"] == 100.0
    assert "no ground surface: ground taken at the lowest floor level, z = 100.00" in result["warnings"]


@pytest.mark.parametrize("reason, shown", [("server room", "(server room)"), ("", "(no reason given)")])
def test_regularly_occupied_spaces_left_out_of_the_views_case_are_warned(two_room_folder, reason, shown):
    header, first = SPACES_CSV.splitlines()
    (two_room_folder / "spaces.csv").write_text(
        header + ",views_reason\n" + first + ",\n"
        + SPACES_CSV_SECOND_ROOM.replace(",1,1,\n", f",1,0,,{reason}\n")
    , newline="\n")

    path, printed = _run(two_room_folder, tracer=TwoRoomTracer())

    result = json.loads(path.read_text())
    (warning,) = [w for w in result["warnings"] if "not ticked in views_base" in w]
    assert f"'Room Core' {shown}" in warning
    assert "certain space types" in warning
    assert warning in printed
    assert [s["room"] for s in result["spaces"]] == ["Room North"]


def test_no_exclusion_warning_when_every_occupied_space_is_ticked(folder):
    path, _ = _run(folder)

    assert not any("not ticked in views_" in w for w in json.loads(path.read_text())["warnings"])


def test_views_run_without_any_daylight_case(folder, monkeypatch):
    """prepare frames every exterior pane whatever the case, so the views
    command needs no include_<case>: spaces with none still run."""
    import dataclasses

    real = views_module.load_spaces
    monkeypatch.setattr(
        views_module, "load_spaces", lambda path: [dataclasses.replace(s, include={}) for s in real(path)]
    )

    path, _ = _run(folder)

    assert json.loads(path.read_text())["quality_area_pct"] == 100.0


def test_views_ignore_the_daylight_case_ticks(folder):
    (folder / "spaces.csv").write_text(SPACES_CSV.replace(",no,,1,1,", ",no,,0,1,"), newline="\n")

    path, _ = _run(folder)

    assert json.loads(path.read_text())["quality_area_pct"] == 100.0


def test_an_angular_step_that_truncates_the_band_is_refused(folder, capsys):
    raw = yaml.safe_load((folder / "project.yaml").read_text())
    raw["views"]["angular_step_deg"] = 4.0
    (folder / "project.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), newline="\n")

    with pytest.raises(ValueError, match=r"must divide half the vertical span \(30 deg\)"):
        _run(folder)


def test_the_plans_carry_the_projects_north_rotation(folder):
    raw = yaml.safe_load((folder / "project.yaml").read_text())
    raw["site"]["north_rotation_deg"] = 30.0
    (folder / "project.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), newline="\n")

    path, _ = _run(folder)

    assert 'class="north"' in (path.parent / "plan_0_quality.svg").read_text()
    assert "rotate(-30)" in (path.parent / "plan_0_quality.svg").read_text()


def test_main_passes_processes_to_the_tracer(folder, monkeypatch):
    seen = {}

    def fake_run_views(folder, case=None, *, processes=1, **kw):
        seen["processes"] = processes
        return folder

    monkeypatch.setattr(views_module, "run_views", fake_run_views)

    assert views_module.main([str(folder), "--processes", "8"]) == 0
    assert seen["processes"] == 8
    assert views_module.main([str(folder)]) == 0
    assert seen["processes"] == 1


def test_main_refuses_processes_below_one(folder, capsys):
    assert views_module.main([str(folder), "--processes", "0"]) == 1
    assert "processes" in capsys.readouterr().err


def test_the_glazing_family_alone_changes_the_room_fingerprint(tmp_path):
    """A see-through modifier moved from glazing_interior to a plain role
    (still marked clear) keeps every other fingerprint field -- pass_through,
    exterior, classes -- but it is no longer an interior glass layer, so the
    room must be recomputed."""
    from leed_daylight.build.grid import RoomGrid
    from leed_daylight.build.view_rays import panorama

    octree = tmp_path / "views.oct"
    octree.write_bytes(b"fake octree")
    grid = RoomGrid(room="Room North", points=np.array([[3.0, 2.0, 1.1]]), spacing_m=0.6,
                    cell_area_m2=0.36, room_area_m2=24.0, storey=0, level_z=0.0)
    directions, weights = panorama(30.0, 60.0)
    tracer = FakeTracer()
    settings = dict(
        classes={"glass_clear": "clear", "glass_part": "clear", "wall_paint": "blocks"},
        tops={"window_ext": 2.0}, pass_through=frozenset({"glass_clear", "glass_part"}),
        exterior=frozenset({"glass_clear"}), skip=2.0, z_ground=0.0, max_interior_glass=1,
        head_multiple=3.0, object_distance=7.5, cache_dir=tmp_path / "engine",
    )
    roles = {"glass_clear": "glazing", "glass_part": "glazing_interior", "wall_paint": "wall"}

    def run(role_of_part):
        views_module._room_points(tracer, octree, grid, directions, weights,
                                  roles={**roles, "glass_part": role_of_part}, **settings)
        return tracer.trace_calls

    first = run("glazing_interior")
    assert run("glazing_interior") == first
    assert run("fixed_shading") > first


def _yaml_error():
    try:
        yaml.safe_load("views: [unclosed\nnext: 1\n")
    except yaml.YAMLError as exc:
        return exc
    raise AssertionError("expected a YAML error")


@pytest.mark.parametrize("make_error", [
    _yaml_error,
    lambda: __import__("shapely.errors", fromlist=["GEOSException"]).GEOSException(
        "TopologyException: side location conflict\nat 1 2"),
    lambda: PermissionError(13, "Permission denied", "/nowhere/views_results.json"),
    lambda: OSError(28, "No space left on device"),
])
def test_main_prints_yaml_geometry_and_os_errors_as_one_line(folder, capsys, monkeypatch, make_error):
    error = make_error()

    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(views_module, "run_views", boom)

    assert views_module.main([str(folder)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert err.count("\n") == 1


def test_the_real_tracer_needs_only_oconv_and_rtrace_and_searches_the_scene_folder(folder, monkeypatch):
    """Quality Views runs only oconv and rtrace, so it checks only those two
    binaries; relative .cal/.xml references resolve through RAYPATH as in
    the run command."""
    seen = {}

    class _Stop(Exception):
        pass

    def fake_find_binaries(search_path=None, names=None):
        # the real signature: a positional tuple would land in search_path
        seen["search_path"] = search_path
        seen["names"] = tuple(names) if names is not None else None
        return {}

    def fake_subprocess_runner(search_dirs=()):
        seen["search_dirs"] = tuple(search_dirs)
        raise _Stop

    monkeypatch.setattr(views_module, "find_binaries", fake_find_binaries)
    monkeypatch.setattr(views_module, "subprocess_runner", fake_subprocess_runner)

    with pytest.raises(_Stop):
        views_module.run_views(folder, out=io.StringIO())

    assert seen["search_path"] is None
    assert seen["names"] == ("oconv", "rtrace")
    assert folder.resolve() in seen["search_dirs"]


def test_room_detection_warnings_reach_the_views_results(folder, monkeypatch):
    real = views_module.scene_rooms

    def detect_with_warning(project, prims, roles):
        detection = real(project, prims, roles)
        return detection.__class__(detection.rooms, detection.warnings + ("storey height check: invented",))

    monkeypatch.setattr(views_module, "scene_rooms", detect_with_warning)

    path, printed = _run(folder)

    assert "storey height check: invented" in json.loads(path.read_text())["warnings"]
    assert "storey height check: invented" in printed


def test_the_results_file_records_the_input_fingerprints(folder):
    from leed_daylight.project.config import load_project
    from leed_daylight.project.inputs import input_fingerprints

    path, _ = _run(folder)

    result = json.loads(path.read_text())
    assert result["inputs"] == input_fingerprints(load_project(folder / "project.yaml"), folder)


def test_an_input_edited_during_the_run_leaves_the_results_stale(folder):
    """Final review I2: the fingerprints describe the inputs as read, not as
    they are when the results are written."""
    from leed_daylight.project.config import load_project
    from leed_daylight.project.inputs import stale_inputs

    class EditingTracer(FakeTracer):
        def trace(self, *args, **kwargs):
            rules = folder / "credit_rules.yaml"
            rules.write_text(rules.read_text() + "# edited during the run\n", newline="\n")
            return super().trace(*args, **kwargs)

    path, _ = _run(folder, tracer=EditingTracer())

    recorded = json.loads(path.read_text())["inputs"]
    assert stale_inputs(recorded, load_project(folder / "project.yaml"), folder) == ["credit_rules.yaml"]


# =============================================================================
# roof-glass design 5.11: roof glass and diffusing glass block views
# =============================================================================

#: An invented 2 m x 2 m flat skylight 0.5 m above the ceiling, glazed with
#: the same glass_clear as the south window.
SKYLIGHT_RAD = """
glass_clear polygon skylight_roof
0
0
12  2 1 3.5   4 1 3.5   4 3 3.5   2 3 3.5
"""

#: SCENE_RAD without its south window.
SCENE_RAD_NO_WINDOW = SCENE_RAD.split("glass_clear polygon window_ext")[0]

MATERIALS_CSV_CLEAR_TAG = """\
modifier,primitive_type,role,note,view
wall_paint,plastic,wall,,
ceiling_tile,plastic,ceiling,,
floor_carpet,plastic,floor,,
glass_clear,glass,glazing,,clear
"""


def _skylight_folder(tmp_path, write_example, *, scene=SCENE_RAD + SKYLIGHT_RAD, materials=MATERIALS_CSV):
    def mutate(d):
        d["views"] = dict(VIEWS)
        d["glazing"]["dirt_factor"] = {"vertical": 0.90, "skylight": 0.85}
        d["glazing"]["frame_share"]["skylight"] = 0.10
        d["glazing"]["skylight_well"] = "modelled"
        return d
    project_dir = write_example(tmp_path, mutate).parent
    (project_dir / "scene.rad").write_text(scene, newline="\n")
    (project_dir / "materials.csv").write_text(materials, newline="\n")
    (project_dir / "spaces.csv").write_text(SPACES_CSV, newline="\n")
    (project_dir / "credit_rules.yaml").write_text(RULES_YAML, newline="\n")
    return project_dir


class RecordingTracer(FakeTracer):
    """FakeTracer that records the see-through sets it was given; the
    analytic answer is the south window's, as in FakeTracer."""

    def __init__(self):
        super().__init__()
        self.seen = {}

    def trace(self, octree, origins, directions, *, pass_through, exterior, **kw):
        self.seen["pass_through"] = pass_through
        self.seen["exterior"] = exterior
        return super().trace(octree, origins, directions, pass_through=pass_through,
                             exterior=frozenset({"glass_clear"}), **kw)


def _views_scene(folder):
    from leed_daylight.scene.reader import read_scene

    path = folder / "_run" / "views" / "model" / "views_scene.rad"
    return {p.name: p.modifier for p in read_scene([path], "m")}


def test_views_prepare_the_model_without_the_ambient_check(folder, monkeypatch):
    seen = {}
    real = views_module.prepare

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(views_module, "prepare", spy)

    _run(folder)

    assert seen.get("check_ambient") is False


def test_a_skylight_sharing_the_window_glass_blocks_views_and_the_window_still_sees(
    tmp_path, write_example,
):
    from leed_daylight.build.view_model import BLOCKED_VIEW_MODIFIER

    folder = _skylight_folder(tmp_path, write_example)
    tracer = RecordingTracer()

    path, _ = _run(folder, tracer)

    modifiers = _views_scene(folder)
    assert modifiers["skylight_roof"] == BLOCKED_VIEW_MODIFIER
    assert modifiers["window_ext"] == "glass_clear"
    assert "glass_clear" in tracer.seen["exterior"]
    assert BLOCKED_VIEW_MODIFIER not in tracer.seen["pass_through"]
    assert json.loads(path.read_text())["quality_area_pct"] == 100.0


def test_a_clear_tag_on_glass_a_skylight_uses_is_warned_once(tmp_path, write_example):
    folder = _skylight_folder(tmp_path, write_example, materials=MATERIALS_CSV_CLEAR_TAG)

    path, printed = _run(folder, RecordingTracer())

    warnings = [w for w in json.loads(path.read_text())["warnings"] if "blocks views" in w]
    assert len(warnings) == 1
    assert warnings[0].startswith("glass_clear: marked 'clear' in materials.csv")
    assert "skylight_roof" in warnings[0]
    assert warnings[0] in printed


def test_a_storey_whose_only_clear_glass_is_a_skylight_is_warned(tmp_path, write_example):
    folder = _skylight_folder(tmp_path, write_example, scene=SCENE_RAD_NO_WINDOW + SKYLIGHT_RAD)

    path, _ = _run(folder, RecordingTracer())

    assert (
        "storey 0: ticked spaces but no exterior vision glass on this storey"
        in json.loads(path.read_text())["warnings"]
    )


def test_the_method_notes_say_roof_and_diffusing_glass_are_not_vision_glazing(folder):
    path, _ = _run(folder)

    notes = json.loads(path.read_text())["method_notes"]
    assert views_module.ROOF_GLASS_NOTE in notes
    assert "not vision glazing" in views_module.ROOF_GLASS_NOTE


@pytest.mark.parametrize("parts", [
    ("_run",), ("_run", "views"), ("_run", "views", "prepared"), ("_run", "views", "prepared", "blinds"),
    ("_run", "views", "prepared", "grids"), ("_run", "views", "engine"),
])
def test_views_never_delete_through_a_linked_parent(folder, tmp_path, parts):
    """Re-review OOS-1: a link planted at a folder the views run cleans or
    replaces files in is refused, by name, before anything there changes."""
    import shutil

    from leed_daylight.engine.jobs import UnsafePath

    _run(folder)
    inside = folder.joinpath(*parts)
    outside = tmp_path / "outside"
    shutil.move(inside, outside)
    symlink_or_skip(inside, outside)
    (outside / "keep.fingerprint.json").write_text("outside the project; never deleted\n", newline="\n")
    before = {p.relative_to(outside).as_posix(): p.read_bytes() for p in outside.rglob("*") if p.is_file()}

    with pytest.raises(UnsafePath) as caught:
        _run(folder)

    assert str(caught.value) == (
        f"{folder.resolve().joinpath(*parts)}: is a symbolic link; "
        "nothing is written or deleted through a link; delete it and try again"
    )
    assert {p.relative_to(outside).as_posix(): p.read_bytes() for p in outside.rglob("*") if p.is_file()} == before


# -- frosted interior glass with a blank primitive_type cell (final review I2) ----------------------

#: An invented frosted partition inside the room: a diffusing ``trans`` whose
#: diffuse transmittance (0.9 x 0.9) is well above the 0.40 view limit.
FROSTED_RAD = """
void trans glass_frosted
0
0
7  0.9 0.9 0.9  0 0  0.9 0

glass_frosted polygon partition_frosted
0
0
12  1 1 0.5   1 3 0.5   1 3 2.5   1 1 2.5
"""


def _frosted_folder(tmp_path, write_example, view):
    materials = MATERIALS_CSV_CLEAR_TAG.replace("glass_clear,glass,glazing,,clear", "glass_clear,glass,glazing,,")
    materials += f"glass_frosted,,glazing_interior,,{view}\n"
    return _skylight_folder(tmp_path, write_example, scene=SCENE_RAD + FROSTED_RAD, materials=materials)


def test_frosted_interior_glass_with_a_blank_type_cell_blocks_views(tmp_path, write_example):
    folder = _frosted_folder(tmp_path, write_example, view="")
    tracer = RecordingTracer()

    _run(folder, tracer)

    assert "glass_frosted" not in tracer.seen["pass_through"]
    assert "glass_clear" in tracer.seen["pass_through"]


def test_a_clear_view_on_frosted_interior_glass_with_a_blank_type_cell_is_refused(tmp_path, write_example):
    from leed_daylight.build.view_model import ViewModelError

    folder = _frosted_folder(tmp_path, write_example, view="clear")

    with pytest.raises(ViewModelError, match=(
        r"glass_frosted: a 'trans' with role 'glazing_interior' is frosted glass, never vision "
        r"glazing, so its view must be empty or 'blocks', got 'clear'"
    )):
        _run(folder, RecordingTracer())


def test_a_failure_while_writing_points_leaves_the_old_results_whole(folder, monkeypatch):
    path, _ = _run(folder)
    old = {name: (path.parent / name).read_bytes() for name in ("views_results.json", "points.csv")}
    project = folder / "project.yaml"
    project.write_text(project.read_text().replace("eye_height_m: 1.1", "eye_height_m: 1.2"), newline="\n")

    def fail(part, grids, results):
        part.write_text("room,\n", newline="\n")
        raise OSError("disk full")

    monkeypatch.setattr(views_module, "_write_points", fail)
    with pytest.raises(OSError):
        _run(folder)

    assert {name: (path.parent / name).read_bytes() for name in old} == old
    assert not list(path.parent.glob("*.part"))


def test_a_failure_replacing_points_never_leaves_new_results_beside_old_points(folder, monkeypatch):
    path, _ = _run(folder)
    old_points = (path.parent / "points.csv").read_bytes()
    project = folder / "project.yaml"
    project.write_text(project.read_text().replace("eye_height_m: 1.1", "eye_height_m: 1.2"), newline="\n")
    real = views_module.replace_from_part

    def fail_on_points(part, target):
        if target.name == "points.csv":
            raise PermissionError("points.csv is open in Excel")
        real(part, target)

    monkeypatch.setattr(views_module, "replace_from_part", fail_on_points)
    with pytest.raises(PermissionError):
        _run(folder)

    assert (path.parent / "points.csv").read_bytes() == old_points
    assert not (path.parent / "views_results.json").exists()
    assert not list(path.parent.glob("*.part"))
