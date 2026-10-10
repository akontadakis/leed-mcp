# tests/test_deliver_data.py
"""deliver/data.py: one case's results for the reports (reports spec 4, 5.2)."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from deliver_folder import CASE, ROOM, build_deliver_project, build_roof_glass_project, edit_results
from leed_daylight.build.glazing_class import PaneClass
from leed_daylight.build.groups import WindowGroup
from leed_daylight.constants import TOOL_VERSION
from leed_daylight.deliver.data import DeliverError, _group_line, load_case, read_points
from leed_daylight.project.config import load_project
from leed_daylight.project.inputs import input_fingerprints
from leed_daylight.scene.primitives import Primitive
from option2_folder import SCENE_RAD, build_option2_project


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    return build_deliver_project(tmp_path, write_example, write_wea)


def test_all_three_options_load(folder):
    case = load_case(folder)

    assert case.case == CASE
    assert set(case.options) == {"option1", "option2", "views"}
    assert dict(case.skipped) == {}
    (storey,) = case.storeys
    assert storey.label == "L0"
    assert storey.rooms == (ROOM,)
    assert case.storey_of(ROOM) == "L0"
    assert case.storeys_for("views") == (storey,)
    (ring,) = storey.outlines[ROOM]
    assert ring.shape[1] == 2
    (group,) = case.groups
    assert group.room == ROOM and group.storey_label == "L0"
    (segment,) = group.segments
    assert sorted(segment[:, 0].tolist()) == [2.0, 4.0]
    assert "glass_clear" in case.glazing
    assert case.rules["option2"]["bands"][0]["points"] == 1
    points = case.options["option1"].points
    assert {"da_pct", "sda_pass", "ase_hours", "ase_exceeds"} <= set(points.columns)
    assert points.columns["sda_pass"].dtype == bool
    assert points.columns["sda_pass"].any() and not points.columns["sda_pass"].all()


def test_a_missing_option_is_skipped_by_name(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=("option1",))

    case = load_case(folder)

    assert set(case.options) == {"option1"}
    assert "drivers.option2" in case.skipped["option2"]
    assert "drivers.views" in case.skipped["views"]


def test_one_stale_option_is_skipped_and_the_others_load(folder):
    edit_results(folder, "option2", lambda p: p["inputs"].update({"spaces.csv": "0" * 64}))

    case = load_case(folder)

    assert set(case.options) == {"option1", "views"}
    assert "out of date" in case.skipped["option2"]
    assert "spaces.csv" in case.skipped["option2"]


def test_results_without_fingerprints_are_skipped(folder):
    edit_results(folder, "views", lambda p: p.pop("inputs"))

    case = load_case(folder)

    assert "no input fingerprints" in case.skipped["views"]


def test_a_missing_points_csv_is_skipped_like_a_missing_results_file(folder):
    """F14: a results file whose points.csv is missing skips only that
    option, naming the rerun command -- it must not abort the whole case."""
    (folder / "_run" / "option2" / CASE / "points.csv").unlink()

    case = load_case(folder)

    assert set(case.options) == {"option1", "views"}
    assert "points.csv" in case.skipped["option2"]
    assert "drivers.option2" in case.skipped["option2"]


def test_everything_stale_is_refused(folder):
    rules = folder / "credit_rules.yaml"
    rules.write_text(rules.read_text() + "# edited\n", newline="\n")

    with pytest.raises(DeliverError, match="credit_rules.yaml"):
        load_case(folder)


def test_no_results_at_all_is_refused(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=())

    with pytest.raises(DeliverError, match="no usable results"):
        load_case(folder)


def test_an_unknown_case_is_refused(folder):
    with pytest.raises(DeliverError, match="unknown case 'nope'"):
        load_case(folder, "nope")


def test_a_different_tool_version_is_a_warning(folder):
    edit_results(folder, "option1", lambda p: p.update({"tool_version": "0.0.0"}))

    case = load_case(folder)

    assert "option1" in case.options
    assert any("0.0.0" in w for w in case.warnings)


def test_points_csv_room_names_with_commas_and_quotes(tmp_path):
    path = tmp_path / "points.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["room", "x", "y", "passes", "lux", "note"])
        writer.writerow(['Office, East "A"', "1.0", "2.0", "yes", "350.5", "a"])
        writer.writerow(["Room North", "3.0", "4.0", "no", "", "b"])

    table = read_points(path)

    assert table.rooms == ('Office, East "A"', "Room North")
    assert table.xy.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert table.columns["passes"].tolist() == [True, False]
    assert table.columns["lux"][0] == 350.5 and np.isnan(table.columns["lux"][1])
    assert table.columns["note"].tolist() == ["a", "b"]
    assert table.mask({'Office, East "A"'}).tolist() == [True, False]


def test_a_points_csv_without_coordinates_is_refused(tmp_path):
    path = tmp_path / "points.csv"
    path.write_text("room,value\nRoom North,1\n", newline="\n")

    with pytest.raises(DeliverError, match="'x'"):
        read_points(path)


# -- F15: a views-only case prepares with the views run's own model --------

#: Two more rooms with no glazing of their own, never touching Room North's
#: floor (so scene.storeys.detect_rooms keeps all three apart): "Room
#: South", a full metre south of it, and "Room Narrow", a 0.5 m wide strip
#: two metres north of it -- too narrow for a single sensor point inside
#: the grid's clearance, which ``build.prepare`` refuses for a regularly
#: occupied space INCLUDED in the case it is given.
_SCENE_RAD_THREE_ROOMS = SCENE_RAD + """\

floor_carpet polygon floor_room_south
0
0
12  0 -5 0   6 -5 0   6 -1 0   0 -1 0

floor_carpet polygon floor_room_narrow
0
0
12  0 6 0   6 6 0   6 6.5 0   0 6.5 0
"""

#: Room Narrow is regularly occupied and included in the one daylight case
#: ("base_case") -- so preparing that case's model (the pre-F15 bug: "the
#: first daylight case") refuses it for having no sensor point -- but it
#: is NOT ticked for the views-only case "only". A correct views-only load
#: must prepare with the views ticks (Room North only) and never grid Room
#: Narrow.
_SPACES_CSV_VIEWS_ONLY = """\
floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,view_preserving_shades,merge_into,include_base_case,views_only,reason
L0,Room North,office,yes,24,3.0,2.0,0.0,no,,1,1,
L0,Room South,office,no,24,3.0,-3.0,0.0,no,,1,0,
L0,Room Narrow,office,yes,3,3.0,6.25,0.0,no,,1,0,
"""


@pytest.fixture
def views_only_folder(tmp_path, write_example, write_wea):
    folder = build_option2_project(
        tmp_path, write_example, write_wea,
        spaces_csv=_SPACES_CSV_VIEWS_ONLY, scene_rad=_SCENE_RAD_THREE_ROOMS,
    )
    case_dir = folder / "_run" / "views" / "only"
    case_dir.mkdir(parents=True)
    payload = {
        "tool_version": TOOL_VERSION, "case": "only",
        "inputs": input_fingerprints(load_project(folder / "project.yaml"), folder),
    }
    (case_dir / "views_results.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    , newline="\n")
    with open(case_dir / "points.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["room", "x", "y"])
        writer.writerow([ROOM, "1.0", "2.0"])
    return folder


def test_a_views_only_case_prepares_with_its_own_model(views_only_folder):
    """Pre-F15, this raised: the views-only case "only" is not a daylight
    case, so the buggy code prepared the model for "base_case" instead --
    which includes Room Narrow, whose missing sensor point build.prepare
    refuses. The fix prepares with the views ticks
    (project.spaces.views_prepare_spaces), which never include Room
    Narrow, so this must load cleanly."""
    case = load_case(views_only_folder, "only")

    assert case.case == "only"
    assert set(case.options) == {"views"}
    (storey,) = case.storeys
    assert set(storey.rooms) == {"Room North", "Room South", "Room Narrow"}
    assert {g.room for g in case.groups} == {"Room North"}


def test_the_views_only_fixture_cannot_prepare_its_daylight_case(views_only_folder, tmp_path):
    """The fixture keeps its purpose: preparing "base_case" is refused, so
    the test above passes only when the views-only case prepares with its
    own ticks."""
    from leed_daylight.build.grid import GridError
    from leed_daylight.deliver.data import prepare_case_model
    from leed_daylight.project.spaces import load_spaces

    project = load_project(views_only_folder / "project.yaml")
    spaces = load_spaces(views_only_folder / "spaces.csv")

    with pytest.raises(GridError, match=r"room 'Room Narrow' \(area 3.00 m2\) yields no sensor point"):
        prepare_case_model(project, spaces, "base_case", tmp_path / "model")


def _edit_rules_mid_run(folder):
    """What a user editing an input during a long run does: the credit rules
    change on disk after they were read, while the engine is working."""
    path = folder / "credit_rules.yaml"
    path.write_text(path.read_text() + "# edited during the run\n", newline="\n")


@pytest.mark.parametrize("option", ["option1", "option2"])
def test_an_input_edited_during_the_run_leaves_the_results_stale(tmp_path, write_example, write_wea, option):
    """Final review I2: fingerprints are taken when the inputs are read, not
    when the results are written hours later."""
    import io

    import leed_daylight.drivers.option2 as option2_module
    import leed_daylight.drivers.run as run_module
    from deliver_folder import Option1Fake, Option2Fake
    from option2_folder import load_chain

    folder = build_deliver_project(tmp_path, write_example, write_wea, options=())
    _, prepared, _, _, _ = load_chain(folder)
    if option == "option1":
        class Editing(Option1Fake):
            def direct(self, group, hours):
                _edit_rules_mid_run(folder)
                return super().direct(group, hours)
        run_module.run_folder(folder, CASE, engine=Editing(prepared), out=io.StringIO())
    else:
        class Editing(Option2Fake):
            def illuminance(self, moment):
                _edit_rules_mid_run(folder)
                return super().illuminance(moment)
        option2_module.option2_folder(folder, CASE, engine=Editing(prepared), out=io.StringIO())

    with pytest.raises(DeliverError, match="credit_rules.yaml changed"):
        load_case(folder, options=(option,))


# -- Final review I5: hand-edited or truncated files are refused by name ----

@pytest.mark.parametrize("content, why", [
    (b'{"case": "base_ca', "not valid JSON"),
    (b"[]\n", "a JSON object"),
    (b"\xff\xfe\x00not text", "not UTF-8"),
])
def test_a_broken_results_file_is_refused_naming_the_file(folder, content, why):
    path = folder / "_run" / "option2" / CASE / "option2_results.json"
    path.write_bytes(content)

    with pytest.raises(DeliverError) as excinfo:
        load_case(folder)
    assert str(path) in str(excinfo.value) and why in str(excinfo.value)


@pytest.mark.parametrize("row, why", [
    ("Room North,1.0\n", "fewer fields"),
    ("Room North,1.0,2.0,9,9\n", "more fields"),
    ("Room North,east,2.0,9\n", "'x'"),
])
def test_a_bad_points_csv_row_is_refused_naming_file_and_line(tmp_path, row, why):
    path = tmp_path / "points.csv"
    path.write_text("room,x,y,lux\nRoom North,1.0,2.0,300\n" + row, newline="\n")

    with pytest.raises(DeliverError) as excinfo:
        read_points(path)
    message = str(excinfo.value)
    assert str(path) in message and "line 3" in message and why in message


def test_a_points_csv_that_is_not_utf8_is_refused_by_name(tmp_path):
    path = tmp_path / "points.csv"
    path.write_bytes(b"room,x,y\n\xff\xfe,1,2\n")

    with pytest.raises(DeliverError, match="not UTF-8"):
        read_points(path)


# -- Final review I6: two storeys, and a space the case leaves out ----------

def test_the_two_storey_fixture_loads_with_the_excluded_room_left_out(tmp_path, write_example, write_wea):
    from deliver_folder import EXCLUDED_ROOM, UPPER_ROOM, build_two_storey_project

    case = load_case(build_two_storey_project(tmp_path, write_example, write_wea))

    assert dict(case.skipped) == {}
    assert [(s.label, s.rooms) for s in case.storeys] == [("L0", (ROOM, EXCLUDED_ROOM)), ("L1", (UPPER_ROOM,))]
    for option, data in case.options.items():
        assert {s["room"] for s in data.results["spaces"]} == {ROOM, UPPER_ROOM}, option
        assert set(data.points.rooms) == {ROOM, UPPER_ROOM}, option
        assert [s.label for s in case.storeys_for(option)] == ["L0", "L1"]
    assert {g.storey_label for g in case.groups} == {"L0", "L1"}
    assert "intent" in case.rules and "intent" in case.rules["views"]
    assert case.rules["option1"]["space_rule_sda_pct"] == [55.0, 75.0]


# -- Task 11: kind, tilt and one roof-glass outline per pane -----------------

def _pane(name, *xyz):
    return Primitive("glass_roof", "polygon", name, (), (), tuple(float(v) for v in xyz), "test")


def _group(group_id, openings, classes, *, kind, azimuth, tilt):
    return WindowGroup(
        group_id=group_id, room="Room North", storey=0, azimuth_bin_deg=azimuth,
        openings=tuple(openings), outward_normals=tuple(np.array([0.0, 0.0, 1.0]) for _ in openings),
        area_m2=float(len(openings)), kind=kind, tilt_deg=tilt, pane_classes=tuple(classes),
    )


_FLAT_ROOF = PaneClass(tilt_deg=0.0, tilt_class="skylight", position="roof", diffusing=False, kind="skylight")


def test_a_window_group_keeps_its_plan_segment_and_azimuth():
    window = _pane("window_ext", 2, 0, 1, 4, 0, 1, 4, 0, 2, 2, 0, 2)
    line = _group_line(_group("Room_North_180", [window], (), kind="window", azimuth=180.0, tilt=90.0), "L0")

    assert line.kind == "window" and line.azimuth_deg == 180.0 and line.tilt_deg == 90.0
    (segment,) = line.segments
    assert sorted(segment[:, 0].tolist()) == [2.0, 4.0]
    assert line.outlines == ()
    assert line.openings == 1
    assert line.tilt_classes == ("vertical",)


def test_two_separate_skylights_of_one_room_stay_two_outlines():
    a = _pane("sky_a", 1, 1.5, 3.4, 2, 1.5, 3.4, 2, 2.5, 3.4, 1, 2.5, 3.4)
    b = _pane("sky_b", 4, 1.5, 3.4, 5, 1.5, 3.4, 5, 2.5, 3.4, 4, 2.5, 3.4)
    group = _group("Room_North_sky", [a, b], (_FLAT_ROOF, _FLAT_ROOF), kind="skylight", azimuth=None, tilt=0.0)

    line = _group_line(group, "L0")

    assert line.azimuth_deg is None
    assert line.segments == ()
    assert line.outlines == (
        ((1.0, 1.5), (2.0, 1.5), (2.0, 2.5), (1.0, 2.5)),
        ((4.0, 1.5), (5.0, 1.5), (5.0, 2.5), (4.0, 2.5)),
    )
    assert line.openings == 2
    assert line.tilt_classes == ("skylight",)


def test_a_repeated_vertex_is_dropped_from_a_roof_outline():
    pane = _pane("sky_c", 1, 1, 3.4, 2, 1, 3.4, 2, 1, 3.4, 2, 2, 3.4, 1, 2, 3.4, 1, 1, 3.4)
    group = _group("Room_North_sky", [pane], (_FLAT_ROOF,), kind="skylight", azimuth=None, tilt=0.0)

    assert _group_line(group, "L0").outlines == (((1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)),)


def test_a_group_of_facade_and_roof_panes_draws_each_its_own_way():
    wall = _pane("diffuser_wall", 2, 0, 1, 4, 0, 1, 4, 0, 2, 2, 0, 2)
    roof = _pane("diffuser_roof", 4, 1.5, 3.4, 5, 1.5, 3.4, 5, 2.5, 3.4, 4, 2.5, 3.4)
    facade_class = PaneClass(tilt_deg=90.0, tilt_class="vertical", position="perimeter", diffusing=True,
                             kind="diffusing")
    roof_class = PaneClass(tilt_deg=0.0, tilt_class="skylight", position="roof", diffusing=True, kind="diffusing")
    group = _group("Room_North_dif", [wall, roof], (facade_class, roof_class), kind="diffusing",
                   azimuth=None, tilt=45.0)

    line = _group_line(group, "L0")

    (segment,) = line.segments
    assert sorted(segment[:, 0].tolist()) == [2.0, 4.0]
    assert line.outlines == (((4.0, 1.5), (5.0, 1.5), (5.0, 2.5), (4.0, 2.5)),)
    assert line.openings == 2
    assert line.tilt_classes == ("vertical", "skylight")


def test_monitor_glass_is_outlined_as_a_plan_line():
    pane = _pane("monitor_east", 3, 1, 3.4, 3, 3, 3.4, 3, 3, 4.0, 3, 1, 4.0)
    monitor = PaneClass(tilt_deg=90.0, tilt_class="vertical", position="roof", diffusing=False, kind="monitor")

    line = _group_line(_group("Room_North_m090", [pane], (monitor,), kind="monitor", azimuth=90.0, tilt=90.0), "L0")

    assert line.segments == ()
    assert line.outlines == (((3.0, 1.0), (3.0, 3.0)),)
    assert line.azimuth_deg == 90.0


def test_roof_glass_groups_carry_kind_tilt_and_a_plan_outline(tmp_path, write_example, write_wea):
    case = load_case(build_roof_glass_project(tmp_path, write_example, write_wea))

    by_kind = {g.kind: g for g in case.groups}
    assert set(by_kind) == {"window", "skylight", "diffusing"}
    window, sky, dif = by_kind["window"], by_kind["skylight"], by_kind["diffusing"]
    assert window.outlines == () and len(window.segments) == 1
    assert isinstance(window.azimuth_deg, float) and window.tilt_deg == pytest.approx(90.0)
    assert sky.group_id == "Room_North_sky" and sky.azimuth_deg is None
    assert sky.segments == () and sky.openings == 1 and sky.tilt_classes == ("skylight",)
    assert sky.tilt_deg == pytest.approx(0.0)
    (sky_outline,) = sky.outlines
    assert sorted(sky_outline) == [(1.0, 1.5), (1.0, 2.5), (2.0, 1.5), (2.0, 2.5)]
    assert dif.group_id == "Room_North_dif" and dif.azimuth_deg is None
    (dif_outline,) = dif.outlines
    assert sorted(dif_outline) == [(4.0, 1.5), (4.0, 2.5), (5.0, 1.5), (5.0, 2.5)]


def test_option1_results_without_window_groups_are_skipped_run_again(folder):
    """Results from before roof glass was supported have no "groups": the
    report cannot tell a skylight from a window, so they are stale."""
    edit_results(folder, "option1", lambda p: p.pop("groups"))

    case = load_case(folder)

    assert "option1" not in case.options
    reason = case.skipped["option1"]
    assert reason.startswith("option1 results for case 'base_case'")
    assert "run again" in reason and "drivers.run" in reason
    assert set(case.options) == {"option2", "views"}
