"""build.view_model: see-through classes, pane tops, eye-height grids and
the black, blind-free views scene (QV spec 5.2, 6.1, 6.2).

The prepared-model tests use a fictional 6 m x 4 m x 3 m room with one
exterior window and one vertical furniture panel, invented and round.
"""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from leed_daylight.build.glazing_class import PaneClass
from leed_daylight.build.groups import WindowGroup
from leed_daylight.build.prepare import PreparedModel, prepare
from leed_daylight.build.view_model import (
    BLOCKED_VIEW_MODIFIER,
    ViewModelError,
    blocked_clear_warnings,
    blocked_panes,
    copy_rows,
    ground_plane,
    pane_tops,
    room_file_stem,
    view_classes,
    view_grids,
    vision_panes,
    write_view_scene,
)
from leed_daylight.project.config import load_project
from leed_daylight.project.materials import MaterialRow, load_materials
from leed_daylight.project.spaces import load_spaces, match_to_rooms
from leed_daylight.scene.primitives import Primitive
from leed_daylight.scene.reader import read_scene
from leed_daylight.scene.storeys import detect_rooms
from leed_daylight.scene.validate import validate_scene

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

void plastic desk_panel
0
0
5  0.4 0.4 0.4  0 0

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

desk_panel polygon screen_panel
0
0
12  1 2 0   2 2 0   2 2 1.5   1 2 1.5

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
desk_panel,plastic,furniture,
glass_clear,glass,glazing,
"""

SPACES_CSV = """\
floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,view_preserving_shades,merge_into,include_base_case,views_base,reason
L0,Room North,office,yes,24,3.0,2.0,0.0,no,,1,1,
"""


@pytest.fixture
def chain(tmp_path, write_example):
    project_dir = write_example(tmp_path).parent
    (project_dir / "scene.rad").write_text(SCENE_RAD, newline="\n")
    (project_dir / "materials.csv").write_text(MATERIALS_CSV, newline="\n")
    (project_dir / "spaces.csv").write_text(SPACES_CSV, newline="\n")
    project = load_project(project_dir / "project.yaml")
    prims = read_scene(project.scene.files, project.scene.units)
    roles = load_materials(project_dir / "materials.csv")
    validate_scene(prims, project.scene.units, roles).raise_if_errors()
    rooms = detect_rooms(prims, roles)
    spaces = load_spaces(project_dir / "spaces.csv")
    prepared = prepare(project, prims, roles, rooms, spaces, "base_case", project_dir / "_prepared")
    return project, prims, roles, rooms, spaces, prepared, project_dir


def _row(modifier, ptype, role, view=""):
    return MaterialRow(modifier=modifier, primitive_type=ptype, role=role, note="", view=view)


def test_view_classes_defaults_and_overrides():
    rows = [
        _row("glass_clear", "glass", "glazing"),
        _row("glass_dark", "glass", "glazing"),
        _row("glass_part", "glass", "glazing_interior"),
        _row("frosted", "trans", "wall"),
        _row("screen", "plastic", "fixed_shading", "clear"),
        _row("trees", "plastic", "context", "nature"),
        _row("wall_paint", "plastic", "wall"),
        _row("glass_unused", "glass", "glazing"),
    ]
    vt = {"glass_clear": 0.41, "glass_dark": 0.40, "glass_part": 0.8}

    classes = view_classes(rows, vt, vlt_min_exclusive=0.40)

    assert classes == {
        "glass_clear": "clear", "glass_dark": "blocks", "glass_part": "clear",
        "frosted": "blocks", "screen": "clear", "trees": "nature",
        "wall_paint": "blocks", "glass_unused": "blocks",
    }


def test_clear_on_glass_at_or_below_the_vlt_limit_is_refused():
    rows = [_row("glass_dark", "glass", "glazing", "clear")]

    with pytest.raises(ViewModelError, match=r"glass_dark.*0\.30.*0\.40"):
        view_classes(rows, {"glass_dark": 0.30}, vlt_min_exclusive=0.40)


def test_pane_tops_are_the_top_of_the_framed_glass(chain):
    *_, prepared, _ = chain
    (group,) = prepared.groups
    (glass,) = prepared.glass_by_group[group.group_id]

    tops = pane_tops(prepared)

    assert tops == {"window_ext": pytest.approx(float(np.max(glass[:, 2])))}
    assert 1.0 < tops["window_ext"] <= 2.0


def test_ground_plane_is_the_lowest_ground_polygon():
    ground = Primitive("site_ground", "polygon", "g1", (), (), (0, 0, -0.5, 1, 0, -0.5, 1, 1, -0.5), "t")
    other = Primitive("site_ground", "polygon", "g2", (), (), (0, 0, 0, 1, 0, 0, 1, 1, 0), "t")
    roles = {"site_ground": "ground"}

    z, warnings = ground_plane([ground, other], roles, storey_levels=[0.0, 3.5], room_levels={"Room A": 0.0})

    assert z == -0.5
    assert warnings == []


def test_without_ground_the_plane_is_the_lowest_floor_level_with_a_warning():
    # A model in datum heights: floors near +100 m, no ground surface.
    z, warnings = ground_plane([], {}, storey_levels=[104.0, 100.0], room_levels={"Room A": 104.0})

    assert z == 100.0
    assert warnings == ["no ground surface: ground taken at the lowest floor level, z = 100.00"]


def test_a_room_below_the_ground_plane_is_warned():
    ground = Primitive("site_ground", "polygon", "g1", (), (), (0, 0, 0, 1, 0, 0, 1, 1, 0), "t")

    z, warnings = ground_plane([ground], {"site_ground": "ground"}, storey_levels=[-3.0, 0.0],
                               room_levels={"Basement": -3.0, "Room A": 0.0})

    assert z == 0.0
    assert len(warnings) == 1
    assert "Basement" in warnings[0] and "below the ground plane" in warnings[0]


def test_ground_plane_needs_ground_or_a_floor():
    with pytest.raises(ViewModelError, match="no ground surface and no floor level"):
        ground_plane([], {}, storey_levels=[], room_levels={})


def test_view_grids_sit_at_eye_height(chain):
    project, prims, roles, rooms, spaces, prepared, _ = chain
    matched = match_to_rooms(spaces, rooms.rooms)

    (grid,) = view_grids(spaces, matched, project.grid, 1.1, "base")

    assert grid.room == "Room North"
    assert grid.points.shape[0] > 0
    assert np.allclose(grid.points[:, 2], 1.1)
    assert grid.room_area_m2 == pytest.approx(24.0)


def test_view_grids_grid_every_piece_of_a_merged_space(chain):
    """merge-split spec 7: a merged space gets one eye-height grid over
    all its pieces (the merge_into refusal is gone)."""
    project, prims, roles, rooms, spaces, prepared, _ = chain
    matched = match_to_rooms(spaces, rooms.rooms)
    (north,) = spaces
    (room,) = rooms.rooms
    annex = dataclasses.replace(north, room="Annex", merge_into="Room North")
    annex_room = dataclasses.replace(room, room_id="S0_R1", outline=room.outline + np.array([10.0, 0.0]))
    (alone,) = view_grids(spaces, matched, project.grid, 1.1, "base")

    (grid,) = view_grids([north, annex], {**matched, "Annex": annex_room}, project.grid, 1.1, "base")

    n = len(alone.points)
    assert grid.room == "Room North"
    assert len(grid.points) == 2 * n
    assert np.array_equal(grid.points[:n], alone.points)
    assert np.allclose(grid.points[n:, :2], alone.points[:, :2] + np.array([10.0, 0.0]))
    assert np.allclose(grid.points[:, 2], 1.1)
    assert grid.room_area_m2 == pytest.approx(48.0)


def test_the_views_scene_is_black_blind_free_and_follows_the_furniture_setting(chain):
    project, prims, roles, rooms, spaces, prepared, project_dir = chain

    scene, materials = write_view_scene(prepared, roles, moveable_furniture="exclude", out_dir=project_dir / "vm")
    scene_prims = read_scene([scene], "m")
    material_prims = read_scene([materials], "m")

    names = {p.name for p in scene_prims}
    assert "window_ext" in names
    assert "screen_panel" not in names
    assert not any("_blind_" in n for n in names)
    assert material_prims
    for p in material_prims:
        assert (p.modifier, p.ptype, p.fargs) == ("void", "plastic", (0.0, 0.0, 0.0, 0.0, 0.0))

    scene_kept, _ = write_view_scene(prepared, roles, moveable_furniture="include", out_dir=project_dir / "vm2")
    assert "screen_panel" in {p.name for p in read_scene([scene_kept], "m")}


def test_room_file_stem_is_filesystem_safe():
    assert "/" not in room_file_stem("A/B Room")


def test_vt_is_compared_at_four_decimals_so_a_round_trip_does_not_make_glass_clear():
    rows = [_row("glass_edge", "glass", "glazing"), _row("glass_above", "glass", "glazing")]
    # source: invented; 0.40 written as a 5-digit transmissivity and recovered
    vt = {"glass_edge": 0.40000043, "glass_above": 0.4001}  # source: invented, see above

    classes = view_classes(rows, vt, vlt_min_exclusive=0.40)

    assert classes == {"glass_edge": "blocks", "glass_above": "clear"}


def test_clear_override_is_refused_when_vt_rounds_to_the_limit():
    rows = [_row("glass_edge", "glass", "glazing", "clear")]

    with pytest.raises(ViewModelError, match=r"glass_edge.*not above"):
        view_classes(rows, {"glass_edge": 0.40000043}, vlt_min_exclusive=0.40)  # source: invented round trip
    ok = view_classes([_row("glass_above", "glass", "glazing", "clear")], {"glass_above": 0.4001},  # source: invented
                      vlt_min_exclusive=0.40)
    assert ok == {"glass_above": "clear"}


# =============================================================================
# roof-glass design 5.11: only perimeter, non-diffusing glass is vision
# glazing. Every pane below is invented and round.
# =============================================================================

_WINDOW = PaneClass(90.0, "vertical", "perimeter", False, "window")
_SLOPED_FACADE = PaneClass(60.0, "sloped", "perimeter", False, "sloped")
_SKYLIGHT = PaneClass(0.0, "skylight", "roof", False, "skylight")
_MONITOR = PaneClass(90.0, "vertical", "roof", False, "monitor")
_DIFFUSING = PaneClass(90.0, "vertical", "perimeter", True, "diffusing")


def _pane(name, modifier, corners):
    return Primitive(modifier, "polygon", name, (), (), tuple(float(v) for c in corners for v in c), "test")


def _group(group_id, panes, classes, *, kind):
    return WindowGroup(
        group_id=group_id, room="Room North", storey=0,
        azimuth_bin_deg=None if kind in ("skylight", "diffusing") else 180.0,
        openings=tuple(panes), outward_normals=tuple(np.array([0.0, -1.0, 0.0]) for _ in panes),
        area_m2=2.0, kind=kind, tilt_deg=classes[0].tilt_deg if classes else 90.0,
        pane_classes=tuple(classes),
    )


def _model(tmp_path, groups, effective_roles=None):
    return PreparedModel(
        root=tmp_path, case="views", scene_file=tmp_path / "scene.rad",
        materials_file=tmp_path / "materials.rad", blind_files={}, grid_files={},
        manifest_path=tmp_path / "manifest.json", manifest={}, groups=tuple(groups), grids=(),
        glass_by_group={g.group_id: tuple(o.points for o in g.openings) for g in groups},
        effective_roles=types.MappingProxyType(dict(effective_roles or {})),
    )


def _roof_glass_model(tmp_path):
    """A window, a sloped facade pane, a skylight, a roof monitor and a
    diffusing facade panel; the window and the monitor share glass_clear,
    the skylight uses its copy."""
    window = _pane("window_ext", "glass_clear", [(2, 0, 1), (4, 0, 1), (4, 0, 2), (2, 0, 2)])
    sloped = _pane("sloped_ext", "glass_clear__sloped", [(0, 1, 1), (0, 3, 1), (0.5, 3, 2), (0.5, 1, 2)])
    skylight = _pane("skylight_roof", "glass_clear__skylight",
                     [(2, 1, 3.5), (4, 1, 3.5), (4, 3, 3.5), (2, 3, 3.5)])
    monitor = _pane("monitor_roof", "glass_clear", [(1, 2, 3.5), (5, 2, 3.5), (5, 2, 4.5), (1, 2, 4.5)])
    diffuser = _pane("diffuser_ext", "panel_dif", [(6, 1, 1), (6, 3, 1), (6, 3, 2.5), (6, 1, 2.5)])
    groups = [
        _group("room_north_180", [window], [_WINDOW], kind="window"),
        _group("room_north_dif", [diffuser], [_DIFFUSING], kind="diffusing"),
        _group("room_north_m180", [monitor], [_MONITOR], kind="monitor"),
        _group("room_north_s270", [sloped], [_SLOPED_FACADE], kind="sloped"),
        _group("room_north_sky", [skylight], [_SKYLIGHT], kind="skylight"),
    ]
    roles = {
        "glass_clear": "glazing", "glass_clear__sloped": "glazing",
        "glass_clear__skylight": "glazing", "panel_dif": "glazing_diffusing",
    }
    return _model(tmp_path, groups, roles)


def test_only_perimeter_clear_glass_is_vision_glazing(tmp_path):
    model = _roof_glass_model(tmp_path)

    assert vision_panes(model) == {"window_ext", "sloped_ext"}
    assert blocked_panes(model) == {"skylight_roof", "monitor_roof", "diffuser_ext"}


def test_a_group_without_pane_classes_is_all_vision(tmp_path):
    window = _pane("window_ext", "glass_clear", [(2, 0, 1), (4, 0, 1), (4, 0, 2), (2, 0, 2)])
    model = _model(tmp_path, [_group("room_north_180", [window], [], kind="window")])

    assert vision_panes(model) == {"window_ext"}
    assert blocked_panes(model) == frozenset()


def test_head_height_comes_from_vision_panes_only(tmp_path):
    tops = pane_tops(_roof_glass_model(tmp_path))

    # the skylight (3.5), the monitor (4.5) and the diffuser (2.5) set none
    assert tops == {"window_ext": 2.0, "sloped_ext": 2.0}


def test_diffusing_glazing_always_blocks():
    rows = [_row("panel_dif", "trans", "glazing_diffusing"), _row("glass_clear", "glass", "glazing")]

    classes = view_classes(rows, {"panel_dif": 0.70, "glass_clear": 0.70}, vlt_min_exclusive=0.40)

    assert classes == {"panel_dif": "blocks", "glass_clear": "clear"}


def test_a_material_copy_is_classed_like_its_original():
    rows = [_row("glass_clear", "glass", "glazing", "clear"), _row("wall_paint", "plastic", "wall")]
    roles = {"glass_clear": "glazing", "glass_clear__sloped": "glazing", "wall_paint": "wall"}

    extra = copy_rows(rows, roles)

    assert extra == [_row("glass_clear__sloped", "glass", "glazing", "clear")]
    classes = view_classes([*rows, *extra], {"glass_clear": 0.70, "glass_clear__sloped": 0.70},
                           vlt_min_exclusive=0.40)
    assert classes["glass_clear__sloped"] == "clear"


def test_a_clear_tag_on_glass_that_blocked_panes_use_is_warned_once(tmp_path):
    model = _roof_glass_model(tmp_path)
    rows = [_row("glass_clear", "glass", "glazing", "clear"), _row("panel_dif", "trans", "glazing_diffusing")]

    warnings = blocked_clear_warnings(model, rows)

    # the monitor and the skylight (through its copy) both use glass_clear
    assert len(warnings) == 1
    assert warnings[0].startswith("glass_clear: marked 'clear' in materials.csv")
    assert "monitor_roof" in warnings[0] and "blocks views" in warnings[0]
    assert blocked_clear_warnings(model, [_row("glass_clear", "glass", "glazing")]) == []


def test_a_blocked_pane_wears_the_black_blocking_modifier(chain):
    *_, roles, _, _, prepared, project_dir = chain

    scene, materials = write_view_scene(
        prepared, roles, moveable_furniture="include", out_dir=project_dir / "vm",
        blocked=frozenset({"window_ext"}),
    )

    by_name = {p.name: p for p in read_scene([scene], "m")}
    assert by_name["window_ext"].modifier == BLOCKED_VIEW_MODIFIER
    material = {p.name: p for p in read_scene([materials], "m")}[BLOCKED_VIEW_MODIFIER]
    assert (material.modifier, material.ptype, material.fargs) == ("void", "plastic", (0.0,) * 5)


def test_nothing_blocked_adds_no_blocking_modifier(chain):
    *_, roles, _, _, prepared, project_dir = chain

    _, materials = write_view_scene(prepared, roles, moveable_furniture="include", out_dir=project_dir / "vm")

    assert BLOCKED_VIEW_MODIFIER not in {p.name for p in read_scene([materials], "m")}


def test_a_scene_that_already_uses_the_blocking_name_is_refused(chain):
    *_, roles, _, _, prepared, project_dir = chain
    clashing = project_dir / "clashing_materials.rad"
    clashing.write_text(
        prepared.materials_file.read_text()
        + f"\nvoid plastic {BLOCKED_VIEW_MODIFIER}\n0\n0\n5  0.5 0.5 0.5  0 0\n"
    , newline="\n")
    model = dataclasses.replace(prepared, materials_file=clashing)

    with pytest.raises(ViewModelError, match=BLOCKED_VIEW_MODIFIER):
        write_view_scene(model, roles, moveable_furniture="include", out_dir=project_dir / "vm",
                         blocked=frozenset({"window_ext"}))


def test_a_listed_material_named_like_a_copy_keeps_its_own_row(tmp_path):
    # glass_grey__sloped is a material of its own in materials.csv, not a
    # copy of glass_grey: it gets no synthetic row and keeps its own class.
    rows = [_row("glass_grey", "glass", "glazing", "clear"), _row("glass_grey__sloped", "glass", "glazing")]
    roles = {"glass_grey": "glazing", "glass_grey__sloped": "glazing"}

    assert copy_rows(rows, roles) == []
    classes = view_classes(rows, {"glass_grey": 0.70, "glass_grey__sloped": 0.30}, vlt_min_exclusive=0.40)
    assert classes["glass_grey__sloped"] == "blocks"

    skylight = _pane("skylight_roof", "glass_grey__sloped", [(2, 1, 3.5), (4, 1, 3.5), (4, 3, 3.5), (2, 3, 3.5)])
    model = _model(tmp_path, [_group("room_north_sky", [skylight], [_SKYLIGHT], kind="skylight")], roles)
    assert blocked_clear_warnings(model, rows) == []
