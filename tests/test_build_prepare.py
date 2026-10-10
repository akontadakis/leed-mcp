# tests/test_build_prepare.py
"""Tests for leed_daylight.build.prepare (T3.12): the prepared model.

The fixture (`tests/fixtures/mixed_glazing.rad`) is a fictional two-room
building, invented and round throughout (phase3-global.md #5): office_a is
6 x 4 x 3 m with one exterior window in its south wall; office_b is 4 x 4 x
3 m, separated from office_a by a 0.2 m gap so the two floor slabs never
touch (touching slabs would union into one room). The wall between them
carries one interior partition window -- spec 7's integration test.
"""
from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import re
import shutil
from types import MappingProxyType

import numpy as np
import pytest

from leed_daylight.build.prepare import PrepareError, prepare
from leed_daylight.project.config import (
    Analysis,
    Blinds,
    Glazing,
    Grid,
    Option2Spec,
    Performance,
    Project,
    RadianceParams,
    Reflectance,
    Site,
    SceneSpec,
    Sky,
    load_project,
)
from leed_daylight.project.config import ProjectError
from leed_daylight.project.spaces import Space
from leed_daylight.scene.primitives import Primitive
from leed_daylight.scene.reader import read_scene
from leed_daylight.scene.storeys import RoomDetection, detect_rooms
from leed_daylight.scene.validate import SceneError
from leed_daylight.units import transmissivity, transmittance_of

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
MIXED_GLAZING = FIXTURES / "mixed_glazing.rad"

ROLES = {
    "wall_paint": "wall",
    "ceiling_tile": "ceiling",
    "floor_carpet": "floor",
    "glass_clear": "glazing",
    "glass_partition": "glazing_interior",
}


# -- shared fixture helpers ---------------------------------------------


def make_project(
    root,
    *,
    frame_share=MappingProxyType({"default": 0.1}),
    frame_reflectance=0.3,
    frame_style="constant_width",
    reveal_depth_m=0.15,
    reveal_reflectance=0.5,
    dirt_factor=0.9,
    vt_includes_dirt=False,
    blinds=Blinds(0.05, 0.0, 0.5),
    grid=Grid(spacing_m=0.5, work_plane_m=0.75, min_clearance_m=0.1),
    reflectance=Reflectance(policy="keep_as_modelled", by_role=MappingProxyType({})),
    warnings=("a fictional project-level warning",),
    standard="LM-83-12",
    units="m",
    skylight_well=None,
    skylight_well_reflectance=None,
    ambient_full="-ab 6",
    direct="-dc 1",
) -> Project:
    return Project(
        name="Fictional Office",
        credit="LEED v4 BD+C, EQ Credit Daylight",
        root=pathlib.Path(root),
        site=Site(
            latitude_N=40.0,
            longitude_E=-75.0,
            meridian_radiance=75.0,
            north_rotation_deg=15.0,
            north_rotation_source="Fictional: taken from the site plan north arrow",
        ),
        scene=SceneSpec(units=units, files=()),
        standard=standard,
        credit_rules_path=pathlib.Path(root) / "credit_rules.yaml",
        weather_file=pathlib.Path(root) / "weather.wea",
        glazing=Glazing(
            dirt_factor=dirt_factor,
            frame_share=frame_share,
            frame_reflectance=frame_reflectance,
            frame_style=frame_style,
            reveal_depth_m=reveal_depth_m,
            reveal_reflectance=reveal_reflectance,
            vt_includes_dirt=vt_includes_dirt,
            skylight_well=skylight_well,
            skylight_well_reflectance=skylight_well_reflectance,
        ),
        reflectance=reflectance,
        grid=grid,
        analysis=Analysis(
            hours="08:00-18:00", daylight_saving="none",
            aggregation="floor_area", moveable_furniture="exclude",
        ),
        blinds=blinds,
        sky=Sky(basis="Reinhart", MF=1),
        radiance=RadianceParams(ambient_draft="-ab 2", ambient_full=ambient_full, direct=direct),
        performance=Performance(trace_scope="all", linked_zones=MappingProxyType({})),
        option2=Option2Spec(
            equinox_dates=(), clearest_day_window_days=0, times=(),
            sky="CIE clear with sun", lux_min=300.0, lux_max=3000.0,
            blinds="exclude", time_basis="UNVERIFIED",
        ),
        warnings=warnings,
    )


def make_spaces(*, include_office_b_in_variant=False):
    return [
        Space(
            floor="L0", room="office_a", space_type="office", regularly_occupied=True,
            area_m2=24.0, centroid_x=3.0, centroid_y=2.0, level_z=0.0,
            view_preserving_shades=False, merge_into=None,
            include={"base_case": True, "variant": True}, reason="",
        ),
        Space(
            floor="L0", room="office_b", space_type="office", regularly_occupied=True,
            area_m2=16.0, centroid_x=8.2, centroid_y=2.0, level_z=0.0,
            view_preserving_shades=False, merge_into=None,
            include={"base_case": True, "variant": include_office_b_in_variant}, reason="",
        ),
    ]


def load_mixed_glazing():
    prims = read_scene([MIXED_GLAZING], "m")
    rooms = detect_rooms(prims, ROLES)
    assert len(rooms.rooms) == 2
    return prims, rooms


def do_prepare(tmp_path, *, project_kwargs=None, case="base_case", spaces=None):
    prims, rooms = load_mixed_glazing()
    out_dir = tmp_path / "prepared"
    project = make_project(out_dir, **(project_kwargs or {}))
    if spaces is None:
        spaces = make_spaces()
    result = prepare(project, prims, ROLES, rooms, spaces, case, out_dir)
    return result


# -- the brief's eight cases ----------------------------------------------


def test_a_prepared_model_is_written(tmp_path):
    result = do_prepare(tmp_path)
    assert result.scene_file.is_file()
    assert result.materials_file.is_file()
    assert len(result.blind_files) == 1
    for path in result.blind_files.values():
        assert path.is_file()
    assert len(result.grid_files) == 2
    for path in result.grid_files.values():
        assert path.is_file()
    assert result.manifest_path.is_file()


def test_interior_glazing_gets_no_frame_no_reveal_no_blind_and_no_group(tmp_path):
    result = do_prepare(tmp_path)

    assert len(result.manifest["groups"]) == 1

    (blind_path,) = result.blind_files.values()
    blind_text = blind_path.read_text()
    assert "window_int" not in blind_text
    assert "glass_partition" not in blind_text

    scene_text = result.scene_file.read_text()
    assert "glass_partition polygon window_int" in scene_text
    # The interior opening's own vertices are untouched (no frame inset).
    assert "6.1 1.0 1.0 6.1 2.0 1.0 6.1 2.0 2.0 6.1 1.0 2.0" in scene_text
    assert "window_int_frame_0" not in scene_text
    assert "window_int_reveal_0" not in scene_text


def test_the_manifest_records_the_four_glazing_values(tmp_path):
    result = do_prepare(tmp_path)
    glazing = result.manifest["glazing"]
    assert set(glazing) == {"glass_clear", "glass_partition"}
    for record in glazing.values():
        assert set(record) >= {
            "visible_transmittance", "dirt_factor", "net_transmittance", "tn",
        }
        assert 0.0 < record["visible_transmittance"] <= 1.0
        assert 0.0 < record["tn"] <= 1.0
    # Exterior glazing gets the dirt factor; interior does not.
    assert glazing["glass_clear"]["dirt_factor"] == pytest.approx(0.9)
    assert glazing["glass_partition"]["dirt_factor"] == pytest.approx(1.0)


def test_the_manifest_records_the_north_rotation_source(tmp_path):
    result = do_prepare(tmp_path)
    assert result.manifest["site"]["north_rotation_source"] == (
        "Fictional: taken from the site plan north arrow"
    )
    assert result.manifest["site"]["north_rotation_deg"] == pytest.approx(15.0)


def test_achieved_glass_area_is_within_half_a_percent(tmp_path):
    result = do_prepare(tmp_path)
    openings = result.manifest["openings"]
    target = sum(o["target_glass_area_m2"] for o in openings.values())
    achieved = sum(o["achieved_glass_area_m2"] for o in openings.values())
    assert achieved == pytest.approx(target, rel=0.005)


def test_preparing_twice_is_byte_identical(tmp_path):
    result1 = do_prepare(tmp_path)
    result2 = do_prepare(tmp_path)
    assert result1.manifest["files"] == result2.manifest["files"]
    assert result1.scene_file.read_bytes() == result2.scene_file.read_bytes()
    assert result1.materials_file.read_bytes() == result2.materials_file.read_bytes()
    assert result1.manifest_path.read_bytes() == result2.manifest_path.read_bytes()


def _all_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _all_strings(k)
            yield from _all_strings(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _all_strings(item)


def test_every_manifest_path_is_relative(tmp_path):
    result = do_prepare(tmp_path)
    for path in result.manifest["files"]:
        assert not path.startswith("/"), path
    raw = result.manifest_path.read_text()
    parsed = json.loads(raw)
    for s in _all_strings(parsed):
        assert not s.startswith(str(tmp_path)), s


def test_a_second_case_produces_a_different_grid_set(tmp_path):
    base = do_prepare(tmp_path / "base", case="base_case")
    variant = do_prepare(tmp_path / "variant", case="variant")

    assert len(base.grid_files) == 2
    assert len(variant.grid_files) == 1
    assert set(variant.grid_files) == {"office_a"}

    assert base.scene_file.read_bytes() == variant.scene_file.read_bytes()
    assert base.materials_file.read_bytes() == variant.materials_file.read_bytes()


# -- controller-mandated: the prepared model reads back cleanly -----------


def test_the_prepared_model_reads_back_cleanly(tmp_path):
    result = do_prepare(tmp_path)
    written_surface_count = result.scene_file.read_text().count(" polygon ")

    readback = read_scene([result.scene_file, result.materials_file], "m")
    readback_surfaces = [p for p in readback if p.is_surface]
    assert len(readback_surfaces) == written_surface_count
    assert written_surface_count > 0

    # C1: the count alone does not catch two surfaces silently sharing one
    # name (the last one written would win and the other's geometry would
    # vanish with the count unchanged). Check that an untouched surface's
    # own geometry round-trips exactly, by name, not just that a polygon
    # with that name exists.
    by_name = {p.name: p for p in readback_surfaces}
    assert by_name["wall_partition"].points == pytest.approx(
        np.array([[6.1, 0.0, 0.0], [6.1, 4.0, 0.0], [6.1, 4.0, 3.0], [6.1, 0.0, 3.0]])
    )
    assert by_name["floor_office_b"].points == pytest.approx(
        np.array([[6.2, 0.0, 0.0], [10.2, 0.0, 0.0], [10.2, 4.0, 0.0], [6.2, 4.0, 0.0]])
    )


# -- idempotence and cleanup ------------------------------------------------


def test_stale_grid_file_is_removed_when_a_case_excludes_a_room(tmp_path):
    out_dir = tmp_path / "prepared"
    do_prepare(tmp_path, case="base_case")
    assert (out_dir / "grids" / "office_b.pts").exists()

    do_prepare(tmp_path, case="variant")
    assert not (out_dir / "grids" / "office_b.pts").exists()
    assert (out_dir / "grids" / "office_a.pts").exists()


# -- hostile-input / refusal pass -------------------------------------------


def test_out_dir_that_is_a_file_is_refused(tmp_path):
    prims, rooms = load_mixed_glazing()
    out_dir = tmp_path / "not_a_dir"
    out_dir.write_text("i am a file", newline="\n")
    project = make_project(out_dir)
    with pytest.raises(PrepareError, match="not_a_dir"):
        prepare(project, prims, ROLES, rooms, make_spaces(), "base_case", out_dir)


def test_an_empty_scene_is_refused(tmp_path):
    project = make_project(tmp_path / "prepared")
    with pytest.raises(ValueError):
        prepare(project, [], ROLES, RoomDetection((), ()), [], "base_case", tmp_path / "prepared")


def test_an_unknown_case_is_refused(tmp_path):
    prims, rooms = load_mixed_glazing()
    project = make_project(tmp_path / "prepared")
    with pytest.raises(ProjectError, match="nonexistent_case"):
        prepare(project, prims, ROLES, rooms, make_spaces(), "nonexistent_case", tmp_path / "prepared")


def test_an_unmatched_space_is_refused_by_name(tmp_path):
    prims, rooms = load_mixed_glazing()
    project = make_project(tmp_path / "prepared")
    ghost = Space(
        floor="L0", room="ghost_room", space_type="office", regularly_occupied=True,
        area_m2=10.0, centroid_x=500.0, centroid_y=500.0, level_z=0.0,
        view_preserving_shades=False, merge_into=None,
        include={"base_case": True, "variant": True}, reason="",
    )
    with pytest.raises(ProjectError, match="ghost_room"):
        prepare(project, prims, ROLES, rooms, make_spaces() + [ghost], "base_case", tmp_path / "prepared")


def test_missing_frame_share_default_is_refused(tmp_path):
    prims, rooms = load_mixed_glazing()
    project = make_project(tmp_path / "prepared", frame_share=MappingProxyType({}))
    with pytest.raises(PrepareError, match="default"):
        prepare(project, prims, ROLES, rooms, make_spaces(), "base_case", tmp_path / "prepared")


def test_zero_reveal_depth_produces_no_reveal_material_or_quads(tmp_path):
    result = do_prepare(tmp_path, project_kwargs={"reveal_depth_m": 0.0})
    materials_text = result.materials_file.read_text()
    scene_text = result.scene_file.read_text()
    assert "leed_reveal" not in materials_text
    assert "leed_reveal" not in scene_text
    assert "_reveal_" not in scene_text


def test_a_generated_name_collision_is_refused(tmp_path):
    prims, rooms = load_mixed_glazing()
    # A rogue primitive that will collide with the ring quad the exterior
    # window's frame must generate.
    colliding = Primitive(
        "wall_paint", "polygon", "window_ext_frame_0", (), (),
        (0.0, 5.0, 0.0, 1.0, 5.0, 0.0, 1.0, 5.0, 1.0), "test",
    )
    project = make_project(tmp_path / "prepared")
    with pytest.raises(PrepareError, match="window_ext_frame_0"):
        prepare(project, prims + [colliding], ROLES, rooms, make_spaces(), "base_case", tmp_path / "prepared")


def test_a_degenerate_glazing_opening_is_refused_by_name(tmp_path):
    # Collinear -- zero area. validate_scene catches this before window
    # grouping is ever attempted, so the whole scene is refused by name,
    # naming the broken opening, before anything is written.
    prims, rooms = load_mixed_glazing()
    collinear = (2.0, 0.0, 1.0, 3.0, 0.0, 1.0, 4.0, 0.0, 1.0)
    broken = [
        p if p.name != "window_ext" else Primitive(
            p.modifier, p.ptype, p.name, p.sargs, p.iargs, collinear, p.source,
        )
        for p in prims
    ]
    out_dir = tmp_path / "prepared"
    project = make_project(out_dir)
    with pytest.raises(SceneError, match="window_ext"):
        prepare(project, broken, ROLES, rooms, make_spaces(), "base_case", out_dir)
    assert not out_dir.exists()


def test_a_duplicated_vertex_opening_is_prepared_without_it(tmp_path):
    # A duplicated consecutive vertex adds no corner (M1): the frame and
    # the reveal are both built on the outline without it, rather than
    # the frame solver refusing a vertex count shapely never returns.
    prims, rooms = load_mixed_glazing()
    with_duplicate = (
        2.0, 0.0, 1.0,  # duplicated
        2.0, 0.0, 1.0,
        4.0, 0.0, 1.0,
        4.0, 0.0, 2.0,
        2.0, 0.0, 2.0,
    )
    broken = [
        p if p.name != "window_ext" else Primitive(
            p.modifier, p.ptype, p.name, p.sargs, p.iargs, with_duplicate, p.source,
        )
        for p in prims
    ]
    out_dir = tmp_path / "prepared"
    project = make_project(out_dir)
    model = prepare(project, broken, ROLES, rooms, make_spaces(), "base_case", out_dir)
    record = model.manifest["openings"]["window_ext"]
    assert record["achieved_glass_area_m2"] == pytest.approx(
        record["target_glass_area_m2"], rel=0.005
    )


def test_a_scene_that_fails_validation_is_refused_before_anything_is_written(tmp_path):
    prims, rooms = load_mixed_glazing()
    out_dir = tmp_path / "prepared"
    # A surface whose modifier has no role at all (materials.csv gap).
    bad = list(prims) + [
        Primitive("nobody_home", "polygon", "orphan", (), (), (0, 0, 0, 1, 0, 0, 1, 1, 0), "test")
    ]
    project = make_project(out_dir)
    with pytest.raises(SceneError, match="orphan"):
        prepare(project, bad, ROLES, rooms, make_spaces(), "base_case", out_dir)
    assert not out_dir.exists()


# -- C1: repeated surface names never silently replace each other's geometry


def test_a_repeated_surface_name_is_refused_naming_both_positions(tmp_path):
    prims, rooms = load_mixed_glazing()
    # A second "wall_partition" with DIFFERENT geometry, appended after the
    # first. Radiance allows two surfaces to share a name; this tool must
    # refuse it by name rather than silently keep only the later geometry
    # under that name (the reproduction: it used to replace the original
    # partition wall, and the manifest hash certified the corrupted file).
    duplicate = Primitive(
        "wall_paint", "polygon", "wall_partition", (), (),
        (0.0, 20.0, 0.0, 1.0, 20.0, 0.0, 1.0, 20.0, 1.0), "test",
    )
    out_dir = tmp_path / "prepared"
    project = make_project(out_dir)
    with pytest.raises(PrepareError, match="wall_partition") as excinfo:
        prepare(project, list(prims) + [duplicate], ROLES, rooms, make_spaces(), "base_case", out_dir)
    message = str(excinfo.value)
    assert "position" in message
    assert not out_dir.exists()


def test_two_glazing_openings_sharing_a_name_are_refused_clearly(tmp_path):
    # Before the fix, two glazing openings sharing a name produced a
    # misleading collision message blaming the generated "..._frame_0"
    # name instead of the real problem (the shared opening name itself).
    # The C1 fix catches this earlier, by name, before any frame is ever
    # generated.
    prims, rooms = load_mixed_glazing()
    duplicate_window = Primitive(
        "glass_clear", "polygon", "window_ext", (), (),
        (2.0, 4.0, 1.0, 4.0, 4.0, 1.0, 4.0, 4.0, 2.0, 2.0, 4.0, 2.0), "test",
    )
    out_dir = tmp_path / "prepared"
    project = make_project(out_dir)
    with pytest.raises(PrepareError) as excinfo:
        prepare(
            project, list(prims) + [duplicate_window], ROLES, rooms,
            make_spaces(), "base_case", out_dir,
        )
    message = str(excinfo.value)
    assert "window_ext" in message
    assert "window_ext_frame_0" not in message


# -- I3: PreparedModel carries groups, grids and framed glass (P26) ---------


def test_the_prepared_model_carries_groups_grids_and_framed_glass(tmp_path):
    result = do_prepare(tmp_path)

    assert [g.group_id for g in result.groups] == sorted(g.group_id for g in result.groups)
    assert len(result.grids) == 2

    (group,) = result.groups
    assert set(result.glass_by_group) == {group.group_id}
    glass = result.glass_by_group[group.group_id]
    assert len(glass) == len(group.openings)
    for outline in glass:
        assert outline.shape[1] == 3

    for room_grid in result.grids:
        assert isinstance(room_grid.storey, int)
        assert isinstance(room_grid.level_z, float)


# -- I4: two space names slugging to the same grid file name ----------------


def test_two_spaces_slugging_to_the_same_grid_name_are_refused(tmp_path):
    prims, rooms = load_mixed_glazing()
    spaces = [
        Space(
            floor="L0", room="office a", space_type="office", regularly_occupied=True,
            area_m2=24.0, centroid_x=3.0, centroid_y=2.0, level_z=0.0,
            view_preserving_shades=False, merge_into=None,
            include={"base_case": True, "variant": True}, reason="",
        ),
        Space(
            floor="L0", room="office_a", space_type="office", regularly_occupied=True,
            area_m2=16.0, centroid_x=8.2, centroid_y=2.0, level_z=0.0,
            view_preserving_shades=False, merge_into=None,
            include={"base_case": True, "variant": True}, reason="",
        ),
    ]
    project = make_project(tmp_path / "prepared")
    with pytest.raises(PrepareError, match="office a") as excinfo:
        prepare(project, prims, ROLES, rooms, spaces, "base_case", tmp_path / "prepared")
    assert "office_a" in str(excinfo.value)


# -- I5: a scene with no exterior glazing is refused; a glazing-less room is warned


def test_a_scene_with_no_exterior_glazing_is_refused(tmp_path):
    prims, rooms = load_mixed_glazing()
    no_windows = [p for p in prims if p.name != "window_ext"]
    out_dir = tmp_path / "prepared"
    project = make_project(out_dir)
    with pytest.raises(PrepareError, match="exterior glazing"):
        prepare(project, no_windows, ROLES, rooms, make_spaces(), "base_case", out_dir)
    assert not out_dir.exists()


def test_a_room_with_no_glazing_is_named_in_the_manifest_warnings(tmp_path):
    # office_b has no exterior glazing of its own (only office_a's window
    # and the interior partition, which never forms a window group), so it
    # must be named in the manifest's own warnings (spec 6.0).
    result = do_prepare(tmp_path)
    assert any("office_b" in w for w in result.manifest["warnings"])


# -- I6: project.yaml cannot smuggle NaN/Infinity into the manifest ---------


def test_a_non_finite_manifest_write_is_refused(tmp_path, monkeypatch):
    import leed_daylight.build.prepare as prepare_module

    real_dumps = prepare_module.json.dumps

    def poisoned_dumps(manifest, **kwargs):
        manifest = dict(manifest)
        manifest["totals"] = dict(manifest["totals"])
        manifest["totals"]["regularly_occupied_area_m2"] = float("nan")
        return real_dumps(manifest, **kwargs)

    monkeypatch.setattr(prepare_module.json, "dumps", poisoned_dumps)
    with pytest.raises(ValueError):
        do_prepare(tmp_path)
    # Nothing is written until every check has passed, the JSON one too.
    assert not (tmp_path / "prepared").exists()


def _snapshot(folder):
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def test_a_non_finite_manifest_leaves_an_earlier_prepared_model_unchanged(tmp_path, monkeypatch):
    import leed_daylight.build.prepare as prepare_module

    do_prepare(tmp_path)
    before = _snapshot(tmp_path / "prepared")
    assert "manifest.json" in before

    real_dumps = prepare_module.json.dumps

    def poisoned_dumps(manifest, **kwargs):
        manifest = dict(manifest)
        manifest["totals"] = dict(manifest["totals"])
        manifest["totals"]["grid_point_count"] = float("inf")
        return real_dumps(manifest, **kwargs)

    monkeypatch.setattr(prepare_module.json, "dumps", poisoned_dumps)
    with pytest.raises(ValueError):
        # A different frame share would change scene.rad if it were written.
        do_prepare(tmp_path, project_kwargs={"frame_share": MappingProxyType({"default": 0.2})})
    assert _snapshot(tmp_path / "prepared") == before


def test_every_file_is_written_by_rename_and_the_manifest_last(tmp_path, monkeypatch):
    import leed_daylight.build.prepare as prepare_module

    real_replace = prepare_module.os.replace
    targets = []

    def recording_replace(src, dst):
        assert pathlib.Path(src).parent == pathlib.Path(dst).parent
        targets.append(pathlib.Path(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(prepare_module.os, "replace", recording_replace)
    result = do_prepare(tmp_path)

    root = tmp_path / "prepared"
    written = {p for p in root.rglob("*") if p.is_file()}
    assert set(targets) == written
    assert targets[-1] == result.manifest_path
    assert not [p for p in root.rglob("*") if p.name.endswith(".tmp")]


# -- merge_into: one space of two pieces (merge-split spec 7, 8) ------------


def test_a_merged_space_prepares_with_one_grid_and_groups_per_piece(tmp_path):
    prims, rooms = load_mixed_glazing()
    spaces = make_spaces()
    merged = dataclasses.replace(spaces[1], merge_into="office_a")
    spaces = [spaces[0], merged]
    project = make_project(tmp_path / "prepared")

    result = prepare(project, prims, ROLES, rooms, spaces, "base_case", tmp_path / "prepared")

    assert [(g.group_id, g.room, g.piece) for g in result.groups] == [("office_a_p1_180", "office_a", 0)]
    (grid,) = result.grids
    assert grid.room == "office_a"
    assert grid.room_area_m2 == pytest.approx(40.0)
    assert set(result.grid_files) == {"office_a"}
    assert result.manifest["rooms"]["office_a"]["point_count"] == len(grid.points)
    layout = result.manifest["layout"]
    assert list(layout) == ["office_a"]
    assert layout["office_a"]["origin"] == "merged"
    assert [p["room_id"] for p in layout["office_a"]["pieces"]] == ["S0_R0", "S0_R1"]
    assert [p["area_m2"] for p in layout["office_a"]["pieces"]] == pytest.approx([24.0, 16.0])
    assert result.layout.space("office_a").rows == ("office_a", "office_b")


def test_a_piece_row_in_an_unselected_case_is_left_out(tmp_path):
    # office_b is a piece of office_a, and neither is in the variant case:
    # the variant model grids neither and records no layout, while the
    # window still groups on its piece.
    prims, rooms = load_mixed_glazing()
    spaces = make_spaces(include_office_b_in_variant=False)
    office_a = dataclasses.replace(spaces[0], include={"base_case": True, "variant": False})
    merged = dataclasses.replace(spaces[1], merge_into="office_a")
    result = prepare(
        make_project(tmp_path / "prepared"), prims, ROLES, rooms, [office_a, merged],
        "variant", tmp_path / "prepared",
    )
    assert result.manifest_path.is_file()
    assert result.grids == ()
    assert "layout" not in result.manifest
    assert [g.group_id for g in result.groups] == ["office_a_p1_180"]


# -- M5: a reveal failure inside prepare names the opening -------------------


def test_a_reveal_failure_inside_prepare_names_the_opening(tmp_path, monkeypatch):
    import leed_daylight.build.prepare as prepare_module
    from leed_daylight.build.reveals import RevealError

    def boom(*args, **kwargs):
        raise RevealError("synthetic failure for this test")

    monkeypatch.setattr(prepare_module, "reveal_walls", boom)

    prims, rooms = load_mixed_glazing()
    out_dir = tmp_path / "prepared"
    project = make_project(out_dir)
    with pytest.raises(PrepareError, match="window_ext"):
        prepare(project, prims, ROLES, rooms, make_spaces(), "base_case", out_dir)


# -- M10: a real load_project() output feeds prepare() end to end -----------


def test_load_project_output_feeds_prepare_end_to_end(tmp_path, write_wea):
    # Every other test in this file hand-builds its Project through
    # make_project(); this is the one path that goes through the real
    # config.py loader (parsing, cross-field checks) before prepare() ever
    # sees it -- the same shape of gap that let C1 through unnoticed, since
    # every phase-3 test used a hand-built Project.
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    shutil.copy(MIXED_GLAZING, project_dir / "scene.rad")
    # A full-year .wea for the site below (40 N, 75 W): checks.py reads it.
    write_wea(project_dir / "weather" / "example.wea", latitude=40.0, longitude_W=75.0)
    (project_dir / "credit_rules.yaml").write_text('version: "Example rules v0"\n', newline="\n")

    yaml_text = """
project:
  name: "Fictional Office"
  credit: "LEED v4 BD+C, EQ Credit Daylight"
site:
  latitude_N: 40.0
  longitude_E: -75.0
  meridian_radiance: 75.0
  north_rotation_deg: 15.0
  north_rotation_source: "Fictional: taken from the site plan north arrow"
scene:
  units: m
  files: ["scene.rad"]
standard:
  daylight: "LM-83-12"
  credit_rules: "credit_rules.yaml"
weather:
  file: "weather/example.wea"
glazing:
  dirt_factor: 0.9
  frame_share: {default: 0.1}
  frame_reflectance: 0.3
  frame_style: constant_width
  reveal_depth_m: 0.15
  reveal_reflectance: 0.5
  vt_includes_dirt: false
reflectance:
  policy: keep_as_modelled
grid:
  spacing_m: 0.5
  work_plane_m: 0.75
  min_clearance_m: 0.1
analysis:
  hours: "08:00-18:00"
  daylight_saving: none
  aggregation: floor_area
  moveable_furniture: exclude
performance:
  trace_scope: all
  linked_zones: {}
sky:
  basis: Reinhart
  MF: 1
radiance:
  ambient:
    draft: "-ab 2"
    full: "-ab 6"
  direct: "-dc 1"
blinds:
  transmittance_diffuse: 0.05
  transmittance_specular: 0.0
  room_side_reflectance: 0.5
option2:
  equinox_dates: []
  clearest_day_window_days: 0
  times: []
  sky: "CIE clear with sun"
  lux_min: 300.0
  lux_max: 3000.0
  blinds: exclude
  time_basis: "UNVERIFIED"
"""
    (project_dir / "project.yaml").write_text(yaml_text, newline="\n")
    project = load_project(project_dir / "project.yaml")

    prims, rooms = load_mixed_glazing()
    out_dir = tmp_path / "prepared"
    result = prepare(project, prims, ROLES, rooms, make_spaces(), "base_case", out_dir)
    assert result.manifest_path.is_file()
    assert len(result.manifest["groups"]) == 1


def test_override_with_a_missing_role_reflectance_is_warned_in_the_manifest(tmp_path):
    prims, rooms = load_mixed_glazing()
    out_dir = tmp_path / "prepared"
    project = make_project(
        out_dir,
        reflectance=Reflectance(policy="override", by_role=MappingProxyType({"ceiling": 0.7})),
    )
    model = prepare(project, prims, ROLES, rooms, make_spaces(), "base_case", out_dir)
    hits = [w for w in model.manifest["warnings"] if "policy 'override'" in w]
    assert len(hits) == 1
    assert "'floor', 'wall'" in hits[0]


# -- reveals on floor-to-ceiling curtain-wall panes -------------------------


def test_a_floor_to_ceiling_curtain_wall_gets_reveals_only_on_its_outer_jambs(tmp_path):
    prims, rooms = load_mixed_glazing()
    window = next(p for p in prims if p.name == "window_ext")
    left = dataclasses.replace(
        window, name="pane_left", fargs=(2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0, 3.0, 2.0, 0.0, 3.0)
    )
    right = dataclasses.replace(
        window, name="pane_right", fargs=(3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 0.0, 3.0, 3.0, 0.0, 3.0)
    )
    scene = [p for p in prims if p.name != "window_ext"] + [left, right]
    out_dir = tmp_path / "prepared"
    prepare(make_project(out_dir), scene, ROLES, rooms, make_spaces(), "base_case", out_dir)

    reveal_names = [
        line.split()[2]
        for line in (out_dir / "scene.rad").read_text().splitlines()
        if line.startswith("leed_reveal polygon")
    ]
    # No sill (on the floor), no head (at the ceiling), no joint at x = 3:
    # each pane keeps one jamb, inward and outward.
    assert sorted(reveal_names) == [
        "pane_left_reveal_0", "pane_left_reveal_1",
        "pane_right_reveal_0", "pane_right_reveal_1",
    ]


def test_words_with_spaces_quotes_and_hashes_survive_prepare_and_read_back(tmp_path):
    # A pattern the prepared materials keep untouched, whose string
    # arguments hold a space, a hash and a quote: written back quoted
    # where needed, so read_scene gets the same words.
    prims, rooms = load_mixed_glazing()
    words = ("dim pattern", "fic#tional.cal", "it's")
    pattern = Primitive("void", "brightfunc", "invented_pattern", words, (), (), "test")
    out_dir = tmp_path / "prepared"
    project = make_project(out_dir)

    result = prepare(project, [pattern, *prims], ROLES, rooms, make_spaces(), "base_case", out_dir)

    readback = read_scene([result.materials_file], "m")
    [kept] = [p for p in readback if p.name == "invented_pattern"]
    assert kept.sargs == words


# -- roof glass, diffusing glass and the order of work (spec 5.4-5.8, 4.3) --
#
# The same invented building with roof glass added. office_a's ceiling is
# at z = 3.0; a 2 x 1 m skylight sits at z = 3.5 over the opening
# x 2..4, y 1.5..2.5, and (unless a test asks for the closed ceiling) the
# ceiling is four strips around that opening.

SKYLIGHT_A = (2.0, 1.5, 3.5, 4.0, 1.5, 3.5, 4.0, 2.5, 3.5, 2.0, 2.5, 3.5)
# Rises 0.5 m over 1 m of plan: tilt atan(1 / 0.5) from flat, about 27
# degrees, so sloped class; 2 m in from every wall of office_a, so roof.
SLOPED_A = (2.0, 1.5, 3.5, 4.0, 1.5, 3.5, 4.0, 2.5, 4.0, 2.0, 2.5, 4.0)
# A vertical diffusing panel in office_a's south wall, beside window_ext.
PANEL_EXT = (4.5, 0.0, 1.0, 5.5, 0.0, 1.0, 5.5, 0.0, 2.0, 4.5, 0.0, 2.0)

SKY_PANE = ("glass_clear", "skylight_a", SKYLIGHT_A)
SLOPED_PANE = ("glass_clear", "sloped_a", SLOPED_A)
DIFFUSING_PANEL = ("diffuser_white", "panel_ext", PANEL_EXT)

DIFFUSER = Primitive(
    "void", "trans", "diffuser_white", (), (), (0.8, 0.8, 0.8, 0.0, 0.0, 0.5, 0.0), "test"
)
ROOF_ROLES = {**ROLES, "diffuser_white": "glazing_diffusing"}

SKY_KW = {
    "dirt_factor": MappingProxyType({"vertical": 0.9, "skylight": 0.85}),
    "frame_share": MappingProxyType({"default": 0.1, "skylight": 0.05}),
    "skylight_well": 0.4,
    "skylight_well_reflectance": 0.7,
}


def _ceiling_strips_around(x0, x1, y0, y1):
    """office_a's ceiling (0..6 x 0..4 at z = 3) less the opening, drawn
    facing down like the fixture's own ceiling."""
    rects = [(0.0, 6.0, 0.0, y0), (0.0, 6.0, y1, 4.0), (0.0, x0, y0, y1), (x1, 6.0, y0, y1)]
    return [
        Primitive(
            "ceiling_tile", "polygon", f"ceiling_office_a_{k}", (), (),
            (xa, ya, 3.0, xa, yb, 3.0, xb, yb, 3.0, xb, ya, 3.0), "test",
        )
        for k, (xa, xb, ya, yb) in enumerate(rects)
    ]


def roof_scene(panes, *, open_ceiling=True, diffuser=False):
    prims = read_scene([MIXED_GLAZING], "m")
    if open_ceiling:
        prims = [p for p in prims if p.name != "ceiling_office_a"]
        prims += _ceiling_strips_around(2.0, 4.0, 1.5, 2.5)
    head = [DIFFUSER] if diffuser else []
    extra = [Primitive(mod, "polygon", name, (), (), pts, "test") for mod, name, pts in panes]
    prims = head + prims + extra
    rooms = detect_rooms(prims, ROOF_ROLES)
    return prims, rooms


def roof_prepare(
    tmp_path, panes, *, open_ceiling=True, diffuser=False, check_ambient=True, **project_kwargs
):
    prims, rooms = roof_scene(panes, open_ceiling=open_ceiling, diffuser=diffuser)
    out_dir = tmp_path / "prepared"
    project = make_project(out_dir, **project_kwargs)
    return prepare(
        project, prims, ROOF_ROLES, rooms, make_spaces(), "base_case", out_dir,
        check_ambient=check_ambient,
    )


def _readback(result):
    return {p.name: p for p in read_scene([result.scene_file, result.materials_file], "m")}


# spec 5.4 step 3: one material, two tilt classes


def test_one_glass_material_on_a_wall_window_and_a_skylight_gives_two_materials(tmp_path):
    result = roof_prepare(tmp_path, [SKY_PANE], **SKY_KW)

    materials = [p.name for p in read_scene([result.materials_file], "m")]
    assert materials[materials.index("glass_clear") + 1] == "glass_clear__skylight"

    readback = _readback(result)
    vt = transmittance_of(0.6)  # the fixture's glass_clear carries transmissivity 0.6
    assert readback["glass_clear"].fargs == pytest.approx((transmissivity(vt * 0.9),) * 3)
    assert readback["glass_clear__skylight"].fargs == pytest.approx(
        (transmissivity(vt * 0.85),) * 3
    )
    assert readback["skylight_a"].modifier == "glass_clear__skylight"
    assert readback["window_ext"].modifier == "glass_clear"

    assert result.effective_roles["glass_clear__skylight"] == "glazing"
    assert result.manifest["openings"]["skylight_a"]["modifier"] == "glass_clear__skylight"
    assert result.manifest["openings"]["skylight_a"]["role"] == "glazing"
    glazing = result.manifest["glazing"]
    assert glazing["glass_clear"]["class"] == "vertical"
    assert glazing["glass_clear__skylight"]["class"] == "skylight"
    assert glazing["glass_clear__skylight"]["transmission"] == "specular"
    assert glazing["glass_clear__skylight"]["dirt_factor"] == pytest.approx(0.85)

    # The prepared groups keep the UNFRAMED original outline and differ
    # only in carrying the new modifier.
    (sky,) = [g for g in result.groups if g.kind == "skylight"]
    assert sky.group_id == "office_a_sky"
    (opening,) = sky.openings
    assert opening.modifier == "glass_clear__skylight"
    assert opening.fargs == SKYLIGHT_A
    # A clear skylight keeps a blind (spec D1); the open ceiling gives no warning.
    assert "office_a_sky" in result.blind_files
    assert not any("ceiling surface" in w for w in result.manifest["warnings"])


def test_a_copy_name_already_in_the_scene_is_refused(tmp_path):
    clash = ("wall_paint", "glass_clear__skylight", (20.0, 20.0, 0.0, 21.0, 20.0, 0.0, 21.0, 20.0, 1.0))
    with pytest.raises(PrepareError, match="glass_clear__skylight"):
        roof_prepare(tmp_path, [SKY_PANE, clash], **SKY_KW)
    assert not (tmp_path / "prepared").exists()


def test_a_missing_skylight_dirt_key_is_refused(tmp_path):
    kw = dict(SKY_KW, dirt_factor=MappingProxyType({"vertical": 0.9}))
    with pytest.raises(PrepareError, match=r"no 'skylight' value, but pane skylight_a is skylight class"):
        roof_prepare(tmp_path, [SKY_PANE], **kw)


# spec 5.5: frames


def test_the_skylight_frame_share_wins_over_the_role_key(tmp_path):
    kw = dict(
        SKY_KW,
        frame_share=MappingProxyType({"default": 0.1, "glazing": 0.2, "skylight": 0.05}),
    )
    result = roof_prepare(tmp_path, [SKY_PANE], **kw)
    openings = result.manifest["openings"]
    assert openings["skylight_a"]["frame_share"] == pytest.approx(0.05)
    assert openings["window_ext"]["frame_share"] == pytest.approx(0.2)


def test_a_missing_skylight_frame_share_is_refused(tmp_path):
    kw = dict(SKY_KW, frame_share=MappingProxyType({"default": 0.1, "glazing": 0.2}))
    with pytest.raises(PrepareError, match=r"glazing\.frame_share: no 'skylight' entry, but pane skylight_a"):
        roof_prepare(tmp_path, [SKY_PANE], **kw)
    assert not (tmp_path / "prepared").exists()


# spec 5.6: wells and reveals


def test_a_missing_skylight_well_is_refused(tmp_path):
    kw = dict(SKY_KW, skylight_well=None, skylight_well_reflectance=None)
    with pytest.raises(PrepareError, match=r"glazing\.skylight_well: missing, but pane skylight_a"):
        roof_prepare(tmp_path, [SKY_PANE], **kw)


def test_a_modelled_well_builds_no_well_walls(tmp_path):
    kw = dict(SKY_KW, skylight_well="modelled", skylight_well_reflectance=None)
    result = roof_prepare(tmp_path, [SKY_PANE], **kw)
    readback = _readback(result)
    assert not any("_well_" in name for name in readback)
    assert "leed_skylight_well" not in readback
    assert not any(name.startswith("skylight_a_reveal_") for name in readback)
    assert result.manifest["skylight_well"] == "modelled"


def test_a_well_depth_builds_one_level_wall_per_edge(tmp_path):
    result = roof_prepare(tmp_path, [SKY_PANE], **SKY_KW)
    readback = _readback(result)
    wells = sorted(name for name in readback if name.startswith("skylight_a_well_"))
    assert wells == [f"skylight_a_well_{i}" for i in range(4)]
    for name in wells:
        quad = readback[name]
        assert quad.modifier == "leed_skylight_well"
        # 3.5 (the pane's lowest vertex) - 0.4 (the well depth) = 3.1.
        assert quad.points[:, 2].min() == pytest.approx(3.1)
        assert quad.points[:, 2].max() == pytest.approx(3.5)
    assert readback["leed_skylight_well"].fargs == pytest.approx((0.7, 0.7, 0.7, 0.0, 0.0))
    # A skylight gets a well, never a reveal, even with reveals on.
    assert not any(name.startswith("skylight_a_reveal_") for name in readback)
    assert result.manifest["skylight_well"] == {"depth_m": 0.4, "reflectance": 0.7}


def test_a_ceiling_under_a_skylight_is_warned(tmp_path):
    result = roof_prepare(tmp_path, [SKY_PANE], open_ceiling=False, **SKY_KW)
    assert (
        "a ceiling surface lies under skylight skylight_a; its light is blocked"
        in result.manifest["warnings"]
    )


# An L-shaped skylight at z = 3.5: arms 3 x 1 (x 1..4, y 0.5..1.5) and
# 1 x 2 (x 1..2, y 1.5..3.5). Its centroid, (2.1, 1.6), lies outside the L.
SKYLIGHT_L = (
    1.0, 0.5, 3.5, 4.0, 0.5, 3.5, 4.0, 1.5, 3.5,
    2.0, 1.5, 3.5, 2.0, 3.5, 3.5, 1.0, 3.5, 3.5,
)
# office_a's ceiling (0..6 x 0..4 at z = 3) less an L-shaped hole under the glass.
_L_CEILING = [
    (0.0, 6.0, 0.0, 0.5), (0.0, 6.0, 3.5, 4.0), (0.0, 1.0, 0.5, 3.5),
    (4.0, 6.0, 0.5, 1.5), (2.0, 6.0, 1.5, 3.5),
]


@pytest.mark.parametrize("hole", [True, False])
def test_an_l_shaped_skylight_is_checked_from_a_point_inside_it(tmp_path, hole):
    prims, rooms = roof_scene([("glass_clear", "skylight_l", SKYLIGHT_L)], open_ceiling=False)
    ceiling = [p for p in prims if p.name == "ceiling_office_a"]
    assert len(ceiling) == 1
    if hole:
        prims = [p for p in prims if p.name != "ceiling_office_a"] + [
            Primitive(
                "ceiling_tile", "polygon", f"ceiling_office_a_{k}", (), (),
                (xa, ya, 3.0, xa, yb, 3.0, xb, yb, 3.0, xb, ya, 3.0), "test",
            )
            for k, (xa, xb, ya, yb) in enumerate(_L_CEILING)
        ]
    rooms = detect_rooms(prims, ROOF_ROLES)
    out_dir = tmp_path / "prepared"
    result = prepare(
        make_project(out_dir, **SKY_KW), prims, ROOF_ROLES, rooms, make_spaces(),
        "base_case", out_dir,
    )
    warned = [w for w in result.manifest["warnings"] if "ceiling surface" in w]
    # With the hole cut under the glass the light is not blocked, although
    # the centroid lies over solid ceiling; with the ceiling closed it is.
    assert warned == ([] if hole else [
        "a ceiling surface lies under skylight skylight_l; its light is blocked"
    ])


def test_two_skylight_panes_sharing_a_ridge_get_no_well_wall_on_it(tmp_path):
    # Two 10-degree panes of office_a meeting at a ridge at x = 3.
    rise = math.tan(math.radians(10.0))
    top = 3.5 + rise
    west = (2.0, 1.5, 3.5, 3.0, 1.5, top, 3.0, 2.5, top, 2.0, 2.5, 3.5)
    east = (3.0, 1.5, top, 4.0, 1.5, 3.5, 4.0, 2.5, 3.5, 3.0, 2.5, top)
    result = roof_prepare(
        tmp_path,
        [("glass_clear", "sky_west", west), ("glass_clear", "sky_east", east)],
        **SKY_KW,
    )
    classes = result.manifest["openings"]
    assert classes["sky_west"]["class"]["tilt_class"] == "skylight"
    assert classes["sky_east"]["class"]["tilt_class"] == "skylight"
    readback = _readback(result)
    for pane in ("sky_west", "sky_east"):
        wells = sorted(name for name in readback if name.startswith(f"{pane}_well_"))
        # Four edges, one of them the shared ridge: three well walls.
        assert wells == [f"{pane}_well_{i}" for i in range(3)]
        for name in wells:
            xs = readback[name].points[:, 0]
            assert not np.allclose(xs, 3.0), f"{name} is a well wall on the ridge"


def test_a_sloped_roof_pane_gets_a_room_side_reveal_only(tmp_path):
    result = roof_prepare(
        tmp_path, [SLOPED_PANE],
        dirt_factor=MappingProxyType({"vertical": 0.9, "sloped": 0.9}),
        frame_share=MappingProxyType({"default": 0.1, "sloped": 0.1}),
    )
    (sloped,) = [g for g in result.groups if g.kind == "sloped"]
    assert sloped.group_id.startswith("office_a_s")
    reveal_names = sorted(n for n in _readback(result) if n.startswith("sloped_a_reveal_"))
    # Four edges, inner half only (none at the floor or the ceiling).
    assert reveal_names == [f"sloped_a_reveal_{i}" for i in range(4)]
    # No skylight-class glass: no well key needed, recorded as null.
    assert result.manifest["skylight_well"] is None


# spec 5.8: no blind for diffusing glass


def test_a_diffusing_facade_panel_is_framed_and_revealed_but_has_no_blind(tmp_path):
    result = roof_prepare(tmp_path, [DIFFUSING_PANEL], diffuser=True)

    (dif,) = [g for g in result.groups if g.kind == "diffusing"]
    assert dif.group_id == "office_a_dif"
    assert dif.azimuth_bin_deg is None
    assert dif.pane_classes[0].position == "perimeter"

    scene_text = result.scene_file.read_text()
    for i in range(4):
        assert f"panel_ext_frame_{i}" in scene_text
    # Today's reveal, both sides of the glass: 4 edges x 2 = 8 quads.
    assert "panel_ext_reveal_0" in scene_text
    assert "panel_ext_reveal_7" in scene_text

    assert "office_a_dif" not in result.blind_files
    assert not (result.root / "blinds" / "office_a_dif.rad").exists()
    group = result.manifest["groups"]["office_a_dif"]
    assert group["blind_polygon_count"] == 0
    assert group["kind"] == "diffusing"
    assert group["azimuth_bin_deg"] is None
    assert result.manifest["glazing"]["diffuser_white"]["transmission"] == "diffuse"
    # The clear window beside it keeps its blind.
    assert len(result.blind_files) == 1


# spec 4.3: the -ab check


@pytest.mark.parametrize(
    "ambient_full, direct, option",
    [
        ("-aa 0.1 -ad 1024", "-dc 1", "ambient.full"),  # no -ab at all: names both
        ("-ab 1", "-dc 1", "ambient.full"),
        ("-ab 3 -ab 1", "-dc 1", "ambient.full"),  # the last -ab wins
        ("-ab 3", "-ab 1 -dc 1", "direct"),  # direct comes after ambient.full
    ],
)
def test_a_low_or_missing_ab_is_refused_with_diffusing_glass(tmp_path, ambient_full, direct, option):
    with pytest.raises(PrepareError, match=rf"^radiance\.{re.escape(option)}: .*-ab 2 or more"):
        roof_prepare(
            tmp_path, [DIFFUSING_PANEL], diffuser=True,
            ambient_full=ambient_full, direct=direct,
        )
    assert not (tmp_path / "prepared").exists()


def test_an_options_file_is_refused_with_diffusing_glass(tmp_path):
    with pytest.raises(PrepareError, match=r"^radiance\.ambient\.full: '@full\.opt' reads an options file"):
        roof_prepare(tmp_path, [DIFFUSING_PANEL], diffuser=True, ambient_full="@full.opt")


def test_an_options_file_in_direct_names_radiance_direct(tmp_path):
    with pytest.raises(PrepareError, match=r"^radiance\.direct: '@direct\.opt' reads an options file"):
        roof_prepare(
            tmp_path, [DIFFUSING_PANEL], diffuser=True, ambient_full="-ab 2", direct="@direct.opt",
        )


@pytest.mark.parametrize(
    "ambient_full, direct",
    [("-ab 2", "-dc 1"), ("-ab 1", "-ab 2 -dc 1"), ("-ab 6 -ad 1024", "-dc 1")],
)
def test_ab_two_or_more_passes_with_diffusing_glass(tmp_path, ambient_full, direct):
    result = roof_prepare(
        tmp_path, [DIFFUSING_PANEL], diffuser=True, ambient_full=ambient_full, direct=direct,
    )
    assert result.manifest_path.is_file()


@pytest.mark.parametrize("ambient_full", ["-ab 1", "@full.opt"])
def test_without_diffusing_glass_ab_is_not_checked(tmp_path, ambient_full):
    result = do_prepare(tmp_path, project_kwargs={"ambient_full": ambient_full})
    assert result.manifest_path.is_file()


def test_check_ambient_false_skips_the_ab_check(tmp_path):
    result = roof_prepare(
        tmp_path, [DIFFUSING_PANEL], diffuser=True, check_ambient=False, ambient_full="-ab 0",
    )
    assert result.manifest_path.is_file()


# spec 5.12: the manifest gate


def test_an_all_vertical_scene_writes_no_new_manifest_keys(tmp_path):
    result = do_prepare(tmp_path)
    manifest = result.manifest
    assert "skylight_well" not in manifest
    for group in manifest["groups"].values():
        assert "kind" not in group and "tilt_deg" not in group
    for opening in manifest["openings"].values():
        assert "class" not in opening
    for record in manifest["glazing"].values():
        assert "class" not in record and "transmission" not in record
    assert dict(result.effective_roles) == ROLES


def test_roof_glass_adds_the_class_keys(tmp_path):
    result = roof_prepare(tmp_path, [SKY_PANE], **SKY_KW)
    manifest = result.manifest
    sky = manifest["groups"]["office_a_sky"]
    assert sky["kind"] == "skylight"
    assert sky["tilt_deg"] == pytest.approx(0.0)
    assert sky["azimuth_bin_deg"] is None
    assert manifest["openings"]["skylight_a"]["class"] == {
        "tilt_deg": 0.0, "tilt_class": "skylight", "position": "roof",
        "diffusing": False, "kind": "skylight",
    }
    window_class = manifest["openings"]["window_ext"]["class"]
    assert (window_class["tilt_class"], window_class["position"], window_class["kind"]) == (
        "vertical", "perimeter", "window",
    )
    (window_group_id,) = [gid for gid, g in manifest["groups"].items() if g["kind"] == "window"]
    assert manifest["groups"][window_group_id]["azimuth_bin_deg"] is not None


@pytest.mark.parametrize(
    "ambient_full, direct, option, message",
    [
        ("-aa 0.1 -ab", "-dc 1", "ambient.full", r"-ab needs a whole number after it"),
        ("-ab two", "-dc 1", "ambient.full", r"-ab needs a whole number after it"),
        ('-ab "2', "-dc 1", "ambient.full", r"cannot be read as options"),
        ("-ab 2", "-dc 1 -ab", "direct", r"-ab needs a whole number after it"),
        ("-ab 2", '-dc "1', "direct", r"cannot be read as options"),
    ],
)
def test_an_unreadable_ab_is_refused_with_diffusing_glass(
    tmp_path, ambient_full, direct, option, message
):
    with pytest.raises(PrepareError, match=rf"^radiance\.{re.escape(option)}: {message}"):
        roof_prepare(
            tmp_path, [DIFFUSING_PANEL], diffuser=True, ambient_full=ambient_full, direct=direct,
        )
    assert not (tmp_path / "prepared").exists()


def test_a_well_depth_without_skylight_glass_writes_no_well_material(tmp_path):
    result = do_prepare(
        tmp_path, project_kwargs={"skylight_well": 0.4, "skylight_well_reflectance": 0.7}
    )
    assert "leed_skylight_well" not in result.materials_file.read_text()
    assert "skylight_well" not in result.manifest
