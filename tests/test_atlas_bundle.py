# tests/test_atlas_bundle.py
"""The atlas bundle (atlas spec 5, 6, 7) and bundle.npz.

Built on the invented project of tests/deliver_folder.py and its roof-glass
and two-storey variants. The prepared scene is read in metres whatever the
project's units, so a scene drawn in millimetres gives the same bundle.
Every number is invented and round.
"""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from atlas_folder import atlas_inputs, edit_points, millimetre_copy
from deliver_folder import build_deliver_project, build_roof_glass_project, build_two_storey_project
from leed_daylight.deliver.atlas.bundle import (
    BELOW, SITE, Bundle, RoomLabel, build_bundle, classify, storey_of_z, write_npz,
)
from leed_daylight.deliver.atlas.colours import KIND_RGBA, KINDS
from leed_daylight.deliver.data import DeliverError
from leed_daylight.deliver.figures.spec import NEUTRAL_RGB
from leed_daylight.scene.reader import read_scene


# -- kinds -----------------------------------------------------------------

def _kinds_by_name(prepared) -> dict[str, str]:
    return {prim.name: kind for prim, kind in classify(prepared)}


def test_every_surface_of_the_prepared_scene_gets_exactly_one_kind(tmp_path, write_example, write_wea):
    folder = build_roof_glass_project(tmp_path, write_example, write_wea, options=())
    _, prepared, _ = atlas_inputs(folder, tmp_path / "work")

    classified = classify(prepared)
    scene = read_scene([prepared.scene_file], "m")
    blinds = read_scene(list(prepared.blind_files.values()), "m")

    assert [prim.name for prim, _ in classified] == [p.name for p in scene] + [p.name for p in blinds]
    assert all(kind in KINDS for _, kind in classified)
    kinds = _kinds_by_name(prepared)
    assert kinds["diffuser_a"] == "glass_diffusing"  # a diffusing roof panel: diffusing wins over roof
    assert kinds["skylight_a"] == "glass_roof"
    assert kinds["window_ext"] == "glass_window"
    assert kinds["skylight_a_well_0"] == "skylight_well"
    assert kinds["window_ext_reveal_0"] == "reveal"
    assert kinds["skylight_a_frame_0"] == "frame"
    assert kinds["Room_North_sky_blind_0"] == "blind"
    assert kinds["wall_south_below"] == "wall"
    assert kinds["floor_room_north"] == "floor"
    assert kinds["ceiling_north_south_strip"] == "ceiling"


def test_role_ignore_is_not_drawn(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=())
    _, prepared, _ = atlas_inputs(folder, tmp_path / "work")
    roles = dict(prepared.effective_roles, wall_paint="ignore")

    kinds = _kinds_by_name(dataclasses.replace(prepared, effective_roles=types.MappingProxyType(roles)))

    assert "wall_room_north_south" not in kinds
    assert "floor_room_north" in kinds


def test_a_surface_with_no_role_is_refused(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=())
    _, prepared, _ = atlas_inputs(folder, tmp_path / "work")
    roles = {m: r for m, r in prepared.effective_roles.items() if m != "wall_paint"}

    with pytest.raises(DeliverError, match="surface 'wall_room_north_south' has modifier 'wall_paint', which has no role"):
        classify(dataclasses.replace(prepared, effective_roles=types.MappingProxyType(roles)))


# -- storeys ---------------------------------------------------------------

def test_a_surface_exactly_at_a_level_goes_up():
    levels = (0.0, 4.0)
    assert storey_of_z(4.0, levels) == 1
    assert storey_of_z(4.0 - 1e-9, levels) == 1  # a millimetre scene's rounding still goes up
    assert storey_of_z(3.9, levels) == 0
    assert storey_of_z(0.0, levels) == 0
    assert storey_of_z(40.0, levels) == 1  # the top storey has no upper bound


def test_two_labels_on_one_level_take_the_first():
    levels = (0.0, 0.0, 4.0)
    assert storey_of_z(0.0, levels) == 0
    assert storey_of_z(2.0, levels) == 0
    assert storey_of_z(4.0, levels) == 2


def test_below_the_lowest_level():
    assert storey_of_z(-0.5, (0.0, 4.0)) == BELOW == -2
    assert SITE == -1


def test_surfaces_take_the_storey_of_their_lowest_point(tmp_path, write_example, write_wea):
    folder = build_two_storey_project(tmp_path, write_example, write_wea, options=())
    _, _, bundle = atlas_inputs(folder, tmp_path / "work")

    assert bundle.storeys == ("L0", "L1")
    assert bundle.levels == (0.0, 4.0)
    floors = bundle.surfaces["floor"]
    assert floors.storey.tolist() == [0, 0, 1]  # floor_room_upper lies exactly at level 4.0
    assert bundle.surfaces["ceiling"].storey.tolist() == [0, 1]
    assert floors.counts.tolist() == [4, 4, 4]
    assert floors.verts.shape == (12, 3)
    assert floors.verts[8:].tolist() == [[0.0, 0.0, 4.0], [6.0, 0.0, 4.0], [6.0, 4.0, 4.0], [0.0, 4.0, 4.0]]
    assert floors.counts.dtype == np.int32 and floors.storey.dtype == np.int16


def test_site_kinds_have_no_storey(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=())
    case, prepared, _ = atlas_inputs(folder, tmp_path / "work")
    roles = dict(prepared.effective_roles, wall_paint="context")

    bundle = build_bundle(case, dataclasses.replace(prepared, effective_roles=types.MappingProxyType(roles)))

    assert "wall" not in bundle.surfaces
    assert bundle.surfaces["context"].storey.tolist() == [SITE]


def test_rooms_are_labelled_included_or_excluded(tmp_path, write_example, write_wea):
    folder = build_two_storey_project(tmp_path, write_example, write_wea, options=())
    _, _, bundle = atlas_inputs(folder, tmp_path / "work")

    assert bundle.rooms == (
        RoomLabel("Room North", 0, 3.0, 2.0, 0.0, True),
        RoomLabel("Store East", 0, 10.0, 2.0, 0.0, False),
        RoomLabel("Room Upper", 1, 3.0, 2.0, 4.0, True),
    )


# -- points ----------------------------------------------------------------

def test_model_check_points_are_the_grid(tmp_path, write_example, write_wea):
    folder = build_two_storey_project(tmp_path, write_example, write_wea, options=())
    case, prepared, bundle = atlas_inputs(folder, tmp_path / "work")

    grid = bundle.grid
    assert grid.xyz.shape == (sum(g.points.shape[0] for g in prepared.grids), 3)
    assert np.array_equal(grid.xyz, np.vstack([g.points for g in prepared.grids]))
    assert grid.room == tuple(g.room for g in prepared.grids for _ in range(g.points.shape[0]))
    assert sorted(set(zip(grid.room, grid.storey.tolist()))) == [("Room North", 0), ("Room Upper", 1)]
    assert not grid.values.any()
    assert (grid.rgb == np.array(NEUTRAL_RGB, dtype=np.uint8)).all()
    assert bundle.layers == ()
    assert bundle.cell_m == 0.6
    assert bundle.work_plane_m == case.project.grid.work_plane_m


def test_result_points_come_from_points_csv(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea)
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    layers = {layer.key: points for layer, points in bundle.layers}
    for key, points in layers.items():
        option, column = key.split(".", 1)
        table = case.options[option].points
        assert np.array_equal(points.xyz[:, :2], table.xy)
        assert np.array_equal(points.xyz[:, 2], table.columns["z"])
        assert np.array_equal(points.values, np.asarray(table.columns[column], dtype=float))
        assert points.room == table.rooms
        assert points.rgb.shape == (len(table.rooms), 3)
    assert layers["option1.sda_pass"].values.tolist()[:2] == [1.0, 1.0]


def test_points_take_their_storey_from_the_room_not_the_storey_column(tmp_path, write_example, write_wea):
    folder = build_two_storey_project(tmp_path, write_example, write_wea, options=("option1",))

    def renumber(header, rows):
        for row in rows:
            row[header.index("storey")] = "9"

    edit_points(folder, "option1", renumber)
    _, _, bundle = atlas_inputs(folder, tmp_path / "work")

    points = dict((layer.key, p) for layer, p in bundle.layers)["option1.da_pct"]
    assert sorted(set(zip(points.room, points.storey.tolist()))) == [("Room North", 0), ("Room Upper", 1)]


def test_a_stray_room_is_left_out_with_the_warning(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=("option1",))

    def add_stray(header, rows):
        stray = list(rows[0])
        stray[header.index("room")] = "Room Nowhere"
        rows.append(stray)

    edit_points(folder, "option1", add_stray)
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    assert "option1: 1 point(s) name a room that is not in spaces.csv; they are left out of the plans" in case.warnings
    points = dict((layer.key, p) for layer, p in bundle.layers)["option1.da_pct"]
    assert len(case.options["option1"].points.rooms) == points.xyz.shape[0] + 1
    assert "Room Nowhere" not in points.room


def test_a_missing_column_is_refused(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=("option1",))

    def drop(header, rows):
        at = header.index("ase_hours")
        del header[at]
        for row in rows:
            del row[at]

    edit_points(folder, "option1", drop)
    with pytest.raises(DeliverError, match=r"points.csv: no 'ase_hours' column; rerun python -m leed_daylight.drivers.run"):
        atlas_inputs(folder, tmp_path / "work")


# -- units (Review Focus 2) --------------------------------------------------

def _assert_same_bundle(a: Bundle, b: Bundle) -> None:
    assert (a.storeys, a.levels, a.rooms, a.cell_m, a.work_plane_m) == (b.storeys, b.levels, b.rooms, b.cell_m,
                                                                        b.work_plane_m)
    assert list(a.surfaces) == list(b.surfaces)
    for kind in a.surfaces:
        np.testing.assert_allclose(a.surfaces[kind].verts, b.surfaces[kind].verts, rtol=0.0, atol=1e-9)
        assert np.array_equal(a.surfaces[kind].counts, b.surfaces[kind].counts)
        assert np.array_equal(a.surfaces[kind].storey, b.surfaces[kind].storey)
    pairs = [(a.grid, b.grid)] + [(pa, pb) for (_, pa), (_, pb) in zip(a.layers, b.layers)]
    assert [layer for layer, _ in a.layers] == [layer for layer, _ in b.layers]
    for pa, pb in pairs:
        np.testing.assert_allclose(pa.xyz, pb.xyz, rtol=0.0, atol=1e-9)
        assert (pa.room, pa.storey.tolist(), pa.values.tolist(), pa.rgb.tolist()) == (
            pb.room, pb.storey.tolist(), pb.values.tolist(), pb.rgb.tolist())


def test_a_millimetre_scene_gives_the_same_bundle(tmp_path, write_example, write_wea):
    folder = build_roof_glass_project(tmp_path, write_example, write_wea, options=())
    mm = millimetre_copy(folder, tmp_path / "mm")
    assert "12 0 0 0 6000 0 0 6000 4000 0 0 4000 0" in (mm / "scene.rad").read_text()

    _, _, metres = atlas_inputs(folder, tmp_path / "work_m")
    _, _, millimetres = atlas_inputs(mm, tmp_path / "work_mm")

    _assert_same_bundle(metres, millimetres)
    assert "glass_diffusing" in millimetres.surfaces
    assert float(millimetres.surfaces["floor"].verts[:, 0].max()) == pytest.approx(6.0)


# -- bundle.npz --------------------------------------------------------------

def test_write_npz_holds_the_keys_the_blender_script_reads(tmp_path, write_example, write_wea):
    folder = build_two_storey_project(tmp_path, write_example, write_wea)
    _, _, bundle = atlas_inputs(folder, tmp_path / "work")
    path = tmp_path / "out" / "bundle.npz"
    path.parent.mkdir()

    write_npz(bundle, path)

    with np.load(path, allow_pickle=False) as npz:
        data = {key: npz[key] for key in npz.files}
    kinds = [k for k in KINDS if k in bundle.surfaces]
    n_layers = len(bundle.layers)
    expected = {"storeys", "levels", "kinds", "kind_rgba", "grid_v", "grid_s", "grid_room", "layer_keys",
                "layer_titles", "room_names", "room_xyz", "room_s", "room_included", "cell_m", "work_plane_m"}
    expected |= {f"k_{k}_{s}" for k in kinds for s in ("v", "n", "s")}
    expected |= {f"l{i}_{s}" for i in range(n_layers) for s in ("v", "s", "room", "value", "rgb")}
    assert set(data) == expected
    assert data["storeys"].tolist() == ["L0", "L1"]
    assert data["levels"].tolist() == [0.0, 4.0]
    assert data["kinds"].tolist() == kinds
    assert data["kind_rgba"].tolist() == [list(KIND_RGBA[k]) for k in kinds]
    assert data["kind_rgba"].dtype == np.uint8
    floor = bundle.surfaces["floor"]
    assert data["k_floor_v"].dtype == np.float32 and np.array_equal(data["k_floor_v"], floor.verts.astype(np.float32))
    assert data["k_floor_n"].dtype == np.int32 and data["k_floor_n"].tolist() == floor.counts.tolist()
    assert data["k_floor_s"].dtype == np.int16 and data["k_floor_s"].tolist() == [0, 0, 1]
    assert data["grid_v"].dtype == np.float32 and data["grid_v"].shape == bundle.grid.xyz.shape
    assert data["grid_s"].dtype == np.int16
    assert data["grid_room"].tolist() == list(bundle.grid.room)
    assert n_layers == 12
    assert data["layer_keys"].tolist() == [layer.key for layer, _ in bundle.layers]
    assert data["layer_titles"].tolist() == [layer.title for layer, _ in bundle.layers]
    layer, points = bundle.layers[2]
    assert data["l2_value"].dtype == np.float64 and data["l2_value"].tolist() == points.values.tolist()
    assert data["l2_rgb"].dtype == np.uint8 and data["l2_rgb"].tolist() == points.rgb.tolist()
    assert data["l2_room"].tolist() == list(points.room)
    assert data["l2_s"].dtype == np.int16 and data["l2_v"].dtype == np.float32
    assert data["room_names"].tolist() == ["Room North", "Store East", "Room Upper"]
    assert data["room_xyz"].tolist() == [[3.0, 2.0, 0.0], [10.0, 2.0, 0.0], [3.0, 2.0, 4.0]]
    assert data["room_s"].tolist() == [0, 0, 1] and data["room_s"].dtype == np.int16
    assert data["room_included"].tolist() == [True, False, True]
    assert data["cell_m"].shape == () and float(data["cell_m"]) == 0.6
    assert data["work_plane_m"].shape == () and data["work_plane_m"].dtype == np.float64


def test_write_npz_of_a_model_check_has_no_layer(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=())
    _, _, bundle = atlas_inputs(folder, tmp_path / "work")
    path = tmp_path / "bundle.npz"

    write_npz(bundle, path)

    with np.load(path, allow_pickle=False) as npz:
        assert npz["layer_keys"].shape == (0,)
        assert npz["layer_titles"].shape == (0,)
        assert not any(key.startswith("l0_") for key in npz.files)
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".npz") == ["bundle.npz"]
