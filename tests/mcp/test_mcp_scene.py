# tests/mcp/test_mcp_scene.py
"""The scene tool (spec 6.2): draft-tolerant scan, the materials template, rooms."""
import io

import numpy as np
import pytest

from conftest import symlink_or_skip
from leed_daylight.drivers.materials import materials_folder
from leed_daylight.drivers.rooms import rooms_folder
from leed_daylight.scene.storeys import Room
from leed_mcp.adapters import scene as adapter
from leed_mcp.errors import Refused
from deliver_folder import build_roof_glass_project, build_two_storey_project
from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture
from merge_split_folder import build_split_project
from option2_folder import build_option2_project

pytestmark = pytest.mark.anyio


def _draft(d):
    d["site"]["latitude_N"] = "FILL_ME"


async def test_scan_works_while_values_outside_the_scene_are_fill_me(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea, mutate=_draft)
    result = await call(tool_server(tmp_path), "scene", {"action": "scan", "folder": str(folder)})
    data = result.structured_content
    assert result.is_error is False and codes(result) == []
    assert data["units"] == "m" and data["files"] == ["scene.rad"] and data["surfaces"] == 4
    glass = next(m for m in data["materials"] if m["modifier"] == "glass_clear")
    assert glass == {"modifier": "glass_clear", "primitive_type": "glass", "surfaces": 1, "area_m2": 2.0,
                     "guessed_role": "glazing", "role": "glazing"}
    assert data["storeys"] == [{"index": 0, "level_z": 0.0, "rooms": 1, "area_m2": 24.0}]


async def test_scan_without_materials_lists_materials_and_says_storeys_need_them(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    (folder / "materials.csv").unlink()
    result = await call(tool_server(tmp_path), "scene", {"action": "scan", "folder": str(folder)})
    assert result.structured_content["storeys"] is None
    assert all(m["role"] is None for m in result.structured_content["materials"])
    assert codes(result) == [("FILE_MISSING", "storeys")]


async def test_scan_reports_a_role_missing_as_a_scene_problem(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    lines = (folder / "materials.csv").read_text().splitlines()
    (folder / "materials.csv").write_text("\n".join(l for l in lines if not l.startswith("glass_clear")) + "\n")
    result = await call(tool_server(tmp_path), "scene", {"action": "scan", "folder": str(folder)})
    assert ("SCENE_PROBLEM", "problems") in codes(result)
    assert result.structured_content["storeys"] is None


async def test_scan_refuses_fill_me_in_the_scene_section(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea,
                                   mutate=lambda d: d["scene"].update(units="FILL_ME"))
    result = await call(tool_server(tmp_path), "scene", {"action": "scan", "folder": str(folder)})
    assert codes(result) == [("INPUT_INVALID", None)]
    assert "scene.units still says FILL_ME" in result.structured_content["reason"]


async def test_an_xform_target_outside_the_folder_is_refused_before_reading(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    (tmp_path / "outside.rad").write_text("void plastic leak\n0\n0\n5 1 1 1 0 0\n")
    with open(folder / "scene.rad", "a") as handle:
        handle.write("\n!xform ../outside.rad\n")
    result = await call(tool_server(tmp_path), "scene", {"action": "scan", "folder": str(folder)})
    assert codes(result) == [("SCENE_INCLUDE_OUTSIDE", None)]
    assert "leak" not in str(result.structured_content)


async def test_a_scene_file_link_leading_out_is_refused(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    (tmp_path / "real.rad").write_text((folder / "scene.rad").read_text())
    (folder / "scene.rad").unlink()
    symlink_or_skip(folder / "scene.rad", tmp_path / "real.rad")
    result = await call(tool_server(tmp_path), "scene", {"action": "rooms", "folder": str(folder)})
    assert codes(result) == [("SCENE_INCLUDE_OUTSIDE", None)]


def test_an_xform_inside_the_folder_passes_the_check(tmp_path):
    folder = tmp_path.resolve()
    (folder / "wing.rad").write_text("")
    (folder / "main.rad").write_text("!xform -t 10 0 0 wing.rad\n!xform -rz 90 -t 0 0 3 wing.rad\n")
    assert adapter.check_inside(folder, [folder / "main.rad"]) is None


async def test_materials_template_matches_the_cli_byte_for_byte(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    cli = materials_folder(folder, out=io.StringIO()).read_bytes()
    (folder / "_run" / "materials_template.csv").unlink()
    result = await call(tool_server(tmp_path), "scene", {"action": "materials_template", "folder": str(folder)})
    assert (folder / "_run" / "materials_template.csv").read_bytes() == cli
    row = next(r for r in result.structured_content["rows"] if r["modifier"] == "glass_clear")
    assert row == {"modifier": "glass_clear", "primitive_type": "glass",
                   "note": "guess: glazing; 1 surfaces, 2.00 m2"}


async def test_rooms_without_materials_is_file_missing(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    (folder / "materials.csv").unlink()
    result = await call(tool_server(tmp_path), "scene", {"action": "rooms", "folder": str(folder)})
    assert codes(result) == [("FILE_MISSING", "folder")]


async def test_rooms_gives_outline_centroid_and_the_facade_pane(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea, mutate=_draft)
    data = (await call(tool_server(tmp_path), "scene", {"action": "rooms", "folder": str(folder)})).structured_content
    (room,) = data["rooms"]
    assert room["label"] == "R1" and room["storey"] == 0 and room["area_m2"] == 24.0
    assert room["centroid"] == [3.0, 2.0] and len(room["outline"]) == 4 and room["split"] is False
    (pane,) = data["panes"]
    assert pane["room"] == "R1" and pane["kind"] == "window" and pane["area_m2"] == 2.0
    assert pane["openings"] == [{"name": "window_ext", "shape": "segment", "points": [[2.0, 0.0], [4.0, 0.0]]}]
    assert (folder / "_run" / "spaces_template.csv").is_file()


async def test_rooms_applies_splits_and_the_template_matches_the_cli(tmp_path, write_example, write_wea):
    folder = build_split_project(tmp_path, write_example, write_wea)
    cli = rooms_folder(folder, out=io.StringIO()).read_bytes()
    (folder / "_run" / "spaces_template.csv").unlink()
    data = (await call(tool_server(tmp_path), "scene", {"action": "rooms", "folder": str(folder)})).structured_content
    assert [(r["label"], r["area_m2"], r["split"]) for r in data["rooms"]] == [("R1", 18.0, True), ("R2", 6.0, True)]
    assert data["split_lines"] == [{"level_z": 0.0, "points": [[4.5, -1.0], [4.5, 5.0]], "note": "invented west edge"}]
    assert (folder / "_run" / "spaces_template.csv").read_bytes() == cli


async def test_rooms_gives_roof_panes_as_polygons(tmp_path, write_example, write_wea):
    folder = build_roof_glass_project(tmp_path, write_example, write_wea)
    data = (await call(tool_server(tmp_path), "scene", {"action": "rooms", "folder": str(folder)})).structured_content
    shapes = {p["kind"]: p["openings"][0]["shape"] for p in data["panes"]}
    assert shapes == {"window": "segment", "skylight": "polygon", "diffusing": "polygon"}


async def test_rooms_filters_by_storey_and_refuses_an_unknown_one(tmp_path, write_example, write_wea):
    folder = build_two_storey_project(tmp_path, write_example, write_wea, options=())
    server = tool_server(tmp_path)
    data = (await call(server, "scene", {"action": "rooms", "folder": str(folder), "storey": 1})).structured_content
    assert data["storeys"] == [0, 1] and [r["label"] for r in data["rooms"]] == ["R3"]
    assert [p["room"] for p in data["panes"]] == ["R3"]
    result = await call(server, "scene", {"action": "rooms", "folder": str(folder), "storey": 7})
    assert codes(result) == [("INPUT_INVALID", "storey")]


async def test_rooms_cuts_outlines_at_the_vertex_budget(tmp_path, write_example, write_wea, monkeypatch):
    monkeypatch.setattr(adapter, "MAX_TOTAL_VERTICES", 6)
    folder = build_two_storey_project(tmp_path, write_example, write_wea, options=())
    result = await call(tool_server(tmp_path), "scene", {"action": "rooms", "folder": str(folder)})
    outlines = [r["outline"] is not None for r in result.structured_content["rooms"]]
    assert outlines == [True, False, False]
    assert ("RESULT_TRUNCATED", "rooms") in codes(result)


def test_a_ring_over_500_vertices_is_simplified():
    angles = np.linspace(0, 2 * np.pi, 1200, endpoint=False)
    circle = np.column_stack([5 + 5 * np.cos(angles), 5 + 5 * np.sin(angles)])
    room = Room(room_id="S0_R0", storey=0, level_z=0.0, outline=circle, area_m2=78.5, ceiling_height_m=None)
    outline, holes, simplified = adapter.rings(room)
    assert simplified is True and 3 <= len(outline) <= adapter.MAX_RING_VERTICES and holes == []


async def test_scene_takes_storey_only_for_rooms(tmp_path):
    result = await call(tool_server(tmp_path), "scene", {"action": "scan", "folder": str(tmp_path), "storey": 0})
    assert codes(result) == [("UNKNOWN_ARGUMENT", "storey")]


def test_an_absolute_self_including_scene_file_passes_and_one_outside_is_refused(tmp_path):
    folder = tmp_path / "p"
    folder.mkdir()
    folder = folder.resolve()
    (folder / "loop.rad").write_text("!xform loop.rad\n")
    assert adapter.check_inside(folder, [folder / "loop.rad"]) is None
    (tmp_path / "out.rad").write_text("")
    with pytest.raises(Refused):
        adapter.check_inside(folder, [tmp_path / "out.rad"])


@pytest.mark.parametrize("name", ["project.yaml", "materials.csv", "splits.csv"])
async def test_a_linked_project_file_is_path_unsafe_and_never_read(tmp_path, write_example, write_wea, name):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    outside = tmp_path / "outside_copy"
    outside.write_bytes((folder / name).read_bytes() if (folder / name).exists() else b"")
    (folder / name).parent.mkdir(exist_ok=True)
    (folder / name).unlink(missing_ok=True)
    symlink_or_skip(folder / name, outside)
    result = await call(tool_server(tmp_path), "scene", {"action": "rooms", "folder": str(folder)})
    assert codes(result) == [("PATH_UNSAFE", None)]


async def test_a_modifier_a_spreadsheet_would_run_is_refused_with_a_rename_hint(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    for path in [folder / "scene.rad"]:
        path.write_text(path.read_text().replace("wall_paint", "-wall_paint"))
    result = await call(tool_server(tmp_path), "scene", {"action": "materials_template", "folder": str(folder)})
    assert codes(result) == [("INPUT_INVALID", "rows")]
    assert "-wall_paint" in result.structured_content["reason"] and "rename" in result.structured_content["reason"]
    assert not (folder / "_run" / "materials_template.csv").exists()
