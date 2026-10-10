# tests/mcp/test_mcp_inputs_materials.py
"""inputs create and update of materials.csv (spec 6.3)."""
import hashlib

import pytest

from conftest import symlink_or_skip
from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture
from option2_folder import MATERIALS_CSV, build_option2_project

pytestmark = pytest.mark.anyio

ROLES = [{"modifier": "wall_paint", "role": "wall"}, {"modifier": "ceiling_tile", "role": "ceiling"},
         {"modifier": "floor_carpet", "role": "floor"}, {"modifier": "glass_clear", "role": "glazing"}]


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _inputs(tmp_path, folder, action, **arguments):
    return await call(tool_server(tmp_path), "inputs", {"action": action, "folder": str(folder),
                                                       "file": "materials", **arguments})


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    (folder / "materials.csv").unlink()
    return folder


async def test_create_writes_every_scene_modifier_with_its_role(tmp_path, folder):
    result = await _inputs(tmp_path, folder, "create", rows=ROLES, expected_sha256=None)
    data = result.structured_content
    assert result.is_error is False and codes(result) == []
    assert (folder / "materials.csv").read_text() == (
        "modifier,primitive_type,role,note\nfloor_carpet,plastic,floor,\nceiling_tile,plastic,ceiling,\n"
        "wall_paint,plastic,wall,\nglass_clear,glass,glazing,\n")
    assert data["sha256"] == _sha(folder / "materials.csv") and data["backup"] is None
    assert data["content"]["rows"][3] == {"modifier": "glass_clear", "primitive_type": "glass",
                                          "role": "glazing", "note": ""}


async def test_create_refuses_a_modifier_without_a_role_and_writes_nothing(tmp_path, folder):
    rows = ROLES[:3] + [{"modifier": "glass_clear", "role": ""}]
    result = await _inputs(tmp_path, folder, "create", rows=rows)
    assert codes(result) == [("ARGUMENT_MISSING", "rows")]
    assert "these have none: glass_clear" in result.structured_content["reason"]
    assert not (folder / "materials.csv").exists()


@pytest.mark.parametrize("row, field, words", [
    ({"modifier": "marble", "role": "floor"}, "rows", "'marble' is not a modifier of the scene"),
    ({"modifier": "wall_paint", "role": "wall", "primitive_type": "metal"}, "rows", "comes from the scene"),
    ({"modifier": "wall_paint", "role": "wall", "colour": "white"}, "rows", "unknown key(s) colour"),
    ({"modifier": "wall_paint", "role": 3}, "rows", "role must be text"),
    ({"modifier": "wall_paint", "role": "wal"}, None, "materials.csv"),
    ({"modifier": "wall_paint", "role": "wall", "note": "nul\x00here"}, "rows",
     "rows[4] (wall_paint): note holds a line break or control character"),
    ({"modifier": "wall_paint", "role": "wall", "note": "two\nlines"}, "rows", "note holds a line break"),
    ({"modifier": "wall_paint", "role": "wall", "view": "clear\u2028"}, "rows", "view holds a line break"),
    ({"modifier": "wall_paint", "role": "wall", "note": "see <!-- auto:spaces -->"}, "rows", "note holds '<!--'"),
    ({"modifier": "wall_paint", "role": "=wall"}, "rows", "rows[4] (wall_paint): role starts with '='"),
    ({"modifier": "wall_paint", "role": "wall", "note": " @SUM(1)"}, "rows", "note starts with '@'"),
    ({"modifier": "wall_paint", "role": "wall", "view": "-1"}, "rows", "view starts with '-'"),
])
async def test_create_refuses_a_bad_row(tmp_path, folder, row, field, words):
    result = await _inputs(tmp_path, folder, "create", rows=ROLES[1:] + [row])
    assert codes(result) == [("INPUT_INVALID", field)]
    assert words in result.structured_content["reason"]
    assert not (folder / "materials.csv").exists()


async def test_create_over_an_existing_file_needs_its_sha256(tmp_path, folder):
    (folder / "materials.csv").write_text(MATERIALS_CSV, newline="\n")
    refused = await _inputs(tmp_path, folder, "create", rows=ROLES)
    assert codes(refused) == [("INPUT_CHANGED_ON_DISK", "expected_sha256")]
    result = await _inputs(tmp_path, folder, "create", rows=ROLES, expected_sha256=_sha(folder / "materials.csv"))
    assert result.is_error is False and open(result.structured_content["backup"]).read() == MATERIALS_CSV


async def test_the_view_column_is_written_only_when_a_row_has_a_view(tmp_path, folder):
    rows = ROLES[:3] + [{"modifier": "glass_clear", "role": "glazing", "view": "clear"}]
    await _inputs(tmp_path, folder, "create", rows=rows)
    lines = (folder / "materials.csv").read_text().splitlines()
    assert lines[0] == "modifier,primitive_type,role,note,view" and lines[4] == "glass_clear,glass,glazing,,clear"


async def test_update_changes_the_rows_it_names(tmp_path, folder):
    (folder / "materials.csv").write_text(MATERIALS_CSV, newline="\n")
    result = await _inputs(tmp_path, folder, "update", expected_sha256=_sha(folder / "materials.csv"),
                           rows=[{"modifier": "wall_paint", "note": "painted plaster, invented"}])
    assert result.is_error is False
    assert "wall_paint,plastic,wall,\"painted plaster, invented\"" in (folder / "materials.csv").read_text()


async def test_update_adds_a_scene_modifier_the_file_lacks(tmp_path, folder):
    (folder / "materials.csv").write_text(MATERIALS_CSV.replace("glass_clear,glass,glazing,\n", ""), newline="\n")
    result = await _inputs(tmp_path, folder, "update", expected_sha256=_sha(folder / "materials.csv"),
                           rows=[{"modifier": "glass_clear", "role": "glazing"}])
    assert result.is_error is False
    assert (folder / "materials.csv").read_text().endswith("glass_clear,glass,glazing,\n")


async def test_update_without_the_file_is_file_missing(tmp_path, folder):
    result = await _inputs(tmp_path, folder, "update", rows=[{"modifier": "wall_paint", "role": "wall"}])
    assert codes(result) == [("FILE_MISSING", "file")]


async def test_a_linked_materials_csv_is_never_written_through(tmp_path, folder):
    (tmp_path / "real.csv").write_text(MATERIALS_CSV, newline="\n")
    symlink_or_skip(folder / "materials.csv", tmp_path / "real.csv")
    result = await _inputs(tmp_path, folder, "update", expected_sha256=_sha(tmp_path / "real.csv"),
                           rows=[{"modifier": "wall_paint", "role": "ceiling"}])
    assert codes(result) == [("PATH_UNSAFE", None)]
    assert (tmp_path / "real.csv").read_text() == MATERIALS_CSV


async def test_rows_is_for_materials_and_spaces_only(tmp_path, folder):
    result = await call(tool_server(tmp_path), "inputs", {"action": "create", "folder": str(folder),
                                                         "file": "project", "rows": ROLES})
    assert codes(result) == [("INPUT_INVALID", "file")]
    assert "works on these files: materials" in result.structured_content["reason"]


async def test_update_writes_a_vlt_column_and_a_run_stays_current(tmp_path, write_example, write_wea):
    from deliver_folder import CASE, build_deliver_project
    project = build_deliver_project(tmp_path, write_example, write_wea)
    path = project / "materials.csv"
    result = await _inputs(tmp_path, project, "update", expected_sha256=_sha(path),
                           rows=[{"modifier": "glass_clear", "vlt": 62}])
    assert result.is_error is False and codes(result) == []
    lines = path.read_text().splitlines()
    assert lines[0] == "modifier,primitive_type,role,note,vlt"
    assert "glass_clear,glass,glazing,,62" in lines
    summary = await call(tool_server(tmp_path), "results",
                         {"action": "summary", "folder": str(project), "case": CASE, "credit": "option1"})
    assert summary.structured_content["inputs"] == "current"


@pytest.mark.parametrize("layout", ["crlf_bom", "empty_view"])
async def test_adding_a_vlt_by_update_keeps_the_materials_digest(tmp_path, folder, layout):
    from leed_daylight.project.inputs import materials_digest
    path = folder / "materials.csv"
    if layout == "crlf_bom":
        path.write_bytes(b"\xef\xbb\xbf" + MATERIALS_CSV.replace("\n", "\r\n").encode("utf-8"))
    else:
        lines = MATERIALS_CSV.splitlines()
        path.write_text("\n".join([lines[0] + ",view"] + [line + "," for line in lines[1:]]) + "\n", newline="\n")
    before_digest, before = materials_digest(path), path.read_bytes()
    result = await _inputs(tmp_path, folder, "update", expected_sha256=_sha(path),
                           rows=[{"modifier": "glass_clear", "vlt": 62}])
    assert result.is_error is False and codes(result) == []
    after = path.read_bytes()
    assert b",vlt" in after and after != before
    assert materials_digest(path) == before_digest
    if layout == "crlf_bom":
        assert after.startswith(b"\xef\xbb\xbf") and after.count(b"\r\n") == before.count(b"\r\n")
    else:
        assert after.decode("utf-8").splitlines()[0] == "modifier,primitive_type,role,note,view,vlt"


async def test_update_keeps_the_other_rows_view_note_and_vlt(tmp_path, folder):
    path = folder / "materials.csv"
    path.write_text("modifier,primitive_type,role,note,view,vlt\n"
                    "floor_carpet,plastic,floor,carpet note,,\n"
                    "ceiling_tile,plastic,ceiling,,,\n"
                    "wall_paint,plastic,wall,,blocks,\n"
                    "glass_clear,glass,glazing,clear glass,clear,60\n", newline="\n")
    result = await _inputs(tmp_path, folder, "update", expected_sha256=_sha(path),
                           rows=[{"modifier": "ceiling_tile", "note": "invented"}])
    assert result.is_error is False
    assert path.read_text() == ("modifier,primitive_type,role,note,view,vlt\n"
                                "floor_carpet,plastic,floor,carpet note,,\n"
                                "ceiling_tile,plastic,ceiling,invented,,\n"
                                "wall_paint,plastic,wall,,blocks,\n"
                                "glass_clear,glass,glazing,clear glass,clear,60\n")


async def test_create_refuses_a_scene_modifier_a_spreadsheet_would_run(tmp_path, folder):
    for path in [folder / "scene.rad"]:
        path.write_text(path.read_text().replace("wall_paint", "=wall_paint"), newline="\n")
    roles = [{**row, "modifier": "=wall_paint"} if row["modifier"] == "wall_paint" else row for row in ROLES]
    result = await _inputs(tmp_path, folder, "create", rows=roles, expected_sha256=None)
    assert codes(result) == [("INPUT_INVALID", "rows")] and "rename" in result.structured_content["reason"]
    assert not (folder / "materials.csv").exists()
