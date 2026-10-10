# tests/mcp/test_mcp_inputs_spaces_create.py
"""inputs(action="create", file="spaces") (spec 6.3)."""
import hashlib

import pytest

from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture
from merge_split_folder import SPLITS_CSV
from option2_folder import SPACES_CSV, build_option2_project

pytestmark = pytest.mark.anyio

ROW = {"label": "R1", "room": "Lab 1", "space_type": "lab", "regularly_occupied": True,
       "view_preserving_shades": False, "include_base": True}


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _create(tmp_path, folder, rows, **extra):
    return await call(tool_server(tmp_path), "inputs", {"action": "create", "folder": str(folder), "file": "spaces",
                                                       "rows": rows, **extra})


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    (folder / "spaces.csv").unlink()
    return folder


async def test_a_new_spaces_csv_takes_geometry_from_the_rooms_and_the_rest_from_rows(tmp_path, folder):
    result = await _create(tmp_path, folder, [ROW])
    data = result.structured_content
    assert result.is_error is False and codes(result) == []
    assert (folder / "spaces.csv").read_text() == (
        "floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,view_preserving_shades,"
        "merge_into,include_base,reason\nL0,Lab 1,lab,yes,24.00,3.00,2.00,0.00,no,,1,\n")
    assert data["sha256"] == _sha(folder / "spaces.csv") and data["backup"] is None
    assert data["content"]["rows"][0]["room"] == "Lab 1" and "unmatched_rows" in data


async def test_a_room_without_a_row_is_refused_and_no_placeholder_is_written(tmp_path, folder):
    result = await _create(tmp_path, folder, [])
    assert codes(result) == [("ARGUMENT_MISSING", "rows")]
    assert "R1 (room, space_type, regularly_occupied, view_preserving_shades, include_base)" in (
        result.structured_content["reason"])
    assert not (folder / "spaces.csv").exists()


async def test_a_row_missing_one_column_is_refused(tmp_path, folder):
    row = {k: v for k, v in ROW.items() if k != "regularly_occupied"}
    result = await _create(tmp_path, folder, [row])
    assert "R1 (regularly_occupied)" in result.structured_content["reason"]
    assert not (folder / "spaces.csv").exists()


@pytest.mark.parametrize("change, words", [
    ({"label": "R9"}, "label 'R9' is not a room of the scene; the rooms are R1"),
    ({"area_m2": "30"}, "area_m2 comes from the scene's rooms"),
    ({"colour": "red"}, "spaces.csv has no column 'colour'"),
    ({"room": 101}, 'given as "101"'),
    ({"regularly_occupied": "yes"}, "regularly_occupied takes true or false"),
    ({"room": "Lab\r1"}, "rows (R1): room holds a line break or control character"),
    ({"room": "Lab\x0c1"}, "rows (R1): room holds a line break or control character"),
    ({"space_type": "lab\u2029"}, "rows (R1): space_type holds a line break"),
    ({"room": "Lab <!-- x"}, "rows (R1): room holds '<!--'"),
    ({"room": "=HYPERLINK(1)"}, "rows (R1): room starts with '='"),
])
async def test_a_bad_row_is_refused(tmp_path, folder, change, words):
    result = await _create(tmp_path, folder, [{**ROW, **change}])
    assert codes(result) == [("INPUT_INVALID", "rows")] and words in result.structured_content["reason"]


async def test_a_room_name_like_101_is_written_as_text(tmp_path, folder):
    await _create(tmp_path, folder, [{**ROW, "room": "101"}])
    assert "\nL0,101,lab," in (folder / "spaces.csv").read_text()


async def test_replacing_carries_each_matched_rows_values(tmp_path, folder):
    (folder / "spaces.csv").write_text(SPACES_CSV)
    result = await _create(tmp_path, folder, [{"label": "R1", "space_type": "classroom"}],
                           expected_sha256=_sha(folder / "spaces.csv"))
    assert result.is_error is False and result.structured_content["unmatched_rows"] == []
    assert (folder / "spaces.csv").read_text().splitlines()[1] == (
        "L0,Room North,classroom,yes,24.00,3.00,2.00,0.00,no,,1,")
    assert open(result.structured_content["backup"]).read() == SPACES_CSV


async def test_after_a_split_the_old_row_is_listed_and_the_halves_need_rows(tmp_path, folder):
    (folder / "spaces.csv").write_text(SPACES_CSV)
    (folder / "splits.csv").write_text(SPLITS_CSV)
    sha = _sha(folder / "spaces.csv")
    refused = await _create(tmp_path, folder, [], expected_sha256=sha)
    assert codes(refused) == [("ARGUMENT_MISSING", "rows")] and "R1 (" in refused.structured_content["reason"]
    base = {k: v for k, v in ROW.items() if k != "include_base"}
    halves = [{**base, "label": "R1", "room": "West", "include_base_case": True},
              {**base, "label": "R2", "room": "Strip", "include_base_case": False}]
    result = await _create(tmp_path, folder, halves, expected_sha256=sha)
    assert result.is_error is False and result.structured_content["unmatched_rows"] == ["Room North"]
    assert [r["room"] for r in result.structured_content["content"]["rows"]] == ["West", "Strip"]


async def test_two_rooms_with_one_name_are_refused_by_the_loader(tmp_path, folder):
    (folder / "splits.csv").write_text(SPLITS_CSV)
    result = await _create(tmp_path, folder, [ROW, {**ROW, "label": "R2"}])
    assert codes(result) == [("INPUT_INVALID", None)] and "appears twice" in result.structured_content["reason"]
    assert not (folder / "spaces.csv").exists()


async def test_create_spaces_needs_materials(tmp_path, folder):
    (folder / "materials.csv").unlink()
    assert codes(await _create(tmp_path, folder, [ROW])) == [("FILE_MISSING", "folder")]


async def test_an_old_file_without_centroids_lists_every_room_as_unmatched(tmp_path, folder):
    (folder / "spaces.csv").write_text("floor,room,space_type\nL0,Room North,office\nL0, ,lab\nL1,Room West,lab\n")
    result = await _create(tmp_path, folder, [ROW], expected_sha256=_sha(folder / "spaces.csv"))
    assert result.is_error is False and codes(result) == []
    assert result.structured_content["unmatched_rows"] == ["Room North", "Room West"]


@pytest.mark.parametrize("old, why", [
    ("floor,room,space_type\nL0,Room North,office\n".encode("utf-16"), "not UTF-8 text"),
    (b"floor,name,space_type\nL0,Room North,office\n", "it has no room column"),
], ids=["utf-16", "no-room-column"])
async def test_an_old_file_that_cannot_be_matched_gives_a_warning(tmp_path, folder, old, why):
    (folder / "spaces.csv").write_bytes(old)
    result = await _create(tmp_path, folder, [ROW], expected_sha256=_sha(folder / "spaces.csv"))
    assert result.is_error is False and codes(result) == [("INPUT_INVALID", "content")]
    advisory = result.structured_content["advisories"][0]
    assert advisory["severity"] == "warn" and advisory["message"].startswith(
        "spaces.csv: the old file could not be matched (")
    assert why in advisory["message"]
    assert advisory["message"].endswith("; its values were not carried; the backup holds them")
    assert result.structured_content["unmatched_rows"] == []
