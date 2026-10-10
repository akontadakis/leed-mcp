# tests/mcp/test_mcp_inputs_spaces_update.py
"""inputs(action="update", file="spaces") (spec 6.3)."""
import hashlib

import pytest

from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture
from merge_split_folder import SPLITS_CSV, build_merge_project
from option2_folder import build_option2_project

pytestmark = pytest.mark.anyio


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _update(tmp_path, folder, **change):
    return await call(tool_server(tmp_path), "inputs", {
        "action": "update", "folder": str(folder), "file": "spaces",
        "expected_sha256": _sha(folder / "spaces.csv"), **change})


async def test_rename_follows_into_merge_into_and_columns_change(tmp_path, write_example, write_wea):
    folder = build_merge_project(tmp_path, write_example, write_wea)
    result = await _update(tmp_path, folder, rows=[
        {"floor": "L0", "room": "Room North", "rename_to": "Open Office", "space_type": "open office"}])
    assert result.is_error is False and codes(result) == []
    lines = (folder / "spaces.csv").read_text().splitlines()
    assert lines[1].startswith("L0,Open Office,open office,yes,24,")
    assert lines[2].split(",")[9] == "Open Office"


async def test_add_and_remove_a_case(tmp_path, write_example, write_wea):
    folder = build_merge_project(tmp_path, write_example, write_wea)
    added = await _update(tmp_path, folder, add_case={"name": "scheme b", "kind": "include", "value": False},
                          rows=[{"floor": "L0", "room": "Room North", "include_scheme b": True},
                                {"floor": "L0", "room": "Room East", "include_scheme b": True}])
    assert added.is_error is False
    assert added.structured_content["content"]["columns"][10:13] == [
        "include_base_case", "include_scheme b", "views_base_case"]
    assert [r["include_scheme b"] for r in added.structured_content["content"]["rows"]] == ["1", "1"]
    removed = await _update(tmp_path, folder, remove_case="include_scheme b")
    assert removed.is_error is False and "include_scheme b" not in (folder / "spaces.csv").read_text()


async def test_a_first_views_case_goes_after_the_include_columns(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    result = await _update(tmp_path, folder, add_case={"name": "base_case", "kind": "views", "value": True})
    assert result.structured_content["content"]["columns"][10:13] == ["include_base_case", "views_base_case",
                                                                      "reason"]


@pytest.mark.parametrize("change, field, words", [
    ({"rows": [{"floor": "L1", "room": "Room North", "space_type": "lab"}]}, "rows", "no row on floor 'L1'"),
    ({"rows": [{"floor": "L0", "room": "Room North", "area_m2": "30"}]}, "rows", "area_m2 comes from"),
    ({"rows": [{"floor": "L0", "room": "Room North", "rename_to": " "}]}, "rows", "rename_to is blank"),
    ({"add_case": {"name": "a/b", "kind": "include", "value": True}}, "add_case", "not a safe case name"),
    ({"add_case": {"name": "base_case", "kind": "include", "value": True}}, "add_case", "already has"),
    ({"remove_case": "reason"}, "remove_case", "not a case column"),
    ({"remove_case": "include_base_case"}, None, "no include_<case> column"),
    ({"rows": [{"floor": "L0", "room": "Room East", "include_base_case": False}]}, None, "a piece row must"),
    ({"rows": [{"floor": "L0", "room": "Room North", "space_type": "lab\nnext"}]}, "rows",
     "rows[1]: space_type holds a line break or control character"),
    ({"rows": [{"floor": "L0", "room": "Room North", "reason": "tab\there"}]}, "rows",
     "rows[1]: reason holds a line break or control character"),
    ({"rows": [{"floor": "L0", "room": "Room North", "reason": "nul\x00here"}]}, "rows", "reason holds"),
    ({"rows": [{"floor": "L0", "room": "Room North", "rename_to": "A\u2028B"}]}, "rows",
     "rows[1]: rename_to holds a line break"),
    ({"rows": [{"floor": "L0", "room": "Room North", "rename_to": "A\x85B"}]}, "rows", "rename_to holds"),
    ({"rows": [{"floor": "L0", "room": "Room North", "rename_to": "A<!-- auto:spaces -->"}]}, "rows",
     "rows[1]: rename_to holds '<!--'"),
    ({"rows": [{"floor": "L0", "room": "Room North", "space_type": " =1+1"}]}, "rows",
     "rows[1]: space_type starts with '='"),
    ({"rows": [{"floor": "L0", "room": "Room North", "rename_to": "@SUM(1)"}]}, "rows",
     "rows[1]: rename_to starts with '@'"),
    ({"rows": [{"floor": "L0", "room": "Room North", "reason": "+cmd"}]}, "rows", "reason starts with '+'"),
    ({"rows": [{"floor": "L0", "room": "Room North", "reason": "-cmd"}]}, "rows", "reason starts with '-'"),
])
async def test_a_refused_update_writes_nothing(tmp_path, write_example, write_wea, change, field, words):
    folder = build_merge_project(tmp_path, write_example, write_wea)
    before = (folder / "spaces.csv").read_bytes()
    result = await _update(tmp_path, folder, **change)
    assert codes(result) == [("INPUT_INVALID", field)] and words in result.structured_content["reason"]
    assert (folder / "spaces.csv").read_bytes() == before


async def test_a_stale_file_after_a_split_is_refused_and_create_is_named(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    (folder / "splits.csv").write_text(SPLITS_CSV, newline="\n")
    result = await _update(tmp_path, folder, rows=[{"floor": "L0", "room": "Room North", "space_type": "lab"}])
    assert codes(result) == [("INPUT_INVALID", None)]
    assert "inputs(action='create', file='spaces') rebuilds the rows" in result.structured_content["reason"]


async def test_update_spaces_needs_a_change(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    assert ("ARGUMENT_MISSING", "rows") in codes(await _update(tmp_path, folder))
