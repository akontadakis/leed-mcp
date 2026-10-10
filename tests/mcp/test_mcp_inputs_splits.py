# tests/mcp/test_mcp_inputs_splits.py
"""inputs(action="update", file="splits") (spec 6.3)."""
import hashlib

import pytest

from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture
from merge_split_folder import SPLITS_CSV, build_split_project
from option2_folder import build_option2_project

pytestmark = pytest.mark.anyio

WEST = {"level_z": 0.0, "points": [[4.5, -1.0], [4.5, 5.0]], "note": "invented west edge"}


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _update(tmp_path, folder, lines, **extra):
    return await call(tool_server(tmp_path), "inputs", {"action": "update", "folder": str(folder),
                                                       "file": "splits", "lines": lines, **extra})


async def test_a_first_line_writes_splits_csv_and_names_the_stale_rows(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    result = await _update(tmp_path, folder, [WEST], expected_sha256=None)
    data = result.structured_content
    assert result.is_error is False and codes(result) == []
    assert (folder / "splits.csv").read_text() == "level_z,points,note\n0,4.5 -1; 4.5 5,invented west edge\n"
    assert data["content"] == {"lines": [WEST]} and data["backup"] is None
    assert data["unmatched_rows"] == ["Room North"]


async def test_a_line_that_cuts_no_room_is_refused_and_nothing_is_written(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    result = await _update(tmp_path, folder, [{**WEST, "points": [[20.0, -1.0], [20.0, 5.0]]}])
    assert codes(result) == [("INPUT_INVALID", None)] and "splits.csv" in result.structured_content["reason"]
    assert not (folder / "splits.csv").exists()


async def test_an_empty_list_leaves_the_header_and_backs_up_the_lines(tmp_path, write_example, write_wea):
    folder = build_split_project(tmp_path, write_example, write_wea)
    result = await _update(tmp_path, folder, [], expected_sha256=_sha(folder / "splits.csv"))
    assert (folder / "splits.csv").read_text() == "level_z,points,note\n"
    assert open(result.structured_content["backup"]).read() == SPLITS_CSV
    assert result.structured_content["unmatched_rows"] == ["Room West", "Room Strip"]


@pytest.mark.parametrize("line", [
    {"level_z": 0.0, "points": [[4.5, -1.0]]},
    {"level_z": 0.0, "points": [[4.5, -1.0, 0.0], [4.5, 5.0, 0.0]]},
    {"points": [[4.5, -1.0], [4.5, 5.0]]},
])
async def test_a_malformed_line_is_refused_by_the_input_schema(tmp_path, write_example, write_wea, line):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    result = await _update(tmp_path, folder, [line])
    assert codes(result)[0][0] == "INPUT_INVALID" and codes(result)[0][1].startswith("lines")


@pytest.mark.parametrize("note", ["=cmd|' /C calc'!A1", "+1", "-1", "@SUM(A1)", "two\nlines"])
async def test_a_note_a_spreadsheet_would_run_is_refused_and_nothing_is_written(
        tmp_path, write_example, write_wea, note):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    result = await _update(tmp_path, folder, [{**WEST, "note": note}])
    assert codes(result) == [("INPUT_INVALID", "rows")] and "note" in result.structured_content["reason"]
    assert not (folder / "splits.csv").exists()
