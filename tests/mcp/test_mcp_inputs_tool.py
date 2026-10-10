# tests/mcp/test_mcp_inputs_tool.py
"""The inputs tool as a whole: every write under the folder lock and the
spec 10 path rule, both protocol eras, output safety (spec 6.3, 9, 10)."""
import hashlib

import pytest
from mcp.client import Client

from leed_daylight.deliver.narrative import TEMPLATE
from leed_daylight.engine.jobs import acquire_lock, release_lock
from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture
from option2_folder import MATERIALS_CSV, build_option2_project

pytestmark = pytest.mark.anyio


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _writes(folder):
    """One call of every write action, each valid on the option2 project."""
    return {
        "update project": {"file": "project", "values": {"project.name": "B"}, "source": "typed by the user",
                           "expected_sha256": _sha(folder / "project.yaml")},
        "create materials": {"file": "materials", "expected_sha256": _sha(folder / "materials.csv"), "rows": [
            {"modifier": m, "role": r} for m, r in (("wall_paint", "wall"), ("ceiling_tile", "ceiling"),
                                                    ("floor_carpet", "floor"), ("glass_clear", "glazing"))]},
        "update materials": {"file": "materials", "expected_sha256": _sha(folder / "materials.csv"),
                             "rows": [{"modifier": "wall_paint", "note": "x"}]},
        "create spaces": {"file": "spaces", "expected_sha256": _sha(folder / "spaces.csv"), "rows": []},
        "update spaces": {"file": "spaces", "expected_sha256": _sha(folder / "spaces.csv"),
                          "rows": [{"floor": "L0", "room": "Room North", "space_type": "lab"}]},
        "update splits": {"file": "splits", "lines": []},
        "update glare_narrative": {"file": "glare_narrative", "expected_sha256": _sha(folder / "glare_narrative.md"),
                                   "prose": {"1": "Confirmed."}},
    }


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    (folder / "glare_narrative.md").write_text(TEMPLATE, newline="\n")
    return folder


@pytest.mark.parametrize("which", ["update project", "create materials", "update materials", "create spaces",
                                   "update spaces", "update splits", "update glare_narrative"])
async def test_every_write_works_on_the_project(tmp_path, folder, which):
    action, _ = which.split()
    result = await call(tool_server(tmp_path), "inputs", {"action": action, "folder": str(folder),
                                                         **_writes(folder)[which]})
    assert result.is_error is False, result.structured_content


@pytest.mark.parametrize("which", ["update project", "create materials", "update spaces", "update splits",
                                   "update glare_narrative"])
async def test_every_write_is_refused_while_a_job_holds_the_folder(tmp_path, folder, which):
    action, _ = which.split()
    arguments = {"action": action, "folder": str(folder), **_writes(folder)[which]}
    before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    info = acquire_lock(folder, "run")
    try:
        result = await call(tool_server(tmp_path), "inputs", arguments)
    finally:
        release_lock(folder, info)
    assert codes(result) == [("LOCK_HELD", None)]
    assert {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()} == before


@pytest.mark.parametrize("which", ["create materials", "update materials", "create spaces", "update spaces",
                                   "update splits", "update glare_narrative"])
async def test_every_write_refuses_a_project_yaml_path_outside_the_folder(tmp_path, folder, which):
    action, _ = which.split()
    arguments = {"action": action, "folder": str(folder), **_writes(folder)[which]}
    text = (folder / "project.yaml").read_text().replace("weather/example.wea", "../elsewhere.wea")
    (folder / "project.yaml").write_text(text, newline="\n")
    result = await call(tool_server(tmp_path), "inputs", arguments)
    assert codes(result) == [("PATH_OUTSIDE_ROOTS", None)] and "weather.file" in result.structured_content["reason"]


async def test_an_argument_the_file_does_not_take_is_unknown(tmp_path, folder):
    result = await call(tool_server(tmp_path), "inputs", {
        "action": "update", "folder": str(folder), "file": "materials", "rows": [], "source": "x"})
    assert codes(result) == [("UNKNOWN_ARGUMENT", "source")]


async def test_text_from_a_file_is_cleaned_of_control_characters(tmp_path, folder):
    (folder / "materials.csv").write_text(MATERIALS_CSV.replace("wall_paint,plastic,wall,",
                                                                "wall_paint,plastic,wall,\x1b[31mred\x07"), newline="\n")
    result = await call(tool_server(tmp_path), "inputs", {"action": "read", "folder": str(folder),
                                                         "file": "materials"})
    assert result.structured_content["content"]["rows"][0]["note"] == "[31mred"


async def test_a_legacy_client_can_read_and_write(tmp_path, folder):
    async with Client(tool_server(tmp_path), mode="legacy") as client:
        read = await client.call_tool("inputs", {"action": "read", "folder": str(folder), "file": "materials"})
        written = await client.call_tool("inputs", {
            "action": "update", "folder": str(folder), "file": "materials",
            "expected_sha256": read.structured_content["sha256"], "rows": [{"modifier": "wall_paint", "note": "x"}]})
    assert read.is_error is False and written.is_error is False


@pytest.mark.parametrize("arguments, field", [
    ({"action": "update", "file": "materials", "rows": [{"modifier": "wall_paint"}] * 5001}, "rows"),
    ({"action": "update", "file": "splits", "lines": [{"level_z": 0.0, "points": [[0, 0], [1, 0]]}] * 501}, "lines"),
    ({"action": "update", "file": "splits", "lines": [{"level_z": 0.0, "points": [[0, 0]] * 1001}]},
     "lines.0.points"),
])
async def test_an_oversized_argument_is_refused_by_the_input_schema(tmp_path, folder, arguments, field):
    before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    result = await call(tool_server(tmp_path), "inputs", {"folder": str(folder), **arguments})
    assert codes(result) == [("INPUT_INVALID", field)] and "input schema" in result.structured_content["reason"]
    assert {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()} == before


async def test_rows_at_the_cap_pass_the_input_schema(tmp_path, folder):
    result = await call(tool_server(tmp_path), "inputs", {
        "action": "update", "folder": str(folder), "file": "materials",
        "expected_sha256": _sha(folder / "materials.csv"), "rows": [{"modifier": "wall_paint", "note": "x"}] * 5000})
    assert codes(result) == [("INPUT_INVALID", "rows")] and "given twice" in result.structured_content["reason"]


async def test_a_csv_read_shows_at_most_5000_rows_and_says_so(tmp_path, folder):
    rows = "".join(f"m{n},plastic,wall,\n" for n in range(5001))
    (folder / "materials.csv").write_text("modifier,primitive_type,role,note\n" + rows, newline="\n")
    result = await call(tool_server(tmp_path), "inputs", {"action": "read", "folder": str(folder),
                                                         "file": "materials"})
    data = result.structured_content
    assert result.is_error is False and len(data["content"]["rows"]) == 5000
    assert data["content"]["rows"][-1]["modifier"] == "m4999"
    assert ("RESULT_TRUNCATED", "content") in codes(result)
    cut = [a for a in data["advisories"] if a["code"] == "RESULT_TRUNCATED"][0]
    assert cut["severity"] == "warn" and cut["message"] == (
        "materials.csv has 5001 rows; only the first 5000 are shown.")
