# tests/mcp/test_mcp_inputs_project.py
"""inputs(action="update", file="project") (spec 6.3, spec 10)."""
import hashlib
import json

import pytest
import yaml

from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture
from option2_folder import build_option2_project

pytestmark = pytest.mark.anyio

SOURCE = "typed by the user in chat"


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _update(tmp_path, folder, **change):
    arguments = {"action": "update", "folder": str(folder), "file": "project",
                 "expected_sha256": _sha(folder / "project.yaml"), "source": SOURCE, **change}
    return await call(tool_server(tmp_path), "inputs", arguments)


def _draft(d):
    d["site"]["latitude_N"] = "FILL_ME"
    d["analysis"]["hours"] = "FILL_ME"


async def test_update_sets_values_keeps_times_text_and_logs_the_source(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea, mutate=_draft)
    old = (folder / "project.yaml").read_bytes()
    result = await _update(tmp_path, folder, values={"analysis.hours": "08:00-17:00", "project.name": "Example B"})
    data = result.structured_content
    assert result.is_error is False and codes(result) == []
    text = (folder / "project.yaml").read_text()
    assert 'hours: "08:00-17:00"' in text and yaml.safe_load(text)["analysis"]["hours"] == "08:00-17:00"
    assert data["sha256"] == _sha(folder / "project.yaml") and data["content"]["project"]["name"] == "Example B"
    assert data["fill_me"] == ["site.latitude_N"]
    assert data["backup"].startswith(str(folder / "_run" / "backup" / "project.yaml."))
    assert open(data["backup"], "rb").read() == old
    (line,) = [json.loads(x) for x in (folder / "_run" / "input_log.jsonl").read_text().splitlines()]
    assert data["log"] == str(folder / "_run" / "input_log.jsonl")
    assert (line["file"], line["set"], line["unset"], line["source"]) == (
        "project.yaml", ["analysis.hours", "project.name"], [], SOURCE)


async def test_a_whole_section_can_be_set_and_an_optional_key_removed(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    views = {"eye_height_m": 1.2, "eye_height_source": "Example: seated eye height"}
    result = await _update(tmp_path, folder, values={"views": views}, unset=["reflectance.ground"])
    raw = yaml.safe_load((folder / "project.yaml").read_text())
    assert result.is_error is False
    assert raw["views"] == views and "ground" not in raw["reflectance"]


async def test_calculator_units_can_be_set_through_the_tool(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    result = await _update(tmp_path, folder, values={"calculator.units": "SI"})
    assert result.is_error is False
    text = (folder / "project.yaml").read_text()
    assert yaml.safe_load(text)["calculator"] == {"units": "SI"}
    assert 'calculator:\n  units: "SI"' in text  # the writer quotes strings


async def test_a_wrong_sha256_is_refused_and_nothing_is_written(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    before = (folder / "project.yaml").read_bytes()
    result = await _update(tmp_path, folder, values={"project.name": "B"}, expected_sha256="0" * 64)
    assert codes(result) == [("INPUT_CHANGED_ON_DISK", "expected_sha256")]
    assert "wizard" not in result.structured_content["reason"]
    assert (folder / "project.yaml").read_bytes() == before and not (folder / "_run" / "input_log.jsonl").exists()


@pytest.mark.parametrize("change, code, field", [
    ({"values": {"site.latitude": 1.0}}, "INPUT_INVALID", "values"),
    ({"values": {"glazing.frame_style": "bogus"}}, "INPUT_INVALID", "values"),
    ({"unset": ["site.latitude_N"]}, "INPUT_INVALID", "values"),
    ({"values": {"project.name": "B"}, "source": "   "}, "INPUT_INVALID", "source"),
    ({"values": {"weather.file": "../outside.wea"}}, "PATH_OUTSIDE_ROOTS", None),
    ({}, "ARGUMENT_MISSING", "unset"),
    ({"values": {}, "unset": []}, "ARGUMENT_MISSING", "unset"),
    ({"values": {}}, "ARGUMENT_MISSING", "values"),
    ({"unset": []}, "ARGUMENT_MISSING", "values"),
])
async def test_a_refused_change_writes_and_logs_nothing(tmp_path, write_example, write_wea, change, code, field):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    before = (folder / "project.yaml").read_bytes()
    result = await _update(tmp_path, folder, **change)
    assert (code, field) in codes(result)
    assert (folder / "project.yaml").read_bytes() == before and not (folder / "_run" / "input_log.jsonl").exists()


async def test_source_is_required(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    result = await call(tool_server(tmp_path), "inputs", {
        "action": "update", "folder": str(folder), "file": "project", "values": {"project.name": "B"},
        "expected_sha256": _sha(folder / "project.yaml")})
    assert codes(result) == [("ARGUMENT_MISSING", "source")]


async def test_hand_comments_refuse_the_write_until_the_user_agrees(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    with open(folder / "project.yaml", "a") as handle:
        handle.write("# checked with the architect\n")
    refused = await _update(tmp_path, folder, values={"project.name": "B"})
    assert codes(refused) == [("HAND_COMMENTS_WOULD_BE_LOST", "drop_hand_comments")]
    assert "# checked with the architect" in refused.structured_content["reason"]
    result = await _update(tmp_path, folder, values={"project.name": "B"}, drop_hand_comments=True)
    assert result.is_error is False and "checked with the architect" not in (folder / "project.yaml").read_text()


async def test_a_path_a_hand_edit_put_outside_can_be_moved_back_in(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea,
                                   mutate=lambda d: d["weather"].update(file="../elsewhere.wea"))
    still_outside = await _update(tmp_path, folder, values={"project.name": "B"})
    assert codes(still_outside) == [("PATH_OUTSIDE_ROOTS", None)]
    assert "weather.file" in still_outside.structured_content["reason"]
    result = await _update(tmp_path, folder, values={"weather.file": "weather/example.wea"})
    assert result.is_error is False


async def test_update_without_project_yaml_is_file_missing(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    result = await call(tool_server(tmp_path), "inputs", {
        "action": "update", "folder": str(folder), "file": "project", "values": {"project.name": "B"},
        "source": SOURCE})
    assert codes(result) == [("FILE_MISSING", "file")]
