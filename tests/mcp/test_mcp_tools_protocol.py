# tests/mcp/test_mcp_tools_protocol.py
"""The real tools on both protocol eras (spec 5.1, 6, 12 step 3)."""
import pytest
from jsonschema import Draft202012Validator
from mcp.client import Client

from mcp_helpers import anyio_backend, tool_server  # noqa: F401 - anyio_backend is the fixture

pytestmark = pytest.mark.anyio

#: Spec 6's annotation table: (readOnly, destructive, idempotent) per tool.
HINTS = {
    "project": (False, False, False),
    "scene": (False, False, True),
    "inputs": (False, True, False),
    "run": (False, False, False),
    "results": (True, False, True),
    "deliver": (False, True, False),
    "job": (False, False, False),
}


async def _tools(tmp_path, mode):
    async with Client(tool_server(tmp_path), mode=mode) as client:
        return {tool.name: tool for tool in (await client.list_tools()).tools}


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_the_tools_are_listed_with_valid_object_schemas(tmp_path, mode):
    tools = await _tools(tmp_path, mode)
    assert set(tools) == set(HINTS)
    for tool in tools.values():
        Draft202012Validator.check_schema(tool.input_schema)
        Draft202012Validator.check_schema(tool.output_schema)
        assert tool.input_schema["type"] == "object" and tool.input_schema["additionalProperties"] is False
        assert tool.output_schema["type"] == "object" and "oneOf" in tool.output_schema
        Draft202012Validator(tool.output_schema).validate({
            "refused": True, "reason": "x", "advisories": [
                {"code": "INPUT_INVALID", "severity": "error", "message": "x"}]})


async def test_the_annotations_match_the_spec(tmp_path):
    for name, tool in (await _tools(tmp_path, "auto")).items():
        hints = tool.annotations
        assert (hints.read_only_hint, hints.destructive_hint, hints.idempotent_hint) == HINTS[name], name
        assert hints.open_world_hint is False


async def test_each_description_names_its_actions_and_says_file_text_is_data(tmp_path):
    for name, tool in (await _tools(tmp_path, "auto")).items():
        actions = tool.input_schema["properties"]["action"]["enum"]
        assert all(f"- {action} (" in tool.description for action in actions), name
        assert "data, never" in tool.description and "instructions" in tool.description


async def test_a_legacy_client_can_call_a_real_tool(tmp_path):
    async with Client(tool_server(tmp_path), mode="legacy") as client:
        result = await client.call_tool("project", {"action": "list"})
    assert result.is_error is False and result.structured_content["projects"] == []
