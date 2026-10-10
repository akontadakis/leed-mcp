# tests/mcp/test_server_dispatch.py
"""LeedMCPServer.call_tool's order (spec 5.1)."""
import json

import pytest
from jsonschema import Draft202012Validator
from mcp.client import Client
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS

from leed_mcp.ratelimit import RateLimiter
from leed_mcp.server import SERVER_NAME, SERVER_TITLE, build_server
from probe import register_probe

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    """asyncio only. A tests/mcp/conftest.py would shadow tests/conftest.py,
    which the copied tests import as ``conftest``."""
    return "asyncio"


def _server(**kwargs):
    server = build_server(**kwargs)
    register_probe(server)
    return server


async def _call(server, arguments):
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        result = await client.call_tool("probe", arguments)
    return tools["probe"], result


def _codes(result):
    return [(a["code"], a.get("field")) for a in result.structured_content["advisories"]]


async def test_identity():
    async with Client(_server()) as client:
        assert client.server_info.name == SERVER_NAME
        assert client.server_info.title == SERVER_TITLE
        assert client.server_info.version


async def test_an_unknown_tool_is_a_json_rpc_invalid_params_error():
    async with Client(_server()) as client:
        with pytest.raises(MCPError) as caught:
            await client.call_tool("no_such_tool", {})
    assert caught.value.code == INVALID_PARAMS


async def test_an_echo_validates_against_the_output_schema():
    tool, result = await _call(_server(), {"action": "echo", "text": "ab", "count": 2})
    assert result.is_error is False
    assert result.structured_content == {"action": "echo", "text": "abab"}
    Draft202012Validator(tool.output_schema).validate(result.structured_content)


async def test_an_unknown_argument_is_refused_and_validates():
    tool, result = await _call(_server(), {"action": "echo", "text": "a", "txet": "b"})
    assert result.is_error is True
    assert _codes(result) == [("UNKNOWN_ARGUMENT", "txet")]
    Draft202012Validator(tool.output_schema).validate(result.structured_content)


async def test_wrong_type_is_refused_without_echoing_the_value():
    tool, result = await _call(_server(), {"action": "echo", "text": "a",
                                           "count": "/Users/someone/private"})
    assert result.is_error is True
    assert _codes(result) == [("INPUT_INVALID", "count")]
    assert "/Users/someone/private" not in json.dumps(result.model_dump(mode="json"))
    Draft202012Validator(tool.output_schema).validate(result.structured_content)


async def test_a_missing_argument_is_argument_missing():
    _, result = await _call(_server(), {"action": "echo"})
    assert _codes(result) == [("ARGUMENT_MISSING", "text")]


async def test_an_exception_is_internal_error_with_the_type_only():
    tool, result = await _call(_server(), {"action": "boom"})
    assert result.is_error is True
    assert _codes(result) == [("INTERNAL_ERROR", None)]
    text = json.dumps(result.model_dump(mode="json"))
    assert "RuntimeError" in text and "secret.rad" not in text and "/Users/someone" not in text
    Draft202012Validator(tool.output_schema).validate(result.structured_content)


async def test_the_rate_limit_refuses_with_the_seconds_to_wait():
    server = _server(rate_limiter=RateLimiter(default=2))
    async with Client(server) as client:
        for _ in range(2):
            assert (await client.call_tool("probe", {"action": "echo", "text": "a"})).is_error is False
        result = await client.call_tool("probe", {"action": "echo", "text": "a"})
    assert _codes(result) == [("RATE_LIMITED", None)]
    assert "seconds" in result.structured_content["reason"]


async def test_published_input_schemas_forbid_extra_properties():
    async with Client(_server()) as client:
        (tool,) = [t for t in (await client.list_tools()).tools if t.name == "probe"]
    assert tool.input_schema["additionalProperties"] is False
    assert tool.description.startswith("Test-only tool.")


@pytest.mark.parametrize("text", ["null", "true", '["a", "b"]', '{"a": 1}', "[a sentence in brackets]"])
async def test_a_text_argument_that_looks_like_json_stays_text(text):
    _, result = await _call(_server(), {"action": "echo", "text": text})
    assert result.is_error is False
    assert result.structured_content == {"action": "echo", "text": text}


async def test_a_list_sent_as_json_text_is_decoded_before_the_check(tmp_path):
    from leed_mcp.server import register_tools

    server = build_server(roots=[], state=tmp_path / "state")
    register_tools(server)
    async with Client(server) as client:
        result = await client.call_tool("inputs", {
            "action": "create", "folder": str(tmp_path / "p"), "file": "materials",
            "expected_sha256": None, "rows": '[{"modifier": "glass_a", "role": "glazing"}]'})
    codes = [a["code"] for a in result.structured_content["advisories"]]
    assert "INPUT_INVALID" not in codes
    assert codes == ["ROOTS_NOT_CONFIGURED"]
