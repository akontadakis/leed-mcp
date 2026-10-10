# tests/mcp/test_protocol_eras.py
"""Both protocol eras are served (spec 5.1)."""
import pytest
from jsonschema import Draft202012Validator
from mcp.client import Client

from leed_mcp.server import SERVER_NAME, build_server
from probe import register_probe

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    """asyncio only. A tests/mcp/conftest.py would shadow tests/conftest.py,
    which the copied tests import as ``conftest``."""
    return "asyncio"


def _server():
    server = build_server()
    register_probe(server)
    return server


async def test_a_modern_client_gets_2026_07_28():
    async with Client(_server()) as client:
        assert client.protocol_version == "2026-07-28"
        assert client.server_info.name == SERVER_NAME


async def test_a_legacy_client_lists_and_calls_tools():
    async with Client(_server(), mode="legacy") as client:
        assert client.protocol_version == "2025-11-25"
        assert client.server_info.name == SERVER_NAME
        names = [t.name for t in (await client.list_tools()).tools]
        result = await client.call_tool("probe", {"action": "echo", "text": "a"})
    assert names == ["probe"]
    assert result.structured_content == {"action": "echo", "text": "a"}


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_every_schema_is_valid_2020_12_with_an_object_root(mode):
    async with Client(_server(), mode=mode) as client:
        tools = (await client.list_tools()).tools
    for tool in tools:
        Draft202012Validator.check_schema(tool.input_schema)
        Draft202012Validator.check_schema(tool.output_schema)
        assert tool.input_schema["type"] == "object"
        assert tool.output_schema["type"] == "object", tool.name
