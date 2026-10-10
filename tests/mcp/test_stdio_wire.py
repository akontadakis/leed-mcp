# tests/mcp/test_stdio_wire.py
"""Over real stdio: discover, a bad version, and a stdout that carries only JSON-RPC."""
import json

import pytest

from rawwire import MODERN_META, RawServer


@pytest.fixture
def wire():
    server = RawServer()
    yield server
    server.close()


def _all_json_rpc(lines):
    for line in lines:
        message = json.loads(line)
        assert message.get("jsonrpc") == "2.0", line


def test_discover_names_2026_07_28_and_the_server(wire):
    answer = wire.request("server/discover")
    result = answer["result"]
    assert "2026-07-28" in result["supportedVersions"]
    assert result["_meta"]["io.modelcontextprotocol/serverInfo"]["name"] == "leed-mcp"


def test_an_unsupported_version_is_minus_32022(wire):
    answer = wire.request("server/discover",
                          meta={**MODERN_META, "io.modelcontextprotocol/protocolVersion": "1999-01-01"})
    assert answer["error"]["code"] == -32022


def test_stdout_carries_only_json_rpc(wire):
    wire.request("server/discover")
    answer = wire.request("tools/call", {"name": "probe", "arguments": {"action": "noisy", "text": "a"}})
    assert answer["result"]["structuredContent"] == {"action": "noisy", "text": "a"}
    _all_json_rpc(wire.stdout_lines())
    stderr = wire.close()
    _all_json_rpc(wire.stdout_lines())  # again after exit: a buffered print flushed then would land here
    assert "a stray print from a tool" in stderr
    assert "a child writing to fd 1" in stderr


def test_an_unknown_tool_over_the_wire_is_minus_32602(wire):
    wire.request("server/discover")
    answer = wire.request("tools/call", {"name": "no_such_tool", "arguments": {}})
    assert answer["error"]["code"] == -32602
