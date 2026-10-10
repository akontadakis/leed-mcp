# tests/mcp/test_mcp_main_wire.py
"""The server ``main`` starts, over stdio: seven tools, three resources and
three prompts on both protocol eras (spec 5.1, 7)."""
import pytest

from rawwire import HERE, MODERN_META, RawServer

LEGACY_INIT = {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "rawwire", "version": "0"}}
TOOLS = ["project", "scene", "inputs", "run", "results", "deliver", "job"]
RESOURCES = ["leed://docs/running-a-project", "leed://templates/project", "leed://templates/credit-rules"]
PROMPTS = ["new_project", "check_before_run", "explain_results"]


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    return {"LEED_MCP_ROOTS": str(root), "LEED_MCP_STATE": str(tmp_path / "state")}


def _lists(server, meta):
    return ([t["name"] for t in server.request("tools/list", meta=meta)["result"]["tools"]],
            [r["uri"] for r in server.request("resources/list", meta=meta)["result"]["resources"]],
            [p["name"] for p in server.request("prompts/list", meta=meta)["result"]["prompts"]])


def test_a_modern_client_sees_everything_main_registers(env):
    server = RawServer(HERE / "main_server.py", env=env)
    try:
        capabilities = server.request("server/discover")["result"]["capabilities"]
        assert {"tools", "resources", "prompts"} <= set(capabilities)
        assert _lists(server, MODERN_META) == (TOOLS, RESOURCES, PROMPTS)
    finally:
        server.close()


def test_a_legacy_client_sees_everything_main_registers(env):
    server = RawServer(HERE / "main_server.py", env=env)
    try:
        init = server.request("initialize", LEGACY_INIT, meta=None)["result"]
        server.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert {"tools", "resources", "prompts"} <= set(init["capabilities"])
        assert _lists(server, None) == (TOOLS, RESOURCES, PROMPTS)
    finally:
        server.close()
