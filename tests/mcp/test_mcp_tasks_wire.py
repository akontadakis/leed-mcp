# tests/mcp/test_mcp_tasks_wire.py
"""The Tasks extension on a real stdio connection (spec 8.3, 12 steps 2 and 5).

With tests/mcp/test_mcp_tasks_create.py they replace the plan 1 spike
(tests/mcp/test_tasks_spike.py): its checks, on the real extension.
"""
import pytest

from mcp_helpers import DECLARING, TASKS
from rawwire import MODERN_META, RawServer

LEGACY_INIT = {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "rawwire", "version": "0"}}
UNKNOWN = {"taskId": "0" * 32}
ACKNOWLEDGED = "notifications/subscriptions/acknowledged"


@pytest.fixture
def wire():
    server = RawServer()
    yield server
    server.close()


def _legacy(server):
    init = server.request("initialize", LEGACY_INIT, meta=None)
    server.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    return init["result"]


def test_discover_advertises_the_extension_and_initialize_does_not(wire):
    assert wire.request("server/discover")["result"]["capabilities"]["extensions"] == {TASKS: {}}
    legacy = RawServer()
    try:
        assert "extensions" not in _legacy(legacy)["capabilities"]
    finally:
        legacy.close()


@pytest.mark.parametrize("method", ["tasks/get", "tasks/update", "tasks/cancel"])
def test_each_task_method_by_era_and_declaration(wire, method):
    assert wire.request(method, UNKNOWN, meta=MODERN_META)["error"]["code"] == -32021
    assert wire.request(method, UNKNOWN, meta=DECLARING)["error"]["code"] == -32602
    legacy = RawServer()
    try:
        _legacy(legacy)
        assert legacy.request(method, UNKNOWN, meta=None)["error"]["code"] == -32601
    finally:
        legacy.close()


def test_an_earlier_declaration_does_not_count(wire):
    assert wire.request("tasks/get", UNKNOWN, meta=DECLARING)["error"]["code"] == -32602
    error = wire.request("tasks/get", UNKNOWN, meta=MODERN_META)["error"]
    assert error["code"] == -32021 and error["data"] == {"requiredCapabilities": {"extensions": {TASKS: {}}}}


@pytest.mark.parametrize("method", ["tasks/result", "tasks/list"])
def test_the_methods_2026_07_28_removed_are_not_served(wire, method):
    assert wire.request(method, UNKNOWN, meta=DECLARING)["error"]["code"] == -32601


def test_listen_for_task_ids_without_declaring_is_32021(wire):
    answer = wire.request("subscriptions/listen", {"notifications": {"taskIds": ["0" * 32]}}, meta=MODERN_META)
    assert answer["error"]["code"] == -32021
    assert answer["error"]["data"] == {"requiredCapabilities": {"extensions": {TASKS: {}}}}


def test_listen_for_an_empty_task_ids_list_without_declaring_is_32021(wire):
    answer = wire.request("subscriptions/listen", {"notifications": {"taskIds": []}}, meta=MODERN_META)
    assert answer["error"]["code"] == -32021


def test_a_declaring_listen_is_acknowledged_without_task_ids(wire):
    wire.send({"jsonrpc": "2.0", "id": 90, "method": "subscriptions/listen", "params": {
        "notifications": {"taskIds": ["0" * 32], "toolsListChanged": True}, "_meta": DECLARING}})
    ack = wire.notification(ACKNOWLEDGED)
    assert ack["params"]["notifications"] == {"toolsListChanged": True}


def test_a_listen_without_task_ids_needs_no_declaration(wire):
    wire.send({"jsonrpc": "2.0", "id": 91, "method": "subscriptions/listen", "params": {
        "notifications": {"toolsListChanged": True}, "_meta": MODERN_META}})
    assert wire.notification(ACKNOWLEDGED)["params"]["notifications"] == {"toolsListChanged": True}
