# tests/mcp/mcp_helpers.py
"""Shared helpers for the MCP tool tests. Not a conftest.py: one in
tests/mcp would shadow tests/conftest.py, which the copied tests import as
``conftest`` (plan 1, Task 10). A test module imports ``anyio_backend``
from here so pytest finds the fixture."""
from __future__ import annotations

import contextlib
import pathlib

import anyio
import pytest
from jsonschema import Draft202012Validator
from mcp.client import Client
from mcp.client._memory import InMemoryTransport
from mcp.server.runner import modern_on_request
from mcp.shared.direct_dispatcher import create_direct_dispatcher_pair
from mcp.shared.exceptions import MCPError
from mcp.shared.jsonrpc_dispatcher import JSONRPCDispatcher

from leed_mcp.server import build_server, register_tools
from rawwire import MODERN_META


@pytest.fixture
def anyio_backend():
    """asyncio only."""
    return "asyncio"


def tool_server(*roots: pathlib.Path, state: pathlib.Path | None = None, launcher=None, signaller=None,
                clock=None):
    """A server with the real tools, whose roots are ``roots`` (resolved).
    Its job index is ``state``, by default ``<first root>.state`` beside the
    first root, so no test touches ``~/.leed-mcp``. ``launcher``,
    ``signaller`` and ``clock`` replace the real ones (``tests/job_fakes.py``)."""
    resolved = tuple(pathlib.Path(r).resolve() for r in roots)
    if state is None:
        state = resolved[0].with_name(resolved[0].name + ".state") if resolved else None
    server = build_server(roots=resolved, state=state, launcher=launcher, signaller=signaller,
                          clock=clock)
    register_tools(server)
    return server


async def call(server, name: str, arguments: dict):
    """Call one tool over the SDK's in-process client and check the result,
    refusals included, against the tool's published output schema."""
    async with Client(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        result = await client.call_tool(name, arguments)
    Draft202012Validator(tools[name].output_schema).validate(result.structured_content)
    return result


def codes(result) -> list[tuple[str, str | None]]:
    """``(code, field)`` of every advisory, in order."""
    return [(a["code"], a.get("field")) for a in result.structured_content.get("advisories", [])]


TASKS = "io.modelcontextprotocol/tasks"
#: A modern request's ``_meta`` that declares the Tasks extension
#: (``MODERN_META`` declares nothing).
DECLARING = {**MODERN_META, "io.modelcontextprotocol/clientCapabilities": {"extensions": {TASKS: {}}}}


async def _no_messages(*_args) -> None:
    return None


@contextlib.asynccontextmanager
async def wire(server, *, legacy: bool = False):
    """Raw JSON-RPC requests to ``server`` in process, as the SDK's own
    ``Client`` connects (``modern_on_request``; the in-memory transport and
    ``initialize`` for ``legacy``). Yields ``send(method, params=None,
    meta=MODERN_META)``, which returns the result as a dict; a JSON-RPC error
    raises ``MCPError``, outside the task group so it is not wrapped in an
    exception group. On a legacy connection ``meta`` defaults to None and is
    sent as ``_meta`` only when given. The SDK
    serves only its low-level server in process, as ``Client`` reads it."""
    failure = None
    async with anyio.create_task_group() as group, contextlib.AsyncExitStack() as stack:
        if legacy:
            read, write = await stack.enter_async_context(InMemoryTransport(server))
            peer = JSONRPCDispatcher(read, write)
            await group.start(peer.run, _no_messages, _no_messages)
            await peer.send_raw_request("initialize", {
                "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "wire", "version": "0"}})
            await peer.notify("notifications/initialized", None)

            async def send(method, params=None, meta=None):
                params = dict(params or {})
                if meta is not None:
                    params["_meta"] = meta
                return await peer.send_raw_request(method, params)
        else:
            low = server._lowlevel_server
            state = await stack.enter_async_context(low.lifespan(low))
            client, served = create_direct_dispatcher_pair(raise_handler_exceptions=False)
            stack.callback(served.close)
            await group.start(served.run, modern_on_request(low, state), _no_messages)
            await group.start(client.run, _no_messages, _no_messages)

            async def send(method, params=None, meta=MODERN_META):
                return await client.send_raw_request(method, {**(params or {}), "_meta": meta})
        try:
            yield send
        except MCPError as error:
            failure = error
        group.cancel_scope.cancel()
    if failure is not None:
        raise failure
