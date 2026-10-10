# tests/mcp/probe.py
"""A test-only tool for the server core: no real tool exists in plan 1."""
from __future__ import annotations

import subprocess
import sys
from typing import Any, Literal

from mcp_types import ToolAnnotations
from pydantic import RootModel

from leed_mcp.envelope import LeedResult, Refusal, emit, one_of_schema, refuse_missing


class ProbeBase(LeedResult):
    action: str


class ProbeEcho(ProbeBase):
    action: Literal["echo", "noisy"]
    text: str


class ProbeResponse(RootModel[ProbeEcho | Refusal]):
    """What `probe` returns: the echo, or a refusal."""

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return one_of_schema(super().model_json_schema(*args, **kwargs), ProbeBase)


def probe(action: Literal["echo", "boom", "noisy"], text: str | None = None,
          count: int = 1) -> ProbeResponse:
    """Test-only tool.

    action: echo returns text repeated count times; boom raises; noisy
    prints and runs a child that writes to fd 1, then echoes.
    """
    if action == "boom":
        raise RuntimeError("a private path /Users/someone/client_project/secret.rad")
    if (missing := refuse_missing(f"probe(action={action!r})", text=text)) is not None:
        return missing
    if action == "noisy":
        print("a stray print from a tool")
        subprocess.run([sys.executable, "-c", "print('a child writing to fd 1')"], check=True)
    return emit(ProbeEcho(action=action, text=text * count))


def register_probe(server) -> None:
    server.add_tool(probe, annotations=ToolAnnotations(
        read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False))
