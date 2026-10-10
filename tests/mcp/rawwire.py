# tests/mcp/rawwire.py
"""A raw newline-delimited JSON-RPC client over a server's stdio.

It sees every byte the server writes to stdout, which the SDK client hides,
so a test can assert that stdout carries only JSON-RPC.
"""
from __future__ import annotations

import json
import os
import pathlib
import queue
import subprocess
import sys
import threading
from typing import Any

HERE = pathlib.Path(__file__).parent
MODERN_VERSION = "2026-07-28"
MODERN_META = {
    "io.modelcontextprotocol/protocolVersion": MODERN_VERSION,
    "io.modelcontextprotocol/clientInfo": {"name": "rawwire", "version": "0"},
    "io.modelcontextprotocol/clientCapabilities": {},
}


class RawServer:
    def __init__(self, script: pathlib.Path = HERE / "probe_server.py", env: dict | None = None):
        self._proc = subprocess.Popen(
            [sys.executable, str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env={**os.environ, **(env or {})},
        )
        self._lines: list[str] = []
        self._queue: queue.Queue[str] = queue.Queue()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self._next_id = 0

    def _read(self) -> None:
        for raw in self._proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            self._lines.append(line)
            self._queue.put(line)

    def send(self, message: dict[str, Any]) -> None:
        self._proc.stdin.write((json.dumps(message) + "\n").encode())
        self._proc.stdin.flush()

    def request(self, method: str, params: dict[str, Any] | None = None, *,
                meta: dict[str, Any] | None = MODERN_META, timeout: float = 20.0) -> dict[str, Any]:
        """Send one request and return its response (a dict with `result` or `error`)."""
        self._next_id += 1
        params = dict(params or {})
        if meta is not None:
            params["_meta"] = {**meta, **params.get("_meta", {})}
        self.send({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params})
        while True:
            message = json.loads(self._queue.get(timeout=timeout))
            if message.get("id") == self._next_id:
                return message

    def notification(self, method: str, *, timeout: float = 20.0) -> dict[str, Any]:
        """The next notification named ``method``; other messages before it are skipped."""
        while True:
            message = json.loads(self._queue.get(timeout=timeout))
            if message.get("method") == method:
                return message

    def stdout_lines(self) -> list[str]:
        return list(self._lines)

    def close(self) -> str:
        """Stop the server; return its stderr."""
        self._proc.stdin.close()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._reader.join(timeout=10)  # every stdout line, including any written at exit, is read
        return self._proc.stderr.read().decode("utf-8", "replace")
