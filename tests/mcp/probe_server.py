# tests/mcp/probe_server.py
"""The leed-mcp server with only the test probe tool, over stdio, for wire tests."""
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
# The repository root too: on macOS the editable install's .pth file can carry the
# `hidden` flag, which Python skips, so the installed path cannot be relied on here.
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parents[2]))

from leed_mcp.server import build_server  # noqa: E402
from probe import register_probe  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    server = build_server()
    register_probe(server)
    server.run(transport="stdio")
