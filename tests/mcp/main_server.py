# tests/mcp/main_server.py
"""The real leed-mcp server, as the ``leed-mcp`` script starts it (``main``), for wire tests."""
import pathlib
import sys

# The repository root: on macOS the editable install's .pth file can carry the
# `hidden` flag, which Python skips, so the installed path cannot be relied on here.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from leed_mcp.server import main  # noqa: E402

if __name__ == "__main__":
    main()
