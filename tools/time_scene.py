#!/usr/bin/env python3
# tools/time_scene.py
"""Time ``scene(action="scan")`` and ``scene(action="rooms")`` on one project folder (spec 13, Q1).

    .venv/bin/python tools/time_scene.py "<project folder>"

Q1 asks whether the two scene actions stay under 30 s on a large scene; one
that does not moves to a job. This runs the server in process, with the
folder itself as its only root and a temporary job index, calls each action
once, and prints one line per action, then the verdict. It starts no Radiance
program. ``rooms`` writes ``_run/spaces_template.csv`` in the folder, as it
always does; nothing else is written. It returns 0 when both actions
answered, 1 when one was refused, 2 for a bad argument. A developer script:
not part of the server and not shipped.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import tempfile
import time
from collections.abc import Callable, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The repository first: on macOS the editable install's .pth file can carry the
# `hidden` flag, which Python skips (spec 13, N3).
sys.path.insert(0, str(ROOT))

from mcp.client import Client  # noqa: E402

from leed_mcp.paths import ROOTS_ENV_VAR, roots_from_env  # noqa: E402
from leed_mcp.server import build_server, register_tools  # noqa: E402

#: Q1's bound, in seconds.
LIMIT_S = 30.0
ACTIONS = ("scan", "rooms")


async def _time(server, folder: pathlib.Path, clock: Callable[[], float]) -> list[tuple[str, float, dict]]:
    timed = []
    async with Client(server) as client:
        for action in ACTIONS:
            start = clock()
            result = await client.call_tool("scene", {"action": action, "folder": str(folder)})
            timed.append((action, clock() - start, result.structured_content))
    return timed


def _line(action: str, seconds: float, data: dict) -> str:
    if action == "scan":
        problems = sum(a["code"] == "SCENE_PROBLEM" for a in data.get("advisories", []))
        storeys = "no storeys (no materials.csv)" if data["storeys"] is None else f"{len(data['storeys'])} storeys"
        return f"scene scan: {seconds:.1f} s, {len(data['materials'])} materials, {storeys}, {problems} problems"
    return f"scene rooms: {seconds:.1f} s, {len(data['rooms'])} rooms"


def main(argv: Sequence[str] | None = None, clock: Callable[[], float] = time.perf_counter) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print('usage: tools/time_scene.py "<project folder>"', file=sys.stderr)
        return 2
    folder = pathlib.Path(argv[0]).resolve()
    if not folder.is_dir():
        print(f"error: {folder}: no such folder", file=sys.stderr)
        return 2
    roots = roots_from_env({ROOTS_ENV_VAR: str(folder)})
    if not roots:
        print(f"error: {folder} cannot be a root: it is this repository, holds it, or lies inside it",
              file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as state:
        server = build_server(roots=roots, state=pathlib.Path(state) / "state")
        register_tools(server)
        timed = asyncio.run(_time(server, folder, clock))
    for action, seconds, data in timed:
        if data.get("refused"):
            print(f"scene {action}: refused ({data['advisories'][0]['code']}): {data['reason']}")
            return 1
        print(_line(action, seconds, data))
    over = [f"scene {action}" for action, seconds, _ in timed if seconds > LIMIT_S]
    if over:
        print(f"Q1: {' and '.join(over)} took over {LIMIT_S:.0f} s: spec Q1 moves it to a job")
    else:
        print(f"Q1: scene scan and scene rooms each took under {LIMIT_S:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
