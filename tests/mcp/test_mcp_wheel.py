# tests/mcp/test_mcp_wheel.py
"""Spec 3: the wheel installs into a fresh environment and
``project(action="create")`` runs from it, so the package data ships and
nothing depends on this checkout or on an editable install's ``.pth``; the
guide resource (spec 7) reads from the installed package.
From the same install, a real ``prepare`` job runs to its end (spec 8.1,
10): a child process started with no shell, ``-P`` and a cleaned
environment imports the installed pipeline and writes only to its log."""
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from conftest import write_example_project, write_wea_file
from option2_folder import build_option2_project

ROOT = pathlib.Path(__file__).resolve().parents[2]
UV = shutil.which("uv")

pytestmark = pytest.mark.skipif(UV is None, reason="uv is not installed")

#: Run with the fresh environment's Python in isolated mode (-I): no
#: PYTHONPATH, no user site, and the current folder is not on the path.
SCRIPT = """
import asyncio, json, pathlib, sys
import leed_daylight, leed_mcp
from mcp.client import Client
from leed_mcp.server import build_server, register_tools
from leed_mcp.static_resources import register_resources

root = pathlib.Path(sys.argv[1]).resolve()

async def main():
    server = build_server(roots=(root,))
    register_tools(server)
    register_resources(server)
    async with Client(server) as client:
        result = await client.call_tool("project", {"action": "create", "folder": str(root / "p")})
        guide = (await client.read_resource("leed://docs/running-a-project")).contents[0].text
    print(json.dumps({"leed_mcp": leed_mcp.__file__, "leed_daylight": leed_daylight.__file__,
                      "result": result.structured_content, "guide": guide.splitlines()[0]}))

asyncio.run(main())
"""

#: Start a real prepare job and wait for its end, as a client would.
JOB_SCRIPT = """
import asyncio, json, pathlib, sys
from mcp.client import Client
from leed_mcp.server import build_server, register_tools

folder = pathlib.Path(sys.argv[1]).resolve()

async def main():
    server = build_server(roots=(folder.parent,))
    register_tools(server)
    async with Client(server) as client:
        started = await client.call_tool("run", {"action": "prepare", "folder": str(folder)})
        job_id = started.structured_content["job_id"]
        for _ in range(12):
            done = await client.call_tool("job", {"action": "wait", "job_id": job_id, "timeout_s": 10})
            if done.structured_content["state"] != "running":
                break
        logs = await client.call_tool("job", {"action": "logs", "job_id": job_id})
    print(json.dumps({"done": done.structured_content, "lines": logs.structured_content["lines"]}))

asyncio.run(main())
"""


def _run(*argv, **kwargs):
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV")}
    return subprocess.run(argv, check=True, capture_output=True, text=True, env=env, **kwargs)


@pytest.fixture(scope="module")
def python(tmp_path_factory):
    """The Python of a fresh environment with only the wheel installed."""
    tmp = tmp_path_factory.mktemp("wheel")
    dist = tmp / "dist"
    _run(UV, "build", "--offline", "--out-dir", str(dist), str(ROOT))
    (wheel,) = dist.glob("leed_mcp-*.whl")
    venv = tmp / "venv"
    _run(UV, "venv", str(venv), "--python", sys.executable)
    windows = sys.platform == "win32"
    python = venv / ("Scripts" if windows else "bin") / ("python.exe" if windows else "python")
    _run(UV, "pip", "install", "--offline", "--python", str(python), str(wheel))
    return python


def test_the_wheel_installs_and_creates_a_project(tmp_path, python):
    root = tmp_path / "root"
    root.mkdir()
    out = _run(str(python), "-I", "-c", SCRIPT, str(root), cwd=tmp_path)
    data = json.loads(out.stdout.splitlines()[-1])
    for module in ("leed_mcp", "leed_daylight"):
        assert pathlib.Path(data[module]).is_relative_to(python.parent.parent), data[module]
    assert data["result"]["written"] == ["project.yaml", "credit_rules.yaml", "model/", "weather/"]
    template = ROOT / "leed_daylight" / "templates" / "project_template.yaml"
    assert (root / "p" / "project.yaml").read_bytes() == template.read_bytes()
    assert data["guide"] == "# Running a project with leed-mcp"


def test_the_installed_server_runs_a_real_prepare_job(tmp_path, python):
    folder = build_option2_project(tmp_path, write_example_project, write_wea_file)
    out = _run(str(python), "-I", "-c", JOB_SCRIPT, str(folder), cwd=tmp_path)
    data = json.loads(out.stdout.splitlines()[-1])
    done = data["done"]
    assert (done["state"], done["exit_code"]) == ("succeeded", 0), data["lines"]
    finished = done["finished"]
    assert (finished["rooms"], finished["window_groups"], finished["superseded"]) == (1, 1, False)
    assert finished["grid_points"] > 0
    assert any(line.startswith("prepared model written to") for line in data["lines"])
    assert out.stdout.splitlines()[-1].startswith("{")
