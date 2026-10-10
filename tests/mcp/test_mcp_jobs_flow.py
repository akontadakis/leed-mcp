# tests/mcp/test_mcp_jobs_flow.py
"""The job tools together: both protocol eras and output safety (spec 5.1, 9, 12 steps 3 and 5)."""
import pytest
from mcp.client import Client

from job_fakes import FakeLauncher
from mcp_helpers import anyio_backend, call, tool_server  # noqa: F401 - anyio_backend is the fixture
from option2_folder import build_option2_project

pytestmark = pytest.mark.anyio


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    return build_option2_project(tmp_path, write_example, write_wea)


async def test_a_legacy_client_starts_a_job_and_follows_it(tmp_path, folder):
    launcher = FakeLauncher(["progress: rooms 1/1"])
    async with Client(tool_server(tmp_path, launcher=launcher), mode="legacy") as client:
        started = await client.call_tool("run", {"action": "prepare", "folder": str(folder)})
        launcher.processes[0].finish(0)
        done = await client.call_tool("job", {"action": "wait", "job_id": started.structured_content["job_id"],
                                              "timeout_s": 5})
    assert started.is_error is False and started.structured_content["state"] == "started"
    assert done.structured_content["state"] == "succeeded"


async def test_log_text_is_cleaned_of_control_characters(tmp_path, folder):
    launcher = FakeLauncher(["\x1b[31mred\x07 line", "progress: \x1b[1mbold\x1b[0m"])
    server = tool_server(tmp_path, launcher=launcher)
    started = await call(server, "run", {"action": "prepare", "folder": str(folder)})
    job_id = started.structured_content["job_id"]
    logs = (await call(server, "job", {"action": "logs", "job_id": job_id})).structured_content
    status = (await call(server, "job", {"action": "status", "job_id": job_id})).structured_content
    assert logs["lines"] == ["[31mred line", "progress: [1mbold[0m"]
    assert status["progress"] == "[1mbold[0m"


async def test_a_long_log_line_is_cut_and_marked(tmp_path, folder):
    launcher = FakeLauncher(["x" * 2500])
    server = tool_server(tmp_path, launcher=launcher)
    started = await call(server, "run", {"action": "prepare", "folder": str(folder)})
    logs = (await call(server, "job", {"action": "logs", "job_id": started.structured_content["job_id"]}))
    assert logs.structured_content["lines"][0].endswith("[cut: 500 more characters]")


async def test_two_waits_on_one_server_do_not_block_each_other(tmp_path, folder):
    import anyio

    launcher = FakeLauncher()
    server = tool_server(tmp_path, launcher=launcher)
    started = await call(server, "run", {"action": "prepare", "folder": str(folder)})
    job_id = started.structured_content["job_id"]
    results = []

    async def wait():
        results.append(await call(server, "job", {"action": "wait", "job_id": job_id, "timeout_s": 10}))

    async with anyio.create_task_group() as group:
        group.start_soon(wait)
        group.start_soon(wait)
        await anyio.sleep(0.3)
        launcher.processes[0].finish(0)
    assert [r.structured_content["state"] for r in results] == ["succeeded", "succeeded"]
