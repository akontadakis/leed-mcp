# tests/mcp/test_mcp_tasks_create.py
"""run and deliver answer a declaring request with a CreateTaskResult (spec 8.3)."""
import pytest

from job_fakes import FakeLauncher
from leed_daylight.deliver.data import result_path
from leed_mcp.jobindex import JobIndex
from mcp_helpers import DECLARING, anyio_backend, tool_server, wire  # noqa: F401 - anyio_backend is the fixture
from option2_folder import build_option2_project
from rawwire import MODERN_META

pytestmark = pytest.mark.anyio

TASK_KEYS = {"resultType", "taskId", "status", "statusMessage", "createdAt", "lastUpdatedAt", "ttlMs",
             "pollIntervalMs", "_meta"}


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    path = result_path(folder, "option1", "base_case")
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    return folder


def _server(tmp_path, launcher=None):
    return tool_server(tmp_path, launcher=launcher or FakeLauncher(("progress: rooms 1/1",)))


async def _call(server, name, arguments, meta=DECLARING, *, legacy=False):
    async with wire(server, legacy=legacy) as send:
        return await send("tools/call", {"name": name, "arguments": arguments}, meta=meta)


@pytest.mark.parametrize("name, action", [("run", "prepare"), ("deliver", "atlas")])
async def test_a_declaring_start_gets_a_flat_create_task_result(tmp_path, folder, name, action):
    server = _server(tmp_path)
    task = await _call(server, name, {"action": action, "folder": str(folder)})
    assert set(task) == TASK_KEYS
    assert (task["resultType"], task["status"]) == ("task", "working")
    assert task["statusMessage"] == f"the {action} job started on case base_case"
    assert (task["ttlMs"], task["pollIntervalMs"]) == (30 * 24 * 3600 * 1000, 5000)
    record = JobIndex(tmp_path.with_name(tmp_path.name + ".state")).read(task["taskId"])
    assert (record.action, record.created) == (action, task["createdAt"])
    assert task["lastUpdatedAt"] == task["createdAt"]
    async with wire(server) as send:
        got = await send("tasks/get", {"taskId": task["taskId"]}, meta=DECLARING)
    assert (got["taskId"], got["status"]) == (task["taskId"], "working")


async def test_a_request_that_does_not_declare_gets_the_started_branch(tmp_path, folder):
    result = await _call(_server(tmp_path), "run", {"action": "prepare", "folder": str(folder)}, meta=MODERN_META)
    assert result["resultType"] == "complete" and result["isError"] is False
    assert result["structuredContent"]["state"] == "started"


async def test_a_legacy_request_gets_the_started_branch(tmp_path, folder):
    # The declaration alone: a legacy connection refuses the 2026-07-28 envelope keys of DECLARING.
    key = "io.modelcontextprotocol/clientCapabilities"
    declaration = {key: DECLARING[key]}
    result = await _call(_server(tmp_path), "run", {"action": "prepare", "folder": str(folder)}, declaration,
                         legacy=True)
    assert "resultType" not in result and result["structuredContent"]["state"] == "started"


async def test_a_declaring_request_that_is_refused_gets_the_refusal(tmp_path, folder):
    launcher = FakeLauncher()
    result = await _call(_server(tmp_path, launcher), "run", {"action": "prepare", "folder": str(folder),
                                                              "processes": 2})
    assert result["isError"] is True and result["structuredContent"]["refused"] is True
    assert result["structuredContent"]["advisories"][0]["code"] == "UNKNOWN_ARGUMENT"
    assert launcher.calls == []


async def test_a_second_start_while_a_job_runs_is_refused_not_a_task(tmp_path, folder):
    server = _server(tmp_path)
    first = await _call(server, "run", {"action": "prepare", "folder": str(folder)})
    second = await _call(server, "run", {"action": "prepare", "folder": str(folder)})
    assert first["resultType"] == "task"
    assert second["isError"] is True and second["structuredContent"]["advisories"][0]["code"] == "JOB_RUNNING"


async def test_no_other_tool_returns_a_task(tmp_path, folder):
    server = _server(tmp_path)
    await _call(server, "run", {"action": "prepare", "folder": str(folder)})
    listed = await _call(server, "job", {"action": "list"})
    assert listed["resultType"] == "complete" and listed["structuredContent"]["total"] == 1
