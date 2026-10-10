# tests/mcp/test_mcp_tasks.py
"""tasks/get, tasks/update and tasks/cancel (spec 8.2, 8.3) over raw JSON-RPC in process."""
import dataclasses
import datetime
import json

import pytest
from jsonschema import Draft202012Validator
from mcp.client import Client
from mcp.shared.exceptions import MCPError

from job_fakes import FakeClock, FakeLauncher, FakeSignaller
from leed_mcp.jobindex import JobIndex
from leed_mcp.tasks import POLL_INTERVAL_MS, TTL_MS
from mcp_helpers import DECLARING, TASKS, anyio_backend, call, tool_server, wire  # noqa: F401 - the fixture
from option2_folder import build_option2_project
from rawwire import MODERN_META

pytestmark = pytest.mark.anyio

METHODS = ("tasks/get", "tasks/update", "tasks/cancel")


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    return build_option2_project(tmp_path, write_example, write_wea)


class Rig:
    """One server with the pipeline's fakes; jobs start through ``run``."""

    def __init__(self, tmp_path, *, lines=("progress: rooms 1/1",)):
        self.launcher = FakeLauncher(lines)
        self.clock = FakeClock()
        self.signaller = FakeSignaller(self.clock, ends_on="interrupt")
        self.server = tool_server(tmp_path, launcher=self.launcher, signaller=self.signaller, clock=self.clock)
        self.index = JobIndex(tmp_path.with_name(tmp_path.name + ".state"))

    async def start(self, folder):
        result = await call(self.server, "run", {"action": "prepare", "folder": str(folder)})
        self.signaller.process = self.launcher.processes[-1]
        return result.structured_content["job_id"]

    async def task(self, method, task_id, meta=DECLARING, **params):
        async with wire(self.server) as send:
            return await send(method, {"taskId": task_id, **params}, meta=meta)


async def _error(rig, method, task_id, meta=DECLARING, **params):
    with pytest.raises(MCPError) as raised:
        await rig.task(method, task_id, meta, **params)
    return raised.value.error


async def test_a_running_job_is_a_working_task_with_its_progress(tmp_path, folder):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    task = await rig.task("tasks/get", job_id)
    assert (task["resultType"], task["taskId"], task["status"], task["statusMessage"]) == (
        "complete", job_id, "working", "rooms 1/1")
    created = rig.index.read(job_id).created
    assert (task["createdAt"], task["lastUpdatedAt"]) == (created, created)
    assert (task["ttlMs"], task["pollIntervalMs"]) == (TTL_MS, POLL_INTERVAL_MS) == (30 * 24 * 3600 * 1000, 5000)
    assert "result" not in task


async def test_a_succeeded_job_is_a_completed_task_whose_result_is_its_end(tmp_path, folder):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    rig.launcher.processes[0].finish(0)
    task = await rig.task("tasks/get", job_id)
    assert (task["status"], task["statusMessage"]) == ("completed", "the prepare job succeeded")
    result = task["result"]
    assert result["isError"] is False
    ended = result["structuredContent"]
    assert (ended["action"], ended["state"], ended["job_id"]) == ("prepare", "succeeded", job_id)
    assert (ended["finished"]["verb"], ended["finished"]["exit_code"]) == ("prepare", 0) and "reason" not in ended
    assert task["lastUpdatedAt"] == datetime.datetime.fromisoformat("2026-01-01T14:30:00").astimezone(
        datetime.timezone.utc).isoformat(timespec="seconds")
    async with Client(rig.server) as client:
        schema = {t.name: t for t in (await client.list_tools()).tools}["run"].output_schema
    Draft202012Validator(schema).validate(ended)


async def test_a_failed_job_is_a_completed_task_with_is_error(tmp_path, folder):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    rig.launcher.processes[0].finish(1)
    task = await rig.task("tasks/get", job_id)
    assert (task["status"], task["result"]["isError"]) == ("completed", True)
    ended = task["result"]["structuredContent"]
    assert (ended["state"], ended["reason"]) == ("failed", "the job exited with code 1")
    assert task["statusMessage"] == "the prepare job failed: the job exited with code 1"


async def test_tasks_cancel_stops_the_job_and_the_task_is_cancelled(tmp_path, folder):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    answer = await rig.task("tasks/cancel", job_id)
    assert answer["resultType"] == "complete" and set(answer) == {"resultType", "_meta"}
    assert [name for name, _, _ in rig.signaller.sent] == ["interrupt"]
    task = await rig.task("tasks/get", job_id)
    assert (task["status"], task["statusMessage"]) == ("cancelled", "the prepare job was cancelled")
    assert "result" not in task


@pytest.mark.parametrize("change, code", [
    ({"start_time": None}, "STOP_UNCONFIRMED"),
    ({"log": "_run/jobs/20260101-120000_prepare.log"}, "JOB_NOT_CURRENT"),
])
async def test_a_refused_stop_is_acknowledged_and_the_task_says_why(tmp_path, folder, change, code):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    current = folder / "_run" / "jobs" / "current.json"
    current.write_text(json.dumps({**json.loads(current.read_text()), **change}), newline="\n")
    assert (await rig.task("tasks/cancel", job_id))["resultType"] == "complete"
    assert rig.signaller.sent == []
    task = await rig.task("tasks/get", job_id)
    assert task["status"] == "working" and task["statusMessage"].startswith(f"not cancelled ({code}): ")


async def test_cancelling_an_ended_job_is_acknowledged_and_changes_nothing(tmp_path, folder):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    rig.launcher.processes[0].finish(0)
    assert (await rig.task("tasks/cancel", job_id))["resultType"] == "complete"
    assert rig.signaller.sent == [] and (await rig.task("tasks/get", job_id))["status"] == "completed"


async def test_tasks_update_is_acknowledged_and_its_input_is_ignored(tmp_path, folder):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    answer = await rig.task("tasks/update", job_id, inputResponses={"a": {"action": "accept"}})
    assert answer["resultType"] == "complete" and set(answer) == {"resultType", "_meta"}
    assert (await rig.task("tasks/get", job_id))["status"] == "working"


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("task_id", ["0" * 32, "../x", "A" * 32])
async def test_an_unknown_task_is_invalid_params(tmp_path, method, task_id):
    error = await _error(Rig(tmp_path), method, task_id)
    assert error.code == -32602 and error.message == "Failed to retrieve task: task not found"


@pytest.mark.parametrize("method", METHODS)
async def test_an_expired_task_is_invalid_params(tmp_path, folder, method):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    record = rig.index.read(job_id)
    old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=31)
    with rig.index.locked():
        rig.index.write(dataclasses.replace(record, created=old.isoformat(timespec="seconds")))
    error = await _error(rig, method, job_id)
    assert error.code == -32602 and error.message == "Failed to retrieve task: task has expired"


@pytest.mark.parametrize("method", METHODS)
async def test_a_task_outside_the_roots_is_invalid_params(tmp_path, folder, method):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    other = tmp_path.with_name(tmp_path.name + "-other")
    other.mkdir()
    elsewhere = Rig(other)
    clock = FakeClock()
    signaller = FakeSignaller(clock)
    elsewhere.server = tool_server(other, state=rig.index.state, signaller=signaller, clock=clock)
    error = await _error(elsewhere, method, job_id)
    assert error.code == -32602
    assert signaller.sent == []
    assert rig.launcher.processes[0].poll() is None


@pytest.mark.parametrize("method", METHODS)
async def test_a_request_that_does_not_declare_gets_32021(tmp_path, folder, method):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    error = await _error(rig, method, job_id, meta=MODERN_META)
    assert error.code == -32021
    assert error.data == {"requiredCapabilities": {"extensions": {TASKS: {}}}}


@pytest.mark.parametrize("method", METHODS)
async def test_a_legacy_connection_gets_method_not_found(tmp_path, folder, method):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    async with wire(rig.server, legacy=True) as send:
        with pytest.raises(MCPError) as raised:
            await send(method, {"taskId": job_id})
    assert raised.value.error.code == -32601


async def test_a_record_read_error_is_internal_and_leaves_the_task_unchanged(tmp_path, folder, monkeypatch):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)

    def broken(self, job_id):
        raise OSError("the disk is not there")

    with monkeypatch.context() as patch:
        patch.setattr(JobIndex, "read", broken)
        error = await _error(rig, "tasks/get", job_id)
    assert error.code == -32603 and "disk" not in error.message
    assert (await rig.task("tasks/get", job_id))["status"] == "working"


async def test_a_stored_finished_snapshot_that_does_not_fit_is_internal(tmp_path, folder):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    rig.launcher.processes[0].finish(0)
    assert (await rig.task("tasks/get", job_id))["status"] == "completed"
    record = rig.index.read(job_id)
    assert record.finished is not None
    with rig.index.locked():
        rig.index.write(dataclasses.replace(record, finished={**record.finished, "unexpected": 1}))
    error = await _error(rig, "tasks/get", job_id)
    assert error.code == -32603
    assert error.message == ("Failed to retrieve task: its job record could not be read; the task is unchanged, "
                             "try again")


async def test_a_refused_stop_note_is_dropped_when_the_task_ends(tmp_path, folder):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    current = folder / "_run" / "jobs" / "current.json"
    current.write_text(json.dumps({**json.loads(current.read_text()), "start_time": None}), newline="\n")
    assert (await rig.task("tasks/cancel", job_id))["resultType"] == "complete"
    tasks = rig.server._extensions[0]
    assert job_id in tasks._notes
    rig.launcher.processes[0].finish(0)
    assert (await rig.task("tasks/get", job_id))["status"] == "completed"
    assert job_id not in tasks._notes


async def test_a_status_message_has_no_control_characters(tmp_path, folder):
    rig = Rig(tmp_path, lines=("progress: \x1b[31mrooms 1/1\x07",))
    job_id = await rig.start(folder)
    assert (await rig.task("tasks/get", job_id))["statusMessage"] == "[31mrooms 1/1"


async def test_a_long_progress_line_is_cut_in_the_status_message(tmp_path, folder):
    rig = Rig(tmp_path, lines=("progress: " + "r" * 3000,))
    job_id = await rig.start(folder)
    message = (await rig.task("tasks/get", job_id))["statusMessage"]
    assert message == "r" * 2000 + " [cut: 1000 more characters]"


async def test_a_request_without_a_task_id_is_invalid_params(tmp_path):
    async with wire(Rig(tmp_path).server) as send:
        with pytest.raises(MCPError) as raised:
            await send("tasks/get", {}, meta=DECLARING)
    assert raised.value.error.code == -32602


async def test_a_new_server_instance_follows_a_task_another_started(tmp_path, folder):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    later = Rig(tmp_path)
    assert (await later.task("tasks/get", job_id))["status"] == "working"
    later.signaller.process = rig.launcher.processes[0]
    assert (await later.task("tasks/cancel", job_id))["resultType"] == "complete"
    assert (await later.task("tasks/get", job_id))["status"] == "cancelled"



async def test_the_output_schemas_carry_the_ended_branch(tmp_path):
    async with Client(Rig(tmp_path).server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    ended = {"action": "atlas", "state": "failed", "job_id": "0" * 32, "reason": "the job exited with code 1",
             "finished": {"verb": "atlas", "state": "failed", "exit_code": 1, "ended": None, "superseded": False}}
    for name in ("run", "deliver"):
        Draft202012Validator(tools[name].output_schema).validate(ended)
