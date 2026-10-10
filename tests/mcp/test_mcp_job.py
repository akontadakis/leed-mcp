# tests/mcp/test_mcp_job.py
"""job (spec 6.7) through the tool, and project status naming the job (spec 8.1)."""
import dataclasses
import datetime
import json
import threading

import pytest
from mcp.client import Client

from job_fakes import FakeClock, FakeLauncher, FakeSignaller
from leed_daylight.engine.jobs import Jobs
from leed_mcp.jobindex import JobIndex
from leed_mcp.tools import job as job_tool
from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture
from option2_folder import build_option2_project

pytestmark = pytest.mark.anyio


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    return build_option2_project(tmp_path, write_example, write_wea)


class Rig:
    """One server with the pipeline's fakes, and the jobs it starts."""

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

    async def job(self, action, **arguments):
        return await call(self.server, "job", {"action": action, **arguments})


async def test_status_of_a_running_job_then_of_its_end(tmp_path, folder):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    running = (await rig.job("status", job_id=job_id)).structured_content
    assert (running["state"], running["job_action"], running["verb"], running["case"]) == (
        "running", "prepare", "prepare", "base_case")
    assert running["progress"] == "rooms 1/1" and "finished" not in running and running["elapsed_s"] >= 0
    rig.launcher.processes[0].finish(0)
    done = (await rig.job("status", job_id=job_id)).structured_content
    assert (done["state"], done["exit_code"], done["ended"]) == ("succeeded", 0, "2026-01-01T14:30:00")
    assert done["finished"]["state"] == "succeeded" and done["finished"]["verb"] == "prepare"


async def test_a_failed_job_says_why(tmp_path, folder):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    rig.launcher.processes[0].finish(1)
    data = (await rig.job("status", job_id=job_id)).structured_content
    assert (data["state"], data["exit_code"], data["reason"]) == ("failed", 1, "the job exited with code 1")


def _finish_now(process, code=0):
    """End ``process`` with an end record whose time is now, so the files
    the job wrote fall between its start and its end."""
    process.finish(code)
    process.end_record.write_text(json.dumps(
        {"exit_code": code, "ended": datetime.datetime.now().isoformat(timespec="seconds")}))


async def test_the_finished_snapshot_survives_a_later_job(tmp_path, folder):
    rig = Rig(tmp_path)
    manifest = folder / "_run" / "manifest.json"

    def write(rooms, points):
        manifest.write_text(json.dumps({"rooms": rooms, "totals": {"grid_point_count": points, "group_count": 1}}))

    first = await rig.start(folder)
    write({"Room North": {}}, 40)
    _finish_now(rig.launcher.processes[0])
    seen = (await rig.job("status", job_id=first)).structured_content["finished"]
    assert (seen["rooms"], seen["grid_points"], seen["window_groups"]) == (1, 40, 1)
    second = await rig.start(folder)
    write({}, 0)
    _finish_now(rig.launcher.processes[1])
    assert (await rig.job("status", job_id=first)).structured_content["finished"] == seen
    assert (await rig.job("status", job_id=second)).structured_content["finished"]["rooms"] == 0


async def test_logs_of_an_older_job_read_its_own_log(tmp_path, folder):
    rig = Rig(tmp_path, lines=("first job line",))
    first = await rig.start(folder)
    rig.launcher.processes[0].finish(0)
    rig.launcher.lines = ("second job line",)
    await rig.start(folder)
    data = (await rig.job("logs", job_id=first, tail=5)).structured_content
    assert data["lines"] == ["first job line"] and data["log"].endswith("_prepare.log")
    assert (data["radiance_log"], data["radiance_lines"]) == (None, None)


async def test_wait_answers_when_the_job_ends(tmp_path, folder):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    threading.Timer(0.3, rig.launcher.processes[0].finish, args=(0,)).start()
    data = (await rig.job("wait", job_id=job_id, timeout_s=10)).structured_content
    assert (data["action"], data["state"]) == ("wait", "succeeded")


async def test_wait_answers_running_when_its_time_is_up(tmp_path, folder, monkeypatch):
    monkeypatch.setattr(job_tool, "POLL_S", 0.05)
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    assert (await rig.job("wait", job_id=job_id, timeout_s=1)).structured_content["state"] == "running"


async def test_wait_sends_progress_that_always_increases(tmp_path, folder, monkeypatch):
    monkeypatch.setattr(job_tool, "POLL_S", 0.05)
    monkeypatch.setattr(job_tool, "PROGRESS_S", 0.1)
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    record = rig.index.read(job_id)
    created = datetime.datetime.fromisoformat(record.created) - datetime.timedelta(seconds=40)
    with rig.index.locked():
        rig.index.write(dataclasses.replace(record, created=created.isoformat()))
    sent = []

    async def progress(value, total, message):
        sent.append((value, total, message))

    async with Client(rig.server) as client:
        await client.call_tool("job", {"action": "wait", "job_id": job_id, "timeout_s": 3},
                               progress_callback=progress)
    values = [value for value, _, _ in sent]
    assert len(values) >= 2 and values == sorted(set(values)) and values[0] >= 40
    assert all(total is None and message == "rooms 1/1" for _, total, message in sent)


async def test_wait_progress_message_has_no_control_characters(tmp_path, folder, monkeypatch):
    monkeypatch.setattr(job_tool, "POLL_S", 0.05)
    monkeypatch.setattr(job_tool, "PROGRESS_S", 0.1)
    rig = Rig(tmp_path, lines=("progress: \x1b[31mrooms 1/1\x07",))
    job_id = await rig.start(folder)
    sent = []

    async def progress(value, total, message):
        sent.append(message)

    async with Client(rig.server) as client:
        await client.call_tool("job", {"action": "wait", "job_id": job_id, "timeout_s": 5},
                               progress_callback=progress)
    assert sent and all(message is not None for message in sent)
    assert all(ch == "\n" or ch == "\t" or ch.isprintable() for message in sent for ch in message)
    assert sent[0] == "[31mrooms 1/1"


async def test_wait_for_an_unknown_job_is_not_found(tmp_path):
    result = await call(tool_server(tmp_path), "job", {"action": "wait", "job_id": "0" * 32, "timeout_s": 1})
    assert codes(result) == [("JOB_NOT_FOUND", "job_id")]


async def test_cancel_stops_the_job(tmp_path, folder):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    data = (await rig.job("cancel", job_id=job_id)).structured_content
    assert (data["action"], data["state"], data["exit_code"]) == ("cancel", "cancelled", 130)
    assert [name for name, _, _ in rig.signaller.sent] == ["interrupt"]


async def test_list_shows_the_jobs_newest_first(tmp_path, write_example, write_wea):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = build_option2_project(tmp_path / "a", write_example, write_wea)
    second = build_option2_project(tmp_path / "b", write_example, write_wea)
    rig = Rig(tmp_path)
    one = await rig.start(first)
    rig.launcher.processes[0].finish(0)
    record = rig.index.read(one)
    with rig.index.locked():
        rig.index.write(dataclasses.replace(record, created=(
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)).isoformat()))
    two = await rig.start(second)
    every = (await rig.job("list")).structured_content
    assert [j["job_id"] for j in every["jobs"]] == [two, one] and every["total"] == 2
    only = (await rig.job("list", folder=str(first), limit=1)).structured_content
    assert [j["job_id"] for j in only["jobs"]] == [one]


@pytest.mark.parametrize("arguments, code, field", [
    ({"action": "status"}, "ARGUMENT_MISSING", "job_id"),
    ({"action": "status", "job_id": "../x"}, "INPUT_INVALID", "job_id"),
    ({"action": "wait", "job_id": "0" * 32, "timeout_s": 51}, "INPUT_INVALID", "timeout_s"),
    ({"action": "logs", "job_id": "0" * 32, "tail": 401}, "INPUT_INVALID", "tail"),
    ({"action": "list", "job_id": "0" * 32}, "UNKNOWN_ARGUMENT", "job_id"),
    ({"action": "wait", "job_id": "0" * 32, "tail": 3}, "UNKNOWN_ARGUMENT", "tail"),
])
async def test_job_arguments_are_checked(tmp_path, arguments, code, field):
    assert (code, field) in codes(await call(tool_server(tmp_path), "job", arguments))


async def test_a_legacy_client_can_follow_a_job(tmp_path, folder):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    async with Client(rig.server, mode="legacy") as client:
        result = await client.call_tool("job", {"action": "status", "job_id": job_id})
    assert result.is_error is False and result.structured_content["state"] == "running"


async def test_project_status_names_the_indexed_job_and_marks_a_job_started_outside(tmp_path, folder):
    rig = Rig(tmp_path)
    job_id = await rig.start(folder)
    status = (await call(rig.server, "project", {"action": "status", "folder": str(folder)})).structured_content
    assert (status["job"]["job_id"], status["job"]["outside_server"]) == (job_id, False)
    rig.launcher.processes[0].finish(0)
    Jobs(folder, launcher=FakeLauncher()).start("prepare", ["--case", "base_case"])
    status = (await call(rig.server, "project", {"action": "status", "folder": str(folder)})).structured_content
    assert (status["job"]["job_id"], status["job"]["outside_server"]) == (None, True)
