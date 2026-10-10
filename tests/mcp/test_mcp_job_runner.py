# tests/mcp/test_mcp_job_runner.py
"""Starting, watching and stopping jobs through the index (spec 8.1, 8.2,
6.3 rule 2, 6.7), with the pipeline's fakes (``tests/job_fakes.py``)."""
import contextlib
import dataclasses
import datetime
import hashlib
import json
import os
import sys

import pytest

from job_fakes import FakeClock, FakeLauncher, FakeSignaller, hold_lock
from leed_daylight.engine.jobs import JobError, Jobs
from leed_mcp.adapters import job_runner
from leed_mcp.adapters.job_runner import JobRunner
from leed_mcp.adapters.project_state import project_folders
from leed_mcp.errors import Refused
from leed_mcp.jobindex import JobIndex
from leed_mcp.server import build_server, main
from mcp_helpers import anyio_backend, call, codes, tool_server  # noqa: F401 - anyio_backend is the fixture
from option2_folder import build_option2_project

ARGS = ["--case", "base_case"]


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    return build_option2_project(tmp_path, write_example, write_wea)


class Rig:
    """A runner with fakes over an index beside ``tmp_path``."""

    def __init__(self, tmp_path, *, lines=("progress: rooms 1/1",), wait_s=90.0):
        self.launcher = FakeLauncher(lines)
        self.clock = FakeClock()
        self.signaller = FakeSignaller(self.clock, ends_on="interrupt")
        self.index = JobIndex(tmp_path.with_name(tmp_path.name + ".state"), wait_s=wait_s)
        self.runner = JobRunner(self.index, launcher=self.launcher, signaller=self.signaller, clock=self.clock)
        self.roots = (tmp_path.resolve(),)

    def start(self, folder):
        record = self.runner.start(folder, self.roots, "prepare", "prepare", "base_case", ARGS)
        self.signaller.process = self.launcher.processes[-1]
        return record


def _code(refused: Refused) -> tuple[str, str]:
    content = refused.result.structured_content
    return content["advisories"][0]["code"], content["reason"]


def test_a_start_writes_the_record_before_it_returns(tmp_path, folder):
    rig = Rig(tmp_path)
    record = rig.start(folder)
    assert rig.index.read(record.job_id) == record
    assert (record.folder, record.action, record.verb, record.case, record.args) == (
        str(folder), "prepare", "prepare", "base_case", tuple(ARGS))
    assert record.log.startswith("_run/jobs/") and record.log.endswith("_prepare.log") and record.pid == os.getpid()
    assert datetime.datetime.fromisoformat(record.created).utcoffset() == datetime.timedelta(0)
    assert rig.launcher.calls[0]["argv"][-4:] == ["prepare", str(folder), *ARGS]


def test_the_child_gets_the_environment_without_python_path_variables(tmp_path, folder, monkeypatch):
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        monkeypatch.setenv(name, "/invented")
    rig = Rig(tmp_path)
    rig.start(folder)
    env = rig.launcher.calls[0]["env"]
    assert not {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"} & set(env) and env["PATH"] == os.environ["PATH"]


def test_one_job_runs_at_a_time_across_two_servers(tmp_path, write_example, write_wea):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = build_option2_project(tmp_path / "a", write_example, write_wea)
    second = build_option2_project(tmp_path / "b", write_example, write_wea)
    one, two = Rig(tmp_path), Rig(tmp_path)
    started = one.start(first)
    with pytest.raises(Refused) as caught:
        two.start(second)
    code, reason = _code(caught.value)
    assert code == "JOB_RUNNING" and started.job_id in reason and two.launcher.calls == []
    one.launcher.processes[0].finish(0)
    assert two.start(second).folder == str(second)


def test_a_cli_job_blocks_a_start_in_its_folder_and_in_any_other(tmp_path, write_example, write_wea):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = build_option2_project(tmp_path / "a", write_example, write_wea)
    second = build_option2_project(tmp_path / "b", write_example, write_wea)
    hold_lock(first)
    rig = Rig(tmp_path)
    rig.roots = ((tmp_path / "a").resolve(), (tmp_path / "b").resolve())
    with pytest.raises(Refused) as here:
        rig.start(first)
    with pytest.raises(Refused) as elsewhere:
        rig.start(second)
    assert _code(here.value)[0] == "LOCK_HELD" and "run is already working on this folder" in _code(here.value)[1]
    assert _code(elsewhere.value)[0] == "JOB_RUNNING" and "a job started outside this server" in (
        _code(elsewhere.value)[1])
    assert rig.launcher.calls == []


def test_a_record_that_cannot_be_written_stops_the_job(tmp_path, folder, monkeypatch):
    rig = Rig(tmp_path)

    def broken(self, record):
        rig.signaller.process = rig.launcher.processes[-1]
        raise OSError("invented disk full")

    monkeypatch.setattr(JobIndex, "write", broken)
    with pytest.raises(Refused) as caught:
        rig.runner.start(folder, rig.roots, "prepare", "prepare", "base_case", ARGS)
    code, reason = _code(caught.value)
    assert code == "INTERNAL_ERROR" and "it was stopped" in reason
    assert [name for name, _, _ in rig.signaller.sent] == ["interrupt"]


def test_a_failed_start_writes_nothing(tmp_path, folder):
    rig = Rig(tmp_path)

    def refuse(argv, *, env, cwd, log):
        raise OSError("invented: the program cannot start")

    rig.runner = JobRunner(rig.index, launcher=refuse)
    with pytest.raises(OSError, match="cannot start"):
        rig.runner.start(folder, rig.roots, "prepare", "prepare", "base_case", ARGS)
    assert rig.index.records() == []


def test_an_expired_record_is_deleted_on_the_next_start(tmp_path, folder):
    rig = Rig(tmp_path)
    old = rig.start(folder)
    rig.launcher.processes[0].finish(0)
    with rig.index.locked():
        rig.index.write(dataclasses.replace(old, created="2025-01-01T10:00:00+00:00"))
    assert rig.runner.view(old.job_id, rig.roots).state.state == "expired"
    rig.start(folder)
    assert rig.index.read(old.job_id) is None


def test_a_busy_index_is_lock_held(tmp_path, folder):
    holder, rig = Rig(tmp_path), Rig(tmp_path, wait_s=0.2)
    with holder.index.locked():
        with pytest.raises(Refused) as caught:
            rig.start(folder)
    assert _code(caught.value)[0] == "LOCK_HELD" and "another leed-mcp server" in _code(caught.value)[1]


def test_an_input_write_is_refused_while_a_job_runs_on_its_folder(tmp_path, write_example, write_wea):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = build_option2_project(tmp_path / "a", write_example, write_wea)
    second = build_option2_project(tmp_path / "b", write_example, write_wea)
    rig = Rig(tmp_path)
    rig.start(first)
    with pytest.raises(Refused) as caught:
        with rig.runner.guard_write(first):
            pass
    assert _code(caught.value)[0] == "LOCK_HELD" and "is running on this folder" in _code(caught.value)[1]
    with rig.runner.guard_write(second):
        pass
    rig.launcher.processes[0].finish(0)
    with rig.runner.guard_write(first):
        pass


@pytest.mark.anyio
async def test_the_inputs_tool_writes_under_the_guard(tmp_path, folder):
    launcher = FakeLauncher()
    server = tool_server(tmp_path, launcher=launcher)
    server.jobs.start(folder, server.roots, "prepare", "prepare", "base_case", ARGS)
    sha = hashlib.sha256((folder / "materials.csv").read_bytes()).hexdigest()
    update = {"action": "update", "folder": str(folder), "file": "materials", "expected_sha256": sha,
              "rows": [{"modifier": "wall_paint", "note": "invented"}]}
    assert codes(await call(server, "inputs", update)) == [("LOCK_HELD", "folder")]
    assert (await call(server, "inputs", {"action": "read", "folder": str(folder), "file": "materials"})
            ).is_error is False
    launcher.processes[0].finish(0)
    assert (await call(server, "inputs", update)).is_error is False


def test_view_builds_the_finished_snapshot_once(tmp_path, folder):
    rig = Rig(tmp_path)
    record = rig.start(folder)
    assert rig.runner.view(record.job_id, rig.roots).record.finished is None
    rig.launcher.processes[0].finish(0)
    view = rig.runner.view(record.job_id, rig.roots)
    assert (view.state.state, view.record.finished["state"]) == ("succeeded", "succeeded")
    assert rig.index.read(record.job_id).finished == view.record.finished
    assert view.progress == "rooms 1/1" and view.elapsed_s >= 0


@pytest.mark.parametrize("change, words", [
    (None, "not in the job index"),
    ("outside", "its folder is outside the configured roots"),
    ("gone", "its project folder is gone"),
])
def test_a_job_that_cannot_be_reached_is_not_found(tmp_path, folder, change, words):
    rig = Rig(tmp_path)
    record = rig.start(folder)
    job_id, roots = record.job_id, rig.roots
    if change is None:
        job_id = "0" * 32
    elif change == "outside":
        (tmp_path / "other").mkdir()
        roots = ((tmp_path / "other").resolve(),)
    else:
        folder.rename(tmp_path / "moved")
    with pytest.raises(Refused) as caught:
        rig.runner.view(job_id, roots)
    assert _code(caught.value)[0] == "JOB_NOT_FOUND" and words in _code(caught.value)[1]


def test_views_lists_the_jobs_inside_the_roots_newest_first(tmp_path, folder):
    rig = Rig(tmp_path)
    first = rig.start(folder)
    rig.launcher.processes[0].finish(0)
    with rig.index.locked():
        rig.index.write(dataclasses.replace(first, created=(
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)).isoformat()))
    second = rig.start(folder)
    assert [v.record.job_id for v in rig.runner.views(rig.roots)] == [second.job_id, first.job_id]
    (tmp_path / "other").mkdir()
    assert rig.runner.views(((tmp_path / "other").resolve(),)) == []


def test_cancel_stops_the_job_and_records_it(tmp_path, folder):
    rig = Rig(tmp_path)
    record = rig.start(folder)
    view = rig.runner.cancel(record.job_id, rig.roots)
    assert (view.state.state, view.state.exit_code) == ("cancelled", 130)
    assert [name for name, _, _ in rig.signaller.sent] == ["interrupt"]
    assert rig.index.read(record.job_id).cancel_requested is True


def test_a_new_server_instance_cancels_a_job_another_started(tmp_path, folder):
    rig = Rig(tmp_path)
    record = rig.start(folder)
    later = JobRunner(rig.index, signaller=rig.signaller, clock=rig.clock)
    assert later.view(record.job_id, rig.roots).state.state == "running"
    assert later.cancel(record.job_id, rig.roots).state.state == "cancelled"


def _current(folder, **change):
    path = folder / "_run" / "jobs" / "current.json"
    record = json.loads(path.read_text())
    for key, value in change.items():
        record[key] = value(record[key]) if callable(value) else value
    path.write_text(json.dumps(record), newline="\n")


@pytest.mark.parametrize("change, code", [
    ({"log": lambda v: v.replace("_prepare.log", "-2_prepare.log"),
      "end_record": lambda v: v.replace("_prepare.end", "-2_prepare.end")}, "JOB_NOT_CURRENT"),
    ({"start_time": None}, "STOP_UNCONFIRMED"),
    ({"start_time": lambda v: v + 100}, "JOB_NOT_CURRENT"),
])
def test_cancel_sends_nothing_it_cannot_confirm(tmp_path, folder, change, code):
    rig = Rig(tmp_path)
    record = rig.start(folder)
    _current(folder, **change)
    with pytest.raises(Refused) as caught:
        rig.runner.cancel(record.job_id, rig.roots)
    assert _code(caught.value)[0] == code and rig.signaller.sent == []
    assert rig.index.read(record.job_id).cancel_requested is False


def test_cancel_rechecks_the_current_job_under_the_index_lock(tmp_path, folder, monkeypatch):
    rig = Rig(tmp_path)
    record = rig.start(folder)
    locked = rig.index.locked

    @contextlib.contextmanager
    def newer_job_starts_first():
        # another server starts a newer job while this cancel waited for the lock
        _current(folder, log=lambda v: v.replace("_prepare.log", "-2_prepare.log"),
                 end_record=lambda v: v.replace("_prepare.end", "-2_prepare.end"), pid=os.getpid() + 1)
        with locked():
            yield

    monkeypatch.setattr(rig.index, "locked", newer_job_starts_first)
    with pytest.raises(Refused) as caught:
        rig.runner.cancel(record.job_id, rig.roots)
    assert _code(caught.value)[0] == "JOB_NOT_CURRENT" and rig.signaller.sent == []
    assert rig.index.read(record.job_id).cancel_requested is False


def test_cancel_of_an_ended_job_changes_nothing(tmp_path, folder):
    rig = Rig(tmp_path)
    record = rig.start(folder)
    rig.launcher.processes[0].finish(0)
    assert rig.runner.cancel(record.job_id, rig.roots).state.state == "succeeded" and rig.signaller.sent == []


def test_find_matches_the_folders_current_job_to_the_index(tmp_path, folder):
    rig = Rig(tmp_path)
    record = rig.start(folder)
    assert rig.runner.find(folder, record.log) == record
    assert rig.runner.find(folder, "_run/jobs/20260101-120000_prepare.log") is None


def test_the_server_reads_its_state_folder_and_creates_nothing_until_used(tmp_path):
    server = build_server(roots=(), state=tmp_path / "state")
    assert server.jobs.index.state == tmp_path / "state" and not (tmp_path / "state").exists()


def test_the_server_exits_when_the_state_lies_inside_a_root(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("LEED_MCP_ROOTS", str(tmp_path))
    monkeypatch.setenv("LEED_MCP_STATE", str(tmp_path / "state"))
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 1 and "lies inside the root" in caplog.text


def _two_projects(tmp_path, write_example, write_wea):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    return (build_option2_project(tmp_path / "a", write_example, write_wea),
            build_option2_project(tmp_path / "b", write_example, write_wea))


def _rewrite(rig, record, **change):
    with rig.index.locked():
        rig.index.write(dataclasses.replace(record, **change))


def test_a_missing_root_is_skipped(tmp_path, folder, caplog):
    roots = ((tmp_path / "other" / "missing").resolve(), tmp_path.resolve())
    assert project_folders(roots) == [folder] and caplog.text.count(" skipped: ") == 1
    rig = Rig(tmp_path)
    rig.roots = roots
    assert rig.start(folder).folder == str(folder)


@pytest.mark.skipif(sys.platform == "win32", reason="chmod(0) does not make a folder unreadable on Windows")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a chmod(0) folder")
def test_an_unreadable_root_is_skipped(tmp_path, folder, caplog):
    closed = tmp_path / "other" / "closed"
    closed.mkdir(parents=True)
    closed.chmod(0)
    try:
        roots = (closed.resolve(), tmp_path.resolve())
        assert project_folders(roots) == [folder] and caplog.text.count(" skipped: ") == 1
        rig = Rig(tmp_path)
        rig.roots = roots
        assert rig.start(folder).folder == str(folder)
    finally:
        closed.chmod(0o700)


def test_job_running_never_names_a_folder_outside_this_servers_roots(tmp_path, write_example, write_wea):
    first, second = _two_projects(tmp_path, write_example, write_wea)
    other, here = Rig(tmp_path), Rig(tmp_path)
    other.roots, here.roots = ((tmp_path / "a").resolve(),), ((tmp_path / "b").resolve(),)
    started = other.start(first)
    with pytest.raises(Refused) as caught:
        here.start(second)
    code, reason = _code(caught.value)
    assert code == "JOB_RUNNING" and started.job_id in reason and "another leed-mcp server" in reason
    assert str(tmp_path / "a") not in reason and "base_case" not in reason and "prepare" not in reason


def test_a_malformed_record_blocks_no_start_no_write_and_no_list(tmp_path, write_example, write_wea, caplog):
    first, second = _two_projects(tmp_path, write_example, write_wea)
    rig = Rig(tmp_path)
    record = rig.start(first)
    data = json.loads(record.to_json())
    data["created"] = "not a time"
    (rig.index.state / "jobs" / f"{record.job_id}.json").write_text(json.dumps(data), newline="\n")
    assert rig.runner.views(rig.roots) == []
    with rig.runner.guard_write(first):
        pass
    assert rig.start(second).folder == str(second) and "left out" in caplog.text


def test_a_malformed_end_time_leaves_status_and_list_working(tmp_path, folder):
    rig = Rig(tmp_path)
    record = rig.start(folder)
    rig.launcher.processes[0].finish(0)
    (folder / record.end_record).write_text(json.dumps({"exit_code": 0, "ended": "not a time"}), newline="\n")
    view = rig.runner.view(record.job_id, rig.roots)
    assert (view.state.state, view.state.ended, view.record.finished["ended"]) == ("succeeded", None, None)
    assert [v.record.job_id for v in rig.runner.views(rig.roots)] == [record.job_id]


def test_a_job_without_a_start_time_counts_as_running_only_while_its_folder_lock_is_held(
        tmp_path, write_example, write_wea):
    first, second = _two_projects(tmp_path, write_example, write_wea)
    rig = Rig(tmp_path)
    rig.roots = ((tmp_path / "b").resolve(),)
    _rewrite(rig, rig.start(first), start_time=None)
    with rig.runner.guard_write(first):
        pass
    hold_lock(first)
    with pytest.raises(Refused) as caught:
        with rig.runner.guard_write(first):
            pass
    assert _code(caught.value)[0] == "LOCK_HELD"


def test_a_snapshot_too_large_to_store_is_shown_and_not_stored(tmp_path, folder, monkeypatch):
    rig = Rig(tmp_path)
    record = rig.start(folder)
    rig.launcher.processes[0].finish(0)
    huge = {"verb": "prepare", "state": "succeeded", "files": ["x" * 1000] * 1100}
    monkeypatch.setattr(job_runner, "finished", lambda record, state: huge)
    view = rig.runner.view(record.job_id, rig.roots)
    assert view.record.finished == huge and rig.index.read(record.job_id).finished is None


def test_a_refused_stop_does_not_leave_cancel_requested(tmp_path, folder, monkeypatch):
    rig = Rig(tmp_path)
    record = rig.start(folder)

    def refuse(self, **_):
        raise JobError("invented: the start time cannot be read this time")

    monkeypatch.setattr(Jobs, "stop", refuse)
    with pytest.raises(Refused) as caught:
        rig.runner.cancel(record.job_id, rig.roots)
    assert _code(caught.value)[0] == "STOP_UNCONFIRMED" and rig.signaller.sent == []
    assert rig.index.read(record.job_id).cancel_requested is False


def test_cancel_of_a_job_whose_index_record_has_no_start_time_is_unconfirmed(tmp_path, folder):
    rig = Rig(tmp_path)
    record = rig.start(folder)
    _rewrite(rig, record, start_time=None)
    with pytest.raises(Refused) as caught:
        rig.runner.cancel(record.job_id, rig.roots)
    assert _code(caught.value)[0] == "STOP_UNCONFIRMED" and rig.signaller.sent == []
    assert rig.index.read(record.job_id).cancel_requested is False
