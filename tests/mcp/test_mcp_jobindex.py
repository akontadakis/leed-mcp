# tests/mcp/test_mcp_jobindex.py
"""The machine-wide job index (spec 5.3, 8.1)."""
import dataclasses
import inspect
import json
import os
import stat
import sys
import threading

import pytest

from leed_daylight.engine.jobs import UnsafePath
from leed_mcp import jobindex
from leed_mcp.jobindex import IndexBusy, IndexRecord, JobIndex, StateInsideRoot, state_from_env

JOB = "a" * 32


def _record(job_id=JOB, created="2026-01-01T12:00:00+00:00", **change):
    record = IndexRecord(job_id=job_id, folder="/invented/p", action="prepare", verb="prepare",
                         case="base_case", args=("--case", "base_case"),
                         log="_run/jobs/20260101-120000_prepare.log",
                         end_record="_run/jobs/20260101-120000_prepare.end.json", pid=10, start_time=100.0,
                         created=created)
    return dataclasses.replace(record, **change)


def test_the_state_folder_defaults_to_leed_mcp_in_the_home_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert state_from_env((), {}) == (tmp_path / ".leed-mcp").resolve()
    assert state_from_env((), {"LEED_MCP_STATE": str(tmp_path / "s")}) == (tmp_path / "s").resolve()


@pytest.mark.parametrize("value, words", [("relative/state", "must be an absolute path"),
                                          ("{root}/state", "lies inside the root")])
def test_a_relative_state_or_one_inside_a_root_is_refused(tmp_path, value, words):
    root = tmp_path.resolve()
    with pytest.raises(StateInsideRoot, match=words):
        state_from_env((root,), {"LEED_MCP_STATE": value.format(root=root)})


def test_a_record_is_written_private_and_read_back(tmp_path):
    index = JobIndex(tmp_path / "state")
    with index.locked():
        index.write(_record())
    assert index.read(JOB) == _record()
    if sys.platform != "win32":  # POSIX modes
        assert stat.S_IMODE((tmp_path / "state").stat().st_mode) == 0o700
        assert stat.S_IMODE((tmp_path / "state" / "jobs").stat().st_mode) == 0o700
        assert stat.S_IMODE((tmp_path / "state" / "jobs" / f"{JOB}.json").stat().st_mode) == 0o600
    assert not list((tmp_path / "state" / "jobs").glob("*.part"))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX owner, modes and process groups")
def test_an_existing_state_folder_keeps_its_mode(tmp_path):
    (tmp_path / "state").mkdir(mode=0o755)
    (tmp_path / "state").chmod(0o755)
    with JobIndex(tmp_path / "state").locked():
        pass
    assert stat.S_IMODE((tmp_path / "state").stat().st_mode) == 0o755
    assert stat.S_IMODE((tmp_path / "state" / "jobs").stat().st_mode) == 0o700


def test_records_come_newest_first_and_a_broken_one_is_left_out(tmp_path, caplog):
    index = JobIndex(tmp_path / "state")
    with index.locked():
        index.write(_record("b" * 32, created="2026-01-02T12:00:00+00:00"))
        index.write(_record("c" * 32, created="2026-01-03T12:00:00+00:00"))
    (tmp_path / "state" / "jobs" / ("d" * 32 + ".json")).write_text('{"job_id": "not whole"}')
    assert [r.job_id for r in index.records()] == ["c" * 32, "b" * 32]
    assert "left out" in caplog.text
    with pytest.raises(ValueError, match="not a job record"):
        index.read("d" * 32)


def test_an_unknown_job_reads_as_none_and_a_bad_id_is_refused(tmp_path):
    index = JobIndex(tmp_path / "state")
    assert index.read(JOB) is None and index.records() == []
    with pytest.raises(ValueError, match="not a job id"):
        index.read("../../etc/passwd")


def test_delete_removes_a_record(tmp_path):
    index = JobIndex(tmp_path / "state")
    with index.locked():
        index.write(_record())
        index.delete(JOB)
    assert index.read(JOB) is None


def test_the_lock_orders_two_servers_and_gives_up_after_its_wait(tmp_path):
    first, second = JobIndex(tmp_path / "state"), JobIndex(tmp_path / "state", wait_s=0.2)
    with first.locked():
        with pytest.raises(IndexBusy, match="another leed-mcp server held the job index"):
            with second.locked():
                pass
    with second.locked():
        pass


def test_the_lock_is_held_across_threads_of_one_server(tmp_path):
    index = JobIndex(tmp_path / "state", wait_s=0.2)
    entered = threading.Event()
    release = threading.Event()

    def hold():
        with index.locked():
            entered.set()
            release.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    entered.wait(5)
    try:
        with pytest.raises(IndexBusy):
            with index.locked():
                pass
    finally:
        release.set()
        thread.join()


def test_a_record_with_a_wrong_field_type_is_not_whole():
    data = json.loads(_record().to_json())
    data["pid"] = True
    with pytest.raises(ValueError, match="not a job record"):
        IndexRecord.from_json(json.dumps(data).encode())


@pytest.mark.parametrize("field, value", [
    ("created", "not a time"), ("created", "2026-01-01T12:00:00"), ("pid", 0), ("pid", 2 ** 31),
])
def test_a_record_with_a_time_or_pid_that_cannot_be_used_is_not_whole(tmp_path, caplog, field, value):
    data = json.loads(_record().to_json())
    data[field] = value
    with pytest.raises(ValueError, match="not a job record"):
        IndexRecord.from_json(json.dumps(data).encode())
    index = JobIndex(tmp_path / "state")
    with index.locked():
        index.write(_record("b" * 32))
    (tmp_path / "state" / "jobs" / f"{JOB}.json").write_text(json.dumps(data))
    assert [r.job_id for r in index.records()] == ["b" * 32] and "left out" in caplog.text


def test_a_record_too_large_to_read_back_is_never_written(tmp_path):
    index = JobIndex(tmp_path / "state")
    big = _record(finished={"files": ["x" * 1000] * 1100})
    with index.locked():
        with pytest.raises(ValueError, match="larger than"):
            index.write(big)
    assert list((tmp_path / "state" / "jobs").iterdir()) == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX owner, modes and process groups")
@pytest.mark.parametrize("where", ["state", "jobs", "jobs.lock"])
def test_a_state_folder_others_can_write_is_refused(tmp_path, where):
    index = JobIndex(tmp_path / "state")
    with index.locked():
        index.write(_record())
    (tmp_path / "state" if where == "state" else tmp_path / "state" / where).chmod(0o777)
    with pytest.raises(UnsafePath, match="others can write it"):
        with index.locked():
            pass
    if where != "jobs.lock":
        with pytest.raises(UnsafePath, match="others can write it"):
            index.records()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX owner, modes and process groups")
def test_a_state_folder_of_another_user_is_refused(tmp_path, monkeypatch):
    index = JobIndex(tmp_path / "state")
    with index.locked():
        pass
    other = os.getuid() + 1
    monkeypatch.setattr(jobindex.os, "getuid", lambda: other)
    with pytest.raises(UnsafePath, match="belongs to another user"):
        with index.locked():
            pass


def test_the_owner_check_is_skipped_on_windows(tmp_path, monkeypatch):
    monkeypatch.delattr(jobindex.os, "getuid", raising=False)
    jobindex._private(tmp_path, os.lstat(tmp_path), platform="win32")  # no AttributeError, no UnsafePath


def test_the_index_lock_no_longer_imports_fcntl_itself():
    assert "fcntl" not in inspect.getsource(jobindex)


def test_a_state_folder_inside_a_root_is_refused_when_only_the_case_differs(tmp_path):
    root = (tmp_path / "Projects").resolve()
    root.mkdir()
    swapped = root.parent / "pROJECTS"
    if not swapped.exists():
        pytest.skip("the file system tells names that differ in case apart")
    with pytest.raises(StateInsideRoot, match="lies inside the root"):
        state_from_env((root,), {"LEED_MCP_STATE": str(swapped / "state")})
