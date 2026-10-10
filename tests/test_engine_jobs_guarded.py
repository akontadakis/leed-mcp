# tests/test_engine_jobs_guarded.py
"""A crafted project folder cannot steer the job runner (security
review M1, L1, L2).

- M1: Stop, Quit with stop and Ctrl-C with stop signal only a job this
  wizard started, or a job found again whose recorded start time matches
  the live process; otherwise they refuse with one plain line and send
  nothing.
- L1: a job record's ``log`` and ``end_record`` must be
  ``_run/jobs/<stamp>_<verb>.log`` and ``.end.json``, read inside the
  folder; any other record is no record.
- L2: the lock file and the job record are read without following a link,
  without waiting on a FIFO, only as a plain file of at most 4 KiB.

Every signal goes to a ``FakeSignaller``, also the server's (the real one
is patched out), so nothing is ever sent to a real process. Every folder is
built fresh in ``tmp_path``.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import threading

import pytest

from conftest import symlink_or_skip
from job_fakes import FakeClock, FakeLauncher, FakeSignaller, fixed_now, hold_lock
from leed_daylight.engine import jobs as jobs_module
from leed_daylight.engine.jobs import (
    CURRENT,
    JOBS_DIR,
    LOCK_FILE,
    JobError,
    Jobs,
    acquire_lock,
    lock_state,
    process_start_time,
    release_lock,
)

#: How long a read may take before the test calls it blocked, in seconds.
BLOCKED_S = 5.0
UNCONFIRMED = "cannot be confirmed as this folder's run job"


@pytest.fixture
def folder(tmp_path):
    path = tmp_path / "project"
    path.mkdir()
    return path


@pytest.fixture
def outside(tmp_path):
    path = tmp_path / "outside"
    path.mkdir()
    (path / "secret.txt").write_text("a file outside the project, never shown\n", newline="\n")
    return path


def _jobs(folder, launcher=None, *, signaller=None):
    return Jobs(folder, launcher=launcher or FakeLauncher(), now=fixed_now("2026-01-01T14:02:05"),
                signaller=signaller or FakeSignaller(FakeClock()))


def _jobs_dir(folder) -> pathlib.Path:
    return folder.joinpath(*JOBS_DIR)


def _record(folder, **changes) -> pathlib.Path:
    """A job record of this test's own (live) process, with ``changes``."""
    jobs_dir = _jobs_dir(folder)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / "20260101-140205_run.log").write_text("a line of the job\n", newline="\n")
    data = {
        "verb": "run", "args": ["--case", "base"], "pid": os.getpid(),
        "start_time": process_start_time(os.getpid()),
        "log": "_run/jobs/20260101-140205_run.log",
        "end_record": "_run/jobs/20260101-140205_run.end.json",
        "started": "2026-01-01T14:02:05",
        **changes,
    }
    (jobs_dir / CURRENT).write_text(json.dumps(data), newline="\n")
    return jobs_dir / CURRENT


def _within(seconds: float, func):
    """``func()``'s result, or fail when it has not returned in time."""
    box: dict = {}

    def run():
        box["value"] = func()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"blocked for more than {seconds} s"
    return box["value"]


# --- M1: stop signals only a checked process ---------------------------------

def test_a_found_job_whose_start_time_is_not_recorded_is_never_signalled(folder):
    _record(folder, start_time=None)
    clock = FakeClock()
    signaller = FakeSignaller(clock)
    jobs_ = _jobs(folder, signaller=signaller)
    assert jobs_.status().state == "running"
    with pytest.raises(JobError, match=UNCONFIRMED):
        jobs_.stop(sleep=clock.sleep, monotonic=clock.monotonic)
    assert signaller.sent == []


def test_a_found_job_whose_live_start_time_cannot_be_read_is_never_signalled(folder, monkeypatch):
    _record(folder)
    monkeypatch.setattr(jobs_module, "process_start_time", lambda pid: None)
    clock = FakeClock()
    signaller = FakeSignaller(clock)
    jobs_ = _jobs(folder, signaller=signaller)
    assert jobs_.status().state == "running"
    with pytest.raises(JobError, match=UNCONFIRMED):
        jobs_.stop(sleep=clock.sleep, monotonic=clock.monotonic)
    assert signaller.sent == []


def test_this_runner_s_own_job_is_stopped_even_without_a_start_time(folder, monkeypatch):
    monkeypatch.setattr(jobs_module, "process_start_time", lambda pid: None)
    clock = FakeClock()
    launcher = FakeLauncher()
    jobs_ = _jobs(folder, launcher)
    signaller = FakeSignaller(clock, ends_on="interrupt")
    jobs_._signaller = signaller
    assert jobs_.start("run").start_time is None
    signaller.process = launcher.processes[0]
    status = jobs_.stop(sleep=clock.sleep, monotonic=clock.monotonic)
    assert [name for name, _, _ in signaller.sent] == ["interrupt"]
    assert status.state == "finished"


def test_a_found_job_whose_start_time_matches_is_stopped(folder):
    _record(folder)
    clock = FakeClock()
    signaller = FakeSignaller(clock)
    _jobs(folder, signaller=signaller).stop(sleep=clock.sleep, monotonic=clock.monotonic)
    assert [name for name, _, _ in signaller.sent] == ["interrupt", "terminate", "kill"]


def test_a_group_the_user_may_not_signal_is_refused_quietly(monkeypatch):
    def refused(pid, sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(jobs_module.os, "killpg", refused, raising=False)
    assert jobs_module._GroupSignaller().interrupt(12345) is None


# --- L1: the record's paths stay inside the folder ---------------------------

@pytest.mark.parametrize("changes", [
    {"log": "OUTSIDE"},
    {"log": "_run/jobs/../../../outside/secret.txt"},
    {"log": "_run/jobs/20260101-140205_atlas.log"},
    {"log": "_run/jobs/notes.txt"},
    {"end_record": "_run/jobs/20260101-140205_other.end.json"},
    {"end_record": "OUTSIDE"},
])
def test_a_record_whose_paths_are_not_its_own_job_files_is_no_record(folder, outside, changes):
    changes = {k: (str(outside / "secret.txt") if v == "OUTSIDE" else v) for k, v in changes.items()}
    _record(folder, **changes)
    jobs_ = _jobs(folder)
    assert jobs_.status().state == "none"
    assert jobs_.read_log(0).text == ""


def test_a_jobs_folder_that_is_a_link_is_never_read(folder, outside):
    _record(outside / "elsewhere")
    (outside / "elsewhere" / "_run" / "jobs" / "20260101-140205_run.log").write_text("outside text\n", newline="\n")
    (folder / "_run").mkdir()
    symlink_or_skip(folder / "_run" / "jobs", outside / "elsewhere" / "_run" / "jobs")
    jobs_ = _jobs(folder)
    assert jobs_.status().state == "none"
    assert jobs_.read_log(0).text == ""


def test_a_log_that_is_a_link_is_never_read(folder, outside):
    _record(folder)
    log = _jobs_dir(folder) / "20260101-140205_run.log"
    log.unlink()
    symlink_or_skip(log, outside / "secret.txt")
    assert _jobs(folder).read_log(0).text == ""


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this system")
def test_a_log_that_is_a_fifo_never_blocks(folder):
    _record(folder)
    log = _jobs_dir(folder) / "20260101-140205_run.log"
    log.unlink()
    os.mkfifo(log)
    assert _within(BLOCKED_S, lambda: _jobs(folder).read_log(0)).text == ""


# --- L2: bounded, non-blocking reads of the lock and the record --------------

@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this system")
def test_a_lock_that_is_a_fifo_never_blocks(folder):
    os.mkfifo(folder / LOCK_FILE)
    state, info = _within(BLOCKED_S, lambda: lock_state(folder))
    assert state in ("starting", "stale") and info is None


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this system")
def test_an_old_fifo_lock_is_cleared_as_stale_without_blocking(folder):
    os.mkfifo(folder / LOCK_FILE)
    os.utime(folder / LOCK_FILE, (0, 0))
    err = io.StringIO()
    info = _within(BLOCKED_S, lambda: acquire_lock(folder, "run", err=err))
    assert "cleared a stale lock that could not be read" in err.getvalue()
    release_lock(folder, info)
    assert not (folder / LOCK_FILE).exists()


def test_a_lock_that_is_a_link_is_not_followed(folder, outside):
    hold_lock(outside)
    symlink_or_skip(folder / LOCK_FILE, outside / LOCK_FILE)
    assert lock_state(outside)[0] == "held"
    assert lock_state(folder)[0] != "held"


def test_a_lock_larger_than_4_kib_is_not_read(folder):
    info = hold_lock(folder)
    data = (folder / LOCK_FILE).read_text()
    (folder / LOCK_FILE).write_text(data + " " * 5000, newline="\n")
    assert lock_state(folder) != ("held", info)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this system")
def test_a_record_that_is_a_fifo_never_blocks(folder):
    jobs_dir = _jobs_dir(folder)
    jobs_dir.mkdir(parents=True)
    os.mkfifo(jobs_dir / CURRENT)
    assert _within(BLOCKED_S, lambda: _jobs(folder).status()).state == "none"


def test_a_record_larger_than_4_kib_is_not_read(folder):
    path = _record(folder)
    path.write_text(path.read_text() + " " * 5000, newline="\n")
    assert _jobs(folder).status().state == "none"


# --- Re-review OOS-2: the Radiance log tail is read like the job log ----------

def _record_of(folder, verb: str) -> None:
    """A job record of ``verb``, of this test's own (live) process."""
    jobs_dir = _jobs_dir(folder)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / f"20260101-140205_{verb}.log").write_text("a line of the job\n", newline="\n")
    (jobs_dir / CURRENT).write_text(json.dumps({
        "verb": verb, "args": [], "pid": os.getpid(), "start_time": process_start_time(os.getpid()),
        "log": f"_run/jobs/20260101-140205_{verb}.log",
        "end_record": f"_run/jobs/20260101-140205_{verb}.end.json",
        "started": "2026-01-01T14:02:05",
    }), newline="\n")


def test_the_radiance_tail_is_shown(folder):
    _record_of(folder, "run")
    (folder / "_run" / "radiance.log").write_text("rtrace: a warning\n", newline="\n")
    assert _jobs(folder).read_log(0).radiance_tail == "rtrace: a warning"


@pytest.mark.parametrize("verb, rel", [("run", ("radiance.log",)), ("option2", ("option2", "radiance.log")),
                                       ("views", ("views", "radiance.log"))])
def test_a_radiance_log_that_is_a_link_is_not_read(folder, outside, verb, rel):
    _record_of(folder, verb)
    path = folder.joinpath("_run", *rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    symlink_or_skip(path, outside / "secret.txt")
    assert _jobs(folder).read_log(0).radiance_tail == ""


@pytest.mark.parametrize("verb, parent", [("option2", "option2"), ("views", "views")])
def test_a_radiance_log_under_a_linked_folder_is_not_read(folder, outside, verb, parent):
    _record_of(folder, verb)
    (outside / "radiance.log").write_text("text of a file outside the project\n", newline="\n")
    symlink_or_skip(folder / "_run" / parent, outside)
    assert _jobs(folder).read_log(0).radiance_tail == ""


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this system")
def test_a_radiance_log_that_is_a_fifo_never_blocks(folder):
    _record_of(folder, "run")
    os.mkfifo(folder / "_run" / "radiance.log")
    assert _within(BLOCKED_S, lambda: _jobs(folder).read_log(0)).radiance_tail == ""


# --- Re-review OOS-3: a non-file at current.json refuses the start ------------

@pytest.mark.parametrize("make", [os.mkdir, pytest.param(getattr(os, "mkfifo", None), marks=pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="no FIFOs on this system"))], ids=["folder", "fifo"])
def test_a_non_file_at_the_job_record_refuses_the_start_before_any_launch(folder, make):
    launcher = FakeLauncher()
    _jobs_dir(folder).mkdir(parents=True)
    make(_jobs_dir(folder) / CURRENT)

    with pytest.raises(OSError) as caught:
        _jobs(folder, launcher).start("atlas")

    assert str(caught.value) == (
        f"{_jobs_dir(folder).resolve() / CURRENT}: is not a plain file; move it away and try again"
    )
    assert launcher.calls == []
    assert not list(_jobs_dir(folder).glob("*.log"))
