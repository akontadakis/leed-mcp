# tests/test_engine_jobs.py
"""The wizard's jobs (spec 8.1): start, the live log, the end record, stop,
a crash, a second start, a restart, the verb list and the log pruning.

Every job here runs through the fakes of ``tests/job_fakes.py``: the
launcher writes scripted lines and starts nothing; the signaller records
signals and sends none. The one real child process (``scan`` on the
invented one-room project, which needs no Radiance) is the last test.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import signal
import sys

import pytest

import leed_daylight
from leed_daylight.constants import JOB_RECORD_ENV
from leed_daylight.engine import jobs
from leed_daylight.engine.jobs import (
    CURRENT,
    JOBS_DIR,
    KEEP_LOGS,
    KILL_GRACE_S,
    LOG_CHUNK,
    RADIANCE_TAIL_LINES,
    STOP_GRACE_S,
    VERBS,
    EndRecord,
    JobError,
    Jobs,
    LockHeld,
    job_argv,
    job_env,
    lock_message,
    popen_launcher,
    process_start_time,
    read_end_record,
)
from job_fakes import DEAD_PID, FakeClock, FakeLauncher, FakeSignaller, fixed_now, hold_lock
from option2_folder import build_option2_project

#: The real ``taskkill`` call, taken before the test guard (``conftest``)
#: replaces it, for the one test that checks its command line with a fake
#: ``subprocess.run``.
REAL_TASKKILL = jobs._TaskkillSignaller.__dict__["_taskkill"]


@pytest.fixture
def folder(tmp_path):
    path = tmp_path / "project"
    path.mkdir()
    return path


def _jobs(folder, launcher=None, *, now="2026-01-01T14:02:05", signaller=None):
    return Jobs(folder, launcher=launcher or FakeLauncher(), now=fixed_now(now),
                signaller=signaller or FakeSignaller(FakeClock()))


def _jobs_dir(folder) -> pathlib.Path:
    return folder.joinpath(*JOBS_DIR)


def test_the_verb_list_is_the_eight_writing_verbs_a_job_runs():
    assert VERBS == ("prepare", "atlas", "run", "option2", "views", "report", "calculator", "package")


def test_the_job_argv_runs_the_verb_unbuffered_without_a_shell(folder):
    """``-P``: the project folder (the job's working directory) is kept off
    ``sys.path``, so a file there cannot shadow a library."""
    assert job_argv("run", folder, ("--case", "base_case")) == [
        sys.executable, "-P", "-u", "-m", "leed_daylight", "run", str(folder), "--case", "base_case",
    ]


def test_the_job_environment_carries_the_end_record_path(tmp_path):
    env = job_env(tmp_path / "x.end.json", {"KEPT": "1"})
    assert env == {
        "KEPT": "1", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
        JOB_RECORD_ENV: str(tmp_path / "x.end.json"),
    }


def test_start_writes_the_job_record_and_the_log(folder):
    launcher = FakeLauncher(["case: base_case"])
    record = _jobs(folder, launcher).start("run", ("--case", "base_case"))

    assert record.verb == "run" and record.args == ("--case", "base_case")
    assert record.pid == os.getpid()
    assert abs(record.start_time - process_start_time(os.getpid())) < 1e-6
    assert record.log == "_run/jobs/20260101-140205_run.log"
    assert record.end_record == "_run/jobs/20260101-140205_run.end.json"
    assert record.started == "2026-01-01T14:02:05"
    written = json.loads((_jobs_dir(folder) / CURRENT).read_text())
    assert written == {**dataclasses.asdict(record), "args": ["--case", "base_case"]}
    assert (folder / record.log).read_text() == "case: base_case\n"
    assert not list(_jobs_dir(folder).glob("*.part"))

    (call,) = launcher.calls
    assert call["argv"] == job_argv("run", folder, ("--case", "base_case"))
    assert call["cwd"] == folder
    assert call["env"][JOB_RECORD_ENV] == str(folder / record.end_record)
    assert call["env"]["PYTHONUNBUFFERED"] == "1"
    assert call["env"]["PYTHONIOENCODING"] == "utf-8"


def test_a_verb_outside_the_list_is_refused_and_nothing_is_written(folder):
    launcher = FakeLauncher()
    with pytest.raises(JobError, match=r"'scan' cannot be run as a job; the verbs are prepare, atlas, run"):
        _jobs(folder, launcher).start("scan")
    assert launcher.calls == []
    assert not (folder / "_run").exists()


def test_the_log_is_read_from_the_last_offset(folder):
    jobs_ = _jobs(folder, FakeLauncher(["one", "two"]))
    record = jobs_.start("run")
    first = jobs_.read_log(0)
    assert (first.text, first.offset) == ("one\ntwo\n", 8)

    with open(folder / record.log, "ab") as handle:
        handle.write(b"three\n")
    second = jobs_.read_log(first.offset)
    assert (second.text, second.offset) == ("three\n", 14)
    assert jobs_.read_log(second.offset).text == ""


def test_a_utf8_character_split_across_two_reads_is_kept_whole(folder):
    jobs_ = _jobs(folder)
    record = jobs_.start("run")
    with open(folder / record.log, "ab") as handle:
        handle.write("area 2".encode() + "\u00b2".encode()[:1])
    first = jobs_.read_log(0)
    assert (first.text, first.offset) == ("area 2", 6)

    with open(folder / record.log, "ab") as handle:
        handle.write("\u00b2".encode()[1:] + b"\n")
    second = jobs_.read_log(first.offset)
    assert (second.text, second.offset) == ("\u00b2\n", 9)


def test_bytes_that_are_not_utf8_are_replaced(folder):
    jobs_ = _jobs(folder)
    record = jobs_.start("run")
    (folder / record.log).write_bytes(b"bad \xff byte\n")
    assert jobs_.read_log(0).text == "bad \ufffd byte\n"


def test_a_read_gives_at_most_one_chunk(folder):
    jobs_ = _jobs(folder)
    record = jobs_.start("run")
    (folder / record.log).write_bytes(b"x" * (LOG_CHUNK + 10))
    first = jobs_.read_log(0)
    assert first.offset == LOG_CHUNK
    assert jobs_.read_log(first.offset).offset == LOG_CHUNK + 10


def test_a_negative_offset_is_refused(folder):
    jobs_ = _jobs(folder)
    jobs_.start("run")
    with pytest.raises(JobError, match="offset"):
        jobs_.read_log(-1)


def test_the_latest_progress_line_of_the_whole_log_is_given(folder):
    jobs_ = _jobs(folder, FakeLauncher([
        "progress: residual pass", "progress: window group 1 of 4 (direct)", "note: something else",
    ]))
    jobs_.start("run")
    first = jobs_.read_log(0)
    assert first.progress == "window group 1 of 4 (direct)"
    # Read again from the end: nothing new, and the latest progress line is still known.
    assert jobs_.read_log(first.offset).progress == "window group 1 of 4 (direct)"


def test_a_progress_line_not_yet_ended_is_not_given(folder):
    jobs_ = _jobs(folder, FakeLauncher(["progress: residual pass"]))
    record = jobs_.start("run")
    with open(folder / record.log, "ab") as handle:
        handle.write(b"progress: window gro")
    assert jobs_.read_log(0).progress == "residual pass"


def test_no_progress_line_gives_none(folder):
    jobs_ = _jobs(folder, FakeLauncher(["case: base_case"]))
    jobs_.start("run")
    assert jobs_.read_log(0).progress is None


def test_the_tail_of_the_verbs_radiance_log_is_given(folder):
    log = folder / "_run" / "option2" / "radiance.log"
    log.parent.mkdir(parents=True)
    log.write_text("".join(f"rtrace line {i}\n" for i in range(30)), newline="\n")
    jobs_ = _jobs(folder)
    jobs_.start("option2")
    tail = jobs_.read_log(0).radiance_tail
    assert tail.splitlines() == [f"rtrace line {i}" for i in range(30 - RADIANCE_TAIL_LINES, 30)]


def test_a_verb_without_a_radiance_log_has_an_empty_tail(folder):
    (folder / "_run").mkdir()
    (folder / "_run" / "radiance.log").write_text("rtrace line\n", newline="\n")
    jobs_ = _jobs(folder)
    jobs_.start("report")
    assert jobs_.read_log(0).radiance_tail == ""


def test_no_job_reads_an_empty_log(folder):
    chunk = _jobs(folder).read_log(0)
    assert (chunk.text, chunk.offset, chunk.progress, chunk.radiance_tail) == ("", 0, None, "")


def test_a_running_job_is_running(folder):
    launcher = FakeLauncher()
    jobs_ = _jobs(folder, launcher)
    record = jobs_.start("run")
    status = jobs_.status()
    assert status.state == "running" and status.record == record
    assert (status.exit_code, status.ended, status.holder) == (None, None, None)
    assert jobs_.running()


def test_a_finished_job_shows_its_end_record(folder):
    launcher = FakeLauncher()
    jobs_ = _jobs(folder, launcher)
    record = jobs_.start("run")
    launcher.processes[0].finish(0)

    status = jobs_.status()
    assert status.state == "finished" and status.record == record
    assert (status.exit_code, status.ended) == (0, "2026-01-01T14:30:00")
    assert not jobs_.running()
    assert read_end_record(folder / record.end_record) == EndRecord(0, "2026-01-01T14:30:00")


def test_a_job_with_no_end_record_and_no_process_ended_without_a_record(folder):
    launcher = FakeLauncher()
    jobs_ = _jobs(folder, launcher)
    jobs_.start("run")
    launcher.processes[0].finish(-9, record=False)

    status = jobs_.status()
    assert status.state == "ended_without_record"
    assert status.exit_code == -9 and status.ended is None


def test_a_crashed_job_is_told_after_a_restart(folder):
    _jobs(folder, FakeLauncher(pid=DEAD_PID)).start("run")
    status = _jobs(folder).status()
    assert status.state == "ended_without_record"
    assert status.record.pid == DEAD_PID and status.exit_code is None


def test_a_malformed_end_record_counts_as_none(folder):
    (folder / "x.end.json").write_text("{not json", newline="\n")
    assert read_end_record(folder / "x.end.json") is None
    assert read_end_record(folder / "missing.end.json") is None


def test_a_second_start_is_refused(folder):
    launcher = FakeLauncher()
    jobs_ = _jobs(folder, launcher)
    jobs_.start("run")
    with pytest.raises(JobError, match=r"^a job is already running: run, started 14:02$"):
        jobs_.start("report")
    assert len(launcher.calls) == 1


def test_a_start_after_the_job_finished_is_allowed(folder):
    launcher = FakeLauncher()
    jobs_ = _jobs(folder, launcher)
    first = jobs_.start("run")
    launcher.processes[0].finish(0)
    second = jobs_.start("report")
    assert second.log == "_run/jobs/20260101-140205-2_report.log"
    assert second.end_record == "_run/jobs/20260101-140205-2_report.end.json"
    assert (folder / first.log).exists()


def test_a_start_is_refused_while_the_folder_lock_is_held(folder):
    info = hold_lock(folder, "run")
    launcher = FakeLauncher()
    with pytest.raises(LockHeld) as refused:
        _jobs(folder, launcher).start("report")
    assert str(refused.value) == lock_message(info)
    assert launcher.calls == []
    assert not (folder / "_run").exists()


def test_a_lock_held_outside_any_job_shows_busy(folder):
    info = hold_lock(folder, "run")
    status = _jobs(folder).status()
    assert status.state == "busy" and status.holder == info and status.record is None


def test_a_running_job_is_found_again_after_a_restart(folder):
    launcher = FakeLauncher(["progress: residual pass"])
    record = _jobs(folder, launcher).start("run", ("--case", "base_case"))

    again = _jobs(folder, FakeLauncher())
    status = again.status()
    assert status.state == "running" and status.record == record
    assert again.read_log(0).text == "progress: residual pass\n"
    with pytest.raises(JobError, match="a job is already running: run"):
        again.start("report")

    launcher.processes[0].finish(0)
    assert again.status().state == "finished"


def test_a_reused_process_id_is_not_trusted_as_a_running_job(folder):
    record = _jobs(folder).start("run")
    current = _jobs_dir(folder) / CURRENT
    shifted = {**json.loads(current.read_text()), "start_time": record.start_time - 10.0}
    current.write_text(json.dumps(shifted), newline="\n")
    assert _jobs(folder).status().state == "ended_without_record"


def test_a_job_whose_start_time_could_not_be_read_counts_as_running_while_its_process_lives(folder):
    record = _jobs(folder).start("run")
    current = _jobs_dir(folder) / CURRENT
    current.write_text(json.dumps({**json.loads(current.read_text()), "start_time": None}), newline="\n")
    status = _jobs(folder).status()
    assert status.state == "running" and status.record == dataclasses.replace(record, start_time=None)


def test_stop_sends_sigint_then_sigterm_then_sigkill(folder):
    clock = FakeClock()
    signaller = FakeSignaller(clock)
    launcher = FakeLauncher()
    jobs_ = _jobs(folder, launcher, signaller=signaller)
    jobs_.start("run")
    signaller.process = launcher.processes[0]

    status = jobs_.stop(sleep=clock.sleep, monotonic=clock.monotonic)

    pid = os.getpid()
    assert [(name, p) for name, p, _ in signaller.sent] == [
        ("interrupt", pid), ("terminate", pid), ("kill", pid),
    ]
    times = [t for _, _, t in signaller.sent]
    assert times[0] == 0.0
    assert times[1] == pytest.approx(STOP_GRACE_S, abs=0.5)
    assert times[2] == pytest.approx(STOP_GRACE_S + KILL_GRACE_S, abs=0.5)
    assert status.state == "ended_without_record" and status.exit_code == -9


def test_stop_ends_early_when_the_job_stops_on_sigint(folder):
    clock = FakeClock()
    signaller = FakeSignaller(clock, ends_on="interrupt")
    launcher = FakeLauncher()
    jobs_ = _jobs(folder, launcher, signaller=signaller)
    jobs_.start("run")
    signaller.process = launcher.processes[0]

    status = jobs_.stop(sleep=clock.sleep, monotonic=clock.monotonic)
    assert [name for name, _, _ in signaller.sent] == ["interrupt"]
    assert status.state == "finished" and status.exit_code == 130


def test_stop_checks_the_process_id_and_start_time_first(folder):
    record = _jobs(folder).start("run")
    current = _jobs_dir(folder) / CURRENT
    current.write_text(json.dumps({**json.loads(current.read_text()), "start_time": record.start_time - 10.0}), newline="\n")

    clock = FakeClock()
    signaller = FakeSignaller(clock)
    status = _jobs(folder, signaller=signaller).stop(sleep=clock.sleep, monotonic=clock.monotonic)
    assert signaller.sent == []
    assert status.state == "ended_without_record"


def test_stop_without_a_job_sends_nothing(folder):
    clock = FakeClock()
    signaller = FakeSignaller(clock)
    status = _jobs(folder, signaller=signaller).stop(sleep=clock.sleep, monotonic=clock.monotonic)
    assert signaller.sent == [] and status.state == "none"


def test_a_stop_after_a_restart_signals_the_recorded_process(folder):
    launcher = FakeLauncher()
    _jobs(folder, launcher).start("run")
    clock = FakeClock()
    signaller = FakeSignaller(clock, launcher.processes[0], ends_on="interrupt")
    status = _jobs(folder, signaller=signaller).stop(sleep=clock.sleep, monotonic=clock.monotonic)
    assert [name for name, _, _ in signaller.sent] == ["interrupt"]
    assert status.state == "finished" and status.exit_code == 130


def test_only_fifty_logs_are_kept(folder):
    jobs_dir = _jobs_dir(folder)
    jobs_dir.mkdir(parents=True)
    for second in range(55):
        stamp = f"20251201-1000{second:02d}"
        (jobs_dir / f"{stamp}_run.log").write_text("old\n", newline="\n")
        (jobs_dir / f"{stamp}_run.end.json").write_text('{"exit_code": 0, "ended": "x"}', newline="\n")
    names = sorted(p.name for p in jobs_dir.glob("*.log"))
    oldest = names[:6]

    record = _jobs(folder).start("run")

    logs = sorted(p.name for p in jobs_dir.glob("*.log"))
    assert len(logs) == KEEP_LOGS
    assert pathlib.Path(record.log).name in logs
    for name in oldest:
        assert not (jobs_dir / name).exists()
        assert not (jobs_dir / name.replace(".log", ".end.json")).exists()


def test_logs_of_the_same_second_are_pruned_in_the_order_they_were_made(folder):
    jobs_dir = _jobs_dir(folder)
    jobs_dir.mkdir(parents=True)
    for i in range(KEEP_LOGS):
        (jobs_dir / f"20260101-120000-{i + 2}_run.log").write_text("old\n", newline="\n")
    (jobs_dir / "20260101-120000_run.log").write_text("first\n", newline="\n")

    _jobs(folder).start("run")

    assert not (jobs_dir / "20260101-120000_run.log").exists()
    assert not (jobs_dir / "20260101-120000-2_run.log").exists()
    assert (jobs_dir / f"20260101-120000-{KEEP_LOGS + 1}_run.log").exists()


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="this system has no SIGKILL to send")
def test_the_posix_signaller_signals_the_process_group(monkeypatch):
    sent = []
    monkeypatch.setattr(jobs.os, "killpg", lambda pid, sig: sent.append((pid, sig)), raising=False)
    signaller = jobs.default_signaller("darwin")
    signaller.interrupt(4000)
    signaller.terminate(4000)
    signaller.kill(4000)
    assert sent == [(4000, signal.SIGINT), (4000, signal.SIGTERM), (4000, signal.SIGKILL)]


def test_the_posix_signaller_ignores_a_process_already_gone(monkeypatch):
    def gone(pid, sig):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(jobs.os, "killpg", gone, raising=False)
    # Controller ruling (pre-flight F6): assert the call's own behaviour,
    # not just that no exception is raised.
    assert jobs.default_signaller("darwin").interrupt(4000) is None


def test_the_windows_signaller_uses_taskkill(monkeypatch):
    ran = []
    monkeypatch.setattr(jobs.subprocess, "run", lambda argv, **kw: ran.append(argv))
    monkeypatch.setattr(jobs._TaskkillSignaller, "_taskkill", REAL_TASKKILL)
    signaller = jobs.default_signaller("win32")
    signaller.interrupt(4000)
    signaller.terminate(4000)
    signaller.kill(4000)
    assert ran == [
        ["taskkill", "/T", "/PID", "4000"],
        ["taskkill", "/T", "/F", "/PID", "4000"],
        ["taskkill", "/T", "/F", "/PID", "4000"],
    ]


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_the_real_launcher_detaches_the_job(monkeypatch, tmp_path, platform):
    seen = {}

    def fake_popen(argv, **kwargs):
        seen.update(kwargs, argv=argv)
        return "process"

    monkeypatch.setattr(jobs.subprocess, "Popen", fake_popen)
    with open(tmp_path / "x.log", "wb") as log:
        assert popen_launcher(["a"], env={"E": "1"}, cwd=tmp_path, log=log, platform=platform) == "process"
    assert seen["argv"] == ["a"] and seen["env"] == {"E": "1"} and seen["cwd"] == tmp_path
    assert seen["stdin"] == jobs.subprocess.DEVNULL and seen["stderr"] == jobs.subprocess.STDOUT
    assert seen["stdout"].name == str(tmp_path / "x.log")
    if platform == "win32":
        assert seen["creationflags"] == 0x00000008 | 0x00000200
        assert "start_new_session" not in seen
    else:
        assert seen["start_new_session"] is True
        assert "creationflags" not in seen


def test_one_real_child_runs_scan(tmp_path, write_example, write_wea, monkeypatch):
    """The real launcher, argv and environment: ``leed-daylight scan`` on the
    invented project in a child process; the child's own ``cli.main``
    writes the end record. ``scan`` starts no Radiance. A ``yaml.py`` in the
    project folder (the child's working directory) would stop the child if
    it shadowed the real library; ``-P`` keeps the folder off ``sys.path``."""
    folder = build_option2_project(tmp_path, write_example, write_wea)
    (folder / "yaml.py").write_text('raise SystemExit("a file in the project folder shadowed a library")\n', newline="\n")
    # The child imports the same leed_daylight as this test.
    monkeypatch.setenv("PYTHONPATH", str(pathlib.Path(leed_daylight.__file__).parent.parent))
    end = tmp_path / "scan.end.json"
    with open(tmp_path / "scan.log", "wb") as log:
        process = popen_launcher(job_argv("scan", folder, ()), env=job_env(end), cwd=folder, log=log)
    assert process.wait(timeout=120) == 0
    record = read_end_record(end)
    assert record is not None and record.exit_code == 0
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", record.ended)
    assert "problems: 0" in (tmp_path / "scan.log").read_text(encoding="utf-8")
    assert not (folder / "_run" / "jobs").exists()


def test_the_unconfirmed_stop_text_names_no_wizard():
    from leed_daylight.engine.jobs import STOP_UNCONFIRMED
    text = STOP_UNCONFIRMED.format(pid=1, verb="run", why="x")
    assert "wizard" not in text
    assert "cannot be confirmed as this folder's run job" in text
    assert "stop it from the terminal" in text
