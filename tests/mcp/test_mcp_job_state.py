# tests/mcp/test_mcp_job_state.py
"""What a job in the index is now (spec 8.2) and what it left (spec 6.4, 6.6)."""
import dataclasses
import datetime
import json
import os

import pytest

from conftest import symlink_or_skip
from job_fakes import DEAD_PID, write_end_record
from leed_daylight.engine.jobs import process_start_time
from leed_mcp.adapters import job_state as job_state_module
from leed_mcp.adapters.job_state import JobState, finished, job_state, log_tail, progress_line, radiance_tail
from leed_mcp.jobindex import IndexRecord

#: The end record's time is local and naive, as ``cli.main`` writes it; the
#: start is UTC, as the server records it: 30 minutes before, on any zone.
ENDED = "2026-01-01T14:30:00"
CREATED = (datetime.datetime.fromisoformat(ENDED).astimezone(datetime.timezone.utc)
           - datetime.timedelta(minutes=30)).isoformat(timespec="seconds")
NOW = datetime.datetime.fromisoformat(CREATED) + datetime.timedelta(hours=1)


def _job(tmp_path, verb="prepare", *, pid=None, cancel=False, created=CREATED,
         lines=("progress: rooms 1/1",)):
    folder = tmp_path / "p"
    (folder / "_run" / "jobs").mkdir(parents=True, exist_ok=True)
    base = f"_run/jobs/20260101-140000_{verb}"
    (folder / f"{base}.log").write_text("".join(f"{line}\n" for line in lines), newline="\n")
    pid = os.getpid() if pid is None else pid
    return IndexRecord(job_id="a" * 32, folder=str(folder), action=verb, verb=verb, case="base_case",
                       args=("--case", "base_case"), log=f"{base}.log", end_record=f"{base}.end.json", pid=pid,
                       start_time=process_start_time(pid), created=created, cancel_requested=cancel)


def _end(record, code):
    write_end_record(os.path.join(record.folder, record.end_record), code)


def test_a_job_whose_process_lives_is_running(tmp_path):
    assert job_state(_job(tmp_path), now=NOW) == JobState("running", None, None, None)


@pytest.mark.parametrize("code, cancel, state", [
    (0, False, "succeeded"), (0, True, "succeeded"), (1, False, "failed"), (1, True, "cancelled"),
    (130, True, "cancelled"),
])
def test_the_end_record_decides(tmp_path, code, cancel, state):
    record = _job(tmp_path, cancel=cancel)
    _end(record, code)
    found = job_state(record, now=NOW)
    assert (found.state, found.exit_code, found.ended) == (state, code, ENDED)
    assert (found.reason is not None) == (state == "failed")


def test_a_job_that_died_without_an_end_record_failed_unless_it_was_cancelled(tmp_path):
    died = job_state(_job(tmp_path, pid=DEAD_PID), now=NOW)
    assert died.state == "failed" and "without an end record" in died.reason
    assert job_state(_job(tmp_path, pid=DEAD_PID, cancel=True), now=NOW).state == "cancelled"


def test_a_job_older_than_30_days_or_without_its_log_is_expired(tmp_path):
    old = _job(tmp_path, created=(NOW - datetime.timedelta(days=31)).isoformat())
    assert job_state(old, now=NOW).state == "expired"
    pruned = _job(tmp_path)
    os.unlink(os.path.join(pruned.folder, pruned.log))
    assert job_state(pruned, now=NOW).state == "expired"


def test_a_linked_log_folder_is_never_read(tmp_path):
    record = _job(tmp_path)
    jobs = tmp_path / "p" / "_run" / "jobs"
    jobs.rename(tmp_path / "elsewhere")
    symlink_or_skip(jobs, tmp_path / "elsewhere")
    assert job_state(record, now=NOW).state == "expired"
    assert log_tail(record, 10) == [] and progress_line(record) is None


def test_the_log_tail_and_the_latest_progress_line_come_from_the_jobs_own_log(tmp_path):
    record = _job(tmp_path, lines=("starting", "progress: rooms 1/2", "a line", "progress: rooms 2/2", "done"))
    assert log_tail(record, 2) == ["progress: rooms 2/2", "done"]
    assert progress_line(record) == "rooms 2/2"


def test_the_radiance_tail_is_for_the_radiance_verbs_only(tmp_path):
    assert radiance_tail(_job(tmp_path)) == (None, None)
    record = _job(tmp_path, "run")
    (tmp_path / "p" / "_run" / "radiance.log").write_text("".join(f"line {n}\n" for n in range(30)), newline="\n")
    path, lines = radiance_tail(record)
    assert path == str(tmp_path / "p" / "_run" / "radiance.log") and lines[0] == "line 10" and len(lines) == 20


def _ended_at(path, text=ENDED, offset=0.0):
    stamp = datetime.datetime.fromisoformat(text).timestamp() + offset
    os.utime(path, (stamp, stamp))


def test_prepare_leaves_its_counts(tmp_path):
    record = _job(tmp_path)
    manifest = tmp_path / "p" / "_run" / "manifest.json"
    manifest.write_text(json.dumps({"rooms": {"Room North": {}}, "totals": {"grid_point_count": 40,
                                                                              "group_count": 1}}), newline="\n")
    _ended_at(manifest, offset=-60)
    out = finished(record, JobState("succeeded", 0, ENDED, None))
    assert out == {"verb": "prepare", "state": "succeeded", "exit_code": 0, "ended": ENDED, "superseded": False,
                   "manifest": str(manifest), "rooms": 1, "grid_points": 40, "window_groups": 1}


def test_a_file_changed_after_the_end_is_superseded(tmp_path):
    record = _job(tmp_path)
    manifest = tmp_path / "p" / "_run" / "manifest.json"
    manifest.write_text("{}", newline="\n")
    _ended_at(manifest, offset=60)
    out = finished(record, JobState("succeeded", 0, ENDED, None))
    assert out["superseded"] is True and "rooms" not in out


def test_a_run_leaves_its_results_and_score(tmp_path):
    record = _job(tmp_path, "run")
    results = tmp_path / "p" / "_run" / "option1" / "base_case" / "results.json"
    results.parent.mkdir(parents=True)
    results.write_text(json.dumps({"score": {"points": 2, "band": "two points", "reason": "x"}}), newline="\n")
    _ended_at(results, offset=-60)
    out = finished(record, JobState("succeeded", 0, ENDED, None))
    assert out["results"] == str(results) and out["score"] == {"points": 2, "band": "two points"}


def test_a_report_lists_the_files_it_wrote_and_the_marks_left(tmp_path):
    record = _job(tmp_path, "report")
    reports = tmp_path / "p" / "_run" / "reports" / "base_case"
    (reports / "option1").mkdir(parents=True)
    old, new = reports / "old.docx", reports / "option1" / "report.docx"
    old.write_text("x", newline="\n")
    new.write_text("x", newline="\n")
    _ended_at(old, offset=-3600)
    _ended_at(new, offset=-60)
    (tmp_path / "p" / "glare_narrative.md").write_text("[ADD a] [CONFIRM b]\n", newline="\n")
    out = finished(record, JobState("succeeded", 0, ENDED, None))
    assert out["files"] == [str(new)] and out["marks_left"] == 2


def test_without_an_end_record_nothing_is_read(tmp_path):
    out = finished(_job(tmp_path, "run", pid=DEAD_PID), JobState("failed", None, None, "killed"))
    assert out == {"verb": "run", "state": "failed", "exit_code": None, "ended": None, "superseded": False}


def _old_file(tmp_path, verb, offset):
    """The file a ``verb`` job reads its snapshot from, written ``offset``
    seconds from the end record's time, as an earlier job left it."""
    if verb == "prepare":
        path = tmp_path / "p" / "_run" / "manifest.json"
        path.write_text(json.dumps({"rooms": {"Room North": {}}, "totals": {"grid_point_count": 40,
                                                                           "group_count": 1}}), newline="\n")
    else:
        path = tmp_path / "p" / "_run" / "option1" / "base_case" / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"score": {"points": 2, "band": "two points"}}), newline="\n")
    _ended_at(path, offset=offset)
    return path


@pytest.mark.parametrize("verb", ["prepare", "run"])
@pytest.mark.parametrize("state, code", [("failed", 1), ("cancelled", 130)])
@pytest.mark.parametrize("offset", [-86400, -60])
def test_a_failed_or_cancelled_job_never_reports_counts_or_a_score(tmp_path, verb, state, code, offset):
    record = _job(tmp_path, verb)
    _old_file(tmp_path, verb, offset)
    out = finished(record, JobState(state, code, ENDED, None))
    assert out == {"verb": verb, "state": state, "exit_code": code, "ended": ENDED, "superseded": False}


@pytest.mark.parametrize("verb", ["prepare", "run"])
def test_a_succeeded_job_over_a_file_from_before_its_start_reports_nothing_from_it(tmp_path, verb):
    record = _job(tmp_path, verb)
    _old_file(tmp_path, verb, -3600)  # 30 minutes before the job started
    out = finished(record, JobState("succeeded", 0, ENDED, None))
    assert out == {"verb": verb, "state": "succeeded", "exit_code": 0, "ended": ENDED, "superseded": False}


def test_a_deliver_file_list_is_cut_and_marked_and_its_total_is_exact(tmp_path, monkeypatch):
    monkeypatch.setattr(job_state_module, "FILES_MAX", 2)
    record = _job(tmp_path, "atlas")
    atlas = tmp_path / "p" / "_run" / "atlas"
    atlas.mkdir()
    for n in range(5):
        (atlas / f"page{n}.svg").write_text("x", newline="\n")
        _ended_at(atlas / f"page{n}.svg", offset=-60)
    out = finished(record, JobState("succeeded", 0, ENDED, None))
    assert out["files"] == [str(atlas / "page0.svg"), str(atlas / "page1.svg"), "[cut: 3 more files]"]
    assert out["files_total"] == 5


@pytest.mark.parametrize("ended", ["not a time", "9999-12-31T23:59:59"])
def test_an_end_time_that_cannot_be_used_counts_as_none(tmp_path, ended):
    record = _job(tmp_path, "run")
    with open(os.path.join(record.folder, record.end_record), "w") as handle:
        json.dump({"exit_code": 0, "ended": ended}, handle)
    _old_file(tmp_path, "run", -60)
    state = job_state(record, now=NOW)
    assert state == JobState("succeeded", 0, None, None)
    assert finished(record, state) == {"verb": "run", "state": "succeeded", "exit_code": 0, "ended": None,
                                       "superseded": False}


def test_a_job_once_seen_ended_is_not_running_again_when_its_pid_is_reused(tmp_path):
    record = dataclasses.replace(_job(tmp_path), start_time=None, finished={"state": "failed"})
    found = job_state(record, now=NOW, alive=lambda pid: True, start_time=lambda pid: None)
    assert found.state == "failed" and "without an end record" in found.reason


def test_an_end_record_from_before_the_job_started_is_ignored(tmp_path):
    record = _job(tmp_path)
    _end(record, 0)
    _ended_at(os.path.join(record.folder, record.end_record), offset=-3600)  # 30 minutes before the start
    assert job_state(record, now=NOW).state == "running"


def test_an_end_record_written_while_the_process_check_ran_is_read(tmp_path):
    record = _job(tmp_path, pid=DEAD_PID)

    def ends_now(pid):
        _end(record, 0)  # the job writes its end record and exits during the check
        return False

    assert job_state(record, now=NOW, alive=ends_now).state == "succeeded"


def test_a_failed_job_reports_the_error_line_of_its_log(tmp_path):
    text = "calc.xlsm is open in another program; close calc.xlsm in Excel or Word, then try again"
    record = _job(tmp_path, lines=("progress: rooms 1/1", f"error: {text}"))
    _end(record, 1)
    state = job_state(record, now=NOW)
    assert state.state == "failed"
    assert state.reason == text


def test_a_failed_job_without_an_error_line_keeps_its_exit_code(tmp_path):
    record = _job(tmp_path)
    _end(record, 1)
    assert job_state(record, now=NOW).reason == "the job exited with code 1"
