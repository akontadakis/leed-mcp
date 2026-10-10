# leed_mcp/adapters/job_state.py
"""What a job in the index is now (spec 8.2), and what it left (spec 6.4).

State is computed when asked, from the record's own end record and the
process check (pid plus start time, as ``engine.jobs.Jobs`` does): there is
no scan at start-up. Every file is read as the pipeline reads job files:
its folders real ones and not links, the file plain and not a link, never
waited on, bounded.
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
from collections.abc import Callable
from typing import Any, NamedTuple

from leed_daylight.deliver.data import result_path
from leed_daylight.engine.jobs import (
    JOBS_DIR, LOG_CHUNK, PROGRESS_PREFIX, RADIANCE_LOGS, RADIANCE_TAIL_LINES, START_TIME_TOLERANCE_S,
    checked_dir, open_plain, process_alive, process_start_time, read_end_record,
)
from leed_daylight.scene.input_file import INPUT_FILE_MAX
from leed_mcp.adapters.narrative_file import NAME as NARRATIVE, count_marks
from leed_mcp.jobindex import IndexRecord

__all__ = [
    "CREDIT_OF", "FILES_MAX", "OUTPUTS", "RETENTION_S", "TERMINAL", "JobState", "finished", "job_state", "log_tail",
    "progress_line", "radiance_tail",
]

#: Records older than this are expired (spec 8.1), in seconds.
RETENTION_S = 30 * 24 * 3600
#: The states a job never leaves.
TERMINAL = frozenset({"succeeded", "failed", "cancelled", "expired"})
#: The results file each run verb writes (``deliver.data.result_path``).
CREDIT_OF = {"run": "option1", "option2": "option2", "views": "views"}
#: Each deliver verb's output folder under the project (spec 6.6).
OUTPUTS = {"report": ("_run", "reports", "{case}"), "atlas": ("_run", "atlas"), "package": ("submission",),
           "calculator": ("_run", "calculator", "{case}")}
#: A file changed this long after the job's ``ended`` (whole seconds,
#: local) still counts as the job's own: the end record's time is cut to
#: the second, after the last file was written.
_ENDED_SLACK_S = START_TIME_TOLERANCE_S
#: At most this many files in a deliver job's ``finished.files``; the rest
#: are counted (``files_total``) and the cut is marked. Chosen: 500 paths at
#: macOS's 1,024-byte path limit keep a record far below its 1 MiB cap.
FILES_MAX = 500


class JobState(NamedTuple):
    """``state`` (spec 8.2), the exit code and end time when the end record
    is there, and for ``failed`` the reason."""

    state: str
    exit_code: int | None
    ended: str | None
    reason: str | None


def _job_file(folder: pathlib.Path, rel: str) -> pathlib.Path | None:
    """``<folder>/_run/jobs/<name>`` for a record's ``rel``, with ``_run``
    and ``_run/jobs`` real folders; None when they are not."""
    prefix = "/".join(JOBS_DIR) + "/"
    if not rel.startswith(prefix) or "/" in rel[len(prefix):]:
        return None
    try:
        return checked_dir(folder, *JOBS_DIR, make=False) / rel[len(prefix):]
    except OSError:
        return None


def _alive(record: IndexRecord, alive: Callable[[int], bool], start_time: Callable[[int], float | None]) -> bool:
    """The process with the record's pid lives and, when both start times
    are known, started when the record says (``Jobs._alive``)."""
    if not alive(record.pid):
        return False
    actual = start_time(record.pid)
    if record.start_time is None or actual is None:
        return True
    return abs(actual - record.start_time) <= START_TIME_TOLERANCE_S


def _utc(text: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(text)


def _ended(text: str) -> str | None:
    """An end record's ``ended`` when it is a time this computer can use,
    else None (security audit L3: one bad end record breaks nothing)."""
    try:
        moment = datetime.datetime.fromisoformat(text)
        moment.astimezone(datetime.timezone.utc)
        moment.timestamp()
    except (ValueError, OverflowError, OSError):
        return None
    return text


def _end_record(folder: pathlib.Path, record: IndexRecord):
    """The job's end record, or None when it is missing, not whole, or
    older than the job (planted before it started; security audit L4)."""
    path = _job_file(folder, record.end_record)
    end = None if path is None else read_end_record(path)
    if end is None:
        return None
    try:
        if os.lstat(path).st_mtime < _utc(record.created).timestamp() - START_TIME_TOLERANCE_S:
            return None
    except OSError:
        return None
    return end


def job_state(record: IndexRecord, *, now: datetime.datetime | None = None,
              alive: Callable[[int], bool] = process_alive,
              start_time: Callable[[int], float | None] = process_start_time) -> JobState:
    """The record's state now (spec 8.2). ``expired``: created more than
    ``RETENTION_S`` ago, or its log is gone (the pipeline keeps 50)."""
    folder = pathlib.Path(record.folder)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    log = _job_file(folder, record.log)
    if (now - _utc(record.created)).total_seconds() > RETENTION_S or log is None or not os.path.lexists(log):
        return JobState("expired", None, None, None)
    end = _end_record(folder, record)
    # A stored snapshot means the job was once seen ended: a process that
    # took its pid since is not it (security audit L2).
    if end is None and record.finished is None and _alive(record, alive, start_time):
        return JobState("running", None, None, None)
    if end is None:
        # The job may have written its end record and ended between the
        # first look and the process check (code review M3).
        end = _end_record(folder, record)
    if end is not None:
        ended = _ended(end.ended)
        if end.exit_code == 0:
            return JobState("succeeded", 0, ended, None)
        if record.cancel_requested:
            return JobState("cancelled", end.exit_code, ended, None)
        return JobState("failed", end.exit_code, ended,
                        _last_error(folder, record) or f"the job exited with code {end.exit_code}")
    if record.cancel_requested:
        return JobState("cancelled", None, None, None)
    return JobState("failed", None, None,
                    "the job's process ended without an end record (it was killed, or the computer restarted)")


def _tail_bytes(path: pathlib.Path) -> tuple[bytes, bool]:
    """The last ``LOG_CHUNK`` bytes of ``path`` and whether more came
    before; empty when it cannot be read as a plain file."""
    try:
        with open_plain(path) as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - LOG_CHUNK))
            return handle.read(LOG_CHUNK), size > LOG_CHUNK
    except OSError:
        return b"", False


def _lines(path: pathlib.Path | None) -> list[str]:
    if path is None:
        return []
    data, cut = _tail_bytes(path)
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[1:] if cut and lines else lines


#: The prefix of the error line every pipeline command prints before it exits 1.
_ERROR_PREFIX = "error: "


def _last_error(folder: pathlib.Path, record: IndexRecord) -> str | None:
    """The text of the last ``error: `` line of the job's log, if any."""
    for line in reversed(_lines(_job_file(folder, record.log))):
        if line.startswith(_ERROR_PREFIX):
            return line[len(_ERROR_PREFIX):].strip() or None
    return None


def log_tail(record: IndexRecord, tail: int) -> list[str]:
    """The last ``tail`` lines of the job's own log (spec 6.7: the path in
    its record, not the folder's latest job)."""
    return _lines(_job_file(pathlib.Path(record.folder), record.log))[-tail:]


def progress_line(record: IndexRecord) -> str | None:
    """The text after ``progress: `` on the last line of the job's log
    that has it."""
    for line in reversed(_lines(_job_file(pathlib.Path(record.folder), record.log))):
        if line.startswith(PROGRESS_PREFIX):
            return line[len(PROGRESS_PREFIX):].strip()
    return None


def radiance_tail(record: IndexRecord) -> tuple[str | None, list[str] | None]:
    """``(path, last lines)`` of the verb's Radiance log; ``(None, None)``
    for a verb that runs no Radiance."""
    rel = RADIANCE_LOGS.get(record.verb)
    if rel is None:
        return None, None
    folder = pathlib.Path(record.folder)
    try:
        path = checked_dir(folder, *rel[:-1], make=False) / rel[-1]
    except OSError:
        return str(folder.joinpath(*rel)), []
    return str(path), _lines(path)[-RADIANCE_TAIL_LINES:]


def _read_json(folder: pathlib.Path, path: pathlib.Path) -> dict[str, Any] | None:
    """A JSON object file under ``folder``, read without following a link;
    None when it is missing, not plain, too large or not an object."""
    try:
        checked_dir(folder, *path.parent.relative_to(folder).parts, make=False)
        with open_plain(path) as handle:
            data = handle.read(INPUT_FILE_MAX + 1)
        content = json.loads(data.decode("utf-8")) if len(data) <= INPUT_FILE_MAX else None
    except (OSError, ValueError):
        return None
    return content if isinstance(content, dict) else None


def _changed_after(path: pathlib.Path, ended: float) -> bool:
    try:
        return os.lstat(path).st_mtime > ended + _ENDED_SLACK_S
    except OSError:
        return False


def _from_before(path: pathlib.Path, created: float) -> bool:
    """Whether ``path`` was last written before the job started (an
    earlier job's file), or cannot be looked at."""
    try:
        return os.lstat(path).st_mtime < created - _ENDED_SLACK_S
    except OSError:
        return True


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _score(content: dict[str, Any]) -> dict[str, Any]:
    score = content.get("score") if isinstance(content.get("score"), dict) else {}
    points, band = score.get("points"), score.get("band")
    return {"points": _count(points), "band": band if isinstance(band, str) else None}


def _files(folder: pathlib.Path, parts: tuple[str, ...], since: float, until: float) -> tuple[list[str], int]:
    """The files under ``<folder>/<parts>`` modified between ``since`` and
    ``until`` (with the slack), sorted, at most ``FILES_MAX`` of them with
    the cut marked, and how many there are; links are listed, never
    followed."""
    try:
        top = checked_dir(folder, *parts, make=False)
    except OSError:
        return [], 0
    found = []
    for here, dirs, names in os.walk(top, followlinks=False):
        dirs.sort()
        for name in names:
            path = pathlib.Path(here) / name
            try:
                modified = os.lstat(path).st_mtime
            except OSError:
                continue
            if since - _ENDED_SLACK_S <= modified <= until + _ENDED_SLACK_S:
                found.append(str(path))
    found.sort()
    if len(found) > FILES_MAX:
        return [*found[:FILES_MAX], f"[cut: {len(found) - FILES_MAX} more files]"], len(found)
    return found, len(found)


def finished(record: IndexRecord, state: JobState) -> dict[str, Any]:
    """The ``finished`` branch of an ended job (spec 6.4, 6.6), built once
    and stored: a later job overwrites the files it reads. A file modified
    after the job's ``ended`` time is not used: ``superseded: true`` and no
    score or counts. Without an end record nothing is read. The manifest
    and a results file are read only for a job that succeeded and only
    when written since it started: a failed or cancelled run never reports
    an earlier run's counts or score as its own (code review I1)."""
    folder = pathlib.Path(record.folder)
    out: dict[str, Any] = {"verb": record.verb, "state": state.state, "exit_code": state.exit_code,
                           "ended": state.ended, "superseded": False}
    if state.ended is None:
        return out
    ended = datetime.datetime.fromisoformat(state.ended).timestamp()
    created = _utc(record.created).timestamp()
    if record.verb == "prepare":
        if state.state != "succeeded":
            return out
        path = folder / "_run" / "manifest.json"
        if _changed_after(path, ended):
            return {**out, "superseded": True}
        if _from_before(path, created):
            return out
        manifest = _read_json(folder, path) or {}
        totals = manifest.get("totals") if isinstance(manifest.get("totals"), dict) else {}
        rooms = manifest.get("rooms")
        return {**out, "manifest": str(path), "rooms": len(rooms) if isinstance(rooms, dict) else None,
                "grid_points": _count(totals.get("grid_point_count")),
                "window_groups": _count(totals.get("group_count"))}
    if record.verb in CREDIT_OF:
        if state.state != "succeeded":
            return out
        path = result_path(folder, CREDIT_OF[record.verb], record.case)
        if _changed_after(path, ended):
            return {**out, "superseded": True}
        if _from_before(path, created):
            return out
        content = _read_json(folder, path)
        if content is None:
            return out
        return {**out, "results": str(path), "score": _score(content)}
    parts = tuple(part.format(case=record.case) for part in OUTPUTS[record.verb])
    out["files"], out["files_total"] = _files(folder, parts, created, ended)
    if record.verb == "report":
        try:
            with open_plain(folder / NARRATIVE) as handle:
                out["marks_left"] = count_marks(handle.read(INPUT_FILE_MAX).decode("utf-8", errors="replace"))
        except OSError:
            out["marks_left"] = None
    return out
