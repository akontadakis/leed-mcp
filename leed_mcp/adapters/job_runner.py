# leed_mcp/adapters/job_runner.py
"""Starting, watching and stopping jobs through the index (spec 8.1, 8.2, 6.7).

D6: one job at a time on this machine. A start takes the index lock, then
refuses while any indexed job runs (``JOB_RUNNING``) or while a project
folder ``project list`` finds holds a live folder lock (a job started
outside this server): ``LOCK_HELD`` for the folder itself, ``JOB_RUNNING``
for another. Then ``Jobs(folder).start`` launches the child and the record
is written before the lock is let go, so the record exists before any
reply. An ``inputs`` write takes the same lock (``guard_write``) and is
refused while an indexed job runs on its folder, so no write lands between
a start's checks and the child taking the folder lock.

The child's environment is the server's without ``PYTHONPATH``,
``PYTHONHOME`` and ``PYTHONSTARTUP`` (spec 10), so it imports this
repository's ``leed_daylight``: ``Jobs.start`` takes no environment, so
the launcher it is given cleans the one it builds.
"""
from __future__ import annotations

import contextlib
import dataclasses
import datetime
import logging
import pathlib
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, NamedTuple, Protocol

from leed_daylight.engine.jobs import JobError, Jobs, Launcher, Signaller, lock_message, lock_state, popen_launcher
from leed_mcp.adapters.job_state import TERMINAL, JobState, finished, job_state, progress_line
from leed_mcp.adapters.project_state import project_folders
from leed_mcp.envelope import refusal
from leed_mcp.errors import Refused, one_line
from leed_mcp.jobindex import IndexBusy, IndexRecord, JobIndex
from leed_mcp.paths import PathOutsideRoots, RootsNotConfigured, contain

__all__ = ["CLEANED_ENV", "Clock", "JobRunner", "JobView", "clean_env"]

#: Removed from the child's environment (spec 10).
CLEANED_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")
#: What computing a malformed record's state can raise (security audit L3).
_MALFORMED = (ValueError, TypeError, OverflowError, OSError)

logger = logging.getLogger(__name__)


def clean_env(env: Mapping[str, str]) -> dict[str, str]:
    """``env`` without ``CLEANED_ENV``."""
    return {key: value for key, value in env.items() if key not in CLEANED_ENV}


class Clock(Protocol):
    """How a stop waits (``Jobs.stop``): ``tests/job_fakes.FakeClock`` in tests."""

    def sleep(self, seconds: float) -> None: ...

    def monotonic(self) -> float: ...


class _RealClock:
    sleep = staticmethod(time.sleep)
    monotonic = staticmethod(time.monotonic)


class JobView(NamedTuple):
    """A record and what it is now: its state, the latest progress line
    and the seconds since it started (to its end, once ended)."""

    record: IndexRecord
    state: JobState
    progress: str | None
    elapsed_s: float


def _busy(exc: IndexBusy) -> Refused:
    return Refused(refusal("LOCK_HELD", one_line(exc)))


def _inside(record: IndexRecord, roots: Sequence[pathlib.Path]) -> bool:
    """Whether the record's folder is one inside this server's roots."""
    try:
        return contain(record.folder, roots) == pathlib.Path(record.folder)
    except (PathOutsideRoots, RootsNotConfigured):
        return False


def _not_found(job_id: str, why: str) -> Refused:
    return Refused(refusal("JOB_NOT_FOUND", f"job {job_id}: {why}; job action='list' shows the jobs.",
                           field="job_id"))


class JobRunner:
    """The jobs of one server: its index, how a child starts, and how a
    stop signals and waits (the last three replaced by fakes in tests)."""

    def __init__(self, index: JobIndex, *, launcher: Launcher | None = None, signaller: Signaller | None = None,
                 clock: Clock | None = None) -> None:
        self.index = index
        inner = launcher if launcher is not None else popen_launcher

        def launch(argv, *, env, cwd, log):
            return inner(argv, env=clean_env(env), cwd=cwd, log=log)

        self._launcher = launch
        self._signaller = signaller
        self._clock = clock if clock is not None else _RealClock()

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        try:
            with self.index.locked():
                yield
        except IndexBusy as exc:
            raise _busy(exc) from None

    def _running(self) -> list[IndexRecord]:
        """The indexed jobs that run now. Expired records are deleted (spec
        8.1, on the next start). A record whose state cannot be computed is
        left out with a warning. A record without a start time counts only
        while its folder's lock is held: its pid alone may be another
        process's by now (security audit L2). The caller holds the lock."""
        running = []
        for record in self.index.records():
            try:
                state = job_state(record).state
                if state == "running" and record.start_time is None:
                    state = "running" if lock_state(pathlib.Path(record.folder))[0] in ("held", "starting") else (
                        "failed")
            except _MALFORMED as exc:
                logger.warning("job record %s left out: %s", record.job_id, exc)
                continue
            if state == "expired":
                self.index.delete(record.job_id)
            elif state not in TERMINAL:
                running.append(record)
        return running

    def _refuse_busy(self, folder: pathlib.Path, roots: Sequence[pathlib.Path]) -> None:
        running = self._running()
        if running:
            record = running[0]
            if not _inside(record, roots):
                # Another server's job: its folder and case are not this
                # server's to show (security audit L6).
                raise Refused(refusal(
                    "JOB_RUNNING", f"job {record.job_id}, started by another leed-mcp server (or the terminal), "
                    "is running elsewhere, outside this server's roots; one job runs at a time. Try again once "
                    "it has ended.", field="folder"))
            raise Refused(refusal(
                "JOB_RUNNING", f"job {record.job_id} ({record.action}, case {record.case}) is running in "
                f"{record.folder}; one job runs at a time. Wait for it (job action='wait') or cancel it.",
                field="folder"))
        for other in [folder, *(f for f in project_folders(roots) if f != folder)]:
            state, info = lock_state(other)
            if state not in ("held", "starting"):
                continue
            if other == folder:
                raise Refused(refusal("LOCK_HELD", f"{folder}: {lock_message(info)}.", field="folder"))
            raise Refused(refusal(
                "JOB_RUNNING", f"a job started outside this server is working on {other} ({lock_message(info)}); "
                "one job runs at a time.", field="folder"))

    def start(self, folder: pathlib.Path, roots: Sequence[pathlib.Path], action: str, verb: str, case: str,
              args: Sequence[str]) -> IndexRecord:
        """Start ``verb`` with ``args`` on ``folder`` and index it (spec 8.1,
        start order). A failed record write stops the job just started and
        refuses ``INTERNAL_ERROR``; a failed start writes nothing."""
        with self._locked():
            self._refuse_busy(folder, roots)
            jobs = Jobs(folder, launcher=self._launcher, signaller=self._signaller)
            started = jobs.start(verb, args)
            record = IndexRecord(
                job_id=uuid.uuid4().hex, folder=str(folder), action=action, verb=verb, case=case,
                args=tuple(started.args), log=started.log, end_record=started.end_record, pid=started.pid,
                start_time=started.start_time,
                created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
            try:
                self.index.write(record)
            except OSError as exc:
                with contextlib.suppress(JobError, OSError):
                    jobs.stop(sleep=self._clock.sleep, monotonic=self._clock.monotonic)
                raise Refused(refusal(
                    "INTERNAL_ERROR", f"the job started but its record could not be written ({one_line(exc)}); "
                    "it was stopped.")) from None
        return record

    @contextlib.contextmanager
    def guard_write(self, folder: pathlib.Path) -> Iterator[None]:
        """The index lock for an ``inputs`` write (spec 6.3 rule 2):
        ``LOCK_HELD`` while an indexed job runs on ``folder``."""
        with self._locked():
            for record in self._running():
                if pathlib.Path(record.folder) == folder:
                    raise Refused(refusal(
                        "LOCK_HELD", f"job {record.job_id} ({record.action}) is running on this folder; an input "
                        "file is not changed while a job reads them. Wait for it or cancel it.", field="folder"))
            yield

    def _record(self, job_id: str, roots: Sequence[pathlib.Path]) -> IndexRecord:
        """The record of ``job_id``, its folder still inside the roots
        (spec 8.1, recovery); ``JOB_NOT_FOUND`` otherwise."""
        record = self.index.read(job_id)
        if record is None:
            raise _not_found(job_id, "not in the job index")
        try:
            folder = contain(record.folder, roots)
        except (PathOutsideRoots, RootsNotConfigured):
            raise _not_found(job_id, "its folder is outside the configured roots") from None
        if folder != pathlib.Path(record.folder) or not folder.is_dir():
            raise _not_found(job_id, "its project folder is gone")
        return record

    def view(self, job_id: str, roots: Sequence[pathlib.Path]) -> JobView:
        """The job now. The first time it is seen ended, its ``finished``
        snapshot is built and stored in its record (spec 6.4)."""
        record = self._record(job_id, roots)
        state = job_state(record)
        if state.state in TERMINAL and state.state != "expired" and record.finished is None:
            with self._locked():
                record = self.index.read(job_id) or record
                if record.finished is None:
                    record = dataclasses.replace(record, finished=finished(record, state))
                    try:
                        self.index.write(record)
                    except (ValueError, OSError) as exc:  # too large, or the disk refused: shown, built again next time
                        logger.warning("job %s: its finished snapshot is not stored: %s", job_id, exc)
        created = datetime.datetime.fromisoformat(record.created)
        end = datetime.datetime.now(datetime.timezone.utc)
        if state.ended is not None:
            end = datetime.datetime.fromisoformat(state.ended).astimezone(datetime.timezone.utc)
        return JobView(record, state, progress_line(record), max(0.0, (end - created).total_seconds()))

    def views(self, roots: Sequence[pathlib.Path], folder: pathlib.Path | None = None) -> list[JobView]:
        """Every job inside the roots (only ``folder``'s, when given), newest first."""
        found = []
        for record in self.index.records():
            if folder is not None and pathlib.Path(record.folder) != folder:
                continue
            try:
                found.append(self.view(record.job_id, roots))
            except Refused as exc:
                # Only a job outside the roots or gone is left out; any other
                # refusal (LOCK_HELD) answers the list, so no job vanishes silently.
                advisories = (exc.result.structured_content or {}).get("advisories", [])
                if any(a.get("code") == "JOB_NOT_FOUND" for a in advisories):
                    continue
                raise
            except (ValueError, TypeError, OverflowError) as exc:
                logger.warning("job record %s left out: %s", record.job_id, exc)
        return found

    def cancel(self, job_id: str, roots: Sequence[pathlib.Path]) -> JobView:
        """Stop the job (spec 6.7): only while the folder's ``current.json``
        names it, with the index record's pid, start time, log and end
        record (``JOB_NOT_CURRENT``), and its process is confirmed as the
        job's (``STOP_UNCONFIRMED``, also when the index record has no start
        time); ``cancel_requested`` is recorded first, and set back when the
        stop is refused. A job that is not running answers its status
        unchanged."""
        record = self._record(job_id, roots)
        if job_state(record).state != "running":
            return self.view(job_id, roots)
        jobs = Jobs(pathlib.Path(record.folder), signaller=self._signaller)
        try:
            # The decisive check and the stop are under the index lock: no
            # other server can start a newer job in this folder between them.
            with self._locked():
                latest = self.index.read(job_id) or record
                if job_state(latest).state == "running":
                    if latest.start_time is None:
                        raise JobError(f"job {job_id}: its start time was not recorded, so its process cannot be "
                                       "confirmed as the job's; no signal is sent")
                    current = jobs.status().record
                    if (current is None or current.log != latest.log or current.end_record != latest.end_record
                            or current.pid != latest.pid
                            or (current.start_time is not None and current.start_time != latest.start_time)):
                        raise Refused(refusal(
                            "JOB_NOT_CURRENT", f"job {job_id} is not the folder's latest job any more "
                            f"({record.folder}/_run/jobs/current.json names another), so no signal is sent.",
                            field="job_id"))
                    jobs.check_stop()
                    self.index.write(dataclasses.replace(latest, cancel_requested=True))
                    try:
                        jobs.stop(sleep=self._clock.sleep, monotonic=self._clock.monotonic)
                    except JobError:
                        # No signal went out: the job is not being cancelled (code review M4).
                        self.index.write(latest)
                        raise
        except JobError as exc:
            raise Refused(refusal("STOP_UNCONFIRMED", one_line(exc), field="job_id")) from None
        return self.view(job_id, roots)

    def find(self, folder: pathlib.Path, log: str) -> IndexRecord | None:
        """The indexed job of ``folder`` whose log is ``log`` (as the folder's
        ``current.json`` names it), or None: a job started outside this server."""
        return next((r for r in self.index.records() if pathlib.Path(r.folder) == folder and r.log == log), None)

    def summary(self, record: IndexRecord) -> dict[str, Any]:
        """The record's identity, for a reply."""
        return {"job_id": record.job_id, "action": record.action, "verb": record.verb, "folder": record.folder,
                "case": record.case, "log": str(pathlib.Path(record.folder) / record.log),
                "created": record.created}
