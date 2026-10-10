# tests/job_fakes.py
"""Stand-ins for a job's process, launcher, signaller and clock (wizard
spec 8.1, "Tests use an injectable launcher").

``FakeLauncher`` writes its scripted lines to the job log and hands back a
``FakeProcess`` that runs until the test calls ``finish``. The fake process
carries this test's own process id, so the liveness and start-time checks
of ``engine.jobs`` see a live process with a real start time; nothing ever
signals it, because every test that stops a job passes a
``FakeSignaller``. ``hold_lock`` writes a folder lock naming this process,
as a job started from the wizard or a verb typed in a terminal holds it.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import os
import pathlib
import threading

from leed_daylight.constants import JOB_RECORD_ENV
from leed_daylight.engine.jobs import LOCK_FILE, LockInfo, process_start_time

#: Above the largest process id of macOS (99998) and Linux (4194304).
DEAD_PID = 2 ** 22 + 1


def write_end_record(path: pathlib.Path, code: int) -> None:
    """The end record ``cli.main`` writes, as the child would."""
    path = pathlib.Path(path)
    path.write_text(json.dumps({"exit_code": code, "ended": "2026-01-01T14:30:00"}))


class FakeProcess:
    """A job process that ends when the test says so."""

    def __init__(self, pid: int, end_record: pathlib.Path | None = None):
        self.pid = pid
        self.end_record = end_record
        self._code: int | None = None
        self._done = threading.Event()

    def finish(self, code: int, *, record: bool = True) -> None:
        """End with ``code``; ``record`` writes the end record first, as
        ``cli.main`` does; without it the process crashed or was killed."""
        if record and self.end_record is not None:
            write_end_record(self.end_record, code)
        self._code = code
        self._done.set()

    def poll(self) -> int | None:
        return self._code

    def wait(self, timeout: float | None = None) -> int:
        self._done.wait(timeout)
        return self._code


class FakeLauncher:
    """Writes ``lines`` to the log and returns a running ``FakeProcess``."""

    def __init__(self, lines=(), *, pid: int | None = None):
        self.lines = tuple(lines)
        self.pid = os.getpid() if pid is None else pid
        self.calls: list[dict] = []
        self.processes: list[FakeProcess] = []

    def __call__(self, argv, *, env, cwd, log):
        self.calls.append({"argv": list(argv), "env": dict(env), "cwd": cwd, "log": log.name})
        log.write("".join(f"{line}\n" for line in self.lines).encode("utf-8"))
        log.flush()
        process = FakeProcess(self.pid, pathlib.Path(env[JOB_RECORD_ENV]))
        self.processes.append(process)
        return process


class FakeClock:
    """``monotonic`` and ``sleep`` on one invented clock."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@dataclasses.dataclass
class FakeSignaller:
    """Records every signal with the clock's time; ``ends_on`` names the
    signal that ends the process (``None``: none does)."""
    clock: FakeClock
    process: FakeProcess | None = None
    ends_on: str | None = "kill"
    sent: list = dataclasses.field(default_factory=list)

    def _send(self, name: str, pid: int) -> None:
        self.sent.append((name, pid, self.clock.now))
        if self.process is not None and name == self.ends_on:
            self.process.finish(130 if name == "interrupt" else -9, record=name == "interrupt")

    def interrupt(self, pid: int) -> None:
        self._send("interrupt", pid)

    def terminate(self, pid: int) -> None:
        self._send("terminate", pid)

    def kill(self, pid: int) -> None:
        self._send("kill", pid)


def fixed_now(text: str = "2026-01-01T14:02:05"):
    """A ``now`` that always gives the same local time."""
    moment = datetime.datetime.fromisoformat(text)
    return lambda: moment


def hold_lock(folder: pathlib.Path, verb: str = "run") -> LockInfo:
    """A lock naming this very process, as a job or a terminal verb holds it."""
    info = LockInfo(os.getpid(), process_start_time(os.getpid()), verb, "2026-01-01T14:02:00")
    (pathlib.Path(folder) / LOCK_FILE).write_text(json.dumps(dataclasses.asdict(info)))
    return info
