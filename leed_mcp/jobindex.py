# leed_mcp/jobindex.py
"""The machine-wide job index (spec 8.1).

``<state>/jobs/<job_id>.json`` holds one record per job this server (any
instance of it on this machine) started. ``<state>`` is ``LEED_MCP_STATE``,
default ``~/.leed-mcp``; it lies outside every root and is made ``0700``
when the index is first used, never at start-up. Every change holds the
index lock, ``<state>/jobs.lock`` (``flock``; ``msvcrt.locking`` on Windows), so two servers (Claude
Desktop and Claude Code) never race; a record is written to a new
``.part`` file, flushed to disk and moved into place.
"""
from __future__ import annotations

import contextlib
import dataclasses
import datetime
import json
import logging
import os
import pathlib
import re
import stat
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from leed_daylight.engine.filelock import try_lock, unlock
from leed_daylight.engine.jobs import UnsafePath, checked_dir, open_plain
from leed_mcp.paths import inside_root

__all__ = [
    "DEFAULT_STATE", "IndexBusy", "IndexRecord", "JOB_ID", "JobIndex", "LOCK_WAIT_S", "STATE_ENV_VAR",
    "StateInsideRoot", "state_from_env",
]

STATE_ENV_VAR = "LEED_MCP_STATE"
#: The index folder when ``LEED_MCP_STATE`` is not set (spec 5.3).
DEFAULT_STATE = "~/.leed-mcp"
#: A job id: ``uuid4().hex`` (spec 8.1).
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
#: How long a call waits for the index lock, in seconds. Chosen: longer than
#: the longest hold, an ``inputs import_weather`` (``epw2wea``'s 60 s limit).
LOCK_WAIT_S = 90.0
#: How often the lock is asked for again while another server holds it.
_LOCK_RETRY_S = 0.05
#: The most bytes one record may hold, written or read. Chosen: a record
#: with its ``finished`` snapshot (a file list, capped) stays far below it.
_RECORD_MAX = 1_048_576
#: A process id above this is not one (``os.kill`` takes a C int).
_PID_MAX = 2 ** 31
_NEW_FILE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
_OPEN_LOCK = (os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
              | getattr(os, "O_BINARY", 0))

logger = logging.getLogger(__name__)


class StateInsideRoot(ValueError):
    """``LEED_MCP_STATE`` is not usable: not absolute, or inside a root."""


def _aware(text: str) -> bool:
    """Whether ``text`` is an ISO 8601 time with its offset."""
    try:
        return datetime.datetime.fromisoformat(text).tzinfo is not None
    except ValueError:
        return False


def _private(path: pathlib.Path, info: os.stat_result, *, platform: str | None = None) -> None:
    """``UnsafePath`` (``PATH_UNSAFE``) unless ``path`` is this user's and
    no one else can write it, as ssh asks of its folder (security audit L5).
    Windows has no owner id or group-write bit to check: the folder is
    under the user's own profile, so the check is skipped there."""
    if (platform or sys.platform) == "win32":
        return
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise UnsafePath(f"{path}: belongs to another user or others can write it; the job index is not "
                         f"kept there. Set {STATE_ENV_VAR} to a private folder, or leave it unset for ~/.leed-mcp.")


class IndexBusy(Exception):
    """Another server held the index lock for longer than ``LOCK_WAIT_S``."""


def state_from_env(roots: Sequence[pathlib.Path], environ: Mapping[str, str] | None = None) -> pathlib.Path:
    """The index folder from ``LEED_MCP_STATE`` (default ``~/.leed-mcp``),
    resolved; nothing is created. ``StateInsideRoot`` when the value is not
    absolute or the folder lies inside a root (spec 5.3): the server then
    exits with the message on stderr."""
    raw = (os.environ if environ is None else environ).get(STATE_ENV_VAR, "").strip()
    path = pathlib.Path(raw) if raw else pathlib.Path(DEFAULT_STATE).expanduser()
    if not path.is_absolute():
        raise StateInsideRoot(f"{STATE_ENV_VAR} must be an absolute path; got {raw!r}.")
    path = path.resolve()
    for root in roots:
        if inside_root(path, root):
            raise StateInsideRoot(
                f"{STATE_ENV_VAR} ({path}) lies inside the root {root}; the job index must be outside "
                "every project folder. Set it to a folder elsewhere, or leave it unset for ~/.leed-mcp.")
    return path


@dataclasses.dataclass(frozen=True)
class IndexRecord:
    """One job (spec 8.1). ``action`` is the tool action (``option1``),
    ``verb`` the pipeline verb it starts (``run``); ``log``, ``end_record``,
    ``pid`` and ``start_time`` come from the pipeline's ``JobRecord``;
    ``created`` is UTC with its offset."""

    job_id: str
    folder: str
    action: str
    verb: str
    case: str
    args: tuple[str, ...]
    log: str
    end_record: str
    pid: int
    start_time: float | None
    created: str
    cancel_requested: bool = False
    finished: dict[str, Any] | None = None

    def to_json(self) -> bytes:
        data = {**dataclasses.asdict(self), "args": list(self.args)}
        return (json.dumps(data, indent=2, allow_nan=False) + "\n").encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> IndexRecord:
        """The record in ``data``; ``ValueError`` when it is not a whole one."""
        raw = json.loads(data.decode("utf-8"))
        names = {field.name for field in dataclasses.fields(cls)}
        if not isinstance(raw, dict) or set(raw) != names:
            raise ValueError("not a job record")
        texts = ("job_id", "folder", "action", "verb", "case", "log", "end_record", "created")
        start = raw["start_time"]
        if (not all(isinstance(raw[key], str) for key in texts) or not JOB_ID.fullmatch(raw["job_id"])
                or not isinstance(raw["args"], list) or not all(isinstance(a, str) for a in raw["args"])
                or isinstance(raw["pid"], bool) or not isinstance(raw["pid"], int) or not 0 < raw["pid"] < _PID_MAX
                or not _aware(raw["created"])
                or not (start is None or (isinstance(start, (int, float)) and not isinstance(start, bool)))
                or not isinstance(raw["cancel_requested"], bool)
                or not (raw["finished"] is None or isinstance(raw["finished"], dict))):
            raise ValueError("not a job record")
        return cls(**{**raw, "args": tuple(raw["args"]),
                      "start_time": None if start is None else float(start)})


class JobIndex:
    """The records under ``state`` and the lock that orders every change."""

    def __init__(self, state: pathlib.Path, *, wait_s: float = LOCK_WAIT_S) -> None:
        self.state = pathlib.Path(state)
        self.wait_s = wait_s

    def _jobs_dir(self, *, make: bool) -> pathlib.Path:
        """``<state>/jobs``, never through a link. ``make`` creates what is
        missing: the state folder ``0700`` (a folder already there keeps
        its mode) and ``jobs`` in it, always ``0700``. The state folder and
        ``jobs``, when there, must be this user's and not writable by
        others (``UnsafePath``)."""
        if make:
            with contextlib.suppress(FileExistsError):
                self.state.mkdir(mode=0o700, parents=True)
        try:
            info = os.lstat(self.state)
        except FileNotFoundError:
            if make:
                raise
            return self.state / "jobs"
        if not stat.S_ISDIR(info.st_mode):
            raise UnsafePath(f"{self.state}: is not a folder; the job index cannot be kept there")
        _private(self.state, info)
        jobs = checked_dir(self.state, "jobs", make=make)
        if os.path.lexists(jobs):
            _private(jobs, os.lstat(jobs))
        if make:
            os.chmod(jobs, 0o700)
        return jobs

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the index lock for the body of a ``with``. ``IndexBusy``
        after ``wait_s`` seconds of waiting for another server."""
        self._jobs_dir(make=True)
        fd = os.open(self.state / "jobs.lock", _OPEN_LOCK, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise UnsafePath(f"{self.state / 'jobs.lock'}: is not a plain file; move it away")
            _private(self.state / "jobs.lock", info)
            deadline = time.monotonic() + self.wait_s
            while not try_lock(fd):
                if time.monotonic() >= deadline:
                    raise IndexBusy(
                        f"another leed-mcp server held the job index for more than {self.wait_s:g} s; "
                        "try again in a moment")
                time.sleep(_LOCK_RETRY_S)
            try:
                yield
            finally:
                unlock(fd)
        finally:
            os.close(fd)

    def _path(self, job_id: str) -> pathlib.Path:
        if not JOB_ID.fullmatch(job_id):
            raise ValueError(f"{job_id!r} is not a job id")
        return self._jobs_dir(make=False) / f"{job_id}.json"

    def write(self, record: IndexRecord) -> None:
        """Save ``record``: a new ``.part`` file, flushed, moved into place.
        The caller holds the lock. A record larger than ``read`` takes is
        ``ValueError`` before any file is made."""
        data = record.to_json()
        if len(data) > _RECORD_MAX:
            raise ValueError(f"job {record.job_id}: its record would be larger than {_RECORD_MAX} bytes")
        path = self._jobs_dir(make=True) / f"{record.job_id}.json"
        part = path.with_name(path.name + ".part")
        with contextlib.suppress(FileNotFoundError):
            os.unlink(part)
        fd = os.open(part, _NEW_FILE, 0o600)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(part, path)

    def read(self, job_id: str) -> IndexRecord | None:
        """The record of ``job_id``, or None when there is none. A record
        that is not a plain file, too large or not whole raises
        ``ValueError``; an I/O error propagates."""
        path = self._path(job_id)
        try:
            with open_plain(path) as handle:
                data = handle.read(_RECORD_MAX + 1)
        except FileNotFoundError:
            return None
        if len(data) > _RECORD_MAX:
            raise ValueError(f"{path}: is larger than {_RECORD_MAX} bytes")
        try:
            return IndexRecord.from_json(data)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"{path}: {exc}") from None

    def records(self) -> list[IndexRecord]:
        """Every readable record, newest first; one that cannot be read is
        left out with a warning on stderr."""
        try:
            jobs = self._jobs_dir(make=False)
            names = sorted(p.stem for p in jobs.glob("*.json") if JOB_ID.fullmatch(p.stem))
        except (FileNotFoundError, NotADirectoryError):
            return []
        found = []
        for name in names:
            try:
                record = self.read(name)
            except (OSError, ValueError) as exc:
                logger.warning("job record %s left out: %s", name, exc)
                continue
            if record is not None:
                found.append(record)
        return sorted(found, key=lambda r: r.created, reverse=True)

    def delete(self, job_id: str) -> None:
        """Remove the record of ``job_id``. The caller holds the lock."""
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._path(job_id))
