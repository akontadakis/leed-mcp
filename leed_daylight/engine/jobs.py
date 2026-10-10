# leed_daylight/engine/jobs.py
"""The folder lock (wizard spec 8.2) and the wizard's jobs (spec 8.1).

``<folder>/.leed-daylight.lock`` says one command is writing to a project
folder. It is taken by the ``main()`` of ``prepare``, ``atlas``, ``run``,
``option2``, ``views``, ``report`` and ``package`` -- from the terminal or
from the wizard alike -- after the argument and folder checks, and released
in a ``finally``. Library functions never take it, so ``report`` building
the atlas in-process does not lock twice; it is not re-entrant.

The file is created atomically (``O_CREAT | O_EXCL``) and holds the process
id, the process start time, the verb and the wall-clock start, for the
refusal's message. Its holder keeps it open with the kernel's lock on it
(``flock``; ``msvcrt.locking`` on Windows), which the kernel frees when the
holder's process ends however it ends (security review L6). A lock file no
command holds so is stale and is taken over in place, with a one-line note;
nothing is renamed aside, so no command can lose another's lock. Only a lock
whose text names a process that still lives -- the process exists and its
start time equals the recorded one within ``START_TIME_TOLERANCE_S``, or a
start time is unknown -- is still kept as held without the kernel lock, as a
lock written by an earlier version of this tool, which takes none.

A job is one writing verb run as a separate process for the wizard:
``python -P -u -m leed_daylight <verb> <folder> ...`` with no shell (``-P``:
the project folder, its working directory, is not put on ``sys.path``, so a
file there cannot stand in for a library), its output
in ``_run/jobs/<stamp>_<verb>.log``, its record in ``_run/jobs/current.json``
and its end record (written by the child's own ``cli.main``) beside the log.
A job is found again after the wizard restarts, from the record, the end
record and the same process id and start-time check as the lock's holder.

This is the one module that starts, signals or inspects another process
(``ps`` for a start time, the job processes, ``taskkill`` on Windows).
"""
from __future__ import annotations

import contextlib
import dataclasses
import datetime
import errno
import json
import os
import pathlib
import re
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import BinaryIO, Protocol

from ..constants import JOB_RECORD_ENV
from .filelock import try_lock as _try_lock
from .filelock import unlock as _unlock
from .in_use import replace_explained

LOCK_FILE = ".leed-daylight.lock"
#: A recorded start time and the live process's may differ by this much
#: (``ps`` gives whole seconds).
START_TIME_TOLERANCE_S = 2.0
#: A lock file that cannot be read yet is a command still writing it while
#: it is younger than this, in seconds; older, it is stale.
YOUNG_LOCK_S = 5.0
#: How often a create, a stale lock cleared meanwhile, is tried.
_ATTEMPTS = 5

#: Windows: rights to read a process's times (PROCESS_QUERY_LIMITED_INFORMATION).
_WINDOWS_QUERY = 0x1000
#: Windows FILETIME ticks (100 ns) from 1601-01-01 to 1970-01-01.
_WINDOWS_EPOCH_TICKS = 116444736000000000


class LockHeld(ValueError):
    """Another command holds the folder lock."""


# ---------------------------------------------------------------------------
# Safe writes (security review H1): never through a link, never outside
# ---------------------------------------------------------------------------

#: Windows opens files in text mode unless told otherwise, which changes
#: line endings and stops reading at Ctrl-Z; every low-level open here is binary.
_BINARY = getattr(os, "O_BINARY", 0)
#: A new file: created here, never an existing one, never through a link.
_NEW_FILE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | _BINARY


class UnsafePath(OSError):
    """A path the wizard would write through that is a link or not a
    plain file or folder; the text is one line saying what to do."""


def is_link(path) -> bool:
    """Whether ``path`` is a symbolic link or, on Windows, an NTFS junction
    (``os.path.isjunction``, Python 3.12 on; re-review of OOS-1)."""
    isjunction = getattr(os.path, "isjunction", None)
    return os.path.islink(path) or (isjunction is not None and isjunction(path))


def refuse_link(path) -> None:
    """Refuse (``UnsafePath``) a path that is a symbolic link or a junction."""
    if is_link(path):
        raise UnsafePath(
            f"{path}: is a symbolic link; nothing is written or deleted through a link; delete it and try again"
        )


def refuse_not_file(path) -> None:
    """Refuse (``UnsafePath``) a path that is there and is not a plain
    file: a folder, a FIFO, a link (re-review OOS-3)."""
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        raise UnsafePath(f"{path}: is not a plain file; move it away and try again")


def checked_dir(folder, *parts: str, make: bool = True) -> pathlib.Path:
    """``<folder>/<parts...>`` under the resolved ``folder``, each part a
    real folder and never a link, so the result stays inside the project
    folder. ``make`` creates the missing parts; without it only the parts
    that exist are checked (the path is still the whole one) and nothing is
    created: a verb calls it so before it deletes, renames or replaces
    anything under the path (re-review OOS-1)."""
    path = pathlib.Path(folder).resolve()
    missing = False
    for part in parts:
        if part in ("", ".", "..") or "/" in part or "\\" in part:
            raise UnsafePath(f"{part!r}: is not a folder name inside the project folder")
        path = path / part
        if missing:
            continue
        refuse_link(path)
        if not os.path.lexists(path):
            if not make:
                missing = True
                continue
            with contextlib.suppress(FileExistsError):
                path.mkdir()
            refuse_link(path)
        if not path.is_dir():
            raise UnsafePath(f"{path}: is not a folder; move it away and try again")
    return path


def checked_below(root, path) -> pathlib.Path:
    """``checked_dir(root, <each part of path below root>, make=False)``:
    every folder from ``root`` down to ``path`` that exists is a real
    folder and not a link, and nothing is created. ``path`` lies lexically
    under ``root`` (built from it). Called before a cache file or its
    fingerprint sidecar in ``path`` is deleted or replaced."""
    return checked_dir(root, *pathlib.Path(path).relative_to(root).parts, make=False)


def replace_file(root, path, data: bytes) -> pathlib.Path:
    """Write ``data`` to ``path`` under ``root`` as the safe save does
    (security review H1): every folder from ``root`` down to ``path``'s
    folder checked (``checked_below``; it must exist), a link at ``path``
    itself refused, the bytes written to a new ``<path>.part`` (created
    exclusively, never through a link) and moved into place. Returns the
    path written, inside the resolved ``root``."""
    path = pathlib.Path(path)
    target = checked_below(root, path.parent) / path.name
    refuse_link(target)
    part = target.with_name(target.name + ".part")
    write_new(part, data)
    replace_from_part(part, target)
    return target


def clear_leftover(path) -> None:
    """Remove a leftover file at ``path`` (an earlier crash's ``.part``);
    a link there is refused, not removed."""
    refuse_link(path)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


def write_new(path, data: bytes, *, mode: int = 0o666, leftover: bool = True) -> None:
    """Write ``data`` to a new file at ``path``, created exclusively and
    never through a link. ``leftover``: a plain file already there (an
    earlier crash's ``.part``) is removed first; without it any file there
    is refused. A link there is always refused."""
    if leftover:
        clear_leftover(path)
    fd = os.open(path, _NEW_FILE, mode)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def replace_from_part(part, target) -> None:
    """``os.replace(part, target)`` once ``part`` is checked to be a plain
    file and not a link, and ``target`` is not a link."""
    refuse_link(part)
    refuse_link(target)
    if not stat.S_ISREG(os.lstat(part).st_mode):
        raise UnsafePath(f"{part}: is not a plain file; delete it and try again")
    replace_explained(part, target)


def append_private(path, text: str) -> None:
    """Append ``text`` to ``path``, readable by its owner only when it is
    new; never through a link, and only to a plain file (a FIFO there is
    refused at once, never waited on)."""
    refuse_link(path)
    flags = (os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0) | _BINARY)
    fd = os.open(path, flags, 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise UnsafePath(f"{path}: is not a plain file; move it away and try again")
    with os.fdopen(fd, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


#: The most bytes the lock file or the job record may hold (security review L2).
SMALL_FILE_MAX = 4096
#: Opening a file only to read it: never through a link, never waiting on a FIFO.
_READ_PLAIN = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | _BINARY
#: What ``open`` gives for a link under ``O_NOFOLLOW`` (macOS ``ELOOP``, some BSDs ``EMLINK``).
_LINK_ERRNOS = (errno.ELOOP, errno.EMLINK)


def open_plain(path, *, platform: str | None = None) -> BinaryIO:
    """``path`` opened for reading as a plain file: a link, a FIFO, a
    folder or any other non-plain file is refused (``UnsafePath``) without
    waiting; ``FileNotFoundError`` when there is none. Windows has no
    ``O_NOFOLLOW``, so a link or junction is refused by name first there,
    and again after the open when the name is now a link or no longer
    names the file opened (one put in place between the check and the
    open); a folder, which Windows refuses to open, is refused too."""
    windows = (platform or sys.platform) == "win32"
    if windows:
        refuse_link(path)
    try:
        fd = os.open(path, _READ_PLAIN)
    except OSError as exc:
        if exc.errno in _LINK_ERRNOS:
            raise UnsafePath(f"{path}: is a symbolic link; it is not read") from None
        if windows and isinstance(exc, PermissionError) and os.path.isdir(path):
            raise UnsafePath(f"{path}: is not a plain file; it is not read") from None
        raise
    if windows:
        try:
            here, there = os.lstat(path), os.fstat(fd)
        except OSError:
            os.close(fd)
            raise
        if stat.S_ISLNK(here.st_mode) or (here.st_dev, here.st_ino) != (there.st_dev, there.st_ino):
            os.close(fd)
            raise UnsafePath(f"{path}: is a symbolic link; it is not read")
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise UnsafePath(f"{path}: is not a plain file; it is not read")
    return os.fdopen(fd, "rb")


def read_small(path, limit: int = SMALL_FILE_MAX) -> bytes:
    """The bytes of the plain file ``path`` (``open_plain``), refused
    (``UnsafePath``) when it holds more than ``limit`` bytes."""
    with open_plain(path) as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise UnsafePath(f"{path}: is larger than {limit} bytes; it is not read")
    return data


@dataclasses.dataclass(frozen=True)
class LockInfo:
    """What the lock file holds (as JSON with the same keys): the holder's
    process id, its process start time (epoch seconds), its verb, and the
    local time the lock was taken, ISO to the second."""
    pid: int
    start_time: float
    verb: str
    started: str


def _windows_process(pid: int):
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(_WINDOWS_QUERY, False, pid)
    return kernel32, handle


def process_start_time(pid: int, *, platform: str = sys.platform) -> float | None:
    """When process ``pid`` started, in epoch seconds; None when there is no
    such process. macOS: ``ps -o lstart=`` read as local time. Windows:
    ``GetProcessTimes`` (written, never run on the machines this was
    built on)."""
    if pid <= 0:
        return None
    if platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32, handle = _windows_process(pid)
        if not handle:
            return None
        try:
            created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
            if not kernel32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                            ctypes.byref(kernel), ctypes.byref(user)):
                return None
            ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
            return (ticks - _WINDOWS_EPOCH_TICKS) / 1e7
        finally:
            kernel32.CloseHandle(handle)
    try:
        done = subprocess.run(
            ["/bin/ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=10, check=False,
            env=dict(os.environ, LC_ALL="C"),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = done.stdout.strip()
    if done.returncode != 0 or not text:
        return None
    try:
        return time.mktime(time.strptime(text, "%a %b %d %H:%M:%S %Y"))
    except ValueError:
        return None


def process_alive(pid: int, *, platform: str = sys.platform) -> bool:
    """Whether process ``pid`` exists. macOS: ``os.kill(pid, 0)`` succeeds
    or is refused for permission. Windows: ``OpenProcess`` succeeds."""
    if pid <= 0:
        return False
    if platform == "win32":
        kernel32, handle = _windows_process(pid)
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _parse(data: bytes) -> LockInfo | None:
    """The lock in ``data``, or None when it is not a whole lock."""
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(raw, dict) or set(raw) != {"pid", "start_time", "verb", "started"}:
        return None
    pid, start, verb, started = raw["pid"], raw["start_time"], raw["verb"], raw["started"]
    if isinstance(pid, bool) or not isinstance(pid, int):
        return None
    if isinstance(start, bool) or not isinstance(start, (int, float)):
        return None
    if not isinstance(verb, str) or not isinstance(started, str):
        return None
    return LockInfo(pid, float(start), verb, started)


def _holder(info: LockInfo, alive, start_time) -> str:
    """``"alive"``: the process exists with the recorded start time;
    ``"unchecked"``: it exists but a start time is unknown (the lock records
    none, or none can be read); ``"dead"``: no such process, or another
    process now has its id."""
    if not alive(info.pid):
        return "dead"
    actual = start_time(info.pid)
    if actual is None or info.start_time <= 0.0:
        return "unchecked"
    return "alive" if abs(actual - info.start_time) <= START_TIME_TOLERANCE_S else "dead"


def _holder_alive(info: LockInfo, alive, start_time) -> bool:
    """Whether the lock's holder counts as alive: checked or not."""
    return _holder(info, alive, start_time) != "dead"


def _lock_bytes(path: pathlib.Path) -> bytes:
    """The lock file's bytes, read bounded and without waiting
    (``read_small``); a lock that is a link, not a plain file or too large
    is never a lock a command wrote, so it reads as empty (not a whole
    lock). ``FileNotFoundError`` when there is none."""
    try:
        return read_small(path)
    except UnsafePath:
        return b""


def _fd_bytes(fd: int) -> bytes:
    """The bytes of the open lock file ``fd``, read bounded; more than
    ``SMALL_FILE_MAX`` reads as empty (not a whole lock)."""
    os.lseek(fd, 0, os.SEEK_SET)
    data = os.read(fd, SMALL_FILE_MAX + 1)
    return b"" if len(data) > SMALL_FILE_MAX else data


def _kernel_held(path: pathlib.Path) -> bool | None:
    """Whether another open file holds the kernel lock of the plain lock
    file ``path``: asked with a shared lock given back at once. None when
    there is no plain file to ask."""
    try:
        with open_plain(path) as handle:
            if not _try_lock(handle.fileno(), shared=True):
                return True
            _unlock(handle.fileno())
            return False
    except OSError:
        return None


def _classify(data: bytes, modified: float, alive, start_time, now) -> tuple[str, LockInfo | None]:
    """A lock's state by its text alone, for a lock no command holds by the
    kernel: a lock written by an earlier version of this tool, which takes
    no kernel lock, is still respected while its process lives."""
    info = _parse(data)
    if info is None:
        return ("starting" if now() - modified < YOUNG_LOCK_S else "stale"), None
    return ("held" if _holder_alive(info, alive, start_time) else "stale"), info


def _read(folder: pathlib.Path, alive, start_time, now) -> tuple[str, LockInfo | None, bytes | None]:
    path = pathlib.Path(folder) / LOCK_FILE
    try:
        modified = os.lstat(path).st_mtime
        data = _lock_bytes(path)
    except FileNotFoundError:
        return "free", None, None
    if _kernel_held(path):
        # A command holds it now. While it is still writing its text (or
        # is taking over a stale lock in place) the text is not yet its own.
        info = _parse(data)
        if info is None or not _holder_alive(info, alive, start_time):
            return "starting", None, data
        return "held", info, data
    state, info = _classify(data, modified, alive, start_time, now)
    return state, info, data


def lock_state(folder: pathlib.Path, *, alive: Callable[[int], bool] = process_alive,
               start_time: Callable[[int], float | None] = process_start_time,
               now: Callable[[], float] = time.time) -> tuple[str, LockInfo | None]:
    """``("free", None)``; ``("held", info)``; ``("stale", info)``;
    ``("starting", None)`` for a lock that cannot be read yet and is younger
    than ``YOUNG_LOCK_S`` (older, it is ``("stale", None)``)."""
    state, info, _ = _read(folder, alive, start_time, now)
    return state, info


def _clock(started: str) -> str:
    try:
        return datetime.datetime.fromisoformat(started).strftime("%H:%M")
    except ValueError:
        return started


def lock_message(info: LockInfo | None, *, unchecked_lock: pathlib.Path | None = None) -> str:
    """The refusal a second command gets; ``None`` is a lock still being
    written. ``unchecked_lock``, the lock file's path, says the holder's
    process exists but its start time could not be checked."""
    if info is None:
        return "another command is starting on this folder; try again in a few seconds"
    text = (
        f"{info.verb} is already working on this folder "
        f"(started {_clock(info.started)}, process {info.pid})"
    )
    if unchecked_lock is None:
        return text
    return (
        f"{text}; process {info.pid} exists but its start time could not be checked, so the "
        f"lock is kept; if process {info.pid} is not a leed-daylight command, delete "
        f"{unchecked_lock} by hand"
    )


def _refusal(folder: pathlib.Path, info: LockInfo | None, alive, start_time) -> LockHeld:
    if info is not None and _holder(info, alive, start_time) == "unchecked":
        return LockHeld(lock_message(info, unchecked_lock=pathlib.Path(folder) / LOCK_FILE))
    return LockHeld(lock_message(info))


#: The open lock files this process holds, by lock path and lock.
_HELD: dict[tuple[str, LockInfo], int] = {}
#: How often, and how far apart in seconds, a command asks again for a kernel
#: lock that a reader holds for a moment (``lock_state`` asks with a shared lock).
_LOCK_TRIES = 20
_LOCK_RETRY_S = 0.005
#: Opening an existing lock file: never through a link, never waiting on a FIFO.
_OPEN_LOCK = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | _BINARY


def _lock_now(fd: int) -> bool:
    """``_try_lock`` (exclusive), asked again for a moment while a reader
    holds it."""
    for attempt in range(_LOCK_TRIES):
        if _try_lock(fd):
            return True
        if attempt + 1 < _LOCK_TRIES:
            time.sleep(_LOCK_RETRY_S)
    return False


def _same_file(fd: int, path: pathlib.Path) -> bool:
    """Whether ``path`` still names the open file ``fd`` (a lock released
    meanwhile is deleted, and a new one may stand at its name)."""
    try:
        here = os.lstat(path)
    except FileNotFoundError:
        return False
    there = os.fstat(fd)
    return (here.st_dev, here.st_ino) == (there.st_dev, there.st_ino)


def _write_lock(fd: int, info: LockInfo) -> None:
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, json.dumps(dataclasses.asdict(info)).encode("utf-8"))


def _clear_planted(folder: pathlib.Path, path: pathlib.Path) -> bool:
    """Delete the link, FIFO or other non-plain file at the lock's name (no
    command ever writes one); True when it was still there. Clearers take
    the folder's own kernel lock first, so two of them never both delete,
    and one never deletes a plain lock another created meanwhile."""
    guard = None if sys.platform == "win32" else os.open(folder, os.O_RDONLY)
    try:
        if guard is not None:
            import fcntl

            fcntl.flock(guard, fcntl.LOCK_EX)
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            return False
        if stat.S_ISREG(mode) or stat.S_ISDIR(mode):
            return False
        os.unlink(path)
        return True
    finally:
        if guard is not None:
            os.close(guard)


def _folder_refusal(path: pathlib.Path) -> LockHeld:
    """The refusal for a folder at the lock's name (re-review OOS-4)."""
    return LockHeld(f"{path}: is a folder, not a lock file; move it away and try again")


def _stale_note(info: LockInfo | None) -> str:
    if info is None:
        return "note: cleared a stale lock that could not be read"
    return (
        f"note: cleared a stale lock left by {info.verb} "
        f"(process {info.pid}, started {_clock(info.started)})"
    )


def acquire_lock(folder, verb: str, *, err=None, on_stale: Callable[[], None] | None = None,
                 alive: Callable[[int], bool] = process_alive,
                 start_time: Callable[[int], float | None] = process_start_time) -> LockInfo:
    """Take the lock of ``folder`` for ``verb``, or raise ``LockHeld``.

    The lock file is created exclusively, or opened when it is there, and
    held by the kernel lock (``_try_lock``) until ``release_lock``: a file
    another command holds so is refused. A file nobody holds so is the lock
    of a command that has ended, unless its text names a process that still
    lives (an earlier version of this tool) or it cannot be read yet and is
    young; otherwise it is stale: ``on_stale`` (a test hook) runs, and the
    file is taken over in place with a note to ``err`` (default
    ``sys.stderr``). Nothing is ever renamed aside, so no command can lose
    another's lock (security review L6). A link or FIFO at the lock's name
    is old junk, cleared as stale; a folder there is refused."""
    folder = pathlib.Path(folder)
    path = folder / LOCK_FILE
    pid = os.getpid()
    mine = LockInfo(pid, start_time(pid) or 0.0, verb,
                    datetime.datetime.now().isoformat(timespec="seconds"))
    for _ in range(_ATTEMPTS):
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | _BINARY, 0o644)
            created = True
        except FileExistsError:
            created = False
            try:
                found = os.lstat(path)
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(found.st_mode):
                raise _folder_refusal(path) from None
            if not stat.S_ISREG(found.st_mode):
                if time.time() - found.st_mtime < YOUNG_LOCK_S:
                    raise LockHeld(lock_message(None)) from None
                if on_stale is not None:
                    on_stale()
                if _clear_planted(folder, path):
                    print(_stale_note(None), file=err if err is not None else sys.stderr)
                continue
            try:
                fd = os.open(path, _OPEN_LOCK)
            except FileNotFoundError:
                continue
            except OSError as exc:
                if exc.errno in _LINK_ERRNOS or exc.errno == errno.EISDIR:
                    continue  # replaced meanwhile: look again
                raise
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                continue
            if not created:
                state, _ = _classify(_fd_bytes(fd), os.fstat(fd).st_mtime, alive, start_time, time.time)
                if state == "stale" and on_stale is not None:
                    on_stale()
            if not _lock_now(fd):
                os.close(fd)
                state, info, _ = _read(folder, alive, start_time, time.time)
                if state in ("free", "stale"):
                    continue  # its holder let go meanwhile: try again
                raise _refusal(folder, info, alive, start_time)
            if not _same_file(fd, path):
                os.close(fd)
                continue
            if not created:
                data = _fd_bytes(fd)
                state, info = _classify(data, os.fstat(fd).st_mtime, alive, start_time, time.time)
                if state != "stale":
                    os.close(fd)
                    raise _refusal(folder, info, alive, start_time)
                print(_stale_note(info), file=err if err is not None else sys.stderr)
            _write_lock(fd, mine)
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(fd)
            raise
        _HELD[(str(path), mine)] = fd
        return mine
    raise LockHeld(lock_message(None))


def release_lock(folder, info: LockInfo, *, platform: str = sys.platform) -> None:
    """Remove the lock only when it is still ``info``'s, then give back the
    kernel lock. The file is deleted while it is still held, so a command
    waiting on it finds it gone (``_same_file``) and creates a new one.
    Windows (``_release_windows``) cannot delete an open file: the kernel
    lock is given back first."""
    path = pathlib.Path(folder) / LOCK_FILE
    fd = _HELD.pop((str(path), info), None)
    if platform == "win32":
        _release_windows(path, fd, info)
        return
    try:
        if _parse(_lock_bytes(path)) == info:
            path.unlink()
    except FileNotFoundError:
        pass
    finally:
        if fd is not None:
            os.close(fd)


def _release_windows(path: pathlib.Path, fd: int | None, info: LockInfo) -> None:
    """Windows: empty the lock file while the kernel lock is still held,
    give the kernel lock back, close the file, then delete it, asked again
    for a moment while a reader (``lock_state``) has it open. A file that
    still cannot be deleted is left empty: a lock that cannot be read, so
    ``starting`` while young and ``stale`` after, never ``held``, and the
    next command takes it over. Only an empty file (the one emptied here)
    or one still holding ``info`` is deleted: once the kernel lock is given
    back another command may take the file over, and its lock stays."""
    emptied = False
    if fd is not None:
        try:
            if _parse(_fd_bytes(fd)) == info:
                os.ftruncate(fd, 0)
                emptied = True
        finally:
            _unlock(fd, platform="win32")
            os.close(fd)
    for attempt in range(_LOCK_TRIES):
        try:
            data = _lock_bytes(path)
            if not (_parse(data) == info or (emptied and data == b"")):
                return
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt + 1 < _LOCK_TRIES:
                time.sleep(_LOCK_RETRY_S)


@contextlib.contextmanager
def folder_lock(folder, verb: str, *, err=None) -> Iterator[LockInfo]:
    """The lock of ``folder`` for the body of a ``with``; released however
    the body ends, ``KeyboardInterrupt`` included."""
    info = acquire_lock(folder, verb, err=err)
    try:
        yield info
    finally:
        release_lock(folder, info)


# ---------------------------------------------------------------------------
# Jobs (wizard spec 8.1)
# ---------------------------------------------------------------------------

#: The verbs a job may run, and no other.
VERBS = ("prepare", "atlas", "run", "option2", "views", "report", "calculator", "package")
#: The folder of job logs, end records and the job record, under the project.
JOBS_DIR = ("_run", "jobs")
#: The record of the job started last.
CURRENT = "current.json"
#: How many job logs, each with its end record, are kept.
KEEP_LOGS = 50
#: A stop waits this long after SIGINT before SIGTERM, in seconds ...
STOP_GRACE_S = 10.0
#: ... and this long after SIGTERM before SIGKILL.
KILL_GRACE_S = 5.0
#: The most log bytes one read returns.
LOG_CHUNK = 262144
#: The start of a progress line (spec 8.4).
PROGRESS_PREFIX = "progress: "
#: Each verb's Radiance log, under the project folder.
RADIANCE_LOGS = {
    "run": ("_run", "radiance.log"),
    "option2": ("_run", "option2", "radiance.log"),
    "views": ("_run", "views", "radiance.log"),
}
#: How many lines of that log the page shows.
RADIANCE_TAIL_LINES = 20
#: A stop's refusal when the process cannot be confirmed as the job's (security review M1).
STOP_UNCONFIRMED = (
    "process {pid} cannot be confirmed as this folder's {verb} job ({why}), so no signal is sent to "
    "it; stop it from the terminal if it is still running"
)
#: How often a stop looks whether the job has ended, in seconds.
_STOP_POLL_S = 0.2
#: How long a stop waits for the job to be gone after SIGKILL, in seconds.
_KILL_WAIT_S = 1.0
#: Windows process creation flags: DETACHED_PROCESS, CREATE_NEW_PROCESS_GROUP.
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
#: A job log's name: the stamp, a clash number from 2, the verb.
_LOG_NAME = re.compile(r"^(\d{8}-\d{6})(?:-(\d+))?_([a-z0-9]+)\.log$")
#: The folder of the job files, as a record names it.
_JOBS_PREFIX = "/".join(JOBS_DIR) + "/"
#: A record's ``log``: exactly ``_run/jobs/<stamp>[-n]_<verb>.log``.
_RECORD_LOG = re.compile(re.escape(_JOBS_PREFIX) + r"(?P<base>\d{8}-\d{6}(?:-\d+)?_(?P<verb>[a-z0-9]+))\.log")


class JobError(ValueError):
    """A job cannot be started, or a request about one is wrong."""


class JobProcess(Protocol):
    """What a launcher hands back: ``subprocess.Popen`` or a test's fake."""
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class Launcher(Protocol):
    """Starts ``argv`` with ``env`` in ``cwd``, its output to ``log``."""

    def __call__(self, argv: list[str], *, env: dict[str, str], cwd: pathlib.Path,
                 log: BinaryIO) -> JobProcess: ...


def popen_launcher(argv, *, env, cwd, log, platform: str = sys.platform) -> subprocess.Popen:
    """The real launcher: no shell, no input, stdout and stderr to ``log``.
    macOS: a new session, so the job leads its own process group and
    outlives the wizard's terminal. Windows (written, never run here): a
    detached process in its own group."""
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": log, "stderr": subprocess.STDOUT,
              "cwd": cwd, "env": env}
    if platform == "win32":
        kwargs["creationflags"] = _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(argv, **kwargs)


class Signaller(Protocol):
    """The three steps of a stop."""

    def interrupt(self, pid: int) -> None: ...

    def terminate(self, pid: int) -> None: ...

    def kill(self, pid: int) -> None: ...


class _GroupSignaller:
    """macOS: each signal goes to the job's process group, which the job
    leads (``start_new_session``), so Radiance children get it too."""

    @staticmethod
    def _send(pid: int, sig: int) -> None:
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def interrupt(self, pid: int) -> None:
        self._send(pid, signal.SIGINT)

    def terminate(self, pid: int) -> None:
        self._send(pid, signal.SIGTERM)

    def kill(self, pid: int) -> None:
        self._send(pid, signal.SIGKILL)


class _TaskkillSignaller:
    """Windows (written, never run here): ``taskkill /T`` ends the job's
    process tree, ``/F`` by force."""

    @staticmethod
    def _taskkill(pid: int, *flags: str) -> None:
        try:
            subprocess.run(["taskkill", "/T", *flags, "/PID", str(pid)],
                           capture_output=True, check=False, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass

    def interrupt(self, pid: int) -> None:
        self._taskkill(pid)

    def terminate(self, pid: int) -> None:
        self._taskkill(pid, "/F")

    def kill(self, pid: int) -> None:
        self._taskkill(pid, "/F")


def default_signaller(platform: str = sys.platform) -> Signaller:
    """The real signaller of ``platform``."""
    return _TaskkillSignaller() if platform == "win32" else _GroupSignaller()


def job_argv(verb: str, folder: pathlib.Path, args: Sequence[str]) -> list[str]:
    """``python -P -u -m leed_daylight <verb> <folder> <args...>``: a list,
    run with no shell; ``-P`` keeps the working directory (the project
    folder) off ``sys.path``."""
    args = list(args)
    if not all(isinstance(a, str) for a in args):
        raise JobError(f"job arguments must be text, got {args!r}")
    return [sys.executable, "-P", "-u", "-m", "leed_daylight", verb, str(folder), *args]


def job_env(record_path: pathlib.Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """``base`` (default: this process's environment) with unbuffered UTF-8
    output and ``LEED_JOB_RECORD`` naming the end record."""
    env = dict(os.environ if base is None else base)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env[JOB_RECORD_ENV] = str(record_path)
    return env


@dataclasses.dataclass(frozen=True)
class JobRecord:
    """``current.json``: the job's verb and arguments, its process id and
    start time (None when it could not be read), its log and end record
    (relative POSIX paths under the folder) and its local start time."""
    verb: str
    args: tuple[str, ...]
    pid: int
    start_time: float | None
    log: str
    end_record: str
    started: str


@dataclasses.dataclass(frozen=True)
class EndRecord:
    """What ``cli.main`` writes when the verb ends."""
    exit_code: int
    ended: str


@dataclasses.dataclass(frozen=True)
class JobStatus:
    """``state``: ``"none"`` (no job yet), ``"running"``, ``"finished"``
    (the end record is there), ``"ended_without_record"`` (crashed or
    killed), or ``"busy"`` (no job runs, but a process that is not the
    recorded job holds the folder lock: ``holder``)."""
    state: str
    record: JobRecord | None
    exit_code: int | None
    ended: str | None
    holder: LockInfo | None


@dataclasses.dataclass(frozen=True)
class LogChunk:
    """New log text from an offset; the offset after it; the latest
    progress line of the whole log so far; the Radiance log's tail."""
    text: str
    offset: int
    progress: str | None
    radiance_tail: str


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def read_end_record(path) -> EndRecord | None:
    """The end record at ``path``, or None when it is missing, not whole,
    or not a plain file of at most ``SMALL_FILE_MAX`` bytes."""
    try:
        raw = json.loads(read_small(path).decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(raw, dict) or set(raw) != {"exit_code", "ended"}:
        return None
    if not _is_int(raw["exit_code"]) or not isinstance(raw["ended"], str):
        return None
    return EndRecord(raw["exit_code"], raw["ended"])


def _record_paths_ok(raw: dict) -> bool:
    """Whether the record's ``log`` and ``end_record`` are exactly its own
    job's files, ``_run/jobs/<stamp>_<verb>.log`` and ``.end.json``, with
    the record's verb (security review L1)."""
    match = _RECORD_LOG.fullmatch(raw["log"])
    return (match is not None and match.group("verb") == raw["verb"]
            and raw["end_record"] == f"{_JOBS_PREFIX}{match.group('base')}.end.json")


def _read_record(path: pathlib.Path) -> JobRecord | None:
    """The job record at ``path``, read bounded and without waiting
    (``read_small``); None when it is missing, not a plain file of at most
    ``SMALL_FILE_MAX`` bytes, not whole, or names files that are not its
    own job's."""
    try:
        raw = json.loads(read_small(path).decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    fields = {f.name for f in dataclasses.fields(JobRecord)}
    if not isinstance(raw, dict) or set(raw) != fields:
        return None
    start = raw["start_time"]
    texts = (raw["verb"], raw["log"], raw["end_record"], raw["started"])
    if (not all(isinstance(t, str) for t in texts) or not _is_int(raw["pid"])
            or not isinstance(raw["args"], list) or not all(isinstance(a, str) for a in raw["args"])
            or not (start is None or (isinstance(start, (int, float)) and not isinstance(start, bool)))
            or not _record_paths_ok(raw)):
        return None
    return JobRecord(raw["verb"], tuple(raw["args"]), raw["pid"],
                     None if start is None else float(start), raw["log"], raw["end_record"],
                     raw["started"])


def _record_part(path: pathlib.Path) -> pathlib.Path:
    return path.with_name(path.name + ".part")


def _write_record(path: pathlib.Path, record: JobRecord) -> None:
    part = _record_part(path)
    data = {**dataclasses.asdict(record), "args": list(record.args)}
    write_new(part, (json.dumps(data, indent=2) + "\n").encode("utf-8"))
    replace_from_part(part, path)


def _log_names(jobs_dir: pathlib.Path) -> list[pathlib.Path]:
    """The job logs in ``jobs_dir``, oldest first."""
    found = []
    for path in jobs_dir.glob("*.log"):
        match = _LOG_NAME.match(path.name)
        if match:
            found.append(((match.group(1), int(match.group(2) or 1)), path))
    return [path for _, path in sorted(found)]


def _new_names(jobs_dir: pathlib.Path, stamp: str, verb: str) -> tuple[pathlib.Path, pathlib.Path]:
    """The log and end record of a new job: ``<stamp>_<verb>``, or
    ``<stamp>-2_<verb>``, ``-3`` ... when a log of that stamp exists."""
    number = 1
    while True:
        base = stamp if number == 1 else f"{stamp}-{number}"
        if not any(jobs_dir.glob(f"{base}_*.log")):
            return jobs_dir / f"{base}_{verb}.log", jobs_dir / f"{base}_{verb}.end.json"
        number += 1


def _prune_logs(jobs_dir: pathlib.Path, keep: int = KEEP_LOGS) -> None:
    logs = _log_names(jobs_dir)
    for path in logs[:max(0, len(logs) - keep)]:
        path.unlink(missing_ok=True)
        path.with_name(path.name[:-len(".log")] + ".end.json").unlink(missing_ok=True)


def _whole_utf8(data: bytes) -> int:
    """How many bytes of ``data`` end on a whole UTF-8 character."""
    i = len(data) - 1
    while i >= 0 and len(data) - i < 4 and data[i] & 0xC0 == 0x80:
        i -= 1
    if i < 0:
        return len(data)
    lead = data[i]
    if lead >> 5 == 0b110:
        size = 2
    elif lead >> 4 == 0b1110:
        size = 3
    elif lead >> 3 == 0b11110:
        size = 4
    else:
        size = 1
    return i if len(data) - i < size else len(data)


def _latest_progress(path: pathlib.Path, end: int) -> str | None:
    """The text after ``progress: `` on the last whole line before ``end``
    that has it, read backwards in chunks."""
    prefix = PROGRESS_PREFIX.encode("utf-8")
    try:
        handle = open_plain(path)
    except OSError:
        return None
    with handle:
        pos, carry, whole = end, b"", False
        while pos > 0:
            start = max(0, pos - LOG_CHUNK)
            handle.seek(start)
            parts = (handle.read(pos - start) + carry).split(b"\n")
            pos = start
            if not whole:
                if len(parts) == 1:
                    carry = b""  # all of it belongs to the unfinished last line
                    continue
                parts, whole = parts[:-1], True
            carry, lines = (parts[0], parts[1:]) if start > 0 else (b"", parts)
            for line in reversed(lines):
                line = line.rstrip(b"\r")
                if line.startswith(prefix):
                    return line[len(prefix):].decode("utf-8", errors="replace").strip()
    return None


def _radiance_tail(folder: pathlib.Path, verb: str) -> str:
    """The last lines of ``verb``'s Radiance log, read as the job log is:
    its folders real ones and not links, the log itself a plain file and
    not a link, never waited on, at most ``LOG_CHUNK`` bytes; empty when it
    cannot be read so (re-review OOS-2)."""
    rel = RADIANCE_LOGS.get(verb)
    if rel is None:
        return ""
    try:
        path = checked_dir(folder, *rel[:-1], make=False) / rel[-1]
        with open_plain(path) as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - LOG_CHUNK))
            data = handle.read()
    except OSError:
        return ""
    lines = data.decode("utf-8", errors="replace").splitlines()
    if size > LOG_CHUNK and lines:
        lines = lines[1:]
    return "\n".join(lines[-RADIANCE_TAIL_LINES:])


class Jobs:
    """The wizard's one job at a time on ``folder``.

    ``launcher`` starts the process (default ``popen_launcher``);
    ``signaller`` stops it (default ``default_signaller(platform)``);
    ``now`` gives the local time of a start (default
    ``datetime.datetime.now``). A new ``Jobs`` on the same folder finds a
    job started earlier from ``current.json``, its end record and its
    process id and start time.
    """

    def __init__(self, folder: pathlib.Path, *, launcher: Launcher | None = None,
                 signaller: Signaller | None = None,
                 now: Callable[[], datetime.datetime] | None = None,
                 platform: str = sys.platform):
        self.folder = pathlib.Path(folder)
        if launcher is None:
            def launcher(argv, *, env, cwd, log):
                return popen_launcher(argv, env=env, cwd=cwd, log=log, platform=platform)
        self._launcher = launcher
        self._signaller = signaller if signaller is not None else default_signaller(platform)
        self._now = now if now is not None else datetime.datetime.now
        self._guard = threading.Lock()
        self._process: JobProcess | None = None
        self._record: JobRecord | None = None

    @property
    def jobs_dir(self) -> pathlib.Path:
        return self.folder.joinpath(*JOBS_DIR)

    def _job_file(self, rel: str) -> pathlib.Path | None:
        """The path of ``rel`` (``_run/jobs/<name>``, as a checked record
        names it) inside the folder, with ``_run`` and ``_run/jobs`` real
        folders and not links; None when either is not (security review
        L1)."""
        try:
            return checked_dir(self.folder, *JOBS_DIR, make=False) / rel[len(_JOBS_PREFIX):]
        except OSError:
            return None

    def _current(self) -> JobRecord | None:
        if self._record is not None:
            return self._record
        path = self._job_file(_JOBS_PREFIX + CURRENT)
        return None if path is None else _read_record(path)

    def _own(self, record: JobRecord) -> bool:
        return self._process is not None and self._record == record

    def _stop_refusal(self, record: JobRecord) -> JobError | None:
        """None when a signal may go to the job's process: this wizard's
        own child, or a job found again whose recorded start time matches
        the live process's. Otherwise the refusal: the process id may be
        another program's (security review M1)."""
        if self._own(record):
            return None
        if record.start_time is None:
            why = "the job record holds no start time"
        else:
            actual = process_start_time(record.pid)
            if actual is not None and abs(actual - record.start_time) <= START_TIME_TOLERANCE_S:
                return None
            why = "its start time cannot be read"
        return JobError(STOP_UNCONFIRMED.format(pid=record.pid, verb=record.verb, why=why))

    def check_stop(self) -> None:
        """Refuse (``JobError``) to stop a running job whose process cannot
        be confirmed as the job's (``_stop_refusal``); nothing is sent."""
        with self._guard:
            status = self._status()
            if status.state == "running":
                refusal = self._stop_refusal(status.record)
                if refusal is not None:
                    raise refusal

    def _alive(self, record: JobRecord) -> bool:
        """Whether the job's process still runs: this wizard's own child by
        ``poll``; another's by its process id and start time, where a start
        time that cannot be checked counts as running (as the lock does)."""
        if self._own(record):
            return self._process.poll() is None
        if not process_alive(record.pid):
            return False
        actual = process_start_time(record.pid)
        if record.start_time is None or actual is None:
            return True
        return abs(actual - record.start_time) <= START_TIME_TOLERANCE_S

    def _status(self) -> JobStatus:
        record = self._current()
        state, exit_code, ended = "none", None, None
        if record is not None:
            end_path = self._job_file(record.end_record)
            end = None if end_path is None else read_end_record(end_path)
            if end is not None:
                state, exit_code, ended = "finished", end.exit_code, end.ended
            elif self._alive(record):
                state = "running"
            else:
                state = "ended_without_record"
                if self._own(record):
                    exit_code = self._process.poll()
        holder = None
        if state != "running":
            lock, info = lock_state(self.folder)
            if lock == "held":
                state, holder = "busy", info
        return JobStatus(state, record, exit_code, ended, holder)

    def status(self) -> JobStatus:
        """The job's state now; this wizard's own child is polled first."""
        with self._guard:
            return self._status()

    def running(self) -> bool:
        return self.status().state == "running"

    def start(self, verb: str, args: Sequence[str] = ()) -> JobRecord:
        """Start ``verb`` on the folder with ``args`` (built by the caller
        from checked values). Refused for a verb not in ``VERBS``, while a
        job runs, and while the folder lock is held."""
        if verb not in VERBS:
            raise JobError(f"{verb!r} cannot be run as a job; the verbs are {', '.join(VERBS)}")
        args = tuple(args)
        argv = job_argv(verb, self.folder, args)
        with self._guard:
            current = self._status()
            if current.state == "running":
                raise JobError(
                    f"a job is already running: {current.record.verb}, "
                    f"started {_clock(current.record.started)}"
                )
            state, info, _ = _read(self.folder, process_alive, process_start_time, time.time)
            if state in ("held", "starting"):
                raise _refusal(self.folder, info, process_alive, process_start_time)
            # Every path of the job stays inside the folder: no link is followed (security review H1).
            jobs_dir = checked_dir(self.folder, *JOBS_DIR)
            refuse_link(jobs_dir / CURRENT)
            refuse_not_file(jobs_dir / CURRENT)
            clear_leftover(_record_part(jobs_dir / CURRENT))
            moment = self._now()
            log_path, end_path = _new_names(jobs_dir, moment.strftime("%Y%m%d-%H%M%S"), verb)
            with open(log_path, "xb") as log:
                process = self._launcher(argv, env=job_env(end_path), cwd=self.folder, log=log)
            record = JobRecord(
                verb, args, process.pid, process_start_time(process.pid),
                log_path.relative_to(jobs_dir.parent.parent).as_posix(),
                end_path.relative_to(jobs_dir.parent.parent).as_posix(),
                moment.isoformat(timespec="seconds"),
            )
            _write_record(jobs_dir / CURRENT, record)
            self._process, self._record = process, record
            # Reap the child as soon as it ends, so it never lingers.
            threading.Thread(target=process.wait, name=f"job-{verb}", daemon=True).start()
            _prune_logs(jobs_dir)
            return record

    def read_log(self, offset: int) -> LogChunk:
        """The job log from byte ``offset``: at most ``LOG_CHUNK`` bytes,
        ending on a whole UTF-8 character."""
        if not _is_int(offset) or offset < 0:
            raise JobError(f"the log offset must be a whole number of 0 or more, got {offset!r}")
        with self._guard:
            record = self._current()
        path = None if record is None else self._job_file(record.log)
        if path is None:
            return LogChunk("", offset, None, "")
        try:
            with open_plain(path) as handle:
                handle.seek(offset)
                data = handle.read(LOG_CHUNK)
        except OSError:  # missing, or a link or FIFO planted at the log's name: never read
            data = b""
        cut = _whole_utf8(data)
        end = offset + cut
        return LogChunk(data[:cut].decode("utf-8", errors="replace"), end,
                        _latest_progress(path, end), _radiance_tail(self.folder, record.verb))

    def stop(self, *, sleep: Callable[[float], None] = time.sleep,
             monotonic: Callable[[], float] = time.monotonic) -> JobStatus:
        """Stop the running job, once its process id and start time are
        checked: SIGINT (the verb keeps what it finished and exits 130),
        SIGTERM after ``STOP_GRACE_S``, SIGKILL after ``KILL_GRACE_S`` more.
        Windows: ``taskkill /T``, then ``/F``. A job whose process cannot be
        confirmed as the job's is refused (``check_stop``)."""
        status = self.status()
        if status.state != "running":
            return status
        refusal = self._stop_refusal(status.record)
        if refusal is not None:
            raise refusal
        pid = status.record.pid
        steps = (
            (self._signaller.interrupt, STOP_GRACE_S),
            (self._signaller.terminate, KILL_GRACE_S),
            (self._signaller.kill, _KILL_WAIT_S),
        )
        for send, grace in steps:
            send(pid)
            deadline = monotonic() + grace
            while monotonic() < deadline:
                if not self.running():
                    return self.status()
                sleep(_STOP_POLL_S)
            if not self.running():
                break
        return self.status()
