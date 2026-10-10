# leed_daylight/engine/filelock.py
"""The kernel lock on an open file, one way per system (spec 2026-10-09 A1).

macOS and Linux: ``flock``. Windows: ``msvcrt.locking`` on one byte far past
any text, so the text stays readable to whoever is refused. The kernel frees
the lock when its holder's process ends, however it ends (security review L6).
Used by the folder lock (``engine.jobs``) and the job index (``leed_mcp.jobindex``).
"""
from __future__ import annotations

import contextlib
import os
import sys

__all__ = ["WINDOWS_LOCK_OFFSET", "try_lock", "unlock"]

#: Windows: the byte of the lock file that ``msvcrt.locking`` locks, far past its text.
WINDOWS_LOCK_OFFSET = 2 ** 30


def try_lock(fd: int, *, shared: bool = False, platform: str | None = None) -> bool:
    """Take the kernel lock of the open file ``fd`` without waiting: True
    when taken, False when another open file holds it. Windows has no shared
    lock; ``shared`` takes the same exclusive one there."""
    if (platform or sys.platform) == "win32":
        import msvcrt

        os.lseek(fd, WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def unlock(fd: int, *, platform: str | None = None) -> None:
    """Give back the kernel lock ``try_lock`` took on ``fd``."""
    if (platform or sys.platform) == "win32":
        import msvcrt

        os.lseek(fd, WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        with contextlib.suppress(OSError):
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)
