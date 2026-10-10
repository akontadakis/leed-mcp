# tests/test_engine_jobs_lock.py
"""The folder lock (wizard spec 8.2): atomic creation, the holder check by
process id and start time, stale locks cleared by one clearer only.

Every folder is a fresh ``tmp_path``. Liveness and start times are injected
where a test needs a process that is dead or reused; the real ``ps`` is
asked only about this test's own process.
"""
from __future__ import annotations

import io
import json
import os
import re
import time

import pytest

from leed_daylight.engine import jobs
from leed_daylight.engine.filelock import try_lock, unlock
from leed_daylight.engine.jobs import (
    LOCK_FILE,
    START_TIME_TOLERANCE_S,
    YOUNG_LOCK_S,
    LockHeld,
    LockInfo,
    acquire_lock,
    folder_lock,
    lock_message,
    lock_state,
    process_alive,
    process_start_time,
    release_lock,
)

#: Above the largest process id of macOS (99998) and Linux (4194304).
DEAD_PID = 2 ** 22 + 1
ME = os.getpid()


def _write_lock(folder, info: LockInfo) -> None:
    (folder / LOCK_FILE).write_text(json.dumps(
        {"pid": info.pid, "start_time": info.start_time, "verb": info.verb, "started": info.started}
    ))


def _only_me_alive(pid: int) -> bool:
    return pid == ME


def _leftovers(folder) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.name.startswith(LOCK_FILE + "."))


def test_a_free_folder_locks_and_unlocks(tmp_path):
    info = acquire_lock(tmp_path, "run")

    raw = json.loads((tmp_path / LOCK_FILE).read_text())
    assert set(raw) == {"pid", "start_time", "verb", "started"}
    assert raw == {"pid": ME, "start_time": info.start_time, "verb": "run", "started": info.started}
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", info.started)
    assert lock_state(tmp_path) == ("held", info)

    release_lock(tmp_path, info)

    assert not (tmp_path / LOCK_FILE).exists()
    assert lock_state(tmp_path) == ("free", None)


def test_the_lock_file_sits_in_the_folder_never_in_run(tmp_path):
    with folder_lock(tmp_path, "option2"):
        assert (tmp_path / LOCK_FILE).is_file()
        assert not (tmp_path / "_run").exists()


def test_the_start_time_of_this_process_is_what_the_lock_records(tmp_path):
    info = acquire_lock(tmp_path, "run")

    own = process_start_time(ME)
    assert own is not None
    assert abs(own - info.start_time) <= START_TIME_TOLERANCE_S
    assert own <= time.time()
    release_lock(tmp_path, info)


def test_a_process_that_does_not_exist_has_no_start_time_and_is_not_alive():
    assert process_start_time(DEAD_PID) is None
    assert not process_alive(DEAD_PID)
    assert not process_alive(0)
    assert not process_alive(-1)
    assert process_alive(ME)


def test_a_stale_lock_is_cleared_with_a_note(tmp_path):
    _write_lock(tmp_path, LockInfo(DEAD_PID, 1000.0, "option2", "2026-01-01T09:30:00"))
    err = io.StringIO()

    info = acquire_lock(tmp_path, "run", err=err)

    assert err.getvalue() == (
        f"note: cleared a stale lock left by option2 (process {DEAD_PID}, started 09:30)\n"
    )
    assert info.pid == ME and info.verb == "run"
    assert json.loads((tmp_path / LOCK_FILE).read_text())["verb"] == "run"
    assert _leftovers(tmp_path) == []


def test_a_reused_process_id_is_not_trusted(tmp_path):
    own = process_start_time(ME)
    _write_lock(tmp_path, LockInfo(ME, own - 10.0, "views", "2026-01-01T08:00:00"))
    err = io.StringIO()

    assert lock_state(tmp_path)[0] == "stale"
    info = acquire_lock(tmp_path, "report", err=err)

    assert info.verb == "report"
    assert err.getvalue().startswith(f"note: cleared a stale lock left by views (process {ME}, ")


def test_a_start_time_within_the_tolerance_is_the_holder(tmp_path):
    own = process_start_time(ME)
    held = LockInfo(ME, own - 1.0, "views", "2026-01-01T08:00:00")
    _write_lock(tmp_path, held)

    with pytest.raises(LockHeld) as caught:
        acquire_lock(tmp_path, "report")

    assert str(caught.value) == f"views is already working on this folder (started 08:00, process {ME})"
    assert lock_state(tmp_path) == ("held", held)


def _no_start_time(pid: int) -> None:
    return None


def test_a_live_holder_whose_start_time_cannot_be_read_is_held(tmp_path):
    held = LockInfo(ME, 1000.0, "run", "2026-01-01T14:02:00")
    _write_lock(tmp_path, held)
    before = (tmp_path / LOCK_FILE).read_bytes()
    err = io.StringIO()

    with pytest.raises(LockHeld) as caught:
        acquire_lock(tmp_path, "report", err=err, start_time=_no_start_time)

    assert str(caught.value) == (
        f"run is already working on this folder (started 14:02, process {ME}); process {ME} "
        "exists but its start time could not be checked, so the lock is kept; if process "
        f"{ME} is not a leed-daylight command, delete {tmp_path / LOCK_FILE} by hand"
    )
    assert lock_state(tmp_path, start_time=_no_start_time) == ("held", held)
    assert (tmp_path / LOCK_FILE).read_bytes() == before
    assert err.getvalue() == ""
    assert _leftovers(tmp_path) == []


def test_a_live_holder_with_no_recorded_start_time_is_held(tmp_path):
    held = LockInfo(ME, 0.0, "views", "2026-01-01T08:00:00")
    _write_lock(tmp_path, held)

    with pytest.raises(LockHeld, match=r"its start time could not be checked, so the lock is kept"):
        acquire_lock(tmp_path, "report")

    assert lock_state(tmp_path) == ("held", held)


def test_a_dead_holder_is_stale_even_when_start_times_cannot_be_read(tmp_path):
    _write_lock(tmp_path, LockInfo(DEAD_PID, 0.0, "run", "2026-01-01T09:30:00"))
    err = io.StringIO()

    info = acquire_lock(tmp_path, "run", err=err, start_time=_no_start_time)

    assert info.start_time == 0.0
    assert err.getvalue().startswith(f"note: cleared a stale lock left by run (process {DEAD_PID}, ")
    with pytest.raises(LockHeld, match="could not be checked"):
        acquire_lock(tmp_path, "report")


def test_two_clearers_racing_cannot_both_win(tmp_path):
    _write_lock(tmp_path, LockInfo(DEAD_PID, 1000.0, "run", "2026-01-01T09:30:00"))
    err = io.StringIO()
    won = []

    def second_clearer():
        won.append(acquire_lock(tmp_path, "report", err=err, alive=_only_me_alive))

    with pytest.raises(LockHeld) as caught:
        acquire_lock(tmp_path, "package", err=err, on_stale=second_clearer, alive=_only_me_alive)

    assert len(won) == 1 and won[0].verb == "report"
    assert str(caught.value) == lock_message(won[0])
    assert json.loads((tmp_path / LOCK_FILE).read_text())["verb"] == "report"
    assert err.getvalue().count("note: cleared a stale lock") == 1
    assert _leftovers(tmp_path) == []


def test_a_lock_cleared_by_another_meanwhile_is_taken(tmp_path):
    _write_lock(tmp_path, LockInfo(DEAD_PID, 1000.0, "run", "2026-01-01T09:30:00"))
    err = io.StringIO()

    def other_clears():
        (tmp_path / LOCK_FILE).unlink()

    info = acquire_lock(tmp_path, "atlas", err=err, on_stale=other_clears, alive=_only_me_alive)

    assert info.verb == "atlas"
    assert err.getvalue() == ""
    assert _leftovers(tmp_path) == []


def test_an_unreadable_young_lock_is_held_and_an_old_one_is_stale(tmp_path):
    (tmp_path / LOCK_FILE).write_text("")

    with pytest.raises(LockHeld) as caught:
        acquire_lock(tmp_path, "run")
    assert str(caught.value) == "another command is starting on this folder; try again in a few seconds"
    assert lock_state(tmp_path) == ("starting", None)

    old = time.time() - YOUNG_LOCK_S - 5
    os.utime(tmp_path / LOCK_FILE, (old, old))
    assert lock_state(tmp_path) == ("stale", None)
    err = io.StringIO()
    info = acquire_lock(tmp_path, "run", err=err)

    assert err.getvalue() == "note: cleared a stale lock that could not be read\n"
    assert lock_state(tmp_path) == ("held", info)


def test_lock_state_reads_the_age_from_the_clock_given(tmp_path):
    (tmp_path / LOCK_FILE).write_text("{\"pid\": ")
    modified = (tmp_path / LOCK_FILE).stat().st_mtime

    assert lock_state(tmp_path, now=lambda: modified + 1.0) == ("starting", None)
    assert lock_state(tmp_path, now=lambda: modified + YOUNG_LOCK_S + 1.0) == ("stale", None)


def test_release_lock_leaves_another_holders_lock(tmp_path):
    info = acquire_lock(tmp_path, "run")
    other = LockInfo(ME, info.start_time, "report", "2026-01-01T10:00:00")
    _write_lock(tmp_path, other)

    release_lock(tmp_path, info)

    assert lock_state(tmp_path) == ("held", other)
    release_lock(tmp_path, other)
    assert lock_state(tmp_path) == ("free", None)


def test_the_lock_is_not_re_entrant(tmp_path):
    with folder_lock(tmp_path, "run") as info:
        with pytest.raises(LockHeld, match=r"^run is already working on this folder \(started \d{2}:\d{2}, process \d+\)$"):
            acquire_lock(tmp_path, "run")
        assert lock_state(tmp_path) == ("held", info)
    assert lock_state(tmp_path) == ("free", None)


def test_a_lock_is_released_after_keyboard_interrupt(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with folder_lock(tmp_path, "run"):
            raise KeyboardInterrupt

    assert not (tmp_path / LOCK_FILE).exists()


def test_lock_held_is_a_value_error():
    assert issubclass(LockHeld, ValueError)
    assert jobs.LOCK_FILE == ".leed-daylight.lock"
    assert lock_message(LockInfo(51234, 0.0, "run", "2026-01-01T14:02:09")) == (
        "run is already working on this folder (started 14:02, process 51234)"
    )


# --- security review L6: the kernel holds the lock (flock), re-review OOS-4 --

def _kernel_held(path) -> bool:
    """Whether some open file holds the kernel lock of ``path`` (``flock``,
    or ``msvcrt.locking`` on Windows)."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        if not try_lock(fd):
            return True
        unlock(fd)
    finally:
        os.close(fd)
    return False


def _no_rename(*args, **kwargs):
    pytest.fail("a lock was renamed aside")


def test_a_taken_lock_is_held_by_the_kernel_and_freed_on_release(tmp_path):
    info = acquire_lock(tmp_path, "run")
    assert _kernel_held(tmp_path / LOCK_FILE)

    release_lock(tmp_path, info)

    assert not (tmp_path / LOCK_FILE).exists()


def test_a_lock_the_kernel_holds_is_never_cleared_whatever_its_text(tmp_path):
    # Another command holds the kernel lock; its file still names a dead process.
    _write_lock(tmp_path, LockInfo(DEAD_PID, 1000.0, "run", "2026-01-01T09:30:00"))
    before = (tmp_path / LOCK_FILE).read_bytes()
    holder = os.open(tmp_path / LOCK_FILE, os.O_RDWR | getattr(os, "O_BINARY", 0))
    assert try_lock(holder)
    err = io.StringIO()
    try:
        with pytest.raises(LockHeld) as caught:
            acquire_lock(tmp_path, "report", err=err)
        assert str(caught.value) == "another command is starting on this folder; try again in a few seconds"
        assert lock_state(tmp_path)[0] == "starting"
    finally:
        os.close(holder)
    assert (tmp_path / LOCK_FILE).read_bytes() == before
    assert err.getvalue() == ""
    assert _leftovers(tmp_path) == []


def test_a_stale_lock_is_taken_over_in_place_never_renamed(tmp_path, monkeypatch):
    _write_lock(tmp_path, LockInfo(DEAD_PID, 1000.0, "option2", "2026-01-01T09:30:00"))
    inode = os.lstat(tmp_path / LOCK_FILE).st_ino
    monkeypatch.setattr(jobs.os, "rename", _no_rename)
    err = io.StringIO()

    info = acquire_lock(tmp_path, "run", err=err)

    assert err.getvalue() == f"note: cleared a stale lock left by option2 (process {DEAD_PID}, started 09:30)\n"
    assert os.lstat(tmp_path / LOCK_FILE).st_ino == inode
    assert json.loads((tmp_path / LOCK_FILE).read_text())["verb"] == "run"
    assert _kernel_held(tmp_path / LOCK_FILE)
    assert lock_state(tmp_path) == ("held", info)
    release_lock(tmp_path, info)
    assert _leftovers(tmp_path) == [] and not (tmp_path / LOCK_FILE).exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this system")
def test_a_planted_fifo_is_cleared_without_a_rename(tmp_path, monkeypatch):
    os.mkfifo(tmp_path / LOCK_FILE)
    os.utime(tmp_path / LOCK_FILE, (0, 0))
    monkeypatch.setattr(jobs.os, "rename", _no_rename)
    err = io.StringIO()

    info = acquire_lock(tmp_path, "run", err=err)

    assert err.getvalue() == "note: cleared a stale lock that could not be read\n"
    assert lock_state(tmp_path) == ("held", info)
    release_lock(tmp_path, info)
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize("age", [0, YOUNG_LOCK_S + 60], ids=["young", "old"])
def test_a_folder_at_the_lock_is_refused_with_one_plain_line(tmp_path, age):
    (tmp_path / LOCK_FILE).mkdir()
    (tmp_path / LOCK_FILE / "kept.txt").write_text("kept\n")
    old = time.time() - age
    os.utime(tmp_path / LOCK_FILE, (old, old))

    with pytest.raises(LockHeld) as caught:
        acquire_lock(tmp_path, "run")

    assert str(caught.value) == (
        f"{tmp_path / LOCK_FILE}: is a folder, not a lock file; move it away and try again"
    )
    assert (tmp_path / LOCK_FILE / "kept.txt").read_text() == "kept\n"
    assert _leftovers(tmp_path) == []
