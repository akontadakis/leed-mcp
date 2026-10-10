# tests/test_engine_filelock.py
"""The shared kernel lock (spec 2026-10-09 A1)."""
import os

from leed_daylight.engine import filelock, jobs


def test_a_second_open_file_cannot_take_a_held_lock(tmp_path):
    path = tmp_path / "x.lock"
    first = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    second = os.open(path, os.O_RDWR)
    try:
        assert filelock.try_lock(first)
        assert not filelock.try_lock(second)
        filelock.unlock(first)
        assert filelock.try_lock(second)
        filelock.unlock(second)
    finally:
        os.close(first)
        os.close(second)


def test_the_job_lock_uses_the_shared_one():
    assert jobs._try_lock is filelock.try_lock
    assert jobs._unlock is filelock.unlock
