# tests/test_engine_jobs_portable.py
"""Low-level file opens behave the same on every system (spec 2026-10-09 A1)."""
import dataclasses
import json
import os
import sys

import pytest

from conftest import symlink_or_skip
from leed_daylight.engine import jobs
from leed_daylight.engine.jobs import UnsafePath, append_private, open_plain, write_new

#: LF, CRLF and Ctrl-Z: the bytes Windows text mode would change or stop at.
AWKWARD = b"a\nb\r\n\x1ac"


def test_bytes_written_and_read_back_are_identical(tmp_path):
    path = tmp_path / "f.bin"
    write_new(path, AWKWARD)
    with open_plain(path) as handle:
        assert handle.read() == AWKWARD


def test_every_low_level_flag_set_is_binary(monkeypatch):
    for name in ("_NEW_FILE", "_READ_PLAIN", "_OPEN_LOCK"):
        assert getattr(jobs, name) & jobs._BINARY == jobs._BINARY, name


def test_appended_text_keeps_lf(tmp_path):
    path = tmp_path / "log.txt"
    append_private(path, "one\ntwo\n")
    assert path.read_bytes() == b"one\ntwo\n"


def test_windows_refuses_a_link_without_o_nofollow(tmp_path, monkeypatch):
    target = tmp_path / "real.txt"
    target.write_bytes(b"x")
    link = tmp_path / "link.txt"
    symlink_or_skip(link, target)
    monkeypatch.setattr(jobs, "_READ_PLAIN", os.O_RDONLY | jobs._BINARY)  # as on Windows: no O_NOFOLLOW
    with pytest.raises(UnsafePath, match="symbolic link"):
        open_plain(link, platform="win32")


def test_windows_refuses_a_link_that_appears_after_the_link_check(tmp_path, monkeypatch):
    target = tmp_path / "real.txt"
    target.write_bytes(b"x")
    link = tmp_path / "link.txt"
    symlink_or_skip(link, target)
    monkeypatch.setattr(jobs, "_READ_PLAIN", os.O_RDONLY | jobs._BINARY)  # as on Windows: no O_NOFOLLOW
    monkeypatch.setattr(jobs, "is_link", lambda path: False)  # the link is put there after the check
    with pytest.raises(UnsafePath, match="symbolic link"):
        open_plain(link, platform="win32")


def test_windows_refuses_a_file_swapped_in_between_the_open_and_the_check(tmp_path, monkeypatch):
    path, other = tmp_path / "f.txt", tmp_path / "other.txt"
    path.write_bytes(b"first")
    other.write_bytes(b"second")
    real_open, real_lstat, opened, swapped = os.open, os.lstat, [], []

    def open_then_swapped(name, flags, *args):
        fd = real_open(name, flags, *args)
        opened.append(fd)
        swapped.append(True)  # from here the name is taken to be another file
        return fd

    def lstat_of_the_swapped_name(name, *args, **kwargs):
        if swapped and os.fspath(name) == os.fspath(path):
            return real_lstat(other)  # Windows will not replace an open file: the swap is simulated
        return real_lstat(name, *args, **kwargs)

    monkeypatch.setattr(jobs.os, "open", open_then_swapped)
    monkeypatch.setattr(jobs.os, "lstat", lstat_of_the_swapped_name)
    with pytest.raises(UnsafePath, match="symbolic link"):
        open_plain(path, platform="win32")
    monkeypatch.undo()
    with pytest.raises(OSError):
        os.fstat(opened[0])  # closed before the refusal


def test_windows_refuses_a_folder_that_cannot_be_opened(tmp_path, monkeypatch):
    folder = tmp_path / "sub.csv"
    folder.mkdir()

    def refused(name, flags, *args):
        raise PermissionError(13, "Permission denied", str(name))  # what Windows gives for a folder

    monkeypatch.setattr(jobs.os, "open", refused)
    with pytest.raises(UnsafePath, match=r"sub\.csv: is not a plain file; it is not read"):
        open_plain(folder, platform="win32")
    with pytest.raises(PermissionError):
        open_plain(tmp_path / "absent", platform="win32")  # a real permission problem stays one
    with pytest.raises(PermissionError):
        open_plain(folder, platform="darwin")


def test_windows_release_tolerates_a_lock_file_another_reader_has_open(tmp_path, monkeypatch):
    info = jobs.acquire_lock(tmp_path, "prepare")
    lock = tmp_path / jobs.LOCK_FILE
    monkeypatch.setattr(jobs, "_unlock", lambda fd, platform=None: None)
    monkeypatch.setattr(jobs, "_LOCK_RETRY_S", 0.0)
    tries = []

    def busy(self, *args, **kwargs):
        tries.append(self)
        raise PermissionError(13, "the file is open in another process")

    monkeypatch.setattr(type(tmp_path), "unlink", busy)
    jobs.release_lock(tmp_path, info, platform="win32")  # no PermissionError

    assert len(tries) == jobs._LOCK_TRIES  # asked again while the reader holds it
    assert not any(key[1] == info for key in jobs._HELD)  # this process holds it no more
    assert lock.read_bytes() == b""  # the holder's text is gone: never read as held
    state, _ = jobs.lock_state(tmp_path)
    assert state == "starting"  # an empty young file; older, it is stale
    os.utime(lock, (lock.stat().st_mtime - jobs.YOUNG_LOCK_S - 1,) * 2)
    assert jobs.lock_state(tmp_path) == ("stale", None)
    monkeypatch.undo()
    again = jobs.acquire_lock(tmp_path, "run")  # the same process takes it over
    assert jobs.lock_state(tmp_path)[0] == "held"
    jobs.release_lock(tmp_path, again)
    assert not lock.exists()


def test_windows_release_deletes_the_lock_once_the_reader_lets_go(tmp_path, monkeypatch):
    info = jobs.acquire_lock(tmp_path, "prepare")
    lock = tmp_path / jobs.LOCK_FILE
    monkeypatch.setattr(jobs, "_unlock", lambda fd, platform=None: None)
    monkeypatch.setattr(jobs, "_LOCK_RETRY_S", 0.0)
    real_unlink = type(tmp_path).unlink
    tries = []

    def busy_once(self, *args, **kwargs):
        tries.append(self)
        if len(tries) == 1:
            raise PermissionError(13, "the file is open in another process")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "unlink", busy_once)
    jobs.release_lock(tmp_path, info, platform="win32")
    assert len(tries) == 2 and not lock.exists()
    assert jobs.lock_state(tmp_path) == ("free", None)


def test_windows_release_leaves_another_commands_lock(tmp_path, monkeypatch):
    info = jobs.acquire_lock(tmp_path, "prepare")
    lock = tmp_path / jobs.LOCK_FILE
    other = jobs.LockInfo(info.pid + 1, 0.0, "run", info.started)
    monkeypatch.setattr(jobs, "_unlock", lambda fd, platform=None: None)
    real_close = os.close

    def close_then_taken(fd):
        real_close(fd)
        if not lock.read_bytes():  # another command takes it over between unlock and delete
            lock.write_text(json.dumps(dataclasses.asdict(other)))

    monkeypatch.setattr(jobs.os, "close", close_then_taken)
    jobs.release_lock(tmp_path, info, platform="win32")
    monkeypatch.undo()
    assert jobs._parse(lock.read_bytes()) == other


# --- Windows only: real files another handle has open (never run on macOS) --

WINDOWS = pytest.mark.skipif(sys.platform != "win32", reason="Windows: files another handle has open")


@WINDOWS
def test_windows_a_lock_a_reader_has_open_is_released_and_taken_again(tmp_path):
    info = jobs.acquire_lock(tmp_path, "run")
    lock = tmp_path / jobs.LOCK_FILE
    reader = open(lock, "rb")
    try:
        jobs.release_lock(tmp_path, info)  # the file cannot be deleted while the reader has it open
    finally:
        reader.close()
    assert jobs.lock_state(tmp_path)[0] != "held"
    if lock.exists():  # left empty: starting while young, then stale
        assert lock.read_bytes() == b""
        os.utime(lock, (lock.stat().st_mtime - jobs.YOUNG_LOCK_S - 1,) * 2)
    again = jobs.acquire_lock(tmp_path, "prepare")
    assert jobs.lock_state(tmp_path) == ("held", again)
    jobs.release_lock(tmp_path, again)
    assert not lock.exists()


@WINDOWS
def test_windows_a_folder_is_not_a_plain_file(tmp_path):
    (tmp_path / "sub.csv").mkdir()
    with pytest.raises(UnsafePath, match="is not a plain file"):
        open_plain(tmp_path / "sub.csv")


@WINDOWS
def test_windows_a_junction_is_refused(tmp_path):
    import _winapi

    target_dir, link = tmp_path / "real", tmp_path / "link"
    target_dir.mkdir()
    (target_dir / "f").write_bytes(b"x")
    _winapi.CreateJunction(str(target_dir), str(link))
    with pytest.raises(UnsafePath, match="symbolic link"):
        open_plain(link)
