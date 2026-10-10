# tests/test_engine_blender.py
"""Finding and starting Blender (atlas spec 12, 15).

Every test that runs Blender uses ``FakeRunner``: it records the command
line, writes the text a Blender would print into the log, and (when told
to) writes the file the build script would save. No test here starts a
real Blender. The three tests of ``subprocess_blender_runner`` itself run
only the running Python. The version text of the real Blender 5.1.1 (its
first lines) is copied below with invented build dates.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

from conftest import NO_BLENDER
from leed_daylight.engine.blender import (
    MAC_DEFAULT,
    MIN_VERSION,
    TIMEOUT_S,
    BlenderFailed,
    BlenderNotFound,
    BlenderTooOld,
    check_version,
    find_blender,
    parse_version,
    run_blender,
    subprocess_blender_runner,
)

#: The first lines ``Blender --version`` prints (5.1.1; dates invented).
VERSION_5_1_1 = (
    "Blender 5.1.1\n"
    "\tbuild date: 2000-01-01\n"
    "\tbuild time: 00:00:00\n"
    "\tbuild branch: blender-v5.1-release\n"
    "\tbuild platform: Darwin\n"
)


class FakeRunner:
    """Records every call. ``--version`` prints ``version``; any other
    command prints ``output``, writes the ``.part`` file (the last
    argument) when ``save`` is set, and returns ``code``, or raises
    ``TimeoutError`` when ``timeout`` is set."""

    def __init__(self, *, version=VERSION_5_1_1, version_code=0, code=0, save=True, timeout=False,
                 output="building\n"):
        self.calls: list[tuple[list[str], pathlib.Path, float]] = []
        self.version, self.version_code = version, version_code
        self.code, self.save, self.timeout, self.output = code, save, timeout, output

    def run(self, argv, *, log_path, timeout_s):
        argv = [str(a) for a in argv]
        self.calls.append((argv, pathlib.Path(log_path), timeout_s))
        if argv[1:] == ["--version"]:
            pathlib.Path(log_path).write_text(self.version)
            return self.version_code
        pathlib.Path(log_path).write_text(self.output)
        if self.timeout:
            raise TimeoutError("ran too long")
        if self.save:
            pathlib.Path(argv[-1]).write_bytes(b"BLENDER-v501")
        return self.code


def _executable(path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


# --- find_blender -----------------------------------------------------------

def test_the_blender_variable_wins(tmp_path):
    exe = _executable(tmp_path / "chosen" / "blender")
    found = find_blender({"BLENDER": str(exe)}, which=lambda name: "/elsewhere/blender",
                         mac_default=_executable(tmp_path / "mac" / "Blender"), platform="darwin")
    assert found == exe


def test_the_blender_variable_set_to_a_missing_path_is_refused_by_name(tmp_path):
    missing = tmp_path / "nowhere" / "blender"
    with pytest.raises(BlenderNotFound, match="BLENDER") as info:
        find_blender({"BLENDER": str(missing)}, which=lambda name: "/elsewhere/blender",
                     mac_default=_executable(tmp_path / "mac" / "Blender"), platform="darwin")
    assert str(missing) in str(info.value)


def test_then_blender_on_the_path(tmp_path):
    exe = _executable(tmp_path / "bin" / "blender")
    seen = []

    def which(name):
        seen.append(name)
        return str(exe)

    assert find_blender({}, which=which, mac_default=tmp_path / "none", platform="linux") == exe
    assert seen == ["blender"]


def test_then_the_mac_default_on_a_mac_only(tmp_path):
    mac = _executable(tmp_path / "Blender.app" / "Contents" / "MacOS" / "Blender")
    assert find_blender({}, which=lambda name: None, mac_default=mac, platform="darwin") == mac
    with pytest.raises(BlenderNotFound):
        find_blender({}, which=lambda name: None, mac_default=mac, platform="linux",
                     linux_roots=(tmp_path / "none",))


def test_not_found_names_every_place_looked(tmp_path):
    mac = tmp_path / "Blender.app" / "Contents" / "MacOS" / "Blender"
    with pytest.raises(BlenderNotFound) as info:
        find_blender({}, which=lambda name: None, mac_default=mac, platform="darwin")
    text = str(info.value)
    assert "BLENDER" in text and "PATH" in text and str(mac) in text


def test_an_empty_blender_variable_counts_as_unset(tmp_path):
    exe = _executable(tmp_path / "bin" / "blender")
    assert find_blender({"BLENDER": ""}, which=lambda name: str(exe), platform="linux") == exe


def test_the_defaults():
    assert MIN_VERSION == (4, 2)
    assert TIMEOUT_S == 600
    assert MAC_DEFAULT == pathlib.Path("/Applications/Blender.app/Contents/MacOS/Blender")


def test_under_the_test_guard_no_blender_is_found():
    # Review Focus 3: tests/conftest.py sets BLENDER to a path that does
    # not exist, so the finder stops there, whatever the machine has; also
    # with LEED_BLENDER_SMOKE=1, which lifts the guard for the smoke test
    # only (final review M2).
    with pytest.raises(BlenderNotFound) as info:
        find_blender()
    assert str(NO_BLENDER) in str(info.value)


# --- parse_version and check_version ------------------------------------------

@pytest.mark.parametrize("text, expected", [
    (VERSION_5_1_1, (5, 1, 1)),
    ("Blender 4.2.0 LTS\n", (4, 2, 0)),
    ("Blender 4.2\n", (4, 2, 0)),
    ("Read prefs: /somewhere/userpref.blend\nBlender 4.5.3\n", (4, 5, 3)),
    ("Blender 3.6.9\nBlender 5.0.0\n", (3, 6, 9)),
    ("no version here\n", None),
    ("", None),
])
def test_parse_version(text, expected):
    assert parse_version(text) == expected


def test_check_version_reads_the_real_first_line(tmp_path):
    runner = FakeRunner()
    exe = tmp_path / "blender"
    assert check_version(exe, runner) == (5, 1, 1)
    (argv, _, _), = runner.calls
    assert argv == [str(exe), "--version"]


def test_an_older_blender_is_refused_with_both_versions(tmp_path):
    with pytest.raises(BlenderTooOld) as info:
        check_version(tmp_path / "blender", FakeRunner(version="Blender 4.1.1\n"))
    assert "4.1.1" in str(info.value) and "4.2" in str(info.value)


@pytest.mark.parametrize("runner", [
    FakeRunner(version="Something else 9.9\n"),
    FakeRunner(version_code=1),
], ids=["unreadable", "non-zero exit"])
def test_an_unreadable_version_counts_as_too_old(tmp_path, runner):
    with pytest.raises(BlenderTooOld, match="could not read"):
        check_version(tmp_path / "blender", runner)


def test_a_version_check_that_hangs_counts_as_unreadable(tmp_path):
    class Hangs:
        def run(self, argv, *, log_path, timeout_s):
            raise TimeoutError("ran too long")

    with pytest.raises(BlenderTooOld, match="could not read"):
        check_version(tmp_path / "blender", Hangs())


# --- run_blender ----------------------------------------------------------------

def _paths(tmp_path):
    return (tmp_path / "blender", tmp_path / "blender_build.py", tmp_path / "bundle.npz",
            tmp_path / "out" / "model.blend", tmp_path / "out" / "blender.log")


def test_success_renames_the_part_file(tmp_path):
    exe, script, bundle, out, log = _paths(tmp_path)
    out.parent.mkdir()
    runner = FakeRunner()
    run_blender(exe, script, bundle, out, log_path=log, runner=runner)
    (argv, log_path, timeout_s), = runner.calls
    assert argv == [str(exe), "-b", "--factory-startup", "--python-exit-code", "1",
                    "--python", str(script), "--", str(bundle), str(out) + ".part"]
    assert log_path == log and timeout_s == TIMEOUT_S
    assert out.read_bytes() == b"BLENDER-v501"
    assert sorted(p.name for p in out.parent.iterdir()) == ["blender.log", "model.blend"]


def test_an_old_part_file_is_removed_first(tmp_path):
    exe, script, bundle, out, log = _paths(tmp_path)
    out.parent.mkdir()
    (out.parent / "model.blend.part").write_bytes(b"left over")
    with pytest.raises(BlenderFailed):
        run_blender(exe, script, bundle, out, log_path=log, runner=FakeRunner(save=False))
    assert not (out.parent / "model.blend.part").exists()


def test_a_non_zero_exit_fails_naming_the_log(tmp_path):
    exe, script, bundle, out, log = _paths(tmp_path)
    out.parent.mkdir()
    with pytest.raises(BlenderFailed) as info:
        run_blender(exe, script, bundle, out, log_path=log, runner=FakeRunner(code=1))
    assert info.value.log_path == log
    assert "exited 1" in str(info.value) and str(log) in str(info.value)
    assert sorted(p.name for p in out.parent.iterdir()) == ["blender.log"]


def test_a_timeout_fails_naming_the_log(tmp_path):
    exe, script, bundle, out, log = _paths(tmp_path)
    out.parent.mkdir()
    runner = FakeRunner(timeout=True)
    with pytest.raises(BlenderFailed) as info:
        run_blender(exe, script, bundle, out, log_path=log, runner=runner, timeout_s=5)
    assert info.value.log_path == log
    assert "5 s" in str(info.value) and str(log) in str(info.value)
    assert runner.calls[0][2] == 5
    assert not out.exists()


def test_a_zero_exit_with_no_saved_file_is_a_failure(tmp_path):
    exe, script, bundle, out, log = _paths(tmp_path)
    out.parent.mkdir()
    with pytest.raises(BlenderFailed, match="model.blend.part") as info:
        run_blender(exe, script, bundle, out, log_path=log, runner=FakeRunner(save=False))
    assert info.value.log_path == log
    assert not out.exists()


def test_a_blender_that_cannot_start_is_a_failure(tmp_path):
    exe, script, bundle, out, log = _paths(tmp_path)
    out.parent.mkdir()

    class CannotStart:
        def run(self, argv, *, log_path, timeout_s):
            raise FileNotFoundError(2, "No such file or directory", str(argv[0]))

    with pytest.raises(BlenderFailed, match="could not start"):
        run_blender(exe, script, bundle, out, log_path=log, runner=CannotStart())


# --- subprocess_blender_runner: only the running Python ----------------------

def test_the_real_runner_writes_the_output_to_the_log(tmp_path):
    log = tmp_path / "logs" / "blender.log"
    argv = [sys.executable, "-c", "print('Blender 5.1.1')"]
    code = subprocess_blender_runner().run(argv, log_path=log, timeout_s=10)
    assert code == 0
    assert log.read_text() == "Blender 5.1.1\n"


def test_the_real_runner_returns_a_non_zero_exit(tmp_path):
    argv = [sys.executable, "-c", "raise SystemExit(1)"]
    assert subprocess_blender_runner().run(argv, log_path=tmp_path / "b.log", timeout_s=10) == 1


def test_the_real_runner_stops_a_process_that_runs_too_long(tmp_path):
    argv = [sys.executable, "-c", "import time; time.sleep(5)"]
    with pytest.raises(TimeoutError):
        subprocess_blender_runner().run(argv, log_path=tmp_path / "b.log", timeout_s=0.2)


def _exe(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_windows_picks_the_newest_blender_under_program_files(tmp_path):
    base = tmp_path / "Program Files" / "Blender Foundation"
    _exe(base / "Blender 4.1" / "blender.exe")
    _exe(base / "Blender 4.9" / "blender.exe")
    newest = _exe(base / "Blender 4.10" / "blender.exe")
    found = find_blender({"ProgramFiles": str(tmp_path / "Program Files")}, which=lambda name: None,
                         platform="win32")
    assert found == newest


def test_windows_with_no_blender_names_where_it_looked(tmp_path):
    with pytest.raises(BlenderNotFound, match="Blender Foundation"):
        find_blender({"ProgramFiles": str(tmp_path)}, which=lambda name: None, platform="win32")


def test_linux_finds_blender_in_opt_when_not_on_path(tmp_path):
    snap, opt = tmp_path / "snap" / "bin", tmp_path / "opt"
    _exe(opt / "blender-4.2.3-linux-x64" / "blender")
    newest = _exe(opt / "blender-4.4.0-linux-x64" / "blender")
    found = find_blender({}, which=lambda name: None, platform="linux", linux_roots=(snap, opt))
    assert found == newest


def test_linux_prefers_the_snap(tmp_path):
    snap, opt = tmp_path / "snap" / "bin", tmp_path / "opt"
    snapped = _exe(snap / "blender")
    _exe(opt / "blender-4.4.0-linux-x64" / "blender")
    assert find_blender({}, which=lambda name: None, platform="linux", linux_roots=(snap, opt)) == snapped
