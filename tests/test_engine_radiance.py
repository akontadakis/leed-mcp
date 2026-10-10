# tests/test_engine_radiance.py
"""Tests for the Radiance process boundary (layer 4).

Every test here uses a FakeRunner -- never a real subprocess, never a
Radiance binary. The tests that exercise ``subprocess_runner()`` directly
(not through ``Radiance``) run only the running Python or a nonexistent
path, per the phase-4 ban on ever executing a Radiance binary, even
``-version``.
"""
from __future__ import annotations

import os
import pathlib
import re
import shlex
import sys

import pytest

from conftest import symlink_or_skip
from leed_daylight.engine.radiance import (
    REQUIRED_BINARIES,
    Radiance,
    RadianceError,
    RadianceNotFound,
    RunResult,
    find_binaries,
    subprocess_runner,
)
from stub_program import make_program


class FakeRunner:
    """Records every call; answers ``rtrace -version`` and everything else
    from configured, fixed results. Never touches a real process."""

    def __init__(self, *, version="fake_radiance 1.2.3", exit_code=0,
                 stdout=b"", stderr=b"", version_exit_code=0):
        self.calls: list[list[str]] = []
        self.version = version
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.version_exit_code = version_exit_code

    def run(self, argv, *, stdin=None, stdout_path=None, cwd=None):
        self.calls.append(list(argv))
        if "-version" in argv:
            if self.version_exit_code != 0:
                return RunResult(self.version_exit_code, b"", b"could not get version\n")
            return RunResult(0, self.version.encode("utf-8"), b"")
        if stdout_path is not None:
            pathlib.Path(stdout_path).write_bytes(self.stdout)
            return RunResult(self.exit_code, b"", self.stderr)
        return RunResult(self.exit_code, self.stdout, self.stderr)


#: Pins the log line's exit-status and wall-time fields exactly, so a test
#: can check the real values rather than a substring that a wall time could
#: satisfy by accident (e.g. "1" appearing inside "0.001s").
_LOG_LINE_RE = re.compile(r"-> exit (-?\d+) \(([0-9]+\.[0-9]+)s\)")


def _make_stub(directory: pathlib.Path, name: str) -> pathlib.Path:
    """A minimal program standing in for a Radiance binary."""
    return make_program(directory, name, "")


# --- the six required rows ---------------------------------------------


def test_every_command_reaches_the_log(tmp_path):
    log_path = tmp_path / "run.log"
    runner = FakeRunner()
    radiance = Radiance(runner, log_path)

    argv1 = ["oconv", "scene.rad"]
    argv2 = ["gendaymtx", "-m", "1", "weather.wea"]
    argv3 = ["rtrace", "-I", "octree"]
    radiance.run(argv1)
    radiance.run(argv2)
    radiance.run(argv3)

    lines = log_path.read_text().splitlines()
    # Line 0 is the session header, line 1 the version (M3), captured
    # before the first command is logged; the three run() calls produced
    # the three lines after those two.
    assert lines[0].startswith("=== Radiance session")
    command_lines = lines[2:]
    assert len(command_lines) == 3
    assert shlex.join(argv1) in command_lines[0]
    assert shlex.join(argv2) in command_lines[1]
    assert shlex.join(argv3) in command_lines[2]


def test_the_version_is_captured_once_and_logged(tmp_path):
    log_path = tmp_path / "run.log"
    runner = FakeRunner(version="fake_radiance 1.2.3 fictional build")
    radiance = Radiance(runner, log_path)

    assert radiance.version == "fake_radiance 1.2.3 fictional build"
    assert radiance.version == "fake_radiance 1.2.3 fictional build"
    radiance.run(["oconv", "scene.rad"])
    radiance.run(["gendaymtx", "-m", "1", "weather.wea"])

    version_calls = [call for call in runner.calls if "-version" in call]
    assert len(version_calls) == 1

    # Line 0 is the session header (M3); the version is the first line of
    # the session's own block, line 1.
    lines = log_path.read_text().splitlines()
    assert lines[0].startswith("=== Radiance session")
    assert "fake_radiance 1.2.3 fictional build" in lines[1]


def test_a_non_zero_exit_is_a_named_error(tmp_path):
    log_path = tmp_path / "run.log"
    stderr = b"error: bad octree\nmore detail\n"
    runner = FakeRunner(exit_code=1, stderr=stderr)
    radiance = Radiance(runner, log_path)

    argv = ["rcontrib", "-I", "bad_octree"]
    with pytest.raises(RadianceError) as excinfo:
        radiance.run(argv)

    err = excinfo.value
    assert not isinstance(err, RadianceNotFound)
    assert err.argv == tuple(argv)
    assert err.returncode == 1
    assert "bad octree" in err.stderr_tail
    assert shlex.join(argv) in str(err)
    assert "bad octree" in str(err)


def test_missing_binaries_are_all_named(tmp_path):
    _make_stub(tmp_path, "oconv")

    with pytest.raises(RadianceNotFound) as excinfo:
        find_binaries(search_path=str(tmp_path))

    message = str(excinfo.value)
    for name in REQUIRED_BINARIES:
        if name == "oconv":
            continue
        assert name in message
    assert str(tmp_path) in message


def test_all_binaries_present_is_accepted(tmp_path):
    stubs = {name: _make_stub(tmp_path, name) for name in REQUIRED_BINARIES}

    found = find_binaries(search_path=str(tmp_path))

    assert set(found.keys()) == set(REQUIRED_BINARIES)
    for name, path in found.items():
        assert path == stubs[name]
        assert path.exists()


def test_nothing_runs_without_a_runner():
    with pytest.raises(TypeError):
        Radiance()


# --- hostile inputs (contract step 9) -----------------------------------


def test_version_capture_failure_is_a_named_error(tmp_path):
    log_path = tmp_path / "run.log"
    runner = FakeRunner(version_exit_code=1)
    radiance = Radiance(runner, log_path)

    with pytest.raises(RadianceError, match="version"):
        radiance.run(["oconv", "scene.rad"])


def test_a_failed_version_probe_is_logged_before_the_error_is_raised(tmp_path):
    log_path = tmp_path / "run.log"
    runner = FakeRunner(version_exit_code=1)
    radiance = Radiance(runner, log_path)

    with pytest.raises(RadianceError, match="version"):
        radiance.run(["oconv", "scene.rad"])

    lines = log_path.read_text().splitlines()
    # Line 0 is the session header (M3); the failed version probe is line 1.
    assert len(lines) == 2
    assert lines[0].startswith("=== Radiance session")
    assert "rtrace" in lines[1]
    assert "-version" in lines[1]
    match = _LOG_LINE_RE.search(lines[1])
    assert match is not None, f"no exit-status/wall-time field in {lines[1]!r}"
    assert match.group(1) == "1"
    assert float(match.group(2)) >= 0.0


def test_a_bare_str_argv_is_refused(tmp_path):
    log_path = tmp_path / "run.log"
    radiance = Radiance(FakeRunner(), log_path)

    with pytest.raises(RadianceError, match="bare str"):
        radiance.run("oconv")

    # Refused before anything reaches the runner or the log.
    assert not log_path.exists()


def test_a_bare_bytes_argv_is_refused(tmp_path):
    log_path = tmp_path / "run.log"
    radiance = Radiance(FakeRunner(), log_path)

    with pytest.raises(RadianceError, match="bare bytes"):
        radiance.run(b"oconv")

    assert not log_path.exists()


def test_empty_argv_is_refused(tmp_path):
    log_path = tmp_path / "run.log"
    radiance = Radiance(FakeRunner(), log_path)

    with pytest.raises(RadianceError, match="argv"):
        radiance.run([])


def test_a_non_str_non_pathlike_argv_item_is_refused(tmp_path):
    log_path = tmp_path / "run.log"
    radiance = Radiance(FakeRunner(), log_path)

    with pytest.raises(RadianceError, match="oconv"):
        radiance.run(["oconv", 42])


def test_a_pathlike_argv_item_is_converted(tmp_path):
    log_path = tmp_path / "run.log"
    runner = FakeRunner()
    radiance = Radiance(runner, log_path)

    scene_path = tmp_path / "scene.rad"
    radiance.run(["oconv", scene_path])

    real_calls = [call for call in runner.calls if "-version" not in call]
    assert real_calls[0] == ["oconv", str(scene_path)]


def test_non_bytes_stdin_is_refused(tmp_path):
    log_path = tmp_path / "run.log"
    radiance = Radiance(FakeRunner(), log_path)

    with pytest.raises(RadianceError, match="stdin"):
        radiance.run(["oconv", "scene.rad"], stdin="not bytes")


def test_log_parent_directory_is_created(tmp_path):
    log_path = tmp_path / "nested" / "does" / "not" / "exist" / "run.log"
    radiance = Radiance(FakeRunner(), log_path)

    radiance.run(["oconv", "scene.rad"])

    assert log_path.exists()


def test_the_log_appends_across_radiance_instances_and_never_truncates(tmp_path):
    """M3: a resumed run builds a NEW ``Radiance`` instance over the SAME
    log path; the file must never be truncated, or the commands behind
    every matrix the resume reuses (rather than re-traces) would lose
    their audit-trail line."""
    log_path = tmp_path / "run.log"
    log_path.write_text("stale content from a previous session\n")
    radiance = Radiance(FakeRunner(), log_path)

    radiance.run(["oconv", "scene.rad"])
    content = log_path.read_text()
    assert "stale content from a previous session" in content
    first_write_lines = content.splitlines()

    # A brand-new instance (a resume) over the SAME log path must APPEND a
    # new session block, never discard the first one.
    radiance2 = Radiance(FakeRunner(), log_path)
    radiance2.run(["gendaymtx", "-m", "1", "weather.wea"])
    second_write_lines = log_path.read_text().splitlines()
    assert second_write_lines[: len(first_write_lines)] == first_write_lines
    assert len(second_write_lines) > len(first_write_lines)


def test_the_version_is_the_first_line_of_each_sessions_own_block(tmp_path):
    """M3: each ``Radiance`` instance writes its own session-header line
    followed by its own version line before its first command -- so a
    resumed run's version stays first within ITS block, even though the
    log file as a whole keeps growing across sessions."""
    log_path = tmp_path / "run.log"
    radiance1 = Radiance(FakeRunner(version="fake_radiance session one"), log_path)
    radiance1.run(["oconv", "scene.rad"])
    radiance2 = Radiance(FakeRunner(version="fake_radiance session two"), log_path)
    radiance2.run(["gendaymtx", "-m", "1", "weather.wea"])

    lines = log_path.read_text().splitlines()
    session_starts = [i for i, line in enumerate(lines) if line.startswith("=== Radiance session")]
    assert len(session_starts) == 2
    assert lines[session_starts[0] + 1] == "fake_radiance session one"
    assert lines[session_starts[1] + 1] == "fake_radiance session two"
    # The first session's own oconv line comes before the second session's
    # header -- the commands behind a resume's reused matrices are kept.
    assert any("oconv" in line for line in lines[: session_starts[1]])


def test_a_successful_commands_log_line_has_exact_exit_status_and_wall_time(tmp_path):
    log_path = tmp_path / "run.log"
    runner = FakeRunner(exit_code=0)
    radiance = Radiance(runner, log_path)

    radiance.run(["oconv", "scene.rad"])

    # Line 0 is the session header, line 1 the version; the command is the
    # line after those two.
    command_line = log_path.read_text().splitlines()[2]
    match = _LOG_LINE_RE.search(command_line)
    assert match is not None, f"no exit-status/wall-time field in {command_line!r}"
    assert match.group(1) == "0"
    assert float(match.group(2)) >= 0.0


def test_a_failed_command_is_logged_before_the_error_is_raised(tmp_path):
    log_path = tmp_path / "run.log"
    runner = FakeRunner(exit_code=1, stderr=b"boom\n")
    radiance = Radiance(runner, log_path)

    with pytest.raises(RadianceError):
        radiance.run(["rcontrib", "-I", "octree"])

    lines = log_path.read_text().splitlines()
    # Lines 0 and 1 are the session header and the version; the failed
    # command is the one after those two.
    command_lines = lines[2:]
    assert len(command_lines) == 1
    assert "rcontrib" in command_lines[0]
    match = _LOG_LINE_RE.search(command_lines[0])
    assert match is not None, f"no exit-status/wall-time field in {command_lines[0]!r}"
    assert match.group(1) == "1"
    assert float(match.group(2)) >= 0.0


def test_stderr_tail_is_last_20_lines(tmp_path):
    log_path = tmp_path / "run.log"
    all_lines = [f"stderr line {i}" for i in range(25)]
    stderr = ("\n".join(all_lines) + "\n").encode("utf-8")
    runner = FakeRunner(exit_code=1, stderr=stderr)
    radiance = Radiance(runner, log_path)

    with pytest.raises(RadianceError) as excinfo:
        radiance.run(["rcontrib", "-I", "octree"])

    tail = excinfo.value.stderr_tail
    tail_lines = tail.splitlines()
    assert len(tail_lines) == 20
    assert tail_lines == all_lines[-20:]
    assert "stderr line 0" not in tail


def test_stderr_tail_at_exactly_20_lines_is_kept_in_full(tmp_path):
    log_path = tmp_path / "run.log"
    all_lines = [f"stderr line {i}" for i in range(20)]
    stderr = ("\n".join(all_lines) + "\n").encode("utf-8")
    runner = FakeRunner(exit_code=1, stderr=stderr)
    radiance = Radiance(runner, log_path)

    with pytest.raises(RadianceError) as excinfo:
        radiance.run(["rcontrib", "-I", "octree"])

    tail_lines = excinfo.value.stderr_tail.splitlines()
    assert tail_lines == all_lines
    assert "stderr line 19" in excinfo.value.stderr_tail


# --- find_binaries edge behaviour ---------------------------------------


def test_find_binaries_with_no_stubs_names_all_seven(tmp_path):
    with pytest.raises(RadianceNotFound) as excinfo:
        find_binaries(search_path=str(tmp_path))

    message = str(excinfo.value)
    for name in REQUIRED_BINARIES:
        assert name in message


# --- subprocess_runner(): only the running Python, or a missing path


def test_subprocess_runner_runs_a_harmless_command(tmp_path):
    runner = subprocess_runner()
    result = runner.run([sys.executable, "-c", "print('hello')"])
    assert result.returncode == 0
    assert b"hello" in result.stdout


def test_subprocess_runner_reports_a_nonzero_exit(tmp_path):
    runner = subprocess_runner()
    result = runner.run([sys.executable, "-c", "raise SystemExit(1)"])
    assert result.returncode != 0


def test_subprocess_runner_reports_a_missing_executable_as_a_named_error(tmp_path):
    runner = subprocess_runner()
    missing = tmp_path / "does-not-exist-binary"
    with pytest.raises(RadianceError, match="does-not-exist-binary"):
        runner.run([str(missing)])


def test_subprocess_runner_writes_stdout_to_a_file(tmp_path):
    runner = subprocess_runner()
    out_path = tmp_path / "out.txt"
    result = runner.run([sys.executable, "-c", "print('to a file')"], stdout_path=out_path)
    assert result.stdout == b""
    assert out_path.read_bytes().strip() == b"to a file"


# --- find_binaries(names=...): checking a subset ------------------------


def test_find_binaries_checks_only_the_names_given(tmp_path):
    _make_stub(tmp_path, "oconv")
    _make_stub(tmp_path, "rtrace")

    found = find_binaries(search_path=str(tmp_path), names=("oconv", "rtrace"))

    assert set(found) == {"oconv", "rtrace"}


def test_find_binaries_subset_still_names_every_missing_one(tmp_path):
    _make_stub(tmp_path, "oconv")

    with pytest.raises(RadianceNotFound) as excinfo:
        find_binaries(search_path=str(tmp_path), names=("oconv", "rtrace", "rcontrib"))

    message = str(excinfo.value)
    assert "rtrace" in message and "rcontrib" in message
    assert "gendaymtx" not in message


# --- a process that never starts reaches the log ------------------------


class _NoStartRunner(FakeRunner):
    """Answers -version, then fails to start every other command the way
    subprocess_runner() does (a RadianceError, no exit status)."""

    def run(self, argv, *, stdin=None, stdout_path=None, cwd=None):
        if "-version" in argv:
            return super().run(argv, stdin=stdin, stdout_path=stdout_path, cwd=cwd)
        self.calls.append(list(argv))
        raise RadianceError(f"could not start {shlex.join(argv)}: no such file", argv=argv)


class _OSErrorRunner(FakeRunner):
    def run(self, argv, *, stdin=None, stdout_path=None, cwd=None):
        if "-version" in argv:
            return super().run(argv, stdin=stdin, stdout_path=stdout_path, cwd=cwd)
        raise FileNotFoundError(2, "No such file or directory")


@pytest.mark.parametrize("runner_cls", [_NoStartRunner, _OSErrorRunner])
def test_a_command_that_fails_to_start_is_logged(tmp_path, runner_cls):
    log_path = tmp_path / "run.log"
    radiance = Radiance(runner_cls(), log_path)
    argv = ["oconv", "scene.rad"]

    with pytest.raises(RadianceError):
        radiance.run(argv)

    last = log_path.read_text().splitlines()[-1]
    assert shlex.join(argv) in last
    assert "failed to start" in last


def test_a_version_probe_that_fails_to_start_is_logged(tmp_path):
    class _Dead(FakeRunner):
        def run(self, argv, *, stdin=None, stdout_path=None, cwd=None):
            raise RadianceError("could not start rtrace -version: gone", argv=argv)

    log_path = tmp_path / "run.log"
    radiance = Radiance(_Dead(), log_path)

    with pytest.raises(RadianceError):
        radiance.run(["oconv", "scene.rad"])

    last = log_path.read_text().splitlines()[-1]
    assert "rtrace -version" in last and "failed to start" in last


# --- RAYPATH: extra search directories for relative scene references ---


def test_subprocess_runner_prepends_search_dirs_to_raypath(tmp_path, monkeypatch):
    import subprocess as subprocess_module

    seen = {}

    def fake_run(argv, **kwargs):
        seen["env"] = kwargs.get("env")
        return subprocess_module.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr("leed_daylight.engine.radiance.subprocess.run", fake_run)
    monkeypatch.setenv("RAYPATH", "/opt/ray/lib")
    scene_dir, model_dir = tmp_path / "scene", tmp_path / "model"

    subprocess_runner(search_dirs=(scene_dir, model_dir)).run(["oconv", "x.rad"])

    assert seen["env"]["RAYPATH"] == os.pathsep.join(
        [str(scene_dir), str(model_dir), "/opt/ray/lib"]
    )
    assert seen["env"]["PATH"] == os.environ["PATH"]


def _capture_env(monkeypatch):
    import subprocess as subprocess_module

    seen = {}

    def fake_run(argv, **kwargs):
        seen["env"] = kwargs.get("env")
        return subprocess_module.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr("leed_daylight.engine.radiance.subprocess.run", fake_run)
    return seen


def _fake_install(tmp_path, monkeypatch, *, with_lib=True, standard_lib=False):
    """An invented Radiance install under tmp_path: <root>/bin/rtrace and
    (optionally) <root>/lib; `shutil.which` finds rtrace there only."""
    root = tmp_path / "radiance"
    (root / "bin").mkdir(parents=True)
    if with_lib:
        (root / "lib").mkdir()
        (root / "lib" / "rayinit.cal").write_text("{ invented }\n")
    standard = tmp_path / "usr_local_lib_ray"
    if standard_lib:
        standard.mkdir()
    monkeypatch.setattr("leed_daylight.engine.radiance.STANDARD_RAYLIBS",
                        {key: (standard,) for key in ("darwin", "linux", "win32")})

    def fake_which(name, mode=os.F_OK | os.X_OK, path=None):
        return str(root / "bin" / name) if name in ("rtrace", "oconv") else None

    monkeypatch.setattr("leed_daylight.engine.radiance.shutil.which", fake_which)
    monkeypatch.delenv("RAYPATH", raising=False)
    return root, standard


def test_with_no_raypath_the_library_is_found_next_to_the_binaries(tmp_path, monkeypatch):
    seen = _capture_env(monkeypatch)
    root, _ = _fake_install(tmp_path, monkeypatch)
    scene = tmp_path / "scene"

    subprocess_runner(search_dirs=(scene,)).run(["oconv", "x.rad"])

    # the install's own lib folder, never a /usr/local/lib/ray that is not there
    assert seen["env"]["RAYPATH"] == os.pathsep.join([str(scene), ".", str(root / "lib")])


def test_with_no_raypath_the_standard_library_is_kept_when_it_exists(tmp_path, monkeypatch):
    seen = _capture_env(monkeypatch)
    root, standard = _fake_install(tmp_path, monkeypatch, standard_lib=True)
    scene = tmp_path / "scene"

    subprocess_runner(search_dirs=(scene,)).run(["oconv", "x.rad"])

    assert seen["env"]["RAYPATH"] == os.pathsep.join(
        [str(scene), ".", str(root / "lib"), str(standard)]
    )


def test_with_no_raypath_a_symlinked_binary_leads_to_its_real_library(tmp_path, monkeypatch):
    # /usr/local/bin/rtrace -> /usr/local/radiance/bin/rtrace, in miniature:
    # the generic <prefix>/lib beside the symlink is not a Radiance library.
    seen = _capture_env(monkeypatch)
    root, _ = _fake_install(tmp_path, monkeypatch)
    (root / "bin" / "rtrace").write_text("")
    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib").mkdir()
    symlink_or_skip(prefix / "bin" / "rtrace", root / "bin" / "rtrace")
    monkeypatch.setattr(
        "leed_daylight.engine.radiance.shutil.which",
        lambda name, mode=None, path=None: str(prefix / "bin" / name) if name == "rtrace" else None,
    )
    scene = tmp_path / "scene"

    subprocess_runner(search_dirs=(scene,)).run(["oconv", "x.rad"])

    assert seen["env"]["RAYPATH"] == os.pathsep.join(
        [str(scene), ".", str((root / "lib").resolve())]
    )


def test_with_no_raypath_and_no_library_folder_only_the_current_folder_is_searched(
    tmp_path, monkeypatch,
):
    seen = _capture_env(monkeypatch)
    _fake_install(tmp_path, monkeypatch, with_lib=False)
    scene = tmp_path / "scene"

    subprocess_runner(search_dirs=(scene,)).run(["oconv", "x.rad"])

    assert seen["env"]["RAYPATH"] == os.pathsep.join([str(scene), "."])


def test_with_no_library_folder_the_radiance_log_warns(tmp_path, monkeypatch):
    # RAYPATH, once set, replaces the library path compiled into the
    # binaries; with no library found the scene folders must still go on
    # it, so the run log says the compiled-in library is no longer searched.
    _capture_env(monkeypatch)
    _fake_install(tmp_path, monkeypatch, with_lib=False)
    runner = subprocess_runner(search_dirs=(tmp_path / "scene",))
    log = tmp_path / "radiance.log"
    rad = Radiance(runner, log)

    rad.run(["oconv", "x.rad"])

    lines = log.read_text().splitlines()
    assert lines[0] == "=== Radiance session ==="
    (warning,) = [line for line in lines if "WARNING" in line]
    assert "Radiance library not found next to rtrace" in warning
    assert "RAYPATH set to scene folders only" in warning
    assert lines.index(warning) == 2  # after the version line


@pytest.mark.parametrize("case", ["library found", "raypath set", "no search dirs"])
def test_the_library_warning_is_not_written_otherwise(tmp_path, monkeypatch, case):
    _capture_env(monkeypatch)
    _fake_install(tmp_path, monkeypatch, with_lib=(case == "library found"))
    if case == "raypath set":
        monkeypatch.setenv("RAYPATH", "/opt/ray/lib")
    dirs = () if case == "no search dirs" else (tmp_path / "scene",)
    log = tmp_path / "radiance.log"

    Radiance(subprocess_runner(search_dirs=dirs), log).run(["oconv", "x.rad"])

    assert "WARNING" not in log.read_text()


def test_subprocess_runner_without_search_dirs_leaves_the_env_alone(monkeypatch):
    import subprocess as subprocess_module

    seen = {}

    def fake_run(argv, **kwargs):
        seen["env"] = kwargs.get("env")
        return subprocess_module.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr("leed_daylight.engine.radiance.subprocess.run", fake_run)

    subprocess_runner().run(["oconv", "x.rad"])

    assert seen["env"] is None


def test_a_child_without_input_never_reads_the_parent_s_stdin():
    """leed-mcp: the parent's stdin is the JSON-RPC stream, so a child run
    without input gets the null device. fd 0 is made a pipe holding a line
    for the length of the call: a child that inherited it would read
    b'leak\\n'; with the null device it reads b''."""
    import os
    import sys
    from leed_daylight.engine import radiance
    read_end, write_end = os.pipe()
    os.write(write_end, b"leak\n")
    os.close(write_end)
    saved = os.dup(0)
    os.dup2(read_end, 0)
    try:
        result = radiance.subprocess_runner().run(
            [sys.executable, "-c", "import sys; print(repr(sys.stdin.buffer.readline()))"])
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        os.close(read_end)
    assert result.returncode == 0
    assert result.stdout.strip() == b"b''"


def test_windows_runs_radiance_with_one_process(capsys, monkeypatch):
    from leed_daylight.engine import radiance

    monkeypatch.setattr(radiance, "_TOLD_ONE_PROCESS", False)
    assert radiance.process_count(8, platform="win32") == 1
    assert radiance.process_count(8, platform="win32") == 1
    err = capsys.readouterr().err
    assert err.count("Radiance runs with one process on Windows") == 1
    assert radiance.process_count(8, platform="darwin") == 8
    assert radiance.process_count(8, platform="linux") == 8


def test_the_library_fallbacks_are_per_system(monkeypatch, tmp_path):
    from leed_daylight.engine import radiance

    lib = tmp_path / "Radiance" / "lib"
    lib.mkdir(parents=True)
    monkeypatch.setitem(radiance.STANDARD_RAYLIBS, "win32", (lib,))
    raypath = radiance.default_raypath(search_path=str(tmp_path / "empty"), platform="win32")
    assert raypath.split(os.pathsep) == [".", str(lib)]


def test_the_coefficient_command_has_no_n_on_windows():
    from leed_daylight.engine.coefficients import _processes_args

    assert _processes_args(4, platform="win32") == []
    assert _processes_args(4, platform="linux") == ["-n", "4"]
