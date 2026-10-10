# leed_daylight/engine/radiance.py
"""The Radiance process boundary: the only module that shells out (layer 4).

Nothing in this module knows what a window group, a room or a LEED point
is; it runs Radiance command lines. ``Radiance`` writes every command it
runs -- argv, exit status and wall time -- to a run log, so a submission's
audit trail exists even when nothing above this layer asks for it. The
Radiance version is captured once, from ``rtrace -version``, on first use,
and is always the log's first line, because a report needs to say which
Radiance built its numbers.

A non-zero exit never becomes a silent empty result: it is a
``RadianceError`` carrying the failing command line and the tail of its
stderr. A command that never starts is logged too (``-> failed to start``)
before its ``RadianceError`` is raised. ``find_binaries`` refuses once, naming every missing binary at
once, rather than letting a person chase one missing tool at a time.

The real process runner (``subprocess_runner``) lives here too, but tests
never call it against a Radiance binary; they use it only against harmless
system commands (``/bin/echo``, ``/usr/bin/false``, a nonexistent path).
Production code injects a ``Runner`` into ``Radiance``; there is no
default, because injection is the design, not a convenience.
"""
from __future__ import annotations

import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

#: Every external binary a run of this tool may need. Order is the order
#: ``find_binaries`` reports a missing one in.
REQUIRED_BINARIES: tuple[str, ...] = (
    "oconv", "gendaymtx", "gensky", "rcontrib", "rfluxmtx", "rtrace", "xform",
)

#: How many trailing lines of stderr a RadianceError keeps.
_STDERR_TAIL_LINES = 20

#: The library folders of standard installs, per system, used only when
#: they exist (:func:`default_raypath`).
STANDARD_RAYLIBS: dict[str, tuple[pathlib.Path, ...]] = {
    "darwin": (pathlib.Path("/usr/local/lib/ray"),),
    "linux": (pathlib.Path("/usr/local/lib/ray"), pathlib.Path("/usr/local/radiance/lib")),
    "win32": (pathlib.Path(r"C:\Radiance\lib"),),
}

#: Whether the one-process notice was printed in this process already.
_TOLD_ONE_PROCESS = False


def process_count(asked: int, *, platform: str | None = None) -> int:
    """How many processes ``rtrace``/``rcontrib`` may use: ``asked``, but
    one on Windows, whose Radiance builds cannot split a run over processes.
    The first cut in a process is said once on stderr (the job's log)."""
    global _TOLD_ONE_PROCESS
    if (platform or sys.platform) == "win32" and asked > 1:
        if not _TOLD_ONE_PROCESS:
            print(f"note: Radiance runs with one process on Windows (asked for {asked})", file=sys.stderr)
            _TOLD_ONE_PROCESS = True
        return 1
    return asked

#: The binaries whose install folder locates Radiance's own library, in
#: the order they are looked for (:func:`default_raypath`).
_LIBRARY_PROBES: tuple[str, ...] = ("rtrace", "oconv")

#: A file every Radiance library folder holds (the function-file start-up
#: definitions); a folder without it is not a Radiance library.
_LIBRARY_MARKER = "rayinit.cal"


class RadianceError(RuntimeError):
    """A Radiance command line could not be run, or exited non-zero.

    Carries the argv it tried to run, the exit code (``None`` when the
    process never started), and the last ``_STDERR_TAIL_LINES`` lines of
    stderr, so a caller can act on the failure without re-parsing the
    message string.
    """

    def __init__(self, message: str, *, argv: Sequence[object] = (),
                 returncode: int | None = None, stderr_tail: str = "") -> None:
        super().__init__(message)
        self.argv: tuple[str, ...] = tuple(str(item) for item in argv)
        self.returncode = returncode
        self.stderr_tail = stderr_tail


class RadianceNotFound(RadianceError):
    """One or more binaries in ``REQUIRED_BINARIES`` are not on the search path."""


@dataclass(frozen=True)
class RunResult:
    """What a ``Runner`` hands back for one command."""

    returncode: int
    stdout: bytes  # b"" when stdout_path was given (the runner wrote it there)
    stderr: bytes


class Runner(Protocol):
    def run(self, argv: Sequence[str], *, stdin: bytes | None = None,
            stdout_path: pathlib.Path | None = None, cwd: pathlib.Path | None = None
            ) -> RunResult: ...


def _stderr_tail(stderr: bytes) -> str:
    """The last _STDERR_TAIL_LINES lines of stderr, decoded leniently."""
    text = stderr.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-_STDERR_TAIL_LINES:])


def find_binaries(
    search_path: str | None = None, names: Sequence[str] = REQUIRED_BINARIES,
) -> dict[str, pathlib.Path]:
    """Locate every binary in ``names`` (default ``REQUIRED_BINARIES``).

    ``search_path`` is a ``PATH``-style string; ``None`` means the
    process's own ``PATH``. ``names`` lets a caller that needs only some
    of the tools (e.g. ``("oconv", "rtrace")``) check just those. Refuses
    with ``RadianceNotFound`` naming every missing binary, in ``names``
    order, plus the path that was searched -- never one at a time.
    """
    found: dict[str, pathlib.Path] = {}
    missing: list[str] = []
    for name in names:
        located = shutil.which(name, path=search_path)
        if located is None:
            missing.append(name)
        else:
            found[name] = pathlib.Path(located)
    if missing:
        searched = search_path if search_path is not None else os.environ.get("PATH", "")
        raise RadianceNotFound(
            # Quoted, not repr: a Windows path would show doubled backslashes.
            f"Radiance binaries not found on search path '{searched}': "
            f"{', '.join(missing)}",
            argv=tuple(missing),
        )
    return found


def default_raypath(search_path: str | None = None, *, platform: str | None = None) -> str:
    """Radiance's library search path for when ``RAYPATH`` is not set.

    Setting ``RAYPATH`` replaces the library path compiled into the
    binaries, so it must name the library the binaries themselves would
    use: ``<folder of rtrace (or oconv)>/../lib`` (or ``../lib/ray``),
    e.g. ``/usr/local/radiance/lib``, following a symlinked binary to its
    real install first; a candidate counts only if it holds
    ``_LIBRARY_MARKER``, so a generic ``/usr/local/lib`` next to a
    symlink in ``/usr/local/bin`` is never taken. Then the standard library folders of this system,
    only those that exist -- never a folder that is not there. Always starts
    with ``.``. ``search_path`` is the ``PATH`` the binaries are looked up
    on (``None``: the process's own).
    """
    parts = ["."]
    for name in _LIBRARY_PROBES:
        located = shutil.which(name, path=search_path)
        if located is None:
            continue
        binary = pathlib.Path(located)
        prefixes = dict.fromkeys([binary.resolve().parent.parent, binary.parent.parent])
        candidates = [p / sub for p in prefixes for sub in ("lib", "lib/ray")]
        for candidate in candidates:
            if (candidate / _LIBRARY_MARKER).is_file():
                parts.append(os.fspath(candidate))
                break
        break
    for folder in STANDARD_RAYLIBS.get(platform or sys.platform, STANDARD_RAYLIBS["linux"]):
        if folder.is_dir() and os.fspath(folder) not in parts:
            parts.append(os.fspath(folder))
    return os.pathsep.join(parts)


#: The run-log line written when ``RAYPATH`` had to be set with no
#: Radiance library on it (:func:`raypath_env`).
NO_LIBRARY_WARNING = (
    "WARNING: Radiance library not found next to rtrace; RAYPATH set to "
    "scene folders only (the library path compiled into Radiance is no "
    "longer searched -- set RAYPATH to include it)"
)


def raypath_env(
    search_dirs: Sequence[os.PathLike | str], base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """A copy of ``base_env`` (default ``os.environ``) whose ``RAYPATH``
    starts with ``search_dirs``, followed by the existing ``RAYPATH`` or,
    when none is set, the library path of the Radiance install found on
    ``PATH`` (:func:`default_raypath`). When that finds no library (only
    ``.``), ``RAYPATH`` is still set -- the scene folders must be on it --
    and :func:`raypath_warnings` says so."""
    env = dict(os.environ if base_env is None else base_env)
    existing = env.get("RAYPATH") or default_raypath(env.get("PATH"))
    env["RAYPATH"] = os.pathsep.join([*(os.fspath(d) for d in search_dirs), existing])
    return env


def raypath_warnings(
    search_dirs: Sequence[os.PathLike | str], base_env: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """``(NO_LIBRARY_WARNING,)`` when :func:`raypath_env` would set
    ``RAYPATH`` without any Radiance library on it: ``search_dirs`` given,
    ``RAYPATH`` unset, and :func:`default_raypath` found no library.
    Setting ``RAYPATH`` replaces the compiled-in library path, so a
    function file such as ``rayinit.cal`` may then not be found."""
    env = os.environ if base_env is None else base_env
    if not search_dirs or env.get("RAYPATH"):
        return ()
    if default_raypath(env.get("PATH")) == ".":
        return (NO_LIBRARY_WARNING,)
    return ()


def _stdin_kwargs(stdin: bytes | None) -> dict:
    """``input=`` when there is input, else the null device: a child never
    inherits the parent's stdin (leed-mcp: that is the JSON-RPC stream).
    ``subprocess.run`` refuses ``input`` and ``stdin`` together."""
    return {"input": stdin} if stdin is not None else {"stdin": subprocess.DEVNULL}


def subprocess_runner(search_dirs: Sequence[os.PathLike | str] = ()) -> Runner:
    """The real ``Runner``, over ``subprocess.run``.

    No shell, ``check=False``, stderr always captured; stdout goes to
    ``stdout_path`` when given, otherwise it is captured too. An
    executable that cannot even be started (``OSError``, typically "no
    such file") reaches the caller as a ``RadianceError`` naming the argv,
    never a raw ``FileNotFoundError``.

    ``search_dirs`` are prepended to ``RAYPATH`` in every child's
    environment (:func:`raypath_env`). Commands run in the process's own
    working directory, so a scene's relative ``.cal``, ``.xml`` or data
    file reference resolves only through ``RAYPATH``; the caller passes the
    scene files' folder(s) and the prepared model folder here. With no
    ``search_dirs`` the child inherits the environment unchanged.

    The runner's ``warnings`` (:func:`raypath_warnings`) are written to the
    run log by ``Radiance``, once per session, after the version line.
    """
    env = raypath_env(search_dirs) if search_dirs else None
    setup_warnings = raypath_warnings(search_dirs)

    class _SubprocessRunner:
        warnings: tuple[str, ...] = setup_warnings

        def run(self, argv: Sequence[str], *, stdin: bytes | None = None,
                 stdout_path: pathlib.Path | None = None,
                 cwd: pathlib.Path | None = None) -> RunResult:
            plain_argv = [os.fspath(item) if isinstance(item, os.PathLike) else item
                          for item in argv]
            try:
                if stdout_path is not None:
                    with open(stdout_path, "wb") as stdout_file:
                        completed = subprocess.run(
                            plain_argv, **_stdin_kwargs(stdin), stdout=stdout_file,
                            stderr=subprocess.PIPE, cwd=cwd, check=False,
                            shell=False, env=env,
                        )
                    stdout_bytes = b""
                else:
                    completed = subprocess.run(
                        plain_argv, **_stdin_kwargs(stdin), stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, cwd=cwd, check=False,
                        shell=False, env=env,
                    )
                    stdout_bytes = completed.stdout or b""
            except OSError as exc:
                raise RadianceError(
                    f"could not start {shlex.join(plain_argv)}: {exc}",
                    argv=plain_argv,
                ) from exc
            return RunResult(
                returncode=completed.returncode,
                stdout=stdout_bytes,
                stderr=completed.stderr or b"",
            )

    return _SubprocessRunner()


#: The line written once per ``Radiance`` instance, right before its
#: version line (M3): a resumed run creates a NEW ``Radiance`` instance
#: over the SAME log path, and the log must APPEND rather than truncate,
#: so this marks where one session's commands end and the next begins.
_SESSION_HEADER = "=== Radiance session ==="


class Radiance:
    """Runs Radiance command lines through an injected ``Runner``, logging
    every one of them and converting a non-zero exit into a named error.

    The log file is never truncated (M3): a resumed run builds a NEW
    ``Radiance`` instance over the SAME log path, and every command behind
    a matrix this instance reuses (rather than re-traces) already has its
    audit-trail line in there from an earlier session -- truncating on
    first write would silently erase it. Each instance instead APPENDS a
    session-header line, then its own version line, before its first
    command line, so the version stays the first line of every session's
    own block even though the file as a whole keeps growing.
    """

    def __init__(self, runner: Runner, log_path: pathlib.Path) -> None:
        self._runner = runner
        self._log_path = pathlib.Path(log_path)
        self._version: str | None = None
        self._log_started = False

    @property
    def version(self) -> str:
        """The Radiance version, from ``rtrace -version``, captured once."""
        if self._version is None:
            self._version = self._capture_version()
        return self._version

    def _capture_version(self) -> str:
        argv = ["rtrace", "-version"]
        start = time.monotonic()
        try:
            result = self._runner.run(argv, stdin=None, stdout_path=None, cwd=None)
        except RadianceError:
            self._log_failed_start(argv, start)
            raise
        except OSError as exc:
            self._log_failed_start(argv, start)
            raise RadianceError(
                f"could not run {shlex.join(argv)}: {exc}", argv=argv,
            ) from exc
        wall_time = time.monotonic() - start
        if result.returncode != 0:
            tail = _stderr_tail(result.stderr)
            # The probe ran (it produced an exit status), so it is logged
            # like any other command before the error it raises -- the
            # same "never a silent failure" rule that applies to run().
            self._write_log_line(
                f"{shlex.join(argv)} -> exit {result.returncode} "
                f"({wall_time:.3f}s)"
            )
            raise RadianceError(
                f"{shlex.join(argv)} exited {result.returncode}: {tail}",
                argv=argv, returncode=result.returncode, stderr_tail=tail,
            )
        version = result.stdout.decode("utf-8", errors="replace").strip()
        self._write_log_line(version)
        return version

    def run(self, argv: Sequence[object], *, stdin: bytes | None = None,
            stdout_path: pathlib.Path | None = None,
            cwd: pathlib.Path | None = None) -> bytes:
        """Run one Radiance command line.

        Returns stdout (``b""`` when ``stdout_path`` was given). Raises
        ``RadianceError`` on a non-zero exit, carrying the argv and the
        stderr tail. The version is captured (and logged) before the first
        command line is logged, even if ``.version`` was never read
        directly.
        """
        clean_argv = self._normalise_argv(argv)
        if stdin is not None and not isinstance(stdin, (bytes, bytearray)):
            raise RadianceError(
                f"stdin must be bytes or None, got {type(stdin).__name__}",
                argv=clean_argv,
            )

        _ = self.version  # ensures the version line precedes every command line

        start = time.monotonic()
        try:
            result = self._runner.run(
                clean_argv, stdin=stdin, stdout_path=stdout_path, cwd=cwd,
            )
        except RadianceError:
            # subprocess_runner() turns a process that never started into a
            # RadianceError; it still belongs in the audit trail.
            self._log_failed_start(clean_argv, start)
            raise
        except OSError as exc:
            self._log_failed_start(clean_argv, start)
            raise RadianceError(
                f"could not run {shlex.join(clean_argv)}: {exc}", argv=clean_argv,
            ) from exc
        wall_time = time.monotonic() - start

        self._write_log_line(
            f"{shlex.join(clean_argv)} -> exit {result.returncode} "
            f"({wall_time:.3f}s)"
        )

        if result.returncode != 0:
            tail = _stderr_tail(result.stderr)
            raise RadianceError(
                f"{shlex.join(clean_argv)} exited {result.returncode}: {tail}",
                argv=clean_argv, returncode=result.returncode, stderr_tail=tail,
            )
        return result.stdout

    def _normalise_argv(self, argv: Sequence[object]) -> tuple[str, ...]:
        if isinstance(argv, (str, bytes)):
            # A bare str/bytes is itself a Sequence[str]/Sequence[int]: left
            # unchecked, list("oconv") silently becomes ['o', 'c', 'o', 'n',
            # 'v'] instead of refusing the caller's mistake.
            kind = "str" if isinstance(argv, str) else "bytes"
            raise RadianceError(
                f"argv must be a sequence of strings, not a bare {kind}: "
                f"{argv!r} (wrap it in a list, e.g. ['oconv', ...])",
                argv=(),
            )
        items = list(argv) if argv is not None else []
        if not items:
            raise RadianceError("argv must not be empty", argv=())
        normalised: list[str] = []
        for item in items:
            if isinstance(item, str):
                normalised.append(item)
            elif isinstance(item, os.PathLike):
                normalised.append(os.fspath(item))
            else:
                raise RadianceError(
                    "argv items must be str or os.PathLike, got "
                    f"{type(item).__name__}: {item!r} in {items!r}",
                    argv=tuple(str(i) for i in items),
                )
        return tuple(normalised)

    def _log_failed_start(self, argv: Sequence[str], start: float) -> None:
        """The log line for a command that never started (no exit status)."""
        wall_time = time.monotonic() - start
        self._write_log_line(f"{shlex.join(argv)} -> failed to start ({wall_time:.3f}s)")

    def _write_log_line(self, text: str) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8", newline="\n") as handle:
            first = not self._log_started
            if first:
                self._log_started = True
                handle.write(_SESSION_HEADER + "\n")
            handle.write(text + "\n")
            if first:
                # A runner's own set-up warnings (e.g. the RAYPATH one),
                # after the session's first line -- the version line.
                for warning in getattr(self._runner, "warnings", ()):
                    handle.write(warning + "\n")
