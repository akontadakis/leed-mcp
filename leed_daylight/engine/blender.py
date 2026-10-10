# leed_daylight/engine/blender.py
"""The Blender process boundary (atlas spec 12): finding and starting Blender.

Like ``radiance.py``, this module knows nothing about rooms, storeys or
results; it finds a Blender, reads its version and runs one build script
in the background. ``deliver/atlas`` decides what the script builds.

Search order: the ``BLENDER`` environment variable (when set it must name
a file; nothing else is tried, so the tests can rule a real Blender out),
then ``blender`` on ``PATH``, then the standard install folders of
macOS, Windows or Linux. Accepted versions: 4.2 and newer, read from the first line of
``blender --version`` that matches ``Blender <major>.<minor>[.<patch>]``.

The build runs as ``<exe> -b --factory-startup --python-exit-code 1
--python <script> -- <bundle> <out>.part``. Without ``--python-exit-code
1`` Blender exits 0 when the script fails. Blender's output goes to the
log. The script saves to ``<out>.part``; only after a zero exit with that
file present is it renamed to ``<out>``, so a half-written file never
carries the final name. A non-zero exit, a timeout, or a zero exit with no
``.part`` file is a :class:`BlenderFailed` naming the log.

Tests inject a fake ``BlenderRunner``; the real one
(``subprocess_blender_runner``) is used by tests only on harmless system
commands.
"""
from __future__ import annotations

import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Protocol

#: The oldest Blender the build script supports (``surface_render_method``).
MIN_VERSION: tuple[int, int] = (4, 2)
#: Where the macOS installer puts Blender.
MAC_DEFAULT = pathlib.Path("/Applications/Blender.app/Contents/MacOS/Blender")
#: Where Linux installs put Blender when it is not on ``PATH``: the snap
#: first, then the unpacked archives under ``/opt``.
LINUX_ROOTS: tuple[pathlib.Path, ...] = (pathlib.Path("/snap/bin"), pathlib.Path("/opt"))
#: Seconds one build may take before it is stopped.
TIMEOUT_S = 600
#: Seconds ``blender --version`` may take.
_VERSION_TIMEOUT_S = 60

_VERSION_LINE = re.compile(r"Blender (\d+)\.(\d+)(?:\.(\d+))?")
_FOLDER_VERSION = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")


class BlenderNotFound(Exception):
    """No Blender at any place searched."""


class BlenderTooOld(Exception):
    """The Blender found is older than ``MIN_VERSION``, or its version could not be read."""


class BlenderFailed(Exception):
    """A build ran and did not produce the file; ``log_path`` holds Blender's output."""

    def __init__(self, message: str, *, log_path: pathlib.Path) -> None:
        super().__init__(message)
        self.log_path = pathlib.Path(log_path)


class BlenderRunner(Protocol):
    def run(self, argv: Sequence[str], *, log_path: pathlib.Path, timeout_s: float) -> int:
        """Run ``argv`` with its output (stdout and stderr) written to
        ``log_path``; the exit code. Raises ``TimeoutError`` after
        ``timeout_s`` seconds, ``OSError`` when it cannot start."""
        ...


def _folder_version(path: pathlib.Path) -> tuple[int, int, int]:
    """The version in the name of ``path``'s folder (``Blender 4.4``,
    ``blender-4.2.3-linux-x64``) as numbers, so 4.10 sorts after 4.9;
    (0, 0, 0) when it has none."""
    match = _FOLDER_VERSION.search(path.parent.name)
    if not match:
        return (0, 0, 0)
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def _newest(paths) -> list[pathlib.Path]:
    return sorted((p for p in paths if p.is_file()), key=_folder_version, reverse=True)


def _program_files(environ: Mapping[str, str]) -> pathlib.Path:
    return pathlib.Path(environ.get("ProgramFiles", r"C:\Program Files")) / "Blender Foundation"


def system_blenders(platform: str, environ: Mapping[str, str], *,
                    linux_roots: tuple[pathlib.Path, ...] = LINUX_ROOTS,
                    mac_default: pathlib.Path = MAC_DEFAULT) -> list[pathlib.Path]:
    """The Blender programs in the system folders of ``platform``, best
    first: the newest version by folder name. Only files that exist."""
    if platform == "darwin":
        return [pathlib.Path(mac_default)] if pathlib.Path(mac_default).is_file() else []
    if platform == "win32":
        base = _program_files(environ)
        return _newest(base.glob("Blender */blender.exe")) if base.is_dir() else []
    snap, *others = linux_roots
    found = [snap / "blender"] if (snap / "blender").is_file() else []
    for root in others:
        if root.is_dir():
            found += _newest(root.glob("blender*/blender"))
    return found


def _system_places(platform: str, environ: Mapping[str, str], linux_roots, mac_default) -> str:
    """Where ``system_blenders`` looked, for the refusal."""
    if platform == "darwin":
        return f"none at {mac_default}"
    if platform == "win32":
        return f"none under {_program_files(environ)}"
    return "none in " + ", ".join(str(root) for root in linux_roots)


def find_blender(environ: Mapping[str, str] | None = None, *, which=shutil.which,
                 mac_default: pathlib.Path = MAC_DEFAULT, platform: str = sys.platform,
                 linux_roots: tuple[pathlib.Path, ...] = LINUX_ROOTS) -> pathlib.Path:
    """The Blender executable: ``BLENDER``, then ``blender`` on ``PATH``,
    then the system folders (:func:`system_blenders`). ``BLENDER`` set to a path that is not a
    file is refused naming it; nothing else is tried then."""
    environ = os.environ if environ is None else environ
    chosen = environ.get("BLENDER", "")
    if chosen:
        path = pathlib.Path(chosen).expanduser()
        if not path.is_file():
            raise BlenderNotFound(f"BLENDER is set to {chosen}, which is not a file; "
                                  "set it to the Blender program, or unset it")
        return path
    located = which("blender")
    if located:
        return pathlib.Path(located)
    places = ["BLENDER is not set", "no 'blender' on PATH"]
    found = system_blenders(platform, environ, linux_roots=linux_roots, mac_default=mac_default)
    if found:
        return found[0]
    places.append(_system_places(platform, environ, linux_roots, mac_default))
    raise BlenderNotFound(f"no Blender found ({', '.join(places)}); install Blender "
                          f"{MIN_VERSION[0]}.{MIN_VERSION[1]} or newer, or set BLENDER to its program")


def parse_version(text: str) -> tuple[int, int, int] | None:
    """(major, minor, patch) from the first line of ``text`` that matches
    ``Blender <major>.<minor>[.<patch>]``; patch 0 when absent; ``None``
    when no line matches."""
    for line in text.splitlines():
        match = _VERSION_LINE.search(line)
        if match:
            major, minor, patch = match.groups()
            return int(major), int(minor), int(patch or 0)
    return None


def subprocess_blender_runner() -> BlenderRunner:
    """The real ``BlenderRunner``, over ``subprocess.run``: no shell, no
    input, stdout and stderr both into the log (truncated first); the
    process is killed when it runs past ``timeout_s``."""

    class _SubprocessBlenderRunner:
        def run(self, argv: Sequence[str], *, log_path: pathlib.Path, timeout_s: float) -> int:
            plain = [os.fspath(item) for item in argv]
            log_path = pathlib.Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "wb") as log:
                try:
                    completed = subprocess.run(
                        plain, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        timeout=timeout_s, check=False, shell=False,
                    )
                except subprocess.TimeoutExpired:
                    raise TimeoutError(f"{shlex.join(plain)} ran longer than {timeout_s:g} s") from None
            return completed.returncode

    return _SubprocessBlenderRunner()


def _version_text(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def check_version(exe, runner: BlenderRunner) -> tuple[int, int, int]:
    """The version of ``exe`` from ``exe --version``. Refuses with
    :class:`BlenderTooOld` a version older than ``MIN_VERSION``, and a
    version that cannot be read (no matching line, a non-zero exit, a
    program that cannot start or does not answer)."""
    minimum = _version_text(MIN_VERSION)
    with tempfile.TemporaryDirectory() as tmp:
        log = pathlib.Path(tmp) / "version.log"
        try:
            code = runner.run([os.fspath(exe), "--version"], log_path=log, timeout_s=_VERSION_TIMEOUT_S)
        except (OSError, TimeoutError) as exc:
            raise BlenderTooOld(f"could not read the version of {exe} ({exc}); "
                                f"Blender {minimum} or newer is needed") from None
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    version = parse_version(text) if code == 0 else None
    if version is None:
        why = f"it exited {code}" if code != 0 else "no 'Blender <version>' line in its output"
        raise BlenderTooOld(f"could not read the version of {exe} ({why}); "
                            f"Blender {minimum} or newer is needed")
    if version[:2] < MIN_VERSION:
        raise BlenderTooOld(f"Blender {_version_text(version)} at {exe} is older than {minimum}; "
                            f"install Blender {minimum} or newer")
    return version


def run_blender(exe, script: pathlib.Path, bundle: pathlib.Path, out: pathlib.Path, *,
                log_path: pathlib.Path, runner: BlenderRunner | None = None,
                timeout_s: float = TIMEOUT_S) -> None:
    """Run ``script`` in a background Blender on ``bundle``; the script
    saves ``<out>.part``, renamed here to ``out`` on success. Raises
    :class:`BlenderFailed` on a non-zero exit, a timeout, a Blender that
    cannot start, or a zero exit with no ``.part`` file."""
    runner = subprocess_blender_runner() if runner is None else runner
    out = pathlib.Path(out)
    part = out.with_name(out.name + ".part")
    part.unlink(missing_ok=True)
    argv = [os.fspath(exe), "-b", "--factory-startup", "--python-exit-code", "1",
            "--python", os.fspath(script), "--", os.fspath(bundle), os.fspath(part)]
    try:
        code = runner.run(argv, log_path=pathlib.Path(log_path), timeout_s=timeout_s)
    except TimeoutError:
        part.unlink(missing_ok=True)
        raise BlenderFailed(f"Blender did not finish within {timeout_s:g} s and was stopped; "
                            f"its output is in {log_path}", log_path=log_path) from None
    except OSError as exc:
        raise BlenderFailed(f"could not start {exe} ({exc})", log_path=log_path) from None
    if code != 0:
        part.unlink(missing_ok=True)
        raise BlenderFailed(f"Blender exited {code}; its output is in {log_path}", log_path=log_path)
    if not part.is_file():
        raise BlenderFailed(f"Blender exited 0 but saved no {part.name}; its output is in {log_path}",
                            log_path=log_path)
    os.replace(part, out)
