# leed_daylight/cli.py
"""The ``leed-daylight`` command (layer 7, CLI spec 3): one command, one verb
per step.

    leed-daylight <verb> [arguments]
    python -m leed_daylight <verb> [arguments]

A thin dispatcher (spec decision C5): it reads only the verb, imports that
verb's module from ``drivers/`` and returns
``module.main(rest, prog="leed-daylight <verb>")``. Every flag is defined
once, in the verb's own module, so the old ``python -m
leed_daylight.drivers.<name>`` commands and this one cannot drift apart.
Modules are imported only when their verb runs, so ``--help`` stays fast.

When ``LEED_JOB_RECORD`` names a path (the wizard sets it for every job it
starts, wizard spec 8.1), ``main`` writes the verb's exit code and end time
there however the verb ends, so a finished job is told from a crashed one
even when the wizard was closed meanwhile.
"""
from __future__ import annotations

import contextlib
import datetime
import importlib
import json
import os
import sys
from collections.abc import Sequence

from .constants import JOB_RECORD_ENV, TOOL_VERSION

#: (verb, one line), in work order. The verb is also the module name under
#: ``leed_daylight/drivers/``.
VERBS: tuple[tuple[str, str], ...] = (
    ("init", "Make a new project folder with a project.yaml to fill in"),
    ("scan", "Check the Radiance scene and list its materials, storeys and rooms"),
    ("materials", "Write a materials.csv template from the materials in the scene"),
    ("rooms", "Write a spaces.csv template from the rooms in the scene"),
    ("prepare", "Build the prepared model (frames, blinds, grid) without Radiance"),
    ("run", "Run Option 1 (sDA and ASE)"),
    ("option2", "Run Option 2 (illuminance at two times on two days)"),
    ("views", "Run Quality Views"),
    ("report", "Write the Word reports, plots and glare narrative"),
    ("atlas", "Write the floor atlas page and, with Blender, a Blender file"),
    ("calculator", "Fill the USGBC LEED v4.1 Daylight and Quality Views Calculator"),
    ("package", "Write the checksummed submission folder"),
)

#: The exit code argparse uses for a usage error.
USAGE_ERROR = 2
#: The shell's exit code for a process stopped by Ctrl-C (128 + SIGINT).
INTERRUPTED = 130


def _help() -> str:
    width = max(len(verb) for verb, _ in VERBS)
    lines = [
        "usage: leed-daylight <verb> [arguments]",
        "",
        "LEED v4.1 BD+C daylight and views credits from a Radiance scene.",
        "",
        "verbs, in the order you use them:",
    ]
    lines += [f"  {verb:<{width}}  {text}" for verb, text in VERBS]
    lines += [
        "",
        "leed-daylight <verb> --help   shows the verb's own arguments",
        "leed-daylight --version       shows the tool version",
    ]
    return "\n".join(lines)


#: A new file: created here, never an existing one, never through a link.
_NEW_FILE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)


def _write_end_record(path: str, code: int) -> None:
    """Write ``{"exit_code", "ended"}`` to ``path`` through a ``.part``
    file created new and never through a link (a leftover plain ``.part``
    is removed first; a link at either name is refused); a failure only
    warns."""
    part = f"{path}.part"
    ended = datetime.datetime.now().isoformat(timespec="seconds")
    data = json.dumps({"exit_code": code, "ended": ended}).encode("utf-8")
    try:
        for name in (part, path):
            if os.path.islink(name):
                raise OSError(f"{name}: is a symbolic link")
        with contextlib.suppress(FileNotFoundError):
            os.unlink(part)
        with os.fdopen(os.open(part, _NEW_FILE, 0o666), "wb") as handle:
            handle.write(data)
        os.replace(part, path)
    except OSError as exc:
        print(f"warning: could not write the job end record {path}: {exc}", file=sys.stderr)


def _exit_code(code: object) -> int:
    """The process exit code ``SystemExit(code)`` gives."""
    if code is None:
        return 0
    if isinstance(code, int) and not isinstance(code, bool):
        return code
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    record = os.environ.get(JOB_RECORD_ENV)
    if not record:
        return _dispatch(argv)
    try:
        code = _dispatch(argv)
    except KeyboardInterrupt:
        _write_end_record(record, INTERRUPTED)
        raise
    except SystemExit as exc:
        _write_end_record(record, _exit_code(exc.code))
        raise
    except BaseException:
        _write_end_record(record, 1)
        raise
    _write_end_record(record, _exit_code(code))
    return code


def _dispatch(argv: Sequence[str] | None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_help())
        return 0
    if argv[0] == "--version":
        print(f"leed-daylight {TOOL_VERSION}")
        return 0
    verb, rest = argv[0], argv[1:]
    known = [name for name, _ in VERBS]
    if verb not in known:
        print(f"error: unknown verb {verb!r}; known verbs are {', '.join(known)}", file=sys.stderr)
        return USAGE_ERROR
    try:
        module = importlib.import_module(f"leed_daylight.drivers.{verb}")
        return module.main(rest, prog=f"leed-daylight {verb}")
    except KeyboardInterrupt:
        print("stopped; run the same command again to resume", file=sys.stderr)
        return INTERRUPTED
