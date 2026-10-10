# leed_mcp/adapters/weather_import.py
"""``inputs(action="import_weather")`` (spec 6.3): an EnergyPlus ``.epw``
inside the project folder to ``weather/<name>.wea``.

Radiance's ``epw2wea`` runs inline, with no shell, its stdin the null
device and a 60 s limit, into a temporary file in ``_run/``. It reads a
copy of the ``.epw`` in ``_run/``, made under the folder lock from bytes
read without following a link (``input_files.read_bytes``), so a link
put in place after the check is never followed. The result is
checked with the pipeline's ``read_wea`` and saved as every input file is
(``input_files.write_file``): the folder lock, a planted link refused, an
existing ``.wea`` of that name backed up. project.yaml is not changed;
``weather.file`` is set with ``inputs(action="update", file="project")``.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import uuid
from typing import Any

from leed_daylight.engine.jobs import checked_dir, write_new
from leed_daylight.project.spaces import is_safe_case_name
from leed_daylight.project.weather import read_wea
from leed_daylight.scene.input_file import read_input_bytes
from leed_mcp.adapters.input_files import Written, digest, read_bytes, write_file
from leed_mcp.adapters.results_files import refuse_linked
from leed_mcp.envelope import refusal
from leed_mcp.errors import Refused

__all__ = ["PROGRAM", "TIMEOUT_S", "import_weather"]

PROGRAM = "epw2wea"
#: epw2wea's time limit, seconds (spec 6.3).
TIMEOUT_S = 60.0
#: The last lines of epw2wea's stderr a refusal keeps.
_STDERR_LINES = 20


def _epw(folder: pathlib.Path, epw: str) -> pathlib.Path:
    """The ``.epw`` file: a path inside ``folder``, relative to it or
    absolute, reached through no link."""
    path = pathlib.Path(epw)
    lexical = path if path.is_absolute() else folder / path
    if path.is_absolute() and folder not in lexical.parents:
        # Through a linked prefix (/tmp -> /private/tmp on macOS): the
        # lexical ancestor that is the folder is replaced by the folder, so
        # a link inside the folder is still refused below.
        for ancestor in path.parents:
            if ancestor.resolve() == folder:
                lexical = folder / path.relative_to(ancestor)
                break
    if folder not in lexical.parents or folder not in lexical.resolve().parents:
        raise Refused(refusal("PATH_OUTSIDE_ROOTS", f"epw {epw!r} is not inside the project folder ({folder}); "
                              "put the file there first.", field="epw"))
    refuse_linked(folder, lexical)
    if lexical.suffix.lower() != ".epw":
        raise Refused(refusal("INPUT_INVALID", f"epw {epw!r} is not an .epw file.", field="epw"))
    if not lexical.is_file():
        raise Refused(refusal("FILE_MISSING", f"{lexical}: no such file.", field="epw"))
    return lexical


def _run_epw2wea(program: str, epw: pathlib.Path, out: pathlib.Path) -> None:
    argv = [program, str(epw), str(out)]
    try:
        done = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=TIMEOUT_S,
                              check=False, shell=False)
    except subprocess.TimeoutExpired:
        raise Refused(refusal("RADIANCE_FAILED", f"epw2wea did not finish within {TIMEOUT_S:g} s; it was "
                              "stopped.")) from None
    except OSError as exc:
        raise Refused(refusal("RADIANCE_FAILED", f"epw2wea could not start: {exc}")) from None
    if done.returncode != 0:
        tail = "\n".join(done.stderr.decode("utf-8", "replace").splitlines()[-_STDERR_LINES:])
        raise Refused(refusal("RADIANCE_FAILED", f"epw2wea exited with code {done.returncode}: {tail}"))


def import_weather(folder: pathlib.Path, epw: str, name: str | None) -> tuple[Written, pathlib.Path]:
    """Convert ``epw`` to ``weather/<name>.wea`` (``name`` defaults to the
    file's stem); returns the file written and the ``.epw`` read."""
    source = _epw(folder, epw)
    stem = source.stem if name is None else name
    if not is_safe_case_name(stem) or stem.startswith("."):
        raise Refused(refusal("INPUT_INVALID", f"name {stem!r} is not one file name: no / \\ : or control "
                              "character, and not starting with a dot.", field="name"))
    program = shutil.which(PROGRAM)
    if program is None:
        raise Refused(refusal("RADIANCE_NOT_FOUND", f"{PROGRAM} is not on this server's PATH; add the Radiance "
                              "bin folder to PATH in the client's server configuration."))
    target = f"weather/{stem}.wea"
    checked_dir(folder, "weather")

    def build(old: bytes | None, raw: Any) -> bytes:
        data = read_bytes(folder, source.relative_to(folder).as_posix())
        if data is None:
            raise Refused(refusal("FILE_MISSING", f"{source}: no such file.", field="epw"))
        run = checked_dir(folder, "_run")
        base = f"epw2wea-{uuid.uuid4().hex}"
        copy, tmp = run / f"{base}.epw", run / f"{base}.wea"
        try:
            write_new(copy, data)
            _run_epw2wea(program, copy, tmp)
            refuse_linked(folder, tmp)
            return read_input_bytes(tmp)
        finally:
            copy.unlink(missing_ok=True)
            tmp.unlink(missing_ok=True)

    written = write_file(folder, target, digest(read_bytes(folder, target)), build, read_wea)
    return written, source
