# leed_mcp/adapters/project_state.py
"""What a project folder holds, read without loading the project (spec 6.1).

Every reader here tolerates a draft: a ``FILL_ME`` value, a missing file or
an unreadable ``spaces.csv`` becomes part of the answer, never a refusal.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
import shutil
from collections.abc import Callable, Sequence
from typing import Any

from leed_daylight.constants import FILL_ME
from leed_daylight.deliver.data import result_path
from leed_daylight.drivers.option2 import OPTION2_BINARIES
from leed_daylight.drivers.views import VIEWS_BINARIES
from leed_daylight.engine.blender import BlenderNotFound, find_blender
from leed_daylight.engine.jobs import Jobs, lock_state
from leed_daylight.engine.radiance import REQUIRED_BINARIES
from leed_daylight.project.spaces import case_names, load_spaces, views_case_names
from leed_daylight.scene.input_file import read_input_text
from leed_mcp.adapters.input_files import load_yaml
from leed_mcp.adapters.results_files import CREDITS, InputsOnDisk, check_inputs, read_result, refuse_linked
from leed_mcp.errors import Refused, one_line

logger = logging.getLogger(__name__)

__all__ = [
    "FILE_KINDS", "KNOWN_FILES", "blender_path", "calculator_state", "fill_me_keys", "folder_files", "job_state", "lock_holder",
    "project_cases", "project_folders", "radiance_missing", "read_project_yaml", "report_libraries",
    "results_found", "results_state",
]

#: The files ``project status`` reports, and whether a run needs each.
KNOWN_FILES = (("project.yaml", True), ("materials.csv", True), ("spaces.csv", True),
               ("splits.csv", False), ("glare_narrative.md", False))
#: ``project files`` kinds by suffix (spec 6.1); every other file is ``other``.
FILE_KINDS = {".rad": "scene", ".wea": "weather", ".epw": "weather", ".png": "drawing",
              ".jpg": "drawing", ".jpeg": "drawing", ".pdf": "drawing", ".svg": "drawing"}
#: Folders ``project files`` leaves out: the pipeline's own outputs.
_SKIPPED = ("_run", "submission")
#: ``folder_files`` stops listing after this many entries. Chosen.
FILES_MAX = 20_000
#: The Radiance programs each run action needs (spec 6.1, ``project status``).
_PROGRAMS = {"prepare": (), "option1": REQUIRED_BINARIES, "option2": OPTION2_BINARIES,
             "views": VIEWS_BINARIES}
#: Report libraries by distribution name -> import name.
_LIBRARIES = {"python-docx": "docx", "matplotlib": "matplotlib", "Pillow": "PIL",
              "cairosvg": "cairosvg"}


def read_project_yaml(folder: pathlib.Path) -> dict[str, Any] | None:
    """project.yaml parsed, or None when it is absent. A file that is not a
    YAML mapping raises ``ValueError``; a YAML error propagates."""
    path = folder / "project.yaml"
    refuse_linked(folder, path)
    if not path.is_file():
        return None
    raw = load_yaml(read_input_text(path, encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("project.yaml: project.yaml must be a mapping at the top level")
    return raw


def fill_me_keys(value: Any, path: str = "") -> list[str]:
    """Every dotted key whose value is still ``FILL_ME``, in file order.
    Ported from ``project.config._fill_me_paths`` (PORTED.md)."""
    if isinstance(value, dict):
        found: list[str] = []
        for key, item in value.items():
            found += fill_me_keys(item, f"{path}.{key}" if path else str(key))
        return found
    if isinstance(value, list):
        found = []
        for index, item in enumerate(value):
            found += fill_me_keys(item, f"{path}[{index}]")
        return found
    return [path] if isinstance(value, str) and value.strip() == FILL_ME else []


def project_cases(folder: pathlib.Path) -> tuple[tuple[str, ...], tuple[str, ...]] | str:
    """``(include cases, views cases)`` from spaces.csv, or the reason they
    cannot be read."""
    path = folder / "spaces.csv"
    try:
        refuse_linked(folder, path)
        if not path.is_file():
            return "spaces.csv does not exist yet"
        spaces = load_spaces(path)
    except (OSError, ValueError) as exc:
        return one_line(exc)
    return case_names(spaces), views_case_names(spaces)


def results_state(folder: pathlib.Path, credit: str, case: str,
                  disk: Callable[[], InputsOnDisk] | None = None) -> dict[str, Any]:
    """Whether the credit's results exist for ``case`` and whether their
    inputs changed: ``{"exists", "inputs", "changed", "outside"}``, with
    ``inputs`` one of current, changed, unchecked, unreadable, or None.
    ``disk`` gives the inputs read once for the whole folder."""
    state = {"case": case, "credit": credit, "exists": False, "inputs": None, "changed": [], "outside": []}
    if not result_path(folder, credit, case).exists():
        return state
    try:
        _, content = read_result(folder, credit, case)
    except (Refused, OSError, ValueError):
        return {**state, "exists": True, "inputs": "unreadable"}
    recorded = content.get("inputs")
    if not isinstance(recorded, dict):
        return {**state, "exists": True, "inputs": "unreadable"}
    check = check_inputs(recorded, folder, disk() if disk is not None else None)
    return {**state, "exists": True, "inputs": check.state, "changed": check.changed, "outside": check.outside}


def results_found(folder: pathlib.Path) -> dict[str, list[str]]:
    """Per credit, the cases that have a results file (no fingerprints)."""
    found: dict[str, list[str]] = {}
    for credit in CREDITS:
        parent = result_path(folder, credit, "x").parent.parent
        cases = []
        if parent.is_dir() and not parent.is_symlink():
            cases = sorted(entry.name for entry in parent.iterdir()
                           if entry.is_dir() and not entry.is_symlink()
                           and result_path(folder, credit, entry.name).is_file())
        found[credit] = cases
    return found


def lock_holder(folder: pathlib.Path) -> dict[str, Any]:
    """The folder lock: ``{"state", "verb", "pid", "started"}``."""
    state, info = lock_state(folder)
    return {"state": state, "verb": info.verb if info else None, "pid": info.pid if info else None,
            "started": info.started if info else None}


def job_state(folder: pathlib.Path) -> dict[str, Any]:
    """The folder's latest job as the pipeline records it (``_run/jobs/current.json``)."""
    status = Jobs(folder).status()
    record = status.record
    return {"state": status.state, "verb": record.verb if record else None,
            "started": record.started if record else None, "log": record.log if record else None,
            "exit_code": status.exit_code, "ended": status.ended}


def radiance_missing() -> dict[str, list[str]]:
    """Per run action, the Radiance programs not found on this server's PATH."""
    return {action: [name for name in names if shutil.which(name) is None]
            for action, names in _PROGRAMS.items()}


def report_libraries() -> dict[str, bool]:
    """Per report library, whether it is installed (found, never imported)."""
    return {dist: importlib.util.find_spec(module) is not None for dist, module in _LIBRARIES.items()}


def blender_path() -> str | None:
    """Blender's path as the pipeline finds it, or None. Blender is not started."""
    try:
        return str(find_blender())
    except BlenderNotFound:
        return None


def calculator_state() -> str:
    """``not configured``, ``V02`` or ``unsupported: <reason>`` for LEED_MCP_CALCULATOR."""
    from leed_mcp.adapters.run_checks import calculator_template
    try:
        calculator_template()
    except Refused as exc:
        content = exc.result.structured_content
        if content["advisories"][0]["code"] == "CALCULATOR_NOT_CONFIGURED":
            return "not configured"
        return f"unsupported: {content['reason']}"
    return "V02"


def project_folders(roots: Sequence[pathlib.Path]) -> list[pathlib.Path]:
    """Every folder holding a project.yaml at or directly inside a root, in
    root order, each once (``project list``; spec 8.1's D6 check). A root
    that is missing or cannot be read is skipped with a warning, so the
    other roots still count (code review M5)."""
    folders: list[pathlib.Path] = []
    for root in roots:
        try:
            candidates = [root, *sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink())]
        except OSError as exc:
            logger.warning("root %s skipped: %s", root, one_line(exc))
            continue
        folders += [p for p in candidates if (p / "project.yaml").is_file() and p not in folders]
    return folders


def folder_files(folder: pathlib.Path) -> tuple[list[dict[str, Any]], bool]:
    """``(entries, complete)``: the files under ``folder`` except ``_run/``
    and ``submission/``, sorted by path: ``{"path", "size", "kind", "link"}``.
    Folder links are not followed; a file link is listed with its own size
    and ``link: true``. The walk stops after ``FILES_MAX`` entries
    (``complete`` is then False); an entry that vanishes between the listing
    and its ``lstat`` is skipped."""
    entries: list[dict[str, Any]] = []
    for top, dirs, names in os.walk(folder, followlinks=False):
        here = pathlib.Path(top)
        if here == folder:
            dirs[:] = [d for d in dirs if d not in _SKIPPED]
        dirs.sort()
        for name in sorted(names) + [d for d in dirs if (here / d).is_symlink()]:
            if len(entries) >= FILES_MAX:
                return sorted(entries, key=lambda e: e["path"]), False
            path = here / name
            try:
                info = path.lstat()
            except OSError:
                continue
            entries.append({
                "path": path.relative_to(folder).as_posix(), "size": info.st_size,
                "kind": FILE_KINDS.get(path.suffix.lower(), "other"), "link": path.is_symlink(),
            })
    return sorted(entries, key=lambda e: e["path"]), True
