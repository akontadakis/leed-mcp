# leed_mcp/adapters/results_files.py
"""Reading a credit's results safely, and whether its inputs changed since.

Results live at ``deliver.data.result_path(folder, credit, case)`` with
``points.csv`` beside them. Each read refuses a link anywhere from the
project folder down (``engine.jobs.checked_below``, ``open_plain``) and a
file over ``INPUT_FILE_MAX``: the pipeline's own readers
(``deliver.data.read_json_object``) apply neither.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import pathlib
from collections.abc import Mapping
from typing import Any, Literal, NamedTuple

import yaml

from leed_daylight.deliver.data import result_path
from leed_daylight.engine.jobs import UnsafePath, checked_below, is_link, open_plain
from leed_daylight.project.config import load_project
from leed_daylight.project.inputs import _project_yaml_digest, input_fingerprints
from leed_daylight.scene.input_file import INPUT_FILE_MAX, read_input_bytes, too_large
from leed_mcp.envelope import refusal
from leed_mcp.errors import Refused, one_line

__all__ = ["CREDITS", "InputsCheck", "InputsOnDisk", "check_inputs", "inputs_on_disk", "outside_inputs", "points_path", "read_points",
           "read_result", "refuse_linked"]

#: The credits a results file exists for (``deliver.data.OPTIONS``).
CREDITS = ("option1", "option2", "views")
#: The rerun action per credit, for the RESULTS_MISSING reason.
_RERUN = {"option1": "run(action='option1')", "option2": "run(action='option2')",
          "views": "run(action='views')"}


def _read_guarded(folder: pathlib.Path, path: pathlib.Path) -> bytes:
    """The bytes of ``path``: no link from ``folder`` down, a plain file, at
    most ``INPUT_FILE_MAX`` bytes. ``FileNotFoundError`` when it is absent."""
    checked_below(folder, path.parent)
    with open_plain(path) as handle:
        data = handle.read(INPUT_FILE_MAX + 1)
    if len(data) > INPUT_FILE_MAX:
        raise too_large(path, INPUT_FILE_MAX)
    return data


def refuse_linked(folder: pathlib.Path, path: pathlib.Path) -> None:
    """``UnsafePath`` when ``path``, or a folder from ``folder`` down to it, is
    a link: a project file is never read through one (spec 9, ``PATH_UNSAFE``;
    spec 10). The pipeline's readers follow links, so this runs first."""
    checked_below(folder, path.parent)
    if is_link(path):
        raise UnsafePath(f"{path}: is a symbolic link; it is not read")


def read_result(folder: pathlib.Path, credit: str, case: str) -> tuple[pathlib.Path, dict[str, Any]]:
    """``(path, content)`` of the credit's results file, or ``Refused``:
    ``RESULTS_MISSING`` when there is none, ``INPUT_INVALID`` when it is not
    a JSON object."""
    path = result_path(folder, credit, case)
    try:
        data = _read_guarded(folder, path)
    except FileNotFoundError:
        raise Refused(refusal(
            "RESULTS_MISSING",
            f"no {credit} results for case {case!r} ({path} not found); run {_RERUN[credit]} first.",
            field="case")) from None
    try:
        content = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Refused(refusal("INPUT_INVALID", f"{path}: not valid JSON ({one_line(exc)})")) from None
    if not isinstance(content, dict):
        raise Refused(refusal("INPUT_INVALID", f"{path}: the top level is not a JSON object"))
    return path, content


def points_path(folder: pathlib.Path, credit: str, case: str) -> pathlib.Path:
    return result_path(folder, credit, case).parent / "points.csv"


def _cell(text: str) -> Any:
    """A points.csv cell as JSON: yes/no as a bool, a finite number as a
    number, ``inf``/``nan`` and an empty cell as None, else the text."""
    if text in ("yes", "no"):
        return text == "yes"
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() and "." not in text else number


def read_points(folder: pathlib.Path, credit: str, case: str) -> tuple[pathlib.Path, list[str], list[dict[str, Any]]]:
    """``(path, columns, rows)`` of the credit's points.csv, or ``Refused``."""
    path = points_path(folder, credit, case)
    try:
        data = _read_guarded(folder, path)
    except FileNotFoundError:
        raise Refused(refusal(
            "RESULTS_MISSING", f"no points.csv for {credit} case {case!r} ({path} not found).",
            field="case")) from None
    try:
        reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""))
        rows = [row for row in reader if row]
    except (UnicodeDecodeError, csv.Error) as exc:
        raise Refused(refusal("INPUT_INVALID", f"{path}: not a readable CSV file ({one_line(exc)})")) from None
    if not rows:
        raise Refused(refusal("INPUT_INVALID", f"{path}: empty file; expected a header row"))
    columns, body = rows[0], rows[1:]
    bad = next((n for n, row in enumerate(body, 2) if len(row) != len(columns)), None)
    if bad is not None:
        raise Refused(refusal("INPUT_INVALID", f"{path} line {bad}: expected {len(columns)} fields"))
    # ``room`` is the space's name, text even when it reads as a number.
    return path, columns, [{name: value if name == "room" else _cell(value)
                            for name, value in zip(columns, row)} for row in body]


def _inside(folder: pathlib.Path, path: str | pathlib.Path) -> bool:
    return folder in pathlib.Path(path).resolve().parents


def _digest(folder: pathlib.Path, key: str) -> str | None:
    """The current digest of one recorded input, or None when it is
    unreadable or lies outside ``folder`` (spec 10: never read)."""
    path = pathlib.Path(key) if pathlib.Path(key).is_absolute() else folder / key
    if not _inside(folder, path):
        return None
    try:
        if key == "project.yaml":
            return _project_yaml_digest(path)
        return hashlib.sha256(read_input_bytes(path)).hexdigest()
    except (OSError, ValueError, yaml.YAMLError):
        return None


class InputsCheck(NamedTuple):
    """Whether a run's inputs changed: ``state``, the ``changed`` keys, and
    the files project.yaml names ``outside`` the folder (never read)."""

    state: Literal["current", "changed", "unchecked"]
    changed: list[str]
    outside: list[str]


def outside_inputs(project, folder: pathlib.Path) -> list[str]:
    """The input files project.yaml names that resolve outside ``folder``,
    resolved and sorted. Spec 10 keeps every such path inside the folder."""
    named = [project.weather_file, project.credit_rules_path, *project.scene.files]
    if project.grid.points_file is not None:
        named.append(project.grid.points_file)
    return sorted({str(pathlib.Path(p).resolve()) for p in named if not _inside(folder, p)})


class InputsOnDisk(NamedTuple):
    """The inputs as they are now, read once for any number of results:
    whether the project ``loads``, the inputs named ``outside`` the folder,
    the ``current`` fingerprints (None when an input cannot be read), and a
    ``cache`` of single-key digests for that case."""

    loads: bool
    outside: list[str]
    current: dict[str, str] | None
    cache: dict[str, str | None]


def inputs_on_disk(folder: pathlib.Path) -> InputsOnDisk:
    """Load project.yaml and fingerprint its inputs, once (spec 10: nothing
    is read when an input is named outside the folder)."""
    try:
        project = load_project(folder / "project.yaml")
    except (OSError, ValueError, yaml.YAMLError):
        return InputsOnDisk(False, [], None, {})
    outside = outside_inputs(project, folder)
    if outside:
        return InputsOnDisk(True, outside, None, {})
    try:
        current = input_fingerprints(project, folder)
    except (OSError, ValueError):
        current = None
    return InputsOnDisk(True, [], current, {})


def check_inputs(recorded: Mapping[str, str], folder: pathlib.Path,
                 disk: InputsOnDisk | None = None) -> InputsCheck:
    """Whether the inputs changed since the results were computed.

    When the project no longer loads (for example a value set back to
    ``FILL_ME``), ``project.yaml`` itself changed. When project.yaml names an
    input outside the folder, nothing is read and the check is
    ``unchecked``. Otherwise every key whose digest differs between
    ``recorded`` and the disk, as ``stale_inputs`` compares them; when an
    input cannot be read, each recorded key is compared on its own and an
    unreadable one counts as changed. Pass ``disk`` to check several results
    against one reading of the inputs."""
    disk = disk if disk is not None else inputs_on_disk(folder)
    if not disk.loads:
        return InputsCheck("changed", ["project.yaml"], [])
    if disk.outside:
        return InputsCheck("unchecked", [], disk.outside)
    if disk.current is not None:
        changed = sorted(k for k in set(recorded) | set(disk.current) if recorded.get(k) != disk.current.get(k))
    else:
        for key in recorded:
            if key not in disk.cache:
                disk.cache[key] = _digest(folder, key)
        changed = sorted(key for key, digest in recorded.items() if disk.cache[key] != digest)
    return InputsCheck("changed" if changed else "current", changed, [])
