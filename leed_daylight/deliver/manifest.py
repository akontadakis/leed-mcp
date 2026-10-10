# leed_daylight/deliver/manifest.py
"""The report manifest: what the report step made, read by the package step
(reports spec 8, 9).

``<project folder>/_run/reports/<case>/report_manifest.json`` ties each
option's reports to the results and ``points.csv`` they were made from, and
(Option 1) to the glare narrative. ``drivers/report.py`` writes it and
``deliver/package.py`` reads it; its name, its entry keys, its writer and
its reader live here, once (final review M2), because ``drivers/`` may
import ``deliver/`` but not the other way round.

The reader refuses a hand-edited or truncated manifest by name (final
review I5): the top level is an object, ``options`` and ``skipped`` are
objects, every option entry is an object, and every ``files`` item is a
relative POSIX path with no ``..`` -- so no entry can steer a copy outside
the reports folder or the package being built.

The ``atlas`` entry (atlas spec 11) sits beside ``options``: the files the
atlas wrote (relative to ``_run/atlas/<case>/``, ``blender.log`` never among
them), the input fingerprints it was built from, and each option it showed
with the sha256 of that option's results file and ``points.csv``. The
reader checks its shape and its paths by the same rule.
"""
from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable, Mapping

from ..project.inputs import sha256_file
from .data import DeliverError, read_json_object

REPORT_MANIFEST = "report_manifest.json"
#: The key of the atlas entry in the report manifest.
ATLAS_KEY = "atlas"
_ATLAS_RERUN = "run python -m leed_daylight.drivers.atlas again"


def _safe_relative(name) -> bool:
    """True when ``name`` is a non-empty relative POSIX path with no ``..``
    part and no backslash."""
    if not isinstance(name, str) or not name or "\\" in name:
        return False
    path = pathlib.PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def read_report_manifest(path, *, error: type[DeliverError] = DeliverError, check_atlas: bool = True) -> dict:
    """The report manifest at ``path``, its shape checked; ``error`` (a
    :class:`DeliverError` subclass) names the file and the fault.

    ``check_atlas=False`` is for the two commands that replace the
    ``atlas`` entry (``atlas`` and ``report``; final review M1): the entry
    is dropped unread, so a bad one is replaced instead of refusing the
    command that would replace it. ``package`` keeps the full check."""
    path = pathlib.Path(path)
    try:
        manifest = read_json_object(path)
    except DeliverError as exc:
        raise error(str(exc)) from None
    manifest.setdefault("options", {})
    manifest.setdefault("skipped", {})
    for key in ("options", "skipped"):
        if not isinstance(manifest[key], dict):
            raise error(f"{path}: {key!r} is a {type(manifest[key]).__name__}, not an object; "
                        "delete the file and run python -m leed_daylight.drivers.report again")
    for option, entry in manifest["options"].items():
        if not isinstance(entry, dict):
            raise error(f"{path}: the {option!r} entry is a {type(entry).__name__}, not an object; "
                        "delete the file and run python -m leed_daylight.drivers.report again")
        files = entry.get("files", [])
        if not isinstance(files, list):
            raise error(f"{path}: the {option!r} entry's 'files' is a {type(files).__name__}, not a list; "
                        "run python -m leed_daylight.drivers.report again")
        bad = [f for f in files if not _safe_relative(f)]
        if bad:
            raise error(f"{path}: the {option!r} entry lists {bad[0]!r}, which is not a relative path "
                        "inside the reports folder; run python -m leed_daylight.drivers.report again")
    if not check_atlas:
        manifest.pop(ATLAS_KEY, None)
    elif ATLAS_KEY in manifest:
        _check_atlas_entry(path, manifest[ATLAS_KEY], error)
    return manifest


def _check_atlas_entry(path: pathlib.Path, entry, error: type[DeliverError]) -> None:
    """The ``atlas`` entry's shape: ``files`` a list of safe relative paths,
    ``inputs`` an object of strings, ``options`` an object of objects that
    each hold a ``results_sha256`` and a ``points_sha256`` string."""
    def refuse(fault: str) -> None:
        raise error(f"{path}: the 'atlas' entry {fault}; {_ATLAS_RERUN}")

    if not isinstance(entry, dict):
        refuse(f"is a {type(entry).__name__}, not an object")
    files = entry.get("files")
    if not isinstance(files, list):
        refuse(f"has 'files' as a {type(files).__name__}, not a list")
    bad = [f for f in files if not _safe_relative(f)]
    if bad:
        refuse(f"lists {bad[0]!r}, which is not a relative path inside the atlas folder")
    inputs = entry.get("inputs")
    if not isinstance(inputs, dict) or not all(isinstance(v, str) for v in inputs.values()):
        refuse("has no 'inputs' object of fingerprints")
    options = entry.get("options")
    if not isinstance(options, dict):
        refuse("has no 'options' object")
    for option, shas in options.items():
        if not isinstance(shas, dict) or not all(
            isinstance(shas.get(key), str) for key in ("results_sha256", "points_sha256")
        ):
            refuse(f"has no results_sha256 and points_sha256 for {option!r}")


def atlas_entry(files: Iterable[str], inputs: Mapping[str, str], options: Mapping[str, pathlib.Path],
                presentation: str | None = None) -> dict:
    """The atlas entry: ``files`` (relative POSIX paths, sorted), the input
    fingerprints ``inputs``, and for each option in ``options`` (option to
    its results file) the sha256 of that file and of the ``points.csv``
    beside it. ``presentation`` (``presentation_digest``) is recorded as
    ``presentation_sha256`` when given."""
    entry = {
        "files": sorted(files),
        "inputs": dict(sorted(inputs.items())),
        "options": {
            option: {
                "results_sha256": sha256_file(results_path),
                "points_sha256": sha256_file(pathlib.Path(results_path).parent / "points.csv"),
            }
            for option, results_path in sorted(options.items())
        },
    }
    if presentation is not None:
        entry["presentation_sha256"] = presentation
    return entry


def report_entry(results_path: pathlib.Path, files, *, narrative_path: pathlib.Path | None = None,
                 case: str | None = None, presentation: str | None = None) -> dict:
    """One option's manifest entry: the sha256 of its results file and of
    the ``points.csv`` beside it, the files written (relative POSIX paths),
    and for Option 1 the narrative's sha256 and the case it was rendered
    for (pre-flight ruling F6). ``presentation`` (``presentation_digest``)
    is recorded as ``presentation_sha256`` when given."""
    entry = {
        "results_sha256": sha256_file(results_path),
        "points_sha256": sha256_file(pathlib.Path(results_path).parent / "points.csv"),
        "files": sorted(files),
    }
    if narrative_path is not None:
        entry["narrative_sha256"] = sha256_file(narrative_path)
        entry["narrative_case"] = case
    if presentation is not None:
        entry["presentation_sha256"] = presentation
    return entry


def write_report_manifest(path, manifest: dict) -> None:
    pathlib.Path(path).write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        newline="\n",
    )
