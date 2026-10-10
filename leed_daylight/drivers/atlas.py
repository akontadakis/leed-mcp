# leed_daylight/drivers/atlas.py
"""The ``atlas`` command (atlas spec 3, 11, 12): the floor atlas page, its
data bundle and, when a Blender 4.2 or newer is found, a Blender file.

    leed-daylight atlas "<project folder>" [--case <name>] [--no-blender]

It writes ``<folder>/_run/atlas/<case>/`` (emptied first): ``atlas.html``,
``bundle.npz`` and, with Blender, ``model.blend`` and ``blender.log``.
Before building it prints the case summary ``prepare`` prints (rooms, grid
points, window groups, warnings), so this one command is the model check
before a run. It shows every result that exists and is current for the
case; with none, the page is the model check and every option is listed
with the reason it is not shown. It prints every file written, each option
not shown, every warning and one Blender line.

The atlas entry is merged into ``_run/reports/<case>/report_manifest.json``
(the manifest is created when no report has run yet); every other entry is
kept as it was. The manifest is read before anything is written, so a
malformed one is refused first; its old ``atlas`` entry is not checked,
since this command replaces it.

Exit codes (spec 12): 0 when the page is written, whether or not Blender
was found, new enough or asked for; 1 on a refusal (``error: ...``) and
when Blender ran and failed (the page is kept; the line names the log).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence

import yaml
from shapely.errors import ShapelyError

from ..deliver.atlas import AtlasResult, build_atlas
from ..deliver.data import load_case
from ..deliver.manifest import ATLAS_KEY, REPORT_MANIFEST, read_report_manifest, write_report_manifest
from ..engine.jobs import folder_lock
from ._folder import REPO_ROOT, check_folder
from .prepare import _print_summary


class AtlasCommandError(ValueError):
    """The atlas command cannot proceed from these inputs."""


def _read_manifest(folder: pathlib.Path, case: str) -> tuple[pathlib.Path, dict]:
    """The case's report manifest path and its content (a new one when no
    report has run), read before anything is written, so a malformed one
    is refused first (final review C5). Its ``atlas`` entry is left out
    unread: this command replaces it (final review M1)."""
    path = folder / "_run" / "reports" / case / REPORT_MANIFEST
    if path.is_file():
        return path, read_report_manifest(path, check_atlas=False)
    return path, {"case": case, "options": {}, "skipped": {}}


def _merge_entry(path: pathlib.Path, manifest: dict, entry: dict) -> None:
    """Set the atlas entry of the manifest read by ``_read_manifest``,
    keeping the rest, and write it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest[ATLAS_KEY] = entry
    write_report_manifest(path, manifest)


def atlas_folder(folder, case: str | None = None, *, blender: bool = True, out=None) -> AtlasResult:
    """Build the atlas of ``case`` and merge its entry into the report
    manifest; print the summary, the files, the skipped options, the
    warnings and the Blender line to ``out`` (default ``sys.stdout``,
    resolved here)."""
    out = sys.stdout if out is None else out
    folder = pathlib.Path(folder)
    check_folder(folder, REPO_ROOT, AtlasCommandError)

    def summary(prepared) -> None:
        _print_summary(prepared.manifest, prepared.root, out=out, groups=prepared.groups, root_line=False)

    data = load_case(folder, case, require_results=False)
    manifest_path, manifest = _read_manifest(folder, data.case)
    result = build_atlas(folder, data.case, blender=blender, data=data, on_prepared=summary, out=out)
    _merge_entry(manifest_path, manifest, result.entry)
    for option, reason in result.skipped.items():
        print(f"skipped {option}: {reason}", file=out)
    for warning in result.warnings:
        print(f"warning: {warning}", file=out)
    if result.blender and not result.blender_failed:
        print(result.blender, file=out)
    return result


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog or "python -m leed_daylight.drivers.atlas",
        description="Write the floor atlas page and, with Blender 4.2 or newer, a Blender file.",
    )
    parser.add_argument("folder", help="the project folder")
    parser.add_argument("--case", default=None, help="the case (default: the first case in spaces.csv)")
    parser.add_argument("--no-blender", action="store_true", help="do not look for Blender or build model.blend")
    args = parser.parse_args(argv)
    try:
        check_folder(pathlib.Path(args.folder), REPO_ROOT, AtlasCommandError)
        with folder_lock(pathlib.Path(args.folder), "atlas"):
            result = atlas_folder(args.folder, args.case, blender=not args.no_blender)
    except (ValueError, OSError, yaml.YAMLError, ShapelyError) as exc:
        text = " ".join(line.strip() for line in str(exc).splitlines() if line.strip())
        print(f"error: {text or type(exc).__name__}", file=sys.stderr)
        return 1
    if result.blender_failed:
        print(f"error: {result.blender}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
