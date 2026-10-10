# leed_daylight/drivers/prepare.py
"""The ``prepare`` verb (CLI spec 4.4): build the prepared model, no Radiance.

    leed-daylight prepare "<project folder>" [--case <name>]

It runs the same chain as ``run`` up to ``build.prepare`` -- load_project,
read_scene, load_materials, validate_scene(..., declared_types=...).raise_if_errors(),
scene_rooms (room detection, then the lines of splits.csv), load_spaces, the
case -- and writes the prepared model into ``<folder>/_run/``. Then it prints
the counts from the manifest: rooms, grid points, regularly occupied area,
window groups, each room's own (with the pieces of a split or merged space),
and every warning. A cheap check before an hours-long run. No Radiance binary is
looked for or run; ``run`` rebuilds the same model anyway.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter
from collections.abc import Sequence

import yaml
from shapely.errors import ShapelyError

from ..build.glazing_class import KINDS
from ..build.groups import WindowGroup
from ..build.prepare import prepare
from ..engine.jobs import folder_lock
from ..project.config import load_project
from ..project.layout import scene_rooms
from ..project.materials import declared_types, load_materials
from ..project.spaces import load_spaces
from ..project.splits import SPLITS_FILE
from ..scene.reader import read_scene
from ..scene.validate import validate_scene
from ._folder import PREPARED_DIRS, REPO_ROOT, check_folder, check_output_dirs
from .run import _resolve_case

_REQUIRED_FILES = ("project.yaml", "materials.csv", "spaces.csv")


class PrepareCommandError(ValueError):
    """The prepare command cannot proceed from these inputs."""


def _kind_breakdown(groups: Sequence[WindowGroup]) -> str:
    """`` (window 3, skylight 1, diffusing 1)``, kinds in KINDS order; empty
    when every group is a vertical facade window, so an all-window project
    prints exactly what it printed before roof glass existed."""
    if all(g.kind == "window" for g in groups):
        return ""
    counts = Counter(g.kind for g in groups)
    return " (" + ", ".join(f"{kind} {counts[kind]}" for kind in KINDS if counts[kind]) + ")"


def _group_line(group: WindowGroup) -> str:
    panes = len(group.openings)
    blind = "no blind (diffusing)" if group.kind == "diffusing" else "blind yes"
    return (
        f"  {group.group_id}: {panes} pane{'' if panes == 1 else 's'}, "
        f"{group.area_m2:.1f} m2, tilt {group.tilt_deg:.1f}, {blind}"
    )


def _piece_text(piece: dict) -> str:
    """``S0_R2 12.00 m2``; a split half adds its splits.csv line or lines."""
    text = f"{piece['room_id']} {float(piece['area_m2']):.2f} m2"
    rows = piece.get("splits_rows")
    if rows:
        which = "line" if len(rows) == 1 else "lines"
        text += f" ({SPLITS_FILE} {which} {', '.join(str(row) for row in rows)})"
    return text


def _print_summary(
    manifest: dict, root: pathlib.Path, *, out, groups: Sequence[WindowGroup] = (),
    root_line: bool = True,
) -> None:
    """The case summary: rooms, grid points, window groups, every warning;
    then, with ``root_line``, where the prepared model was written (the
    ``atlas`` verb prepares into a temporary folder and leaves it out). A
    space in the manifest's ``layout`` (split or merged) gets one more line
    listing its pieces."""
    totals = manifest["totals"]
    rooms = manifest["rooms"]
    layout = manifest.get("layout", {})
    groups_by_room: dict[str, list[str]] = {}
    for group_id, group in manifest["groups"].items():
        groups_by_room.setdefault(group["room"], []).append(group_id)
    print(f"case: {manifest['case']}", file=out)
    print(
        f"rooms: {len(rooms)}, grid points: {totals['grid_point_count']}, "
        f"regularly occupied area: {totals['regularly_occupied_area_m2']:.2f} m2, "
        f"window groups: {totals['group_count']}{_kind_breakdown(groups)}",
        file=out,
    )
    for group in groups:
        if group.kind != "window":
            print(_group_line(group), file=out)
    for room, info in rooms.items():
        ids = ", ".join(sorted(groups_by_room.get(room, []))) or "none"
        print(f"  {room}: {info['area_m2']:.2f} m2, {info['point_count']} points, window groups: {ids}",
              file=out)
        if room in layout:
            pieces = ", ".join(_piece_text(piece) for piece in layout[room]["pieces"])
            print(f"    pieces ({layout[room]['origin']}): {pieces}", file=out)
    print(f"warnings: {len(manifest['warnings'])}", file=out)
    for warning in manifest["warnings"]:
        print(f"  - {warning}", file=out)
    if root_line:
        print(f"prepared model written to {root}", file=out)


def prepare_folder(folder, case: str | None = None, *, out=None) -> pathlib.Path:
    """Build the prepared model of ``case`` into ``<folder>/_run/`` and
    return the manifest path."""
    out = sys.stdout if out is None else out
    folder = pathlib.Path(folder)
    check_folder(folder, REPO_ROOT, PrepareCommandError)
    for name in _REQUIRED_FILES:
        if not (folder / name).is_file():
            raise PrepareCommandError(f"{folder}: missing required file {name!r}")

    project = load_project(folder / "project.yaml")
    prims = read_scene(project.scene.files, project.scene.units)
    roles = load_materials(folder / "materials.csv")
    validate_scene(
        prims, project.scene.units, roles, declared_types=declared_types(folder / "materials.csv"),
    ).raise_if_errors()
    rooms = scene_rooms(project, prims, roles)
    spaces = load_spaces(folder / "spaces.csv")
    case = _resolve_case(spaces, case)
    check_output_dirs(folder, *(("_run", *parts) for parts in PREPARED_DIRS))
    prepared = prepare(project, prims, roles, rooms, spaces, case, folder / "_run")
    _print_summary(prepared.manifest, prepared.root, out=out, groups=prepared.groups)
    return prepared.manifest_path


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog or "python -m leed_daylight.drivers.prepare",
        description="Build the prepared model (frames, blinds, grid) without Radiance.",
    )
    parser.add_argument("folder", help="the project folder (project.yaml, materials.csv, spaces.csv)")
    parser.add_argument("--case", default=None, help="the case (default: the first case in spaces.csv)")
    args = parser.parse_args(argv)
    try:
        check_folder(pathlib.Path(args.folder), REPO_ROOT, PrepareCommandError)
        with folder_lock(pathlib.Path(args.folder), "prepare"):
            prepare_folder(args.folder, args.case)
    except (ValueError, OSError, yaml.YAMLError, ShapelyError) as exc:
        text = " ".join(line.strip() for line in str(exc).splitlines() if line.strip())
        print(f"error: {text or type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
