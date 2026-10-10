# leed_daylight/drivers/scan.py
"""The ``scan`` verb (CLI spec 4.2): check the scene and say what is in it.

    leed-daylight scan "<project folder>"

It needs ``project.yaml`` (and the files it names); ``materials.csv`` is
optional. It prints the scene files, the surface count, the scene's size in
metres (so a wrong ``scene.units`` shows at once), the material table, and
every scene warning and problem. With ``materials.csv`` and no problem, it
also prints each storey's level, room count and floor area, after the lines
of ``splits.csv`` when that file exists (preceded by ``split lines: N``); a
split line that is refused is one more problem (``split problems: 1``).
Without ``materials.csv``, ``splits.csv`` is not read. It writes nothing.
One or more problems: ``error: the scene has N problem(s), listed above``
and exit code 1.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence

import numpy as np
import yaml
from shapely.errors import ShapelyError

from ..project.config import load_project
from ..project.layout import scene_rooms
from ..project.materials import declared_types, load_materials
from ..project.splits import SplitError, load_splits
from ..scene.inventory import inventory
from ..scene.reader import read_scene
from ..scene.validate import validate_scene
from ._folder import REPO_ROOT, check_folder


class ScanError(ValueError):
    """The scan command cannot proceed from these inputs."""


def _extent(prims) -> tuple[float, float, float] | None:
    """The x, y and z extent in metres of every readable surface vertex, or
    None when there is none (validation then names the scene as empty)."""
    blocks = []
    for prim in prims:
        if not prim.is_surface:
            continue
        try:
            pts = prim.points
        except ValueError:
            continue  # a malformed polygon; validation names it
        if pts is not None and len(pts) and np.isfinite(pts).all():
            blocks.append(np.asarray(pts, dtype=float).reshape(-1, 3))
    if not blocks:
        return None
    allpts = np.vstack(blocks)
    span = allpts.max(axis=0) - allpts.min(axis=0)
    return float(span[0]), float(span[1]), float(span[2])


def _print_materials(prims, roles, *, out) -> None:
    print("materials:", file=out)
    print(f"  {'modifier':<28} {'type':<10} {'surfaces':>8} {'area m2':>10}  role", file=out)
    for row in inventory(prims):
        role = "(no materials.csv)" if roles is None else roles.get(row.name, "(not in materials.csv)")
        print(f"  {row.name:<28} {row.ptype or '-':<10} {row.surfaces:>8} {row.area_m2:>10.2f}  {role}",
              file=out)


def _print_storeys(detection, *, out) -> None:
    by_storey: dict[int, list] = {}
    for room in detection.rooms:
        by_storey.setdefault(room.storey, []).append(room)
    for storey in sorted(by_storey):
        rooms = by_storey[storey]
        level = min(room.level_z for room in rooms)
        area = sum(room.area_m2 for room in rooms)
        print(f"storey {storey}: level {level:.2f} m, {len(rooms)} room(s), {area:.2f} m2", file=out)
    print(f"room warnings: {len(detection.warnings)}", file=out)
    for warning in detection.warnings:
        print(f"  - {warning}", file=out)


def scan_folder(folder, *, out=None) -> int:
    """Check ``folder``'s scene and print what is in it. Returns the number
    of scene problems found (0 means the scene passed)."""
    out = sys.stdout if out is None else out
    folder = pathlib.Path(folder)
    check_folder(folder, REPO_ROOT, ScanError)
    if not (folder / "project.yaml").is_file():
        raise ScanError(f"{folder}: missing required file 'project.yaml'")

    project = load_project(folder / "project.yaml")
    prims = read_scene(project.scene.files, project.scene.units)
    materials_path = folder / "materials.csv"
    roles = load_materials(materials_path) if materials_path.is_file() else None

    for path in project.scene.files:
        print(f"scene file: {path}", file=out)
    print(f"surfaces: {sum(1 for p in prims if p.is_surface)}", file=out)
    extent = _extent(prims)
    if extent is not None:
        print(f"size: x {extent[0]:.2f} m, y {extent[1]:.2f} m, z {extent[2]:.2f} m "
              f"(scene.units: {project.scene.units})", file=out)
    _print_materials(prims, roles, out=out)

    report = validate_scene(
        prims, project.scene.units, roles,
        declared_types=declared_types(materials_path) if roles is not None else None,
    )
    print(f"warnings: {len(report.warnings)}", file=out)
    for warning in report.warnings:
        print(f"  - {warning}", file=out)
    print(f"problems: {len(report.errors)}", file=out)
    for error in report.errors:
        print(f"  - {error}", file=out)

    if report.errors or roles is None:
        return len(report.errors)
    try:
        detection = scene_rooms(project, prims, roles)
    except SplitError as exc:
        print("split problems: 1", file=out)
        print(f"  - {exc}", file=out)
        return len(report.errors) + 1
    splits = load_splits(project.root)
    if splits is not None:
        print(f"split lines: {len(splits)}", file=out)
    _print_storeys(detection, out=out)
    return len(report.errors)


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog or "python -m leed_daylight.drivers.scan",
        description="Check the Radiance scene and list its materials, storeys and rooms.",
    )
    parser.add_argument("folder", help="the project folder (project.yaml; materials.csv optional)")
    args = parser.parse_args(argv)
    try:
        problems = scan_folder(args.folder)
    except (ValueError, OSError, yaml.YAMLError, ShapelyError) as exc:
        text = " ".join(line.strip() for line in str(exc).splitlines() if line.strip())
        print(f"error: {text or type(exc).__name__}", file=sys.stderr)
        return 1
    if problems:
        print(f"error: the scene has {problems} problem(s), listed above", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
