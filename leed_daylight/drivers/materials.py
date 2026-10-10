# leed_daylight/drivers/materials.py
"""The ``materials`` verb (CLI spec 4.3): write a materials.csv template.

    leed-daylight materials "<project folder>"

It reads the scene ``project.yaml`` names and writes
``<folder>/_run/materials_template.csv`` in the exact materials.csv header,
one row per modifier a surface uses, largest area first. The ``role``
column is empty in every row, so ``load_materials`` refuses the template
until a person has chosen every role (``scene.inventory``: a guessed role
in a LEED submission is a substitution). The guess and its evidence go in
``note``. It never opens or writes ``<folder>/materials.csv``.
"""
from __future__ import annotations

import argparse
import csv
import io
import pathlib
import sys
from collections.abc import Sequence

import yaml
from shapely.errors import ShapelyError

from ..engine.jobs import checked_dir, replace_file
from ..project.config import load_project
from ..scene.inventory import inventory
from ..scene.reader import read_scene
from ._folder import REPO_ROOT, check_folder

#: The materials.csv header ``project.materials`` requires, in order.
TEMPLATE_HEADER = ("modifier", "primitive_type", "role", "note")


class MaterialsCommandError(ValueError):
    """The materials command cannot proceed from these inputs."""


def _note(row) -> str:
    guess = row.guessed_role or "none"
    note = f"guess: {guess}; {row.surfaces} surfaces, {row.area_m2:.2f} m2"
    if not row.ptype:
        # A surface names this modifier but the scene never defines it;
        # `scan` refuses such a scene by name.
        note += "; not defined in the scene"
    return note


def materials_folder(folder, *, out=None) -> pathlib.Path:
    """Write ``<folder>/_run/materials_template.csv`` and return its path."""
    out = sys.stdout if out is None else out
    folder = pathlib.Path(folder)
    check_folder(folder, REPO_ROOT, MaterialsCommandError)
    if not (folder / "project.yaml").is_file():
        raise MaterialsCommandError(f"{folder}: missing required file 'project.yaml'")

    project = load_project(folder / "project.yaml")
    prims = read_scene(project.scene.files, project.scene.units)
    rows = inventory(prims)

    # Never through a link at _run or at the template (security review H1's class).
    checked_dir(folder, "_run")
    handle = io.StringIO(newline="")
    writer = csv.writer(handle)
    writer.writerow(TEMPLATE_HEADER)
    for row in rows:
        writer.writerow([row.name, row.ptype, "", _note(row)])
    template = folder / "_run" / "materials_template.csv"
    replace_file(folder, template, handle.getvalue().encode("utf-8"))

    for row in rows:
        print(f"  {row.name:<28} {row.ptype or '-':<10} {_note(row)}", file=out)
    print(f"template written to {template}", file=out)
    print("copy it to materials.csv and fill every role", file=out)
    return template


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog or "python -m leed_daylight.drivers.materials",
        description="Write a materials.csv template from the materials in the scene.",
    )
    parser.add_argument("folder", help="the project folder (project.yaml)")
    args = parser.parse_args(argv)
    try:
        materials_folder(args.folder)
    except (ValueError, OSError, yaml.YAMLError, ShapelyError) as exc:
        text = " ".join(line.strip() for line in str(exc).splitlines() if line.strip())
        print(f"error: {text or type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
