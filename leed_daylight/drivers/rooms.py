# leed_daylight/drivers/rooms.py
"""The `rooms` verb (layer 7, task ROOMS): list the rooms `detect_rooms`
finds in a Radiance model, after the lines of splits.csv when that file
exists (``project.layout.scene_rooms``), and write a starter spaces.csv
template a person edits into the real spaces.csv. Each half of a split
room gets its own row, with its own centroid.

spaces.csv (`project.spaces.load_spaces`) is where a person records a
room's name, type and occupancy, matched back onto a detected room by its
centroid and level -- `project.spaces`'s own docstring explains why: a
detected room's own id is a sort-order index, reassigned from scratch on
every scan, so persisting an edit by that index would silently attach it to
the wrong room later. Before this command existed nothing showed a person
which centroid and level to type in; this prints them and writes a template
with a placeholder label (`R1`, `R2`, ...) per row, ready to rename::

    .venv/bin/python -m leed_daylight.drivers.rooms "<project folder>"

The folder holds `project.yaml` and `materials.csv` (unlike `run.py`,
`spaces.csv` is NOT required -- this command exists to help write it, and
never reads or overwrites an existing one). ``rooms_folder`` runs the same
intake chain as ``run.py``, as far as ``scene_rooms``: ``load_project`` ->
``read_scene`` -> ``load_materials`` -> ``validate_scene(...)
.raise_if_errors()`` BEFORE ``scene_rooms``. With splits.csv, the first
printed line says how many split lines were applied. ``main(argv)`` is the thin CLI
wrapper: it parses arguments, calls ``rooms_folder``, and turns any chain
refusal -- and any malformed YAML, shapely geometry failure or file-system
error on the way -- into a one-line ``error: ...`` on stderr and exit code
1 -- never a traceback.
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
from shapely.geometry import Polygon

from ..engine.jobs import checked_dir, replace_file
from ..project.config import load_project
from ..project.layout import scene_rooms
from ..project.materials import declared_types, load_materials
from ..project.splits import SPLITS_FILE, load_splits
from ..scene.reader import read_scene
from ..scene.storeys import Room
from ..scene.validate import validate_scene
from ._folder import REPO_ROOT, check_folder

#: The header ``project.spaces.load_spaces`` accepts, with a single
#: ``include_<case>`` column named ``base`` -- this template always
#: describes one case, which a person can rename or extend once they open
#: it. Every value in a row is invented from the detected room's own
#: geometry, never a client default: `floor`, `room` and `area_m2`/
#: `centroid_x`/`centroid_y`/`level_z` come straight from the scan;
#: `space_type`, `regularly_occupied`, `view_preserving_shades` and
#: `include_base` are fixed placeholders a person is expected to edit.
SPACES_TEMPLATE_HEADER = (
    "floor", "room", "space_type", "regularly_occupied", "area_m2",
    "centroid_x", "centroid_y", "level_z", "view_preserving_shades",
    "merge_into", "include_base", "reason",
)

#: The two files a project folder must hold for this command (checked, and
#: named, in this order). Unlike `run.py`'s own `_REQUIRED_FILES`,
#: `spaces.csv` is deliberately absent: this command exists to help write
#: one.
_REQUIRED_FILES = ("project.yaml", "materials.csv")


class RoomsError(ValueError):
    """The rooms command cannot proceed from these inputs."""


def _check_folder(folder: pathlib.Path) -> None:
    check_folder(folder, REPO_ROOT, RoomsError)


def _check_required_files(folder: pathlib.Path) -> None:
    for name in _REQUIRED_FILES:
        if not (folder / name).is_file():
            raise RoomsError(f"{folder}: missing required file {name!r}")


def _centroid(room: Room) -> tuple[float, float]:
    """The 2-D outline centroid, via shapely -- the same value
    `project.spaces.match_to_rooms` computes from a room's own outline when
    it matches a space back onto it."""
    point = Polygon(room.outline).centroid
    return point.x, point.y


def _sort_key(room: Room) -> tuple[int, float, float, float]:
    cx, cy = _centroid(room)
    return (room.storey, room.level_z, cx, cy)


def _labelled_rooms(rooms: Sequence[Room]) -> list[tuple[str, Room]]:
    """Every room, sorted by storey then level then centroid (x, y), paired
    with its label (`R1`, `R2`, ...) in that sorted order."""
    ordered = sorted(rooms, key=_sort_key)
    return [(f"R{i + 1}", room) for i, room in enumerate(ordered)]


def _print_rooms(
    labelled: Sequence[tuple[str, Room]], warnings: Sequence[str], *, out,
) -> None:
    for label, room in labelled:
        cx, cy = _centroid(room)
        print(
            f"{label}: storey {room.storey}, area {room.area_m2:.2f} m2, "
            f"centroid ({cx:.2f}, {cy:.2f}), level_z {room.level_z:.2f}",
            file=out,
        )
    print(f"warnings: {len(warnings)}", file=out)
    for warning in warnings:
        print(f"  - {warning}", file=out)


def _write_template(folder: pathlib.Path, labelled: Sequence[tuple[str, Room]]) -> pathlib.Path:
    """Write `<folder>/_run/spaces_template.csv`, overwriting any previous
    template -- never `<folder>/spaces.csv` itself, which this function
    never opens. Never through a link at `_run` or at the template
    (security review H1's class)."""
    checked_dir(folder, "_run")
    f = io.StringIO(newline="")
    writer = csv.writer(f)
    writer.writerow(SPACES_TEMPLATE_HEADER)
    for label, room in labelled:
        cx, cy = _centroid(room)
        writer.writerow([
            f"L{room.storey}", label, "office", "yes",
            f"{room.area_m2:.2f}", f"{cx:.2f}", f"{cy:.2f}", f"{room.level_z:.2f}",
            "no", "", "1", "",
        ])
    template_path = folder / "_run" / "spaces_template.csv"
    replace_file(folder, template_path, f.getvalue().encode("utf-8"))
    return template_path


def rooms_folder(folder, *, out=sys.stdout) -> pathlib.Path:
    """List every room `detect_rooms` finds in `folder`, split by
    splits.csv when it exists, and write
    `<folder>/_run/spaces_template.csv`. Returns the template's path.

    Never touches an existing `<folder>/spaces.csv` -- only the template is
    written, overwritten on each run.
    """
    if not isinstance(folder, (str, pathlib.Path)):
        raise RoomsError(
            f"folder: expected a str or pathlib.Path, got "
            f"{type(folder).__name__}: {folder!r}"
        )
    folder = pathlib.Path(folder)
    _check_folder(folder)
    _check_required_files(folder)

    project = load_project(folder / "project.yaml")
    prims = read_scene(project.scene.files, project.scene.units)
    roles = load_materials(folder / "materials.csv")
    validate_scene(
        prims, project.scene.units, roles, declared_types=declared_types(folder / "materials.csv"),
    ).raise_if_errors()
    detection = scene_rooms(project, prims, roles)
    splits = load_splits(project.root)
    if splits is not None:
        print(f"split lines applied: {len(splits)} ({SPLITS_FILE})", file=out)

    labelled = _labelled_rooms(detection.rooms)
    _print_rooms(labelled, detection.warnings, out=out)

    template_path = _write_template(folder, labelled)
    print(f"template written to {template_path}", file=out)
    print("copy it to spaces.csv and edit the room names", file=out)

    return template_path


def _one_line(exc: BaseException) -> str:
    """``exc``'s message with every run of whitespace (newlines included)
    collapsed to one space, so a refusal is always a single line."""
    return " ".join(str(exc).split())


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog or "python -m leed_daylight.drivers.rooms",
        description=(
            "List the rooms detected in a Radiance model and write a "
            "starter spaces.csv template."
        ),
    )
    parser.add_argument(
        "folder", help="the project folder (project.yaml, materials.csv)"
    )
    args = parser.parse_args(argv)

    try:
        rooms_folder(args.folder)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except yaml.YAMLError as exc:
        print(f"error: project.yaml is not valid YAML: {_one_line(exc)}", file=sys.stderr)
        return 1
    except ShapelyError as exc:
        print(
            f"error: geometry failure while detecting rooms ({type(exc).__name__}): "
            f"{_one_line(exc)}",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(f"error: file system: {_one_line(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
