# leed_mcp/adapters/scene.py
"""Draft-tolerant scene reading for the ``scene`` tool (spec 6.2).

Ported from the browser wizard's ``steps/scene.py`` and ``steps/rooms.py``
(PORTED.md lists each function and its source).
These read only project.yaml's ``scene:`` section (``load_scene_settings``),
the scene files and, where roles are needed, ``materials.csv``, so they
work while other values are still ``FILL_ME``. The CLI drivers
(``drivers/scan.py``, ``materials.py``, ``rooms.py``) call ``load_project``
and refuse a draft, so they are not used.

Two underscore-private pipeline names are imported on purpose: ``scene.reader``'s
``_tokens_of`` and ``_parse_xform``, so the containment check reads the
``!xform`` targets exactly as ``read_scene`` will (pinned by
``tests/mcp/test_mcp_scene.py``).
"""
from __future__ import annotations

import csv
import io
import itertools
import pathlib
import shlex
from collections.abc import Sequence
from typing import Any

from shapely.errors import ShapelyError
from shapely.geometry import Polygon

from leed_daylight.build.groups import GroupError, window_groups
from leed_daylight.drivers.materials import TEMPLATE_HEADER
from leed_daylight.drivers.rooms import SPACES_TEMPLATE_HEADER
from leed_daylight.engine.jobs import checked_dir, replace_file
from leed_daylight.project.config import SceneSpec, load_scene_settings
from leed_daylight.project.layout import SplitHalf, apply_splits
from leed_daylight.project.materials import declared_types, load_materials
from leed_daylight.project.splits import SPLITS_FILE, load_splits
from leed_daylight.scene.input_file import read_input_text
from leed_daylight.scene.inventory import inventory
from leed_daylight.scene.reader import _parse_xform, _tokens_of, read_scene
from leed_daylight.scene.storeys import Room, RoomDetection, detect_rooms
from leed_daylight.scene.validate import validate_scene
from leed_mcp.adapters.input_files import check_cell_text
from leed_mcp.adapters.results_files import refuse_linked
from leed_mcp.envelope import refusal
from leed_mcp.errors import Refused, one_line

__all__ = [
    "MAX_RING_VERTICES", "MAX_TOTAL_VERTICES", "check_inside", "detected", "labelled", "load_scene",
    "materials_template", "rings", "rooms", "scan", "scene_and_roles",
]

#: At most this many vertices per outline or hole ring (spec 6.2). Chosen.
MAX_RING_VERTICES = 500
#: Without ``storey``, at most this many outline vertices in all (spec 6.2). Chosen.
MAX_TOTAL_VERTICES = 2000
#: How deep the containment check follows ``!xform`` includes; the reader
#: refuses a nest deeper than its own cap (``MAX_INCLUDE_DEPTH``, 32).
_MAX_DEPTH = 64


def check_inside(folder: pathlib.Path, files: Sequence[pathlib.Path]) -> None:
    """Refuse (``SCENE_INCLUDE_OUTSIDE``) when a scene file, or any file an
    ``!xform`` reaches, resolves outside ``folder``. Checked before the
    scene is read. A file that cannot be read or parsed is left to
    ``read_scene``, which reports it with its own message."""
    seen: set[pathlib.Path] = set()

    def walk(path: pathlib.Path, depth: int) -> None:
        resolved = path.resolve()
        if folder not in resolved.parents:
            raise Refused(refusal(
                "SCENE_INCLUDE_OUTSIDE",
                f"{path}: the scene reads a file outside the project folder ({folder}); every scene "
                "file and !xform target must be inside it."))
        if resolved in seen or depth > _MAX_DEPTH or not resolved.is_file():
            return
        seen.add(resolved)
        try:
            tokens = list(_tokens_of(read_input_text(resolved), resolved))
        except (OSError, ValueError):
            return
        for kind, token, _ in tokens:
            if kind != "!":
                continue
            try:
                parts = shlex.split(token)
                if not parts or parts[0] != "xform":
                    continue
                _, _, targets = _parse_xform(parts[1:])
            except (ValueError, TypeError, IndexError):
                continue
            for target in targets:
                walk(path.parent / target, depth + 1)

    for file in files:
        walk(pathlib.Path(file), 0)


def load_scene(folder: pathlib.Path) -> tuple[SceneSpec, list]:
    """``(scene settings, primitives)``: project.yaml's ``scene:`` only, the
    containment check, then the reader. ``FILE_MISSING`` without project.yaml."""
    path = folder / "project.yaml"
    refuse_linked(folder, path)
    if not path.is_file():
        raise Refused(refusal("FILE_MISSING", f"{folder}: no project.yaml; run project(action='create') first.",
                              field="folder"))
    spec = load_scene_settings(path)
    check_inside(folder, spec.files)
    return spec, read_scene(spec.files, spec.units)


def scene_and_roles(folder: pathlib.Path) -> tuple[list, dict[str, str]]:
    """The scene's primitives and materials.csv's roles, once the scene has
    no problem: what room detection needs. ``FILE_MISSING`` without
    materials.csv; ``INPUT_INVALID`` while ``validate_scene`` finds errors."""
    spec, prims = load_scene(folder)
    path = folder / "materials.csv"
    refuse_linked(folder, path)
    if not path.is_file():
        raise Refused(refusal("FILE_MISSING", f"{folder}: rooms need materials.csv with every role set; "
                              "write it first (scene(action='materials_template') gives the rows).",
                              field="folder"))
    roles = load_materials(path)
    errors = validate_scene(prims, spec.units, roles, declared_types=declared_types(path)).errors
    if errors:
        raise Refused(refusal("INPUT_INVALID", f"the scene has {len(errors)} problem(s), the first: "
                              f"{errors[0]}; scene(action='scan') lists them all."))
    return prims, roles


def detected(folder: pathlib.Path) -> tuple[list, dict[str, str], RoomDetection]:
    """``(primitives, roles, rooms)``: the rooms every verb finds, with
    ``splits.csv``'s lines applied (``project.layout.scene_rooms``)."""
    prims, roles = scene_and_roles(folder)
    return prims, roles, _detect(folder, prims, roles)


def _detect(folder: pathlib.Path, prims, roles) -> RoomDetection:
    """Rooms with ``splits.csv``'s lines applied (wizard ``detected_rooms``)."""
    detection = detect_rooms(prims, roles)
    refuse_linked(folder, folder / SPLITS_FILE)
    lines = load_splits(folder)
    return apply_splits(detection, lines, prims, roles) if lines else detection


def _centroid(room: Room) -> tuple[float, float]:
    point = Polygon(room.outline).centroid
    return point.x, point.y


def labelled(rooms: Sequence[Room]) -> list[tuple[str, Room]]:
    """``R1``, ``R2``, ... in (storey, level, centroid x, centroid y) order,
    as ``drivers/rooms.py`` labels them."""
    ordered = sorted(rooms, key=lambda r: (r.storey, r.level_z, *_centroid(r)))
    return [(f"R{i + 1}", room) for i, room in enumerate(ordered)]


def _relative(folder: pathlib.Path, path: pathlib.Path) -> str:
    return pathlib.Path(path).resolve().relative_to(folder).as_posix()


def scan(folder: pathlib.Path) -> dict[str, Any]:
    """The scene check (wizard ``_problems`` and ``_table``; storeys as
    ``drivers/scan.py`` prints them). ``storeys`` is None without a usable
    ``materials.csv`` or while the scene has problems."""
    spec, prims = load_scene(folder)
    path = folder / "materials.csv"
    roles = declared = None
    problems: list[str] = []
    if path.is_file():
        try:
            refuse_linked(folder, path)
            roles, declared = load_materials(path), declared_types(path)
        except (OSError, ValueError) as exc:
            problems.append(one_line(exc))
    problems += validate_scene(prims, spec.units, roles, declared_types=declared).errors
    storeys, warnings = None, []
    if roles is not None and not problems:
        try:
            detection = _detect(folder, prims, roles)
        except (OSError, ValueError, ShapelyError) as exc:
            problems.append(one_line(exc))
        else:
            warnings = list(detection.warnings)
            storeys = []
            for index, group in itertools.groupby(sorted(detection.rooms, key=lambda r: r.storey),
                                                  key=lambda r: r.storey):
                group = list(group)
                storeys.append({"index": index, "level_z": round(min(r.level_z for r in group), 3),
                                "rooms": len(group), "area_m2": round(sum(r.area_m2 for r in group), 2)})
    materials = [{
        "modifier": info.name, "primitive_type": info.ptype or None, "surfaces": info.surfaces,
        "area_m2": round(info.area_m2, 2), "guessed_role": info.guessed_role or None,
        "role": roles.get(info.name) if roles is not None else None,
    } for info in inventory(prims)]
    return {"units": spec.units, "files": [_relative(folder, f) for f in spec.files],
            "surfaces": sum(1 for p in prims if p.is_surface), "materials": materials,
            "storeys": storeys, "problems": problems, "warnings": warnings,
            "has_materials": roles is not None}


def _note(info) -> str:
    """Ported from ``drivers/materials.py`` ``_note``."""
    note = f"guess: {info.guessed_role or 'none'}; {info.surfaces} surfaces, {info.area_m2:.2f} m2"
    return note + ("; not defined in the scene" if not info.ptype else "")


def checked_modifiers(rows) -> None:
    """Refuse (``INPUT_INVALID``) a scene modifier name that would be a
    formula or break a line in the materials CSV files; the person renames
    it in the model."""
    for info in rows:
        try:
            check_cell_text(info.name, "the scene", "modifier name")
        except Refused:
            raise Refused(refusal(
                "INPUT_INVALID", f"modifier {info.name!r} cannot be written to a CSV file safely "
                "(it starts with =, +, - or @, or holds a control character or '<!--'); "
                "rename the modifier in the model.", field="rows")) from None


def materials_template(folder: pathlib.Path) -> dict[str, Any]:
    """Write ``_run/materials_template.csv`` byte for byte as
    ``leed-daylight materials`` does, and return its rows."""
    _, prims = load_scene(folder)
    rows = inventory(prims)
    checked_modifiers(rows)
    checked_dir(folder, "_run")
    handle = io.StringIO(newline="")
    writer = csv.writer(handle)
    writer.writerow(TEMPLATE_HEADER)
    for info in rows:
        writer.writerow([info.name, info.ptype, "", _note(info)])
    written = replace_file(folder, folder / "_run" / "materials_template.csv", handle.getvalue().encode("utf-8"))
    return {"file": str(written), "rows": [
        {"modifier": info.name, "primitive_type": info.ptype or None, "note": _note(info)} for info in rows]}


def _ring(points) -> list[list[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in points]


def rings(room: Room, max_vertices: int = MAX_RING_VERTICES) -> tuple[list, list, bool]:
    """``(outline, holes, simplified)`` in metres rounded to 1 mm, each ring
    at most ``max_vertices`` long: a longer one is simplified (shapely,
    topology kept) with a tolerance doubled from 1 mm until it fits."""
    if max(len(r) for r in (room.outline, *room.holes)) <= max_vertices:
        return _ring(room.outline), [_ring(h) for h in room.holes], False
    polygon, tolerance = Polygon(room.outline, list(room.holes)), 0.001
    while True:
        simple = polygon.simplify(tolerance, preserve_topology=True)
        outline = list(simple.exterior.coords)[:-1]
        holes = [list(ring.coords)[:-1] for ring in simple.interiors]
        if max(len(r) for r in (outline, *holes)) <= max_vertices:
            return _ring(outline), [_ring(h) for h in holes], True
        tolerance *= 2


def _write_spaces_template(folder: pathlib.Path, rooms_: Sequence[tuple[str, Room]]) -> pathlib.Path:
    """``_run/spaces_template.csv`` byte for byte as ``leed-daylight rooms``
    writes it (ported from ``drivers/rooms.py`` ``_write_template``)."""
    checked_dir(folder, "_run")
    handle = io.StringIO(newline="")
    writer = csv.writer(handle)
    writer.writerow(SPACES_TEMPLATE_HEADER)
    for label, room in rooms_:
        cx, cy = _centroid(room)
        writer.writerow([f"L{room.storey}", label, "office", "yes", f"{room.area_m2:.2f}", f"{cx:.2f}",
                         f"{cy:.2f}", f"{room.level_z:.2f}", "no", "", "1", ""])
    return replace_file(folder, folder / "_run" / "spaces_template.csv", handle.getvalue().encode("utf-8"))


def _opening(prim, kind: str) -> dict[str, Any]:
    """A pane in plan: a facade window as the segment between its two
    farthest-apart vertices, any roof kind as its polygon."""
    plan = [(float(x), float(y)) for x, y, _ in prim.points]
    if kind != "window":
        return {"name": prim.name, "shape": "polygon", "points": _ring(plan)}
    far = max(itertools.combinations(plan, 2), key=lambda p: (p[0][0] - p[1][0]) ** 2 + (p[0][1] - p[1][1]) ** 2)
    return {"name": prim.name, "shape": "segment", "points": _ring(far)}


def rooms(folder: pathlib.Path, storey: int | None) -> dict[str, Any]:
    """Rooms with splits applied, their panes and the split lines (wizard
    ``plan``); writes ``_run/spaces_template.csv``. Needs ``materials.csv``."""
    prims, roles = scene_and_roles(folder)
    detection = _detect(folder, prims, roles)
    named = labelled(detection.rooms)
    template = _write_spaces_template(folder, named)
    warnings = list(detection.warnings)
    try:
        groups, more = window_groups(prims, roles, dict(named), all_rooms=[room for _, room in named])
        warnings += more
    except GroupError as exc:
        groups = []
        warnings.append(one_line(exc))
    storeys = sorted({room.storey for _, room in named})
    if storey is not None and storey not in storeys:
        raise Refused(refusal("INPUT_INVALID", f"storey {storey} is not in the scene; its storeys are "
                              f"{', '.join(map(str, storeys)) or 'none'}.", field="storey"))
    out_rooms, budget, truncated = [], 0, False
    for label, room in named:
        if storey is not None and room.storey != storey:
            continue
        outline, holes, simplified = rings(room)
        size = len(outline) + sum(len(h) for h in holes)
        if storey is None and (truncated or budget + size > MAX_TOTAL_VERTICES):
            outline, holes, truncated = None, None, True
        else:
            budget += size
        cx, cy = _centroid(room)
        out_rooms.append({
            "label": label, "room_id": room.room_id, "storey": room.storey,
            "level_z": round(room.level_z, 3), "area_m2": round(room.area_m2, 2),
            "centroid": [round(cx, 3), round(cy, 3)], "outline": outline, "holes": holes,
            "split": isinstance(room, SplitHalf), "simplified": simplified,
        })
    panes = [{
        "group_id": group.group_id, "room": group.room, "storey": group.storey, "kind": group.kind,
        "area_m2": round(group.area_m2, 2), "openings": [_opening(p, group.kind) for p in group.openings],
    } for group in groups if storey is None or group.storey == storey]
    lines = load_splits(folder) or ()
    return {"storeys": storeys, "rooms": out_rooms, "panes": panes, "warnings": warnings,
            "truncated": truncated, "template": str(template),
            "split_lines": [{"level_z": round(line.level_z, 3), "points": _ring(line.points),
                             "note": line.note} for line in lines]}
