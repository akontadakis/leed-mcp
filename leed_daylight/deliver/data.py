# leed_daylight/deliver/data.py
"""One case's results, ready for the reports (reports spec 4, 5).

``load_case`` reads each option's results file and ``points.csv``, refuses
results whose input fingerprints no longer match the files on disk (that
option is skipped, by name), and adds what the reports need from the project:
the storeys and room outlines from the scene, the window groups and the
glazing record from a model prepared afresh into a temporary folder.

Pre-flight ruling F7: this deliberately does not read ``_run/manifest.json``
(spec 4's own wording). ``_run`` holds whichever case ran last, so its
manifest can describe a different case than the one being loaded here;
``prepare_case_model`` below runs ``build.prepare`` again, with no Radiance
involved, into a throwaway temporary folder instead. That costs a few
seconds on a large model but is correct for every case, not just the last
one run.

Merge-split spec 7: the storeys list space rows only (a piece row is part of
its space); a space's outlines are those of every piece, its label sits on
its largest piece, and the edge two halves of a split room share is kept as
a split line to draw dashed. ``layout_note`` gives the one sentence the
reports add for a space that is not one detected room as found.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import pathlib
import tempfile
import types
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from shapely.geometry import Polygon

from ..build.prepare import PreparedModel, prepare, split_hint
from ..constants import TOOL_VERSION
from ..project.config import Project, load_project
from ..project.inputs import stale_inputs
from ..project.layout import LaidOutSpace, SpaceLayout, SplitHalf, scene_rooms
from ..project.materials import declared_types, load_materials
from ..project.rules import load_rules
from ..project.spaces import (
    VIEWS_PREPARE_CASE, Space, case_names, load_spaces, match_to_rooms, space_rows, views_case_names,
    views_prepare_spaces,
)
from ..project.splits import SPLITS_FILE
from ..scene.reader import read_scene
from ..scene.storeys import Room
from ..scene.validate import validate_scene

OPTIONS = ("option1", "option2", "views")
_RESULTS = {
    "option1": ("option1", "results.json"),
    "option2": ("option2", "option2_results.json"),
    "views": ("views", "views_results.json"),
}
COMMANDS = {
    "option1": "python -m leed_daylight.drivers.run",
    "option2": "python -m leed_daylight.drivers.option2",
    "views": "python -m leed_daylight.drivers.views",
}


class DeliverError(ValueError):
    """The reports or the package cannot be made from these results."""


def result_path(folder, option: str, case: str) -> pathlib.Path:
    sub, name = _RESULTS[option]
    return pathlib.Path(folder) / "_run" / sub / case / name


@dataclasses.dataclass(frozen=True, eq=False)
class PointTable:
    rooms: tuple[str, ...]
    xy: np.ndarray
    columns: Mapping[str, np.ndarray]

    def mask(self, rooms: Iterable[str]) -> np.ndarray:
        wanted = set(rooms)
        return np.array([r in wanted for r in self.rooms], dtype=bool)


def _column(values: list[str]) -> np.ndarray:
    if values and all(v in ("yes", "no") for v in values):
        return np.array([v == "yes" for v in values], dtype=bool)
    try:
        return np.array([float(v) if v != "" else np.nan for v in values], dtype=float)
    except ValueError:
        return np.array(values, dtype=object)


def read_json_object(path) -> dict:
    """The JSON object in ``path``; a :class:`DeliverError` naming the file
    for text that is not UTF-8, is not valid JSON (a truncated file from a
    killed run, say) or is not an object at the top level (final review I5)."""
    path = pathlib.Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise DeliverError(f"{path}: not UTF-8 text ({exc.reason} at byte {exc.start})") from None
    except json.JSONDecodeError as exc:
        raise DeliverError(f"{path}: not valid JSON ({exc})") from None
    if not isinstance(payload, dict):
        raise DeliverError(f"{path}: expected a JSON object at the top level, found a {type(payload).__name__}")
    return payload


def read_points(path) -> PointTable:
    """The per-point table in ``path``. A row with fewer or more fields than
    the header, or a non-numeric ``x``/``y``, is refused naming the file and
    the line (final review I5)."""
    path = pathlib.Path(path)
    if not path.is_file():
        raise DeliverError(f"{path}: not found")
    rows, lines = [], []
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            for row in reader:
                rows.append(row)
                lines.append(reader.line_num)
    except UnicodeDecodeError as exc:
        raise DeliverError(f"{path}: not UTF-8 text ({exc.reason})") from None
    except csv.Error as exc:
        raise DeliverError(f"{path}: not a readable CSV file ({exc})") from None
    for need in ("room", "x", "y"):
        if need not in header:
            raise DeliverError(f"{path}: no {need!r} column")
    xy = np.empty((len(rows), 2), dtype=float)
    for i, (row, line) in enumerate(zip(rows, lines)):
        if None in row:
            raise DeliverError(f"{path}, line {line}: more fields than the header's {len(header)}")
        if any(v is None for v in row.values()):
            raise DeliverError(f"{path}, line {line}: fewer fields than the header's {len(header)}")
        for j, name in enumerate(("x", "y")):
            try:
                xy[i, j] = float(row[name])
            except ValueError:
                raise DeliverError(f"{path}, line {line}: {name!r} is {row[name]!r}, not a number") from None
    rooms = tuple(r["room"] for r in rows)
    columns = {name: _column([r[name] for r in rows]) for name in header if name not in ("room", "x", "y")}
    return PointTable(rooms, xy, types.MappingProxyType(columns))


@dataclasses.dataclass(frozen=True, eq=False)
class OptionData:
    name: str
    results: Mapping
    points: PointTable
    folder: pathlib.Path
    results_path: pathlib.Path


@dataclasses.dataclass(frozen=True, eq=False)
class StoreyPlan:
    """One storey of the plans. ``rooms`` are its space rows (spaces.csv
    order); ``outlines[name]`` every ring of every piece of that space;
    ``centroids[name]`` where its name is drawn; ``split_lines[name]`` the
    drawn line of every cut that made a split piece of it (only for a space
    with one)."""
    label: str
    level_z: float
    rooms: tuple[str, ...]
    outlines: Mapping[str, tuple[np.ndarray, ...]]
    centroids: Mapping[str, tuple[float, float]]
    split_lines: Mapping[str, tuple[np.ndarray, ...]] = dataclasses.field(default_factory=lambda: types.MappingProxyType({}))


@dataclasses.dataclass(frozen=True, eq=False)
class GroupLine:
    """One window group as the reports draw and list it (skylights spec 7).

    ``segments`` holds the plan line of each facade (``perimeter``) pane;
    ``outlines`` one plan outline per roof pane (its cleaned vertices in
    drawn order; the two plan ends for a vertical roof-monitor pane), each
    drawn dashed.
    ``azimuth_deg`` is ``None`` for skylight and diffusing groups.
    ``openings`` counts every pane; ``tilt_classes`` lists the distinct tilt
    classes of the panes, in pane order."""
    group_id: str
    room: str
    storey_label: str
    azimuth_deg: float | None
    segments: tuple[np.ndarray, ...]
    kind: str = "window"
    tilt_deg: float = 90.0
    outlines: tuple[tuple[tuple[float, float], ...], ...] = ()
    openings: int = 0
    tilt_classes: tuple[str, ...] = ()
    #: The scene glass materials of the group's panes, in pane order, each once (calculator spec D3).
    materials: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, eq=False)
class CaseData:
    folder: pathlib.Path
    case: str
    created: datetime.date
    project: Project
    rules: Mapping
    spaces: tuple[Space, ...]
    storeys: tuple[StoreyPlan, ...]
    groups: tuple[GroupLine, ...]
    glazing: Mapping[str, Mapping]
    options: Mapping[str, OptionData]
    skipped: Mapping[str, str]
    warnings: tuple[str, ...]
    #: Space name to its ``layout_note``, only for a space that is split or
    #: merged: with the blind clause (Option 1, the atlas) -- but only when
    #: the space has at least one window group that can carry a blind; a
    #: `diffusing` group, or a space with no group, gets the plain sentence
    #: instead (controller ruling, pre-flight F2).
    layout_notes: Mapping[str, str] = dataclasses.field(default_factory=lambda: types.MappingProxyType({}))
    #: ...and without it (Option 2 and Quality Views, which use no blinds).
    layout_notes_no_blinds: Mapping[str, str] = dataclasses.field(default_factory=lambda: types.MappingProxyType({}))
    #: ...and with no grid clause either (controller ruling, pre-flight
    #: F2b), for a space this case has no grid for (excluded from it): the
    #: atlas uses this map instead of the two above for such a space.
    layout_notes_no_grid: Mapping[str, str] = dataclasses.field(default_factory=lambda: types.MappingProxyType({}))

    def storey(self, label: str) -> StoreyPlan:
        for plan in self.storeys:
            if plan.label == label:
                return plan
        raise DeliverError(f"no storey {label!r}; storeys are {', '.join(s.label for s in self.storeys)}")

    def storey_of(self, room: str) -> str:
        for plan in self.storeys:
            if room in plan.rooms:
                return plan.label
        raise DeliverError(f"room {room!r} is not in spaces.csv")

    def storeys_for(self, option: str) -> tuple[StoreyPlan, ...]:
        rooms = set(self.options[option].points.rooms)
        return tuple(s for s in self.storeys if rooms & set(s.rooms))


def _label_point(space: Space, laid: LaidOutSpace) -> tuple[float, float]:
    """Where a space's name is drawn: its spaces.csv centroid for one piece;
    for several, the centroid of the largest piece's outline (the first on a
    tie), or a point inside that piece when the centroid falls outside it."""
    if len(laid.pieces) == 1:
        return float(space.centroid_x), float(space.centroid_y)
    largest = max(laid.pieces, key=lambda piece: piece.area_m2)
    point = Polygon(largest.outline).centroid
    if not largest.footprint.covers(point):
        point = largest.footprint.representative_point()
    return float(point.x), float(point.y)


def _split_lines(laid: LaidOutSpace) -> tuple[np.ndarray, ...]:
    """The drawn line of every cut of the space's split pieces, in piece
    order: the arrays themselves, so a line two pieces share is one object."""
    return tuple(cut.drawn for piece in laid.pieces if isinstance(piece, SplitHalf) for cut in piece.cuts)


def _storeys(spaces: Sequence[Space], layout: SpaceLayout) -> tuple[StoreyPlan, ...]:
    by_floor: dict[str, list[Space]] = {}
    for space in space_rows(spaces):
        by_floor.setdefault(space.floor, []).append(space)
    plans = []
    for label, members in by_floor.items():
        laid = {s.room: layout.space(s.room) for s in members}
        outlines = {
            name: tuple(
                np.asarray(ring, dtype=float)[:, :2]
                for piece in space.pieces for ring in (piece.outline, *piece.holes)
            )
            for name, space in laid.items()
        }
        lines = {name: _split_lines(space) for name, space in laid.items() if _split_lines(space)}
        plans.append(StoreyPlan(
            label=label, level_z=min(float(s.level_z) for s in members),
            rooms=tuple(s.room for s in members), outlines=types.MappingProxyType(outlines),
            centroids=types.MappingProxyType({s.room: _label_point(s, laid[s.room]) for s in members}),
            split_lines=types.MappingProxyType(lines),
        ))
    return tuple(sorted(plans, key=lambda p: (p.level_z, p.label)))


#: A count of pieces in a layout note: two to nine in words, above nine in digits.
_COUNT_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def _lines_text(rows: Sequence[int]) -> str:
    """``line 2``, ``lines 2 and 3``, ``lines 2, 3 and 4``."""
    if len(rows) == 1:
        return f"line {rows[0]}"
    return "lines " + ", ".join(str(row) for row in rows[:-1]) + f" and {rows[-1]}"


def layout_note(space: LaidOutSpace, *, blinds: bool = True, grid: bool = True) -> str | None:
    """The one sentence the reports add for a split or merged space
    (merge-split spec 7); ``None`` for a space that is one detected room.
    ``blinds=False`` leaves out the blind clause, for a report whose option
    uses no blinds (Option 2, Quality Views). ``grid=False`` (controller
    ruling, pre-flight F2b) leaves out the grid clause too, and with it the
    blind clause, for a space with no grid in this case (excluded from it,
    or included but not regularly occupied); a split space's sentence,
    which states neither, is unaffected."""
    if space.origin == "detected":
        return None
    count = _COUNT_WORDS.get(len(space.pieces), str(len(space.pieces)))
    rows = sorted({cut.row for piece in space.pieces if isinstance(piece, SplitHalf) for cut in piece.cuts})
    if space.origin == "split":
        return f"{space.name} was split from a detected room by {_lines_text(rows)} of {SPLITS_FILE}."
    if not grid:
        tail = "."
    elif blinds:
        tail = "; each keeps its own grid, and blinds respond to the whole space."
    else:
        tail = "; each keeps its own grid."
    if space.origin == "merged":
        return f"{space.name} is {count} detected rooms treated as one space{tail}"
    # Review M5: the halves may come from more than one detected room; a
    # half's original room is its source outline, shared by identity.
    sources = {id(piece.source_outline) for piece in space.pieces if isinstance(piece, SplitHalf)}
    source = "a detected room" if len(sources) == 1 else "detected rooms"
    return (
        f"{space.name} is {count} rooms treated as one space, one or more of them split from "
        f"{source} by {_lines_text(rows)} of {SPLITS_FILE}{tail}"
    )


def _plan_segment(opening) -> np.ndarray:
    """The two plan vertices of an opening that lie farthest apart."""
    verts = np.asarray(opening.fargs, dtype=float).reshape(-1, 3)[:, :2]
    gaps = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=-1)
    i, j = np.unravel_index(int(np.argmax(gaps)), gaps.shape)
    return np.array([verts[i], verts[j]])


def _plan_outline(opening) -> tuple[tuple[float, float], ...]:
    """One roof pane's plan outline: the x, y of its vertices in drawn
    order, a vertex equal to the one before it (or a closing copy of the
    first) dropped. A pane whose plan area is zero (vertical roof-monitor
    glass) gives its two plan ends instead."""
    ring: list[tuple[float, float]] = []
    for x, y, _ in np.asarray(opening.fargs, dtype=float).reshape(-1, 3):
        point = (round(float(x), 9), round(float(y), 9))
        if not ring or point != ring[-1]:
            ring.append(point)
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring.pop()
    twice_area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]))
    if abs(twice_area) <= 1e-12:
        a, b = _plan_segment(opening)
        return ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
    return tuple(ring)


def _group_line(group, storey_label: str) -> GroupLine:
    """One prepared window group as a :class:`GroupLine`. A pane at position
    ``roof`` gets its own dashed outline; every other pane (and every pane
    of a group with no pane classes, as master built them) is a plan
    segment."""
    classes = tuple(group.pane_classes)
    on_roof = [bool(classes) and classes[i].position == "roof" for i in range(len(group.openings))]
    roof = [o for o, r in zip(group.openings, on_roof) if r]
    facade = [o for o, r in zip(group.openings, on_roof) if not r]
    return GroupLine(
        group_id=group.group_id, room=group.room, storey_label=storey_label,
        azimuth_deg=None if group.azimuth_bin_deg is None else float(group.azimuth_bin_deg),
        segments=tuple(_plan_segment(o) for o in facade),
        kind=group.kind, tilt_deg=float(group.tilt_deg),
        outlines=tuple(_plan_outline(o) for o in roof),
        openings=len(group.openings),
        tilt_classes=tuple(dict.fromkeys(c.tilt_class for c in classes)) if classes else ("vertical",),
        materials=tuple(dict.fromkeys(o.modifier for o in group.openings)),
    )


@dataclasses.dataclass(frozen=True)
class CaseModel:
    """What :func:`prepare_case_model` hands back: the model it prepared,
    every space matched to the room it detected on the scene (split lines
    applied), and the spaces laid out over those rooms."""
    prepared: PreparedModel
    matched: Mapping[str, Room]
    layout: SpaceLayout


def prepare_case_model(project: Project, spaces: Sequence[Space], case: str, out_dir) -> CaseModel:
    """Read the scene, match it to ``spaces``, and prepare the model for
    ``case`` into ``out_dir`` (``build.prepare``, no Radiance) -- the one
    scene-to-prepare chain ``load_case`` and ``deliver.package`` (Task 12)
    share (F11).

    ``case`` may be a daylight case (an ``include_<case>`` column) or a
    Quality-Views-only case (a ``views_<case>`` column with no daylight case
    of the same name). A views-only case is prepared with
    :func:`~leed_daylight.project.spaces.views_prepare_spaces`, exactly as
    ``drivers.views`` prepares its own model, so its outlines, groups and
    packaged model come from the model the views run actually used (F15) --
    never from the first daylight case, which could include a space the
    views run never included.

    The rooms are found as every command finds them
    (``project.layout.scene_rooms``: detection, then splits.csv), and
    ``layout`` is the prepared model's own space layout.
    """
    prims = read_scene(project.scene.files, project.scene.units)
    roles = load_materials(project.root / "materials.csv")
    validate_scene(
        prims, project.scene.units, roles, declared_types=declared_types(project.root / "materials.csv"),
    ).raise_if_errors()
    rooms = scene_rooms(project, prims, roles)
    matched = match_to_rooms(spaces, rooms.rooms, hint=split_hint(rooms.rooms))

    if case in case_names(spaces):
        prep_spaces, prep_case = list(spaces), case
    elif case in views_case_names(spaces):
        prep_spaces, prep_case = views_prepare_spaces(spaces, case), VIEWS_PREPARE_CASE
    else:
        known = ", ".join(dict.fromkeys((*case_names(spaces), *views_case_names(spaces))))
        raise DeliverError(f"unknown case {case!r}; known cases are {known}")

    prepared = prepare(project, prims, roles, rooms, prep_spaces, prep_case, out_dir)
    return CaseModel(prepared=prepared, matched=types.MappingProxyType(matched), layout=prepared.layout)


def check_option(folder: pathlib.Path, project: Project, option: str, case: str) -> str | None:
    """None when ``option``'s results for ``case`` are usable, else the
    reason ``load_case`` gives for leaving the option out: no results file,
    no input fingerprints, inputs changed since (``stale_inputs``), no
    window-group list (Option 1), or no ``points.csv`` beside it. The
    wizard's run step tells current results from stale ones with it."""
    folder = pathlib.Path(folder)
    path = result_path(folder, option, case)
    if not path.is_file():
        return f"no {option} results for case {case!r} ({path} not found); run {COMMANDS[option]}"
    results = read_json_object(path)
    recorded = results.get("inputs")
    if not isinstance(recorded, dict):
        return f"{path} has no input fingerprints (written by an earlier version); rerun {COMMANDS[option]}"
    stale = stale_inputs(recorded, project, folder)
    if stale:
        return (
            f"{option} results for case {case!r} are out of date: {', '.join(stale)} changed "
            f"since they were computed; rerun {COMMANDS[option]}"
        )
    if option == "option1" and not isinstance(results.get("groups"), list):
        # Skylights spec 5.9: results written before roof glass was
        # supported carry no window-group kinds, so a skylight could
        # not be told from a window in the report.
        return (
            f"{option} results for case {case!r} have no window-group list ({path} was written "
            f"before roof and diffusing glass were supported); run again: {COMMANDS[option]}"
        )
    points_path = path.parent / "points.csv"
    if not points_path.is_file():
        # F14: a results file with no points.csv beside it is skipped
        # like a missing results file, not an abort of the whole case.
        return (
            f"{option} results for case {case!r} have no points.csv "
            f"({points_path} not found); rerun {COMMANDS[option]}"
        )
    return None


def load_case(folder, case: str | None = None, *, options: Sequence[str] = OPTIONS,
              today: datetime.date | None = None, require_results: bool = True) -> CaseData:
    """One case's results and model. An option whose results are missing,
    out of date or unreadable is left out of ``options`` and its reason put
    in ``skipped``. When no option is left, ``require_results`` (the default,
    as ``report`` and ``package`` need) refuses with a ``DeliverError``;
    ``require_results=False`` (the atlas model check, atlas spec 5) returns
    the case with an empty ``options`` instead."""
    folder = pathlib.Path(folder)
    if not (folder / "project.yaml").is_file():
        raise DeliverError(f"{folder}: no project.yaml")
    unknown = [o for o in options if o not in OPTIONS]
    if unknown:
        raise DeliverError(f"unknown option {unknown[0]!r}; expected one of {', '.join(OPTIONS)}")
    project = load_project(folder / "project.yaml")
    spaces = tuple(load_spaces(folder / "spaces.csv"))
    daylight_cases = case_names(spaces)
    known = tuple(dict.fromkeys((*daylight_cases, *views_case_names(spaces))))
    if case is None:
        if not known:
            raise DeliverError("spaces.csv: no cases found")
        case = known[0]
    elif case not in known:
        raise DeliverError(f"unknown case {case!r}; known cases are {', '.join(known)}")

    loaded: dict[str, OptionData] = {}
    skipped: dict[str, str] = {}
    warnings: list[str] = []
    for option in options:
        reason = check_option(folder, project, option, case)
        if reason is not None:
            skipped[option] = reason
            continue
        path = result_path(folder, option, case)
        points_path = path.parent / "points.csv"
        results = read_json_object(path)
        if results.get("tool_version") != TOOL_VERSION:
            warnings.append(
                f"{path.name}: written by tool version {results.get('tool_version')!r}; "
                f"this is version {TOOL_VERSION!r}"
            )
        loaded[option] = OptionData(
            option, types.MappingProxyType(results), read_points(points_path), path.parent, path,
        )
    if not loaded and require_results:
        reasons = "; ".join(skipped[o] for o in options) or "no option was asked for"
        raise DeliverError(f"no usable results for case {case!r}: {reasons}")

    names = {s.room for s in spaces}
    for option, data in loaded.items():
        strays = sum(1 for r in data.points.rooms if r not in names)
        if strays:
            warnings.append(
                f"{option}: {strays} point(s) name a room that is not in spaces.csv; "
                "they are left out of the plans"
            )

    with tempfile.TemporaryDirectory() as tmp:
        model = prepare_case_model(project, spaces, case, pathlib.Path(tmp))
        prepared = model.prepared
        glazing = {m: types.MappingProxyType(dict(r)) for m, r in prepared.manifest.get("glazing", {}).items()}
        storeys = _storeys(spaces, model.layout)
        label_of = {room: s.label for s in storeys for room in s.rooms}
        groups = tuple(_group_line(g, label_of.get(g.room, "")) for g in prepared.groups)
        laid = [space for space in model.layout.spaces if space.origin != "detected"]
        # Controller ruling (pre-flight F2): Option 1 states the blind
        # clause only for a space with at least one window group that can
        # carry a blind -- a `diffusing` group never has one (run.py's own
        # "blind": group.kind != "diffusing"), and a space with no group
        # has none. `GroupLine.room` is the space name (never a piece id),
        # so this reads directly off `groups`.
        has_blind = {g.room for g in groups if g.kind != "diffusing"}
        notes = {space.name: layout_note(space, blinds=space.name in has_blind) for space in laid}
        notes_no_blinds = {space.name: layout_note(space, blinds=False) for space in laid}
        notes_no_grid = {space.name: layout_note(space, grid=False) for space in laid}

    return CaseData(
        folder=folder, case=case, created=today or datetime.date.today(), project=project,
        rules=types.MappingProxyType(load_rules(project.credit_rules_path)), spaces=spaces,
        storeys=storeys, groups=groups, glazing=types.MappingProxyType(glazing),
        options=types.MappingProxyType(loaded), skipped=types.MappingProxyType(skipped),
        warnings=tuple(warnings), layout_notes=types.MappingProxyType(notes),
        layout_notes_no_blinds=types.MappingProxyType(notes_no_blinds),
        layout_notes_no_grid=types.MappingProxyType(notes_no_grid),
    )
