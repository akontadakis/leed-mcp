# leed_daylight/project/spaces.py
"""Load spaces.csv: the wizard's per-space edits, matched onto detected rooms.

`scene.storeys.detect_rooms` finds a candidate room for every connected patch
of floor, but it has no idea what a person calls that room, whether it is
regularly occupied, or which of several credit scenarios it belongs to.
spaces.csv is where a person records those answers, and this module is the
only place that reads the file back and reconciles it against a fresh scan.

A detected room's own identifier is a sort-order index, not a stable
identity: it is reassigned from scratch, in area order, every time
`detect_rooms` runs, so adding or resizing one room can shift every other
room's index. Persisting a person's edits by that index would silently
attach them to the wrong room after the next scan. Spec 6.1 solves this by
persisting the room's centroid instead, so this module matches a space back
to a room by its 2-D outline centroid -- and, because two rooms stacked on
different storeys can share a centroid, also by level -- and never by that
sort-order index. A test in `tests/test_project_spaces.py` pins that down
by scanning this file's own source text for that identifier's field name.

**Merged spaces** (merge-split spec 3.2). A row with ``merge_into`` set is a
*piece row* of the space its chain ends at (A into B, B into C: both A and
B are pieces of C); that last row is the *space row*. A piece row must carry
its space row's ``floor`` label and every ``include_<case>`` and
``views_<case>`` tick, so a space is selected or left out whole; its own
``space_type``, ``regularly_occupied`` and ``view_preserving_shades`` are not
used (``piece_row_warnings`` names each one that differs). ``included`` and
``views_included`` return space rows only, so every consumer that makes one
thing per selected row makes one per space.
"""
from __future__ import annotations

import csv
import io
import math
import pathlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from shapely.geometry import Polygon

from ..scene.storeys import Room
from .config import ProjectError
from .materials import read_csv_text

#: Spec 6.1: user edits "survive a re-scan by matching on room centroid
#: within 0.5 m".
CENTROID_TOLERANCE_M: float = 0.5

#: A candidate room must also sit on the same storey as the space, within
#: this z tolerance. Equals `scene.storeys.detect_rooms`'s own default
#: `z_tolerance` -- a storey's own z span -- so a space is never matched
#: across two rooms `detect_rooms` itself would treat as different storeys.
LEVEL_TOLERANCE_M: float = 0.05

#: Fixed columns, in order, before and after the `include_<case>` columns.
_FIXED_BEFORE = (
    "floor", "room", "space_type", "regularly_occupied", "area_m2",
    "centroid_x", "centroid_y", "level_z", "view_preserving_shades",
    "merge_into",
)
_FIXED_AFTER = ("reason",)
_FIXED = _FIXED_BEFORE + _FIXED_AFTER
_VIEWS_REASON = "views_reason"

_TRUE = frozenset({"1", "yes", "true"})
_FALSE = frozenset({"0", "no", "false"})

#: The characters a case name may not hold: it becomes one folder name under
#: ``_run`` and ``submission`` (security review H2), so no path separator,
#: no drive colon, no control character.
_UNSAFE_IN_CASE_NAME = re.compile(r"[/\\:\x00-\x1f\x7f]")
#: Why a case name is refused, after the name.
CASE_NAME_RULE = (
    "is not a safe case name; it becomes a folder name, so it may not be . or .. "
    "or hold / \\ : or a control character"
)


def is_safe_case_name(name: str) -> bool:
    """Whether ``name`` (an ``include_<case>`` or ``views_<case>`` case) is
    one folder name that stays where it is put: not empty, not ``.`` or
    ``..``, no ``/``, ``\\``, ``:`` or control character (security review
    H2). Every other name is kept as it is, spaces included."""
    return bool(name) and name not in (".", "..") and not _UNSAFE_IN_CASE_NAME.search(name)


@dataclass(frozen=True)
class Space:
    floor: str
    room: str  # the user's name; the stable identity, paired with the centroid
    space_type: str
    regularly_occupied: bool
    area_m2: float
    centroid_x: float  # metres, scene coordinates
    centroid_y: float
    level_z: float  # metres, scene coordinates
    view_preserving_shades: bool  # Option 2 only; see spec 5.3 and 6.7 item 6
    merge_into: str | None
    include: dict[str, bool]  # case name -> included
    reason: str
    #: Quality Views: views case name -> ticked. Empty when spaces.csv has
    #: no views_<case> column. Independent of `include` (spec 5.3).
    views: dict[str, bool] = field(default_factory=dict)
    #: Why a regularly occupied space is left out of Quality Views.
    views_reason: str = ""


def _fail(detail: str) -> None:
    raise ProjectError(f"spaces.csv: {detail}") from None


def _fail_at(line_no: int, detail: str) -> None:
    raise ProjectError(f"spaces.csv line {line_no}: {detail}") from None


def _check_header(fieldnames) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """Validate the header; return (daylight cases, views cases, has views_reason).

    The header is `_FIXED_BEFORE`, one or more `include_<case>` columns,
    zero or more `views_<case>` columns, `_FIXED_AFTER`, then optionally
    `views_reason`.
    """
    normalized = tuple(name.strip() for name in (fieldnames or ()))

    missing = [c for c in _FIXED if c not in normalized]
    if missing:
        _fail(
            f"missing column {missing[0]!r}; columns must be "
            f"{','.join(_FIXED_BEFORE)},include_<case>...,{','.join(_FIXED_AFTER)}, "
            f"got {','.join(normalized)}"
        )

    has_views_reason = _VIEWS_REASON in normalized
    middle = [name for name in normalized if name not in _FIXED and name != _VIEWS_REASON]

    def _is_case_column(name: str, prefix: str) -> bool:
        return name.startswith(prefix) and len(name) > len(prefix)

    bad = [n for n in middle if not (_is_case_column(n, "include_") or _is_case_column(n, "views_"))]
    if bad:
        _fail(
            f"unexpected column {bad[0]!r}; every column besides "
            f"{','.join(_FIXED_BEFORE)},{','.join(_FIXED_AFTER)},{_VIEWS_REASON} must be "
            f"'include_<case>' or 'views_<case>' with a non-empty case name, got {','.join(normalized)}"
        )
    include_cols = [n for n in middle if n.startswith("include_")]
    views_cols = [n for n in middle if n.startswith("views_")]
    if not include_cols:
        _fail(
            "no include_<case> column found; at least one is required "
            f"(the first is the base case), got {','.join(normalized)}"
        )

    expected = (
        _FIXED_BEFORE + tuple(include_cols) + tuple(views_cols) + _FIXED_AFTER
        + ((_VIEWS_REASON,) if has_views_reason else ())
    )
    if normalized != expected:
        _fail(f"columns must be exactly {','.join(expected)}, got {','.join(normalized)}")

    case_order = tuple(name[len("include_"):] for name in include_cols)
    for case in case_order:
        if case_order.count(case) > 1:
            _fail(f"case {case!r} has more than one include_{case} column")
    views_order = tuple(name[len("views_"):] for name in views_cols)
    for case in views_order:
        if views_order.count(case) > 1:
            _fail(f"views case {case!r} has more than one views_{case} column")
    for kind, names in (("case", case_order), ("views case", views_order)):
        for case in names:
            if not is_safe_case_name(case):
                _fail(f"{kind} {case!r} {CASE_NAME_RULE}; rename its column")

    return case_order, views_order, has_views_reason


def _parse_bool(value: str, line_no: int, column: str) -> bool:
    lowered = value.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    _fail_at(
        line_no,
        f"column {column!r} must be one of 1/0/yes/no/true/false (any case), got {value!r}",
    )


def _parse_float(value: str, line_no: int, column: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        _fail_at(line_no, f"column {column!r} must be a number, got {value!r}")
    if not math.isfinite(parsed):
        _fail_at(line_no, f"column {column!r} must be a finite number, got {value!r}")
    return parsed


def _check_merge_targets(spaces: Sequence[Space]) -> None:
    names = {s.room for s in spaces}
    for s in spaces:
        if s.merge_into is not None and s.merge_into not in names:
            _fail(
                f"space {s.room!r}: merge_into {s.merge_into!r} is not the name "
                "of another space in this file"
            )


def _check_merge_cycles(spaces: Sequence[Space]) -> None:
    """Refuse a merge cycle, including a self-merge, naming every space in it.

    Targets are already known to exist (`_check_merge_targets` runs first),
    so following `targets[node]` here never falls off the map.
    """
    targets = {s.room: s.merge_into for s in spaces}
    resolved: set[str] = set()
    for start in targets:
        if start in resolved:
            continue
        order: list[str] = []
        position: dict[str, int] = {}
        node: str | None = start
        while node is not None and node not in resolved:
            if node in position:
                cycle = order[position[node]:]
                _fail(f"merge cycle: {' -> '.join(cycle)} -> {cycle[0]}")
            position[node] = len(order)
            order.append(node)
            node = targets[node]
        resolved.update(order)


def _flag(value: bool) -> str:
    return "1" if value else "0"


def _check_piece_rows(
    spaces: Sequence[Space], lines: Mapping[str, int],
    case_order: Sequence[str], views_order: Sequence[str],
) -> None:
    """Refuse a piece row whose floor label, or any include_/views_ tick,
    differs from its space row's; the first differing column is named."""
    by_name = {s.room: s for s in spaces}
    owner = space_of(spaces)
    for piece in spaces:
        if piece.merge_into is None:
            continue
        space = by_name[owner[piece.room]]
        columns = [("floor", repr(piece.floor), repr(space.floor))] if piece.floor != space.floor else []
        columns += [
            (f"include_{case}", _flag(piece.include[case]), _flag(space.include[case]))
            for case in case_order if piece.include[case] != space.include[case]
        ]
        columns += [
            (f"views_{case}", _flag(piece.views[case]), _flag(space.views[case]))
            for case in views_order if piece.views[case] != space.views[case]
        ]
        if columns:
            column, own, theirs = columns[0]
            _fail(
                f"space {piece.room!r} (line {lines[piece.room]}) is a piece of {space.room!r} "
                f"(line {lines[space.room]}) but its {column} differs ({own}, not {theirs}); "
                "a piece row must have the same floor and include_/views_ ticks as its space row"
            )


def _check_subset_of_base(spaces: Sequence[Space], case_order: Sequence[str]) -> None:
    """Spec 5.3: a scenario case must be a subset of the base (first) case."""
    if len(case_order) < 2:
        return
    base = case_order[0]
    for scenario in case_order[1:]:
        for s in spaces:
            if s.include[scenario] and not s.include[base]:
                _fail(
                    f"space {s.room!r}: included in 'include_{scenario}' but not "
                    f"in 'include_{base}'; a scenario must be a subset of the base case"
                )


def load_spaces(path: str | pathlib.Path) -> list[Space]:
    """Load, validate and return every space in spaces.csv, in file order.

    Either a fully valid list of `Space` comes back, or `ProjectError` is
    raised naming the offending line, column and value (or the object, for
    a cross-row refusal); there is no partly valid state.
    """
    path = pathlib.Path(path)
    # UTF-8 (BOM stripped), cp1252 and ';'-delimited files refused by name:
    # the same reader materials.csv uses (T3.3).
    f = io.StringIO(read_csv_text(path, _fail), newline="")

    spaces: list[Space] = []
    with f:
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration:
            _fail("empty file; expected a header row")
        case_order, views_order, has_views_reason = _check_header(raw_header)
        header = tuple(name.strip() for name in raw_header)
        n_prefix = len(_FIXED_BEFORE)

        seen_rooms: dict[str, int] = {}
        # Physical FILE line numbers (header = line 1), from csv.reader's
        # own line_num, not a record count -- a blank line or a multi-line
        # quoted cell must not shift every later line number (T3.3 fix 1).
        last_line = reader.line_num
        while True:
            try:
                raw_row = next(reader)
            except StopIteration:
                break
            line_no = last_line + 1
            last_line = reader.line_num
            if raw_row == []:
                # A genuinely blank physical line: not a data row.
                continue

            if len(raw_row) != len(header):
                _fail_at(
                    line_no,
                    f"row has {len(raw_row)} field(s), expected {len(header)} "
                    f"({','.join(header)}): {raw_row!r}",
                )

            cells = [c.strip() for c in raw_row]
            if not any(cells):
                # Excel exports a row of commas for a cleared row: not a data row.
                continue
            room = cells[1]
            if not room:
                _fail_at(line_no, "room is blank; every row needs a room name")

            regularly_occupied = _parse_bool(cells[3], line_no, "regularly_occupied")
            area_m2 = _parse_float(cells[4], line_no, "area_m2")
            centroid_x = _parse_float(cells[5], line_no, "centroid_x")
            centroid_y = _parse_float(cells[6], line_no, "centroid_y")
            level_z = _parse_float(cells[7], line_no, "level_z")
            view_preserving_shades = _parse_bool(cells[8], line_no, "view_preserving_shades")
            merge_into = cells[9] or None
            include = {
                case: _parse_bool(cells[n_prefix + i], line_no, f"include_{case}")
                for i, case in enumerate(case_order)
            }
            views_start = n_prefix + len(case_order)
            views = {
                case: _parse_bool(cells[views_start + j], line_no, f"views_{case}")
                for j, case in enumerate(views_order)
            }
            reason = cells[views_start + len(views_order)]
            views_reason = cells[-1] if has_views_reason else ""
            # A piece row carries its space row's views ticks whatever its
            # own regularly_occupied says; that value only warns
            # (piece_row_warnings), so this rule is for space rows.
            if not regularly_occupied and merge_into is None:
                ticked = [case for case, on in views.items() if on]
                if ticked:
                    _fail_at(
                        line_no,
                        f"space {room!r}: ticked in 'views_{ticked[0]}' but regularly_occupied "
                        "is no; Quality Views counts regularly occupied floor area only",
                    )

            if area_m2 <= 0:
                _fail_at(line_no, f"space {room!r}: area_m2 must be > 0, got {area_m2!r}")

            if room in seen_rooms:
                _fail(
                    f"room {room!r} appears twice, at line {seen_rooms[room]} "
                    f"and line {line_no}"
                )
            seen_rooms[room] = line_no

            spaces.append(Space(
                floor=cells[0],
                room=room,
                space_type=cells[2],
                regularly_occupied=regularly_occupied,
                area_m2=area_m2,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                level_z=level_z,
                view_preserving_shades=view_preserving_shades,
                merge_into=merge_into,
                include=include,
                reason=reason,
                views=views,
                views_reason=views_reason,
            ))

    if not spaces:
        _fail("no spaces")

    _check_merge_targets(spaces)
    _check_merge_cycles(spaces)
    _check_piece_rows(spaces, seen_rooms, case_order, views_order)
    _check_subset_of_base(spaces, case_order)

    return spaces


def space_rows(spaces: Sequence[Space]) -> list[Space]:
    """The space rows (``merge_into`` empty), in file order."""
    return [s for s in spaces if s.merge_into is None]


def space_of(spaces: Sequence[Space]) -> dict[str, str]:
    """Every row's name mapped to the name of the space row its
    ``merge_into`` chain ends at (itself for a space row)."""
    _check_merge_targets(spaces)
    _check_merge_cycles(spaces)
    targets = {s.room: s.merge_into for s in spaces}
    owner: dict[str, str] = {}
    for name in targets:
        node = name
        while targets[node] is not None:
            node = targets[node]
        owner[name] = node
    return owner


def _shown(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def piece_row_warnings(spaces: Sequence[Space]) -> tuple[str, ...]:
    """One warning per piece row and column (``space_type``,
    ``regularly_occupied``, ``view_preserving_shades``) whose value differs
    from its space row's: the space row's value is the one used."""
    by_name = {s.room: s for s in spaces}
    owner = space_of(spaces)
    warnings: list[str] = []
    for piece in spaces:
        if piece.merge_into is None:
            continue
        space = by_name[owner[piece.room]]
        for column in ("space_type", "regularly_occupied", "view_preserving_shades"):
            own, theirs = getattr(piece, column), getattr(space, column)
            if own != theirs:
                warnings.append(
                    f"space {piece.room!r} is a piece of {space.room!r}: its {column} "
                    f"{_shown(own)!r} differs from the space row's {_shown(theirs)!r}; "
                    "the space row's value is used"
                )
    return tuple(warnings)


def check_linked_zones(spaces: Sequence[Space], linked_zones: Mapping) -> None:
    """Refuse a ``performance.linked_zones`` entry that names a piece row:
    a linked zone is a space, named by its space row. Any other name (an
    unknown one, a non-string) is left to
    ``engine.coefficients.sensors_in_scope``, as before."""
    owner = space_of(spaces)
    for key, names in linked_zones.items():
        for name in names:
            if isinstance(name, str) and owner.get(name, name) != name:
                raise ProjectError(
                    f"performance.linked_zones storey {key}: {name!r} is a piece of space "
                    f"{owner[name]!r} (merge_into); name the space row {owner[name]!r}"
                )


def case_names(spaces: Sequence[Space]) -> tuple[str, ...]:
    """Every case name, in file order (the first is the base case)."""
    if not spaces:
        return ()
    return tuple(spaces[0].include.keys())


def included(spaces: Sequence[Space], case: str) -> list[Space]:
    """Every space row included in `case` (piece rows are left out), or a
    refusal naming the known cases."""
    known = case_names(spaces)
    if case not in known:
        _fail(f"unknown case {case!r}; known cases are {', '.join(known)}")
    return [s for s in space_rows(spaces) if s.include[case]]


def views_case_names(spaces: Sequence[Space]) -> tuple[str, ...]:
    """Every Quality Views case name, in file order; () when there is none."""
    if not spaces:
        return ()
    return tuple(spaces[0].views.keys())


def views_included(spaces: Sequence[Space], case: str) -> list[Space]:
    """Every space row ticked in `views_<case>` (piece rows are left out),
    or a refusal."""
    known = views_case_names(spaces)
    if not known:
        _fail("no views_<case> column; add one (for example views_base) to run Quality Views")
    if case not in known:
        _fail(f"unknown views case {case!r}; known views cases are {', '.join(known)}")
    return [s for s in space_rows(spaces) if s.views[case]]


#: The case name `build.prepare` is given when preparing a model for Quality
#: Views (moved here from `drivers.views` -- reports spec F15): not a real
#: spaces.csv case, just a fixed label both `drivers.views` and
#: `deliver.data.prepare_case_model` (a views-only case) hand it.
VIEWS_PREPARE_CASE: str = "views"


def views_prepare_spaces(spaces: Sequence[Space], case: str) -> list[Space]:
    """The spaces with one synthetic daylight case, `VIEWS_PREPARE_CASE`,
    that includes exactly the spaces ticked in views_<case>.

    `build.prepare` frames every exterior pane whatever the case; the case
    only selects which spaces it grids and checks. Handing it the views
    ticks keeps a views-derived model independent of any include_<case>
    column. `drivers.views` uses this for its own run, and
    `deliver.data.prepare_case_model` uses it again for a views-only case
    (reports spec 8, F15), so a views-only case's outlines, groups and
    packaged model come from the same model the views run actually used --
    never from the first daylight case, which could exclude a merged space
    the views run never included either.
    """
    return [replace(s, include={VIEWS_PREPARE_CASE: s.views[case]}) for s in spaces]


def _outline_centroid(room: Room) -> tuple[float, float]:
    centroid = Polygon(room.outline).centroid
    return centroid.x, centroid.y


def _describe(room: Room) -> str:
    """Describe a room for a refusal message -- by storey, level and
    centroid, never by its own sort-order identifier."""
    cx, cy = _outline_centroid(room)
    return f"a room on storey {room.storey} at level {room.level_z:.2f} m, centroid ({cx:.2f}, {cy:.2f})"


def match_to_rooms(
    spaces: Sequence[Space],
    rooms: Sequence[Room],
    tolerance_m: float = CENTROID_TOLERANCE_M,
    *,
    hint: str = "",
) -> dict[str, Room]:
    """Match every space to exactly one room, by level then by centroid.

    A room is a candidate for a space only when it is within
    `LEVEL_TOLERANCE_M` of the space's level AND within `tolerance_m` of the
    space's centroid (both boundaries inclusive). Zero candidates, or more
    than one, is a refusal naming the space -- ambiguity is never resolved
    by picking the nearest candidate, because it means the scan changed the
    plan and a person has to look. The returned mapping is injective: two
    spaces are never matched to the same room.

    ``hint`` is appended to every refusal (``build.prepare`` passes one
    when split lines changed the rooms); ``""`` keeps the messages as they
    were.
    """
    matched: dict[str, Room] = {}
    claimed_by: dict[int, str] = {}

    for space in spaces:
        on_level = [r for r in rooms if abs(space.level_z - r.level_z) <= LEVEL_TOLERANCE_M]
        if not on_level:
            _fail(
                f"space {space.room!r}: no room found at level_z={space.level_z:.2f} m "
                f"(checked {len(rooms)} room(s)){hint}"
            )

        distances = [
            math.hypot(space.centroid_x - cx, space.centroid_y - cy)
            for cx, cy in (_outline_centroid(r) for r in on_level)
        ]
        candidates = [r for r, d in zip(on_level, distances) if d <= tolerance_m]

        if not candidates:
            nearest = min(distances)
            _fail(
                f"space {space.room!r}: nearest room on its level is {nearest:.2f} m "
                f"away, outside the {tolerance_m:.2f} m tolerance{hint}"
            )
        if len(candidates) > 1:
            described = "; ".join(_describe(r) for r in candidates)
            _fail(f"space {space.room!r} matches more than one room: {described}{hint}")

        room = candidates[0]
        key = id(room)
        if key in claimed_by:
            _fail(
                f"space {space.room!r} and space {claimed_by[key]!r} both match "
                f"{_describe(room)}{hint}"
            )
        claimed_by[key] = space.room
        matched[space.room] = room

    return matched


#: A spaces.csv area more than this share away from the scene room's own
#: area is named in a warning. Results always use the scene area; a large
#: gap usually means the space matched a different room than intended.
AREA_MISMATCH_SHARE: float = 0.05


def area_mismatches(spaces: Sequence[Space], matched: dict[str, Room]) -> tuple[str, ...]:
    """Warn for every space whose spaces.csv area_m2 is more than
    `AREA_MISMATCH_SHARE` away from its matched room's scene area."""
    warnings: list[str] = []
    for space in spaces:
        room = matched.get(space.room)
        if room is None:
            continue
        if abs(space.area_m2 - room.area_m2) > AREA_MISMATCH_SHARE * room.area_m2:
            warnings.append(
                f"space {space.room!r}: spaces.csv area_m2 is {space.area_m2:.2f} m2 but the "
                f"matched scene room is {room.area_m2:.2f} m2 (more than "
                f"{AREA_MISMATCH_SHARE:.0%} apart); results use the scene area, so check "
                "that the space matched the room you meant"
            )
    return tuple(warnings)
