# leed_daylight/drivers/run_option1.py
"""The full LEED Option 1 run (layer 7; spec 6.6): blind schedule, sDA with
blinds operating, ASE with blinds open, per space and per project, both
aggregations, score and glare-narrative trigger.

This is the `run` verb's business logic for Option 1, with no command-line
front end of its own -- plan 3's `cli.py` calls it. It never touches a
Radiance binary itself: every daylight number it uses comes from
`Option1Engine`, an injectable seam that exactly matches
`engine.coefficients.Option1Tracer`'s own three lux-only methods
(`direct`, `sky`, `residual`). When `engine` is not supplied, the real tracer is built from the
project's sky and sun matrices; every test in this module injects a FAKE
engine instead and never exercises that path.

The twelve-step sequence (spec 6.6, controller ruling):

1. The occupied-hour mask (`metrics.occupancy.occupied_mask`), with the
   daylight-saving dates of `metrics.occupancy.DST_YEAR` (2025; a TMY file
   has no year of its own), recorded in `Option1Result.daylight_saving_year`.
2. Two checks: the engine and the mask agree on the hour count; the mask's
   own hour count agrees with what `analysis.hours` says it should be
   ("a different number means the mask is wrong", spec 6.9). Ruling P51
   keeps this second check. Its one known cause -- an hours window
   opening at 00:00 while daylight saving is in effect, which lost an
   hour on every DST day -- is now refused earlier by
   `metrics.occupancy.occupied_mask` (no window may open before 03:00
   under daylight saving), so the check is defensive again.
3. Two cross-checks: `case` matches `prepared.case`; `prepared.grids`'
   rooms are exactly the case's regularly occupied spaces.
4. Direct-sun lux (blinds open), every window group.
5. The blind schedule, per room (LM-83-12 clause 2.2.6), using the ROOM
   area from the scene as the trigger area (spec 6.10), or, with
   ``blinds.trigger_area: grid``, the area the room's sensor points stand
   for (point count x ``grid.spacing_m`` squared,
   ``RoomGrid.cell_area_m2``). A group whose room has no grid keeps its
   blinds open every hour, named in a warning. A diffusing group (kind
   "diffusing") has no blind at all: it is open every hour, never enters
   the closing order and is never named in that warning. A room over
   LM-83-12 2.2.6's 10,000 sf blind analysis area keeps one schedule
   (spec ruling), named in a warning that LM-83 would split it. Ruling
   P52: if two groups owning the same room disagree on how many of that
   room's own sensor points they see, that is a named refusal, not a raw
   `np.stack` failure.
6. Sky lux (blinds open, and blinds closed only where the schedule
   actually closes), every window group.
7. Per room: sDA lux follows the blind schedule; ASE lux always uses the
   blinds-open direct lux, never the schedule (R-P43) -- LM-83 defines ASE
   with blinds open regardless of when they would really close.
7b. The residual pass (run before step 4): sky and direct-sun lux with
   EVERY group's openings blacked out and no blinds, i.e. light that
   reaches a sensor through no window group (a skylight not labelled as
   glazing, an unglazed gap). Each group's own trace blacks out only the
   OTHER groups' openings (P34), so it already carries this light once,
   blinds open or closed alike (a closed blind covers only its own
   group's openings). It is therefore not added on top; it is used
   to (a) take the residual direct sun back out of each group's own
   direct lux before the blind trigger (step 5), clamped at 0 lux --
   no blind controls it, so it never closes one -- and (b) step 7c.
7c. Per room covered by n group traces, per point-hour: sDA lux = sum
   of the n traces (as scheduled) + (1 - n) x residual sky lux; ASE lux
   = sum of the n direct traces + (1 - n) x residual direct lux; both
   clamped at 0 lux (the traces are separate Monte Carlo samples). n = 1
   needs no correction. The room's residual share is taken against the
   blinds-open annual sky illuminance with the residual counted once. A
   room whose share is over `RESIDUAL_WARNING_SHARE`, or where residual
   direct sun alone reaches the ASE lux at any sensor, is named in a
   warning.
8. The LM-83 metrics, per room.
9. Both aggregations (area-weighted and point-weighted), for sDA and for
   ASE, always both, never only one (spec 6.10) -- from each room's exact
   integer point counts (`metrics.aggregate.aggregate_counts`); every
   band, minimum and trigger comparison afterwards is made at
   `metrics.scoring.PCT_DECIMALS` precision.
10. The score (`metrics.scoring.score_option1`), against whichever
    aggregation `analysis.aggregation` names as the headline.
11. The per-space glare-narrative trigger (spec defect S1): a PER-SPACE
    check, independent of the LM-83 reporting band a space's own ASE
    percentage falls into (carry-forward item 9).
12. Warnings: the prepared model's own, the blind-trigger-area note, every
    per-group warning from step 5, every residual warning from step 7c,
    and (for a "points" headline) a note that LEED Table 1 is defined on
    the floor-area average.

Every threshold used here comes from `rules`; this module contains no
literal 300, 0.50, 1000 or 250.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from typing import Protocol

import numpy as np

from ..build.groups import WindowGroup
from ..build.prepare import PreparedModel
from ..engine.coefficients import Option1Tracer
from ..engine.radiance import Radiance
from ..engine.sky import sky_matrix, sun_matrix
from ..metrics.aggregate import aggregate_counts
from ..metrics.blinds import MAX_ANALYSIS_AREA_M2, blind_schedule
from ..metrics.occupancy import DST_YEAR, HOURS_RE, occupied_mask
from ..metrics.option1 import ase_band, ase_exceeds, ase_hours, daylight_autonomy, sda_pass
from ..metrics.scoring import Option1Score, score_option1, spaces_needing_glare_narrative
from ..project.config import Project
from ..project.spaces import Space, included


class Option1Error(ValueError):
    """Option 1 cannot be run to completion from these inputs."""


#: A room is named in a warning when the residual pass (step 7b) delivers
#: more than this share of its blinds-open annual sky illuminance. Not a
#: LEED threshold: a check that the window groups account for the daylight.
#: The occupied-hours window LEED v4.1 Table 1 is defined on.
LEED_HOURS = "08:00-18:00"

RESIDUAL_WARNING_SHARE = 0.01

#: The note step 12 writes for each ``blinds.trigger_area``.
_TRIGGER_AREA_NOTES = {
    "room": "the blind trigger area is the room area from the scene, not the grid area",
    "grid": (
        "the blind trigger area is the area the sensor points stand for (points x grid spacing "
        "squared), not the room area"
    ),
}

#: The name the residual pass is reported under in a refusal.
_RESIDUAL_LABEL = "residual"

#: The window-group kind that has no blind (roof-glass design D1): open
#: every hour, never in LM-83's closing order, and the engine is never asked
#: for its blinds-closed trace (it has no blind file).
_NO_BLIND_KIND = "diffusing"


class Option1Engine(Protocol):
    """The engine seam (R-P37): exactly the surface of
    `engine.coefficients.Option1Tracer` that this driver calls."""

    @property
    def n_hours(self) -> int: ...

    def direct(
        self, group: WindowGroup, hours: np.ndarray,
    ) -> tuple[tuple[str, ...], np.ndarray]: ...

    def sky(
        self, group: WindowGroup, hours: np.ndarray, open_schedule: np.ndarray,
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray | None]: ...

    def residual(
        self, hours: np.ndarray,
    ) -> Iterable[tuple[tuple[str, ...], np.ndarray, np.ndarray]]:
        """Per trace unit: ``(sensor_index, sky_lux, direct_lux)`` with every
        group's openings blacked out and no blinds; the units together
        cover every grid exactly once."""
        ...


@dataclasses.dataclass(frozen=True)
class SpaceResult:
    room: str
    area_m2: float  # the room's area, not the grid's
    points: int  # the room's own sensor-point count
    sda_pct: float
    ase_pct: float
    ase_band: str  # the LM-83 reporting label
    needs_glare_narrative: bool  # ASE over the credit's per-space trigger
    # the residual pass's share of the room's blinds-open annual sky
    # illuminance, in percent (step 7b)
    residual_sky_pct: float = 0.0


@dataclasses.dataclass(frozen=True)
class Option1Result:
    case: str
    occupied_hours: int
    spaces: tuple[SpaceResult, ...]
    sda_area_pct: float
    sda_point_pct: float
    ase_area_pct: float
    ase_point_pct: float
    headline: str  # "floor_area" or "points", from analysis.aggregation
    score: Option1Score
    narrative_spaces: tuple[str, ...]
    blind_schedule: dict[str, np.ndarray]  # group_id -> (hours,) bool, True = open
    warnings: tuple[str, ...]
    # the calendar year the daylight-saving changeover dates came from
    daylight_saving_year: int = DST_YEAR
    #: One dict per sensor point, in grid order (reports spec 3.1): room,
    #: storey, x, y, z, da_pct (share of occupied hours at or above the sDA
    #: lux, in percent), sda_pass, ase_hours (blinds open), ase_exceeds.
    point_rows: tuple[dict, ...] = ()


def _select_room_rows(sensor_index: Sequence[str], lux: np.ndarray, room: str) -> np.ndarray:
    """The rows of `lux` whose `sensor_index` names `room`; an ordinary
    empty `(0, hours)` slice when the group's scope misses the room."""
    if len(sensor_index) == 0:
        return lux[:0]
    keep = np.array([name == room for name in sensor_index])
    return lux[keep]


def _check_engine_shape(
    kind: str, group_id: str, sensor_index: Sequence[str], lux: np.ndarray, n_selected: int,
) -> None:
    """M2: an injected engine's returned ``lux`` array must have one row
    per ``sensor_index`` entry and one column per occupied hour -- a named
    refusal naming the group and what was seen, never a raw `IndexError`/
    `ValueError` several lines later in `_select_room_rows` or `np.where`.
    """
    if lux.shape[0] != len(sensor_index):
        raise Option1Error(
            f"group {group_id!r}: engine.{kind} returned {lux.shape[0]} lux "
            f"row(s) but {len(sensor_index)} sensor_index entries; they "
            "must match"
        )
    if lux.shape[1] != n_selected:
        raise Option1Error(
            f"group {group_id!r}: engine.{kind} returned {lux.shape[1]} lux "
            f"column(s), expected {n_selected} (the occupied-hour count)"
        )


def _parse_hour_bounds(hours: str) -> tuple[int, int]:
    match = HOURS_RE.match(hours)
    if not match:
        raise Option1Error(
            f"project.analysis.hours: expected 'HH:MM-HH:MM', got {hours!r}"
        )
    start_h, _start_m, end_h, _end_m = (int(g) for g in match.groups())
    return start_h, end_h


def _build_engine(
    rad: Radiance, project: Project, prepared: PreparedModel, processes: int = 1,
) -> Option1Tracer:
    sky_dir = prepared.root / "engine" / "sky"
    sky_path = sky_matrix(
        rad, project.weather_file, sky_dir / "sky.mtx",
        project.sky.MF, project.site.north_rotation_deg,
    )
    suns_path = sun_matrix(
        rad, project.weather_file, sky_dir / "suns.mtx",
        project.sky.MF, project.site.north_rotation_deg,
    )
    return Option1Tracer(
        rad, project, prepared, sky=sky_path, suns=suns_path, processes=processes,
    )


def run_option1(
    rad: Radiance, project: Project, prepared: PreparedModel,
    spaces: Sequence[Space], rules: dict, case: str,
    *, engine: Option1Engine | None = None, processes: int = 1,
    progress: Callable[[str], None] | None = None,
) -> Option1Result:
    """The twelve-step Option 1 run (module docstring). ``processes`` is
    the ``rfluxmtx``/``rcontrib`` ``-n`` count for the real engine; it is
    unused when ``engine`` is injected. ``progress``, when given, is called
    with one short text after each finished unit of work (wizard spec 8.4):
    the residual pass, then each window group's direct and sky traces."""
    if engine is None:
        engine = _build_engine(rad, project, prepared, processes)

    # (1) the occupied-hour mask. The daylight-saving changeover dates are
    # taken from the DST_YEAR calendar (a TMY file has no year of its own);
    # the year used is carried into the result's settings.
    mask = occupied_mask(
        project.analysis.hours, project.analysis.daylight_saving, DST_YEAR,
    )

    # (2) R-P38: the engine and the mask must agree on the hour count, and
    # the mask's own occupied-hour count must match what analysis.hours says.
    if engine.n_hours != len(mask):
        raise Option1Error(
            f"engine.n_hours ({engine.n_hours}) does not match the "
            f"occupied-hour mask's length ({len(mask)}); a different "
            "number means the mask, or the engine's sky matrix, is wrong"
        )
    start_h, end_h = _parse_hour_bounds(project.analysis.hours)
    expected_occupied = (len(mask) // 24) * (end_h - start_h)
    actual_occupied = int(mask.sum())
    if actual_occupied != expected_occupied:
        message = (
            f"occupied_mask produced {actual_occupied} occupied hours, "
            f"expected {expected_occupied} from analysis.hours "
            f"{project.analysis.hours!r}; a different number means the "
            "mask is wrong"
        )
        # Controller ruling P51: the refusal stays, as a guard against a
        # wrong mask. Its one known cause (a window opening before 03:00
        # under daylight saving) is now refused by occupied_mask itself,
        # and a window ending at 24:00 is represented correctly (24:00
        # clock time is 23:00 standard time under daylight saving), so
        # this is reachable only through a bug in the mask.
        raise Option1Error(message)
    n_selected = actual_occupied

    # (3) R-P41: cross-checks between the case, the prepared model and spaces.
    if case != prepared.case:
        raise Option1Error(
            f"case {case!r} does not match prepared.case {prepared.case!r}"
        )
    occupied_spaces = {s.room for s in included(spaces, case) if s.regularly_occupied}
    grid_rooms = {g.room for g in prepared.grids}
    if occupied_spaces != grid_rooms:
        missing_grids = sorted(occupied_spaces - grid_rooms)
        extra_grids = sorted(grid_rooms - occupied_spaces)
        raise Option1Error(
            "prepared.grids does not match the case's regularly occupied "
            f"spaces: missing a grid for {missing_grids}, and a grid "
            f"exists with no matching regularly occupied space for "
            f"{extra_grids}"
        )

    # I2: per-room accumulators, sized once up front, so no more than ONE
    # group's own full (potentially whole-building) lux array is ever
    # alive at a time -- a `trace_scope: all` run no longer holds every
    # group's own array simultaneously (groups x points x hours would not
    # fit in memory on a buildings-scale project).
    sda_lux_by_room: dict[str, np.ndarray] = {
        grid.room: np.zeros((int(grid.points.shape[0]), n_selected), dtype=np.float64)
        for grid in prepared.grids
    }
    ase_lux_by_room: dict[str, np.ndarray] = {
        grid.room: np.zeros((int(grid.points.shape[0]), n_selected), dtype=np.float64)
        for grid in prepared.grids
    }

    # (7b, run first) the residual pass: light through no window group,
    # always open, never scheduled. Each group's own isolated trace already
    # carries it once (only OTHER groups' openings are blacked out there,
    # P34), so it is kept per room here and used (a) to take it back out
    # of each group's own direct lux before the blind trigger (step 5) --
    # no blind controls it -- and (b) to remove the n - 1 surplus copies
    # from a room that n group traces cover (step 7c).
    residual_sky_by_room: dict[str, np.ndarray] = {}
    residual_direct_by_room: dict[str, np.ndarray] = {}
    for sensor_index, residual_sky, residual_direct in engine.residual(mask):
        _check_engine_shape(
            "residual (sky)", _RESIDUAL_LABEL, sensor_index, residual_sky, n_selected,
        )
        _check_engine_shape(
            "residual (direct)", _RESIDUAL_LABEL, sensor_index, residual_direct, n_selected,
        )
        for grid in prepared.grids:
            room_sky = _select_room_rows(sensor_index, residual_sky, grid.room)
            if room_sky.shape[0] == 0:
                continue
            if grid.room in residual_sky_by_room:
                raise Option1Error(
                    f"room {grid.room!r}: engine.residual covered it in more "
                    "than one trace unit; the residual must be counted exactly once"
                )
            expected = sda_lux_by_room[grid.room].shape[0]
            if room_sky.shape[0] != expected:
                raise Option1Error(
                    f"engine.residual returned {room_sky.shape[0]} row(s) for "
                    f"room {grid.room!r}, expected {expected} (the room's own "
                    "sensor-point count)"
                )
            # Kept for the whole run, so stored as float32 (half the
            # memory; the boolean row selection already made a copy, so
            # no second one is taken). Promoted to float64 wherever it is
            # added or summed.
            residual_sky_by_room[grid.room] = np.asarray(room_sky, dtype=np.float32)
            residual_direct_by_room[grid.room] = np.asarray(
                _select_room_rows(sensor_index, residual_direct, grid.room),
                dtype=np.float32,
            )
    uncovered = [g.room for g in prepared.grids if g.room not in residual_sky_by_room]
    if uncovered:
        raise Option1Error(
            f"engine.residual did not cover room(s) {uncovered}; every room "
            "needs its residual pass"
        )
    if progress is not None:
        progress("residual pass")

    # (4) direct-sun lux (blinds open), every window group -- folded into
    # the room ASE accumulators as each group's own lux arrives, and only
    # the group's OWN-ROOM rows kept afterward (for the blind schedule,
    # step 5); the full (possibly whole-scope) array is never retained.
    own_room_direct: dict[str, np.ndarray] = {}
    own_room_counts: dict[str, dict[str, int]] = {}  # room -> {group_id: count}
    # how many group traces cover each room: each one carries the residual
    direct_traces: dict[str, int] = {grid.room: 0 for grid in prepared.grids}
    n_groups = len(prepared.groups)
    for number, group in enumerate(prepared.groups, start=1):
        sensor_index, direct_lux = engine.direct(group, mask)
        _check_engine_shape("direct", group.group_id, sensor_index, direct_lux, n_selected)
        own_rows = _select_room_rows(sensor_index, direct_lux, group.room)
        own_room_direct[group.group_id] = own_rows

        if group.room in ase_lux_by_room:
            # Ruling P52: every group owning this room must see the same
            # number of the room's own sensor points, checked and raised
            # here -- BEFORE any accumulation below -- so a disagreement
            # is always a named refusal listing every group's own count,
            # never a raw, unnamed numpy broadcast error from `+=`.
            counts = own_room_counts.setdefault(group.room, {})
            counts[group.group_id] = own_rows.shape[0]
            if len(set(counts.values())) > 1:
                raise Option1Error(
                    f"room {group.room!r}: its window groups disagree on "
                    f"how many of the room's own sensor points they see: "
                    f"{counts}"
                )

        for grid in prepared.grids:
            room_direct = _select_room_rows(sensor_index, direct_lux, grid.room)
            if room_direct.shape[0] == 0:
                continue
            expected = ase_lux_by_room[grid.room].shape[0]
            if room_direct.shape[0] != expected:
                raise Option1Error(
                    f"group {group.group_id!r}: engine.direct returned "
                    f"{room_direct.shape[0]} row(s) for room {grid.room!r}, "
                    f"expected {expected} (the room's own sensor-point count)"
                )
            ase_lux_by_room[grid.room] += room_direct
            direct_traces[grid.room] += 1
        # `direct_lux` (and `sensor_index`) are not referenced again after
        # this iteration -- nothing keeps every group's full array alive.
        if progress is not None:
            progress(f"window group {number} of {n_groups} (direct)")

    # (5) R-P40: the blind schedule, per room, LM-83-12 clause 2.2.6 -- the
    # trigger area is the ROOM area from the scene (spec 6.10), or with
    # blinds.trigger_area "grid" the area the sensor points stand for.
    # Row-count agreement was already checked and raised above (P52); this
    # only builds the schedule from what step 4 already collected. Each
    # group's own direct lux carries the residual direct sun (step 7b),
    # which comes through none of its windows and no blind controls: it is
    # taken back out (clamped at 0 lux, Monte Carlo noise) so a blind
    # closes only for sun through its own group.
    ase_lux_threshold = rules["option1"]["ase"]["lux"]
    schedule: dict[str, np.ndarray] = {}
    group_warnings: list[str] = []
    for grid in prepared.grids:
        # LM-83-12 2.2.6 caps a blind analysis area at 10,000 sf, split by
        # facade orientation; the ruling here is one schedule per room, so
        # a larger room is kept whole and named.
        if grid.room_area_m2 > MAX_ANALYSIS_AREA_M2:
            group_warnings.append(
                f"room {grid.room!r}: {grid.room_area_m2:.1f} m2 is over "
                f"LM-83-12 2.2.6's 10,000 sf ({MAX_ANALYSIS_AREA_M2:.2f} m2) "
                "blind analysis area; LM-83 requires splitting it by facade "
                "orientation, and this result pools the whole room into one "
                "blind schedule"
            )
        own_groups = [g for g in prepared.groups if g.room == grid.room]
        for g in own_groups:
            if g.kind == _NO_BLIND_KIND:
                schedule[g.group_id] = np.ones(n_selected, dtype=bool)
        own_groups = [g for g in own_groups if g.kind != _NO_BLIND_KIND]
        if not own_groups:
            continue
        room_residual = residual_direct_by_room[grid.room].astype(np.float64)
        rows = [
            np.maximum(own_room_direct[g.group_id] - room_residual, 0.0)
            for g in own_groups
        ]
        stacked = np.stack(rows, axis=0)  # (own_groups, points, hours)
        # "grid": the room's own sensor points only, each standing for one cell.
        trigger_area_m2 = (
            grid.points.shape[0] * grid.cell_area_m2
            if project.blinds.trigger_area == "grid"
            else grid.room_area_m2
        )
        room_schedule = blind_schedule(stacked, trigger_area_m2, ase_lux_threshold)
        for g, row in zip(own_groups, room_schedule):
            schedule[g.group_id] = row

    for group in prepared.groups:
        if group.group_id in schedule:
            continue
        schedule[group.group_id] = np.ones(n_selected, dtype=bool)
        if group.kind == _NO_BLIND_KIND:
            continue  # no blind to keep open, so nothing to warn about
        group_warnings.append(
            f"{group.group_id}: room {group.room!r} has no sensor grid; "
            "its blinds are kept open every hour"
        )

    # (6)+(7) sky lux, blinds open and (where the schedule actually closes)
    # blinds closed, every window group -- folded into the room sDA
    # accumulators as each group's own lux arrives (I2), the same way step
    # 4 did for direct/ASE above. sDA lux follows the blind schedule; ASE
    # lux (accumulated in step 4) is always the blinds-open direct lux
    # (R-P43), whatever the schedule says.
    open_sky_total: dict[str, float] = {grid.room: 0.0 for grid in prepared.grids}
    sky_traces: dict[str, int] = {grid.room: 0 for grid in prepared.grids}
    for number, group in enumerate(prepared.groups, start=1):
        group_schedule = schedule[group.group_id]
        sensor_index, open_lux, closed_lux = engine.sky(group, mask, group_schedule)
        _check_engine_shape("sky (open)", group.group_id, sensor_index, open_lux, n_selected)
        if closed_lux is None and not np.all(group_schedule):
            raise Option1Error(
                f"group {group.group_id!r}: its blind schedule closes in "
                "some occupied hour, but the engine returned no "
                "blinds-closed lux for it"
            )
        if closed_lux is not None:
            _check_engine_shape("sky (closed)", group.group_id, sensor_index, closed_lux, n_selected)
        for grid in prepared.grids:
            room_open = _select_room_rows(sensor_index, open_lux, grid.room)
            if room_open.shape[0] == 0:
                continue
            expected = sda_lux_by_room[grid.room].shape[0]
            if room_open.shape[0] != expected:
                raise Option1Error(
                    f"group {group.group_id!r}: engine.sky returned "
                    f"{room_open.shape[0]} row(s) for room {grid.room!r}, "
                    f"expected {expected} (the room's own sensor-point count)"
                )
            open_sky_total[grid.room] += float(room_open.sum())
            if closed_lux is not None:
                room_closed = _select_room_rows(sensor_index, closed_lux, grid.room)
                combined = np.where(group_schedule[None, :], room_open, room_closed)
            else:
                combined = room_open
            sda_lux_by_room[grid.room] += combined
            sky_traces[grid.room] += 1
        # `open_lux`/`closed_lux` are not referenced again after this
        # iteration.
        if progress is not None:
            progress(f"window group {number} of {n_groups} (sky)")

    # (7c) the residual, counted exactly once. A room covered by n group
    # traces holds n copies of it -- in the blinds-open and the
    # blinds-closed traces alike, since a closed blind covers only its own
    # group's openings -- so per point-hour: total = sum of the n traces
    # + (1 - n) x residual. n = 1 needs no correction (the residual pass
    # only detects); n = 0 adds it once. Clamped at 0 lux: the traces and
    # the residual are separate Monte Carlo samples.
    residual_pct: dict[str, float] = {}
    residual_warnings: list[str] = []
    ase_lux = rules["option1"]["ase"]["lux"]
    for grid in prepared.grids:
        room_sky = residual_sky_by_room[grid.room]
        room_direct = residual_direct_by_room[grid.room]
        sky_surplus = 1 - sky_traces[grid.room]
        direct_surplus = 1 - direct_traces[grid.room]
        if sky_surplus != 0:
            sda_lux_by_room[grid.room] += sky_surplus * room_sky.astype(np.float64)
            np.maximum(sda_lux_by_room[grid.room], 0.0, out=sda_lux_by_room[grid.room])
        if direct_surplus != 0:
            ase_lux_by_room[grid.room] += direct_surplus * room_direct.astype(np.float64)
            np.maximum(ase_lux_by_room[grid.room], 0.0, out=ase_lux_by_room[grid.room])

        # The share of the room's blinds-open annual sky illuminance, the
        # residual counted once in the total.
        residual = float(room_sky.sum(dtype=np.float64))
        total = open_sky_total[grid.room] + sky_surplus * residual
        share = min(max(residual / total, 0.0), 1.0) if total > 0.0 else 0.0
        residual_pct[grid.room] = 100.0 * share
        lit = int((room_direct >= ase_lux).any(axis=-1).sum())
        if share > RESIDUAL_WARNING_SHARE or lit > 0:
            message = (
                f"light reaches room {grid.room!r} through surfaces outside "
                f"every window group ({100.0 * share:.1f} % of its sky "
                "illuminance); check for skylights or gaps"
            )
            if lit > 0:
                message += (
                    f"; direct sun through them reaches {lit} sensor point(s) "
                    f"at {ase_lux} lux or more"
                )
            residual_warnings.append(message)

    # (8) the LM-83 metrics, per room -- every threshold from rules.
    sda_lux_threshold = rules["option1"]["sda"]["lux"]
    sda_fraction = rules["option1"]["sda"]["fraction"]
    ase_hour_limit = rules["option1"]["ase"]["hours"]

    point_rows: list[dict] = []
    rooms_data: list[dict] = []
    for grid in prepared.grids:
        da = daylight_autonomy(sda_lux_by_room[grid.room], sda_lux_threshold)
        passes = sda_pass(da, sda_fraction)
        hours_over = ase_hours(ase_lux_by_room[grid.room], ase_lux_threshold)
        exceeds = ase_exceeds(hours_over, ase_hour_limit)

        for i, (x, y, z) in enumerate(grid.points):
            point_rows.append({
                "room": grid.room, "storey": int(grid.storey),
                "x": float(x), "y": float(y), "z": float(z),
                "da_pct": 100.0 * float(da[i]), "sda_pass": bool(passes[i]),
                "ase_hours": int(hours_over[i]), "ase_exceeds": bool(exceeds[i]),
            })

        n_points = int(passes.shape[0])
        n_passing = int(passes.sum())
        n_exceeding = int(exceeds.sum())
        sda_pct = 100.0 * n_passing / n_points
        ase_pct = 100.0 * n_exceeding / n_points

        rooms_data.append({
            "room": grid.room,
            "area_m2": grid.room_area_m2,
            "points": n_points,
            "sda_pct": sda_pct,
            "ase_pct": ase_pct,
            "ase_band": ase_band(ase_pct),
            "sda_passing": n_passing,
            "ase_exceeding": n_exceeding,
        })

    # (9) both aggregations, for sDA and for ASE, always both -- weighted
    # from each room's exact integer point counts (never from the rounded
    # float percentage), so a room exactly on a band edge stays on it.
    areas = [d["area_m2"] for d in rooms_data]
    totals = [d["points"] for d in rooms_data]
    sda_agg = aggregate_counts(areas, [d["sda_passing"] for d in rooms_data], totals)
    ase_agg = aggregate_counts(areas, [d["ase_exceeding"] for d in rooms_data], totals)

    # (10) R-P42: the score, against whichever aggregation is the headline.
    headline_warnings: list[str] = []
    if project.analysis.aggregation == "floor_area":
        headline = "floor_area"
        headline_pct = sda_agg["area_pct"]
    else:
        headline = "points"
        headline_pct = sda_agg["point_pct"]
        headline_warnings.append(
            "analysis.aggregation is 'points'; LEED Table 1 is defined on "
            "the floor-area average, not the point-weighted figure"
        )
    score = score_option1(
        headline_pct, [d["sda_pct"] for d in rooms_data],
        rules["option1"]["sda"], rules["max_points"],
    )
    # Table 1 is defined on 08:00-18:00 and the floor-area average; any
    # other setting still computes, but its percentages are not a LEED result.
    basis_problems = []
    if project.analysis.hours != LEED_HOURS:
        basis_problems.append(
            f"analysis.hours is {project.analysis.hours!r}; LEED Option 1 is defined on {LEED_HOURS}"
        )
    if project.analysis.aggregation != "floor_area":
        basis_problems.append(
            f"analysis.aggregation is {project.analysis.aggregation!r}; LEED Option 1 is "
            "defined on the floor-area average"
        )
    if basis_problems:
        score = Option1Score(
            points=None, band=None, additional_point_awarded=False,
            reason="not scored: " + "; ".join(basis_problems),
        )
        if project.analysis.hours != LEED_HOURS:
            # The aggregation warning above already says its own part.
            headline_warnings.append(
                f"analysis.hours is {project.analysis.hours!r}, not {LEED_HOURS}; "
                "LEED Option 1 is defined on the latter, so no points are awarded"
            )

    # (11) the per-space glare-narrative trigger (spec defect S1) -- kept
    # apart from the LM-83 reporting band above (carry-forward item 9).
    per_space_ase_pct = {d["room"]: d["ase_pct"] for d in rooms_data}
    narrative_trigger = rules["option1"]["ase"]["per_space_narrative_trigger_pct"]
    narrative_spaces = spaces_needing_glare_narrative(per_space_ase_pct, narrative_trigger)
    narrative_set = set(narrative_spaces)

    space_results = tuple(
        SpaceResult(
            room=d["room"], area_m2=d["area_m2"], points=d["points"],
            sda_pct=d["sda_pct"], ase_pct=d["ase_pct"], ase_band=d["ase_band"],
            needs_glare_narrative=d["room"] in narrative_set,
            residual_sky_pct=residual_pct[d["room"]],
        )
        for d in rooms_data
    )

    # (12) warnings: the prepared model's own, the blind-trigger-area note,
    # every per-group warning from step 5, then the headline warning.
    warnings = (
        tuple(prepared.manifest.get("warnings", ()))
        + (_TRIGGER_AREA_NOTES[project.blinds.trigger_area],)
        + tuple(group_warnings)
        + tuple(residual_warnings)
        + tuple(headline_warnings)
    )

    return Option1Result(
        case=case,
        occupied_hours=n_selected,
        spaces=space_results,
        sda_area_pct=sda_agg["area_pct"],
        sda_point_pct=sda_agg["point_pct"],
        ase_area_pct=ase_agg["area_pct"],
        ase_point_pct=ase_agg["point_pct"],
        headline=headline,
        score=score,
        narrative_spaces=narrative_spaces,
        blind_schedule=schedule,
        warnings=warnings,
        daylight_saving_year=DST_YEAR,
        point_rows=tuple(point_rows),
    )
