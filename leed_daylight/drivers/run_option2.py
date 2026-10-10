# leed_daylight/drivers/run_option2.py
"""The LEED v4.1 Option 2 run (layer 7; Option 2 design, section 4).

1. Refuse, by name, any Option 2 setting this tool cannot run
   (``project.checks.check_option2``), a case that does not match the
   prepared model, or grids that do not match the case's regularly
   occupied spaces.
2. Pick the clearest day near each equinox from the weather file
   (``metrics.clearest_day``).
3. One sky moment per day and time: the irradiance at the hour
   (interpolated), the sun's position; refuse a sun at or below the
   horizon and zero diffuse light, since gendaylit would write an
   all-zero error sky and exit 0.
4. Lux at every grid point for every moment, from the engine.
5. Per time, the two days' average; per space, the band (no upper limit
   with view-preserving shades); both aggregations; the score on the
   floor-area figure.

Every threshold comes from ``project.option2`` or ``rules``.
"""
from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from ..build.prepare import PreparedModel
from ..engine.point_trace import trace_points
from ..engine.radiance import Radiance
from ..engine.sky import weather_sky
from ..metrics.aggregate import aggregate_counts
from ..metrics.clearest_day import ClearestDayError, DayChoice, choose_clearest_day, irradiance_at
from ..metrics.option2 import band_counts, two_day_average
from ..metrics.scoring import Score, score
from ..metrics.solar import day_of_year, solar_position
from ..project.checks import check_option2, parse_mmdd, parse_whole_hour
from ..project.config import Project
from ..project.spaces import Space, included
from ..project.weather import Weather


#: The checked hours LEED v4.1 Option 2 is defined at.
LEED_HOURS = (9, 15)


class Option2Error(ValueError):
    """Option 2 cannot be run to completion from these inputs."""


@dataclasses.dataclass(frozen=True)
class SkyMoment:
    label: str  # "MM-DD_HH00"
    month: int
    day: int
    hour: int
    dni: float  # W/m2, interpolated at the hour
    dhi: float
    altitude_deg: float
    azimuth_deg: float


class Option2Engine(Protocol):
    def illuminance(self, moment: SkyMoment) -> np.ndarray:
        """Lux at every grid point, ``prepared.grids`` order, concatenated."""


class RadianceOption2Engine:
    """The real engine: one gendaylit sky and one cached rtrace per moment."""

    def __init__(self, rad: Radiance, project: Project, prepared: PreparedModel,
                 work_dir: pathlib.Path, processes: int = 1) -> None:
        self._rad = rad
        self._project = project
        self._prepared = prepared
        self._work_dir = pathlib.Path(work_dir)
        self._processes = processes
        self._points = np.vstack([g.points for g in prepared.grids])

    def illuminance(self, moment: SkyMoment) -> np.ndarray:
        sky = weather_sky(
            self._rad, self._work_dir / "skies" / f"{moment.label}.sky",
            moment.month, moment.day, float(moment.hour), moment.dni, moment.dhi,
            self._project.site, self._project.site.north_rotation_deg,
        )
        return trace_points(
            self._rad, materials=self._prepared.materials_file,
            scene=self._prepared.scene_file, sky=sky, points=self._points,
            ambient=self._project.radiance.ambient_full,
            direct=self._project.radiance.direct,
            work_dir=self._work_dir / "engine", name=moment.label,
            processes=self._processes,
        )


@dataclasses.dataclass(frozen=True)
class SpaceResult2:
    room: str
    area_m2: float  # the room's area from the scene, not the grid's
    points: int
    passing: int
    pass_pct: float
    n_low: int
    n_high: int
    upper_limit_applied: bool


@dataclasses.dataclass(frozen=True)
class Option2Result:
    case: str
    days: tuple[DayChoice, ...]
    moments: tuple[SkyMoment, ...]
    spaces: tuple[SpaceResult2, ...]
    area_pct: float
    point_pct: float
    score: Score
    warnings: tuple[str, ...]
    point_rows: tuple[dict, ...]
    settings: dict


def _check_case(prepared: PreparedModel, spaces: Sequence[Space], case: str) -> None:
    if case != prepared.case:
        raise Option2Error(f"case {case!r} does not match prepared.case {prepared.case!r}")
    occupied = {s.room for s in included(spaces, case) if s.regularly_occupied}
    grid_rooms = {g.room for g in prepared.grids}
    if occupied != grid_rooms:
        raise Option2Error(
            "prepared.grids does not match the case's regularly occupied spaces: "
            f"missing a grid for {sorted(occupied - grid_rooms)}, and a grid with no "
            f"regularly occupied space for {sorted(grid_rooms - occupied)}"
        )
    if not prepared.grids:
        raise Option2Error(f"case {case!r} has no regularly occupied space to score")


def _moments(project: Project, weather: Weather, days: Sequence[DayChoice]) -> list[SkyMoment]:
    site = project.site
    hours = [parse_whole_hour(t) for t in project.option2.times]
    moments = []
    for choice in days:
        month, day = choice.chosen.month, choice.chosen.day
        for hour in hours:
            dni, dhi = irradiance_at(
                weather.direct_normal, weather.diffuse_horizontal,
                month=month, day=day, hour=hour,
            )
            altitude, azimuth = solar_position(
                day_of_year(month, day), float(hour),
                site.latitude_N, site.longitude_E, site.meridian_radiance,
            )
            when = f"{month:02d}-{day:02d} at {hour:02d}:00"
            if altitude <= 0.0:
                raise Option2Error(
                    f"{when} (weather-file time) the sun is below the horizon "
                    f"(altitude {altitude:.1f} deg); gendaylit would write an all-zero sky"
                )
            if dhi <= 0.0:
                raise Option2Error(
                    f"{when} (weather-file time) the diffuse irradiance is 0 W/m2 in the "
                    "weather file; gendaylit would write an all-zero sky"
                )
            moments.append(SkyMoment(
                label=f"{month:02d}-{day:02d}_{hour:02d}00", month=month, day=day,
                hour=hour, dni=dni, dhi=dhi, altitude_deg=altitude, azimuth_deg=azimuth,
            ))
    return moments


def plan_option2(
    project: Project, weather: Weather,
) -> tuple[tuple[DayChoice, ...], tuple[SkyMoment, ...]]:
    """Steps 2 and 3 alone: the clearest day near each equinox and one sky
    moment per day and time (day-major, then time). Pure: no Radiance runs
    and nothing is written, so the option2 command refuses these inputs
    before it prepares the model. Raises Option2Error."""
    spec = project.option2
    site = project.site
    try:
        days = tuple(
            choose_clearest_day(
                weather.direct_normal, weather.diffuse_horizontal,
                equinox=parse_mmdd(text), window_days=spec.clearest_day_window_days,
                latitude_N=site.latitude_N, longitude_E=site.longitude_E,
                meridian_radiance=site.meridian_radiance,
            )
            for text in spec.equinox_dates
        )
    except ClearestDayError as exc:
        raise Option2Error(str(exc)) from None
    return days, tuple(_moments(project, weather, days))


def run_option2(
    rad: Radiance | None, project: Project, prepared: PreparedModel,
    spaces: Sequence[Space], rules: dict, case: str, *, weather: Weather,
    engine: Option2Engine | None = None, work_dir: pathlib.Path | None = None,
    processes: int = 1, progress: Callable[[str], None] | None = None,
) -> Option2Result:
    """The Option 2 run (module docstring). ``rad`` and ``work_dir`` are
    used only to build the real engine when ``engine`` is None.
    ``progress``, when given, is called with one short text after each sky
    (wizard spec 8.4)."""
    errors = check_option2(project)
    if errors:
        raise Option2Error("Option 2 cannot run: " + "; ".join(errors))
    _check_case(prepared, spaces, case)
    spec = project.option2
    site = project.site
    # The checked hours are the weather file's own hours (decision 40), and
    # the sun is placed with site.meridian_radiance; the two must be one clock.
    if abs(weather.time_zone - site.meridian_radiance) > 1e-6:
        raise Option2Error(
            f"the weather file's time_zone ({weather.time_zone:g}) differs from "
            f"project.yaml site.meridian_radiance ({site.meridian_radiance:g}); Option 2 "
            "checks the weather file's own hours, so both must name the same time zone"
        )

    days, moments = plan_option2(project, weather)

    if engine is None:
        if rad is None or work_dir is None:
            raise Option2Error("run_option2 needs rad and work_dir when no engine is given")
        engine = RadianceOption2Engine(rad, project, prepared, work_dir, processes)

    n_points = sum(int(g.points.shape[0]) for g in prepared.grids)
    lux_by_label: dict[str, np.ndarray] = {}
    for number, moment in enumerate(moments, start=1):
        lux = np.asarray(engine.illuminance(moment), dtype=np.float64)
        if lux.shape != (n_points,):
            raise Option2Error(
                f"sky {moment.label}: the engine returned lux of shape {lux.shape}, "
                f"expected ({n_points},), one per grid point"
            )
        lux_by_label[moment.label] = lux
        if progress is not None:
            progress(f"sky {number} of {len(moments)} ({moment.label})")

    # (points, times) per day, then the two-day average. The slicing relies on
    # plan_option2's moments being day-major, then time.
    n_times = len(spec.times)
    per_day = [
        np.stack([lux_by_label[m.label] for m in moments[i * n_times:(i + 1) * n_times]], axis=1)
        for i in range(len(days))
    ]
    average = two_day_average(per_day[0], per_day[1])

    by_room = {s.room: s for s in included(spaces, case)}
    space_results: list[SpaceResult2] = []
    point_rows: list[dict] = []
    start = 0
    for grid in prepared.grids:
        n = int(grid.points.shape[0])
        rows = slice(start, start + n)
        upper = None if by_room[grid.room].view_preserving_shades else spec.lux_max
        counts = band_counts(average[rows], spec.lux_min, upper)
        passing = int(counts.passing.sum())
        space_results.append(SpaceResult2(
            room=grid.room, area_m2=float(grid.room_area_m2), points=n, passing=passing,
            pass_pct=100.0 * passing / n, n_low=counts.n_low, n_high=counts.n_high,
            upper_limit_applied=upper is not None,
        ))
        for i in range(n):
            point = grid.points[i]
            point_rows.append({
                "room": grid.room,
                "x": float(point[0]), "y": float(point[1]), "z": float(point[2]),
                "lux": {m.label: float(lux_by_label[m.label][start + i]) for m in moments},
                "average": {t: float(average[start + i, j]) for j, t in enumerate(spec.times)},
                "passes": bool(counts.passing[i]),
            })
        start += n

    aggregated = aggregate_counts(
        [s.area_m2 for s in space_results],
        [s.passing for s in space_results],
        [s.points for s in space_results],
    )
    warnings = [*prepared.manifest.get("warnings", ()), *(d.warning for d in days if d.warning)]
    settings = {
        "equinox_dates": list(spec.equinox_dates),
        "clearest_day_window_days": spec.clearest_day_window_days,
        "times": list(spec.times),
        "time_basis": spec.time_basis,
        "sky": spec.sky,
        "lux_min": spec.lux_min,
        "lux_max": spec.lux_max,
        "blinds": spec.blinds,
        "radiance_ambient": project.radiance.ambient_full,
        "radiance_direct": project.radiance.direct,
    }
    # The rules file holds the LEED band and times; a project that changes
    # them still computes, but its share is not a LEED result.
    basis_problems = []
    rules_option2 = rules["option2"]
    rules_band = (rules_option2["lux_min"], rules_option2["lux_max"])
    if (spec.lux_min, spec.lux_max) != tuple(float(v) for v in rules_band):
        basis_problems.append(
            f"the project's band is {spec.lux_min:g} to {spec.lux_max:g} lux, "
            f"but LEED Option 2 is {rules_band[0]:g} to {rules_band[1]:g} lux"
        )
    if tuple(sorted(parse_whole_hour(t) for t in spec.times)) != LEED_HOURS:
        basis_problems.append(
            f"option2.times is {list(spec.times)!r}; LEED Option 2 is defined at "
            "9:00 and 15:00"
        )
    if basis_problems:
        result_score = Score(
            points=None, band=None, reason="not scored: " + "; ".join(basis_problems),
        )
        warnings.extend(f"{p}, so no points are awarded" for p in basis_problems)
    else:
        result_score = score(aggregated["area_pct"], rules_option2["bands"])
    return Option2Result(
        case=case, days=days, moments=tuple(moments), spaces=tuple(space_results),
        area_pct=aggregated["area_pct"], point_pct=aggregated["point_pct"],
        score=result_score,
        warnings=tuple(warnings), point_rows=tuple(point_rows), settings=settings,
    )
