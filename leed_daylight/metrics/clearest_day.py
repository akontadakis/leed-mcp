# leed_daylight/metrics/clearest_day.py
"""Option 2's two days, and the irradiance at a checked time (layer 5, pure).

LEED v4.1 Option 2: "Select one day within 15 days of September 21 and one
day within 15 days of March 21 that represent the clearest sky condition."
The day is picked from the weather file's own hourly irradiance (Option 2
design, 4.1): the most summed direct normal irradiance over its 24 records
wins; an exact tie goes to the day closer to the equinox, then the earlier
day. The Perez clearness index (4.2) is computed for every candidate as a
check; when it prefers another day, the choice stands and a warning says so.

Arrays in, frozen results out: ``project.weather`` reads the file.
"""
from __future__ import annotations

import datetime
import math
from dataclasses import dataclass

import numpy as np

from .solar import day_of_year, solar_position

HOURS_PER_DAY = 24
HOURS_IN_YEAR = 8760
#: An hour counts toward a day's Perez index only with the sun at least this
#: high; near the horizon the zenith term dominates and the index is noise.
MIN_EPSILON_ALTITUDE_DEG = 5.0
#: Perez et al. (1990), the zenith coefficient of the clearness index.
PEREZ_KAPPA = 1.041
#: How many best candidates a choice records.
TOP_CANDIDATES = 5
#: Any non-leap year; a typical-meteorological-year file has no year.
_YEAR = 2025


class ClearestDayError(ValueError):
    """The clearest day cannot be picked from these inputs."""


@dataclass(frozen=True)
class DayCandidate:
    month: int
    day: int
    #: Days from the equinox date; negative is before it.
    offset_days: int
    #: The day's 24 direct normal records summed, Wh/m2.
    direct_normal_wh: float
    #: The day's mean hourly Perez clearness; None with no qualifying hour.
    epsilon: float | None


@dataclass(frozen=True)
class DayChoice:
    equinox: tuple[int, int]
    chosen: DayCandidate
    #: The best TOP_CANDIDATES by the ranking, best first.
    top: tuple[DayCandidate, ...]
    #: The candidate with the highest Perez index; None if none has one.
    epsilon_best: DayCandidate | None
    warning: str | None
    #: The chosen day's 24 records, W/m2: direct normal, diffuse horizontal.
    hourly_direct_normal: tuple[float, ...]
    hourly_diffuse_horizontal: tuple[float, ...]


def _year_arrays(direct_normal, diffuse_horizontal) -> tuple[np.ndarray, np.ndarray]:
    dni = np.asarray(direct_normal, dtype=np.float64)
    dhi = np.asarray(diffuse_horizontal, dtype=np.float64)
    if dni.shape != (HOURS_IN_YEAR,) or dhi.shape != (HOURS_IN_YEAR,):
        raise ClearestDayError(
            f"irradiance must hold {HOURS_IN_YEAR} hours each, got "
            f"{dni.shape} and {dhi.shape}"
        )
    return dni, dhi


def _mmdd(month: int, day: int) -> str:
    return f"{month:02d}-{day:02d}"


def perez_epsilon(dni: float, dhi: float, altitude_deg: float) -> float | None:
    """Perez sky clearness epsilon for one hour. None when the diffuse
    irradiance is not positive or the sun is below MIN_EPSILON_ALTITUDE_DEG."""
    if dhi <= 0.0 or altitude_deg < MIN_EPSILON_ALTITUDE_DEG:
        return None
    zenith = math.radians(90.0 - altitude_deg)
    k = PEREZ_KAPPA * zenith ** 3
    return ((dhi + dni) / dhi + k) / (1.0 + k)


def _candidate_dates(equinox: tuple[int, int], window_days: int) -> list[tuple[int, int, int]]:
    centre = datetime.date(_YEAR, *equinox)
    dates = []
    for offset in range(-window_days, window_days + 1):
        date = centre + datetime.timedelta(days=offset)
        if date.year != _YEAR:
            raise ClearestDayError(
                f"the {window_days}-day window around {_mmdd(*equinox)} runs outside "
                "one calendar year"
            )
        dates.append((date.month, date.day, offset))
    return dates


def _rank_key(c: DayCandidate):
    return (-c.direct_normal_wh, abs(c.offset_days), c.offset_days)


def _epsilon_key(c: DayCandidate):
    return (-c.epsilon, abs(c.offset_days), c.offset_days)


def choose_clearest_day(
    direct_normal, diffuse_horizontal, *, equinox: tuple[int, int], window_days: int,
    latitude_N: float, longitude_E: float, meridian_radiance: float,
) -> DayChoice:
    dni, dhi = _year_arrays(direct_normal, diffuse_horizontal)
    candidates: list[DayCandidate] = []
    for month, day, offset in _candidate_dates(tuple(equinox), window_days):
        doy = day_of_year(month, day)
        start = (doy - 1) * HOURS_PER_DAY
        day_dni = dni[start:start + HOURS_PER_DAY]
        day_dhi = dhi[start:start + HOURS_PER_DAY]
        hourly = []
        for h in range(HOURS_PER_DAY):
            altitude, _ = solar_position(
                doy, h + 0.5, latitude_N, longitude_E, meridian_radiance,
            )
            value = perez_epsilon(float(day_dni[h]), float(day_dhi[h]), altitude)
            if value is not None:
                hourly.append(value)
        candidates.append(DayCandidate(
            month=month, day=day, offset_days=offset,
            direct_normal_wh=float(day_dni.sum()),
            epsilon=sum(hourly) / len(hourly) if hourly else None,
        ))

    ranked = sorted(candidates, key=_rank_key)
    chosen = ranked[0]
    label = _mmdd(*equinox)
    if chosen.direct_normal_wh <= 0.0:
        raise ClearestDayError(
            f"no day within {window_days} days of {label} has any direct sun in the "
            "weather file; Option 2 needs a clear day near each equinox"
        )
    with_epsilon = [c for c in candidates if c.epsilon is not None]
    epsilon_best = min(with_epsilon, key=_epsilon_key) if with_epsilon else None
    warning = None
    if epsilon_best is not None and (epsilon_best.month, epsilon_best.day) != (chosen.month, chosen.day):
        chosen_eps = "none" if chosen.epsilon is None else f"{chosen.epsilon:.2f}"
        warning = (
            f"clearest day near {label}: direct sun picks "
            f"{_mmdd(chosen.month, chosen.day)} ({chosen.direct_normal_wh:.0f} Wh/m2, "
            f"clearness {chosen_eps}), but the Perez clearness index prefers "
            f"{_mmdd(epsilon_best.month, epsilon_best.day)} (clearness "
            f"{epsilon_best.epsilon:.2f}, {epsilon_best.direct_normal_wh:.0f} Wh/m2); "
            "the direct-sun choice stands"
        )
    start = (day_of_year(chosen.month, chosen.day) - 1) * HOURS_PER_DAY
    return DayChoice(
        equinox=(int(equinox[0]), int(equinox[1])),
        chosen=chosen,
        top=tuple(ranked[:TOP_CANDIDATES]),
        epsilon_best=epsilon_best,
        warning=warning,
        hourly_direct_normal=tuple(float(v) for v in dni[start:start + HOURS_PER_DAY]),
        hourly_diffuse_horizontal=tuple(float(v) for v in dhi[start:start + HOURS_PER_DAY]),
    )


def irradiance_at(
    direct_normal, diffuse_horizontal, *, month: int, day: int, hour: int,
) -> tuple[float, float]:
    """DNI and DHI at exactly ``hour``:00, W/m2: the mean of the two
    half-hour records either side of it (Option 2 design, 4.3 item 2)."""
    dni, dhi = _year_arrays(direct_normal, diffuse_horizontal)
    if isinstance(hour, bool) or not isinstance(hour, int) or not 1 <= hour <= 23:
        raise ClearestDayError(
            f"hour must be a whole hour from 1 to 23, got {hour!r}"
        )
    after = (day_of_year(month, day) - 1) * HOURS_PER_DAY + hour  # stamped hour + 0.5
    return (
        float((dni[after - 1] + dni[after]) / 2.0),
        float((dhi[after - 1] + dhi[after]) / 2.0),
    )
