# leed_daylight/metrics/solar.py
"""Where the sun is, to Radiance 6.1's own formulas (``common/sun.c``), so
this tool and ``gensky``/``gendaylit`` agree on it (layer 5, pure).

Used by ``metrics.clearest_day`` for the Perez clearness index and by the
Option 2 driver to refuse a checked time with the sun down. Conventions are
Radiance's: longitude and meridian are degrees WEST positive inside the
formulas (so the east-positive ``longitude_E`` is negated here, exactly as
``engine.sky`` negates it for ``gensky -o``); the hour is standard time in
the meridian's time zone; the azimuth is degrees from south, positive
toward west, as ``gensky`` prints it.
"""
from __future__ import annotations

import calendar
import math

#: Any non-leap year; a typical-meteorological-year file has no year.
_YEAR = 2025


def day_of_year(month: int, day: int) -> int:
    """1-based day of a non-leap year. A ``ValueError`` for a date that
    does not exist (29 February included)."""
    if not 1 <= month <= 12 or not 1 <= day <= calendar.monthrange(_YEAR, month)[1]:
        raise ValueError(f"month {month} day {day} is not a date in a non-leap year")
    return sum(calendar.monthrange(_YEAR, m)[1] for m in range(1, month)) + day


def solar_position(
    day_of_year: int, hour: float, latitude_N: float, longitude_E: float,
    meridian_radiance: float,
) -> tuple[float, float]:
    """``(altitude_deg, azimuth_deg)`` at standard-time ``hour`` of
    ``day_of_year``, from Radiance's ``stadj``, ``sdec``, ``salt`` and
    ``sazi``."""
    latitude = math.radians(latitude_N)
    longitude_w = math.radians(-longitude_E)
    meridian_w = math.radians(meridian_radiance)
    jd = day_of_year
    solar_time = (
        hour
        + 0.170 * math.sin(4 * math.pi / 373 * (jd - 80))
        - 0.129 * math.sin(2 * math.pi / 355 * (jd - 8))
        + 12 * (meridian_w - longitude_w) / math.pi
    )
    # source: Radiance common/sun.c sdec(), with 2*pi/365 as Radiance 6.1a's gensky computes it
    declination = 0.4093 * math.sin(2 * math.pi / 365 * (jd - 81))
    hour_angle = solar_time * (math.pi / 12)
    altitude = math.asin(
        math.sin(latitude) * math.sin(declination)
        - math.cos(latitude) * math.cos(declination) * math.cos(hour_angle)
    )
    azimuth = -math.atan2(
        math.cos(declination) * math.sin(hour_angle),
        -math.cos(latitude) * math.sin(declination)
        - math.sin(latitude) * math.cos(declination) * math.cos(hour_angle),
    )
    return math.degrees(altitude), math.degrees(azimuth)
