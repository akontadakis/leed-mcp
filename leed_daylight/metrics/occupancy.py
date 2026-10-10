"""The occupied-hour mask.

The weather file is in local standard time.  The analysis window is in local
clock time.  Inside the daylight-saving period the clock runs one hour ahead,
so the standard-time window moves one hour earlier.  Getting this wrong shifts
every annual result without any error appearing.

The clocks change at about 02:00 local time (US: 02:00 local; EU: 01:00
UTC, which is 01:00 to 03:00 local standard time depending on the zone).
Under a daylight-saving rule this module therefore refuses an occupied
window that opens before 03:00 (``OccupancyError``): only a window opening
at 03:00 or later is sure to open after the change on both changeover days.
For every window it accepts, the whole of the day daylight saving starts is
daylight time during occupied hours, and the whole of the day it ends is
already standard time: the period is ``start <= day < end``.  With
``daylight_saving: none`` any window is accepted.

The calendar year is ``DST_YEAR`` (2025) unless a caller passes another.  A
typical-meteorological-year weather file has no real year of its own; its
8760 hours are laid on the 2025 calendar (a non-leap year starting on a
Wednesday) only to find the two changeover Sundays, which move by at most a
few days from year to year.  The Option 1 run records the year it used in
``results.json`` under ``settings.daylight_saving_year``.
"""
from __future__ import annotations

import datetime
import re

import numpy as np

HOURS_RE = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$")
HOURS_IN_YEAR = 8760

#: The calendar year the daylight-saving changeover dates are taken from for a
#: typical-meteorological-year weather file.  See the module docstring.
DST_YEAR = 2025

#: Under daylight saving, the earliest clock hour an occupied window may
#: open at: after the changeover (at most 03:00 local) on both days.
EARLIEST_DST_START_H = 3


class OccupancyError(ValueError):
    """The occupied-hours window cannot be turned into a mask."""


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime.date:
    """The nth (1-based) given weekday of a month.  Monday is 0."""
    first = datetime.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + datetime.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> datetime.date:
    last = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
    return last - datetime.timedelta(days=(last.weekday() - weekday) % 7)


def dst_bounds(year: int, rule: str):
    """The date daylight saving starts and the date it ends, or None.

    Occupied hours on the start date are daylight time; occupied hours on
    the end date are already standard time (both changes happen at about
    02:00, before any occupied window ``occupied_mask`` accepts under
    daylight saving opens).
    """
    if rule == "none":
        return None
    if rule == "EU":
        return _last_weekday(year, 3, 6), _last_weekday(year, 10, 6)
    if rule == "US":
        return _nth_weekday(year, 3, 6, 2), _nth_weekday(year, 11, 6, 1)
    raise ValueError(f"unknown daylight saving rule {rule!r}; expected EU, US or none")


def occupied_mask(hours: str, daylight_saving: str, year: int = DST_YEAR) -> np.ndarray:
    """Boolean mask of length 8760, indexed by local standard time.

    Hour ``i`` covers the interval ``[i, i+1)`` of standard time on day ``i//24``.
    ``year`` only fixes the daylight-saving changeover dates; it defaults to
    ``DST_YEAR`` (2025) for a typical-meteorological-year weather file.
    Under a daylight-saving rule, a window opening before
    ``EARLIEST_DST_START_H`` (03:00) is refused (module docstring).
    """
    match = HOURS_RE.match(hours)
    if not match:
        raise OccupancyError(f"hours must look like '08:00-18:00', got {hours!r}")
    start_h, start_m, end_h, end_m = (int(g) for g in match.groups())
    if (start_m, end_m) != (0, 0):
        raise OccupancyError(f"hours must be whole hours, got {hours!r}")
    if not 0 <= start_h < end_h <= 24:
        raise OccupancyError(f"hours must run forwards within one day, got {hours!r}")

    bounds = dst_bounds(year, daylight_saving)
    if bounds is not None and start_h < EARLIEST_DST_START_H:
        raise OccupancyError(
            f"analysis.hours {hours!r} opens before "
            f"{EARLIEST_DST_START_H:02d}:00 under daylight saving "
            f"{daylight_saving!r}; the clocks change at about 02:00 local "
            "time, inside this window on the two changeover days, which the "
            f"mask does not model -- open the window at "
            f"{EARLIEST_DST_START_H:02d}:00 or later, or set "
            "daylight_saving to none"
        )
    jan_1 = datetime.date(year, 1, 1)
    mask = np.zeros(HOURS_IN_YEAR, dtype=bool)
    for index in range(HOURS_IN_YEAR):
        day = jan_1 + datetime.timedelta(days=index // 24)
        standard_hour = index % 24
        # End bound exclusive: the clocks go back before the occupied
        # window opens, so the end date is standard time all occupied day.
        shift = 1 if bounds and bounds[0] <= day < bounds[1] else 0
        clock_hour = standard_hour + shift
        mask[index] = start_h <= clock_hour < end_h
    return mask
