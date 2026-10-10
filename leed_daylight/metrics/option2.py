# leed_daylight/metrics/option2.py
"""Option 2: point-in-time illuminance inside a band.

A point counts only if it is inside the band at every required time, not at one
of them.  The percentage of regularly occupied floor area that passes is what
the credit is scored on, and the area weighting happens in ``aggregate``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def band_pass(lux: np.ndarray, lux_min: float, lux_max: float) -> np.ndarray:
    """Points inside ``[lux_min, lux_max]`` at every time.  Both edges count.

    ``lux`` is point-major: ``(points, times)``.  Returns one bool per point.
    """
    if lux_min > lux_max:
        raise ValueError(f"lux_min {lux_min} is above lux_max {lux_max}")
    lux = np.asarray(lux, dtype=np.float64)
    if lux.ndim != 2:
        raise ValueError(f"illuminance needs two dimensions (points, times), got {lux.ndim}")
    if lux.shape[1] == 0:
        raise ValueError("need at least one time")
    if np.any(lux < 0.0):
        raise ValueError("illuminance contains a negative value")
    return ((lux >= lux_min) & (lux <= lux_max)).all(axis=-1)


def two_day_average(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """LEED: "Use the average of the hourly value for the two selected
    days." Both are ``(points, times)``; the mean per point and time."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(
            f"the two days' illuminance must have the same shape, got "
            f"{first.shape} and {second.shape}"
        )
    return (first + second) / 2.0


@dataclass(frozen=True)
class BandCounts:
    #: One bool per point: inside the band at every time.
    passing: np.ndarray
    #: Points below the minimum at one time or more.
    n_low: int
    #: Points above the maximum at one time or more (a point may be both).
    n_high: int


def band_counts(lux: np.ndarray, lux_min: float, lux_max: float | None) -> BandCounts:
    """``band_pass`` plus which way the failures miss. ``lux_max`` None
    means no upper limit: a space with view-preserving automatic glare
    control "may demonstrate compliance for only the minimum 300 lux"."""
    upper = np.inf if lux_max is None else lux_max
    passing = band_pass(lux, lux_min, upper)
    lux = np.asarray(lux, dtype=np.float64)
    n_low = int((lux < lux_min).any(axis=-1).sum())
    n_high = 0 if lux_max is None else int((lux > lux_max).any(axis=-1).sum())
    return BandCounts(passing=passing, n_low=n_low, n_high=n_high)
