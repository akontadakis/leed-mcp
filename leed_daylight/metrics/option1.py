# leed_daylight/metrics/option1.py
"""sDA and ASE, per point, to IES LM-83-12.

Both take illuminance in lux over the occupied hours only.  sDA uses the fully
inter-reflected result with the blinds operating.  ASE uses the direct-sun-only
result with the blinds open and the fixed shading still in the model.
"""
from __future__ import annotations

import numpy as np

from .scoring import comparable_pct

SDA_LUX = 300.0
SDA_FRACTION = 0.50
ASE_LUX = 1000.0
ASE_HOURS = 250

#: LM-83-12 3.1.3 per-space ASE reporting labels, as (upper bound, bound is
#: inclusive, label).  The standard says under 3 % is clearly acceptable, under
#: 7 % is neutral (nominally acceptable) and over 10 % is unsatisfactory; it
#: does not class 7-10 %, and the label says so rather than inventing a class.
#: Bounds: < 3, < 7, <= 10, > 10.  The last band is open-ended.
ASE_BANDS = (
    (3.0, False, "under 3 %: clearly acceptable"),
    (7.0, False, "3-7 %: neutral"),
    (10.0, True, "7-10 %: not classified by LM-83-12"),
    (float("inf"), True, "over 10 %: unsatisfactory"),
)


def _check(lux: np.ndarray, what: str) -> np.ndarray:
    lux = np.asarray(lux, dtype=np.float64)
    if lux.ndim != 2:
        raise ValueError(f"{what} needs two dimensions (points, hours), got {lux.ndim}")
    if lux.shape[1] == 0:
        raise ValueError("no occupied hours; the mask selected nothing")
    if np.any(lux < 0.0):
        raise ValueError(f"{what} contains a negative illuminance")
    return lux


def daylight_autonomy(lux: np.ndarray, threshold: float) -> np.ndarray:
    """Fraction of occupied hours at or above the threshold, per point."""
    return (_check(lux, "illuminance") >= threshold).mean(axis=-1)


def sda_pass(da: np.ndarray, fraction: float) -> np.ndarray:
    """Points that meet the daylight autonomy fraction.  True means the point passes."""
    return np.asarray(da, dtype=np.float64) >= fraction


def ase_hours(direct_lux: np.ndarray, threshold: float) -> np.ndarray:
    """Occupied hours of direct sunlight at or above the threshold, per point."""
    return (_check(direct_lux, "direct illuminance") >= threshold).sum(axis=-1)


def ase_exceeds(hours: np.ndarray, limit: int) -> np.ndarray:
    """Points exceeding the hour limit.  'More than', so the limit itself does not exceed."""
    return np.asarray(hours) > limit


def ase_band(pct: float) -> str:
    """The LM-83-12 3.1.3 label for a space's ASE percentage.

    Under 3 % is clearly acceptable; 3 % up to (not including) 7 % is
    neutral; 7 % to 10 % inclusive is not classed by the standard; over
    10 % is unsatisfactory.  The edges follow the standard's own "under" and
    "over" wording, and the comparison is made at
    ``scoring.PCT_DECIMALS`` precision so a float error cannot move a space
    across an edge.
    """
    pct = comparable_pct(pct)
    for upper, inclusive, label in ASE_BANDS:
        if pct < upper or (inclusive and pct == upper):
            return label
    return ASE_BANDS[-1][2]
