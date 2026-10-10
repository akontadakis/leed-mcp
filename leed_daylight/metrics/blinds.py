# leed_daylight/metrics/blinds.py
"""The LM-83-12 clause 2.2.6 hourly blind protocol.

Every occupied hour, a space counts the grid points receiving 1,000 lux or more
of direct sunlight with all blinds open.  If that count exceeds the space's
threshold, its window groups close one at a time, the group that lights the most
points on its own going first, until the count is at or below the threshold.

The schedule this produces is what the sDA run consumes, so an error here moves
sDA without moving anything that looks wrong.
"""
from __future__ import annotations

import numpy as np

SQFT_PER_SQM = 10.763910416709722  # source: square feet per square metre, exactly 1 / 0.3048^2

#: (upper bound in square feet, allowed lit points).  Above the last bound the
#: allowance is a fraction of the point count instead.
AREA_BANDS = ((200.0, 1), (500.0, 3), (1000.0, 5))
LARGE_SPACE_FRACTION = 0.02

#: LM-83-12 2.2.6: a blind analysis area never exceeds 10,000 sf and is
#: considered by facade orientation.  The Option 1 run keeps one schedule per
#: room (spec ruling) and names every room over this area in a warning.
MAX_ANALYSIS_AREA_SQFT = 10000.0
MAX_ANALYSIS_AREA_M2 = MAX_ANALYSIS_AREA_SQFT / SQFT_PER_SQM


def allowed_lit_points(area_m2: float, n_points: int) -> float:
    """The largest number of lit points the space may have before blinds close."""
    if n_points < 1:
        raise ValueError("a space needs at least one point")
    area_sqft = area_m2 * SQFT_PER_SQM
    for upper, allowed in AREA_BANDS:
        if area_sqft < upper:
            return float(allowed)
    return LARGE_SPACE_FRACTION * n_points


def blind_schedule(
    direct_by_group: np.ndarray, area_m2: float, lit_threshold: float
) -> np.ndarray:
    """Which window groups are open, per hour.  True means open.

    ``direct_by_group`` is direct-sun illuminance in lux, shaped
    ``(n_groups, n_points, n_hours)``.  ``lit_threshold`` is the direct
    illuminance, in lux, at or above which a point counts as lit.
    """
    direct = np.asarray(direct_by_group, dtype=np.float64)
    if direct.ndim != 3:
        raise ValueError(
            f"direct illuminance needs three dimensions "
            f"(groups, points, hours), got {direct.ndim}"
        )
    n_groups, n_points, n_hours = direct.shape
    allowed = allowed_lit_points(area_m2, n_points)

    # How many points each group lights on its own, per hour.  This is the
    # closing order LM-83 asks for: most-lit group first.
    own_lit = (direct >= lit_threshold).sum(axis=1)          # (groups, hours)

    schedule = np.ones((n_groups, n_hours), dtype=bool)
    for hour in range(n_hours):
        lit = ((direct[:, :, hour].sum(axis=0)) >= lit_threshold).sum()
        if lit <= allowed:
            continue
        for group in np.argsort(-own_lit[:, hour], kind="stable"):
            schedule[group, hour] = False
            open_sum = direct[schedule[:, hour], :, hour].sum(axis=0)
            lit = (open_sum >= lit_threshold).sum() if schedule[:, hour].any() else 0
            if lit <= allowed:
                break
    return schedule
