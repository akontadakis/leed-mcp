# leed_daylight/metrics/aggregate.py
"""Rolling per-space results up to a storey or a project.

Two figures exist and they are not the same.  Floor-area weighting is the
reported one; point weighting is printed beside it so a large difference is
visible.  They agree only when every space has the same point density.

:func:`aggregate_counts` is the one the Option 1 run uses: it weights each
space's exact ``passing / total`` point counts by its floor area in exact
rational arithmetic (``fractions.Fraction``) and rounds to float once, at the
end.  Six spaces each at exactly 11 of 20 points therefore aggregate to
exactly 55.0, whatever their areas -- re-weighting the float 0.55 can give
55 minus one unit in the last place, which falls below a 55 % band.
"""
from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction


def _check_pairs(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) != len(b):
        raise ValueError(f"inputs must be the same length, got {len(a)} and {len(b)}")
    if not a:
        raise ValueError("need at least one space")


def area_weighted(areas: Sequence[float], fractions: Sequence[float]) -> float:
    """Percentage of floor area passing, weighted by each space's floor area."""
    _check_pairs(areas, fractions)
    if any(area < 0 for area in areas):
        raise ValueError("a space has a negative floor area")
    if any(not 0.0 <= f <= 1.0 for f in fractions):
        raise ValueError("every passing fraction must be between 0 and 1")
    total = sum(areas)
    if total <= 0.0:
        raise ValueError("total area is zero; cannot weight by area")
    return 100.0 * sum(a * f for a, f in zip(areas, fractions)) / total


def point_weighted(passing: Sequence[int], totals: Sequence[int]) -> float:
    """Percentage of grid points passing, ignoring floor area."""
    _check_pairs(passing, totals)
    if any(p < 0 or t < 0 for p, t in zip(passing, totals)):
        raise ValueError("a space has a negative point count")
    if any(p > t for p, t in zip(passing, totals)):
        raise ValueError("a space has more passing points than points")
    total = sum(totals)
    if total <= 0:
        raise ValueError("total points is zero; cannot weight by points")
    return 100.0 * sum(passing) / total


def aggregate(
    areas: Sequence[float],
    fractions: Sequence[float],
    passing: Sequence[int],
    totals: Sequence[int],
) -> dict[str, float]:
    """Both figures together, so a report can never show one without the other."""
    if len(areas) != len(passing):
        raise ValueError(
            f"areas/fractions describe {len(areas)} spaces but "
            f"passing/totals describe {len(passing)} spaces"
        )
    return {
        "area_pct": area_weighted(areas, fractions),
        "point_pct": point_weighted(passing, totals),
    }


def aggregate_counts(
    areas: Sequence[float], passing: Sequence[int], totals: Sequence[int],
) -> dict[str, float]:
    """Both figures from exact integer point counts, never from a float
    fraction.

    Each space's passing fraction is ``Fraction(passing, total)``, computed
    once; the area weighting is done in exact rational arithmetic over the
    areas' own float values, and each figure is converted to float once, at
    the end.  A space with no points is refused, since its fraction is
    undefined.
    """
    _check_pairs(areas, passing)
    _check_pairs(passing, totals)
    if any(area < 0 for area in areas):
        raise ValueError("a space has a negative floor area")
    if any(p < 0 or t < 0 for p, t in zip(passing, totals)):
        raise ValueError("a space has a negative point count")
    if any(p > t for p, t in zip(passing, totals)):
        raise ValueError("a space has more passing points than points")
    if any(t == 0 for t in totals):
        raise ValueError("every space needs at least one point")
    exact_areas = [Fraction(area) for area in areas]
    total_area = sum(exact_areas, Fraction(0))
    if total_area <= 0:
        raise ValueError("total area is zero; cannot weight by area")
    weighted = sum(
        (a * Fraction(p, t) for a, p, t in zip(exact_areas, passing, totals)),
        Fraction(0),
    )
    return {
        "area_pct": float(100 * weighted / total_area),
        "point_pct": float(Fraction(100 * sum(passing), sum(totals))),
    }
