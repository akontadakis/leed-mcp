"""Turning a percentage into credit points.

A band set of UNVERIFIED means the LEED text has not been encoded yet.  The
score is then None, never zero, so a missing threshold can never be mistaken
for a failed credit.

This module stays layer 5: it imports nothing but ``constants`` and the
standard library (``dataclasses``, ``collections.abc``).  It never reads a
rules file itself -- the caller in a higher layer loads and validates the
rules mapping and passes the already-validated pieces in.

Band edges.  Every percentage is compared to a threshold (a band's
``min_pct``, ``every_space_min_pct``, the per-space narrative trigger) only
after rounding to ``PCT_DECIMALS`` decimal places (:func:`comparable_pct`).
A result that is exactly on an edge in exact arithmetic but lands a few
units in the last place below it in float (55 % re-weighted by floor area
can come out one unit in the last place under 55) is then still on the
edge.  Nine decimals of a percentage is far finer than one grid point in any
real space, yet well above float noise, so the rounding can never hide a
genuine shortfall (a result of 55 less 5e-7 is below the edge, not on it).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from leed_daylight.constants import UNVERIFIED

#: The precision, in decimal places of a percentage, at which a result is
#: compared to a threshold.  See the module docstring.
PCT_DECIMALS = 9


def comparable_pct(pct: float) -> float:
    """``pct`` rounded to ``PCT_DECIMALS`` places: the value every band,
    minimum and trigger comparison in this package uses."""
    return round(float(pct), PCT_DECIMALS)


@dataclass(frozen=True)
class Score:
    points: int | None
    band: str | None
    reason: str


def score(pct: float, bands) -> Score:
    """Points for a percentage, against ascending bands or UNVERIFIED.

    The comparison uses :func:`comparable_pct`, never the raw float.
    """
    pct = comparable_pct(pct)
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"percentage must be between 0 and 100, got {pct}")
    if bands == UNVERIFIED:
        return Score(
            points=None,
            band=None,
            reason=(
                f"result is {pct:.1f} %, but the point bands are {UNVERIFIED}: "
                "the LEED credit text has not been encoded yet"
            ),
        )
    best = None
    for band in sorted(bands, key=lambda b: float(b["min_pct"])):
        if pct >= comparable_pct(band["min_pct"]):
            best = band
    if best is None:
        return Score(0, None, f"result is {pct:.1f} %, below the lowest band")
    return Score(
        points=int(best["points"]),
        band=f"{float(best['min_pct'])} % or more",
        reason=f"result is {pct:.1f} %, which reaches the {best['min_pct']} % band",
    )


@dataclass(frozen=True)
class Option1Score:
    points: int | None                 # None while the bands are UNVERIFIED -- never 0
    band: str | None
    additional_point_awarded: bool
    reason: str


def _check_percentage(value: float, what: str) -> None:
    if not 0.0 <= comparable_pct(value) <= 100.0:
        raise ValueError(f"{what} must be between 0 and 100, got {value}")


def score_option1(
    average_pct: float,
    per_space_pct: Sequence[float],
    sda_rules: dict,
    max_points: int,
) -> Option1Score:
    """Option 1 (sDA) points, including the conditional additional point.

    ``sda_rules`` is the already-validated ``option1.sda`` mapping (``lux``,
    ``fraction``, ``bands``, and an optional ``additional_point``).  The band
    lookup itself is delegated to :func:`score`, never re-implemented here.

    The additional point (LEED v4.1 Table 1's last row) is awarded only when
    the average-sDA band result is *exactly* ``only_if_points`` and
    every regularly occupied space reaches ``every_space_min_pct``. At a
    higher band the same condition is "exemplary performance", not an extra
    point, and the credit as a whole is still capped at ``max_points``.

    An empty ``per_space_pct`` is refused: "every regularly occupied space"
    over an empty set would be vacuously true, and a vacuous extra LEED
    point is the worst failure this function could produce.
    """
    _check_percentage(average_pct, "average_pct")
    if not per_space_pct:
        raise ValueError(
            "per_space_pct must not be empty: the additional-point condition is "
            "'every regularly occupied space', and an empty set would satisfy "
            "that vacuously"
        )
    for i, pct in enumerate(per_space_pct):
        _check_percentage(pct, f"per_space_pct[{i}]")

    band_score = score(average_pct, sda_rules["bands"])
    if band_score.points is None:
        return Option1Score(
            points=None,
            band=None,
            additional_point_awarded=False,
            reason=band_score.reason,
        )

    additional_point = sda_rules.get("additional_point")
    if additional_point is None:
        return Option1Score(
            points=band_score.points,
            band=band_score.band,
            additional_point_awarded=False,
            reason=(
                f"{band_score.reason}; the rules file defines no additional-point "
                "rule for option 1"
            ),
        )
    if additional_point == UNVERIFIED:
        return Option1Score(
            points=band_score.points,
            band=band_score.band,
            additional_point_awarded=False,
            reason=f"{band_score.reason}; the additional point is {UNVERIFIED}",
        )

    only_if_points = int(additional_point["only_if_points"])
    every_space_min_pct = float(additional_point["every_space_min_pct"])
    additional_points = int(additional_point["points"])

    every_space_meets_minimum = all(
        comparable_pct(pct) >= comparable_pct(every_space_min_pct) for pct in per_space_pct
    )

    if band_score.points == only_if_points and every_space_meets_minimum:
        total_points = min(band_score.points + additional_points, max_points)
        return Option1Score(
            points=total_points,
            band=band_score.band,
            additional_point_awarded=True,
            reason=(
                f"{band_score.reason}; every regularly occupied space reaches "
                f"{every_space_min_pct} %, the additional point is awarded"
            ),
        )

    # Not awarded. The band-vs-condition comparison has three outcomes
    # (below / equal / above only_if_points) and the per-space check has
    # two (meets / does not meet every_space_min_pct); every one of the
    # five remaining cells below must say something TRUE about why, never
    # a generic "requires N points" that would be false when the band is
    # already above the requirement and a space, not a point shortfall, is
    # what blocks the award.
    if band_score.points == only_if_points:
        # every_space_meets_minimum is False here (the True case awarded above).
        reason = (
            f"{band_score.reason}; not every regularly occupied space reaches "
            f"{every_space_min_pct} %, so the additional point is not awarded"
        )
    elif band_score.points > only_if_points and every_space_meets_minimum:
        reason = (
            f"{band_score.reason}; every regularly occupied space reaches "
            f"{every_space_min_pct} %, but at {band_score.points} points this is "
            "exemplary performance, not an additional point"
        )
    elif band_score.points > only_if_points:
        # Above the condition, but a space still fails it: the extra band
        # points do not excuse a non-compliant space, and the surplus band
        # points must not be reported as if they were a deficiency.
        reason = (
            f"{band_score.reason}; at {band_score.points} points this would be "
            "exemplary performance, but not every regularly occupied space reaches "
            f"{every_space_min_pct} %, so the additional point does not apply"
        )
    elif every_space_meets_minimum:
        # Below the condition, spaces are compliant: the points shortfall
        # alone is why it is not awarded.
        reason = (
            f"{band_score.reason}; every regularly occupied space reaches "
            f"{every_space_min_pct} %, but the additional point requires "
            f"{only_if_points} points, not {band_score.points}"
        )
    else:
        # Below the condition AND a space fails it: state both true facts.
        reason = (
            f"{band_score.reason}; the additional point requires "
            f"{only_if_points} points, not {band_score.points}, and not every "
            f"regularly occupied space reaches {every_space_min_pct} % either"
        )

    return Option1Score(
        points=band_score.points,
        band=band_score.band,
        additional_point_awarded=False,
        reason=reason,
    )


def spaces_needing_glare_narrative(
    per_space_ase_pct: Mapping[str, float], trigger_pct: float
) -> tuple[str, ...]:
    """Names of the spaces whose ASE percentage exceeds ``trigger_pct``.

    Strictly greater than -- a space exactly at the trigger does not need
    the narrative.  Compared at :func:`comparable_pct` precision, so a float
    error cannot push a space that is exactly at the trigger over it.  This is a per-space check (spec defect S1): the project
    ASE figure is still computed and reported elsewhere, but it triggers
    nothing here. Names come back in the mapping's own iteration order.
    """
    return tuple(
        name for name, pct in per_space_ase_pct.items()
        if comparable_pct(pct) > comparable_pct(trigger_pct)
    )
