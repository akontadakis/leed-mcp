# tests/test_scoring.py
import pytest

from leed_daylight.constants import UNVERIFIED
from leed_daylight.metrics.scoring import score, score_option1, spaces_needing_glare_narrative

BANDS = [
    {"min_pct": 40.0, "points": 1},
    {"min_pct": 55.0, "points": 2},
    {"min_pct": 75.0, "points": 3},
]


def test_below_the_lowest_band_scores_zero():
    result = score(39.9, BANDS)
    assert result.points == 0
    assert result.band is None


def test_the_highest_band_reached_wins():
    assert score(40.0, BANDS).points == 1
    assert score(54.9, BANDS).points == 1
    assert score(55.0, BANDS).points == 2
    assert score(99.0, BANDS).points == 3


def test_the_band_is_named_in_the_result():
    assert score(55.0, BANDS).band == "55.0 % or more"


def test_unverified_bands_award_no_points_and_say_why():
    result = score(80.0, "UNVERIFIED")
    assert result.points is None
    assert "UNVERIFIED" in result.reason
    assert "80.0" in result.reason


def test_a_percentage_outside_0_to_100_is_refused():
    with pytest.raises(ValueError, match="percentage"):
        score(101.0, BANDS)


# --- score_option1 -----------------------------------------------------
#
# Mirrors LEED v4.1 Table 1: three ascending bands plus a conditional
# additional point that only fires at exactly 2 band points, when every
# regularly occupied space is at or above 55 %.

ADDITIONAL_POINT = {
    "every_space_min_pct": 55.0,
    "only_if_points": 2,
    "points": 1,
}
SDA_RULES = {"bands": BANDS, "additional_point": ADDITIONAL_POINT}
MAX_POINTS = 3


def test_two_points_plus_the_additional_point():
    result = score_option1(60.0, [56.0, 70.0, 99.0], SDA_RULES, MAX_POINTS)
    assert result.points == 3
    assert result.additional_point_awarded is True


def test_one_low_space_loses_the_additional_point():
    result = score_option1(60.0, [56.0, 54.9, 99.0], SDA_RULES, MAX_POINTS)
    assert result.points == 2
    assert result.additional_point_awarded is False


def test_a_space_at_exactly_55_keeps_the_additional_point():
    result = score_option1(60.0, [55.0, 80.0], SDA_RULES, MAX_POINTS)
    assert result.additional_point_awarded is True


def test_three_points_is_exemplary_not_an_extra_point():
    result = score_option1(80.0, [90.0, 95.0], SDA_RULES, MAX_POINTS)
    assert result.points == 3
    assert result.additional_point_awarded is False
    assert "exemplary" in result.reason


def test_one_point_does_not_earn_the_additional_point():
    result = score_option1(45.0, [90.0, 95.0], SDA_RULES, MAX_POINTS)
    assert result.points == 1
    assert result.additional_point_awarded is False


def test_below_the_lowest_band_is_zero_points():
    result = score_option1(30.0, [10.0, 20.0], SDA_RULES, MAX_POINTS)
    assert result.points == 0
    assert result.additional_point_awarded is False


# NOTE: the brief names this test the same as the existing
# test_unverified_bands_award_no_points_and_say_why above, but that name is
# already taken in this module by the score() version of the same case
# (Python cannot have two functions with one name in one module -- the
# second definition would silently shadow the first, deleting its
# coverage). Distinguished here with a "_score_option1" suffix; behaviour
# and intent are exactly the brief's row.
def test_unverified_bands_award_no_points_and_say_why_score_option1():
    unverified_rules = {"bands": UNVERIFIED, "additional_point": ADDITIONAL_POINT}
    result = score_option1(80.0, [90.0, 95.0], unverified_rules, MAX_POINTS)
    assert result.points is None
    assert result.band is None
    assert result.additional_point_awarded is False
    assert UNVERIFIED in result.reason


def test_an_empty_space_list_is_refused():
    with pytest.raises(ValueError):
        score_option1(60.0, [], SDA_RULES, MAX_POINTS)


def test_no_additional_point_rule_awards_band_points_only():
    rules_without_additional_point = {"bands": BANDS}
    result = score_option1(60.0, [56.0, 70.0], rules_without_additional_point, MAX_POINTS)
    assert result.points == 2
    assert result.additional_point_awarded is False
    assert "no additional-point rule" in result.reason


def test_an_unverified_additional_point_awards_band_points_only():
    rules_with_unverified_additional_point = {"bands": BANDS, "additional_point": UNVERIFIED}
    result = score_option1(
        60.0, [56.0, 70.0], rules_with_unverified_additional_point, MAX_POINTS
    )
    assert result.points == 2
    assert result.additional_point_awarded is False
    assert UNVERIFIED in result.reason


# --- the additional-point reason must be true in every cell of the grid --
#
# Fix round 1, Important finding: the band-vs-only_if_points
# comparison has three outcomes (below / equal / above) and the per-space
# check has two (meets / does not meet every_space_min_pct). Only the
# equal+meets cell awards the point; the other five must each say something
# true about *why not* -- in particular the old code's catch-all "else"
# branch claimed a points deficiency ("requires 2 points, not 3") even when
# the band was ABOVE the requirement and the real blocker was a
# non-compliant space. avg 45.0 -> band 1 (below); avg 60.0 -> band 2
# (equal); avg 80.0 -> band 3 (above). only_if_points is 2.
@pytest.mark.parametrize(
    "average_pct, per_space_pct, expected_points, expected_awarded, must_contain, must_not_contain",
    [
        pytest.param(
            60.0, [60.0, 70.0], 3, True, ("is awarded",), (), id="equal-meets-awarded"
        ),
        pytest.param(
            60.0,
            [60.0, 10.0],
            2,
            False,
            ("not every regularly occupied space reaches",),
            ("exemplary", "requires"),
            id="equal-not_meets",
        ),
        pytest.param(
            80.0,
            [60.0, 70.0],
            3,
            False,
            ("exemplary performance, not an additional point",),
            ("requires",),
            id="above-meets-exemplary",
        ),
        pytest.param(
            # The reviewer's exact repro for the Important finding.
            80.0,
            [90.0, 10.0],
            3,
            False,
            ("would be exemplary performance", "not every regularly occupied space reaches"),
            ("requires 2 points, not 3",),
            id="above-not_meets",
        ),
        pytest.param(
            45.0,
            [60.0, 70.0],
            1,
            False,
            ("every regularly occupied space reaches", "requires 2 points, not 1"),
            ("exemplary",),
            id="below-meets",
        ),
        pytest.param(
            45.0,
            [60.0, 10.0],
            1,
            False,
            ("requires 2 points, not 1", "not every regularly occupied space reaches"),
            ("exemplary",),
            id="below-not_meets",
        ),
    ],
)
def test_the_additional_point_reason_is_true_in_every_grid_cell(
    average_pct, per_space_pct, expected_points, expected_awarded, must_contain, must_not_contain
):
    result = score_option1(average_pct, per_space_pct, SDA_RULES, MAX_POINTS)
    assert result.points == expected_points
    assert result.additional_point_awarded is expected_awarded
    for phrase in must_contain:
        assert phrase in result.reason
    for phrase in must_not_contain:
        assert phrase not in result.reason


# --- spaces_needing_glare_narrative --------------------------------------


def test_the_narrative_trigger_is_per_space():
    per_space_ase_pct = {"A": 9.9, "B": 10.0, "C": 10.1}
    assert spaces_needing_glare_narrative(per_space_ase_pct, 10.0) == ("C",)


def test_an_empty_mapping_needs_no_narrative():
    assert spaces_needing_glare_narrative({}, 10.0) == ()


# --- band edges and float error ------------------------------------------
#
# A percentage is compared to a band threshold after rounding to
# PCT_DECIMALS decimals, so a float error far below that precision can never
# move a result across a band edge.


def test_a_float_error_below_the_precision_does_not_drop_a_band():
    from leed_daylight.metrics.scoring import PCT_DECIMALS

    assert PCT_DECIMALS == 9
    assert score(55.0 - 1e-12, BANDS).points == 2
    # A real shortfall at the stated precision still drops the band.
    assert score(55.0 - 1e-5, BANDS).points == 1


def test_the_every_space_check_tolerates_float_error():
    just_under = 55.0 - 1e-12
    result = score_option1(just_under, [just_under, 60.0], SDA_RULES, MAX_POINTS)
    assert result.points == 3
    assert result.additional_point_awarded is True


def test_the_narrative_trigger_tolerates_float_error():
    assert spaces_needing_glare_narrative({"A": 10.0 + 1e-12, "B": 10.5}, 10.0) == ("B",)


def test_a_percentage_a_hair_over_100_is_not_refused():
    assert score(100.0 + 1e-12, BANDS).points == 3


def test_a_share_just_under_a_band_edge_does_not_reach_it():
    assert score(55.0 - 5e-7, BANDS).points == 1
    assert score(55.0 + 1e-10, BANDS).points == 2
