# tests/test_aggregate.py
import pytest

from leed_daylight.metrics.aggregate import aggregate, area_weighted, point_weighted


def test_area_weighted_of_a_single_space_is_its_own_fraction():
    assert area_weighted([100.0], [0.42]) == pytest.approx(42.0)


def test_area_weighted_favours_the_larger_space():
    # 90 m2 fully passing, 10 m2 failing.
    assert area_weighted([90.0, 10.0], [1.0, 0.0]) == pytest.approx(90.0)


def test_point_weighted_ignores_area():
    # The small space has most of the points, so the point figure is much lower.
    # source: 90/990 exactly
    assert point_weighted([90, 0], [90, 900]) == pytest.approx(9.090909, abs=1e-5)


def test_the_two_figures_differ_when_point_density_differs():
    areas = [90.0, 10.0]
    fractions = [1.0, 0.0]
    totals = [90, 900]
    passing = [90, 0]
    out = aggregate(areas, fractions, passing, totals)
    assert out["area_pct"] == pytest.approx(90.0)
    # source: 90/990 exactly
    assert out["point_pct"] == pytest.approx(9.090909, abs=1e-5)


def test_the_two_figures_agree_at_equal_density():
    out = aggregate([50.0, 50.0], [1.0, 0.0], [100, 0], [100, 100])
    assert out["area_pct"] == pytest.approx(50.0)
    assert out["point_pct"] == pytest.approx(50.0)


def test_empty_input_is_refused():
    with pytest.raises(ValueError, match="at least one space"):
        area_weighted([], [])


def test_zero_total_area_is_refused():
    with pytest.raises(ValueError, match="total area"):
        area_weighted([0.0, 0.0], [1.0, 1.0])


def test_negative_area_is_refused():
    with pytest.raises(ValueError, match="negative"):
        area_weighted([10.0, -1.0], [1.0, 1.0])


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="same length"):
        area_weighted([10.0, 20.0], [1.0])


def test_fraction_out_of_range_is_refused():
    with pytest.raises(ValueError, match="between 0 and 1"):
        area_weighted([10.0], [1.5])


def test_zero_total_points_is_refused():
    with pytest.raises(ValueError, match="total points"):
        point_weighted([0], [0])


def test_aggregate_refuses_mismatched_space_counts():
    areas = [10.0, 20.0, 30.0]
    fractions = [1.0, 0.0, 0.5]
    passing = [5, 5]
    totals = [10, 10]
    with pytest.raises(ValueError) as excinfo:
        aggregate(areas, fractions, passing, totals)
    message = str(excinfo.value)
    assert "3" in message
    assert "2" in message


def test_area_weighted_from_counts_is_exact():
    """Every space at exactly 11/20 must aggregate to exactly 55.0, whatever
    the invented areas; float re-weighting of 0.55 lands one unit in the last place under 55."""
    from leed_daylight.metrics.aggregate import aggregate_counts

    areas = [53.3, 66.7, 37.3, 24.6, 45.1, 73.1]
    out = aggregate_counts(areas, [11] * 6, [20] * 6)
    assert out["area_pct"] == 55.0
    assert out["point_pct"] == 55.0


def test_aggregate_counts_refuses_bad_counts():
    from leed_daylight.metrics.aggregate import aggregate_counts

    with pytest.raises(ValueError, match="more passing points"):
        aggregate_counts([10.0], [3], [2])
    with pytest.raises(ValueError, match="at least one point"):
        aggregate_counts([10.0], [0], [0])
