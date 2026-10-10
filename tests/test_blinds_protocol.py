# tests/test_blinds_protocol.py
import numpy as np
import pytest

from leed_daylight.metrics.blinds import allowed_lit_points, blind_schedule
from leed_daylight.metrics.option1 import ASE_LUX

# source: square foot, exactly 0.3048^2 m2
SQM_PER_SQFT = 0.09290304


def sqft(value: float) -> float:
    return value * SQM_PER_SQFT


def test_thresholds_follow_the_four_area_bands():
    # Under 200 sf: more than 1 point triggers, so 1 is allowed.
    assert allowed_lit_points(sqft(199), 50) == 1
    assert allowed_lit_points(sqft(200), 50) == 3
    assert allowed_lit_points(sqft(499), 50) == 3
    assert allowed_lit_points(sqft(500), 50) == 5
    assert allowed_lit_points(sqft(999), 50) == 5
    # 1,000 sf and over: more than 2 % of the points.
    assert allowed_lit_points(sqft(1000), 500) == pytest.approx(10.0)


def test_zero_points_is_refused():
    with pytest.raises(ValueError, match="at least one point"):
        allowed_lit_points(50.0, 0)


def test_all_groups_stay_open_when_nothing_is_lit():
    direct = np.zeros((2, 10, 3))
    assert blind_schedule(direct, sqft(300), ASE_LUX).all()


def test_one_group_closes_when_the_threshold_is_exceeded():
    # 300 sf allows 3 lit points.  Group 0 lights 5 points, group 1 lights none.
    direct = np.zeros((2, 10, 1))
    direct[0, :5, 0] = 2000.0
    schedule = blind_schedule(direct, sqft(300), ASE_LUX)
    assert schedule[0, 0] == False  # noqa: E712 - explicit closed
    assert schedule[1, 0] == True   # noqa: E712 - explicit open


def test_the_most_lit_group_closes_first():
    direct = np.zeros((2, 10, 1))
    direct[0, :3, 0] = 2000.0   # lights 3
    direct[1, :9, 0] = 2000.0   # lights 9, so this one closes first
    schedule = blind_schedule(direct, sqft(300), ASE_LUX)
    assert schedule[1, 0] == False  # noqa: E712
    assert schedule[0, 0] == True   # noqa: E712 - closing group 1 was enough


def test_groups_close_until_the_count_is_low_enough():
    direct = np.zeros((3, 10, 1))
    direct[0, :5, 0] = 2000.0
    direct[1, 5:9, 0] = 2000.0
    direct[2, 9:, 0] = 2000.0
    schedule = blind_schedule(direct, sqft(199), ASE_LUX)  # allows 1 lit point
    assert schedule[:, 0].sum() == 1  # only the smallest contributor stays open


def test_every_hour_is_decided_independently():
    direct = np.zeros((1, 10, 2))
    direct[0, :5, 0] = 2000.0   # hour 0 exceeds
    schedule = blind_schedule(direct, sqft(300), ASE_LUX)
    assert schedule[0, 0] == False  # noqa: E712
    assert schedule[0, 1] == True   # noqa: E712


def test_all_groups_may_close():
    direct = np.full((2, 10, 1), 5000.0)
    assert not blind_schedule(direct, sqft(199), ASE_LUX).any()


def test_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="three dimensions"):
        blind_schedule(np.zeros((10, 3)), 50.0, ASE_LUX)


def test_blind_schedule_requires_a_lit_threshold():
    direct = np.zeros((1, 5, 1))
    with pytest.raises(TypeError):
        blind_schedule(direct, 50.0)
