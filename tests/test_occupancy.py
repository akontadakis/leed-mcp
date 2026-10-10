import datetime

import numpy as np
import pytest

from leed_daylight.metrics.occupancy import dst_bounds, occupied_mask


def test_eu_dst_bounds_are_the_last_sundays():
    start, end = dst_bounds(2025, "EU")
    assert start == datetime.date(2025, 3, 30)
    assert end == datetime.date(2025, 10, 26)
    assert start.weekday() == 6 and end.weekday() == 6


def test_us_dst_bounds_are_second_march_and_first_november_sunday():
    start, end = dst_bounds(2025, "US")
    assert start == datetime.date(2025, 3, 9)
    assert end == datetime.date(2025, 11, 2)


def test_no_daylight_saving_returns_none():
    assert dst_bounds(2025, "none") is None


def test_mask_length_is_a_full_non_leap_year():
    assert occupied_mask("08:00-18:00", "none").shape == (8760,)


def test_ten_hours_a_day_gives_3650_hours():
    assert occupied_mask("08:00-18:00", "none").sum() == 3650
    assert occupied_mask("08:00-18:00", "EU").sum() == 3650


def test_without_dst_the_window_is_standard_time():
    mask = occupied_mask("08:00-18:00", "none")
    assert mask[7] is np.False_ or not mask[7]
    assert mask[8] and mask[17]
    assert not mask[18]


def test_inside_eu_dst_the_window_shifts_one_hour_earlier_in_standard_time():
    mask = occupied_mask("08:00-18:00", "EU")
    june_1 = (datetime.date(2025, 6, 1) - datetime.date(2025, 1, 1)).days * 24
    # 09:00 clock time is 08:00 standard time, so standard hour 7 is occupied.
    assert mask[june_1 + 7]
    assert not mask[june_1 + 6]
    assert mask[june_1 + 16]
    assert not mask[june_1 + 17]


def test_january_is_unshifted_under_eu_rules():
    mask = occupied_mask("08:00-18:00", "EU")
    assert mask[8] and not mask[7]


def test_a_shorter_window_gives_fewer_hours():
    assert occupied_mask("09:00-17:00", "none").sum() == 8 * 365


def test_bad_window_is_refused():
    with pytest.raises(ValueError, match="hours"):
        occupied_mask("18:00-08:00", "none")
    with pytest.raises(ValueError, match="hours"):
        occupied_mask("8 to 6", "none")


def test_unknown_daylight_saving_rule_is_refused():
    with pytest.raises(ValueError, match="daylight saving"):
        occupied_mask("08:00-18:00", "Mars")


def _day_start(date):
    return (date - datetime.date(2025, 1, 1)).days * 24


@pytest.mark.parametrize("rule", ["EU", "US"])
def test_the_day_dst_ends_is_standard_time_during_occupied_hours(rule):
    # The clocks go back at about 02:00, before the occupied window opens,
    # so every occupied hour of that day is already standard time.
    _start, end = dst_bounds(2025, rule)
    mask = occupied_mask("08:00-18:00", rule)
    base = _day_start(end)
    assert not mask[base + 7]
    assert mask[base + 8] and mask[base + 17]
    assert not mask[base + 18]


@pytest.mark.parametrize("rule", ["EU", "US"])
def test_the_day_dst_starts_is_daylight_time_during_occupied_hours(rule):
    # The clocks go forward at about 02:00, before the occupied window.
    start, _end = dst_bounds(2025, rule)
    mask = occupied_mask("08:00-18:00", rule)
    base = _day_start(start)
    assert mask[base + 7] and mask[base + 16]
    assert not mask[base + 17]
    # The day before is still standard time.
    before = base - 24
    assert not mask[before + 7] and mask[before + 17]


def test_the_default_year_is_the_stated_tmy_year():
    from leed_daylight.metrics.occupancy import DST_YEAR

    assert DST_YEAR == 2025
    assert (occupied_mask("08:00-18:00", "US") == occupied_mask("08:00-18:00", "US", DST_YEAR)).all()


@pytest.mark.parametrize("rule", ["EU", "US"])
@pytest.mark.parametrize("hours", ["00:00-24:00", "01:00-09:00", "02:00-10:00"])
def test_a_window_opening_before_three_under_daylight_saving_is_refused(rule, hours):
    # The clocks change at about 02:00 (EU: 01:00 UTC; US: 02:00 local);
    # a window opening before 03:00 would straddle the change on the two
    # changeover days, which the per-day rule does not model.
    from leed_daylight.metrics.occupancy import OccupancyError

    with pytest.raises(OccupancyError, match="03:00") as exc_info:
        occupied_mask(hours, rule)
    assert hours in str(exc_info.value) and rule in str(exc_info.value)


def test_a_window_opening_at_three_under_daylight_saving_is_accepted():
    assert occupied_mask("03:00-11:00", "US").sum() == 365 * 8


def test_an_early_window_without_daylight_saving_is_accepted():
    assert occupied_mask("00:00-24:00", "none").all()
