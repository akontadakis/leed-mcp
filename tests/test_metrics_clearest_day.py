# tests/test_metrics_clearest_day.py
"""Tests for metrics.clearest_day: invented, round irradiance on an
invented site (42 N, 80 W, meridian 75 W)."""
from __future__ import annotations

import numpy as np
import pytest

from leed_daylight.metrics.clearest_day import (
    HOURS_IN_YEAR,
    ClearestDayError,
    choose_clearest_day,
    irradiance_at,
    perez_epsilon,
)
from leed_daylight.metrics.solar import day_of_year

SITE = {"latitude_N": 42.0, "longitude_E": -80.0, "meridian_radiance": 75.0}


def _year(clear_days, *, dni=700.0, dhi=100.0, dhi_by_day=None):
    """8760 hours: `dni` from 07:00 to 17:00 on each day in `clear_days`
    (a dict (month, day) -> dni overrides), `dhi` from 06:00 to 18:00 every day."""
    direct = np.zeros(HOURS_IN_YEAR)
    diffuse = np.zeros(HOURS_IN_YEAR)
    for doy in range(1, 366):
        start = (doy - 1) * 24
        diffuse[start + 6:start + 18] = dhi
    for (month, day), value in dict(clear_days).items():
        start = (day_of_year(month, day) - 1) * 24
        direct[start + 7:start + 17] = value
    for (month, day), value in dict(dhi_by_day or {}).items():
        start = (day_of_year(month, day) - 1) * 24
        diffuse[start + 6:start + 18] = value
    return direct, diffuse


def test_perez_epsilon_matches_gendaylit():
    # `gendaylit 3 21 9 -a 52 -o -5 -m -15 -W 500 100` printed
    # "Solar altitude and azimuth: 19.3 -62.5" and the epsilon asserted below.
    # source: gendaylit 6.1a output above; the altitude is metrics.solar's unrounded value for it
    assert perez_epsilon(500.0, 100.0, 19.335458) == pytest.approx(2.6932, abs=1e-3)


def test_perez_epsilon_is_none_without_diffuse_light_or_with_a_low_sun():
    assert perez_epsilon(500.0, 0.0, 30.0) is None
    assert perez_epsilon(500.0, 100.0, 4.0) is None


def test_perez_epsilon_of_an_overcast_hour_is_one():
    assert perez_epsilon(0.0, 100.0, 30.0) == pytest.approx(1.0)


def test_the_day_with_the_most_direct_sun_wins():
    direct, diffuse = _year({(3, 28): 700.0, (3, 10): 500.0})
    choice = choose_clearest_day(direct, diffuse, equinox=(3, 21), window_days=15, **SITE)
    assert (choice.chosen.month, choice.chosen.day) == (3, 28)
    assert choice.chosen.offset_days == 7
    assert choice.chosen.direct_normal_wh == pytest.approx(7000.0)
    assert choice.equinox == (3, 21)
    assert [(c.month, c.day) for c in choice.top[:2]] == [(3, 28), (3, 10)]
    assert len(choice.top) == 5


def test_a_day_outside_the_window_is_never_chosen():
    direct, diffuse = _year({(4, 6): 900.0, (3, 28): 700.0})
    choice = choose_clearest_day(direct, diffuse, equinox=(3, 21), window_days=15, **SITE)
    assert (choice.chosen.month, choice.chosen.day) == (3, 28)


def test_a_tie_goes_to_the_day_closer_to_the_equinox():
    direct, diffuse = _year({(3, 10): 700.0, (3, 25): 700.0})
    choice = choose_clearest_day(direct, diffuse, equinox=(3, 21), window_days=15, **SITE)
    assert (choice.chosen.month, choice.chosen.day) == (3, 25)


def test_an_equal_distance_tie_goes_to_the_earlier_day():
    direct, diffuse = _year({(3, 18): 700.0, (3, 24): 700.0})
    choice = choose_clearest_day(direct, diffuse, equinox=(3, 21), window_days=15, **SITE)
    assert (choice.chosen.month, choice.chosen.day) == (3, 18)


def test_a_window_with_no_direct_sun_is_refused():
    direct, diffuse = _year({})
    with pytest.raises(ClearestDayError, match="03-21"):
        choose_clearest_day(direct, diffuse, equinox=(3, 21), window_days=15, **SITE)


def test_no_warning_when_both_measures_agree():
    direct, diffuse = _year({(3, 28): 700.0})
    choice = choose_clearest_day(direct, diffuse, equinox=(3, 21), window_days=15, **SITE)
    assert choice.warning is None
    assert (choice.epsilon_best.month, choice.epsilon_best.day) == (3, 28)


def test_a_warning_when_the_perez_index_prefers_another_day():
    # 03-28 has more direct sun; 03-20 has less, but far less diffuse
    # light, so its Perez index is higher. Direct sun still picks.
    direct, diffuse = _year(
        {(3, 28): 700.0, (3, 20): 600.0}, dhi_by_day={(3, 20): 20.0},
    )
    choice = choose_clearest_day(direct, diffuse, equinox=(3, 21), window_days=15, **SITE)
    assert (choice.chosen.month, choice.chosen.day) == (3, 28)
    assert (choice.epsilon_best.month, choice.epsilon_best.day) == (3, 20)
    assert "03-28" in choice.warning and "03-20" in choice.warning


def test_a_wrong_length_year_is_refused():
    with pytest.raises(ClearestDayError, match="8760"):
        choose_clearest_day(np.zeros(10), np.zeros(10), equinox=(3, 21), window_days=15, **SITE)


def test_irradiance_at_the_hour_is_the_mean_of_the_records_either_side():
    direct = np.zeros(HOURS_IN_YEAR)
    diffuse = np.zeros(HOURS_IN_YEAR)
    start = (day_of_year(3, 28) - 1) * 24
    direct[start + 8] = 600.0   # the record stamped 8.5
    direct[start + 9] = 800.0   # the record stamped 9.5
    diffuse[start + 8] = 100.0
    diffuse[start + 9] = 120.0
    direct[start + 14] = 400.0  # the record stamped 14.5
    direct[start + 15] = 200.0  # the record stamped 15.5
    diffuse[start + 14] = 90.0
    diffuse[start + 15] = 70.0
    assert irradiance_at(direct, diffuse, month=3, day=28, hour=9) == (700.0, 110.0)
    assert irradiance_at(direct, diffuse, month=3, day=28, hour=15) == (300.0, 80.0)


@pytest.mark.parametrize("hour", [0, 24, 9.5, True])
def test_irradiance_at_refuses_an_hour_without_two_neighbours(hour):
    with pytest.raises(ClearestDayError):
        irradiance_at(np.zeros(HOURS_IN_YEAR), np.zeros(HOURS_IN_YEAR), month=3, day=28, hour=hour)
