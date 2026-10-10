# tests/test_metrics_solar.py
"""Tests for metrics.solar. The pinned values are what Radiance 6.1a's own
gensky printed for invented sites (e.g. `gensky 3 21 9 -a 52 -o -5 -m -15`
prints "Solar altitude and azimuth: 19.3 -62.5"); gensky rounds to 0.1."""
from __future__ import annotations

import math

import pytest

from leed_daylight.metrics.solar import day_of_year, solar_position


def test_day_of_year():
    assert day_of_year(1, 1) == 1
    assert day_of_year(3, 21) == 80
    assert day_of_year(9, 14) == 257
    assert day_of_year(12, 31) == 365
    with pytest.raises(ValueError):
        day_of_year(2, 29)


@pytest.mark.parametrize(
    "month, day, hour, lat, lon_e, meridian, altitude, azimuth",
    [
        (3, 21, 9.0, 52.0, 5.0, -15.0, 19.3, -62.5),
        (9, 14, 15.0, 40.0, -3.7, 0.0, 36.5, 56.8),
        (3, 28, 9.0, 42.0, -80.0, 75.0, 29.5, -63.6),
        (3, 28, 15.0, 42.0, -80.0, 75.0, 37.4, 51.9),
        (9, 14, 9.0, 42.0, -80.0, 75.0, 31.2, -61.7),
        (6, 21, 12.0, 50.0, 10.0, -15.0, 63.1, -10.9),
    ],
)
def test_matches_what_radiance_gensky_printed(month, day, hour, lat, lon_e, meridian, altitude, azimuth):
    alt, azi = solar_position(day_of_year(month, day), hour, lat, lon_e, meridian)
    assert alt == pytest.approx(altitude, abs=0.06)
    assert azi == pytest.approx(azimuth, abs=0.06)


def test_solar_noon_on_the_zero_declination_day_is_ninety_minus_latitude():
    # Day 81 has zero declination under the 2*pi/365 formula. With the
    # meridian on the site's longitude, solar time is standard time plus
    # the equation of time; choose the standard hour that makes it 12.
    jd = 81
    equation_of_time = (
        0.170 * math.sin(4 * math.pi / 373 * (jd - 80))
        - 0.129 * math.sin(2 * math.pi / 355 * (jd - 8))
    )
    alt, azi = solar_position(jd, 12.0 - equation_of_time, 40.0, -75.0, 75.0)
    assert alt == pytest.approx(50.0, abs=1e-9)
    assert azi == pytest.approx(0.0, abs=1e-9)
