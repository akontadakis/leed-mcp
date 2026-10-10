# tests/test_project_weather.py
"""Tests for project.weather.read_wea: invented, round irradiance only."""
from __future__ import annotations

import calendar

import numpy as np
import pytest

from leed_daylight.project.weather import HOURS_IN_YEAR, WeatherError, read_wea


def _lines(dni_of, dhi_of):
    lines = []
    index = 0
    for month in range(1, 13):
        for day in range(1, calendar.monthrange(2023, month)[1] + 1):
            for hour in range(24):
                lines.append(f"{month} {day} {hour + 0.5} {dni_of(index)} {dhi_of(index)}")
                index += 1
    return lines


def test_every_hour_is_read_in_order(tmp_path, write_wea):
    path = write_wea(
        tmp_path / "site.wea",
        hours=_lines(lambda i: float(i % 7) * 100.0, lambda i: float(i % 5) * 10.0),
    )
    weather = read_wea(path)
    assert weather.direct_normal.shape == (HOURS_IN_YEAR,)
    assert weather.diffuse_horizontal.shape == (HOURS_IN_YEAR,)
    assert weather.direct_normal[0] == 0.0
    assert weather.direct_normal[3] == 300.0
    assert weather.diffuse_horizontal[4] == 40.0
    assert weather.direct_normal[HOURS_IN_YEAR - 1] == float((HOURS_IN_YEAR - 1) % 7) * 100.0


def test_a_short_file_is_refused(tmp_path, write_wea):
    path = write_wea(tmp_path / "site.wea", hours=_lines(lambda i: 0.0, lambda i: 0.0)[:-1])
    with pytest.raises(WeatherError, match="8759"):
        read_wea(path)


def test_a_negative_irradiance_is_refused(tmp_path, write_wea):
    path = write_wea(
        tmp_path / "site.wea",
        hours=_lines(lambda i: -1.0 if i == 10 else 0.0, lambda i: 0.0),
    )
    with pytest.raises(WeatherError, match="negative"):
        read_wea(path)


def test_the_arrays_are_float64(tmp_path, write_wea):
    path = write_wea(tmp_path / "site.wea")
    weather = read_wea(path)
    assert weather.direct_normal.dtype == np.float64
    assert weather.diffuse_horizontal.dtype == np.float64


def test_the_header_time_zone_is_read(tmp_path, write_wea):
    path = write_wea(tmp_path / "site.wea", time_zone=60.0)
    assert read_wea(path).time_zone == 60.0


def test_a_file_without_a_time_zone_is_refused(tmp_path, write_wea):
    path = write_wea(tmp_path / "site.wea")
    text = path.read_text().replace("time_zone 75.0\n", "")
    path.write_text(text)
    with pytest.raises(WeatherError, match="time_zone"):
        read_wea(path)


def test_units_other_than_irradiance_are_refused_by_name(tmp_path, write_wea):
    path = write_wea(tmp_path / "site.wea")
    text = path.read_text().replace("weather_data_file_units 1\n", "weather_data_file_units 2\n")
    path.write_text(text)
    with pytest.raises(WeatherError, match="weather_data_file_units"):
        read_wea(path)


def test_a_file_without_a_units_line_is_refused(tmp_path, write_wea):
    path = write_wea(tmp_path / "site.wea")
    text = path.read_text().replace("weather_data_file_units 1\n", "")
    path.write_text(text)
    with pytest.raises(WeatherError, match="weather_data_file_units"):
        read_wea(path)
