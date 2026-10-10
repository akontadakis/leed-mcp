# leed_daylight/project/weather.py
"""The weather file's hourly irradiance, for Option 2 (layer 2).

``project.checks`` has already held the ``.wea`` to its shape when the
project loaded: six header lines, then exactly 8,760 data lines from
1 January, each stamped at its half hour, in order. This module only
reads columns 4 and 5 (direct normal and diffuse horizontal irradiance,
W/m2) into two arrays in hour-of-year order, so the pure day picker in
``metrics.clearest_day`` never touches a file.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

#: One data line per hour of a non-leap year.
HOURS_IN_YEAR = 8760
#: The .wea header's weather_data_file_units value for columns 4 and 5 in
#: W/m2, direct normal and diffuse horizontal irradiance (epw2wea writes it).
IRRADIANCE_UNITS = 1


class WeatherError(ValueError):
    """The weather file's irradiance cannot be used as read."""


@dataclass(frozen=True)
class Weather:
    #: (8760,) W/m2, the .wea's 4th column, hour-of-year order.
    direct_normal: np.ndarray
    #: (8760,) W/m2, the .wea's 5th column, hour-of-year order.
    diffuse_horizontal: np.ndarray
    #: The header's time_zone: the meridian the hours are stamped in,
    #: degrees WEST positive (Radiance's convention, as meridian_radiance).
    time_zone: float


def read_wea(path) -> Weather:
    """Read the header's time zone, and the direct normal and diffuse
    horizontal irradiance of every data line; refuse a
    weather_data_file_units other than IRRADIANCE_UNITS. A header line
    never has five fields that read as numbers, so it is skipped as a data
    line."""
    path = pathlib.Path(path)
    rows: list[tuple[float, float]] = []
    time_zone: float | None = None
    units: str | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split()
        if len(parts) == 2 and parts[0] == "time_zone":
            try:
                time_zone = float(parts[1])
            except ValueError:
                raise WeatherError(
                    f"weather file {path.name}: time_zone {parts[1]!r} is not a number"
                ) from None
            continue
        if parts and parts[0] == "weather_data_file_units":
            units = " ".join(parts[1:])
            continue
        if len(parts) < 5:
            continue
        try:
            int(parts[0])
            int(parts[1])
            float(parts[2])
            direct = float(parts[3])
            diffuse = float(parts[4])
        except ValueError:
            continue
        rows.append((direct, diffuse))
    if time_zone is None:
        raise WeatherError(f"weather file {path.name}: no time_zone header line")
    if units is None:
        raise WeatherError(f"weather file {path.name}: no weather_data_file_units header line")
    try:
        irradiance_units = float(units) == IRRADIANCE_UNITS
    except ValueError:
        irradiance_units = False
    if not irradiance_units:
        raise WeatherError(
            f"weather file {path.name}: weather_data_file_units is {units!r}, expected "
            f"{IRRADIANCE_UNITS} (columns 4 and 5 in W/m2, direct normal and diffuse "
            "horizontal irradiance)"
        )
    if len(rows) != HOURS_IN_YEAR:
        raise WeatherError(
            f"weather file {path.name}: {len(rows)} data line(s), expected "
            f"{HOURS_IN_YEAR}"
        )
    values = np.array(rows, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise WeatherError(f"weather file {path.name}: an irradiance is not a finite number")
    if np.any(values < 0.0):
        first = int(np.argwhere(values < 0.0)[0][0])
        raise WeatherError(
            f"weather file {path.name}: a negative irradiance at data line {first + 1}"
        )
    return Weather(
        direct_normal=values[:, 0].copy(),
        diffuse_horizontal=values[:, 1].copy(),
        time_zone=time_zone,
    )
