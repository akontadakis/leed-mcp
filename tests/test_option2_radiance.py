# tests/test_option2_radiance.py
"""Option 2 against the real Radiance on an invented site. Skipped when
gensky, gendaylit, oconv or rtrace is not on PATH."""
from __future__ import annotations

import io
import json
import re
import shutil
import subprocess

import pytest

from leed_daylight.drivers.option2 import option2_folder
from leed_daylight.metrics.clearest_day import perez_epsilon
from leed_daylight.metrics.solar import day_of_year, solar_position
from option2_folder import build_option2_project

needs_radiance = pytest.mark.skipif(
    any(shutil.which(b) is None for b in ("gensky", "gendaylit", "oconv", "rtrace", "xform")),
    reason="Radiance is not on PATH",
)
_SUN_RE = re.compile(r"Solar altitude and azimuth:\s+(-?[\d.]+)\s+(-?[\d.]+)")
_EPS_RE = re.compile(r"epsilon, delta, atmospheric precipitable water content\s*:\s*([\d.]+)")


@needs_radiance
@pytest.mark.parametrize("month, day, hour, lat, lon_e, meridian", [
    (3, 21, 9, 52.0, 5.0, -15.0),
    (9, 14, 15, 40.0, -3.7, 0.0),
    (3, 28, 9, 42.0, -80.0, 75.0),
    (12, 21, 12, -33.0, 151.0, -150.0),
])
def test_the_solar_position_matches_gensky(month, day, hour, lat, lon_e, meridian):
    out = subprocess.run(
        ["gensky", str(month), str(day), str(hour), "-a", str(lat), "-o", str(-lon_e), "-m", str(meridian)],
        capture_output=True, text=True, check=True,
    ).stdout
    altitude, azimuth = (float(v) for v in _SUN_RE.search(out).groups())
    alt, azi = solar_position(day_of_year(month, day), float(hour), lat, lon_e, meridian)
    assert alt == pytest.approx(altitude, abs=0.06)
    assert azi == pytest.approx(azimuth, abs=0.06)


@needs_radiance
def test_the_perez_epsilon_matches_gendaylit():
    out = subprocess.run(
        ["gendaylit", "9", "14", "15", "-a", "40", "-o", "3.7", "-m", "0", "-W", "600", "80"],
        capture_output=True, text=True, check=True,
    ).stdout
    altitude = float(_SUN_RE.search(out).group(1))
    epsilon = float(_EPS_RE.search(out).group(1))
    assert perez_epsilon(600.0, 80.0, altitude) == pytest.approx(epsilon, abs=2e-3)


@needs_radiance
def test_an_invented_room_runs_end_to_end(tmp_path, write_example, write_wea):
    def quick(d):
        d["radiance"]["ambient"]["full"] = "-ab 1 -ad 256 -lw 1.0e-3"

    folder = build_option2_project(tmp_path, write_example, write_wea, mutate=quick)
    path = option2_folder(folder, out=io.StringIO())
    payload = json.loads(path.read_text())
    assert payload["radiance_version"].startswith("RADIANCE")
    assert [d["chosen"]["date"] for d in payload["days"]] == ["03-28", "09-14"]
    (space,) = payload["spaces"]
    assert space["points"] > 0
    rows = (folder / "_run" / "option2" / "base_case" / "points.csv").read_text().splitlines()[1:]
    lux = [float(v) for row in rows for v in row.split(",")[4:8]]
    assert all(v > 0.0 for v in lux)
    log = (folder / "_run" / "option2" / "radiance.log").read_text()
    assert "gendaylit" in log and "rtrace" in log

    # A second run with nothing changed traces nothing again.
    def traces():
        text = (folder / "_run" / "option2" / "radiance.log").read_text()
        return sum(1 for line in text.splitlines() if line.startswith("rtrace -n"))

    first = traces()
    assert first == 4
    option2_folder(folder, out=io.StringIO())
    assert traces() == first
