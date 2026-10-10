# tests/test_engine_weather_sky.py
"""Tests for engine.sky.weather_sky with a fake runner. The canned
gendaylit text is the shape Radiance 6.1a printed for an invented site."""
from __future__ import annotations

import pathlib

import pytest

from leed_daylight.engine.radiance import Radiance, RunResult
from leed_daylight.engine.sky import SKY_AND_GROUND_GLOW, SkyError, weather_sky
from leed_daylight.project.config import Site

SITE = Site(
    latitude_N=52.0, longitude_E=5.0, meridian_radiance=-15.0,
    north_rotation_deg=0.0, north_rotation_source="fictional",
)

GOOD_SKY = b"""\
# gendaylit 3 21 9 -a 52 -o -5 -m -15 -W 500 100
void light solar
0
0
3 3.4e+06 3.4e+06 3.4e+06

solar source sun
0
0
4 0.8 -0.4 0.3 0.533

void brightfunc skyfunc
2 skybright perezlum.cal
0
10 2.8e+01 7.9e+00 -0.9 -0.3 12.3 -3.2 0.04 0.8 -0.4 0.3
"""

ERROR_SKY = b"""\
# gendaylit 3 21 5 -a 52 -o -5 -m -15 -W 0 10
void brightfunc skyfunc
2 skybright perezlum.cal
0
10 0.00 0.00  0.000 0.000 0.000 0.000 0.000  0.89 0.35 -0.28
"""


class FakeRunner:
    def __init__(self, sky_text=GOOD_SKY):
        self.calls = []
        self.stdins = []
        self.sky_text = sky_text

    def run(self, argv, *, stdin=None, stdout_path=None, cwd=None):
        argv = list(argv)
        self.calls.append(argv)
        self.stdins.append(stdin)
        if "-version" in argv:
            return RunResult(0, b"fake_radiance 1.0", b"")
        out = self.sky_text if argv[0] == "gendaylit" else (stdin or b"")
        if stdout_path is not None:
            pathlib.Path(stdout_path).write_bytes(out)
            return RunResult(0, b"", b"")
        return RunResult(0, out, b"")


def _call(tmp_path, runner, **overrides):
    args = dict(month=3, day=21, hour=9.0, dni=500.0, dhi=100.0, site=SITE, north_rotation_deg=10.0)
    args.update(overrides)
    rad = Radiance(runner, tmp_path / "run.log")
    return weather_sky(rad, tmp_path / "skies" / "03-21_0900.sky", **args)


def test_gendaylit_carries_the_site_and_the_irradiance(tmp_path):
    runner = FakeRunner()
    _call(tmp_path, runner)
    (argv,) = [c for c in runner.calls if c[0] == "gendaylit"]
    assert argv[:4] == ["gendaylit", "3", "21", "9.0"]
    assert argv[argv.index("-a") + 1] == "52.0"
    assert argv[argv.index("-o") + 1] == "-5.0"
    assert argv[argv.index("-m") + 1] == "-15.0"
    assert argv[argv.index("-W") + 1:argv.index("-W") + 3] == ["500.0", "100.0"]


def test_the_sky_and_ground_glow_are_appended_and_the_sky_is_rotated(tmp_path):
    runner = FakeRunner()
    out = _call(tmp_path, runner)
    (xform_index,) = [i for i, c in enumerate(runner.calls) if c[0] == "xform"]
    assert runner.calls[xform_index] == ["xform", "-rz", "10.0"]
    stdin = runner.stdins[xform_index].decode()
    assert stdin.startswith(GOOD_SKY.decode())
    assert stdin.endswith(SKY_AND_GROUND_GLOW)
    assert "sky_glow source sky" in SKY_AND_GROUND_GLOW
    assert "ground_glow source ground" in SKY_AND_GROUND_GLOW
    assert out.read_text() == stdin


def test_an_error_sky_is_refused(tmp_path):
    runner = FakeRunner(sky_text=ERROR_SKY)
    with pytest.raises(SkyError, match="error sky"):
        _call(tmp_path, runner)


def test_the_error_sky_message_names_every_cause(tmp_path):
    runner = FakeRunner(sky_text=ERROR_SKY)
    with pytest.raises(SkyError) as caught:
        _call(tmp_path, runner)
    message = str(caught.value)
    assert "below the horizon" in message
    assert "zero" in message
    assert "solar constant" in message


@pytest.mark.parametrize("dni, dhi", [(500.0, 0.0), (-1.0, 100.0), (500.0, float("nan"))])
def test_unusable_irradiance_is_refused_before_gendaylit_runs(tmp_path, dni, dhi):
    runner = FakeRunner()
    with pytest.raises(SkyError):
        _call(tmp_path, runner, dni=dni, dhi=dhi)
    assert not [c for c in runner.calls if c[0] == "gendaylit"]
