# tests/test_engine_sky.py
"""Tests for the sky matrix, the sun matrix and the clear sky (layer 4).

Every test uses a FakeRunner that never touches a real process: it
records every argv it is given, answers ``rtrace -version`` (``Radiance``
runs this first, always), and -- for a ``gendaymtx -D ... -M ...`` call --
writes canned Radiance primitive text to whatever path follows ``-D`` and
a canned modifier list to whatever path follows ``-M``, standing in for
what a real ``gendaymtx -D`` run would have written there itself.

A fictional site only: latitude 50.0, longitude 10.0, meridian -15.0.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from leed_daylight.engine.radiance import Radiance, RunResult
from leed_daylight.engine.sky import (
    SkyError,
    Suns,
    clear_sky,
    read_suns,
    sky_matrix,
    sun_matrix,
)
from leed_daylight.project.config import Site

FICTIONAL_SITE = Site(
    latitude_N=50.0,
    longitude_E=10.0,
    meridian_radiance=-15.0,
    north_rotation_deg=0.0,
    north_rotation_source="fictional",
)

#: Two suns, round fictional numbers only -- never a real weather value.
_SUNS_TEXT = """\
void light solar600
0
0
3 500 500 500

solar600 source sun600
0
0
4 0.5 0.5 0.7 0.5

void light solar1200
0
0
3 300 300 300

solar1200 source sun1200
0
0
4 0.1 0.2 0.9 0.3
"""
_MODIFIERS_TEXT = "solar600\nsolar1200\n"


class FakeRunner:
    """Records every call; answers ``rtrace -version`` and a ``-D``/``-M``
    gendaymtx call by writing canned text to the paths it finds in argv."""

    def __init__(self, *, version="fake_radiance 1.2.3", exit_code=0,
                 stdout=b"fake matrix bytes", stderr=b"",
                 suns_text="", modifiers_text=""):
        self.calls: list[list[str]] = []
        self.version = version
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.suns_text = suns_text
        self.modifiers_text = modifiers_text

    def run(self, argv, *, stdin=None, stdout_path=None, cwd=None):
        argv = list(argv)
        self.calls.append(argv)
        if "-version" in argv:
            return RunResult(0, self.version.encode("utf-8"), b"")
        if "-D" in argv:
            suns_path = pathlib.Path(argv[argv.index("-D") + 1])
            suns_path.write_text(self.suns_text, encoding="utf-8", newline="\n")
        if "-M" in argv:
            mod_path = pathlib.Path(argv[argv.index("-M") + 1])
            mod_path.write_text(self.modifiers_text, encoding="utf-8", newline="\n")
        if stdout_path is not None:
            pathlib.Path(stdout_path).write_bytes(self.stdout)
            return RunResult(self.exit_code, b"", self.stderr)
        return RunResult(self.exit_code, self.stdout, self.stderr)


def _gendaymtx_calls(runner):
    return [c for c in runner.calls if c and c[0] == "gendaymtx"]


#: A fictional, round DAYSIM-style .wea header -- no data lines, so this
#: alone yields NO time steps (final-fix I3's own weather-file reader
#: skips every one of these six lines: none has three numeric leading
#: tokens).
_WEA_HEADER = (
    "place fictional_site\n"
    "latitude 50.0\n"
    "longitude -10.0\n"
    "time_zone -15\n"
    "site_elevation 0.0\n"
    "weather_data_file_units 1\n"
)


def _wea_data_lines(hours) -> str:
    """``month day hour dni dhi`` lines, one per ``hours`` value, Jan 1st,
    round fictional irradiance -- the shape gendaymtx(1) documents for a
    DAYSIM .wea file."""
    return "\n".join(f"1 1 {h} 0 0" for h in hours) + "\n"


def _make_wea(tmp_path, name="fictional.wea") -> pathlib.Path:
    """A fictional .wea with 24 WHOLE-HOUR Jan-1st data lines (hours 0..23):
    since day 1's minute-of-year is exactly ``hour * 60``, line index i
    (0-based) is minute-of-year ``i * 60``, i.e. the SAME value the old
    ``minute // 60`` formula gave -- so every test below that names an
    hour by a sun's own minute (e.g. minute 600 -> hour 10) still gets
    that same index under final-fix I3's own weather-file matching.
    """
    wea = tmp_path / name
    wea.write_text(_WEA_HEADER + _wea_data_lines(range(24)), encoding="utf-8", newline="\n")
    return wea


# --- the eight required rows --------------------------------------------


def test_the_sky_matrix_command_is_gendaymtx_with_the_basis(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sky.mtx"
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")

    sky_matrix(rad, wea, out, mf=4, north_rotation_deg=0.0)

    calls = _gendaymtx_calls(runner)
    assert len(calls) == 1
    argv = calls[0]
    assert argv[0] == "gendaymtx"
    assert "-m" in argv and argv[argv.index("-m") + 1] == "4"


def test_the_north_rotation_reaches_gendaymtx(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sky.mtx"
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")

    sky_matrix(rad, wea, out, mf=1, north_rotation_deg=30.0)

    argv = _gendaymtx_calls(runner)[0]
    assert "-r" in argv and argv[argv.index("-r") + 1] == "30.0"


def test_a_zero_rotation_is_still_explicit(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sky.mtx"
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")

    sky_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)

    argv = _gendaymtx_calls(runner)[0]
    assert "-r" in argv and argv[argv.index("-r") + 1] == "0.0"


def test_the_sun_matrix_uses_the_direct_flag(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")

    sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)

    argv = _gendaymtx_calls(runner)[0]
    assert "-D" in argv


def test_the_sky_and_the_sun_use_the_same_rotation(tmp_path):
    wea = _make_wea(tmp_path)
    sky_out = tmp_path / "sky.mtx"
    sun_out = tmp_path / "sun.mtx"
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")

    sky_matrix(rad, wea, sky_out, mf=2, north_rotation_deg=45.0)
    sun_matrix(rad, wea, sun_out, mf=2, north_rotation_deg=45.0)

    calls = _gendaymtx_calls(runner)
    assert len(calls) == 2
    for argv in calls:
        assert "-r" in argv and argv[argv.index("-r") + 1] == "45.0"


def test_the_geometry_is_not_also_rotated(tmp_path):
    wea = _make_wea(tmp_path)
    sky_out = tmp_path / "sky.mtx"
    sun_out = tmp_path / "sun.mtx"
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")

    sky_matrix(rad, wea, sky_out, mf=1, north_rotation_deg=10.0)
    sun_matrix(rad, wea, sun_out, mf=1, north_rotation_deg=10.0)

    for argv in runner.calls:
        joined = " ".join(argv)
        assert "xform" not in joined
        assert "-rz" not in joined


def test_a_second_call_with_the_same_inputs_is_skipped(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sky.mtx"
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")

    sky_matrix(rad, wea, out, mf=3, north_rotation_deg=0.0)
    sky_matrix(rad, wea, out, mf=3, north_rotation_deg=0.0)

    assert len(_gendaymtx_calls(runner)) == 1


def test_gensky_carries_the_site_and_the_clear_sky_flag(tmp_path):
    out = tmp_path / "clear.sky"
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")

    clear_sky(rad, out, month=6, day=21, hour=12.0,
               site=FICTIONAL_SITE, north_rotation_deg=0.0)

    gensky_calls = [c for c in runner.calls if c and c[0] == "gensky"]
    assert len(gensky_calls) == 1
    argv = gensky_calls[0]
    assert "+s" in argv
    assert "-a" in argv and argv[argv.index("-a") + 1] == "50.0"
    assert "-o" in argv and argv[argv.index("-o") + 1] == "-10.0"
    assert "-m" in argv and argv[argv.index("-m") + 1] == "-15.0"


# --- sun_matrix: parsing, three files, unit radiance ----------------------


def test_sun_matrix_writes_three_files_and_returns_the_first(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    runner = FakeRunner(suns_text=_SUNS_TEXT, modifiers_text=_MODIFIERS_TEXT)
    rad = Radiance(runner, tmp_path / "run.log")

    result = sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)

    assert result == out
    assert out.exists()
    assert (tmp_path / "sun.mtx.mod").exists()
    assert (tmp_path / "sun.mtx.suns.json").exists()


def test_sun_matrix_resets_light_rgb_to_unit_radiance(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    runner = FakeRunner(suns_text=_SUNS_TEXT, modifiers_text=_MODIFIERS_TEXT)
    rad = Radiance(runner, tmp_path / "run.log")

    sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)

    written = out.read_text(encoding="utf-8")
    assert "500 500 500" not in written
    assert "300 300 300" not in written
    assert "1 1 1" in written


def test_sun_matrix_mod_file_preserves_gendaymtx_order(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    runner = FakeRunner(suns_text=_SUNS_TEXT, modifiers_text=_MODIFIERS_TEXT)
    rad = Radiance(runner, tmp_path / "run.log")

    sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)

    lines = (tmp_path / "sun.mtx.mod").read_text(encoding="utf-8").splitlines()
    assert lines == ["solar600", "solar1200"]


def test_sun_matrix_json_carries_true_radiance_and_hour(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    runner = FakeRunner(suns_text=_SUNS_TEXT, modifiers_text=_MODIFIERS_TEXT)
    rad = Radiance(runner, tmp_path / "run.log")

    sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)

    records = json.loads((tmp_path / "sun.mtx.suns.json").read_text())
    assert records == [
        {"modifier": "solar600", "hour": 10, "rgb": [500.0, 500.0, 500.0]},
        {"modifier": "solar1200", "hour": 20, "rgb": [300.0, 300.0, 300.0]},
    ]


def test_read_suns_round_trips_the_json_sidecar(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    runner = FakeRunner(suns_text=_SUNS_TEXT, modifiers_text=_MODIFIERS_TEXT)
    rad = Radiance(runner, tmp_path / "run.log")
    sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)

    suns = read_suns(out)

    assert isinstance(suns, Suns)
    assert suns.modifiers == ("solar600", "solar1200")
    assert isinstance(suns.hours, np.ndarray)
    assert suns.hours.tolist() == [10, 20]
    assert suns.rgb.shape == (2, 3)
    assert suns.rgb.tolist() == [[500.0, 500.0, 500.0], [300.0, 300.0, 300.0]]


def test_a_zero_radiance_sun_is_kept(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    # minute 0 -> Jan 1st, hour 0.0 -- _make_wea's own first data line.
    suns_text = (
        "void light solar1\n0\n0\n3 0 0 0\n\n"
        "solar1 source sun0\n0\n0\n4 0.1 0.2 0.9 0.5\n"
    )
    runner = FakeRunner(suns_text=suns_text, modifiers_text="solar1\n")
    rad = Radiance(runner, tmp_path / "run.log")

    sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)

    records = json.loads((tmp_path / "sun.mtx.suns.json").read_text())
    assert records == [{"modifier": "solar1", "hour": 0, "rgb": [0.0, 0.0, 0.0]}]


def test_sun_matrix_with_no_sun_up_hours_writes_empty_outputs(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    runner = FakeRunner(suns_text="", modifiers_text="")
    rad = Radiance(runner, tmp_path / "run.log")

    sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)

    assert json.loads((tmp_path / "sun.mtx.suns.json").read_text()) == []
    assert (tmp_path / "sun.mtx.mod").read_text(encoding="utf-8") == ""


def test_a_second_sun_matrix_call_with_the_same_inputs_is_skipped(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    runner = FakeRunner(suns_text=_SUNS_TEXT, modifiers_text=_MODIFIERS_TEXT)
    rad = Radiance(runner, tmp_path / "run.log")

    sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)
    sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)

    assert len(_gendaymtx_calls(runner)) == 1


def test_a_changed_mf_forces_a_rebuild(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sky.mtx"
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")

    sky_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)
    sky_matrix(rad, wea, out, mf=2, north_rotation_deg=0.0)

    assert len(_gendaymtx_calls(runner)) == 2


def test_a_radiance_upgrade_rebuilds_the_sky_and_sun_matrices(tmp_path):
    wea = _make_wea(tmp_path)
    sky_out = tmp_path / "sky.mtx"
    sun_out = tmp_path / "sun.mtx"
    runner = FakeRunner(suns_text=_SUNS_TEXT, modifiers_text=_MODIFIERS_TEXT)
    rad = Radiance(runner, tmp_path / "run.log")
    sky_matrix(rad, wea, sky_out, mf=1, north_rotation_deg=0.0)
    sun_matrix(rad, wea, sun_out, mf=1, north_rotation_deg=0.0)
    calls = len(_gendaymtx_calls(runner))

    runner2 = FakeRunner(suns_text=_SUNS_TEXT, modifiers_text=_MODIFIERS_TEXT)
    runner2.version = "other_radiance 9.9"
    rad2 = Radiance(runner2, tmp_path / "run2.log")
    sky_matrix(rad2, wea, sky_out, mf=1, north_rotation_deg=0.0)
    sun_matrix(rad2, wea, sun_out, mf=1, north_rotation_deg=0.0)

    assert len(_gendaymtx_calls(runner2)) == calls


# --- sun_matrix parsing refusals ------------------------------------------


def test_a_listed_modifier_with_no_light_is_a_named_refusal(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    runner = FakeRunner(suns_text=_SUNS_TEXT, modifiers_text="solar_missing\n")
    rad = Radiance(runner, tmp_path / "run.log")

    with pytest.raises(SkyError, match="solar_missing"):
        sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)


def test_a_light_with_no_source_is_a_named_refusal(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    suns_text = "void light solar1\n0\n0\n3 100 100 100\n"
    runner = FakeRunner(suns_text=suns_text, modifiers_text="solar1\n")
    rad = Radiance(runner, tmp_path / "run.log")

    with pytest.raises(SkyError, match="source"):
        sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)


def test_a_non_integer_minute_is_a_named_refusal(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    suns_text = (
        "void light solarBAD\n0\n0\n3 100 100 100\n\n"
        "solarBAD source sunBAD\n0\n0\n4 0.1 0.2 0.9 0.5\n"
    )
    runner = FakeRunner(suns_text=suns_text, modifiers_text="solarBAD\n")
    rad = Radiance(runner, tmp_path / "run.log")

    with pytest.raises(SkyError, match="minute"):
        sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)


def test_mismatched_argument_counts_are_a_named_refusal(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    # Declares 3 real args for the light but the file ends after 2: the
    # declared count and the tokens actually present disagree.
    suns_text = "void light solar1\n0\n0\n3 100 100\n"
    runner = FakeRunner(suns_text=suns_text, modifiers_text="solar1\n")
    rad = Radiance(runner, tmp_path / "run.log")

    with pytest.raises(SkyError, match="disagree"):
        sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)


# --- hostile inputs (contract step 9) -------------------------------------


def test_a_non_int_mf_is_a_named_refusal(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sky.mtx"
    rad = Radiance(FakeRunner(), tmp_path / "run.log")

    with pytest.raises(SkyError, match="mf"):
        sky_matrix(rad, wea, out, mf=4.0, north_rotation_deg=0.0)


def test_a_bool_mf_is_a_named_refusal(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sky.mtx"
    rad = Radiance(FakeRunner(), tmp_path / "run.log")

    with pytest.raises(SkyError, match="mf"):
        sky_matrix(rad, wea, out, mf=True, north_rotation_deg=0.0)


def test_a_zero_mf_is_a_named_refusal(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sky.mtx"
    rad = Radiance(FakeRunner(), tmp_path / "run.log")

    with pytest.raises(SkyError, match="mf"):
        sky_matrix(rad, wea, out, mf=0, north_rotation_deg=0.0)


def test_a_nan_rotation_is_a_named_refusal(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sky.mtx"
    rad = Radiance(FakeRunner(), tmp_path / "run.log")

    with pytest.raises(SkyError, match="north_rotation_deg"):
        sky_matrix(rad, wea, out, mf=1, north_rotation_deg=math.nan)


def test_an_infinite_rotation_is_a_named_refusal(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    rad = Radiance(FakeRunner(), tmp_path / "run.log")

    with pytest.raises(SkyError, match="north_rotation_deg"):
        sun_matrix(rad, wea, out, mf=1, north_rotation_deg=math.inf)


def test_a_missing_wea_file_is_a_named_error(tmp_path):
    missing = tmp_path / "does_not_exist.wea"
    out = tmp_path / "sky.mtx"
    rad = Radiance(FakeRunner(), tmp_path / "run.log")

    with pytest.raises(FileNotFoundError, match="wea"):
        sky_matrix(rad, missing, out, mf=1, north_rotation_deg=0.0)


def test_clear_sky_refuses_a_month_out_of_range(tmp_path):
    out = tmp_path / "clear.sky"
    rad = Radiance(FakeRunner(), tmp_path / "run.log")

    with pytest.raises(SkyError, match="month"):
        clear_sky(rad, out, month=13, day=1, hour=12.0,
                   site=FICTIONAL_SITE, north_rotation_deg=0.0)


def test_clear_sky_refuses_a_day_that_does_not_exist(tmp_path):
    out = tmp_path / "clear.sky"
    rad = Radiance(FakeRunner(), tmp_path / "run.log")

    # A non-leap year: February has 28 days, never 29.
    with pytest.raises(SkyError, match="day"):
        clear_sky(rad, out, month=2, day=29, hour=12.0,
                   site=FICTIONAL_SITE, north_rotation_deg=0.0)


def test_clear_sky_refuses_an_hour_above_24(tmp_path):
    out = tmp_path / "clear.sky"
    rad = Radiance(FakeRunner(), tmp_path / "run.log")

    with pytest.raises(SkyError, match="hour"):
        clear_sky(rad, out, month=6, day=21, hour=24.1,
                   site=FICTIONAL_SITE, north_rotation_deg=0.0)


def test_clear_sky_refuses_a_non_finite_hour(tmp_path):
    out = tmp_path / "clear.sky"
    rad = Radiance(FakeRunner(), tmp_path / "run.log")

    with pytest.raises(SkyError, match="hour"):
        clear_sky(rad, out, month=6, day=21, hour=math.nan,
                   site=FICTIONAL_SITE, north_rotation_deg=0.0)


def test_clear_sky_boundary_hours_are_accepted(tmp_path):
    out = tmp_path / "clear.sky"
    rad = Radiance(FakeRunner(), tmp_path / "run.log")

    clear_sky(rad, out, month=1, day=1, hour=0.0,
               site=FICTIONAL_SITE, north_rotation_deg=0.0)
    clear_sky(rad, out, month=12, day=31, hour=24.0,
               site=FICTIONAL_SITE, north_rotation_deg=0.0)

    assert out.exists()


def test_read_suns_missing_sidecar_is_a_named_refusal(tmp_path):
    missing_out = tmp_path / "never_built.mtx"

    with pytest.raises(SkyError, match="suns.json"):
        read_suns(missing_out)


def test_read_suns_malformed_json_is_a_named_refusal(tmp_path):
    out = tmp_path / "sun.mtx"
    (tmp_path / "sun.mtx.suns.json").write_text("{not valid json", encoding="utf-8", newline="\n")

    with pytest.raises(SkyError, match="suns.json"):
        read_suns(out)


def test_clear_sky_rotates_via_xform_rz_on_the_sky_only(tmp_path):
    out = tmp_path / "clear.sky"
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")

    clear_sky(rad, out, month=6, day=21, hour=12.0,
               site=FICTIONAL_SITE, north_rotation_deg=15.0)

    xform_calls = [c for c in runner.calls if c and c[0] == "xform"]
    assert len(xform_calls) == 1
    argv = xform_calls[0]
    assert "-rz" in argv and argv[argv.index("-rz") + 1] == "15.0"


# --- fix round 1 -----------------------------------------------------------
# review item 2: a changed rotation, and a changed .wea, each force a
# rebuild -- for both sky_matrix and sun_matrix.


def test_a_changed_rotation_forces_a_sky_matrix_rebuild(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sky.mtx"
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")

    sky_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)
    sky_matrix(rad, wea, out, mf=1, north_rotation_deg=30.0)

    assert len(_gendaymtx_calls(runner)) == 2


def test_a_changed_wea_forces_a_sky_matrix_rebuild(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sky.mtx"
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")

    sky_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)
    wea.write_text("place fictional_site\nlatitude 50.0\nchanged\n", encoding="utf-8", newline="\n")
    sky_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)

    assert len(_gendaymtx_calls(runner)) == 2


def test_a_changed_rotation_forces_a_sun_matrix_rebuild(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    runner = FakeRunner(suns_text=_SUNS_TEXT, modifiers_text=_MODIFIERS_TEXT)
    rad = Radiance(runner, tmp_path / "run.log")

    sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)
    sun_matrix(rad, wea, out, mf=1, north_rotation_deg=30.0)

    assert len(_gendaymtx_calls(runner)) == 2


def test_a_changed_wea_forces_a_sun_matrix_rebuild(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    runner = FakeRunner(suns_text=_SUNS_TEXT, modifiers_text=_MODIFIERS_TEXT)
    rad = Radiance(runner, tmp_path / "run.log")

    sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)
    # Changed bytes (an extra comment line, still with every hour-0..23
    # data line intact so sun600/sun1200 still match), forcing a rebuild.
    wea.write_text(wea.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8", newline="\n")
    sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)

    assert len(_gendaymtx_calls(runner)) == 2


# review item 3: light/source primitives must declare zero string and zero
# integer args; otherwise a declared-count overrun could silently shift
# the parse into the next primitive's tokens.


def test_a_light_with_string_args_is_a_named_refusal(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    # The light declares 1 string arg it should never have.
    suns_text = (
        "void light solar1\n1 bogus\n0\n3 100 100 100\n\n"
        "solar1 source sun1\n0\n0\n4 0.1 0.2 0.9 0.5\n"
    )
    runner = FakeRunner(suns_text=suns_text, modifiers_text="solar1\n")
    rad = Radiance(runner, tmp_path / "run.log")

    with pytest.raises(SkyError, match="disagree"):
        sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)


def test_a_source_with_int_args_is_a_named_refusal(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    # The source declares 1 integer arg it should never have.
    suns_text = (
        "void light solar1\n0\n0\n3 100 100 100\n\n"
        "solar1 source sun1\n0\n1 7\n4 0.1 0.2 0.9 0.5\n"
    )
    runner = FakeRunner(suns_text=suns_text, modifiers_text="solar1\n")
    rad = Radiance(runner, tmp_path / "run.log")

    with pytest.raises(SkyError, match="disagree"):
        sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)


# review item 1: read_suns must never leak a raw KeyError/TypeError/numpy
# ValueError for a wrong-shaped .suns.json -- every mismatch is a named
# SkyError naming the file and what was wrong.


def _write_suns_json(tmp_path, out_name, payload) -> pathlib.Path:
    out = tmp_path / out_name
    (tmp_path / f"{out_name}.suns.json").write_text(
        json.dumps(payload), encoding="utf-8",
    newline="\n")
    return out


def test_read_suns_refuses_a_json_document_that_is_not_a_list(tmp_path):
    out = _write_suns_json(tmp_path, "sun.mtx", {"not": "a list"})

    with pytest.raises(SkyError, match="list"):
        read_suns(out)


def test_read_suns_refuses_a_record_that_is_not_an_object(tmp_path):
    out = _write_suns_json(tmp_path, "sun.mtx", ["not an object"])

    with pytest.raises(SkyError, match="object"):
        read_suns(out)


def test_read_suns_refuses_a_record_missing_modifier(tmp_path):
    out = _write_suns_json(
        tmp_path, "sun.mtx", [{"hour": 10, "rgb": [1.0, 1.0, 1.0]}],
    )

    with pytest.raises(SkyError, match="modifier"):
        read_suns(out)


def test_read_suns_refuses_a_record_missing_hour(tmp_path):
    out = _write_suns_json(
        tmp_path, "sun.mtx", [{"modifier": "solar1", "rgb": [1.0, 1.0, 1.0]}],
    )

    with pytest.raises(SkyError, match="hour"):
        read_suns(out)


def test_read_suns_refuses_a_record_missing_rgb(tmp_path):
    out = _write_suns_json(
        tmp_path, "sun.mtx", [{"modifier": "solar1", "hour": 10}],
    )

    with pytest.raises(SkyError, match="rgb"):
        read_suns(out)


def test_read_suns_refuses_an_rgb_with_the_wrong_length(tmp_path):
    out = _write_suns_json(
        tmp_path, "sun.mtx",
        [{"modifier": "solar1", "hour": 10, "rgb": [1.0, 1.0]}],
    )

    with pytest.raises(SkyError, match="rgb"):
        read_suns(out)


def test_read_suns_refuses_a_non_numeric_rgb_component(tmp_path):
    out = _write_suns_json(
        tmp_path, "sun.mtx",
        [{"modifier": "solar1", "hour": 10, "rgb": [1.0, "bad", 1.0]}],
    )

    with pytest.raises(SkyError, match="rgb"):
        read_suns(out)


def test_read_suns_refuses_a_non_finite_rgb_component(tmp_path):
    out = _write_suns_json(
        tmp_path, "sun.mtx",
        [{"modifier": "solar1", "hour": 10, "rgb": [1.0, math.nan, 1.0]}],
    )

    with pytest.raises(SkyError, match="rgb"):
        read_suns(out)


def test_read_suns_refuses_a_negative_hour(tmp_path):
    out = _write_suns_json(
        tmp_path, "sun.mtx",
        [{"modifier": "solar1", "hour": -1, "rgb": [1.0, 1.0, 1.0]}],
    )

    with pytest.raises(SkyError, match="hour"):
        read_suns(out)


def test_read_suns_refuses_a_non_int_hour(tmp_path):
    out = _write_suns_json(
        tmp_path, "sun.mtx",
        [{"modifier": "solar1", "hour": 10.5, "rgb": [1.0, 1.0, 1.0]}],
    )

    with pytest.raises(SkyError, match="hour"):
        read_suns(out)


# =============================================================================
# final-fix I3: a sun's minute is matched to the WEATHER FILE'S OWN
# time-step line, not assumed to be a whole-hour-of-day stamp
# =============================================================================


def test_sun_matrix_maps_minute_to_the_whole_hour_weather_line_index(tmp_path):
    # _make_wea's own convention: 24 whole-hour Jan-1st lines (0.0..23.0),
    # so line index i IS minute-of-year i * 60 -- sun600 -> index 10,
    # sun1200 -> index 20 (the same values the old minute // 60 gave).
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    runner = FakeRunner(suns_text=_SUNS_TEXT, modifiers_text=_MODIFIERS_TEXT)
    rad = Radiance(runner, tmp_path / "run.log")

    sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)

    records = json.loads((tmp_path / "sun.mtx.suns.json").read_text())
    assert [r["hour"] for r in records] == [10, 20]


def test_sun_matrix_maps_minute_to_the_half_hour_weather_line_index(tmp_path):
    # 24 HALF-hour Jan-1st lines (0.5, 1.5, ..., 23.5): line index i is
    # hour (i + 0.5), so minute-of-year i * 60 + 30. sun630 (minute 630,
    # hour 10.5) -> index 10; sun1230 (minute 1230, hour 20.5) -> index 20
    # -- the exact indices a whole-hour file gave those same minutes-past-
    # the-half-hour, proving the match is against the FILE, not a fixed
    # hour-of-day arithmetic.
    wea = tmp_path / "half_hour.wea"
    wea.write_text(
        _WEA_HEADER + _wea_data_lines(h + 0.5 for h in range(24)), encoding="utf-8",
    newline="\n")
    out = tmp_path / "sun.mtx"
    suns_text = _SUNS_TEXT.replace("sun600", "sun630").replace("sun1200", "sun1230")
    runner = FakeRunner(suns_text=suns_text, modifiers_text=_MODIFIERS_TEXT)
    rad = Radiance(runner, tmp_path / "run.log")

    sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)

    records = json.loads((tmp_path / "sun.mtx.suns.json").read_text())
    assert [r["hour"] for r in records] == [10, 20]


def test_a_sun_with_no_matching_weather_line_is_a_named_refusal(tmp_path):
    wea = _make_wea(tmp_path)  # whole-hour lines only; minute 601 never occurs
    out = tmp_path / "sun.mtx"
    suns_text = _SUNS_TEXT.replace("source sun600", "source sun601")
    runner = FakeRunner(suns_text=suns_text, modifiers_text=_MODIFIERS_TEXT)
    rad = Radiance(runner, tmp_path / "run.log")

    with pytest.raises(SkyError, match="601"):
        sun_matrix(rad, wea, out, mf=1, north_rotation_deg=0.0)


# =============================================================================
# final-fix M1: the old fingerprint sidecar is deleted BEFORE a rebuild
# =============================================================================


class _SidecarCheckingRunner(FakeRunner):
    """Like FakeRunner, but asserts the output's own sidecar is already
    gone the instant gendaymtx is invoked -- proving deletion happens
    BEFORE the rebuild, not after (M1)."""

    def __init__(self, sidecar_path, **kwargs):
        super().__init__(**kwargs)
        self._sidecar_path = sidecar_path

    def run(self, argv, *, stdin=None, stdout_path=None, cwd=None):
        if argv and argv[0] == "gendaymtx":
            assert not self._sidecar_path.exists(), (
                "the old sidecar must be deleted before the rebuild starts"
            )
        return super().run(argv, stdin=stdin, stdout_path=stdout_path, cwd=cwd)


def test_the_old_sidecar_is_deleted_before_sky_matrix_rebuilds(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sky.mtx"
    sky_matrix(Radiance(FakeRunner(), tmp_path / "run.log"), wea, out, mf=1, north_rotation_deg=0.0)
    sidecar = tmp_path / "sky.mtx.fingerprint.json"
    assert sidecar.exists()

    checking = _SidecarCheckingRunner(sidecar)
    rad2 = Radiance(checking, tmp_path / "run2.log")
    sky_matrix(rad2, wea, out, mf=2, north_rotation_deg=0.0)  # mf changed -> rebuild

    assert sidecar.exists()  # the rebuild wrote a fresh one afterward


def test_the_old_sidecar_is_deleted_before_sun_matrix_rebuilds(tmp_path):
    wea = _make_wea(tmp_path)
    out = tmp_path / "sun.mtx"
    runner = FakeRunner(suns_text=_SUNS_TEXT, modifiers_text=_MODIFIERS_TEXT)
    sun_matrix(Radiance(runner, tmp_path / "run.log"), wea, out, mf=1, north_rotation_deg=0.0)
    sidecar = tmp_path / "sun.mtx.fingerprint.json"
    assert sidecar.exists()

    checking = _SidecarCheckingRunner(
        sidecar, suns_text=_SUNS_TEXT, modifiers_text=_MODIFIERS_TEXT,
    )
    rad2 = Radiance(checking, tmp_path / "run2.log")
    sun_matrix(rad2, wea, out, mf=2, north_rotation_deg=0.0)  # mf changed -> rebuild

    assert sidecar.exists()  # the rebuild wrote a fresh one afterward


# --- an .epw given where a .wea is needed is refused before gendaymtx runs


@pytest.mark.parametrize("make", [sky_matrix, sun_matrix])
@pytest.mark.parametrize("name", ["fictional.epw", "fictional.csv", "fictional"])
def test_a_weather_file_that_is_not_wea_is_refused_before_gendaymtx(tmp_path, make, name):
    weather = tmp_path / name
    weather.write_text("LOCATION,Fictional,,,,0,50.0,10.0,1.0,0.0\n", newline="\n")
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")

    with pytest.raises(SkyError, match=r"\.wea") as exc_info:
        make(rad, weather, tmp_path / "out.mtx", 1, 0.0)

    assert str(weather) in str(exc_info.value)
    assert _gendaymtx_calls(runner) == []


def test_an_upper_case_wea_suffix_is_accepted(tmp_path):
    wea = _make_wea(tmp_path, name="FICTIONAL.WEA")
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")

    sky_matrix(rad, wea, tmp_path / "sky.mtx", 1, 0.0)

    assert len(_gendaymtx_calls(runner)) == 1
