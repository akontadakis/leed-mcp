# tests/test_engine_point_trace.py
"""Tests for engine.point_trace.trace_points with a fake runner: rtrace
answers one invented RGB line per input line."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from leed_daylight.engine.point_trace import PointTraceError, trace_points
from leed_daylight.engine.radiance import Radiance, RunResult
from leed_daylight.units import LUX_WEIGHTS

POINTS = np.array([[1.0, 1.0, 0.8], [2.0, 1.0, 0.8], [3.0, 1.0, 0.8]])


class FakeRunner:
    def __init__(self, rgb=(1.0, 2.0, 3.0), short_by=0):
        self.calls = []
        self.rgb = rgb
        self.short_by = short_by

    def run(self, argv, *, stdin=None, stdout_path=None, cwd=None):
        argv = list(argv)
        self.calls.append(argv)
        if "-version" in argv:
            return RunResult(0, b"fake_radiance 1.0", b"")
        if argv[0] == "oconv":
            pathlib.Path(stdout_path).write_bytes(b"fake octree")
            return RunResult(0, b"", b"")
        if argv[0] == "rtrace":
            n = len(stdin.decode().splitlines()) - self.short_by
            line = " ".join(str(v) for v in self.rgb)
            return RunResult(0, ("\n".join([line] * n) + "\n").encode(), b"")
        raise AssertionError(f"unexpected command {argv}")


def _files(tmp_path, sky_text="sky one"):
    materials = tmp_path / "materials.rad"
    scene = tmp_path / "scene.rad"
    sky = tmp_path / "03-28_0900.sky"
    materials.write_text("materials", newline="\n")
    scene.write_text("scene", newline="\n")
    sky.write_text(sky_text, newline="\n")
    return materials, scene, sky


def _trace(tmp_path, runner, sky, processes=1):
    """One trace through a fresh Radiance, as a resumed run would make."""
    rad = Radiance(runner, tmp_path / "run.log")
    return trace_points(
        rad, materials=tmp_path / "materials.rad", scene=tmp_path / "scene.rad", sky=sky,
        points=POINTS, ambient="-ab 2 -ad 512", direct="-dc 1 -dt 0",
        work_dir=tmp_path / "engine", name="03-28_0900", processes=processes,
    )


def _traces(runner):
    """rtrace calls that traced points; `rtrace -version` is Radiance's own probe."""
    return [c for c in runner.calls if c[0] == "rtrace" and "-version" not in c]


def test_lux_is_the_weighted_rgb_of_every_point(tmp_path):
    _, _, sky = _files(tmp_path)
    lux = _trace(tmp_path, FakeRunner(rgb=(1.0, 2.0, 3.0)), sky)
    expected = float(np.array([1.0, 2.0, 3.0]) @ LUX_WEIGHTS)
    assert lux.shape == (3,)
    assert np.allclose(lux, expected)


def test_the_octree_holds_materials_scene_and_sky_and_rtrace_gets_the_settings(tmp_path, monkeypatch):
    monkeypatch.setattr("leed_daylight.engine.radiance.sys.platform", "linux")

    materials, scene, sky = _files(tmp_path)
    runner = FakeRunner()
    _trace(tmp_path, runner, sky, processes=2)
    (oconv,) = [c for c in runner.calls if c[0] == "oconv"]
    assert oconv[1:] == [str(materials), str(scene), str(sky)]
    (rtrace,) = _traces(runner)
    assert rtrace[:3] == ["rtrace", "-n", "2"]
    for flag in ("-h", "-I+", "-ov", "-ab", "512", "-dc"):
        assert flag in rtrace


def test_an_unchanged_trace_is_reused(tmp_path):
    _, _, sky = _files(tmp_path)
    runner = FakeRunner()
    _trace(tmp_path, runner, sky)
    _trace(tmp_path, runner, sky)
    assert len(_traces(runner)) == 1


def test_a_changed_sky_is_traced_again(tmp_path):
    _, _, sky = _files(tmp_path)
    runner = FakeRunner()
    _trace(tmp_path, runner, sky)
    sky.write_text("sky two", newline="\n")
    _trace(tmp_path, runner, sky)
    assert len(_traces(runner)) == 2


def test_a_short_rtrace_answer_is_refused(tmp_path):
    _, _, sky = _files(tmp_path)
    with pytest.raises(PointTraceError, match="2 line"):
        _trace(tmp_path, FakeRunner(short_by=1), sky)


def test_no_points_is_refused(tmp_path):
    materials, scene, sky = _files(tmp_path)
    rad = Radiance(FakeRunner(), tmp_path / "run.log")
    with pytest.raises(PointTraceError, match="no points"):
        trace_points(
            rad, materials=materials, scene=scene, sky=sky, points=np.zeros((0, 3)),
            ambient="-ab 2", direct="-dc 1", work_dir=tmp_path / "engine", name="x",
        )
