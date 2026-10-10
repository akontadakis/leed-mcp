"""engine.view_trace: marching view rays through the model with rtrace.

A fake Runner answers rtrace for a toy world of planes x = c, each with a
modifier and a surface name, like Radiance would with -oLpNsm (the normal
of every plane is +x). It never
runs a Radiance binary. A ray that hits nothing is answered with a length
of 1e10, the value rtrace is expected to print for a miss; the smoke run
(Task 12) confirms this against real Radiance.
"""
from __future__ import annotations

import numpy as np
import pytest

from leed_daylight.engine.radiance import Radiance, RunResult
from leed_daylight.engine.view_trace import (
    MISS,
    UNUSED,
    RadianceViewTracer,
    ViewTraceError,
    build_view_octree,
    trace_views,
)

PARTITION = (2.0, "glass_part", "part_1")
PANE = (4.0, "glass_clear", "pane_1")
SCREEN = (5.0, "screen", "screen_1")
TOWER = (20.0, "context", "tower_1")
PASS = frozenset({"glass_part", "glass_clear"})
EXTERIOR = frozenset({"glass_clear"})


class FakeRunner:
    def __init__(self, planes, drop_last_line=False, round_point=False):
        self.planes = planes
        self.drop_last_line = drop_last_line
        self.round_point = round_point
        self.calls = []

    def run(self, argv, *, stdin=None, stdout_path=None, cwd=None):
        argv = list(argv)
        self.calls.append(argv)
        if argv[:2] == ["rtrace", "-version"]:
            return RunResult(0, b"RADIANCE 6.1a fake\n", b"")
        if argv[0] == "oconv":
            stdout_path.write_bytes(b"fake octree " + " ".join(argv).encode())
            return RunResult(0, b"", b"")
        lines = []
        for row in stdin.decode().splitlines():
            if not row.strip():
                continue
            ox, oy, oz, dx, dy, dz = map(float, row.split())
            best = None
            for x, modifier, surface in self.planes:
                if dx == 0:
                    continue
                t = (x - ox) / dx
                if t > 1e-9 and (best is None or t < best[0]):
                    best = (t, modifier, surface)
            if best is None:
                lines.append("1e+10\t0\t0\t0\t0\t0\t0\t*\t*")
            else:
                t, modifier, surface = best
                px, py, pz = ox + t * dx, oy + t * dy, oz + t * dz
                if self.round_point:
                    lines.append(f"{t}\t{px:.1f}\t{py:.1f}\t{pz:.1f}\t1\t0\t0\t{surface}\t{modifier}")
                else:
                    lines.append(f"{t}\t{px}\t{py}\t{pz}\t1\t0\t0\t{surface}\t{modifier}")
        if self.drop_last_line:
            lines = lines[:-1]
        return RunResult(0, ("\n".join(lines) + "\n").encode(), b"")


def _rad(tmp_path, planes, **kw):
    runner = FakeRunner(planes, **kw)
    return Radiance(runner, tmp_path / "radiance.log"), runner


def _trace(rad, tmp_path, directions, **kw):
    directions = np.asarray(directions, dtype=float)
    origins = np.tile([0.0, 0.0, 1.0], (directions.shape[0], 1))
    settings = dict(pass_through=PASS, exterior=EXTERIOR, skip_within_m=2.0, max_segments=5)
    settings.update(kw)
    return trace_views(rad, tmp_path / "views.oct", origins, directions, **settings)


def _names(segments, i):
    return [segments.modifiers[k] if k >= 0 else k for k in segments.modifier[i]]


def test_a_ray_marches_through_glass_skips_a_near_screen_and_stops_at_a_building(tmp_path):
    rad, _ = _rad(tmp_path, [PARTITION, PANE, SCREEN, TOWER])

    segments = _trace(rad, tmp_path, [[1, 0, 0]])

    assert _names(segments, 0) == ["glass_part", "glass_clear", "screen", "context", UNUSED]
    assert [segments.surfaces[k] for k in segments.surface[0][:4]] == ["part_1", "pane_1", "screen_1", "tower_1"]
    assert segments.hit[0, :4, 0] == pytest.approx([2.0, 4.0, 5.0, 20.0])
    assert np.isnan(segments.hit[0, 4]).all()


def test_without_the_cut_off_the_screen_is_where_the_ray_stops(tmp_path):
    rad, _ = _rad(tmp_path, [PARTITION, PANE, SCREEN, TOWER])

    segments = _trace(rad, tmp_path, [[1, 0, 0]], skip_within_m=0.0)

    assert _names(segments, 0) == ["glass_part", "glass_clear", "screen", UNUSED, UNUSED]


def test_a_ray_that_hits_nothing_is_a_miss(tmp_path):
    rad, _ = _rad(tmp_path, [PANE])

    segments = _trace(rad, tmp_path, [[-1, 0, 0]])

    assert segments.modifier[0].tolist() == [MISS, UNUSED, UNUSED, UNUSED, UNUSED]
    assert segments.surface[0, 0] == -1
    assert np.isnan(segments.hit[0, 0]).all()


def test_an_opaque_surface_before_the_glass_stops_the_ray(tmp_path):
    rad, _ = _rad(tmp_path, [(1.0, "wall_paint", "wall_1"), PANE])

    segments = _trace(rad, tmp_path, [[1, 0, 0]])

    assert _names(segments, 0) == ["wall_paint", UNUSED, UNUSED, UNUSED, UNUSED]


def test_max_segments_bounds_the_march(tmp_path):
    rad, _ = _rad(tmp_path, [PARTITION, PANE, SCREEN, TOWER])

    segments = _trace(rad, tmp_path, [[1, 0, 0]], max_segments=2)

    assert segments.modifier.shape == (1, 2)
    assert _names(segments, 0) == ["glass_part", "glass_clear"]


def test_chunking_gives_the_same_result(tmp_path):
    rad, runner = _rad(tmp_path, [PARTITION, PANE, SCREEN, TOWER])
    directions = [[1, 0, 0], [-1, 0, 0], [1, 0.1, 0]]

    whole = _trace(rad, tmp_path, directions)
    calls_whole = len(runner.calls)
    chunked = _trace(rad, tmp_path, directions, chunk_rays=1)

    assert len(runner.calls) - calls_whole > 3
    assert np.array_equal(whole.modifier, chunked.modifier)
    assert np.allclose(whole.hit, chunked.hit, equal_nan=True)


def test_the_rtrace_command_line(tmp_path):
    rad, runner = _rad(tmp_path, [PANE])

    _trace(rad, tmp_path, [[1, 0, 0]])

    argv = next(a for a in runner.calls if a[0] == "rtrace" and "-version" not in a)
    assert argv[-1] == str(tmp_path / "views.oct")
    assert "-oLpNsm" in argv
    assert argv[argv.index("-ab") + 1] == "0"


def test_a_short_rtrace_output_is_refused(tmp_path):
    rad, _ = _rad(tmp_path, [PANE], drop_last_line=True)

    with pytest.raises(ViewTraceError, match=r"rtrace returned 1 ray results, expected 2"):
        _trace(rad, tmp_path, [[1, 0, 0], [1, 0.1, 0]])


def test_exterior_must_be_a_subset_of_pass_through(tmp_path):
    rad, _ = _rad(tmp_path, [PANE])

    with pytest.raises(ViewTraceError, match="exterior"):
        _trace(rad, tmp_path, [[1, 0, 0]], pass_through=frozenset())


def test_the_octree_is_built_once_and_rebuilt_when_the_scene_changes(tmp_path):
    rad, runner = _rad(tmp_path, [])
    scene = tmp_path / "views_scene.rad"
    materials = tmp_path / "views_materials.rad"
    scene.write_text("# scene one\n", newline="\n")
    materials.write_text("# materials\n", newline="\n")

    first = build_view_octree(rad, scene, materials, tmp_path / "engine")
    build_view_octree(rad, scene, materials, tmp_path / "engine")
    oconv_calls = [a for a in runner.calls if a[0] == "oconv"]
    scene.write_text("# scene two\n", newline="\n")
    build_view_octree(rad, scene, materials, tmp_path / "engine")

    assert first == tmp_path / "engine" / "views.oct"
    assert oconv_calls == [["oconv", str(materials), str(scene)]]
    assert len([a for a in runner.calls if a[0] == "oconv"]) == 2


def test_the_hit_point_is_computed_not_read_from_rtrace(tmp_path):
    exact_rad, _ = _rad(tmp_path, [PARTITION, PANE, SCREEN, TOWER])
    rounded_rad, _ = _rad(tmp_path, [PARTITION, PANE, SCREEN, TOWER], round_point=True)
    direction = [[1, 0.13, 0.07]]

    exact = _trace(exact_rad, tmp_path, direction)
    rounded = _trace(rounded_rad, tmp_path, direction)

    assert _names(rounded, 0) == _names(exact, 0)
    assert rounded.hit[0, :4] == pytest.approx(exact.hit[0, :4])


def test_a_mixed_batch_of_exited_and_not_yet_exited_rays_matches_separate_traces(tmp_path):
    planes = [
        (1.0, "glass_part", "part_extra"),
        (2.0, "glass_part", "part_1"),
        (4.0, "glass_clear", "pane_1"),
        (5.0, "screen", "screen_1"),
        (20.0, "context", "tower_1"),
    ]
    origins = np.array([[1.5, 0.0, 1.0], [0.0, 0.0, 1.0]])
    directions = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    settings = dict(pass_through=PASS, exterior=EXTERIOR, skip_within_m=2.0, max_segments=5)

    combined_rad, _ = _rad(tmp_path, planes)
    combined = trace_views(combined_rad, tmp_path / "views.oct", origins, directions, **settings)
    alone0_rad, _ = _rad(tmp_path, planes)
    alone0 = trace_views(alone0_rad, tmp_path / "views.oct", origins[:1], directions[:1], **settings)
    alone1_rad, _ = _rad(tmp_path, planes)
    alone1 = trace_views(alone1_rad, tmp_path / "views.oct", origins[1:], directions[1:], **settings)

    assert _names(combined, 0) == _names(alone0, 0)
    assert _names(combined, 1) == _names(alone1, 0)
    assert combined.hit[0] == pytest.approx(alone0.hit[0], nan_ok=True)
    assert combined.hit[1] == pytest.approx(alone1.hit[0], nan_ok=True)


def test_the_radiance_view_tracer_wraps_both(tmp_path):
    rad, _ = _rad(tmp_path, [PANE])
    tracer = RadianceViewTracer(rad)
    scene = tmp_path / "s.rad"
    materials = tmp_path / "m.rad"
    scene.write_text("", newline="\n")
    materials.write_text("", newline="\n")

    octree = tracer.octree(scene, materials, tmp_path / "engine")
    segments = tracer.trace(
        octree, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]),
        pass_through=PASS, exterior=EXTERIOR, skip_within_m=2.0, max_segments=3,
    )

    assert tracer.radiance_version == "RADIANCE 6.1a fake"
    assert segments.modifiers == ("glass_clear",)


def test_normals_are_returned_per_segment(tmp_path):
    rad, _ = _rad(tmp_path, [PARTITION, PANE, TOWER])

    segments = _trace(rad, tmp_path, [[1, 0, 0], [-1, 0, 0]], skip_within_m=0.0)

    assert segments.normal.shape == segments.hit.shape
    assert segments.normal[0, :3].tolist() == [[1.0, 0.0, 0.0]] * 3
    assert np.isnan(segments.normal[1, 0]).all()


def test_the_second_face_of_the_pane_is_passed_for_a_near_grazing_ray(tmp_path):
    # Faces 0.02 m apart; along this ray (84 deg off the normal) about 0.2 m.
    planes = [PANE, (4.02, "glass_clear", "pane_1_back"), TOWER]
    rad, _ = _rad(tmp_path, planes)

    segments = _trace(rad, tmp_path, [[1, 10, 0]], skip_within_m=0.0)

    assert _names(segments, 0)[:3] == ["glass_clear", "glass_clear", "context"]


def test_a_second_pane_beyond_the_merge_gap_stops_a_grazing_ray(tmp_path):
    planes = [PANE, (4.5, "glass_clear", "pane_2"), TOWER]
    rad, _ = _rad(tmp_path, planes)

    segments = _trace(rad, tmp_path, [[1, 10, 0]], skip_within_m=0.0)

    assert _names(segments, 0)[:3] == ["glass_clear", "glass_clear", UNUSED]


def test_rtrace_runs_one_process_by_default_and_n_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr("leed_daylight.engine.radiance.sys.platform", "linux")

    rad, runner = _rad(tmp_path, [PANE])

    _trace(rad, tmp_path, [[1, 0, 0]])
    first = len(runner.calls)
    _trace(rad, tmp_path, [[1, 0, 0]], processes=6)

    one = runner.calls[first - 1]
    six = runner.calls[-1]
    assert one[one.index("-n") + 1] == "1"
    assert six[six.index("-n") + 1] == "6"
    assert six[-1] == str(tmp_path / "views.oct")


@pytest.mark.parametrize("processes", [0, -2])
def test_processes_below_one_are_refused(tmp_path, processes):
    rad, _ = _rad(tmp_path, [PANE])

    with pytest.raises(ViewTraceError, match="processes"):
        _trace(rad, tmp_path, [[1, 0, 0]], processes=processes)


def test_the_radiance_view_tracer_passes_its_processes(tmp_path, monkeypatch):
    monkeypatch.setattr("leed_daylight.engine.radiance.sys.platform", "linux")

    rad, runner = _rad(tmp_path, [PANE])
    tracer = RadianceViewTracer(rad, processes=4)

    tracer.trace(tmp_path / "views.oct", np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]),
                 pass_through=PASS, exterior=EXTERIOR, skip_within_m=2.0, max_segments=3)

    argv = next(a for a in runner.calls if a[0] == "rtrace" and "-version" not in a)
    assert argv[argv.index("-n") + 1] == "4"
