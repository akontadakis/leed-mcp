# tests/test_skylights_radiance.py
"""Roof glass against the real Radiance (skylights spec, section 9: the
smoke run). Run only on purpose: it needs Radiance on PATH and
LEED_SKYLIGHT_SMOKE=1, and it takes minutes.

One invented, closed 6 m x 4 m room (tests/deliver_folder.py,
CLOSED_SKYLIGHT_SCENE_RAD: four solid walls, no window) with one clear
skylight and one diffusing panel in its roof, on the invented site and
weather of tests/option2_folder.py (direct sun on two clear days). The room
must be closed: through an open side the sun reaches the points without
passing any glass, and the diffuser's direct sun is then not zero. It
records the diffuser's direct sun (expected: exactly zero, spec D8) next to
the clear skylight's (expected: above zero on some point and hour, so the
trace is not empty), and the diffuser's sky light at two -ad values (to show
the ambient noise).
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from deliver_folder import CLOSED_SKYLIGHT_SCENE_RAD, build_roof_glass_project
from leed_daylight.drivers.run import _raypath_dirs
from leed_daylight.drivers.run_option1 import _build_engine
from leed_daylight.engine.radiance import Radiance, find_binaries, subprocess_runner
from leed_daylight.metrics.occupancy import occupied_mask
from option2_folder import load_chain

needs_smoke = pytest.mark.skipif(
    os.environ.get("LEED_SKYLIGHT_SMOKE") != "1"
    or any(shutil.which(b) is None for b in ("oconv", "rfluxmtx", "rcontrib", "gendaymtx", "rtrace")),
    reason="the skylight smoke run needs Radiance on PATH and LEED_SKYLIGHT_SMOKE=1",
)


def _tracer(folder):
    project, prepared, _, _, _ = load_chain(folder)
    find_binaries()
    rad = Radiance(
        subprocess_runner(search_dirs=_raypath_dirs(project, prepared.root)),
        folder / "_run" / "radiance.log",
    )
    hours = occupied_mask(project.analysis.hours, project.analysis.daylight_saving)
    return prepared, _build_engine(rad, project, prepared), hours


def _group(prepared, kind):
    (group,) = [g for g in prepared.groups if g.kind == kind]
    return group


@needs_smoke
def test_the_diffusing_panel_adds_no_direct_sun(tmp_path, write_example, write_wea):
    folder = build_roof_glass_project(
        tmp_path, write_example, write_wea, options=(), scene_rad=CLOSED_SKYLIGHT_SCENE_RAD,
        ambient_full="-ab 2 -ad 512 -lw 1e-3",
    )
    prepared, tracer, hours = _tracer(folder)

    _, diffuser = tracer.direct(_group(prepared, "diffusing"), hours)
    _, clear = tracer.direct(_group(prepared, "skylight"), hours)

    print(f"\ndirect sun, highest lux at any point and hour: diffusing panel {diffuser.max():.1f}, "
          f"clear skylight {clear.max():.1f}")
    assert diffuser.max() == 0.0
    assert clear.max() > 0.0


@needs_smoke
def test_the_diffuser_sky_light_at_two_ambient_divisions(tmp_path, write_example, write_wea):
    means = {}
    for ad in (512, 2048):
        root = tmp_path / f"ad{ad}"
        root.mkdir()
        folder = build_roof_glass_project(
            root, write_example, write_wea, options=(), scene_rad=CLOSED_SKYLIGHT_SCENE_RAD,
            ambient_full=f"-ab 2 -ad {ad} -lw 1e-4",
        )
        prepared, tracer, hours = _tracer(folder)
        _, lux, closed = tracer.sky(_group(prepared, "diffusing"), hours, np.ones(int(hours.sum()), dtype=bool))
        assert closed is None
        means[ad] = float(lux.mean())

    change = abs(means[2048] - means[512]) / means[2048]
    print(f"\ndiffuser sky light, mean lux over points and occupied hours: -ad 512 {means[512]:.1f}, "
          f"-ad 2048 {means[2048]:.1f}, difference {100.0 * change:.1f} %")
    assert means[512] > 0.0 and means[2048] > 0.0
