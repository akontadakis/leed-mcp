# tests/test_atlas_blender_smoke.py
"""The opt-in Blender smoke run (atlas spec 12, 15). Skipped by default.

    LEED_BLENDER_SMOKE=1 .venv/bin/python -m pytest tests/test_atlas_blender_smoke.py -q

Only with ``LEED_BLENDER_SMOKE=1``, and only for the tests of this module
(marked ``blender_smoke``), does ``tests/conftest.py`` leave ``BLENDER``
alone, so the real Blender (``BLENDER``, ``blender`` on
``PATH`` or the macOS application) is found. The invented two-storey
project's bundle is built with ``blender_build.py`` through
``engine.blender.run_blender``; a second Blender run opens the saved file
and prints what it holds, which is compared with the bundle: objects per
collection, the view layers, the blended glass and the per-corner colours.
Every number is invented and round.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from atlas_folder import atlas_inputs
from conftest import BLENDER_SMOKE_ENV
from deliver_folder import build_two_storey_project
from leed_daylight.deliver.atlas import bundle as bundle_module
from leed_daylight.deliver.atlas.bundle import write_npz
from leed_daylight.engine.blender import (
    TIMEOUT_S, check_version, find_blender, run_blender, subprocess_blender_runner,
)

pytestmark = [
    pytest.mark.skipif(
        os.environ.get(BLENDER_SMOKE_ENV) != "1",
        reason=f"the opt-in Blender smoke run; set {BLENDER_SMOKE_ENV}=1",
    ),
    # The conftest guard leaves BLENDER alone for this mark only.
    pytest.mark.blender_smoke,
]

SCRIPT = pathlib.Path(bundle_module.__file__).resolve().parent / "blender_build.py"

#: Run inside the second Blender, on the saved file.
INSPECT = r"""
import json
import bpy

scene = bpy.context.scene
report = {
    "top": {c.name: len(c.all_objects) for c in scene.collection.children},
    "sub": {s.name: len(s.objects) for c in scene.collection.children for s in c.children},
    "view_layers": [v.name for v in scene.view_layers],
    "excluded": {v.name: sorted(lc.name for top in v.layer_collection.children for lc in top.children
                                if lc.exclude) for v in scene.view_layers},
    "context_hidden": [v.layer_collection.children["CONTEXT"].hide_viewport for v in scene.view_layers],
    "glass": {m.name: m.surface_render_method for m in bpy.data.materials if m.name.startswith("atlas glass")},
    "colours": sorted({(a.data_type, a.domain) for o in bpy.data.objects if o.type == "MESH"
                       for a in o.data.color_attributes}),
    "hidden": len([o for o in bpy.data.objects if o.hide_get()]),
    "texts": sorted(o.name for o in bpy.data.objects if o.type == "FONT"),
    "shading": sorted({s.shading.type for sc in bpy.data.screens for a in sc.areas for s in a.spaces
                       if s.type == "VIEW_3D"}),
}
print("ATLAS-REPORT " + json.dumps(report, sort_keys=True))
"""


def test_the_real_blender_builds_the_model(tmp_path, write_example, write_wea):
    folder = build_two_storey_project(tmp_path, write_example, write_wea)
    _, _, bundle = atlas_inputs(folder, tmp_path / "work")
    npz = tmp_path / "bundle.npz"
    write_npz(bundle, npz)
    runner = subprocess_blender_runner()
    exe = find_blender()
    version = check_version(exe, runner)
    print(f"Blender {'.'.join(map(str, version))} at {exe}")

    out, log = tmp_path / "model.blend", tmp_path / "blender.log"
    run_blender(exe, SCRIPT, npz, out, log_path=log, runner=runner)
    assert out.is_file() and out.stat().st_size > 0
    assert not (tmp_path / "model.blend.part").exists()

    inspect_log = tmp_path / "inspect.log"
    code = runner.run([str(exe), "-b", "--factory-startup", str(out), "--python-exit-code", "1",
                       "--python-expr", INSPECT], log_path=inspect_log, timeout_s=TIMEOUT_S)
    text = inspect_log.read_text(encoding="utf-8", errors="replace")
    assert code == 0, text
    line = next(line for line in text.splitlines() if line.startswith("ATLAS-REPORT "))
    report = json.loads(line.split(" ", 1)[1])
    print(json.dumps(report, sort_keys=True))

    # Two storeys. MODEL: wall, floor, ceiling and window glass on each;
    # ADDED: frame, reveal and blind on each; ROOMS: three room names and a
    # grid on each storey; RESULTS: the 12 layers of the three options on
    # each storey. No context or ground in the invented scene.
    assert report["top"] == {"MODEL": 8, "ADDED": 6, "CONTEXT": 0, "ROOMS": 5, "RESULTS": 24}
    assert report["sub"] == {
        "MODEL L0": 4, "MODEL L1": 4, "ADDED L0": 3, "ADDED L1": 3,
        "ROOMS L0": 3, "ROOMS L1": 2, "RESULTS L0": 12, "RESULTS L1": 12,
    }
    assert len(bundle.layers) == 12
    assert report["view_layers"] == ["whole building", "storey L0", "storey L1"]
    assert report["excluded"] == {
        "whole building": [],
        "storey L0": ["ADDED L1", "MODEL L1", "RESULTS L1", "ROOMS L1"],
        "storey L1": ["ADDED L0", "MODEL L0", "RESULTS L0", "ROOMS L0"],
    }
    assert report["context_hidden"] == [True, True, True]
    assert report["glass"] == {"atlas glass_window": "BLENDED"}
    assert report["colours"] == [["FLOAT_COLOR", "CORNER"]]
    assert report["hidden"] == 22  # every result layer but the first, on both storeys
    assert report["texts"] == ["room Room North", "room Room Upper", "room Store East"]
    assert report["shading"] == ["MATERIAL"]
