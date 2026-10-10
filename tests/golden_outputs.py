# tests/golden_outputs.py
"""Golden command outputs and results: merging and splitting rooms must leave a
project that uses neither byte for byte the same (merge-split spec 8, 9).

The invented project of ``tests/deliver_folder.py`` (one 6 m x 4 m room, fake
engines, real drivers) is built, and thirteen outputs are taken from it:

- the printed output of the ``rooms``, ``scan`` and ``prepare`` commands
  (``rooms.txt``, ``scan.txt``, ``prepare.txt``) and the template ``rooms``
  writes (``spaces_template.csv``);
- the Option 1 results (``option1/results.json``, ``option1/points.csv``) and
  the Option 2 results (``option2/option2_results.json``,
  ``option2/points.csv``), as the real drivers write them with the fake
  engines of ``tests/deliver_folder.py``;
- the Quality Views results (``views/views_results.json``,
  ``views/points.csv``) and its plans (``views/plan_0_quality.svg``,
  ``views/plan_0_type2.svg``, ``views/plan_0_type3.svg``), run through the
  real ``leed_daylight.drivers.views`` driver with a fake tracer (below)
  instead of ``tests/deliver_folder.py``'s hand-written views results, which
  are not built by any driver. ``credit_rules.yaml`` has no ``views:``
  section on master (nothing needs one before this), so one is appended,
  in memory, to the copy this module writes; every value in it is invented
  and round, matching ``tests/test_views_command.py``'s own fixture.

Every occurrence of the project folder's path (as given, and resolved) is
replaced by ``<folder>``; nothing else in them depends on the machine. They
were written once, on master before the merge-split work, into
``tests/fixtures/golden_outputs/``; ``tests/test_golden_outputs.py`` builds
them again on the code under test and compares byte for byte.

Generator mode (run once, on the commit the goldens are taken from; it
refuses to overwrite an existing golden folder)::

    PYTHONPATH=.:tests .venv/bin/python tests/golden_outputs.py --write
"""
from __future__ import annotations

import io
import os
import pathlib
import re
import sys
import tempfile

import numpy as np

from conftest import write_example_project, write_wea_file
from deliver_folder import CASE, build_deliver_project
from leed_daylight.drivers.prepare import prepare_folder
from leed_daylight.drivers.rooms import rooms_folder
from leed_daylight.drivers.scan import scan_folder
from leed_daylight.drivers.views import run_views
from leed_daylight.engine.view_trace import MISS, UNUSED, RaySegments

GOLDEN_DIR = pathlib.Path(__file__).parent / "fixtures" / "golden_outputs"
FOLDER = "<folder>"

#: The invented project's one storey (the room's floor level, not the "L0"
#: text of spaces.csv): every views plan file name embeds it.
_VIEWS_STOREY = 0
_VIEWS_PLAN_LAYERS = ("quality", "type2", "type3")

#: Every golden file, by its path under GOLDEN_DIR.
OUTPUTS = (
    "rooms.txt",
    "spaces_template.csv",
    "scan.txt",
    "prepare.txt",
    "option1/results.json",
    "option1/points.csv",
    "option2/option2_results.json",
    "option2/points.csv",
    "views/views_results.json",
    "views/points.csv",
    *(f"views/plan_{_VIEWS_STOREY}_{layer}.svg" for layer in _VIEWS_PLAN_LAYERS),
)

#: The result files, by golden name: where the run drivers write them under ``<folder>/_run``.
_RESULTS = {
    "option1/results.json": f"option1/{CASE}/results.json",
    "option1/points.csv": f"option1/{CASE}/points.csv",
    "option2/option2_results.json": f"option2/{CASE}/option2_results.json",
    "option2/points.csv": f"option2/{CASE}/points.csv",
    "views/views_results.json": f"views/{CASE}/views_results.json",
    "views/points.csv": f"views/{CASE}/points.csv",
    **{
        f"views/plan_{_VIEWS_STOREY}_{layer}.svg": f"views/{CASE}/plan_{_VIEWS_STOREY}_{layer}.svg"
        for layer in _VIEWS_PLAN_LAYERS
    },
}

#: Appended to the invented project's credit_rules.yaml (which has no
#: views: section on master) so the real views driver can run; every value
#: is invented and round, the same ones tests/test_views_command.py uses.
_VIEWS_RULES_YAML = """
views:
  version: "Example views rules v0"
  quality_definition: "type2_and_type3"
  threshold_pct: 75.0
  points: 1
  vlt_min_exclusive: 0.40
  head_height_multiple: 3.0
  object_distance_m: 7.5
  obstruction_exception_m: 0.30
  method: {distance_statistic: "median", min_outside_distance_m: 2.0, panorama_vertical_deg: 60.0}
"""


class _FakeViewsTracer:
    """A fake ``leed_daylight.drivers.views.ViewTracer``: no Radiance binary
    is ever looked for or run. Matches ``tests/test_views_command.py``'s own
    fixture: a ray heading south that meets the window opening (x 2-4 m,
    z 1-2 m, the only exterior glazing of ``tests/deliver_folder.py``'s
    invented room) passes the glass and leaves the scene; any other ray
    hits the south wall.
    """

    radiance_version = None

    def octree(self, scene: pathlib.Path, materials: pathlib.Path, work_dir: pathlib.Path) -> pathlib.Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        path = work_dir / "views.oct"
        path.write_bytes(scene.read_bytes() + materials.read_bytes())
        return path

    def trace(self, octree, origins, directions, *, pass_through, exterior, skip_within_m, max_segments):
        n = origins.shape[0]
        modifier = np.full((n, max_segments), UNUSED, dtype=np.int32)
        surface = np.full((n, max_segments), -1, dtype=np.int32)
        hit = np.full((n, max_segments, 3), np.nan)
        south = directions[:, 1] < 0
        t = np.where(south, -origins[:, 1] / np.where(south, directions[:, 1], 1.0), 1.0)
        at = origins + t[:, None] * directions
        window = south & (at[:, 0] >= 2) & (at[:, 0] <= 4) & (at[:, 2] >= 1) & (at[:, 2] <= 2)
        modifier[:, 0] = np.where(window, 0, 1)
        surface[:, 0] = np.where(window, 0, 1)
        hit[:, 0] = at
        modifier[window, 1] = MISS
        return RaySegments(
            ("glass_clear", "wall_paint"), ("window_ext", "wall_room_north_south"), modifier, surface, hit,
        )


def mask_folder(text: str, folder: pathlib.Path) -> str:
    """``text`` with the project folder's path, resolved and as given,
    replaced by ``<folder>`` (resolved first: on macOS the temporary folder
    ``/var/...`` resolves to ``/private/var/...``, which contains it), and
    Windows path separators after it made ``/``."""
    masked = text.replace(str(folder.resolve()), FOLDER).replace(str(folder), FOLDER)
    if os.sep != "/":
        masked = re.sub(re.escape(FOLDER) + r"[^\s]*", lambda m: m.group(0).replace(os.sep, "/"), masked)
    return masked


def _printed(command, folder: pathlib.Path) -> str:
    out = io.StringIO()
    command(folder, out=out)
    return out.getvalue()


def render_outputs(tmp_path) -> dict[str, bytes]:
    """Build the invented project under ``tmp_path``, run Option 1 and
    Option 2 with the fake engines, Quality Views with the fake tracer, then
    ``rooms``, ``scan`` and ``prepare``: golden name -> bytes, the folder
    path masked."""
    folder = build_deliver_project(
        pathlib.Path(tmp_path), write_example_project, write_wea_file, options=("option1", "option2"),
    )
    rules_path = folder / "credit_rules.yaml"
    rules_path.write_text(rules_path.read_text() + _VIEWS_RULES_YAML)
    run_views(folder, CASE, tracer=_FakeViewsTracer(), out=io.StringIO())
    texts = {
        name: (folder / "_run" / relative).read_text(encoding="utf-8")
        for name, relative in _RESULTS.items()
    }
    texts["rooms.txt"] = _printed(rooms_folder, folder)
    texts["spaces_template.csv"] = (folder / "_run" / "spaces_template.csv").read_text(encoding="utf-8")
    texts["scan.txt"] = _printed(scan_folder, folder)
    texts["prepare.txt"] = _printed(prepare_folder, folder)
    return {name: mask_folder(texts[name], folder).encode("utf-8") for name in OUTPUTS}


def write_goldens(golden_dir=GOLDEN_DIR) -> None:
    """Generator mode: refuses, before writing anything, if ``golden_dir`` exists."""
    golden_dir = pathlib.Path(golden_dir)
    if golden_dir.exists():
        raise SystemExit(
            f"{golden_dir} already exists; goldens are written once, on the commit "
            "they are taken from, never refreshed"
        )
    with tempfile.TemporaryDirectory() as tmp:
        rendered = render_outputs(tmp)
    for name in OUTPUTS:
        path = golden_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rendered[name])
    print(f"wrote {len(OUTPUTS)} files to {golden_dir}")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: PYTHONPATH=.:tests .venv/bin/python tests/golden_outputs.py --write")
    write_goldens()
