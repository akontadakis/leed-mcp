# tests/test_merge_split_radiance.py
"""Merging and splitting rooms against the real Radiance (merge-split spec
10: the opt-in run). Run only on purpose: it needs Radiance on PATH and
LEED_RADIANCE_SMOKE=1, and it takes minutes::

    LEED_RADIANCE_SMOKE=1 .venv/bin/python -m pytest tests/test_merge_split_radiance.py -q

The invented building is the combined project of
``tests/merge_split_folder.py``: Room North cut along x = 4.5 m by
splits.csv into Room West (a split space, with the south window) and Room
Strip, and Room Strip a piece of Room East (a split+merged space, with
Room East's window and the pane in the gap). ``run``, ``option2``,
``views`` and ``report`` (the Word reports; no Blender file) run end to
end on it; the test checks one result row per space row and prints what
the commands printed. Ambient parameters are the draft ones, so the
numbers are a smoke check, not a result.
"""
from __future__ import annotations

import io
import json
import os
import shutil

import pytest

from deliver_folder import TWO_STOREY_CREDIT_RULES
from leed_daylight.drivers.option2 import OPTION2_BINARIES, option2_folder
from leed_daylight.drivers.report import report_folder
from leed_daylight.drivers.run import run_folder
from leed_daylight.drivers.views import run_views
from leed_daylight.engine.radiance import REQUIRED_BINARIES
from merge_split_folder import CASE, build_combined_project

#: Every binary the run, option2 and views commands look for.
BINARIES = tuple(dict.fromkeys((*REQUIRED_BINARIES, *OPTION2_BINARIES)))

needs_smoke = pytest.mark.skipif(
    os.environ.get("LEED_RADIANCE_SMOKE") != "1" or any(shutil.which(b) is None for b in BINARIES),
    reason="the merge-split Radiance run needs Radiance on PATH and LEED_RADIANCE_SMOKE=1",
)

SPACES = ["Room West", "Room East"]


def _draft(d):
    d["radiance"]["ambient"]["full"] = d["radiance"]["ambient"]["draft"]


@needs_smoke
def test_a_split_and_a_merged_space_run_end_to_end(tmp_path, write_example, write_wea):
    folder = build_combined_project(tmp_path, write_example, write_wea, mutate=_draft)
    (folder / "credit_rules.yaml").write_text(TWO_STOREY_CREDIT_RULES, newline="\n")
    out = io.StringIO()

    option1 = json.loads(run_folder(folder, CASE, out=out).read_text())
    option2 = json.loads(option2_folder(folder, CASE, out=out).read_text())
    views = json.loads(run_views(folder, CASE, out=out).read_text())
    reports = report_folder(folder, CASE, out=out, blender=False)

    print("\n" + out.getvalue())
    for result in (option1, option2, views):
        assert [s["room"] for s in result["spaces"]] == SPACES
    assert [(s["room"], s["area_m2"]) for s in option1["spaces"]] == [("Room West", 18.0), ("Room East", 22.0)]
    assert sorted(g["group_id"] for g in option1["groups"]) == ["Room_East_p1_180", "Room_West_180"]
    assert views["excluded_spaces"] == []
    for option in ("option1", "option2", "views"):
        assert (reports / option / f"00_PROJECT_{option}.docx").is_file()
