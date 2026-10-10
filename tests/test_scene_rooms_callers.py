# tests/test_scene_rooms_callers.py
"""Every command that finds rooms applies splits.csv (merge-split spec 4):
``drivers/prepare.py``, ``run.py``, ``option2.py``, ``views.py``,
``rooms.py``, ``scan.py`` and ``deliver/data.prepare_case_model`` call
``project.layout.scene_rooms``.

Each test runs one caller through its public entry on the split project of
``tests/merge_split_folder.py`` (Room North cut along x = 4.5 m into Room
West and Room Strip) and checks that a half's space comes out. Before the
callers switched, the rows of the two halves matched no detected room.
Fake engines only; no Radiance binary is looked for or run.
"""
from __future__ import annotations

import csv
import io
import json

import pytest

import leed_daylight.drivers.option2 as option2_module
import leed_daylight.drivers.run as run_module
import leed_daylight.drivers.views as views_module
from deliver_folder import Option1Fake, Option2Fake
from leed_daylight.deliver.data import prepare_case_model
from leed_daylight.drivers.prepare import prepare_folder
from leed_daylight.drivers.rooms import rooms_folder
from leed_daylight.drivers.scan import scan_folder
from leed_daylight.project.config import ProjectError, load_project
from leed_daylight.project.spaces import load_spaces
from merge_split_folder import CASE, HEADER, build_split_project, prepared_model
from test_views_command import RULES_YAML, FakeTracer

HALVES = ["Room West", "Room Strip"]


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    return build_split_project(tmp_path, write_example, write_wea)


def test_prepare_applies_splits_csv(folder):
    prepare_folder(folder, CASE, out=io.StringIO())

    manifest = json.loads((folder / "_run" / "manifest.json").read_text())
    assert sorted(manifest["rooms"]) == sorted(HALVES)
    assert manifest["layout"]["Room West"]["pieces"][0]["room_id"] == "S0_R0a"


def test_run_applies_splits_csv(folder):
    engine = Option1Fake(prepared_model(folder))

    path = run_module.run_folder(folder, CASE, engine=engine, out=io.StringIO())

    assert [s["room"] for s in json.loads(path.read_text())["spaces"]] == HALVES


def test_option2_applies_splits_csv(folder):
    engine = Option2Fake(prepared_model(folder))

    path = option2_module.option2_folder(folder, CASE, engine=engine, out=io.StringIO())

    assert [s["room"] for s in json.loads(path.read_text())["spaces"]] == HALVES


def test_views_applies_splits_csv(folder):
    (folder / "credit_rules.yaml").write_text(RULES_YAML)

    path = views_module.run_views(folder, CASE, tracer=FakeTracer(), out=io.StringIO())

    assert [s["room"] for s in json.loads(path.read_text())["spaces"]] == HALVES


def test_rooms_applies_splits_csv(folder):
    path = rooms_folder(folder, out=io.StringIO())

    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["area_m2"], r["centroid_x"]) for r in rows] == [("18.00", "2.25"), ("6.00", "5.25")]


def test_scan_applies_splits_csv(folder):
    out = io.StringIO()

    assert scan_folder(folder, out=out) == 0

    assert "storey 0: level 0.00 m, 2 room(s), 24.00 m2" in out.getvalue()


def test_prepare_case_model_applies_splits_csv(folder, tmp_path):
    project = load_project(folder / "project.yaml")

    model = prepare_case_model(project, load_spaces(folder / "spaces.csv"), CASE, tmp_path / "model")

    assert model.matched["Room West"].room_id == "S0_R0a"
    assert [s.name for s in model.layout.spaces] == HALVES
    assert model.layout.space("Room Strip").origin == "split"


def test_prepare_case_model_refuses_a_stale_row_with_the_rooms_hint(folder, tmp_path):
    """Review M1: the atlas (which runs before any result) gives the same
    advice as prepare for a row that still holds the whole room's centroid."""
    (folder / "spaces.csv").write_text(HEADER + "L0,Room North,office,yes,24,3.0,2.0,0.0,no,,1,1,\n")
    project = load_project(folder / "project.yaml")

    with pytest.raises(ProjectError) as excinfo:
        prepare_case_model(project, load_spaces(folder / "spaces.csv"), CASE, tmp_path / "model")

    assert str(excinfo.value) == (
        "spaces.csv: space 'Room North': nearest room on its level is 0.75 m away, outside the "
        "0.50 m tolerance; splits.csv changed the rooms: run the rooms command again and copy "
        "each half's centroid into spaces.csv"
    )
