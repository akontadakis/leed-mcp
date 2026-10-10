# tests/test_merge_split_runs.py
"""Option 1, Option 2 and Quality Views on merged and split spaces
(merge-split spec 7: blinds, results; spec 4: the consumers), through the
real drivers with fake engines. No Radiance binary is looked for or run.

The projects are those of ``tests/merge_split_folder.py``: the merge
project (Room North and Room East, one space of two pieces, 40 m2, with a
pane in the gap between them that ties between the two pieces) and the
split project (Room North cut along x = 4.5 m into Room West and Room
Strip). Every number is invented and round.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pytest

import leed_daylight.drivers.option2 as option2_module
import leed_daylight.drivers.run as run_module
import leed_daylight.drivers.run_option1 as run_option1_module
import leed_daylight.drivers.views as views_module
from deliver_folder import N_HOURS, Option1Fake, Option2Fake
from leed_daylight.metrics.blinds import allowed_lit_points, blind_schedule
from merge_split_folder import (
    CASE, build_combined_project, build_merge_project, build_split_project, prepared_model,
)
from test_views_command import RULES_YAML, FakeTracer

#: Direct sun on a lit point, lux: over the ASE threshold (1,000 lux).
LIT_LUX = 2000.0
#: The x that divides the two pieces of the merged space (the middle of the gap).
GAP_X = 6.5


@pytest.fixture
def merged(tmp_path, write_example, write_wea):
    return build_merge_project(tmp_path, write_example, write_wea)


@pytest.fixture
def split(tmp_path, write_example, write_wea):
    return build_split_project(tmp_path, write_example, write_wea)


def _views_run(folder):
    (folder / "credit_rules.yaml").write_text(RULES_YAML, newline="\n")
    path = views_module.run_views(folder, CASE, tracer=FakeTracer(), out=io.StringIO())
    return path, json.loads(path.read_text())


def test_a_merged_space_runs_end_to_end(merged, monkeypatch):
    prepared = prepared_model(merged)
    (grid,) = prepared.grids
    calls = []

    def spy(direct_by_group, area_m2, lit_threshold):
        calls.append((direct_by_group.shape, area_m2))
        return blind_schedule(direct_by_group, area_m2, lit_threshold)

    monkeypatch.setattr(run_option1_module, "blind_schedule", spy)

    path = run_module.run_folder(merged, CASE, engine=Option1Fake(prepared), out=io.StringIO())

    result = json.loads(path.read_text())
    # One space, two pieces, one grid.
    assert [(s["room"], s["area_m2"]) for s in result["spaces"]] == [("Room North", 40.0)]
    assert grid.room == "Room North" and grid.room_area_m2 == 40.0
    west = int((grid.points[:, 0] < GAP_X).sum())
    assert 0 < west < grid.points.shape[0]
    assert result["spaces"][0]["points"] == grid.points.shape[0]
    # Groups per piece; the gap pane, tied between the pieces, goes with piece 1.
    assert [(g["group_id"], g["room"]) for g in result["groups"]] == [
        ("Room_North_p1_180", "Room North"), ("Room_North_p2_180", "Room North"),
    ]
    assert [[o.name for o in g.openings] for g in prepared.groups] == [
        ["window_ext", "window_gap"], ["window_east"],
    ]
    # One blind schedule for the space: both groups, all its points, its area.
    assert calls == [((2, grid.points.shape[0], 3650), 40.0)]


class LitPoints:
    """Direct sun of LIT_LUX, every occupied hour, on the first three points
    of Room North (through group p1) and the first point of Room East
    (through group p2); 200 lux of sky light with blinds open, 50 closed;
    no residual light."""

    n_hours = N_HOURS

    def __init__(self, prepared):
        (grid,) = prepared.grids
        n = grid.points.shape[0]
        self._index = (grid.room,) * n
        self._n = n
        west = np.flatnonzero(grid.points[:, 0] < GAP_X)
        east = np.flatnonzero(grid.points[:, 0] > GAP_X)
        self.lit = {"Room_North_p1_180": west[:3], "Room_North_p2_180": east[:1]}
        self.pieces = (west, east)

    def direct(self, group, hours):
        lux = np.zeros((self._n, int(hours.sum())))
        lux[self.lit[group.group_id]] = LIT_LUX
        return self._index, lux

    def sky(self, group, hours, open_schedule):
        shape = (self._n, int(hours.sum()))
        return self._index, np.full(shape, 200.0), np.full(shape, 50.0)

    def residual(self, hours):
        zeros = np.zeros((self._n, int(hours.sum())))
        return [(self._index, zeros, zeros)]


def test_blinds_close_when_the_space_is_counted_whole(merged):
    prepared = prepared_model(merged)
    engine = LitPoints(prepared)
    # Room North (24 m2) may have 3 lit points and Room East (16 m2) 1; the
    # space (40 m2) may have 3, and has 4.
    assert (allowed_lit_points(24.0, 10), allowed_lit_points(16.0, 10), allowed_lit_points(40.0, 10)) == (3, 1, 3)
    hours = 4
    for piece, group_id in zip(engine.pieces, ("Room_North_p1_180", "Room_North_p2_180")):
        direct = np.zeros((1, len(piece), hours))
        direct[0, np.isin(piece, engine.lit[group_id])] = LIT_LUX
        area = 24.0 if group_id.endswith("p1_180") else 16.0
        assert blind_schedule(direct, area, 1000.0).all()  # counted per piece: open

    path = run_module.run_folder(merged, CASE, engine=engine, out=io.StringIO())

    closed = json.loads(path.read_text())["blinds_closed_hours"]
    # Counted per space: the group lighting the most points closes every hour.
    assert closed == {"Room_North_p1_180": 3650, "Room_North_p2_180": 0}


def test_option1_and_option2_grid_checks_pass_on_a_merged_space(merged):
    prepared = prepared_model(merged)

    option1 = run_module.run_folder(merged, CASE, engine=Option1Fake(prepared), out=io.StringIO())
    option2 = option2_module.option2_folder(merged, CASE, engine=Option2Fake(prepared), out=io.StringIO())

    for path in (option1, option2):
        assert [s["room"] for s in json.loads(path.read_text())["spaces"]] == ["Room North"]


def test_the_views_excluded_list_leaves_out_piece_rows(merged):
    _, result = _views_run(merged)

    assert result["excluded_spaces"] == []
    assert not any("not ticked in views_" in w for w in result["warnings"])
    assert [s["room"] for s in result["spaces"]] == ["Room North"]


@pytest.mark.parametrize("build, rooms", [
    (build_merge_project, ["Room North"]),
    (build_split_project, ["Room West", "Room Strip"]),
])
def test_option2_on_a_merged_and_a_split_space(tmp_path, write_example, write_wea, build, rooms):
    folder = build(tmp_path, write_example, write_wea)
    prepared = prepared_model(folder)

    path = option2_module.option2_folder(folder, CASE, engine=Option2Fake(prepared), out=io.StringIO())

    result = json.loads(path.read_text())
    assert [s["room"] for s in result["spaces"]] == rooms
    assert [s["points"] for s in result["spaces"]] == [g.points.shape[0] for g in prepared.grids]
    lines = (path.parent / "points.csv").read_text().splitlines()
    assert len(lines) - 1 == sum(g.points.shape[0] for g in prepared.grids)


def test_views_on_a_merged_space_draw_every_piece(merged):
    path, result = _views_run(merged)

    (space,) = result["spaces"]
    assert (space["room"], space["area_m2"]) == ("Room North", 40.0)
    for layer in ("type2", "type3", "quality"):
        svg = (path.parent / f"plan_0_{layer}.svg").read_text()
        assert svg.count('class="room"') == 2


def test_views_on_a_split_space_give_one_row_per_half(split):
    path, result = _views_run(split)

    assert [(s["room"], s["area_m2"]) for s in result["spaces"]] == [("Room West", 18.0), ("Room Strip", 6.0)]
    assert (path.parent / "plan_0_quality.svg").read_text().count('class="room"') == 2


def test_the_radiance_smoke_project_prepares(tmp_path, write_example, write_wea):
    # The project of tests/test_merge_split_radiance.py, prepared without Radiance.
    folder = build_combined_project(tmp_path, write_example, write_wea)

    prepared = prepared_model(folder)

    assert {name: entry["origin"] for name, entry in prepared.manifest["layout"].items()} == {
        "Room West": "split", "Room East": "split+merged",
    }
    assert [g.group_id for g in prepared.groups] == ["Room_East_p1_180", "Room_West_180"]
