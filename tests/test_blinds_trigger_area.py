# tests/test_blinds_trigger_area.py
"""``blinds.trigger_area``: which area picks a space's LM-83-12 2.2.6 tier.

``room`` (the default, and the only behaviour before this key): the room
area from the scene. ``grid``: the area the space's sensor points stand
for, the point count times ``grid.spacing_m`` squared -- the basis of a
study whose blind tiers were set from its grid. Every room, area and lux
value is invented and round.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from test_run_option1 import FULL_MASK_LEN, N_HOURS, FakeEngine, _group, _prepared, _project, _rules, _space

from leed_daylight.build.grid import RoomGrid
from leed_daylight.drivers.run_option1 import run_option1
from leed_daylight.project.config import ProjectError, check_project_draft, load_project

ROOM = "Room North"
ROOM_WARNING = "the blind trigger area is the room area from the scene, not the grid area"
GRID_WARNING = (
    "the blind trigger area is the area the sensor points stand for (points x grid spacing "
    "squared), not the room area"
)


def _with_trigger_area(tmp_path, write_example, value):
    def mutate(d):
        d["blinds"]["trigger_area"] = value
        return d

    return write_example(tmp_path, mutate)


# -- the key -------------------------------------------------------------------------


def test_the_key_is_optional_and_room_by_default(example_project_dir):
    assert load_project(example_project_dir / "project.yaml").blinds.trigger_area == "room"


@pytest.mark.parametrize("value", ["room", "grid"])
def test_room_and_grid_are_read(tmp_path, write_example, value):
    assert load_project(_with_trigger_area(tmp_path, write_example, value)).blinds.trigger_area == value


def test_another_value_is_refused(tmp_path, write_example):
    path = _with_trigger_area(tmp_path, write_example, "floor")

    with pytest.raises(ProjectError, match=r"blinds\.trigger_area is 'floor'; expected one of \['grid', 'room'\]"):
        load_project(path)


def test_the_draft_check_lists_a_fill_me_trigger_area_as_unfinished(tmp_path, write_example):
    path = _with_trigger_area(tmp_path, write_example, "FILL_ME")

    assert check_project_draft(path) == ((), ("blinds.trigger_area",))


# -- the run ---------------------------------------------------------------------------


def _run(tmp_path, write_example, trigger_area):
    """One room of 30 m2 (323 sf: three lit points allowed) whose four
    sensor points stand for 1.44 m2 (16 sf: one lit point allowed). In the
    first occupied hour two points get direct sun through the one group."""
    project = _project(tmp_path, write_example)
    project = dataclasses.replace(project, blinds=dataclasses.replace(project.blinds, trigger_area=trigger_area))
    grid = RoomGrid(room=ROOM, points=np.zeros((4, 3)), spacing_m=0.6, cell_area_m2=0.36,
                    room_area_m2=30.0, storey=0, level_z=0.0)
    group = _group("group_a", ROOM)
    rules = _rules()
    direct = np.zeros((4, N_HOURS))
    direct[:2, 0] = rules["option1"]["ase"]["lux"] * 2
    idx = (ROOM,) * 4
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"group_a": (idx, direct)},
        sky={"group_a": (idx, np.full((4, N_HOURS), 400.0), np.full((4, N_HOURS), 100.0))},
    )
    prepared = _prepared(tmp_path, [group], [grid])
    return run_option1(None, project, prepared, [_space(ROOM)], rules, "base_case", engine=engine)


def test_the_room_area_keeps_two_lit_points_under_its_tier(tmp_path, write_example):
    result = _run(tmp_path, write_example, "room")

    assert result.blind_schedule["group_a"].all()
    assert ROOM_WARNING in result.warnings
    assert GRID_WARNING not in result.warnings


def test_the_grid_area_closes_the_blind_for_two_lit_points(tmp_path, write_example):
    result = _run(tmp_path, write_example, "grid")

    schedule = result.blind_schedule["group_a"]
    assert not schedule[0]
    assert schedule[1:].all()
    assert GRID_WARNING in result.warnings
    assert ROOM_WARNING not in result.warnings


def test_the_grid_area_never_changes_the_area_weighting(tmp_path, write_example):
    (tmp_path / "room").mkdir()
    (tmp_path / "grid").mkdir()
    room = _run(tmp_path / "room", write_example, "room")
    grid = _run(tmp_path / "grid", write_example, "grid")

    assert [s.area_m2 for s in room.spaces] == [s.area_m2 for s in grid.spaces] == [30.0]
