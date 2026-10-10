# tests/acceptance/test_trace_scope_equivalence.py
"""The trace_scope equivalence proof: spec 6.11's proof obligation.

`performance.trace_scope: storey` is an optimisation over the default,
slower `all` scope: for each window group it traces only the sensor points
on its own storey (plus whatever `performance.linked_zones` names as also
lit from that storey), instead of every sensor point in the model. Spec
6.11 makes this an obligation, not a hope: on the fixture, `storey` and
`all` must agree within 0.01 percentage points on both sDA and ASE before
`storey` may be a project's default -- because a storey with an unrecorded
`linked_zones` entry (a room lit through an internal opening the
project.yaml forgot to name) is exactly the case `storey` scope would get
wrong and `all` would not.

Two of the three tests below need the private fixture (as
`tests/acceptance/conftest.py` resolves it) and are marked
`@pytest.mark.acceptance`; with the fixture unset they skip, same as every
other acceptance test (Global Constraint 9: the user decides when the
simulation runs, never this module). They are also genuinely expensive --
`all` is the slow scope `storey` exists to avoid tracing -- so this module
runs the pipeline chain through `run_option1` twice on the SAME prepared
model (once per scope), via `conftest.py`'s `run_option1_for_prepared`
(task 4.8 review, finding 1: there is exactly ONE copy of that chain,
shared with `conftest.py`'s own `_option1_run`, so the two can never
silently drift apart), never runs it a third time on its own initiative,
and never sets `LEED_ACCEPTANCE_FIXTURE` or executes a Radiance binary
itself.

The remaining tests are synthetic and need no fixture, so they run in
every ordinary suite: `test_the_failure_message_names_the_worst_room` and
the two `_worst_room_message` refusal-path tests exercise that pure
helper on invented rooms (task 4.8 review, finding 2); and
`test_project_with_trace_scope_only_replaces_trace_scope` exercises
`conftest.py`'s `_project_with_trace_scope` on a real, ordinary
(non-fixture) `Project` built from `tests/fixtures/project_example.yaml`
via the shared `write_example` fixture, proving that substituting
`trace_scope` never touches anything else in the `Project` it builds for
`run_option1_for_prepared`.

No fixture value is written here. The only literals this file owns are
`TRACE_SCOPE_TOLERANCE_PCT` (decision-shaped, from spec 6.11, not a fact
about any building) and the invented room names/numbers inside the
`SYNTHETIC_` constants.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import pytest

from leed_daylight.project.config import load_project

from .conftest import _project_with_trace_scope, expected_aggregation, run_option1_for_prepared

#: Spec 6.11: twenty times tighter than SDA_ASE_TOLERANCE_PCT
#: (tests/acceptance/test_fixture_option1.py), because this compares the
#: tool against itself, not against an independent measurement.
TRACE_SCOPE_TOLERANCE_PCT = 0.01

#: Invented rooms, storeys and sDA-like percentages for
#: `test_the_failure_message_names_the_worst_room` -- no fixture value.
#: "Room B", on storey 1, has by far the largest gap between the two
#: scopes, as if its window group were missing a `linked_zones` entry.
SYNTHETIC_WORST_ROOM_TRIAL = {
    "all": {"Room A": 90.0, "Room B": 82.0, "Room C": 75.0},
    "storey": {"Room A": 90.0, "Room B": 55.0, "Room C": 74.0},
    "room_storey": {"Room A": 0, "Room B": 1, "Room C": 1},
}

#: Two rooms named on one side and not the other -- invented, for
#: `test_worst_room_message_refuses_mismatched_room_sets` (task 4.8
#: review, finding 2).
SYNTHETIC_WORST_ROOM_MISMATCH = {
    "all": {"Room A": 90.0},
    "storey": {"Room D": 90.0},
    "room_storey": {"Room A": 0},
}

#: No rooms at all, for `test_worst_room_message_refuses_empty_mappings`
#: (task 4.8 review, finding 2).
SYNTHETIC_WORST_ROOM_EMPTY = {"all": {}, "storey": {}, "room_storey": {}}


def _worst_room_message(
    scope_a: Mapping[str, float], scope_b: Mapping[str, float],
    room_storey: Mapping[str, int],
) -> str:
    """The room with the largest absolute difference between two per-room
    percentage mappings (sDA or ASE, whichever the caller is comparing),
    naming its storey and the difference -- used as the assertion message
    when the two `trace_scope` runs disagree by more than
    `TRACE_SCOPE_TOLERANCE_PCT`, so a failure points straight at the storey
    most likely to hold the unrecorded `linked_zones` entry.

    A pure function: no fixture, no I/O, callable on invented data.
    """
    rooms = sorted(scope_a)
    if rooms != sorted(scope_b) or rooms != sorted(room_storey):
        raise ValueError(
            "_worst_room_message: scope_a, scope_b and room_storey must all "
            f"name the same rooms; got {sorted(scope_a)}, {sorted(scope_b)}, "
            f"{sorted(room_storey)}"
        )
    if not rooms:
        raise ValueError(
            "_worst_room_message: scope_a, scope_b and room_storey are all "
            "empty; there is no room to name as the worst"
        )
    worst_room = max(rooms, key=lambda room: abs(scope_a[room] - scope_b[room]))
    diff = abs(scope_a[worst_room] - scope_b[worst_room])
    storey = room_storey[worst_room]
    return (
        f"largest per-room difference at storey {storey}, room {worst_room!r}: "
        f"{diff:.4f} percentage points (all={scope_a[worst_room]:.2f}, "
        f"storey={scope_b[worst_room]:.2f})"
    )


@pytest.fixture(scope="session")
def _option1_run_all(_prepared):
    fixture, project, prims, rooms, spaces, prepared = _prepared
    result = run_option1_for_prepared(
        fixture, project, prepared, spaces,
        trace_scope="all", log_name="radiance-trace-scope-all.log",
    )
    return fixture, result


@pytest.fixture(scope="session")
def _option1_run_storey(_prepared):
    fixture, project, prims, rooms, spaces, prepared = _prepared
    result = run_option1_for_prepared(
        fixture, project, prepared, spaces,
        trace_scope="storey", log_name="radiance-trace-scope-storey.log",
    )
    return fixture, result


def _room_storey_map(prepared) -> dict[str, int]:
    return {grid.room: grid.storey for grid in prepared.grids}


@pytest.mark.acceptance
def test_storey_scope_equals_all_scope_for_sda(_prepared, _option1_run_all, _option1_run_storey):
    fixture, project, prims, rooms, spaces, prepared = _prepared
    _, result_all = _option1_run_all
    _, result_storey = _option1_run_storey

    aggregation = expected_aggregation(fixture.expected)
    actual_all = result_all.sda_area_pct if aggregation == "floor_area" else result_all.sda_point_pct
    actual_storey = (
        result_storey.sda_area_pct if aggregation == "floor_area" else result_storey.sda_point_pct
    )
    diff = abs(actual_all - actual_storey)

    room_storey = _room_storey_map(prepared)
    by_room_all = {s.room: s.sda_pct for s in result_all.spaces}
    by_room_storey = {s.room: s.sda_pct for s in result_storey.spaces}
    detail = _worst_room_message(by_room_all, by_room_storey, room_storey)

    assert diff <= TRACE_SCOPE_TOLERANCE_PCT, (
        f"sDA ({aggregation}) differs between trace_scope 'all' ({actual_all:.4f}) "
        f"and 'storey' ({actual_storey:.4f}) by {diff:.4f} percentage points, over "
        f"the {TRACE_SCOPE_TOLERANCE_PCT} pp tolerance; {detail}"
    )


@pytest.mark.acceptance
def test_storey_scope_equals_all_scope_for_ase(_prepared, _option1_run_all, _option1_run_storey):
    fixture, project, prims, rooms, spaces, prepared = _prepared
    _, result_all = _option1_run_all
    _, result_storey = _option1_run_storey

    aggregation = expected_aggregation(fixture.expected)
    actual_all = result_all.ase_area_pct if aggregation == "floor_area" else result_all.ase_point_pct
    actual_storey = (
        result_storey.ase_area_pct if aggregation == "floor_area" else result_storey.ase_point_pct
    )
    diff = abs(actual_all - actual_storey)

    room_storey = _room_storey_map(prepared)
    by_room_all = {s.room: s.ase_pct for s in result_all.spaces}
    by_room_storey = {s.room: s.ase_pct for s in result_storey.spaces}
    detail = _worst_room_message(by_room_all, by_room_storey, room_storey)

    assert diff <= TRACE_SCOPE_TOLERANCE_PCT, (
        f"ASE ({aggregation}) differs between trace_scope 'all' ({actual_all:.4f}) "
        f"and 'storey' ({actual_storey:.4f}) by {diff:.4f} percentage points, over "
        f"the {TRACE_SCOPE_TOLERANCE_PCT} pp tolerance; {detail}"
    )


def test_the_failure_message_names_the_worst_room():
    trial = SYNTHETIC_WORST_ROOM_TRIAL
    message = _worst_room_message(trial["all"], trial["storey"], trial["room_storey"])
    assert "Room B" in message
    assert "storey 1" in message


def test_worst_room_message_refuses_mismatched_room_sets():
    trial = SYNTHETIC_WORST_ROOM_MISMATCH
    with pytest.raises(ValueError, match="must all name the same rooms"):
        _worst_room_message(trial["all"], trial["storey"], trial["room_storey"])


def test_worst_room_message_refuses_empty_mappings():
    trial = SYNTHETIC_WORST_ROOM_EMPTY
    with pytest.raises(ValueError, match="there is no room to name as the worst"):
        _worst_room_message(trial["all"], trial["storey"], trial["room_storey"])


def test_project_with_trace_scope_only_replaces_trace_scope(tmp_path, write_example):
    """`conftest._project_with_trace_scope` (task 4.8 review, finding 1),
    run against a real, ordinary `Project` built from the fictional example
    project -- never the private fixture, never a fixture value, so this
    runs in every ordinary suite. Proves the override happens only through
    `dataclasses.replace`: `None` returns the same object untouched, and a
    real scope produces a new object whose `performance.trace_scope` is the
    override, `performance.linked_zones` is untouched, and every other
    field -- undone by a second `dataclasses.replace` back to the
    original's own `performance` -- reproduces the original project
    exactly.
    """
    project_path = write_example(tmp_path)
    project = load_project(project_path)

    unchanged = _project_with_trace_scope(project, None)
    assert unchanged is project

    scoped = _project_with_trace_scope(project, "storey")
    assert scoped is not project
    assert scoped.performance.trace_scope == "storey"
    assert scoped.performance.trace_scope != project.performance.trace_scope
    assert scoped.performance.linked_zones == project.performance.linked_zones
    # Undo the substitution: every field other than performance.trace_scope
    # must be exactly the original project's, proving dataclasses.replace
    # touched nothing else.
    assert dataclasses.replace(scoped, performance=project.performance) == project
