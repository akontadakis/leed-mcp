# tests/acceptance/test_fixture_option1.py
"""The acceptance gate: spec 10's phase 4 exit condition.

Every test below except the first two needs the private fixture
(``conftest.FIXTURE_ENV_VAR``) and is marked ``@pytest.mark.acceptance``; in
an ordinary run, with the variable unset, each of those skips by way of the
session-scoped fixtures in ``conftest.py``. Nothing here ever sets the
variable, creates a fixture folder, or runs Radiance -- the user decides
when the acceptance run happens (Global Constraint 9).

The fixture may hold several runs (``conftest.py``); the session runs the one
``LEED_ACCEPTANCE_RUN`` names. A run without an sDA or ASE target has no
pass mark: its two gate tests skip, saying so, after the run has written
its ``acceptance_result.json``. A run without a window-group or grid-point
count does not check it.

No fixture value (a path, a room name, a building name, a number) is
written here. Every number this file compares against comes out of
``expected.json`` at test time; the only literal this file owns is the
tolerance constant below, which is a rule from decision D9, not a fact
about any building.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from .conftest import (
    FIXTURE_ENV_VAR,
    AcceptanceFixtureError,
    _resolve_fixture,
    expected_aggregation,
    expected_count,
    expected_number,
    expected_number_list,
    expected_optional_count,
    expected_rooms,
    expected_str,
    expected_target,
    frame_area_m2,
)

#: Decision D9: "the only independent proof available" is compared within
#: half a percentage point, on both sDA and ASE.
SDA_ASE_TOLERANCE_PCT = 0.5

ACCEPTANCE_DIR = pathlib.Path(__file__).parent

#: A count/areas-list pair whose lengths disagree, for the cross-check test
#: below. Invented, round numbers -- no fixture value.
SYNTHETIC_ROOM_COUNT_AREA_MISMATCH = {"rooms": {"count": 2, "areas_m2": []}}

#: A fictional 1 m^2 square opening whose glass covers half of it -- shape
#: and area invented only to drive `frame_area_m2`'s missing-opening
#: refusal (M8); no fixture value.
SYNTHETIC_FRAME_INPUTS = {
    "square_points": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
    "achieved_glass_area_m2": 0.5,
}


def test_the_acceptance_test_skips_without_the_fixture(monkeypatch):
    monkeypatch.delenv(FIXTURE_ENV_VAR, raising=False)
    with pytest.raises(pytest.skip.Exception, match=FIXTURE_ENV_VAR):
        _resolve_fixture()


@pytest.mark.acceptance
def test_option1_reproduces_the_fixture_sda(_option1_run):
    fixture, result = _option1_run
    aggregation = expected_aggregation(fixture.expected)
    actual_pct = result.sda_area_pct if aggregation == "floor_area" else result.sda_point_pct
    expected_pct = expected_target(fixture.expected, "option1.sda_pct")
    if expected_pct is None:
        pytest.skip(f"run {fixture.run!r} has no sDA target (no pass mark); got {actual_pct:.3f}")
    diff = abs(actual_pct - expected_pct)
    assert diff <= SDA_ASE_TOLERANCE_PCT, (
        f"sDA ({aggregation}) is {diff:.2f} percentage points off "
        f"expected.json: got {actual_pct:.2f}, expected {expected_pct:.2f}"
    )


@pytest.mark.acceptance
def test_option1_reproduces_the_fixture_ase(_option1_run):
    fixture, result = _option1_run
    aggregation = expected_aggregation(fixture.expected)
    actual_pct = result.ase_area_pct if aggregation == "floor_area" else result.ase_point_pct
    expected_pct = expected_target(fixture.expected, "option1.ase_pct")
    if expected_pct is None:
        pytest.skip(f"run {fixture.run!r} has no ASE target (no pass mark); got {actual_pct:.3f}")
    diff = abs(actual_pct - expected_pct)
    assert diff <= SDA_ASE_TOLERANCE_PCT, (
        f"ASE ({aggregation}) is {diff:.2f} percentage points off "
        f"expected.json: got {actual_pct:.2f}, expected {expected_pct:.2f}"
    )


@pytest.mark.acceptance
def test_the_room_count_and_areas_match(_prepared):
    fixture, project, prims, rooms, spaces, prepared = _prepared
    # expected_rooms() also cross-checks: len(areas_m2) must equal count.
    _room_count, expected_areas = expected_rooms(fixture.expected)
    tolerance = expected_number(fixture.expected, "tolerances.room_area_m2")

    actual_areas = sorted(room.area_m2 for room in rooms.rooms)
    assert len(actual_areas) == _room_count, (
        f"detect_rooms found {len(actual_areas)} room(s); expected.json "
        f"records {_room_count}"
    )
    diffs = [abs(a - e) for a, e in zip(actual_areas, expected_areas)]
    assert all(d <= tolerance for d in diffs), (
        "room areas differ from expected.json beyond tolerances.room_area_m2: "
        f"got {actual_areas}, expected {expected_areas}"
    )


@pytest.mark.acceptance
def test_the_prepared_model_matches(_prepared):
    fixture, project, prims, rooms, spaces, prepared = _prepared
    expected_glass = expected_number(fixture.expected, "prepared.glass_area_m2")
    expected_frame = expected_number(fixture.expected, "prepared.frame_area_m2")
    expected_groups = expected_optional_count(fixture.expected, "prepared.group_count")
    expected_points = expected_optional_count(fixture.expected, "prepared.grid_point_count")
    glass_tolerance = expected_number(fixture.expected, "tolerances.glass_area_m2")
    frame_tolerance = expected_number(fixture.expected, "tolerances.frame_area_m2")

    openings = prepared.manifest["openings"]
    prims_by_name = {p.name: p for p in prims}
    glass_area = sum(record["achieved_glass_area_m2"] for record in openings.values())
    frame_area = frame_area_m2(prims_by_name, openings)
    group_count = prepared.manifest["totals"]["group_count"]
    grid_point_count = prepared.manifest["totals"]["grid_point_count"]

    assert abs(glass_area - expected_glass) <= glass_tolerance, (
        "prepared glass area differs from expected.json beyond "
        "tolerances.glass_area_m2"
    )
    assert abs(frame_area - expected_frame) <= frame_tolerance, (
        "prepared frame area differs from expected.json beyond "
        "tolerances.frame_area_m2"
    )
    assert expected_groups is None or group_count == expected_groups, (
        f"prepared model has {group_count} window group(s); expected.json "
        f"records {expected_groups}"
    )
    assert expected_points is None or grid_point_count == expected_points, (
        f"prepared model has {grid_point_count} grid point(s); expected.json "
        f"records {expected_points}"
    )


# -- expected.json validation, exercised directly (no fixture needed) -----
#
# These call the `expected_*` helpers on hand-built dicts. None of them
# touch LEED_ACCEPTANCE_FIXTURE, so they run in every ordinary suite, same
# as `test_the_acceptance_test_skips_without_the_fixture` above.


def test_expected_number_names_a_missing_key():
    with pytest.raises(AcceptanceFixtureError, match=r"option1\.sda_pct"):
        expected_number({"option1": {}}, "option1.sda_pct")


def test_expected_number_names_a_wrong_type():
    with pytest.raises(AcceptanceFixtureError, match=r"option1\.sda_pct"):
        expected_number({"option1": {"sda_pct": "not a number"}}, "option1.sda_pct")


def test_expected_number_rejects_a_boolean():
    with pytest.raises(AcceptanceFixtureError, match=r"option1\.sda_pct"):
        expected_number({"option1": {"sda_pct": True}}, "option1.sda_pct")


def test_expected_number_rejects_a_non_finite_value():
    with pytest.raises(AcceptanceFixtureError, match=r"option1\.sda_pct"):
        expected_number({"option1": {"sda_pct": float("nan")}}, "option1.sda_pct")


def test_expected_count_rejects_a_boolean():
    with pytest.raises(AcceptanceFixtureError, match=r"rooms\.count"):
        expected_count({"rooms": {"count": True}}, "rooms.count")


def test_expected_count_rejects_a_negative_value():
    with pytest.raises(AcceptanceFixtureError, match=r"rooms\.count"):
        expected_count({"rooms": {"count": -1}}, "rooms.count")


def test_expected_count_rejects_a_non_integer():
    with pytest.raises(AcceptanceFixtureError, match=r"rooms\.count"):
        expected_count({"rooms": {"count": "many"}}, "rooms.count")


def test_expected_str_rejects_a_non_string():
    with pytest.raises(AcceptanceFixtureError, match=r"option1\.case"):
        expected_str({"option1": {"case": 1}}, "option1.case")


def test_expected_number_list_names_a_missing_key():
    with pytest.raises(AcceptanceFixtureError, match=r"rooms\.areas_m2"):
        expected_number_list({"rooms": {}}, "rooms.areas_m2")


def test_expected_number_list_rejects_a_non_list():
    with pytest.raises(AcceptanceFixtureError, match=r"rooms\.areas_m2"):
        expected_number_list({"rooms": {"areas_m2": "not a list"}}, "rooms.areas_m2")


def test_expected_number_list_rejects_a_non_numeric_element():
    with pytest.raises(AcceptanceFixtureError, match=r"rooms\.areas_m2\[0\]"):
        expected_number_list({"rooms": {"areas_m2": ["x"]}}, "rooms.areas_m2")


def test_expected_number_list_rejects_a_non_finite_element():
    with pytest.raises(AcceptanceFixtureError, match=r"rooms\.areas_m2\[0\]"):
        expected_number_list({"rooms": {"areas_m2": [float("inf")]}}, "rooms.areas_m2")


def test_expected_aggregation_names_a_missing_key():
    with pytest.raises(AcceptanceFixtureError, match=r"option1\.aggregation"):
        expected_aggregation({"option1": {}})


def test_expected_aggregation_rejects_an_unknown_value():
    with pytest.raises(AcceptanceFixtureError, match=r"option1\.aggregation"):
        expected_aggregation({"option1": {"aggregation": "by_room"}})


def test_expected_rooms_names_a_count_areas_mismatch():
    with pytest.raises(AcceptanceFixtureError, match=r"rooms\.areas_m2"):
        expected_rooms(SYNTHETIC_ROOM_COUNT_AREA_MISMATCH)


class _SyntheticPrim:
    """Stands in for a `scene.primitives.Primitive` -- only `.points` is
    needed by `frame_area_m2`."""

    def __init__(self, points):
        self.points = points


def test_frame_area_m2_names_a_missing_opening():
    # M8: a manifest opening with no matching scene primitive must be a
    # named AcceptanceFixtureError, never a raw KeyError.
    prims_by_name = {
        "opening_a": _SyntheticPrim(SYNTHETIC_FRAME_INPUTS["square_points"]),
    }
    openings = {
        "opening_b": {"achieved_glass_area_m2": SYNTHETIC_FRAME_INPUTS["achieved_glass_area_m2"]},
    }

    with pytest.raises(AcceptanceFixtureError, match="opening_b"):
        frame_area_m2(prims_by_name, openings)


def _tolerance_or_synthetic_names(tree: ast.Module) -> set[int]:
    """Object ids of every ``ast.Constant`` inside a module-level assignment
    whose target name contains ``TOLERANCE`` or starts with ``SYNTHETIC_``
    (the confidentiality rule's two carve-outs)."""
    allowed: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
            value = node.value
        else:
            continue
        if value is None or not any("TOLERANCE" in n or n.startswith("SYNTHETIC_") for n in names):
            continue
        for sub in ast.walk(value):
            if isinstance(sub, ast.Constant):
                allowed.add(id(sub))
    return allowed


def _numeric_literal_findings(path: pathlib.Path) -> list[tuple[pathlib.Path, int, object]]:
    """Every numeric literal in ``path`` other than 0 and 1, outside a
    TOLERANCE/SYNTHETIC_ constant -- a fixture value hiding in plain sight."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    allowed = _tolerance_or_synthetic_names(tree)
    findings: list[tuple[pathlib.Path, int, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if value in (0, 1):
            continue
        if id(node) in allowed:
            continue
        findings.append((path, node.lineno, value))
    return findings


def test_no_fixture_value_is_hard_coded():
    findings = []
    for path in sorted(ACCEPTANCE_DIR.glob("*.py")):
        findings += _numeric_literal_findings(path)
    assert not findings, (
        "numeric literal(s) found in tests/acceptance/ outside a "
        "TOLERANCE/SYNTHETIC_ constant -- a fixture value must never be "
        "hard-coded here:\n"
        + "\n".join(f"{p}:{line}: {value!r}" for p, line, value in findings)
    )
