# tests/test_run_option1.py
"""Tests for leed_daylight.drivers.run_option1 (T4.6): the full Option 1
run -- blind schedule, sDA with blinds operating, ASE with blinds open,
per space and per project, both aggregations, score and glare-narrative
trigger.

Every test injects a FAKE engine (``FakeEngine``, matching the
``Option1Engine`` seam exactly -- the same surface as
``engine.coefficients.Option1Tracer``) that returns canned lux arrays. No
test here builds the real engine or runs a Radiance binary; ``rad`` is
always ``None``. ``Project`` comes from the real, fictional
``tests/fixtures/project_example.yaml`` (via conftest's ``write_example``)
so the analysis/site values a test checks against are the real loaded
ones; ``rules`` comes from the real, public
``leed_daylight/templates/credit_rules_v41_bdc_nc.yaml`` via ``load_rules``
so no LEED threshold is retyped here either. ``PreparedModel``,
``WindowGroup`` and ``RoomGrid`` are hand-built per the brief's controller
ruling, since phase 4 never opens a real scene in this task.

Every room, area and point pattern below is invented and round (a
fictional two-room office, a fictional single room); nothing is copied
from any real project.
"""
from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest

from leed_daylight.build.grid import RoomGrid
from leed_daylight.build.groups import WindowGroup
from leed_daylight.build.prepare import PreparedModel
from leed_daylight.constants import UNVERIFIED
from leed_daylight.drivers.run_option1 import Option1Error, run_option1
from leed_daylight.project.config import load_project
from leed_daylight.project.rules import load_rules
from leed_daylight.project.spaces import Space

TEMPLATE_RULES_PATH = (
    pathlib.Path(__file__).parent.parent
    / "leed_daylight" / "templates" / "credit_rules_v41_bdc_nc.yaml"
)

#: Spec 6.9's own check: 08:00-18:00 every day of a non-leap year is
#: 365 x 10 occupied hours, whatever the DST rule (a shift only moves
#: which standard-time hours are occupied, never how many).
N_HOURS = 3650
#: leed_daylight.metrics.occupancy.HOURS_IN_YEAR -- the mask's own length,
#: independent of the analysis window.
FULL_MASK_LEN = 8760


# =============================================================================
# fixtures and small builders
# =============================================================================


def _project(tmp_path, write_example, *, aggregation=None, daylight_saving=None, hours=None):
    project_path = write_example(tmp_path)
    project = load_project(project_path)
    analysis = project.analysis
    return dataclasses.replace(
        project,
        analysis=dataclasses.replace(
            analysis,
            aggregation=aggregation or analysis.aggregation,
            daylight_saving=daylight_saving or analysis.daylight_saving,
            hours=hours or analysis.hours,
        ),
    )


def _rules():
    return load_rules(TEMPLATE_RULES_PATH)


def _space(room, *, area_m2=20.0, regularly_occupied=True, included_=True, case="base_case"):
    return Space(
        floor="L0", room=room, space_type="office",
        regularly_occupied=regularly_occupied, area_m2=area_m2,
        centroid_x=0.0, centroid_y=0.0, level_z=0.0,
        view_preserving_shades=False, merge_into=None,
        include={case: included_}, reason="",
    )


def _grid(room, n_points, *, room_area_m2=20.0, storey=0, spacing_m=0.6, cell_area_m2=0.36):
    points = np.zeros((n_points, 3), dtype=np.float64)
    return RoomGrid(
        room=room, points=points, spacing_m=spacing_m, cell_area_m2=cell_area_m2,
        room_area_m2=room_area_m2, storey=storey, level_z=float(storey) * 3.0,
    )


def _group(group_id, room, *, storey=0):
    return WindowGroup(
        group_id=group_id, room=room, storey=storey, azimuth_bin_deg=180.0,
        openings=(), outward_normals=(), area_m2=1.0,
    )


def _prepared(root, groups, grids, *, case="base_case", warnings=()):
    root = pathlib.Path(root)
    return PreparedModel(
        root=root, case=case,
        scene_file=root / "scene.rad", materials_file=root / "materials.rad",
        blind_files={}, grid_files={}, manifest_path=root / "manifest.json",
        manifest={"warnings": list(warnings)},
        groups=tuple(groups), grids=tuple(grids), glass_by_group={},
    )


class FakeEngine:
    """The ``Option1Engine`` seam: canned lux, keyed by group_id. Never
    touches a Radiance binary, never validates its own inputs -- a test
    controls exactly what each group returns.
    """

    def __init__(self, n_hours, *, direct, sky, residual=None):
        self._n_hours = n_hours
        self._direct = direct  # group_id -> (sensor_index, lux)
        self._sky = sky  # group_id -> (sensor_index, open_lux, closed_lux_or_None)
        # [(sensor_index, sky_lux, direct_lux), ...], one per trace unit;
        # None -> one unit of zero lux over every room the groups see.
        self._residual = residual
        self.residual_calls = 0

    @property
    def n_hours(self):
        return self._n_hours

    def direct(self, group, hours):
        return self._direct[group.group_id]

    def sky(self, group, hours, open_schedule):
        return self._sky[group.group_id]

    def residual(self, hours):
        self.residual_calls += 1
        if self._residual is not None:
            return self._residual
        rows: dict[str, int] = {}
        n_selected = 0
        for sensor_index, lux in self._direct.values():
            n_selected = lux.shape[1]
            for room in dict.fromkeys(sensor_index):
                rows.setdefault(room, sum(1 for r in sensor_index if r == room))
        index = tuple(room for room, n in rows.items() for _ in range(n))
        zeros = np.zeros((len(index), n_selected))
        return [(index, zeros, zeros.copy())]


# =============================================================================
# the brief's twelve rows
# =============================================================================


def test_a_full_year_of_office_hours_is_3650_occupied_hours(tmp_path, write_example):
    project = _project(tmp_path, write_example, hours="08:00-18:00", daylight_saving="EU")
    rules = _rules()
    room = "Room North"
    grid = _grid(room, 2, room_area_m2=20.0)
    group = _group("group_a", room)
    prepared = _prepared(tmp_path, [group], [grid])
    spaces = [_space(room)]
    idx = (room,) * 2
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"group_a": (idx, np.zeros((2, N_HOURS)))},
        sky={"group_a": (idx, np.full((2, N_HOURS), 400.0), None)},
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    assert result.occupied_hours == 3650


def test_a_wrong_hour_count_is_refused(tmp_path, write_example):
    project = _project(tmp_path, write_example)
    rules = _rules()
    room = "Room North"
    grid = _grid(room, 1)
    group = _group("group_a", room)
    prepared = _prepared(tmp_path, [group], [grid])
    spaces = [_space(room)]
    # The fake engine's own hour count (8784) disagrees with the occupied
    # mask's length (8760, HOURS_IN_YEAR) -- "a different number means the
    # mask is wrong".
    engine = FakeEngine(8784, direct={}, sky={})

    with pytest.raises(ValueError, match=r"8784"):
        run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)


def test_sda_uses_the_blind_schedule_and_ase_does_not(tmp_path, write_example):
    project = _project(tmp_path, write_example)
    rules = _rules()
    ase_lux = rules["option1"]["ase"]["lux"]
    sda_lux = rules["option1"]["sda"]["lux"]
    room = "Room North"
    n_points = 4
    idx = (room,) * n_points

    direct = np.zeros((n_points, N_HOURS))
    n_bright = 2000  # more than half the occupied hours: bright, direct sun
    direct[:, :n_bright] = ase_lux * 2.0  # every point lit, every bright hour

    open_lux = np.full((n_points, N_HOURS), sda_lux + 100.0)  # always passes alone
    closed_lux = np.full((n_points, N_HOURS), sda_lux - 100.0)  # always fails alone

    group = _group("group_a", room)
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"group_a": (idx, direct)},
        sky={"group_a": (idx, open_lux, closed_lux)},
    )
    spaces = [_space(room)]

    # Small room (< 200 sqft): LM-83's own area band allows only 1 lit
    # point, so all 4 points lit forces the sole group shut every bright
    # hour. Large room (500-1000 sqft): the band allows 5, so 4 lit points
    # never forces it shut.
    small = _prepared(tmp_path / "small", [group], [_grid(room, n_points, room_area_m2=5.0)])
    large = _prepared(tmp_path / "large", [group], [_grid(room, n_points, room_area_m2=60.0)])

    result_small = run_option1(None, project, small, spaces, rules, "base_case", engine=engine)
    result_large = run_option1(None, project, large, spaces, rules, "base_case", engine=engine)

    assert result_small.sda_area_pct < result_large.sda_area_pct
    assert result_small.ase_area_pct == pytest.approx(result_large.ase_area_pct)


def test_both_aggregations_are_reported(tmp_path, write_example):
    project = _project(tmp_path, write_example)
    rules = _rules()
    sda_lux = rules["option1"]["sda"]["lux"]

    room_a, room_b = "Room North", "Room South"
    grid_a = _grid(room_a, 2, room_area_m2=10.0)
    grid_b = _grid(room_b, 8, room_area_m2=10.0)
    group_a = _group("group_a", room_a)
    group_b = _group("group_b", room_b)
    prepared = _prepared(tmp_path, [group_a, group_b], [grid_a, grid_b])
    spaces = [_space(room_a), _space(room_b)]

    idx_a, idx_b = (room_a,) * 2, (room_b,) * 8
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={
            "group_a": (idx_a, np.zeros((2, N_HOURS))),
            "group_b": (idx_b, np.zeros((8, N_HOURS))),
        },
        sky={
            "group_a": (idx_a, np.full((2, N_HOURS), sda_lux + 100.0), None),
            "group_b": (idx_b, np.full((8, N_HOURS), sda_lux - 100.0), None),
        },
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    # Equal areas (10 and 10): area-weighted = 50 %.
    # Very different point counts (2 and 8): point-weighted = 20 %.
    assert result.sda_area_pct != result.sda_point_pct
    assert result.sda_area_pct == pytest.approx(50.0)
    assert result.sda_point_pct == pytest.approx(20.0)


def test_the_headline_follows_the_configuration(tmp_path, write_example):
    project = _project(tmp_path, write_example, aggregation="points")
    rules = _rules()
    sda_lux = rules["option1"]["sda"]["lux"]
    room = "Room North"
    grid = _grid(room, 2, room_area_m2=20.0)
    group = _group("group_a", room)
    prepared = _prepared(tmp_path, [group], [grid])
    spaces = [_space(room)]
    idx = (room,) * 2
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"group_a": (idx, np.zeros((2, N_HOURS)))},
        sky={"group_a": (idx, np.full((2, N_HOURS), sda_lux + 100.0), None)},
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    assert result.headline == "points"
    assert result.sda_area_pct == pytest.approx(100.0)
    assert result.sda_point_pct == pytest.approx(100.0)
    assert any("floor-area" in w for w in result.warnings)
    assert result.score.points is None
    assert "floor-area" in result.score.reason


def test_hours_other_than_8_to_18_are_not_scored(tmp_path, write_example):
    project = _project(tmp_path, write_example, hours="09:00-17:00")
    rules = _rules()
    sda_lux = rules["option1"]["sda"]["lux"]
    room = "Room North"
    grid = _grid(room, 2, room_area_m2=20.0)
    group = _group("group_a", room)
    prepared = _prepared(tmp_path, [group], [grid])
    idx = (room,) * 2
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"group_a": (idx, np.zeros((2, 2920)))},
        sky={"group_a": (idx, np.full((2, 2920), sda_lux + 100.0), None)},
    )

    result = run_option1(None, project, prepared, [_space(room)], rules, "base_case", engine=engine)

    assert result.sda_area_pct == pytest.approx(100.0)
    assert result.score.points is None
    assert any("09:00-17:00" in w and "08:00-18:00" in w for w in result.warnings)


def test_the_floor_area_is_the_room_s_not_the_grid_s(tmp_path, write_example):
    project = _project(tmp_path, write_example)
    rules = _rules()
    sda_lux = rules["option1"]["sda"]["lux"]

    room_a, room_b = "Room North", "Room South"
    # Room A: large room area (90 m2), a tiny grid footprint (1 point).
    grid_a = _grid(room_a, 1, room_area_m2=90.0, spacing_m=1.0, cell_area_m2=1.0)
    # Room B: small room area (10 m2), a larger grid footprint (1 point).
    grid_b = _grid(room_b, 1, room_area_m2=10.0, spacing_m=3.0, cell_area_m2=9.0)
    group_a = _group("group_a", room_a)
    group_b = _group("group_b", room_b)
    prepared = _prepared(tmp_path, [group_a, group_b], [grid_a, grid_b])
    spaces = [_space(room_a), _space(room_b)]

    idx_a, idx_b = (room_a,) * 1, (room_b,) * 1
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={
            "group_a": (idx_a, np.zeros((1, N_HOURS))),
            "group_b": (idx_b, np.zeros((1, N_HOURS))),
        },
        sky={
            "group_a": (idx_a, np.full((1, N_HOURS), sda_lux + 100.0), None),  # passes
            "group_b": (idx_b, np.full((1, N_HOURS), sda_lux - 100.0), None),  # fails
        },
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    # If the GRID's own footprint (1 and 9) were used instead of the room
    # area (90 and 10), the weighted result would be 10 %, not 90 %.
    assert result.sda_area_pct == pytest.approx(90.0)


def test_the_thresholds_come_from_the_rules(tmp_path, write_example):
    project = _project(tmp_path, write_example)
    room = "Room North"
    grid = _grid(room, 2, room_area_m2=20.0)
    group = _group("group_a", room)
    prepared = _prepared(tmp_path, [group], [grid])
    spaces = [_space(room)]
    idx = (room,) * 2
    lux_value = 400.0
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"group_a": (idx, np.zeros((2, N_HOURS)))},
        sky={"group_a": (idx, np.full((2, N_HOURS), lux_value), None)},
    )

    rules_default = _rules()  # sda.lux 300: 400 >= 300, passes
    rules_high = _rules()
    rules_high["option1"]["sda"]["lux"] = 500.0  # 400 < 500, fails

    result_default = run_option1(
        None, project, prepared, spaces, rules_default, "base_case", engine=engine,
    )
    result_high = run_option1(
        None, project, prepared, spaces, rules_high, "base_case", engine=engine,
    )

    assert result_default.sda_area_pct != result_high.sda_area_pct
    assert result_default.sda_area_pct == pytest.approx(100.0)
    assert result_high.sda_area_pct == pytest.approx(0.0)


def test_the_per_space_narrative_trigger_is_applied(tmp_path, write_example):
    project = _project(tmp_path, write_example)
    rules = _rules()
    ase_lux = rules["option1"]["ase"]["lux"]
    ase_hour_limit = rules["option1"]["ase"]["hours"]

    # Straddling the credit's default 10 % per-space trigger, on either
    # side, with small round point counts (spec defect S1: this is a
    # PER-SPACE check).
    room_high, room_low = "Room High", "Room Low"
    n = 20
    over_high, over_low = 3, 1  # 15 % and 5 %

    def _direct_for(n_over):
        arr = np.zeros((n, N_HOURS))
        bright_hours = ase_hour_limit + 50
        arr[:n_over, :bright_hours] = ase_lux * 2.0
        return arr

    grid_high = _grid(room_high, n, room_area_m2=1000.0)
    grid_low = _grid(room_low, n, room_area_m2=1000.0)
    group_high = _group("group_high", room_high)
    group_low = _group("group_low", room_low)
    prepared = _prepared(tmp_path, [group_high, group_low], [grid_high, grid_low])
    spaces = [_space(room_high), _space(room_low)]

    idx_high, idx_low = (room_high,) * n, (room_low,) * n
    filler = np.full((n, N_HOURS), 1.0)  # sDA value irrelevant to this test
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={
            "group_high": (idx_high, _direct_for(over_high)),
            "group_low": (idx_low, _direct_for(over_low)),
        },
        sky={
            "group_high": (idx_high, filler, filler.copy()),
            "group_low": (idx_low, filler.copy(), filler.copy()),
        },
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    assert result.narrative_spaces == (room_high,)


def test_the_ase_band_and_the_narrative_trigger_are_independent(tmp_path, write_example):
    project = _project(tmp_path, write_example)
    rules = _rules()
    ase_lux = rules["option1"]["ase"]["lux"]
    ase_hour_limit = rules["option1"]["ase"]["hours"]

    room = "Room North"
    n = 10
    over_count = 1  # exactly 10 % of 10 points

    direct = np.zeros((n, N_HOURS))
    bright_hours = ase_hour_limit + 50
    direct[:over_count, :bright_hours] = ase_lux * 2.0

    grid = _grid(room, n, room_area_m2=1000.0)
    group = _group("group_a", room)
    prepared = _prepared(tmp_path, [group], [grid])
    spaces = [_space(room)]
    idx = (room,) * n
    filler = np.full((n, N_HOURS), 1.0)
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"group_a": (idx, direct)},
        sky={"group_a": (idx, filler, filler.copy())},
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    assert result.spaces[0].ase_pct == pytest.approx(10.0)
    assert result.spaces[0].ase_band == "7-10 %: not classified by LM-83-12"
    assert result.spaces[0].needs_glare_narrative is False


def test_the_score_comes_from_score_option1(tmp_path, write_example):
    project = _project(tmp_path, write_example)  # default aggregation: floor_area
    rules = _rules()
    sda_lux = rules["option1"]["sda"]["lux"]

    room = "Room North"
    n = 100
    n_pass = 60  # exactly 60 % of points pass

    open_lux = np.zeros((n, N_HOURS))
    open_lux[:n_pass, :] = sda_lux + 100.0
    open_lux[n_pass:, :] = 0.0

    grid = _grid(room, n, room_area_m2=20.0)
    group = _group("group_a", room)
    prepared = _prepared(tmp_path, [group], [grid])
    spaces = [_space(room)]
    idx = (room,) * n
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"group_a": (idx, np.zeros((n, N_HOURS)))},
        sky={"group_a": (idx, open_lux, None)},
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    assert result.score.points == 3
    assert result.score.additional_point_awarded is True


def test_unverified_bands_still_report_percentages(tmp_path, write_example):
    project = _project(tmp_path, write_example)
    rules = _rules()
    rules["option1"]["sda"]["bands"] = UNVERIFIED

    room = "Room North"
    n = 10
    grid = _grid(room, n, room_area_m2=20.0)
    group = _group("group_a", room)
    prepared = _prepared(tmp_path, [group], [grid])
    spaces = [_space(room)]
    idx = (room,) * n
    open_lux = np.full((n, N_HOURS), rules["option1"]["sda"]["lux"] + 100.0)
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"group_a": (idx, np.zeros((n, N_HOURS)))},
        sky={"group_a": (idx, open_lux, None)},
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    assert result.sda_area_pct == pytest.approx(100.0)
    assert result.sda_point_pct == pytest.approx(100.0)
    assert result.ase_area_pct == pytest.approx(0.0)
    assert result.ase_point_pct == pytest.approx(0.0)
    assert result.score.points is None


def test_an_excluded_space_contributes_nothing(tmp_path, write_example):
    project = _project(tmp_path, write_example)
    rules = _rules()
    sda_lux = rules["option1"]["sda"]["lux"]

    room_in, room_out = "Room North", "Room South"
    grid_in = _grid(room_in, 2, room_area_m2=20.0)
    group_in = _group("group_in", room_in)
    # Excluded from the case: no grid for it, but still one of
    # prepared.groups -- exercises "a group whose room has no grid keeps
    # blinds open every hour" (ruling item 5) at the same time.
    group_out = _group("group_out", room_out)
    prepared = _prepared(tmp_path, [group_in, group_out], [grid_in])
    spaces = [
        _space(room_in, included_=True),
        _space(room_out, included_=False),
    ]

    idx_in, idx_out = (room_in,) * 2, (room_out,) * 3
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={
            "group_in": (idx_in, np.zeros((2, N_HOURS))),
            "group_out": (idx_out, np.zeros((3, N_HOURS))),
        },
        sky={
            "group_in": (idx_in, np.full((2, N_HOURS), sda_lux + 100.0), None),
            "group_out": (idx_out, np.full((3, N_HOURS), sda_lux + 100.0), None),
        },
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    assert [s.room for s in result.spaces] == [room_in]
    assert result.sda_area_pct == pytest.approx(100.0)
    assert "group_out" in result.blind_schedule
    assert bool(result.blind_schedule["group_out"].all())
    assert any("group_out" in w for w in result.warnings)


# =============================================================================
# hostile-input pass: refusals this driver defines itself
# =============================================================================


def test_a_case_mismatch_is_refused(tmp_path, write_example):
    project = _project(tmp_path, write_example)
    rules = _rules()
    room = "Room North"
    grid = _grid(room, 1)
    group = _group("group_a", room)
    prepared = _prepared(tmp_path, [group], [grid], case="base_case")
    spaces = [_space(room)]
    idx = (room,)
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"group_a": (idx, np.zeros((1, N_HOURS)))},
        sky={"group_a": (idx, np.zeros((1, N_HOURS)), None)},
    )

    with pytest.raises(Option1Error, match="other_case"):
        run_option1(None, project, prepared, spaces, rules, "other_case", engine=engine)


def test_a_grid_and_space_mismatch_is_refused(tmp_path, write_example):
    project = _project(tmp_path, write_example)
    rules = _rules()
    room_a, room_b = "Room North", "Room South"
    # prepared.grids has room_a; the only regularly-occupied included
    # space is room_b -- the two sets disagree.
    grid_a = _grid(room_a, 1)
    group_a = _group("group_a", room_a)
    prepared = _prepared(tmp_path, [group_a], [grid_a])
    spaces = [_space(room_b, included_=True)]
    idx = (room_a,)
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"group_a": (idx, np.zeros((1, N_HOURS)))},
        sky={"group_a": (idx, np.zeros((1, N_HOURS)), None)},
    )

    with pytest.raises(Option1Error, match=room_b):
        run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)


def test_a_missing_closed_lux_when_the_schedule_closes_is_refused(tmp_path, write_example):
    project = _project(tmp_path, write_example)
    rules = _rules()
    ase_lux = rules["option1"]["ase"]["lux"]
    room = "Room North"
    n = 2
    direct = np.full((n, N_HOURS), ase_lux * 2.0)  # always lit
    # Small room (< 200 sqft): allows only 1 lit point, so 2 lit points
    # force the group shut every occupied hour.
    grid = _grid(room, n, room_area_m2=5.0)
    group = _group("group_a", room)
    prepared = _prepared(tmp_path, [group], [grid])
    spaces = [_space(room)]
    idx = (room,) * n
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"group_a": (idx, direct)},
        sky={"group_a": (idx, np.zeros((n, N_HOURS)), None)},  # closed_lux missing
    )

    with pytest.raises(Option1Error, match="group_a"):
        run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)


# =============================================================================
# fix round 1 (review findings, controller rulings P51 and P52)
# =============================================================================


def test_a_window_opening_at_midnight_under_daylight_saving_is_refused(tmp_path, write_example):
    """Ruling P51's case -- "00:00-24:00" under EU daylight saving, which
    lost an hour on every DST day -- is now refused by occupied_mask
    itself (a window opening before 03:00 under DST), by name, before any
    count is taken."""
    from leed_daylight.metrics.occupancy import OccupancyError

    project = _project(tmp_path, write_example, hours="00:00-24:00", daylight_saving="EU")
    engine = FakeEngine(FULL_MASK_LEN, direct={}, sky={})

    with pytest.raises(OccupancyError, match="03:00"):
        run_option1(
            None, project, _prepared(tmp_path, [], []), [], _rules(), "base_case",
            engine=engine,
        )


def test_a_window_ending_at_24_00_under_daylight_saving_runs(tmp_path, write_example):
    """08:00-24:00 under EU daylight saving: 24:00 clock time is 23:00
    standard time inside the daylight-saving period, which the mask
    represents -- 365 x 16 occupied hours, the last one standard 22:00
    in summer and 23:00 in winter."""
    captured = {}

    class CapturingEngine(FakeEngine):
        def direct(self, group, hours):
            captured["mask"] = hours
            return super().direct(group, hours)

    n_hours = 365 * 16
    project = _project(tmp_path, write_example, hours="08:00-24:00", daylight_saving="EU")
    room = "Room North"
    idx = (room,) * 2
    engine = CapturingEngine(
        FULL_MASK_LEN,
        direct={"group_a": (idx, np.zeros((2, n_hours)))},
        sky={"group_a": (idx, np.full((2, n_hours), 400.0), None)},
    )
    prepared = _prepared(tmp_path, [_group("group_a", room)], [_grid(room, 2)])

    result = run_option1(
        None, project, prepared, [_space(room)], _rules(), "base_case", engine=engine,
    )

    assert result.occupied_hours == n_hours
    assert result.spaces[0].sda_pct == 100.0
    mask = captured["mask"]
    july_1 = 181 * 24  # day index 181 of a non-leap year, daylight time
    jan_1 = 0
    assert mask[july_1 + 7] and mask[july_1 + 22] and not mask[july_1 + 23]
    assert not mask[jan_1 + 7] and mask[jan_1 + 8] and mask[jan_1 + 23]


def test_a_mask_whose_hour_count_disagrees_with_the_window_is_refused(
    tmp_path, write_example, monkeypatch,
):
    """Ruling P51's refusal stays: a mask one hour short of what
    analysis.hours says is refused with both counts named."""
    import leed_daylight.drivers.run_option1 as run_option1_module
    from leed_daylight.metrics.occupancy import occupied_mask as real_mask

    def short_mask(hours, daylight_saving, year):
        mask = real_mask(hours, daylight_saving, year)
        mask[np.flatnonzero(mask)[0]] = False
        return mask

    monkeypatch.setattr(run_option1_module, "occupied_mask", short_mask)
    project = _project(tmp_path, write_example, hours="08:00-18:00", daylight_saving="EU")
    engine = FakeEngine(FULL_MASK_LEN, direct={}, sky={})

    with pytest.raises(Option1Error) as exc_info:
        run_option1(
            None, project, _prepared(tmp_path, [], []), [], _rules(), "base_case",
            engine=engine,
        )

    message = str(exc_info.value)
    assert "3649" in message and "3650" in message and "mask is wrong" in message


def test_groups_disagreeing_on_a_room_s_point_count_is_refused(tmp_path, write_example):
    """Ruling P52: two groups owning the same room must see the same
    number of that room's own sensor points; a malformed fake engine that
    returns different counts must be refused by name, never surface a raw
    numpy error from ``np.stack``.
    """
    project = _project(tmp_path, write_example)
    rules = _rules()
    room = "Room North"
    grid = _grid(room, 4, room_area_m2=20.0)
    group_a = _group("group_a", room)
    group_b = _group("group_b", room)
    prepared = _prepared(tmp_path, [group_a, group_b], [grid])
    spaces = [_space(room)]
    # Malformed: group_a sees all 4 of the room's points, group_b sees
    # only 3 -- the two groups disagree on the room's own point count.
    idx_a = (room,) * 4
    idx_b = (room,) * 3
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={
            "group_a": (idx_a, np.zeros((4, N_HOURS))),
            "group_b": (idx_b, np.zeros((3, N_HOURS))),
        },
        sky={
            "group_a": (idx_a, np.zeros((4, N_HOURS)), None),
            "group_b": (idx_b, np.zeros((3, N_HOURS)), None),
        },
    )

    with pytest.raises(Option1Error) as exc_info:
        run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    message = str(exc_info.value)
    assert room in message
    assert "group_a" in message
    assert "group_b" in message


# =============================================================================
# final-fix M2: a malformed engine's returned shapes are a named refusal,
# never a raw IndexError/ValueError three lines later
# =============================================================================


def test_a_direct_row_count_mismatch_from_the_engine_is_refused(tmp_path, write_example):
    project = _project(tmp_path, write_example)
    rules = _rules()
    room = "Room North"
    grid = _grid(room, 2, room_area_m2=20.0)
    group = _group("group_a", room)
    prepared = _prepared(tmp_path, [group], [grid])
    spaces = [_space(room)]
    idx = (room,) * 2
    # Malformed: 3 sensor_index entries claimed but the lux array only has
    # 2 rows -- rows and sensor_index disagree.
    bad_idx = (room,) * 3
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"group_a": (bad_idx, np.zeros((2, N_HOURS)))},
        sky={"group_a": (idx, np.zeros((2, N_HOURS)), None)},
        # a well-formed residual: the residual pass runs first
        residual=[(idx, np.zeros((2, N_HOURS)), np.zeros((2, N_HOURS)))],
    )

    with pytest.raises(Option1Error, match="group_a"):
        run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)


def test_a_sky_column_count_mismatch_from_the_engine_is_refused(tmp_path, write_example):
    project = _project(tmp_path, write_example)
    rules = _rules()
    room = "Room North"
    grid = _grid(room, 2, room_area_m2=20.0)
    group = _group("group_a", room)
    prepared = _prepared(tmp_path, [group], [grid])
    spaces = [_space(room)]
    idx = (room,) * 2
    # Malformed: open_lux has the wrong number of hour columns -- not the
    # occupied-hour count run_option1 computed.
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"group_a": (idx, np.zeros((2, N_HOURS)))},
        sky={"group_a": (idx, np.zeros((2, N_HOURS - 1)), None)},
    )

    with pytest.raises(Option1Error, match="group_a"):
        run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)


# =============================================================================
# band edges: float error must never move a result across a band
# =============================================================================


def test_six_rooms_each_at_exactly_55_pct_score_three_points(tmp_path, write_example):
    """Six invented rooms, each with exactly 11 of 20 points passing (55 %).
    Re-weighting 0.55 by these areas in float lands one unit in the last place under 55, which
    used to score 1 point instead of 2 + the additional point."""
    project = _project(tmp_path, write_example)
    rules = _rules()
    sda_lux = rules["option1"]["sda"]["lux"]
    areas = [53.3, 66.7, 37.3, 24.6, 45.1, 73.1]
    n, n_pass = 20, 11

    groups, grids, spaces, direct, sky = [], [], [], {}, {}
    for i, area in enumerate(areas):
        room = f"Room {i}"
        groups.append(_group(f"group_{i}", room))
        grids.append(_grid(room, n, room_area_m2=area))
        spaces.append(_space(room, area_m2=area))
        idx = (room,) * n
        open_lux = np.zeros((n, N_HOURS))
        open_lux[:n_pass, :] = sda_lux + 100.0
        direct[f"group_{i}"] = (idx, np.zeros((n, N_HOURS)))
        sky[f"group_{i}"] = (idx, open_lux, None)
    prepared = _prepared(tmp_path, groups, grids)
    engine = FakeEngine(FULL_MASK_LEN, direct=direct, sky=sky)

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    assert result.sda_area_pct == 55.0
    assert result.score.points == 3
    assert result.score.additional_point_awarded is True


def test_a_room_over_10000_sf_is_named_in_a_warning(tmp_path, write_example):
    """LM-83-12 2.2.6: a blind analysis area never exceeds 10,000 sf
    (929.03 m2). The run keeps one schedule per room (spec ruling) but
    must say the result pools a room that LM-83 would split."""
    project = _project(tmp_path, write_example)
    rules = _rules()
    big, small = "Room Big", "Room Small"
    grids = [_grid(big, 2, room_area_m2=1000.0), _grid(small, 2, room_area_m2=900.0)]
    groups = [_group("group_big", big), _group("group_small", small)]
    prepared = _prepared(tmp_path, groups, grids)
    spaces = [_space(big), _space(small)]
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={
            "group_big": ((big,) * 2, np.zeros((2, N_HOURS))),
            "group_small": ((small,) * 2, np.zeros((2, N_HOURS))),
        },
        sky={
            "group_big": ((big,) * 2, np.full((2, N_HOURS), 400.0), None),
            "group_small": ((small,) * 2, np.full((2, N_HOURS), 400.0), None),
        },
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    over = [w for w in result.warnings if "10,000 sf" in w]
    assert len(over) == 1
    assert big in over[0] and "split" in over[0] and "pools" in over[0]


# =============================================================================
# the residual pass (7b): light through no window group. Each group's own
# isolated trace blacks out only the OTHER groups' openings, so it already
# carries the residual once -- the fakes below include it in every group's
# lux, as the real engine does -- and a room seen by n group traces holds n
# copies: the driver subtracts n - 1.
# =============================================================================


def _two_group_room(
    tmp_path, write_example, *, residual_sky, residual_direct, n=4,
    group_sky=100.0, n_groups=2,
):
    """One room, ``n_groups`` window groups, each group's own light
    (``group_sky``) too dim for sDA alone and with no direct sun of its
    own. Every group's fake lux INCLUDES the residual, as the real
    engine's isolated trace does; the residual unit's own lux is supplied
    by the test."""
    project = _project(tmp_path, write_example)
    rules = _rules()
    room = "Room North"
    grid = _grid(room, n, room_area_m2=20.0)
    groups = [_group(f"group_{i}", room) for i in range(n_groups)]
    prepared = _prepared(tmp_path, groups, [grid])
    idx = (room,) * n
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={g.group_id: (idx, np.full((n, N_HOURS), residual_direct)) for g in groups},
        sky={
            g.group_id: (idx, np.full((n, N_HOURS), group_sky + residual_sky), None)
            for g in groups
        },
        residual=[(idx, np.full((n, N_HOURS), residual_sky), np.full((n, N_HOURS), residual_direct))],
    )
    return project, rules, prepared, [_space(room)], engine, room


def test_the_residual_is_counted_once_not_once_per_group(tmp_path, write_example):
    # Group traces: 100 + 150 lux each (own light + residual), 600 lux
    # direct sun each (all of it residual). Residual: 150 sky, 600 direct.
    # Once: 100 + 100 + 150 = 350 lux passes sDA (300); 600 lux direct
    # stays under ASE (1000). The traces summed alone give 500 and 1200;
    # with the residual added on top, 650 and 1800.
    project, rules, prepared, spaces, engine, room = _two_group_room(
        tmp_path, write_example, residual_sky=150.0, residual_direct=600.0,
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    assert engine.residual_calls == 1
    assert result.spaces[0].sda_pct == 100.0
    assert result.spaces[0].ase_pct == 0.0
    # 150 / (100 + 100 + 150)
    assert result.spaces[0].residual_sky_pct == pytest.approx(100.0 * 150.0 / 350.0)


def test_a_double_counted_residual_does_not_pass_sda(tmp_path, write_example):
    # Group traces 100 + 60 lux each, residual 60: the true total is
    # 100 + 100 + 60 = 260 lux, under sDA's 300. Two copies of the
    # residual (320 lux) would wrongly pass.
    project, rules, prepared, spaces, engine, room = _two_group_room(
        tmp_path, write_example, residual_sky=60.0, residual_direct=0.0,
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    assert result.spaces[0].sda_pct == 0.0
    assert result.spaces[0].residual_sky_pct == pytest.approx(100.0 * 60.0 / 260.0)


def test_one_group_needs_no_residual_correction(tmp_path, write_example):
    # One group trace: 250 + 100 residual = 350 lux, already the total.
    # Adding the residual again (450) or taking it away (250, failing sDA)
    # would both be wrong.
    project, rules, prepared, spaces, engine, room = _two_group_room(
        tmp_path, write_example, residual_sky=100.0, residual_direct=0.0,
        group_sky=250.0, n_groups=1,
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    assert result.spaces[0].sda_pct == 100.0
    assert result.spaces[0].residual_sky_pct == pytest.approx(100.0 * 100.0 / 350.0)


def test_one_group_with_residual_direct_sun_keeps_its_blinds_for_its_own_sun(
    tmp_path, write_example,
):
    # The group's own direct lux (residual included) is 1200 at every
    # point; 1200 - 600 residual = 600 lux of its own, under the 1000 lux
    # trigger: the blinds stay open. ASE still sees the full 1200.
    project, rules, prepared, spaces, engine, room = _two_group_room(
        tmp_path, write_example, residual_sky=0.0, residual_direct=600.0,
        n_groups=1,
    )
    engine._direct["group_0"] = (engine._direct["group_0"][0], np.full((4, N_HOURS), 1200.0))

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    assert result.blind_schedule["group_0"].all()
    assert result.spaces[0].ase_pct == 100.0


def test_the_residual_correction_is_clamped_at_zero(tmp_path, write_example):
    # Monte Carlo noise: three group traces of 10 lux each and a residual
    # traced at 20 lux give 30 - 2 x 20 = -10 lux. It is clamped to 0, so
    # with an sDA threshold of 0 lux every point passes; unclamped, none
    # would.
    project, rules, prepared, spaces, engine, room = _two_group_room(
        tmp_path, write_example, residual_sky=20.0, residual_direct=0.0,
        group_sky=-10.0, n_groups=3,
    )
    rules["option1"]["sda"]["lux"] = 0.0

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    assert result.spaces[0].sda_pct == 100.0


def test_the_residual_reaches_ase(tmp_path, write_example):
    project, rules, prepared, spaces, engine, room = _two_group_room(
        tmp_path, write_example, residual_sky=0.0,
        residual_direct=rules_ase_lux() + 500.0,
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    assert result.spaces[0].ase_pct == 100.0
    # Blinds never close for light no blind controls.
    assert all(sched.all() for sched in result.blind_schedule.values())


def rules_ase_lux():
    return _rules()["option1"]["ase"]["lux"]


def test_a_large_residual_share_is_named_in_a_warning(tmp_path, write_example):
    project, rules, prepared, spaces, engine, room = _two_group_room(
        tmp_path, write_example, residual_sky=10.0, residual_direct=0.0,
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    (warning,) = [w for w in result.warnings if "outside every window group" in w]
    # 10 / 210 = 4.8 %
    assert room in warning
    assert "4.8 % of its sky illuminance" in warning
    assert "check for skylights or gaps" in warning


def test_a_tiny_residual_share_is_not_warned(tmp_path, write_example):
    project, rules, prepared, spaces, engine, room = _two_group_room(
        tmp_path, write_example, residual_sky=1.0, residual_direct=0.0,
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    # 1 / 201 is under 1 %
    assert not [w for w in result.warnings if "outside every window group" in w]


def test_residual_direct_sun_alone_is_warned(tmp_path, write_example):
    project, rules, prepared, spaces, engine, room = _two_group_room(
        tmp_path, write_example, residual_sky=0.0, residual_direct=rules_ase_lux(),
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    (warning,) = [w for w in result.warnings if "outside every window group" in w]
    assert room in warning and "direct sun" in warning


def test_a_room_the_residual_covers_twice_is_refused(tmp_path, write_example):
    project, rules, prepared, spaces, engine, room = _two_group_room(
        tmp_path, write_example, residual_sky=0.0, residual_direct=0.0,
    )
    unit = engine._residual[0]
    engine._residual = [unit, unit]

    with pytest.raises(Option1Error, match="more than one trace unit"):
        run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)


def test_a_room_the_residual_misses_is_refused(tmp_path, write_example):
    project, rules, prepared, spaces, engine, room = _two_group_room(
        tmp_path, write_example, residual_sky=0.0, residual_direct=0.0,
    )
    engine._residual = []

    with pytest.raises(Option1Error, match="did not cover"):
        run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)


def test_the_residual_kept_in_float32_changes_results_only_within_float32(
    tmp_path, write_example,
):
    # The residual is held for the whole run in float32 (memory); a value
    # float32 cannot represent exactly (100 / 3 lux) moves the share only
    # at float32 precision, and the sums stay float64.
    residual = 100.0 / 3.0
    project, rules, prepared, spaces, engine, room = _two_group_room(
        tmp_path, write_example, residual_sky=residual, residual_direct=0.0,
        group_sky=150.0,
    )

    result = run_option1(None, project, prepared, spaces, rules, "base_case", engine=engine)

    # 150 + 150 + 100/3 = 333.3 lux passes sDA's 300
    assert result.spaces[0].sda_pct == 100.0
    assert result.spaces[0].residual_sky_pct == pytest.approx(
        100.0 * residual / (300.0 + residual), rel=1e-6,
    )


def _capture_metric_inputs(monkeypatch):
    """Record the per-room sDA and ASE lux arrays the driver hands to the
    metrics, in grid order."""
    import leed_daylight.drivers.run_option1 as run_option1_module

    captured = {"sda": [], "ase": []}
    real_da = run_option1_module.daylight_autonomy
    real_ase = run_option1_module.ase_hours

    def da(lux, threshold):
        captured["sda"].append(np.array(lux))
        return real_da(lux, threshold)

    def ase(lux, threshold):
        captured["ase"].append(np.array(lux))
        return real_ase(lux, threshold)

    monkeypatch.setattr(run_option1_module, "daylight_autonomy", da)
    monkeypatch.setattr(run_option1_module, "ase_hours", ase)
    return captured


def test_closed_blinds_and_the_residual_give_the_hand_worked_lux(
    tmp_path, write_example, monkeypatch,
):
    # One room, two groups, residual 50 lux sky and 300 lux direct in
    # every trace. group_0's own sun is 1200 lux on even hours, so its
    # blinds close then; group_1 has none of its own and stays open.
    #   sky, group_0: open 200 + 50, closed 40 + 50; group_1: 100 + 50
    #   sDA even: 90 + 150 - 50 = 190;  odd: 250 + 150 - 50 = 350
    #   ASE even: 1500 + 300 - 300 = 1500;  odd: 300 + 300 - 300 = 300
    captured = _capture_metric_inputs(monkeypatch)
    project = _project(tmp_path, write_example)
    room, n = "Room North", 4
    idx = (room,) * n
    direct_0 = np.full((n, N_HOURS), 300.0)
    direct_0[:, ::2] += 1200.0
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={
            "group_0": (idx, direct_0),
            "group_1": (idx, np.full((n, N_HOURS), 300.0)),
        },
        sky={
            "group_0": (idx, np.full((n, N_HOURS), 250.0), np.full((n, N_HOURS), 90.0)),
            "group_1": (idx, np.full((n, N_HOURS), 150.0), None),
        },
        residual=[(idx, np.full((n, N_HOURS), 50.0), np.full((n, N_HOURS), 300.0))],
    )
    groups = [_group("group_0", room), _group("group_1", room)]
    prepared = _prepared(tmp_path, groups, [_grid(room, n)])

    result = run_option1(
        None, project, prepared, [_space(room)], _rules(), "base_case", engine=engine,
    )

    schedule_0 = result.blind_schedule["group_0"]
    assert not schedule_0[::2].any() and schedule_0[1::2].all()
    assert result.blind_schedule["group_1"].all()
    (sda,), (ase,) = captured["sda"], captured["ase"]
    np.testing.assert_allclose(sda[:, ::2], 190.0)
    np.testing.assert_allclose(sda[:, 1::2], 350.0)
    np.testing.assert_allclose(ase[:, ::2], 1500.0)
    np.testing.assert_allclose(ase[:, 1::2], 300.0)
    # 50 / (250 + 150 - 50)
    assert result.spaces[0].residual_sky_pct == pytest.approx(100.0 * 50.0 / 350.0)


def test_rooms_covered_by_one_and_by_three_traces_each_count_the_residual_once(
    tmp_path, write_example, monkeypatch,
):
    # Room A: group_a only (n = 1), whose trace also reaches room B.
    # Room B: group_a plus its own group_b1 and group_b2 (n = 3).
    # Residual 50 lux sky, 300 lux direct, in every trace.
    #   B's own groups each add 800 lux of sun on even hours: 1600 together
    #   crosses 1000, one closing (group_b1, first in order) is enough.
    #   A sDA: 200 (no correction);  ASE: 300
    #   B sDA even: 200 + 70 + 150 - 2 x 50 = 320;  odd: 200 + 150 + 150 - 100 = 400
    #   B ASE even: 300 + 1100 + 1100 - 600 = 1900;  odd: 300
    captured = _capture_metric_inputs(monkeypatch)
    project = _project(tmp_path, write_example)
    room_a, room_b, n = "Room A", "Room B", 6
    idx_ab, idx_b = (room_a,) * n + (room_b,) * n, (room_b,) * n
    direct_b = np.full((n, N_HOURS), 300.0)
    direct_b[:, ::2] += 800.0
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={
            "group_a": (idx_ab, np.full((2 * n, N_HOURS), 300.0)),
            "group_b1": (idx_b, direct_b),
            "group_b2": (idx_b, direct_b.copy()),
        },
        sky={
            "group_a": (idx_ab, np.full((2 * n, N_HOURS), 200.0), None),
            "group_b1": (idx_b, np.full((n, N_HOURS), 150.0), np.full((n, N_HOURS), 70.0)),
            "group_b2": (idx_b, np.full((n, N_HOURS), 150.0), np.full((n, N_HOURS), 70.0)),
        },
        residual=[(idx_ab, np.full((2 * n, N_HOURS), 50.0), np.full((2 * n, N_HOURS), 300.0))],
    )
    groups = [_group("group_a", room_a), _group("group_b1", room_b), _group("group_b2", room_b)]
    prepared = _prepared(tmp_path, groups, [_grid(room_a, n), _grid(room_b, n)])

    result = run_option1(
        None, project, prepared, [_space(room_a), _space(room_b)], _rules(), "base_case",
        engine=engine,
    )

    assert not result.blind_schedule["group_b1"][::2].any()
    assert result.blind_schedule["group_b1"][1::2].all()
    assert result.blind_schedule["group_b2"].all()
    assert result.blind_schedule["group_a"].all()
    (sda_a, sda_b), (ase_a, ase_b) = captured["sda"], captured["ase"]
    np.testing.assert_allclose(sda_a, 200.0)
    np.testing.assert_allclose(ase_a, 300.0)
    np.testing.assert_allclose(sda_b[:, ::2], 320.0)
    np.testing.assert_allclose(sda_b[:, 1::2], 400.0)
    np.testing.assert_allclose(ase_b[:, ::2], 1900.0)
    np.testing.assert_allclose(ase_b[:, 1::2], 300.0)
    by_room = {s.room: s for s in result.spaces}
    assert (by_room[room_a].sda_pct, by_room[room_b].sda_pct) == (0.0, 100.0)
    assert (by_room[room_a].ase_pct, by_room[room_b].ase_pct) == (0.0, 100.0)
    # the totals: equal rooms, one of each
    assert result.sda_point_pct == 50.0 and result.sda_area_pct == 50.0
    assert result.ase_point_pct == 50.0 and result.ase_area_pct == 50.0
    # A: 50 / 200;  B: 50 / (200 + 150 + 150 - 2 x 50)
    assert by_room[room_a].residual_sky_pct == pytest.approx(25.0)
    assert by_room[room_b].residual_sky_pct == pytest.approx(12.5)


# =============================================================================
# diffusing glazing (roof-glass design, section 5.9): a diffusing group is
# traced like any other, carries the residual like any other, but has no
# blind -- open every hour, never in LM-83's closing order, and no "no
# sensor grid" warning. Other roof-glass kinds keep today's behaviour.
# =============================================================================


def _kind_group(group_id, room, kind, *, storey=0):
    """A window group of ``kind``; skylight and diffusing groups have no
    facade azimuth and lie flat in these fakes."""
    flat = kind in ("skylight", "diffusing")
    return WindowGroup(
        group_id=group_id, room=room, storey=storey,
        azimuth_bin_deg=None if flat else 180.0,
        openings=(), outward_normals=(), area_m2=1.0,
        kind=kind, tilt_deg=0.0 if flat else 90.0,
    )


def _spy_blind_schedule(monkeypatch):
    """Record how many groups each call to metrics.blinds.blind_schedule
    is given; the real function still computes the schedule."""
    import leed_daylight.drivers.run_option1 as run_option1_module

    calls = []
    real = run_option1_module.blind_schedule

    def spy(direct_by_group, area_m2, lit_threshold):
        calls.append(int(np.asarray(direct_by_group).shape[0]))
        return real(direct_by_group, area_m2, lit_threshold)

    monkeypatch.setattr(run_option1_module, "blind_schedule", spy)
    return calls


class ScheduleRecordingEngine(FakeEngine):
    """FakeEngine that keeps the open schedule each group's sky call got."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.schedules = {}

    def sky(self, group, hours, open_schedule):
        self.schedules[group.group_id] = np.array(open_schedule, dtype=bool)
        return super().sky(group, hours, open_schedule)


def _empty_trace():
    """A trace whose sensor scope holds no point (a room with no grid)."""
    return ((), np.zeros((0, N_HOURS)))


def test_a_diffusing_group_counts_the_residual_once_and_has_no_blind(
    tmp_path, write_example, monkeypatch,
):
    # One room, 4 points, 20 m2 (215 sf: LM-83 allows 3 lit points).
    # R, the residual: 50 lux sky, 300 lux direct, in every trace.
    # Window group trace = W + R, W = 1200 lux of its own sun on even hours;
    #   sky open 200 + 50, closed 40 + 50.
    # Diffusing group trace = R (its own glass adds no direct sun);
    #   sky 150 + 50; no blind.
    #   blind trigger, window: (W + R) - R = 1200 on even hours, 4 lit > 3: closes
    #   ASE even: 1500 + 300 + (1 - 2) x 300 = 1500 = W + R;  odd: 300 = R
    #   sDA even: 90 + 200 - 50 = 240;  odd: 250 + 200 - 50 = 400
    captured = _capture_metric_inputs(monkeypatch)
    calls = _spy_blind_schedule(monkeypatch)
    project = _project(tmp_path, write_example)
    room, n = "Room North", 4
    idx = (room,) * n
    window_direct = np.full((n, N_HOURS), 300.0)
    window_direct[:, ::2] += 1200.0
    engine = ScheduleRecordingEngine(
        FULL_MASK_LEN,
        direct={
            "room_north_180": (idx, window_direct),
            "room_north_dif": (idx, np.full((n, N_HOURS), 300.0)),
        },
        sky={
            "room_north_180": (idx, np.full((n, N_HOURS), 250.0), np.full((n, N_HOURS), 90.0)),
            "room_north_dif": (idx, np.full((n, N_HOURS), 200.0), None),
        },
        residual=[(idx, np.full((n, N_HOURS), 50.0), np.full((n, N_HOURS), 300.0))],
    )
    groups = [_group("room_north_180", room), _kind_group("room_north_dif", room, "diffusing")]
    prepared = _prepared(tmp_path, groups, [_grid(room, n, room_area_m2=20.0)])

    result = run_option1(
        None, project, prepared, [_space(room)], _rules(), "base_case", engine=engine,
    )

    # only the window group enters LM-83's closing order
    assert calls == [1]
    window = result.blind_schedule["room_north_180"]
    assert not window[::2].any() and window[1::2].all()
    assert result.blind_schedule["room_north_dif"].all()
    # the engine is never asked for a blinds-closed trace of the diffuser
    assert engine.schedules["room_north_dif"].all()
    (sda,), (ase,) = captured["sda"], captured["ase"]
    np.testing.assert_allclose(ase[:, ::2], 1500.0)
    np.testing.assert_allclose(ase[:, 1::2], 300.0)
    np.testing.assert_allclose(sda[:, ::2], 240.0)
    np.testing.assert_allclose(sda[:, 1::2], 400.0)
    assert not any("no sensor grid" in w for w in result.warnings)


def test_a_room_whose_only_group_is_diffusing_is_open_every_hour_with_no_warning(
    tmp_path, write_example, monkeypatch,
):
    # One diffusing group, residual 50 lux sky and 300 lux direct; its
    # trace: sky 300 + 50, direct R only. n = 1: no correction.
    #   ASE = 300;  sDA = 350
    captured = _capture_metric_inputs(monkeypatch)
    calls = _spy_blind_schedule(monkeypatch)
    project = _project(tmp_path, write_example)
    room, n = "Room North", 4
    idx = (room,) * n
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"room_north_dif": (idx, np.full((n, N_HOURS), 300.0))},
        sky={"room_north_dif": (idx, np.full((n, N_HOURS), 350.0), None)},
        residual=[(idx, np.full((n, N_HOURS), 50.0), np.full((n, N_HOURS), 300.0))],
    )
    prepared = _prepared(
        tmp_path, [_kind_group("room_north_dif", room, "diffusing")], [_grid(room, n)],
    )

    result = run_option1(
        None, project, prepared, [_space(room)], _rules(), "base_case", engine=engine,
    )

    assert calls == []
    assert result.blind_schedule["room_north_dif"].all()
    assert not any("no sensor grid" in w for w in result.warnings)
    (sda,), (ase,) = captured["sda"], captured["ase"]
    np.testing.assert_allclose(ase, 300.0)
    np.testing.assert_allclose(sda, 350.0)


def test_a_diffusing_group_over_a_room_without_a_grid_is_open_with_no_warning(
    tmp_path, write_example,
):
    project = _project(tmp_path, write_example)
    room, n = "Room North", 2
    idx = (room,) * n
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"room_north_180": (idx, np.zeros((n, N_HOURS))), "room_store_dif": _empty_trace()},
        sky={
            "room_north_180": (idx, np.full((n, N_HOURS), 400.0), None),
            "room_store_dif": (*_empty_trace(), None),
        },
    )
    groups = [_group("room_north_180", room), _kind_group("room_store_dif", "Room Store", "diffusing")]
    prepared = _prepared(tmp_path, groups, [_grid(room, n)])

    result = run_option1(
        None, project, prepared, [_space(room)], _rules(), "base_case", engine=engine,
    )

    assert result.blind_schedule["room_store_dif"].all()
    assert not any("room_store_dif" in w for w in result.warnings)


def test_a_skylight_over_a_room_without_a_grid_stays_open_with_the_usual_warning(
    tmp_path, write_example,
):
    # A skylight has a blind (design D1): over a room with no sensor grid
    # it is kept open with today's warning, exactly as a window group there.
    project = _project(tmp_path, write_example)
    room, n = "Room North", 2
    idx = (room,) * n
    engine = FakeEngine(
        FULL_MASK_LEN,
        direct={"room_north_180": (idx, np.zeros((n, N_HOURS))), "room_store_sky": _empty_trace()},
        sky={
            "room_north_180": (idx, np.full((n, N_HOURS), 400.0), None),
            "room_store_sky": (*_empty_trace(), None),
        },
    )
    groups = [_group("room_north_180", room), _kind_group("room_store_sky", "Room Store", "skylight")]
    prepared = _prepared(tmp_path, groups, [_grid(room, n)])

    result = run_option1(
        None, project, prepared, [_space(room)], _rules(), "base_case", engine=engine,
    )

    assert result.blind_schedule["room_store_sky"].all()
    assert (
        "room_store_sky: room 'Room Store' has no sensor grid; its blinds are kept open every hour"
        in result.warnings
    )
