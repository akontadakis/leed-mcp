# tests/test_run_option2.py
"""Tests for drivers.run_option2 with a FAKE engine: no Radiance binary runs."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from leed_daylight.drivers.run_option2 import Option2Error, run_option2
from option2_folder import ROOM, SCENE_RAD, build_option2_project, load_chain, sunny_wea_lines

LABELS = ["03-28_0900", "03-28_1500", "09-14_0900", "09-14_1500"]


class FakeEngine:
    """Canned lux per sky label, the same value at every point."""

    def __init__(self, n_points, lux_by_label, short_by=0):
        self.n_points = n_points
        self.lux_by_label = lux_by_label
        self.short_by = short_by
        self.moments = []

    def illuminance(self, moment):
        self.moments.append(moment)
        return np.full(self.n_points - self.short_by, float(self.lux_by_label[moment.label]))


def _run(tmp_path, write_example, write_wea, lux_by_label, *, short_by=0, **build):
    folder = build_option2_project(tmp_path, write_example, write_wea, **build)
    project, prepared, spaces, rules, weather = load_chain(folder)
    n = sum(int(g.points.shape[0]) for g in prepared.grids)
    engine = FakeEngine(n, lux_by_label, short_by=short_by)
    result = run_option2(
        None, project, prepared, spaces, rules, "base_case", weather=weather, engine=engine,
    )
    return result, engine, n


def test_the_two_clear_days_and_four_skies_are_used(tmp_path, write_example, write_wea):
    result, engine, _ = _run(tmp_path, write_example, write_wea, dict.fromkeys(LABELS, 500.0))
    assert [m.label for m in engine.moments] == LABELS
    assert [(d.chosen.month, d.chosen.day) for d in result.days] == [(3, 28), (9, 14)]
    first = engine.moments[0]
    assert (first.month, first.day, first.hour) == (3, 28, 9)
    assert (first.dni, first.dhi) == (700.0, 100.0)
    assert first.altitude_deg == pytest.approx(29.5, abs=0.06)


def test_the_two_days_are_averaged_before_the_band(tmp_path, write_example, write_wea):
    # 09:00: March 200, September 600 -> 400 passes; 15:00: 500 and 500.
    lux = {"03-28_0900": 200.0, "09-14_0900": 600.0, "03-28_1500": 500.0, "09-14_1500": 500.0}
    result, _, n = _run(tmp_path, write_example, write_wea, lux)
    (space,) = result.spaces
    assert space.room == ROOM
    assert (space.passing, space.points, space.pass_pct) == (n, n, 100.0)
    assert result.area_pct == pytest.approx(100.0)
    assert result.score.points == 3


def test_too_bright_points_fail_high(tmp_path, write_example, write_wea):
    lux = {"03-28_0900": 500.0, "09-14_0900": 500.0, "03-28_1500": 4000.0, "09-14_1500": 4000.0}
    result, _, n = _run(tmp_path, write_example, write_wea, lux)
    (space,) = result.spaces
    assert (space.passing, space.n_low, space.n_high) == (0, 0, n)
    assert space.upper_limit_applied is True
    assert result.score.points == 0


def test_view_preserving_shades_remove_the_upper_limit(tmp_path, write_example, write_wea):
    spaces_csv = (
        "floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,"
        "view_preserving_shades,merge_into,include_base_case,reason\n"
        "L0,Room North,office,yes,24,3.0,2.0,0.0,yes,,1,\n"
    )
    lux = {"03-28_0900": 500.0, "09-14_0900": 500.0, "03-28_1500": 4000.0, "09-14_1500": 4000.0}
    result, _, n = _run(tmp_path, write_example, write_wea, lux, spaces_csv=spaces_csv)
    (space,) = result.spaces
    assert (space.passing, space.n_high) == (n, 0)
    assert space.upper_limit_applied is False


def test_every_point_has_a_row(tmp_path, write_example, write_wea):
    result, _, n = _run(tmp_path, write_example, write_wea, dict.fromkeys(LABELS, 500.0))
    assert len(result.point_rows) == n
    row = result.point_rows[0]
    assert row["room"] == ROOM
    assert row["lux"] == dict.fromkeys(LABELS, 500.0)
    assert row["average"] == {"09:00": 500.0, "15:00": 500.0}
    assert row["passes"] is True


def test_an_unready_project_is_refused_by_name(tmp_path, write_example, write_wea):
    def unverified(d):
        d["option2"]["time_basis"] = "UNVERIFIED"

    with pytest.raises(Option2Error, match="option2.time_basis"):
        _run(tmp_path, write_example, write_wea, {}, mutate=unverified)


def test_a_sun_below_the_horizon_is_refused(tmp_path, write_example, write_wea):
    def early(d):
        d["option2"]["times"] = ["05:00", "15:00"]

    with pytest.raises(Option2Error, match="below the horizon"):
        _run(tmp_path, write_example, write_wea, {}, mutate=early)


def test_zero_diffuse_light_at_a_checked_hour_is_refused(tmp_path, write_example, write_wea):
    lines = sunny_wea_lines(dhi_hours=range(10, 18))  # no diffuse light at 08:30 and 09:30
    with pytest.raises(Option2Error, match="09:00"):
        _run(tmp_path, write_example, write_wea, {}, wea_lines=lines)


def test_a_weather_file_with_no_clear_day_is_refused(tmp_path, write_example, write_wea):
    lines = sunny_wea_lines(clear_days={})
    with pytest.raises(Option2Error, match="direct sun"):
        _run(tmp_path, write_example, write_wea, {}, wea_lines=lines)


def test_a_weather_file_in_another_time_zone_is_refused(tmp_path, write_example, write_wea):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    write_wea(
        folder / "weather" / "example.wea", latitude=42.0, longitude_W=80.0,
        time_zone=60.0, hours=sunny_wea_lines(),
    )
    project, prepared, spaces, rules, weather = load_chain(folder)
    with pytest.raises(Option2Error, match="time_zone"):
        run_option2(None, project, prepared, spaces, rules, "base_case", weather=weather,
                    engine=FakeEngine(1, {}))


def test_an_engine_answer_of_the_wrong_length_is_refused(tmp_path, write_example, write_wea):
    with pytest.raises(Option2Error, match="03-28_0900"):
        _run(tmp_path, write_example, write_wea, dict.fromkeys(LABELS, 500.0), short_by=1)


def test_the_settings_record_the_time_basis_and_the_sky(tmp_path, write_example, write_wea):
    result, _, _ = _run(tmp_path, write_example, write_wea, dict.fromkeys(LABELS, 500.0))
    assert result.settings["time_basis"] == "weather_file_local"
    assert result.settings["sky"] == "weather_file"
    assert result.settings["times"] == ["09:00", "15:00"]


# A second invented room, 4 m x 4 m x 3 m, standing apart from the first.
SECOND_ROOM = "Room South"
TWO_ROOM_SCENE = SCENE_RAD + """
floor_carpet polygon floor_room_south
0
0
12  10 0 0   14 0 0   14 4 0   10 4 0

ceiling_tile polygon ceiling_room_south
0
0
12  10 0 3   10 4 3   14 4 3   14 0 3
"""
TWO_ROOM_SPACES = (
    "floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,"
    "view_preserving_shades,merge_into,include_base_case,reason\n"
    "L0,Room North,office,yes,24,3.0,2.0,0.0,no,,1,\n"
    "L0,Room South,office,yes,16,12.0,2.0,0.0,no,,1,\n"
)


class PerPointEngine:
    """Canned lux per grid point, the same at every sky."""

    def __init__(self, lux):
        self.lux = lux

    def illuminance(self, moment):
        return self.lux.copy()


def test_two_rooms_one_passing_and_one_failing_low(tmp_path, write_example, write_wea):
    folder = build_option2_project(
        tmp_path, write_example, write_wea, scene_rad=TWO_ROOM_SCENE, spaces_csv=TWO_ROOM_SPACES,
    )
    project, prepared, spaces, rules, weather = load_chain(folder)
    assert [g.room for g in prepared.grids] == [ROOM, SECOND_ROOM]
    sizes = [int(g.points.shape[0]) for g in prepared.grids]
    lux = np.concatenate([np.full(sizes[0], 500.0), np.full(sizes[1], 100.0)])
    result = run_option2(
        None, project, prepared, spaces, rules, "base_case", weather=weather,
        engine=PerPointEngine(lux),
    )

    first, second = result.spaces
    assert (first.room, first.points, first.passing, first.n_low, first.n_high) == (
        ROOM, sizes[0], sizes[0], 0, 0)
    assert (second.room, second.points, second.passing, second.n_low, second.n_high) == (
        SECOND_ROOM, sizes[1], 0, sizes[1], 0)
    assert second.pass_pct == 0.0
    assert result.area_pct == pytest.approx(100.0 * 24.0 / 40.0)

    rows = result.point_rows
    assert len(rows) == sum(sizes)
    assert {r["room"] for r in rows[:sizes[0]]} == {ROOM}
    tail = rows[sizes[0]:]
    assert {r["room"] for r in tail} == {SECOND_ROOM}
    assert all(r["lux"] == dict.fromkeys(LABELS, 100.0) for r in tail)
    assert all(r["passes"] is False for r in tail)
    second_grid = prepared.grids[1].points
    assert (tail[0]["x"], tail[0]["y"]) == (float(second_grid[0][0]), float(second_grid[0][1]))
    assert tail[0]["x"] > 10.0


def _run_with(tmp_path, write_example, write_wea, *, option2=None, rules_option2=None):
    folder = build_option2_project(tmp_path, write_example, write_wea)
    project, prepared, spaces, rules, weather = load_chain(folder)
    if option2:
        project = dataclasses.replace(project, option2=dataclasses.replace(project.option2, **option2))
    rules["option2"].update(rules_option2 or {})
    n = sum(int(g.points.shape[0]) for g in prepared.grids)
    engine = FakeEngine(n, {**dict.fromkeys(LABELS, 500.0), "03-28_1000": 500.0, "09-14_1000": 500.0})
    return run_option2(
        None, project, prepared, spaces, rules, "base_case", weather=weather, engine=engine,
    )


def test_a_project_band_that_differs_from_the_rules_band_is_not_scored(tmp_path, write_example, write_wea):
    result = _run_with(
        tmp_path, write_example, write_wea,
        rules_option2={"lux_min": 300, "lux_max": 3000}, option2={"lux_min": 200.0},
    )
    assert result.area_pct == pytest.approx(100.0)
    assert result.score.points is None
    assert any("200 to 3000 lux" in w and "300 to 3000 lux" in w for w in result.warnings)


def test_the_rules_band_scores_normally(tmp_path, write_example, write_wea):
    result = _run_with(
        tmp_path, write_example, write_wea,
        rules_option2={"lux_min": 300, "lux_max": 3000},
    )
    assert result.score.points == 3


def test_times_other_than_9_and_15_are_not_scored(tmp_path, write_example, write_wea):
    result = _run_with(tmp_path, write_example, write_wea, option2={"times": ("10:00", "15:00")})
    assert result.score.points is None
    assert any("9:00 and 15:00" in w for w in result.warnings)


def test_the_summary_says_why_nothing_was_awarded(tmp_path, write_example, write_wea):
    import io
    from leed_daylight.drivers.option2 import _print_summary

    result = _run_with(tmp_path, write_example, write_wea, option2={"times": ("10:00", "15:00")})
    out = io.StringIO()
    _print_summary(result, tmp_path / "results.json", out=out)
    assert "score: not awarded: not scored:" in out.getvalue()
    assert "UNVERIFIED" not in out.getvalue()
