# tests/test_option2_config.py
"""Tests for project.checks.check_option2 (Option 2 design, sections 3 and 7).

Every project is the shared fictional example (tests/fixtures/
project_example.yaml) with one option2 key changed.
"""
from __future__ import annotations

import pytest

from leed_daylight.constants import UNVERIFIED
from leed_daylight.project.checks import (
    OPTION2_SKY,
    OPTION2_TIME_BASIS,
    check_option2,
    check_project,
    parse_mmdd,
    parse_whole_hour,
)
from leed_daylight.project.config import ProjectError, load_project


def _load(write_example, tmp_path, **changes):
    def mutate(d):
        d["option2"].update(changes)
        return d

    return load_project(write_example(tmp_path, mutate=mutate))


def test_the_accepted_values_are_the_ones_the_user_chose():
    assert OPTION2_TIME_BASIS == "weather_file_local"
    assert OPTION2_SKY == "weather_file"


def test_the_example_project_is_ready_for_option2(example_project_dir):
    project = load_project(example_project_dir / "project.yaml")
    assert check_option2(project) == ()


@pytest.mark.parametrize(
    "key, value, fragment",
    [
        ("time_basis", UNVERIFIED, "option2.time_basis"),
        ("time_basis", "clock", "option2.time_basis"),
        ("sky", "CIE clear with sun", "option2.sky"),
        ("blinds", "include", "option2.blinds"),
        ("clearest_day_window_days", 31, "option2.clearest_day_window_days"),
        ("clearest_day_window_days", -1, "option2.clearest_day_window_days"),
        ("equinox_dates", ["03-21"], "option2.equinox_dates"),
        ("equinox_dates", ["03-21", "02-30"], "option2.equinox_dates"),
        ("equinox_dates", ["3-21", "09-21"], "option2.equinox_dates"),
        ("equinox_dates", ["01-10", "09-21"], "past 1 January"),
        ("equinox_dates", ["03-21", "12-25"], "past 31 December"),
        ("times", ["09:30", "15:00"], "option2.times"),
        ("times", ["00:00", "15:00"], "option2.times"),
        ("times", ["09:00", "24:00"], "option2.times"),
        ("times", [], "option2.times"),
        ("times", ["09:00", "09:00"], "option2.times"),
        ("equinox_dates", ["03-21", "03-25"], "option2.equinox_dates"),
        ("equinox_dates", ["09-21", "03-21"], "option2.equinox_dates"),
    ],
)
def test_each_setting_the_tool_cannot_run_is_refused_by_name(
    write_example, tmp_path, key, value, fragment,
):
    try:
        project = _load(write_example, tmp_path, **{key: value})
    except ProjectError as exc:
        # load_project already refuses some shapes (e.g. an empty list);
        # that refusal must name the key too.
        assert key in str(exc)
        return
    errors = check_option2(project)
    assert any(fragment in e for e in errors), errors


def test_an_unverified_option2_section_still_loads_for_run_and_views(write_example, tmp_path):
    project = _load(
        write_example, tmp_path, time_basis=UNVERIFIED, sky="CIE clear with sun",
    )
    errors, warnings = check_project(project)
    assert errors == ()
    assert any("time_basis" in w for w in warnings)


def test_parse_mmdd_and_parse_whole_hour():
    assert parse_mmdd("03-21") == (3, 21)
    assert parse_whole_hour("09:00") == 9
    assert parse_whole_hour("15:00") == 15
    with pytest.raises(ValueError):
        parse_mmdd("02-29")
    with pytest.raises(ValueError):
        parse_whole_hour("09:30")
