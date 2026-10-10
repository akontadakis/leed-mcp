# tests/test_project_fill_me.py
"""``load_project`` refuses every value still marked FILL_ME (CLI spec 5).

``leed-daylight init`` writes a project.yaml whose every value is FILL_ME.
A number left as FILL_ME was already refused by its type check, but a text
value was not: ``north_rotation_source: "FILL_ME"`` would have reached a
report. The refusal lists every such value at once.
"""
from __future__ import annotations

import pytest

from leed_daylight.constants import FILL_ME
from leed_daylight.project.config import ProjectError, load_project


def test_fill_me_is_the_marker_text():
    assert FILL_ME == "FILL_ME"


def _write(tmp_path, write_example, mutate):
    return write_example(tmp_path, mutate)


def test_every_fill_me_value_is_named_in_one_refusal(tmp_path, write_example):
    def mutate(raw):
        raw["site"]["north_rotation_source"] = "FILL_ME"
        raw["site"]["latitude_N"] = "FILL_ME"
        return raw

    path = _write(tmp_path, write_example, mutate)
    with pytest.raises(ProjectError) as info:
        load_project(path)
    message = str(info.value)
    assert "2 value(s) still say FILL_ME" in message
    assert "site.latitude_N" in message
    assert "site.north_rotation_source" in message


def test_a_list_entry_is_named_with_its_index(tmp_path, write_example):
    def mutate(raw):
        raw["scene"]["files"] = ["FILL_ME"]
        return raw

    path = _write(tmp_path, write_example, mutate)
    with pytest.raises(ProjectError, match=r"scene\.files\[0\]"):
        load_project(path)


def test_spaces_around_the_marker_still_count(tmp_path, write_example):
    def mutate(raw):
        raw["project"]["name"] = "  FILL_ME "
        return raw

    path = _write(tmp_path, write_example, mutate)
    with pytest.raises(ProjectError, match=r"project\.name"):
        load_project(path)


def test_the_marker_inside_other_text_is_not_refused(tmp_path, write_example):
    def mutate(raw):
        raw["project"]["name"] = "Tower FILL_ME Annex"
        return raw

    path = _write(tmp_path, write_example, mutate)
    assert load_project(path).name == "Tower FILL_ME Annex"


def test_a_complete_project_loads_unchanged(example_project_dir):
    project = load_project(example_project_dir / "project.yaml")
    assert project.site.north_rotation_source.startswith("Fictional example")
