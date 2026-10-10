# tests/test_project_spaces_views.py
"""Quality Views columns in spaces.csv (Quality Views spec 5.3)."""
from __future__ import annotations

import pytest

from leed_daylight.project.config import ProjectError
from leed_daylight.project.spaces import load_spaces, views_case_names, views_included

BASE = (
    "floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,"
    "view_preserving_shades,merge_into"
)


def _write(tmp_path, tail, *rows):
    path = tmp_path / "spaces.csv"
    path.write_text(BASE + tail + "\n" + "".join(r + "\n" for r in rows), encoding="utf-8")
    return path


def test_views_columns_are_read(tmp_path):
    path = _write(
        tmp_path, ",include_base,views_base,views_strict,reason,views_reason",
        "L0,Room North,office,yes,20,2.0,1.5,0.0,no,,1,1,1,,",
        "L0,Gym,gym,yes,50,10.0,5.0,0.0,no,,1,0,0,,gymnasium: may be excluded",
    )

    spaces = load_spaces(path)

    assert views_case_names(spaces) == ("base", "strict")
    assert [s.room for s in views_included(spaces, "base")] == ["Room North"]
    assert spaces[0].include == {"base": True}
    assert spaces[1].views == {"base": False, "strict": False}
    assert spaces[1].reason == ""
    assert spaces[1].views_reason == "gymnasium: may be excluded"


def test_a_file_without_views_columns_still_loads(tmp_path):
    path = _write(tmp_path, ",include_base,reason", "L0,Room North,office,yes,20,2.0,1.5,0.0,no,,1,kept")

    spaces = load_spaces(path)

    assert views_case_names(spaces) == ()
    assert spaces[0].views == {}
    assert spaces[0].views_reason == ""
    assert spaces[0].reason == "kept"
    with pytest.raises(ProjectError, match=r"no views_<case> column"):
        views_included(spaces, "base")


def test_an_unknown_views_case_is_refused(tmp_path):
    path = _write(tmp_path, ",include_base,views_base,reason", "L0,Room North,office,yes,20,2.0,1.5,0.0,no,,1,1,")

    with pytest.raises(ProjectError, match=r"unknown views case 'nope'.*base"):
        views_included(load_spaces(path), "nope")


def test_a_views_tick_on_a_space_not_regularly_occupied_is_refused(tmp_path):
    path = _write(tmp_path, ",include_base,views_base,reason", "L0,Store,storage,no,5,9.0,9.0,0.0,no,,0,1,")

    with pytest.raises(ProjectError, match=r"line 2.*Store.*views_base.*regularly"):
        load_spaces(path)


def test_views_columns_must_follow_the_include_columns(tmp_path):
    path = _write(tmp_path, ",views_base,include_base,reason", "L0,Room North,office,yes,20,2.0,1.5,0.0,no,,1,1,")

    with pytest.raises(ProjectError, match="columns must be exactly"):
        load_spaces(path)


def test_a_repeated_views_case_is_refused(tmp_path):
    path = _write(
        tmp_path, ",include_base,views_base,views_base,reason",
        "L0,Room North,office,yes,20,2.0,1.5,0.0,no,,1,1,1,",
    )

    with pytest.raises(ProjectError, match=r"views_base"):
        load_spaces(path)


def test_an_empty_views_case_name_is_refused(tmp_path):
    path = _write(tmp_path, ",include_base,views_,reason", "L0,Room North,office,yes,20,2.0,1.5,0.0,no,,1,1,")

    with pytest.raises(ProjectError, match=r"unexpected column 'views_'"):
        load_spaces(path)
