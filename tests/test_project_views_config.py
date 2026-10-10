# tests/test_project_views_config.py
"""The optional `views:` section of project.yaml (Quality Views spec 5.1).

Every value is invented and round. The section is optional: a project that
never runs Quality Views has none, and loads exactly as before.
"""
from __future__ import annotations

import pytest

from leed_daylight.project.config import ProjectError, ViewsSpec, load_project


def _with_views(views):
    def mutate(d):
        d["views"] = views
        return d
    return mutate


def test_the_views_section_is_optional(tmp_path, write_example):
    project = load_project(write_example(tmp_path))

    assert project.views is None
    assert not any("views" in w for w in project.warnings)


def test_the_views_section_loads_with_its_two_defaults(tmp_path, write_example):
    path = write_example(
        tmp_path, _with_views({"eye_height_m": 1.1, "eye_height_source": "invented for a test"})
    )

    project = load_project(path)

    assert project.views == ViewsSpec(
        eye_height_m=1.1,
        eye_height_source="invented for a test",
        max_interior_glass=1,
        angular_step_deg=2.0,
    )


def test_every_views_setting_is_read(tmp_path, write_example):
    path = write_example(tmp_path, _with_views({
        "eye_height_m": 1.2,
        "eye_height_source": "invented",
        "max_interior_glass": 0,
        "angular_step_deg": 3.0,
    }))

    views = load_project(path).views

    assert (views.eye_height_m, views.max_interior_glass, views.angular_step_deg) == (1.2, 0, 3.0)


@pytest.mark.parametrize("views, match", [
    ({"eye_height_source": "x"}, r"views\.eye_height_m"),
    ({"eye_height_m": 1.1}, r"views\.eye_height_source"),
    ({"eye_height_m": "tall", "eye_height_source": "x"}, r"views\.eye_height_m must be a number"),
    ({"eye_height_m": 0.3, "eye_height_source": "x"}, r"views\.eye_height_m must be 0\.5 <= h <= 2\.0"),
    ({"eye_height_m": 2.5, "eye_height_source": "x"}, r"views\.eye_height_m must be 0\.5 <= h <= 2\.0"),
    ({"eye_height_m": 1.1, "eye_height_source": "   "}, r"views\.eye_height_source must say"),
    ({"eye_height_m": 1.1, "eye_height_source": "x", "max_interior_glass": 6},
     r"views\.max_interior_glass must be 0 <= n <= 5"),
    ({"eye_height_m": 1.1, "eye_height_source": "x", "max_interior_glass": 1.5},
     r"views\.max_interior_glass must be an integer"),
    ({"eye_height_m": 1.1, "eye_height_source": "x", "angular_step_deg": 0.25},
     r"views\.angular_step_deg must be 0\.5 <= a <= 5\.0"),
    ({"eye_height_m": 1.1, "eye_height_source": "x", "colour": "red"}, r"unknown key 'views\.colour'"),
    ("not a mapping", r"views must be a mapping"),
])
def test_a_bad_views_setting_is_refused(tmp_path, write_example, views, match):
    path = write_example(tmp_path, _with_views(views))

    with pytest.raises(ProjectError, match=match):
        load_project(path)
