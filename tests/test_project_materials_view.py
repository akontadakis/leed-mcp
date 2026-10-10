# tests/test_project_materials_view.py
"""The optional fifth column of materials.csv, `view` (Quality Views spec 5.2)."""
from __future__ import annotations

import pytest

from leed_daylight.project.config import ProjectError
from leed_daylight.project.materials import (
    TAG_ROLES,
    TAG_VIEWS,
    VIEW_VALUES,
    load_materials,
    rows,
    view_overrides,
)

HEADER5 = "modifier,primitive_type,role,note,view\n"


def _write(tmp_path, body, header=HEADER5):
    path = tmp_path / "materials.csv"
    path.write_text(header + body, encoding="utf-8")
    return path


def test_the_vocabularies():
    assert VIEW_VALUES == {"clear", "blocks", "nature", "art", "landmark"}
    assert TAG_VIEWS == {"nature", "art", "landmark"}
    assert TAG_ROLES == {"context", "ground", "fixed_shading", "wall"}


def test_the_view_column_is_optional(tmp_path):
    path = _write(tmp_path, "glass_clear,glass,glazing,\n", header="modifier,primitive_type,role,note\n")

    assert view_overrides(path) == {}
    assert rows(path)[0].view == ""


def test_view_values_are_read(tmp_path):
    path = _write(tmp_path, (
        "glass_clear,glass,glazing,,\n"
        "screen,trans,fixed_shading,,clear\n"
        "trees,plastic,context,,nature\n"
        "mural,plastic,wall,,art\n"
        "tower,plastic,context,,landmark\n"
        "fritted,glass,glazing,,blocks\n"
    ))

    assert view_overrides(path) == {
        "screen": "clear", "trees": "nature", "mural": "art",
        "tower": "landmark", "fritted": "blocks",
    }
    assert load_materials(path)["screen"] == "fixed_shading"


def test_an_unknown_view_value_is_refused(tmp_path):
    path = _write(tmp_path, "wall_paint,plastic,wall,,see-through\n")

    with pytest.raises(ProjectError, match=r"line 2.*wall_paint.*'see-through'"):
        load_materials(path)


def test_a_tag_on_a_glazing_role_is_refused(tmp_path):
    path = _write(tmp_path, "glass_clear,glass,glazing,,nature\n")

    with pytest.raises(ProjectError, match=r"line 2.*glass_clear.*nature.*'glazing'"):
        load_materials(path)


def test_the_view_column_must_come_last(tmp_path):
    path = _write(tmp_path, "glass_clear,glass,glazing,,\n", header="modifier,primitive_type,role,view,note\n")

    with pytest.raises(ProjectError, match="columns must be exactly"):
        load_materials(path)


@pytest.mark.parametrize("view", ["", "blocks"])
def test_a_diffusing_material_may_leave_view_empty_or_block(tmp_path, view):
    path = _write(tmp_path, f"panel_diffuse,trans,glazing_diffusing,,{view}\n")

    assert rows(path)[0].view == view


@pytest.mark.parametrize("view", ["clear", "nature", "art", "landmark"])
def test_any_other_view_on_a_diffusing_material_is_refused(tmp_path, view):
    path = _write(tmp_path, f"panel_diffuse,trans,glazing_diffusing,,{view}\n")

    with pytest.raises(ProjectError, match=r"line 2.*panel_diffuse.*glazing_diffusing"):
        load_materials(path)
