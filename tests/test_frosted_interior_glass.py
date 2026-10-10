# tests/test_frosted_interior_glass.py
"""Frosted interior glass: a diffusing ``trans`` with the role
``glazing_interior``.

Neither the pipeline spec nor the roof-glass spec gives a reason to refuse
it: interior glazing is "glass inside the building envelope" that "lets
light through" and gets no frame, reveal, blind or window group, and the
Quality Views rule already says only ``glass`` or ``dielectric`` can be a
clear view. So it is accepted under the diffusing checks (seven real
arguments, no transmitted specularity), written unchanged (clause 2.2.9's
dirt is an envelope rule), and it always blocks a view. Every value is
invented and round.
"""
from __future__ import annotations

import dataclasses

import pytest
from test_build_materials import GREY_DIFFUSER, make_glazing, trans
from test_build_prepare import ROLES, load_mixed_glazing, make_project, make_spaces

from leed_daylight.build.materials import MaterialError, rebuild_glazing
from leed_daylight.build.prepare import prepare
from leed_daylight.build.view_model import view_classes
from leed_daylight.project.config import ProjectError
from leed_daylight.project.materials import MaterialRow, load_materials

SEE_THROUGH = GREY_DIFFUSER[:6] + (0.5,)


def test_a_frosted_interior_pane_is_written_unchanged():
    prim = trans("frosted_a", GREY_DIFFUSER)

    rebuilt, records, warnings = rebuild_glazing([prim], {"frosted_a": "glazing_interior"}, make_glazing(0.8))

    assert rebuilt == [prim]
    [record] = records
    assert record.transmission == "diffuse"
    assert record.dirt_factor == 1.0
    # Td = c (1 - A4) A6 of the pane's own numbers; a grey's photopic colour c
    # is its channel value. Interior glass gets no dirt, so Td' = Td.
    r, g, b, a4, _, a6, _ = GREY_DIFFUSER
    assert r == g == b
    assert record.visible_transmittance == pytest.approx(r * (1.0 - a4) * a6)
    assert record.net_transmittance == record.visible_transmittance
    assert warnings == ()


def test_a_see_through_trans_is_refused_as_interior_glazing():
    with pytest.raises(MaterialError, match=r"frosted_a: trans transmitted specularity \(7th argument\) is 0\.5"):
        rebuild_glazing([trans("frosted_a", SEE_THROUGH)], {"frosted_a": "glazing_interior"}, make_glazing())


@pytest.mark.parametrize("role", ["glazing", "glazing_curtain_wall"])
def test_a_trans_on_an_exterior_window_role_is_still_refused(role):
    with pytest.raises(MaterialError, match=(
        r"frosted_a: a 'trans' glazing modifier needs the role 'glazing_diffusing' \(exterior\) "
        r"or 'glazing_interior' \(interior\)"
    )):
        rebuild_glazing([trans("frosted_a", GREY_DIFFUSER)], {"frosted_a": role}, make_glazing())


def test_a_frosted_partition_forms_no_group_and_stays_in_the_scene(tmp_path):
    prims, rooms = load_mixed_glazing()
    prims = [
        dataclasses.replace(p, ptype="trans", fargs=GREY_DIFFUSER) if p.name == "glass_partition" else p
        for p in prims
    ]
    out_dir = tmp_path / "prepared"

    result = prepare(make_project(out_dir), prims, ROLES, rooms, make_spaces(), "base_case", out_dir)

    assert len(result.manifest["groups"]) == 1
    assert all("window_int" not in g["openings"] for g in result.manifest["groups"].values())
    assert "glass_partition polygon window_int" in result.scene_file.read_text()
    assert "void trans glass_partition" in result.materials_file.read_text()
    assert result.manifest["glazing"]["glass_partition"]["dirt_factor"] == 1.0


def _row(modifier, ptype, role, view=""):
    return MaterialRow(modifier=modifier, primitive_type=ptype, role=role, note="", view=view)


def test_a_frosted_interior_pane_always_blocks_views():
    rows = [_row("frosted_a", "trans", "glazing_interior"), _row("clear_a", "glass", "glazing_interior")]

    classes = view_classes(rows, {"frosted_a": 0.8, "clear_a": 0.8}, vlt_min_exclusive=0.40)

    assert classes == {"frosted_a": "blocks", "clear_a": "clear"}


def _materials(tmp_path, view):
    path = tmp_path / "materials.csv"
    path.write_text(f"modifier,primitive_type,role,note,view\nfrosted_a,trans,glazing_interior,,{view}\n",
                    encoding="utf-8", newline="\n")
    return path


def test_materials_csv_refuses_a_clear_view_on_frosted_interior_glass(tmp_path):
    with pytest.raises(ProjectError, match=(
        r"materials\.csv line 2: modifier 'frosted_a': a 'trans' with role 'glazing_interior' is "
        r"frosted glass, never vision glazing, so its view must be empty or 'blocks', got 'clear'"
    )):
        load_materials(_materials(tmp_path, "clear"))


def test_materials_csv_accepts_blocks_on_frosted_interior_glass(tmp_path):
    assert load_materials(_materials(tmp_path, "blocks")) == {"frosted_a": "glazing_interior"}


def test_materials_csv_accepts_an_empty_view_on_frosted_interior_glass(tmp_path):
    assert load_materials(_materials(tmp_path, "")) == {"frosted_a": "glazing_interior"}


def test_view_classes_refuses_a_clear_view_on_frosted_interior_glass():
    from leed_daylight.build.view_model import ViewModelError

    rows = [_row("frosted_a", "trans", "glazing_interior", view="clear")]

    with pytest.raises(ViewModelError, match=r"frosted_a: a 'trans' with role 'glazing_interior' is frosted glass"):
        view_classes(rows, {"frosted_a": 0.8}, vlt_min_exclusive=0.40)
