# tests/test_build_materials.py
"""Tests for leed_daylight.build.materials (T3.6).

Every numeric literal below is invented and round, per the confidentiality
rule (phase3-global.md #5): fictional glazing and paint values, never a
value the plan lists as client-derived.
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from leed_daylight.build.glazing_class import PaneClass, kind_of
from leed_daylight.build.materials import (
    MaterialError,
    apply_reflectances,
    copy_name,
    rebuild_glazing,
    rebuild_materials,
    rebuild_materials_by_class,
)
from leed_daylight.project.config import Glazing, Reflectance
from leed_daylight.scene.primitives import Primitive
from leed_daylight.units import LUX_WEIGHTS, transmissivity


def make_glazing(dirt_factor=1.0, vt_includes_dirt=False):
    # Everything but dirt_factor/vt_includes_dirt is irrelevant to this
    # module; the values here are invented round placeholders.
    return Glazing(
        dirt_factor=dirt_factor,
        frame_share=MappingProxyType({"default": 0.0}),
        frame_reflectance=0.0,
        frame_style="constant_width",
        reveal_depth_m=0.0,
        reveal_reflectance=0.0,
        vt_includes_dirt=vt_includes_dirt,
    )


def make_reflectance(policy, **by_role):
    return Reflectance(policy=policy, by_role=MappingProxyType(by_role))


def glass(name, r, g, b, extra=(), modifier="void"):
    return Primitive(modifier, "glass", name, (), (), (r, g, b) + tuple(extra), "test")


def plastic(name, r, g, b, spec=0.0, rough=0.0, modifier="void"):
    return Primitive(modifier, "plastic", name, (), (), (r, g, b, spec, rough), "test")


# -- rebuild_glazing ----------------------------------------------------


def test_the_dirt_factor_is_applied_once():
    # 0.75 and 0.86 are invented; 0.645 is their product.
    tn = transmissivity(0.75)
    prim = glass("glazing_a", tn, tn, tn)
    roles = {"glazing_a": "glazing"}
    _, records, warnings = rebuild_glazing(
        [prim], roles, make_glazing(dirt_factor=0.86, vt_includes_dirt=False)
    )
    assert warnings == ()
    assert len(records) == 1
    assert records[0].visible_transmittance == pytest.approx(0.75, abs=1e-9)
    assert records[0].net_transmittance == pytest.approx(0.645)


def test_vt_includes_dirt_does_not_apply_it_twice():
    tn = transmissivity(0.75)
    prim = glass("glazing_a", tn, tn, tn)
    roles = {"glazing_a": "glazing"}
    _, records, _ = rebuild_glazing(
        [prim], roles, make_glazing(dirt_factor=0.86, vt_includes_dirt=True)
    )
    assert records[0].net_transmittance == pytest.approx(0.75)


def test_the_rebuilt_glass_carries_transmissivity_not_transmittance():
    tn = transmissivity(0.75)
    prim = glass("glazing_a", tn, tn, tn)
    roles = {"glazing_a": "glazing"}
    _, records, _ = rebuild_glazing([prim], roles, make_glazing(dirt_factor=0.86))
    assert records[0].transmissivity > records[0].net_transmittance


def test_interior_glazing_gets_no_dirt_factor():
    tn = transmissivity(0.75)
    ext = glass("glazing_ext", tn, tn, tn)
    interior = glass("glazing_int", tn, tn, tn)
    roles = {"glazing_ext": "glazing", "glazing_int": "glazing_interior"}
    rebuilt, records, _ = rebuild_glazing([ext, interior], roles, make_glazing(dirt_factor=0.86))

    by_modifier = {r.modifier: r for r in records}
    assert by_modifier["glazing_ext"].net_transmittance == pytest.approx(0.645)
    assert by_modifier["glazing_int"].net_transmittance == pytest.approx(0.75)
    assert by_modifier["glazing_int"].dirt_factor == 1.0

    rebuilt_by_name = {p.name: p for p in rebuilt}
    assert rebuilt_by_name["glazing_int"].fargs == interior.fargs
    assert rebuilt_by_name["glazing_ext"].fargs != ext.fargs


def test_a_curtain_wall_role_also_gets_the_dirt_factor():
    tn = transmissivity(0.75)
    prim = glass("cw_glass", tn, tn, tn)
    roles = {"cw_glass": "glazing_curtain_wall"}
    _, records, _ = rebuild_glazing([prim], roles, make_glazing(dirt_factor=0.86))
    assert records[0].net_transmittance == pytest.approx(0.645)


def test_a_fourth_real_argument_survives_unchanged():
    # source: typical Radiance glass refractive index example
    tn = transmissivity(0.75)
    prim = glass("glazing_a", tn, tn, tn, extra=(1.52,))
    roles = {"glazing_a": "glazing"}
    rebuilt, _, _ = rebuild_glazing([prim], roles, make_glazing(dirt_factor=0.86))
    assert rebuilt[0].fargs[3] == pytest.approx(1.52)


@pytest.mark.parametrize("role", ["glazing", "glazing_interior"])
def test_a_refractive_index_other_than_the_standard_one_is_refused(role):
    tn = transmissivity(0.70)
    prim = glass("glazing_n", tn, tn, tn, extra=(1.60,))
    with pytest.raises(MaterialError, match=r"glazing_n.*refractive index 1\.6.*only 1\.52"):
        rebuild_glazing([prim], {"glazing_n": role}, make_glazing(dirt_factor=0.90))


def test_a_glass_modifier_is_never_read_as_a_reflectance():
    # A broken materials.csv assigns an opaque role to a glass modifier.
    tn = transmissivity(0.93)
    prim = glass("broken_glass", tn, tn, tn)
    roles = {"broken_glass": "wall"}
    with pytest.raises(MaterialError, match="broken_glass"):
        rebuild_glazing([prim], roles, make_glazing())


def test_a_dielectric_modifier_with_an_opaque_role_is_refused():
    prim = Primitive("void", "dielectric", "broken_dielectric", (), (), (0.9, 0.9, 0.9, 1.0, 0.0), "test")
    roles = {"broken_dielectric": "ceiling"}
    with pytest.raises(MaterialError, match="broken_dielectric"):
        rebuild_glazing([prim], roles, make_glazing())


def test_a_bsdf_glazing_is_refused_by_name():
    prim = Primitive("void", "BSDF", "smart_glass", ("data.xml",), (), (), "test")
    roles = {"smart_glass": "glazing"}
    with pytest.raises(MaterialError, match="smart_glass"):
        rebuild_glazing([prim], roles, make_glazing())


def test_an_abdsf_glazing_is_refused_by_name():
    prim = Primitive("void", "aBSDF", "smart_glass_2", ("data.xml",), (), (), "test")
    roles = {"smart_glass_2": "glazing"}
    with pytest.raises(MaterialError, match="smart_glass_2"):
        rebuild_glazing([prim], roles, make_glazing())


def test_a_non_glass_type_with_a_glazing_role_is_refused():
    prim = Primitive("void", "trans", "trans_glazing", (), (), (0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5), "test")
    roles = {"trans_glazing": "glazing"}
    with pytest.raises(MaterialError, match="trans_glazing"):
        rebuild_glazing([prim], roles, make_glazing())


def test_a_glass_primitive_with_the_wrong_real_count_is_refused():
    prim = Primitive("void", "glass", "malformed_glass", (), (), (0.9, 0.9), "test")
    roles = {"malformed_glass": "glazing"}
    with pytest.raises(MaterialError, match="malformed_glass"):
        rebuild_glazing([prim], roles, make_glazing())


@pytest.mark.parametrize("role", ["glazing", "glazing_interior"])
@pytest.mark.parametrize("channel", [0, 1, 2])
def test_a_negative_glass_channel_is_refused_even_when_the_average_is_positive(role, channel):
    # Invented values: two clear channels and one negative one; the
    # photopic-weighted average stays positive whichever channel is negative.
    reals = [0.8, 0.8, 0.8]
    reals[channel] = -0.1
    prim = glass("negative_channel_glass", *reals)
    roles = {"negative_channel_glass": role}
    with pytest.raises(MaterialError, match=r"negative_channel_glass.*channel"):
        rebuild_glazing([prim], roles, make_glazing())


@pytest.mark.parametrize("role", ["glazing", "glazing_interior"])
def test_a_glass_channel_above_the_physical_limit_is_refused(role):
    # transmissivity(1.0) is the largest a real glass can carry; an invented
    # 1.2 in one channel passes today because the average stays below it.
    prim = glass("too_clear_glass", 0.5, 1.2, 0.5)
    roles = {"too_clear_glass": role}
    with pytest.raises(MaterialError, match=r"too_clear_glass.*channel"):
        rebuild_glazing([prim], roles, make_glazing())


def test_tinted_glass_warns_but_loads():
    # Three round, deliberately unequal invented values.
    r, g, b = transmissivity(0.70), transmissivity(0.75), transmissivity(0.65)
    prim = glass("tinted_glass", r, g, b)
    roles = {"tinted_glass": "glazing"}
    rebuilt, records, warnings = rebuild_glazing([prim], roles, make_glazing(dirt_factor=1.0))

    assert len(warnings) == 1
    assert "tinted_glass" in warnings[0]
    assert records[0].tinted is True

    new = rebuilt[0]
    assert not (new.fargs[0] == new.fargs[1] == new.fargs[2])


def test_fixed_shading_and_ignore_roles_are_untouched():
    tn = transmissivity(0.75)
    shading = glass("shading_fin", tn, tn, tn)
    ignored = plastic("scrap", 0.10, 0.10, 0.10)
    roles = {"shading_fin": "fixed_shading", "scrap": "ignore"}
    rebuilt, records, warnings = rebuild_glazing([shading, ignored], roles, make_glazing(dirt_factor=0.5))
    assert records == []
    assert warnings == ()
    rebuilt_by_name = {p.name: p for p in rebuilt}
    assert rebuilt_by_name["shading_fin"].fargs == shading.fargs
    assert rebuilt_by_name["scrap"].fargs == ignored.fargs


def test_a_modifier_with_no_role_is_left_alone():
    # Referenced only by another material; materials.csv has no row for it.
    prim = plastic("helper_material", 0.25, 0.25, 0.25)
    rebuilt, records, warnings = rebuild_glazing([prim], {}, make_glazing())
    assert rebuilt == [prim]
    assert records == []
    assert warnings == ()


# -- apply_reflectances ---------------------------------------------------


def test_override_replaces_every_role_reflectance():
    ceiling_p = plastic("ceiling_paint", 0.10, 0.10, 0.10)
    wall_p = plastic("wall_paint", 0.10, 0.10, 0.10)
    floor_p = plastic("floor_paint", 0.10, 0.10, 0.10)
    roles = {"ceiling_paint": "ceiling", "wall_paint": "wall", "floor_paint": "floor"}
    # Invented set, distinct from the values the plan lists as client-derived.
    policy = make_reflectance("override", ceiling=0.75, wall=0.56, floor=0.34)

    rebuilt, warnings = apply_reflectances([ceiling_p, wall_p, floor_p], roles, policy)

    assert warnings == ()
    by_name = {p.name: p for p in rebuilt}
    assert by_name["ceiling_paint"].fargs[:3] == (0.75, 0.75, 0.75)
    assert by_name["wall_paint"].fargs[:3] == (0.56, 0.56, 0.56)
    assert by_name["floor_paint"].fargs[:3] == (0.34, 0.34, 0.34)
    assert by_name["ceiling_paint"].fargs[3] == 0.0
    assert by_name["ceiling_paint"].fargs[4] == 0.0
    # name, modifier and type kept
    assert by_name["ceiling_paint"].modifier == "void"
    assert by_name["ceiling_paint"].ptype == "plastic"


def test_keep_as_modelled_changes_nothing():
    prims = [plastic("wall_paint", 0.10, 0.10, 0.10)]
    roles = {"wall_paint": "wall"}
    policy = make_reflectance("keep_as_modelled", wall=0.56)

    rebuilt, warnings = apply_reflectances(prims, roles, policy)

    assert warnings == ()
    assert rebuilt == prims
    assert rebuilt[0] is prims[0]


def test_fill_gaps_leaves_a_modelled_value_alone():
    modelled = plastic("wall_paint", 0.40, 0.40, 0.40)
    gap = plastic("floor_paint", 0.0, 0.0, 0.0)
    roles = {"wall_paint": "wall", "floor_paint": "floor"}
    policy = make_reflectance("fill_gaps", wall=0.56, floor=0.34)

    rebuilt, warnings = apply_reflectances([modelled, gap], roles, policy)

    assert warnings == ()
    by_name = {p.name: p for p in rebuilt}
    assert by_name["wall_paint"] is modelled
    assert by_name["floor_paint"].fargs[:3] == (0.34, 0.34, 0.34)


def test_fill_gaps_with_fewer_than_five_reals_is_a_gap():
    gap = Primitive("void", "plastic", "sparse_paint", (), (), (0.0, 0.0, 0.0), "test")
    roles = {"sparse_paint": "wall"}
    policy = make_reflectance("fill_gaps", wall=0.56)

    rebuilt, _ = apply_reflectances([gap], roles, policy)

    assert rebuilt[0].fargs[:3] == (0.56, 0.56, 0.56)


def test_a_role_matched_non_plastic_metal_type_is_left_unchanged_and_warned():
    trans_p = Primitive("void", "trans", "trans_wall", (), (), (0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5), "test")
    roles = {"trans_wall": "wall"}
    policy = make_reflectance("override", wall=0.56)

    rebuilt, warnings = apply_reflectances([trans_p], roles, policy)

    assert rebuilt[0] is trans_p
    assert len(warnings) == 1
    assert "trans_wall" in warnings[0]


def test_an_unmatched_role_is_left_alone():
    wall_p = plastic("wall_paint", 0.10, 0.10, 0.10)
    roles = {"wall_paint": "wall"}
    policy = make_reflectance("override", ceiling=0.75)

    rebuilt, warnings = apply_reflectances([wall_p], roles, policy)

    assert rebuilt[0] is wall_p
    # M4: under override a role with no reflectance key keeps its modelled
    # value, and that is said out loud rather than done silently.
    assert len(warnings) == 1
    assert "'wall'" in warnings[0] and "override" in warnings[0]


def test_override_names_every_scene_role_without_a_reflectance_once():
    prims = [
        plastic("wall_paint", 0.10, 0.10, 0.10),
        plastic("wall_paint_2", 0.20, 0.20, 0.20),
        plastic("floor_paint", 0.20, 0.20, 0.20),
        plastic("ceiling_paint", 0.80, 0.80, 0.80),
        plastic("shade", 0.30, 0.30, 0.30),
    ]
    roles = {
        "wall_paint": "wall", "wall_paint_2": "wall", "floor_paint": "floor",
        "ceiling_paint": "ceiling", "shade": "fixed_shading",
    }
    policy = make_reflectance("override", ceiling=0.70)

    _, warnings = apply_reflectances(prims, roles, policy)

    assert len(warnings) == 1
    assert "'floor', 'wall'" in warnings[0]
    assert "ceiling" not in warnings[0]
    assert "fixed_shading" not in warnings[0]


def test_fill_gaps_does_not_warn_about_roles_without_a_reflectance():
    wall_p = plastic("wall_paint", 0.10, 0.10, 0.10)
    policy = make_reflectance("fill_gaps", ceiling=0.75)
    _, warnings = apply_reflectances([wall_p], {"wall_paint": "wall"}, policy)
    assert warnings == ()


# -- rebuild_materials (the wiring) ---------------------------------------


class _FakeProject:
    def __init__(self, glazing, reflectance):
        self.glazing = glazing
        self.reflectance = reflectance


def test_rebuild_materials_wires_glazing_and_reflectances_together():
    tn = transmissivity(0.75)
    glazing_prim = glass("glazing_a", tn, tn, tn)
    wall_p = plastic("wall_paint", 0.10, 0.10, 0.10)
    roles = {"glazing_a": "glazing", "wall_paint": "wall"}
    project = _FakeProject(
        glazing=make_glazing(dirt_factor=0.86),
        reflectance=make_reflectance("override", wall=0.56),
    )

    rebuilt, records, warnings = rebuild_materials([glazing_prim, wall_p], roles, project)

    assert warnings == ()
    assert len(records) == 1
    assert records[0].net_transmittance == pytest.approx(0.645)
    by_name = {p.name: p for p in rebuilt}
    assert by_name["wall_paint"].fargs[:3] == (0.56, 0.56, 0.56)
    # order is preserved
    assert [p.name for p in rebuilt] == ["glazing_a", "wall_paint"]


def test_rebuild_materials_still_refuses_a_glass_wall():
    tn = transmissivity(0.93)
    prim = glass("broken_glass", tn, tn, tn)
    roles = {"broken_glass": "wall"}
    project = _FakeProject(glazing=make_glazing(), reflectance=make_reflectance("keep_as_modelled"))

    with pytest.raises(MaterialError, match="broken_glass"):
        rebuild_materials([prim], roles, project)


# -- dirt per class, material copies, diffusing trans (spec 5.4) ----------
#
# Invented round values throughout: VT 0.70, dirt 0.95 / 0.90 / 0.85, a grey
# 0.8 diffuser with A6 0.5. Arithmetic is shown beside each expectation.

_TILT = {"vertical": 90.0, "sloped": 30.0, "skylight": 0.0}


def pane_class(tilt_class, *, diffusing=False):
    position = "perimeter" if tilt_class == "vertical" else "roof"
    return PaneClass(
        tilt_deg=_TILT[tilt_class],
        tilt_class=tilt_class,
        position=position,
        diffusing=diffusing,
        kind=kind_of(tilt_class, position, diffusing),
    )


def pane(name, modifier):
    # The outline does not matter to this module: an invented 1 x 1 m square.
    return Primitive(
        modifier, "polygon", name, (), (),
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0), "test",
    )


def trans(name, reals):
    return Primitive("void", "trans", name, (), (), tuple(reals), "test")


def classed_project(dirt_factor, vt_includes_dirt=False):
    return _FakeProject(
        glazing=make_glazing(dirt_factor=dirt_factor, vt_includes_dirt=vt_includes_dirt),
        reflectance=make_reflectance("keep_as_modelled"),
    )


DIRT = MappingProxyType({"vertical": 0.95, "sloped": 0.90, "skylight": 0.85})
GREY_DIFFUSER = (0.8, 0.8, 0.8, 0.0, 0.0, 0.5, 0.0)


def test_copy_names():
    assert copy_name("glazing_a", "sloped") == "glazing_a__sloped"
    assert copy_name("glazing_a", "skylight") == "glazing_a__skylight"
    with pytest.raises(ValueError, match="vertical"):
        copy_name("glazing_a", "vertical")


def test_one_glass_on_a_wall_window_and_a_skylight_gives_the_original_and_a_copy():
    tn = transmissivity(0.70)
    original = glass("glazing_a", tn, tn, tn)
    win, sky = pane("win_1", "glazing_a"), pane("sky_1", "glazing_a")
    classed = rebuild_materials_by_class(
        [original, win, sky], {"glazing_a": "glazing"}, classed_project(DIRT),
        {"win_1": pane_class("vertical"), "sky_1": pane_class("skylight")},
    )

    assert [p.name for p in classed.prims] == [
        "glazing_a", "glazing_a__skylight", "win_1", "sky_1",
    ]
    # Surfaces pass through as the very same objects: prepare's
    # identity-keyed lookups (position_by_id) rely on it.
    assert classed.prims[2] is win
    assert classed.prims[3] is sky
    # Original: 0.70 x 0.95 = 0.665. Copy: 0.70 x 0.85 = 0.595.
    assert classed.prims[0].fargs == pytest.approx((transmissivity(0.665),) * 3)
    copy = classed.prims[1]
    assert (copy.modifier, copy.ptype) == ("void", "glass")
    assert copy.fargs == pytest.approx((transmissivity(0.595),) * 3)

    assert [(r.modifier, r.tilt_class) for r in classed.records] == [
        ("glazing_a", "vertical"), ("glazing_a__skylight", "skylight"),
    ]
    assert classed.records[0].net_transmittance == pytest.approx(0.665)
    assert classed.records[1].net_transmittance == pytest.approx(0.595)
    assert classed.records[1].dirt_factor == 0.85
    assert classed.records[1].transmission == "specular"
    assert dict(classed.pane_modifier) == {"win_1": "glazing_a", "sky_1": "glazing_a__skylight"}
    assert classed.effective_roles["glazing_a"] == "glazing"
    assert classed.effective_roles["glazing_a__skylight"] == "glazing"
    assert classed.warnings == ()


def test_copies_sit_right_after_their_original_sloped_before_skylight():
    tn = transmissivity(0.70)
    a, b = glass("glazing_a", tn, tn, tn), glass("glazing_b", tn, tn, tn)
    panes = [pane("sky_1", "glazing_a"), pane("slope_1", "glazing_a"), pane("win_1", "glazing_b")]
    roles = {"glazing_a": "glazing", "glazing_b": "glazing_curtain_wall"}
    classed = rebuild_materials_by_class(
        [a, b, *panes], roles, classed_project(DIRT),
        {
            "sky_1": pane_class("skylight"),
            "slope_1": pane_class("sloped"),
            "win_1": pane_class("vertical"),
        },
    )
    assert [p.name for p in classed.prims[:4]] == [
        "glazing_a", "glazing_a__sloped", "glazing_a__skylight", "glazing_b",
    ]
    # Originals first, in scene order; then the copies.
    assert [r.modifier for r in classed.records] == [
        "glazing_a", "glazing_b", "glazing_a__sloped", "glazing_a__skylight",
    ]
    # Sloped: 0.70 x 0.90 = 0.63.
    assert classed.records[2].net_transmittance == pytest.approx(0.63)
    assert classed.effective_roles["glazing_a__sloped"] == "glazing"


def test_the_records_of_originals_are_exactly_the_all_vertical_records():
    tn = transmissivity(0.70)
    prims = [
        glass("glazing_a", tn, tn, tn), glass("glazing_unused", tn, tn, tn),
        pane("win_1", "glazing_a"), pane("sky_1", "glazing_a"),
    ]
    roles = {"glazing_a": "glazing", "glazing_unused": "glazing"}
    project = classed_project(MappingProxyType({"vertical": 0.95, "skylight": 0.85}))

    _, plain_records, _ = rebuild_materials(prims, roles, project)
    classed = rebuild_materials_by_class(
        prims, roles, project,
        {"win_1": pane_class("vertical"), "sky_1": pane_class("skylight")},
    )
    assert classed.records[:2] == plain_records
    assert [r.modifier for r in classed.records[2:]] == ["glazing_a__skylight"]


def test_a_material_used_only_by_skylights_with_no_vertical_key_is_written_unchanged():
    tn = transmissivity(0.70)
    original = glass("glazing_a", tn, tn, tn)
    classed = rebuild_materials_by_class(
        [original, pane("sky_1", "glazing_a")], {"glazing_a": "glazing"},
        classed_project(MappingProxyType({"skylight": 0.85})),
        {"sky_1": pane_class("skylight")},
    )
    assert classed.prims[0] is original
    assert classed.records[0].dirt_factor == 1.0
    assert classed.records[0].net_transmittance == pytest.approx(0.70)
    # The copy still gets its own class: 0.70 x 0.85 = 0.595.
    assert classed.prims[1].fargs == pytest.approx((transmissivity(0.595),) * 3)


def test_with_a_vertical_key_a_material_used_only_by_skylights_is_rebuilt_as_today():
    tn = transmissivity(0.70)
    classed = rebuild_materials_by_class(
        [glass("glazing_a", tn, tn, tn), pane("sky_1", "glazing_a")], {"glazing_a": "glazing"},
        classed_project(MappingProxyType({"vertical": 0.95, "skylight": 0.85})),
        {"sky_1": pane_class("skylight")},
    )
    # 0.70 x 0.95 = 0.665, although no vertical pane uses it.
    assert classed.prims[0].fargs == pytest.approx((transmissivity(0.665),) * 3)
    assert classed.records[0].dirt_factor == 0.95


def test_a_missing_class_key_is_refused_by_name():
    tn = transmissivity(0.70)
    with pytest.raises(
        MaterialError,
        match=r"glazing\.dirt_factor: no 'skylight' value, but pane sky_1 is skylight class",
    ):
        rebuild_materials_by_class(
            [glass("glazing_a", tn, tn, tn), pane("win_1", "glazing_a"), pane("sky_1", "glazing_a")],
            {"glazing_a": "glazing"},
            classed_project(MappingProxyType({"vertical": 0.95})),
            {"win_1": pane_class("vertical"), "sky_1": pane_class("skylight")},
        )


def test_a_missing_vertical_key_is_refused_when_vertical_glass_exists():
    tn = transmissivity(0.70)
    with pytest.raises(MaterialError, match=r"no 'vertical' value, but pane win_1 is vertical class"):
        rebuild_materials_by_class(
            [glass("glazing_a", tn, tn, tn), pane("win_1", "glazing_a")],
            {"glazing_a": "glazing"},
            classed_project(MappingProxyType({"skylight": 0.85})),
            {"win_1": pane_class("vertical")},
        )


def test_a_single_number_is_refused_with_sloped_glass():
    tn = transmissivity(0.70)
    with pytest.raises(MaterialError, match=r"glazing\.dirt_factor: a single number.*slope_1 is sloped class"):
        rebuild_materials_by_class(
            [glass("glazing_a", tn, tn, tn), pane("slope_1", "glazing_a")],
            {"glazing_a": "glazing"}, classed_project(0.90),
            {"slope_1": pane_class("sloped")},
        )


def test_vt_includes_dirt_accepts_a_single_number_and_applies_no_dirt():
    tn = transmissivity(0.70)
    classed = rebuild_materials_by_class(
        [glass("glazing_a", tn, tn, tn), pane("sky_1", "glazing_a")],
        {"glazing_a": "glazing"}, classed_project(0.90, vt_includes_dirt=True),
        {"sky_1": pane_class("skylight")},
    )
    assert classed.records[1].dirt_factor == 1.0
    assert classed.records[1].net_transmittance == pytest.approx(0.70)
    assert classed.warnings == ()


@pytest.mark.parametrize(
    "tilt_class, value, warned",
    [("sloped", 0.95, True), ("sloped", 0.90, False), ("skylight", 0.90, True), ("skylight", 0.85, False)],
)
def test_a_sloped_or_skylight_factor_above_the_lm83_minimum_is_warned(tilt_class, value, warned):
    tn = transmissivity(0.70)
    classed = rebuild_materials_by_class(
        [glass("glazing_a", tn, tn, tn), pane("p_1", "glazing_a")], {"glazing_a": "glazing"},
        classed_project(MappingProxyType({tilt_class: value})),
        {"p_1": pane_class(tilt_class)},
    )
    hits = [w for w in classed.warnings if w.startswith("glazing.dirt_factor")]
    assert bool(hits) is warned
    if warned:
        assert f"the {tilt_class} value" in hits[0]


def test_no_dirt_warning_when_vt_includes_dirt():
    tn = transmissivity(0.70)
    classed = rebuild_materials_by_class(
        [glass("glazing_a", tn, tn, tn), pane("p_1", "glazing_a")], {"glazing_a": "glazing"},
        classed_project(MappingProxyType({"skylight": 0.95}), vt_includes_dirt=True),
        {"p_1": pane_class("skylight")},
    )
    assert classed.warnings == ()


def test_a_diffusing_trans_keeps_rd_and_scales_only_td():
    # Grey 0.8, A4 = 0, A5 = 0, A6 = 0.5 (A4 < 0.018, so no Fresnel term):
    #   Td = 0.8 x (1 - 0) x 0.5 = 0.4        Rd = 0.8 x (1 - 0) x (1 - 0.5) = 0.4
    # Skylight copy, dirt 0.85:
    #   Td' = 0.85 x 0.4 = 0.34   k = (0.34 + 0.4) / (0.4 + 0.4) = 0.925
    #   channels 0.8 x 0.925 = 0.74   A6' = 0.34 / (0.34 + 0.4) = 0.34 / 0.74
    # Original, vertical dirt 0.95:
    #   Td' = 0.38   k = (0.38 + 0.4) / 0.8 = 0.975   channels 0.78   A6' = 0.38 / 0.78
    diffuser = trans("diffuser_a", GREY_DIFFUSER)
    classed = rebuild_materials_by_class(
        [diffuser, pane("wall_1", "diffuser_a"), pane("sky_1", "diffuser_a")],
        {"diffuser_a": "glazing_diffusing"}, classed_project(DIRT),
        {
            "wall_1": pane_class("vertical", diffusing=True),
            "sky_1": pane_class("skylight", diffusing=True),
        },
    )
    original, copy = classed.prims[0], classed.prims[1]
    assert (copy.name, copy.ptype) == ("diffuser_a__skylight", "trans")
    assert copy.fargs == pytest.approx((0.74, 0.74, 0.74, 0.0, 0.0, 0.34 / 0.74, 0.0))
    assert original.fargs == pytest.approx((0.78, 0.78, 0.78, 0.0, 0.0, 0.38 / 0.78, 0.0))

    record_original, record_copy = classed.records
    assert record_copy.transmission == "diffuse"
    assert record_copy.tilt_class == "skylight"
    assert record_copy.visible_transmittance == pytest.approx(0.4)
    assert record_copy.net_transmittance == pytest.approx(0.34)
    assert record_copy.transmissivity == pytest.approx(0.34 / 0.74)
    assert record_copy.tinted is False
    assert record_original.transmission == "diffuse"
    assert record_original.net_transmittance == pytest.approx(0.38)


def test_a_coloured_diffuser_keeps_its_colour_ratios_and_its_rd():
    # Invented coloured diffuser; A5 = 0.1 (A5 squared 0.01 > 1e-6: no Fresnel).
    r, g, b, a4, a5, a6 = 0.8, 0.6, 0.4, 0.05, 0.1, 0.5
    classed = rebuild_materials_by_class(
        [trans("diffuser_c", (r, g, b, a4, a5, a6, 0.0)), pane("sky_1", "diffuser_c")],
        {"diffuser_c": "glazing_diffusing"},
        classed_project(MappingProxyType({"skylight": 0.85})),
        {"sky_1": pane_class("skylight", diffusing=True)},
    )
    weights = LUX_WEIGHTS / LUX_WEIGHTS.sum()

    def td_rd(channels, spec, trans_):
        c = float(weights @ np.array(channels))
        return c * (1 - spec) * trans_, c * (1 - spec) * (1 - trans_)

    td, rd = td_rd((r, g, b), a4, a6)
    nr, ng, nb, n4, n5, n6, n7 = classed.prims[1].fargs
    td_new, rd_new = td_rd((nr, ng, nb), n4, n6)
    assert td_new == pytest.approx(0.85 * td)
    assert rd_new == pytest.approx(rd)
    assert (nr / ng, ng / nb) == pytest.approx((r / g, g / b))
    assert (n4, n5, n7) == (a4, a5, 0.0)
    assert classed.records[1].tinted is True


@pytest.mark.parametrize(
    "reals, fragment",
    [
        ((0.8, 0.8, 0.8, 0.0, 0.0, 0.5), r"exactly 7 real arguments"),
        ((0.8, 0.8, 0.8, 0.0, 0.0, 0.5, 0.1), r"transmitted specularity \(7th argument\)"),
        ((0.0, 0.8, 0.8, 0.0, 0.0, 0.5, 0.0), r"channel r"),
        ((0.8, 1.1, 0.8, 0.0, 0.0, 0.5, 0.0), r"channel g"),
        ((0.8, 0.8, 0.8, 1.0, 0.1, 0.5, 0.0), r"specularity \(4th argument\)"),
        ((0.8, 0.8, 0.8, -0.1, 0.1, 0.5, 0.0), r"specularity \(4th argument\)"),
        ((0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0), r"transmissivity \(6th argument\)"),
        ((0.8, 0.8, 0.8, 0.0, 0.0, 1.1, 0.0), r"transmissivity \(6th argument\)"),
        ((0.8, 0.8, 0.8, 0.02, 0.0, 0.5, 0.0), r"Fresnel"),
    ],
)
def test_each_diffusing_trans_argument_refusal(reals, fragment):
    with pytest.raises(MaterialError, match=rf"diffuser_a: .*{fragment}"):
        rebuild_glazing(
            [trans("diffuser_a", reals)], {"diffuser_a": "glazing_diffusing"}, make_glazing()
        )


def test_glass_with_the_diffusing_role_is_refused():
    tn = transmissivity(0.70)
    with pytest.raises(MaterialError, match=r"clear_a: role 'glazing_diffusing' needs a Radiance 'trans'"):
        rebuild_glazing([glass("clear_a", tn, tn, tn)], {"clear_a": "glazing_diffusing"}, make_glazing())


@pytest.mark.parametrize("role", ["glazing", "glazing_curtain_wall"])
def test_trans_with_another_glazing_role_is_refused(role):
    with pytest.raises(MaterialError, match=r"diffuser_a: a 'trans' glazing modifier needs the role 'glazing_diffusing'"):
        rebuild_glazing([trans("diffuser_a", GREY_DIFFUSER)], {"diffuser_a": role}, make_glazing())


def test_a_skylight_pane_whose_material_is_not_in_the_scene_is_refused():
    with pytest.raises(MaterialError, match=r"glazing_x: pane sky_1 is skylight class"):
        rebuild_materials_by_class(
            [pane("sky_1", "glazing_x")], {"glazing_x": "glazing"}, classed_project(DIRT),
            {"sky_1": pane_class("skylight")},
        )


def test_rebuild_materials_is_the_all_vertical_path():
    r, g, b = transmissivity(0.70), transmissivity(0.75), transmissivity(0.65)
    prims = [glass("tinted_a", r, g, b), plastic("wall_paint", 0.10, 0.10, 0.10), pane("win_1", "tinted_a")]
    roles = {"tinted_a": "glazing", "wall_paint": "wall"}
    project = _FakeProject(
        glazing=make_glazing(dirt_factor=0.86),
        reflectance=make_reflectance("override", wall=0.56),
    )
    plain = rebuild_materials(prims, roles, project)
    classed = rebuild_materials_by_class(prims, roles, project, {"win_1": pane_class("vertical")})
    assert (classed.prims, classed.records, classed.warnings) == plain
    assert dict(classed.effective_roles) == roles
    assert dict(classed.pane_modifier) == {"win_1": "tinted_a"}
