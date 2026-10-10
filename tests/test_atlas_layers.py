# tests/test_atlas_layers.py
"""Kinds, colours and result layers of the atlas (atlas spec 6, 8).

One colour per kind of geometry; one result layer per row of the spec 8
table, present only when its option's results are current, each with the
colour scale the Word report uses for it. Every number is invented and
round.
"""
from __future__ import annotations

import numpy as np
import pytest

from deliver_folder import build_deliver_project, edit_results
from leed_daylight.deliver.atlas.colours import DEFAULT_OFF, KIND_RGBA, KINDS, ROOM_FILL_RGB
from leed_daylight.deliver.atlas.layers import Layer, layer_rgb, result_layers
from leed_daylight.deliver.data import DeliverError, load_case
from leed_daylight.deliver.figures.spec import FAIL_RGB, PASS_RGB, ColourScale

SKIES = ["03-28_0900", "03-28_1500", "09-14_0900", "09-14_1500"]
ALL_KEYS = [
    "option1.da_pct", "option1.sda_pass", "option1.ase_hours", "option1.ase_exceeds",
    *(f"option2.lux_{sky}" for sky in SKIES), "option2.passes",
    "views.type2", "views.type3", "views.quality",
]


def test_every_kind_has_one_colour():
    assert KINDS == (
        "wall", "floor", "ceiling", "fixed_shading", "furniture", "glass_window", "glass_curtain_wall",
        "glass_interior", "glass_diffusing", "glass_roof", "frame", "reveal", "skylight_well", "blind",
        "context", "ground",
    )
    assert list(KIND_RGBA) == list(KINDS)
    for rgba in KIND_RGBA.values():
        assert len(rgba) == 4 and all(isinstance(v, int) and 0 <= v <= 255 for v in rgba)
    assert DEFAULT_OFF == frozenset({"ceiling", "blind", "context", "ground"})
    assert len(ROOM_FILL_RGB) == 3


def test_every_layer_of_a_current_option_is_listed_in_table_order(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea)

    layers = result_layers(load_case(folder, require_results=False))

    assert [layer.key for layer in layers] == ALL_KEYS
    assert [layer.column for layer in layers] == [key.split(".", 1)[1] for key in ALL_KEYS]
    assert [layer.option for layer in layers] == [key.split(".", 1)[0] for key in ALL_KEYS]
    assert [layer.title for layer in layers] == [
        "Daylight autonomy", "sDA pass", "ASE hours", "ASE over limit",
        *(f"Illuminance, {sky}" for sky in SKIES), "Option 2 pass",
        "Type 2 view", "Type 3 view", "Quality View",
    ]


def test_the_scales_are_the_word_reports_scales(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea)

    layers = {layer.key: layer for layer in result_layers(load_case(folder, require_results=False))}

    assert layers["option1.da_pct"].scale == ColourScale(
        0.0, 100.0, "% of occupied hours at 300 lux or more", marks=(50.0,))
    assert layers["option1.ase_hours"].scale == ColourScale(
        0.0, 500.0, "occupied hours at 1000 lux or more, blinds open", marks=(250.0,))
    assert layers["option2.lux_03-28_0900"].scale == ColourScale(
        30.0, 9000.0, "lux (log scale)", marks=(300.0, 3000.0), log=True)
    assert [layers[k].fail_when for k in ("option1.sda_pass", "option1.ase_exceeds", "option2.passes",
                                          "views.type2", "views.type3", "views.quality")] == [
        False, True, False, False, False, False]
    assert all(layers[k].scale is None for k in ("option1.sda_pass", "option2.passes", "views.quality"))
    assert layers["option1.ase_exceeds"].legend == ("within the limit", "over the limit")
    assert layers["option2.passes"].legend == ("in the band", "outside the band")
    assert layers["views.type2"].legend == ("view", "no view")


def test_an_option_with_no_results_has_no_layer_and_keeps_its_reason(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=("option1",))

    case = load_case(folder, require_results=False)

    assert [layer.key for layer in result_layers(case)] == ALL_KEYS[:4]
    assert sorted(case.skipped) == ["option2", "views"]
    assert "no option2 results for case 'base_case'" in case.skipped["option2"]


def test_a_stale_option_has_no_layer_and_keeps_its_reason(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea)
    edit_results(folder, "views", lambda payload: payload["inputs"].update({"project.yaml": "0" * 64}))

    case = load_case(folder, require_results=False)

    assert [layer.key for layer in result_layers(case)] == ALL_KEYS[:-3]
    assert "views results for case 'base_case' are out of date: project.yaml changed" in case.skipped["views"]


def test_a_case_with_no_results_has_no_layer(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=())

    assert result_layers(load_case(folder, require_results=False)) == ()


def test_a_layer_has_a_scale_or_a_fail_value():
    scale = ColourScale(0.0, 10.0, "x")
    with pytest.raises(DeliverError, match="layer a.b: give a colour scale or a fail value, not both or neither"):
        Layer("a.b", "a", "A", "b", None, None)
    with pytest.raises(DeliverError, match="layer a.b: give a colour scale or a fail value"):
        Layer("a.b", "a", "A", "b", scale, True)


def test_layer_colours():
    scale = ColourScale(0.0, 10.0, "x")
    number = Layer("a.n", "a", "N", "n", scale, None)
    passes = Layer("a.p", "a", "P", "p", None, False)
    exceeds = Layer("a.e", "a", "E", "e", None, True)
    values = np.array([0.0, 1.0, 10.0])

    assert layer_rgb(number, values).tolist() == [list(scale.rgb(v)) for v in values]
    assert layer_rgb(passes, values).tolist() == [list(FAIL_RGB), list(PASS_RGB), list(PASS_RGB)]
    assert layer_rgb(exceeds, values).tolist() == [list(PASS_RGB), list(FAIL_RGB), list(FAIL_RGB)]
    assert layer_rgb(number, values).dtype == np.uint8
    assert layer_rgb(passes, np.array([])).shape == (0, 3)
