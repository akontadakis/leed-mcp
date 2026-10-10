# tests/test_scene_inventory.py
import pathlib

import pytest

from leed_daylight.scene.inventory import inventory
from leed_daylight.scene.reader import read_scene

FIX = pathlib.Path(__file__).parent / "fixtures"


def infos():
    return {i.name: i for i in inventory(read_scene([FIX / "simple_room.rad"], "m"))}


def test_every_used_modifier_appears():
    assert set(infos()) == {"wall_paint", "window_glass"}


def test_surface_counts_and_areas_are_summed():
    paint = infos()["wall_paint"]
    assert paint.surfaces == 2
    assert paint.area_m2 == pytest.approx(24.0)      # 4x3 floor plus 4x3 ceiling


def test_glass_is_recognised_and_measured():
    glass = infos()["window_glass"]
    assert glass.kind == "glass"
    assert glass.surfaces == 1
    assert glass.area_m2 == pytest.approx(3.0)       # 2.0 m x 1.5 m


def test_plastic_is_opaque_and_keeps_its_rgb():
    paint = infos()["wall_paint"]
    assert paint.kind == "opaque"
    assert paint.rgb == pytest.approx((0.5, 0.5, 0.5))


def test_glass_is_guessed_as_glazing():
    assert infos()["window_glass"].guessed_role == "glazing"


def test_opaque_is_guessed_as_wall():
    assert infos()["wall_paint"].guessed_role == "wall"


def test_results_are_sorted_by_area_descending():
    areas = [i.area_m2 for i in inventory(read_scene([FIX / "simple_room.rad"], "m"))]
    assert areas == sorted(areas, reverse=True)


def test_an_undefined_modifier_is_reported_with_no_type():
    from leed_daylight.scene.primitives import Primitive

    orphan = Primitive("ghost", "polygon", "p", (), (), (0, 0, 0, 1, 0, 0, 1, 1, 0), "x")
    info = {i.name: i for i in inventory([orphan])}["ghost"]
    assert info.ptype == "" and info.kind == "other"
    assert info.guessed_role == ""


def test_a_malformed_polygon_does_not_crash_the_inventory():
    from leed_daylight.scene.primitives import Primitive

    paint = Primitive("void", "plastic", "paint", (), (), (0.5, 0.5, 0.5, 0, 0), "test")
    bad = Primitive("paint", "polygon", "bad", (), (), (0, 0, 0, 1, 0, 0, 1, 1), "test")
    [row] = inventory([paint, bad])
    assert row.name == "paint"
    assert row.surfaces == 1
    assert row.area_m2 == 0.0


def _trans_row(fargs, ptype="trans"):
    from leed_daylight.scene.primitives import Primitive

    material = Primitive("void", ptype, "panel", (), (), fargs, "test")
    surface = Primitive("panel", "polygon", "p", (), (), (0, 0, 0, 1, 0, 0, 1, 1, 0), "test")
    [row] = inventory([material, surface])
    return row


def test_a_trans_with_no_specular_transmission_is_guessed_as_diffusing_glazing():
    row = _trans_row((0.8, 0.8, 0.8, 0.0, 0.0, 0.5, 0.0))
    assert row.kind == "trans"
    assert row.guessed_role == "glazing_diffusing"


@pytest.mark.parametrize("fargs", [
    (0.8, 0.8, 0.8, 0.0, 0.0, 0.5, 0.2),     # some light passes straight through
    (0.8, 0.8, 0.8, 0.0, 0.0, 0.5),          # not a well-formed trans
    (0.8, 0.8, 0.8, 0.0, 0.0, 0.5, 0.0, 1.0),
])
def test_any_other_trans_is_still_guessed_as_fixed_shading(fargs):
    assert _trans_row(fargs).guessed_role == "fixed_shading"


def test_a_trans2_is_still_guessed_as_fixed_shading():
    row = _trans_row((0.8, 0.8, 0.8, 0.0, 0.0, 0.5, 0.0), ptype="trans2")
    assert row.guessed_role == "fixed_shading"
