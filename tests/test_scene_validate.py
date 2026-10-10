import pathlib

import pytest

from leed_daylight.scene.primitives import Primitive
from leed_daylight.scene.reader import read_scene
from leed_daylight.scene.validate import SceneError, validate_scene

FIX = pathlib.Path(__file__).parent / "fixtures"


def square(modifier="wall_paint", name="p", z=0.0, size=1.0):
    return Primitive(
        modifier, "polygon", name, (), (),
        (0, 0, z, size, 0, z, size, size, z, 0, size, z), "test",
    )


def paint():
    return Primitive("void", "plastic", "wall_paint", (), (), (0.5, 0.5, 0.5, 0, 0), "test")


def test_the_example_scene_is_clean():
    report = validate_scene(read_scene([FIX / "simple_room.rad"], "m"), "m")
    assert report.errors == ()
    report.raise_if_errors()


def test_an_undefined_modifier_is_an_error():
    report = validate_scene([square(modifier="ghost")], "m")
    assert any("ghost" in e for e in report.errors)


def test_void_is_always_defined():
    report = validate_scene([paint(), square()], "m")
    assert report.errors == ()


def test_a_degenerate_polygon_is_an_error():
    flat = Primitive("wall_paint", "polygon", "line", (), (),
                     (0, 0, 0, 1, 0, 0, 2, 0, 0), "test")
    report = validate_scene([paint(), flat], "m")
    assert any("degenerate" in e for e in report.errors)


def test_a_non_planar_polygon_is_an_error():
    bent = Primitive("wall_paint", "polygon", "bent", (), (),
                     (0, 0, 0, 1, 0, 0, 1, 1, 0.01, 0, 1, 0), "test")
    report = validate_scene([paint(), bent], "m")
    assert any("planar" in e for e in report.errors)


def test_a_one_millimetre_bend_is_tolerated():
    bent = Primitive("wall_paint", "polygon", "bent", (), (),
                     # source: the test's own 0.5 mm out-of-plane offset
                     (0, 0, 0, 1, 0, 0, 1, 1, 0.0005, 0, 1, 0), "test")
    assert validate_scene([paint(), bent], "m").errors == ()


def test_duplicate_coplanar_surfaces_are_an_error():
    report = validate_scene([paint(), square(name="a"), square(name="b")], "m")
    assert any("duplicate" in e for e in report.errors)


def test_a_scene_with_no_surface_is_an_error():
    assert any("no surface" in e for e in validate_scene([paint()], "m").errors)


def test_a_plausible_metric_scene_is_not_called_implausible():
    report = validate_scene(read_scene([FIX / "simple_room.rad"], "m"), "m")
    assert not any("implausible size" in e for e in report.errors)


def test_a_millimetre_sized_model_is_refused_as_implausible():
    # The primitive is already in metres; a 0.001 m square is what a 1 mm
    # building looks like once `read_scene` has converted it.
    tiny = square(size=0.001)
    report = validate_scene([paint(), tiny], "mm")
    matches = [e for e in report.errors if "implausible size" in e]
    assert len(matches) == 1
    assert "'mm'" in matches[0]


def test_a_degenerate_sliver_in_a_millimetre_scene_is_refused(tmp_path):
    # source: the test's own 2 micron offset, giving 1e-12 m2
    sliver_y = 0.000002
    scene = tmp_path / "s.rad"
    scene.write_text(
        "void plastic wall_paint\n0\n0\n5 0.5 0.5 0.5 0 0\n\n"
        f"wall_paint polygon sliver.1\n0\n0\n9  0 0 0   1 0 0   1 {sliver_y} 0\n"
    , newline="\n")
    prims = read_scene([scene], "mm")
    report = validate_scene(prims, "mm")
    assert any("degenerate polygon" in e for e in report.errors)


def test_an_unknown_unit_is_an_error():
    assert any("unit" in e for e in validate_scene([paint(), square()], "furlong").errors)


def test_a_modifier_without_a_role_is_an_error_when_roles_are_given():
    report = validate_scene([paint(), square()], "m", roles={})
    assert any("no role" in e for e in report.errors)


def test_roles_that_cover_everything_pass():
    assert validate_scene([paint(), square()], "m", roles={"wall_paint": "floor"}).errors == ()


def test_a_declared_type_that_differs_from_the_scene_is_an_error():
    # materials.csv says glass, the scene defines plastic: the role was
    # chosen for a different material than the one the run will use.
    report = validate_scene(
        [paint(), square()], "m",
        roles={"wall_paint": "glazing"}, declared_types={"wall_paint": "glass"},
    )
    matches = [e for e in report.errors if "wall_paint" in e and "materials.csv" in e]
    assert len(matches) == 1
    assert "'glass'" in matches[0] and "'plastic'" in matches[0]


def test_a_declared_type_that_matches_the_scene_passes():
    report = validate_scene(
        [paint(), square()], "m",
        roles={"wall_paint": "wall"}, declared_types={"wall_paint": "plastic"},
    )
    assert report.errors == ()


def test_a_declared_type_is_checked_against_the_last_definition():
    # Radiance keeps the last definition of a name; so does the check.
    glass = Primitive("void", "glass", "wall_paint", (), (), (0.8, 0.8, 0.8), "test")
    report = validate_scene(
        [glass, paint(), square()], "m",
        roles={"wall_paint": "wall"}, declared_types={"wall_paint": "plastic"},
    )
    assert not any("materials.csv" in e for e in report.errors)


def test_a_declared_type_for_a_modifier_the_scene_never_defines_is_skipped():
    report = validate_scene(
        [paint(), square()], "m",
        roles={"wall_paint": "wall", "unused": "wall"},
        declared_types={"wall_paint": "plastic", "unused": "metal"},
    )
    assert report.errors == ()


def test_raise_if_errors_raises_and_names_everything():
    with pytest.raises(SceneError) as excinfo:
        validate_scene([square(modifier="ghost")], "m").raise_if_errors()
    assert "ghost" in str(excinfo.value)


def test_a_sphere_is_refused_by_name():
    dome = Primitive("wall_paint", "sphere", "dome.1", (), (), (0, 0, 0, 1), "test")
    report = validate_scene([paint(), dome], "m")
    matches = [e for e in report.errors if "unsupported surface type" in e]
    assert len(matches) == 1
    assert "sphere" in matches[0]


def test_a_sphere_only_scene_is_not_also_called_implausibly_small():
    dome = Primitive("wall_paint", "sphere", "dome.1", (), (), (0, 0, 0, 1), "test")
    report = validate_scene([paint(), dome], "m")
    assert not any("implausible size" in e for e in report.errors)


def test_a_malformed_polygon_does_not_destroy_the_report():
    # 11 reals: not a multiple of 3, so Primitive.points raises ValueError.
    malformed = Primitive(
        "wall_paint", "polygon", "bad.1", (), (),
        (0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0), "test",
    )
    orphan = square(modifier="ghost", name="orphan.1")
    report = validate_scene([paint(), malformed, orphan], "m")
    assert len(report.errors) == 2


def test_a_pentagon_and_a_star_from_the_same_points_are_not_duplicates():
    a, b, c, d, e = (0, 0, 0), (2, 0, 0), (3, 1, 0), (1, 2, 0), (-1, 1, 0)
    pentagon = Primitive(
        "wall_paint", "polygon", "pentagon.1", (), (),
        a + b + c + d + e, "test",
    )
    star = Primitive(
        "wall_paint", "polygon", "star.1", (), (),
        a + c + e + b + d, "test",
    )
    report = validate_scene([paint(), pentagon, star], "m")
    assert not any("duplicate" in e for e in report.errors)


def test_the_same_polygon_wound_backwards_is_a_duplicate():
    forward = square(name="forward")
    reversed_pts = Primitive(
        "wall_paint", "polygon", "backward", (), (),
        (0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0), "test",
    )
    report = validate_scene([paint(), forward, reversed_pts], "m")
    assert sum("duplicate" in e for e in report.errors) == 1


def test_the_same_polygon_rotated_in_its_list_is_a_duplicate():
    original = square(name="original")
    rotated = Primitive(
        "wall_paint", "polygon", "rotated", (), (),
        (1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0), "test",
    )
    report = validate_scene([paint(), original, rotated], "m")
    assert sum("duplicate" in e for e in report.errors) == 1


def test_a_clean_scene_still_passes():
    report = validate_scene(read_scene([FIX / "simple_room.rad"], "m"), "m")
    assert report.errors == ()


def test_a_zero_vertex_polygon_does_not_crash_the_span_check():
    # 0 reals: a multiple of 3, so Primitive.points does not raise -- it
    # returns an empty (0, 3) array, which must not reach np.vstack/max/min.
    zero = Primitive("wall_paint", "polygon", "zero.1", (), (), (), "test")
    report = validate_scene([paint(), zero], "m")
    assert any("fewer than three vertices" in e for e in report.errors)


def test_a_scene_with_only_malformed_polygons_is_not_also_called_implausibly_small():
    # Every polygon-typed primitive fails to parse, so there is no valid
    # polygon to compute a span from at all -- unlike the mixed malformed
    # scene above, nothing here masks a spurious zero-span error.
    bad1 = Primitive(
        "wall_paint", "polygon", "bad.1", (), (),
        (0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0), "test",
    )
    bad2 = Primitive(
        "wall_paint", "polygon", "bad.2", (), (),
        (0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0), "test",
    )
    report = validate_scene([paint(), bad1, bad2], "m")
    assert not any("implausible size" in e for e in report.errors)
    assert sum("reals, not a multiple of 3" in e for e in report.errors) == 2


def test_span_and_the_per_primitive_loop_agree_on_zero_vertex_and_malformed_polygons():
    # _span and the per-primitive loop used to hand-roll separate copies of
    # the same `.points` guard; pin that a zero-vertex polygon and a
    # malformed polygon are each reported exactly once, by the loop, and
    # never distort the span computed from the one valid polygon.
    zero = Primitive("wall_paint", "polygon", "zero.1", (), (), (), "test")
    malformed = Primitive(
        "wall_paint", "polygon", "bad.1", (), (),
        (0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0), "test",
    )
    report = validate_scene([paint(), square(), zero, malformed], "m")
    assert not any("implausible size" in e for e in report.errors)
    assert sum("fewer than three vertices" in e for e in report.errors) == 1
    assert sum("reals, not a multiple of 3" in e for e in report.errors) == 1


def test_a_non_finite_vertex_is_named_and_reported_once():
    # A NaN vertex used to be caught only by luck, as an "implausible size"
    # blamed on the declared unit, and to raise numpy RuntimeWarnings along
    # the way.  It must instead be named as what it is, and only that.
    nan_poly = Primitive(
        "wall_paint", "polygon", "corrupt.1", (), (),
        (float("nan"),) * 12, "test",
    )
    report = validate_scene([paint(), nan_poly], "m")
    assert len(report.errors) == 1
    assert "corrupt.1" in report.errors[0]
    assert "non-finite" in report.errors[0]
    assert "implausible size" not in report.errors[0]
    assert "degenerate" not in report.errors[0]
    assert "planar" not in report.errors[0]


def test_an_infinite_vertex_is_reported_as_non_finite():
    inf_poly = Primitive(
        "wall_paint", "polygon", "inf.1", (), (),
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, float("inf"), 1.0, 0.0), "test",
    )
    report = validate_scene([paint(), inf_poly], "m")
    assert any("inf.1" in e and "non-finite" in e for e in report.errors)
    assert not any("implausible size" in e for e in report.errors)


def test_a_non_finite_vertex_does_not_pollute_the_span_of_other_polygons():
    nan_poly = Primitive(
        "wall_paint", "polygon", "corrupt.1", (), (),
        (float("nan"),) * 12, "test",
    )
    report = validate_scene([paint(), square(), nan_poly], "m")
    assert not any("implausible size" in e for e in report.errors)
    assert sum("corrupt.1" in e for e in report.errors) == 1


@pytest.mark.parametrize("ptype", ["source", "instance", "mesh", "blob"])
def test_a_non_scene_primitive_type_is_refused_by_name(ptype):
    odd = Primitive("void", ptype, "odd.1", (), (), (0.0, 0.0, 1.0, 0.5), "test")
    report = validate_scene([paint(), square(), odd], "m")
    assert any("odd.1" in e and ptype in e and "not supported in the scene" in e
               for e in report.errors), report.errors


@pytest.mark.parametrize("ptype", ["colorpict", "texfunc", "mixfunc", "brightfunc"])
def test_a_pattern_texture_or_mixture_is_not_refused_as_unknown(ptype):
    mod = Primitive("void", ptype, "m.1", (), (), (), "test")
    report = validate_scene([mod, paint(), square()], "m")
    assert not any("m.1" in e for e in report.errors), report.errors


def _poly(name, pts, modifier="wall_paint"):
    return Primitive(modifier, "polygon", name, (), (),
                     tuple(float(v) for p in pts for v in p), "test")


def test_a_bow_tie_floor_is_refused():
    # Two edges cross at a point inside both: not a simple polygon.
    bow = _poly("bow", [(0, 0, 0), (3, 0, 0), (0, 2, 0), (1, 3, 0)])
    report = validate_scene([paint(), bow], "m")
    assert any("bow" in e and "crosses itself" in e for e in report.errors), report.errors


def test_a_bow_tie_wall_is_refused_in_its_own_plane():
    bow = _poly("bow", [(0, 0, 0), (0, 3, 0), (0, 0, 2), (0, 1, 3)])
    report = validate_scene([paint(), bow], "m")
    assert any("bow" in e and "crosses itself" in e for e in report.errors), report.errors


def test_a_keyhole_polygon_is_not_refused():
    # Radiance's usual way of writing a polygon with a hole: the outer loop,
    # a seam in, the hole the other way round, the same seam back out. The
    # seam edges overlap but nothing crosses.
    keyhole = _poly("key", [
        (0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0), (0, 0, 0),
        (1, 1, 0), (1, 3, 0), (3, 3, 0), (3, 1, 0), (1, 1, 0),
    ])
    report = validate_scene([paint(), keyhole], "m")
    assert report.errors == ()


def test_a_surface_may_come_before_its_material():
    # [geometry.rad, materials.rad] is fine: the prepared model writes
    # every material into materials.rad, read ahead of the surfaces.
    report = validate_scene([square(), paint()], "m")
    assert report.errors == ()


def test_a_material_using_a_pattern_defined_after_it_is_refused():
    # Modifier definitions keep their reading order in materials.rad, so
    # a material that uses a later pattern would fail in oconv.
    pattern = Primitive("void", "brightfunc", "dim_pattern", ("0.5",), (), (), "b.rad")
    tinted = Primitive("dim_pattern", "plastic", "wall_paint", (), (),
                       (0.5, 0.5, 0.5, 0, 0), "a.rad")
    report = validate_scene([tinted, pattern, square()], "m")
    assert any("wall_paint" in e and "dim_pattern" in e and "before it is defined" in e
               for e in report.errors), report.errors


def test_a_surface_with_an_undefined_modifier_is_still_refused():
    report = validate_scene([square(modifier="ghost_paint"), paint()], "m")
    assert any("ghost_paint" in e and "never defined" in e for e in report.errors), report.errors


def test_a_material_modified_by_an_undefined_pattern_is_refused():
    tinted = Primitive("ghost_pattern", "plastic", "wall_paint", (), (),
                       (0.5, 0.5, 0.5, 0, 0), "test")
    report = validate_scene([tinted, square()], "m")
    assert any("ghost_pattern" in e for e in report.errors), report.errors


def test_a_modifier_defined_twice_differently_is_refused():
    other = Primitive("void", "plastic", "wall_paint", (), (), (0.2, 0.2, 0.2, 0, 0), "b.rad")
    report = validate_scene([paint(), other, square()], "m")
    assert any("wall_paint" in e and "defined twice" in e and "b.rad" in e
               for e in report.errors), report.errors


def test_the_same_definition_read_twice_is_not_refused():
    report = validate_scene([paint(), paint(), square()], "m")
    assert report.errors == ()
