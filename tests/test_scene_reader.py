import pathlib

import numpy as np
import pytest

from leed_daylight import geometry as g
from leed_daylight.scene.reader import XformError, read_scene

FIX = pathlib.Path(__file__).parent / "fixtures"


def test_the_same_scene_in_feet_and_metres_differs_by_the_foot_squared():
    prims_m = [p for p in read_scene([FIX / "wing.rad"], "m") if p.is_surface]
    prims_ft = [p for p in read_scene([FIX / "wing.rad"], "ft") if p.is_surface]
    area_m = g.area(prims_m[0].points)
    area_ft = g.area(prims_ft[0].points)
    # source: international foot, exactly 0.3048 m
    assert area_ft == pytest.approx(area_m * 0.3048**2, rel=1e-12)


def test_a_millimetre_scene_is_read_in_metres(tmp_path):
    scene = tmp_path / "s.rad"
    scene.write_text(
        "wall_paint polygon sq.1\n0\n0\n"
        "12  0 0 0   1000 0 0   1000 1000 0   0 1000 0\n"
    , newline="\n")
    prims = [p for p in read_scene([scene], "mm") if p.is_surface]
    assert g.area(prims[0].points) == pytest.approx(1.0)


def test_units_is_required():
    with pytest.raises(TypeError):
        read_scene([FIX / "simple_room.rad"])


def test_an_unknown_unit_is_refused():
    with pytest.raises(ValueError, match="unknown length unit"):
        read_scene([FIX / "simple_room.rad"], "furlong")


def test_the_factor_is_applied_after_the_transform(tmp_path):
    (tmp_path / "sq.rad").write_text(
        "wall_paint polygon sq.1\n0\n0\n"
        "12  0 0 0   1000 0 0   1000 1000 0   0 1000 0\n"
    , newline="\n")
    scene = tmp_path / "s.rad"
    scene.write_text("!xform -t 1000 0 0 sq.rad\n", newline="\n")
    prims = [p for p in read_scene([scene], "mm") if p.is_surface]
    xs = prims[0].points[:, 0]
    assert xs.min() == pytest.approx(1.0)
    assert xs.max() == pytest.approx(2.0)


def test_reads_every_primitive():
    prims = read_scene([FIX / "simple_room.rad"], "m")
    assert [p.name for p in prims] == [
        "wall_paint", "window_glass", "floor.1", "ceiling.1", "win.1"
    ]


def test_records_modifier_type_and_arguments():
    prims = {p.name: p for p in read_scene([FIX / "simple_room.rad"], "m")}
    paint = prims["wall_paint"]
    assert paint.modifier == "void" and paint.ptype == "plastic"
    assert paint.fargs == (0.5, 0.5, 0.5, 0.0, 0.0)
    assert paint.sargs == () and paint.iargs == ()


def test_polygon_points_have_the_right_shape():
    prims = {p.name: p for p in read_scene([FIX / "simple_room.rad"], "m")}
    pts = prims["floor.1"].points
    assert pts.shape == (4, 3)
    assert np.allclose(pts[2], [4, 3, 0])


def test_a_material_has_no_points():
    prims = {p.name: p for p in read_scene([FIX / "simple_room.rad"], "m")}
    assert prims["wall_paint"].points is None
    assert prims["wall_paint"].is_surface is False
    assert prims["floor.1"].is_surface is True


def test_comments_and_blank_lines_are_ignored():
    prims = read_scene([FIX / "simple_room.rad"], "m")
    assert all(not p.name.startswith("#") for p in prims)


def test_source_file_is_recorded():
    prims = read_scene([FIX / "simple_room.rad"], "m")
    assert prims[0].source.endswith("simple_room.rad")


def test_xform_translate():
    prims = [p for p in read_scene([FIX / "with_xform.rad"], "m") if p.is_surface]
    assert np.allclose(prims[0].points[0], [10, 0, 0])


def test_xform_rotate_then_translate():
    prims = [p for p in read_scene([FIX / "with_xform.rad"], "m") if p.is_surface]
    # -rz 90 sends (2, 0, 0) to (0, 2, 0); then -t 0 0 3 lifts it.
    assert np.allclose(prims[1].points[1], [0, 2, 3], atol=1e-9)


def test_xform_scale_scales_geometry(tmp_path):
    (tmp_path / "wing.rad").write_text((FIX / "wing.rad").read_text(), newline="\n")
    scene = tmp_path / "s.rad"
    scene.write_text("!xform -s 2 wing.rad\n", newline="\n")
    prims = [p for p in read_scene([scene], "m") if p.is_surface]
    assert np.allclose(prims[0].points[1], [4, 0, 0])


def test_xform_prefix_renames(tmp_path):
    (tmp_path / "wing.rad").write_text((FIX / "wing.rad").read_text(), newline="\n")
    scene = tmp_path / "s.rad"
    scene.write_text("!xform -n blockA wing.rad\n", newline="\n")
    prims = [p for p in read_scene([scene], "m") if p.is_surface]
    assert prims[0].name == "blockA.wing_floor.1"


def test_mirroring_keeps_the_area_and_flips_the_normal(tmp_path):
    from leed_daylight import geometry as g

    (tmp_path / "wing.rad").write_text((FIX / "wing.rad").read_text(), newline="\n")
    scene = tmp_path / "s.rad"
    scene.write_text("!xform -mx wing.rad\n", newline="\n")
    prims = [p for p in read_scene([scene], "m") if p.is_surface]
    assert g.area(prims[0].points) == pytest.approx(4.0)
    assert np.allclose(g.normal(prims[0].points), [0, 0, 1])


def test_an_unsupported_xform_option_is_refused(tmp_path):
    (tmp_path / "wing.rad").write_text((FIX / "wing.rad").read_text(), newline="\n")
    scene = tmp_path / "s.rad"
    scene.write_text("!xform -a 4 -t 1 0 0 wing.rad\n", newline="\n")
    with pytest.raises(XformError, match="-a"):
        read_scene([scene], "m")


def test_a_non_xform_command_is_refused(tmp_path):
    scene = tmp_path / "s.rad"
    scene.write_text("!genbox void b 1 1 1\n", newline="\n")
    with pytest.raises(XformError, match="genbox"):
        read_scene([scene], "m")


def test_a_truncated_primitive_is_refused(tmp_path):
    scene = tmp_path / "s.rad"
    scene.write_text("void plastic p\n0\n0\n5 0.5 0.5\n", newline="\n")
    with pytest.raises(ValueError, match="ran out"):
        read_scene([scene], "m")


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_scene([tmp_path / "nope.rad"], "m")


def test_a_sphere_inside_a_transform_is_refused(tmp_path):
    (tmp_path / "dome.rad").write_text("void sphere ball.1\n0\n0\n4  0 0 0  1\n", newline="\n")
    scene = tmp_path / "s.rad"
    scene.write_text("!xform -t 1 0 0 dome.rad\n", newline="\n")
    with pytest.raises(XformError) as exc_info:
        read_scene([scene], "m")
    message = str(exc_info.value)
    assert "ball.1" in message
    assert "sphere" in message


def test_a_cylinder_inside_a_rotation_is_refused(tmp_path):
    (tmp_path / "pipe.rad").write_text(
        "void cylinder tube.1\n0\n0\n7  0 0 0  0 0 1  0.5\n"
    , newline="\n")
    scene = tmp_path / "s.rad"
    scene.write_text("!xform -rz 90 pipe.rad\n", newline="\n")
    with pytest.raises(XformError) as exc_info:
        read_scene([scene], "m")
    assert "cylinder" in str(exc_info.value)


def test_a_sphere_outside_any_transform_is_read(tmp_path):
    scene = tmp_path / "s.rad"
    scene.write_text("void sphere ball.1\n0\n0\n4  0 0 0  1\n", newline="\n")
    prims = read_scene([scene], "m")
    assert len(prims) == 1
    assert prims[0].fargs == (0.0, 0.0, 0.0, 1.0)


def test_a_polygon_inside_a_transform_is_still_transformed():
    prims = [p for p in read_scene([FIX / "with_xform.rad"], "m") if p.is_surface]
    assert np.allclose(prims[0].points[0], [10, 0, 0])
    # -rz 90 sends (2, 0, 0) to (0, 2, 0); then -t 0 0 3 lifts it.
    assert np.allclose(prims[1].points[1], [0, 2, 3], atol=1e-9)


def test_a_malformed_xform_option_names_the_file(tmp_path):
    (tmp_path / "wing.rad").write_text((FIX / "wing.rad").read_text(), newline="\n")
    scene = tmp_path / "s.rad"
    scene.write_text("!xform -t 1 2 wing.rad\n", newline="\n")
    with pytest.raises(XformError) as exc_info:
        read_scene([scene], "m")
    message = str(exc_info.value)
    assert str(scene) in message
    assert "could not convert string to float" not in message


def test_a_non_numeric_argument_count_names_the_file(tmp_path):
    scene = tmp_path / "s.rad"
    scene.write_text("void plastic p\n0\n0\nx 0.5 0.5 0.5 0.0 0.0\n", newline="\n")
    with pytest.raises(ValueError) as exc_info:
        read_scene([scene], "m")
    assert str(scene) in str(exc_info.value)


def test_an_unsupported_xform_option_names_the_file(tmp_path):
    # _parse_xform's own refusal ("-a is not supported") used to reach the
    # caller as a bare `raise`, so it never got the file-qualifying prefix
    # every other reader error has.
    (tmp_path / "wing.rad").write_text((FIX / "wing.rad").read_text(), newline="\n")
    scene = tmp_path / "s.rad"
    scene.write_text("!xform -a 4 -t 1 0 0 wing.rad\n", newline="\n")
    with pytest.raises(XformError) as exc_info:
        read_scene([scene], "m")
    message = str(exc_info.value)
    assert str(scene) in message
    assert "-a" in message


def test_a_missing_file_argument_names_the_file(tmp_path):
    scene = tmp_path / "s.rad"
    scene.write_text("!xform -t 1 0 0\n", newline="\n")
    with pytest.raises(XformError) as exc_info:
        read_scene([scene], "m")
    message = str(exc_info.value)
    assert str(scene) in message
    assert "no file argument" in message


def test_xform_reads_every_file_it_names_in_order(tmp_path):
    (tmp_path / "a.rad").write_text("paint polygon a.1\n0\n0\n12 0 0 0 1 0 0 1 1 0 0 1 0\n", newline="\n")
    (tmp_path / "b.rad").write_text("paint polygon b.1\n0\n0\n12 0 0 0 2 0 0 2 2 0 0 2 0\n", newline="\n")
    scene = tmp_path / "s.rad"
    scene.write_text("!xform -n blk -t 10 0 0 a.rad b.rad\n", newline="\n")
    prims = [p for p in read_scene([scene], "m") if p.is_surface]
    assert [p.name for p in prims] == ["blk.a.1", "blk.b.1"]
    assert np.allclose(prims[0].points[0], [10, 0, 0])
    assert np.allclose(prims[1].points[1], [12, 0, 0])


def test_xform_prefix_renames_surfaces_only_as_radiance_does(tmp_path):
    # xform.c prefixes the names of non-modifier objects only; a material's
    # name and every line's modifier field are left alone.
    (tmp_path / "part.rad").write_text(
        "void plastic paint\n0\n0\n5 0.5 0.5 0.5 0 0\n"
        "paint polygon floor.1\n0\n0\n12 0 0 0 1 0 0 1 1 0 0 1 0\n"
    , newline="\n")
    scene = tmp_path / "s.rad"
    scene.write_text("!xform -n blk part.rad\n", newline="\n")
    material, surface = read_scene([scene], "m")
    assert material.name == "paint"
    assert material.modifier == "void"
    assert surface.name == "blk.floor.1"
    assert surface.modifier == "paint"


def test_nested_xform_prefixes_join_on_surfaces_only(tmp_path):
    (tmp_path / "inner.rad").write_text(
        "void plastic paint\n0\n0\n5 0.5 0.5 0.5 0 0\n"
        "paint polygon floor.1\n0\n0\n12 0 0 0 1 0 0 1 1 0 0 1 0\n"
    , newline="\n")
    (tmp_path / "mid.rad").write_text("!xform -n b inner.rad\n", newline="\n")
    scene = tmp_path / "s.rad"
    scene.write_text("!xform -n a mid.rad\n", newline="\n")
    material, surface = read_scene([scene], "m")
    assert material.name == "paint"
    assert surface.name == "a.b.floor.1"
    assert surface.modifier == "paint"


def test_a_void_alias_is_a_copy_of_the_material_it_names(tmp_path):
    scene = tmp_path / "s.rad"
    scene.write_text(
        "void glass clear\n0\n0\n3 0.7 0.7 0.7\n"
        "void alias pane clear\n"
        "pane polygon w.1\n0\n0\n12 0 0 0 1 0 0 1 0 1 0 0 1\n"
    , newline="\n")
    clear, pane, window = read_scene([scene], "m")
    assert pane.name == "pane"
    assert pane.ptype == "glass"
    assert pane.fargs == clear.fargs
    assert pane.modifier == "void"
    assert window.modifier == "pane"


def test_an_alias_with_a_modifier_takes_that_modifier(tmp_path):
    scene = tmp_path / "s.rad"
    scene.write_text(
        "void plastic paint\n0\n0\n5 0.5 0.5 0.5 0 0\n"
        "void brightfunc dim\n2 dimfunc dim.cal\n0\n0\n"
        "dim alias dimpaint paint\n"
    , newline="\n")
    *_, alias = read_scene([scene], "m")
    assert alias.name == "dimpaint"
    assert alias.ptype == "plastic"
    assert alias.modifier == "dim"


def test_an_alias_to_an_undefined_name_is_refused(tmp_path):
    scene = tmp_path / "s.rad"
    scene.write_text("void alias pane clear\n", newline="\n")
    with pytest.raises(ValueError) as exc_info:
        read_scene([scene], "m")
    message = str(exc_info.value)
    assert str(scene) in message
    assert "pane" in message and "clear" in message


def test_an_xform_line_continued_with_a_backslash_is_read(tmp_path):
    (tmp_path / "wing.rad").write_text((FIX / "wing.rad").read_text(), newline="\n")
    scene = tmp_path / "s.rad"
    scene.write_text("!xform -t 10 0 0 \\\n    wing.rad\n", newline="\n")
    prims = [p for p in read_scene([scene], "m") if p.is_surface]
    assert np.allclose(prims[0].points[0], [10, 0, 0])


def test_bad_quoting_on_an_xform_line_names_the_file_and_line(tmp_path):
    scene = tmp_path / "s.rad"
    scene.write_text("# header\n!xform -n \"blk wing.rad\n", newline="\n")
    with pytest.raises(XformError) as exc_info:
        read_scene([scene], "m")
    message = str(exc_info.value)
    assert str(scene) in message
    assert "line 2" in message


def test_a_quoted_string_argument_is_one_word(tmp_path):
    scene = tmp_path / "s.rad"
    scene.write_text("void BSDF pane\n6 0 \"pane data.xml\" 0 0 1 .\n0\n0\n", newline="\n")
    [prim] = read_scene([scene], "m")
    assert prim.sargs == ("0", "pane data.xml", "0", "0", "1", ".")


def test_an_unterminated_quote_in_a_primitive_names_the_file_and_line(tmp_path):
    scene = tmp_path / "s.rad"
    scene.write_text("void BSDF pane\n6 0 \"pane.xml 0 0 1 .\n0\n0\n", newline="\n")
    with pytest.raises(ValueError) as exc_info:
        read_scene([scene], "m")
    message = str(exc_info.value)
    assert str(scene) in message
    assert "line 2" in message
    assert "quote" in message


def test_a_quote_inside_a_word_is_an_ordinary_character(tmp_path):
    # Radiance's fgetword treats a quote as special only at a word's start.
    scene = tmp_path / "s.rad"
    scene.write_text("void brightfunc dim\n2 it's x\"y\n0\n0\n", newline="\n")
    [prim] = read_scene([scene], "m")
    assert prim.sargs == ("it's", 'x"y')


def test_quote_word_round_trips_every_awkward_word(tmp_path):
    from leed_daylight.scene.reader import quote_word

    words = ["plain", "two words", "has#hash", "it's here", 'say "hi"', "'lead", "it's", ""]
    scene = tmp_path / "s.rad"
    scene.write_text(
        "void brightfunc dim\n"
        + " ".join([str(len(words)), *(quote_word(w) for w in words)])
        + "\n0\n0\n"
    , newline="\n")
    [prim] = read_scene([scene], "m")
    assert prim.sargs == tuple(words)


def test_quote_word_refuses_a_word_no_quote_can_hold(tmp_path):
    from leed_daylight.scene.reader import quote_word

    with pytest.raises(ValueError, match="both"):
        quote_word("it's \"odd\"")


_POLY = "m polygon p\n0\n0\n9 0 0 0 1 0 0 1 1 0\n"


def test_a_chain_of_includes_deeper_than_the_cap_is_refused(tmp_path):
    from leed_daylight.scene import reader
    depth = reader.MAX_INCLUDE_DEPTH + 2
    for n in range(depth):
        (tmp_path / f"f{n}.rad").write_text(f"!xform f{n + 1}.rad\n", newline="\n")
    (tmp_path / f"f{depth}.rad").write_text(_POLY, newline="\n")
    with pytest.raises(XformError, match="nest too deep"):
        read_scene([tmp_path / "f0.rad"], "m")


def test_includes_that_double_at_every_level_stop_at_the_primitive_cap(tmp_path, monkeypatch):
    from leed_daylight.scene import reader
    monkeypatch.setattr(reader, "MAX_PRIMITIVES", 1000)
    for n in range(20):
        (tmp_path / f"f{n}.rad").write_text(f"!xform f{n + 1}.rad\n!xform f{n + 1}.rad\n", newline="\n")
    (tmp_path / "f20.rad").write_text(_POLY, newline="\n")
    with pytest.raises(XformError, match="primitives"):
        read_scene([tmp_path / "f0.rad"], "m")


def test_a_nest_within_the_caps_still_reads(tmp_path):
    (tmp_path / "a.rad").write_text("!xform b.rad\n!xform b.rad\n", newline="\n")
    (tmp_path / "b.rad").write_text(_POLY, newline="\n")
    assert len(read_scene([tmp_path / "a.rad"], "m")) == 2


def test_a_doubling_chain_of_empty_includes_is_refused_quickly(tmp_path):
    for n in range(30):
        (tmp_path / f"f{n}.rad").write_text(f"!xform f{n + 1}.rad\n!xform f{n + 1}.rad\n", newline="\n")
    (tmp_path / "f30.rad").write_text("", newline="\n")
    with pytest.raises(XformError, match="reads more than"):
        read_scene([tmp_path / "f0.rad"], "m")
