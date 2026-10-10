# tests/test_atlas_blender_build.py
"""The Blender build script, without Blender (atlas spec 12, 15).

``blender_build.py`` runs only inside Blender; these tests read it as a
module and as source. It must import outside Blender without ``bpy``,
compile, import nothing from the package (Blender cannot import it), read
only arrays that ``write_npz`` writes, repeat the bundle's storey codes and
the kinds exactly, and hold no project value. The real Blender run is the
opt-in smoke test (``tests/test_atlas_blender_smoke.py``). Every number is
invented and round.
"""
from __future__ import annotations

import ast
import importlib
import pathlib
import py_compile
import re
import sys

import numpy as np
import pytest

from atlas_folder import atlas_inputs
from deliver_folder import build_two_storey_project
from leed_daylight.deliver.atlas import bundle as bundle_module
from leed_daylight.deliver.atlas.bundle import write_npz
from leed_daylight.deliver.atlas.colours import KINDS
from leed_daylight.engine import blender as engine_blender

SCRIPT = pathlib.Path(bundle_module.__file__).resolve().parent / "blender_build.py"


def _tree() -> ast.Module:
    return ast.parse(SCRIPT.read_text(encoding="utf-8"))


def _module():
    return importlib.import_module("leed_daylight.deliver.atlas.blender_build")


def test_the_module_imports_without_bpy():
    module = _module()
    assert "bpy" not in sys.modules
    assert callable(module.main)
    with pytest.raises(RuntimeError, match="runs only inside Blender 4.2 or newer"):
        module.main(["bundle.npz", "model.blend.part"])


def test_bpy_is_imported_only_inside_main():
    tree = _tree()
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all(alias.name != "bpy" for n in top for alias in n.names)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    inside = [alias.name for n in ast.walk(main) if isinstance(n, ast.Import) for alias in n.names]
    assert "bpy" in inside


def test_it_byte_compiles(tmp_path):
    py_compile.compile(str(SCRIPT), cfile=str(tmp_path / "blender_build.pyc"), doraise=True)


def test_it_imports_nothing_from_the_package():
    modules = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import"
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    assert modules == {"__future__", "sys", "bpy", "numpy"}


def _key_patterns() -> list[tuple[bool, str, re.Pattern]]:
    """Every key the script reads through ``_array(npz, key)``: (is an
    f-string, its source text, its pattern); in an f-string each ``{...}``
    matches any text."""
    found = []
    for node in ast.walk(_tree()):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_array"):
            continue
        key = node.args[1]
        if isinstance(key, ast.Constant):
            found.append((False, key.value, re.compile(re.escape(key.value))))
        else:
            assert isinstance(key, ast.JoinedStr), ast.dump(key)
            parts = [re.escape(v.value) if isinstance(v, ast.Constant) else ".+" for v in key.values]
            found.append((True, ast.unparse(key), re.compile("".join(parts))))
    return found


def test_it_reads_only_arrays_that_write_npz_writes(tmp_path, write_example, write_wea):
    folder = build_two_storey_project(tmp_path, write_example, write_wea)
    _, _, bundle = atlas_inputs(folder, tmp_path / "work")
    write_npz(bundle, tmp_path / "bundle.npz")
    with np.load(tmp_path / "bundle.npz", allow_pickle=False) as npz:
        written = set(npz.files)
    patterns = _key_patterns()
    for _, text, pattern in patterns:
        assert any(pattern.fullmatch(key) for key in written), f"the script reads {text}, which is never written"
    assert {text for formatted, text, _ in patterns if not formatted} == {
        "storeys", "levels", "cell_m", "work_plane_m", "kinds", "kind_rgba", "grid_v", "grid_s",
        "room_xyz", "room_s", "room_included", "room_names", "layer_titles", "layer_keys",
    }
    assert {text for formatted, text, _ in patterns if formatted} == {
        "f'k_{kind}_v'", "f'k_{kind}_n'", "f'k_{kind}_s'",
        "f'l{i}_v'", "f'l{i}_s'", "f'l{i}_value'", "f'l{i}_rgb'",
    }


def test_the_storey_codes_and_kinds_agree_with_the_package():
    module = _module()
    assert (module.SITE, module.BELOW) == (bundle_module.SITE, bundle_module.BELOW)
    assert module.MIN_VERSION == engine_blender.MIN_VERSION
    assert tuple(module.TOP_OF_KIND) == KINDS
    assert module.TOPS == ("MODEL", "ADDED", "CONTEXT", "ROOMS", "RESULTS")
    by_top = {top: {k for k, t in module.TOP_OF_KIND.items() if t == top} for top in module.TOPS}
    assert by_top["ADDED"] == {"frame", "reveal", "skylight_well", "blind"}
    assert by_top["CONTEXT"] == {"context", "ground"}
    assert by_top["MODEL"] == set(KINDS) - by_top["ADDED"] - by_top["CONTEXT"]
    assert by_top["ROOMS"] == by_top["RESULTS"] == set()


def test_glass_is_drawn_blended_where_blender_has_it():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'if hasattr(mat, "surface_render_method"):' in text
    assert 'mat.surface_render_method = "BLENDED"' in text


def test_the_script_arguments_come_after_the_double_dash():
    module = _module()
    assert module.script_args(["blender", "-b", "--python", "blender_build.py", "--", "a.npz", "b.part"]) == [
        "a.npz", "b.part"]
    assert module.script_args(["blender", "-b"]) == []


def test_colours_go_from_srgb_bytes_to_linear():
    module = _module()
    linear = module._linear([[0, 255, 10]])
    assert linear.shape == (1, 3)
    assert linear[0, 0] == 0.0 and linear[0, 1] == pytest.approx(1.0)
    assert linear[0, 2] == pytest.approx(10 / 255 / 12.92)
    assert module._linear([128])[0] == pytest.approx(((128 / 255 + 0.055) / 1.055) ** 2.4)


def test_a_tile_is_a_square_around_its_point():
    verts, counts = _module()._tiles([[10.0, 20.0, 1.0], [0.0, 0.0, 0.0]], 2.0, 0.5)
    assert counts.tolist() == [4, 4]
    assert verts[:4].tolist() == [[9.0, 19.0, 1.5], [11.0, 19.0, 1.5], [11.0, 21.0, 1.5], [9.0, 21.0, 1.5]]


def test_it_holds_no_project_value():
    text = SCRIPT.read_text(encoding="utf-8")
    for value in ("0.762", "3,650", "3650"):
        assert value not in text
    assert not re.search(r"(?<![\d.])0\.6(?!\d)", text), "a fixed grid spacing"
    assert not re.search(r"""(["'])[A-Z0-9]{1,3}\1(\s*,\s*(["'])[A-Z0-9]{1,3}\3){2,}""", text), "a storey code list"
    assert "GROUND FLOOR" not in text.upper()
    assert "http:" not in text and "https:" not in text
