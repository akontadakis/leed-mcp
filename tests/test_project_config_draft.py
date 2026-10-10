# tests/test_project_config_draft.py
"""The wizard's draft check and scene-only reader (wizard spec 7.3, W9).

``check_project_draft`` runs every check of ``load_project`` except the
whole-file FILL_ME refusal: a value still FILL_ME is listed as unfinished,
never reported, and a check that needs it is skipped. ``load_scene_settings``
reads only ``scene:``. ``load_project`` itself still refuses any FILL_ME.

Every folder is built fresh in ``tmp_path``; every value is invented.
"""
from __future__ import annotations

import copy
import re

import pytest
import yaml

from leed_daylight.constants import FILL_ME
from leed_daylight.drivers.init import init_folder
from leed_daylight.project.config import (
    ProjectError,
    SceneSpec,
    check_project_draft,
    load_project,
    load_scene_settings,
)
from leed_daylight.project.materials import EXTERIOR_GLAZING_ROLES, ROLES

SITE = {
    "latitude_N": 42.0, "longitude_E": -80.0, "meridian_radiance": 75.0,
    "north_rotation_deg": 10.0, "north_rotation_source": "Invented: the site plan north arrow",
}
GRID = {"spacing_m": 0.6, "work_plane_m": 0.762, "min_clearance_m": 0.3}


def _fill_me_leaves(value, path: str = "") -> list[str]:
    """Every path whose value is FILL_ME, in file order, written as
    ``load_project`` names them (``scene.files[0]``)."""
    if isinstance(value, dict):
        found = []
        for key, item in value.items():
            found += _fill_me_leaves(item, f"{path}.{key}" if path else str(key))
        return found
    if isinstance(value, list):
        found = []
        for index, item in enumerate(value):
            found += _fill_me_leaves(item, f"{path}[{index}]")
        return found
    return [path] if value == FILL_ME else []


_STEP = re.compile(r"\.?([^.\[\]]+)|\[(\d+)\]")


def _set(raw: dict, path: str, value) -> None:
    """Set ``raw`` at a path such as ``atlas.drawings[0].points[1].pixel``."""
    steps = [int(index) if name == "" else name for name, index in _STEP.findall(path)]
    target = raw
    for step in steps[:-1]:
        target = target[step]
    target[steps[-1]] = value


def _write(folder, raw: dict):
    path = folder / "project.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), newline="\n")
    return path


def _new_project(tmp_path):
    """A folder as ``init`` writes it: the template, with every value FILL_ME."""
    folder = init_folder(tmp_path / "new")
    return folder, yaml.safe_load((folder / "project.yaml").read_text())


# --- check_project_draft ----------------------------------------------------

def test_the_template_has_no_error_and_every_fill_me_is_unfinished(tmp_path):
    folder, raw = _new_project(tmp_path)

    errors, unfinished = check_project_draft(folder / "project.yaml")

    assert errors == ()
    assert unfinished == tuple(_fill_me_leaves(raw))
    assert unfinished[:3] == ("project.name", "project.credit", "site.latitude_N")
    assert "standard.credit_rules" not in unfinished


def test_a_complete_project_has_nothing_to_report(example_project_dir):
    path = example_project_dir / "project.yaml"

    assert check_project_draft(path) == ((), ())
    load_project(path)


def test_a_partial_file_lists_the_rest_as_unfinished(tmp_path):
    folder, raw = _new_project(tmp_path)
    raw["site"] = dict(SITE)
    raw["grid"] = dict(GRID)
    path = _write(folder, raw)

    errors, unfinished = check_project_draft(path)

    assert errors == ()
    assert unfinished == tuple(_fill_me_leaves(raw))
    assert not [p for p in unfinished if p.startswith(("site.", "grid."))]
    assert "analysis.hours" in unfinished


def test_a_filled_value_out_of_range_is_an_error_among_unfinished_keys(tmp_path):
    folder, raw = _new_project(tmp_path)
    raw["grid"] = dict(GRID, spacing_m=0.9)
    path = _write(folder, raw)

    errors, unfinished = check_project_draft(path)

    assert len(errors) == 1
    assert errors[0].startswith("grid.spacing_m must be greater than zero")
    assert "0.9" in errors[0]
    assert "site.latitude_N" in unfinished


def test_a_typing_error_is_the_one_error(tmp_path):
    folder, raw = _new_project(tmp_path)
    raw["grid"] = dict(GRID, spacing_m="wide")
    raw["sky"]["MF"] = 5
    path = _write(folder, raw)

    errors, unfinished = check_project_draft(path)

    assert errors == ("project.yaml: grid.spacing_m must be a number, got 'wide'",)
    assert "site.latitude_N" in unfinished


def test_a_cross_check_on_an_unfinished_key_is_skipped(example_project_dir):
    raw = yaml.safe_load((example_project_dir / "project.yaml").read_text())
    raw["option2"]["lux_min"] = 5000
    raw["option2"]["lux_max"] = FILL_ME
    path = _write(example_project_dir, raw)

    assert check_project_draft(path) == ((), ("option2.lux_max",))

    raw["option2"]["lux_max"] = 3000
    _write(example_project_dir, raw)
    errors, unfinished = check_project_draft(path)
    assert unfinished == ()
    assert len(errors) == 1 and errors[0].startswith("option2.lux_min must be <= option2.lux_max")


def test_an_error_on_a_sibling_of_an_unfinished_key_is_kept(example_project_dir):
    raw = yaml.safe_load((example_project_dir / "project.yaml").read_text())
    raw["glazing"]["frame_share"]["default"] = 1.5
    raw["glazing"]["frame_share"]["glazing_curtain_wall"] = FILL_ME
    raw["scene"]["files"] = [FILL_ME, "missing.rad"]
    path = _write(example_project_dir, raw)

    errors, unfinished = check_project_draft(path)

    assert unfinished == ("scene.files[0]", "glazing.frame_share.glazing_curtain_wall")
    assert [e.split(" ")[0] for e in errors] == ["scene.files[1]:", "glazing.frame_share.default"]


def test_an_error_on_an_ancestor_of_an_unfinished_key_is_skipped(example_project_dir):
    raw = yaml.safe_load((example_project_dir / "project.yaml").read_text())
    raw["atlas"] = {"drawings": [{
        "storey": "L0", "image": "drawings/L0.png",
        "points": [{"pixel": [0.0, 0.0], "model": [0.0, 0.0]},
                   {"pixel": [FILL_ME, 5.0], "model": [6.0, 0.0]}],
    }]}
    path = _write(example_project_dir, raw)

    assert check_project_draft(path) == ((), ("atlas.drawings[0].points[1].pixel[0]",))

    raw["atlas"]["drawings"][0]["points"][1]["pixel"][0] = 3.0
    _write(example_project_dir, raw)
    errors, _ = check_project_draft(path)
    assert len(errors) == 1 and errors[0].startswith("atlas.drawings[0].points: the two pixels are")


def test_a_pixel_error_is_kept_while_a_model_value_is_unfinished(example_project_dir):
    # Both checks share the identical ancestor text "atlas.drawings[0].points:";
    # only the wording after it says which one fired. A real, filled pixel
    # error must survive even though the model branch of the same pair is
    # still FILL_ME (and, filled in with its stand-in, would itself look
    # like a genuine -- but not real -- model error, correctly skipped).
    raw = yaml.safe_load((example_project_dir / "project.yaml").read_text())
    raw["atlas"] = {"drawings": [{
        "storey": "L0", "image": "drawings/L0.png",
        "points": [{"pixel": [0.0, 0.0], "model": [0.0, 0.0]},
                   {"pixel": [5.0, 0.0], "model": [FILL_ME, 0.0]}],
    }]}
    path = _write(example_project_dir, raw)

    errors, unfinished = check_project_draft(path)

    assert unfinished == ("atlas.drawings[0].points[1].model[0]",)
    assert len(errors) == 1 and errors[0].startswith("atlas.drawings[0].points: the two pixels are")


def test_a_model_error_is_kept_while_a_pixel_value_is_unfinished(example_project_dir):
    raw = yaml.safe_load((example_project_dir / "project.yaml").read_text())
    raw["atlas"] = {"drawings": [{
        "storey": "L0", "image": "drawings/L0.png",
        "points": [{"pixel": [0.0, 0.0], "model": [0.0, 0.0]},
                   {"pixel": [FILL_ME, 0.0], "model": [0.3, 0.0]}],
    }]}
    path = _write(example_project_dir, raw)

    errors, unfinished = check_project_draft(path)

    assert unfinished == ("atlas.drawings[0].points[1].pixel[0]",)
    assert len(errors) == 1 and errors[0].startswith("atlas.drawings[0].points: the two model points are")


def test_a_pixel_error_is_still_skipped_when_a_pixel_value_is_itself_unfinished(example_project_dir):
    raw = yaml.safe_load((example_project_dir / "project.yaml").read_text())
    raw["atlas"] = {"drawings": [{
        "storey": "L0", "image": "drawings/L0.png",
        "points": [{"pixel": [0.0, 0.0], "model": [0.0, 0.0]},
                   {"pixel": [5.0, FILL_ME], "model": [6.0, 0.0]}],
    }]}
    path = _write(example_project_dir, raw)

    assert check_project_draft(path) == ((), ("atlas.drawings[0].points[1].pixel[1]",))


def _every_key_project(tmp_path, write_example):
    """The example with every optional and commented key in use."""
    def mutate(d):
        d["glazing"]["dirt_factor"] = {"vertical": 0.9, "sloped": 0.9, "skylight": 0.8}
        for role in sorted(EXTERIOR_GLAZING_ROLES) + ["sloped", "skylight"]:
            d["glazing"]["frame_share"][role] = 0.2
        d["glazing"]["skylight_well"] = 0.4
        d["glazing"]["skylight_well_reflectance"] = 0.7
        for role in sorted(ROLES):
            d["reflectance"][role] = 0.5
        d["performance"]["linked_zones"] = {"L0": ["zone a", "zone b"]}
        d["grid"]["points_file"] = "points.csv"
        (tmp_path / "project" / "points.csv").write_text("space,x,y,z\n", encoding="utf-8", newline="\n")
        d["blinds"]["trigger_area"] = "grid"
        d["views"] = {"eye_height_m": 1.2, "eye_height_source": "Invented: seated eye height",
                      "max_interior_glass": 1, "angular_step_deg": 2.0}
        d["report"] = {"renderer": "pillow", "deviations": ["An invented deviation."],
                       "prepared_by": "An invented author"}
        d["atlas"] = {"drawings": [{
            "storey": "L0", "image": "drawings/L0.png", "opacity": 0.5,
            "points": [{"pixel": [10.0, 10.0], "model": [0.0, 0.0]},
                       {"pixel": [610.0, 10.0], "model": [6.0, 0.0]}],
        }]}
        return d

    return write_example(tmp_path, mutate)


def _every_path(value, path: str = "") -> list[str]:
    """Every path in ``value``: each mapping key, each list item, each leaf."""
    found = [path] if path else []
    if isinstance(value, dict):
        for key, item in value.items():
            found += _every_path(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found += _every_path(item, f"{path}[{index}]")
    return found


def test_every_key_the_loaders_read_has_a_stand_in(tmp_path, write_example):
    path = _every_key_project(tmp_path, write_example)
    complete = yaml.safe_load(path.read_text())
    assert check_project_draft(path) == ((), ())
    sections = set(complete)
    paths = [p for p in _every_path(complete) if p not in sections]
    template = yaml.safe_load(init_folder(tmp_path / "template").joinpath("project.yaml").read_text())
    assert set(_fill_me_leaves(template)) <= set(paths)
    assert {
        "glazing.frame_share.skylight", "glazing.dirt_factor.sloped", "reflectance.furniture",
        "performance.linked_zones.L0[1]", "atlas.drawings[0].points[1].model[0]",
        "report.deviations[0]", "views.angular_step_deg", "grid.points_file",
        "blinds.trigger_area",
    } <= set(paths)

    for key in paths:
        raw = copy.deepcopy(complete)
        _set(raw, key, FILL_ME)
        _write(path.parent, raw)

        assert check_project_draft(path) == ((), (key,)), key


def test_the_draft_check_writes_nothing(tmp_path):
    folder, raw = _new_project(tmp_path)
    raw["grid"] = dict(GRID, spacing_m=0.9)
    _write(folder, raw)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.rglob("*") if p.is_file()}
    folders = sorted(p for p in folder.rglob("*") if p.is_dir())

    check_project_draft(folder / "project.yaml")

    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.rglob("*") if p.is_file()}
    assert after == before
    assert sorted(p for p in folder.rglob("*") if p.is_dir()) == folders


def test_yaml_that_does_not_parse_is_one_error(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text("project: [unclosed\n", newline="\n")

    errors, unfinished = check_project_draft(path)

    assert unfinished == ()
    assert len(errors) == 1 and errors[0].startswith("project.yaml is not valid YAML: ")


def test_a_top_level_that_is_not_a_mapping_is_one_error(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text("- a list\n", newline="\n")

    assert check_project_draft(path) == (("project.yaml: project.yaml must be a mapping at the top level",), ())


def test_a_missing_section_is_an_error_as_in_load_project(tmp_path):
    folder, raw = _new_project(tmp_path)
    del raw["blinds"]
    path = _write(folder, raw)

    errors, _ = check_project_draft(path)

    assert errors == ("project.yaml: missing required key 'blinds'",)
    with pytest.raises(ProjectError, match="still say FILL_ME"):
        load_project(path)


# --- load_scene_settings ------------------------------------------------------

def test_a_complete_scene_section_is_read(example_project_dir):
    scene = load_scene_settings(example_project_dir / "project.yaml")

    assert scene == SceneSpec(units="m", files=(example_project_dir / "scene.rad",))


def test_the_rest_of_the_file_may_still_be_unfinished(tmp_path):
    folder, raw = _new_project(tmp_path)
    (folder / "model" / "geometry.rad").write_text("", newline="\n")
    raw["scene"] = {"units": "mm", "files": ["model/geometry.rad"]}
    path = _write(folder, raw)

    assert load_scene_settings(path) == SceneSpec(units="mm", files=(folder / "model" / "geometry.rad",))


def test_fill_me_in_scene_units_is_refused_by_path(example_project_dir):
    raw = yaml.safe_load((example_project_dir / "project.yaml").read_text())
    raw["scene"]["units"] = FILL_ME
    path = _write(example_project_dir, raw)

    with pytest.raises(ProjectError) as caught:
        load_scene_settings(path)

    assert str(caught.value) == (
        "project.yaml: scene.units still says FILL_ME; set it in project.yaml"
    )


def test_every_fill_me_in_the_scene_is_listed(tmp_path):
    folder, _ = _new_project(tmp_path)

    with pytest.raises(ProjectError) as caught:
        load_scene_settings(folder / "project.yaml")

    assert str(caught.value) == (
        "project.yaml: scene.units, scene.files still say FILL_ME; set them in project.yaml"
    )


def test_an_empty_file_list_is_refused(example_project_dir):
    raw = yaml.safe_load((example_project_dir / "project.yaml").read_text())
    raw["scene"]["files"] = []
    path = _write(example_project_dir, raw)

    with pytest.raises(ProjectError, match=r"^project\.yaml: scene\.files must name at least one \.rad file$"):
        load_scene_settings(path)


def test_a_listed_file_that_does_not_exist_is_refused(example_project_dir):
    raw = yaml.safe_load((example_project_dir / "project.yaml").read_text())
    raw["scene"]["files"] = ["scene.rad", "model/missing.rad"]
    path = _write(example_project_dir, raw)

    with pytest.raises(ProjectError) as caught:
        load_scene_settings(path)

    assert str(caught.value) == (
        "project.yaml: scene.files[1]: file does not exist or is not readable: "
        f"{example_project_dir / 'model' / 'missing.rad'}"
    )


def test_a_file_without_a_scene_section_is_refused(example_project_dir):
    raw = yaml.safe_load((example_project_dir / "project.yaml").read_text())
    del raw["scene"]
    path = _write(example_project_dir, raw)

    with pytest.raises(ProjectError, match=r"missing required key 'scene'"):
        load_scene_settings(path)


def test_load_project_still_refuses_any_fill_me(tmp_path):
    folder, raw = _new_project(tmp_path)
    (folder / "model" / "geometry.rad").write_text("", newline="\n")
    raw["scene"] = {"units": "m", "files": ["model/geometry.rad"]}
    path = _write(folder, raw)
    load_scene_settings(path)

    with pytest.raises(ProjectError, match=r"value\(s\) still say FILL_ME; fill them in: project\.name, "):
        load_project(path)
