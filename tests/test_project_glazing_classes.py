# tests/test_project_glazing_classes.py
"""project.yaml `glazing:` keys for sloped and roof glass (skylights spec 4.2).

Every project is the shared fictional example (tests/fixtures/
project_example.yaml) with its `glazing:` section changed. Every value is
invented and round.
"""
from __future__ import annotations

import types

import pytest

from leed_daylight.build.blinds import BLIND_OFFSET_M
from leed_daylight.project import checks
from leed_daylight.project.config import DIRT_CLASSES, Glazing, ProjectError, load_project


def _load(write_example, tmp_path, **changes):
    def mutate(d):
        d["glazing"].update(changes)
        return d

    return load_project(write_example(tmp_path, mutate=mutate))


def _refused(write_example, tmp_path, **changes) -> str:
    with pytest.raises(ProjectError) as excinfo:
        _load(write_example, tmp_path, **changes)
    return str(excinfo.value)


# -- dirt_factor: one number, or one per class -------------------------------


def test_the_dirt_classes():
    assert DIRT_CLASSES == ("vertical", "sloped", "skylight")


def test_a_dirt_factor_per_class_loads(write_example, tmp_path):
    project = _load(
        write_example, tmp_path,
        dirt_factor={"vertical": 0.95, "sloped": 0.9, "skylight": 0.85},
    )
    glazing = project.glazing
    assert isinstance(glazing.dirt_factor, types.MappingProxyType)
    assert dict(glazing.dirt_factor) == {"vertical": 0.95, "sloped": 0.9, "skylight": 0.85}
    assert glazing.dirt_for("vertical") == 0.95
    assert glazing.dirt_for("sloped") == 0.9
    assert glazing.dirt_for("skylight") == 0.85


def test_one_dirt_factor_serves_every_class(example_project_dir):
    glazing = load_project(example_project_dir / "project.yaml").glazing
    assert glazing.dirt_factor == 0.9
    assert [glazing.dirt_for(c) for c in DIRT_CLASSES] == [0.9, 0.9, 0.9]


def test_a_class_the_mapping_does_not_hold_is_a_key_error(write_example, tmp_path):
    glazing = _load(write_example, tmp_path, dirt_factor={"skylight": 0.85}).glazing
    with pytest.raises(KeyError, match="vertical"):
        glazing.dirt_for("vertical")


def test_an_unknown_dirt_class_is_refused_by_name(write_example, tmp_path):
    message = _refused(write_example, tmp_path, dirt_factor={"vertical": 0.95, "slopped": 0.9})
    assert "glazing.dirt_factor.slopped" in message
    assert "vertical, sloped, skylight" in message


def test_an_empty_dirt_mapping_is_refused(write_example, tmp_path):
    assert "glazing.dirt_factor" in _refused(write_example, tmp_path, dirt_factor={})


@pytest.mark.parametrize("bad", [0.0, 1.5, "0.85"])
def test_a_dirt_class_value_outside_0_to_1_is_refused_by_name(write_example, tmp_path, bad):
    message = _refused(write_example, tmp_path, dirt_factor={"vertical": 0.95, "skylight": bad})
    assert "glazing.dirt_factor.skylight" in message


# -- frame_share: class keys beside role keys --------------------------------


def test_frame_share_accepts_skylight_and_sloped(write_example, tmp_path):
    project = _load(
        write_example, tmp_path,
        frame_share={"default": 0.2, "skylight": 0.1, "sloped": 0.15},
    )
    assert project.glazing.frame_share["skylight"] == 0.1
    assert project.glazing.frame_share["sloped"] == 0.15


def test_a_class_frame_share_outside_0_to_1_is_refused(write_example, tmp_path):
    message = _refused(write_example, tmp_path, frame_share={"default": 0.2, "skylight": 1.0})
    assert "glazing.frame_share.skylight" in message


def test_a_misspelt_class_frame_share_is_still_refused(write_example, tmp_path):
    message = _refused(write_example, tmp_path, frame_share={"default": 0.2, "skylights": 0.1})
    assert "glazing.frame_share.skylights" in message
    assert "skylight" in message and "sloped" in message


# -- skylight_well and its reflectance ---------------------------------------


def test_without_the_well_keys_both_are_none(example_project_dir):
    glazing = load_project(example_project_dir / "project.yaml").glazing
    assert glazing.skylight_well is None
    assert glazing.skylight_well_reflectance is None


def test_a_modelled_well_loads(write_example, tmp_path):
    glazing = _load(write_example, tmp_path, skylight_well="modelled").glazing
    assert glazing.skylight_well == "modelled"
    assert glazing.skylight_well_reflectance is None


def test_a_well_depth_with_its_reflectance_loads(write_example, tmp_path):
    glazing = _load(
        write_example, tmp_path, skylight_well=0.4, skylight_well_reflectance=0.7,
    ).glazing
    assert glazing.skylight_well == 0.4
    assert glazing.skylight_well_reflectance == 0.7


@pytest.mark.parametrize("bad", ["open", "Modelled", None, True, 0.05, 0.0, -0.4])
def test_a_well_that_is_neither_modelled_nor_a_depth_above_the_blind_is_refused(
    write_example, tmp_path, bad,
):
    message = _refused(
        write_example, tmp_path, skylight_well=bad, skylight_well_reflectance=0.7,
    )
    assert "glazing.skylight_well" in message


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_a_well_reflectance_outside_0_to_1_is_refused(write_example, tmp_path, bad):
    message = _refused(
        write_example, tmp_path, skylight_well=0.4, skylight_well_reflectance=bad,
    )
    assert "glazing.skylight_well_reflectance" in message


def test_a_well_reflectance_with_a_modelled_well_is_refused(write_example, tmp_path):
    message = _refused(
        write_example, tmp_path, skylight_well="modelled", skylight_well_reflectance=0.7,
    )
    assert "glazing.skylight_well_reflectance" in message
    assert "modelled" in message


def test_a_well_depth_without_a_reflectance_is_refused(write_example, tmp_path):
    message = _refused(write_example, tmp_path, skylight_well=0.4)
    assert "glazing.skylight_well_reflectance must be stated" in message


def test_a_well_reflectance_without_a_well_is_refused(write_example, tmp_path):
    message = _refused(write_example, tmp_path, skylight_well_reflectance=0.7)
    assert "without glazing.skylight_well" in message


def test_the_shallowest_well_matches_the_blind_offset():
    # checks.py may not import build/, so the 0.05 m is written twice.
    assert checks._MIN_SKYLIGHT_WELL_M == BLIND_OFFSET_M


def test_a_hand_built_glazing_needs_no_new_argument():
    glazing = Glazing(
        dirt_factor=0.9,
        frame_share=types.MappingProxyType({"default": 0.1}),
        frame_reflectance=0.3,
        frame_style="constant_width",
        reveal_depth_m=0.15,
        reveal_reflectance=0.5,
        vt_includes_dirt=False,
    )
    assert glazing.skylight_well is None
    assert glazing.skylight_well_reflectance is None
    assert glazing.dirt_for("skylight") == 0.9
