# tests/test_wizard_project_yaml.py
"""The wizard's project.yaml writer and pre-filled values (wizard spec 6.1,
7.2, W3, W10).

``render`` writes a loaded project.yaml in the template's form; for every
example and fixture project, load, render and load again gives an equal
``Project``. The pre-filled table is held against the template's own
comments, so the two cannot drift. Every value is invented.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest
import yaml

from deliver_folder import build_deliver_project, build_roof_glass_project, build_two_storey_project
from leed_daylight.constants import FILL_ME
from leed_daylight.drivers.init import PROJECT_TEMPLATE, init_folder
from leed_daylight.edit import project_yaml
from leed_daylight.edit.project_yaml import (
    GROUPS,
    PREFILLS,
    apply_prefills,
    comments,
    get_path,
    group_confirmed,
    group_keys,
    hand_comments,
    render,
    set_path,
    template_keys,
)
from leed_daylight.project.config import load_project
from merge_split_folder import build_combined_project, build_merge_project, build_split_project
from option2_folder import build_option2_project

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _every_key_project(tmp_path, write_example, write_wea):
    """An invented project with every commented key in use."""
    def mutate(d):
        d["glazing"]["dirt_factor"] = {"vertical": 0.9, "skylight": 0.8}
        d["glazing"]["frame_share"]["glazing_diffusing"] = 0.1
        d["glazing"]["frame_share"]["skylight"] = 0.2
        d["glazing"]["skylight_well"] = 0.4
        d["glazing"]["skylight_well_reflectance"] = 0.7
        d["reflectance"]["furniture"] = 0.5
        d["performance"]["linked_zones"] = {"L0": ["zone a", "zone b"], "Level 1": ["zone c"]}
        d["views"] = {"eye_height_m": 1.2, "eye_height_source": "Invented: seated eye height",
                      "angular_step_deg": 2.5}
        d["report"] = {"renderer": "pillow", "deviations": ["An invented deviation."],
                       "prepared_by": "An invented author"}
        d["atlas"] = {"drawings": [{
            "storey": "L0", "image": "drawings/L0.png", "opacity": 0.5,
            "points": [{"pixel": [10.0, 10.0], "model": [0.0, 0.0]},
                       {"pixel": [610.0, 10.0], "model": [6.0, 0.0]}],
        }]}
        return d

    return build_option2_project(tmp_path, write_example, write_wea, mutate=mutate)


PROJECTS = {
    "project_example": lambda tmp, ex, wea: ex(tmp).parent,
    "option2_folder": build_option2_project,
    "deliver_folder": lambda tmp, ex, wea: build_deliver_project(tmp, ex, wea, options=()),
    "two_storey": lambda tmp, ex, wea: build_two_storey_project(tmp, ex, wea, options=()),
    "roof_glass": lambda tmp, ex, wea: build_roof_glass_project(tmp, ex, wea, options=()),
    "every_key": _every_key_project,
    "split_folder": build_split_project,
    "merge_folder": build_merge_project,
    "combined_folder": build_combined_project,
}


def test_rendering_the_template_gives_the_template():
    text = PROJECT_TEMPLATE.read_text(encoding="utf-8")

    assert render(yaml.safe_load(text)) == text


def test_the_fixture_folder_holds_no_other_project_yaml():
    # The round trip below covers every project of the tests; a project.yaml
    # added under tests/fixtures/ must be added to PROJECTS.
    assert sorted(FIXTURES.rglob("project.yaml")) == []


@pytest.mark.parametrize("name", sorted(PROJECTS))
def test_the_round_trip_keeps_every_fixture_project(name, tmp_path, write_example, write_wea):
    folder = PROJECTS[name](tmp_path, write_example, write_wea)
    path = folder / "project.yaml"
    before = load_project(path)
    raw = yaml.safe_load(path.read_text())

    text = render(raw)
    path.write_text(text, encoding="utf-8")

    assert load_project(path) == before
    assert yaml.safe_load(text) == raw
    assert render(yaml.safe_load(text)) == text


def test_template_keys_keep_their_template_lines(tmp_path, write_example, write_wea):
    folder = _every_key_project(tmp_path, write_example, write_wea)
    text = render(yaml.safe_load((folder / "project.yaml").read_text()))
    lines = text.splitlines()

    assert lines[:3] == PROJECT_TEMPLATE.read_text().splitlines()[:3]
    assert '  name: "Example Tower"          # free text, printed in reports' in lines
    assert "  spacing_m: 0.6                 # LEED: at most 0.60" in lines
    assert (
        '    draft: "-ab 2 -ad 512 -lw 1e-3" # Radiance ambient settings for a quick look, quoted'
        in lines
    )
    assert '  credit_rules: "credit_rules.yaml"   # written by init: the LEED v4.1 rules template' in lines


def test_keys_the_template_lacks_follow_their_section_in_the_loaded_order(tmp_path, write_example, write_wea):
    folder = _every_key_project(tmp_path, write_example, write_wea)
    lines = render(yaml.safe_load((folder / "project.yaml").read_text())).splitlines()

    start = lines.index("  frame_share:")
    end = lines.index("  frame_reflectance: 0.34        # 0 to 1")
    assert [line for line in lines[start:end] if not line.lstrip().startswith("#")] == [
        "  frame_share:",
        "    default: 0.25                # share of each window's area that is frame, from the manufacturer",
        "    glazing_curtain_wall: 0.15",
        "    glazing_diffusing: 0.1",
        "    skylight: 0.2",
    ]
    well = lines.index("  skylight_well: 0.4")
    assert lines[well - 1].startswith("  # skylight_well_reflectance:")
    assert lines[well + 1:well + 3] == ["  skylight_well_reflectance: 0.7", ""]
    assert "  dirt_factor: {vertical: 0.9, skylight: 0.8} # LM-83-12 clause 2.2.9, between 0 and 1" in lines
    ground = lines.index("  ground: 0.15")
    assert lines[ground + 1:ground + 3] == ["  furniture: 0.5", ""]
    assert (
        '  linked_zones: {L0: ["zone a", "zone b"], "Level 1": ["zone c"]} # {} unless the guide tells you otherwise'
        in lines
    )


def test_the_optional_sections_follow_in_order_in_block_style(tmp_path, write_example, write_wea):
    folder = _every_key_project(tmp_path, write_example, write_wea)
    raw = yaml.safe_load((folder / "project.yaml").read_text())
    raw = {"atlas": raw.pop("atlas"), "notes": {"one": "An invented note."}, **raw}

    text = render(raw)

    tail = text.split('  time_basis: "weather_file_local" # weather_file_local\n', 1)[1]
    assert tail == (
        "\n"
        "views:\n"
        "  eye_height_m: 1.2\n"
        '  eye_height_source: "Invented: seated eye height"\n'
        "  angular_step_deg: 2.5\n"
        "\n"
        "report:\n"
        '  renderer: "pillow"\n'
        '  deviations: ["An invented deviation."]\n'
        '  prepared_by: "An invented author"\n'
        "\n"
        "atlas:\n"
        "  drawings:\n"
        '    - storey: "L0"\n'
        '      image: "drawings/L0.png"\n'
        "      opacity: 0.5\n"
        "      points: [{pixel: [10.0, 10.0], model: [0.0, 0.0]}, {pixel: [610.0, 10.0], model: [6.0, 0.0]}]\n"
        "\n"
        "notes:\n"
        '  one: "An invented note."\n'
    )
    assert yaml.safe_load(text) == raw


def test_a_missing_key_is_written_fill_me_never_the_template_value(example_project_dir):
    raw = yaml.safe_load((example_project_dir / "project.yaml").read_text())
    del raw["standard"]["credit_rules"]
    del raw["radiance"]["ambient"]
    del raw["blinds"]

    loaded = yaml.safe_load(render(raw))

    assert loaded["standard"]["credit_rules"] == FILL_ME
    assert loaded["radiance"]["ambient"] == {"draft": FILL_ME, "full": FILL_ME}
    assert loaded["blinds"] == {
        "transmittance_diffuse": FILL_ME, "transmittance_specular": FILL_ME, "room_side_reflectance": FILL_ME,
    }


def test_a_reflectance_role_left_out_stays_out(example_project_dir):
    """The template says to delete a role line to leave that role as
    modelled: such a role is not written back as FILL_ME."""
    raw = yaml.safe_load((example_project_dir / "project.yaml").read_text())
    del raw["reflectance"]["context"]
    del raw["reflectance"]["ground"]

    text = render(raw)
    (example_project_dir / "project.yaml").write_text(text)

    assert yaml.safe_load(text)["reflectance"] == {"policy": "override", "ceiling": 0.75, "wall": 0.56, "floor": 0.34}
    assert dict(load_project(example_project_dir / "project.yaml").reflectance.by_role) == {
        "ceiling": 0.75, "wall": 0.56, "floor": 0.34,
    }


def test_a_missing_reflectance_policy_or_section_is_still_fill_me(example_project_dir):
    raw = yaml.safe_load((example_project_dir / "project.yaml").read_text())
    del raw["reflectance"]["policy"]
    assert yaml.safe_load(render(raw))["reflectance"]["policy"] == FILL_ME
    del raw["reflectance"]
    assert yaml.safe_load(render(raw))["reflectance"] == {
        "policy": FILL_ME, "ceiling": FILL_ME, "wall": FILL_ME, "floor": FILL_ME, "context": FILL_ME,
        "ground": FILL_ME,
    }


def test_a_template_mapping_held_as_one_value_is_written_as_it(example_project_dir):
    raw = yaml.safe_load((example_project_dir / "project.yaml").read_text())
    raw["glazing"]["frame_share"] = FILL_ME

    text = render(raw)

    assert "  frame_share: FILL_ME" in text.splitlines()
    assert yaml.safe_load(text) == raw


def test_a_float_in_e_notation_is_written_as_a_number(example_project_dir):
    raw = yaml.safe_load((example_project_dir / "project.yaml").read_text())
    raw["reflectance"]["furniture"] = 1e-05
    raw["reflectance"]["context"] = 2.5e-07
    raw["blinds"]["transmittance_specular"] = 1e20

    text = render(raw)
    loaded = yaml.safe_load(text)

    assert "  furniture: 1.0e-05" in text.splitlines()
    assert loaded["reflectance"]["furniture"] == 1e-05
    assert loaded["reflectance"]["context"] == 2.5e-07
    assert loaded["blinds"]["transmittance_specular"] == 1e20
    assert loaded["radiance"]["ambient"]["full"] == "-ab 6 -ad 4096 -lw 1e-5"


def test_values_keep_their_types(example_project_dir):
    raw = yaml.safe_load((example_project_dir / "project.yaml").read_text())
    raw["project"]["name"] = 'An "invented" name: with a # and ü'
    raw["glazing"]["vt_includes_dirt"] = True
    raw["sky"]["MF"] = 4
    raw["report"] = {"prepared_by": None, "deviations": []}

    loaded = yaml.safe_load(render(raw))

    assert loaded == raw
    assert type(loaded["sky"]["MF"]) is int
    assert type(loaded["grid"]["spacing_m"]) is float


# --- Pre-filled values ---------------------------------------------------

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _stated_in(value, comment: str) -> bool:
    """Whether the template comment states ``value``."""
    if isinstance(value, tuple):
        return json.dumps(list(value)) in comment
    if isinstance(value, (int, float)):
        return any(float(n) == float(value) for n in _NUMBER.findall(comment))
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])", comment) is not None


def test_the_prefilled_table_equals_the_template_comments():
    keys = {key.path: key for key in template_keys()}

    assert [p.key for p in PREFILLS] == [
        "grid.spacing_m", "grid.work_plane_m", "analysis.aggregation",
        "option2.equinox_dates", "option2.clearest_day_window_days", "option2.times",
        "option2.lux_min", "option2.lux_max", "option2.sky", "option2.blinds", "option2.time_basis",
        "sky.basis", "sky.MF",
    ]
    assert GROUPS == ("grid", "analysis.aggregation", "option2", "sky")
    assert {p.group for p in PREFILLS} == set(GROUPS)
    for prefill in PREFILLS:
        assert keys[prefill.key].value == FILL_ME, prefill.key
        assert _stated_in(prefill.value, keys[prefill.key].comment), (prefill.key, keys[prefill.key].comment)
        assert prefill.source
    assert group_keys("grid") == ("grid.spacing_m", "grid.work_plane_m")
    assert group_keys("analysis.aggregation") == ("analysis.aggregation",)
    assert len(group_keys("option2")) == 8
    assert group_keys("sky") == ("sky.basis", "sky.MF")
    with pytest.raises(ValueError, match="unknown pre-filled group 'blinds'"):
        group_keys("blinds")


def test_nothing_the_template_leaves_to_the_person_is_prefilled():
    prefilled = {p.key for p in PREFILLS}
    for forbidden in ("glazing.dirt_factor", "glazing.frame_share.default", "grid.min_clearance_m",
                      "blinds.transmittance_diffuse", "radiance.ambient.draft", "analysis.hours",
                      "site.latitude_N", "analysis.daylight_saving"):
        assert forbidden not in prefilled


def test_template_keys_are_dotted_in_template_order():
    keys = template_keys()

    assert keys[0] == project_yaml.TemplateKey("project.name", "free text, printed in reports", FILL_ME)
    paths = [key.path for key in keys]
    assert paths.index("glazing.frame_share.default") < paths.index("glazing.frame_reflectance")
    assert "radiance.ambient.draft" in paths
    assert paths[-1] == "option2.time_basis"
    assert next(k for k in keys if k.path == "standard.credit_rules").value == '"credit_rules.yaml"'


def test_an_unconfirmed_group_stays_fill_me(tmp_path):
    folder = init_folder(tmp_path / "new")
    raw = yaml.safe_load((folder / "project.yaml").read_text())

    unconfirmed = yaml.safe_load(render(apply_prefills(raw, ())))
    confirmed = yaml.safe_load(render(apply_prefills(raw, ("grid", "sky"))))

    assert unconfirmed == raw
    assert get_path(unconfirmed, "grid.spacing_m") == FILL_ME
    assert confirmed["grid"] == {"spacing_m": 0.6, "work_plane_m": 0.762, "min_clearance_m": FILL_ME}
    assert confirmed["sky"] == {"basis": "Reinhart", "MF": 4}
    assert confirmed["option2"] == raw["option2"]
    assert confirmed["analysis"]["aggregation"] == FILL_ME
    for group in GROUPS:
        assert not group_confirmed(raw, group)
    assert group_confirmed(confirmed, "grid") and group_confirmed(confirmed, "sky")


def test_a_group_with_real_values_counts_as_confirmed(example_project_dir):
    path = example_project_dir / "project.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["grid"]["spacing_m"] = 0.5
    raw["option2"]["lux_min"] = 250

    assert all(group_confirmed(raw, group) for group in GROUPS)
    applied = apply_prefills(raw, GROUPS)
    assert applied == raw
    path.write_text(render(applied))
    project = load_project(path)
    assert project.grid.spacing_m == 0.5 and project.option2.lux_min == 250


def test_an_unconfirmed_group_keeps_the_real_values_it_already_has(tmp_path):
    # Review Focus 1: grid.spacing_m typed by hand, grid.work_plane_m still FILL_ME.
    folder = init_folder(tmp_path / "new")
    raw = yaml.safe_load((folder / "project.yaml").read_text())
    raw["grid"]["spacing_m"] = 0.5

    assert not group_confirmed(raw, "grid")
    kept = yaml.safe_load(render(apply_prefills(raw, ())))
    filled = yaml.safe_load(render(apply_prefills(raw, ("grid",))))

    assert kept["grid"] == {"spacing_m": 0.5, "work_plane_m": FILL_ME, "min_clearance_m": FILL_ME}
    assert filled["grid"] == {"spacing_m": 0.5, "work_plane_m": 0.762, "min_clearance_m": FILL_ME}


def test_apply_prefills_leaves_its_input_alone(tmp_path):
    folder = init_folder(tmp_path / "new")
    raw = yaml.safe_load((folder / "project.yaml").read_text())
    before = yaml.safe_load((folder / "project.yaml").read_text())

    applied = apply_prefills(raw, GROUPS)

    assert raw == before
    assert applied["option2"]["equinox_dates"] == ["03-21", "09-21"]
    assert applied["option2"]["times"] == ["09:00", "15:00"]


def test_get_and_set_a_path():
    raw = {"grid": FILL_ME}

    assert get_path(raw, "grid.spacing_m") == FILL_ME
    assert get_path(raw, "sky.MF") == FILL_ME
    set_path(raw, "grid.spacing_m", 0.5)
    set_path(raw, "radiance.ambient.draft", "-ab 1")

    assert raw == {"grid": {"spacing_m": 0.5}, "radiance": {"ambient": {"draft": "-ab 1"}}}
    assert get_path(raw, "radiance.ambient.draft") == "-ab 1"


# --- Comments ------------------------------------------------------------

def test_hand_comments_finds_an_added_comment_and_nothing_on_the_template():
    text = PROJECT_TEMPLATE.read_text()
    added = text.replace(
        "  name: FILL_ME                  # free text, printed in reports\n",
        '  name: "An # invented name"     # our own name for it\n# a note of our own\n',
    )

    assert hand_comments(text) == ()
    assert hand_comments(added) == ("# our own name for it", "# a note of our own")


def test_comments_ignores_a_hash_inside_quotes_or_a_word():
    text = "a: \"x # y\"  # one\nb: it's#here # two\n# three\nc: 'q # r'\n"

    assert comments(text) == ("# one", "# two", "# three")
