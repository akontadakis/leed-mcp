# tests/test_project_config.py
import dataclasses
import pathlib

import pytest

from leed_daylight.project.config import ProjectError, load_project

# The 14 top-level sections of project.yaml (R-P3/R-P6).
TOP_LEVEL_SECTIONS = (
    "project", "site", "scene", "standard", "weather", "glazing",
    "reflectance", "grid", "analysis", "performance", "sky", "radiance",
    "blinds", "option2",
)

# (section, key, permitted values) for every checked enum (R-P4's addendum
# adds standard.daylight to this list).
ENUM_CASES = (
    ("glazing", "frame_style", {"constant_width", "uniform_scale"}),
    ("reflectance", "policy", {"override", "fill_gaps", "keep_as_modelled"}),
    ("analysis", "aggregation", {"floor_area", "points"}),
    ("analysis", "daylight_saving", {"EU", "US", "none"}),
    ("analysis", "moveable_furniture", {"include", "exclude"}),
    ("performance", "trace_scope", {"storey", "all"}),
    ("standard", "daylight", {"LM-83-12", "LM-83-23"}),
)

# (id, mutate, expected path substring, expected value-seen substring) for
# fix round 1: a malformed section must be refused by name, not crash with
# a raw TypeError. Covers a list section, a scalar section, a string
# section, and a non-mapping nested sub-section.
NON_MAPPING_SECTION_CASES = (
    (
        "list_section",
        lambda d: d.__setitem__("site", [{"a": 1}]),
        "site",
        repr([{"a": 1}]),
    ),
    (
        "scalar_section",
        lambda d: d.__setitem__("weather", 5),
        "weather",
        repr(5),
    ),
    (
        "string_section",
        lambda d: d.__setitem__("grid", "oops"),
        "grid",
        repr("oops"),
    ),
    (
        "non_mapping_nested_subsection",
        lambda d: d["radiance"].__setitem__("ambient", "oops"),
        "radiance.ambient",
        repr("oops"),
    ),
)


def test_the_example_project_loads(write_example, tmp_path):
    path = write_example(tmp_path)
    project = load_project(path)

    assert project.site.north_rotation_deg == 10.0
    assert project.scene.units == "m"
    assert project.grid.spacing_m == 0.60


def test_a_missing_required_key_is_refused(write_example, tmp_path):
    def mutate(d):
        del d["site"]["north_rotation_source"]
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="site.north_rotation_source"):
        load_project(path)


@pytest.mark.parametrize("section", TOP_LEVEL_SECTIONS)
def test_every_section_is_required(write_example, tmp_path, section):
    def mutate(d):
        del d[section]
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match=section):
        load_project(path)


def test_an_unknown_unit_is_refused(write_example, tmp_path):
    def mutate(d):
        d["scene"]["units"] = "furlong"
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError) as excinfo:
        load_project(path)
    message = str(excinfo.value)
    assert "furlong" in message
    assert str(sorted(["m", "mm", "cm", "ft", "in"])) in message


@pytest.mark.parametrize("section, key, permitted", ENUM_CASES)
def test_an_unknown_enum_value_is_refused(write_example, tmp_path, section, key, permitted):
    def mutate(d):
        d[section][key] = "bogus"
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError) as excinfo:
        load_project(path)
    message = str(excinfo.value)
    assert "bogus" in message
    assert str(sorted(permitted)) in message


def test_moveable_furniture_is_required(write_example, tmp_path):
    def mutate(d):
        del d["analysis"]["moveable_furniture"]
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="analysis.moveable_furniture"):
        load_project(path)


def test_relative_paths_resolve_against_the_project_file(write_example, tmp_path, monkeypatch):
    project_path = write_example(tmp_path)
    project_dir = project_path.parent

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    project = load_project(project_path)

    assert project.weather_file.is_absolute()
    assert project.weather_file.is_relative_to(project_dir)
    for f in project.scene.files:
        assert f.is_absolute()
        assert f.is_relative_to(project_dir)


def test_an_unknown_key_inside_a_section_is_refused(write_example, tmp_path):
    def mutate(d):
        d["grid"]["spacng_m"] = d["grid"].pop("spacing_m")
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="grid.spacng_m"):
        load_project(path)


def test_an_unknown_top_level_section_is_a_warning(write_example, tmp_path):
    def mutate(d):
        d["future"] = {"foo": "bar"}
        return d

    path = write_example(tmp_path, mutate=mutate)
    project = load_project(path)

    assert any("future" in w for w in project.warnings)


def test_the_project_is_frozen(write_example, tmp_path):
    path = write_example(tmp_path)
    project = load_project(path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        project.site = project.site

    with pytest.raises(TypeError):
        project.glazing.frame_share["default"] = 0.1


@pytest.mark.parametrize(
    "case_id, apply, path, value_seen",
    NON_MAPPING_SECTION_CASES,
    ids=[c[0] for c in NON_MAPPING_SECTION_CASES],
)
def test_a_non_mapping_section_is_refused_by_name(write_example, tmp_path, case_id, apply, path, value_seen):
    def mutate(d):
        apply(d)
        return d

    file_path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError) as excinfo:
        load_project(file_path)
    message = str(excinfo.value)
    assert path in message
    assert "mapping" in message
    assert value_seen in message


# ---------------------------------------------------------------------------
# I6: a float field can never be NaN or Infinity
# ---------------------------------------------------------------------------


def test_a_nan_float_is_refused(write_example, tmp_path):
    def mutate(d):
        d["glazing"]["reveal_depth_m"] = float("nan")
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="glazing.reveal_depth_m"):
        load_project(path)


def test_an_infinite_float_is_refused(write_example, tmp_path):
    def mutate(d):
        d["site"]["north_rotation_deg"] = float("inf")
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="site.north_rotation_deg"):
        load_project(path)


# ---------------------------------------------------------------------------
# reflectance.* and glazing.frame_share.* keys must be real role names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("typo", ["celing", "curtain_wall", "context_buildings"])
def test_an_unknown_reflectance_role_is_refused(write_example, tmp_path, typo):
    def mutate(d):
        d["reflectance"][typo] = 0.5
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError) as excinfo:
        load_project(path)
    message = str(excinfo.value)
    assert "project.yaml" in message
    assert f"reflectance.{typo}" in message
    # The valid names are listed so the fix is obvious.
    assert "ceiling" in message and "policy" in message


@pytest.mark.parametrize("typo", ["glazing_curtainwall", "curtain_wall", "wall", "glazing_interior"])
def test_an_unknown_frame_share_role_is_refused(write_example, tmp_path, typo):
    def mutate(d):
        d["glazing"]["frame_share"][typo] = 0.2
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError) as excinfo:
        load_project(path)
    message = str(excinfo.value)
    assert "project.yaml" in message
    assert f"glazing.frame_share.{typo}" in message
    assert "glazing_curtain_wall" in message and "default" in message


def test_every_role_is_accepted_as_a_reflectance_key(write_example, tmp_path):
    from leed_daylight.project.materials import ROLES

    def mutate(d):
        for role in ROLES:
            d["reflectance"][role] = 0.5
        d["glazing"]["frame_share"]["glazing"] = 0.2
        return d

    project = load_project(write_example(tmp_path, mutate=mutate))
    assert set(project.reflectance.by_role) == set(ROLES)
    assert project.glazing.frame_share["glazing"] == 0.2


# ---------------------------------------------------------------------------
# YAML 1.1 gotchas: 1e3 is text, 15:00 is 900, an unquoted date is a date
# ---------------------------------------------------------------------------


def _write_raw(write_example, tmp_path, old, new):
    path = write_example(tmp_path)
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))
    return path


def test_a_number_written_as_1e3_is_refused_with_a_hint(write_example, tmp_path):
    path = _write_raw(write_example, tmp_path, "lux_max: 3000", "lux_max: 3e3")
    with pytest.raises(ProjectError) as excinfo:
        load_project(path)
    message = str(excinfo.value)
    assert "option2.lux_max" in message
    assert "YAML reads 1e3 as text; write 1.0e3" in message


def test_a_non_numeric_string_gets_no_exponent_hint(write_example, tmp_path):
    def mutate(d):
        d["option2"]["lux_max"] = "lots"
        return d

    with pytest.raises(ProjectError) as excinfo:
        load_project(write_example(tmp_path, mutate=mutate))
    assert "1.0e3" not in str(excinfo.value)


def test_an_unquoted_clock_time_is_refused_with_a_quoting_hint(write_example, tmp_path):
    # yaml.safe_dump quotes '15:00'; unquoted, YAML 1.1 reads it as 900.
    path = _write_raw(write_example, tmp_path, "- '15:00'", "- 15:00")
    with pytest.raises(ProjectError) as excinfo:
        load_project(path)
    message = str(excinfo.value)
    assert "option2.times" in message
    assert "900" in message
    assert "quote" in message and '"15:00"' in message


def test_an_unquoted_date_is_refused_with_a_quoting_hint(write_example, tmp_path):
    path = _write_raw(write_example, tmp_path, "- 03-21", "- 2026-03-21")
    with pytest.raises(ProjectError) as excinfo:
        load_project(path)
    message = str(excinfo.value)
    assert "option2.equinox_dates" in message
    assert "date" in message and "quote" in message


# ---------------------------------------------------------------------------
# The optional `report:` section (reports spec 3.3)
# ---------------------------------------------------------------------------


def test_report_section_is_optional(tmp_path, write_example):
    project = load_project(write_example(tmp_path))
    assert project.report.renderer == "matplotlib"
    assert project.report.deviations == ()
    assert project.report.prepared_by is None


def test_report_section_is_read(tmp_path, write_example):
    def mutate(d):
        d["report"] = {
            "renderer": "pillow",
            "deviations": ["Furniture is not modelled."],
            "prepared_by": "Example Consultants",
        }
        return d

    project = load_project(write_example(tmp_path, mutate))
    assert project.report.renderer == "pillow"
    assert project.report.deviations == ("Furniture is not modelled.",)
    assert project.report.prepared_by == "Example Consultants"


@pytest.mark.parametrize("section, message", [
    ({"renderer": "gnuplot"}, "report.renderer"),
    ({"deviations": "none"}, "report.deviations"),
    ({"colour": "red"}, "report.colour"),
])
def test_report_section_refusals(tmp_path, write_example, section, message):
    def mutate(d):
        d["report"] = section
        return d

    with pytest.raises(ProjectError, match=message):
        load_project(write_example(tmp_path, mutate))


# The radiance option strings reach rcontrib and rtrace as arguments: only
# numeric ambient, limit, direct and sampling options pass, so an option that
# names a file or a command (-o, -af, -e, @file, !cmd) is refused on load.
@pytest.mark.parametrize("key", ["draft", "full", "direct"])
@pytest.mark.parametrize("text", [
    "-o '!touch pwned'", "-af /tmp/amb", "-e /tmp/err", "@opts.txt", "-ab 2 -f x.cal",
    "-ab", "-ab two", "-av 0 0", "-ab 2 extra", "-ab 'unclosed",
])
def test_an_unsafe_or_malformed_radiance_option_is_refused(write_example, tmp_path, key, text):
    def mutate(d):
        if key == "direct":
            d["radiance"]["direct"] = text
        else:
            d["radiance"]["ambient"][key] = text
        return d

    path = write_example(tmp_path, mutate=mutate)
    where = "radiance.direct" if key == "direct" else f"radiance.ambient.{key}"
    with pytest.raises(ProjectError, match=rf"{where.replace('.', '[.]')}"):
        load_project(path)


@pytest.mark.parametrize("text", [
    "", "-ab 6 -ad 4096 -lw 1e-5", "-dc 1 -dt 0 -dj 0", "-ab 2 -aa 0.1 -ar 256 -as 128",
    "-av 0 0 0 -u+", "-u- -st 0.15 -ss 1 -lr -10 -dr 2 -dp 512 -ds 0.2",
])
def test_numeric_radiance_options_load(write_example, tmp_path, text):
    def mutate(d):
        d["radiance"]["ambient"]["full"] = text
        return d

    project = load_project(write_example(tmp_path, mutate=mutate))
    assert project.radiance.ambient_full == text


@pytest.mark.parametrize("text", ["", "-dc 1 -dt 0 -dj 0", "-u- -st 0.15 -ss 1 -dr 2 -dp 512 -ds 0.2"])
def test_d_and_s_radiance_direct_options_load(write_example, tmp_path, text):
    def mutate(d):
        d["radiance"]["direct"] = text
        return d

    project = load_project(write_example(tmp_path, mutate=mutate))
    assert project.radiance.direct == text


# radiance.direct is added to the sun pass and the sky pass after the
# ambient options, so only the direct (-d*) and sampling (-s*) options are
# allowed there: -ab, -ad, -lw and the like would override the pass itself.
@pytest.mark.parametrize("text", ["-ab 1", "-dc 1 -ad 64", "-lw 1e-5", "-aa 0.1", "-av 0 0 0"])
def test_a_direct_option_outside_d_and_s_is_refused(write_example, tmp_path, text):
    def mutate(d):
        d["radiance"]["direct"] = text
        return d

    with pytest.raises(ProjectError, match=r"radiance[.]direct.*only the -d and -s options"):
        load_project(write_example(tmp_path, mutate=mutate))


def _sub(tmp_path, name):
    """A fresh subfolder: write_example makes ``project/`` without parents."""
    (tmp_path / name).mkdir()
    return tmp_path / name


def test_the_calculator_section_is_optional_and_closed(write_example, tmp_path):
    assert load_project(write_example(tmp_path)).calculator.units is None

    def units(d):
        d["calculator"] = {"units": "SI"}
        return d

    assert load_project(write_example(_sub(tmp_path, "a"), mutate=units)).calculator.units == "SI"

    def bad(d):
        d["calculator"] = {"units": "metric"}
        return d

    with pytest.raises(ProjectError, match="calculator.units"):
        load_project(write_example(_sub(tmp_path, "b"), mutate=bad))


def test_setting_units_keeps_runs_current(write_example, tmp_path):
    from leed_daylight.project.inputs import _project_yaml_digest

    def units(d):
        d["calculator"] = {"units": "IP"}
        return d

    assert _project_yaml_digest(write_example(_sub(tmp_path, "a"))) == _project_yaml_digest(
        write_example(_sub(tmp_path, "b"), mutate=units))
