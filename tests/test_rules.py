# tests/test_rules.py
import pathlib
import re

import pytest
import yaml

import leed_daylight
from leed_daylight.project.rules import load_rules

EXAMPLE = pathlib.Path(__file__).parent / "fixtures" / "credit_rules_example.yaml"
SHIPPED_TEMPLATE = (
    pathlib.Path(leed_daylight.__file__).parent
    / "templates"
    / "credit_rules_v41_bdc_nc.yaml"
)


def test_loads_the_example_file():
    rules = load_rules(EXAMPLE)
    assert rules["option1"]["sda"]["lux"] == 300
    assert rules["option1"]["ase"]["hours"] == 250
    assert rules["option2"]["bands"] == "UNVERIFIED"
    assert rules["glare_control_required"] is True


def test_bands_are_sorted_and_ascending(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(
        "version: v\n"
        "max_points: 3\n"
        "option1:\n"
        "  sda: {lux: 300, fraction: 0.5, bands: [{min_pct: 75.0, points: 3},"
        " {min_pct: 40.0, points: 1}]}\n"
        "  ase: {lux: 1000, hours: 250, per_space_narrative_trigger_pct: 10.0}\n"
        "option2: {bands: UNVERIFIED, lux_min: 300, lux_max: 3000}\n"
        "glare_control_required: true\n"
    , newline="\n")
    bands = load_rules(path)["option1"]["sda"]["bands"]
    assert [b["min_pct"] for b in bands] == [40.0, 75.0]


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rules(tmp_path / "nope.yaml")


def test_missing_section_is_refused(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text("version: v\nmax_points: 3\n", newline="\n")
    with pytest.raises(ValueError, match="option1"):
        load_rules(path)


def test_a_band_without_points_is_refused(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(
        "version: v\n"
        "max_points: 3\n"
        "option1:\n"
        "  sda: {lux: 300, fraction: 0.5, bands: [{min_pct: 40.0}]}\n"
        "  ase: {lux: 1000, hours: 250, per_space_narrative_trigger_pct: 10.0}\n"
        "option2: {bands: UNVERIFIED, lux_min: 300, lux_max: 3000}\n"
        "glare_control_required: true\n"
    , newline="\n")
    with pytest.raises(ValueError, match="points"):
        load_rules(path)


def test_a_malformed_nested_section_is_a_named_refusal(tmp_path):
    # Parked plan-1 finding: a present-but-malformed nested section used to
    # raise a raw AttributeError/TypeError/KeyError from indexing straight
    # into it, instead of the named refusal spec 6.12 asks for.
    path = tmp_path / "rules.yaml"
    path.write_text(
        "version: v\n"
        "max_points: 3\n"
        "option1:\n"
        "  sda: \"yes\"\n"
        "  ase: {lux: 1000, hours: 250, per_space_narrative_trigger_pct: 10.0}\n"
        "option2: {bands: UNVERIFIED, lux_min: 300, lux_max: 3000}\n"
        "glare_control_required: true\n"
    , newline="\n")
    with pytest.raises(ValueError, match=r"option1\.sda") as excinfo:
        load_rules(path)
    assert not isinstance(excinfo.value, (AttributeError, TypeError, KeyError))


# Fix round 1, ruling P24: a band's min_pct/points must themselves be
# numbers. Before this fix, float(band["min_pct"]) in the final sort raised
# float()'s own bare ValueError ("could not convert string to float: 'high'")
# instead of a named refusal.
def test_a_band_with_a_non_numeric_min_pct_is_a_named_refusal(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(
        "version: v\n"
        "max_points: 3\n"
        "option1:\n"
        "  sda: {lux: 300, fraction: 0.5, bands: [{min_pct: \"high\", points: 1}]}\n"
        "  ase: {lux: 1000, hours: 250, per_space_narrative_trigger_pct: 10.0}\n"
        "option2: {bands: UNVERIFIED, lux_min: 300, lux_max: 3000}\n"
        "glare_control_required: true\n"
    , newline="\n")
    with pytest.raises(ValueError, match=re.escape("option1.sda.bands[0].min_pct")) as excinfo:
        load_rules(path)
    assert not isinstance(excinfo.value, (AttributeError, TypeError, KeyError))
    assert "could not convert" not in str(excinfo.value)


def _complete_rules() -> dict:
    """A fresh, fully valid rules mapping for the parametrised cases below."""
    return yaml.safe_load(
        "version: v\n"
        "max_points: 3\n"
        "option1:\n"
        "  sda: {lux: 300, fraction: 0.5, bands: [{min_pct: 40.0, points: 1}]}\n"
        "  ase: {lux: 1000, hours: 250, per_space_narrative_trigger_pct: 10.0}\n"
        "option2: {bands: UNVERIFIED, lux_min: 300, lux_max: 3000}\n"
        "glare_control_required: true\n"
    )


# Each case removes or malforms exactly one required nested key from an
# otherwise complete, valid rules mapping. This generalises
# test_a_malformed_nested_section_is_a_named_refusal above to every required
# nested key R-P16 lists, not just option1.sda.
MALFORMED_NESTED_CASES = [
    pytest.param(
        lambda r: r["option1"]["sda"].pop("lux"), "option1.sda.lux", id="sda-lux-missing"
    ),
    pytest.param(
        lambda r: r["option1"]["sda"].__setitem__("fraction", 1.5),
        "option1.sda.fraction",
        id="sda-fraction-out-of-range",
    ),
    pytest.param(
        lambda r: r["option1"]["sda"].pop("bands"),
        "option1.sda.bands",
        id="sda-bands-missing",
    ),
    pytest.param(
        lambda r: r["option1"]["sda"].__setitem__("bands", []),
        "option1.sda.bands",
        id="sda-bands-empty",
    ),
    pytest.param(lambda r: r["option1"].pop("ase"), "option1.ase", id="ase-section-missing"),
    pytest.param(
        lambda r: r["option1"]["ase"].pop("lux"),
        "option1.ase.lux",
        id="ase-lux-missing",
    ),
    pytest.param(
        lambda r: r["option1"]["ase"].pop("hours"),
        "option1.ase.hours",
        id="ase-hours-missing",
    ),
    pytest.param(
        lambda r: r["option1"]["ase"].__setitem__(
            "per_space_narrative_trigger_pct", "ten"
        ),
        "option1.ase.per_space_narrative_trigger_pct",
        id="ase-trigger-not-a-number",
    ),
    pytest.param(lambda r: r.pop("option2"), "option2", id="option2-section-missing"),
    pytest.param(
        lambda r: r["option2"].__setitem__("bands", {"not": "a list"}),
        "option2.bands",
        id="option2-bands-malformed",
    ),
    pytest.param(lambda r: r["option2"].pop("lux_min"), "option2.lux_min", id="option2-lux_min-missing"),
    pytest.param(lambda r: r["option2"].pop("lux_max"), "option2.lux_max", id="option2-lux_max-missing"),
    pytest.param(
        lambda r: r["option2"].__setitem__("lux_min", 3000), "option2.lux_min", id="option2-lux_min-not-below-max",
    ),
    pytest.param(
        lambda r: r["option2"].__setitem__("lux_min", 0), "option2.lux_min", id="option2-lux_min-not-above-zero",
    ),
    pytest.param(lambda r: r.pop("max_points"), "max_points", id="max_points-missing"),
    pytest.param(
        lambda r: r.__setitem__("max_points", True),
        "max_points",
        id="max_points-is-a-bool-not-an-int",
    ),
    pytest.param(
        lambda r: r.__setitem__("glare_control_required", "yes"),
        "glare_control_required",
        id="glare_control_required-not-a-bool",
    ),
    pytest.param(lambda r: r.pop("version"), "version", id="version-missing"),
    pytest.param(
        lambda r: r["option1"]["sda"]["bands"][0].__setitem__("min_pct", "high"),
        "option1.sda.bands[0].min_pct",
        id="sda-band-min_pct-not-a-number",
    ),
    pytest.param(
        lambda r: r["option1"]["sda"]["bands"][0].__setitem__("min_pct", True),
        "option1.sda.bands[0].min_pct",
        id="sda-band-min_pct-is-a-bool-not-a-number",
    ),
    pytest.param(
        lambda r: r["option1"]["sda"]["bands"][0].__setitem__("points", "one"),
        "option1.sda.bands[0].points",
        id="sda-band-points-not-a-number",
    ),
]


@pytest.mark.parametrize("break_it, expected_path", MALFORMED_NESTED_CASES)
def test_each_required_nested_key_is_a_named_refusal(tmp_path, break_it, expected_path):
    rules = _complete_rules()
    break_it(rules)
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(rules), newline="\n")
    with pytest.raises(ValueError, match=re.escape(expected_path)) as excinfo:
        load_rules(path)
    assert not isinstance(excinfo.value, (AttributeError, TypeError, KeyError))


def test_the_shipped_template_loads():
    rules = load_rules(SHIPPED_TEMPLATE)
    sda_bands = [b["min_pct"] for b in rules["option1"]["sda"]["bands"]]
    option2_bands = [b["min_pct"] for b in rules["option2"]["bands"]]
    assert sda_bands == [40.0, 55.0, 75.0]
    assert option2_bands == [55.0, 75.0, 90.0]


def test_the_template_time_basis_is_the_weather_files_local_time():
    rules = load_rules(SHIPPED_TEMPLATE)
    assert rules["option2"]["time_basis"] == "weather_file_local"


# Fix round 1, ruling P24: tie conftest.py's MINIMAL_CREDIT_RULES to
# load_rules's actual contract, so a future rename/requirement change in
# load_rules shows up here instead of the fixture silently drifting out of
# sync with the loader it is meant to stand in for.
def test_the_write_example_fixtures_credit_rules_loads(tmp_path, write_example):
    project_path = write_example(tmp_path)
    load_rules(project_path.parent / "credit_rules.yaml")


def _rules_with_additional_point(tmp_path, additional_point: str) -> pathlib.Path:
    path = tmp_path / "rules.yaml"
    path.write_text(
        "version: v\n"
        "max_points: 3\n"
        "option1:\n"
        "  sda: {lux: 300, fraction: 0.5, bands: [{min_pct: 40.0, points: 1}],"
        f" additional_point: {additional_point}}}\n"
        "  ase: {lux: 1000, hours: 250, per_space_narrative_trigger_pct: 10.0}\n"
        "option2: {bands: UNVERIFIED, lux_min: 300, lux_max: 3000}\n"
        "glare_control_required: true\n"
    , newline="\n")
    return path


def test_the_shipped_additional_point_is_an_exact_points_match():
    rules = load_rules(SHIPPED_TEMPLATE)
    additional = rules["option1"]["sda"]["additional_point"]
    assert additional["only_if_points"] == 2
    assert "only_if_points_at_least" not in additional


def test_the_additional_point_needs_only_if_points(tmp_path):
    path = _rules_with_additional_point(
        tmp_path, "{every_space_min_pct: 55.0, points: 1}"
    )
    with pytest.raises(ValueError, match=r"option1\.sda\.additional_point\.only_if_points"):
        load_rules(path)


def test_the_old_at_least_key_is_refused_by_name(tmp_path):
    path = _rules_with_additional_point(
        tmp_path, "{every_space_min_pct: 55.0, only_if_points_at_least: 2, points: 1}"
    )
    with pytest.raises(ValueError, match="only_if_points_at_least.*renamed.*only_if_points"):
        load_rules(path)


# Pre-flight ruling F8: the report's intent paragraph and Option 1's
# per-space sDA bands come from credit_rules.yaml (a fresh copy of the
# credit's own text), not from a sentence baked into the report code.
def _write_template(tmp_path, mutate) -> pathlib.Path:
    raw = yaml.safe_load(SHIPPED_TEMPLATE.read_text())
    mutate(raw)
    path = tmp_path / "credit_rules.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), newline="\n")
    return path


def test_intent_is_optional(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(_complete_rules()), newline="\n")

    rules = load_rules(path)

    assert "intent" not in rules


def test_a_non_string_intent_is_refused(tmp_path):
    rules = _complete_rules()
    rules["intent"] = 4.1
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(rules), newline="\n")

    with pytest.raises(ValueError, match="intent"):
        load_rules(path)


def test_a_non_string_views_intent_is_refused(tmp_path):
    path = _write_template(tmp_path, lambda raw: raw["views"].__setitem__("intent", 4.1))

    with pytest.raises(ValueError, match=r"views\.intent"):
        load_rules(path)


def test_the_shipped_template_carries_the_intent_sentences():
    rules = load_rules(SHIPPED_TEMPLATE)

    assert "daylight into the space" in rules["intent"]
    assert "quality views" in rules["views"]["intent"].lower()


def test_the_shipped_template_carries_the_option1_space_rule():
    rules = load_rules(SHIPPED_TEMPLATE)

    assert rules["option1"]["space_rule_sda_pct"] == [55.0, 75.0]


# Fix round 1: a bare, non-empty-string check for intent, and full
# validation for option1.space_rule_sda_pct, which used to pass through
# unchecked -- unlike every other key load_rules handles.
@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_intent_is_refused(tmp_path, value):
    rules = _complete_rules()
    rules["intent"] = value
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(rules), newline="\n")

    with pytest.raises(ValueError, match="intent"):
        load_rules(path)


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_views_intent_is_refused(tmp_path, value):
    path = _write_template(tmp_path, lambda raw: raw["views"].__setitem__("intent", value))

    with pytest.raises(ValueError, match=r"views\.intent"):
        load_rules(path)


def test_space_rule_sda_pct_is_optional(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(_complete_rules()), newline="\n")

    rules = load_rules(path)

    assert "space_rule_sda_pct" not in rules["option1"]


@pytest.mark.parametrize("value", [
    pytest.param("not-a-list-at-all", id="a-bare-string"),
    pytest.param([], id="an-empty-list"),
    pytest.param([55.0, "not-a-number"], id="a-non-numeric-entry"),
    pytest.param([0.0, 75.0], id="a-value-at-or-below-zero"),
    pytest.param([55.0, 120.0], id="a-value-above-100"),
    pytest.param([75.0, 55.0], id="not-strictly-increasing"),
    pytest.param([55.0, 55.0], id="a-repeated-value"),
])
def test_a_malformed_space_rule_sda_pct_is_refused_by_name(tmp_path, value):
    rules = _complete_rules()
    rules["option1"]["space_rule_sda_pct"] = value
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(rules), newline="\n")

    with pytest.raises(ValueError, match=re.escape("option1.space_rule_sda_pct")) as excinfo:
        load_rules(path)
    assert not isinstance(excinfo.value, (AttributeError, TypeError, KeyError))
