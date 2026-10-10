# tests/test_rules_views.py
"""The Quality Views section of credit_rules.yaml (Quality Views spec 5.4)."""
from __future__ import annotations

import datetime
import math
import pathlib

import pytest
import yaml

import leed_daylight
from leed_daylight.constants import UNVERIFIED
from leed_daylight.project.rules import load_rules, view_rules, views_verified

TEMPLATE = pathlib.Path(leed_daylight.__file__).parent / "templates" / "credit_rules_v41_bdc_nc.yaml"


def _write(tmp_path, mutate):
    raw = yaml.safe_load(TEMPLATE.read_text())
    mutate(raw)
    path = tmp_path / "credit_rules.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def test_the_template_carries_the_views_section():
    views = view_rules(load_rules(TEMPLATE))

    assert views["quality_definition"] == "type2_and_type3"
    assert views["threshold_pct"] == 75.0
    assert views["points"] == 1
    assert views["vlt_min_exclusive"] == 0.40
    assert views["head_height_multiple"] == 3.0
    assert views["object_distance_m"] == 7.5
    assert views["obstruction_exception_m"] == 0.30
    assert views["method"] == {
        "distance_statistic": "median",
        "min_outside_distance_m": 2.0,
        "panorama_vertical_deg": 60.0,
    }
    assert views_verified(views)


def test_rules_without_views_still_load_but_view_rules_refuses(tmp_path):
    path = _write(tmp_path, lambda raw: raw.pop("views"))

    rules = load_rules(path)

    with pytest.raises(ValueError, match=r"views: missing"):
        view_rules(rules)


def test_an_unverified_credit_value_loads_and_is_reported(tmp_path):
    path = _write(tmp_path, lambda raw: raw["views"].__setitem__("threshold_pct", UNVERIFIED))

    views = view_rules(load_rules(path))

    assert views["threshold_pct"] == UNVERIFIED
    assert not views_verified(views)


def _set(keys, value):
    def mutate(raw):
        target = raw["views"]
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return mutate


@pytest.mark.parametrize("keys, value, match", [
    (("threshold_pct",), 120.0, r"views\.threshold_pct"),
    (("vlt_min_exclusive",), 1.5, r"views\.vlt_min_exclusive"),
    (("head_height_multiple",), 0, r"views\.head_height_multiple"),
    (("object_distance_m",), "far", r"views\.object_distance_m"),
    (("points",), 0, r"views\.points"),
    (("quality_definition",), "type2_or_type3", r"views\.quality_definition"),
    (("method", "distance_statistic"), "mean", r"views\.method\.distance_statistic"),
    (("method", "min_outside_distance_m"), -1.0, r"views\.method\.min_outside_distance_m"),
    (("method", "panorama_vertical_deg"), 200.0, r"views\.method\.panorama_vertical_deg"),
])
def test_a_bad_views_rule_is_refused(tmp_path, keys, value, match):
    path = _write(tmp_path, _set(keys, value))

    with pytest.raises(ValueError, match=match):
        load_rules(path)


@pytest.mark.parametrize("keys, value, match", [
    (("threshold_pct",), math.inf, r"views\.threshold_pct.*inf"),
    (("object_distance_m",), math.nan, r"views\.object_distance_m.*nan"),
    (("method", "min_outside_distance_m"), math.inf, r"views\.method\.min_outside_distance_m.*inf"),
    (("head_height_multiple",), datetime.date(2026, 1, 1), r"views\.head_height_multiple.*date\(2026, 1, 1\)"),
])
def test_a_non_finite_or_non_number_views_value_is_refused_by_name(tmp_path, keys, value, match):
    path = _write(tmp_path, _set(keys, value))

    with pytest.raises(ValueError, match=match):
        load_rules(path)


@pytest.mark.parametrize("value", [datetime.date(2026, 1, 1), 4.1, 2026])
def test_a_non_string_views_version_is_refused(tmp_path, value):
    # An unquoted `version: 2026-01-01` loads as a datetime.date and would
    # only crash later, when results.json is written.
    path = _write(tmp_path, lambda raw: raw["views"].__setitem__("version", value))

    with pytest.raises(ValueError) as excinfo:
        load_rules(path)
    message = str(excinfo.value)
    assert "views.version" in message
    assert "quote" in message
