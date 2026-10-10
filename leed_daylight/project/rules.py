"""credit_rules.yaml: the only place LEED thresholds live.

Anything not yet taken from the LEED credit text carries the literal value
UNVERIFIED.  The tool still computes and prints the percentage; it simply
awards no points and says why.

Every key the Option 1 run consumes is required.  A missing or malformed one
is refused with a ValueError naming its dotted path (e.g. ``option1.sda``,
``option1.ase.hours``) and what was expected -- never a raw
KeyError/TypeError/AttributeError from indexing straight into a
present-but-malformed section.
"""
from __future__ import annotations

import math
import pathlib

import yaml

from leed_daylight.constants import UNVERIFIED


def _join(where: str, key: str) -> str:
    return f"{where}.{key}" if where else key


def _missing(where: str, expected: str) -> None:
    raise ValueError(f"{where}: missing, expected {expected}")


def _malformed(where: str, expected: str, value) -> None:
    raise ValueError(f"{where}: expected {expected}, got {value!r}")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value) -> bool:
    return _is_number(value) and math.isfinite(value)


def _mapping(value, where: str) -> dict:
    if not isinstance(value, dict):
        _malformed(where, "a mapping", value)
    return value


def _get_mapping(parent: dict, key: str, where: str) -> dict:
    full = _join(where, key)
    if key not in parent:
        _missing(full, "a mapping")
    return _mapping(parent[key], full)


def _get_number(parent: dict, key: str, where: str):
    full = _join(where, key)
    if key not in parent:
        _missing(full, "a number")
    value = parent[key]
    if not _is_number(value):
        _malformed(full, "a number", value)
    return value


def _get_int(parent: dict, key: str, where: str, *, positive: bool = False) -> int:
    full = _join(where, key)
    expected = "a positive int" if positive else "an int"
    if key not in parent:
        _missing(full, expected)
    value = parent[key]
    if isinstance(value, bool) or not isinstance(value, int):
        _malformed(full, expected, value)
    if positive and value <= 0:
        _malformed(full, expected, value)
    return value


def _get_bool(parent: dict, key: str, where: str) -> bool:
    full = _join(where, key)
    if key not in parent:
        _missing(full, "a bool")
    value = parent[key]
    if not isinstance(value, bool):
        _malformed(full, "a bool", value)
    return value


def _check_bands(bands, where: str):
    """Validate a bands value: UNVERIFIED, or a non-empty list of bands.

    ``where`` is already the full dotted path to the bands key itself
    (e.g. ``option1.sda.bands``). Each band's own ``min_pct``/``points`` are
    checked as numbers here too (fix round 1, ruling P24): a band whose
    ``min_pct`` was not a number used to reach ``float(band["min_pct"])`` in
    the sort below and surface float()'s own bare message instead of a
    named refusal.
    """
    if bands == UNVERIFIED:
        return UNVERIFIED
    if not isinstance(bands, list) or not bands:
        _malformed(where, f"a non-empty list or {UNVERIFIED}", bands)
    for i, band in enumerate(bands):
        band_where = f"{where}[{i}]"
        band = _mapping(band, band_where)
        _get_number(band, "min_pct", band_where)
        _get_number(band, "points", band_where)
    return sorted(bands, key=lambda b: float(b["min_pct"]))


def _check_sda(sda: dict, where: str) -> None:
    _get_number(sda, "lux", where)
    fraction = _get_number(sda, "fraction", where)
    fraction_path = _join(where, "fraction")
    if not (0.0 < fraction <= 1.0):
        _malformed(fraction_path, "a number in (0, 1]", fraction)
    bands_path = _join(where, "bands")
    if "bands" not in sda:
        _missing(bands_path, f"a list or {UNVERIFIED}")
    sda["bands"] = _check_bands(sda["bands"], bands_path)
    if "additional_point" in sda:
        _check_additional_point(sda["additional_point"], _join(where, "additional_point"))


def _check_additional_point(additional_point, where: str) -> None:
    if additional_point == UNVERIFIED:
        return
    additional_point = _mapping(additional_point, where)
    _get_number(additional_point, "every_space_min_pct", where)
    if "only_if_points_at_least" in additional_point:
        # The old name read as ">=", but the credit (and the scoring) is an
        # exact match: the extra point applies only at exactly this many
        # band points; above it the same condition is exemplary performance.
        raise ValueError(
            f"{_join(where, 'only_if_points_at_least')}: this key was renamed "
            "only_if_points -- the additional point applies only at exactly "
            "that many band points, not at least that many"
        )
    _get_int(additional_point, "only_if_points", where)
    _get_int(additional_point, "points", where)


def _check_ase(ase: dict, where: str) -> None:
    _get_number(ase, "lux", where)
    _get_number(ase, "hours", where)
    _get_number(ase, "per_space_narrative_trigger_pct", where)
    # project_threshold_pct is optional and may be null (v4.1 removed the
    # project ASE threshold); it is not otherwise validated here.


#: The views keys that come from the credit text. Any of them may be
#: UNVERIFIED; the tool then computes but awards no point.
_VIEWS_CREDIT_KEYS = (
    "threshold_pct", "vlt_min_exclusive", "head_height_multiple",
    "object_distance_m", "obstruction_exception_m",
)


def _views_number(views: dict, key: str, check, expected: str) -> None:
    full = f"views.{key}"
    if key not in views:
        _missing(full, f"a number or {UNVERIFIED}")
    value = views[key]
    if value == UNVERIFIED:
        return
    if not _is_finite_number(value) or not check(value):
        _malformed(full, expected, value)


def _check_intent(mapping: dict, where: str) -> None:
    """``intent`` is optional everywhere it may appear (pre-flight ruling
    F8): the reports read the credit's intent sentence from here rather than
    carry their own copy, so a project without it still loads -- the report
    then omits the paragraph and warns instead of refusing outright. When it
    is present it must be a non-empty string (fix round 1): an empty or
    blank sentence would silently produce a blank paragraph instead of the
    warning a missing key gets.
    """
    if "intent" not in mapping:
        return
    value = mapping["intent"]
    if not isinstance(value, str) or not value.strip():
        _malformed(_join(where, "intent"), "a non-empty string", value)


def _check_space_rule_sda_pct(option1: dict, where: str) -> None:
    """``space_rule_sda_pct`` is optional (pre-flight ruling F8; Task 7
    reads it for its per-space sDA bands), but when present it must be a
    non-empty list of numbers in (0, 100], strictly increasing -- fix round
    1: this key was passed through unvalidated, unlike every other key
    ``load_rules`` handles.
    """
    key = "space_rule_sda_pct"
    if key not in option1:
        return
    full = _join(where, key)
    expected = "a non-empty list of numbers in (0, 100], strictly increasing"
    value = option1[key]
    if not isinstance(value, list) or not value:
        _malformed(full, expected, value)
    if not all(_is_finite_number(v) and 0.0 < v <= 100.0 for v in value):
        _malformed(full, expected, value)
    if any(a >= b for a, b in zip(value, value[1:])):
        _malformed(full, expected, value)


def _check_views(views: dict) -> None:
    _check_intent(views, "views")
    if "version" not in views:
        _missing("views.version", "a value")
    if not isinstance(views["version"], str):
        # An unquoted `version: 2026-01-01` loads as a datetime.date, which
        # only fails much later, when results.json is written.
        _malformed(
            "views.version",
            "a string (quote it in credit_rules.yaml, e.g. version: \"2026-01-01\")",
            views["version"],
        )
    definition = views.get("quality_definition")
    if definition != "type2_and_type3":
        _malformed("views.quality_definition", "'type2_and_type3'", definition)
    _views_number(views, "threshold_pct", lambda v: 0 < v <= 100, "a number in (0, 100]")
    _views_number(views, "vlt_min_exclusive", lambda v: 0 <= v < 1, "a number in [0, 1)")
    _views_number(views, "head_height_multiple", lambda v: v > 0, "a positive number")
    _views_number(views, "object_distance_m", lambda v: v > 0, "a positive number")
    _views_number(views, "obstruction_exception_m", lambda v: v >= 0, "a number >= 0")
    _get_int(views, "points", "views", positive=True)
    method = _get_mapping(views, "method", "views")
    statistic = method.get("distance_statistic")
    if statistic != "median":
        _malformed("views.method.distance_statistic", "'median'", statistic)
    cut_off = _get_number(method, "min_outside_distance_m", "views.method")
    if not (math.isfinite(cut_off) and cut_off >= 0):
        _malformed("views.method.min_outside_distance_m", "a number >= 0", cut_off)
    vertical = _get_number(method, "panorama_vertical_deg", "views.method")
    if not (math.isfinite(vertical) and 0 < vertical <= 180):
        _malformed("views.method.panorama_vertical_deg", "a number in (0, 180]", vertical)


def view_rules(rules: dict) -> dict:
    """The validated `views` section of a loaded rules mapping, or a refusal."""
    if "views" not in rules:
        _missing("views", "a mapping; credit_rules.yaml has no Quality Views section")
    return rules["views"]


def views_verified(views: dict) -> bool:
    """True when every credit-text value of the views section is stated."""
    return all(views[key] != UNVERIFIED for key in _VIEWS_CREDIT_KEYS)


def load_rules(path: str | pathlib.Path) -> dict:
    """Read and validate credit_rules.yaml.  Bands come back sorted ascending."""
    path = pathlib.Path(path)
    rules = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(rules, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    if "version" not in rules:
        _missing("version", "a value")
    _check_intent(rules, "")

    _get_int(rules, "max_points", "", positive=True)

    option1 = _get_mapping(rules, "option1", "")
    sda = _get_mapping(option1, "sda", "option1")
    _check_sda(sda, "option1.sda")
    ase = _get_mapping(option1, "ase", "option1")
    _check_ase(ase, "option1.ase")
    _check_space_rule_sda_pct(option1, "option1")

    option2 = _get_mapping(rules, "option2", "")
    bands_path = "option2.bands"
    if "bands" not in option2:
        _missing(bands_path, f"a list or {UNVERIFIED}")
    option2["bands"] = _check_bands(option2["bands"], bands_path)
    # The band Option 2 is judged on: a rules file without it would let a project's own band pass as LEED's.
    lux_min = _get_number(option2, "lux_min", "option2")
    lux_max = _get_number(option2, "lux_max", "option2")
    if not (0 < lux_min < lux_max):
        _malformed("option2.lux_min", "a number above 0 and below option2.lux_max", lux_min)

    _get_bool(rules, "glare_control_required", "")

    if "views" in rules:
        _check_views(_mapping(rules["views"], "views"))

    return rules
