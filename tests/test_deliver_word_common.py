# tests/test_deliver_word_common.py
"""Shared report sections (reports spec 6)."""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

pytest.importorskip("docx", reason="the report extras are not installed: pip install -e '.[report]'")

from deliver_folder import ROOM, build_deliver_project, doc_headings, doc_text
from leed_daylight.deliver.data import load_case
from leed_daylight.deliver.figures.base import get_renderer, render_bar
from leed_daylight.deliver.word import common, kit
from leed_daylight.project.config import ReportSpec


@pytest.fixture
def case(tmp_path, write_example, write_wea):
    return load_case(build_deliver_project(tmp_path, write_example, write_wea))


def _with_rules(case, rules: dict):
    return dataclasses.replace(case, rules=types.MappingProxyType(rules))


def test_new_doc_page_layout():
    # python-docx stores page geometry in twips, so a value set in Cm reads
    # back a few hundred EMU (a fraction of a hair's width) off; round to
    # 0.1 cm, well above that rounding, rather than compare Length objects.
    doc = kit.new_doc("Example Tower, page")
    section = doc.sections[0]

    assert round(section.page_width.cm, 1) == 21.0
    assert round(section.page_height.cm, 1) == 29.7
    assert round(section.left_margin.cm, 1) == round(section.right_margin.cm, 1) == 1.8
    assert round(section.top_margin.cm, 1) == 1.8
    assert round(section.bottom_margin.cm, 1) == 1.8  # F16: 1.8 cm on every side, not 1.5
    normal = doc.styles["Normal"]
    assert normal.font.name == "Calibri"
    assert normal.font.size == kit.Pt(10)
    footer_text = section.footer.paragraphs[0].text
    assert footer_text.startswith("Example Tower, page")


def test_small_formatters():
    assert common.safe_name("Level 1 / East") == "Level_1_East"
    assert common.safe_name("///") == "storey"
    assert common.pct(None) == "not computed"
    assert common.pct(54.321) == "54.3 %"
    assert common.area(1234.56) == "1,234.6"
    assert common.score_text({"points": 2, "band": "55 %"}) == "2 (55 %)"
    assert common.score_text({"points": 0, "band": None}) == "0 (below the lowest band)"
    assert common.score_text({"points": None, "reason": "bands UNVERIFIED"}) == "not awarded (bands UNVERIFIED)"


def test_new_report_title_block(case):
    project = dataclasses.replace(case.project, report=ReportSpec(prepared_by="Example Consultants"))
    built = common.new_report(dataclasses.replace(case, project=project), "option1", "storey L0")

    doc = built.document
    assert doc_headings(doc)[0] == ("Title", f"{common.OPTION_TITLES['option1']}, storey L0")
    text = doc_text(doc)
    assert "Project: Example Tower" in text
    assert "Case: base_case" in text
    assert "Prepared by: Example Consultants" in text


@pytest.mark.parametrize("option, needle", [
    ("option1", "300"), ("option2", "300"), ("views", "Quality Views"),
])
def test_intent_and_requirements(case, option, needle):
    built = common.new_report(case, option)
    common.intent_and_requirements(built, case, option)

    headings = [t for _, t in doc_headings(built.document)]
    assert headings[1:3] == ["Intent", "Requirements"]
    assert needle in doc_text(built.document)


# Pre-flight ruling F8: the intent sentence comes from credit_rules.yaml,
# not from a sentence hard-coded in common.py. Both the present and the
# absent case are tested.
DAYLIGHT_INTENT = "Connect building occupants with the outdoors."
VIEWS_INTENT = "Give building occupants a connection to the outdoors."


@pytest.mark.parametrize("option, key_path", [
    ("option1", ("intent",)), ("option2", ("intent",)), ("views", ("views", "intent")),
])
def test_intent_present_is_read_from_the_credit_rules_file(case, option, key_path):
    rules = dict(case.rules)
    if key_path == ("intent",):
        rules["intent"] = DAYLIGHT_INTENT
        expected = DAYLIGHT_INTENT
    else:
        rules["views"] = {**dict(rules.get("views", {})), "intent": VIEWS_INTENT}
        expected = VIEWS_INTENT
    with_intent = _with_rules(case, rules)
    built = common.new_report(with_intent, option)

    common.intent_and_requirements(built, with_intent, option)
    common.deviations_and_warnings(built, with_intent, with_intent.options[option])

    text = doc_text(built.document)
    assert expected in text
    assert "no intent text" not in text


@pytest.mark.parametrize("option", ["option1", "option2", "views"])
def test_intent_absent_omits_the_paragraph_and_warns(case, option):
    # `case`'s rules come from MINIMAL_CREDIT_RULES (tests/conftest.py),
    # which deliberately carries no intent text at all.
    assert "intent" not in case.rules
    built = common.new_report(case, option)

    common.intent_and_requirements(built, case, option)
    common.deviations_and_warnings(built, case, case.options[option])

    text = doc_text(built.document)
    assert "the credit rules file has no intent text" in text


# Task 9 ruling: the Quality View points band comes from credit_rules.yaml
# views.points, never a literal fallback. Both the present and the absent
# case are tested, mirroring the F8 intent tests above.
def test_views_points_present_is_read_from_the_credit_rules_file(case):
    rules = dict(case.rules)
    rules["views"] = {**dict(rules.get("views", {})), "threshold_pct": 75.0, "points": 2}
    with_points = _with_rules(case, rules)
    built = common.new_report(with_points, "views")

    common.intent_and_requirements(built, with_points, "views")
    common.deviations_and_warnings(built, with_points, with_points.options["views"])

    text = doc_text(built.document)
    assert "Floor area with a Quality View" in text
    assert "75" in text and "2" in text
    assert common.NO_VIEWS_POINTS_WARNING not in text


def test_views_points_absent_omits_the_band_and_warns(case):
    # A threshold with no points value used to fall back to a literal 1
    # silently; it must now warn instead of fabricating a point count.
    rules = dict(case.rules)
    rules["views"] = {**dict(rules.get("views", {})), "threshold_pct": 75.0}
    with_threshold = _with_rules(case, rules)
    built = common.new_report(with_threshold, "views")

    common.intent_and_requirements(built, with_threshold, "views")
    common.deviations_and_warnings(built, with_threshold, with_threshold.options["views"])

    text = doc_text(built.document)
    assert "Floor area with a Quality View" not in text
    assert common.NO_VIEWS_POINTS_WARNING in text


def test_deviations_come_from_project_yaml_and_warnings_from_the_results(case):
    project = dataclasses.replace(case.project, report=ReportSpec(deviations=("Furniture is not modelled.",)))
    case = dataclasses.replace(case, project=project, warnings=("an invented case warning",))
    built = common.new_report(case, "option1")

    common.deviations_and_warnings(built, case, case.options["option1"])

    text = doc_text(built.document)
    assert "Furniture is not modelled." in text
    assert "an invented case warning" in text
    assert "the blind trigger area is the room area" in text  # a warning the Option 1 run always writes


def test_no_deviations_is_said_plainly(case):
    built = common.new_report(case, "option2")
    common.deviations_and_warnings(built, case, case.options["option2"])
    assert "None stated in project.yaml" in doc_text(built.document)


@pytest.fixture
def two_storey_case(tmp_path, write_example, write_wea):
    from deliver_folder import build_two_storey_project
    return load_case(build_two_storey_project(tmp_path, write_example, write_wea))


@pytest.mark.parametrize("option", ["option1", "option2", "views"])
def test_storey_plan_keeps_only_that_storeys_points(two_storey_case, option):
    """Final review I6: with two storeys the filter really filters."""
    from deliver_folder import UPPER_ROOM

    case = two_storey_case
    data = case.options[option]
    lower, upper = case.storeys
    total = data.points.xy.shape[0]
    specs = {s.label: common.storey_plan(case, s, data, title="t", passed=np.ones(total, dtype=bool))
             for s in (lower, upper)}

    n_lower, n_upper = (int(data.points.mask(s.rooms).sum()) for s in (lower, upper))
    assert 0 < n_lower < total and n_lower + n_upper == total
    assert specs["L0"].points_xy.shape[0] == n_lower
    assert specs["L1"].points_xy.shape[0] == n_upper
    assert [label[0] for label in specs["L1"].labels] == [UPPER_ROOM]
    assert specs["L0"].north_rotation_deg == case.project.site.north_rotation_deg

    empty = common.storey_plan(case, lower, data, title="zones", points=False)
    assert empty.points_xy.shape == (0, 2)


@pytest.mark.parametrize("option", ["option1", "option2", "views"])
def test_storey_plan_draws_only_the_spaces_in_the_analysis(two_storey_case, option):
    """Final review I3 (Ruling FR-I3): a space the case leaves out is
    neither outlined nor labelled on the plans."""
    from deliver_folder import EXCLUDED_ROOM

    case = two_storey_case
    lower = case.storey("L0")
    assert EXCLUDED_ROOM in lower.rooms

    spec = common.storey_plan(case, lower, case.options[option], title="zones", points=False)

    assert [label[0] for label in spec.labels] == [ROOM]
    assert len(spec.outlines) == len(lower.outlines[ROOM])


def test_storey_averages_and_bar(case):
    option = case.options["option1"]
    ((label, count, area_m2, value),) = common.storey_averages(case, option, "sda_pct")
    assert (label, count, area_m2) == ("L0", 1, 24.0)
    assert 0.0 < value < 100.0
    bar = common.storey_bar(case, option, title="sDA", key="sda_pct", unit="%", threshold=55.0)
    assert bar.categories == ("L0",)


def test_add_figure_records_the_png(case):
    built = common.new_report(case, "option1")
    renderer = get_renderer("pillow")
    bar = common.storey_bar(case, case.options["option1"], title="sDA", key="sda_pct", unit="%", threshold=55.0)
    png = render_bar(bar, renderer)

    common.add_figure(built, "bar.png", png, "An invented caption.")

    assert built.figures == [("bar.png", png)]
    assert len(built.document.inline_shapes) == 1
    assert "An invented caption." in doc_text(built.document)


def test_add_figure_fits_a_tall_picture_to_the_maximum_height(case):
    """Final review I4 (Ruling FR-I4): a picture taller than the page's
    figure room is placed by its height, so it never runs off the page."""
    import io

    from PIL import Image

    from leed_daylight.deliver.figures.spec import MAX_PRINT_HEIGHT_CM, PRINT_WIDTH_CM

    buffer = io.BytesIO()
    Image.new("RGB", (400, 1600), "white").save(buffer, format="PNG")
    built = common.new_report(case, "option1")

    common.add_figure(built, "tall.png", buffer.getvalue(), "An invented caption.")

    (shape,) = built.document.inline_shapes
    assert PRINT_WIDTH_CM == kit.USABLE
    assert shape.height.cm <= MAX_PRINT_HEIGHT_CM + 0.01
    assert shape.width.cm <= kit.USABLE + 0.01
