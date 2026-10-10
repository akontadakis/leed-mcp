# tests/test_deliver_word_views.py
"""The Quality Views project and storey reports (reports spec 6.3, 6.4).

Pre-flight ruling F10: the spaces table and the excluded-spaces table are
checked by row count and by the numbers in named cells (via the shared
``table_after`` helper), not by substring presence alone.
"""
from __future__ import annotations

import dataclasses
import types

import pytest

pytest.importorskip("docx", reason="the report extras are not installed: pip install -e '.[report]'")

from deliver_folder import ROOM, build_deliver_project, doc_headings, doc_text, edit_results, table_after
from leed_daylight.deliver.data import load_case
from leed_daylight.deliver.figures.base import get_renderer
from leed_daylight.deliver.word import common, views


@pytest.fixture
def folder(tmp_path, write_example, write_wea):
    return build_deliver_project(tmp_path, write_example, write_wea, options=("views",))


@pytest.fixture
def case(folder):
    return load_case(folder)


@pytest.fixture
def renderer():
    return get_renderer("pillow")


def _ordered_headings(doc) -> list[str]:
    """Title, level-1 and level-2 heading text (reports spec order checks;
    F11: via the shared ``doc_headings`` helper). There is no level-3
    heading in a Quality Views report, unlike Option 1 and Option 2."""
    return [text for style, text in doc_headings(doc) if style != "Heading 3"]


def test_project_report_sections_in_order(case, renderer):
    built = views.build_project(case, renderer)

    assert _ordered_headings(built.document)[1:] == [
        "Intent", "Requirements", "Input data", "Method", "Summary results", "Deviations",
        "Warnings", "Spaces", "Excluded spaces", "Results by storey", "Floor plans", "Storey L0",
    ]


def test_project_report_figures(case, renderer):
    built = views.build_project(case, renderer)

    names = [name for name, _ in built.figures]
    assert names == ["storeys_quality.png", "L0_type2.png", "L0_type3.png", "L0_quality.png"]
    assert len(built.document.inline_shapes) == len(names)


def test_project_report_method_notes_and_input_data_come_from_the_results(case, renderer):
    results = case.options["views"].results
    settings = results["settings"]
    built = views.build_project(case, renderer)

    text = doc_text(built.document)
    assert "Invented method note." in [p.text for p in built.document.paragraphs]
    assert str(settings["eye_height_source"]) in text
    assert f"{float(settings['eye_height_m']):g} m above the floor" in text
    assert ROOM in text


def test_project_report_summary_numbers_come_from_the_results(case, renderer):
    results = case.options["views"].results
    built = views.build_project(case, renderer)

    text = doc_text(built.document)
    assert f"{results['quality_area_pct']:.1f} %" in text
    assert f"{results['type2_area_pct']:.1f} %" in text
    assert f"{results['type3_area_pct']:.1f} %" in text


def test_project_report_spaces_table_has_one_row_per_room_with_the_invented_percentages(case, renderer):
    # F10: the named room's Type 2 / Type 3 / Quality View percentages must
    # equal the invented results, in the right cell of the right row -- not
    # merely present somewhere in the document.
    results = case.options["views"].results
    (room,) = [s for s in results["spaces"] if s["room"] == ROOM]
    built = views.build_project(case, renderer)

    table = table_after(built.document, "Spaces", level=1)
    assert len(table.rows) == 1 + len(results["spaces"]) == 2  # header + the one invented room
    row = [c.text for c in table.rows[1].cells]
    assert row[0] == "L0"
    assert row[1] == ROOM
    assert row[2] == f"{float(room['area_m2']):,.1f}"
    assert row[3] == str(room["points"])
    assert row[4] == f"{room['type2_pct']:.1f} %"
    assert row[5] == f"{room['type3_pct']:.1f} %"
    assert row[6] == f"{room['quality_pct']:.1f} %"
    # The invented fake fixture never has everyone pass, so this is a real
    # check, not a trivial 0 % or 100 %.
    assert 0.0 < room["quality_pct"] < 100.0


def test_no_excluded_spaces_is_said_plainly(case, renderer):
    results = case.options["views"].results
    assert results["excluded_spaces"] == []
    built = views.build_project(case, renderer)

    paragraphs = [p.text for p in built.document.paragraphs]
    assert "None: every regularly occupied space is in the analysis." in paragraphs


def test_excluded_spaces_are_listed_with_their_reason(folder, renderer):
    # F10: row count plus the room and its reason in the right cells, not
    # just present somewhere in the document.
    edit_results(folder, "views", lambda p: p.update(
        {"excluded_spaces": [{"room": "Store Room", "reason": "not regularly occupied"}]}))
    case = load_case(folder)

    built = views.build_project(case, renderer)

    table = table_after(built.document, "Excluded spaces", level=1)
    assert len(table.rows) == 2  # header + the one excluded space
    assert [c.text for c in table.rows[0].cells] == ["Space", "Reason"]
    assert [c.text for c in table.rows[1].cells] == ["Store Room", "not regularly occupied"]
    # The included room still appears in the (unaffected) Spaces table.
    spaces_table = table_after(built.document, "Spaces", level=1)
    assert ROOM in [c.text for row in spaces_table.rows for c in row.cells]


def test_excluded_spaces_with_no_reason_says_so(folder, renderer):
    edit_results(folder, "views", lambda p: p.update(
        {"excluded_spaces": [{"room": "Store Room", "reason": None}]}))
    case = load_case(folder)

    built = views.build_project(case, renderer)

    table = table_after(built.document, "Excluded spaces", level=1)
    assert [c.text for c in table.rows[1].cells] == ["Store Room", "no reason given"]


def test_storey_report_sections_in_order(case, renderer):
    built = views.build_storey(case, "L0", renderer)

    assert _ordered_headings(built.document)[1:] == [
        "Intent", "Requirements", "Input data", "Method", "Summary results, storey L0",
        "Deviations", "Warnings", "Spaces", "Floor plans", "Storey L0",
        "The project behind this storey", "Results by storey",
    ]
    text = doc_text(built.document)
    assert "Scope of this report." in text
    assert "L0 (this report)" in text
    # Spec 6.4: the storey report carries the same sections cut to one
    # storey; "Excluded spaces" is a project-wide concept and is left out.
    assert "Excluded spaces" not in _ordered_headings(built.document)


def test_storey_report_figures_and_no_project_wide_chart(case, renderer):
    built = views.build_storey(case, "L0", renderer)

    names = [name for name, _ in built.figures]
    assert names == ["L0_type2.png", "L0_type3.png", "L0_quality.png"]  # no storeys_quality.png (chart=False)


def test_storey_report_numbers_come_from_the_results(case, renderer):
    results = case.options["views"].results
    (room,) = [s for s in results["spaces"] if s["room"] == ROOM]
    built = views.build_storey(case, "L0", renderer)

    text = doc_text(built.document)
    assert f"{room['quality_pct']:.1f} %" in text

    project_table = table_after(built.document, "The project behind this storey", level=1)
    assert [c.text for c in project_table.rows[0].cells] == [
        "Project Quality View", f"{results['quality_area_pct']:.1f} %",
    ]

    storey_table = table_after(built.document, "Results by storey", level=1)
    assert len(storey_table.rows) == 2  # header + the one storey, marked, no total
    storey_row = [c.text for c in storey_table.rows[1].cells]
    assert storey_row[0] == "L0 (this report)"
    assert storey_row[3] == f"{results['quality_area_pct']:.1f} %"


def test_by_storey_chart_builds_with_no_threshold_line_when_the_credit_rules_have_no_views_section(case, renderer):
    # Task 9 ruling: any view threshold or band comes from the credit rules
    # / results, never a literal in code. With no `views:` section in
    # credit_rules.yaml (this fixture's default), the bar chart still
    # builds with no threshold line rather than a fabricated default.
    assert "views" not in case.rules

    built = views.build_project(case, renderer)

    assert [name for name, _ in built.figures][0] == "storeys_quality.png"


def test_no_views_points_value_omits_the_band_and_warns(case, renderer):
    # Task 9 ruling: a `views` section with a threshold but no `points`
    # value used to fall back to a literal 1 silently; it must now warn
    # instead of fabricating a point count.
    rules = dict(case.rules)
    rules["views"] = {"threshold_pct": 75.0}
    with_rules = dataclasses.replace(case, rules=types.MappingProxyType(rules))

    built = views.build_project(with_rules, renderer)

    text = doc_text(built.document)
    assert "Floor area with a Quality View" not in text
    assert common.NO_VIEWS_POINTS_WARNING in text
