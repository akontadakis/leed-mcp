# tests/test_deliver_word_option2.py
"""The Option 2 project and storey reports (reports spec 6.2, 6.4).

Pre-flight ruling F9: spec 6.2 asks for, near each equinox, "the candidates
with direct-normal irradiance and Perez clearness, and any day-choice
warning" -- not just the chosen day. ``test_..._candidates_table...`` reads
the candidates straight from the invented results' own ``days[].top``, and
``test_..._day_choice_warning...`` injects a warning shaped like the one
``metrics.clearest_day`` writes to check it lands under the right equinox's
candidates table, not just anywhere in the document.

Pre-flight ruling F10: the four-skies and spaces tables are checked by row
count and by the numbers in named cells (via the shared ``table_after``
helper), not by substring presence alone.
"""
from __future__ import annotations

import dataclasses
import types

import pytest

pytest.importorskip("docx", reason="the report extras are not installed: pip install -e '.[report]'")

from deliver_folder import ROOM, build_deliver_project, doc_headings, doc_text, table_after
from leed_daylight.deliver.data import load_case
from leed_daylight.deliver.figures.base import get_renderer
from leed_daylight.deliver.word import option2


@pytest.fixture
def case(tmp_path, write_example, write_wea):
    return load_case(build_deliver_project(tmp_path, write_example, write_wea, options=("option2",)))


@pytest.fixture
def renderer():
    return get_renderer("pillow")


def _ordered_headings(doc) -> list[str]:
    """Title, level-1 and level-2 heading text (reports spec order checks);
    level 3 -- the per-equinox "Candidates near ..." tables -- is left out,
    exactly as the per-storey "Spaces" table is left out in the Option 1
    tests (F11: via the shared ``doc_headings`` helper)."""
    return [text for style, text in doc_headings(doc) if style != "Heading 3"]


def test_project_report_sections_in_order(case, renderer):
    built = option2.build_project(case, renderer)

    assert _ordered_headings(built.document)[1:] == [
        "Intent", "Requirements", "Input data", "Method", "The chosen days", "The skies",
        "Summary results", "Deviations", "Warnings", "Spaces", "Results by storey",
        "Floor plans", "Storey L0",
    ]


def test_project_report_names_the_days_and_skies(case, renderer):
    built = option2.build_project(case, renderer)

    text = doc_text(built.document)
    assert "03-28" in text and "09-14" in text  # the invented clear days
    assert "03-28_0900" in text
    assert ROOM in text


def test_project_report_figures(case, renderer):
    built = option2.build_project(case, renderer)

    names = [name for name, _ in built.figures]
    assert names == [
        "storeys_pass.png",
        "L0_sky_03-28_0900.png", "L0_sky_03-28_1500.png", "L0_sky_09-14_0900.png", "L0_sky_09-14_1500.png",
        "L0_average_09_00.png", "L0_average_15_00.png", "L0_pass.png",
    ]
    assert len(built.document.inline_shapes) == len(names)


def test_storey_report_sections_in_order(case, renderer):
    built = option2.build_storey(case, "L0", renderer)

    assert _ordered_headings(built.document)[1:] == [
        "Intent", "Requirements", "Input data", "Method", "Summary results, storey L0",
        "Deviations", "Warnings", "Spaces", "Floor plans", "Storey L0",
        "The project behind this storey", "Results by storey",
    ]


def test_project_report_candidates_table_reads_top_from_the_results(case, renderer):
    # F9: a candidates table per equinox, from days[].top, with numbers
    # checked against the invented results -- not just the chosen day.
    results = case.options["option2"].results
    built = option2.build_project(case, renderer)

    for day in results["days"]:
        equinox = day["equinox"]
        table = table_after(built.document, f"Candidates near {equinox}", level=3)
        assert len(table.rows) == 1 + len(day["top"])  # header + one row per candidate
        assert [c.text for c in table.rows[0].cells] == [
            "Date", "Offset days", "Direct normal Wh/m2", "Perez clearness",
        ]
        first = [c.text for c in table.rows[1].cells]
        top0 = day["top"][0]
        # ranked[0] is always the chosen day (metrics.clearest_day._rank_key).
        assert top0["date"] == day["chosen"]["date"]
        assert first[0] == top0["date"]
        assert first[1] == str(top0["offset_days"])
        assert first[2] == f"{float(top0['direct_normal_wh_m2']):,.0f}"
        assert first[3] == f"{float(top0['perez_epsilon']):.2f}"


def test_project_report_day_choice_warning_appears_under_its_equinox(case, renderer):
    # F9: "any day-choice warning" -- this fixture's own two clear days never
    # trigger one (metrics.clearest_day never disagrees with itself), so a
    # warning shaped exactly like the one that module writes is injected
    # (same technique as the Option 1 tests' _with_space_rule: replace the
    # results mapping, never edit the credit rules or invent a number).
    warning = "clearest day near 03-21: direct sun picks 03-28 (invented test warning); the direct-sun choice stands"
    other_warning = "clearest day near 09-21: an unrelated invented warning that must not appear near 03-21"
    results = dict(case.options["option2"].results)
    results["warnings"] = [*results["warnings"], warning, other_warning]
    option = dataclasses.replace(case.options["option2"], results=types.MappingProxyType(results))
    with_warning = dataclasses.replace(case, options=types.MappingProxyType({"option2": option}))

    built = option2.build_project(with_warning, renderer)

    paragraphs = [p.text for p in built.document.paragraphs]
    start = paragraphs.index("Candidates near 03-21")
    end = paragraphs.index("Candidates near 09-21")
    assert warning in paragraphs[start:end]
    assert other_warning not in paragraphs[start:end]
    # Also reaches the general Warnings section further down (common.py).
    assert warning in doc_text(built.document)


def test_project_report_skies_table_has_four_rows_with_the_invented_irradiance(case, renderer):
    # F10: row count plus the numbers in named cells, not substring presence.
    results = case.options["option2"].results
    built = option2.build_project(case, renderer)

    table = table_after(built.document, "The skies", level=1)
    assert len(table.rows) == 1 + len(results["skies"]) == 5  # header + the four skies
    for row, sky in zip(table.rows[1:], results["skies"]):
        cells = [c.text for c in row.cells]
        assert cells[0] == sky["label"]
        assert cells[1] == f"{float(sky['direct_normal_w_m2']):g}"
        assert cells[2] == f"{float(sky['diffuse_horizontal_w_m2']):g}"


def test_project_report_spaces_table_has_one_row_per_room_with_the_invented_pass_pct(case, renderer):
    # F10: the named room's pass % must equal the invented results, in the
    # right cell of the right row -- not merely present somewhere in the doc.
    results = case.options["option2"].results
    (room,) = [s for s in results["spaces"] if s["room"] == ROOM]
    built = option2.build_project(case, renderer)

    table = table_after(built.document, "Spaces", level=1)
    assert len(table.rows) == 1 + len(results["spaces"]) == 2  # header + the one invented room
    row = [c.text for c in table.rows[1].cells]
    assert row[0] == "L0"
    assert row[1] == ROOM
    assert row[2] == f"{float(room['area_m2']):,.1f}"
    assert row[3] == str(room["points"])
    assert row[4] == f"{room['pass_pct']:.1f} %"
    assert row[5] == str(room["n_low"])
    assert row[6] == str(room["n_high"])
    assert row[7] == ("applied" if room["upper_limit_applied"] else "not applied")
    # The invented fake engine (deliver_folder.Option2Fake) never passes
    # everyone and never fails everyone, so this is a real check, not a
    # trivial 0 % or 100 %.
    assert 0.0 < room["pass_pct"] < 100.0


def test_the_summary_says_sensor_points_not_leed_points(case, renderer):
    """Final review M4: "Points in the band" sat next to "Points" (LEED points)."""
    doc = option2.build_project(case, renderer).document

    labels = [row.cells[0].text for row in table_after(doc, "Summary results").rows]
    assert "Sensor points in the band" in labels and "Points in the band" not in labels
