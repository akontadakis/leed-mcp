# tests/test_deliver_word_option1.py
"""The Option 1 project and storey reports (reports spec 6.1, 6.4).

Pre-flight ruling F8: the `case` fixture's credit rules (option2_folder.py's
CREDIT_RULES) carry no `option1.space_rule_sda_pct`, exactly like they carry
no `intent` (see tests/test_deliver_word_common.py). So the default `case`
here has NO "sDA by space" section -- that section, and its warning, are
tested with a case whose rules add the key back. Spec 6.4 ("the same
sections, cut to one storey") means the storey report carries this section
too, scoped to that storey's rooms; both are tested here (fix round 1).

Pre-flight ruling F9: the per-storey spaces table (spec 6.1 item 11) is
checked with the shared ``table_after`` helper (fix round 1: promoted from
a private copy here into tests/deliver_folder.py, F11), which locates the
table that immediately follows a given heading in document order, so row
counts and named cells (pre-flight ruling F10) are checked on the right
table -- not just a substring anywhere in the document.
"""
from __future__ import annotations

import dataclasses
import types

import pytest

pytest.importorskip("docx", reason="the report extras are not installed: pip install -e '.[report]'")

from deliver_folder import ROOM, build_deliver_project, build_roof_glass_project, doc_headings, doc_text, table_after
from leed_daylight.deliver.data import load_case
from leed_daylight.deliver.figures.base import get_renderer
from leed_daylight.deliver.word import option1


@pytest.fixture
def case(tmp_path, write_example, write_wea):
    return load_case(build_deliver_project(tmp_path, write_example, write_wea, options=("option1",)))


@pytest.fixture
def renderer():
    return get_renderer("pillow")


def _ordered_headings(doc) -> list[str]:
    """Title, level-1 and level-2 heading text, via the shared
    ``doc_headings`` helper (level 3 -- table headings such as "Window
    groups" and the per-storey "Spaces" table -- is left out, as it never
    takes part in the section-order checks below)."""
    return [text for style, text in doc_headings(doc) if style != "Heading 3"]


def _room_result(case):
    (space,) = [s for s in case.options["option1"].results["spaces"] if s["room"] == ROOM]
    return space


def _with_space_rule(case, limits):
    rules = dict(case.rules)
    rules["option1"] = {**dict(rules["option1"]), "space_rule_sda_pct": limits}
    return dataclasses.replace(case, rules=types.MappingProxyType(rules))


def test_project_report_sections_in_order(case, renderer):
    # F8: this fixture's credit rules have no space_rule_sda_pct (like they
    # have no intent, tests/test_deliver_word_common.py), so "sDA by space"
    # is left out of the section order.
    assert "space_rule_sda_pct" not in case.rules["option1"]
    built = option1.build_project(case, renderer)

    assert _ordered_headings(built.document)[1:] == [
        "Intent", "Requirements", "Input data", "Method", "Summary results", "Deviations",
        "Warnings", "Spaces", "Results by storey", "Spaces with the highest ASE",
        "Spaces that need a glare narrative", "Floor plans", "Storey L0",
    ]


def test_project_report_omits_the_space_rule_and_warns_when_the_key_is_absent(case, renderer):
    built = option1.build_project(case, renderer)

    assert "sDA by space" not in _ordered_headings(built.document)
    assert option1.NO_SPACE_RULE_WARNING in doc_text(built.document)


def test_project_report_space_rule_reads_limits_from_the_credit_rules(case, renderer):
    with_rule = _with_space_rule(case, [55.0, 75.0])
    room = _room_result(with_rule)

    built = option1.build_project(with_rule, renderer)

    assert "sDA by space" in _ordered_headings(built.document)
    assert option1.NO_SPACE_RULE_WARNING not in doc_text(built.document)
    table = table_after(built.document, "sDA by space", level=1)
    meeting_55 = 1 if room["sda_pct"] >= 55.0 else 0
    meeting_75 = 1 if room["sda_pct"] >= 75.0 else 0
    assert len(table.rows) == 3  # header + one row per limit
    assert [c.text for c in table.rows[1].cells] == ["sDA 55 % or more", f"{meeting_55} of 1", "0.0 %"]
    assert [c.text for c in table.rows[2].cells] == ["sDA 75 % or more", f"{meeting_75} of 1", "0.0 %"]


def test_project_report_space_rule_uses_a_different_limit_list(case, renderer):
    with_rule = _with_space_rule(case, [10.0])
    room = _room_result(with_rule)

    built = option1.build_project(with_rule, renderer)

    table = table_after(built.document, "sDA by space", level=1)
    assert len(table.rows) == 2  # header + one row for the single limit
    assert room["sda_pct"] >= 10.0  # the invented room passes this low bar
    assert [c.text for c in table.rows[1].cells] == ["sDA 10 % or more", "1 of 1", "100.0 %"]


def test_project_report_numbers_come_from_the_results(case, renderer):
    results = case.options["option1"].results
    room = _room_result(case)
    built = option1.build_project(case, renderer)

    text = doc_text(built.document)
    assert f"{results['sda_area_pct']:.1f} % (headline)" in text
    assert ROOM in text
    assert "glass_clear" in text
    assert "24.0" in text

    spaces_table = table_after(built.document, "Spaces", level=1)
    # F10: header + one row per included room + a total row.
    assert len(spaces_table.rows) == 3
    row = [c.text for c in spaces_table.rows[1].cells]
    assert row[0] == "L0"
    assert row[1] == ROOM
    assert row[2] == "24.0"
    assert row[3] == str(room["points"])
    assert row[4] == f"{room['sda_pct']:.1f} %"
    assert row[5] == f"{room['ase_pct']:.1f} %"
    assert row[6] == room["ase_band"]
    total_row = [c.text for c in spaces_table.rows[2].cells]
    assert total_row[0] == "Total"
    assert total_row[2] == "24.0"
    assert total_row[3] == str(room["points"])

    storey_table = table_after(built.document, "Results by storey", level=1)
    assert len(storey_table.rows) == 2  # header + one storey, no mark and no total
    storey_row = [c.text for c in storey_table.rows[1].cells]
    assert storey_row[0] == "L0"
    assert storey_row[1] == "1"
    assert storey_row[3] == f"{results['sda_area_pct']:.1f} %"
    assert storey_row[4] == f"{results['ase_area_pct']:.1f} %"


def test_project_report_figures(case, renderer):
    built = option1.build_project(case, renderer)

    names = [name for name, _ in built.figures]
    assert names == ["storeys_sda.png", "L0_zones.png", "L0_grid.png", "L0_sda.png", "L0_ase.png"]
    assert len(built.document.inline_shapes) == 5
    assert all(png[:4] == b"\x89PNG" for _, png in built.figures)


def test_storey_report_sections_in_order(case, renderer):
    built = option1.build_storey(case, "L0", renderer)

    assert _ordered_headings(built.document)[1:] == [
        "Intent", "Requirements", "Input data", "Method", "Summary results, storey L0",
        "Deviations", "Warnings", "Spaces", "Floor plans", "Storey L0",
        "The project behind this storey", "Results by storey",
    ]
    text = doc_text(built.document)
    assert "Scope of this report." in text
    assert "L0 (this report)" in text


def test_storey_report_omits_the_space_rule_and_warns_when_the_key_is_absent(case, renderer):
    built = option1.build_storey(case, "L0", renderer)

    assert "sDA by space" not in _ordered_headings(built.document)
    assert option1.NO_SPACE_RULE_WARNING in doc_text(built.document)


def test_storey_report_space_rule_reads_limits_scoped_to_the_storey(case, renderer):
    # Controller ruling, fix round 1: spec 6.4 says a storey report carries
    # "the same sections, cut to one storey" -- spec 6.1 item 8's space
    # rule included, computed over this storey's own rooms and area.
    with_rule = _with_space_rule(case, [55.0, 75.0])
    room = _room_result(with_rule)

    built = option1.build_storey(with_rule, "L0", renderer)

    assert _ordered_headings(built.document)[1:] == [
        "Intent", "Requirements", "Input data", "Method", "Summary results, storey L0",
        "Deviations", "Warnings", "Spaces", "sDA by space", "Floor plans", "Storey L0",
        "The project behind this storey", "Results by storey",
    ]
    assert option1.NO_SPACE_RULE_WARNING not in doc_text(built.document)
    table = table_after(built.document, "sDA by space", level=1)
    meeting_55 = 1 if room["sda_pct"] >= 55.0 else 0
    meeting_75 = 1 if room["sda_pct"] >= 75.0 else 0
    assert len(table.rows) == 3
    # This fixture has only one storey, so "scoped to the storey" and
    # "project-wide" happen to be the same one room -- the share is still
    # computed from spaces_of(option, rooms=storey.rooms), not the whole
    # project, per _space_rule's `rooms` parameter.
    assert [c.text for c in table.rows[1].cells] == ["sDA 55 % or more", f"{meeting_55} of 1", "0.0 %"]
    assert [c.text for c in table.rows[2].cells] == ["sDA 75 % or more", f"{meeting_75} of 1", "0.0 %"]


def test_the_glare_narrative_list_names_spaces_over_the_trigger(case, renderer):
    results = dict(case.options["option1"].results)
    results["spaces"] = [dict(s, ase_pct=25.0, needs_glare_narrative=True) for s in results["spaces"]]
    results["narrative_spaces"] = [ROOM]
    option = dataclasses.replace(case.options["option1"], results=types.MappingProxyType(results))
    case = dataclasses.replace(case, options=types.MappingProxyType({"option1": option}))

    built = option1.build_project(case, renderer)

    paragraphs = [p.text for p in built.document.paragraphs]
    start = paragraphs.index("Spaces that need a glare narrative")
    assert ROOM in paragraphs[start + 1]


def test_storey_section_has_a_spaces_table_scoped_to_the_storey_before_window_groups(case, renderer):
    # F9: spec 6.1 item 11 -- per storey, after the four plan figures, "a
    # spaces table and a window-group table". This checks both the project
    # report's per-storey subsection and the storey report itself.
    room = _room_result(case)

    project_built = option1.build_project(case, renderer)
    project_table = table_after(project_built.document, "Spaces", level=3)
    assert len(project_table.rows) == 3  # header + the one room on storey L0 + total
    row = [c.text for c in project_table.rows[1].cells]
    assert row[1] == ROOM
    assert row[4] == f"{room['sda_pct']:.1f} %"
    window_groups = table_after(project_built.document, "Window groups", level=3)
    assert [c.text for c in window_groups.rows[0].cells] == [
        "Group", "Space", "Type", "Tilt", "Facade azimuth", "Openings", "Blind closed hours",
    ]

    storey_built = option1.build_storey(case, "L0", renderer)
    storey_table = table_after(storey_built.document, "Spaces", level=3)
    assert len(storey_table.rows) == 3
    assert [c.text for c in storey_table.rows[1].cells][:2] == ["L0", ROOM]


@pytest.fixture
def roof_case(tmp_path, write_example, write_wea):
    return load_case(build_roof_glass_project(tmp_path, write_example, write_wea))


def _window_id(case):
    (group,) = [g for g in case.groups if g.kind == "window"]
    return group.group_id


def test_the_window_group_table_gives_type_tilt_and_no_azimuth_for_roof_glass(roof_case, renderer):
    built = option1.build_storey(roof_case, "L0", renderer)

    table = table_after(built.document, "Window groups", level=3)
    rows = {row.cells[0].text: [c.text for c in row.cells] for row in table.rows[1:]}
    assert set(rows) == {_window_id(roof_case), "Room_North_dif", "Room_North_sky"}
    window = rows[_window_id(roof_case)]
    assert window[1:4] == [ROOM, "window", "90 deg"]
    assert window[4].endswith(" deg") and window[5] == "1"
    assert rows["Room_North_sky"][1:] == [ROOM, "skylight", "0 deg", option1.NO_AZIMUTH, "1", "0"]
    assert rows["Room_North_dif"][1:] == [ROOM, "diffusing", "0 deg", option1.NO_AZIMUTH, "1",
                                          option1.NO_BLIND_DIFFUSING]
    assert option1.NO_AZIMUTH == "–"
    assert option1.NO_BLIND_DIFFUSING == "no blind (diffusing)"


def test_the_inputs_give_dirt_frame_and_well_per_class_in_use(roof_case):
    rows = dict(option1._input_rows(roof_case, roof_case.options["option1"]))

    assert rows["Dirt factor by glass tilt"] == "vertical 0.95, skylight 0.85"
    assert rows["Frame share by glass tilt"] == "skylight 10%; vertical glass by glazing role, as in the Frames row"
    assert rows["Skylight well"] == "built by the tool: straight vertical walls 0.4 m deep, reflectance 0.7"
    assert rows["Glazing diffuser_panel__skylight"].endswith(", diffusing")
    assert not rows["Glazing glass_roof__skylight"].endswith(", diffusing")


def test_a_vertical_only_project_has_no_class_rows(case):
    rows = dict(option1._input_rows(case, case.options["option1"]))

    assert not {"Dirt factor by glass tilt", "Frame share by glass tilt", "Skylight well"} & set(rows)


_TILT = "Glass tilt is measured from flat"
_DIRT = "when manufacturer's data are used"
_ROOF_WINDOW = "Clear roof glass is read as an exterior window"
_GROUPS = "this is a declared simplification"
_DIRECT = "Diffusing glass adds no direct sunlight"
_GUIDE = "Diffused and translucent glazing systems do not require glare-control devices"
_AMBIENT = "at least two (-ab 6)"
_BLACKOUT = "replaced by black"
_WELL = "straight vertical walls 0.4 m deep under each skylight, reflectance 0.7"
_ALL = (_TILT, _DIRT, _ROOF_WINDOW, _GROUPS, _DIRECT, _GUIDE, _AMBIENT, _BLACKOUT, _WELL)


def test_the_method_text_prints_the_readings_that_apply(roof_case, renderer):
    text = "\n".join(option1._method_lines(roof_case))

    for phrase in _ALL:
        assert phrase in text, phrase
    built = option1.build_project(roof_case, renderer)
    assert _GUIDE in doc_text(built.document)


def test_the_method_text_of_a_vertical_only_project_is_unchanged(case):
    lines = option1._method_lines(case)

    assert len(lines) == 4
    assert not any(phrase in "\n".join(lines) for phrase in _ALL)


def test_the_diffusing_sentences_need_diffusing_glass(roof_case):
    clear_only = dataclasses.replace(roof_case, groups=tuple(g for g in roof_case.groups if g.kind != "diffusing"))

    text = "\n".join(option1._method_lines(clear_only))

    for phrase in (_TILT, _ROOF_WINDOW, _GROUPS, _WELL):
        assert phrase in text, phrase
    for phrase in (_DIRECT, _GUIDE, _AMBIENT, _BLACKOUT):
        assert phrase not in text, phrase


def test_the_grid_plan_draws_roof_glass_as_dashed_outlines(roof_case, renderer, monkeypatch):
    specs = []
    real = option1.render_plan

    def spy(spec, painter):
        specs.append(spec)
        return real(spec, painter)

    monkeypatch.setattr(option1, "render_plan", spy)
    option1.build_storey(roof_case, "L0", renderer)

    (grid,) = [s for s in specs if s.title.endswith("sensor grid and window groups")]
    assert sorted(p.label for p in grid.polygons) == ["Room_North_dif", "Room_North_sky"]
    assert all(p.dashed for p in grid.polygons)
    assert [label for label, _ in grid.segments] == [_window_id(roof_case)]
    others = [s for s in specs if s is not grid]
    assert all(s.polygons == () for s in others)


def test_two_separate_skylights_are_two_dashed_polygons_labelled_once(roof_case, renderer, monkeypatch):
    second = ((4.0, 3.0), (5.0, 3.0), (5.0, 3.5), (4.0, 3.5))
    groups = tuple(
        dataclasses.replace(g, outlines=(*g.outlines, second), openings=g.openings + 1)
        if g.kind == "skylight" else g
        for g in roof_case.groups
    )
    case = dataclasses.replace(roof_case, groups=groups)
    specs = []
    real = option1.render_plan

    def spy(spec, painter):
        specs.append(spec)
        return real(spec, painter)

    monkeypatch.setattr(option1, "render_plan", spy)
    built = option1.build_storey(case, "L0", renderer)

    (grid,) = [s for s in specs if s.title.endswith("sensor grid and window groups")]
    assert len(grid.polygons) == 3 and all(p.dashed for p in grid.polygons)
    assert [p.label for p in grid.polygons if p.label != "Room_North_dif"] == ["Room_North_sky", ""]
    assert grid.polygons[[p.label for p in grid.polygons].index("")].points == second
    table = table_after(built.document, "Window groups", level=3)
    rows = {row.cells[0].text: [c.text for c in row.cells] for row in table.rows[1:]}
    assert rows["Room_North_sky"][5] == "2"
