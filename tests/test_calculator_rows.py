"""The case's results as calculator rows (calculator spec 5.1, D3, D4)."""
import pytest

from calculator_standin import add_vlt as _add_vlt
from deliver_folder import CASE, build_deliver_project
from leed_daylight.deliver.calculator.rows import M2_PER_FT2, calculator_rows
from leed_daylight.deliver.data import load_case


@pytest.fixture
def case(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea)
    _add_vlt(folder)
    return load_case(folder, CASE)


def test_one_daylight_row_per_regularly_occupied_space(case):
    rows = calculator_rows(case, "SI")
    (row,) = rows.daylight
    assert (row.id, row.description, row.area, row.shades) == ("Room North", "office", 24.0, "No")
    space = case.options["option1"].results["spaces"][0]
    assert row.sda == round(space["sda_pct"], 1) and row.ase == round(space["ase_pct"], 1)
    assert row.in_band + row.above <= row.area


def test_views_rows_split_by_the_points(case):
    rows = calculator_rows(case, "SI")
    by_id = {r.id: r for r in rows.views}
    # deliver_folder: west half quality (median 12 m), east half no line of sight.
    assert set(by_id) == {"Room North (view, distance)", "Room North (no line of sight)"}
    assert by_id["Room North (view, distance)"].sight == "Yes"
    assert by_id["Room North (view, distance)"].content.startswith("objects at least 25 feet")
    assert by_id["Room North (no line of sight)"].sight == "No"
    assert by_id["Room North (no line of sight)"].content == "other (view does not qualify)"
    assert {r.vlt for r in rows.views} == {62.0}
    assert sum(r.area for r in rows.views) == pytest.approx(24.0, abs=0.11)
    tool = case.options["views"].results["quality_area_pct"]
    assert rows.expected["views_pct"] == pytest.approx(tool, abs=0.5)


def test_ip_areas_are_square_feet(case):
    si, ip = calculator_rows(case, "SI"), calculator_rows(case, "IP")
    assert ip.daylight[0].area == round(24.0 / M2_PER_FT2, 1)
    assert ip.expected["total_area"] == pytest.approx(si.expected["total_area"] / M2_PER_FT2, rel=1e-3)
    assert ip.daylight[0].sda == si.daylight[0].sda


def test_a_missing_credit_leaves_its_columns_empty_and_says_so(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=("views",))
    _add_vlt(folder)
    rows = calculator_rows(load_case(folder, CASE), "SI")
    (row,) = rows.daylight
    assert row.sda is None and row.in_band is None and row.area == 24.0
    assert any("option1" in w for w in rows.warnings) and any("option2" in w for w in rows.warnings)


def test_a_vlt_of_exactly_40_is_warned(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea)
    _add_vlt(folder, "40")
    rows = calculator_rows(load_case(folder, CASE), "SI")
    assert any("exactly 40" in w for w in rows.warnings)


def test_the_atrium_cap_is_always_named(case):
    assert any("atri" in w for w in calculator_rows(case, "SI").warnings)


def test_a_name_with_markup_characters_is_kept_as_text(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea)
    _add_vlt(folder)
    for name in ("spaces.csv",):
        path = folder / name
        path.write_text(path.read_text(encoding="utf-8").replace("Room North", "R&D <North>"), encoding="utf-8", newline="\n")
    # The rename makes every result stale; rows then come from spaces.csv only.
    rows = calculator_rows(load_case(folder, CASE, require_results=False), "SI")
    assert rows.daylight[0].id == "R&D <North>"


def test_average_sda_divides_by_every_row_like_the_calculator(case):
    import dataclasses
    import types

    base = case.spaces[0]
    extra = dataclasses.replace(base, room="Room Extra", area_m2=12.0)
    case2 = dataclasses.replace(case, spaces=(*case.spaces, extra))
    rows = calculator_rows(case2, "SI")
    with_sda = [r for r in rows.daylight if r.sda is not None]
    assert len(rows.daylight) == 2 and len(with_sda) == 1
    total = sum(r.area for r in rows.daylight)
    # H16 = SUMPRODUCT(F, D) / SUM(D): the row without an sDA still counts in the divisor.
    assert rows.expected["average_sda"] == round(with_sda[0].sda * with_sda[0].area / total, 1)


def test_a_space_missing_from_option1_results_is_warned(case):
    import dataclasses
    import types

    opt = case.options["option1"]
    results = dict(opt.results, spaces=[])
    case2 = dataclasses.replace(case, options=types.MappingProxyType(
        {**case.options, "option1": dataclasses.replace(opt, results=results)}))
    rows = calculator_rows(case2, "SI")
    assert rows.daylight[0].sda is None and rows.expected["average_sda"] is None
    assert any("Room North: not in the Option 1 results" in w for w in rows.warnings)


def test_a_chained_merge_adds_every_piece_to_the_area(case):
    import dataclasses

    from leed_daylight.deliver.calculator.rows import _space_area

    base = case.spaces[0]
    b = dataclasses.replace(base, room="B", merge_into=base.room, area_m2=5.0)
    c = dataclasses.replace(base, room="C", merge_into="B", area_m2=2.0)
    assert _space_area((base, b, c), base.room) == base.area_m2 + 7.0


def test_units_must_be_ip_or_si(case):
    with pytest.raises(ValueError):
        calculator_rows(case, "metric")


def _with_points(case, option, **columns):
    """``case`` with ``option``'s per-point columns replaced (one value per point)."""
    import dataclasses
    import types

    import numpy as np

    opt = case.options[option]
    n = len(opt.points.rooms)
    cols = dict(opt.points.columns)
    for key, values in columns.items():
        cols[key] = np.resize(np.asarray(values), n)
    points = dataclasses.replace(opt.points, columns=types.MappingProxyType(cols))
    return dataclasses.replace(case, options=types.MappingProxyType(
        {**case.options, option: dataclasses.replace(opt, points=points)}))


def test_a_point_above_then_below_the_band_is_in_neither_k_nor_l(case):
    import dataclasses

    times = [k for k in case.options["option2"].points.columns if k.startswith("average_")]
    assert len(times) >= 2
    values = {k: [5000.0] if i == 0 else [200.0] for i, k in enumerate(times)}
    shaded = dataclasses.replace(case.spaces[0], view_preserving_shades=True)
    case2 = dataclasses.replace(_with_points(case, "option2", **values), spaces=(shaded, *case.spaces[1:]))
    (row,) = calculator_rows(case2, "SI").daylight
    assert row.in_band == 0.0 and row.above == 0.0
    # Above 3,000 lux once and at least 300 lux every time: L, and it counts with shades.
    values = {k: [5000.0] if i == 0 else [400.0] for i, k in enumerate(times)}
    case3 = dataclasses.replace(_with_points(case, "option2", **values), spaces=(shaded, *case.spaces[1:]))
    rows = calculator_rows(case3, "SI")
    assert rows.daylight[0].above == 24.0 and rows.expected["daylit_area"] == 24.0


def test_views_rows_for_content_and_for_a_view_that_does_not_qualify(case):
    # Half the points: a quality view nearer than the object distance; the other half: line of sight, no quality.
    case2 = _with_points(case, "views", quality=[True, False], type3=[True, True], median_outside_m=[2.0, 2.0])
    rows = calculator_rows(case2, "SI")
    by_id = {r.id: r for r in rows.views}
    assert set(by_id) == {"Room North (view, content)", "Room North (view does not qualify)"}
    content, other = by_id["Room North (view, content)"], by_id["Room North (view does not qualify)"]
    assert (content.sight, content.content) == ("Yes", "nature")
    assert (other.sight, other.content) == ("Yes", "other (view does not qualify)")
    assert rows.expected["views_area"] == content.area


def test_daylight_area_is_the_scene_room_area_like_the_views_rows(case):
    # spaces.csv says 24 m2; the results (the scene's room) say otherwise.
    results = case.options["option1"].results
    results["spaces"][0]["area_m2"] = 21.0
    case.options["option2"].results["spaces"][0]["area_m2"] = 21.0

    rows = calculator_rows(case, "SI")

    assert rows.daylight[0].area == 21.0
    assert sum(r.area for r in rows.views) == pytest.approx(24.0, abs=0.11)  # views from their own results


def test_daylight_area_falls_back_to_spaces_csv_without_an_option_row(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=("views",))
    _add_vlt(folder)

    assert calculator_rows(load_case(folder, CASE), "SI").daylight[0].area == 24.0


@pytest.mark.parametrize("raw, written", [("39.96", 39.9), ("39.99", 39.9), ("40", 39.9), ("40.04", 40.1),
                                          ("40.1", 40.1), ("62", 62.0)])
def test_the_written_vlt_keeps_the_tool_s_pass_or_fail(tmp_path, write_example, write_wea, raw, written):
    folder = build_deliver_project(tmp_path, write_example, write_wea)
    _add_vlt(folder, raw)

    rows = calculator_rows(load_case(folder, CASE), "SI")

    assert {r.vlt for r in rows.views} == {written}


def test_a_vlt_the_tool_fails_is_not_warned_as_exactly_40_when_written_below(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea)
    _add_vlt(folder, "39.96")

    assert not any("exactly 40" in w for w in calculator_rows(load_case(folder, CASE), "SI").warnings)


def test_a_vlt_written_as_a_fraction_is_warned(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea)
    _add_vlt(folder, "0.62")

    assert any("percentage" in w for w in calculator_rows(load_case(folder, CASE), "SI").warnings)


@pytest.mark.parametrize("option", ["option1", "option2"])
def test_an_option_not_on_the_leed_basis_is_warned_with_its_reason(case, option):
    import dataclasses
    import types

    opt = case.options[option]
    score = {"points": None, "band": None, "reason": "not scored: the basis differs"}
    case2 = dataclasses.replace(case, options=types.MappingProxyType(
        {**case.options, option: dataclasses.replace(opt, results=dict(opt.results, score=score))}))
    rows = calculator_rows(case2, "SI")
    assert any(w.startswith(f"{option}: no points are awarded") and "not scored: the basis differs" in w
               for w in rows.warnings)
    assert not any("no points are awarded" in w for w in calculator_rows(case, "SI").warnings)
