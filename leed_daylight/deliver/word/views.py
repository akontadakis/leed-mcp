# leed_daylight/deliver/word/views.py
"""The Quality Views reports (reports spec 6.3, 6.4): one project report and
one report per storey, following the shared shape ``option1.py`` and
``option2.py`` already use (Tasks 7-8).

Any threshold, band or point count printed here comes from the credit rules
or from the results file, never from a literal in this module (Task 9
ruling; see ``word/common.py``'s docstring for the points-band case).
"""
from __future__ import annotations

from ..data import CaseData, OptionData, StoreyPlan
from ..figures.base import Renderer, render_bar, render_plan
from . import common, kit
from .common import Built, area, pct, score_text

OPTION = "views"
_LAYERS = (
    ("type2", "Type 2 view", "Points with a Type 2 view."),
    ("type3", "Type 3 view", "Points with a Type 3 view."),
    ("quality", "Quality View", "Points with a Quality View: both Type 2 and Type 3."),
)


def _option(case: CaseData) -> OptionData:
    return case.options[OPTION]


def _input_rows(case: CaseData, option: OptionData) -> list[tuple[str, str]]:
    s = option.results["settings"]
    return [
        ("Eye height", f"{float(s['eye_height_m']):g} m above the floor"),
        ("Source of the eye height", str(s["eye_height_source"])),
        ("Interior glass a view may cross", str(s["max_interior_glass"])),
        ("Ray spacing", f"{float(s['angular_step_deg']):g} deg, {int(s['rays_per_point']):,} rays per point"),
        ("Ground level", f"{float(s['ground_z_m']):g} m"),
        ("Moveable furniture", str(s["moveable_furniture"])),
        ("Sensor grid", f"{case.project.grid.spacing_m:g} m spacing, {option.points.xy.shape[0]:,} points"),
    ]


def _summary(built: Built, option: OptionData, heading: str) -> None:
    r = option.results
    built.document.add_heading(heading, level=1)
    kit.kv_table(built.document, [
        ("Quality View, floor-area weighted", pct(r["quality_area_pct"])),
        ("Quality View, point weighted", pct(r["quality_point_pct"])),
        ("Type 2 view, floor-area weighted", pct(r["type2_area_pct"])),
        ("Type 3 view, floor-area weighted", pct(r["type3_area_pct"])),
        ("Floor area m2", area(r["total_area_m2"])),
        ("Points", score_text(r["score"])),
    ])


def _spaces_table(built: Built, case: CaseData, option: OptionData, rooms=None) -> None:
    built.document.add_heading("Spaces", level=1)
    kit.table(
        built.document, ["Storey", "Space", "Area m2", "Points", "Type 2", "Type 3", "Quality View"],
        [(case.storey_of(s["room"]), s["room"], area(s["area_m2"]), str(s["points"]), pct(s["type2_pct"]),
          pct(s["type3_pct"]), pct(s["quality_pct"])) for s in common.spaces_of(option, rooms)],
        aligns=["left", "left", "right", "right", "right", "right", "right"],
    )


def _excluded(built: Built, option: OptionData) -> None:
    built.document.add_heading("Excluded spaces", level=1)
    excluded = list(option.results.get("excluded_spaces", []))
    if excluded:
        kit.table(built.document, ["Space", "Reason"],
                  [(e["room"], e.get("reason") or "no reason given") for e in excluded])
    else:
        built.document.add_paragraph("None: every regularly occupied space is in the analysis.")


def _by_storey(built: Built, case: CaseData, option: OptionData, renderer: Renderer, mark: str | None,
               chart: bool) -> None:
    built.document.add_heading("Results by storey", level=1)
    kit.table(built.document, ["Storey", "Spaces", "Area m2", "Quality View"],
              [(label + (" (this report)" if label == mark else ""), str(count), area(total), pct(value))
               for label, count, total, value in common.storey_averages(case, option, "quality_pct")],
              aligns=["left", "right", "right", "right"])
    if chart:
        threshold = case.rules.get("views", {}).get("threshold_pct")
        threshold = float(threshold) if isinstance(threshold, (int, float)) else None
        spec = common.storey_bar(case, option, title="Quality View, by storey", key="quality_pct",
                                 unit="%", threshold=threshold)
        common.add_figure(built, "storeys_quality.png", render_bar(spec, renderer),
                          "Share of floor area with a Quality View, by storey. The line marks the credit threshold.")


def _storey_section(built: Built, case: CaseData, option: OptionData, storey: StoreyPlan,
                    renderer: Renderer) -> None:
    built.document.add_heading(f"Storey {storey.label}", level=2)
    name = common.safe_name(storey.label)
    for key, title, text in _LAYERS:
        passed = option.points.columns[key] != 0
        spec = common.storey_plan(case, storey, option, title=f"Storey {storey.label}: {title}", passed=passed,
                                  legend=("view", "no view"))
        common.add_figure(built, f"{name}_{key}.png", render_plan(spec, renderer), text)


def _opening(built: Built, case: CaseData, option: OptionData) -> None:
    common.intent_and_requirements(built, case, OPTION)
    common.input_data(built, case, option, _input_rows(case, option))
    common.method(
        built,
        list(option.results.get("method_notes", []))
        + common.layout_notes(case, common.analysed(option), blinds=False),
    )


def build_project(case: CaseData, renderer: Renderer) -> Built:
    option = _option(case)
    built = common.new_report(case, OPTION)
    _opening(built, case, option)
    _summary(built, option, "Summary results")
    common.deviations_and_warnings(built, case, option)
    _spaces_table(built, case, option)
    _excluded(built, option)
    _by_storey(built, case, option, renderer, mark=None, chart=True)
    built.document.add_heading("Floor plans", level=1)
    for storey in case.storeys_for(OPTION):
        _storey_section(built, case, option, storey, renderer)
    return built


def build_storey(case: CaseData, label: str, renderer: Renderer) -> Built:
    option = _option(case)
    storey = case.storey(label)
    built = common.new_report(case, OPTION, f"storey {label}")
    common.scope_note(built, label)
    _opening(built, case, option)
    built.document.add_heading(f"Summary results, storey {label}", level=1)
    for row_label, count, total, value in common.storey_averages(case, option, "quality_pct"):
        if row_label == label:
            kit.kv_table(built.document, [("Spaces", str(count)), ("Floor area m2", area(total)),
                                          ("Quality View, floor-area weighted", pct(value))])
    common.deviations_and_warnings(built, case, option)
    _spaces_table(built, case, option, rooms=storey.rooms)
    built.document.add_heading("Floor plans", level=1)
    _storey_section(built, case, option, storey, renderer)
    built.document.add_heading("The project behind this storey", level=1)
    kit.kv_table(built.document, [("Project Quality View", pct(option.results["quality_area_pct"])),
                                  ("Points", score_text(option.results["score"]))])
    _by_storey(built, case, option, renderer, mark=label, chart=False)
    return built
