# leed_daylight/deliver/word/option2.py
"""The Option 2 reports (reports spec 6.2, 6.4): one project report and one
report per storey.

Pre-flight ruling F9: spec 6.2 asks for, near each equinox, "the chosen day
... the candidates with direct-normal irradiance and Perez clearness, and
any day-choice warning". ``_days`` adds a candidates table per equinox, read
from the results' own ``days[].top`` (the ranked candidates
``metrics.clearest_day.choose_clearest_day`` already keeps), with the
matching day-choice warning -- filtered from ``option.results["warnings"]``
by its fixed "clearest day near <equinox>:" prefix, since a day's own
warning is folded into the case-level warnings list, not kept per day in the
results file -- printed underneath it.
"""
from __future__ import annotations

from ..data import CaseData, OptionData, StoreyPlan
from ..figures.base import Renderer, render_bar, render_plan
from ..figures.spec import ColourScale
from . import common, kit
from .common import Built, area, pct, score_text

OPTION = "option2"


def _option(case: CaseData) -> OptionData:
    return case.options[OPTION]


def _settings(option: OptionData):
    return option.results["settings"]


def _input_rows(case: CaseData, option: OptionData) -> list[tuple[str, str]]:
    s = _settings(option)
    return [
        ("Days", f"the clearest day within {s['clearest_day_window_days']} days of each of "
                 f"{', '.join(s['equinox_dates'])}, from the weather file"),
        ("Times", f"{', '.join(s['times'])}, as the weather file stamps them ({s['time_basis']})"),
        ("Sky", f"{s['sky']}: built from the weather file's own irradiance at each time"),
        ("Illuminance band", f"{float(s['lux_min']):g} to {float(s['lux_max']):g} lux"),
        ("Blinds", str(s["blinds"])),
        ("Sensor grid", f"{case.project.grid.spacing_m:g} m spacing, work plane "
                        f"{case.project.grid.work_plane_m:g} m, {option.points.xy.shape[0]:,} points"),
        ("Radiance parameters", f"{s['radiance_ambient']} {s['radiance_direct']}"),
    ]


def _method_lines(option: OptionData) -> list[str]:
    s = _settings(option)
    return [
        "The day with the most direct normal irradiance near each equinox is chosen from the weather file; "
        "the Perez clearness index is computed beside it as a check.",
        "Illuminance is traced at every sensor point for each day and time, with blinds excluded.",
        f"The two days are averaged per time. A point passes when both averages lie between "
        f"{float(s['lux_min']):g} and {float(s['lux_max']):g} lux; in a space with view-preserving "
        "automatic shades only the lower limit applies.",
        "Space values are shares of points; the project value is weighted by floor area.",
    ]


def _epsilon_text(value) -> str:
    """A candidate's Perez clearness, or blank when it has no qualifying
    hour (``metrics.clearest_day.DayCandidate.epsilon`` is ``None`` then)."""
    return "" if value is None else f"{float(value):.2f}"


def _days(built: Built, option: OptionData) -> None:
    doc = built.document
    doc.add_heading("The chosen days", level=1)
    days = option.results["days"]
    rows = []
    for day in days:
        chosen, best = day["chosen"], day.get("perez_best")
        rows.append((
            day["equinox"], chosen["date"], f"{float(chosen['direct_normal_wh_m2']):,.0f}",
            _epsilon_text(chosen["perez_epsilon"]), best["date"] if best else "",
        ))
    kit.table(doc, ["Equinox", "Chosen day", "Direct normal Wh/m2", "Perez clearness", "Clearest by Perez"],
              rows, aligns=["left", "left", "right", "right", "left"])
    # Pre-flight ruling F9: the candidates behind each choice, and the
    # day-choice warning for that equinox (if any), right under its table.
    warnings = list(option.results.get("warnings", []))
    for day in days:
        equinox = day["equinox"]
        doc.add_heading(f"Candidates near {equinox}", level=3)
        kit.table(
            doc, ["Date", "Offset days", "Direct normal Wh/m2", "Perez clearness"],
            [(c["date"], str(c["offset_days"]), f"{float(c['direct_normal_wh_m2']):,.0f}",
              _epsilon_text(c["perez_epsilon"]))
             for c in day["top"]],
            aligns=["left", "right", "right", "right"],
        )
        matching = [w for w in warnings if w.startswith(f"clearest day near {equinox}:")]
        if matching:
            kit.bullets(doc, matching)
    doc.add_heading("The skies", level=1)
    kit.table(doc, ["Sky", "Direct normal W/m2", "Diffuse horizontal W/m2", "Sun altitude", "Sun azimuth"],
              [(m["label"], f"{float(m['direct_normal_w_m2']):g}", f"{float(m['diffuse_horizontal_w_m2']):g}",
                f"{float(m['sun_altitude_deg']):.1f} deg", f"{float(m['sun_azimuth_deg']):.1f} deg")
               for m in option.results["skies"]],
              aligns=["left", "right", "right", "right", "right"])


def _summary(built: Built, option: OptionData, heading: str) -> None:
    r = option.results
    built.document.add_heading(heading, level=1)
    kit.kv_table(built.document, [
        ("Floor area in the band", pct(r["area_pct"])),
        ("Sensor points in the band", pct(r["point_pct"])),
        ("Points", score_text(r["score"])),
    ])


def _spaces_table(built: Built, case: CaseData, option: OptionData, rooms=None) -> None:
    spaces = common.spaces_of(option, rooms)
    built.document.add_heading("Spaces", level=1)
    kit.table(
        built.document, ["Storey", "Space", "Area m2", "Points", "In band", "Too dark", "Too bright", "Upper limit"],
        [(case.storey_of(s["room"]), s["room"], area(s["area_m2"]), str(s["points"]), pct(s["pass_pct"]),
          str(s["n_low"]), str(s["n_high"]), "applied" if s["upper_limit_applied"] else "not applied")
         for s in spaces],
        aligns=["left", "left", "right", "right", "right", "right", "right", "left"],
    )


def _by_storey(built: Built, case: CaseData, option: OptionData, renderer: Renderer, mark: str | None,
               chart: bool) -> None:
    built.document.add_heading("Results by storey", level=1)
    kit.table(built.document, ["Storey", "Spaces", "Area m2", "In band"],
              [(label + (" (this report)" if label == mark else ""), str(count), area(total), pct(value))
               for label, count, total, value in common.storey_averages(case, option, "pass_pct")],
              aligns=["left", "right", "right", "right"])
    if chart:
        bands = case.rules["option2"]["bands"]
        threshold = float(bands[0]["min_pct"]) if isinstance(bands, list) and bands else None
        spec = common.storey_bar(case, option, title="Floor area in the band, by storey", key="pass_pct",
                                 unit="%", threshold=threshold)
        common.add_figure(built, "storeys_pass.png", render_bar(spec, renderer),
                          "Share of floor area in the band, by storey. The line marks the lowest point band.")


def _storey_section(built: Built, case: CaseData, option: OptionData, storey: StoreyPlan,
                    renderer: Renderer) -> None:
    built.document.add_heading(f"Storey {storey.label}", level=2)
    s = _settings(option)
    lux_min, lux_max = float(s["lux_min"]), float(s["lux_max"])
    scale = ColourScale(max(1.0, lux_min / 10.0), lux_max * 3.0, "lux (log scale)", marks=(lux_min, lux_max),
                        log=True)
    columns = option.points.columns
    name = common.safe_name(storey.label)
    for sky in option.results["skies"]:
        label = sky["label"]
        spec = common.storey_plan(case, storey, option, title=f"Storey {storey.label}: illuminance, {label}",
                                  values=columns[f"lux_{label}"], scale=scale)
        common.add_figure(built, f"{name}_sky_{label}.png", render_plan(spec, renderer),
                          f"Illuminance at {label}. The marks on the legend are the band limits.")
    for time in s["times"]:
        spec = common.storey_plan(case, storey, option, title=f"Storey {storey.label}: two-day average, {time}",
                                  values=columns[f"average_{time}"], scale=scale)
        common.add_figure(built, f"{name}_average_{common.safe_name(time)}.png", render_plan(spec, renderer),
                          f"Average of the two days at {time}.")
    spec = common.storey_plan(case, storey, option, title=f"Storey {storey.label}: points in the band",
                              passed=columns["passes"], legend=("in the band", "outside the band"))
    common.add_figure(built, f"{name}_pass.png", render_plan(spec, renderer),
                      "Points whose averages at every time lie in the band.")


def _opening(built: Built, case: CaseData, option: OptionData) -> None:
    common.intent_and_requirements(built, case, OPTION)
    common.input_data(built, case, option, _input_rows(case, option))
    common.method(built, _method_lines(option) + common.layout_notes(case, common.analysed(option), blinds=False))


def build_project(case: CaseData, renderer: Renderer) -> Built:
    option = _option(case)
    built = common.new_report(case, OPTION)
    _opening(built, case, option)
    _days(built, option)
    _summary(built, option, "Summary results")
    common.deviations_and_warnings(built, case, option)
    _spaces_table(built, case, option)
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
    for row_label, count, total, value in common.storey_averages(case, option, "pass_pct"):
        if row_label == label:
            kit.kv_table(built.document, [("Spaces", str(count)), ("Floor area m2", area(total)),
                                          ("Floor area in the band", pct(value))])
    common.deviations_and_warnings(built, case, option)
    _spaces_table(built, case, option, rooms=storey.rooms)
    built.document.add_heading("Floor plans", level=1)
    _storey_section(built, case, option, storey, renderer)
    built.document.add_heading("The project behind this storey", level=1)
    kit.kv_table(built.document, [("Project floor area in the band", pct(option.results["area_pct"])),
                                  ("Points", score_text(option.results["score"]))])
    _by_storey(built, case, option, renderer, mark=label, chart=False)
    return built
