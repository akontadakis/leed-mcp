# leed_daylight/deliver/word/option1.py
"""The Option 1 reports (reports spec 6.1, 6.4): one project report and one
report per storey.

Pre-flight ruling F8: the two per-space sDA limits (spec 6.1 item 8, "spaces
at sDA 55 % and 75 %") are not literal numbers in this module. They come
from ``case.rules["option1"]["space_rule_sda_pct"]`` (validated by
``project.rules``), so a credit rules file without that key still produces
a report -- ``_space_rule`` is simply left out and a warning recorded on
``Built`` explains why. Spec 6.4 says a storey report carries "the same
sections, cut to one storey", so ``build_storey`` runs the same rule too,
scoped to that storey's rooms.

Pre-flight ruling F9: spec 6.1 item 11 asks for, per storey, "a spaces
table and a window-group table" alongside the four plan figures.
``_storey_section`` -- shared by the project report's per-storey
subsections and the storey report -- adds a spaces table scoped to that
storey's rooms immediately before the window-group table.
"""
from __future__ import annotations

import shlex
from collections.abc import Sequence

from ...build.glazing_class import TILT_CLASSES
from ..data import CaseData, OptionData, StoreyPlan
from ..figures.base import Renderer, render_bar, render_plan
from ..figures.spec import ColourScale, PlanPolygon
from . import common, kit
from .common import Built, area, pct, score_text

OPTION = "option1"

#: The warning added to the Warnings section when the credit rules file has
#: no per-space sDA limits for Option 1 (pre-flight ruling F8).
NO_SPACE_RULE_WARNING = "the credit rules file has no option1.space_rule_sda_pct; the sDA-by-space rule is omitted"

#: The window-group table's azimuth cell for a skylight or diffusing group,
#: which has none (skylights spec 7), and its blind cell for diffusing glass.
NO_AZIMUTH = "–"
NO_BLIND_DIFFUSING = "no blind (diffusing)"
#: The window-group table's Type cell per group kind (skylights spec 5.1).
KIND_LABELS = {
    "window": "window", "sloped": "sloped", "monitor": "roof monitor",
    "skylight": "skylight", "diffusing": "diffusing",
}


def _classes_in_use(case: CaseData) -> tuple[str, ...]:
    """The tilt classes of every exterior pane, in ``TILT_CLASSES`` order."""
    present = {c for g in case.groups for c in g.tilt_classes}
    return tuple(c for c in TILT_CLASSES if c in present)


def _last_ab(options: str) -> int | None:
    """The last ``-ab`` value on a Radiance command line (the last wins)."""
    tokens = shlex.split(options)
    value = None
    for flag, arg in zip(tokens, tokens[1:]):
        if flag == "-ab":
            try:
                value = int(arg)
            except ValueError:
                value = None
    return value


def _option(case: CaseData) -> OptionData:
    return case.options[OPTION]


def _input_rows(case: CaseData, option: OptionData) -> list[tuple[str, str]]:
    p, r = case.project, option.results
    rows = [
        ("Analysis hours", f"{p.analysis.hours}, daylight saving {p.analysis.daylight_saving}, "
                           f"{r['occupied_hours']:,} occupied hours"),
        ("Sensor grid", f"{p.grid.spacing_m:g} m spacing, work plane {p.grid.work_plane_m:g} m, "
                        f"{p.grid.min_clearance_m:g} m clear of walls, {option.points.xy.shape[0]:,} points"),
    ]
    for modifier, record in sorted(case.glazing.items()):
        rows.append((f"Glazing {modifier}",
                     f"visible transmittance {float(record['visible_transmittance']):.2f}, dirt factor "
                     f"{float(record['dirt_factor']):.2f}, net {float(record['net_transmittance']):.3f}"
                     + (", diffusing" if record.get("transmission") == "diffuse" else "")))
    g, b = p.glazing, p.blinds
    rows += [
        ("Frames", ", ".join(f"{k} {float(v):.0%}" for k, v in g.frame_share.items())
                   + f" of each opening; reflectance {g.frame_reflectance:g}"),
        ("Reveals", f"{g.reveal_depth_m:g} m deep, reflectance {g.reveal_reflectance:g}"),
    ]
    classes = _classes_in_use(case)
    tilted = [c for c in classes if c != "vertical"]
    if tilted:
        if g.vt_includes_dirt:
            dirt = "included in each glass's visible transmittance (vt_includes_dirt)"
        else:
            dirt = ", ".join(f"{c} {g.dirt_for(c):.2f}" for c in classes)
        rows.append(("Dirt factor by glass tilt", dirt))
        share = ", ".join(f"{c} {float(g.frame_share[c]):.0%}" for c in tilted)
        if "vertical" in classes:
            share += "; vertical glass by glazing role, as in the Frames row"
        rows.append(("Frame share by glass tilt", share))
    if "skylight" in classes:
        well = ("modelled in the scene" if g.skylight_well == "modelled" else
                f"built by the tool: straight vertical walls {float(g.skylight_well):g} m deep, "
                f"reflectance {float(g.skylight_well_reflectance):g}")
        rows.append(("Skylight well", well))
    rows += [
        ("Blinds", f"diffuse transmittance {b.transmittance_diffuse:g}, specular {b.transmittance_specular:g}, "
                   f"room-side reflectance {b.room_side_reflectance:g}"),
        ("Reflectances", ", ".join(f"{k} {float(v):g}" for k, v in p.reflectance.by_role.items())
                         + f" (policy {p.reflectance.policy})"),
        ("Sky", f"{p.sky.basis} MF:{p.sky.MF}"),
        ("Radiance parameters", f"{p.radiance.ambient_full} {p.radiance.direct}"),
    ]
    return rows


def _glass_method_lines(case: CaseData) -> list[str]:
    """The skylights spec's section 2 readings that apply to this scene
    (each printed only when its glass is present) and the 5.9 limit."""
    p = case.project
    kinds = {g.kind for g in case.groups}
    classes = _classes_in_use(case)
    lines: list[str] = []
    if {"sloped", "skylight"} & set(classes):
        lines.append(
            "Glass tilt is measured from flat (0 degrees flat, 90 degrees vertical): vertical 85 to 90 degrees, "
            "sloped 20 to under 85, skylight under 20. LM-83-12 2.2.9 gives the sloped band as \"85 to 20 "
            "degrees of vertical\" and skylights as \"less than a 20 degree slope\"; reading both from flat "
            "is the only reading with no gap and no overlap."
        )
        if not p.glazing.vt_includes_dirt:
            lines.append(
                "LM-83-12 2.2.9 states its dirt minimums \"when manufacturer's data are used\"; they are applied "
                "here because every glass visible transmittance entered is a product value."
            )
    if any(g.outlines and g.kind != "diffusing" for g in case.groups):
        lines.append(
            "Clear roof glass is read as an exterior window under LM-83-12 2.2.6 and has its own operable blind."
        )
    if kinds & {"skylight", "sloped"}:
        lines.append(
            "Window groups: one skylight group per space joins all its clear panes under 20 degrees of tilt, "
            "whatever their azimuth, and a sloped group joins the space's panes of 20 to under 85 degrees in one "
            "22.5-degree azimuth bin. Both are wider than LM-83-12 2.2.6's \"coplanar ... similar shadow "
            "patterns\", and every group's 22.5-degree bin is wider than its 15 degrees; this is a declared "
            "simplification."
        )
    if "diffusing" in kinds:
        ab = _last_ab(f"{p.radiance.ambient_full} {p.radiance.direct}")
        lines += [
            "Diffusing glass adds no direct sunlight: LM-83-12 2.2.6 defines direct sunlight as direct beam "
            "sunlight from a zero-bounce calculation, and light scattered by a diffuser is not a beam; the "
            "zero-bounce trace through it gives zero, so it adds nothing to ASE or to the blind schedule.",
            "No blind is modelled on diffusing glass. The LEED v4 BD+C Reference Guide, EQ Credit Daylight, says "
            "\"Diffused and translucent glazing systems do not require glare-control devices\"; the credit scored "
            "is v4.1, whose credit text says nothing on this, and the v4 guidance is read as still applying "
            "because v4.1 keeps the same device requirement.",
            "Light through diffusing glass reaches the sensors only through ambient bounces, so the simulation "
            f"uses at least two (-ab {ab}); it is sampled by ambient rays and carries more noise than light "
            "through clear glass.",
        ]
        rooms = {g.room for g in case.groups if g.kind == "diffusing"}
        if any(g.room in rooms and g.kind != "diffusing" for g in case.groups):
            lines.append(
                "Each window group is traced with every other group's glass replaced by black, so light one "
                "group lets in that another group's inner glass face would reflect is lost. Diffusing glass "
                "reflects more than clear glass, so sDA may be slightly low in spaces with both clear glass "
                "and diffusing glass."
            )
    if "skylight" in classes:
        well = p.glazing.skylight_well
        if well == "modelled":
            lines.append("Skylight wells are modelled in the scene, as LM-83-12 2.2.9 asks.")
        elif well is not None:
            lines.append(
                f"Skylight wells are built by the tool as straight vertical walls {float(well):g} m deep under "
                f"each skylight, reflectance {float(p.glazing.skylight_well_reflectance):g}; a splayed or "
                "differently shaped real well is not represented (a declared simplification)."
            )
    return lines


def _method_lines(case: CaseData) -> list[str]:
    rules = case.rules["option1"]
    return [
        f"Annual simulation to {case.project.standard} with Radiance: daylight coefficients per window "
        "group, one sun per occupied hour for direct sunlight.",
        f"sDA uses the hourly blind schedule of {case.project.standard}; ASE uses blinds open.",
        f"A point passes sDA when it reaches {rules['sda']['lux']} lux for at least "
        f"{100.0 * float(rules['sda']['fraction']):.0f} % of occupied hours; it counts for ASE when it "
        f"receives {rules['ase']['lux']} lux of direct sunlight for more than {rules['ase']['hours']} hours.",
        "Space values are shares of points; project values are weighted by floor area and by points.",
        *_glass_method_lines(case),
    ]


def _summary(built: Built, option: OptionData, heading: str) -> None:
    r = option.results
    headline = r["headline"]
    built.document.add_heading(heading, level=1)
    kit.kv_table(built.document, [
        ("sDA, floor-area weighted", pct(r["sda_area_pct"]) + (" (headline)" if headline == "floor_area" else "")),
        ("sDA, point weighted", pct(r["sda_point_pct"]) + (" (headline)" if headline == "points" else "")),
        ("ASE, floor-area weighted", pct(r["ase_area_pct"])),
        ("ASE, point weighted", pct(r["ase_point_pct"])),
        ("Occupied hours", f"{r['occupied_hours']:,}"),
        ("Points", score_text(r["score"])),
        ("Additional point (every space at the minimum)",
         "awarded" if r["score"].get("additional_point_awarded") else "not awarded"),
    ])


def _spaces_table(built: Built, case: CaseData, option: OptionData, rooms=None, *, level: int = 1) -> None:
    spaces = common.spaces_of(option, rooms)
    built.document.add_heading("Spaces", level=level)
    rows = [
        (case.storey_of(s["room"]), s["room"], area(s["area_m2"]), str(s["points"]), pct(s["sda_pct"]),
         pct(s["ase_pct"]), s["ase_band"], pct(s.get("residual_sky_pct", 0.0)))
        for s in spaces
    ]
    total_area = sum(float(s["area_m2"]) for s in spaces)
    total = ("Total", "", area(total_area), str(sum(int(s["points"]) for s in spaces)), "", "", "", "")
    kit.table(built.document, ["Storey", "Space", "Area m2", "Points", "sDA", "ASE", "ASE band", "Residual sky"],
              rows, aligns=["left", "left", "right", "right", "right", "right", "left", "right"], total=total)


def _space_rule_limits(built: Built, case: CaseData) -> Sequence[float] | None:
    """The per-space sDA limits (pre-flight ruling F8), or ``None`` -- with a
    Warnings-section note appended to ``built.warnings`` -- when the credit
    rules file carries none. Shared by ``build_project`` and
    ``build_storey``: spec 6.4 says a storey report carries "the same
    sections, cut to one storey", which includes spec 6.1 item 8's space
    rule, computed over just that storey's rooms and area."""
    limits = case.rules["option1"].get("space_rule_sda_pct")
    if limits is None:
        built.warnings.append(NO_SPACE_RULE_WARNING)
    return limits


def _space_rule(built: Built, case: CaseData, option: OptionData, limits: Sequence[float], rooms=None) -> None:
    built.document.add_heading("sDA by space", level=1)
    spaces = common.spaces_of(option, rooms)
    total = sum(float(s["area_m2"]) for s in spaces)
    rows = []
    for limit in limits:
        meeting = [s for s in spaces if float(s["sda_pct"]) >= float(limit)]
        share = 100.0 * sum(float(s["area_m2"]) for s in meeting) / total if total else 0.0
        rows.append((f"sDA {float(limit):g} % or more", f"{len(meeting)} of {len(spaces)}", pct(share)))
    kit.table(built.document, ["Rule", "Spaces", "Share of floor area"], rows, aligns=["left", "right", "right"])


def _by_storey(built: Built, case: CaseData, option: OptionData, renderer: Renderer, mark: str | None,
               chart: bool) -> None:
    built.document.add_heading("Results by storey", level=1)
    sda = {row[0]: row for row in common.storey_averages(case, option, "sda_pct")}
    ase = {row[0]: row for row in common.storey_averages(case, option, "ase_pct")}
    rows = [
        (label + (" (this report)" if label == mark else ""), str(count), area(total), pct(value),
         pct(ase[label][3]))
        for label, (_, count, total, value) in sda.items()
    ]
    kit.table(built.document, ["Storey", "Spaces", "Area m2", "sDA", "ASE"], rows,
              aligns=["left", "right", "right", "right", "right"])
    if chart:
        bands = case.rules["option1"]["sda"]["bands"]
        threshold = float(bands[0]["min_pct"]) if isinstance(bands, list) and bands else None
        spec = common.storey_bar(case, option, title="sDA by storey, floor-area weighted", key="sda_pct",
                                 unit="%", threshold=threshold)
        common.add_figure(built, "storeys_sda.png", render_bar(spec, renderer),
                          "sDA by storey. The line marks the lowest point band.")


def _worst_ase(built: Built, case: CaseData, option: OptionData, limit: int = 15) -> None:
    built.document.add_heading("Spaces with the highest ASE", level=1)
    spaces = sorted(common.spaces_of(option), key=lambda s: -float(s["ase_pct"]))[:limit]
    kit.table(built.document, ["Storey", "Space", "Area m2", "ASE", "sDA"],
              [(case.storey_of(s["room"]), s["room"], area(s["area_m2"]), pct(s["ase_pct"]), pct(s["sda_pct"]))
               for s in spaces], aligns=["left", "left", "right", "right", "right"])


def _narrative(built: Built, case: CaseData, option: OptionData) -> None:
    built.document.add_heading("Spaces that need a glare narrative", level=1)
    names = list(option.results.get("narrative_spaces", []))
    trigger = float(case.rules["option1"]["ase"]["per_space_narrative_trigger_pct"])
    if names:
        built.document.add_paragraph(f"ASE over {trigger:g} %: " + ", ".join(names) + ". See glare_narrative.md.")
    else:
        built.document.add_paragraph(f"None: no space has ASE over {trigger:g} %.")


def _storey_section(built: Built, case: CaseData, option: OptionData, storey: StoreyPlan,
                    renderer: Renderer) -> None:
    doc = built.document
    doc.add_heading(f"Storey {storey.label}", level=2)
    rules = case.rules["option1"]
    columns = option.points.columns
    name = common.safe_name(storey.label)
    here = [g for g in case.groups if g.storey_label == storey.label]
    segments = [(g.group_id, seg) for g in here for seg in g.segments]
    # One dashed outline per roof pane; only a group's first carries its id.
    polygons = [PlanPolygon(points=outline, dashed=True, label=g.group_id if i == 0 else "")
                for g in here for i, outline in enumerate(g.outlines)]
    if polygons:
        grid_caption = ("Sensor points and the window groups: facade glass as solid lines, roof glass as "
                        "dashed outlines. Each group has its own operable blind"
                        + (", except diffusing glass, which needs none." if any(g.kind == "diffusing" for g in here)
                           else "."))
    else:
        grid_caption = "Sensor points and the window groups; each group has its own operable blind."
    hours = float(rules["ase"]["hours"])
    plans = [
        ("zones", common.storey_plan(case, storey, option, title=f"Storey {storey.label}: spaces", points=False),
         "Regularly occupied spaces in the analysis."),
        ("grid", common.storey_plan(case, storey, option, title=f"Storey {storey.label}: sensor grid and window groups",
                                    segments=segments, polygons=polygons),
         grid_caption),
        ("sda", common.storey_plan(
            case, storey, option, title=f"Storey {storey.label}: daylit share of occupied hours",
            values=columns["da_pct"],
            scale=ColourScale(0.0, 100.0, f"% of occupied hours at {rules['sda']['lux']} lux or more",
                              marks=(100.0 * float(rules["sda"]["fraction"]),))),
         "sDA. The mark on the legend is the pass level."),
        ("ase", common.storey_plan(
            case, storey, option, title=f"Storey {storey.label}: hours of direct sunlight",
            values=columns["ase_hours"],
            scale=ColourScale(0.0, 2.0 * hours, f"occupied hours at {rules['ase']['lux']} lux or more, blinds open",
                              marks=(hours,))),
         "ASE. The mark on the legend is the hour limit."),
    ]
    for key, spec, text in plans:
        common.add_figure(built, f"{name}_{key}.png", render_plan(spec, renderer), text)
    # Pre-flight ruling F9: a spaces table scoped to this storey, before the
    # window-group table (spec 6.1 item 11).
    _spaces_table(built, case, option, rooms=storey.rooms, level=3)
    doc.add_heading("Window groups", level=3)
    closed = option.results.get("blinds_closed_hours", {})
    rows = [
        (g.group_id, g.room, KIND_LABELS.get(g.kind, g.kind), f"{g.tilt_deg:.0f} deg",
         NO_AZIMUTH if g.azimuth_deg is None else f"{g.azimuth_deg:g} deg",
         str(g.openings or len(g.segments)),
         NO_BLIND_DIFFUSING if g.kind == "diffusing" else str(closed.get(g.group_id, "")))
        for g in here
    ]
    kit.table(doc, ["Group", "Space", "Type", "Tilt", "Facade azimuth", "Openings", "Blind closed hours"], rows,
              aligns=["left", "left", "left", "right", "right", "right", "right"])


def _opening(built: Built, case: CaseData, option: OptionData) -> None:
    common.intent_and_requirements(built, case, OPTION)
    common.input_data(built, case, option, _input_rows(case, option))
    common.method(built, _method_lines(case) + common.layout_notes(case, common.analysed(option)))


def build_project(case: CaseData, renderer: Renderer) -> Built:
    option = _option(case)
    built = common.new_report(case, OPTION)
    _opening(built, case, option)
    limits = _space_rule_limits(built, case)
    _summary(built, option, "Summary results")
    common.deviations_and_warnings(built, case, option)
    _spaces_table(built, case, option)
    if limits is not None:
        _space_rule(built, case, option, limits)
    _by_storey(built, case, option, renderer, mark=None, chart=True)
    _worst_ase(built, case, option)
    _narrative(built, case, option)
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
    rows = {r[0]: r for r in common.storey_averages(case, option, "sda_pct")}
    ase = {r[0]: r for r in common.storey_averages(case, option, "ase_pct")}
    if label in rows:
        kit.kv_table(built.document, [
            ("Spaces", str(rows[label][1])), ("Floor area m2", area(rows[label][2])),
            ("sDA, floor-area weighted", pct(rows[label][3])), ("ASE, floor-area weighted", pct(ase[label][3])),
        ])
    limits = _space_rule_limits(built, case)
    common.deviations_and_warnings(built, case, option)
    _spaces_table(built, case, option, rooms=storey.rooms)
    if limits is not None:
        _space_rule(built, case, option, limits, rooms=storey.rooms)
    built.document.add_heading("Floor plans", level=1)
    _storey_section(built, case, option, storey, renderer)
    built.document.add_heading("The project behind this storey", level=1)
    r = option.results
    kit.kv_table(built.document, [
        ("Project sDA, floor-area weighted", pct(r["sda_area_pct"])),
        ("Project ASE, floor-area weighted", pct(r["ase_area_pct"])),
        ("Points", score_text(r["score"])),
    ])
    _by_storey(built, case, option, renderer, mark=label, chart=False)
    return built
