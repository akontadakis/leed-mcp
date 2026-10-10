# leed_daylight/deliver/word/common.py
"""Report sections shared by Option 1, Option 2 and Quality Views (reports
spec 6): the title block, intent and requirements, input data, method,
deviations and warnings, figures, and per-storey plan and chart helpers.

Pre-flight ruling F8: the credit's intent sentence is not carried as a
literal string in this module. Spec R5 and section 4 say the report's
wording comes from the credit rules file, so ``intent_and_requirements``
reads it from ``case.rules`` (top-level ``intent`` for Option 1 and
Option 2, ``views.intent`` for Quality Views) and simply omits the
paragraph -- with a warning recorded on ``Built`` for the Warnings section
-- when the credit rules file carries none.

Task 9 ruling: the same applies to the Quality Views points band. When
``views.threshold_pct`` is stated but ``views.points`` is not,
``intent_and_requirements`` used to fall back to a literal ``1`` silently;
it now omits the band table and warns (``NO_VIEWS_POINTS_WARNING``)
instead, since a fabricated point count would be worse than none.

Merge-split spec 7: the Method section of every report ends with one
sentence per analysed space that is split or merged (``layout_notes``; the
blind clause only in the Option 1 report, the one option with blinds), and
a storey plan draws each split line of its spaces once, dashed.
"""
from __future__ import annotations

import dataclasses
import io
import re
from collections.abc import Iterable, Mapping

import numpy as np
from docx.shared import Cm

from ...constants import UNVERIFIED
from ..data import CaseData, OptionData, StoreyPlan
from ..figures.spec import MAX_PRINT_HEIGHT_CM, BarSpec, ColourScale, PlanPolygon, PlanSpec
from . import kit

OPTION_TITLES = {
    "option1": "EQ Credit Daylight, Option 1: annual simulation",
    "option2": "EQ Credit Daylight, Option 2: illuminance calculation",
    "views": "EQ Credit Quality Views",
}

#: The warning added to the Warnings section when the credit rules file has
#: no intent text for the option being reported (pre-flight ruling F8).
NO_INTENT_WARNING = "the credit rules file has no intent text"

#: The warning added to the Warnings section when Quality Views has a
#: threshold but the credit rules file's `views` section carries no
#: `points` value for it (Task 9 ruling: a missing points value is warned
#: about, never silently defaulted to 1).
NO_VIEWS_POINTS_WARNING = "the credit rules file's views section has no points value"


@dataclasses.dataclass
class Built:
    document: object
    figures: list[tuple[str, bytes]] = dataclasses.field(default_factory=list)
    #: Messages raised while building earlier sections (so far: a missing
    #: intent sentence) that belong in the Warnings section built later.
    warnings: list[str] = dataclasses.field(default_factory=list)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "storey"


def pct(value) -> str:
    return "not computed" if value is None else f"{float(value):.1f} %"


def area(value) -> str:
    return f"{float(value):,.1f}"


def score_text(score: Mapping) -> str:
    if score.get("points") is None:
        return f"not awarded ({score.get('reason') or 'no reason recorded'})"
    band = score.get("band") or "below the lowest band"
    return f"{score['points']} ({band})"


def new_report(case: CaseData, option: str, scope: str = "") -> Built:
    doc = kit.new_doc(f"{case.project.name}, {OPTION_TITLES[option]}, case {case.case}, page")
    doc.add_heading(OPTION_TITLES[option] + (f", {scope}" if scope else ""), level=0)
    lines = [
        f"Project: {case.project.name}", f"Credit: {case.project.credit}",
        f"Case: {case.case}", f"Date: {case.created.strftime('%d %B %Y')}",
    ]
    if case.project.report.prepared_by:
        lines.append(f"Prepared by: {case.project.report.prepared_by}")
    for line in lines:
        doc.add_paragraph(line)
    return Built(doc)


def _band_table(doc, bands, label: str) -> None:
    if not isinstance(bands, list):
        doc.add_paragraph(f"Point bands: {bands}; no points are awarded until they are verified.")
        return
    kit.table(doc, [label, "Points"], [(f"{float(b['min_pct']):g} % or more", str(b["points"])) for b in bands],
              widths=[9.0, 3.0], aligns=["left", "right"])


def intent_and_requirements(built: Built, case: CaseData, option: str) -> None:
    doc, rules = built.document, case.rules
    doc.add_heading("Intent", level=1)
    intent = rules.get("views", {}).get("intent") if option == "views" else rules.get("intent")
    if intent:
        doc.add_paragraph(intent)
    else:
        built.warnings.append(NO_INTENT_WARNING)
    doc.add_heading("Requirements", level=1)
    source = rules.get("views", {}) if option == "views" else rules
    kit.kv_table(doc, [
        ("Rating system", source.get("version") or "not stated in credit_rules.yaml"),
        ("Source of the thresholds", source.get("source") or "not stated in credit_rules.yaml"),
    ])
    if option == "option1":
        sda, ase = rules["option1"]["sda"], rules["option1"]["ase"]
        share = 100.0 * float(sda["fraction"])
        kit.bullets(doc, [
            f"Spatial daylight autonomy sDA{sda['lux']}/{share:.0f}%: the share of regularly occupied floor "
            f"area that reaches at least {sda['lux']} lux from daylight for at least {share:.0f} % of the "
            "occupied hours, with blinds operated as the standard prescribes.",
            f"Annual sunlight exposure ASE{ase['lux']},{ase['hours']}: the share of floor area that receives "
            f"at least {ase['lux']} lux of direct sunlight for more than {ase['hours']} occupied hours, blinds open.",
            f"A regularly occupied space with ASE over {float(ase['per_space_narrative_trigger_pct']):g} % needs "
            "a narrative of how the design addresses glare.",
        ])
        _band_table(doc, sda["bands"], "Average sDA")
    elif option == "option2":
        o2 = case.project.option2
        kit.bullets(doc, [
            f"Illuminance between {o2.lux_min:g} and {o2.lux_max:g} lux at {' and '.join(o2.times)} on the "
            f"clearest day within {o2.clearest_day_window_days} days of each of {', '.join(o2.equinox_dates)}, "
            "averaged over the two days, with blinds excluded from the model.",
            "A space with view-preserving automatic glare control need meet only the lower limit.",
        ])
        _band_table(doc, rules["option2"]["bands"], "Floor area in the band")
    else:
        views = rules.get("views", {})
        threshold = views.get("threshold_pct", UNVERIFIED)
        kit.bullets(doc, [
            "A Quality View is a view that meets the Type 2 and Type 3 tests of LEED v4.1 EQ Credit Quality Views.",
            f"The credit asks for a Quality View from {threshold} % of the regularly occupied floor area.",
        ])
        if threshold != UNVERIFIED:
            points = views.get("points")
            if points is None:
                built.warnings.append(NO_VIEWS_POINTS_WARNING)
                doc.add_paragraph(
                    "Points: not stated in credit_rules.yaml (views.points); no points are "
                    "awarded until it is verified."
                )
            else:
                _band_table(doc, [{"min_pct": threshold, "points": points}], "Floor area with a Quality View")


def site_rows(case: CaseData, option_data: OptionData) -> list[tuple[str, str]]:
    p, s = case.project, case.project.site
    return [
        ("Project", p.name),
        ("Site", f"latitude {s.latitude_N:g}, longitude {s.longitude_E:g} (east positive), "
                 f"time meridian {s.meridian_radiance:g}"),
        ("Weather file", p.weather_file.name),
        ("True north", f"{s.north_rotation_deg:g} deg from the model +Y axis, counter-clockwise"),
        ("Source of true north", s.north_rotation_source),
        ("Daylight standard", p.standard),
        ("Engine", f"Radiance {option_data.results.get('radiance_version') or '(version not recorded)'}"),
    ]


def input_data(built: Built, case: CaseData, option_data: OptionData, extra_rows) -> None:
    built.document.add_heading("Input data", level=1)
    kit.kv_table(built.document, site_rows(case, option_data) + list(extra_rows))


def method(built: Built, lines) -> None:
    built.document.add_heading("Method", level=2)
    kit.bullets(built.document, list(lines))


def layout_notes(case: CaseData, rooms: Iterable[str], *, blinds: bool = True) -> list[str]:
    """The layout sentence of each of ``rooms`` that is split or merged, in
    ``rooms`` order (``deliver.data.layout_note``); ``blinds=False`` gives
    the sentences without the blind clause, for an option with no blinds."""
    notes = case.layout_notes if blinds else case.layout_notes_no_blinds
    return [notes[room] for room in rooms if room in notes]


def analysed(option_data: OptionData) -> list[str]:
    """The spaces this option analysed in this case, in its results' order."""
    return [s["room"] for s in option_data.results.get("spaces", [])]


def deviations_and_warnings(built: Built, case: CaseData, option_data: OptionData) -> None:
    doc = built.document
    doc.add_heading("Deviations", level=1)
    if case.project.report.deviations:
        kit.bullets(doc, list(case.project.report.deviations))
    else:
        doc.add_paragraph("None stated in project.yaml (report.deviations).")
    doc.add_heading("Warnings", level=1)
    warnings = list(built.warnings) + list(option_data.results.get("warnings", [])) + list(case.warnings)
    if warnings:
        kit.bullets(doc, warnings)
    else:
        doc.add_paragraph("None.")


def scope_note(built: Built, storey_label: str) -> None:
    kit.note(built.document, "Scope of this report. ",
             f"This report shows storey {storey_label} only. The credit is judged on the whole project; "
             "the storey figures are for diagnosis. The project results follow at the end.")


def _png_size(png: bytes) -> tuple[int, int]:
    """(width, height) in pixels from a PNG's IHDR chunk."""
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


def add_figure(built: Built, name: str, png: bytes, caption_text: str) -> None:
    """The figure at the usable width -- or, when that would make it taller
    than ``MAX_PRINT_HEIGHT_CM``, at that height (final review I4)."""
    built.figures.append((name, png))
    width_px, height_px = _png_size(png)
    if width_px and kit.USABLE * height_px / width_px > MAX_PRINT_HEIGHT_CM:
        built.document.add_picture(io.BytesIO(png), height=Cm(MAX_PRINT_HEIGHT_CM))
    else:
        built.document.add_picture(io.BytesIO(png), width=Cm(kit.USABLE))
    kit.caption(built.document, caption_text)


def spaces_of(option_data: OptionData, rooms: Iterable[str] | None = None) -> list[dict]:
    spaces = list(option_data.results.get("spaces", []))
    if rooms is None:
        return spaces
    wanted = set(rooms)
    return [s for s in spaces if s["room"] in wanted]


def storey_averages(case: CaseData, option_data: OptionData, key: str) -> list[tuple[str, int, float, float]]:
    rows = []
    for storey in case.storeys:
        members = spaces_of(option_data, storey.rooms)
        total = sum(float(s["area_m2"]) for s in members)
        if members and total > 0.0:
            weighted = sum(float(s["area_m2"]) * float(s[key]) for s in members) / total
            rows.append((storey.label, len(members), total, weighted))
    return rows


def storey_plan(case: CaseData, storey: StoreyPlan, option_data: OptionData, *, title: str,
                values=None, scale: ColourScale | None = None, passed=None, segments=(),
                polygons: Iterable[PlanPolygon] = (), legend=("passes", "fails"),
                points: bool = True) -> PlanSpec:
    """One storey's plan. Final review I3 (Ruling FR-I3): only the spaces
    this option analysed in this case (its results' ``spaces``) are
    outlined and labelled; corridors, spaces that are not regularly
    occupied and spaces left out of the case are not drawn. ``polygons``
    are drawn over the plan (roof glass, dashed; skylights spec 7)."""
    table = option_data.points
    analysed = {s["room"] for s in option_data.results.get("spaces", [])}
    rooms = [room for room in storey.rooms if room in analysed]
    mask = table.mask(rooms) if points else np.zeros(len(table.rooms), dtype=bool)
    outlines = tuple(ring for room in rooms for ring in storey.outlines.get(room, ()))
    labels = tuple((room, *storey.centroids[room]) for room in rooms)
    # A line two halves share is one array: drawn once.
    split_lines = {id(line): line for room in rooms for line in storey.split_lines.get(room, ())}
    return PlanSpec(
        title=title, outlines=outlines, labels=labels, points_xy=table.xy[mask],
        cell_m=case.project.grid.spacing_m, north_rotation_deg=case.project.site.north_rotation_deg,
        values=None if values is None else np.asarray(values, dtype=float)[mask], scale=scale,
        passed=None if passed is None else np.asarray(passed, dtype=bool)[mask],
        segments=tuple(segments), polygons=tuple(polygons), legend_labels=tuple(legend),
        split_lines=tuple(split_lines.values()),
    )


def storey_bar(case: CaseData, option_data: OptionData, *, title: str, key: str, unit: str,
               threshold: float | None) -> BarSpec:
    rows = storey_averages(case, option_data, key)
    return BarSpec(title=title, categories=tuple(r[0] for r in rows), values=tuple(r[3] for r in rows),
                   unit=unit, threshold=threshold)
