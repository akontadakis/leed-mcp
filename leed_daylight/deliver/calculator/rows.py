"""A case's results as calculator rows (calculator spec 5.1, D3, D4).

Daylight: one row per regularly occupied space row of the case, in
spaces.csv order. Quality Views: up to four rows per space, split by the
points of ``views/<case>/points.csv``: "view, distance", "view, content",
"no line of sight", "view does not qualify". Each row's area is the space's
area times its share of points, so the calculator's totals equal the tool's.
A credit with no current results leaves its columns empty and is named.
"""
from __future__ import annotations

import dataclasses
import decimal

import numpy as np

from ...project.materials import TAG_VIEWS, VLT_ROLES, load_materials, view_overrides, vlt_values, vlt_warnings
from ...project.spaces import case_names, space_of, space_rows, views_case_names
from ..data import CaseData, DeliverError
from .v02 import CONTENT

__all__ = ["DaylightRow", "M2_PER_FT2", "Rows", "ViewsRow", "calculator_rows", "expected_summary"]

M2_PER_FT2 = 0.09290304  # source: 1 ft = 0.3048 m exactly, so 1 ft2 = 0.09290304 m2


@dataclasses.dataclass(frozen=True)
class DaylightRow:
    id: str
    description: str
    area: float
    sda: float | None
    ase: float | None
    in_band: float | None
    above: float | None
    shades: str


@dataclasses.dataclass(frozen=True)
class ViewsRow:
    id: str
    description: str
    area: float
    sight: str
    vlt: float | None
    content: str


@dataclasses.dataclass(frozen=True)
class Rows:
    daylight: tuple[DaylightRow, ...]
    views: tuple[ViewsRow, ...]
    #: What the Summary sheet should show once Excel recalculates.
    expected: dict[str, float | None]
    warnings: tuple[str, ...]


def _r(value: float | None) -> float | None:
    return None if value is None else round(float(value), 1)


def _space_area(spaces, name: str) -> float:
    """The space's floor area: its row and every piece merged into it, directly or through a chain."""
    owner = space_of(spaces)
    return sum(s.area_m2 for s in spaces if owner[s.room] == name)


def _written_vlt(vlt: float) -> float:
    """The VLT to write, to one decimal, with the tool's pass/fail: the tool
    passes a glazing above 40 %, the calculator at 40 % or more, so a value
    the tool fails is written below 40 (39.96 -> 39.9, 40 -> 39.9) and one it
    passes is never written as 40.0 (40.04 -> 40.1)."""
    if vlt <= 40.0:
        exact = decimal.Decimal(str(vlt))
        return min(float(exact.quantize(decimal.Decimal("0.1"), rounding=decimal.ROUND_FLOOR)), 39.9)
    return max(round(vlt, 1), 40.1)


def _by_room(case: CaseData, option: str) -> dict[str, dict]:
    if option not in case.options:
        return {}
    return {s["room"]: s for s in case.options[option].results.get("spaces", [])}


def _option2_areas(case: CaseData, name: str, area: float) -> tuple[float, float] | tuple[None, None]:
    """(area from 300 to 3,000 lux at every time, area above 3,000 lux at
    any time and at least 300 lux at every time), from the per-point
    averages, area-weighted by points: in_band + above is the tool's passing
    area for a space with view-preserving shades (drivers/run_option2.py)."""
    if "option2" not in case.options:
        return None, None
    table = case.options["option2"].points
    mask = table.mask([name])
    if not mask.any():
        return None, None
    low, high = case.project.option2.lux_min, case.project.option2.lux_max
    lux = np.stack([v[mask] for k, v in table.columns.items() if k.startswith("average_")], axis=1).astype(float)
    in_band = ((lux >= low) & (lux <= high)).all(axis=1)
    above = (lux > high).any(axis=1) & (lux >= low).all(axis=1)
    n = mask.sum()
    return area * in_band.sum() / n, area * above.sum() / n


def _space_vlt(case: CaseData, name: str, roles: dict[str, str], vlts: dict[str, float],
               warnings: list[str]) -> float | None:
    """The lowest VLT of the space's own vision glazing (D3); with no window
    of its own, the lowest of the project's, named in a warning."""
    own = [m for g in case.groups if g.room == name for m in g.materials if roles.get(m) in VLT_ROLES]
    if not own:
        own = [m for m, role in roles.items() if role in VLT_ROLES]
        warnings.append(f"{name}: no window of its own; its VLT is the lowest of the project's vision glazing")
    values = [vlts[m] for m in own if m in vlts]
    return min(values) if values else None


def expected_summary(daylight, views, option2_ran: bool) -> dict[str, float | None]:
    """What the Summary sheet should show once Excel recalculates these rows."""
    total = sum(r.area for r in daylight)
    with_sda = [r for r in daylight if r.sda is not None]
    daylit = sum((r.in_band or 0.0) + ((r.above or 0.0) if r.shades == "Yes" else 0.0) for r in daylight)
    views_total = sum(r.area for r in views)
    viewed = sum(r.area for r in views if r.sight == "Yes" and r.vlt is not None and r.vlt >= 40
                 and r.content != CONTENT["none"])
    return {
        "total_area": _r(total) if daylight else None,
        "average_sda": _r(sum(r.sda * r.area for r in with_sda) / total) if with_sda and total else None,
        "daylit_area": _r(daylit) if option2_ran else None,
        "daylit_pct": _r(100.0 * daylit / total) if total and option2_ran else None,
        "views_area": _r(viewed) if views else None,
        "views_total": _r(views_total) if views else None,
        "views_pct": _r(100.0 * viewed / views_total) if views_total else None,
    }


def calculator_rows(case: CaseData, units: str) -> Rows:
    if units not in ("IP", "SI"):
        raise ValueError(f"units must be 'IP' or 'SI', not {units!r}")
    scale = 1.0 if units == "SI" else 1.0 / M2_PER_FT2
    warnings: list[str] = [f"{option}: {reason}" for option, reason in case.skipped.items()]
    for option in ("option1", "option2"):
        # Not on the LEED basis: the values below are still filled, but the tool awards no points.
        score = case.options[option].results.get("score", {}) if option in case.options else {}
        if score and score.get("points") is None:
            warnings.append(f"{option}: no points are awarded, yet its values are filled in and Excel will "
                            f"score them: {score.get('reason')}")
    warnings.append("the 30 % cap on views into interior atria is not checked; check it by hand")
    warnings.append("Daylight!H17 of the USGBC calculator counts the sDA column, not ASE; "
                    "the Summary sheet's ASE line is the one to read")
    by_name = {s.room: s for s in case.spaces}
    materials_path = case.folder / "materials.csv"
    roles, vlts = load_materials(materials_path), vlt_values(materials_path)
    warnings.extend(vlt_warnings(materials_path))
    tags = {v for v in view_overrides(materials_path).values() if v in TAG_VIEWS}

    daylight: list[DaylightRow] = []
    if case.case in case_names(case.spaces):
        o1, o2 = _by_room(case, "option1"), _by_room(case, "option2")
        for space in space_rows(case.spaces):
            if not space.include.get(case.case) or not space.regularly_occupied:
                continue
            one = o1.get(space.room, {})
            # The scene's room area, as every other output uses; spaces.csv only when no option has the space.
            recorded = one.get("area_m2", o2.get(space.room, {}).get("area_m2"))
            area = float(recorded) if recorded is not None else _space_area(case.spaces, space.room)
            in_band, above = _option2_areas(case, space.room, area)
            daylight.append(DaylightRow(
                id=space.room, description=space.space_type, area=_r(area * scale),
                sda=_r(one.get("sda_pct")), ase=_r(one.get("ase_pct")),
                in_band=_r(None if in_band is None else in_band * scale),
                above=_r(None if above is None else above * scale),
                shades="Yes" if space.view_preserving_shades else "No",
            ))
            if space.room not in o1 and "option1" in case.options:
                warnings.append(f"{space.room}: not in the Option 1 results")
            if space.room not in o2 and "option2" in case.options:
                warnings.append(f"{space.room}: not in the Option 2 results")

    views: list[ViewsRow] = []
    if "views" in case.options and case.case in views_case_names(case.spaces):
        table = case.options["views"].points
        # The distance the run itself used: the views driver records its rules (QV spec 6).
        limit = case.options["views"].results.get("settings", {}).get("rules", {}).get("object_distance_m")
        if limit is None:
            raise DeliverError("the Quality Views results record no object distance (settings.rules."
                               "object_distance_m); run Quality Views again")
        limit = float(limit)
        content = CONTENT["nature"] if not tags or "nature" in tags else CONTENT["art"]
        if "nature" in tags and tags - {"nature"}:
            warnings.append("the project tags both nature and art or landmarks; the 'view, content' rows say "
                            "'nature': check each one")
        for result in case.options["views"].results.get("spaces", []):
            name = result["room"]
            mask = table.mask([name])
            if not mask.any():
                continue
            space = by_name.get(name)
            description = space.space_type if space else ""
            area = float(result["area_m2"])
            quality = table.columns["quality"][mask].astype(bool)
            type3 = table.columns["type3"][mask].astype(bool)
            far = table.columns["median_outside_m"][mask].astype(float) >= limit
            n = mask.sum()
            vlt = _space_vlt(case, name, roles, vlts, warnings)
            if vlt is not None and vlt == 40.0:
                warnings.append(f"{name}: VLT exactly 40 % passes in the calculator (40 or more) but not in this "
                                "tool (above 40 %); 39.9 is written so the calculator fails it too")
            vlt = None if vlt is None else _written_vlt(vlt)
            groups = (
                ("view, distance", quality & far, "Yes", CONTENT["distance"]),
                ("view, content", quality & ~far, "Yes", content),
                ("no line of sight", ~type3, "No", CONTENT["none"]),
                ("view does not qualify", type3 & ~quality, "Yes", CONTENT["none"]),
            )
            for label, chosen, sight, text in groups:
                if chosen.any():
                    views.append(ViewsRow(id=f"{name} ({label})", description=description,
                                          area=_r(area * chosen.sum() / n * scale), sight=sight, vlt=_r(vlt), content=text))

    expected = expected_summary(daylight, views, "option2" in case.options)
    return Rows(tuple(daylight), tuple(views), expected, tuple(warnings))
