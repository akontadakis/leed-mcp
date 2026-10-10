# leed_daylight/scene/inventory.py
"""What materials the scene uses, and how much surface each one covers.

The inventory is what the material-mapping page shows.  A guess is offered so
the common case is one click, but nothing runs until a human has confirmed
every role: a guessed reflectance in a LEED submission is a substitution, and
substitutions have to be declared.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .. import geometry
from .primitives import Primitive

GLASS_TYPES = frozenset({"glass", "dielectric", "BSDF", "aBSDF"})
TRANS_TYPES = frozenset({"trans", "trans2"})
OPAQUE_TYPES = frozenset({"plastic", "metal", "plastic2", "metal2", "mirror"})
LIGHT_TYPES = frozenset({"light", "glow", "spotlight", "illum"})

GUESS = {"glass": "glazing", "trans": "fixed_shading", "opaque": "wall"}


@dataclass(frozen=True)
class ModifierInfo:
    name: str
    ptype: str
    kind: str
    rgb: tuple[float, float, float] | None
    surfaces: int
    area_m2: float
    guessed_role: str


def _kind(ptype: str) -> str:
    if ptype in GLASS_TYPES:
        return "glass"
    if ptype in TRANS_TYPES:
        return "trans"
    if ptype in OPAQUE_TYPES:
        return "opaque"
    if ptype in LIGHT_TYPES:
        return "light"
    return "other"


def _guessed_role(kind: str, material: Primitive | None) -> str:
    """The role offered for a modifier. A plain ``trans`` with exactly
    seven reals whose seventh (the transmitted specularity) is 0 lets no
    light straight through, so it is guessed as diffusing glazing; any
    other ``trans`` (a shade, a louvre) stays ``fixed_shading``."""
    if (
        kind == "trans"
        and material is not None
        and material.ptype == "trans"
        and len(material.fargs) == 7
        and material.fargs[6] == 0.0
    ):
        return "glazing_diffusing"
    return GUESS.get(kind, "")


def inventory(prims: Sequence[Primitive]) -> list[ModifierInfo]:
    """One row per modifier that at least one surface uses."""
    defined = {p.name: p for p in prims if p.is_material}
    counts: dict[str, int] = {}
    areas: dict[str, float] = {}
    for prim in prims:
        if not prim.is_surface:
            continue
        counts[prim.modifier] = counts.get(prim.modifier, 0) + 1
        area = 0.0
        try:
            pts = prim.points
            if pts is not None and len(pts) >= 3 and np.isfinite(pts).all():
                area = geometry.area(pts)
        except ValueError:
            # A malformed or degenerate polygon adds no area here; validation
            # refuses it by name, so it never reaches a build.
            pass
        areas[prim.modifier] = areas.get(prim.modifier, 0.0) + area

    rows = []
    for name, count in counts.items():
        material = defined.get(name)
        ptype = material.ptype if material else ""
        kind = _kind(ptype)
        rgb = None
        if material and kind in ("opaque", "glass", "trans") and len(material.fargs) >= 3:
            rgb = tuple(material.fargs[:3])
        rows.append(
            ModifierInfo(
                name=name,
                ptype=ptype,
                kind=kind,
                rgb=rgb,
                surfaces=count,
                area_m2=areas.get(name, 0.0),
                guessed_role=_guessed_role(kind, material),
            )
        )
    return sorted(rows, key=lambda r: r.area_m2, reverse=True)
