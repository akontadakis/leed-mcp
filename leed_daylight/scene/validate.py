# leed_daylight/scene/validate.py
"""Spec section 6.0: check the scene before anything is built from it.

Every check here guards a number that would otherwise be quietly wrong.  Nothing
is repaired.  A scene that fails is returned to the person who made it, with the
offending object named.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LinearRing

from .. import geometry
from ..units import LENGTH_TO_M
from .primitives import MATERIAL_TYPES, OTHER_MODIFIER_TYPES, SURFACE_TYPES, Primitive

#: Primitive types a Radiance file may hold that this tool refuses by name,
#: each with the reason and the fix.
_REFUSED_TYPES = {
    "source": "the sky and the sun come from the weather file, so a source "
              "here would be an extra sun in every octree; remove it",
    "instance": "its octree is not read, scaled or checked by this tool; "
                "expand it into polygons in the scene",
    "mesh": "its triangles are not read, scaled or checked by this tool; "
            "convert it to polygons (obj2rad without -m, say)",
}
_KNOWN_TYPES = SURFACE_TYPES | MATERIAL_TYPES | OTHER_MODIFIER_TYPES

PLANARITY_TOLERANCE_M = 0.001
DEGENERATE_AREA_M2 = 1e-9
MIN_BUILDING_SPAN_M = 1.0
MAX_BUILDING_SPAN_M = 2000.0


class SceneError(ValueError):
    """The scene cannot be used as it stands."""


@dataclass(frozen=True)
class Report:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def raise_if_errors(self) -> None:
        if self.errors:
            raise SceneError(
                f"{len(self.errors)} problem(s) in the scene:\n  "
                + "\n  ".join(self.errors)
            )


def _points_or_none(prim: Primitive) -> tuple[np.ndarray | None, str | None]:
    """Unify the `.points` guard that `_span` and the per-primitive loop both need.

    Returns `(points, None)` when `Primitive.points` succeeds -- `points` is
    itself `None` for a non-polygon primitive, which is not an error.
    Returns `(None, message)` when `Primitive.points` raises `ValueError` (a
    malformed polygon), with `message` taken from that exception so a caller
    can still name the primitive.  This unifies the *access* only; each call
    site still decides for itself what a refusal means.
    """
    try:
        return prim.points, None
    except ValueError as exc:
        return None, str(exc)


def _span(prims: Sequence[Primitive]) -> tuple[float, bool]:
    """Bounding-box span across every primitive that actually yields vertices.

    Returns `(span, has_vertices)`.  `has_vertices` is true only when at
    least one primitive contributed at least one usable vertex -- a
    non-polygon type, a malformed polygon, an empty (zero-vertex) polygon,
    and a polygon with a non-finite (NaN or Inf) vertex all contribute
    nothing, so `validate_scene` can gate the size check on real geometry
    rather than on raw `ptype` membership.
    """
    corners = []
    for p in prims:
        if p.ptype != "polygon":
            continue
        pts, _ = _points_or_none(p)
        if pts is not None and len(pts) > 0 and np.isfinite(pts).all():
            corners.append(pts)
    if not corners:
        return 0.0, False
    stacked = np.vstack(corners)
    span = float(np.linalg.norm(stacked.max(axis=0) - stacked.min(axis=0)))
    return span, True


def crossing_edges(pts: np.ndarray) -> tuple[int, int] | None:
    """The first pair of edges that properly cross, in the polygon's plane.

    Only a proper crossing -- two non-adjacent edges meeting at a point
    inside both -- counts.  A keyhole polygon (Radiance's usual way of
    writing a surface with a hole: a seam in, the hole, the same seam
    out) has edges that overlap or touch but never cross, so it passes;
    a bow-tie does not.  GEOS's ``is_simple`` screens the common case
    first, so the pairwise test runs only on a ring that is not simple.
    """
    flat = geometry.to_plane(pts, geometry.plane_basis(pts))
    if LinearRing(flat).is_simple:
        return None
    n = len(flat)
    scale = float(np.ptp(flat, axis=0).max()) or 1.0
    eps = 1e-9 * scale * scale

    def cross(o, a, b) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    for i in range(n):
        a, b = flat[i], flat[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue                                  # adjacent through the wrap
            c, d = flat[j], flat[(j + 1) % n]
            d1, d2 = cross(a, b, c), cross(a, b, d)
            d3, d4 = cross(c, d, a), cross(c, d, b)
            if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and (
                (d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)
            ):
                return i, j
    return None


def _canonical_cycle_key(pts: np.ndarray) -> tuple:
    """The smallest rotation of `pts`, or of its reverse, as rounded tuples.

    Treats the same physical surface as a duplicate regardless of which
    vertex an exporter started from or which way it wound the loop, while
    leaving a genuinely different vertex order (a star built from a
    pentagon's points, say) alone.
    """
    rounded = tuple(tuple(v) for v in np.round(pts, 6))
    n = len(rounded)
    reversed_seq = tuple(reversed(rounded))
    return min(
        seq[i:] + seq[:i] for seq in (rounded, reversed_seq) for i in range(n)
    )


def _modifier_order(prims: Sequence[Primitive]) -> list[str]:
    """Modifier references and definitions.

    A SURFACE may use a modifier defined anywhere in the scene: the
    prepared model writes every modifier into ``materials.rad``, which
    oconv reads ahead of the surfaces, so ``[geometry.rad,
    materials.rad]`` builds fine. ORDER is enforced among modifier
    definitions only: a material or pattern using another modifier must
    come after it, because the prepared model keeps their reading order
    and oconv refuses a forward reference. A modifier defined nowhere is
    refused either way. A modifier name defined twice with different
    contents is refused too -- oconv would silently use whichever came
    last, so the reflectance in the report would depend on file order.
    The same definition read twice (a material file included by two
    transforms, say) is harmless and passes.
    """
    errors: list[str] = []
    defined: dict[str, Primitive] = {}
    all_names = {p.name for p in prims if not p.is_surface}
    for prim in prims:
        if prim.modifier != "void" and prim.modifier not in all_names:
            errors.append(f"{prim.name}: modifier {prim.modifier!r} is never defined")
        elif prim.modifier != "void" and not prim.is_surface and prim.modifier not in defined:
            errors.append(
                f"{prim.name} ({prim.source}): modifier {prim.modifier!r} is "
                "used before it is defined; oconv refuses this, so move "
                "the definition above its first use"
            )
        if prim.is_surface:
            continue
        first = defined.get(prim.name)
        if first is None:
            defined[prim.name] = prim
        elif (first.modifier, first.ptype, first.sargs, first.iargs, first.fargs) != (
            prim.modifier, prim.ptype, prim.sargs, prim.iargs, prim.fargs
        ):
            errors.append(
                f"modifier {prim.name!r} is defined twice with different contents, "
                f"in {first.source} and in {prim.source}; oconv would use the "
                "last one silently, so rename one or remove it"
            )
    return errors


def validate_scene(
    prims: Sequence[Primitive],
    declared_units: str,
    roles: dict[str, str] | None = None,
    declared_types: dict[str, str] | None = None,
) -> Report:
    """Run every scene check.  Returns the findings; it never raises by itself.

    `declared_types` is materials.csv's modifier -> primitive_type (blank
    cells left out).  A declared type that differs from the scene's own
    definition of that material is an error: the role was chosen for a
    different material than the one the run will use.

    `declared_units` is a label only: every primitive is already in metres,
    converted once by `read_scene`.  It is checked against the known unit
    set and named in the implausible-size message, and nothing else.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if declared_units not in LENGTH_TO_M:
        errors.append(
            f"unknown length unit {declared_units!r}; expected one of {sorted(LENGTH_TO_M)}"
        )

    surfaces = [p for p in prims if p.is_surface]
    if not surfaces:
        errors.append("the scene has no surface at all")

    for prim in prims:
        if prim.ptype in _REFUSED_TYPES:
            errors.append(
                f"{prim.name} ({prim.source}): a {prim.ptype!r} is not supported "
                f"in the scene; {_REFUSED_TYPES[prim.ptype]}"
            )
        elif prim.ptype not in _KNOWN_TYPES:
            errors.append(
                f"{prim.name} ({prim.source}): primitive type {prim.ptype!r} is "
                "not supported in the scene; it is not a Radiance surface, "
                "material, pattern, texture or mixture this tool knows"
            )

    for prim in surfaces:
        if prim.ptype != "polygon" and prim.ptype not in _REFUSED_TYPES:
            errors.append(
                f"{prim.name}: unsupported surface type {prim.ptype!r}; "
                "this tool supports polygon geometry only"
            )

    span, has_vertices = _span(prims)
    if has_vertices and not MIN_BUILDING_SPAN_M <= span <= MAX_BUILDING_SPAN_M:
        errors.append(
            f"implausible size: the model spans {span:.3f} m after converting from "
            f"{declared_units!r}. Expected between {MIN_BUILDING_SPAN_M} m and "
            f"{MAX_BUILDING_SPAN_M} m. Check scene.units."
        )

    errors.extend(_modifier_order(prims))

    seen: dict[tuple, str] = {}
    for prim in surfaces:
        pts, err = _points_or_none(prim)
        if err is not None:
            errors.append(err)
            continue
        if pts is None:
            continue
        if len(pts) < 3:
            errors.append(f"{prim.name}: polygon has fewer than three vertices")
            continue
        if not np.isfinite(pts).all():
            errors.append(f"{prim.name}: polygon has a non-finite vertex (NaN or Inf)")
            continue
        try:
            if geometry.area(pts) < DEGENERATE_AREA_M2:
                errors.append(f"{prim.name}: degenerate polygon, zero area")
                continue
            if geometry.planarity_error(pts) > PLANARITY_TOLERANCE_M:
                errors.append(
                    f"{prim.name}: not planar within {PLANARITY_TOLERANCE_M * 1000:.0f} mm"
                )
            elif (crossing := crossing_edges(pts)) is not None:
                i, j = crossing
                errors.append(
                    f"{prim.name}: polygon crosses itself (edge {i} crosses edge "
                    f"{j}), so its area is wrong; redraw it as a simple polygon"
                )
                continue
        except ValueError as exc:
            errors.append(f"{prim.name}: degenerate polygon ({exc})")
            continue
        key = (prim.modifier, _canonical_cycle_key(pts))
        if key in seen:
            errors.append(f"{prim.name}: duplicate of {seen[key]}, same modifier and vertices")
        else:
            seen[key] = prim.name

    if roles is not None:
        for modifier in sorted({p.modifier for p in surfaces}):
            if modifier not in roles:
                errors.append(f"modifier {modifier!r} has no role in materials.csv")

    if declared_types is not None:
        # The last definition wins, as it does in Radiance.
        defined = {p.name: p.ptype for p in prims if p.is_material}
        for modifier in sorted(declared_types):
            actual = defined.get(modifier)
            if actual is not None and actual != declared_types[modifier]:
                errors.append(
                    f"modifier {modifier!r}: materials.csv says primitive type "
                    f"{declared_types[modifier]!r}, but the scene defines it as "
                    f"{actual!r}; check the row and its role"
                )

    return Report(tuple(errors), tuple(warnings))
