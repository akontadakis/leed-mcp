# leed_daylight/build/materials.py
"""Rebuild glazing and opaque materials for the prepared model (spec 5.1, 6.8).

A modeller's glass and paint carry whatever the modelling tool happened to
write. This module turns that into what the run actually needs:

* Exterior glazing gets the LM-83-12 clause 2.2.9 dirt factor (unless the
  scene already carries it), and its visible transmittance is converted to
  the Radiance transmissivity a ``glass`` primitive needs -- confusing the
  two makes the glass about 8 % too dark (spec 6.8).
* Opaque surfaces get their reflectance set according to the project's
  policy: every role rebuilt, only the gaps filled, or nothing touched.

Only two Radiance primitive types are ever read as a reflectance
(``plastic``, ``metal``); a ``glass``/``dielectric`` primitive's real
arguments are a transmissivity triple, and reading them as RGB reflectance
would silently turn a window into a mirror. Every branch here dispatches on
``Primitive.ptype``, never on the field name or the role alone, so that
confusion cannot happen (see phase3-global.md's account of the
``ModifierInfo.rgb`` finding).

A material primitive's ``name`` is the modifier name surfaces reference, so
a material's role is ``roles.get(prim.name)``. A material absent from
``roles`` is left untouched -- it may be referenced only by another
material (a glow behind a light fixture, for instance).

Dirt per tilt class (spec 5.4): ``rebuild_materials_by_class`` is told
each exterior pane's class. An original glazing modifier is rebuilt with the
vertical factor (or written unchanged when there is none); a modifier used
by a sloped or skylight pane gets a copy named ``<modifier>__sloped`` or
``<modifier>__skylight`` with that class's factor, written right after
it. ``rebuild_materials`` is the all-vertical path and writes exactly what it
wrote before classes existed. Diffusing glazing (role ``glazing_diffusing``)
is a Radiance ``trans`` with no see-through part; dirt scales only its
diffuse transmission. Frosted interior glass is the same kind of ``trans``
with the role ``glazing_interior``: checked the same way, written unchanged.
"""
from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping

from ..project.materials import EXTERIOR_GLAZING_ROLES, ROLES
from ..scene.primitives import Primitive
from ..units import LUX_WEIGHTS, transmissivity, transmittance_of
from .glazing_class import NON_VERTICAL_TILT_CLASSES, TILT_CLASSES

if typing.TYPE_CHECKING:
    from ..project.config import Glazing, Reflectance
    from .glazing_class import PaneClass

#: The three glazing roles this module ever rebuilds or records. Interior
#: glazing is included because it still gets a record, even though its
#: primitive is left unchanged (clause 2.2.9 is about the envelope only).
_GLAZING_ROLES: frozenset[str] = EXTERIOR_GLAZING_ROLES | frozenset({"glazing_interior"})

#: Roles that name a real opaque surface. A glazing-typed primitive
#: (glass/dielectric/BSDF/aBSDF) carrying one of these is always a mistake
#: in materials.csv, never a legitimate case.
_OPAQUE_ROLES: frozenset[str] = frozenset(
    {"wall", "ceiling", "floor", "furniture", "context", "ground"}
)

#: Roles this module never touches; someone else's job, or none at all.
_UNTOUCHED_ROLES: frozenset[str] = frozenset({"fixed_shading", "ignore"})

assert _GLAZING_ROLES | _OPAQUE_ROLES | _UNTOUCHED_ROLES == ROLES

#: Primitive types whose real arguments are a transmissivity triple, never
#: a reflectance -- the shape the opaque-role guard below refuses.
_GLAZING_TYPES: frozenset[str] = frozenset({"glass", "dielectric", "BSDF", "aBSDF"})

#: Primitive types whose real arguments are a diffuse reflectance triple.
#: ``plastic2``/``metal2``/``trans``/... are different shapes and are never
#: read as a reflectance here.
_REFLECTANCE_TYPES: frozenset[str] = frozenset({"plastic", "metal"})

#: The CIE photopic weights, normalised to sum to 1 -- reused from
#: ``units.LUX_WEIGHTS`` rather than typed again, so there is exactly one
#: photopic constant in the codebase.
_PHOTOPIC_WEIGHTS = LUX_WEIGHTS / LUX_WEIGHTS.sum()


#: The refractive index Radiance's glass transmissivity formula (and so
#: ``units.transmissivity``) assumes; the default when ``glass`` has 3 reals.
_GLASS_INDEX: float = 1.52  # source: Radiance glass default refractive index
_GLASS_INDEX_TOL: float = 1e-6


class MaterialError(ValueError):
    """A material cannot be rebuilt as the project describes it."""


#: Tilt classes that get their own copy of a glazing material (spec 5.4);
#: a vertical pane keeps the original modifier. Aliased from
#: ``glazing_class.NON_VERTICAL_TILT_CLASSES`` rather than retyped, so
#: there is exactly one definition of "every TILT_CLASSES entry but
#: vertical" in the codebase.
_COPY_CLASSES: tuple[str, ...] = NON_VERTICAL_TILT_CLASSES

#: LM-83-12 clause 2.2.9 item 1: at least 10 % dirt loss on sloped glazing
#: and 15 % on skylights. A factor above these is warned about, never
#: refused (spec D2). No vertical entry: a new vertical warning would
#: change existing manifests (spec 5.12).
_DIRT_WARN_ABOVE: dict[str, float] = {
    "sloped": 0.90,  # source: LM-83-12 clause 2.2.9 item 1, 10 % for sloped glazing
    "skylight": 0.85,  # source: LM-83-12 clause 2.2.9 item 1, 15 % for skylights
}

#: Radiance's normal.c adds an angle-dependent Fresnel term to a ``trans``
#: exactly when its roughness squared is at most FTINY and its specularity
#: at least FRESTHRESH; the normal-incidence Td and Rd below would then
#: misstate the glass, so that combination is refused.
_RADIANCE_FTINY: float = 1e-6  # source: Radiance FTINY
_RADIANCE_FRESTHRESH: float = 0.017999  # source: Radiance normal.c FRESTHRESH


@dataclasses.dataclass(frozen=True)
class GlazingRecord:
    modifier: str
    visible_transmittance: float
    dirt_factor: float
    net_transmittance: float  # VT * dirt, or VT when dirt is not applied
    transmissivity: float  # what the rebuilt glass primitive carries
    tinted: bool  # True when the three reals differed (R-P17)
    #: "vertical" for every original modifier; "sloped" or "skylight" for
    #: a copy (spec 5.4).
    tilt_class: str = "vertical"
    #: "specular" for glass; "diffuse" for a glazing_diffusing trans, whose
    #: visible_transmittance is Td, net_transmittance Td' and
    #: transmissivity the rebuilt sixth argument.
    transmission: str = "specular"


@dataclasses.dataclass(frozen=True)
class ClassedMaterials:
    """``rebuild_materials_by_class``'s result (spec 5.4 step 3)."""

    #: Rebuilt primitives in input order; each copy right after its original.
    prims: list[Primitive]
    #: One record per original glazing modifier (exactly as the all-vertical
    #: path writes them), then one per copy.
    records: list[GlazingRecord]
    warnings: tuple[str, ...]
    #: Exterior pane name -> the modifier that pane is written with.
    pane_modifier: types.MappingProxyType
    #: ``roles`` plus one entry per copy, carrying its original's role.
    effective_roles: types.MappingProxyType


class _GlassParts(typing.NamedTuple):
    r: float
    g: float
    b: float
    extra: tuple[float, ...]
    vt: float
    tinted: bool
    tn_weighted: float


class _TransParts(typing.NamedTuple):
    r: float
    g: float
    b: float
    a4: float  # specularity
    a5: float  # roughness
    a6: float  # transmissivity (diffuse share of what is transmitted)
    td: float
    rd: float
    tinted: bool


def copy_name(modifier: str, tilt_class: str) -> str:
    """The name of ``modifier``'s copy for a sloped or skylight pane."""
    if tilt_class not in _COPY_CLASSES:
        raise ValueError(
            f"tilt class {tilt_class!r} has no material copy; only "
            f"{', '.join(_COPY_CLASSES)} do"
        )
    return f"{modifier}__{tilt_class}"


def _role_of(prim: Primitive, roles) -> str | None:
    """A material's role, or ``None`` for a non-material or unlisted one."""
    if not prim.is_material:
        return None
    return roles.get(prim.name)


def _glass_reals(prim: Primitive) -> tuple[float, float, float, tuple[float, ...]]:
    reals = prim.fargs
    if len(reals) not in (3, 4):
        raise MaterialError(
            f"{prim.name}: a glass primitive must carry 3 real arguments "
            f"(r g b transmissivity) or 4 (plus refractive index), got {len(reals)}"
        )
    r, g, b = reals[0], reals[1], reals[2]
    # Each channel on its own, not only the weighted average: a negative
    # channel averaged with two clear ones would otherwise pass and reach
    # Radiance as a light-creating surface.
    upper = transmissivity(1.0)
    for label, value in zip("rgb", (r, g, b)):
        if not 0.0 <= value <= upper:
            raise MaterialError(
                f"{prim.name}: glass channel {label} is {value!r}; every channel "
                f"must be a transmissivity between 0 and {upper:.6g}"
            )
    if len(reals) == 4 and abs(reals[3] - _GLASS_INDEX) > _GLASS_INDEX_TOL:
        # units.transmissivity/transmittance_of are Radiance's formula for
        # n = 1.52 only; keeping another index while converting with that
        # formula would give the glass the wrong transmittance.
        raise MaterialError(
            f"glass {prim.name} has refractive index {reals[3]!r}; only "
            f"{_GLASS_INDEX:g} is supported for the VT conversion"
        )
    return r, g, b, tuple(reals[3:])


def _recover_visible_transmittance(prim: Primitive, r: float, g: float, b: float):
    """Return ``(vt, tinted, tn_weighted)`` recovered from a glass triple.

    Equal channels: the recovered transmissivity is just ``r``. Unequal
    channels are legitimate tinted glass (required shape 5), not a
    refusal: the recovered VT is the photopic-weighted average of the
    three, and the caller is told to warn.
    """
    tinted = not (r == g == b)
    tn_weighted = (
        float(_PHOTOPIC_WEIGHTS[0] * r + _PHOTOPIC_WEIGHTS[1] * g + _PHOTOPIC_WEIGHTS[2] * b)
        if tinted
        else r
    )
    try:
        vt = transmittance_of(tn_weighted)
    except ValueError as exc:
        raise MaterialError(
            f"{prim.name}: recovered transmissivity {tn_weighted!r} is not a value "
            f"transmittance_of() can invert: {exc}"
        ) from None
    return vt, tinted, tn_weighted


def _check_glazing_type(prim: Primitive, role: str) -> None:
    """Refuse a primitive type the glazing role cannot carry (spec 5.4)."""
    if prim.ptype in ("BSDF", "aBSDF"):
        raise MaterialError(
            f"{prim.name}: a {prim.ptype!r} glazing modifier is not supported; its "
            "arguments are not a transmissivity triple and there is nothing safe "
            "to do with it"
        )
    if role == "glazing_diffusing":
        if prim.ptype != "trans":
            raise MaterialError(
                f"{prim.name}: role 'glazing_diffusing' needs a Radiance 'trans' "
                f"primitive (diffusing glass with no see-through part), got "
                f"{prim.ptype!r}"
            )
        return
    if prim.ptype == "trans":
        if role == "glazing_interior":
            # Frosted interior glass: no spec reason refuses it (acceptance
            # spec 4.3); ``_trans_parts`` checks it as a diffuser.
            return
        raise MaterialError(
            f"{prim.name}: a 'trans' glazing modifier needs the role "
            f"'glazing_diffusing' (exterior) or 'glazing_interior' (interior), not {role!r}"
        )
    if prim.ptype != "glass":
        raise MaterialError(
            f"{prim.name}: only 'glass' is supported for glazing roles, got "
            f"primitive type {prim.ptype!r}"
        )


def _glass_parts(prim: Primitive) -> _GlassParts:
    r, g, b, extra = _glass_reals(prim)
    vt, tinted, tn_weighted = _recover_visible_transmittance(prim, r, g, b)
    return _GlassParts(r, g, b, extra, vt, tinted, tn_weighted)


def _trans_parts(prim: Primitive) -> _TransParts:
    """Check a diffusing ``trans`` and return its Td and Rd (spec 5.4).

    With ``c`` the photopic-weighted colour: ``Td = c (1 - A4) A6`` and
    ``Rd = c (1 - A4)(1 - A6)`` (Radiance normal.c, the diffuse transmitted
    and diffuse reflected parts at normal incidence).
    """
    reals = prim.fargs
    if len(reals) != 7:
        raise MaterialError(
            f"{prim.name}: a diffusing 'trans' must carry exactly 7 real "
            "arguments (r g b specularity roughness transmissivity "
            f"transmitted-specularity), got {len(reals)}"
        )
    r, g, b, a4, a5, a6, a7 = (float(x) for x in reals)
    if a7 != 0.0:
        raise MaterialError(
            f"{prim.name}: trans transmitted specularity (7th argument) is "
            f"{a7!r}; diffusing glazing must have 0 there, since any other "
            "value lets a see-through image pass"
        )
    for label, value in zip("rgb", (r, g, b)):
        if not 0.0 < value <= 1.0:
            raise MaterialError(
                f"{prim.name}: trans channel {label} is {value!r}; every "
                "channel must be above 0 and at most 1"
            )
    if not 0.0 <= a4 < 1.0:
        raise MaterialError(
            f"{prim.name}: trans specularity (4th argument) is {a4!r}; it must "
            "be at least 0 and below 1"
        )
    if not 0.0 < a6 <= 1.0:
        raise MaterialError(
            f"{prim.name}: trans transmissivity (6th argument) is {a6!r}; it "
            "must be above 0 and at most 1"
        )
    if a5 * a5 <= _RADIANCE_FTINY and a4 >= _RADIANCE_FRESTHRESH:
        raise MaterialError(
            f"{prim.name}: trans roughness {a5!r} with specularity {a4!r} gets "
            "Radiance's angle-dependent Fresnel term, so its normal-incidence "
            "transmittance would misstate the glass; give a roughness above "
            "0.001 or a specularity below 0.018"
        )
    c = float(_PHOTOPIC_WEIGHTS[0] * r + _PHOTOPIC_WEIGHTS[1] * g + _PHOTOPIC_WEIGHTS[2] * b)
    td = c * (1.0 - a4) * a6
    rd = c * (1.0 - a4) * (1.0 - a6)
    return _TransParts(r, g, b, a4, a5, a6, td, rd, not (r == g == b))


def _vertical_dirt(glazing: "Glazing") -> float | None:
    """The factor an original modifier is rebuilt with, or ``None`` when
    the mapping has no ``vertical`` key (the original is then written
    unchanged, spec 5.4)."""
    if glazing.vt_includes_dirt:
        return 1.0
    if not isinstance(glazing.dirt_factor, Mapping):
        return glazing.dirt_factor
    return glazing.dirt_factor.get("vertical")


def _class_dirt(glazing: "Glazing", tilt_class: str, pane: str) -> float:
    """The dirt factor for one tilt class present in the scene, or a
    refusal naming the missing key and one pane of that class."""
    if glazing.vt_includes_dirt:
        return 1.0
    if not isinstance(glazing.dirt_factor, Mapping):
        if tilt_class == "vertical":
            return glazing.dirt_factor
        raise MaterialError(
            "glazing.dirt_factor: a single number applies only when every "
            f"exterior pane is vertical, but pane {pane} is {tilt_class} "
            "class; give a mapping with the keys vertical, sloped and "
            "skylight that the scene needs"
        )
    try:
        return glazing.dirt_for(tilt_class)
    except KeyError:
        raise MaterialError(
            f"glazing.dirt_factor: no {tilt_class!r} value, but pane {pane} "
            f"is {tilt_class} class"
        ) from None


def _unchanged_record(prim: Primitive, parts) -> GlazingRecord:
    """The record of a primitive written unchanged: interior glazing, or
    an exterior original with no vertical factor."""
    if isinstance(parts, _TransParts):
        return GlazingRecord(
            modifier=prim.name,
            visible_transmittance=parts.td,
            dirt_factor=1.0,
            net_transmittance=parts.td,
            transmissivity=parts.a6,
            tinted=parts.tinted,
            transmission="diffuse",
        )
    return GlazingRecord(
        modifier=prim.name,
        visible_transmittance=parts.vt,
        dirt_factor=1.0,
        net_transmittance=parts.vt,
        transmissivity=parts.tn_weighted,
        tinted=parts.tinted,
    )


def _rebuilt(
    prim: Primitive, parts, dirt: float, tilt_class: str
) -> tuple[Primitive, GlazingRecord]:
    """``prim`` with ``dirt`` applied, and its record."""
    if isinstance(parts, _TransParts):
        # Dirt scales Td only; Rd is kept. Every channel times k and A6
        # replaced, so c' (1 - A4) A6' = Td' and c' (1 - A4)(1 - A6') = Rd.
        td_new = dirt * parts.td
        k = (td_new + parts.rd) / (parts.td + parts.rd)
        a6_new = td_new / (td_new + parts.rd)
        new = dataclasses.replace(
            prim,
            fargs=(parts.r * k, parts.g * k, parts.b * k, parts.a4, parts.a5, a6_new, 0.0),
        )
        return new, GlazingRecord(
            modifier=prim.name,
            visible_transmittance=parts.td,
            dirt_factor=dirt,
            net_transmittance=td_new,
            transmissivity=a6_new,
            tinted=parts.tinted,
            tilt_class=tilt_class,
            transmission="diffuse",
        )
    net_transmittance = parts.vt * dirt
    new_transmissivity = transmissivity(net_transmittance)
    if parts.tinted:
        scale = new_transmissivity / parts.tn_weighted
        new_reals = tuple(x * scale for x in (parts.r, parts.g, parts.b))
    else:
        new_reals = (new_transmissivity, new_transmissivity, new_transmissivity)
    new = dataclasses.replace(prim, fargs=new_reals + parts.extra)
    return new, GlazingRecord(
        modifier=prim.name,
        visible_transmittance=parts.vt,
        dirt_factor=dirt,
        net_transmittance=net_transmittance,
        transmissivity=new_transmissivity,
        tinted=parts.tinted,
        tilt_class=tilt_class,
    )


def _glazing_pass(prims, roles, glazing, pane_classes):
    """Spec 5.4 step 3 for glazing. Returns ``(rebuilt, records, warnings,
    pane_modifier, copy_roles)``. A pane absent from ``pane_classes``
    counts as vertical, so ``{}`` is the all-vertical path."""
    classes_by_modifier: dict[str, set[str]] = {}
    first_pane: dict[str, str] = {}
    first_pane_of: dict[tuple[str, str], str] = {}
    pane_modifier: dict[str, str] = {}
    for p in prims:
        if not p.is_surface or roles.get(p.modifier) not in EXTERIOR_GLAZING_ROLES:
            continue
        pane_class = pane_classes.get(p.name)
        tilt_class = "vertical" if pane_class is None else pane_class.tilt_class
        classes_by_modifier.setdefault(p.modifier, set()).add(tilt_class)
        first_pane.setdefault(tilt_class, p.name)
        first_pane_of.setdefault((p.modifier, tilt_class), p.name)
        pane_modifier[p.name] = (
            p.modifier if tilt_class == "vertical" else copy_name(p.modifier, tilt_class)
        )

    class_dirt = {
        tilt_class: _class_dirt(glazing, tilt_class, first_pane[tilt_class])
        for tilt_class in TILT_CLASSES
        if tilt_class in first_pane
    }
    vertical_dirt = _vertical_dirt(glazing)

    rebuilt: list[Primitive] = []
    records: list[GlazingRecord] = []
    copy_records: list[GlazingRecord] = []
    warnings: list[str] = []
    copy_roles: dict[str, str] = {}
    built: set[tuple[str, str]] = set()

    for prim in prims:
        role = _role_of(prim, roles)

        if role in _OPAQUE_ROLES:
            if prim.ptype in _GLAZING_TYPES:
                raise MaterialError(
                    f"{prim.name}: a {prim.ptype!r} primitive cannot carry the opaque "
                    f"role {role!r}; its real arguments are a transmissivity triple, "
                    "never a reflectance"
                )
            rebuilt.append(prim)
            continue

        if role not in _GLAZING_ROLES:
            rebuilt.append(prim)
            continue

        _check_glazing_type(prim, role)
        if prim.ptype == "trans":
            parts = _trans_parts(prim)
        else:
            parts = _glass_parts(prim)
            if parts.tinted:
                warnings.append(
                    f"{prim.name}: tinted glazing (unequal r/g/b transmissivity); the "
                    "recovered visible transmittance is the photopic-weighted average"
                )

        if role == "glazing_interior" or vertical_dirt is None:
            # Interior glazing: LM-83-12 clause 2.2.9 is about the envelope.
            # An exterior original with no vertical factor: written unchanged.
            rebuilt.append(prim)
            records.append(_unchanged_record(prim, parts))
        else:
            new, record = _rebuilt(prim, parts, vertical_dirt, "vertical")
            rebuilt.append(new)
            records.append(record)

        for tilt_class in _COPY_CLASSES:
            if tilt_class not in classes_by_modifier.get(prim.name, ()):
                continue
            name = copy_name(prim.name, tilt_class)
            new, record = _rebuilt(prim, parts, class_dirt[tilt_class], tilt_class)
            rebuilt.append(dataclasses.replace(new, name=name))
            copy_records.append(dataclasses.replace(record, modifier=name))
            copy_roles[name] = role
            built.add((prim.name, tilt_class))

    for (modifier, tilt_class), pane in first_pane_of.items():
        if tilt_class in _COPY_CLASSES and (modifier, tilt_class) not in built:
            raise MaterialError(
                f"{modifier}: pane {pane} is {tilt_class} class and needs a copy "
                f"of this material, but the scene defines no material {modifier!r}"
            )

    if not glazing.vt_includes_dirt:
        for tilt_class in _COPY_CLASSES:
            value = class_dirt.get(tilt_class)
            limit = _DIRT_WARN_ABOVE[tilt_class]
            if value is not None and value > limit:
                warnings.append(
                    f"glazing.dirt_factor: the {tilt_class} value {value:g} gives "
                    "less dirt loss than LM-83-12 clause 2.2.9's minimum for "
                    f"{tilt_class} glass (a factor of {limit:g} or lower)"
                )

    return rebuilt, records + copy_records, tuple(warnings), pane_modifier, copy_roles


def rebuild_glazing(
    prims: list[Primitive], roles: dict, glazing: "Glazing"
) -> tuple[list[Primitive], list[GlazingRecord], tuple[str, ...]]:
    """Apply the dirt factor and recover transmittance for every glazing role.

    The all-vertical path: every exterior pane counts as vertical class.
    Returns the rebuilt primitives (input order kept, rebuilt primitives in
    place of the originals), one ``GlazingRecord`` per glazing-role
    modifier seen, and any warnings (tinted glass, per required shape 5).
    """
    rebuilt, records, warnings, _, _ = _glazing_pass(prims, roles, glazing, {})
    return rebuilt, records, warnings


def _has_usable_reflectance(prim: Primitive) -> bool:
    """A plastic/metal primitive already carries a value worth keeping."""
    reals = prim.fargs
    return len(reals) >= 5 and max(reals[0], reals[1], reals[2]) > 0.0


def _reflectance_primitive(prim: Primitive, value: float) -> Primitive:
    if not 0.0 <= value <= 1.0:
        # config.py's reflectance.by_role is not range-checked there; this
        # is the backstop so no primitive ever ends up outside [0, 1].
        raise MaterialError(f"{prim.name}: reflectance {value!r} is outside [0, 1]")
    return dataclasses.replace(prim, fargs=(value, value, value, 0.0, 0.0))


def apply_reflectances(
    prims: list[Primitive], roles: dict, reflectance: "Reflectance"
) -> tuple[list[Primitive], tuple[str, ...]]:
    """Apply the project's reflectance policy to every opaque material.

    Only ``plastic``/``metal`` primitives are ever read as a reflectance
    (required shape 4). Under ``override``, an opaque role present in the
    scene with no reflectance value keeps its modelled value, and one
    warning lists every such role (M4). Input order is kept, rebuilt primitives in place of
    the originals.
    """
    if reflectance.policy == "keep_as_modelled":
        return list(prims), ()

    rebuilt: list[Primitive] = []
    warnings: list[str] = []
    unset_roles: set[str] = set()

    for prim in prims:
        role = _role_of(prim, roles)
        if role is None or role not in reflectance.by_role:
            if role in _OPAQUE_ROLES:
                unset_roles.add(role)
            rebuilt.append(prim)
            continue

        if prim.ptype not in _REFLECTANCE_TYPES:
            warnings.append(
                f"{prim.name}: role {role!r} has a reflectance value, but this is a "
                f"{prim.ptype!r} primitive, not plastic/metal; left unchanged"
            )
            rebuilt.append(prim)
            continue

        value = reflectance.by_role[role]
        if reflectance.policy == "override":
            rebuilt.append(_reflectance_primitive(prim, value))
        else:  # "fill_gaps"
            if _has_usable_reflectance(prim):
                rebuilt.append(prim)
            else:
                rebuilt.append(_reflectance_primitive(prim, value))

    if reflectance.policy == "override" and unset_roles:
        # M4: "override" reads as "every surface gets the project's value";
        # a role with no value keeps whatever the modeller painted, so say so.
        listed = ", ".join(repr(r) for r in sorted(unset_roles))
        warnings.append(
            f"reflectance: policy 'override' but no reflectance is given for "
            f"role(s) {listed} present in the scene; those surfaces keep "
            "their modelled reflectance"
        )

    return rebuilt, tuple(warnings)


def rebuild_materials_by_class(
    prims: list[Primitive],
    roles: Mapping[str, str],
    project,
    pane_classes: Mapping[str, "PaneClass"],
) -> ClassedMaterials:
    """Spec 5.4 step 3: glazing per tilt class, then the reflectance policy.

    ``pane_classes`` maps each exterior pane's name to its ``PaneClass``
    (Task 3); a pane absent from it counts as vertical. Returns the rebuilt
    primitives (each copy right after its original), the records (originals
    first, exactly as ``rebuild_materials`` writes them, then copies), the
    warnings, the modifier each exterior pane is to be written with, and the
    effective role map (``roles`` plus each copy with its original's role).
    """
    glazed, records, glazing_warnings, pane_modifier, copy_roles = _glazing_pass(
        prims, roles, project.glazing, pane_classes
    )
    final, reflectance_warnings = apply_reflectances(glazed, roles, project.reflectance)
    effective = dict(roles)
    effective.update(copy_roles)
    return ClassedMaterials(
        prims=final,
        records=records,
        warnings=glazing_warnings + reflectance_warnings,
        pane_modifier=types.MappingProxyType(pane_modifier),
        effective_roles=types.MappingProxyType(effective),
    )


def rebuild_materials(
    prims: list[Primitive], roles: dict, project
) -> tuple[list[Primitive], list[GlazingRecord], tuple[str, ...]]:
    """Rebuild glazing, then apply the reflectance policy, in one pass.

    The all-vertical path: every exterior pane counts as vertical class, so
    the output is exactly what this function wrote before tilt classes
    existed (spec 5.12).
    """
    classed = rebuild_materials_by_class(prims, roles, project, {})
    return classed.prims, classed.records, classed.warnings
