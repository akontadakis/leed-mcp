# leed_daylight/build/blinds.py
"""Build the operable blind polygon and material for each window group.

Spec 6.5 and spec deviation S4: one operable blind per exterior opening,
per window group -- never a single blind spanning a whole facade, because
Radiance needs one polygon per opening to attach a per-opening schedule
later. Exterior glazing only: ``glazing_interior`` never reaches a
``WindowGroup`` (``build.groups``), so it can never get a blind through
this module either -- there is nothing to special-case here. Fixed
shading is a different surface entirely and is never touched by anything
in this module: neither public function here reads or returns the scene's
primitive list, so there is nothing for either of them to modify.

**The polygon.** ``blind_polygons`` offsets each opening's own outline by
``BLIND_OFFSET_M`` along the INWARD direction, i.e. ``-outward_normals[k]``
(controller ruling) -- a pure translation, so the blind is exactly the
size and shape of the glass it covers, just set back into the room.
Offsetting outward would put the blind outside the building. A skylight
(a pane whose ``group.pane_classes`` entry has the ``skylight`` tilt class)
is the one exception: its blind is its outline moved straight DOWN by
``BLIND_OFFSET_M``, not along its tilted normal, so it stays inside the
vertical well under it (skylights design, section 5.8). A group built
without ``pane_classes`` behaves as before.

**The material.** ``blind_material`` converts the project's own optical
values -- ``Blinds.transmittance_diffuse`` (Td), ``Blinds.
transmittance_specular`` (Ts), ``Blinds.room_side_reflectance`` (Rd) -- into
the arguments a Radiance ``trans`` primitive needs (Radiance reference
manual, "Translucent materials"):

    modifier trans id
    0
    0
    7  red green blue  spec rough  trans tspec

With specular reflectance fixed at 0 (spec 6.5 does not model a specular
component on the room side of a fabric blind), the total light the shade
returns or passes on is ``c = Rd + Td + Ts``. That total is what a
Radiance ``trans`` primitive's colour channel actually represents; ``A6``
("trans") is the fraction of ``c`` that is transmitted rather than
reflected, and ``A7`` ("tspec") is the fraction of *that* transmission
which is specular rather than diffuse:

    c  = Rd + Td + Ts
    A6 = (Td + Ts) / c
    A7 = Ts / (Td + Ts)         (0 when Td + Ts == 0)

so the primitive's 7 real arguments are ``c c c 0 0 A6 A7``. This is
invertible exactly (used by this module's own tests to recover the
project's Rd/Td/Ts from the built primitive):

    Rd = c * (1 - A6)
    Td = c * A6 * (1 - A7)
    Ts = c * A6 * A7

``c == 0`` (an inert, fully absorbing shade) or ``c > 1`` (returning more
light than it received) is refused: neither is a shade a real project
would specify. There is NO built-in default for Rd, Td or Ts anywhere in
this module (Global Constraint 6): the LM-83-12 clause 2.2.7 values are a
declared substitution the project states in ``blinds:`` and the report
reproduces, never a number this module assumes on the project's behalf.
``blind_material`` therefore takes no default argument at all -- calling
it with nothing is a bare ``TypeError``, by construction, not a case this
module has to detect.

Every numeric literal here besides the formula's own structure is invented
and round (phase3-global.md #5): ``BLIND_OFFSET_M`` and the edge-length
tolerance are plain, round numbers, not read off a real project.
"""
from __future__ import annotations

import math
import typing

import numpy as np

from .. import geometry
from ..scene.primitives import Primitive

if typing.TYPE_CHECKING:
    from ..project.config import Blinds
    from .groups import WindowGroup

#: Spec 6.5: the blind sits this far from the glass, offset inward into
#: the room.
BLIND_OFFSET_M: float = 0.05

#: An opening edge shorter than this (a duplicated consecutive vertex)
#: would make the blind's own outline degenerate along that side; refused
#: rather than silently produced (same tolerance and reasoning as
#: ``build.reveals``).
_EDGE_LENGTH_TOLERANCE_M: float = 1e-9

#: Sentinel distinguishing "field missing entirely" from "field present
#: but not a usable number", so a malformed ``blinds`` object gets one
#: named refusal either way, never a raw ``AttributeError``.
_MISSING = object()


class BlindError(ValueError):
    """A blind polygon or material cannot be built from these inputs."""


def _validated_edges(pts: np.ndarray, name: str) -> None:
    count = len(pts)
    for i in range(count):
        j = (i + 1) % count
        edge_length = float(np.linalg.norm(pts[j] - pts[i]))
        if edge_length < _EDGE_LENGTH_TOLERANCE_M:
            raise BlindError(
                f"{name}: edge {i} (vertex {i} to vertex {j}) has length "
                f"{edge_length:.3g} m, shorter than the "
                f"{_EDGE_LENGTH_TOLERANCE_M:g} m tolerance; a blind on it "
                "would have a degenerate side"
            )


def _validated_opening_points(p, index: int) -> np.ndarray:
    """Return ``p``'s polygon vertices, or raise ``BlindError`` naming it."""
    name = getattr(p, "name", None) or f"openings[{index}]"
    try:
        raw = p.points
    except ValueError as exc:
        raise BlindError(f"{name}: {exc}") from None
    except AttributeError as exc:
        raise BlindError(
            f"{name}: expected an opening with a .points property, got "
            f"{p!r}: {exc}"
        ) from None
    if raw is None:
        raise BlindError(
            f"{name}: opening has no polygon points (not a polygon primitive)"
        )
    try:
        pts = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BlindError(
            f"{name}: opening points are not a valid array of numbers: {exc}"
        ) from None
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        raise BlindError(
            f"{name}: expected opening points of shape (n, 3) with n >= 3, "
            f"got shape {pts.shape}"
        )
    if not np.all(np.isfinite(pts)):
        raise BlindError(f"{name}: opening polygon has non-finite coordinates")
    _validated_edges(pts, name)
    try:
        area = geometry.area(pts)
    except ValueError as exc:
        raise BlindError(f"{name}: {exc}") from None
    if area < geometry.DEGENERATE:
        raise BlindError(f"{name}: opening polygon is degenerate (zero area)")
    return pts


def _validated_normal(n_raw, index: int) -> np.ndarray:
    try:
        n = np.asarray(n_raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BlindError(
            f"outward_normals[{index}]: not a valid vector of numbers: {exc}"
        ) from None
    if n.shape != (3,):
        raise BlindError(
            f"outward_normals[{index}]: expected a 3-vector, got {n_raw!r} "
            f"with shape {n.shape}"
        )
    if not np.all(np.isfinite(n)):
        raise BlindError(
            f"outward_normals[{index}]: expected finite components, got {n_raw!r}"
        )
    length = float(np.linalg.norm(n))
    if length < geometry.DEGENERATE:
        raise BlindError(
            f"outward_normals[{index}]: expected a nonzero vector, got "
            f"{n_raw!r} (zero length)"
        )
    return n / length


def blind_polygons(group: "WindowGroup") -> tuple[np.ndarray, ...]:
    """One (n, 3) polygon per opening in ``group``, offset inward.

    Each opening's own outline is translated by ``BLIND_OFFSET_M`` along
    ``-outward_normals[k]`` (controller ruling): a pure translation, so
    the blind is parallel to its opening, matches its area exactly, and
    keeps the opening's own vertex order. ``group.openings`` and
    ``group.outward_normals`` must be the same length (they are aligned
    1:1 by construction in ``build.groups``); a mismatch, a malformed
    opening (not enough vertices, a zero-length edge, a degenerate or
    non-finite polygon), or a malformed normal (wrong shape, zero length,
    non-finite or non-numeric) is a :class:`BlindError` naming the
    opening or the normal's index.

    An opening whose ``group.pane_classes`` entry has the ``skylight`` tilt
    class is translated by ``(0, 0, -BLIND_OFFSET_M)`` instead (spec 5.8).
    ``pane_classes`` empty means no opening is a skylight; otherwise it
    must be aligned 1:1 with ``openings`` too.

    A group with no openings returns an empty tuple.
    """
    openings = group.openings
    normals = group.outward_normals
    group_id = getattr(group, "group_id", "?")
    if len(openings) != len(normals):
        raise BlindError(
            f"group {group_id!r}: {len(openings)} opening(s) but "
            f"{len(normals)} outward_normals; they must be aligned 1:1"
        )
    pane_classes = tuple(getattr(group, "pane_classes", ()))
    if pane_classes and len(pane_classes) != len(openings):
        raise BlindError(
            f"group {group_id!r}: {len(openings)} opening(s) but "
            f"{len(pane_classes)} pane_classes; they must be aligned 1:1"
        )

    blinds = []
    for k in range(len(openings)):
        pts = _validated_opening_points(openings[k], k)
        n = _validated_normal(normals[k], k)
        if pane_classes and pane_classes[k].tilt_class == "skylight":
            blinds.append(pts - np.array([0.0, 0.0, BLIND_OFFSET_M]))
        else:
            blinds.append(pts - BLIND_OFFSET_M * n)
    return tuple(blinds)


def _as_finite_float(value, field: str) -> float:
    if value is _MISSING:
        raise BlindError(
            f"blinds.{field}: expected a Blinds object with a {field!r} "
            "field, none found"
        )
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise BlindError(
            f"blinds.{field}: expected a real number, got {value!r}: {exc}"
        ) from None
    if not math.isfinite(v):
        raise BlindError(
            f"blinds.{field}: expected a finite number, got {value!r}"
        )
    return v


def blind_material(blinds: "Blinds") -> Primitive:
    """Build the ``trans`` primitive for the project's own blind product.

    ``blinds`` must carry ``room_side_reflectance`` (Rd),
    ``transmittance_diffuse`` (Td) and ``transmittance_specular`` (Ts) --
    see the module docstring for the conversion and its source. There is
    no default: calling this with no argument is a bare ``TypeError``, and
    a ``blinds`` missing one of the three fields, or carrying a
    non-numeric or non-finite value, is a named :class:`BlindError`
    instead of a raw ``AttributeError``/``TypeError`` from deeper inside.

    ``Rd + Td + Ts`` (``c``) must be in ``(0, 1]``: zero would be an
    inert, fully absorbing shade (nothing to model as a `trans` material),
    and more than 1 would mean the shade returns more light than it
    receives. Either is refused, naming all three values.
    """
    rd = _as_finite_float(
        getattr(blinds, "room_side_reflectance", _MISSING), "room_side_reflectance"
    )
    td = _as_finite_float(
        getattr(blinds, "transmittance_diffuse", _MISSING), "transmittance_diffuse"
    )
    ts = _as_finite_float(
        getattr(blinds, "transmittance_specular", _MISSING), "transmittance_specular"
    )

    c = rd + td + ts
    if c == 0.0 or c > 1.0:
        raise BlindError(
            "blinds: room_side_reflectance + transmittance_diffuse + "
            "transmittance_specular must be in (0, 1] -- a shade cannot "
            f"return more light than it receives, or none at all; got "
            f"room_side_reflectance={rd!r}, transmittance_diffuse={td!r}, "
            f"transmittance_specular={ts!r} (sum {c!r})"
        )

    diffuse_and_specular = td + ts
    a6 = diffuse_and_specular / c
    a7 = ts / diffuse_and_specular if diffuse_and_specular != 0.0 else 0.0

    return Primitive(
        modifier="void",
        ptype="trans",
        name="leed_blind",
        sargs=(),
        iargs=(),
        fargs=(c, c, c, 0.0, 0.0, a6, a7),
        source="build.blinds",
    )
