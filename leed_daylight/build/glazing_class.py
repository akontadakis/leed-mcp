# leed_daylight/build/glazing_class.py
"""Classify every exterior glazing pane by tilt (skylights spec 5.1).

LM-83-12 clause 2.2.9 treats glass by its slope: vertical windows, sloped
glazing "between 85 to 20 degrees of vertical", and skylights "at less
than a 20 degree slope". This module reads that as a tilt measured from
flat -- 0 is a flat roof light, 90 a wall window -- the one reading with
no gap and no overlap between the three classes:

* ``vertical``: tilt 85 to 90 degrees;
* ``sloped``: tilt 20 to under 85 degrees;
* ``skylight``: tilt under 20 degrees.

The tilt is rounded to 6 decimals before it is compared, so a pane drawn
at exactly 20 or 85 degrees lands in the class a person would expect, not
in the one a last-bit floating-point error picks.

The kind -- what a pane's window group is called, whether it gets a blind,
whether it is vision glazing -- follows from the class, the position
(``perimeter`` facade glass or ``roof`` glass, decided by room matching in
``build.groups``) and whether the pane's role is ``glazing_diffusing``.
Geometry (which side is outside, the reveal or well, where the blind
hangs) is keyed on the class and position only, never on the kind.

Pure: no scene, no room, no file. Every numeric literal is a threshold the
standard states, not a value from a real project.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: LM-83-12 2.2.9: glass at 85 degrees from flat or steeper is vertical.
VERTICAL_MIN_TILT_DEG: float = 85.0
#: LM-83-12 2.2.9: glass under 20 degrees from flat is a skylight.
SKYLIGHT_MAX_TILT_DEG: float = 20.0

TILT_CLASSES: tuple[str, ...] = ("vertical", "sloped", "skylight")
KINDS: tuple[str, ...] = ("window", "sloped", "monitor", "skylight", "diffusing")
POSITIONS: tuple[str, ...] = ("perimeter", "roof")

#: The two classes that need their own dirt key, `frame_share` key and
#: material copy (spec D2, D3, 5.4, 5.5) -- every ``TILT_CLASSES`` entry
#: other than ``vertical``. Defined once here, derived from ``TILT_CLASSES``
#: rather than retyped, so later tasks (materials, frames, blinds) import
#: this instead of each spelling out the pair on its own.
NON_VERTICAL_TILT_CLASSES: tuple[str, ...] = TILT_CLASSES[1:]

#: Decimals the tilt is rounded to before it is compared with a threshold.
_TILT_DECIMALS = 6


@dataclass(frozen=True)
class PaneClass:
    #: Degrees from flat, 0 to 90, rounded to 6 decimals.
    tilt_deg: float
    #: One of ``TILT_CLASSES``.
    tilt_class: str
    #: One of ``POSITIONS``.
    position: str
    #: True when the pane's role is ``glazing_diffusing``.
    diffusing: bool
    #: One of ``KINDS``.
    kind: str


def tilt_deg(normal) -> float:
    """Degrees between the pane's plane and a flat roof: 0 flat, 90 vertical.

    ``degrees(acos(|n_z|))`` of the unit normal, rounded to 6 decimals.
    The normal's sign does not matter: a pane drawn facing down has the
    same tilt as the same pane drawn facing up. A normal that is not a
    finite, non-zero 3-vector is a ``ValueError``.
    """
    try:
        n = np.asarray(normal, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"normal: not a vector of numbers: {exc}") from None
    if n.shape != (3,) or not np.all(np.isfinite(n)):
        raise ValueError(f"normal: expected a finite 3-vector, got {normal!r}")
    length = float(np.linalg.norm(n))
    if length == 0.0:
        raise ValueError("normal: a zero vector has no direction")
    nz = min(1.0, abs(float(n[2])) / length)
    return round(math.degrees(math.acos(nz)), _TILT_DECIMALS)


def tilt_class_of(tilt: float) -> str:
    """``vertical`` from 85 degrees, ``sloped`` from 20, else ``skylight``."""
    if not 0.0 <= tilt <= 90.0:
        raise ValueError(f"tilt {tilt!r}: expected degrees from flat, 0 to 90")
    if tilt >= VERTICAL_MIN_TILT_DEG:
        return "vertical"
    if tilt >= SKYLIGHT_MAX_TILT_DEG:
        return "sloped"
    return "skylight"


def kind_of(tilt_class: str, position: str, diffusing: bool) -> str:
    """The pane's kind: ``diffusing`` whenever the flag is set; else
    ``skylight`` or ``sloped`` by class; else ``monitor`` for vertical
    glass on the roof; else ``window``."""
    if tilt_class not in TILT_CLASSES:
        raise ValueError(f"tilt class {tilt_class!r}: expected one of {', '.join(TILT_CLASSES)}")
    if position not in POSITIONS:
        raise ValueError(f"position {position!r}: expected one of {', '.join(POSITIONS)}")
    if diffusing:
        return "diffusing"
    if tilt_class in ("skylight", "sloped"):
        return tilt_class
    if position == "roof":
        return "monitor"
    return "window"


def classify(normal, position: str, diffusing: bool) -> PaneClass:
    """The full class of one pane from its normal as drawn."""
    tilt = tilt_deg(normal)
    tilt_class = tilt_class_of(tilt)
    return PaneClass(
        tilt_deg=tilt,
        tilt_class=tilt_class,
        position=position,
        diffusing=bool(diffusing),
        kind=kind_of(tilt_class, position, diffusing),
    )
