"""Unit and photometric conversions.

Three conversions decide whether every downstream number is right:

1. Scene length to metres.  Declared by the project, never guessed.
2. Visible transmittance to Radiance transmissivity.  The `glass` primitive
   takes transmissivity; using transmittance makes the glass about 8 % too dark.
3. Radiance RGB to lux.  Performed in the engine layer and nowhere else.
"""
from __future__ import annotations

import math

import numpy as np

LENGTH_TO_M: dict[str, float] = {
    "m": 1.0,
    "mm": 0.001,
    "cm": 0.01,
    "ft": 0.3048,  # source: international foot, exactly 0.3048 m
    "in": 0.0254,  # source: international inch, exactly 0.0254 m
}

#: 179 lm/W times the CIE photopic weights (0.265, 0.670, 0.065).
LUX_WEIGHTS = np.array([47.4, 119.9, 11.6], dtype=np.float64)


def to_metres(value, units: str):
    """Convert a length, or an array of lengths, into metres."""
    try:
        factor = LENGTH_TO_M[units]
    except KeyError:
        raise ValueError(
            f"unknown length unit {units!r}; expected one of {sorted(LENGTH_TO_M)}"
        ) from None
    return value * factor


def net_transmittance(vt: float, dirt_factor: float) -> float:
    """Visible transmittance after the LM-83-12 clause 2.2.9 dirt factor."""
    return vt * dirt_factor


def transmissivity(tn: float) -> float:
    """Radiance glass transmissivity for a normal-incidence transmittance."""
    if not 0.0 <= tn <= 1.0:
        raise ValueError(f"transmittance must be between 0 and 1, got {tn}")
    if tn == 0.0:
        return 0.0
    return (
        math.sqrt(0.8402528435 + 0.0072522239 * tn * tn) - 0.9166530661  # source: Radiance glass transmissivity formula
    ) / (0.0036261119 * tn)  # source: Radiance glass transmissivity formula


def transmittance_of(tn: float) -> float:
    """Visible transmittance for a Radiance glass transmissivity.

    The inverse of ``transmissivity()``. That function is monotonic on
    ``[0, 1]``, so the inverse is found by bisection there rather than by
    inverting the formula algebraically -- exact to machine tolerance, and
    needs no new algebra to get wrong.

    Valid domain is ``0 <= tn <= transmissivity(1.0)``, since no visible
    transmittance in ``[0, 1]`` produces a transmissivity above that. A
    value outside the domain is refused, naming the value and the domain.
    """
    upper = transmissivity(1.0)
    if not 0.0 <= tn <= upper:
        raise ValueError(
            f"transmissivity must be between 0 and {upper} to invert, got {tn}"
        )
    if tn == 0.0:
        return 0.0
    lo, hi = 0.0, 1.0
    while hi - lo > 1e-13:
        mid = (lo + hi) / 2.0
        if transmissivity(mid) < tn:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def rgb_to_lux(rgb: np.ndarray) -> np.ndarray:
    """Convert Radiance RGB radiance to illuminance in lux.

    The last axis must be the colour axis and is removed from the result.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.shape[-1] != 3:
        raise ValueError(f"last axis must be 3 (RGB), got shape {rgb.shape}")
    return rgb @ LUX_WEIGHTS
