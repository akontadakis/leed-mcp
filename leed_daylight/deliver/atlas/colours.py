# leed_daylight/deliver/atlas/colours.py
"""The one colour table for kinds of geometry (atlas spec 6).

Every surface of the prepared model is drawn as exactly one kind; the 3D
view of the page and the Blender materials both read ``KIND_RGBA``. The
alpha is the kind's opacity in 3D (glass nearly clear, frames solid).
Result colours are not here: they come from ``deliver/figures/spec.py``,
as in the Word reports. Every value is invented and round.
"""
from __future__ import annotations

KINDS: tuple[str, ...] = (
    "wall", "floor", "ceiling", "fixed_shading", "furniture",
    "glass_window", "glass_curtain_wall", "glass_interior", "glass_diffusing", "glass_roof",
    "frame", "reveal", "skylight_well", "blind", "context", "ground",
)

#: (red, green, blue, alpha), 0 to 255, one per kind, in ``KINDS`` order.
KIND_RGBA: dict[str, tuple[int, int, int, int]] = {
    "wall": (180, 176, 170, 40),
    "floor": (150, 145, 140, 70),
    "ceiling": (200, 200, 195, 40),
    "fixed_shading": (200, 140, 60, 120),
    "furniture": (140, 110, 80, 120),
    "glass_window": (90, 160, 210, 70),
    "glass_curtain_wall": (50, 120, 200, 70),
    "glass_interior": (40, 150, 90, 90),
    "glass_diffusing": (230, 225, 190, 140),
    "glass_roof": (120, 90, 200, 90),
    "frame": (60, 60, 60, 220),
    "reveal": (120, 120, 120, 140),
    "skylight_well": (170, 110, 170, 120),
    "blind": (240, 200, 60, 100),
    "context": (170, 110, 100, 40),
    "ground": (110, 140, 90, 40),
}

#: Kinds hidden in 3D until switched on.
DEFAULT_OFF: frozenset[str] = frozenset({"ceiling", "blind", "context", "ground"})

#: The fill of an included room on an atlas plan.
ROOM_FILL_RGB: tuple[int, int, int] = (236, 239, 243)
