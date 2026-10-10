"""One Radiance primitive.

The format is:

    modifier type identifier
    n_string_args  arg ...
    n_integer_args arg ...
    n_real_args    arg ...
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Primitive types that describe a surface rather than a material or a modifier.
SURFACE_TYPES = frozenset(
    {"polygon", "ring", "sphere", "bubble", "cone", "cup", "cylinder", "tube", "mesh"}
)

#: Primitive types that define a material.
MATERIAL_TYPES = frozenset(
    {
        "plastic", "metal", "trans", "glass", "dielectric", "interface",
        "mirror", "light", "glow", "spotlight", "illum", "BSDF", "aBSDF",
        "plastic2", "metal2", "trans2", "brightfunc", "colorfunc", "mist",
    }
)

#: Every other Radiance modifier type: the materials not listed above, and
#: the patterns, textures and mixtures.  Known, so validation does not
#: refuse them as unknown; not a surface and not one of `MATERIAL_TYPES`.
OTHER_MODIFIER_TYPES = frozenset(
    {
        "plasfunc", "metfunc", "transfunc", "BRTDfunc", "plasdata", "metdata",
        "transdata", "antimatter", "prism1", "prism2", "ashik2", "WGMDfunc",
        "texfunc", "texdata",
        "colordata", "brightdata", "colorpict", "colortext", "brighttext",
        "specfunc", "specdata", "specpict", "spectrum", "specfile",
        "mixfunc", "mixdata", "mixpict", "mixtext",
    }
)


@dataclass(frozen=True)
class Primitive:
    modifier: str
    ptype: str
    name: str
    sargs: tuple[str, ...]
    iargs: tuple[int, ...]
    fargs: tuple[float, ...]
    source: str

    @property
    def is_surface(self) -> bool:
        return self.ptype in SURFACE_TYPES

    @property
    def is_material(self) -> bool:
        return self.ptype in MATERIAL_TYPES

    @property
    def points(self) -> np.ndarray | None:
        """Vertices of a polygon, shape (n, 3).  None for anything else."""
        if self.ptype != "polygon":
            return None
        if len(self.fargs) % 3:
            raise ValueError(
                f"{self.name}: polygon has {len(self.fargs)} reals, not a multiple of 3"
            )
        return np.asarray(self.fargs, dtype=np.float64).reshape(-1, 3)
