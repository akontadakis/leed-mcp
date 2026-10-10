# leed_daylight/deliver/atlas/layers.py
"""The result layers of the atlas (atlas spec 8).

One layer per row of the spec 8 table, only for the options whose results
are current for the case (``case.options``), in table order. A number
layer carries the colour scale the Word report uses for the same column
(``word/option1.py``, ``word/option2.py``), its limits and marks read from
the credit rules and the results file. A pass/fail layer carries
``fail_when``: the column value (true or false) drawn in the fail colour.
The atlas computes no metric; it only colours what ``points.csv`` holds.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..data import CaseData, DeliverError
from ..figures.spec import FAIL_RGB, PASS_RGB, ColourScale


@dataclasses.dataclass(frozen=True)
class Layer:
    key: str  # "<option>.<column>"
    option: str
    title: str
    column: str  # the points.csv column
    scale: ColourScale | None  # number layers
    fail_when: bool | None  # pass/fail layers: the value drawn as a fail
    legend: tuple[str, str] = ("passes", "fails")  # pass/fail layers: the two legend labels

    def __post_init__(self):
        if (self.scale is None) == (self.fail_when is None):
            raise DeliverError(f"layer {self.key}: give a colour scale or a fail value, not both or neither")


def _option1(case: CaseData) -> tuple[Layer, ...]:
    rules = case.rules["option1"]
    sda, ase = rules["sda"], rules["ase"]
    hours = float(ase["hours"])
    return (
        Layer("option1.da_pct", "option1", "Daylight autonomy", "da_pct",
              ColourScale(0.0, 100.0, f"% of occupied hours at {sda['lux']} lux or more",
                          marks=(100.0 * float(sda["fraction"]),)), None),
        Layer("option1.sda_pass", "option1", "sDA pass", "sda_pass", None, False),
        Layer("option1.ase_hours", "option1", "ASE hours", "ase_hours",
              ColourScale(0.0, 2.0 * hours, f"occupied hours at {ase['lux']} lux or more, blinds open",
                          marks=(hours,)), None),
        Layer("option1.ase_exceeds", "option1", "ASE over limit", "ase_exceeds", None, True,
              ("within the limit", "over the limit")),
    )


def _option2(case: CaseData) -> tuple[Layer, ...]:
    results = case.options["option2"].results
    settings = results["settings"]
    lux_min, lux_max = float(settings["lux_min"]), float(settings["lux_max"])
    scale = ColourScale(max(1.0, lux_min / 10.0), lux_max * 3.0, "lux (log scale)", marks=(lux_min, lux_max),
                        log=True)
    skies = tuple(
        Layer(f"option2.lux_{sky['label']}", "option2", f"Illuminance, {sky['label']}", f"lux_{sky['label']}",
              scale, None)
        for sky in results["skies"]
    )
    return skies + (
        Layer("option2.passes", "option2", "Option 2 pass", "passes", None, False,
              ("in the band", "outside the band")),
    )


def _views(case: CaseData) -> tuple[Layer, ...]:
    return tuple(
        Layer(f"views.{column}", "views", title, column, None, False, ("view", "no view"))
        for column, title in (("type2", "Type 2 view"), ("type3", "Type 3 view"), ("quality", "Quality View"))
    )


_BUILDERS = (("option1", _option1), ("option2", _option2), ("views", _views))


def result_layers(case: CaseData) -> tuple[Layer, ...]:
    """Every layer of every current option of ``case``, in spec 8 order."""
    return tuple(layer for option, build in _BUILDERS if option in case.options for layer in build(case))


def layer_rgb(layer: Layer, values: np.ndarray) -> np.ndarray:
    """(n, 3) uint8 colours of ``values``: the ramp colour for a number
    layer, the pass or fail colour for a pass/fail layer (a value that is
    not 0 counts as true)."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if layer.scale is not None:
        rows = [layer.scale.rgb(v) for v in values]
    else:
        fails = (values != 0.0) == layer.fail_when
        rows = [FAIL_RGB if fail else PASS_RGB for fail in fails]
    return np.array(rows, dtype=np.uint8).reshape(-1, 3)
