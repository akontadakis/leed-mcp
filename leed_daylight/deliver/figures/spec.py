# leed_daylight/deliver/figures/spec.py
"""Backend-neutral figure descriptions (reports spec 4).

A figure is described once, as a ``PlanSpec`` or a ``BarSpec``, and turned
into a display list (``Canvas``) of four primitive kinds: ``Rect``, ``Poly``,
``Line`` and ``Text``, in pixel coordinates with y pointing down. Every
layout decision (scale, legend, north arrow, scale bar, labels) is made here,
so the three painters in this package only paint and always agree on where a
point lands.

A plan is drawn with the model's +Y up. ``north_rotation_deg`` is the angle
of true north from the model's +Y axis, counter-clockwise positive (the sense
in which the daylight engine turns the sky), so true north is the model
direction (-sin r, cos r).

Final review I4 (Ruling FR-I4): every size in pixels below (text, title and
legend bands, the north arrow, line widths) is stated for a figure
``_REF_PX`` (800) pixels wide and multiplied by ``width_px / _REF_PX``, so a
figure keeps its proportions at any resolution. Printed at
``PRINT_WIDTH_CM``, the smallest text (12 px at the reference width) is
about 7.4 pt. A plan is never taller than ``MAX_PRINT_HEIGHT_CM`` would
allow at that width: a long floor is drawn at a smaller scale instead.

Atlas spec 8.2: the atlas plans add ``Dot`` (a model-check grid point),
``Image`` (a drawing under the plan) and a hatch on ``Poly`` (an excluded
room), emitted by the ``PlanSpec`` fields ``point_style``, ``underlay``,
``excluded_outlines`` and ``room_fill``. With those fields at their
defaults a figure's display list is exactly what it was before them.

Merge-split spec 7: a ``Line`` may be dashed (``dash``), and ``PlanSpec``
draws each of its ``split_lines`` (the edge two halves of a split room
share) as dashed lines in the room colour, right after the room outlines.
With no split line a plan's display list is exactly what it was before.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np

RGB = tuple[int, int, int]


class FigureError(ValueError):
    """A figure cannot be drawn from this description."""


#: The six-stop blue-to-red ramp, ported from the author's earlier
#: work (PORTED.md): blue, cyan, green, yellow, orange, red.
_STOPS: tuple[tuple[float, RGB], ...] = (
    (0.0, (25, 50, 130)), (0.2, (30, 130, 200)), (0.4, (60, 180, 150)),
    (0.6, (200, 210, 70)), (0.8, (235, 150, 45)), (1.0, (190, 35, 40)),
)
PASS_RGB: RGB = (46, 125, 50)
FAIL_RGB: RGB = (198, 40, 40)
NEUTRAL_RGB: RGB = (150, 150, 150)
ROOM_RGB: RGB = (90, 90, 90)
SEGMENT_RGB: RGB = (110, 50, 160)
BAR_RGB: RGB = (30, 130, 200)
MARK_RGB: RGB = (0, 0, 0)
TEXT_RGB: RGB = (0, 0, 0)
WHITE: RGB = (255, 255, 255)

#: Every figure is placed in the reports at this width (word/kit.USABLE)...
PRINT_WIDTH_CM = 17.4
#: ...and never taller than this: A4's usable height (26.1 cm) less room
#: for the caption below the figure (final review I4, Ruling FR-I4).
MAX_PRINT_HEIGHT_CM = 22.0
#: The width every pixel size in this module is stated for (see above).
_REF_PX = 800.0
TITLE_PX = 24
LEGEND_PX = 48
MARGIN_M = 1.0
_ARROW_PX = 11.0
_GRADIENT_STEPS = 64
#: How a plan draws its grid points: a coloured cell, or a small dot.
POINT_STYLES = ("cell", "dot")
#: A dot's diameter as a share of the grid spacing.
_DOT_SHARE = 0.4
#: The image types an underlay may carry.
UNDERLAY_MIMES = ("image/png", "image/jpeg")


def ramp(t: float) -> RGB:
    """The ramp colour at ``t`` in [0, 1]; values outside are clamped."""
    t = min(max(float(t), 0.0), 1.0)
    for (a, ca), (b, cb) in zip(_STOPS, _STOPS[1:]):
        if a <= t <= b:
            f = (t - a) / (b - a)
            return tuple(int(round(ca[k] + f * (cb[k] - ca[k]))) for k in range(3))
    return _STOPS[-1][1]


@dataclasses.dataclass(frozen=True)
class ColourScale:
    vmin: float
    vmax: float
    unit: str
    marks: tuple[float, ...] = ()
    log: bool = False

    def __post_init__(self):
        if not (math.isfinite(self.vmin) and math.isfinite(self.vmax)):
            raise FigureError(f"colour scale: vmin and vmax must be finite, got {self.vmin}, {self.vmax}")
        if self.vmax <= self.vmin:
            raise FigureError(f"colour scale: vmax ({self.vmax}) must be above vmin ({self.vmin})")
        if self.log and self.vmin <= 0.0:
            raise FigureError(f"colour scale: a log scale needs vmin above 0, got {self.vmin}")

    def fraction(self, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise FigureError(f"colour scale ({self.unit}): {value} is not a finite number")
        value = min(max(value, self.vmin), self.vmax)
        if self.log:
            lo, hi = math.log10(self.vmin), math.log10(self.vmax)
            return (math.log10(value) - lo) / (hi - lo)
        return (value - self.vmin) / (self.vmax - self.vmin)

    def rgb(self, value: float) -> RGB:
        return ramp(self.fraction(value))

    def ticks(self) -> tuple[float, ...]:
        if self.log:
            lo, hi = math.ceil(math.log10(self.vmin)), math.floor(math.log10(self.vmax))
            inner = [10.0 ** k for k in range(lo, hi + 1)]
            return tuple(sorted({self.vmin, *inner, self.vmax}))
        step = (self.vmax - self.vmin) / 4.0
        return tuple(self.vmin + i * step for i in range(5))


@dataclasses.dataclass(frozen=True)
class PlanPolygon:
    """A closed plan outline in metres with a label at its centre; dashed
    for roof glass (skylights spec 7)."""
    points: tuple[tuple[float, float], ...]
    dashed: bool
    label: str

    def __post_init__(self):
        points = tuple((float(x), float(y)) for x, y in self.points)
        if len(points) < 2:
            raise FigureError(f"polygon {self.label!r}: needs at least 2 points, got {len(points)}")
        object.__setattr__(self, "points", points)


@dataclasses.dataclass(frozen=True)
class Underlay:
    """A drawing under a plan (atlas spec 8.1): PNG or JPEG bytes, their size
    in pixels, and the similarity that takes a drawing pixel (u, v; v runs
    down) to model metres: ``x = a*u + c*v + e``, ``y = b*u + d*v + f`` for
    ``model_from_pixel = (a, b, c, d, e, f)``."""
    data: bytes
    mime: str
    width: int
    height: int
    model_from_pixel: tuple[float, float, float, float, float, float]
    opacity: float

    def __post_init__(self):
        if self.mime not in UNDERLAY_MIMES:
            raise FigureError(f"underlay: the image must be image/png or image/jpeg, got {self.mime!r}")
        for name in ("width", "height"):
            if not getattr(self, name) > 0:
                raise FigureError(f"underlay: {name} must be above 0 pixels, got {getattr(self, name)}")
        if not 0.0 <= self.opacity <= 1.0:
            raise FigureError(f"underlay: opacity must be 0 to 1, got {self.opacity}")
        matrix = tuple(self.model_from_pixel)
        if len(matrix) != 6 or not all(math.isfinite(float(v)) for v in matrix):
            raise FigureError(f"underlay: model_from_pixel must be six numbers, got {matrix}")
        object.__setattr__(self, "model_from_pixel", tuple(float(v) for v in matrix))


@dataclasses.dataclass(frozen=True, eq=False)
class PlanSpec:
    title: str
    outlines: tuple[np.ndarray, ...]  # each (n, 2), metres
    labels: tuple[tuple[str, float, float], ...]  # (text, x, y), metres
    points_xy: np.ndarray  # (m, 2), metres
    cell_m: float
    north_rotation_deg: float
    values: np.ndarray | None = None  # (m,), coloured by `scale`
    scale: ColourScale | None = None
    passed: np.ndarray | None = None  # (m,) bool, green or red
    segments: tuple[tuple[str, np.ndarray], ...] = ()  # (label, (2, 2)) lines, metres
    polygons: tuple[PlanPolygon, ...] = ()  # outlines, metres; dashed for roof glass
    legend_labels: tuple[str, str] = ("passes", "fails")
    width_px: int = 1600
    # Atlas spec 8.2; the defaults leave the figure exactly as before.
    underlay: Underlay | None = None  # a drawing under the plan
    excluded_outlines: tuple[np.ndarray, ...] = ()  # each (n, 2), metres; grey and hatched
    room_fill: RGB | None = None  # fill colour of every outline in `outlines`
    point_style: str = "cell"  # "cell" or "dot"
    # Merge-split spec 7; the default leaves the figure exactly as before.
    split_lines: tuple[np.ndarray, ...] = ()  # each (k, 2), metres; drawn dashed

    def __post_init__(self):
        if self.point_style not in POINT_STYLES:
            raise FigureError(
                f"{self.title}: point_style must be one of {', '.join(POINT_STYLES)}, got {self.point_style!r}"
            )
        points = np.asarray(self.points_xy, dtype=float).reshape(-1, 2)
        object.__setattr__(self, "points_xy", points)
        if self.values is not None and self.passed is not None:
            raise FigureError(f"{self.title}: give either values or passed, not both")
        if self.values is not None:
            if self.scale is None:
                raise FigureError(f"{self.title}: values need a colour scale")
            values = np.asarray(self.values, dtype=float).reshape(-1)
            if values.shape[0] != points.shape[0]:
                raise FigureError(f"{self.title}: {points.shape[0]} points but {values.shape[0]} values")
            object.__setattr__(self, "values", values)
        if self.passed is not None:
            passed = np.asarray(self.passed, dtype=bool).reshape(-1)
            if passed.shape[0] != points.shape[0]:
                raise FigureError(f"{self.title}: {points.shape[0]} points but {passed.shape[0]} pass flags")
            object.__setattr__(self, "passed", passed)
        if not (self.cell_m > 0.0):
            raise FigureError(f"{self.title}: cell_m must be above 0, got {self.cell_m}")
        if self.width_px < 200:
            raise FigureError(f"{self.title}: width_px must be at least 200, got {self.width_px}")


@dataclasses.dataclass(frozen=True)
class BarSpec:
    title: str
    categories: tuple[str, ...]
    values: tuple[float, ...]
    unit: str
    threshold: float | None = None
    width_px: int = 1200
    height_px: int = 600

    def __post_init__(self):
        if len(self.categories) != len(self.values):
            raise FigureError(
                f"{self.title}: {len(self.categories)} categories but {len(self.values)} value(s)"
            )
        if not self.categories:
            raise FigureError(f"{self.title}: no category to draw")
        if not all(math.isfinite(float(v)) for v in self.values):
            raise FigureError(f"{self.title}: every value must be a finite number")


@dataclasses.dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float
    fill: RGB


@dataclasses.dataclass(frozen=True)
class Poly:
    points: tuple[tuple[float, float], ...]
    stroke: RGB | None
    fill: RGB | None
    width: float = 1.0
    tag: str = ""
    dash: tuple[float, float] = ()  # (on, off) in pixels; () is a solid line
    hatch: bool = False  # diagonal lines in the stroke colour; drawn by svgtext.to_svg only


@dataclasses.dataclass(frozen=True)
class Dot:
    """A filled circle: a model-check grid point (atlas spec 8)."""
    x: float
    y: float
    r: float
    fill: RGB


@dataclasses.dataclass(frozen=True)
class Image:
    """A PNG or JPEG image drawn through ``matrix``, which takes an image
    pixel to a canvas pixel as SVG ``matrix(a b c d e f)`` does, faded to
    ``opacity`` and clipped to ``clip`` (x, y, width, height in canvas
    pixels) when given. Drawn by ``svgtext.to_svg`` only: the PNG painters
    skip it (a Word figure never carries one)."""
    data: bytes
    mime: str
    width: int
    height: int
    matrix: tuple[float, float, float, float, float, float]
    opacity: float
    clip: tuple[float, float, float, float] | None = None


@dataclasses.dataclass(frozen=True)
class Line:
    x0: float
    y0: float
    x1: float
    y1: float
    stroke: RGB
    width: float = 1.0
    dash: tuple[float, float] = ()  # (on, off) in pixels; () is a solid line


@dataclasses.dataclass(frozen=True)
class Text:
    x: float
    y: float  # the vertical centre of the text
    text: str
    size: float  # pixels
    anchor: str = "start"  # start | middle | end
    colour: RGB = TEXT_RGB


@dataclasses.dataclass(frozen=True)
class Canvas:
    width: int
    height: int
    items: tuple


@dataclasses.dataclass(frozen=True)
class Layout:
    width_px: int
    height_px: int
    x0: float
    y1: float
    scale: float  # pixels per metre
    plan_top: int
    k: float = 1.0  # width_px / _REF_PX: every stated pixel size is multiplied by it

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        return (float(x) - self.x0) * self.scale, (self.y1 - float(y)) * self.scale + self.plan_top


def _extent(spec: PlanSpec) -> np.ndarray:
    parts = [np.asarray(o, dtype=float)[:, :2] for o in spec.outlines if np.asarray(o).size]
    if spec.points_xy.size:
        parts.append(spec.points_xy)
    parts.extend(np.asarray(seg, dtype=float).reshape(-1, 2) for _, seg in spec.segments)
    parts.extend(np.asarray(p.points, dtype=float).reshape(-1, 2) for p in spec.polygons)
    parts.extend(np.asarray(o, dtype=float)[:, :2] for o in spec.excluded_outlines if np.asarray(o).size)
    if not parts:
        raise FigureError(f"{spec.title}: nothing to draw: no outline, no point and no segment")
    return np.vstack(parts)


def plan_layout(spec: PlanSpec) -> Layout:
    xy = _extent(spec)
    x0, y0 = xy.min(axis=0) - MARGIN_M
    x1, y1 = xy.max(axis=0) + MARGIN_M
    k = spec.width_px / _REF_PX
    title, legend = int(round(TITLE_PX * k)), int(round(LEGEND_PX * k))
    # I4: the whole figure, printed PRINT_WIDTH_CM wide, stays within
    # MAX_PRINT_HEIGHT_CM; a long floor is drawn at a smaller scale.
    max_h = int(math.floor(spec.width_px * MAX_PRINT_HEIGHT_CM / PRINT_WIDTH_CM))
    scale = min(spec.width_px / (x1 - x0), (max_h - title - legend) / (y1 - y0))
    plan_h = min(int(math.ceil((y1 - y0) * scale)), max_h - title - legend)
    return Layout(spec.width_px, title + plan_h + legend, float(x0), float(y1), float(scale), title, k)


def _north_arrow(cx: float, cy: float, rotation_deg: float, k: float = 1.0) -> list:
    r = math.radians(rotation_deg)
    # Screen y points down, so a counter-clockwise model turn is clockwise
    # on screen by -r: rotate each arrow vertex by -r in screen terms.
    cos_r, sin_r = math.cos(r), math.sin(r)

    def turn(dx: float, dy: float) -> tuple[float, float]:
        return cx + dx * cos_r + dy * sin_r, cy - dx * sin_r + dy * cos_r

    a = _ARROW_PX * k
    points = (turn(0.0, -a), turn(0.4 * a, a), turn(0.0, 0.5 * a), turn(-0.4 * a, a))
    label_x, label_y = turn(0.0, -a - 10.0 * k)
    return [Poly(points, None, TEXT_RGB, tag="north"), Text(label_x, label_y, "N", 14.0 * k, "middle")]


def _scale_bar(layout: Layout, y: float) -> list:
    k = layout.k
    metres = [m for m in (1, 2, 5, 10, 20, 50, 100, 200) if m * layout.scale <= layout.width_px / 4.0]
    length = metres[-1] if metres else 1
    px = length * layout.scale
    return [Rect(8.0 * k, y - 1.5 * k, px, 3.0 * k, TEXT_RGB),
            Text(8.0 * k + px + 4.0 * k, y, f"{length:g} m", 13.0 * k)]


def _legend(spec: PlanSpec, layout: Layout) -> list:
    k = layout.k
    top = layout.height_px - LEGEND_PX * k
    items = _scale_bar(layout, top + (LEGEND_PX - 10.0) * k)
    left = layout.width_px * 0.35
    if spec.scale is not None:
        scale = spec.scale
        span = layout.width_px * 0.55
        step = span / _GRADIENT_STEPS
        for i in range(_GRADIENT_STEPS):
            items.append(Rect(left + i * step, top + 4.0 * k, step + 0.5, 10.0 * k,
                              ramp((i + 0.5) / _GRADIENT_STEPS)))
        for tick in scale.ticks():
            x = left + scale.fraction(tick) * span
            items.append(Text(x, top + 21.0 * k, f"{tick:g}", 12.0 * k, "middle"))
        for mark in scale.marks:
            x = left + scale.fraction(mark) * span
            items.append(Line(x, top + 1.0 * k, x, top + 17.0 * k, MARK_RGB, 2.0 * k))
        items.append(Text(left, top + 36.0 * k, scale.unit, 13.0 * k))
    elif spec.passed is not None:
        for i, (label, colour) in enumerate(zip(spec.legend_labels, (PASS_RGB, FAIL_RGB))):
            x = left + i * 150.0 * k
            items.append(Rect(x, top + 6.0 * k, 12.0 * k, 12.0 * k, colour))
            items.append(Text(x + 18.0 * k, top + 12.0 * k, label, 13.0 * k))
    return items


def _underlay_image(underlay: Underlay, layout: Layout) -> Image:
    """The underlay as an ``Image``: its pixel-to-model matrix followed by
    the layout's model-to-canvas step, clipped to the plan band (the title
    and the legend stay clear)."""
    a, b, c, d, e, f = underlay.model_from_pixel
    s = layout.scale
    matrix = (s * a, -s * b, s * c, -s * d, s * (e - layout.x0), layout.plan_top + s * (layout.y1 - f))
    legend = int(round(LEGEND_PX * layout.k))
    clip = (0.0, float(layout.plan_top), float(layout.width_px),
            float(layout.height_px - layout.plan_top - legend))
    return Image(underlay.data, underlay.mime, underlay.width, underlay.height, matrix, underlay.opacity, clip)


def plan_canvas(spec: PlanSpec) -> tuple[Canvas, Layout]:
    layout = plan_layout(spec)
    k = layout.k
    items: list = [Text(6.0 * k, layout.plan_top / 2.0, spec.title, 16.0 * k)]
    # Final review I1: the room fills are opaque, so they come before the
    # drawing; drawn over it, they hid it inside every included room.
    if spec.room_fill is not None:
        for outline in spec.outlines:
            ring = np.asarray(outline, dtype=float)[:, :2]
            if ring.size:
                items.append(Poly(tuple(layout.to_px(x, y) for x, y in ring), None, spec.room_fill, 1.0 * k))
    if spec.underlay is not None:
        items.append(_underlay_image(spec.underlay, layout))
    for outline in spec.excluded_outlines:
        ring = np.asarray(outline, dtype=float)[:, :2]
        if ring.size:
            items.append(Poly(tuple(layout.to_px(x, y) for x, y in ring), NEUTRAL_RGB, None, 1.0 * k, hatch=True))
    full = spec.cell_m * layout.scale
    size = full if spec.values is not None else 0.9 * full
    for i, (x, y) in enumerate(spec.points_xy):
        if spec.values is not None:
            colour = spec.scale.rgb(spec.values[i])
        elif spec.passed is not None:
            colour = PASS_RGB if spec.passed[i] else FAIL_RGB
        else:
            colour = NEUTRAL_RGB
        px, py = layout.to_px(x, y)
        if spec.point_style == "dot":
            items.append(Dot(px, py, _DOT_SHARE * full / 2.0, colour))
        else:
            items.append(Rect(px - size / 2.0, py - size / 2.0, size, size, colour))
    for outline in spec.outlines:
        ring = np.asarray(outline, dtype=float)[:, :2]
        if ring.size:
            items.append(Poly(tuple(layout.to_px(x, y) for x, y in ring), ROOM_RGB, None, 1.0 * k))
    for line in spec.split_lines:
        ends = [layout.to_px(x, y) for x, y in np.asarray(line, dtype=float)[:, :2]]
        for (ax, ay), (bx, by) in zip(ends, ends[1:]):
            items.append(Line(ax, ay, bx, by, ROOM_RGB, 1.5 * k, dash=(6.0 * k, 4.0 * k)))
    for label, seg in spec.segments:
        (ax, ay), (bx, by) = (layout.to_px(x, y) for x, y in np.asarray(seg, dtype=float).reshape(2, 2))
        items.append(Line(ax, ay, bx, by, SEGMENT_RGB, 2.5 * k))
        items.append(Text((ax + bx) / 2.0, (ay + by) / 2.0 - 8.0 * k, label, 12.0 * k, "middle", SEGMENT_RGB))
    for polygon in spec.polygons:
        ring = tuple(layout.to_px(x, y) for x, y in polygon.points)
        dash = (8.0 * k, 5.0 * k) if polygon.dashed else ()
        items.append(Poly(ring, SEGMENT_RGB, None, 2.0 * k, dash=dash))
        if polygon.label:
            cx = sum(p[0] for p in ring) / len(ring)
            cy = sum(p[1] for p in ring) / len(ring)
            items.append(Text(cx, cy, polygon.label, 12.0 * k, "middle", SEGMENT_RGB))
    for text, x, y in spec.labels:
        px, py = layout.to_px(x, y)
        items.append(Text(px, py, text, 12.0 * k, "middle"))
    items.extend(_north_arrow(layout.width_px - 24.0 * k, layout.plan_top + 26.0 * k, spec.north_rotation_deg, k))
    items.extend(_legend(spec, layout))
    return Canvas(layout.width_px, layout.height_px, tuple(items)), layout


def bar_canvas(spec: BarSpec) -> Canvas:
    w, h = spec.width_px, spec.height_px
    k = w / _REF_PX  # I4: sizes stated for an 800 px wide chart
    left, right, top, bottom = 56.0 * k, 16.0 * k, 40.0 * k, 36.0 * k
    top_value = max([*map(float, spec.values), float(spec.threshold or 0.0), 1.0]) * 1.1
    plot_w, plot_h = w - left - right, h - top - bottom

    def y_of(value: float) -> float:
        return top + plot_h * (1.0 - float(value) / top_value)

    items: list = [
        Text(6.0 * k, top / 2.0 - 4.0 * k, spec.title, 16.0 * k),
        Line(left, top, left, top + plot_h, TEXT_RGB, 1.0 * k),
        Line(left, top + plot_h, left + plot_w, top + plot_h, TEXT_RGB, 1.0 * k),
    ]
    for i in range(5):
        value = top_value * i / 4.0
        items.append(Text(left - 5.0 * k, y_of(value), f"{value:.0f}", 12.0 * k, "end"))
    items.append(Text(left - 5.0 * k, top - 10.0 * k, spec.unit, 12.0 * k, "end"))
    slot = plot_w / len(spec.values)
    for i, (category, value) in enumerate(zip(spec.categories, spec.values)):
        x = left + i * slot + slot * 0.15
        y = y_of(value)
        items.append(Rect(x, y, slot * 0.7, top + plot_h - y, BAR_RGB))
        items.append(Text(x + slot * 0.35, y - 8.0 * k, f"{float(value):.1f}", 12.0 * k, "middle"))
        items.append(Text(x + slot * 0.35, top + plot_h + 14.0 * k, str(category)[:14], 12.0 * k, "middle"))
    if spec.threshold is not None:
        y = y_of(spec.threshold)
        items.append(Line(left, y, left + plot_w, y, FAIL_RGB, 1.5 * k))
        items.append(Text(left + plot_w, y - 8.0 * k, f"{float(spec.threshold):g}", 12.0 * k, "end", FAIL_RGB))
    return Canvas(w, h, tuple(items))


def rgb_hex(colour: Sequence[int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*colour)


def dash_segments(points, on: float, off: float, closed: bool = True) -> tuple:
    """The drawn pieces of a dashed line through ``points`` (pixels): the
    pattern ``on`` drawn, ``off`` blank starts at the first point and runs
    on across corners, as SVG and matplotlib dash a path. For a painter
    with no dashes of its own (Pillow)."""
    if not (on > 0.0) or off < 0.0:
        raise FigureError(f"dash: the drawn length must be above 0 and the gap at least 0, got {on}, {off}")
    ring = [(float(x), float(y)) for x, y in points]
    if closed and len(ring) > 1:
        ring.append(ring[0])
    period = on + off
    phase = 0.0
    out = []
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0.0:
            continue
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        s = 0.0
        while s < length:
            if phase < on:
                step = min(on - phase, length - s)
                out.append(((x0 + ux * s, y0 + uy * s), (x0 + ux * (s + step), y0 + uy * (s + step))))
            else:
                step = min(period - phase, length - s)
            s += step
            phase = (phase + step) % period
    return tuple(out)
