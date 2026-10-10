# leed_daylight/deliver/figures/mpl.py
"""The matplotlib painter. Uses ``Figure`` and ``FigureCanvasAgg`` directly,
never ``pyplot``, so it keeps no global state. It draws a ``Dot`` and skips
an ``Image`` and a ``Poly``'s hatch (atlas spec 8.2: a Word figure never
carries either)."""
from __future__ import annotations

import io

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle

from .spec import Canvas, Dot, Line, Poly, Rect, Text

_DPI = 100
_POINTS_PER_PX = 72.0 / _DPI
_ALIGN = {"start": "left", "middle": "center", "end": "right"}


def _c(colour):
    return tuple(v / 255.0 for v in colour)


class MatplotlibRenderer:
    name = "matplotlib"

    def paint(self, canvas: Canvas) -> bytes:
        figure = Figure(figsize=(canvas.width / _DPI, canvas.height / _DPI), dpi=_DPI, facecolor="white")
        FigureCanvasAgg(figure)
        axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        axes.set_xlim(0, canvas.width)
        axes.set_ylim(canvas.height, 0)
        axes.axis("off")
        for item in canvas.items:
            if isinstance(item, Rect):
                axes.add_patch(Rectangle((item.x, item.y), item.w, item.h, facecolor=_c(item.fill),
                                         edgecolor="none", linewidth=0))
            elif isinstance(item, Poly):
                patch = Polygon(
                    item.points, closed=True,
                    facecolor=_c(item.fill) if item.fill else "none",
                    edgecolor=_c(item.stroke) if item.stroke else "none",
                    linewidth=item.width * _POINTS_PER_PX,
                )
                if item.dash:
                    # matplotlib scales a dash pattern by the line width
                    # (rcParams "lines.scale_dashes", on by default), so the
                    # pattern is given in line widths: on/width, off/width.
                    on, off = item.dash
                    patch.set_linestyle((0, (on / item.width, off / item.width)))
                axes.add_patch(patch)
            elif isinstance(item, Dot):
                axes.add_patch(Circle((item.x, item.y), item.r, facecolor=_c(item.fill),
                                      edgecolor="none", linewidth=0))
            elif isinstance(item, Line):
                style = {}
                if item.dash:
                    # In line widths, as for a dashed Poly above.
                    on, off = item.dash
                    style["linestyle"] = (0, (on / item.width, off / item.width))
                axes.plot([item.x0, item.x1], [item.y0, item.y1], color=_c(item.stroke),
                          linewidth=item.width * _POINTS_PER_PX, solid_capstyle="butt", **style)
            elif isinstance(item, Text):
                axes.text(item.x, item.y, item.text, fontsize=item.size * _POINTS_PER_PX,
                          ha=_ALIGN[item.anchor], va="center", color=_c(item.colour))
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=_DPI, facecolor="white")
        return buffer.getvalue()
