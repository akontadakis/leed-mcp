# leed_daylight/deliver/figures/svg.py
"""The cairosvg painter: the display list as SVG text (``svgtext.to_svg``),
converted to PNG by cairosvg.

Pre-flight ruling F2: ``to_svg`` itself needs no third-party library and
lives in ``svgtext.py`` so it is importable and testable anywhere. This
module imports ``cairosvg`` at module import time, so that both a missing
``cairosvg`` package (``ImportError``) and a cairo library that cannot load
(``OSError``, cairo present as a Python package but the system cairo library
is missing) are raised while ``base.get_renderer`` imports this module, and
are converted there into a named ``FigureError``.

Atlas spec 8.2: like the other two painters it skips an ``Image`` and a
``Poly``'s hatch (a Word figure never carries either): it paints
``svgtext.for_png(canvas)``.
"""
from __future__ import annotations

import cairosvg

from .spec import Canvas
from .svgtext import for_png, to_svg

__all__ = ["to_svg", "CairoSvgRenderer"]


class CairoSvgRenderer:
    name = "cairosvg"

    def paint(self, canvas: Canvas) -> bytes:
        return cairosvg.svg2png(
            bytestring=to_svg(for_png(canvas)).encode("utf-8"),
            output_width=canvas.width, output_height=canvas.height,
        )
