# leed_daylight/deliver/figures/pil.py
"""The Pillow painter. Text uses Pillow's own scalable default font
(``ImageFont.load_default(size=...)``, Pillow 10.1 or later), so no system
font path is needed. It draws a ``Dot`` and skips an ``Image`` and a
``Poly``'s hatch (atlas spec 8.2: a Word figure never carries either)."""
from __future__ import annotations

import functools
import io

from PIL import Image, ImageDraw, ImageFont

from .spec import WHITE, Canvas, Dot, Line, Poly, Rect, Text, dash_segments

_ANCHOR = {"start": "lm", "middle": "mm", "end": "rm"}


@functools.lru_cache(maxsize=None)
def _font(size: int):
    return ImageFont.load_default(size=size)


class PillowRenderer:
    name = "pillow"

    def paint(self, canvas: Canvas) -> bytes:
        image = Image.new("RGB", (canvas.width, canvas.height), WHITE)
        draw = ImageDraw.Draw(image)
        for item in canvas.items:
            if isinstance(item, Rect):
                draw.rectangle(
                    [item.x, item.y, item.x + max(item.w - 1.0, 0.0), item.y + max(item.h - 1.0, 0.0)],
                    fill=item.fill,
                )
            elif isinstance(item, Poly):
                if item.dash:
                    # Pillow has no dashed line: draw each dash as a segment.
                    if item.fill is not None:
                        draw.polygon([tuple(p) for p in item.points], fill=item.fill)
                    if item.stroke is not None:
                        for a, b in dash_segments(item.points, *item.dash):
                            draw.line([a, b], fill=item.stroke, width=max(1, round(item.width)))
                else:
                    draw.polygon(
                        [tuple(p) for p in item.points], fill=item.fill, outline=item.stroke,
                        width=max(1, round(item.width)),
                    )
            elif isinstance(item, Dot):
                draw.ellipse([item.x - item.r, item.y - item.r, item.x + item.r, item.y + item.r], fill=item.fill)
            elif isinstance(item, Line):
                if item.dash:
                    ends = ((item.x0, item.y0), (item.x1, item.y1))
                    for a, b in dash_segments(ends, *item.dash, closed=False):
                        draw.line([a, b], fill=item.stroke, width=max(1, round(item.width)))
                else:
                    draw.line([(item.x0, item.y0), (item.x1, item.y1)], fill=item.stroke,
                              width=max(1, round(item.width)))
            elif isinstance(item, Text):
                draw.text((item.x, item.y), item.text, fill=item.colour,
                          font=_font(max(6, round(item.size))), anchor=_ANCHOR[item.anchor])
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
