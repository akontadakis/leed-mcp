# leed_daylight/deliver/figures/svgtext.py
"""Pure SVG text generation for a ``Canvas`` (reports spec R3).

``to_svg`` imports no third-party library, so it is importable and testable
on every machine, including one where the system cairo library cannot load.
``figures/svg.py`` builds the cairosvg painter on top of this module.

Atlas spec 8.2: ``Dot`` is a circle; ``Image`` is embedded as a base64 data
URI, faded to its opacity and clipped to its ``clip`` rectangle; a hatched
``Poly`` is drawn, then filled once more with a diagonal-line pattern in its
stroke colour. Pattern and clip ids start with ``id_prefix``, so several
SVGs written into one page keep their ids apart. A canvas with none of
these items is written exactly as before (no ``<defs>``). ``for_png`` is
the canvas the cairosvg painter paints: no image and no hatch, as the
other two painters.

Final review I2: a page that holds many SVGs of one drawing (one plan per
layer of a storey) embeds its bytes once, with ``shared_images_svg``, and
passes ``image_ids`` to ``to_svg``: an ``Image`` whose bytes are named there
is drawn as ``<use href="#id">`` with its own matrix, opacity and clip.
Without ``image_ids`` every ``Image`` is embedded as before.
"""
from __future__ import annotations

import base64
import dataclasses
from collections.abc import Mapping
from xml.sax.saxutils import escape, quoteattr

from .spec import TEXT_RGB, Canvas, Dot, Image, Line, Poly, Rect, Text, rgb_hex

_ANCHOR = {"start": "start", "middle": "middle", "end": "end"}
_FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
#: The hatch pattern's line spacing, in canvas pixels.
_HATCH_PX = 8


def _points(points) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _hatch_colour(item: Poly) -> str:
    return rgb_hex(item.stroke or TEXT_RGB)


def _hatch_defs(canvas: Canvas, id_prefix: str) -> list[str]:
    colours = sorted({_hatch_colour(i) for i in canvas.items if isinstance(i, Poly) and i.hatch})
    if not colours:
        return []
    patterns = "".join(
        f'<pattern id="{id_prefix}hatch-{colour[1:]}" patternUnits="userSpaceOnUse" '
        f'width="{_HATCH_PX}" height="{_HATCH_PX}" patternTransform="rotate(45)">'
        f'<line x1="0" y1="0" x2="0" y2="{_HATCH_PX}" stroke="{colour}" stroke-width="1.5"/></pattern>'
        for colour in colours
    )
    return [f"<defs>{patterns}</defs>"]


def _data_uri(item) -> str:
    return f"data:{item.mime};base64,{base64.b64encode(item.data).decode('ascii')}"


def _image(item: Image, clip_id: str, ref: str | None = None) -> list[str]:
    matrix = " ".join(f"{float(v) + 0.0:.6g}" for v in item.matrix)  # + 0.0: never "-0"
    if ref is None:
        image = (
            f'<image x="0" y="0" width="{item.width}" height="{item.height}" preserveAspectRatio="none" '
            f'transform="matrix({matrix})" opacity="{float(item.opacity):g}" href="{_data_uri(item)}"/>'
        )
    else:
        image = f'<use href="#{ref}" transform="matrix({matrix})" opacity="{float(item.opacity):g}"/>'
    if item.clip is None:
        return [image]
    x, y, w, h = item.clip
    return [
        f'<defs><clipPath id="{clip_id}"><rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}"/>'
        f"</clipPath></defs>",
        f'<g clip-path="url(#{clip_id})">{image}</g>',
    ]


def shared_images_svg(images: Mapping[str, object]) -> str:
    """A hidden, zero-sized ``<svg>`` whose ``<defs>`` hold each image of
    ``images`` (id to anything with ``data``, ``mime``, ``width`` and
    ``height``: an ``Image`` or an ``Underlay``) once, for ``to_svg(...,
    image_ids=...)`` to draw by reference; ``""`` for no image."""
    if not images:
        return ""
    defs = "".join(
        f'<image id="{image_id}" x="0" y="0" width="{image.width}" height="{image.height}" '
        f'preserveAspectRatio="none" href="{_data_uri(image)}"/>'
        for image_id, image in images.items()
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="0" height="0" aria-hidden="true" '
        f'style="position:absolute;overflow:hidden"><defs>{defs}</defs></svg>'
    )


def for_png(canvas: Canvas) -> Canvas:
    """``canvas`` without its images and with no hatch on any polygon."""
    items = tuple(
        dataclasses.replace(item, hatch=False) if isinstance(item, Poly) and item.hatch else item
        for item in canvas.items if not isinstance(item, Image)
    )
    return dataclasses.replace(canvas, items=items)


def to_svg(canvas: Canvas, *, id_prefix: str = "", image_ids: Mapping[bytes, str] | None = None) -> str:
    """``canvas`` as SVG text; ids start with ``id_prefix``. An ``Image``
    whose bytes are a key of ``image_ids`` is drawn by reference to that id
    (see ``shared_images_svg``), any other is embedded."""
    image_ids = image_ids or {}
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas.width}" height="{canvas.height}" '
        f'viewBox="0 0 {canvas.width} {canvas.height}">',
        f'<rect x="0" y="0" width="{canvas.width}" height="{canvas.height}" fill="#ffffff"/>',
    ]
    out.extend(_hatch_defs(canvas, id_prefix))
    images = 0
    for item in canvas.items:
        if isinstance(item, Rect):
            out.append(
                f'<rect x="{item.x:.2f}" y="{item.y:.2f}" width="{item.w:.2f}" height="{item.h:.2f}" '
                f'fill="{rgb_hex(item.fill)}" shape-rendering="crispEdges"/>'
            )
        elif isinstance(item, Poly):
            fill = rgb_hex(item.fill) if item.fill else "none"
            stroke = rgb_hex(item.stroke) if item.stroke else "none"
            dash = f' stroke-dasharray="{item.dash[0]:g} {item.dash[1]:g}"' if item.dash else ""
            out.append(
                f'<polygon points={quoteattr(_points(item.points))} fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{item.width:g}"{dash}/>'
            )
            if item.hatch:
                out.append(
                    f'<polygon points={quoteattr(_points(item.points))} '
                    f'fill="url(#{id_prefix}hatch-{_hatch_colour(item)[1:]})" stroke="none"/>'
                )
        elif isinstance(item, Dot):
            out.append(f'<circle cx="{item.x:.2f}" cy="{item.y:.2f}" r="{item.r:.2f}" fill="{rgb_hex(item.fill)}"/>')
        elif isinstance(item, Image):
            out.extend(_image(item, f"{id_prefix}clip-{images}", image_ids.get(item.data)))
            images += 1
        elif isinstance(item, Line):
            dash = f' stroke-dasharray="{item.dash[0]:g} {item.dash[1]:g}"' if item.dash else ""
            out.append(
                f'<line x1="{item.x0:.2f}" y1="{item.y0:.2f}" x2="{item.x1:.2f}" y2="{item.y1:.2f}" '
                f'stroke="{rgb_hex(item.stroke)}" stroke-width="{item.width:g}"{dash}/>'
            )
        elif isinstance(item, Text):
            out.append(
                f'<text x="{item.x:.2f}" y="{item.y:.2f}" font-family={quoteattr(_FONT)} '
                f'font-size="{item.size:g}" text-anchor="{_ANCHOR[item.anchor]}" '
                f'dominant-baseline="central" fill="{rgb_hex(item.colour)}">{escape(item.text)}</text>'
            )
    out.append("</svg>")
    return "\n".join(out) + "\n"
