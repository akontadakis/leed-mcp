"""A plain SVG plan of Quality Views points (QV spec 6.8): room outlines in
grey, each point a square of the grid spacing, green for pass, red for
fail, and a north arrow. Plain text, no drawing library.

The plan is drawn with the model's +Y up. The north arrow shows true
north: ``north_rotation_deg`` is the angle of true north from the model's
+Y axis, counter-clockwise positive -- the sense in which the daylight
engine turns the sky (``gendaymtx -r``, ``xform -rz``). So true north is
the model direction (-sin r, cos r), and on screen, where SVG's rotate()
turns clockwise, the upward arrow is turned by -r.
"""
from __future__ import annotations

from collections.abc import Sequence
from xml.sax.saxutils import escape, quoteattr

import numpy as np

_PASS = "#2e7d32"
_FAIL = "#c62828"
_ROOM = "#666666"
_MARGIN_M = 1.0
_TITLE_PX = 40
_NORTH_PX = 18


def plan_svg(
    *, title: str, outlines: Sequence[np.ndarray], points_xy: np.ndarray, passed: np.ndarray,
    cell_m: float, north_rotation_deg: float, width_px: int = 1000,
) -> str:
    points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    flags = np.asarray(passed, dtype=bool).reshape(-1)
    if points.shape[0] != flags.shape[0]:
        raise ValueError(f"{points.shape[0]} points but {flags.shape[0]} pass flags")
    shapes = [np.asarray(o, dtype=float)[:, :2] for o in outlines]
    everything = [s for s in shapes if s.size] + ([points] if points.size else [])
    if not everything:
        raise ValueError("nothing to draw: no outline and no point")
    xy = np.vstack(everything)
    x0, y0 = xy.min(axis=0) - _MARGIN_M
    x1, y1 = xy.max(axis=0) + _MARGIN_M
    scale = width_px / (x1 - x0)
    height_px = (y1 - y0) * scale + _TITLE_PX

    def sx(x: float) -> float:
        return (x - x0) * scale

    def sy(y: float) -> float:
        return (y1 - y) * scale + _TITLE_PX

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px:.0f}" '
        f'viewBox="0 0 {width_px} {height_px:.0f}">',
        f"<title>{escape(title)}</title>",
        f'<text x="10" y="26" font-family="sans-serif" font-size="18">{escape(title)}</text>',
    ]
    for shape in shapes:
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in shape)
        out.append(f'<polygon class="room" points={quoteattr(coords)} fill="none" stroke="{_ROOM}" stroke-width="1"/>')
    size = cell_m * scale * 0.9
    for (x, y), ok in zip(points, flags):
        out.append(
            f'<rect class="{"pass" if ok else "fail"}" x="{sx(x) - size / 2:.1f}" y="{sy(y) - size / 2:.1f}" '
            f'width="{size:.1f}" height="{size:.1f}" fill="{_PASS if ok else _FAIL}"/>'
        )
    out.append(_north_arrow(width_px - 2 * _NORTH_PX, _TITLE_PX + 2 * _NORTH_PX, north_rotation_deg))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _north_arrow(cx: float, cy: float, north_rotation_deg: float) -> str:
    """A small arrow at (cx, cy), pointing to true north (module docstring)."""
    turn = -float(north_rotation_deg) + 0.0  # + 0.0: never "-0"
    r = _NORTH_PX
    return (
        f'<g class="north" transform="translate({cx:.1f},{cy:.1f}) rotate({turn:g})">'
        f'<polygon points="0,{-r} {r * 0.4:.1f},{r} 0,{r * 0.5:.1f} {-r * 0.4:.1f},{r}" fill="#000000"/>'
        f'<text x="0" y="{-r - 4}" text-anchor="middle" font-family="sans-serif" font-size="12">N</text>'
        "</g>"
    )
