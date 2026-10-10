# tests/test_deliver_figure_items.py
"""The display-list items the atlas plans need (atlas spec 8, 8.2).

``Dot`` (a model-check grid point), ``Image`` (a drawing under the plan),
a hatch on ``Poly`` (an excluded room), and the ``PlanSpec`` fields that
emit them: ``underlay``, ``excluded_outlines``, ``room_fill`` and
``point_style``. With the new fields at their defaults every figure is
unchanged (tests/test_golden_figures.py). Every number is invented and round.
"""
from __future__ import annotations

import dataclasses
import io

import numpy as np
import pytest

from leed_daylight.deliver.figures.spec import (
    LEGEND_PX, NEUTRAL_RGB, PASS_RGB, Canvas, Dot, FigureError, PlanSpec, Poly, Rect, Underlay,
    plan_canvas, plan_layout,
)
from leed_daylight.deliver.figures.spec import Image as ImageItem
from leed_daylight.deliver.figures.svgtext import for_png, shared_images_svg, to_svg

OUTLINE = np.array([[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]])
EXCLUDED = np.array([[6.0, 0.0], [10.0, 0.0], [10.0, 4.0], [6.0, 4.0]])
POINTS = np.array([[1.5, 1.5], [4.5, 2.5]])
FILL = (240, 240, 240)
#: Not a real image: the display list and the SVG writer never decode it.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
#: 1 pixel = 0.01 m, pixel y down, model y up; pixel (100, 500) is (0, 0).
PIXEL_TO_MODEL = (0.01, 0.0, 0.0, -0.01, -1.0, 5.0)


def _spec(**changes) -> PlanSpec:
    base = dict(
        title="Plan", outlines=(OUTLINE,), labels=(("Room A", 3.0, 2.0),), points_xy=POINTS,
        cell_m=0.5, north_rotation_deg=0.0,
    )
    base.update(changes)
    return PlanSpec(**base)


def _underlay(**changes) -> Underlay:
    base = dict(data=PNG_BYTES, mime="image/png", width=800, height=600,
                model_from_pixel=PIXEL_TO_MODEL, opacity=0.35)
    base.update(changes)
    return Underlay(**base)


def _of(canvas: Canvas, kind) -> list:
    return [item for item in canvas.items if isinstance(item, kind)]


# -- defaults: today's figure --------------------------------------------------


def test_the_new_plan_fields_default_to_todays_figure():
    spec = _spec()
    assert spec.underlay is None
    assert spec.excluded_outlines == ()
    assert spec.room_fill is None
    assert spec.point_style == "cell"
    canvas, _ = plan_canvas(spec)
    assert _of(canvas, Dot) == [] and _of(canvas, ImageItem) == []
    assert not any(poly.hatch for poly in _of(canvas, Poly))


def test_a_poly_is_not_hatched_unless_asked():
    assert Poly(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), NEUTRAL_RGB, None).hatch is False


def test_an_unknown_point_style_is_refused():
    with pytest.raises(FigureError, match="point_style"):
        _spec(point_style="square")


# -- dots, fills, hatches -------------------------------------------------------


def test_the_dot_style_draws_one_dot_per_point_instead_of_a_cell():
    cells, layout = plan_canvas(_spec())
    dots, _ = plan_canvas(_spec(point_style="dot"))
    assert len(_of(dots, Dot)) == 2
    assert len(_of(dots, Rect)) == len(_of(cells, Rect)) - 2
    first = _of(dots, Dot)[0]
    assert (first.x, first.y) == pytest.approx(layout.to_px(1.5, 1.5))
    assert first.fill == NEUTRAL_RGB
    assert 0.0 < 2.0 * first.r < 0.5 * layout.scale  # smaller than a cell


def test_dots_carry_the_pass_and_fail_colours_too():
    canvas, _ = plan_canvas(_spec(point_style="dot", passed=np.array([True, False])))
    assert [dot.fill for dot in _of(canvas, Dot)][0] == PASS_RGB


def test_room_fill_fills_each_outline_under_the_points():
    canvas, layout = plan_canvas(_spec(room_fill=FILL, point_style="dot"))
    filled = [i for i, item in enumerate(canvas.items) if isinstance(item, Poly) and item.fill == FILL]
    assert len(filled) == 1
    fill = canvas.items[filled[0]]
    assert fill.stroke is None
    assert fill.points[0] == pytest.approx(layout.to_px(0.0, 0.0))
    first_dot = next(i for i, item in enumerate(canvas.items) if isinstance(item, Dot))
    assert filled[0] < first_dot


def test_excluded_outlines_are_grey_hatched_and_widen_the_plan():
    spec = _spec(excluded_outlines=(EXCLUDED,))
    canvas, layout = plan_canvas(spec)
    (hatched,) = [poly for poly in _of(canvas, Poly) if poly.hatch]
    assert hatched.stroke == NEUTRAL_RGB
    assert hatched.fill is None
    assert hatched.points[1] == pytest.approx(layout.to_px(10.0, 0.0))
    assert plan_layout(spec).scale < plan_layout(_spec()).scale


# -- the underlay ----------------------------------------------------------------


def test_the_underlay_is_the_first_item_after_the_title():
    canvas, _ = plan_canvas(_spec(underlay=_underlay()))
    assert isinstance(canvas.items[1], ImageItem)
    image = canvas.items[1]
    assert (image.data, image.mime, image.width, image.height, image.opacity) == (
        PNG_BYTES, "image/png", 800, 600, 0.35,
    )


def test_the_image_matrix_takes_a_drawing_pixel_to_its_plan_pixel():
    canvas, layout = plan_canvas(_spec(underlay=_underlay()))
    a, b, c, d, e, f = canvas.items[1].matrix
    ma, mb, mc, md, me, mf = PIXEL_TO_MODEL
    for u, v in ((100.0, 500.0), (700.0, 100.0), (400.0, 300.0)):
        x, y = ma * u + mc * v + me, mb * u + md * v + mf
        assert (a * u + c * v + e, b * u + d * v + f) == pytest.approx(layout.to_px(x, y))


def test_the_underlay_is_clipped_to_the_plan_band():
    canvas, layout = plan_canvas(_spec(underlay=_underlay()))
    legend = round(LEGEND_PX * layout.k)
    assert canvas.items[1].clip == (
        0.0, float(layout.plan_top), float(layout.width_px),
        float(layout.height_px - layout.plan_top - legend),
    )


def test_room_fills_lie_under_the_drawing():
    # Final review I1: an opaque room fill painted over the drawing hid it
    # inside every included room. White, then the fills, then the drawing
    # at its opacity, then the hatch, the points and the outlines.
    canvas, _ = plan_canvas(_spec(underlay=_underlay(), room_fill=FILL, excluded_outlines=(EXCLUDED,),
                                  point_style="dot"))
    image = next(i for i, item in enumerate(canvas.items) if isinstance(item, ImageItem))
    fills = [i for i, item in enumerate(canvas.items) if isinstance(item, Poly) and item.fill == FILL]
    hatched = next(i for i, item in enumerate(canvas.items) if isinstance(item, Poly) and item.hatch)
    first_dot = next(i for i, item in enumerate(canvas.items) if isinstance(item, Dot))
    assert fills and max(fills) < image < hatched < first_dot


@pytest.mark.parametrize("change, message", [
    ({"mime": "image/gif"}, "image/png or image/jpeg"),
    ({"width": 0}, "width"),
    ({"height": -1}, "height"),
    ({"opacity": 1.5}, "opacity"),
    ({"opacity": -0.1}, "opacity"),
    ({"model_from_pixel": (0.01, 0.0, 0.0, -0.01, -1.0)}, "six numbers"),
    ({"model_from_pixel": (0.01, 0.0, 0.0, -0.01, -1.0, float("nan"))}, "six numbers"),
])
def test_a_malformed_underlay_is_refused(change, message):
    with pytest.raises(FigureError, match=message):
        _underlay(**change)


# -- the SVG writer ------------------------------------------------------------


def test_svg_draws_a_dot_as_a_circle():
    canvas = Canvas(100, 100, (Dot(10.0, 20.0, 3.0, PASS_RGB),))
    assert '<circle cx="10.00" cy="20.00" r="3.00" fill="#2e7d32"/>' in to_svg(canvas)


def test_svg_embeds_the_image_as_a_data_uri_clipped_and_faded():
    canvas = Canvas(100, 100, (ImageItem(PNG_BYTES, "image/png", 8, 6, (2.0, 0.0, 0.0, 2.0, 5.0, 10.0),
                                         0.35, clip=(0.0, 10.0, 100.0, 80.0)),))
    svg = to_svg(canvas)
    assert '<clipPath id="clip-0"><rect x="0.00" y="10.00" width="100.00" height="80.00"/></clipPath>' in svg
    assert '<g clip-path="url(#clip-0)">' in svg
    assert 'transform="matrix(2 0 0 2 5 10)"' in svg
    assert 'opacity="0.35"' in svg
    assert 'href="data:image/png;base64,iVBORw0KGgo' in svg
    assert "http:" not in svg.replace('xmlns="http://www.w3.org/2000/svg"', "")


def test_svg_draws_an_image_by_reference_when_asked():
    # Final review I2: a page with many plans of one storey holds its
    # drawing once and draws it in each plan by reference.
    canvas = Canvas(100, 100, (ImageItem(PNG_BYTES, "image/png", 8, 6, (2.0, 0.0, 0.0, 2.0, 5.0, 10.0),
                                         0.35, clip=(0.0, 10.0, 100.0, 80.0)),))
    svg = to_svg(canvas, id_prefix="p-", image_ids={PNG_BYTES: "dwg-a"})
    assert ('<g clip-path="url(#p-clip-0)"><use href="#dwg-a" transform="matrix(2 0 0 2 5 10)" '
            'opacity="0.35"/></g>') in svg
    assert "base64" not in svg and "<image" not in svg
    # An image the mapping does not name is embedded as before.
    assert to_svg(canvas, image_ids={b"other": "dwg-b"}) == to_svg(canvas)


def test_the_shared_images_are_one_hidden_svg_of_definitions():
    svg = shared_images_svg({"dwg-a": _underlay()})
    assert svg.startswith("<svg ") and svg.rstrip().endswith("</svg>")
    assert 'width="0" height="0"' in svg and 'aria-hidden="true"' in svg
    assert svg.count("<image ") == 1
    assert ('<image id="dwg-a" x="0" y="0" width="800" height="600" preserveAspectRatio="none" '
            'href="data:image/png;base64,iVBORw0KGgo') in svg
    assert shared_images_svg({}) == ""


def test_svg_hatches_a_hatched_poly_in_its_stroke_colour():
    square = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    svg = to_svg(Canvas(100, 100, (Poly(square, NEUTRAL_RGB, None, hatch=True),)))
    assert '<pattern id="hatch-969696"' in svg
    assert 'fill="url(#hatch-969696)"' in svg


def test_svg_fills_a_filled_poly():
    square = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    assert 'fill="#f0f0f0" stroke="none"' in to_svg(Canvas(100, 100, (Poly(square, None, FILL),)))


def test_an_id_prefix_keeps_two_svgs_on_one_page_apart():
    square = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    canvas = Canvas(100, 100, (
        ImageItem(PNG_BYTES, "image/png", 8, 6, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), 0.5, clip=(0.0, 0.0, 50.0, 50.0)),
        Poly(square, NEUTRAL_RGB, None, hatch=True),
    ))
    svg = to_svg(canvas, id_prefix="L1-model-")
    assert 'id="L1-model-clip-0"' in svg and 'url(#L1-model-clip-0)' in svg
    assert 'id="L1-model-hatch-969696"' in svg and 'url(#L1-model-hatch-969696)' in svg


def test_svg_of_a_canvas_without_new_items_has_no_definitions():
    canvas, _ = plan_canvas(_spec())
    assert "<defs>" not in to_svg(canvas)


# -- the PNG painters: draw the dot, skip the image and the hatch ---------------


def _painter(name):
    from leed_daylight.deliver.figures.base import get_renderer

    try:
        return get_renderer(name)
    except FigureError as exc:
        pytest.skip(f"{name} renderer not available here: {exc}")


@pytest.mark.parametrize("name", ["matplotlib", "pillow", "cairosvg"])
def test_a_png_painter_draws_the_dot(name):
    pil = pytest.importorskip("PIL.Image", reason="the report extras are not installed")
    painter = _painter(name)
    png = painter.paint(Canvas(200, 100, (Dot(50.0, 50.0, 10.0, PASS_RGB),)))
    image = pil.open(io.BytesIO(png)).convert("RGB")
    assert image.size == (200, 100)
    assert image.getpixel((50, 50)) == PASS_RGB
    assert image.getpixel((150, 50)) == (255, 255, 255)


@pytest.mark.parametrize("name", ["matplotlib", "pillow", "cairosvg"])
def test_a_png_painter_skips_the_image_and_the_hatch(name):
    painter = _painter(name)
    square = ((20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0))
    plain = Canvas(200, 100, (Poly(square, NEUTRAL_RGB, None),))
    with_new = Canvas(200, 100, (
        ImageItem(PNG_BYTES, "image/png", 8, 6, (10.0, 0.0, 0.0, 10.0, 0.0, 0.0), 1.0, clip=(0.0, 0.0, 200.0, 100.0)),
        dataclasses.replace(plain.items[0], hatch=True),
    ))
    assert painter.paint(with_new) == painter.paint(plain)


def test_the_cairosvg_painter_paints_the_canvas_without_image_or_hatch():
    # The cairosvg painter paints ``for_png(canvas)``; tested here without
    # cairo, which many machines lack.
    square = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    dot = Dot(5.0, 5.0, 2.0, PASS_RGB)
    canvas = Canvas(100, 100, (
        ImageItem(PNG_BYTES, "image/png", 8, 6, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), 0.5),
        Poly(square, NEUTRAL_RGB, FILL, hatch=True),
        dot,
    ))
    assert for_png(canvas) == Canvas(100, 100, (Poly(square, NEUTRAL_RGB, FILL), dot))
