# tests/test_deliver_renderers.py
"""Every renderer paints the same display list (reports spec R3, 11).

The pixel under a point's centre must carry that point's colour, so the
three painters agree on where a point lands.
"""
from __future__ import annotations

import importlib
import io

import numpy as np
import pytest

pytest.importorskip("PIL", reason="the report extras are not installed: pip install -e '.[report]'")
from PIL import Image

from leed_daylight.deliver.figures import base as base_module
from leed_daylight.deliver.figures.base import get_renderer, render_bar, render_plan
from leed_daylight.deliver.figures.spec import (
    BAR_RGB, FAIL_RGB, PASS_RGB, SEGMENT_RGB, WHITE, BarSpec, Canvas, ColourScale, FigureError, Line,
    PlanSpec, Poly, Rect, bar_canvas, plan_canvas, ramp,
)
from leed_daylight.project.config import RENDERERS

RENDERER_NAMES = ["matplotlib", "cairosvg", "pillow"]

OUTLINE = np.array([[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]])
POINTS = np.array([[1.5, 1.5], [4.5, 2.5]])


@pytest.fixture(params=RENDERER_NAMES)
def renderer(request):
    try:
        return get_renderer(request.param)
    except FigureError as exc:
        pytest.skip(f"{request.param} renderer not available here: {exc}")


def _image(png: bytes) -> Image.Image:
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    return Image.open(io.BytesIO(png)).convert("RGB")


def _close(pixel, colour, tolerance=8) -> bool:
    return all(abs(int(a) - int(b)) <= tolerance for a, b in zip(pixel, colour))


def _at(image, layout, x, y):
    px, py = layout.to_px(x, y)
    return image.getpixel((int(round(px)), int(round(py))))


def test_pass_fail_plan(renderer):
    spec = PlanSpec(
        title="Pass and fail", outlines=(OUTLINE,), labels=(), points_xy=POINTS, cell_m=1.0,
        north_rotation_deg=0.0, passed=np.array([True, False]),
    )
    canvas, layout = plan_canvas(spec)
    image = _image(render_plan(spec, renderer))

    assert image.size == (canvas.width, canvas.height)
    assert _close(_at(image, layout, 1.5, 1.5), PASS_RGB)
    assert _close(_at(image, layout, 4.5, 2.5), FAIL_RGB)


def test_heat_map_plan(renderer):
    spec = PlanSpec(
        title="Heat map", outlines=(OUTLINE,), labels=(), points_xy=POINTS, cell_m=1.0,
        north_rotation_deg=0.0, values=np.array([0.0, 100.0]), scale=ColourScale(0.0, 100.0, "%"),
    )
    _, layout = plan_canvas(spec)
    image = _image(render_plan(spec, renderer))

    assert _close(_at(image, layout, 1.5, 1.5), ramp(0.0))
    assert _close(_at(image, layout, 4.5, 2.5), ramp(1.0))


def test_bar_chart(renderer):
    spec = BarSpec(title="Bars", categories=("L0", "L1"), values=(40.0, 80.0), unit="%", threshold=55.0)
    canvas = bar_canvas(spec)
    image = _image(render_bar(spec, renderer))

    assert image.size == (canvas.width, canvas.height)


def test_bar_chart_pixels(renderer):
    """Pre-flight ruling F10: the bar-chart pixel colours must be checked,
    not just the image size. The centre of each bar carries the bar colour,
    and the threshold line carries the fail colour."""
    spec = BarSpec(title="Bars", categories=("L0", "L1"), values=(40.0, 80.0), unit="%", threshold=55.0)
    canvas = bar_canvas(spec)
    image = _image(render_bar(spec, renderer))

    bars = [item for item in canvas.items if isinstance(item, Rect) and item.fill == BAR_RGB]
    assert len(bars) == 2
    for bar in bars:
        cx, cy = bar.x + bar.w / 2.0, bar.y + bar.h / 2.0
        pixel = image.getpixel((int(round(cx)), int(round(cy))))
        assert _close(pixel, BAR_RGB), (bar, pixel)

    threshold_lines = [item for item in canvas.items if isinstance(item, Line) and item.stroke == FAIL_RGB]
    assert len(threshold_lines) == 1
    line = threshold_lines[0]
    mx, my = (line.x0 + line.x1) / 2.0, (line.y0 + line.y1) / 2.0
    pixel = image.getpixel((int(round(mx)), int(round(my))))
    assert _close(pixel, FAIL_RGB), pixel


def test_an_unknown_renderer_is_refused():
    with pytest.raises(FigureError, match="unknown renderer 'gnuplot'"):
        get_renderer("gnuplot")


def test_painters_registry_matches_the_project_renderers():
    """Pre-flight ruling F12: the plan's module-level `assert tuple(_PAINTERS)
    == RENDERERS` vanishes under `python -O`; a real test does not."""
    assert tuple(base_module._PAINTERS) == RENDERERS


def test_missing_expected_library_becomes_a_figure_error(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == f"{base_module.__package__}.mpl":
            raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(base_module.importlib, "import_module", fake_import_module)

    with pytest.raises(FigureError, match="matplotlib"):
        get_renderer("matplotlib")


def test_unrelated_import_error_in_a_painter_module_is_not_converted(monkeypatch):
    """Pre-flight ruling F12: a real bug in a painter module (a typo, a stray
    import unrelated to the expected library) must fail loudly, not be
    swallowed into a silent 'renderer not available' skip."""
    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == f"{base_module.__package__}.mpl":
            raise ImportError("cannot import name 'oops'", name="some_unrelated_module")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(base_module.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError) as excinfo:
        get_renderer("matplotlib")
    assert not isinstance(excinfo.value, FigureError)


def test_svg_text_is_escaped_and_complete():
    from leed_daylight.deliver.figures.svgtext import to_svg

    spec = PlanSpec(
        title='Room <A> & "B"', outlines=(OUTLINE,), labels=(), points_xy=POINTS, cell_m=1.0,
        north_rotation_deg=0.0, passed=np.array([True, False]),
    )
    canvas, _ = plan_canvas(spec)
    text = to_svg(canvas)

    assert text.startswith("<svg ")
    assert text.rstrip().endswith("</svg>")
    assert "Room &lt;A&gt; &amp; &quot;B&quot;" in text or "Room &lt;A&gt; &amp; \"B\"" in text
    assert f'width="{canvas.width}"' in text


def test_cairo_library_missing_becomes_a_figure_error(monkeypatch):
    """Pre-flight ruling F2: a cairo OSError (library present as a Python
    package but the system cairo library cannot load) must be a named
    refusal, not a traceback."""
    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == f"{base_module.__package__}.svg":
            raise OSError('cannot load library "cairo-2"')
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(base_module.importlib, "import_module", fake_import_module)

    with pytest.raises(FigureError, match="cairosvg"):
        get_renderer("cairosvg")


# -- Task 11: dashed roof-glass plan polygons --------------------------------

_RING = ((20.0, 20.0), (180.0, 20.0), (180.0, 80.0), (20.0, 80.0))


def test_a_dashed_outline_is_painted_as_dashes(renderer):
    """10 px dashes and 10 px gaps from the first vertex: on the top edge
    x 20-30 is a dash, 30-40 a gap, and so on; the top edge is exactly eight
    periods, so the right edge starts with a dash."""
    canvas = Canvas(200, 100, (Poly(_RING, SEGMENT_RGB, None, 3.0, dash=(10.0, 10.0)),))

    image = _image(renderer.paint(canvas))

    for x in (25, 45, 65):
        assert _close(image.getpixel((x, 20)), SEGMENT_RGB), x
    for x in (35, 55, 75):
        assert _close(image.getpixel((x, 20)), WHITE), x
    assert _close(image.getpixel((180, 25)), SEGMENT_RGB)
    assert _close(image.getpixel((180, 35)), WHITE)


def test_svg_text_writes_a_dash_array_only_for_a_dashed_poly():
    from leed_daylight.deliver.figures.svgtext import to_svg

    canvas = Canvas(200, 100, (
        Poly(_RING, SEGMENT_RGB, None, 3.0, dash=(10.0, 10.0)),
        Poly(_RING, SEGMENT_RGB, None, 3.0),
    ))

    text = to_svg(canvas)

    assert text.count('stroke-dasharray="10 10"') == 1
    assert text.count("stroke-dasharray") == 1
