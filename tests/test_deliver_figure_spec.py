# tests/test_deliver_figure_spec.py
"""Backend-neutral figure descriptions (reports spec 4, figures/spec.py)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from leed_daylight.deliver.figures.spec import (
    FAIL_RGB, PASS_RGB, SEGMENT_RGB, BarSpec, ColourScale, FigureError, PlanPolygon, PlanSpec, Poly, Rect,
    Text, bar_canvas, dash_segments, plan_canvas, ramp,
)

OUTLINE = np.array([[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]])
POINTS = np.array([[1.5, 1.5], [4.5, 2.5]])


def _pass_spec(**kw):
    args = dict(
        title="Test plan", outlines=(OUTLINE,), labels=(("Room North", 3.0, 3.5),),
        points_xy=POINTS, cell_m=1.0, north_rotation_deg=0.0, passed=np.array([True, False]),
    )
    args.update(kw)
    return PlanSpec(**args)


def test_ramp_ends():
    assert ramp(0.0) == (25, 50, 130)
    assert ramp(1.0) == (190, 35, 40)
    assert ramp(-1.0) == ramp(0.0)
    assert ramp(2.0) == ramp(1.0)


def test_colour_scale_clamps_and_log_zero():
    linear = ColourScale(0.0, 100.0, "%")
    assert linear.fraction(-5.0) == 0.0
    assert linear.fraction(150.0) == 1.0
    assert linear.fraction(50.0) == pytest.approx(0.5)
    log = ColourScale(10.0, 10000.0, "lux", log=True)
    assert log.fraction(0.0) == 0.0  # zero lux on a log scale clamps to the bottom
    assert log.fraction(100.0) == pytest.approx(1.0 / 3.0)
    assert log.ticks() == (10.0, 100.0, 1000.0, 10000.0)


def test_colour_scale_refuses_nan():
    with pytest.raises(FigureError, match="not a finite number"):
        ColourScale(0.0, 100.0, "%").fraction(math.nan)


@pytest.mark.parametrize("kw, message", [
    (dict(vmin=1.0, vmax=1.0), "vmax"),
    (dict(vmin=0.0, vmax=10.0, log=True), "log"),
    (dict(vmin=math.nan, vmax=1.0), "finite"),
])
def test_colour_scale_refusals(kw, message):
    with pytest.raises(FigureError, match=message):
        ColourScale(unit="x", **kw)


def test_plan_spec_refuses_values_and_passed_together():
    with pytest.raises(FigureError, match="either"):
        _pass_spec(values=np.array([1.0, 2.0]), scale=ColourScale(0.0, 1.0, "x"))


def test_plan_spec_refuses_values_without_a_scale():
    with pytest.raises(FigureError, match="scale"):
        _pass_spec(passed=None, values=np.array([1.0, 2.0]))


def test_plan_spec_refuses_mismatched_lengths():
    with pytest.raises(FigureError, match="2 points but 3"):
        _pass_spec(passed=np.array([True, False, True]))


def test_plan_spec_refuses_nothing_to_draw():
    spec = _pass_spec(outlines=(), labels=(), points_xy=np.zeros((0, 2)), passed=np.zeros(0, bool))
    with pytest.raises(FigureError, match="nothing to draw"):
        plan_canvas(spec)


def test_plan_canvas_places_every_point_inside_the_plan_area():
    canvas, layout = plan_canvas(_pass_spec())
    for x, y in POINTS:
        px, py = layout.to_px(x, y)
        assert 0 < px < canvas.width
        assert layout.plan_top < py < canvas.height
    fills = [i.fill for i in canvas.items if isinstance(i, Rect)]
    assert PASS_RGB in fills and FAIL_RGB in fills


def test_plan_canvas_labels_rooms_and_titles():
    canvas, _ = plan_canvas(_pass_spec())
    texts = [i.text for i in canvas.items if isinstance(i, Text)]
    assert "Test plan" in texts
    assert "Room North" in texts
    assert "N" in texts


@pytest.mark.parametrize("rotation, west", [(90.0, True), (-90.0, False)])
def test_north_arrow_turns_with_the_rotation(rotation, west):
    canvas, _ = plan_canvas(_pass_spec(north_rotation_deg=rotation))
    (arrow,) = [i for i in canvas.items if isinstance(i, Poly) and i.tag == "north"]
    tip_x = arrow.points[0][0]
    centre_x = sum(p[0] for p in arrow.points[1:]) / 3.0
    # north_rotation_deg is true north from the model +Y axis, counter-clockwise
    # positive, so 90 deg puts true north at -X: the tip points left.
    assert (tip_x < centre_x) is west


def test_heat_map_colours_follow_the_scale():
    spec = _pass_spec(passed=None, values=np.array([0.0, 100.0]), scale=ColourScale(0.0, 100.0, "%", marks=(50.0,)))
    canvas, _ = plan_canvas(spec)
    fills = [i.fill for i in canvas.items if isinstance(i, Rect)]
    assert ramp(0.0) in fills and ramp(1.0) in fills


def test_bar_canvas_has_one_bar_per_category():
    spec = BarSpec(title="Bars", categories=("L0", "L1"), values=(40.0, 80.0), unit="%", threshold=55.0)
    canvas = bar_canvas(spec)
    assert (canvas.width, canvas.height) == (1200, 600)
    bars = [i for i in canvas.items if isinstance(i, Rect) and i.fill == (30, 130, 200)]
    assert len(bars) == 2
    assert bars[1].h > bars[0].h


def test_bar_spec_refuses_mismatched_lengths():
    with pytest.raises(FigureError, match="2 categories but 1 value"):
        BarSpec(title="Bars", categories=("L0", "L1"), values=(40.0,), unit="%")


# -- Final review I4 (Ruling FR-I4): legible and on the page when printed ---

def _printed_pt(size_px: float, canvas) -> float:
    """A text size in points, with the canvas printed PRINT_WIDTH_CM wide."""
    from leed_daylight.deliver.figures.spec import PRINT_WIDTH_CM
    return size_px / canvas.width * PRINT_WIDTH_CM / 2.54 * 72.0


def _tall_spec(**kw):
    floor = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 80.0], [0.0, 80.0]])
    xs, ys = np.meshgrid(np.arange(1.0, 20.0, 2.0), np.arange(1.0, 80.0, 2.0))
    points = np.column_stack([xs.ravel(), ys.ravel()])
    return _pass_spec(outlines=(floor,), points_xy=points, labels=(("Room North", 10.0, 40.0),),
                      passed=np.ones(points.shape[0], dtype=bool), **kw)


@pytest.mark.parametrize("make", [
    lambda: plan_canvas(_pass_spec(segments=(("G1", np.array([[2.0, 0.0], [4.0, 0.0]])),)))[0],
    lambda: plan_canvas(_pass_spec(passed=None, values=np.array([10.0, 90.0]),
                                   scale=ColourScale(0.0, 100.0, "% of hours", marks=(50.0,))))[0],
    lambda: plan_canvas(_tall_spec())[0],
    lambda: bar_canvas(BarSpec("sDA by storey", ("L0", "L1"), (40.0, 60.0), "%", threshold=40.0)),
], ids=["plan", "heat map", "tall plan", "bar chart"])
def test_every_figure_text_is_at_least_7_pt_when_printed(make):
    canvas = make()

    texts = [item for item in canvas.items if isinstance(item, Text)]
    assert texts
    smallest = min(texts, key=lambda t: t.size)
    assert _printed_pt(smallest.size, canvas) >= 7.0, smallest


def test_a_tall_floor_plan_fits_the_page():
    """A 20 m x 80 m floor: printed at the usable width, the whole figure
    stays within the maximum figure height."""
    from leed_daylight.deliver.figures.spec import MAX_PRINT_HEIGHT_CM, PRINT_WIDTH_CM

    canvas, layout = plan_canvas(_tall_spec())

    assert canvas.height / canvas.width * PRINT_WIDTH_CM <= MAX_PRINT_HEIGHT_CM + 1e-9
    for item in canvas.items:
        if isinstance(item, Rect):
            assert 0.0 <= item.y and item.y + item.h <= canvas.height + 1e-6


# -- Task 11: dashed roof-glass plan polygons --------------------------------

_SQUARE = ((1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0))


def test_a_dashed_polygon_becomes_a_dashed_poly_with_its_label():
    canvas, layout = plan_canvas(_pass_spec(polygons=(PlanPolygon(_SQUARE, True, "Room_North_sky"),)))

    (poly,) = [i for i in canvas.items if isinstance(i, Poly) and i.dash]
    assert poly.stroke == SEGMENT_RGB and poly.fill is None
    assert poly.points == tuple(layout.to_px(x, y) for x, y in _SQUARE)
    assert poly.dash[0] > 0.0 and poly.dash[1] > 0.0
    (label,) = [i for i in canvas.items if isinstance(i, Text) and i.text == "Room_North_sky"]
    assert label.colour == SEGMENT_RGB


def test_a_solid_polygon_has_no_dash():
    canvas, _ = plan_canvas(_pass_spec(polygons=(PlanPolygon(_SQUARE, False, "G1"),)))

    (poly,) = [i for i in canvas.items if isinstance(i, Poly) and i.stroke == SEGMENT_RGB]
    assert poly.dash == ()


def test_a_polygon_outside_the_rooms_widens_the_plan():
    far = ((20.0, 1.0), (21.0, 1.0), (21.0, 2.0), (20.0, 2.0))
    canvas, layout = plan_canvas(_pass_spec(polygons=(PlanPolygon(far, True, "G2"),)))

    for x, y in far:
        px, py = layout.to_px(x, y)
        assert 0.0 <= px <= canvas.width
        assert layout.plan_top <= py <= canvas.height


def test_a_polygon_needs_two_points():
    with pytest.raises(FigureError, match="at least 2 points"):
        PlanPolygon(((0.0, 0.0),), True, "G3")


def test_dash_segments_follow_the_pattern_round_a_ring():
    segments = dash_segments(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)), 3.0, 2.0)

    assert len(segments) == 8
    assert sum(math.dist(a, b) for a, b in segments) == pytest.approx(24.0)
    assert np.allclose(segments[0], ((0.0, 0.0), (3.0, 0.0)))
    assert np.allclose(segments[1], ((5.0, 0.0), (8.0, 0.0)))
    assert np.allclose(segments[2], ((10.0, 0.0), (10.0, 3.0)))


def test_dash_segments_carry_the_pattern_across_a_corner():
    segments = dash_segments(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)), 4.0, 2.0, closed=False)

    assert np.allclose(np.array(segments), np.array([
        ((0.0, 0.0), (4.0, 0.0)), ((6.0, 0.0), (10.0, 0.0)),
        ((10.0, 2.0), (10.0, 6.0)), ((10.0, 8.0), (10.0, 10.0)),
    ]))


def test_dash_segments_refuse_a_zero_dash():
    with pytest.raises(FigureError, match="dash"):
        dash_segments(((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0)
