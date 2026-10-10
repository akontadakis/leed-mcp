"""deliver.view_plans: one SVG plan of pass and fail points (QV spec 6.8)."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from leed_daylight.deliver.view_plans import plan_svg

NS = {"s": "http://www.w3.org/2000/svg"}


def _svg(**kw):
    args = dict(
        title="Storey 0, quality: 50.0 % of floor area",
        outlines=[np.array([[0, 0], [6, 0], [6, 4], [0, 4]], dtype=float)],
        points_xy=np.array([[1.0, 1.0], [5.0, 3.0]]),
        passed=np.array([True, False]),
        cell_m=0.6,
        north_rotation_deg=0.0,
    )
    args.update(kw)
    return ET.fromstring(plan_svg(**args))


def test_points_rooms_and_title():
    root = _svg()

    rects = root.findall("s:rect", NS)
    assert [r.get("class") for r in rects] == ["pass", "fail"]
    assert len(root.findall("s:polygon", NS)) == 1
    assert "50.0 %" in root.find("s:title", NS).text


def test_north_is_up():
    rects = _svg().findall("s:rect", NS)

    assert float(rects[1].get("y")) < float(rects[0].get("y"))


def test_the_title_is_escaped():
    root = _svg(title="A & B <plan>")

    assert root.find("s:title", NS).text == "A & B <plan>"


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError, match="2 points but 1 pass flags"):
        plan_svg(title="t", outlines=[], points_xy=np.zeros((2, 2)), passed=np.array([True]), cell_m=0.6,
                 north_rotation_deg=0.0)


def _north_screen_direction(root):
    """The arrow is drawn pointing up (screen -y) and turned by the group's
    rotate(a); SVG turns clockwise on screen, so it points at (sin a, -cos a)."""
    (group,) = root.findall("s:g[@class='north']", NS)
    transform = group.get("transform")
    angle = float(transform.split("rotate(")[1].rstrip(")"))
    assert group.find("s:polygon", NS) is not None
    assert group.find("s:text", NS).text == "N"
    a = np.radians(angle)
    return np.array([np.sin(a), -np.cos(a)])


def test_the_north_arrow_points_up_at_zero_rotation():
    direction = _north_screen_direction(_svg(north_rotation_deg=0.0))

    assert direction == pytest.approx([0.0, -1.0])


def test_the_north_arrow_turns_counter_clockwise_with_the_north_rotation():
    # north_rotation_deg 90: true north is the model's -X (+Y turned 90 deg
    # counter-clockwise, as gendaymtx -r turns the sky), drawn to the left.
    direction = _north_screen_direction(_svg(north_rotation_deg=90.0))

    assert direction == pytest.approx([-1.0, 0.0])

