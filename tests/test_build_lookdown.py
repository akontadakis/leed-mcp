# tests/test_build_lookdown.py
"""Tests for leed_daylight.build.lookdown (skylights spec 5.3, 5.6).

An invented 10 x 10 m plan: a floor at z = 0, a roof at z = 3, walls
3 m high. Every number is round.
"""
from __future__ import annotations

import numpy as np
import pytest

from leed_daylight.build.lookdown import first_hit_below


def square(x0, y0, x1, y1, z):
    return np.array([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], dtype=np.float64)


FLOOR = ("floor", square(0, 0, 10, 10, 0.0))
ROOF = ("roof", square(0, 0, 10, 10, 3.0))


def test_a_ray_lands_on_the_floor_below():
    name, hit = first_hit_below((5.0, 5.0, 2.0), [FLOOR])
    assert name == "floor"
    assert hit == pytest.approx([5.0, 5.0, 0.0])


def test_the_nearest_polygon_below_wins():
    name, hit = first_hit_below((5.0, 5.0, 4.0), [FLOOR, ROOF])
    assert name == "roof"
    assert hit[2] == pytest.approx(3.0)


def test_a_polygon_above_the_origin_is_never_hit():
    name, _ = first_hit_below((5.0, 5.0, 2.0), [ROOF, FLOOR])
    assert name == "floor"


def test_a_polygon_at_the_origin_height_is_not_below_it():
    assert first_hit_below((5.0, 5.0, 0.0), [FLOOR]) is None


def test_a_ray_beside_every_polygon_hits_nothing():
    assert first_hit_below((15.0, 5.0, 2.0), [FLOOR, ROOF]) is None
    assert first_hit_below((5.0, 5.0, 2.0), []) is None


def test_a_vertical_polygon_is_never_hit_even_when_the_ray_runs_along_it():
    wall = ("wall", np.array([(5, 0, 0), (5, 10, 0), (5, 10, 3), (5, 0, 3)], dtype=np.float64))
    assert first_hit_below((5.0, 5.0, 2.0), [wall]) is None


def test_the_winding_of_a_polygon_does_not_matter():
    down_facing = ("floor", FLOOR[1][::-1])
    name, _ = first_hit_below((5.0, 5.0, 2.0), [down_facing])
    assert name == "floor"


def test_a_tilted_polygon_is_hit_at_its_own_height():
    # A 4 x 10 m ramp rising 0.5 m for every metre of x.
    ramp = ("ramp", np.array([(0, 0, 0), (4, 0, 2), (4, 10, 2), (0, 10, 0)], dtype=np.float64))
    name, hit = first_hit_below((2.0, 5.0, 3.0), [ramp])
    assert name == "ramp"
    assert hit == pytest.approx([2.0, 5.0, 1.0])


def test_a_ray_down_a_shared_edge_hits_the_first_polygon_listed():
    west = ("west", square(0, 0, 5, 10, 0.0))
    east = ("east", square(5, 0, 10, 10, 0.0))
    assert first_hit_below((5.0, 5.0, 2.0), [west, east])[0] == "west"
    assert first_hit_below((5.0, 5.0, 2.0), [east, west])[0] == "east"


def test_a_hole_in_an_l_shaped_floor_is_missed():
    # An L: the 5 x 5 m square at the north-east corner is not floor.
    l_floor = ("l_floor", np.array(
        [(0, 0, 0), (10, 0, 0), (10, 5, 0), (5, 5, 0), (5, 10, 0), (0, 10, 0)], dtype=np.float64
    ))
    assert first_hit_below((7.5, 7.5, 2.0), [l_floor]) is None
    assert first_hit_below((2.5, 7.5, 2.0), [l_floor])[0] == "l_floor"


@pytest.mark.parametrize("bad", [(5.0, 5.0), (5.0, float("nan"), 2.0)])
def test_a_malformed_origin_is_refused(bad):
    with pytest.raises(ValueError, match="origin"):
        first_hit_below(bad, [FLOOR])


def test_a_malformed_polygon_is_refused_by_name():
    with pytest.raises(ValueError, match="broken"):
        first_hit_below((5.0, 5.0, 2.0), [("broken", np.array([(0, 0, 0), (1, 0, 0)]))])
