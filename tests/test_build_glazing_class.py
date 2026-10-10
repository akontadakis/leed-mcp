# tests/test_build_glazing_class.py
"""Tests for leed_daylight.build.glazing_class (skylights spec 5.1).

Every normal is built from an invented tilt: a pane tilted t degrees from
flat has the normal (sin t, 0, cos t), or its negation when drawn facing
down.
"""
from __future__ import annotations

import dataclasses
import math

import pytest

from leed_daylight.build.glazing_class import (
    KINDS,
    NON_VERTICAL_TILT_CLASSES,
    POSITIONS,
    SKYLIGHT_MAX_TILT_DEG,
    TILT_CLASSES,
    VERTICAL_MIN_TILT_DEG,
    PaneClass,
    classify,
    kind_of,
    tilt_class_of,
    tilt_deg,
)


def normal_at(tilt, *, down=False, azimuth=0.0):
    t, a = math.radians(tilt), math.radians(azimuth)
    n = (math.sin(t) * math.sin(a), math.sin(t) * math.cos(a), math.cos(t))
    return tuple(-v for v in n) if down else n


def test_the_thresholds_and_vocabularies():
    assert VERTICAL_MIN_TILT_DEG == 85.0
    assert SKYLIGHT_MAX_TILT_DEG == 20.0
    assert TILT_CLASSES == ("vertical", "sloped", "skylight")
    assert KINDS == ("window", "sloped", "monitor", "skylight", "diffusing")
    assert POSITIONS == ("perimeter", "roof")


def test_the_non_vertical_tilt_classes_are_sloped_and_skylight():
    assert NON_VERTICAL_TILT_CLASSES == ("sloped", "skylight")
    # Derived from TILT_CLASSES, not a second hand-typed literal.
    assert NON_VERTICAL_TILT_CLASSES == TILT_CLASSES[1:]


@pytest.mark.parametrize("tilt, expected", [
    (0.0, "skylight"),
    (19.9, "skylight"),
    (20.0, "sloped"),
    (84.9, "sloped"),
    (85.0, "vertical"),
    (90.0, "vertical"),
])
@pytest.mark.parametrize("down", [False, True])
def test_each_tilt_lands_in_its_class(tilt, expected, down):
    assert tilt_deg(normal_at(tilt, down=down)) == tilt
    assert classify(normal_at(tilt, down=down), "roof", False).tilt_class == expected


def test_the_tilt_does_not_depend_on_the_normal_length_or_azimuth():
    assert tilt_deg((0.0, 0.0, 5.0)) == 0.0
    assert tilt_deg((0.0, -3.0, 0.0)) == 90.0
    assert tilt_deg(normal_at(30.0, azimuth=135.0)) == 30.0


@pytest.mark.parametrize("bad", [(0.0, 0.0, 0.0), (1.0, 0.0), (float("nan"), 0.0, 1.0), "up"])
def test_a_normal_that_is_not_a_direction_is_refused(bad):
    with pytest.raises(ValueError, match="normal"):
        tilt_deg(bad)


@pytest.mark.parametrize("bad", [-1.0, 90.5])
def test_a_tilt_outside_0_to_90_is_refused(bad):
    with pytest.raises(ValueError, match="tilt"):
        tilt_class_of(bad)


@pytest.mark.parametrize("tilt_class, position, diffusing, expected", [
    ("vertical", "perimeter", False, "window"),
    ("vertical", "roof", False, "monitor"),
    ("sloped", "perimeter", False, "sloped"),
    ("sloped", "roof", False, "sloped"),
    ("skylight", "roof", False, "skylight"),
    ("vertical", "perimeter", True, "diffusing"),
    ("sloped", "roof", True, "diffusing"),
    ("skylight", "roof", True, "diffusing"),
])
def test_the_kind_follows_class_position_and_flag(tilt_class, position, diffusing, expected):
    assert kind_of(tilt_class, position, diffusing) == expected


def test_an_unknown_class_or_position_is_refused():
    with pytest.raises(ValueError, match="tilt class"):
        kind_of("flat", "roof", False)
    with pytest.raises(ValueError, match="position"):
        kind_of("skylight", "attic", False)


def test_classify_gives_a_frozen_pane_class():
    pane = classify(normal_at(15.0, down=True), "roof", True)
    assert pane == PaneClass(
        tilt_deg=15.0, tilt_class="skylight", position="roof", diffusing=True, kind="diffusing",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        pane.kind = "window"


def test_a_wall_window_is_a_vertical_perimeter_window():
    pane = classify((0.0, -1.0, 0.0), "perimeter", False)
    assert (pane.tilt_deg, pane.tilt_class, pane.kind) == (90.0, "vertical", "window")
