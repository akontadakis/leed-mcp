import numpy as np
import pytest

from leed_daylight.units import (
    LENGTH_TO_M,
    net_transmittance,
    rgb_to_lux,
    to_metres,
    transmissivity,
    transmittance_of,
)


def test_length_units_cover_the_declared_set():
    assert set(LENGTH_TO_M) == {"m", "mm", "cm", "ft", "in"}


@pytest.mark.parametrize(
    "value,units,expected",
    [(1.0, "m", 1.0), (1000.0, "mm", 1.0), (100.0, "cm", 1.0),
     # source: international foot, exactly 0.3048 m
     (1.0, "ft", 0.3048), (12.0, "in", 0.3048)],
)
def test_to_metres_scalar(value, units, expected):
    assert to_metres(value, units) == pytest.approx(expected, abs=1e-12)


def test_to_metres_array():
    out = to_metres(np.array([0.0, 1000.0, 2500.0]), "mm")
    assert np.allclose(out, [0.0, 1.0, 2.5])


def test_to_metres_rejects_unknown_unit():
    with pytest.raises(ValueError, match="unknown length unit"):
        to_metres(1.0, "furlong")


def test_net_transmittance_applies_the_dirt_factor():
    # Invented values, chosen round so the arithmetic is obvious.
    assert net_transmittance(0.80, 0.90) == pytest.approx(0.72)


def test_transmissivity_matches_the_radiance_formula():
    # A glass primitive takes transmissivity, not transmittance.
    # Both pairs are invented round values checked against the formula itself.
    # source: computed from the spec 6.8 Radiance formula
    assert transmissivity(0.50) == pytest.approx(0.5452, abs=5e-4)
    # source: computed from the spec 6.8 Radiance formula
    assert transmissivity(0.90) == pytest.approx(0.9801, abs=5e-4)


def test_transmissivity_is_always_above_its_transmittance():
    for tn in (0.05, 0.3, 0.6, 0.9):
        assert transmissivity(tn) > tn


def test_transmissivity_of_zero_is_zero():
    assert transmissivity(0.0) == 0.0


def test_transmissivity_rejects_out_of_range():
    with pytest.raises(ValueError, match="transmittance"):
        transmissivity(1.5)


def test_rgb_to_lux_uses_the_cie_weights():
    assert rgb_to_lux(np.array([1.0, 1.0, 1.0])) == pytest.approx(178.9)
    assert rgb_to_lux(np.array([1.0, 0.0, 0.0])) == pytest.approx(47.4)
    assert rgb_to_lux(np.array([0.0, 1.0, 0.0])) == pytest.approx(119.9)
    assert rgb_to_lux(np.array([0.0, 0.0, 1.0])) == pytest.approx(11.6)


def test_rgb_to_lux_removes_the_colour_axis():
    out = rgb_to_lux(np.zeros((4, 7, 3)))
    assert out.shape == (4, 7)


def test_rgb_to_lux_rejects_a_wrong_last_axis():
    with pytest.raises(ValueError, match="last axis"):
        rgb_to_lux(np.zeros((4, 2)))


@pytest.mark.parametrize("t", [0.05, 0.25, 0.50, 0.75, 0.95])
def test_the_transmittance_round_trip(t):
    assert transmittance_of(transmissivity(t)) == pytest.approx(t, abs=1e-9)


def test_transmittance_of_zero_is_zero():
    assert transmittance_of(0.0) == 0.0


def test_transmittance_of_the_upper_bound_is_one():
    upper = transmissivity(1.0)
    assert transmittance_of(upper) == pytest.approx(1.0, abs=1e-9)


def test_transmittance_of_rejects_a_negative_value():
    with pytest.raises(ValueError, match="between 0 and"):
        transmittance_of(-0.1)


def test_transmittance_of_rejects_a_value_above_the_domain():
    upper = transmissivity(1.0)
    with pytest.raises(ValueError, match="between 0 and"):
        transmittance_of(upper + 0.1)


def test_transmittance_of_names_the_value_it_saw():
    with pytest.raises(ValueError, match="1.5"):
        transmittance_of(1.5)
