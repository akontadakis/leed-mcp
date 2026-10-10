# tests/test_option1.py
import numpy as np
import pytest

from leed_daylight.metrics.option1 import (
    ASE_LUX,
    SDA_LUX,
    ase_band,
    ase_exceeds,
    ase_hours,
    daylight_autonomy,
    sda_pass,
)


def test_daylight_autonomy_counts_hours_at_or_above_the_threshold():
    lux = np.array([[299.0, 300.0, 301.0, 0.0]])
    assert daylight_autonomy(lux, SDA_LUX) == pytest.approx([0.5])


def test_daylight_autonomy_is_one_when_always_lit():
    assert daylight_autonomy(np.full((3, 10), 500.0), SDA_LUX) == pytest.approx(
        [1.0, 1.0, 1.0]
    )


def test_sda_pass_returns_true_for_a_passing_point():
    da = np.array([0.49, 0.50, 0.51])
    assert sda_pass(da, fraction=0.50).tolist() == [False, True, True]


def test_sda_threshold_is_configurable():
    assert sda_pass(np.array([0.6]), fraction=0.75).tolist() == [False]


def test_ase_hours_counts_hours_at_or_above_1000_lux():
    direct = np.array([[999.0, 1000.0, 1001.0, 0.0]])
    assert ase_hours(direct, ASE_LUX).tolist() == [2]


def test_ase_exceeds_returns_true_for_an_exceeding_point():
    hours = np.array([250, 251])
    assert ase_exceeds(hours, limit=250).tolist() == [False, True]


def test_ase_limit_is_configurable():
    assert ase_exceeds(np.array([100]), limit=99).tolist() == [True]


def test_ase_bands_follow_lm83():
    # LM-83-12 3.1.3: under 3 % clearly acceptable, under 7 % neutral,
    # over 10 % unsatisfactory; 7-10 % is not classed by the standard.
    assert ase_band(0.0) == "under 3 %: clearly acceptable"
    assert ase_band(2.9) == "under 3 %: clearly acceptable"
    assert ase_band(3.0) == "3-7 %: neutral"
    assert ase_band(6.9) == "3-7 %: neutral"
    assert ase_band(7.0) == "7-10 %: not classified by LM-83-12"
    assert ase_band(10.0) == "7-10 %: not classified by LM-83-12"
    assert ase_band(10.1) == "over 10 %: unsatisfactory"


def test_ase_band_edges_tolerate_float_error():
    assert ase_band(3.0 - 1e-12) == "3-7 %: neutral"
    assert ase_band(10.0 + 1e-12) == "7-10 %: not classified by LM-83-12"


def test_a_one_dimensional_array_is_refused():
    with pytest.raises(ValueError, match="two dimensions"):
        daylight_autonomy(np.array([300.0, 400.0]), SDA_LUX)


def test_no_hours_is_refused():
    with pytest.raises(ValueError, match="no occupied hours"):
        daylight_autonomy(np.zeros((5, 0)), SDA_LUX)


def test_negative_illuminance_is_refused():
    with pytest.raises(ValueError, match="negative"):
        daylight_autonomy(np.array([[-1.0, 500.0]]), SDA_LUX)


def test_the_old_names_are_gone():
    with pytest.raises(ImportError):
        from leed_daylight.metrics.option1 import sda_points  # noqa: F401

    with pytest.raises(ImportError):
        from leed_daylight.metrics.option1 import ase_points  # noqa: F401


def test_daylight_autonomy_requires_a_threshold():
    with pytest.raises(TypeError):
        daylight_autonomy(np.array([[300.0, 400.0]]))


def test_sda_pass_requires_a_fraction():
    with pytest.raises(TypeError):
        sda_pass(np.array([0.5]))


def test_ase_hours_requires_a_threshold():
    with pytest.raises(TypeError):
        ase_hours(np.array([[1000.0, 0.0]]))


def test_ase_exceeds_requires_a_limit():
    with pytest.raises(TypeError):
        ase_exceeds(np.array([250]))
