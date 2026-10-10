# tests/test_option2.py
import numpy as np
import pytest

from leed_daylight.metrics.option2 import BandCounts, band_counts, band_pass, two_day_average


def test_band_pass_is_point_major():
    lux = np.array([[200.0], [300.0], [1500.0], [3000.0], [3100.0]])
    assert band_pass(lux, 300.0, 3000.0).tolist() == [False, True, True, True, False]


def test_a_point_must_be_in_band_at_every_time():
    # This mask is deliberately non-symmetric ([[True, False], [True, True]]):
    # a symmetric (2, 2) case cannot tell a points-major axis=1 reduction apart
    # from a times-major axis=0 one, so it could pass on the transposed bug.
    lux = np.array([[400.0, 4000.0], [400.0, 500.0]])
    assert band_pass(lux, 300.0, 3000.0).tolist() == [False, True]


def test_both_band_edges_are_inclusive():
    lux = np.array([[300.0, 3000.0]])
    assert band_pass(lux, 300.0, 3000.0).tolist() == [True]


def test_a_single_time_is_allowed():
    assert band_pass(np.array([[400.0]]), 300.0, 3000.0).tolist() == [True]


def test_band_pass_rejects_zero_times():
    with pytest.raises(ValueError, match="at least one time"):
        band_pass(np.zeros((5, 0)), 300.0, 3000.0)


def test_band_pass_rejects_a_single_dimension():
    with pytest.raises(ValueError, match="two dimensions") as excinfo:
        band_pass(np.array([300.0, 400.0]), 300.0, 3000.0)
    assert "(points, times)" in str(excinfo.value)


def test_inverted_band_is_refused():
    with pytest.raises(ValueError, match="lux_min"):
        band_pass(np.array([[400.0]]), 3000.0, 300.0)


def test_negative_illuminance_is_refused():
    with pytest.raises(ValueError, match="negative"):
        band_pass(np.array([[-1.0]]), 300.0, 3000.0)


def test_two_day_average_is_the_mean_per_point_and_time():
    march = np.array([[200.0, 500.0], [100.0, 4000.0]])
    september = np.array([[600.0, 500.0], [300.0, 2000.0]])
    assert np.array_equal(two_day_average(march, september), [[400.0, 500.0], [200.0, 3000.0]])


def test_two_day_average_refuses_different_shapes():
    with pytest.raises(ValueError, match="shape"):
        two_day_average(np.zeros((2, 2)), np.zeros((3, 2)))


def test_band_counts_counts_passing_low_and_high():
    lux = np.array([
        [400.0, 500.0],    # passes
        [200.0, 500.0],    # low at the first time
        [400.0, 3500.0],   # high at the second time
        [200.0, 3500.0],   # low and high: counted in both
    ])
    counts = band_counts(lux, 300.0, 3000.0)
    assert isinstance(counts, BandCounts)
    assert counts.passing.tolist() == [True, False, False, False]
    assert counts.n_low == 2
    assert counts.n_high == 2


def test_band_counts_without_an_upper_limit():
    lux = np.array([[400.0, 9000.0], [200.0, 9000.0]])
    counts = band_counts(lux, 300.0, None)
    assert counts.passing.tolist() == [True, False]
    assert counts.n_low == 1
    assert counts.n_high == 0


def test_band_counts_edges_pass():
    counts = band_counts(np.array([[300.0, 3000.0]]), 300.0, 3000.0)
    assert counts.passing.tolist() == [True]
