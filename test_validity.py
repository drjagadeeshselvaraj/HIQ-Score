"""Tests for hiq_score.validity"""
import pytest
from hiq_score.validity import (
    contamination_delta, corrected_hiq,
    norm_drift, drift_corrected_hiq,
    CONTAMINATION_SLOPE,
)


def test_contamination_slope():
    assert contamination_delta(1.0) == pytest.approx(CONTAMINATION_SLOPE)


def test_contamination_zero():
    assert contamination_delta(0.0) == 0.0


def test_contamination_five_percent():
    assert contamination_delta(0.05) == pytest.approx(20.4)


def test_contamination_bad_delta():
    with pytest.raises(ValueError):
        contamination_delta(1.1)


def test_corrected_hiq():
    assert corrected_hiq(130.0, delta=0.05) == pytest.approx(109.6)


def test_norm_drift_one_year():
    assert norm_drift(1) == pytest.approx(3.0)


def test_norm_drift_two_years_high():
    assert norm_drift(2, rate=3.75) == pytest.approx(7.5)


def test_drift_corrected():
    assert drift_corrected_hiq(115.0, years=2) == pytest.approx(109.0)
