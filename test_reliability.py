"""Tests for hiq_score.reliability"""
import math
import pytest
from hiq_score.reliability import reliability_se, spearman_brown, min_questions


def test_se_decreases_with_n():
    assert reliability_se(100) > reliability_se(500)


def test_se_formula():
    # SE = 15 / (s * sqrt(n))
    # With s=0.0321, n=100: SE = 15 / (0.0321 * 10) = 15/0.321 ≈ 46.73
    # This is the SE in the same units as HIQ (HIQ points)
    expected = 15.0 / (0.0321 * math.sqrt(100))
    assert reliability_se(100) == pytest.approx(expected, rel=1e-6)


def test_se_large_n():
    # With 10,000 questions SE should be small
    assert reliability_se(10000) < 5.0


def test_spearman_brown_doubles():
    # Doubling a test with r=0.70 → 0.824
    assert spearman_brown(0.70) == pytest.approx(0.824, abs=0.001)


def test_spearman_brown_triple():
    assert spearman_brown(0.80, factor=3) == pytest.approx(0.923, abs=0.001)


def test_spearman_brown_bad_r():
    with pytest.raises(ValueError):
        spearman_brown(1.0)


def test_min_questions_achieves_target():
    n = min_questions(target_se=5)
    # Verify that n questions achieves the target SE
    assert reliability_se(n) <= 5.0


def test_min_questions_monotone():
    # Tighter SE → more questions needed
    assert min_questions(target_se=3) > min_questions(target_se=5)
