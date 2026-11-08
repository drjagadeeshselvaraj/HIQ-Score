"""Tests for hiq_score.norm"""
import math
import pytest
from hiq_score.norm import loo_norm, jackknife_se


def test_loo_norm_basic_properties():
    """LOO-CV norm with symmetric data should produce sensible m/s."""
    bss = [0.10, 0.12, 0.14, 0.16, 0.18]
    r = loo_norm(bss)
    # Grand LOO mean should equal sample mean for symmetric data
    assert r.mean == pytest.approx(sum(bss) / len(bss), abs=0.001)
    assert r.sd > 0


def test_loo_norm_hiq_range():
    """Systems should straddle 100 when the dataset is spread."""
    bss = [0.10, 0.12, 0.14, 0.16, 0.18]
    r = loo_norm(bss)
    assert min(r.loo_hiq) < 100.0
    assert max(r.loo_hiq) > 100.0


def test_loo_norm_requires_three():
    with pytest.raises(ValueError):
        loo_norm([0.1, 0.2])


def test_loo_norm_returns_correct_count():
    bss = [0.10, 0.12, 0.14, 0.16, 0.18]
    r = loo_norm(bss)
    assert len(r.loo_hiq) == 5
    assert len(r.loo_bss) == 5


def test_loo_norm_se_positive():
    bss = [0.10, 0.12, 0.14, 0.16, 0.18]
    r = loo_norm(bss)
    assert r.se_mean > 0
    assert r.se_sd > 0


def test_jackknife_se_constant():
    # Constant values → SE should be 0
    assert jackknife_se([5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0, abs=1e-10)


def test_jackknife_se_positive():
    se = jackknife_se([0.12, 0.13, 0.11, 0.14, 0.10])
    assert se > 0
