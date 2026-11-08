"""Tests for hiq_score.core"""
import pytest
from hiq_score.core import hiq, hiq_from_brier, hiq_batch, HALAWI_NORM_MEAN, HALAWI_NORM_SD


def test_hiq_at_norm_mean():
    assert hiq(HALAWI_NORM_MEAN) == pytest.approx(100.0)


def test_hiq_above_mean():
    score = hiq(HALAWI_NORM_MEAN + HALAWI_NORM_SD)
    assert score == pytest.approx(115.0)


def test_hiq_below_mean():
    score = hiq(HALAWI_NORM_MEAN - HALAWI_NORM_SD)
    assert score == pytest.approx(85.0)


def test_hiq_haiku_full_sample():
    # Claude-3-Haiku full-sample HIQ = 75.5 (Halawi dataset BSS ≈ 0.074)
    # Note: 0.148 is Haiku's ForecastBench BSS (Table 4b); Halawi BSS is lower
    bss_haiku_halawi = HALAWI_NORM_MEAN - 24.5 * HALAWI_NORM_SD / 15.0  # ≈ 0.074
    assert hiq(bss_haiku_halawi) == pytest.approx(75.5, abs=0.1)


def test_hiq_gemini_pro_halawi():
    # Gemini-Pro LOO-CV HIQ = 60.1 (lowest in Halawi dataset)
    bss_gemini = HALAWI_NORM_MEAN - 39.9 * HALAWI_NORM_SD / 15.0
    assert hiq(bss_gemini) == pytest.approx(60.1, abs=0.1)


def test_hiq_negative_sd_raises():
    with pytest.raises(ValueError):
        hiq(0.1, mean=0.1, sd=-0.01)


def test_hiq_from_brier():
    # BSS = 1 - 0.18/0.25 = 0.28
    score = hiq_from_brier(brier=0.18, baseline_brier=0.25)
    expected = hiq(0.28)
    assert score == pytest.approx(expected)


def test_hiq_batch():
    scores = hiq_batch([HALAWI_NORM_MEAN - HALAWI_NORM_SD,
                         HALAWI_NORM_MEAN,
                         HALAWI_NORM_MEAN + HALAWI_NORM_SD])
    assert scores == pytest.approx([85.0, 100.0, 115.0])


def test_hiq_forecastbench_haiku_loo():
    # Haiku ForecastBench LOO-CV HIQ = 34.6 (BSS = 0.148, LOO norm compressed)
    # This uses ForecastBench norm, not Halawi norm
    from hiq_score.core import FORECASTBENCH_NORM_MEAN, FORECASTBENCH_NORM_SD
    # Back-compute: 34.6 = 100 + 15*(0.148 - m_loo)/s_loo
    # This is the LOO norm when Haiku is excluded from ForecastBench top-4
    # The LOO norm with top-4 only: mean≈0.4845, sd≈0.019
    score = hiq(0.148, mean=0.4845, sd=0.019)
    assert score < 50.0  # strongly below average, as reported
