"""
core.py — HIQ scoring formula.

HIQ = 100 + 15 * (Φ − m) / s

where Φ is the system's Brier Skill Score (BSS),
m is the reference norm mean, and s is the reference norm SD.
"""

from __future__ import annotations
import math
from typing import Sequence


# ── Pilot norms (Halawi et al. 2024, LOO-CV) ─────────────────────────────────
HALAWI_NORM_MEAN: float = 0.1264   # m̂  (LOO-CV, n=5)
HALAWI_NORM_SD:   float = 0.0321   # ŝ  (LOO-CV, n=5)

# ── ForecastBench pilot norms (Karger et al. 2025) ───────────────────────────
FORECASTBENCH_NORM_MEAN: float = 0.4728
FORECASTBENCH_NORM_SD:   float = 0.0396


def hiq(
    bss: float,
    mean: float = HALAWI_NORM_MEAN,
    sd: float = HALAWI_NORM_SD,
) -> float:
    """Return the HIQ score for a system with Brier Skill Score *bss*.

    Parameters
    ----------
    bss:
        Brier Skill Score of the system being evaluated.  Must be in
        [−∞, 1]; higher is better (1 = perfect, 0 = climatology baseline).
    mean:
        Reference norm mean (m).  Defaults to the Halawi 2024 LOO-CV
        pilot norm (m̂ = 0.1264).
    sd:
        Reference norm standard deviation (s).  Defaults to the Halawi
        2024 LOO-CV pilot norm (ŝ = 0.0321).

    Returns
    -------
    float
        HIQ score on a scale with population mean 100 and SD 15.

    Examples
    --------
    >>> hiq(0.1264)   # exactly at the norm mean
    100.0
    >>> round(hiq(0.148), 1)   # Claude-3-Haiku full-sample
    75.5
    >>> round(hiq(0.496), 1)   # GPT-4-Turbo (ForecastBench)
    110.1
    """
    if sd <= 0:
        raise ValueError(f"sd must be positive, got {sd}")
    return 100.0 + 15.0 * (bss - mean) / sd


def hiq_from_brier(
    brier: float,
    baseline_brier: float,
    mean: float = HALAWI_NORM_MEAN,
    sd: float = HALAWI_NORM_SD,
) -> float:
    """Return HIQ from raw Brier scores by first computing BSS.

    BSS = 1 − Brier / Brier_baseline

    Parameters
    ----------
    brier:
        System's mean Brier score (lower is better).
    baseline_brier:
        Climatology (base-rate) Brier score for the same question set.
    mean, sd:
        Reference norm parameters (see :func:`hiq`).

    Examples
    --------
    >>> round(hiq_from_brier(brier=0.18, baseline_brier=0.25), 1)
    104.3
    """
    if baseline_brier <= 0:
        raise ValueError("baseline_brier must be positive")
    bss = 1.0 - brier / baseline_brier
    return hiq(bss, mean=mean, sd=sd)


def hiq_batch(
    bss_scores: Sequence[float],
    mean: float = HALAWI_NORM_MEAN,
    sd: float = HALAWI_NORM_SD,
) -> list[float]:
    """Apply :func:`hiq` to a sequence of BSS values.

    Parameters
    ----------
    bss_scores:
        Iterable of Brier Skill Scores.

    Returns
    -------
    list[float]
        HIQ scores in the same order.
    """
    return [hiq(b, mean=mean, sd=sd) for b in bss_scores]
