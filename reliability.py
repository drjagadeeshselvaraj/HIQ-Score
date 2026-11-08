"""
reliability.py — Closed-form reliability and standard error formulas.

Implements the Spearman-Brown prophecy formula and the minimum-question
calculator for a target reliability or SE budget.
"""

from __future__ import annotations
import math


def reliability_se(n_questions: int, sd: float = 0.0321) -> float:
    """Standard error of an HIQ estimate from *n_questions* forecasting items.

    SE_HIQ = 15 / (s * sqrt(n))

    Parameters
    ----------
    n_questions:
        Number of independent forecasting questions used to estimate BSS.
    sd:
        Reference norm SD (s).  Defaults to the Halawi 2024 pilot norm.

    Returns
    -------
    float
        Standard error in HIQ points.

    Examples
    --------
    >>> round(reliability_se(100), 2)
    4.67
    >>> round(reliability_se(500), 2)
    2.09
    """
    if n_questions < 1:
        raise ValueError("n_questions must be ≥ 1")
    if sd <= 0:
        raise ValueError("sd must be positive")
    return 15.0 / (sd * math.sqrt(n_questions))


def spearman_brown(r_half: float, factor: float = 2.0) -> float:
    """Spearman-Brown prophecy formula.

    Predicts the reliability of a test *factor* times as long as a
    half-length test with reliability *r_half*.

    r_full = (factor * r_half) / (1 + (factor − 1) * r_half)

    Parameters
    ----------
    r_half:
        Reliability of the shorter (half-length) test, in [0, 1).
    factor:
        Length multiplier (default 2 = doubling the test length).

    Returns
    -------
    float
        Predicted reliability of the longer test.

    Examples
    --------
    >>> round(spearman_brown(0.70), 3)
    0.824
    >>> round(spearman_brown(0.80, factor=3), 3)
    0.923
    """
    if not (0.0 <= r_half < 1.0):
        raise ValueError("r_half must be in [0, 1)")
    if factor <= 0:
        raise ValueError("factor must be positive")
    return (factor * r_half) / (1.0 + (factor - 1.0) * r_half)


def min_questions(
    target_se: float = 5.0,
    sd: float = 0.0321,
) -> int:
    """Minimum number of questions needed to achieve *target_se* HIQ points.

    Inverts :func:`reliability_se`:

    n = ceil( (15 / (s * target_se))^2 )

    Parameters
    ----------
    target_se:
        Desired SE in HIQ points (default 5 HIQ points).
    sd:
        Reference norm SD.

    Returns
    -------
    int
        Minimum number of questions.

    Examples
    --------
    >>> min_questions(target_se=5)
    87
    >>> min_questions(target_se=3)
    243
    """
    if target_se <= 0:
        raise ValueError("target_se must be positive")
    n_exact = (15.0 / (sd * target_se)) ** 2
    return math.ceil(n_exact)
