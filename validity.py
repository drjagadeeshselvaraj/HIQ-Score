"""
validity.py — Analytic corrections for four validity threats.

Threats covered
---------------
1. Benchmark contamination  ΔHIQ = 408 δ
2. Stale norm drift         ~3.0–3.75 HIQ points per year
3. (Forking-path inflation  — qualitative; no formula implemented)
4. (Adaptivity Inversion    — qualitative; no formula implemented)
"""

from __future__ import annotations


# ── 1. Contamination correction ───────────────────────────────────────────────

CONTAMINATION_SLOPE: float = 408.0  # HIQ points per unit δ


def contamination_delta(delta: float) -> float:
    """HIQ inflation due to benchmark contamination.

    ΔHIQ = 408 δ

    Parameters
    ----------
    delta:
        Estimated contamination fraction δ ∈ [0, 1], where δ = 1 means
        all test items were seen verbatim during training.

    Returns
    -------
    float
        Upward bias in reported HIQ points.

    Examples
    --------
    >>> contamination_delta(0.05)
    20.4
    >>> contamination_delta(0.10)
    40.8
    """
    if not (0.0 <= delta <= 1.0):
        raise ValueError(f"delta must be in [0, 1], got {delta}")
    return CONTAMINATION_SLOPE * delta


def corrected_hiq(reported_hiq: float, delta: float) -> float:
    """Return HIQ corrected for contamination.

    Parameters
    ----------
    reported_hiq:
        Uncorrected HIQ score.
    delta:
        Contamination fraction (see :func:`contamination_delta`).

    Returns
    -------
    float
        Decontaminated HIQ score.

    Examples
    --------
    >>> round(corrected_hiq(130.0, delta=0.05), 1)
    109.6
    """
    return reported_hiq - contamination_delta(delta)


# ── 2. Norm drift correction ──────────────────────────────────────────────────

DRIFT_RATE_LOW:  float = 3.0    # HIQ points / year (conservative)
DRIFT_RATE_HIGH: float = 3.75   # HIQ points / year (aggressive)


def norm_drift(
    years: float,
    rate: float = DRIFT_RATE_LOW,
) -> float:
    """Expected HIQ score inflation due to norm drift over *years*.

    As frontier systems improve, a fixed norm grows stale: scores quoted
    against an old norm overstate the system's standing relative to the
    current frontier.

    Parameters
    ----------
    years:
        Number of years since the norm was established.
    rate:
        Drift rate in HIQ points per year.  Use ``DRIFT_RATE_LOW`` (3.0)
        for a conservative estimate or ``DRIFT_RATE_HIGH`` (3.75) for an
        aggressive one.

    Returns
    -------
    float
        Approximate upward bias in HIQ points.

    Examples
    --------
    >>> norm_drift(1)
    3.0
    >>> norm_drift(2, rate=3.75)
    7.5
    """
    if years < 0:
        raise ValueError("years must be non-negative")
    return rate * years


def drift_corrected_hiq(
    reported_hiq: float,
    years: float,
    rate: float = DRIFT_RATE_LOW,
) -> float:
    """HIQ corrected for norm drift.

    Parameters
    ----------
    reported_hiq:
        HIQ score quoted against an old norm.
    years:
        Age of the norm in years.
    rate:
        Drift rate (default conservative 3.0 HIQ/year).

    Examples
    --------
    >>> round(drift_corrected_hiq(115.0, years=2), 1)
    109.0
    """
    return reported_hiq - norm_drift(years, rate=rate)
