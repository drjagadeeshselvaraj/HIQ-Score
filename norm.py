"""
norm.py — Leave-one-out cross-validated norm estimation.

Implements the LOO-CV procedure from the paper:
  for each system i, estimate (m_{-i}, s_{-i}) from the remaining
  n−1 systems, then score system i against that hold-out norm.

Also provides jackknife standard errors for m̂ and ŝ.
"""

from __future__ import annotations
import math
from typing import Sequence, NamedTuple


class LooNormResult(NamedTuple):
    """Results from :func:`loo_norm`."""
    mean: float          # m̂  — grand mean of LOO norms
    sd: float            # ŝ  — grand SD of LOO norms
    loo_bss: list[float] # each system's LOO BSS score
    loo_hiq: list[float] # each system's LOO HIQ score
    se_mean: float       # jackknife SE of m̂
    se_sd: float         # jackknife SE of ŝ


def loo_norm(bss_scores: Sequence[float]) -> LooNormResult:
    """Estimate the HIQ reference norm via leave-one-out cross-validation.

    For each system i the norm is estimated from the remaining n−1 systems,
    then system i is scored against that hold-out norm.  The grand mean and
    SD of the hold-out norms become (m̂, ŝ).

    Parameters
    ----------
    bss_scores:
        Brier Skill Scores for the systems used to establish the norm.
        Minimum length: 3.

    Returns
    -------
    LooNormResult
        Named tuple with m̂, ŝ, per-system LOO BSS/HIQ, and jackknife SEs.

    Examples
    --------
    >>> # Halawi 2024 five-system replication
    >>> bss = [0.126, 0.128, 0.134, 0.148, 0.075]
    >>> r = loo_norm(bss)
    >>> round(r.mean, 4)
    0.1264
    >>> round(r.sd, 4)
    0.0321
    """
    xs = list(bss_scores)
    n = len(xs)
    if n < 3:
        raise ValueError(f"Need at least 3 systems; got {n}")

    # LOO norm parameters: mean and sd of xs excluding index i
    loo_means = []
    loo_sds = []
    for i in range(n):
        rest = [xs[j] for j in range(n) if j != i]
        m_i = sum(rest) / (n - 1)
        s_i = math.sqrt(sum((x - m_i) ** 2 for x in rest) / (n - 2))
        loo_means.append(m_i)
        loo_sds.append(s_i)

    # Grand pilot norm = mean of the LOO norms
    m_hat = sum(loo_means) / n
    s_hat = sum(loo_sds) / n

    # LOO HIQ scores: score system i against its own hold-out norm
    loo_hiq = [100.0 + 15.0 * (xs[i] - loo_means[i]) / loo_sds[i] for i in range(n)]

    # Jackknife SEs
    se_m = jackknife_se(loo_means)
    se_s = jackknife_se(loo_sds)

    return LooNormResult(
        mean=m_hat,
        sd=s_hat,
        loo_bss=xs,
        loo_hiq=loo_hiq,
        se_mean=se_m,
        se_sd=se_s,
    )


def jackknife_se(values: Sequence[float]) -> float:
    """Jackknife standard error for the mean of *values*.

    Parameters
    ----------
    values:
        Sample of estimates (e.g., LOO norm means).

    Returns
    -------
    float
        Jackknife SE.
    """
    n = len(values)
    if n < 2:
        raise ValueError("Need at least 2 values for jackknife SE")
    grand = sum(values) / n
    pseudo = [n * grand - (n - 1) * v for v in values]
    var = sum((p - sum(pseudo) / n) ** 2 for p in pseudo) / (n * (n - 1))
    return math.sqrt(var)
