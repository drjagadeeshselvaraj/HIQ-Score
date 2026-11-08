"""
meridian.py — Meridian four-component skill decomposition.

Decomposes a system's Brier Skill Score into four orthogonal components:

  Reach        — proportion of questions attempted
  Efficiency   — mean BSS on attempted questions
  Adaptivity   — correlation between confidence and accuracy
  Breadth      — coverage across question domains

The overall BSS ≈ Reach × Efficiency (Adaptivity and Breadth are
modifiers; exact decomposition depends on data structure).
"""

from __future__ import annotations
import math
from typing import Sequence, NamedTuple


class MeridianProfile(NamedTuple):
    """Four-component Meridian skill profile."""
    reach:      float   # proportion of questions attempted
    efficiency: float   # mean BSS on attempted questions
    adaptivity: float   # Spearman ρ(confidence, accuracy)
    breadth:    float   # fraction of domains with ≥1 attempt
    hiq:        float   # overall HIQ from composite BSS


def meridian_decompose(
    attempted: Sequence[bool],
    bss_per_question: Sequence[float],
    confidences: Sequence[float],
    accuracies: Sequence[float],
    domain_ids: Sequence[int] | None = None,
    norm_mean: float = 0.1264,
    norm_sd: float = 0.0321,
) -> MeridianProfile:
    """Decompose system performance into the four Meridian components.

    Parameters
    ----------
    attempted:
        Boolean sequence; True if the system provided a forecast for
        question i (some systems abstain on hard questions).
    bss_per_question:
        BSS for each question (NaN or 0 for unattempted is acceptable).
    confidences:
        Forecast probability issued by the system for each question.
        Used to compute Adaptivity (ρ between confidence and accuracy).
    accuracies:
        Binary outcome for each question (1 = correct direction,
        0 = incorrect).  Same length as *confidences*.
    domain_ids:
        Integer domain label for each question.  When provided, Breadth
        is computed as the fraction of unique domains with ≥1 attempt.
        If None, Breadth is set to NaN.
    norm_mean, norm_sd:
        Reference norm parameters for the HIQ calculation.

    Returns
    -------
    MeridianProfile
        Named tuple with Reach, Efficiency, Adaptivity, Breadth, HIQ.

    Examples
    --------
    >>> import random; random.seed(0)
    >>> n = 50
    >>> att = [True] * 40 + [False] * 10
    >>> bss = [random.uniform(0.05, 0.25) if a else 0.0 for a in att]
    >>> conf = [random.uniform(0.4, 0.9) for _ in range(n)]
    >>> acc  = [random.randint(0, 1) for _ in range(n)]
    >>> p = meridian_decompose(att, bss, conf, acc)
    >>> round(p.reach, 2)
    0.8
    """
    n = len(attempted)
    if not (len(bss_per_question) == len(confidences) == len(accuracies) == n):
        raise ValueError("All input sequences must have the same length")

    # ── Reach ─────────────────────────────────────────────────────────────────
    reach = sum(attempted) / n

    # ── Efficiency ────────────────────────────────────────────────────────────
    attempted_bss = [bss_per_question[i] for i in range(n) if attempted[i]]
    efficiency = sum(attempted_bss) / len(attempted_bss) if attempted_bss else float("nan")

    # ── Adaptivity (Spearman ρ) ───────────────────────────────────────────────
    adaptivity = _spearman_rho(list(confidences), list(accuracies))

    # ── Breadth ───────────────────────────────────────────────────────────────
    if domain_ids is not None:
        all_domains = set(domain_ids)
        attempted_domains = {domain_ids[i] for i in range(n) if attempted[i]}
        breadth = len(attempted_domains) / len(all_domains) if all_domains else float("nan")
    else:
        breadth = float("nan")

    # ── Overall HIQ from composite BSS (Reach × Efficiency) ──────────────────
    composite_bss = reach * efficiency if not math.isnan(efficiency) else float("nan")
    if not math.isnan(composite_bss):
        hiq_score = 100.0 + 15.0 * (composite_bss - norm_mean) / norm_sd
    else:
        hiq_score = float("nan")

    return MeridianProfile(
        reach=reach,
        efficiency=efficiency,
        adaptivity=adaptivity,
        breadth=breadth,
        hiq=hiq_score,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _rank(values: list[float]) -> list[float]:
    """Return fractional ranks for *values* (1-based, ties averaged)."""
    sorted_vals = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) - 1 and sorted_vals[j + 1][1] == sorted_vals[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation between *x* and *y*."""
    n = len(x)
    if n < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denom = math.sqrt(
        sum((rx[i] - mx) ** 2 for i in range(n))
        * sum((ry[i] - my) ** 2 for i in range(n))
    )
    return num / denom if denom > 0 else float("nan")
