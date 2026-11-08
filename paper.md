---
title: 'hiq-score: A Python package for psychometric scoring of AI forecasting systems'
tags:
  - Python
  - artificial intelligence
  - evaluation
  - psychometrics
  - forecasting
  - Brier score
authors:
  - name: Jagadeesh Selvaraj
    orcid: 0009-0007-2314-4057
    corresponding: true
    email: dr.jagadeeshselvaraj@nextgentpl.com
    affiliation: 1
affiliations:
  - name: NextGen Technologies Private Limited, Chennai, India
    index: 1
date: 2026-11-08
bibliography: paper.bib
---

# Summary

`hiq-score` implements the **Machine Intelligence Quotient (HIQ)** framework,
a psychometrically principled system that converts raw Brier Skill Score (BSS)
forecasting performance into a population-normed scale (mean 100, SD 15):

$$\text{HIQ} = 100 + 15\,\frac{\Phi - m}{s}$$

where $\Phi$ is the system's BSS, $m$ is the reference norm mean, and $s$ is
the reference norm standard deviation [@selvaraj2026]. The norming convention
follows classical psychometrics [@wechsler1955]: an HIQ of 115 is exactly one
standard deviation above the reference mean, making scores immediately
interpretable and commensurable across tasks, time periods, and architectures.

# Statement of need

Scores from MMLU [@hendrycks2021], HELM [@liang2022], and BIG-bench
[@srivastava2022] are on incommensurable scales, conflate task difficulty with
system capability, and carry no reliability bounds. `hiq-score` addresses four
specific gaps simultaneously — no existing framework satisfies all four
[@selvaraj2026]:

1. **Normed scale.** `hiq_score.core` provides `hiq()` and `hiq_from_brier()`
   anchored to two empirically estimated pilot norms (Halawi et al. 2024;
   Karger et al. 2025).

2. **Reliability bounds.** `hiq_score.reliability` provides closed-form
   $\text{SE}_{\text{HIQ}} = 15/(s\sqrt{n})$ and Spearman-Brown reliability
   [@spearman1910], enabling principled sample-size planning via
   `min_questions()`.

3. **Validity corrections.** `hiq_score.validity` implements analytic
   corrections for benchmark contamination ($\Delta\text{HIQ} = 408\delta$,
   where $\delta$ is the contaminated fraction [@sainz2023; @shi2023]) and
   stale norm drift (3.0–3.75 HIQ points per year).

4. **Multi-component profiles.** `hiq_score.meridian` decomposes overall
   performance into four orthogonal Meridian components — Reach, Efficiency,
   Adaptivity, and Breadth — producing a profile invisible to scalar scores.

# Implementation

The library is pure Python (≥ 3.9) with no mandatory dependencies. Five
modules cover the full framework:

| Module | Key functions |
|---|---|
| `hiq_score.core` | `hiq()`, `hiq_from_brier()`, `hiq_batch()` |
| `hiq_score.norm` | `loo_norm()`, `jackknife_se()` |
| `hiq_score.reliability` | `reliability_se()`, `spearman_brown()`, `min_questions()` |
| `hiq_score.validity` | `contamination_delta()`, `norm_drift()` |
| `hiq_score.meridian` | `meridian_decompose()` |

```python
from hiq_score import hiq, loo_norm, reliability_se, contamination_delta

# Score a system against the Halawi 2024 pilot norm (m̂=0.1264, ŝ=0.0321)
print(hiq(bss=0.168))          # 130.6  (GPT-4-1106-Preview)

# Estimate a new norm via leave-one-out cross-validation
r = loo_norm([0.168, 0.140, 0.124, 0.120, 0.080])
print(r.mean, r.sd)            # 0.1264  0.0321

# Plan your question budget
print(reliability_se(400))     # 4.5 HIQ points SE at n=400 questions

# Correct for contamination
print(contamination_delta(0.05))  # +20.4 HIQ points inflation at δ=5%
```

The package ships two reference norms: **Halawi et al. 2024** [@halawi2024],
estimated via leave-one-out cross-validation (LOO-CV) on five AI systems
($\hat{m} = 0.1264$, $\hat{s} = 0.0321$, default), and **ForecastBench /
Karger et al. 2025** [@karger2025] ($\hat{m} = 0.4728$, $\hat{s} = 0.0396$).
LOO-CV HIQ scores on the Halawi dataset range from 60.1 (Gemini-Pro) to 130.6
(GPT-4-1106), confirming the scale captures meaningful discrimination. The
test suite comprises 32 unit tests covering edge cases and numerical accuracy
against values reported in @selvaraj2026.

# Acknowledgements

This work was carried out at NextGen Technologies Private Limited, Chennai,
India. The author thanks the forecasting evaluation community whose open
datasets made empirical validation possible. No external funding was received.

# References
