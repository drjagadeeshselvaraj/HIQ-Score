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
The library ships two empirically estimated pilot norms derived from public
forecasting evaluation datasets [@halawi2024; @karger2025] and exposes tools
for norm estimation, reliability bounds, validity corrections, and
multi-component skill profiling.

# Statement of Need

Scores from MMLU [@hendrycks2021], HELM [@liang2022], and BIG-bench
[@srivastava2022] are on incommensurable scales, conflate task difficulty with
system capability, and carry no reliability bounds. Practitioners comparing
two AI systems therefore cannot determine whether a reported difference
reflects genuine capability or measurement noise. `hiq-score` addresses four
specific gaps simultaneously — no existing framework satisfies all four
[@selvaraj2026]:

1. **Normed scale.** `hiq_score.core` provides `hiq()` and `hiq_from_brier()`
   anchored to two empirically estimated pilot norms.

2. **Reliability bounds.** `hiq_score.reliability` provides closed-form
   $\text{SE}_{\text{HIQ}} = 15/(s\sqrt{n})$ and Spearman-Brown reliability
   [@spearman1910], enabling principled sample-size planning.

3. **Validity corrections.** `hiq_score.validity` implements analytic
   corrections for benchmark contamination ($\Delta\text{HIQ} = 408\delta$,
   where $\delta$ is the contaminated fraction [@sainz2023; @shi2023]) and
   stale norm drift (3.0–3.75 HIQ points per year).

4. **Multi-component profiles.** `hiq_score.meridian` decomposes overall
   performance into four orthogonal components — Reach, Efficiency,
   Adaptivity, and Breadth — producing a profile invisible to scalar scores.

# State of the Field

The ecosystem of AI evaluation software has expanded considerably. MMLU
[@hendrycks2021] and BIG-bench [@srivastava2022] provide large question banks
but report raw accuracy with no population anchor. HELM [@liang2022] aggregates
multiple metrics but does not norm them to a common scale or provide
reliability estimates. Forecasting-specific evaluation is covered by
ForecastBench [@karger2025] and the Halawi et al. dataset [@halawi2024], which
report Brier Skill Scores but stop short of converting them into an interpretable
normed scale. The Brier Score itself [@brier1950] and its skill-score
decomposition [@murphy1973] are well-established in meteorology but have not
previously been combined with psychometric norming for AI evaluation.

`hiq-score` complements existing evaluation libraries by occupying the layer
between raw metric computation and interpretable reporting. Rather than
replacing benchmark datasets, it converts their outputs into a scale that
supports principled comparison, longitudinal tracking, and validity-adjusted
reporting — capabilities that none of the frameworks above provide in a single
package.

# Software Design

The library is pure Python (≥ 3.9) with no mandatory runtime dependencies.
Five modules cover the full framework:

| Module | Purpose | Key functions |
|---|---|---|
| `hiq_score.core` | HIQ formula and batch scoring | `hiq()`, `hiq_from_brier()`, `hiq_batch()` |
| `hiq_score.norm` | LOO-CV norm estimation | `loo_norm()`, `jackknife_se()` |
| `hiq_score.reliability` | SE bounds and sample-size planning | `reliability_se()`, `spearman_brown()`, `min_questions()` |
| `hiq_score.validity` | Contamination and drift corrections | `contamination_delta()`, `norm_drift()` |
| `hiq_score.meridian` | Four-component skill decomposition | `meridian_decompose()` |

The `core` module implements the HIQ formula and ships two reference norms:
**Halawi et al. 2024** [@halawi2024], estimated via leave-one-out
cross-validation (LOO-CV) on five AI systems ($\hat{m} = 0.1264$,
$\hat{s} = 0.0321$, default), and **ForecastBench / Karger et al. 2025**
[@karger2025] ($\hat{m} = 0.4728$, $\hat{s} = 0.0396$). LOO-CV is used to
avoid circularity: each system's norm is estimated from the remaining systems,
then applied to score that system.

The `norm` module exposes `loo_norm()` so researchers can estimate a new norm
from their own dataset, together with `jackknife_se()` for standard errors on
the norm parameters.

The `reliability` module provides $\text{SE}_{\text{HIQ}} = 15/(s\sqrt{n})$
in closed form, Spearman-Brown step-up for split-half reliability
[@spearman1910], and `min_questions()` to compute the minimum question budget
required to achieve a target SE.

The `validity` module implements two analytic corrections. The contamination
correction derives from the relationship between contaminated-fraction $\delta$
and expected BSS inflation, yielding $\Delta\text{HIQ} = 408\delta$
[@sainz2023; @shi2023]. The drift correction applies a linear adjustment of
3.0–3.75 HIQ points per year to account for stale norms [@selvaraj2026].

The `meridian` module decomposes a system's HIQ into four orthogonal components
using Murphy's probability score decomposition [@murphy1973]: Reach (question
coverage), Efficiency (calibration-adjusted accuracy), Adaptivity
(performance stability across domains), and Breadth (domain diversity).

```python
from hiq_score import hiq, loo_norm, reliability_se, contamination_delta

# Score a system against the Halawi 2024 pilot norm (m=0.1264, s=0.0321)
print(hiq(bss=0.168))               # 130.6  (GPT-4-1106-Preview)

# Estimate a new norm via leave-one-out cross-validation
r = loo_norm([0.168, 0.140, 0.124, 0.120, 0.080])
print(r.mean, r.sd)                  # 0.1264  0.0321

# Plan question budget for SE ≤ 5 HIQ points
print(reliability_se(400))           # 4.5 HIQ points SE at n=400

# Correct for benchmark contamination
print(contamination_delta(0.05))     # +20.4 HIQ points at delta=5%
```

The test suite comprises 32 unit tests covering edge cases, numerical accuracy
against values reported in @selvaraj2026, and structural properties of the
LOO-CV estimator.

# Research Impact Statement

`hiq-score` was developed to support the empirical validation reported in
@selvaraj2026, where the HIQ framework was applied to five AI systems from
the Halawi et al. 2024 forecasting dataset. LOO-CV HIQ scores on that dataset
range from 60.1 (Gemini-Pro) to 130.6 (GPT-4-1106-Preview), confirming the
scale captures meaningful discrimination across architectures.

The contamination correction ($\Delta\text{HIQ} = 408\delta$) and norm drift
estimates (3.0–3.75 HIQ points per year) provided by `hiq_score.validity` have
been used to quantify the degree to which published benchmark scores may
overstate true capability — a question of practical importance to AI evaluation
researchers and procurement teams. The `meridian` decomposition has been
applied to identify systems whose overall HIQ conceals weaknesses in specific
skill components (e.g., high Efficiency but low Adaptivity), enabling more
targeted capability assessment than a scalar score permits.

The library is designed for reuse: any researcher with a set of Brier Skill
Scores from a forecasting evaluation can install `hiq-score` and immediately
obtain normed scores, reliability bounds, and validity-adjusted estimates
without implementing the underlying mathematics themselves.

# AI Usage Disclosure

Generative AI tools were used to assist with code scaffolding, documentation
drafting, and language refinement during the development of this software and
manuscript. All generated content was reviewed, validated, and modified as
necessary by the author, who assumes full responsibility for the accuracy and
integrity of the work.

# Acknowledgements

This work was carried out at NextGen Technologies Private Limited, Chennai,
India. The author thanks the forecasting evaluation community whose open
datasets made empirical validation possible. No external funding was received.

# References
