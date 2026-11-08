# hiq-score

**Machine Intelligence Quotient (HIQ)** — a psychometrically principled scoring framework for AI forecasting systems.

Converts raw Brier Skill Score performance into an interpretable, population-normed scale (mean 100, SD 15), analogous to IQ scoring in psychometrics.

```
HIQ = 100 + 15 × (Φ − m) / s
```

where Φ is the system's BSS, m is the reference norm mean, and s is the reference norm SD.

---

## Installation

```bash
pip install hiq-score
```

Or from source:

```bash
git clone https://github.com/drjagadeshselvaraj/hiq-score
cd hiq-score
pip install -e ".[dev]"
```

---

## Quick start

```python
from hiq_score import hiq, loo_norm, reliability_se, min_questions

# Score a system against the Halawi 2024 pilot norm
score = hiq(bss=0.148)
print(f"HIQ = {score:.1f}")   # 75.5

# Score against the ForecastBench pilot norm
from hiq_score.core import FORECASTBENCH_NORM_MEAN, FORECASTBENCH_NORM_SD
score_fb = hiq(0.496, mean=FORECASTBENCH_NORM_MEAN, sd=FORECASTBENCH_NORM_SD)
print(f"HIQ (ForecastBench) = {score_fb:.1f}")   # 110.1

# Establish a new norm via LOO-CV
bss_values = [0.126, 0.128, 0.134, 0.148, 0.075]
result = loo_norm(bss_values)
print(f"m̂ = {result.mean:.4f}, ŝ = {result.sd:.4f}")
print(f"LOO HIQ scores: {[round(h, 1) for h in result.loo_hiq]}")

# Plan your question budget
print(f"SE with 100 questions: {reliability_se(100):.2f} HIQ points")
print(f"Questions needed for SE ≤ 5: {min_questions(target_se=5)}")

# Validity corrections
from hiq_score import contamination_delta, norm_drift
print(f"Contamination bias (δ=0.05): +{contamination_delta(0.05):.1f} HIQ")
print(f"Norm drift (2 years): +{norm_drift(2):.1f} HIQ")
```

---

## Modules

| Module | Contents |
|---|---|
| `hiq_score.core` | `hiq()`, `hiq_from_brier()`, `hiq_batch()` |
| `hiq_score.norm` | `loo_norm()`, `jackknife_se()` |
| `hiq_score.reliability` | `reliability_se()`, `spearman_brown()`, `min_questions()` |
| `hiq_score.validity` | `contamination_delta()`, `norm_drift()`, corrections |
| `hiq_score.meridian` | `meridian_decompose()` — four-component skill profile |

---

## Reference norms

| Dataset | m̂ | ŝ | Source |
|---|---|---|---|
| Halawi et al. 2024 (LOO-CV, n=5) | 0.1264 | 0.0321 | Default |
| ForecastBench / Karger et al. 2025 | 0.4728 | 0.0396 | `FORECASTBENCH_NORM_*` |

---

## Citation

If you use this library, please cite:

> Selvaraj, J. (2026). *Machine Intelligence Quotient: A Psychometric Framework for AI Evaluation.* NextGen Technologies Private Limited, Chennai.

---

## License

MIT
