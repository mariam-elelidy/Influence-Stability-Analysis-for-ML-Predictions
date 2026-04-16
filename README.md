# Influence & Stability Analysis for ML Predictions

> *A model that scores well on average can still have predictions controlled by a handful of training points. This artifact finds them, measures their impact, and bounds the worst case.*

---

## What this is

A closed-form influence analysis for ridge regression that:

1. **Ranks** every training point by its disproportionate effect on model predictions
2. **Measures** exact prediction shift when the top-K influential points are removed (full retrain, not approximation)
3. **Bounds** the worst-case prediction change: |Δ| ≤ 0.120 after removing top-25
4. **Validates** that influential removal is 2.25× more disruptive than removing random points of the same count

---

## Core result (n=600, d=8, seed=42)

| Metric | Value |
|---|---|
| Base RMSE | 0.6078 |
| High-leverage points | 23 / 480 (4.8%) |
| Top-25 influence share | 31.3% of total influence mass |
| K=25 mean\|Δ\| | 0.0322 (5.3% of RMSE) |
| K=25 max\|Δ\| (bound) | **0.120** (19.7% of RMSE) |
| Influential vs random ratio | **2.25×** |
| Spearman ρ (influence, \|residual\|) | 0.9524 |

---

## Quick start

```bash
pip install numpy

# Defaults: n=600, d=8, λ=1e-3, seed=42, K=[1,5,10,25,50,100]
python influence_stability.py

# Larger scale, tighter random baseline
python influence_stability.py --n 2000 --d 16 --K 10 25 50 --trials 500
```

**CLI arguments:**

| Flag | Default | Description |
|---|---|---|
| `--n` | 600 | Dataset size |
| `--d` | 8 | Feature dimension |
| `--lam` | 0.001 | Ridge λ |
| `--seed` | 42 | Random seed |
| `--K` | 1 5 10 25 50 100 | Removal sizes to evaluate |
| `--trials` | 100 | Random-removal trials for baseline |

---

## How it works

```
Training data (n_tr = 480)
        │
        ├──► Fit ridge: ŵ = (X'X + λI)⁻¹ X'y
        │
        ├──► Leverage:  h_ii = x_i' (X'X + λI)⁻¹ x_i
        │               flag: h_ii > 2d/n → high-leverage
        │
        ├──► Residuals: r_i = y_i - x_i'ŵ
        │
        ├──► Influence: infl_i ∝ r_i² · h_ii / (1 - h_ii)²
        │               normalised to sum = 1.0
        │               (Cook's approximation)
        │
        └──► Stability: for K in [1, 5, 10, 25, 50, 100]:
                          retrain on X_tr \ top-K
                          measure |Δf(x_test)| for all test points
                          compare to 100 random removals of same K
```

---

## Key findings

**Influence ≠ leverage, ≠ residual.** The point with the largest residual in the dataset (1.969) ranks 2nd in influence, not 1st — because its leverage (0.014) is the lowest in the top-10. The interaction between residual magnitude and feature-space position determines influence. Neither dimension alone gives the correct ranking.

**Negative ΔRMSE is diagnostic, not a cleaning signal.** Removing K=10 and K=25 influential points *decreases* test RMSE. These are hard-to-fit legitimate observations; the model was doing its job on them. Removing them makes the remaining problem easier. Treating RMSE improvement as evidence of corruption would eliminate the most challenging training signal.

**The random-removal ceiling validates the scores.** The worst random removal of 25 points across 100 trials (max = 0.028) never reaches the influential mean (0.032). If influence scores were just filtering by residual, the gap would be near zero. The gap confirms that Cook's approximation is identifying structural leverage, not acting as a residual filter.

**The bound is essentially set by K=25.** Removing K=100 (20.8% of training data) produces max|Δ| = 0.128 — only 6% worse than K=25. Influence mass is concentrated; there is no long tail of equally important points.

---

## Outputs

| Output | What it answers |
|---|---|
| Leverage table + flag | "Which points are geometrically extreme?" |
| Influence concentration bar | "How unevenly is influence distributed?" |
| Top-K influence tensor `[idx, share, h_ii, r_i]` | "Which specific points matter most, and why?" |
| Stability table `[K, mean\|Δ\|, max\|Δ\|, RMSE_new, ΔRMSE]` | "How much do predictions shift after removal?" |
| Influential vs random ratio | "Is the influence score actually predictive of instability?" |
| Worst-case bound | "What is the maximum any prediction can move?" |

---

## Relation to the reliability series

| Artifact | Reliability question |
|---|---|
| Split conformal prediction | Does this interval contain the truth with ≥90% frequency? |
| Assumption stress harness | How far does coverage fall when assumptions are violated? |
| **This artifact** | Which training points drive predictions, and what is the worst-case shift? |

These are orthogonal questions. A model can have correct conformal coverage but unstable predictions under training data perturbation — and vice versa.

---

## Extending this

| Extension | Change needed |
|---|---|
| Different base model | Replace `ridge_fit()` and `compute_leverage()` with model-specific versions |
| Neural networks | Use approximate influence functions (Koh & Liang, 2017) in place of Cook's |
| Group influence | Replace single-point removal with joint subset removal to capture correlations |
| Online detection | Stream new training points through the influence pipeline to flag high-influence arrivals |

---

## Repository layout

```
├── README.md               ← this file
├── influence_stability.py  ← implementation
├── output.txt              ← annotated run output with observations
└── writeup.md              ← full technical writeup
```

---

## References

- Cook, R. D. (1977). Detection of influential observations in linear regression. *Technometrics*, 19(1), 15–18.
- Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via influence functions. *ICML*.
- Hampel, F. R. (1974). The influence curve and its role in robust estimation. *JASA*, 69(346), 383–393.

---

*Part of a series on measurable reliability in ML.*
