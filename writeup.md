# Influence & Stability Analysis for ML Predictions

**Author:** Mariam Mohamed Elelidy  
**Topic:** Model Debugging · Influence Functions · Prediction Stability

---

## TL;DR

A model that achieves good average error can still have predictions disproportionately controlled by a small fraction of training data. This artifact identifies those points, measures how much they shift predictions when removed, and bounds the worst-case change — without modifying the model.

**Core result:** 25 training points (5.2% of data) hold 31.3% of total influence mass. Removing them causes 2.25× larger prediction shift than random removal, with empirical worst-case bound |Δ| ≤ 0.120 — 19.7% of base RMSE.

---

## 1. Motivation

Two models can have identical test RMSE yet differ fundamentally in **stability**: one's predictions are spread across all training data; another's are dominated by a handful of points that, if corrupted or removed, shift outputs significantly.

This matters because:
- **Data quality** — if a few corrupted or mislabeled points drive predictions, the model is brittle to cleaning
- **Deployment robustness** — if a data source changes, which predictions will move most?
- **Auditing** — "which training data drove this output?" is increasingly a regulatory question

Standard evaluation (RMSE, accuracy) answers none of these. Influence analysis does.

---

## 2. Testable Claims

**Primary:** A small influential subset holds disproportionate prediction mass. Removing it causes meaningfully larger shift than removing an equal-sized random subset.

**Stability bound:** The worst-case prediction shift from removing the top-25 influential points is bounded exactly by exact retraining: |Δ| ≤ 0.120.

**Diagnostic:** Influence ≠ leverage alone, and ≠ residual magnitude alone. The joint product — Cook's approximation — is required for correct ranking.

---

## 3. Method

### Setup

$$X \in \mathbb{R}^{n \times d} \sim \mathcal{N}(0, I), \quad w^* \sim \mathcal{N}(0, I_d), \quad y = Xw^* + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 0.36)$$

$n = 600$, $d = 8$, split 80% train / 20% test. Base model: ridge regression, $\lambda = 10^{-3}$.

### Leverage

$$h_{ii} = x_i^\top (X^\top X + \lambda I)^{-1} x_i$$

Points with $h_{ii} > 2d/n$ are high-leverage. The mean satisfies $\mathbb{E}[h_{ii}] = d/n$ exactly (from $\text{tr}(H) = d$).

### Influence scores (Cook's approximation)

$$\text{infl}_i \propto \frac{r_i^2 \cdot h_{ii}}{(1 - h_{ii})^2}, \quad r_i = y_i - \hat{y}_i$$

Normalised to sum to 1.0 — each score is the fraction of total influence mass held by point $i$.

### Stability measurement

For removal set $S$: retrain on $X_{\text{tr}} \setminus S$, compute $|\Delta_S(x)| = |\hat{f}_S(x) - \hat{f}(x)|$ for all test points. This is **exact** (full retrain), not a first-order approximation. The bound is tight.

### Random-removal baseline

100 random removals of the same $K$ quantify expected stability under untargeted removal. The ratio $\text{mean}|\Delta_{\text{inf}}| / \mathbb{E}[\text{mean}|\Delta_{\text{rand}}|]$ measures how much more disruptive targeted removal is.

---

## 4. Results

### Leverage

| Statistic | Value |
|---|---|
| Mean $h_{ii}$ | 0.016667 = $d/n$ ✓ |
| Max $h_{ii}$ | 0.051503 |
| Threshold $2d/n$ | 0.033333 |
| High-leverage points | 23 / 480 (4.8%) |

### Influence concentration

| Subset | Share | Cumulative |
|---|---|---|
| Top-1 (0.2%) | 3.30% | 3.30% |
| Top 2–5 (0.8%) | 6.85% | 10.15% |
| Top 6–25 (4.2%) | 21.18% | 31.32% |
| Rest (94.8%) | 68.68% | 100% |

Top-25 points hold **31.3% of influence mass** despite being 5.2% of the training data.

### Stability

| K | mean\|Δ\| | max\|Δ\| | ΔRMSE | vs random |
|---|---|---|---|---|
| 1 | 0.01163 | 0.03665 | +0.00086 | — |
| 5 | 0.01679 | 0.06276 | +0.00098 | — |
| 10 | 0.01459 | 0.05085 | −0.00108 | — |
| **25** | **0.03221** | **0.12003** | **−0.00403** | **2.25×** |
| 50 | 0.02659 | 0.08770 | +0.00156 | — |
| 100 | 0.03915 | 0.12782 | −0.00315 | — |

Random removal baseline (K=25, 100 trials): mean|Δ| = 0.01433 ±0.00447, max_trial = 0.028.
The random max never reaches the influential mean.

---

## 5. Analysis

### Influence requires both dimensions

Point #2 (train_idx=287) has the largest residual in the dataset (1.969) but ranks 2nd, not 1st — because its leverage (0.014) is the lowest in the top-10. Point #1 (train_idx=55) has the 3rd-largest residual combined with higher leverage, and their product places it first.

Debugging by residuals alone will miss the structurally important cases. Leverage alone identifies geometrically extreme points that may have small residuals. Both are required.

### Negative ΔRMSE is diagnostic, not a cleaning signal

Removing K=10 and K=25 influential points *decreases* test RMSE. This is not evidence of corruption. It means:

- The influential points are hard-to-fit, high-residual observations
- Removing them makes the remaining training problem easier
- The model was doing its job on difficult cases; their removal reduces average error on the now-easier test set

Flagging influential points for deletion based on RMSE improvement would remove the training signal on the hardest cases — the opposite of what debugging should do.

### Random-removal ceiling validates the scores

The worst random removal across 100 trials (max = 0.028) never reaches the influential mean (0.032). If influence scores were just sorting by residual magnitude, the random ceiling would be comparable. The gap validates that Cook's approximation is identifying structural leverage, not just filtering large residuals.

### Diminishing marginal returns past K=25

K=100 (20.8% of training data) produces max|Δ| = 0.128 vs K=25 max|Δ| = 0.120 — only 6% worse despite removing 4× as many points. The bound is essentially determined by the top-25. There is no long tail of equally important points.

---

## 6. Stability Bound

Removing the top-25 influential points (5.2% of data, 31.3% of influence mass):

$$\max_{x \in \text{test}} |\hat{f}_{-S_{25}}(x) - \hat{f}(x)| \leq 0.120$$

This is an empirical worst-case bound from exact retraining, valid for this model and dataset. For a base RMSE of 0.608, the bound is 19.7% of the error scale — meaning the most sensitive prediction shifts by less than one-fifth of a typical error when the most influential subset is removed.

---

## 7. Limitations

| Limitation | Detail |
|---|---|
| Cook's is a quadratic approximation | For non-linear models, first-order influence functions (Koh & Liang, 2017) or exact LOO are more appropriate |
| Ridge-specific leverage | The hat matrix formulation assumes a linear model with closed-form solution |
| Exact retrain cost | Stability measurement requires full retraining per K; infeasible for large models |
| Individual influence only | Cook's measures each point in isolation; joint influence of correlated points requires group removal |
| Synthetic data | Influence concentration may differ on real structured data |

---

## 8. Connections to the Conformal Prediction Series

| Question | Artifact |
|---|---|
| "How often will this interval contain the truth?" | Split conformal prediction |
| "Does coverage hold when assumptions are violated?" | Assumption stress harness |
| "Which training points are driving predictions?" | This artifact |

Each layer addresses a different dimension of reliability. A model can have correct coverage yet be unstable to training data composition — and vice versa.

---

## 9. Reproducibility

```bash
pip install numpy

python influence_stability.py                          # defaults
python influence_stability.py --n 2000 --K 10 25 50   # larger scale
python influence_stability.py --trials 500             # tighter baselines
```

All results are deterministic given `--seed`.

---

## 10. Takeaways

> **The point with the largest residual in the dataset ranked second in influence. Intuition that "hardest example" = "most influential" is often wrong.**

Three shifts in thinking from building this analysis:

1. **Influence needs both dimensions.** Residual magnitude and leverage individually give the wrong ranking. The combination — how wrong the model is, weighted by how extreme the feature location is — determines prediction stability. Debugging on residuals alone is structurally incomplete.

2. **RMSE improvement after removal is a warning, not a win.** A model whose test error improves when its hardest training cases are removed is a model that was doing its job. Treating this as evidence of corruption would remove exactly the training signal that matters.

3. **Stability bounds are part of the model's output.** Reporting RMSE = 0.608 without reporting that 5.2% of training data can shift any individual prediction by up to 19.7% of that RMSE is an incomplete picture. The bound belongs alongside the accuracy metric, not in a footnote.

---

## References

- Cook, R. D. (1977). Detection of influential observations in linear regression. *Technometrics*, 19(1), 15–18.
- Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via influence functions. *ICML*.
- Hampel, F. R. (1974). The influence curve and its role in robust estimation. *JASA*, 69(346), 383–393.
- Hoaglin, D. C., & Welsch, R. E. (1978). The hat matrix in regression and ANOVA. *The American Statistician*, 32(1), 17–22.
