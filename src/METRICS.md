# Evaluation Metrics

> Last updated: 2026-05-22

## 1. Combined Accuracy Percentage %

```
Combined = max(0,  100 − MAPE − MAE% − RMSE%)
```

- The higher the better
- Clipped to **[0, 100]** — cannot go negative
- All three terms are mean-demand-normalised percentages, making them directly
  comparable and additive on the same [0, 100] scale

---

### A. Mean Absolute Percentage Error (MAPE)

```math
\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \frac{|\hat{y}_i - y_i|}{|y_i|} \times 100
```

- $y_i$ = actual value, $\hat{y}_i$ = predicted value
- Unit: **%** — already a percentage, used directly in Combined
- The lower the better

---

### B. MAE% — Mean Absolute Error as % of Total Demand

```math
\text{MAE\%} = \frac{\displaystyle\sum_{i=1}^{n} |\hat{y}_i - y_i|}{\displaystyle\sum_{i=1}^{n} y_i} \times 100
\;=\; \frac{\text{MAE}}{\bar{y}} \times 100
```

- Equivalent to dividing MAE by mean actual demand $\bar{y}$
- Unit: **%** — scale-free, used directly in Combined
- Raw **MAE** (ridership counts) is also reported as a standalone diagnostic
- The lower the better

---

### C. RMSE% — Root Mean Squared Error as % of Mean Demand

```math
\text{RMSE\%} = \frac{\text{RMSE}}{\bar{y}} \times 100
\quad \text{where} \quad
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2}
```

- Same mean-demand denominator as MAE%; result is in **%**
- RMSE penalises large errors more heavily than MAE (squared term)
- Raw **RMSE** (ridership counts) is also reported as a standalone diagnostic
- The lower the better

---

## 2. Coefficient of Determination (R²)

```math
R^2 = 1 - \frac{\displaystyle\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\displaystyle\sum_{i=1}^{n}(y_i - \bar{y})^2}
```

- where:
  - $y_i$ = actual value
  - $\hat{y}_i$ = predicted value
  - $\bar{y}$ = mean of actual values
  - `Numerator` = Residual Sum of Squares (SSR) — unexplained variance
  - `Denominator` = Total Sum of Squares (SST) — total variance in the data
- Range: **−∞ to 1**; a model worse than predicting the mean gives R² < 0
- The closer to 1 the better