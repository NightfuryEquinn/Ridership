# Evaluation Metrics

## 1. Combined Accuracy Percentage %

```
Combined = max(0,  100 − MAPE − NMAE% − NRMSE%)
```

- The higher the better
- Clipped to **[0, 100]** — cannot go negative

> **Note:** Raw MAE and RMSE are in ridership counts (e.g. 26,000 riders). They
> cannot be subtracted from 100 directly. Each term must first be converted to a
> percentage via normalisation (see B and C below).

---

### A. Mean Absolute Percentage Error (MAPE)

```math
\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \frac{|\hat{y}_i - y_i|}{|y_i|} \times 100
```

- $y_i$ = actual value, $\hat{y}_i$ = predicted value
- Unit: **%** — already a percentage, used directly in Combined
- The lower the better

---

### B. Normalised Mean Absolute Error (NMAE) %

```math
\text{NMAE} = \frac{\text{MAE}}{\max(y) - \min(y)} \times 100
\quad \text{where} \quad
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |\hat{y}_i - y_i|
```

- Dividing by the **range** (max − min) maps MAE onto the actual dynamic swing
  the model must capture, giving a scale-free percentage in **[0, 100]**
- The lower the better

---

### C. Normalised Root Mean Squared Error (NRMSE) %

```math
\text{NRMSE} = \frac{\text{RMSE}}{\max(y) - \min(y)} \times 100
\quad \text{where} \quad
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2}
```

- Same range normalisation as NMAE; result is in **[0, 100]**
- RMSE penalises large errors more heavily than MAE (squared term)
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