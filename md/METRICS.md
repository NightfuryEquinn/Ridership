# Evaluation Metrics

> Last updated: 2026-07-02

This page explains the five measures used to score every model, and why each one is
included. In plain terms: they check how close the forecasts are, in percentage terms,
in passenger counts, and in how well the model tracks the ups and downs of demand. All
are computed by `src/utils/metrics.py` via `compute_metrics(y_true, y_pred)`. Each
section is written as **Main Idea → Evidence → Analysis → Link**.

---

## 1. Combined Accuracy Percentage (Combined%)

**Main Idea:** The Combined% metric is a composite scalar that aggregates three complementary error perspectives — relative percentage error (MAPE), scale-free absolute error (MAE%), and scale-free squared error (RMSE%) — into a single rank-ordered performance indicator, enabling direct and unambiguous comparison across all 16 models in the study.

```
Combined% = max(0, 100 − MAPE% − MAE% − RMSE%)
```

- Higher is better; clipped to **[0, 100]** — cannot go negative.
- All three constituent terms are mean-demand-normalised percentages, making them dimensionally homogeneous and directly addable on the same scale.

**Evidence:** In a seven-day ridership forecasting context where models differ in how they distribute prediction error — some achieving low MAPE through proportional accuracy on high-ridership days while incurring higher absolute errors on low-ridership days, and others achieving low RMSE by suppressing peak errors at the cost of elevated average error — no single metric captures the full error profile. Strigula (2026) noted that LSTM-family models evaluated on diverse time series benchmarks exhibit markedly different rankings depending on whether MAPE or RMSE is used as the primary criterion, reflecting the metric sensitivity problem in multi-model comparison studies. The Combined% formulation resolves this by penalising models simultaneously on all three dimensions: a model cannot score well on Combined% by optimising one error type at the expense of another.

**Analysis:** The equal weighting of the three components is a deliberate design choice reflecting the absence of a domain-specific priority ordering between proportional, absolute, and squared-error accuracy in transit operations. A transit agency monitoring ridership forecasts is equally concerned with percentage accuracy relative to daily demand (MAPE), average absolute discrepancy in passenger counts (MAE%), and suppression of large forecast failures (RMSE%). The `max(0, ·)` floor prevents numerical artefacts when the three error terms sum to more than 100% — a regime that corresponds to model failure and should not produce negative scores that distort rankings. This clipping is conservative: in practice, all 16 models in this study achieve Combined% well above zero in the default configuration.

**Limitations:** Three properties of the composite should be kept in view when interpreting rankings. (1) *Profile masking* — two models with identical Combined% can have very different error profiles (e.g., one MAPE-dominated, one RMSE-dominated); the constituent metrics are therefore always reported alongside the composite, and ties or near-ties on Combined% should be adjudicated on the constituents. (2) *Floor truncation* — once MAPE% + MAE% + RMSE% exceeds 100, all further degradation is invisible: two badly failing models both score 0 and become indistinguishable; comparisons in the Combined% = 0 regime must fall back to the raw constituents (this occurs only in a handful of MCO/long-lookback stress configurations in this study). (3) *Weighting convention* — the 1:1:1 weighting encodes the absence of a domain priority rather than an empirically derived preference; an agency that prioritises peak-error suppression could justifiably reweight toward RMSE%, which would reorder closely ranked models. Combined% is best read as a screening and ranking convenience, not a substitute for the underlying error metrics.

**Link:** Combined% is the primary sorting key for all model rankings in Chapter 4. Absolute MAE and RMSE (in raw ridership counts) are reported alongside Combined% as diagnostic supplements for interpreting the magnitude of forecast errors in operational units.

---

## 2. Mean Absolute Percentage Error (MAPE%)

**Main Idea:** MAPE expresses the average prediction error as a proportion of actual demand, providing a scale-independent measure of forecast accuracy that is directly interpretable as a percentage of daily ridership.

```math
\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \frac{|\hat{y}_i - y_i|}{|y_i|} \times 100
```

- $y_i$ = actual ridership, $\hat{y}_i$ = predicted ridership.
- Unit: **%** — already a percentage; used directly in Combined%.
- Lower is better.

**Evidence:** Wei et al. (2023) applied MAPE as the primary accuracy metric for subway passenger flow prediction across Shenzhen metro stations and found that temporal attention mechanisms achieve MAPE reductions of approximately 3–7% relative to standard LSTM baselines, demonstrating that MAPE is sufficiently sensitive to detect the incremental accuracy gains characteristic of deep-learning architecture comparisons. In this study, MAPE values range from 5.45% (HMT-TSF, nomco_lb14) to 20.56% (STFGNN, tuned mco_lb28), a range that spans both meaningful accuracy differentiation and failure detection.

**Analysis:** MAPE is particularly well-suited for transit ridership evaluation because it normalises errors by the actual daily ridership, giving proportionally equal weight to prediction accuracy on low-ridership days (weekends, holidays) and high-ridership days (weekday peaks). Without this normalisation, absolute-error metrics would systematically favour models that perform well on high-ridership days while tolerating large relative errors on low-ridership days — a bias that would misrepresent forecasting utility for transit planners who must manage operational capacity across all service days. The denominator $|y_i|$ ensures that a 5,000-passenger error on a 50,000-passenger day (10% MAPE contribution) is weighted equally to a 50,000-passenger error on a 500,000-passenger day.

**Link:** MAPE's proportional normalisation complements MAE%, which uses mean demand as a single global denominator rather than the per-observation actual. Together, both percentage metrics enter Combined% and jointly penalise models that trade off day-specific proportional accuracy against average absolute accuracy.

---

## 3. MAE% — Mean Absolute Error as Percentage of Mean Demand

**Main Idea:** MAE% normalises the absolute daily ridership error by mean demand, yielding a scale-free percentage that quantifies the average magnitude of forecast deviations relative to typical system throughput.

```math
\text{MAE\%} = \frac{\displaystyle\sum_{i=1}^{n} |\hat{y}_i - y_i|}{\displaystyle\sum_{i=1}^{n} y_i} \times 100 \;=\; \frac{\text{MAE}}{\bar{y}} \times 100
```

- $\bar{y}$ = mean actual ridership across all test observations.
- Unit: **%** — scale-free; used directly in Combined%.
- Raw **MAE** (absolute ridership counts) is also reported separately as an operational diagnostic.
- Lower is better.

**Evidence:** Alajmi and Almutairi (2025) evaluated BiLSTM-based traffic congestion forecasting using MAE alongside MAPE and RMSE, finding that MAE provides a more stable ranking criterion than MAPE when the target series contains near-zero observations — a practical concern because MAPE is numerically undefined (or artificially inflated) when $y_i \approx 0$. In this study, MCO-period ridership values can fall to levels where MAPE exhibits instability; MAE% with its global mean denominator $\bar{y}$ remains well-defined and bounded across the full evaluation period, making it particularly important for robustness analysis under the MCO configuration.

**Analysis:** The difference between MAPE and MAE% lies in their denominators: MAPE divides each error by the corresponding observed value $y_i$, while MAE% divides the sum of absolute errors by the sum of observed values $\sum y_i$ (equivalently, divides MAE by mean demand $\bar{y}$). This distinction matters when the target series is heteroscedastic: on low-ridership days (holidays, MCO period), MAPE assigns disproportionately high weight to the same absolute error that MAE% would treat as a small fraction of mean demand. The combined inclusion of both metrics in Combined% ensures that models are penalised for day-specific proportional failure (MAPE) as well as for average absolute error magnitude relative to system scale (MAE%).

**Link:** MAE% enters Combined% alongside MAPE% and RMSE%. The raw absolute MAE in ridership counts (reported alongside Combined%) provides operational context: for example, HMT-TSF's MAE of 49,897 daily boardings (46,323 for the feature-reduced variant) versus Informer's MAE of 69,476 quantifies the improvement in actionable capacity planning terms, independent of the percentage-normalised Combined% comparison.

---

## 4. RMSE% — Root Mean Squared Error as Percentage of Mean Demand

**Main Idea:** RMSE% normalises root mean squared error by mean ridership demand, providing a scale-free error measure that penalises large forecast deviations more heavily than MAE% through the squaring operation, thereby capturing tail-error behaviour in the forecast distribution.

```math
\text{RMSE\%} = \frac{\text{RMSE}}{\bar{y}} \times 100 \qquad \text{where} \qquad \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2}
```

- Same mean-demand denominator $\bar{y}$ as MAE%.
- Unit: **%** — used directly in Combined%.
- Raw **RMSE** (ridership counts) is also reported separately as an operational diagnostic.
- Lower is better.

**Evidence:** Wu et al. (2025) evaluated multi-dynamic temporal graph convolutional networks for traffic flow prediction and demonstrated that RMSE captures model sensitivity to peak-period demand spikes more effectively than MAE, since the squared error term exponentially penalises large deviations that arise when a model misjudges a morning-peak or post-holiday surge. In Malaysian transit ridership, public holiday return surges and unexpected service disruptions can produce single-day demand spikes of 20–40% above the weekly mean; a model that otherwise performs well but consistently fails on these peak events will exhibit inflated RMSE% relative to its MAE%, making RMSE% a valuable signal for identifying architecturally fragile peak-handling behaviour.

**Analysis:** The quadratic nature of RMSE means that a single large error of magnitude 2k contributes 4k² to the sum of squared errors, whereas two moderate errors of magnitude k each contribute only 2k². This property means RMSE% is more sensitive than MAE% to systematic failures on specific event types (holidays, disruptions, regime transitions), and models that concentrate their prediction errors on a small number of high-magnitude days will be penalised more severely on RMSE% than on MAE%. Including both metrics in Combined% therefore ensures that models are jointly evaluated on average absolute accuracy (MAE%) and worst-case daily error severity (RMSE%), providing a more complete picture of operational forecast reliability than either metric alone.

**Link:** RMSE% is the third and final constituent of Combined%. The difference between a model's RMSE% and MAE% values — referred to informally as the error dispersion — indicates the degree to which prediction errors are concentrated in high-magnitude events. Models with RMSE% substantially exceeding MAE% (e.g., CNN-LSTM under MCO, where both R² ≈ −0.05 and RMSE% >> MAE%) signal systematic catastrophic failures on specific days, while models where RMSE% ≈ MAE% exhibit evenly distributed errors.

---

## 5. Coefficient of Determination (R²)

**Main Idea:** R² measures the proportion of variance in actual ridership that is explained by the model's predictions, providing a scale-independent goodness-of-fit indicator that is directly interpretable as the fraction of total demand variation accounted for by the forecast.

```math
R^2 = 1 - \frac{\displaystyle\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\displaystyle\sum_{i=1}^{n}(y_i - \bar{y})^2}
```

- $y_i$ = actual ridership, $\hat{y}_i$ = predicted ridership, $\bar{y}$ = mean of actual values.
- Numerator = Residual Sum of Squares (SSR) — unexplained variance.
- Denominator = Total Sum of Squares (SST) — total variance in the data.
- Range: **−∞ to 1**; R² < 0 indicates performance worse than predicting the mean.
- Higher is better.

**Evidence:** Cui et al. (2025) applied R² as a complement to MAPE and RMSE in urban rail passenger flow prediction, demonstrating that R² provides a direct and interpretable measure of how much of the natural variability in transit demand — driven by weekly periodicity, seasonal effects, holidays, and external shocks — the model successfully captures. In this study, R² ranges from a high of 0.802 (HMT-TSF, nomco_lb14) to below zero for CNN-LSTM and STFGNN under MCO conditions, with the sub-zero values providing an unambiguous diagnostic of model failure: a negative R² indicates that the model's predictions are further from actual ridership, on average in squared terms, than a constant forecast equal to the mean — the simplest possible prediction strategy.

**Analysis:** R² serves a complementary role to Combined% in the evaluation framework. While Combined% penalises errors as percentages and rewards models that minimise all three error components simultaneously, R² measures explained variance and is sensitive to the model's ability to track the shape and direction of ridership dynamics rather than merely their magnitude. A model could, in principle, achieve low Combined% penalty by predicting values close to the mean on every day — a conservative strategy that avoids large errors but provides no operational forecasting value. R² would correctly identify this failure mode by assigning a near-zero or negative value, since the model fails to explain the variance that actually drives transit planning decisions (peak periods, holiday suppression, demand recovery). Chen et al. (2022) similarly found that R² provided additional discriminative power beyond MAE and RMSE when comparing CNN-BiLSTM variants on traffic flow data, since models with near-identical error magnitudes could differ substantially in how well they tracked the temporal structure of the target series.

**Link:** R² is reported alongside Combined% for all model configurations throughout Chapter 4 and serves as the secondary ranking criterion when Combined% values are close (within 0.1%). The fit diagnostics analysis in `DIAGNOSTICS.md` additionally uses the validation/training loss gap ratio and validation drift percentage — both derived from the training dynamics rather than test-set predictions — to assess generalisation quality independently of the five test-set metrics reported here.

---

## References

Alajmi, M. S., & Almutairi, S. M. (2025). Intelligent traffic congestion forecasting using BiLSTM and adaptive secretary bird optimizer for sustainable urban transportation. *Scientific Reports*, *15*. https://doi.org/10.1038/s41598-025-02933-9

Chen, W., Yang, Z., Xu, G., & Sun, Y. (2022). Short-term traffic flow prediction based on CNN-BiLSTM with multicomponent information. *Applied Sciences*, *12*(17), 8714. https://doi.org/10.3390/app12178714

Cui, H., Si, B., Chi, D., Li, Y., Li, G., & Chen, Y. (2025). Short-term passenger flow prediction for urban rail systems: A deep learning approach utilizing multi-source big data. *PLOS ONE*, *20*(1), e0333094. https://doi.org/10.1371/journal.pone.0333094

Strigula, M. (2026). Beyond traditional forecasting methods: Evaluating LSTM performance on diverse time series. *Mathematics*, *14*(5), 838. https://doi.org/10.3390/math14050838

Wei, L., Guo, D., Chen, Z., Yang, J., & Feng, T. (2023). Forecasting short-term passenger flow of subway stations based on the temporal pattern attention mechanism and the long short-term memory network. *ISPRS International Journal of Geo-Information*, *12*(1), 25. https://doi.org/10.3390/ijgi12010025

Wu, Z., Liu, X., & Zhang, X. (2025). Multi dynamic temporal representation graph convolutional network for traffic flow prediction. *Scientific Reports*, *15*, 16734. https://doi.org/10.1038/s41598-025-01157-1
