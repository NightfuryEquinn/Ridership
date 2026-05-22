# Rainfall EDA Results

Exploratory data analysis of Malaysian sub-national rainfall data from `mys_rainfall_subnat_2019_2026.csv` (OCHA HDX / FEWS NET source), processed by `rainfall.py` into a wide-format daily matrix of per-state rainfall in mm.

---

## Distribution Analyses

### extreme_events_by_state.png
**What:** Count or magnitude of extreme rainfall events (above a threshold, e.g., 90th percentile daily mm) per Malaysian state, 2019–2025.
**Analysis:** Northeast Monsoon states (Kelantan `MY03`, Terengganu `MY11`, Pahang `MY06`) show the highest frequency of extreme events, consistent with the November–February northeast monsoon season when these east-coast states receive the highest annual rainfall totals. West-coast states (Selangor `MY10`, Penang `MY07`) show a bimodal extreme-event pattern from both monsoon seasons. This spatial heterogeneity motivates retaining per-state rainfall columns (`rainfall_mm__MY{pcode}`) rather than a national average — ridership impacts in KL may differ markedly from Kota Bharu on the same day.

### extreme_events_distribution.png
**What:** Statistical distribution (histogram or density) of extreme daily rainfall values across all states and years.
**Analysis:** The distribution is highly right-skewed with a long tail of extreme flood events. Malaysia's rainfall follows a log-normal-like distribution with monthly rainfall accumulations ranging from ~30 mm (dry spell) to >600 mm (monsoon flood peak). The `anomaly_rf` columns (retained in `rainfall_combined_final.csv` but excluded from `features_aligned.csv`) quantify deviations from climatological norms — available for future use if anomaly-based features are found to be more predictive than absolute mm values.

### extreme_events_frequency.png
**What:** Annual count of extreme rainfall days per state, 2019–2025.
**Analysis:** No systematic upward trend in frequency is visible at the 6-year study scale, though inter-annual variability is high (La Niña years, e.g., 2021–2022, show notably higher extreme-event counts in east-coast states). The 2021–2022 period coincides with MCO recovery — disentangling rainfall suppression from MCO suppression is handled in the models by including both `is_mco` context and per-state rainfall features simultaneously.

### extreme_events_seasonal.png (if present)
**What:** Monthly distribution of extreme rainfall events aggregated across all states.
**Analysis:** Shows the bimodal rainfall seasonality: Northeast Monsoon peak (November–January, east-coast states) and Southwest Monsoon / inter-monsoon peak (April–May, west-coast states). This seasonal pattern motivates the `month_sin` / `month_cos` cyclical features — the model can implicitly learn the interaction between monsoon season (month encoding) and per-state rainfall values.

---

## Time Series Analyses

### state_time_series.png (if present)
**What:** Multi-panel time series of daily `rainfall_mm` for a representative sample of Malaysian states (e.g., MY01, MY07, MY10).
**Analysis:** The panel confirms temporal coverage is contiguous after the linear interpolation applied in `rainfall.py` (interior gaps ≤ 7 days; edge gaps bfill/ffill). The null count before vs. after interpolation (logged by `rainfall.py`) should show zero remaining nulls in the final `rainfall_wide_daily.csv`. The 15 `rainfall_mm__MY{pcode}` columns in `features_aligned.csv` (MY01–MY17, excluding MY14/MY16) cover all states for which `final`-version observations are available.

### monsoon_seasonality.png (if present)
**What:** Average monthly rainfall by state, visualising the northeast/southwest monsoon pattern.
**Analysis:** The distinct seasonality confirms the value of per-state rainfall features over a single national average: during the Northeast Monsoon (Nov–Jan), Kelantan/Terengganu receive 5–10× more rainfall than Selangor/Penang, which would be masked by averaging. The models are expected to learn this spatial-temporal interaction when processing the 15 rainfall columns alongside the `month_sin`/`month_cos` temporal features.
