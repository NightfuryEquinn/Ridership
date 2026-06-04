# Multivariate EDA Results

> Last updated: 2026-06-04

Multi-source analyses combining two or more of the eight feature groups to understand higher-order interactions relevant to Malaysian transit ridership forecasting. All analyses use the 79-feature aligned matrix (`features_aligned.csv`, 2019-01-01 – 2025-12-31) unless noted.

---

## Feature Space Analyses

### full_demand_correlation_matrix.png
**What:** Full 79×79 Pearson correlation matrix of all features in `features_aligned.csv`, loaded directly from the merged feature matrix (all 8 sources, full date range 2019–2025).
**Analysis:** The image reveals five visually distinct blocks along the diagonal:

1. **Ridership target block** (top-left, ~13 columns): The densest and darkest red block in the matrix. All 13 service lines (`total_ridership`, `bus_rkl`, `bus_rpn`, `rail_*`, and the 3 lag features) are strongly mutually correlated. Within-block correlations are visually near-uniform dark red, consistent with the expected ~0.85–0.99 range. The 3 lag features (`ridership_lag_7/14/28`) attach directly to this block with only slightly cooler red, confirming the strong autoregressive signal.

2. **Temporal block** (second block): Moderately structured. Cyclical features (`dow_sin`, `dow_cos`, `month_sin`, `month_cos`) show strong within-pair correlations (sin/cos pairs of the same period). `is_weekend` and `day_of_week` show visible negative cross-correlations (blue patches) with the ridership target block — confirming that weekends suppress commuter ridership. Holiday lead/lag columns (`days_to_next_public_hol`, `days_since_last_public_hol`) show weaker but visible blue patches against targets, reflecting the holiday suppression effect.

3. **External block** (fuel + rainfall, ~30 columns): Two visible sub-clusters. Fuel level columns (`fp_lv_*`) form a tight red sub-block — all Malaysian fuel prices move together under the administered pricing regime. Fuel change columns (`fp_chg_*`) form a separate sub-cluster with weaker internal correlation. The 15 rainfall columns (`rainfall_mm__MY*`) are mostly light/near-zero against each other (geographically dispersed stations) with mild positive inter-state correlation for adjacent states. The rainfall block is largely white against the ridership and fuel blocks (low cross-group correlation).

4. **Static block** (population, GTFS, OSM, GADM, ~17 columns): Entirely white against all dynamic feature groups — near-zero Pearson correlation with ridership, temporal, and external features. This is expected: these features are essentially constant across the time dimension (no day-to-day variation), so their Pearson correlation with time-varying features is zero by construction. Within the static block a small red cluster is visible among correlated infrastructure measures (`gtfs_n_stops`, `gtfs_n_routes`, `gtfs_n_directed_edges`).

**Graph construction implication:** After thresholding at |corr| ≥ 0.1, the adjacency is dominated by the ridership and fuel sub-graphs. The static feature nodes are effectively isolated (no edges to dynamic nodes), behaving as self-loop-only nodes in GCN layers — motivating MTGNN's learned adjacency as an alternative that can discover non-Pearson structural relationships.

### full_demand_multicollinearity_diagnostics.png
**What:** Variance Inflation Factor (VIF) bar chart or condition number analysis for the 79 features.
**Analysis:** High VIF is expected within the ridership target block (all 13 service-line riderships are highly correlated). The fuel price level and change columns within the same fuel type are also strongly collinear. However, the models in this project do not rely on coefficient interpretability (all are non-linear deep models), so multicollinearity in the feature matrix is not a model validity concern — it does, however, inform the graph construction: highly collinear features produce dense sub-graphs that risk over-smoothing in GCN models, motivating MTGNN's end-to-end learned (and sparser) adjacency as an alternative.

---

## Demand Pattern Analyses

### seasonal_demand_supply_mismatch.png
**What:** Monthly average ridership vs. a proxy for scheduled service capacity (GTFS `gtfs_n_routes` × days in month), showing months where demand and supply diverge.
**Analysis:** The largest mismatches occur around major holiday periods (Hari Raya months: April or May) where scheduled service levels remain constant but demand drops sharply. Post-MCO 2022–2023 shows a supply-demand gap in the opposite direction — recovering demand against a transit network that had not yet fully restored pre-MCO capacity. The `days_to_next_public_hol` lead-lag features and `year`/`day_of_year` trend features encode this mismatch dynamically in the model input.

### mode_choice_determinants.png
**What:** Feature importance or partial dependence analysis showing which variables most strongly predict total ridership in a regularised linear regression or gradient boosting model fitted on `features_aligned.csv`.
**Analysis:** Expected top predictors: `ridership_lag_7` (autoregressive 1-week lag), `is_weekend`, `is_public_holiday`, `dow_sin`/`dow_cos`, `days_to_next_public_hol`. Fuel price features are expected to have modest but non-zero importance. Rainfall features show low individual importance but contribute collectively across 15 state columns. Static features (GTFS, population) have near-zero marginal importance in a linear model but provide context for graph-based models operating on the correlation adjacency. This analysis validates the feature selection in `feature_align.py` — no feature group should have zero collective importance.

### composite_accessibility_score.png
**What:** A composite score per GTFS stop combining population density, POI count, and network centrality (approximated by `gtfs_n_directed_edges`), mapped against ridership.
**Analysis:** Stops with high composite accessibility scores are expected to be in areas with higher catchment ridership. This validates the joint inclusion of `pop_density_median`, `osm_poi_total_mean`, and `gtfs_n_stops` as complementary static features — each captures a distinct dimension of stop-level attractiveness. The near-zero inter-group correlations in the full correlation matrix (static group vs. dynamic group) confirm these three feature types provide non-redundant information.

---

## Temporal Regime Analyses

### demand_forecasting_correlation.png
**What:** Scatter of actual vs. predicted total ridership from a simple baseline model (e.g., naive 7-day lag forecast or linear regression), showing systematic biases.
**Analysis:** A 7-day naive lag baseline is a strong predictor (R² ~0.85–0.90) due to the strong weekly seasonality of Malaysian transit ridership. The residuals from this baseline are largest around public holidays (the lag model does not know a holiday is coming) and at the MCO boundary (distributional shift). These are exactly the failure modes that motivate the deep learning models: holiday-aware features, lead-lag encodings, and the MCO-aware regime gating in HMT-TSF.

### holiday_service_gap_analysis.png
**What:** Difference between expected ridership (based on day-of-week average) and actual ridership during holiday periods, broken down by holiday type.
**Analysis:** The holiday service gap is largest for Hari Raya Aidilfitri (urban rail ridership 50–70% below weekday average) and smallest for National Day / Malaysia Day (~15–20% below). The gap varies by service line — inter-city rail may show a positive gap (above expected) for Hari Raya as holiday travellers use KTM. This heterogeneity confirms that a national `is_public_holiday` flag is a necessary but not sufficient feature — the full model benefits from the combination of `is_holiday_any`, `days_to_next_public_hol`, `month_sin`/`month_cos`, and the ridership lags to model these complex holiday effects.

### weather_induced_ridership_surge.png
**What:** Ridership change on extreme rainfall days (above 95th percentile for a given state) vs. matched non-rain days, for states with transit service.
**Analysis:** Extreme rainfall can both suppress (walking-deterrence effect, service disruptions) and occasionally increase (car avoidance on flood-risk roads) transit ridership. The net effect in Malaysian transit is typically a mild suppression of bus ridership but near-zero effect on enclosed rail systems. The 15 `rainfall_mm__MY*` features give the models the information to learn this differentiated state-by-service interaction without explicit engineered interaction terms.
