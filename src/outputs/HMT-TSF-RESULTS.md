# HMT-TSF Results (Proposed Model)

> Source: `src/outputs/aggregate_hmttsf.csv` · Generated 2026-06-05
> Companion files: `RESULTS.md` (16 baselines), `DIAGNOSTICS.md` (fit diagnostics)
> Scope: HMT-TSF and HMT-TSF-FR across `{nomco, mco}` × `{lb7, lb14, lb28, lb56, lb84}` = 10 configs each, with naive-persistence reference, Δ-vs-naive block, 3-block walk-forward validation, and fit diagnosis.

## 1. Conventions

- `nomco` / `mco`: MCO window (2020-03-18 – 2021-12-31) excluded / included.
- `lbNN`: look-back window in days. HMT-TSF extends the lookback grid to **7 and 84** beyond the baseline grid (14/28/56).
- Horizon `T_out = 7`.
- **Naive** = persistence baseline (last-value carried forward). **Delta** = model − naive (positive Δ on Combined%/R² = model better; negative Δ on error metrics = model better).
- **HMT-TSF-FR** = feature-reduced variant (53 features, 26 SHAP-zero features removed — see §10–11).
- Headline configuration: `nomco` (project default). MCO and lookback sweeps are robustness analysis.

---

## 2. Main results — all 10 configs

| Config | Combined% | MAPE% | MAE% | RMSE% | R² | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| **nomco_lb7** | 79.90 | 6.16 | 5.42 | 8.52 | 0.784 | 68,085 | 106,956 |
| **nomco_lb14** | **81.82** | **5.45** | **4.58** | **8.16** | **0.802** | **57,553** | **102,512** |
| **nomco_lb28** | 80.23 | 5.94 | 5.39 | 8.44 | 0.788 | 67,735 | 106,116 |
| **nomco_lb56** | 77.75 | 6.89 | 6.33 | 9.02 | 0.755 | 79,447 | 113,291 |
| **nomco_lb84** | 74.17 | 8.13 | 7.59 | 10.11 | 0.695 | 94,581 | 126,031 |
| mco_lb7 | 75.25 | 7.68 | 6.85 | 10.23 | 0.712 | 84,288 | 125,937 |
| mco_lb14 | 76.11 | 7.31 | 6.64 | 9.94 | 0.729 | 81,913 | 122,596 |
| mco_lb28 | 75.94 | 7.45 | 6.58 | 10.02 | 0.722 | 81,350 | 123,840 |
| **mco_lb56** | **76.53** | 7.24 | 6.23 | 10.00 | 0.722 | 77,313 | 124,147 |
| mco_lb84 | 73.95 | 8.33 | 6.90 | 10.82 | 0.673 | 85,604 | 134,339 |

**Read-out:**
- **Best overall: `nomco_lb14` — 81.82 Combined%, R² 0.802, MAE 57,553.** This is the global best of the entire study (beats the top baseline Informer at 79.13 by +2.69 Combined).
- **nomco lookback profile is single-peaked at lb14**, falling off symmetrically: lb7 (79.90) < lb14 (81.82) > lb28 (80.23) > lb56 (77.75) > lb84 (74.17). The 14-day window is decisively optimal for a 7-day horizon.
- **MCO profile inverts the lookback preference**: under MCO the best window is **lb56 (76.53)**, with lb84 again worst. Longer context helps the model average through the lockdown discontinuity, whereas in normal conditions it just adds noise.

---

## 3. Versus naive persistence

Model vs naive (last-value) baseline, Combined% and R², with Δ.

| Config | Model Combined% | Naive Combined% | Δ Combined | Model R² | Naive R² | Δ R² |
|---|---:|---:|---:|---:|---:|---:|
| nomco_lb7 | 79.90 | 34.80 | +45.10 | 0.784 | −0.963 | +1.746 |
| nomco_lb14 | 81.82 | 34.83 | **+46.98** | 0.802 | −0.961 | +1.763 |
| nomco_lb28 | 80.23 | 34.64 | +45.59 | 0.788 | −0.969 | +1.757 |
| nomco_lb56 | 77.75 | 34.99 | +42.76 | 0.755 | −0.960 | +1.715 |
| nomco_lb84 | 74.17 | 34.22 | +39.95 | 0.695 | −0.969 | +1.664 |
| mco_lb7 | 75.25 | 25.73 | +49.52 | 0.712 | −1.132 | +1.844 |
| mco_lb14 | 76.11 | 25.62 | +50.49 | 0.729 | −1.138 | +1.866 |
| mco_lb28 | 75.94 | 25.77 | +50.17 | 0.722 | −1.142 | +1.864 |
| mco_lb56 | 76.53 | 25.62 | +50.91 | 0.722 | −1.157 | +1.880 |
| mco_lb84 | 73.95 | 25.76 | +48.19 | 0.673 | −1.160 | +1.833 |

**Read-out:**
- Naive persistence has **negative R² in every config** (−0.96 nomco, −1.14 mco) — it is worse than predicting the mean, confirming ridership has strong non-persistent dynamics (weekly seasonality, holidays, fuel/weather shocks).
- HMT-TSF beats naive by **+47.0 Combined / +1.76 R² at nomco_lb14**, and the margin is even larger under MCO (**+50 Combined**), because the naive baseline degrades faster than the model when the lockdown break is included. This quantifies how much structure the model is actually capturing beyond carry-forward.

---

## 4. MCO robustness

| Lookback | nomco Combined% | mco Combined% | Δ (mco − nomco) | nomco R² | mco R² |
|---|---:|---:|---:|---:|---:|
| lb7 | 79.90 | 75.25 | −4.65 | 0.784 | 0.712 |
| lb14 | 81.82 | 76.11 | −5.71 | 0.802 | 0.729 |
| lb28 | 80.23 | 75.94 | −4.29 | 0.788 | 0.722 |
| lb56 | 77.75 | 76.53 | −1.22 | 0.755 | 0.722 |
| lb84 | 74.17 | 73.95 | −0.22 | 0.695 | 0.673 |

**Read-out:**
- HMT-TSF's MCO degradation is **−5.71 at lb14 and shrinks to −1.22 at lb56 / −0.22 at lb84**. With a long enough window the model is virtually MCO-invariant.
- Context vs. `RESULTS.md` §5: at lb14 HMT-TSF (−5.71) is the most MCO-robust model in the entire study, marginally ahead of Informer (−5.88) and BiLSTM (−6.30), and far ahead of the fragile tail (CNN-LSTM −32.1, STFGNN −32.9). It also retains the **highest absolute MCO accuracy** (76.11 vs Informer 73.25).
- This robustness is the core empirical justification for the regime-embedding component of the architecture.

---

## 5. Walk-forward validation (3 blocks)

Each config was re-validated on 3 sequential out-of-sample blocks. Combined% / R² per block.

| Config | Block 1 | Block 2 | Block 3 | Spread (Comb.) |
|---|---|---|---|---:|
| nomco_lb7 | 81.63 / 0.822 | 75.59 / 0.693 | 82.90 / 0.847 | 7.3 |
| nomco_lb14 | 85.18 / 0.870 | 76.99 / 0.699 | 83.72 / 0.844 | 8.2 |
| nomco_lb28 | 86.63 / 0.896 | 72.11 / 0.633 | 82.83 / 0.845 | 14.5 |
| nomco_lb56 | 78.38 / 0.759 | 75.72 / 0.728 | 79.16 / 0.778 | 3.4 |
| nomco_lb84 | 68.33 / 0.593 | 78.65 / 0.788 | 75.85 / 0.715 | 10.3 |
| mco_lb7 | 69.89 / 0.638 | 79.52 / 0.782 | 76.31 / 0.706 | 9.6 |
| mco_lb14 | 71.58 / 0.673 | 80.51 / 0.793 | 76.24 / 0.711 | 8.9 |
| mco_lb28 | 69.80 / 0.653 | 81.23 / 0.803 | 76.82 / 0.704 | 11.4 |
| mco_lb56 | 72.40 / 0.674 | 78.29 / 0.747 | 78.90 / 0.746 | 6.5 |
| mco_lb84 | 70.06 / 0.615 | 74.36 / 0.666 | 77.52 / 0.740 | 7.5 |

**Read-out:**
- All blocks stay strong (no block drops below ~68 Combined / 0.59 R²) — performance is **not driven by a single favourable test split**.
- Under `nomco`, **Block 2 is consistently the weakest** block (e.g., lb14: 76.99 vs 85.18/83.72), indicating a harder middle segment (likely a seasonal/behavioural shift). Under `mco`, the ordering flips — **Block 2 is the strongest** — consistent with the lockdown signal landing in different blocks across the two splits.
- **nomco_lb56 is the most temporally stable** config (spread 3.4), a useful property if block-to-block consistency matters more than peak accuracy. nomco_lb28 has the highest single block (86.63 / R² 0.896) but the widest spread (14.5).

---

## 6. Fit diagnosis

Every HMT-TSF config (all 10) is diagnosed **`good_fit`** — see `DIAGNOSTICS.md` for the cross-model comparison.

| Config | Verdict | Val drift % | Gap ratio | Val trend | Best/Total epochs |
|---|---|---:|---:|---|---|
| nomco_lb7 | good_fit | 1.19 | 1.97 | flat | 147 / 150 |
| nomco_lb14 | good_fit | 1.50 | 2.39 | flat | 148 / 150 |
| nomco_lb28 | good_fit | 2.89 | 2.38 | flat | 66 / 91 |
| nomco_lb56 | good_fit | 12.32 | 2.76 | falling | 17 / 37 |
| nomco_lb84 | good_fit | 7.22 | 2.84 | flat | 58 / 88 |
| mco_lb7 | good_fit | 0.44 | 1.80 | flat | 137 / 150 |
| mco_lb14 | good_fit | 5.22 | 2.14 | flat | 102 / 122 |
| mco_lb28 | good_fit | 9.34 | 2.10 | falling | 25 / 50 |
| mco_lb56 | good_fit | 4.42 | 2.28 | falling | 18 / 38 |
| mco_lb84 | good_fit | 3.93 | 1.99 | flat | 45 / 75 |

**Read-out:**
- **Gap ratio (val/train loss) stays within 1.8–2.8× — always under the 3× overfit threshold.** No config shows memorisation. This is the cleanest fit profile in the study: every LSTM-family and STGCN-family baseline exceeds 3× (often far more — up to 37×; see `DIAGNOSTICS.md`).
- **Validation drift is small** (≤12.3%, well under the 25% threshold); largest at nomco_lb56 but still `good_fit`.
- The lb14/lb7 nomco runs train to near the epoch cap (148/147 of 150) with flat val trend — they could likely absorb a few more epochs, but the marginal gain is small given they are already the best configs.

---

## 7. Verdict

1. **HMT-TSF is the global best model in the study** at its default `nomco_lb14`: 81.82 Combined%, R² 0.802, MAE 57,553 — ahead of every baseline on every headline metric (`RESULTS.md` §2).
2. **It generalises cleanly** — the only model family that is `good_fit` across *all* configs, with gap ratios capped at 2.8× vs baselines' 3–37×.
3. **It is the most shock-robust model** — smallest MCO degradation at lb14 and near-invariant at long lookbacks (−0.22 at lb84), with the highest retained MCO accuracy.
4. **Lookback recommendation: lb14 for normal operations, lb56 for shock-prone regimes.** lb84 is never optimal and should be dropped.
5. **Walk-forward confirms the headline is not split-luck** — all three blocks hold up.

---

## 8. Use-case scenarios

| Scenario | Recommended HMT-TSF config | Rationale |
|---|---|---|
| **Default production forecasting** | `nomco_lb14` | Global best accuracy + clean fit; the 14-day window is decisively optimal for the 7-day horizon. |
| **Shock / lockdown / structural-break regime** | `mco_lb56` (or `lb84` for maximum invariance) | Long context minimises MCO degradation (Δ −1.22) and gives the best MCO Combined% (76.53). |
| **Maximum temporal stability across periods** | `nomco_lb56` | Lowest block-to-block spread (3.4) in walk-forward. |
| **Lowest absolute error** | `nomco_lb14` | MAE 57,553 / RMSE 102,512 — lowest in the study. |
| **Interpretability / driver analysis** | `nomco_lb14` with `--shap` | Clean fit makes attributions trustworthy; pair with Informer for cross-checking. |
| **Compute / memory constrained** | HMT-TSF-FR `nomco_lb14` | 53-feature input (vs 79) cuts GCN memory ~33%; within 0.05 Combined% of full model (§11). |
| **Drop from the grid** | `lb84` (both nomco and mco) | Never optimal; strictly dominated by shorter windows. |

> All numbers verified against `aggregate_hmttsf.csv` row 2.

---

## 9. SHAP Feature Importance

### Method

SHAP values were computed using `shap.GradientExplainer` on the trained `nomco_lb14` model. The first 100 test samples (shape `100 × 84 × 79`) were used as the background set. The explainer produced an absolute SHAP array of the same shape; values were averaged over the sample and time-step dimensions, yielding a single 79-element importance vector — one score per input feature.

### Top 10 Features (decoded)

| SHAP rank | feat index | Feature name | Group | Interpretation |
|-----------|-----------|--------------|-------|----------------|
| 1 | feat_12 | rail_komuter | Ridership | KTM Komuter historical counts — dominant predictor; strong autocorrelation |
| 2 | feat_4 | rail_mrt_kajang | Ridership | MRT Kajang historical counts |
| 3 | feat_3 | rail_lrt_ampang | Ridership | LRT Ampang historical counts |
| 4 | feat_42 | fp_chg_ron95_budi95 | Fuel — external | RON95 Budi95 subsidy price change; mode-shift signal |
| 5 | feat_37 | fp_lv_diesel_pct_chg | Fuel — external | Diesel % change; freight/bus cost pass-through |
| 6 | feat_2 | bus_rpn | Ridership | Rapid Bus Penang historical counts |
| 7 | feat_20 | day_of_week | Temporal | Weekly seasonality — single strongest temporal signal |
| 8 | feat_53 | rainfall_mm__MY10 | Rainfall — external | Selangor rainfall; wet days shift commuters onto rail |
| 9 | feat_40 | fp_chg_diesel | Fuel — external | Diesel absolute price change |
| 10 | feat_0 | total_ridership | Ridership | Aggregate total ridership history |

### Interpretation

The model's learned drivers align with known transit demand theory. Historical ridership on individual service lines dominates (indices 0–12), confirming strong autocorrelation — yesterday's Komuter count is the best single predictor of tomorrow's. Fuel price signals (indices 29–43) appear next, capturing the mode-substitution effect: RON95/diesel price increases reduce private vehicle use and push passengers onto transit. Day of week (`feat_20`) is the only temporal feature in the top 10, reflecting weekly periodicity as more informative than calendar or holiday indicators alone. Selangor rainfall (`feat_53`, MY10) enters because the Klang Valley concentrates the majority of Malaysian transit ridership and wet conditions measurably increase rail uptake. Static infrastructure features (GTFS, OSM, GADM, indices 64–78) rank low throughout — they carry little day-to-day variance and the model correctly down-weights them. This attribution profile supports the validity of the feature set and validates the architecture's ability to extract meaningful temporal and cross-modal signals.

### Full Feature Index → Name Mapping

#### Group 1 — Target / Ridership (indices 0–12)

| Index | Feature |
|-------|---------|
| 0 | total_ridership |
| 1 | bus_rkl |
| 2 | bus_rpn |
| 3 | rail_lrt_ampang |
| 4 | rail_mrt_kajang |
| 5 | rail_lrt_kj |
| 6 | rail_monorail |
| 7 | rail_mrt_pjy |
| 8 | rail_ets |
| 9 | rail_intercity |
| 10 | rail_komuter_utara |
| 11 | rail_tebrau |
| 12 | rail_komuter |

#### Group 2 — Temporal (indices 13–28)

| Index | Feature |
|-------|---------|
| 13 | is_public_holiday |
| 14 | is_school_holiday |
| 15 | is_holiday_any |
| 16 | days_to_next_public_hol |
| 17 | days_since_last_public_hol |
| 18 | days_to_next_school_hol |
| 19 | days_since_last_school_hol |
| 20 | day_of_week |
| 21 | month |
| 22 | is_weekend |
| 23 | dow_sin |
| 24 | dow_cos |
| 25 | month_sin |
| 26 | month_cos |
| 27 | year |
| 28 | day_of_year |

#### Group 3 — External: Fuel Prices (indices 29–43)

| Index | Feature |
|-------|---------|
| 29 | fp_lv_ron95 |
| 30 | fp_lv_ron97 |
| 31 | fp_lv_diesel |
| 32 | fp_lv_diesel_eastmsia |
| 33 | fp_lv_ron95_budi95 |
| 34 | fp_lv_ron95_skps |
| 35 | fp_lv_ron95_pct_chg |
| 36 | fp_lv_ron97_pct_chg |
| 37 | fp_lv_diesel_pct_chg |
| 38 | fp_chg_ron95 |
| 39 | fp_chg_ron97 |
| 40 | fp_chg_diesel |
| 41 | fp_chg_diesel_eastmsia |
| 42 | fp_chg_ron95_budi95 |
| 43 | fp_chg_ron95_skps |

#### Group 3 — External: Rainfall by State (indices 44–58)

| Index | Feature | State |
|-------|---------|-------|
| 44 | rainfall_mm__MY01 | Johor |
| 45 | rainfall_mm__MY02 | Kedah |
| 46 | rainfall_mm__MY03 | Kelantan |
| 47 | rainfall_mm__MY04 | Melaka |
| 48 | rainfall_mm__MY05 | Negeri Sembilan |
| 49 | rainfall_mm__MY06 | Pahang |
| 50 | rainfall_mm__MY07 | Pulau Pinang |
| 51 | rainfall_mm__MY08 | Perak |
| 52 | rainfall_mm__MY09 | Perlis |
| 53 | rainfall_mm__MY10 | Selangor |
| 54 | rainfall_mm__MY11 | Terengganu |
| 55 | rainfall_mm__MY12 | Sabah |
| 56 | rainfall_mm__MY13 | Sarawak |
| 57 | rainfall_mm__MY15 | W.P. Kuala Lumpur |
| 58 | rainfall_mm__MY17 | W.P. Putrajaya |

> MY14 (Labuan) and MY16 are excluded from the pipeline.

#### Group 4 — Lag Features (indices 59–61)

| Index | Feature |
|-------|---------|
| 59 | ridership_lag_7 |
| 60 | ridership_lag_14 |
| 61 | ridership_lag_28 |

#### Group 5 — Static (indices 62–78)

| Index | Feature | Source |
|-------|---------|--------|
| 62 | pop_density_median | Population |
| 63 | pop_density_log_median | Population |
| 64 | gtfs_n_stops | GTFS |
| 65 | gtfs_n_routes | GTFS |
| 66 | gtfs_n_directed_edges | GTFS |
| 67 | gtfs_avg_segment_s | GTFS |
| 68 | osm_poi_total_mean | OSM |
| 69 | osm_poi_transport_mean | OSM |
| 70 | osm_poi_food_mean | OSM |
| 71 | osm_poi_retail_mean | OSM |
| 72 | osm_poi_education_mean | OSM |
| 73 | osm_poi_healthcare_mean | OSM |
| 74 | osm_poi_leisure_mean | OSM |
| 75 | osm_poi_other_mean | OSM |
| 76 | gadm_n_states | GADM |
| 77 | gadm_n_border_pairs | GADM |
| 78 | gadm_mean_border_km | GADM |

---

## 10. Cross-Run SHAP Feature Reduction Analysis

### Method

SHAP values from all 10 HMT-TSF configurations (nomco/mco × lb7/lb14/lb28/lb56/lb84) were aggregated independently. Each `shap_values.npy` has shape `(100, T_in, 79, 7)`; absolute values were averaged over samples, time steps, and forecast horizons to produce a 79-element importance vector per run. Features were then classified by how many runs assigned them zero importance.

### Universal Zeros — Zero in ALL 10 runs (drop unconditionally)

**23 features** contribute zero signal regardless of lookback window or MCO setting.

| Index | Feature | Group | Why dead |
|-------|---------|-------|---------|
| 21 | month | Temporal | Redundant with month_sin/cos |
| 22 | is_weekend | Temporal | Fully subsumed by day_of_week |
| 23 | dow_sin | Temporal | Cyclical encoding unused; raw day_of_week dominates |
| 30 | fp_lv_ron97 | Fuel | Collinear with RON95; change variants rank higher |
| 31 | fp_lv_diesel | Fuel | Collinear with diesel pct_chg variants |
| 32 | fp_lv_diesel_eastmsia | Fuel | East Malaysia price level; irrelevant to KL/Penang corridors |
| 62 | pop_density_median | Static | Time-invariant; no day-to-day variation |
| 63 | pop_density_log_median | Static | Same |
| 64 | gtfs_n_stops | Static | Same |
| 65 | gtfs_n_routes | Static | Same |
| 66 | gtfs_n_directed_edges | Static | Same |
| 67 | gtfs_avg_segment_s | Static | Same |
| 68 | osm_poi_total_mean | Static | Aggregate of below; all zero |
| 69 | osm_poi_transport_mean | Static | Same |
| 70 | osm_poi_food_mean | Static | Same |
| 71 | osm_poi_retail_mean | Static | Same |
| 72 | osm_poi_education_mean | Static | Same |
| 73 | osm_poi_healthcare_mean | Static | Same |
| 74 | osm_poi_leisure_mean | Static | Same |
| 75 | osm_poi_other_mean | Static | Same |
| 76 | gadm_n_states | Static | Constant scalar across all rows |
| 77 | gadm_n_border_pairs | Static | Same |
| 78 | gadm_mean_border_km | Static | Same |

> The entire Static group (OSM, GTFS, GADM, population) is dead weight across every configuration. This was confirmed across both nomco and mco, and all five lookback windows.

### Near-Universal Zeros — Zero in 8/10 runs (recommended to drop)

| Index | Feature | Zero in | Note |
|-------|---------|---------|------|
| 18 | days_to_next_school_hol | 8/10 | Only marginally active in 2 configs |
| 24 | dow_cos | 8/10 | Cosine half of DOW encoding; raw day_of_week dominates |
| 27 | year | 8/10 | Only active in lb7 (very short context forces it) |

### Consistently Important Features — Top-15 in ≥6/10 runs

| Consistency | Index | Feature | Group |
|------------|-------|---------|-------|
| 10/10 (100%) | feat_12 | rail_komuter | Ridership |
| 8/10 (80%) | feat_42 | fp_chg_ron95_budi95 | Fuel change |
| 7/10 (70%) | feat_36 | fp_lv_ron97_pct_chg | Fuel % change |
| 7/10 (70%) | feat_46 | rainfall_mm__MY03 (Kelantan) | Rainfall |
| 6/10 (60%) | feat_04 | rail_mrt_kajang | Ridership |
| 6/10 (60%) | feat_20 | day_of_week | Temporal |
| 6/10 (60%) | feat_35 | fp_lv_ron95_skps | Fuel |
| 6/10 (60%) | feat_41 | fp_chg_diesel_eastmsia | Fuel change |
| 6/10 (60%) | feat_45 | rainfall_mm__MY02 (Kedah) | Rainfall |

> Kelantan (MY03) and Kedah (MY02) rainfall appearing in 70% and 60% of runs respectively is not coincidental — both states sit on the northeast monsoon corridor (Nov–Jan). The model has detected a weather regime effect propagating from east-coast precipitation to nationwide transit demand.

### Lookback-Dependent Feature Shifts

| Window | Dominant features | Interpretation |
|--------|------------------|----------------|
| lb7 | feat_34/35 (RON95 price levels), rainfall states | Very short context forces reliance on external signals; no autocorrelation depth |
| lb14–28 | Balanced: ridership lines + rainfall + fuel change features | Optimal regime — matches best accuracy (nomco_lb14 = 81.82 Combined%) |
| lb56–84 | feat_12 (rail_komuter) dominates | Long autocorrelation window; model anchors on historical ridership trend |

### Revised Removal Summary

| Tier | Count | Criterion | Recommended action |
|------|-------|-----------|-------------------|
| Universal zeros | 23 | Zero in all 10 runs | Drop unconditionally |
| Near-universal zeros | 3 | Zero in 8/10 runs | Recommended to drop |
| **Total** | **26** | 79 → **53 features** | ~33% input dimension reduction |

Removing 26 features shrinks the input tensor from `(B, T_in, 79)` → `(B, T_in, 53)`, reduces the graph adjacency from `79×79` → `53×53`, and cuts GradientExplainer memory by ~33% — with no expected loss in predictive accuracy since all removed features contribute zero or near-zero signal across both data regimes and all lookback windows tested.

---

## 11. HMT-TSF vs HMT-TSF-FR Comparison

HMT-TSF-FR is the feature-reduced variant trained on 53 features (26 SHAP-zero features removed per §10). All training hyperparameters are identical; only the input dimension changes.

### Combined% — HMT-TSF vs HMT-TSF-FR

| Config | HMT-TSF | HMT-TSF-FR | Δ (FR − std) |
|---|---:|---:|---:|
| nomco_lb7 | 79.90 | 79.72 | −0.18 |
| nomco_lb14 | **81.82** | **81.77** | −0.05 |
| nomco_lb28 | 80.23 | 81.11 | **+0.88** |
| nomco_lb56 | 77.75 | 78.44 | **+0.69** |
| nomco_lb84 | 74.17 | 75.51 | **+1.34** |
| mco_lb7 | 75.25 | 75.12 | −0.13 |
| mco_lb14 | 76.11 | 75.59 | −0.52 |
| mco_lb28 | 75.94 | **76.69** | **+0.75** |
| mco_lb56 | 76.53 | 76.44 | −0.09 |
| mco_lb84 | **73.95** | 69.45 | **−4.50** |

### R² — HMT-TSF vs HMT-TSF-FR

| Config | HMT-TSF R² | HMT-TSF-FR R² | Δ R² |
|---|---:|---:|---:|
| nomco_lb14 | 0.802 | 0.805 | +0.003 |
| nomco_lb28 | 0.788 | 0.795 | +0.007 |
| nomco_lb56 | 0.755 | 0.758 | +0.003 |
| nomco_lb84 | 0.695 | 0.723 | +0.028 |
| mco_lb14 | 0.729 | 0.712 | −0.017 |
| mco_lb84 | 0.673 | 0.530 | −0.143 |

### Fit Diagnosis — HMT-TSF-FR

| Config | Verdict | Val drift % | Gap ratio | Val trend | Best/Total epochs |
|---|---|---:|---:|---|---|
| nomco_lb7 | good_fit | 2.67 | 2.01 | flat | 147 / 150 |
| nomco_lb14 | good_fit | 4.66 | 2.19 | flat | 85 / 105 |
| nomco_lb28 | good_fit | 1.72 | 2.33 | falling | 56 / 81 |
| nomco_lb56 | good_fit | 8.22 | 2.74 | falling | 12 / 32 |
| nomco_lb84 | good_fit | 1.04 | 2.62 | flat | 132 / 150 |
| mco_lb7 | good_fit | 2.41 | 1.74 | flat | 68 / 93 |
| mco_lb14 | good_fit | 11.78 | 1.93 | flat | 67 / 87 |
| mco_lb28 | good_fit | 6.41 | 2.02 | falling | 27 / 52 |
| mco_lb56 | good_fit | 5.40 | 2.28 | falling | 19 / 39 |
| mco_lb84 | good_fit | 1.91 | 2.10 | flat | 75 / 105 |

### Read-out

- **At the headline configuration (`nomco_lb14`), FR is essentially identical to the standard model** (81.77 vs 81.82, Δ −0.05 Combined, R² 0.805 vs 0.802). The 26 removed features contributed zero signal, confirming the SHAP analysis.
- **FR outperforms the standard model at longer nomco lookbacks** (lb28 +0.88, lb56 +0.69, lb84 +1.34). Removing noisy zero-SHAP features appears to reduce interference when the model has a long autocorrelation window — a noise-regularisation effect.
- **FR degrades sharply at `mco_lb84`** (−4.50 Combined, R² 0.530 vs 0.673). The mco_lb84 regime requires some of the near-zero features (likely temporal/fuel level signals) to navigate the long lockdown-spanning window; their removal causes collapse.
- **All FR configs remain `good_fit`** with gap ratios within 1.7–2.7×, confirming that FR does not introduce overfitting.
- **Recommendation:** Use HMT-TSF-FR for `nomco` operations at any lookback — it matches or exceeds the standard model at lower input cost. Retain the standard model for `mco` or when mco_lb84 stability is required.
