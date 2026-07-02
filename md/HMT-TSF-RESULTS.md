# HMT-TSF Results (Proposed Model)

> Last updated: 2026-07-02 · Source: `src/outputs/aggregate_hmttsf.csv`
> Companion files: `RESULTS.md` (16 baselines), `DIAGNOSTICS.md` (fit diagnostics)

This page reports the proposed model's results in full. It covers two versions —
HMT-TSF (all 79 features) and HMT-TSF-FR (a slimmed 53-feature version) — each tested
with and without the COVID period across five look-back windows. Alongside raw scores it
includes a comparison against a naive "same as last week" baseline, a three-block
stability check, and an over-/under-fitting diagnosis.

> **One caveat:** every number is a single run (fixed random seed), so treat very small
> gaps as indicative rather than definitive.

## 1. Conventions

- `nomco` / `mco`: MCO window (2020-03-18 – 2021-12-31) excluded / included.
- `lbNN`: look-back window in days. HMT-TSF extends the lookback grid to **7 and 84** beyond the baseline grid (14/28/56).
- Horizon `T_out = 7`.
- **Naive** = persistence baseline (last-value carried forward). **Delta** = model − naive (positive Δ on Combined%/R² = model better; negative Δ on error metrics = model better).
- **HMT-TSF-FR** = feature-reduced variant (53 features, 26 SHAP-zero features removed — see §10–11).
- Headline configuration: `nomco` (project default). MCO and lookback sweeps are robustness analysis.

---

## 2. Main results — all 10 configs (HMT-TSF, F=79)

| Config | Combined% | MAPE% | MAE% | RMSE% | R² | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| nomco_lb7 | 84.88 | 4.74 | 4.25 | 6.13 | 0.888 | 53,325 | 76,951 |
| **nomco_lb14** | **85.78** | **4.39** | **3.97** | **5.87** | **0.897** | **49,897** | **73,757** |
| nomco_lb28 | 84.23 | 4.94 | 4.55 | 6.28 | 0.883 | 57,268 | 78,910 |
| nomco_lb56 | 82.53 | 5.52 | 5.11 | 6.84 | 0.859 | 64,198 | 85,874 |
| nomco_lb84 | 77.68 | 6.99 | 6.59 | 8.73 | 0.773 | 82,174 | 108,800 |
| mco_lb7 | 78.61 | 6.89 | 6.12 | 8.38 | 0.807 | 75,401 | 103,170 |
| **mco_lb14** | **81.99** | 5.68 | 5.18 | 7.15 | 0.859 | 63,853 | 88,228 |
| mco_lb28 | 79.89 | 6.26 | 5.83 | 8.02 | 0.822 | 72,032 | 99,059 |
| mco_lb56 | 79.47 | 6.68 | 5.91 | 7.95 | 0.825 | 73,334 | 98,598 |
| mco_lb84 | 78.45 | 6.73 | 6.17 | 8.64 | 0.791 | 76,593 | 107,293 |

**Read-out:**
- **Best full-model config: `nomco_lb14` — 85.78 Combined%, R² 0.897, MAE 49,897.** The study-wide best is the feature-reduced variant at the same configuration (HMT-TSF-FR `nomco_lb14`: **86.59**, R² 0.906, MAE 46,323 — §11), which beats the top baseline (Informer tuned, 79.99) by **+6.60 Combined**.
- **nomco lookback profile is single-peaked at lb14**: lb7 (84.88) < lb14 (85.78) > lb28 (84.23) > lb56 (82.53) > lb84 (77.68). The 14-day window remains decisively optimal for a 7-day horizon.
- **Under MCO the lb14 peak persists** (81.99), unlike the pre-fix runs where lb56 led: with the correct future-calendar conditioning, short windows no longer need long context to absorb the lockdown discontinuity. mco_lb84 (78.45) actually *exceeds* nomco_lb84 (77.68) — the only lookback where the MCO condition wins, because nomco_lb84 is the study's sole overfit configuration (§6).

---

## 3. Versus naive persistence

Model vs naive (last-value) baseline, Combined% and R², with Δ.

| Config | Model Combined% | Naive Combined% | Δ Combined | Model R² | Naive R² | Δ R² |
|---|---:|---:|---:|---:|---:|---:|
| nomco_lb7 | 84.88 | 34.80 | +50.09 | 0.888 | −0.963 | +1.851 |
| nomco_lb14 | 85.78 | 34.83 | **+50.94** | 0.897 | −0.961 | +1.858 |
| nomco_lb28 | 84.23 | 34.64 | +49.59 | 0.883 | −0.969 | +1.852 |
| nomco_lb56 | 82.53 | 34.99 | +47.54 | 0.859 | −0.960 | +1.819 |
| nomco_lb84 | 77.68 | 34.22 | +43.46 | 0.773 | −0.969 | +1.742 |
| mco_lb7 | 78.61 | 25.73 | +52.88 | 0.807 | −1.132 | +1.939 |
| mco_lb14 | 81.99 | 25.62 | **+56.37** | 0.859 | −1.137 | +1.997 |
| mco_lb28 | 79.89 | 25.77 | +54.12 | 0.822 | −1.142 | +1.964 |
| mco_lb56 | 79.47 | 25.62 | +53.85 | 0.825 | −1.157 | +1.982 |
| mco_lb84 | 78.45 | 25.76 | +52.69 | 0.791 | −1.160 | +1.952 |

**Read-out:**
- Naive persistence has **negative R² in every config** (−0.96 nomco, −1.14 mco) — it is worse than predicting the mean, confirming ridership has strong non-persistent dynamics (weekly seasonality, holidays, fuel/weather shocks).
- HMT-TSF beats naive by **+50.9 Combined / +1.86 R² at nomco_lb14**, and the margin is even larger under MCO (**+56.4 Combined at mco_lb14**), because the naive baseline degrades faster than the model when the lockdown break is included.

---

## 4. MCO robustness

| Lookback | nomco Combined% | mco Combined% | Δ (mco − nomco) | nomco R² | mco R² |
|---|---:|---:|---:|---:|---:|
| lb7 | 84.88 | 78.61 | −6.27 | 0.888 | 0.807 |
| lb14 | 85.78 | 81.99 | −3.79 | 0.897 | 0.859 |
| lb28 | 84.23 | 79.89 | −4.34 | 0.883 | 0.822 |
| lb56 | 82.53 | 79.47 | −3.06 | 0.859 | 0.825 |
| lb84 | 77.68 | 78.45 | **+0.77** | 0.773 | 0.791 |

**Read-out:**
- HMT-TSF's MCO degradation at the headline lookback is **−3.79** — the smallest lb14 degradation in the study (Informer tuned: −4.64) — and it retains the **highest absolute MCO accuracy of any model (81.99)**, clearing the 75% target with room to spare.
- At lb84 the sign flips (+0.77): the MCO-included condition wins, but only because nomco_lb84 overfits (§6) — not because long context is desirable.
- This robustness is the core empirical justification for the regime-embedding component of the architecture. Note that the FR variant is noticeably **less** MCO-robust (−7.85 at lb14, §11) — the dropped near-constant fuel levels evidently still anchor the model under the structural break.

---

## 5. Walk-forward validation (3 blocks, HMT-TSF)

Each config was re-validated on 3 sequential out-of-sample blocks. Combined% / R² per block.

| Config | Block 1 | Block 2 | Block 3 | Spread (Comb.) |
|---|---|---|---|---:|
| nomco_lb7 | 88.36 / 0.936 | 81.72 / 0.843 | 84.88 / 0.887 | 6.6 |
| nomco_lb14 | 90.62 / 0.960 | 82.14 / 0.846 | 85.12 / 0.887 | 8.5 |
| nomco_lb28 | 87.08 / 0.927 | 78.57 / 0.810 | 87.50 / 0.915 | 8.9 |
| nomco_lb56 | 75.81 / 0.757 | 86.81 / 0.929 | 85.71 / 0.893 | 11.0 |
| nomco_lb84 | 73.44 / 0.707 | 78.28 / 0.803 | 81.50 / 0.819 | 8.1 |
| mco_lb7 | 71.99 / 0.712 | 84.66 / 0.899 | 79.47 / 0.804 | 12.7 |
| mco_lb14 | 77.66 / 0.807 | 87.33 / 0.929 | 81.27 / 0.840 | 9.7 |
| mco_lb28 | 73.74 / 0.740 | 85.94 / 0.901 | 80.27 / 0.825 | 12.2 |
| mco_lb56 | 75.42 / 0.790 | 81.64 / 0.841 | 81.36 / 0.843 | 6.2 |
| mco_lb84 | 76.12 / 0.767 | 79.45 / 0.809 | 79.78 / 0.794 | 3.7 |

**Read-out:**
- All blocks stay strong (no block drops below ~72 Combined / 0.71 R²) — performance is **not driven by a single favourable test split**.
- Under `nomco` at short lookbacks, **Block 2 is the weakest** (e.g., lb14: 82.14 vs 90.62/85.12), indicating a harder middle segment; under `mco`, **Block 1 is the weakest** and Block 2 the strongest — consistent with the lockdown signal landing in different blocks across the two splits.
- **mco_lb84 is the most temporally stable** config (spread 3.7), followed by mco_lb56 (6.2); among nomco configs, lb7 is the most stable (6.6). The headline nomco_lb14 spread is 8.5.

---

## 6. Fit diagnosis

9 of 10 HMT-TSF configs are diagnosed **`good_fit`**; the sole exception is `nomco_lb84` (overfit — validation drift +36.9% after epoch 18). HMT-TSF-FR is `good_fit` in all 10 configs (§11). See `DIAGNOSTICS.md` for the cross-model comparison.

| Config | Verdict | Val drift % | Gap ratio | Val trend | Best/Total epochs |
|---|---|---:|---:|---|---|
| nomco_lb7 | good_fit | 0.00 | 1.69 | flat | 150 / 150 |
| nomco_lb14 | good_fit | 0.61 | 1.57 | flat | 50 / 70 |
| nomco_lb28 | good_fit | 3.77 | 2.03 | flat | 118 / 143 |
| nomco_lb56 | good_fit | 12.33 | 2.64 | flat | 47 / 67 |
| nomco_lb84 | **overfit** | **36.95** | 2.45 | falling | 18 / 48 |
| mco_lb7 | good_fit | 3.34 | 1.72 | flat | 77 / 102 |
| mco_lb14 | good_fit | 4.36 | 1.85 | flat | 77 / 97 |
| mco_lb28 | good_fit | 12.00 | 2.18 | flat | 78 / 103 |
| mco_lb56 | good_fit | 12.85 | 2.35 | falling | 28 / 48 |
| mco_lb84 | good_fit | 3.40 | 2.12 | flat | 96 / 126 |

**Read-out:**
- **Gap ratio (val/train loss) stays within 1.6–2.6× — always under the 3× overfit threshold.** No config shows memorisation; every LSTM-family and STGCN-family baseline exceeds 3× (often far more — up to 37×; see `DIAGNOSTICS.md`).
- The single `overfit` verdict (nomco_lb84) is drift-driven, not gap-driven: validation loss degraded +37% after its early best (epoch 18 of 48). The 84-day window over a 7-day horizon remains the study's clearest "too much context" case — and the FR variant fixes it (nomco_lb84 `good_fit`, +4.26 Combined; §11).
- The headline nomco_lb14 run has the cleanest profile in the study: gap 1.57×, drift 0.61%.

---

## 7. Verdict

1. **HMT-TSF leads the study at its default `nomco_lb14`**: 85.78 Combined%, R² 0.897, MAE 49,897 — and the feature-reduced variant raises the study-wide headline to **86.59 / 0.906 / 46,323**, +6.60 Combined over the best tuned baseline (Informer tuned, 79.99).
2. **It generalises cleanly** — 19 of 20 runs across both variants are `good_fit` (sole exception: full-model nomco_lb84), with gap ratios capped at 2.7× vs baselines' 3–37×.
3. **It is the most shock-robust model** — smallest lb14 MCO degradation (−3.79) and the highest retained MCO accuracy in the study (81.99).
4. **Lookback recommendation: lb14 everywhere.** With corrected future-calendar conditioning, lb14 is optimal in both regimes; lb84 is never optimal and overfits at nomco.
5. **Walk-forward confirms the headline is not split-luck** — all three blocks hold up (≥82.1 Combined at nomco_lb14).

---

## 8. Use-case scenarios

| Scenario | Recommended HMT-TSF config | Rationale |
|---|---|---|
| **Default production forecasting** | HMT-TSF-FR `nomco_lb14` | Study-wide best (86.59 / R² 0.906 / MAE 46,323) at 53-feature input cost. |
| **Shock / lockdown / structural-break regime** | full HMT-TSF `mco_lb14` | Best MCO accuracy in the study (81.99); the full feature set is markedly more MCO-robust than FR (Δ −3.79 vs −7.85). |
| **Maximum temporal stability across periods** | `mco_lb84` (full) | Lowest block-to-block spread (3.7) in walk-forward. |
| **Lowest absolute error** | HMT-TSF-FR `nomco_lb14` | MAE 46,323 / RMSE 70,664 — lowest in the study. |
| **Interpretability / driver analysis** | `nomco_lb14` with `--shap` | Cleanest fit in the study (gap 1.57×) makes attributions trustworthy. |
| **Drop from the grid** | `nomco_lb84` (full) | The study's only overfit config; strictly dominated by shorter windows. |

> All numbers verified against `aggregate_hmttsf.csv` (regenerated 2026-06-11).

---

## 9. SHAP Feature Importance

### Method

SHAP values were computed using `shap.GradientExplainer` on the trained `nomco_lb14` model. The first 100 test samples (shape `100 × 14 × 79`) were used as the background set. The explainer produced an absolute SHAP array of shape `(100, 14, 79, 7)`; values were averaged over the sample, time-step, and horizon dimensions, yielding a single 79-element importance vector — one score per input feature. Feature names are resolved from the aligned CSV column order (`feature_metadata.json → column_order`), the authoritative index→name mapping after the 2026-06-11 fix.

### Top 10 Features — headline run (`nomco_lb14`, by mean |SHAP|)

| SHAP rank | feat index | Feature name | Group | Mean \|SHAP\| | Interpretation |
|-----------|-----------|--------------|-------|---:|----------------|
| 1 | feat_46 | month_cos | Temporal | 0.391 | Annual-cycle position (cosine) — seasonal demand level |
| 2 | feat_45 | month_sin | Temporal | 0.332 | Annual-cycle position (sine) |
| 3 | feat_41 | month | Temporal | 0.309 | Raw month index; reinforces the seasonal signal |
| 4 | feat_12 | total_ridership | Ridership | 0.011 | Aggregate demand history — strongest non-seasonal driver |
| 5 | feat_20 | fp_lv_diesel | Fuel — external | 0.006 | Diesel price level; mode-substitution signal |
| 6 | feat_36 | days_to_next_public_hol | Temporal | 0.003 | Holiday anticipation (pre-holiday dips/surges) |
| 7 | feat_19 | fp_lv_ron97 | Fuel — external | 0.002 | RON97 price level (the unfrozen petrol grade) |
| 8 | feat_55 | rainfall_mm__MY09 | Rainfall — external | 0.001 | Perlis rainfall — northern corridor weather |
| 9 | feat_57 | rainfall_mm__MY11 | Rainfall — external | 0.001 | Terengganu rainfall — monsoon corridor |
| 10 | feat_53 | rainfall_mm__MY07 | Rainfall — external | 0.001 | Penang rainfall — major urban transit market |

### Interpretation

Seasonal-position encodings (`month_cos`, `month_sin`, `month`) carry by far the largest attribution magnitudes, indicating the model anchors its 7-day forecasts on where in the annual demand cycle the window sits — school terms, festive seasons, and the northeast monsoon all phase-lock to the calendar. The strongest non-seasonal driver is `total_ridership` itself, confirming aggregate autocorrelation, followed by the two *unfrozen* fuel price series (diesel and RON97 — the administered RON95 variants are SHAP-zero; §10) and holiday anticipation. Regional rainfall enters at ranks 8–10. Static infrastructure features (indices 62–78) rank at exactly zero throughout — they carry no day-to-day variance and the model correctly ignores them. Note that magnitude and cross-run consistency give complementary views: ranked by how often a feature reaches the top-15 across all ten configurations, `total_ridership` is first (10/10 runs; §10).

### Full Feature Index → Name Mapping (aligned column order)

#### Indices 0–12 — Target / Ridership

| Index | Feature |
|-------|---------|
| 0 | bus_rkl |
| 1 | bus_rpn |
| 2 | rail_lrt_ampang |
| 3 | rail_mrt_kajang |
| 4 | rail_lrt_kj |
| 5 | rail_monorail |
| 6 | rail_mrt_pjy |
| 7 | rail_ets |
| 8 | rail_intercity |
| 9 | rail_komuter_utara |
| 10 | rail_tebrau |
| 11 | rail_komuter |
| 12 | total_ridership |

#### Indices 13–17 — Lag + trend

| Index | Feature |
|-------|---------|
| 13 | ridership_lag_7 |
| 14 | ridership_lag_14 |
| 15 | ridership_lag_28 |
| 16 | year |
| 17 | day_of_year |

#### Indices 18–32 — External: Fuel Prices

| Index | Feature |
|-------|---------|
| 18 | fp_lv_ron95 |
| 19 | fp_lv_ron97 |
| 20 | fp_lv_diesel |
| 21 | fp_lv_diesel_eastmsia |
| 22 | fp_lv_ron95_budi95 |
| 23 | fp_lv_ron95_skps |
| 24 | fp_lv_ron95_pct_chg |
| 25 | fp_lv_ron97_pct_chg |
| 26 | fp_lv_diesel_pct_chg |
| 27 | fp_chg_ron95 |
| 28 | fp_chg_ron97 |
| 29 | fp_chg_diesel |
| 30 | fp_chg_diesel_eastmsia |
| 31 | fp_chg_ron95_budi95 |
| 32 | fp_chg_ron95_skps |

#### Indices 33–46 — Temporal: holiday / cyclical

| Index | Feature |
|-------|---------|
| 33 | is_public_holiday |
| 34 | is_school_holiday |
| 35 | is_holiday_any |
| 36 | days_to_next_public_hol |
| 37 | days_since_last_public_hol |
| 38 | days_to_next_school_hol |
| 39 | days_since_last_school_hol |
| 40 | day_of_week |
| 41 | month |
| 42 | is_weekend |
| 43 | dow_sin |
| 44 | dow_cos |
| 45 | month_sin |
| 46 | month_cos |

#### Indices 47–61 — External: Rainfall by State

| Index | Feature | State |
|-------|---------|-------|
| 47 | rainfall_mm__MY01 | Johor |
| 48 | rainfall_mm__MY02 | Kedah |
| 49 | rainfall_mm__MY03 | Kelantan |
| 50 | rainfall_mm__MY04 | Melaka |
| 51 | rainfall_mm__MY05 | Negeri Sembilan |
| 52 | rainfall_mm__MY06 | Pahang |
| 53 | rainfall_mm__MY07 | Pulau Pinang |
| 54 | rainfall_mm__MY08 | Perak |
| 55 | rainfall_mm__MY09 | Perlis |
| 56 | rainfall_mm__MY10 | Selangor |
| 57 | rainfall_mm__MY11 | Terengganu |
| 58 | rainfall_mm__MY12 | Sabah |
| 59 | rainfall_mm__MY13 | Sarawak |
| 60 | rainfall_mm__MY15 | W.P. Kuala Lumpur |
| 61 | rainfall_mm__MY17 | W.P. Putrajaya |

> MY14 (Labuan) and MY16 are excluded from the pipeline.

#### Indices 62–78 — Static

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

SHAP values from all 10 HMT-TSF configurations (nomco/mco × lb7/lb14/lb28/lb56/lb84) were aggregated independently. Each `shap_values.npy` has shape `(100, T_in, 79, 7)`; absolute values were averaged over samples, time steps, and forecast horizons to produce a 79-element importance vector per run. Features were then classified by how many runs assigned them zero importance. Full table: `src/outputs/shap_crossrun_summary.csv` (regenerate with `python src/utils/shap_crossrun.py`).

### Universal Zeros — zero in ALL 10 runs (23 features)

All 17 static features plus 6 administered/secondary fuel-price columns:

| Index | Feature | Group | Why dead |
|-------|---------|-------|---------|
| 21 | fp_lv_diesel_eastmsia | Fuel | East Malaysia price level; irrelevant to Peninsular corridors |
| 22 | fp_lv_ron95_budi95 | Fuel | Subsidy-scheme price; near-constant |
| 23 | fp_lv_ron95_skps | Fuel | Subsidy-scheme price; near-constant |
| 30 | fp_chg_diesel_eastmsia | Fuel | Change of a near-constant series |
| 31 | fp_chg_ron95_budi95 | Fuel | Change of a near-constant series |
| 32 | fp_chg_ron95_skps | Fuel | Change of a near-constant series |
| 62–63 | pop_density_median / log_median | Static | Time-invariant; no day-to-day variation |
| 64–67 | gtfs_n_stops / n_routes / n_directed_edges / avg_segment_s | Static | Same |
| 68–75 | osm_poi_* (8 categories) | Static | Same |
| 76–78 | gadm_n_states / n_border_pairs / mean_border_km | Static | Constant scalars |

> The entire Static group (OSM, GTFS, GADM, population) is dead weight across every configuration — confirmed across both regimes and all five lookback windows.

### Near-Universal Zeros — zero in 8/10 runs (3 features)

| Index | Feature | Zero in | Note |
|-------|---------|---------|------|
| 18 | fp_lv_ron95 | 8/10 | RON95 was frozen at RM2.05 across the post-MCO training window |
| 24 | fp_lv_ron95_pct_chg | 8/10 | Percentage change of the frozen series |
| 27 | fp_chg_ron95 | 8/10 | Absolute change of the frozen series |

### Consistently Important Features — top-15 in ≥5/10 runs

| Consistency | Index | Feature | Group |
|------------|-------|---------|-------|
| 10/10 (100%) | feat_12 | total_ridership | Ridership |
| 8/10 (80%) | feat_36 | days_to_next_public_hol | Temporal |
| 7/10 (70%) | feat_45 | month_sin | Temporal |
| 7/10 (70%) | feat_04 | rail_lrt_kj | Ridership |
| 6/10 (60%) | feat_46 | month_cos | Temporal |
| 6/10 (60%) | feat_41 | month | Temporal |
| 6/10 (60%) | feat_20 | fp_lv_diesel | Fuel |
| 5/10 (50%) | feat_35 | is_holiday_any | Temporal |
| 5/10 (50%) | feat_34 | is_school_holiday | Temporal |
| 5/10 (50%) | feat_19 | fp_lv_ron97 | Fuel |
| 5/10 (50%) | feat_38 | days_to_next_school_hol | Temporal |
| 5/10 (50%) | feat_37 | days_since_last_public_hol | Temporal |

> Calendar structure dominates the consistent set: holiday anticipation/recovery, seasonal-position encodings, and school terms — alongside aggregate ridership autocorrelation and the two unfrozen fuel grades (diesel, RON97).

### Removal Summary

| Tier | Count | Criterion | Action |
|------|-------|-----------|-------------------|
| Universal zeros | 23 | Zero in all 10 runs | Dropped |
| Near-universal zeros | 3 | Zero in 8/10 runs | Dropped |
| **Total** | **26** | 79 → **53 features** | ~33% input dimension reduction |

Removing 26 features shrinks the input tensor from `(B, T_in, 79)` → `(B, T_in, 53)`, reduces the graph adjacency from `79×79` → `53×53`, and cuts GradientExplainer memory by ~33%. The post-fix runs confirm the removal is not merely lossless under normal conditions but **beneficial** (§11) — while costing MCO robustness.

---

## 11. HMT-TSF vs HMT-TSF-FR Comparison

HMT-TSF-FR is the feature-reduced variant trained on 53 features (26 SHAP-zero features removed per §10). All training hyperparameters are identical; only the input dimension changes.

### Combined% — HMT-TSF vs HMT-TSF-FR

| Config | HMT-TSF | HMT-TSF-FR | Δ (FR − std) |
|---|---:|---:|---:|
| nomco_lb7 | 84.88 | 84.81 | −0.07 |
| nomco_lb14 | 85.78 | **86.59** | **+0.81** |
| nomco_lb28 | 84.23 | 86.07 | **+1.84** |
| nomco_lb56 | 82.53 | 84.04 | **+1.52** |
| nomco_lb84 | 77.68 | 81.94 | **+4.26** |
| mco_lb7 | 78.61 | 77.76 | −0.85 |
| mco_lb14 | **81.99** | 78.74 | −3.25 |
| mco_lb28 | 79.89 | 76.45 | −3.44 |
| mco_lb56 | 79.47 | 75.90 | −3.57 |
| mco_lb84 | 78.45 | 73.23 | −5.22 |

### R² — HMT-TSF vs HMT-TSF-FR

| Config | HMT-TSF R² | HMT-TSF-FR R² | Δ R² |
|---|---:|---:|---:|
| nomco_lb14 | 0.897 | 0.906 | +0.008 |
| nomco_lb28 | 0.883 | 0.903 | +0.020 |
| nomco_lb56 | 0.859 | 0.878 | +0.019 |
| nomco_lb84 | 0.773 | 0.848 | +0.075 |
| mco_lb14 | 0.859 | 0.780 | −0.079 |
| mco_lb84 | 0.791 | 0.680 | −0.112 |

### Fit Diagnosis — HMT-TSF-FR (10/10 `good_fit`)

| Config | Verdict | Val drift % | Gap ratio | Val trend | Best/Total epochs |
|---|---|---:|---:|---|---|
| nomco_lb7 | good_fit | 0.76 | 1.68 | flat | 135 / 150 |
| nomco_lb14 | good_fit | 2.94 | 1.66 | flat | 69 / 89 |
| nomco_lb28 | good_fit | 6.27 | 2.05 | flat | 110 / 135 |
| nomco_lb56 | good_fit | 13.39 | 2.52 | falling | 45 / 65 |
| nomco_lb84 | good_fit | 11.08 | 2.72 | falling | 61 / 91 |
| mco_lb7 | good_fit | 2.69 | 1.70 | flat | 65 / 90 |
| mco_lb14 | good_fit | 4.36 | 1.86 | flat | 75 / 95 |
| mco_lb28 | good_fit | 5.98 | 2.20 | flat | 74 / 99 |
| mco_lb56 | good_fit | 7.90 | 2.62 | flat | 48 / 68 |
| mco_lb84 | good_fit | 3.49 | 2.18 | flat | 96 / 126 |

### Read-out

- **FR is now the study-wide headline**: at `nomco_lb14` it reaches 86.59 Combined% / R² 0.906 / MAE 46,323 — **+0.81 over the full model** and +6.60 over the best tuned baseline. Walk-forward holds (91.39 / 83.26 / 85.70 per block).
- **FR wins every nomco lookback from lb14 up**, with the margin growing with window length (+0.81 → +4.26 at lb84). Removing the 26 zero-signal features acts as noise regularisation under long autocorrelation windows — it even converts the full model's only `overfit` config (nomco_lb84) into a comfortable `good_fit` (+4.26 Combined, R² +0.075).
- **FR loses every mco config** (−0.85 to −5.22), with degradation deepening as the window grows. Under the structural break, the near-constant fuel-level columns evidently provide a stabilising anchor that the reduced model lacks. FR's lb14 MCO degradation is −7.85 vs the full model's −3.79.
- **All FR configs remain `good_fit`** with gap ratios within 1.7–2.7×, confirming that feature reduction does not introduce overfitting.
- **Recommendation:** Use HMT-TSF-FR for `nomco` operations at any lookback ≥14 — it dominates the full model at lower input cost. Retain the full model for `mco`/shock-prone regimes, where its feature redundancy buys robustness.
