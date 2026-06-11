# Repository Revision Audit

> Date: 2026-06-11
> Scope: full-repository review — data cleaning, preprocessing, EDA, feature engineering, model scripts, results, findings, and discussion — for the thesis *"Mitigating Public Transit Unreliability through Deep Learning Analytics and Ridership Forecasting"*.

This document records every issue found, classifies it as **URGENT (fixed in this revision)** or **FUTURE WORK (documented, deliberately not fixed now)**, and lists the verified non-issues so the methodology can be defended confidently.

---

## 1. Urgent issues — fixed in this revision

### U1 · Feature-column-order misalignment (most significant finding)

Commit `34d859c` (2026-06-04) changed the column order emitted by `src/features/feature_align.py` to:

| Indices | Content |
|---------|---------|
| 0–12 | 12 service lines + `total_ridership` (idx 12) |
| 13–15 | `ridership_lag_{7,14,28}` |
| 16–17 | `year`, `day_of_year` |
| 18–32 | fuel prices (×15) |
| 33–46 | holiday flags / lead–lag / cyclical (×14) |
| 47–61 | rainfall (×15) |
| 62–78 | static (×17) |

All sequence tensors and all current model runs were built with this order (verified from `scaler_X.pkl` per-feature ranges), but three consumers still assumed the **old** order (targets → temporal → external → lag → static):

1. **SHAP labels** — `src/utils/shap_analysis.py::load_feature_names()` concatenated semantic groups in metadata order, so every SHAP output (plots, and the rankings narrated in `FINDING.md` §4.8–4.9) carried wrong feature names. Corrected mapping for every feature cited in the findings:

   | Cited label (old) | feat idx | True feature |
   |---|---|---|
   | `rail_komuter` | 12 | `total_ridership` |
   | `rail_mrt_kajang` | 4 | `rail_lrt_kj` |
   | `rail_lrt_ampang` | 3 | `rail_mrt_kajang` |
   | `fp_chg_ron95_budi95` | 42 | `is_weekend` |
   | `fp_lv_diesel_pct_chg` | 37 | `days_since_last_public_hol` |
   | `fp_chg_diesel` | 40 | `day_of_week` |
   | `day_of_week` | 20 | `fp_lv_diesel` |
   | `rainfall_mm__MY10` | 53 | `rainfall_mm__MY07` (Penang) |
   | `fp_lv_ron97_pct_chg` | 36 | `days_to_next_public_hol` |
   | `rainfall_mm__MY03` | 46 | `month_cos` |
   | `rainfall_mm__MY02` | 45 | `month_sin` |

   The per-index SHAP values and consistency statistics are unaffected — only the names were wrong. The corrected narrative actually *strengthens* the findings: the top predictor is system-total autocorrelation, and calendar structure (weekend, day-of-week, holiday lead–lag, seasonal encodings) is the second-most important group. The 26 dropped features in HMT-TSF-FR are, under true labels, 9 near-constant administered fuel-price columns + all 17 static features — the "static features are redundant" conclusion survives unchanged (indices 62–78 align in both orders).

2. **`X_future` content** — `sequence_builder.py` used a hardcoded `slice(13, 29)` for the "known future temporal" tensor, which after the reorder selected **lag + year/day_of_year + 11 fuel columns** instead of calendar features. Future lags and year/day_of_year are legitimately known at forecast time, but future *fuel prices* are not — a mild information leak (largely inert in practice: RON95 was frozen at RM2.05 across the training window). HMT-TSF therefore never received the future day-of-week/holiday context its documentation claimed.

3. **HMT-TSF feature grouping** — `FEAT_GROUPS` in `src/models/hybrid/hmttsf.py` partitioned the input on old boundaries, so its "temporal" encoder actually received lag+trend+fuel, its "lag" encoder received rainfall, etc. The model still trained (any partition works mechanically), but the "5 semantic group encoders" architectural claim was not true of the runs.

**Fixes applied:**
- `shap_analysis.load_feature_names()` now reads names from the features CSV header (authoritative), falling back to the new `column_order` metadata field; `feature_align.py` now writes `column_order` into both metadata JSONs (and the existing JSONs were patched).
- `sequence_builder.py` resolves the 16 temporal columns **by name** from `feature_metadata*.json` (they are non-contiguous), fails loudly if names are missing, and records `temporal_feat_indices`/`temporal_feat_names` in `split_dates.json`.
- `hmttsf.py` `FEAT_GROUPS` now matches the true layout using multi-segment groups; `FeatureGroupEncoder` accepts index segments; `_REDUCED_FEAT_GROUPS` and the temporal-position bookkeeping are now **derived programmatically** from `FEAT_GROUPS` + `_DROPPED_FEAT_INDICES` (no more hand-maintained constants); dropped-index comments carry true names.
- `FINDING.md` §4.8–4.9, `FUTURE.md` §5.2.5/§5.6, and `HMT-TSF.md` were relabelled/corrected.
- Verified end-to-end: sequences rebuilt into a scratch dir contain exactly the 16 calendar features in `X_future`, and a 2-epoch CPU run of the rewired HMT-TSF trains, evaluates, boosts, and saves correctly.

**Blast radius:** the 14 baseline models are numerically unaffected (they consume the 79 features order-agnostically and never use `X_future`). HMT-TSF results and all SHAP artifacts must be regenerated (see §3).

### U2 · HMT-TSF residual booster gated on test metrics (leakage)

`hmttsf.py` fit its residual booster on training residuals but decided whether to **apply** the boost by comparing test-set Combined% with and without it — a test-set-informed decision. Fixed: the apply/skip gate now uses **validation** Combined%; `results.json` now records `no_boost`, `use_catboost`, and `boost_applied`. The booster runs by default (`--no-boost` off), so all existing HMT-TSF results were produced under the old gate.

### U3 · Thesis-title vs. scope gap

The repository forecasts ridership; nothing modelled unreliability, and the word never appeared in the docs. Fixed by reframing rather than new modelling (no public Malaysian supply-side reliability data exists at daily granularity): `README.md` now opens with a scope section positioning 7-day demand forecasting as the analytics layer enabling unreliability mitigation (capacity planning, headway adjustment, demand–supply matching), and `FUTURE.md` carries the demand-side-only limitation plus a supply-side (AVL/on-time-performance) future-work direction.

### U4 · Single-run results presented without caveats

Every reported metric is a single run at seed 42 — no repeated-seed variance, confidence intervals, or Diebold–Mariano tests. Per the revision decision, tooling is **not** added now; instead the caveat is stated prominently in `README.md`, `FINDING.md` (header), `RESULTS.md`, and `HMT-TSF-RESULTS.md`, and multi-seed replication + significance testing is a named future-work item (`FUTURE.md` §5.5).

### U5 · Comparison loader had no configuration check

`comparison_table.load_model_results()` auto-globbed the latest `results.json` per model with no lookback/MCO compatibility check (the smoke test surfaced a live example: a lb14 run compared against lb56 Autoformer/Informer runs). Fixed: the loader prints the detected run's `lookback`/`T_in` and warns on mismatch when the caller passes `expect_lookback` (backwards-compatible).

### U6 · Hardcoded temporal slice in `sequence_builder.py`

Subsumed by U1: the slice is gone; temporal columns are resolved by name with loud failure on mismatch.

### U7 · Lag features bridge the excluded MCO window (undocumented)

In the no-MCO dataset, `ridership_lag_{7,14,28}` at the start of the post-MCO window reference dates inside the excluded MCO period. Deliberate (fabricating values would be worse) but previously undocumented. Now documented in `PIPELINE.md`, `DATA.md`, and as limitation 7 in `FUTURE.md`. Verified empirically: zero NaN in all lag columns of both feature CSVs.

### U8 · Combined% limitations undocumented

`METRICS.md` §1 now documents: profile masking (identical composites can hide different error profiles), floor truncation (all failures beyond 100 % summed error collapse to 0 and become indistinguishable), and the 1:1:1 weighting being a convention rather than an empirically derived preference.

### U9 · Reproducibility gaps

- No `requirements.txt` — added (derived from actual imports).
- Root-level `cuda.py` shadowed the `cuda.bindings` package that newer PyTorch builds import, breaking `import torch` from the repo root in REPL/`-c` contexts — renamed to `check_cuda.py` (docs updated).

---

## 2. Future work — documented, deliberately not fixed now

These are not fixed in this revision either because they would invalidate the 180+ existing run artifacts, or because they are larger refactors that don't change conclusions. Ordered by value.

| # | Item | Why future work |
|---|------|-----------------|
| F1 | **Multi-seed runs + statistical testing** (≥5 seeds per headline config; Diebold–Mariano or bootstrap CIs for pairwise comparisons) | Requires A100 re-runs; most important step toward publication-grade claims |
| F2 | **Supply-side reliability data integration** (AVL, on-time performance, disruption logs) as covariates and as targets | Data does not currently exist publicly for Malaysian operators; closes the thesis-title gap properly |
| F3 | **Shared training-loop module** — ~85–90 % of each of the 29 model scripts is copy-pasted boilerplate; tuned variants are full copies of base scripts | Large refactor; every bugfix currently needs manual replication across files |
| F4 | **Config centralization** — MCO dates, service launch dates, and paths are duplicated across `ridership.py`, `feature_align.py`, and docs | Touching the pipeline invalidates artifacts; bundle with the next full rebuild |
| F5 | **Holiday lead/lag sentinel 99** → −1 or a capped value, documented in metadata | Changes feature values → invalidates all sequences and results |
| F6 | **Full determinism** — add `torch.cuda.manual_seed_all` everywhere (currently HMT-TSF only) and optionally `cudnn.deterministic=True`; `cudnn.benchmark=True` currently allows kernel-level run-to-run variance | Bit-level reproducibility is not load-bearing for conclusions; document instead |
| F7 | **Unit tests** for split integrity, scaler fit boundaries, lag validity, MCO filtering, temporal-column resolution | No test suite exists; the new name-based resolution partially self-checks |
| F8 | **`fuelprice.py` intermediate scaler** is fit on a hardcoded 2022-01-01 cutoff rather than the actual train boundary | Harmless: that scaler is diagnostic-only; models use the sequence-level scalers fit on the train split |
| F9 | **Cyclical-encoding duplication** — dow/month sin/cos computed in both `ridership.py` and `holiday.py` | Implementations are identical; consolidation is cosmetic |
| F10 | **Booster fit on validation residuals** (currently fit on train residuals, gated on validation after U2) | Worth an ablation when HMT-TSF is re-run |
| F11 | **Rainfall interpolation limit (7 days) and OSM catchment radius (500 m)** are unjustified magic numbers | Sensitivity analysis candidates, not defects |

---

## 3. Required local re-runs (A100)

This environment has no GPU; the following must be regenerated locally to bring results in line with the fixed code:

1. **Rebuild all sequence sets** (corrected `X_future`, new `split_dates.json` keys):
   `python src/features/run_pipeline.py --skip-clean --skip-align`
   (plus the lookback-7/84 builds in `PIPELINE.md` if HMT-TSF lb7/lb84 are kept.)
2. **Re-run HMT-TSF and HMT-TSF-FR** across `{nomco, mco} × {lb7, lb14, lb28, lb56, lb84}` (± `--no-feat-reduce`), with `--shap` on the headline configs. Labels in SHAP outputs are correct automatically. Update `HMT-TSF-RESULTS.md`, `FINDING.md` §4.7–4.9, and the README headline tables from the new runs.
3. **Baseline models do not need re-running** — they are unaffected by U1/U2.
4. Optional but recommended while the GPU is warm: F1 (multi-seed) for the top-5 nomco·lb14 models.

---

## 4. Verified non-issues (defensible as-is)

- **No test-set leakage in training**: all 15 model scripts early-stop and select checkpoints on validation loss only; the test loader is touched only after training (e.g., `stlstm.py`).
- **Scaler hygiene**: sequence-level `MinMaxScaler`s for X and y are fit on the train split only; the same applies to the ridership cleaning scaler.
- **Split integrity**: chronological 70/15/15 split with no shuffling; sliding-window construction cannot straddle split boundaries (windows are built per split).
- **Metric correctness**: predictions are inverse-transformed to rider counts before metrics; the MAPE denominator `|y|+1` handles MCO near-zero days; `results.json` values reproduce `Combined% = 100 − MAPE − MAE% − RMSE%` exactly.
- **float16 sequence storage**: safe for MinMax-scaled [0,1] data (~1/1024 quantization, dominated by forecast error).
- **Lag computation**: lags are computed from the full pre-window history; the documented `.bfill()` artifact affects only the first ≤28 days of 2019 in the MCO-included train split, and the code comment describing it is accurate. Zero NaN verified in all lag columns.
- **`run_pipeline.py` / docs / data layout**: feature counts (79), date ranges, MCO window, and model inventory in the documentation all match the implementation.
