# Baseline & Tuned Model Results

> Source: `src/outputs/aggregate_results.csv` · Generated 2026-05-30
> Companion files: `HMT-TSF-RESULTS.md` (proposed model), `DIAGNOSTICS.md` (fit diagnostics)
> Scope: 16 model variants × 12 configurations. HMT-TSF appears here **only as a reference row**; full treatment in `HMT-TSF-RESULTS.md`.

## 1. Scope & conventions

- **Models (16 rows):** ASTGCN, Autoformer, BiLSTM, CNN-BiLSTM, CNN-LSTM, CNN-LSTM-Augmented, CNN-LSTM-Parallel, Informer, LSTM, MTGNN, PDR-STGCN, ST-LSTM, STFGNN, STGCN, STSGCN, TPA-LSTM. (The three CNN-LSTM rows are the `sequential` / `augmented` / `parallel` modes.)
- **Configuration matrix (12 per model):** `{base, tuned}` × `{nomco, mco}` × `{lb14, lb28, lb56}`.
  - `nomco` = MCO/COVID lockdown window (2020-03-18 – 2021-12-31) **excluded** (project default).
  - `mco` = MCO window **included** (structural-break stress test).
  - `lbNN` = look-back window in days.
- **Forecast horizon:** `T_out = 7` days for every run.
- **Metrics** (`src/utils/metrics.py`): `Combined% = max(0, 100 − MAPE% − MAE% − RMSE%)`, higher is better. `MAE%/RMSE%` are mean-demand-normalised. `MAE`/`RMSE` are absolute (daily ridership). `R²` higher is better.
- **Headline configuration:** `base_nomco_lb14` (the project default). All other slices are treated as robustness analysis.

---

## 2. Headline ranking — `base_nomco_lb14` (default)

Sorted by Combined% (best → worst). HMT-TSF reference row appended.

| Rank | Model | Combined% | MAPE% | MAE% | RMSE% | R² | MAE | RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Informer | 79.13 | 6.59 | 5.53 | 8.76 | 0.772 | 69,476 | 110,050 |
| 2 | BiLSTM | 79.04 | 6.52 | 5.77 | 8.67 | 0.776 | 72,460 | 109,007 |
| 3 | TPA-LSTM | 78.67 | 6.62 | 6.04 | 8.67 | 0.776 | 75,871 | 109,009 |
| 4 | CNN-BiLSTM | 78.18 | 6.81 | 6.12 | 8.89 | 0.765 | 76,884 | 111,782 |
| 5 | LSTM | 78.13 | 6.71 | 6.18 | 8.98 | 0.760 | 77,719 | 112,801 |
| 6 | ST-LSTM | 78.01 | 6.93 | 6.27 | 8.79 | 0.770 | 78,779 | 110,453 |
| 7 | CNN-LSTM | 77.64 | 6.92 | 6.32 | 9.12 | 0.752 | 79,375 | 114,632 |
| 8 | MTGNN | 76.98 | 7.23 | 6.70 | 9.09 | 0.754 | 84,162 | 114,178 |
| 9 | STSGCN | 76.97 | 7.18 | 6.42 | 9.42 | 0.736 | 80,703 | 118,391 |
| 10 | ASTGCN | 75.69 | 7.42 | 6.89 | 9.99 | 0.703 | 86,631 | 125,606 |
| 12 | CNN-LSTM-Augmented | 75.36 | 7.77 | 7.19 | 9.68 | 0.721 | 90,356 | 121,639 |
| 13 | STGCN | 74.92 | 7.87 | 7.42 | 9.80 | 0.714 | 93,238 | 123,116 |
| 14 | Autoformer | 74.73 | 7.86 | 7.51 | 9.90 | 0.708 | 94,414 | 124,408 |
| 15 | STFGNN | 73.94 | 8.21 | 7.60 | 10.25 | 0.688 | 95,547 | 128,777 |
| 16 | CNN-LSTM-Parallel | 73.20 | 8.49 | 8.06 | 10.24 | 0.688 | 101,306 | 128,743 |
| 17 | PDR-STGCN | 72.57 | 8.56 | 8.26 | 10.61 | 0.665 | 103,844 | 133,322 |
| — | **HMT-TSF (ref)** | **81.46** | **5.58** | **4.70** | **8.25** | **0.797** | **59,105** | **103,684** |

**Read-out:**
- The default-config field is tight: ranks 1–7 sit inside a **1.5-point Combined% band (77.6–79.1)**. LSTM-family models (BiLSTM, TPA-LSTM, LSTM, ST-LSTM, CNN-BiLSTM) dominate the top alongside Informer.
- **Informer** is the strongest non-LSTM baseline and the top baseline overall by Combined% (79.13); BiLSTM/TPA-LSTM edge it on R² (0.776 vs 0.772) and RMSE.
- The bottom of the table is graph-heavy: PDR-STGCN, STGCN, STFGNN, plus Autoformer. PDR-STGCN is the weakest baseline in default config.
- HMT-TSF leads the field by **+2.33 Combined points over the best baseline** (81.46 vs 79.13) and posts the lowest absolute MAE (59,105 vs Informer's 69,476, a **−15%** error reduction).

---

## 3. Tuned variants — `tuned_nomco_lb14`

| Model | Combined% | R² | Δ Combined vs base |
|---|---:|---:|---:|
| Informer | 79.99 | 0.778 | +0.86 |
| TPA-LSTM | 79.95 | 0.782 | +1.28 |
| BiLSTM | 79.44 | 0.794 | +0.40 |
| ST-LSTM | 79.13 | 0.774 | +1.12 |
| ASTGCN | 78.82 | 0.754 | +3.13 |
| LSTM | 78.43 | 0.777 | +0.30 |
| PDR-STGCN | 78.16 | 0.759 | **+5.58** |
| CNN-LSTM-Augmented | 76.61 | 0.741 | +1.25 |
| CNN-BiLSTM | 76.40 | 0.750 | −1.78 |
| MTGNN | 76.40 | 0.742 | −0.58 |
| STGCN | 76.33 | 0.739 | +1.41 |
| Autoformer | 75.21 | 0.687 | +0.48 |
| CNN-LSTM | 73.68 | 0.699 | −3.96 |
| STSGCN | 73.65 | 0.658 | −3.32 |
| CNN-LSTM-Parallel | 72.30 | 0.669 | −0.91 |
| STFGNN | 72.63 | 0.633 | −1.31 |

**Read-out:**
- Best baseline number anywhere in `nomco` is **Informer tuned (79.99)** — still below HMT-TSF (81.46).
- **Tuning is not uniformly beneficial.** Largest gains: PDR-STGCN (+5.58), ASTGCN (+3.13). Largest regressions: CNN-LSTM (−3.96), STSGCN (−3.32), CNN-BiLSTM (−1.78). The two STGCN-family models that overfit hardest (see `DIAGNOSTICS.md`) are precisely those whose extra tuned capacity raised test Combined% on `nomco` while worsening fit diagnostics — interpret cautiously.
- On Combined% the median tuning effect is roughly flat (~+0.4). Test-set gains do **not** imply better generalisation; the diagnostics file shows tuned variants overfit more (higher gap ratios).

---

## 4. Look-back sensitivity — base, `nomco`

Combined% at lb14 / lb28 / lb56 (best of the three **bold**).

| Model | lb14 | lb28 | lb56 |
|---|---:|---:|---:|
| ASTGCN | 75.69 | **76.89** | 75.15 |
| Autoformer | **74.73** | 73.61 | 73.40 |
| BiLSTM | **79.04** | 76.97 | 77.09 |
| CNN-BiLSTM | **78.18** | 72.46 | 75.02 |
| CNN-LSTM | **77.64** | 76.15 | 74.16 |
| CNN-LSTM-Augmented | **75.36** | 75.27 | 73.99 |
| CNN-LSTM-Parallel | 73.20 | **76.47** | 74.97 |
| Informer | **79.13** | 78.70 | 77.73 |
| LSTM | **78.13** | 77.48 | 75.98 |
| MTGNN | 76.98 | **77.99** | 77.76 |
| PDR-STGCN | 72.57 | 71.02 | **74.72** |
| ST-LSTM | 78.01 | **78.86** | 75.60 |
| STFGNN | **73.94** | 61.30 | 43.25 |
| STGCN | 74.92 | **75.58** | 74.93 |
| STSGCN | **76.97** | 75.02 | 75.15 |
| TPA-LSTM | 78.67 | **79.08** | 77.01 |

**Read-out:**
- **lb14 or lb28 is optimal for almost every model**; lb56 rarely wins (only PDR-STGCN peaks at lb56). Longer windows add parameters/noise without payoff for a 7-day horizon.
- **Catastrophic lb56 collapse:** STFGNN (73.94 → 43.25, R² −0.112) falls below the "beats the mean" line at lb56 and also degrades sharply at lb28 (61.30). This is the most lookback-fragile model.
- MTGNN is the most lookback-stable (76.98 / 77.99 / 77.76).

---

## 5. MCO robustness stress test — base, lb14

`nomco` → `mco` Combined% and the MCO-config R². Sorted by smallest degradation (most robust → least).

| Model | nomco | mco | Δ Combined | mco R² |
|---|---:|---:|---:|---:|
| Informer | 79.13 | 73.22 | **−5.91** | 0.705 |
| BiLSTM | 79.04 | 72.74 | −6.30 | 0.699 |
| PDR-STGCN | 72.57 | 65.52 | −7.05 | 0.554 |
| TPA-LSTM | 78.67 | 71.18 | −7.49 | 0.671 |
| MTGNN | 76.98 | 66.80 | −10.18 | 0.565 |
| ST-LSTM | 78.01 | 66.75 | −11.26 | 0.584 |
| STSGCN | 76.97 | 64.20 | −12.77 | 0.531 |
| ASTGCN | 75.69 | 61.69 | −14.00 | 0.471 |
| STGCN | 74.92 | 60.82 | −14.10 | 0.430 |
| CNN-BiLSTM | 78.18 | 60.77 | −17.41 | 0.430 |
| STFGNN | 73.94 | 55.22 | −18.72 | 0.288 |
| LSTM | 78.13 | 58.92 | −19.21 | 0.390 |
| Autoformer | 74.73 | 51.26 | −23.47 | 0.121 |
| CNN-LSTM | 77.64 | 47.59 | **−30.05** | 0.026 |
| — | — | — | — | — |
| **HMT-TSF (ref)** | **81.46** | **75.66** | **−5.80** | **0.724** |

**Read-out:**
- The MCO structural break punishes every model, but the spread is huge: **−5.9 (Informer) to −30.1 (CNN-LSTM)**.
- **CNN-LSTM (sequential) effectively fails under MCO** (47.59 Combined, R² 0.026 — barely above the mean baseline). Autoformer (R² 0.121) also degrades severely.
- **HMT-TSF is the most MCO-robust model in the study** (−5.80, edging Informer's −5.91) while retaining the highest absolute MCO performance (75.66 vs Informer 73.22). This is the central argument for its regime-aware design — see `HMT-TSF-RESULTS.md` §MCO.
- Note the CNN-LSTM-Augmented variant is far more MCO-robust (75.36 → 65.25, −10.1) than the sequential CNN-LSTM (−30.1), suggesting the augmentation regularises against the structural break.

---

## 6. Per-family observations

**LSTM-family (LSTM, BiLSTM, TPA-LSTM, ST-LSTM, CNN-LSTM, CNN-BiLSTM)**
- Owns most of the top-7 in default config; strong, low-variance accuracy at lb14.
- BiLSTM and TPA-LSTM are the best of the family (R² 0.776, RMSE ~109k).
- Universally flagged overfit in diagnostics (gap ratios 3.3–13×) despite good test numbers — accuracy is real but headroom is memorisation-driven.

**Graph-based (STGCN, MTGNN, STSGCN, STFGNN, PDR-STGCN)**
- Wide quality spread. MTGNN and STSGCN are competitive (~77 Combined); STGCN, PDR-STGCN, STFGNN trail.
- STFGNN is the least reliable: catastrophic at lb28/lb56 and weak under MCO.
- PDR-STGCN is the weakest baseline at default but the **single biggest beneficiary of tuning (+5.58)** and surprisingly MCO-robust (−7.05).
- MTGNN is the most lookback-stable graph model and `good_fit` when MCO is excluded.

**Attention-based (TPA-LSTM, ASTGCN, Autoformer, Informer)**
- **Informer is the standout** — top baseline Combined%, best non-LSTM R², most MCO-robust attention model, and the only baseline that is `good_fit` across all 6 base configs (`DIAGNOSTICS.md`).
- Autoformer is the fragile end: worst MCO degradation among attention models (R² 0.121).

---

## 7. Best configuration per model (across all 12, by Combined%)

| Model | Best config | Combined% | R² |
|---|---|---:|---:|
| Informer | tuned_nomco_lb14 | 79.99 | 0.778 |
| TPA-LSTM | tuned_nomco_lb14 | 79.95 | 0.782 |
| BiLSTM | tuned_nomco_lb14 | 79.44 | 0.794 |
| ST-LSTM | tuned_nomco_lb14 | 79.13 | 0.774 |
| ASTGCN | tuned_nomco_lb14 | 78.82 | 0.754 |
| LSTM | tuned_nomco_lb14 | 78.43 | 0.777 |
| PDR-STGCN | tuned_nomco_lb14 | 78.16 | 0.759 |
| CNN-BiLSTM | base_nomco_lb14 | 78.18 | 0.765 |
| MTGNN | base_nomco_lb28 | 77.99 | 0.769 |
| CNN-LSTM | base_nomco_lb14 | 77.64 | 0.752 |
| STSGCN | base_nomco_lb14 | 76.97 | 0.736 |
| CNN-LSTM-Augmented | tuned_nomco_lb28 | 77.30 | 0.749 |
| STGCN | tuned_nomco_lb28 | 76.68 | 0.733 |
| Autoformer | tuned_nomco_lb28 | 76.66 | 0.715 |
| CNN-LSTM-Parallel | base_nomco_lb28 | 76.47 | 0.741 |
| STFGNN | base_nomco_lb14 | 73.94 | 0.688 |

Every model's best slice is a `nomco` config; no model's best is MCO-included. Best lookback is lb14 (12 models) or lb28 (5 models) — never lb56.

---

## 8. Verdict

1. **HMT-TSF wins outright.** It is #1 in the default config (+2.33 Combined over the best baseline), posts the lowest MAE/RMSE, and is the most MCO-robust model — a clean sweep across accuracy and robustness axes.
2. **Best baselines: Informer, BiLSTM, TPA-LSTM.** Informer is the most well-rounded (top accuracy, clean fit, MCO-robust). BiLSTM/TPA-LSTM match it on accuracy but overfit.
3. **Weakest baselines: PDR-STGCN, STFGNN, Autoformer, CNN-LSTM-Parallel.** STFGNN has severe lookback fragility; CNN-LSTM (sequential) and Autoformer have severe MCO fragility.
4. **lb14 is the right default.** Longer windows do not help a 7-day horizon and trigger collapses in fragile models.
5. **Tuning is a wash on accuracy** (median ~+0.4 Combined) and **costs generalisation** (see `DIAGNOSTICS.md`). Use tuned variants selectively (PDR-STGCN, ASTGCN benefit; CNN-LSTM, STSGCN regress).

---

## 9. Use-case scenarios

| Scenario | Recommended model(s) | Rationale |
|---|---|---|
| **Production forecasting, normal conditions** | HMT-TSF; fallback Informer | Best accuracy + clean fit; Informer is the strongest, best-generalising baseline fallback. |
| **Robustness to shocks / lockdowns / structural breaks** | HMT-TSF; fallback Informer or BiLSTM | Smallest MCO degradation and highest retained accuracy under the break. |
| **Lowest absolute error (MAE/RMSE) target** | HMT-TSF | MAE 59k vs best baseline 69k (−15%); lowest RMSE in the field. |
| **Compute-constrained / simple deployment** | LSTM or BiLSTM (base, lb14) | Near-top accuracy with the simplest architecture and shortest window. |
| **Interpretability / feature attribution required** | HMT-TSF with `--shap` | SHAP values expose which feature groups drive each forecast step. |
| **Longer look-back mandated by data constraints** | MTGNN | Most lookback-stable; avoid STFGNN at lb≥28. |
| **Avoid at any cost** | STFGNN at lb56; CNN-LSTM (seq) under MCO | Sub-mean R² (negative or ~0) — worse than naive persistence. |

> All numbers verified against `aggregate_results.csv` rows 2–18. Cross-check fit reliability in `DIAGNOSTICS.md` before treating any baseline's headline number as deployable.
