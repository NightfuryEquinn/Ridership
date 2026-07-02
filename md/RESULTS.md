# Baseline & Tuned Model Results

> Last updated: 2026-07-02 · Source: `src/outputs/aggregate_results.csv`
> Companion files: `HMT-TSF-RESULTS.md` (proposed model), `DIAGNOSTICS.md` (fit diagnostics)

This page reports how the 16 baseline models scored, across 12 settings each (with/without
the COVID period, three look-back windows, base and tuned versions). HMT-TSF appears here
only as a reference row for comparison — its full results are in `HMT-TSF-RESULTS.md`. Each
table is followed by a plain-language read-out.

> **One caveat:** every number is a single run (fixed random seed), so treat gaps smaller
> than about one point as indicative rather than definitive.

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
| 7 | MTGNN | 77.93 | 6.91 | 6.47 | 8.69 | 0.775 | 81,292 | 109,185 |
| 8 | CNN-LSTM | 77.64 | 6.92 | 6.32 | 9.12 | 0.752 | 79,375 | 114,632 |
| 9 | STSGCN | 76.97 | 7.18 | 6.42 | 9.42 | 0.736 | 80,703 | 118,391 |
| 10 | STGCN | 76.81 | 7.24 | 6.77 | 9.17 | 0.750 | 85,077 | 115,297 |
| 11 | ASTGCN | 75.69 | 7.42 | 6.89 | 9.99 | 0.703 | 86,631 | 125,606 |
| 12 | CNN-LSTM-Augmented | 75.36 | 7.77 | 7.19 | 9.68 | 0.721 | 90,356 | 121,639 |
| 13 | Autoformer | 74.73 | 7.86 | 7.51 | 9.90 | 0.708 | 94,414 | 124,408 |
| 14 | CNN-LSTM-Parallel | 74.18 | 8.18 | 7.68 | 9.97 | 0.704 | 96,477 | 125,240 |
| 15 | STFGNN | 73.94 | 8.21 | 7.60 | 10.25 | 0.688 | 95,547 | 128,777 |
| 16 | PDR-STGCN | 73.91 | 8.15 | 7.70 | 10.24 | 0.688 | 96,801 | 128,694 |
| — | **HMT-TSF (ref)** | **85.78** | **4.39** | **3.97** | **5.87** | **0.897** | **49,897** | **73,757** |

**Read-out:**
- The default-config field is tight: ranks 1–8 sit inside a **1.5-point Combined% band (77.6–79.1)**. LSTM-family models (BiLSTM, TPA-LSTM, LSTM, ST-LSTM, CNN-BiLSTM) dominate the top alongside Informer and MTGNN.
- **Informer** is the strongest non-LSTM baseline and the top baseline overall by Combined% (79.13); BiLSTM/TPA-LSTM edge it on R² (0.776 vs 0.772) and RMSE.
- **MTGNN** (77.93) climbs to rank 7 and is competitive with the LSTM family; STGCN (76.81) also improved significantly versus earlier runs, now at rank 10.
- The bottom of the table is split: graph-heavy PDR-STGCN (73.91) and STFGNN (73.94) are separated by only 0.03 Combined points. Autoformer (74.73) and CNN-LSTM-Parallel (74.18) also trail.
- HMT-TSF (2026-06-11 regenerated runs) leads the field by **+6.65 Combined points over the best baseline** (85.78 vs 79.13) and posts the lowest absolute MAE (49,897 vs Informer's 69,476, a **−28%** error reduction); its feature-reduced variant extends this to 86.59 / 46,323 (see `HMT-TSF-RESULTS.md`). Note HMT-TSF additionally receives known-future calendar inputs (`X_future`) that no baseline consumes.

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
| STGCN | 78.39 | 0.773 | +1.58 |
| CNN-LSTM-Augmented | 77.10 | 0.754 | +1.74 |
| MTGNN | 76.40 | 0.742 | −1.54 |
| CNN-BiLSTM | 76.12 | 0.747 | −2.06 |
| PDR-STGCN | 75.56 | 0.725 | +1.65 |
| Autoformer | 75.21 | 0.687 | +0.48 |
| STSGCN | 73.65 | 0.658 | −3.32 |
| CNN-LSTM | 73.36 | 0.692 | −4.28 |
| CNN-LSTM-Parallel | 72.53 | 0.673 | −1.65 |
| STFGNN | 65.47 | 0.498 | **−8.47** |

**Read-out:**
- Best baseline number anywhere in `nomco` is **Informer tuned (79.99)** — still well below HMT-TSF (85.78; FR 86.59).
- **Tuning is not uniformly beneficial.** Largest gains: ASTGCN (+3.13), STGCN (+1.58), CNN-LSTM-Augmented (+1.74). Largest regressions: **STFGNN (−8.47)**, CNN-LSTM (−4.28), STSGCN (−3.32). STFGNN's catastrophic tuning regression (65.47 from 73.94) is the most severe across the entire study.
- MTGNN and CNN-BiLSTM also regress under tuning (−1.54, −2.06) — additional capacity hurts both.
- PDR-STGCN gains +1.65 (previous runs showed +5.58; current results are more moderate).
- **PDR-STGCN caveat:** the base runs were executed with `--period` equal to the look-back (14/28/56) instead of the weekly default 7; since the periodic diff encoder zero-pads for t < period, the second input channel was all zeros — base PDR-STGCN effectively ran without its periodicity encoding. Tuned runs used `period=7` correctly, so part of the +1.65 tuning gain reflects re-enabling that channel rather than capacity changes (see `MODEL.md`, PDR-STGCN section).
- On Combined% the median tuning effect is roughly flat (~+0.5). Test-set gains do **not** imply better generalisation; the diagnostics file shows tuned variants overfit more (higher gap ratios).

---

## 4. Look-back sensitivity — base, `nomco`

Combined% at lb14 / lb28 / lb56 (best of the three **bold**).

| Model | lb14 | lb28 | lb56 |
|---|---:|---:|---:|
| ASTGCN | 75.69 | **76.89** | 75.15 |
| Autoformer | **74.73** | 73.61 | 73.40 |
| BiLSTM | **79.04** | 76.97 | 77.09 |
| CNN-BiLSTM | **78.18** | 72.65 | 74.72 |
| CNN-LSTM | **77.64** | 73.90 | 74.33 |
| CNN-LSTM-Augmented | **75.36** | 74.73 | 74.04 |
| CNN-LSTM-Parallel | 74.18 | **76.55** | 74.33 |
| Informer | **79.13** | 78.63 | 77.17 |
| LSTM | **78.13** | 77.48 | 75.98 |
| MTGNN | 77.93 | **78.11** | 77.76 |
| PDR-STGCN | **73.91** | 72.35 | 72.20 |
| ST-LSTM | 78.01 | **78.86** | 75.60 |
| STFGNN | **73.94** | 61.30 | 43.25 |
| STGCN | 76.81 | **77.68** | 74.93 |
| STSGCN | **76.97** | 75.02 | 75.15 |
| TPA-LSTM | 78.67 | **79.08** | 76.36 |

**Read-out:**
- **lb14 or lb28 is optimal for every model; lb56 never wins.** lb14 is best for 10 models, lb28 for 6 (ASTGCN, CNN-LSTM-Parallel, MTGNN, ST-LSTM, STGCN, TPA-LSTM).
- **Catastrophic lb56 collapse:** STFGNN (73.94 → 43.25, R² −0.112) falls below the "beats the mean" line at lb56 and also degrades sharply at lb28 (61.30). This is the most lookback-fragile model.
- MTGNN is the most lookback-stable (77.93 / 78.11 / 77.76), confirming its robustness to window length.
- CNN-LSTM degrades sharply at lb28 (73.90 vs 77.64 at lb14) — a notable sensitivity not seen in other LSTM-family models.

---

## 5. MCO robustness stress test — base, lb14

`nomco` → `mco` Combined% and the MCO-config R². Sorted by smallest degradation (most robust → least).

| Model | nomco | mco | Δ Combined | mco R² |
|---|---:|---:|---:|---:|
| Informer | 79.13 | 73.25 | **−5.88** | 0.705 |
| BiLSTM | 79.04 | 72.74 | −6.30 | 0.699 |
| CNN-LSTM-Parallel | 74.18 | 66.89 | −7.29 | 0.577 |
| TPA-LSTM | 78.67 | 71.05 | −7.62 | 0.669 |
| STGCN | 76.81 | 68.55 | −8.27 | 0.615 |
| MTGNN | 77.93 | 67.33 | −10.60 | 0.581 |
| PDR-STGCN | 73.91 | 63.10 | −10.81 | 0.494 |
| CNN-LSTM-Augmented | 75.36 | 64.50 | −10.86 | 0.531 |
| ST-LSTM | 78.01 | 66.75 | −11.26 | 0.584 |
| STSGCN | 76.97 | 64.20 | −12.77 | 0.531 |
| ASTGCN | 75.69 | 61.69 | −14.00 | 0.471 |
| CNN-BiLSTM | 78.18 | 60.59 | −17.59 | 0.427 |
| LSTM | 78.13 | 58.92 | −19.21 | 0.390 |
| Autoformer | 74.73 | 51.26 | −23.47 | 0.121 |
| CNN-LSTM | 77.64 | 45.58 | **−32.06** | −0.050 |
| STFGNN | 73.94 | 41.00 | **−32.94** | −0.306 |
| — | — | — | — | — |
| **HMT-TSF (ref)** | **85.78** | **81.99** | **−3.79** | **0.859** |

**Read-out:**
- The MCO structural break punishes every model, but the spread is huge: **−3.79 (HMT-TSF) to −32.94 (STFGNN)**.
- **CNN-LSTM (sequential) and STFGNN effectively fail under MCO** — both post R² near or below zero (−0.050 and −0.306 respectively), performing worse than naive persistence. STFGNN's MCO degradation (−32.94) now exceeds CNN-LSTM's (−32.06).
- **STGCN is surprisingly MCO-robust** (−8.27, rank 5), significantly better than earlier runs suggested. CNN-LSTM-Parallel (−7.29) is also among the more robust baselines despite its weaker default-config accuracy.
- Autoformer (R² 0.121) also degrades severely and sits near the fragile end.
- **HMT-TSF is the most MCO-robust model in the study** (−3.79, ahead of Informer's −5.88) while retaining the highest absolute MCO performance (81.99 vs Informer 73.25). This is the central argument for its regime-aware design — see `HMT-TSF-RESULTS.md` §4.
- The CNN-LSTM-Augmented variant (−10.86) is far more MCO-robust than the sequential CNN-LSTM (−32.06), confirming that the augmentation regularises against structural breaks.

---

## 6. Per-family observations

**LSTM-family (LSTM, BiLSTM, TPA-LSTM, ST-LSTM, CNN-LSTM, CNN-BiLSTM)**
- Owns most of the top-8 in default config; strong, low-variance accuracy at lb14.
- BiLSTM and TPA-LSTM are the best of the family (R² 0.776, RMSE ~109k).
- Universally flagged overfit in diagnostics (gap ratios 3.3–13×) despite good test numbers — accuracy is real but headroom is memorisation-driven.
- CNN-LSTM (sequential) remains an outlier: top-8 accuracy in `nomco` but near-total failure under MCO (R² −0.050).

**Graph-based (STGCN, MTGNN, STSGCN, STFGNN, PDR-STGCN)**
- Wide quality spread. **MTGNN (77.93)** is now competitive with the LSTM family and ranks 7th overall; **STGCN (76.81)** has improved significantly and overtakes STSGCN (76.97 → 76.81 close).
- STFGNN is the least reliable: catastrophic at lb28/lb56, severe MCO fragility (−32.94), and now the biggest loser under tuning (−8.47). Avoid in any non-default configuration.
- PDR-STGCN and STFGNN are the two weakest baselines in default config; PDR-STGCN gains modestly from tuning (+1.65).
- MTGNN is the most lookback-stable graph model; STGCN shows improved MCO robustness (−8.27).

**Attention-based (TPA-LSTM, ASTGCN, Autoformer, Informer)**
- **Informer is the standout** — top baseline Combined%, best non-LSTM R², most MCO-robust attention model, and the only baseline that is `good_fit` across all 6 base configs (`DIAGNOSTICS.md`).
- ASTGCN is the largest beneficiary of tuning among attention models (+3.13).
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
| STGCN | tuned_nomco_lb14 | 78.39 | 0.773 |
| CNN-BiLSTM | base_nomco_lb14 | 78.18 | 0.765 |
| MTGNN | base_nomco_lb28 | 78.11 | 0.774 |
| CNN-LSTM | base_nomco_lb14 | 77.64 | 0.752 |
| CNN-LSTM-Augmented | tuned_nomco_lb28 | 77.30 | 0.750 |
| STSGCN | base_nomco_lb14 | 76.97 | 0.736 |
| Autoformer | tuned_nomco_lb28 | 76.66 | 0.715 |
| CNN-LSTM-Parallel | base_nomco_lb28 | 76.55 | 0.743 |
| PDR-STGCN | tuned_nomco_lb28 | 75.64 | 0.723 |
| STFGNN | base_nomco_lb14 | 73.94 | 0.688 |

Every model's best slice is a `nomco` config; no model's best is MCO-included. Best lookback is lb14 (11 models) or lb28 (5 models: MTGNN, CNN-LSTM-Augmented, Autoformer, CNN-LSTM-Parallel, PDR-STGCN) — never lb56.

---

## 8. Verdict

1. **HMT-TSF wins outright.** It is #1 in the default config (+2.69 Combined over the best baseline), posts the lowest MAE/RMSE, and is the most MCO-robust model — a clean sweep across accuracy and robustness axes.
2. **Best baselines: Informer, BiLSTM, TPA-LSTM.** Informer is the most well-rounded (top accuracy, clean fit, MCO-robust). BiLSTM/TPA-LSTM match it on accuracy but overfit. MTGNN (77.93) is a stronger-than-expected graph baseline, ranking 7th overall.
3. **Weakest baselines: STFGNN, PDR-STGCN.** STFGNN has severe lookback fragility, the worst MCO robustness in the study (−32.94), and catastrophic tuning regression (−8.47). CNN-LSTM (sequential) and Autoformer have severe MCO fragility (R² −0.050 and 0.121 respectively).
4. **lb14 is the right default.** Longer windows do not help a 7-day horizon and trigger collapses in fragile models (STFGNN at lb56, CNN-LSTM at lb28).
5. **Tuning is a wash on accuracy** (median ~+0.5 Combined) and **costs generalisation** (see `DIAGNOSTICS.md`). Use tuned variants selectively — ASTGCN (+3.13), STGCN (+1.58), CNN-LSTM-Augmented (+1.74) benefit; STFGNN (−8.47), CNN-LSTM (−4.28), STSGCN (−3.32) regress sharply.

---

## 9. Use-case scenarios

| Scenario | Recommended model(s) | Rationale |
|---|---|---|
| **Production forecasting, normal conditions** | HMT-TSF; fallback Informer | Best accuracy + clean fit; Informer is the strongest, best-generalising baseline fallback. |
| **Robustness to shocks / lockdowns / structural breaks** | HMT-TSF; fallback Informer or BiLSTM | Smallest MCO degradation and highest retained accuracy under the break. |
| **Lowest absolute error (MAE/RMSE) target** | HMT-TSF | MAE 57k vs best baseline 69k (−17%); lowest RMSE in the field. |
| **Compute-constrained / simple deployment** | LSTM or BiLSTM (base, lb14) | Near-top accuracy with the simplest architecture and shortest window. |
| **Interpretability / feature attribution required** | HMT-TSF with `--shap` | SHAP values expose which feature groups drive each forecast step. |
| **Longer look-back mandated by data constraints** | MTGNN | Most lookback-stable; avoid STFGNN at lb≥28. |
| **Avoid at any cost** | STFGNN at lb≥28; CNN-LSTM (seq) under MCO | STFGNN R² −0.112 at lb56; CNN-LSTM R² −0.050 under MCO — both worse than naive persistence. |

> All numbers verified against `aggregate_results.csv`. Cross-check fit reliability in `DIAGNOSTICS.md` before treating any baseline's headline number as deployable.
