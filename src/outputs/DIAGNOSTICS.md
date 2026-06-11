# Fit Diagnostics (Overfit / Underfit Analysis)

> Source: `src/outputs/aggregate_diagnosis.csv` (14 baseline/tuned models × 6 configs) + diagnosis columns of `aggregate_hmttsf.csv` (HMT-TSF × 10 configs) · Generated 2026-06-05
> Companion files: `RESULTS.md`, `HMT-TSF-RESULTS.md`
> This document is **descriptive** — it characterises the fit verdicts and their drivers. No remediation recommendations.

## 1. Methodology

Each training run carries a `verdict` (`good_fit` / `overfit`) derived from three signals recorded per run:

- **`val_drift_pct`** — how far final validation loss drifted above its best checkpoint. Threshold **25%**; above this = degradation after the best epoch.
- **`gap_ratio`** — ratio of best validation loss to minimum training loss. Threshold **3×**; above this = memorisation of the training set.
- **train/val divergence** — training loss still declining in the final 10 epochs while validation drifted upward.

A run is flagged **`overfit`** if **any** signal trips (drift > 25% **OR** gap > 3× **OR** divergence). A run is **`good_fit`** only when all three are clear. Supporting fields: `val_trend` (`rising`/`flat`/`falling` train-loss direction), `early_stop`, `best_epoch`, `total_epochs`.

**Note on `best_epoch` vs `total_epochs`:** across every run, `total = best + patience` (patience 15 for base/most-tuned, 25 for STSGCN-tuned). So `total_epochs − best_epoch` is just the patience window, and `early_stop` is `False` everywhere — i.e. the best model was always an earlier checkpoint and the reported drift measures how much validation deteriorated over the patience window before stopping.

Config axes per model: `{base, tuned}` × `{exclude (nomco), include (mco)}` × `{lb14, lb28, lb56}`.

---

## 2. Aggregate verdict rates

| Slice | good_fit | total | rate |
|---|---:|---:|---:|
| All base runs | 22 | 84 | 26% |
| All tuned runs | 12 | 84 | 14% |
| **nomco (exclude MCO)** | 25 | 84 | **30%** |
| **mco (include MCO)** | 9 | 84 | **11%** |
| lb14 (all) | 11 | 56 | 20% |
| lb28 (all) | 11 | 56 | 20% |
| lb56 (all) | 12 | 56 | 21% |
| HMT-TSF (full + FR) | 19 | 20 | **95%** |

**Read-out:**
- **MCO inclusion is the dominant driver of overfitting** — the good-fit rate falls from 30% (nomco) to 11% (mco), nearly a 3× increase in overfit incidence. The lockdown break makes validation loss diverge after the best checkpoint.
- **Tuning lowers the good-fit rate** (26% → 14%). Extra capacity in the tuned variants raises gap ratios and post-best drift; the test-accuracy effect of tuning is mixed (`RESULTS.md` §3) but the generalisation effect is consistently negative.
- **Lookback has only a weak aggregate effect on the verdict** (20–21% across lb14/28/56). Within individual models longer windows can raise `val_drift` (e.g. ASTGCN exclude lb14 `good` → lb56 `overfit`), but it is not a primary axis — model family and MCO status dominate.

---

## 3. Per-model verdict tally

| Model | Base good/6 | Tuned good/6 | Where it holds | Characteristic |
|---|---:|---:|---|---|
| **Informer** | **6** | 0 | All base configs | Only model with perfect base score; tuned 0/6. |
| STSGCN | 4 | 3 | nomco + some mco | Among the cleanest graph models; gaps ~2.0–2.8×. |
| Autoformer | 4 | 3 | nomco mostly | Robust except scattered mco runs. |
| MTGNN | 3 | 4 | nomco (tuned best) | Best-balanced graph model; tightest gaps in study (down to 1.40×). |
| STFGNN | 3 | 1 | nomco-exclude only | Clean when MCO excluded; breaks otherwise. |
| ASTGCN | 2 | 1 | nomco short lb | Good only at exclude/lb14–28. |
| BiLSTM | 0 | 0 | — | Chronic; gaps 3.4–12.8×. |
| CNN-BiLSTM | 0 | 0 | — | Chronic; gaps up to 13.6×. |
| CNN-LSTM | 0 | 0 | — | Chronic; gaps up to 9.8×. |
| LSTM | 0 | 0 | — | Chronic; gaps 3.3–6.7×. |
| ST-LSTM | 0 | 0 | — | Chronic; gaps 3.3–5.3×. |
| TPA-LSTM | 0 | 0 | — | Chronic; gaps 3.2–5.2×. |
| PDR-STGCN | 0 | 0 | — | Extreme; gaps up to **38.6×**. |
| STGCN | 0 | 0 | — | Chronic; gaps up to 11.2×. |

**Two clean groups:**
- **Cleanest:** Informer (perfect base), MTGNN, STSGCN, Autoformer — predominantly attention/graph models with gap ratios near 2×.
- **Chronic overfitters (0/12 across base + tuned):** all six LSTM-family models (BiLSTM, CNN-BiLSTM, CNN-LSTM, LSTM, ST-LSTM, TPA-LSTM) plus the two simple-convolution graph models (STGCN, PDR-STGCN). These post strong test numbers (`RESULTS.md`) but always trip the gap-ratio signal — their headroom is memorisation-driven.

---

## 4. Memorisation extremes (highest `gap_ratio`)

The widest validation/training-loss gaps — the strongest memorisation signals.

| Model | Config | gap_ratio | verdict |
|---|---|---:|---|
| PDR-STGCN (tuned) | exclude_lb56 | **38.61×** | overfit |
| PDR-STGCN (base) | exclude_lb56 | 20.18× | overfit |
| PDR-STGCN (tuned) | include_lb56 | 19.13× | overfit |
| PDR-STGCN (tuned) | exclude_lb28 | 15.03× | overfit |
| CNN-BiLSTM (base) | include_lb56 | 13.51× | overfit |
| BiLSTM (tuned) | exclude_lb56 | 12.77× | overfit |
| PDR-STGCN (tuned) | include_lb28 | 11.22× | overfit |
| CNN-LSTM (tuned) | include_lb56 | 10.56× | overfit |
| STGCN (tuned) | exclude_lb56 | 10.53× | overfit |
| STGCN (tuned) | include_lb56 | 9.38× | overfit |

**Read-out:** PDR-STGCN dominates the memorisation extremes (training loss collapses to near-zero while validation stays high). Gap ratios escalate with lookback (lb56 worst) and with tuning. By contrast the tightest gaps in the study are **MTGNN-tuned (1.40×, 1.50×)**, STFGNN-base (1.72×), and MTGNN-base (1.91×) — all comfortably under 3×.

---

## 5. Post-best degradation extremes (highest `val_drift_pct`)

Largest validation-loss drift above the best checkpoint — the strongest "trained too long / unstable" signals. **Every one is an MCO-included config.**

| Model | Config | val_drift % | gap_ratio | verdict |
|---|---|---:|---:|---|
| ASTGCN (base) | include_lb28 | **565.3%** | 2.93× | overfit |
| ASTGCN (tuned) | include_lb28 | 463.7% | 3.47× | overfit |
| ASTGCN (tuned) | include_lb56 | 389.0% | 3.68× | overfit |
| PDR-STGCN (base) | include_lb56 | 288.3% | 7.06× | overfit |
| ASTGCN (base) | include_lb56 | 243.2% | 4.03× | overfit |
| PDR-STGCN (tuned) | include_lb56 | 234.7% | 19.13× | overfit |
| PDR-STGCN (base) | include_lb28 | 216.0% | 6.63× | overfit |
| CNN-BiLSTM (tuned) | include_lb56 | 170.0% | 7.52× | overfit |
| PDR-STGCN (tuned) | include_lb28 | 144.2% | 11.22× | overfit |
| MTGNN (base) | include_lb28 | 133.7% | 2.62× | overfit |

**Read-out:**
- Drift in the **hundreds of percent** appears only under MCO, concentrated in graph/attention-graph models (ASTGCN, PDR-STGCN) and now also CNN-BiLSTM-tuned. The lockdown discontinuity makes the validation surface unstable; after the best epoch the model rapidly degrades.
- MTGNN-base include_lb28 is flagged `overfit` purely on **drift** (gap ratio 2.62× is within the 3× bar) — illustrating that the verdict is an OR of signals, not gap-ratio alone.

---

## 6. Verdict-driver patterns

1. **MCO inclusion is the primary overfit driver.** good-fit rate 30% → 11%; all extreme-drift runs are `include`. Models that are clean under `nomco` (ASTGCN, Autoformer, MTGNN, STFGNN, STSGCN) routinely flip to `overfit` when the lockdown window is added.
2. **Model family is the second axis.** Attention/learned-adjacency models (Informer, MTGNN, STSGCN, Autoformer) keep gap ratios near 2×. LSTM-family and fixed-convolution graph models (STGCN, PDR-STGCN) sit ≥3× regardless of config.
3. **Tuning worsens generalisation** (good-fit 26% → 14%; gap ratios rise, e.g. BiLSTM 3.4× → 12.8× at lb56). It does not change the qualitative family pattern.
4. **Lookback effect is secondary and within-model.** Aggregate verdict rates are flat across lb14/28/56, but for individual models longer windows tend to raise `val_drift` and `gap_ratio` (clearest in ASTGCN, PDR-STGCN, BiLSTM).
5. **`val_trend` is `falling`/`flat` in almost every overfit run** — training loss was still decreasing while validation diverged, the classic memorisation signature. `rising` appears in 5 runs (BiLSTM-tuned include_lb14, Informer-tuned exclude_lb28, MTGNN-base include_lb28, STFGNN-base include_lb14, STFGNN-tuned exclude_lb56).
6. **No run triggered `early_stop`** — best checkpoints were always saved before the patience window, and the reported drift is the deterioration over that window.

---

## 7. HMT-TSF (proposed model)

HMT-TSF (2026-06-11 regenerated runs — see `REVISION.md`) is `good_fit` on **9 of 10 configs**; the feature-reduced variant on **all 10** — a 19/20 record no baseline family approaches. Full tables in `HMT-TSF-RESULTS.md` §6 and §11.

- **gap_ratio range 1.57× – 2.72×** (both variants) — always under the 3× threshold, including all MCO-included configs. No memorisation signal in any configuration; the headline nomco_lb14 run posts the study's cleanest profile (1.57×, drift 0.61%).
- **The sole exception is full-model nomco_lb84** — `overfit` on validation drift (+36.9% above its early best, epoch 18/48). The 84-day window over a 7-day horizon is the study's clearest "too much context" case; feature reduction repairs it (FR nomco_lb84 `good_fit`, drift 11.1%).
- It is the **only architecture that holds `good_fit` under MCO in every configuration** — where every baseline family above degrades — directly mirroring its MCO-accuracy robustness (`RESULTS.md` §5, `HMT-TSF-RESULTS.md` §4).

| Comparison | Chronic overfitters | Cleanest baseline (Informer) | HMT-TSF (full / FR) |
|---|---|---|---|
| Gap-ratio range | 3.1 – 38.6× | 2.16 – 2.98× | 1.57 – 2.64× / 1.66 – 2.72× |
| good_fit under MCO | none | 3/3 (base) | 5/5 / 5/5 |
| good_fit overall | 0/12 | 6/12 | 9/10 / 10/10 |

> All verdicts, drift values, and gap ratios verified against `aggregate_diagnosis.csv` and the HMT-TSF diagnosis columns of `aggregate_hmttsf.csv` (regenerated 2026-06-11).
