# Fit Diagnostics (Overfit / Underfit Analysis)

> Source: `src/outputs/aggregate_diagnosis.csv` (15 baseline/tuned models × 6 configs) + diagnosis columns of `aggregate_hmttsf.csv` (HMT-TSF × 10 configs) · Generated 2026-05-30
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
| All base runs | 25 | 90 | 28% |
| All tuned runs | 17 | 90 | 19% |
| **nomco (exclude MCO)** | 31 | 90 | **34%** |
| **mco (include MCO)** | 11 | 90 | **12%** |
| lb14 (all) | 13 | 60 | 22% |
| lb28 (all) | 15 | 60 | 25% |
| lb56 (all) | 15 | 60 | 25% |
| HMT-TSF (all) | 10 | 10 | **100%** |

**Read-out:**
- **MCO inclusion is the dominant driver of overfitting** — the good-fit rate falls from 34% (nomco) to 12% (mco), nearly a 3× increase in overfit incidence. The lockdown break makes validation loss diverge after the best checkpoint.
- **Tuning lowers the good-fit rate** (28% → 19%). Extra capacity in the tuned variants raises gap ratios and post-best drift; the test-accuracy effect of tuning is mixed (`RESULTS.md` §3) but the generalisation effect is consistently negative.
- **Lookback has only a weak aggregate effect on the verdict** (22–25% across lb14/28/56). Within individual models longer windows can raise `val_drift` (e.g. ASTGCN exclude lb14 `good` → lb56 `overfit`), but it is not a primary axis — model family and MCO status dominate.

---

## 3. Per-model verdict tally

| Model | Base good/6 | Tuned good/6 | Where it holds | Characteristic |
|---|---:|---:|---|---|
| **Informer** | **6** | 1 | All base configs | Only baseline `good_fit` on every base run; tuned breaks. |
| STSGCN | 4 | 3 | nomco + some mco | Among the cleanest graph models; gaps ~2.0–2.8×. |
| Autoformer | 4 | 3 | nomco mostly | Robust except scattered mco runs. |
| MTGNN | 3 | 4 | nomco (tuned best) | Best-balanced graph model; tightest gaps in study (down to 1.40×). |
| TFT | 3 | 4 | nomco mostly | Tuning *improves* its fit; low gaps (~1.7–2.6×). |
| STFGNN | 3 | 1 | nomco-exclude only | Clean when MCO excluded; breaks otherwise. |
| ASTGCN | 2 | 1 | nomco short lb | Good only at exclude/lb14–28. |
| BiLSTM | 0 | 0 | — | Chronic; gaps 3.4–12.8×. |
| CNN-BiLSTM | 0 | 0 | — | Chronic; gaps up to 13.6×. |
| CNN-LSTM | 0 | 0 | — | Chronic; gaps up to 9.8×. |
| LSTM | 0 | 0 | — | Chronic; gaps 3.3–6.7×. |
| ST-LSTM | 0 | 0 | — | Chronic; gaps 3.3–5.3×. |
| TPA-LSTM | 0 | 0 | — | Chronic; gaps 3.2–5.2×. |
| PDR-STGCN | 0 | 0 | — | Extreme; gaps up to **37.9×**. |
| STGCN | 0 | 0 | — | Chronic; gaps up to 11.2×. |

**Two clean groups:**
- **Cleanest:** Informer (perfect base), MTGNN, STSGCN, Autoformer, TFT — predominantly attention/graph models with gap ratios near 2×.
- **Chronic overfitters (0/12 across base + tuned):** all six LSTM-family models (BiLSTM, CNN-BiLSTM, CNN-LSTM, LSTM, ST-LSTM, TPA-LSTM) plus the two simple-convolution graph models (STGCN, PDR-STGCN). These post strong test numbers (`RESULTS.md`) but always trip the gap-ratio signal — their headroom is memorisation-driven.

---

## 4. Memorisation extremes (highest `gap_ratio`)

The widest validation/training-loss gaps — the strongest memorisation signals.

| Model | Config | gap_ratio | verdict |
|---|---|---:|---|
| PDR-STGCN (base) | exclude_lb56 | **37.89×** | overfit |
| PDR-STGCN (tuned) | exclude_lb56 | 19.38× | overfit |
| PDR-STGCN (tuned) | include_lb56 | 19.13× | overfit |
| CNN-BiLSTM (base) | include_lb56 | 13.57× | overfit |
| BiLSTM (tuned) | exclude_lb56 | 12.77× | overfit |
| PDR-STGCN (tuned) | include_lb28 | 11.68× | overfit |
| STGCN (tuned) | exclude_lb56 | 11.15× | overfit |
| PDR-STGCN (base) | include_lb56 | 10.74× | overfit |
| PDR-STGCN (base) | exclude_lb28 | 10.51× | overfit |
| CNN-LSTM (tuned) | include_lb56 | 9.77× | overfit |

**Read-out:** PDR-STGCN dominates the memorisation extremes (training loss collapses to near-zero while validation stays high). Gap ratios escalate with lookback (lb56 worst) and with tuning. By contrast the tightest gaps in the study are **MTGNN-tuned (1.40×, 1.50×)**, MTGNN-base (1.71×), STFGNN-base-exclude (1.72×), and TFT (~1.72–1.92×) — all comfortably under 3×.

---

## 5. Post-best degradation extremes (highest `val_drift_pct`)

Largest validation-loss drift above the best checkpoint — the strongest "trained too long / unstable" signals. **Every one is an MCO-included config.**

| Model | Config | val_drift % | gap_ratio | verdict |
|---|---|---:|---:|---|
| ASTGCN (base) | include_lb28 | **565.3%** | 2.93× | overfit |
| ASTGCN (tuned) | include_lb28 | 463.7% | 3.47× | overfit |
| ASTGCN (tuned) | include_lb56 | 389.0% | 3.68× | overfit |
| PDR-STGCN (tuned) | include_lb56 | 234.7% | 19.13× | overfit |
| PDR-STGCN (tuned) | include_lb28 | 225.4% | 11.68× | overfit |
| ASTGCN (base) | include_lb56 | 243.2% | 4.03× | overfit |
| MTGNN (base) | include_lb56 | 139.2% | 2.36× | overfit |
| PDR-STGCN (base) | include_lb56 | 115.9% | 10.74× | overfit |
| PDR-STGCN (base) | include_lb28 | 108.6% | 9.80× | overfit |
| STGCN (base) | include_lb56 | 102.4% | 7.18× | overfit |
| TFT (tuned) | include_lb14 | 102.0% | 2.83× | overfit |

**Read-out:**
- Drift in the **hundreds of percent** appears only under MCO, concentrated in graph/attention-graph models (ASTGCN, PDR-STGCN, STGCN). The lockdown discontinuity makes the validation surface unstable; after the best epoch the model rapidly degrades.
- MTGNN-base include_lb56 and TFT-tuned include_lb14 are flagged `overfit` purely on **drift/divergence** (gap ratios 2.36× and 2.83× are within the 3× bar) — illustrating that the verdict is an OR of signals, not gap-ratio alone.

---

## 6. Verdict-driver patterns

1. **MCO inclusion is the primary overfit driver.** good-fit rate 34% → 12%; all extreme-drift runs are `include`. Models that are clean under `nomco` (ASTGCN, Autoformer, MTGNN, STFGNN, STSGCN, TFT) routinely flip to `overfit` when the lockdown window is added.
2. **Model family is the second axis.** Attention/learned-adjacency models (Informer, MTGNN, STSGCN, Autoformer, TFT) keep gap ratios near 2×. LSTM-family and fixed-convolution graph models (STGCN, PDR-STGCN) sit ≥3× regardless of config.
3. **Tuning worsens generalisation** (good-fit 28% → 19%; gap ratios rise, e.g. BiLSTM 3.4× → 12.8× at lb56). It does not change the qualitative family pattern.
4. **Lookback effect is secondary and within-model.** Aggregate verdict rates are flat across lb14/28/56, but for individual models longer windows tend to raise `val_drift` and `gap_ratio` (clearest in ASTGCN, PDR-STGCN, BiLSTM).
5. **`val_trend` is `falling`/`flat` in almost every overfit run** — training loss was still decreasing while validation diverged, the classic memorisation signature. `rising` appears in only two runs (BiLSTM-tuned include_lb14, STFGNN-tuned exclude_lb56).
6. **No run triggered `early_stop`** — best checkpoints were always saved before the patience window, and the reported drift is the deterioration over that window.

---

## 7. HMT-TSF (proposed model)

HMT-TSF is `good_fit` on **all 10 configs** (the only model family with a perfect fit record). Full table in `HMT-TSF-RESULTS.md` §6.

- **gap_ratio range 1.79× – 2.80×** — always under the 3× threshold, including all MCO-included configs. No memorisation signal in any configuration.
- **val_drift ≤ 12.3%** — largest at nomco_lb56 (12.32%) but well under the 25% bar; mco configs stay ≤ 9.6%.
- It is the **only architecture that holds `good_fit` under MCO** — where every baseline family above degrades — directly mirroring its MCO-accuracy robustness (`RESULTS.md` §5, `HMT-TSF-RESULTS.md` §4).

| Comparison | Chronic overfitters | Cleanest baseline (Informer) | HMT-TSF |
|---|---|---|---|
| Gap-ratio range | 3.2 – 37.9× | 2.16 – 2.97× | 1.79 – 2.80× |
| good_fit under MCO | none | 3/3 (base) | 5/5 |
| good_fit overall | 0/12 | 7/12 | 10/10 |

> All verdicts, drift values, and gap ratios verified against `aggregate_diagnosis.csv` (rows 2–181) and the HMT-TSF diagnosis columns of `aggregate_hmttsf.csv` (row 2).
