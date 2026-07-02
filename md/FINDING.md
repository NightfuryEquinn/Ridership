# Chapter 4: Findings and Results

---

## 4.1 Overview of Experimental Evaluation

With the dataset, feature pipeline, model suite, and evaluation framework fully specified in Chapter 3, this chapter presents the empirical results of the comparative study. The central question is whether a purpose-built hybrid architecture can outperform established deep-learning paradigms for seven-day Malaysian transit ridership forecasting, and under what conditions each paradigm succeeds or fails.

The evaluation spans four analytical dimensions. First, default-configuration performance under normal operating conditions (MCO-excluded, 14-day look-back) establishes the baseline ranking. Second, hyperparameter tuning tests whether capacity increases translate into genuine generalisation gains. Third, look-back window sensitivity across 14, 28, and 56 days examines how much historical context a seven-day horizon requires. Fourth, structural-break robustness under the Movement Control Order (MCO) pandemic period (2020-03-18 to 2021-12-31) stress-tests each model against an 85–95% ridership collapse. A dedicated fit-diagnostics analysis assesses generalisation quality independently of test-set accuracy.

Fifteen deep-learning architectures are evaluated: 14 baseline models, reported as 16 ranked variants because CNN-LSTM contributes three independent fusion modes (sequential, augmented, parallel), plus the proposed hybrid HMT-TSF. The 14 baselines span three series — spatio-temporal recurrent (LSTM-family), graph-based, and attention-based — and are compared against HMT-TSF under a shared protocol. All models use AdamW optimisation with decoupled weight decay, HuberLoss objective, a five-epoch linear learning-rate warm-up followed by ReduceLROnPlateau scheduling, and a 70/15/15 chronological train/validation/test split. Performance is ranked by Combined% = max(0, 100 − MAPE% − MAE% − RMSE%), where percentage terms are normalised by mean ridership demand. Supplementary metrics (R², absolute MAE, RMSE) complete the profile. The headline configuration is `base_nomco_lb14` (base architecture, MCO-excluded dataset, 14-day look-back). Fit diagnostics derive from validation drift above the best checkpoint (threshold 25%), the gap ratio between best validation loss and minimum training loss (threshold 3×), and train/validation divergence in the final training epochs. Section 4.2 opens the baseline comparison; subsequent sections extend the analysis through tuning, look-back sensitivity, MCO robustness, generalisation, the proposed model, and interpretability.

---

## 4.2 Baseline Model Performance — Default Configuration (`base_nomco_lb14`)

Under the project default, baseline accuracy is tightly clustered: the top eight models occupy a 1.5 Combined% band, indicating a saturated regime in which architectural differences yield incremental rather than step-change gains. Table 4.2.1 presents the complete ranking; the subsections below interpret it by architectural series.

| Rank | Model              | Combined% | MAPE% | MAE% | RMSE% | R²    | MAE    | RMSE    |
| ---- | ------------------ | --------- | ----- | ---- | ----- | ----- | ------ | ------- |
| 1    | Informer           | 79.13     | 6.59  | 5.53 | 8.76  | 0.772 | 69,476 | 110,050 |
| 2    | BiLSTM             | 79.04     | 6.52  | 5.77 | 8.67  | 0.776 | 72,460 | 109,007 |
| 3    | TPA-LSTM           | 78.67     | 6.62  | 6.04 | 8.67  | 0.776 | 75,871 | 109,009 |
| 4    | CNN-BiLSTM         | 78.18     | 6.81  | 6.12 | 8.89  | 0.764 | 76,884 | 111,782 |
| 5    | LSTM               | 78.13     | 6.71  | 6.18 | 8.98  | 0.760 | 77,719 | 112,801 |
| 6    | ST-LSTM            | 78.01     | 6.93  | 6.27 | 8.79  | 0.770 | 78,779 | 110,453 |
| 7    | MTGNN              | 77.93     | 6.91  | 6.47 | 8.69  | 0.775 | 81,292 | 109,185 |
| 8    | CNN-LSTM           | 77.64     | 6.92  | 6.32 | 9.12  | 0.752 | 79,375 | 114,632 |
| 9    | STSGCN             | 76.97     | 7.18  | 6.42 | 9.42  | 0.736 | 80,703 | 118,391 |
| 10   | STGCN              | 76.81     | 7.24  | 6.77 | 9.17  | 0.750 | 85,077 | 115,297 |
| 11   | ASTGCN             | 75.69     | 7.42  | 6.89 | 9.99  | 0.703 | 86,631 | 125,606 |
| 12   | CNN-LSTM-Augmented | 75.36     | 7.77  | 7.19 | 9.68  | 0.721 | 90,356 | 121,639 |
| 13   | Autoformer         | 74.73     | 7.86  | 7.51 | 9.90  | 0.708 | 94,414 | 124,408 |
| 14   | CNN-LSTM-Parallel  | 74.18     | 8.18  | 7.68 | 9.97  | 0.704 | 96,477 | 125,240 |
| 15   | STFGNN             | 73.94     | 8.21  | 7.60 | 10.25 | 0.688 | 95,547 | 128,777 |
| 16   | PDR-STGCN          | 73.91     | 8.15  | 7.70 | 10.24 | 0.688 | 96,801 | 128,694 |

*Table 4.2.1. Default-configuration (`base_nomco_lb14`) performance of all 16 baseline variants, ranked by Combined%.*

### 4.2.1 LSTM-Family Models

Recurrent architectures are the strongest single-paradigm family for Malaysian ridership under normal conditions. The main idea is that gated sequential memory aligns with the daily and weekly periodicity that Chapter 3's EDA identified as dominant demand drivers. The evidence is structural: LSTM-family models occupy five of the top eight positions, with BiLSTM (79.04%, R² = 0.776) and TPA-LSTM (78.67%, R² = 0.776) leading the series and the base LSTM (78.13%, R² = 0.760) ranking fifth. Barath et al. (2026) report that LSTM achieves consistently competitive accuracy across diverse time series benchmarks as a general-purpose sequential baseline; the present ranking corroborates that claim and establishes a strong performance floor for the comparison chain.

Bidirectional encoding provides a measurable advantage over unidirectional recurrence. BiLSTM's 0.91 Combined% gain over LSTM is consistent with Alajmi and Almutairi (2025), who show that bidirectional windows improve traffic congestion prediction by contextualising mid-sequence events against subsequent observations. TPA-LSTM's near-equal R² to BiLSTM (0.776) validates temporal pattern attention: Wei et al. (2023) demonstrate that one-dimensional CNN pattern attention over LSTM hidden states extracts multi-scale periodic signals more effectively than positional bidirectionality alone for subway passenger flow. CNN-BiLSTM (78.18%) and CNN-LSTM sequential mode (77.64%) confirm that hierarchical local-then-global encoding can approach bidirectional performance; Chen et al. (2022) find that CNN-BiLSTM coupling reduces traffic prediction error beyond either component in isolation. ST-LSTM (78.01%, R² = 0.770) adds a parallel spatial MLP stream and ranks sixth, supporting Cui et al.'s (2025) finding that decoupled spatial encoding improves urban rail passenger-flow prediction when multi-source inputs are present. The analysis, however, must be qualified: Section 4.6 shows that all LSTM-family models are chronic overfitters despite these strong test numbers — a tension that the graph-based results below partially mirror.

### 4.2.2 Graph-Based Models

Graph-based models exhibit wider performance dispersion than the LSTM family, reflecting heterogeneous strategies for encoding inter-feature correlations. MTGNN (77.93%, R² = 0.775) ranks seventh overall and leads the graph series, narrowly ahead of STSGCN (76.97%) and STGCN (76.81%). Wu et al. (2025) show that end-to-end learned asymmetric adjacency with multi-scale dilated inception outperforms static correlation graphs for traffic flow prediction; MTGNN's placement is consistent with that finding, because asymmetric learned edges can represent directed relationships — for example, fuel-price signals influencing ridership without the reverse — that symmetric Pearson graphs cannot.

STGCN benefits from Pearson-correlation feature-graph construction over the 79-feature input and performs better than earlier literature comparisons suggested. Deng (2025) confirms that Chebyshev graph convolution interleaved with temporal gated convolution captures relational dependencies missed by purely recurrent models; nevertheless, STGCN trails the LSTM baseline by 1.32 Combined% (76.81 versus 78.13), indicating that explicit message passing over feature correlations does not overcome recurrent inductive bias at base capacity. STSGCN (76.97%), with its synchronous 3N×3N spatial-temporal adjacency, slightly exceeds STGCN, supporting Chen et al.'s (2023) argument that simultaneous spatial and temporal graph operations better capture coordinated multi-feature regime shifts during public holidays. At the lower end, PDR-STGCN (73.91%, R² = 0.688) and STFGNN (73.94%, R² = 0.688) underperform despite theoretically richer designs; Hu et al. (2026) describe PDR-STGCN's dynamic relational graph and periodicity-aware encoding, yet limited base capacity relative to dual-graph complexity constrains its default-configuration score. STFGNN's primary liabilities emerge in the sensitivity analyses of Sections 4.4 and 4.5 rather than at lb14 alone.

### 4.2.3 Attention-Based Models

Informer (79.13%, R² = 0.772) is the single strongest baseline, narrowly ahead of BiLSTM (79.04%) by 0.09 Combined%. The main idea is that sparse attention with a generative decoder can match recurrent accuracy while regularising against noise. Song et al. (2024) apply a graph-augmented Informer to long-term traffic flow prediction and show that ProbSparse attention achieves competitive accuracy with reduced computational overhead; at T_in = 14, ProbSparse degenerates to near-full attention (approximately 13 of 14 queries active), so the approximation introduces no meaningful accuracy loss while the decoder architecture avoids autoregressive error accumulation. Informer's head-of-field placement, combined with superior MCO robustness (Section 4.5) and a perfect base-configuration fit record (Section 4.6), makes it the most well-rounded baseline.

Autoformer (74.73%, R² = 0.708) ranks 13th, trailing leaders by a substantial margin. Ma and Zhang (2025) report stable Autoformer performance on strongly periodic signals under appropriate sequence lengths; here, spectral decomposition across 79 heterogeneous feature channels under a moderate look-back appears to limit gains relative to simpler recurrent baselines, and MCO fragility (Section 4.5) further reduces practical utility. ASTGCN (75.69%, R² = 0.703) performs moderately; Cui et al. (2023) show that dual spatial-temporal attention improves non-stationary sensitivity over static graph message passing, yet ASTGCN's largest gains arrive only after tuning (+3.13 Combined%, Section 4.3). Together, the three series establish a competitive baseline field against which HMT-TSF is evaluated in Section 4.7.

---

## 4.3 Effect of Hyperparameter Tuning

Hyperparameter tuning does not uniformly improve baseline performance; instead, it produces model-specific accuracy shifts at a consistent generalisation cost documented in Section 4.6. Table 4.3.1 summarises tuned versus base results at `nomco_lb14`.

| Model | Base Combined% | Tuned Combined% | Δ Combined |
| ----- | -------------- | --------------- | ---------- |
| Informer | 79.13 | 79.99 | +0.86 |
| TPA-LSTM | 78.67 | 79.95 | +1.28 |
| BiLSTM | 79.04 | 79.44 | +0.40 |
| ST-LSTM | 78.01 | 79.13 | +1.12 |
| ASTGCN | 75.69 | 78.82 | +3.13 |
| LSTM | 78.13 | 78.43 | +0.30 |
| STGCN | 76.81 | 78.39 | +1.58 |
| CNN-LSTM-Augmented | 75.36 | 77.10 | +1.74 |
| PDR-STGCN | 73.91 | 75.56 | +1.65 |
| Autoformer | 74.73 | 75.21 | +0.48 |
| MTGNN | 77.93 | 76.40 | −1.54 |
| CNN-BiLSTM | 78.18 | 76.12 | −2.06 |
| STSGCN | 76.97 | 73.65 | −3.33 |
| CNN-LSTM | 77.64 | 73.36 | −4.28 |
| CNN-LSTM-Parallel | 74.18 | 72.53 | −1.65 |
| STFGNN | 73.94 | 65.47 | −8.47 |

*Table 4.3.1. Tuning effect at `tuned_nomco_lb14` versus `base_nomco_lb14`.*

The median tuning improvement is approximately +0.5 Combined%, indicating that test-set accuracy is broadly flat while generalisation deteriorates. The largest beneficiaries are ASTGCN (+3.13, to 78.82%), CNN-LSTM-Augmented (+1.74), PDR-STGCN (+1.65), and STGCN (+1.58). For ASTGCN, widening from d_model=64 to 128, expanding attention heads from 4 to 8, and deepening blocks from 2 to 3 increases expressiveness of the dual attention mechanism — consistent with Cui et al. (2023), who report that deeper attention configurations improve sensitivity to non-stationary patterns. STGCN's gain from doubling hidden dimensions to 256 while strengthening regularisation aligns with Deng (2025), who shows that wider spatio-temporal graph networks extract richer relational representations when dropout and weight decay are co-tuned.

The most severe regressions are STFGNN (−8.47, to 65.47%), CNN-LSTM sequential (−4.28), and STSGCN (−3.33). STFGNN's collapse is the study's largest tuning failure: raising hidden dimensions, adding a fusion block, and increasing dropout to 0.35 causes near-total test-split breakdown, suggesting that dual-graph fusion at this training budget memorises training patterns that do not transfer. CNN-LSTM sequential degrades because widened CNN filters (32 → 64) amplify an existing single-stream bottleneck. MTGNN (−1.54) and CNN-BiLSTM (−2.06) also regress, confirming that additional capacity harms some architectures. The best tuned baseline remains Informer (79.99%), still 5.79 Combined% below HMT-TSF — a gap Section 4.7 examines in detail.

---

## 4.4 Look-Back Window Sensitivity

A systematic sweep across lb14, lb28, and lb56 reveals that longer historical context does not improve seven-day-ahead accuracy for most architectures. The principal finding is operational: **lb14 is optimal for 10 of 16 models, lb28 for 6, and lb56 for none.** Table 4.4.1 reports Combined% across the three windows under `base_nomco`.

| Model | lb14 | lb28 | lb56 | Best |
| ----- | ---- | ---- | ---- | ---- |
| Informer | **79.13** | 78.63 | 77.17 | lb14 |
| BiLSTM | **79.04** | 76.97 | 77.09 | lb14 |
| TPA-LSTM | 78.67 | **79.08** | 76.36 | lb28 |
| CNN-BiLSTM | **78.18** | 72.65 | 74.72 | lb14 |
| LSTM | **78.13** | 77.48 | 75.98 | lb14 |
| ST-LSTM | 78.01 | **78.86** | 75.60 | lb28 |
| MTGNN | 77.93 | **78.11** | 77.76 | lb28 |
| CNN-LSTM | **77.64** | 73.90 | 74.33 | lb14 |
| STSGCN | **76.97** | 75.02 | 75.15 | lb14 |
| STGCN | 76.81 | **77.68** | 74.93 | lb28 |
| ASTGCN | 75.69 | **76.89** | 75.15 | lb28 |
| CNN-LSTM-Augmented | **75.36** | 74.73 | 74.04 | lb14 |
| Autoformer | **74.73** | 73.61 | 73.40 | lb14 |
| CNN-LSTM-Parallel | 74.18 | **76.55** | 74.33 | lb28 |
| STFGNN | **73.94** | 61.30 | 43.25 | lb14 |
| PDR-STGCN | **73.91** | 72.35 | 72.20 | lb14 |

*Table 4.4.1. Look-back sensitivity (Combined%) across lb14, lb28, and lb56 for all 16 baseline variants (`base_nomco`). The best window per model is shown in bold: lb14 is optimal for 10 models, lb28 for 6, and lb56 for none.*

Fourteen days span exactly two weekly commuting cycles — the periodicity Chapter 3's EDA and Section 4.8's SHAP analysis identify as the dominant short-horizon structure — while slower seasonal variation is supplied explicitly through calendar features rather than recovered from extended history. Ma and Zhang (2025) similarly report that decomposition-based forecasters require sequence lengths matched to dominant periodicity; here, additional days contribute redundancy and, for fragile architectures, noise.

MTGNN is the most window-stable model (77.93 / 78.11 / 77.76), consistent with Wu et al.'s (2025) account of multi-scale dilated inception preventing any single temporal scale from dominating. STFGNN exhibits the most severe collapse: Combined% falls from 73.94% at lb14 to 43.25% at lb56, with R² of −0.112 — worse than naive mean prediction — because the temporal adjacency matrix constructed from 56-timestep profiles supplies misleading structural priors (Chang et al., 2025). CNN-LSTM sequential degrades sharply at lb28 (73.90% versus 77.64% at lb14) as the two-block CNN at base filter width compresses long-range patterns too aggressively; parallel and augmented modes, which preserve direct raw-feature access or skip connections, avoid this failure mode. HMT-TSF's own profile (Section 4.7) peaks at lb14 under both MCO conditions once future-calendar conditioning is corrected, reinforcing the 14-day recommendation for seven-day horizons.

---

## 4.5 MCO Structural Break Robustness

The MCO period represents a severe distributional discontinuity in which daily boardings fell from 800,000–1,200,000 to 50,000–200,000 under mobility restrictions. Including this window (`mco` configuration) tests whether models trained on pre-break, break, and post-break data can forecast through the transition. The degradation metric Δ Combined% (nomco → mco) serves as the primary robustness indicator; Table 4.5.1 ranks baselines at lb14.

| Model | nomco | mco | Δ Combined | mco R² |
| ----- | ----- | --- | ---------- | ------ |
| Informer | 79.13 | 73.25 | −5.88 | 0.705 |
| BiLSTM | 79.04 | 72.74 | −6.29 | 0.699 |
| CNN-LSTM-Parallel | 74.18 | 66.89 | −7.29 | 0.577 |
| TPA-LSTM | 78.67 | 71.05 | −7.61 | 0.669 |
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
| CNN-LSTM | 77.64 | 45.58 | −32.06 | −0.050 |
| STFGNN | 73.94 | 41.00 | −32.94 | −0.306 |
| **HMT-TSF** | **85.78** | **81.99** | **−3.79** | **0.859** |

*Table 4.5.1. MCO robustness at lb14 for all 16 baseline variants, ranked by Δ Combined% (least to most degradation). HMT-TSF included for reference.*

The degradation spread among baselines runs from −5.88 (Informer) to −32.94 (STFGNN). Informer and BiLSTM are the most robust recurrent baselines; Song et al. (2024) attribute Informer's sparse attention to reduced sensitivity to distributional anomalies because only the most predictive query-key pairs are engaged across regime boundaries. Two models fail in a technically meaningful sense: CNN-LSTM sequential (R² = −0.050) and STFGNN (R² = −0.306) produce forecasts less accurate than predicting the training-set mean — despite CNN-LSTM ranking eighth under normal conditions. Topilin et al. (2025) note that sequential CNN-LSTM architectures can memorise local temporal patterns at the expense of adaptability, consistent with the −32.06 Combined% degradation observed here.

STGCN demonstrates unexpectedly strong MCO robustness (Δ −8.27, mco R² = 0.615), outperforming more expressive models including MTGNN (Δ −10.60). Deng (2025) show that STGCN's fixed Chebyshev graph constrains prediction drift when input distributions shift — a rigidity that limits peak accuracy but caps catastrophic failure. The analysis reveals a recurring expressiveness-stability trade-off: flexible learned representations (STFGNN's dual graphs, CNN-LSTM's CNN-stage encoding) memorise pre-break regularities that become liabilities after the break, while constrained or regime-aware designs transfer more reliably — a theme HMT-TSF addresses through RevIN and regime gating in Section 4.7.

---

## 4.6 Generalisation and Overfitting Analysis

Test-set Combined% alone is an unreliable deployment criterion because several high-ranking baselines achieve accuracy partly through memorisation. Fit diagnostics across all 12 configurations per model (base and tuned × nomco and mco × three look-backs) show that **all six LSTM-family models and both simple-convolution graph models (STGCN, PDR-STGCN) record zero good-fit verdicts** — 0/12 each — despite strong headline scores for BiLSTM, TPA-LSTM, and CNN-BiLSTM.

The gap ratio — best validation loss divided by minimum training loss — is the clearest signal. LSTM-family gap ratios span 3.3× to 13.6×, with CNN-BiLSTM-base under MCO at lb56 reaching 13.51×. PDR-STGCN is the most extreme memoriser, posting 38.61× at tuned_nomco_lb56, indicating training loss near zero while validation loss remains elevated. By contrast, Informer achieves a perfect 6/6 good-fit rate at base configuration — the only baseline with a clean record — with gap ratios of 2.16×–2.98× consistently below the 3× threshold. Song et al. (2024) attribute this partly to ProbSparse attention's implicit regularisation: restricting active queries to a logarithmic subset suppresses fitting of high-frequency training noise.

Among graph models, MTGNN (3/6 good-fit at base, 4/6 tuned) and STSGCN (4/6 base) are cleanest after Informer, with MTGNN posting study-tightest gaps of 1.40×–1.50× in tuned configurations — supporting Wu et al.'s (2025) observation that end-to-end learned adjacency with mix-hop convolution provides intrinsic structural regularisation. HMT-TSF leads the full study at 19/20 good-fit verdicts across both full and feature-reduced variants (Section 4.7.5). Tuning uniformly harms diagnostics: the good-fit rate falls from 26% (base) to 14% (tuned) across all 14 baselines, confirming that expanded capacity raises gap ratios without equivalent accuracy gains. Deployment decisions must therefore pair accuracy with fit quality — a practice Chapter 5 develops into practical recommendations.

---

## 4.7 Proposed Model Performance: HMT-TSF

### 4.7.1 Main Results Across Configurations

HMT-TSF was evaluated across ten configurations covering two MCO conditions and five look-back windows (7, 14, 28, 56, 84 days). Table 4.7.1 presents the full results.

| Configuration  | Combined% | MAPE%    | MAE%     | RMSE%    | R²        | MAE        | RMSE       |
| -------------- | --------- | -------- | -------- | -------- | --------- | ---------- | ---------- |
| nomco_lb7      | 84.88     | 4.74     | 4.25     | 6.13     | 0.888     | 53,325     | 76,951     |
| **nomco_lb14** | **85.78** | **4.39** | **3.97** | **5.87** | **0.897** | **49,897** | **73,757** |
| nomco_lb28     | 84.23     | 4.94     | 4.55     | 6.28     | 0.883     | 57,268     | 78,910     |
| nomco_lb56     | 82.53     | 5.52     | 5.11     | 6.84     | 0.859     | 64,198     | 85,874     |
| nomco_lb84     | 77.68     | 6.99     | 6.59     | 8.73     | 0.773     | 82,174     | 108,800    |
| mco_lb7        | 78.61     | 6.89     | 6.12     | 8.38     | 0.807     | 75,401     | 103,170    |
| **mco_lb14**   | **81.99** | 5.68     | 5.18     | 7.15     | 0.859     | 63,853     | 88,228     |
| mco_lb28       | 79.89     | 6.26     | 5.83     | 8.02     | 0.822     | 72,032     | 99,059     |
| mco_lb56       | 79.47     | 6.68     | 5.91     | 7.95     | 0.825     | 73,334     | 98,598     |
| mco_lb84       | 78.45     | 6.73     | 6.17     | 8.64     | 0.791     | 76,593     | 107,293    |

*Table 4.7.1. HMT-TSF (full, F=79) results across all ten configurations. Bold rows denote best nomco and mco configurations. The feature-reduced variant exceeds these values at every nomco lookback ≥14 (Section 4.9).*

The nomco look-back profile is single-peaked at lb14 (85.78%), declining to 77.68% at lb84. With corrected known-future calendar conditioning (`X_future`), the lb14 peak persists under MCO (81.99%) — short windows no longer require extended context to absorb the lockdown discontinuity once the model receives explicit future calendar structure. HMT-TSF's margin over naive persistence is substantial: persistence achieves Combined% of approximately 34.8 under nomco and 25.7 under mco, with negative R² in every configuration; at nomco_lb14, HMT-TSF exceeds naive by +50.94 Combined% and +1.858 R², quantifying pattern complexity beyond trivial autocorrelation.

### 4.7.2 Comparison Against Baselines

HMT-TSF at nomco_lb14 (Combined% 85.78, R² 0.897, MAE 49,897) leads every baseline on every headline metric. The margin over the best baseline — Informer at 79.13% — is +6.65 Combined% and −19,579 MAE (28.2% reduction in absolute daily error). The best tuned baseline (Informer, 79.99%) remains 5.79 Combined% below HMT-TSF and 6.60 below the feature-reduced variant (86.59%, Section 4.9). One qualification applies: HMT-TSF alone receives the known-future calendar tensor (`X_future`), so the comparison reflects architecture-plus-conditioning versus architecture-only baselines.

The performance advantage stems from parallel multi-modal encoding: multi-scale TCN with WaveNet-style gating captures temporal dynamics; a Pearson-correlation GCN propagates inter-feature dependencies; and a Regime Gating Embedding with K=3 learnable vectors addresses pre-MCO, MCO, and post-MCO distributional variation. Feature Group Fusion independently embeds five semantic feature sources before a learned softmax gate adaptively weights their contributions — a capability no single-paradigm baseline implements.

### 4.7.3 MCO Robustness and Walk-Forward Validation

HMT-TSF is the most MCO-robust model at the headline lookback (Δ −3.79, nomco 85.78 → mco 81.99), ahead of the best baseline (Informer, Δ −5.88). It retains the highest absolute MCO accuracy in the study (81.99%), outperforming the best MCO baseline (Informer tuned, 75.79%) by 6.20 Combined%. Compared to STFGNN (Δ −32.94) and CNN-LSTM sequential (Δ −32.06), HMT-TSF's degradation is roughly an order of magnitude smaller. RevIN removes within-batch amplitude drift while regime gating adjusts prediction priors per operational period; together they enable transfer across the MCO discontinuity. The feature-reduced variant is less MCO-robust (Δ −7.85 at lb14, Section 4.9), indicating that part of this robustness resides in full-feature redundancy rather than architecture alone.

Three sequential non-overlapping out-of-sample validation blocks were constructed to test whether headline test-set performance depends on a single favourable temporal partition. The main idea is that a model fit to one chronological split may exploit idiosyncratic seasonal segments; walk-forward evaluation exposes such dependence before deployment. The evidence shows that no block falls below 71.99 Combined% or 0.707 R² across all ten HMT-TSF configurations. At nomco_lb14, block-level Combined% is 90.62 (Block 1), 82.14 (Block 2), and 85.12 (Block 3); Block 2, likely corresponding to a mid-evaluation seasonal transition, is the weakest segment under nomco at short lookbacks. Under mco, the pattern inverts: Block 1, which absorbs the lockdown-adjacent segment, is weakest while Block 2 is strongest — consistent with regime gating learning to down-weight the discontinuity once sufficient post-break history is available. The analysis further shows that mco_lb84 is the most temporally stable configuration (block spread 3.7 Combined%), while nomco_lb14 spreads 8.5 points between best and worst blocks. Operators prioritising peak accuracy should deploy nomco_lb14 or HMT-TSF-FR at the same window; those prioritising consistency across evaluation periods may prefer longer-context MCO-trained settings, accepting the accuracy penalty documented in Table 4.7.1.

### 4.7.4 Generalisation and Fit Diagnosis

HMT-TSF achieves `good_fit` verdicts on nine of ten full-model configurations and all ten feature-reduced configurations — 19/20 overall, unmatched by any baseline family. Gap ratios range from 1.57× to 2.64×, uniformly below the 3× threshold; nomco_lb14 posts the cleanest profile (gap 1.57×, validation drift 0.61%). The sole exception is nomco_lb84, diagnosed `overfit` on validation drift (+36.9%), confirming that an 84-day window over a 7-day horizon supplies excess context. All five MCO-included configurations achieve `good_fit` in both variants — the only architecture maintaining clean generalisation under the structural break.

---

## 4.8 SHAP Feature Importance Analysis

SHAP values were computed for the trained nomco_lb14 model using `shap.GradientExplainer` with 100 test samples as background. Absolute SHAP values were averaged over samples, time steps, and forecast horizons to produce a 79-element importance vector resolved from `feature_metadata.json`.

The highest-importance features align with established transit demand theory. Annual-cycle encodings dominate attribution magnitude: `month_cos`, `month_sin`, and `month` carry the three largest mean |SHAP| values, anchoring seven-day forecasts on seasonal position — school terms, festive seasons, and the northeast monsoon. `total_ridership` is the strongest non-seasonal driver and the only feature in the top-15 of all ten configurations (100% consistency), confirming target autocorrelation as the dominant data-driven signal. Holiday structure forms the second calendar tier: `days_to_next_public_hol` ranks in the headline top-10 and appears in 80% of configurations, quantifying the pre-holiday dips and post-holiday surges identified in Chapter 3's EDA. Among external covariates, unfrozen fuel grades (`fp_lv_diesel`, `fp_lv_ron97`) and monsoon-corridor rainfall (Perlis, Terengganu, Penang) complete the headline top-10 — consistent with Cui et al. (2025), who identify multi-source fuel and weather signals as secondary urban rail demand drivers.

Cross-configuration aggregation reveals 23 features with zero importance in all ten runs (the entire 17-feature static group plus six administered or East Malaysia fuel-price columns) and three near-zero features (RON95 level and change variants, frozen at RM2.05 post-MCO). These 26 features define the removal set for HMT-TSF-FR (Section 4.9). Their persistent inactivity independently validates the EDA conclusion in Chapter 3 that administered RON95 series and time-invariant spatial descriptors carry no day-to-day discriminative variance in the post-MCO training window. A lookback-dependent shift is observed: at lb7–lb28, calendar features dominate because short context provides limited autocorrelation depth; at lb56–lb84, `total_ridership` and major Klang Valley rail lines (`rail_lrt_kj`, `rail_mrt_kajang`) take top ranks as extended windows let the model anchor on recent ridership trajectory. This shift has a direct modelling implication: feature-importance profiles are not fixed properties of the dataset but co-vary with architectural context length — a point that cautions against interpreting a single SHAP run as the definitive driver hierarchy without cross-configuration corroboration.

---

## 4.9 Feature-Reduced Variant: HMT-TSF-FR

Removing 26 zero- or near-zero-SHAP features — the entire static group plus nine near-constant administered fuel columns — produces HMT-TSF-FR on a 53-feature input. Static descriptors are redundant because time-invariant structural capacity is implicitly encoded in historical ridership given sufficient look-back; administered fuel columns carry near-zero day-to-day variance across the post-MCO training window.

HMT-TSF-FR achieves Combined% of 86.59 at nomco_lb14 (R² 0.906, MAE 46,323), exceeding the full model (85.78) by +0.81 and setting the study-wide headline. The graph adjacency shrinks from 79×79 to 53×53 and explainability memory decreases by approximately 33%. The FR advantage grows with look-back (+0.81 at lb14, +1.84 at lb28, +1.52 at lb56, +4.26 at lb84), repairing nomco_lb84's overfit verdict under feature reduction.

The trade-off emerges under structural break: HMT-TSF-FR loses to the full model in every mco configuration (Δ −0.85 at lb7 to −5.22 at lb84). Near-constant fuel-price columns evidently stabilise predictions under lockdown-spanning distributional shift; the full model should be retained for shock-prone conditions while FR is recommended for normal operations. All HMT-TSF-FR configurations remain `good_fit` with gap ratios within 1.66–2.72×, confirming that feature reduction does not introduce overfitting and that the 53-feature variant is the preferred default for production deployment under the nomco regime — a deployment rule that Chapter 5 formalises alongside the complementary shock-regime recommendation for the full 79-feature model.

---

## 4.10 Summary of Findings

Five thematic conclusions emerge from the empirical evaluation.

**First, HMT-TSF outperforms all 16 baselines.** At nomco_lb14, Combined% reaches 85.78 (86.59 for FR), leading Informer (79.13%) by 6.65 points and reducing MAE by 28.2% (33.3% for FR). The gain reflects multi-scale TCN encoding, GCN message passing, RevIN, regime gating, and known-future calendar conditioning (`X_future`) — an input advantage unique to HMT-TSF.

**Second, HMT-TSF is the most MCO-robust and most generalisable model.** MCO degradation at lb14 is Δ −3.79; retained MCO accuracy (81.99%) leads the field by 6.2 Combined%; and the 19/20 good-fit record is unmatched.

**Third, LSTM-family models dominate the top eight yet overfit systematically.** BiLSTM and TPA-LSTM achieve the highest baseline R² (0.776) but zero good-fit verdicts across all 12 configurations. Informer is the sole baseline with a clean base fit record and the most deployable alternative.

**Fourth, lb14 is decisively optimal for a seven-day horizon.** Longer windows trigger severe collapses (STFGNN at lb56, CNN-LSTM at lb28) and HMT-TSF's only overfit verdict (nomco_lb84).

**Fifth, SHAP analysis confirms interpretable, domain-consistent feature hierarchies** and enables principled reduction to 53 features for normal-regime deployment, with the full 79-feature model retained for shock-prone regimes.

Chapter 5 interprets these findings, discusses implications for Malaysian transit planning, acknowledges limitations, and outlines future work.

---

## References

Alajmi, M. S., & Almutairi, S. M. (2025). Intelligent traffic congestion forecasting using BiLSTM and adaptive secretary bird optimizer for sustainable urban transportation. *Scientific Reports*, *15*. https://doi.org/10.1038/s41598-025-02933-9

Barath, Z., Veres, P., & Banyai, A. (2026). Beyond traditional forecasting methods: Evaluating LSTM performance on diverse time series. *Mathematics*, *14*(5), 838. https://doi.org/10.3390/math14050838

Chang, J., Yin, J., Hao, Y., & Gao, C. (2025). STFDSGCN: Spatio-temporal fusion graph neural network based on dynamic sparse graph convolution GRU for traffic flow forecast. *Sensors*, *25*(11), 3446. https://doi.org/10.3390/s25113446

Chen, L., Ren, Q., Zeng, J., Zou, F., Luo, S., Tian, J., & Xing, Y. (2023). CSFPre: Expressway key sections based on CEEMDAN-STSGCN-FCM during the holidays for traffic flow prediction. *PLOS ONE*, *18*(4), e0283898. https://doi.org/10.1371/journal.pone.0283898

Chen, W., Yang, Z., Xu, G., & Sun, Y. (2022). Short-term traffic flow prediction based on CNN-BiLSTM with multicomponent information. *Applied Sciences*, *12*(17), 8714. https://doi.org/10.3390/app12178714

Cui, H., Si, B., Chi, D., Li, Y., Li, G., & Chen, Y. (2025). Short-term passenger flow prediction for urban rail systems: A deep learning approach utilizing multi-source big data. *PLOS ONE*, *20*(1), e0333094. https://doi.org/10.1371/journal.pone.0333094

Cui, Z., Zhang, J., Noh, G., & Park, H. J. (2023). ADSTGCN: A dynamic adaptive deeper spatio-temporal graph convolutional network for multi-step traffic forecasting. *Sensors*, *23*(15), 6950. https://doi.org/10.3390/s23156950

Deng, H. (2025). Traffic-forecasting model with spatio-temporal kernel. *Electronics*, *14*(7), 1410. https://doi.org/10.3390/electronics14071410

Hu, J., Tang, B., Zhu, L., Li, Y., Hu, J., & Yang, G. (2026). PDR-STGCN: An enhanced STGCN with multi-scale periodic fusion and a dynamic relational graph for traffic forecasting. *Systems*, *14*(1), 102. https://doi.org/10.3390/systems14010102

Ma, X., & Zhang, H. (2025). Time series forecasting method based on multi-scale feature fusion and Autoformer. *Applied Sciences*, *15*(7), 3768. https://doi.org/10.3390/app15073768

Song, Y., Luo, R., Zhou, T., Zhou, C., & Su, R. (2024). Graph attention Informer for long-term traffic flow prediction under the impact of sports events. *Sensors*, *24*(15), 4796. https://doi.org/10.3390/s24154796

Topilin, I., Jiang, J., Feofilova, A., & Beskopylny, N. (2025). Traffic flow prediction via a hybrid CPO-CNN-LSTM-attention architecture. *Smart Cities*, *8*(5), 148. https://doi.org/10.3390/smartcities8050148

Wei, L., Guo, D., Chen, Z., Yang, J., & Feng, T. (2023). Forecasting short-term passenger flow of subway stations based on the temporal pattern attention mechanism and the long short-term memory network. *ISPRS International Journal of Geo-Information*, *12*(1), 25. https://doi.org/10.3390/ijgi12010025

Wu, Z., Liu, X., & Zhang, X. (2025). Multi dynamic temporal representation graph convolutional network for traffic flow prediction. *Scientific Reports*, *15*, 16734. https://doi.org/10.1038/s41598-025-01157-1
