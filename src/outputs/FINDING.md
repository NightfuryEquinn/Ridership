# Chapter 4: Findings and Results

---

## 4.1 Overview of Experimental Evaluation

This chapter reports the empirical findings from a systematic comparative evaluation of 15 deep-learning architectures — 14 baseline models, evaluated as 16 ranked variants because CNN-LSTM contributes three independent fusion modes, plus the proposed hybrid HMT-TSF — for seven-day Malaysian transit ridership forecasting. The evaluation is structured across four analytical dimensions: (1) default-configuration performance under normal operating conditions (MCO-excluded, 14-day look-back); (2) the effect of hyperparameter tuning on both accuracy and generalisation; (3) look-back window sensitivity across three window lengths (14, 28, and 56 days); and (4) structural-break robustness under the Movement Control Order (MCO) pandemic period (2020-03-18 to 2021-12-31). A dedicated fit-diagnostics analysis examines the generalisation quality of all models independently of their test-set accuracy.

The 14 baseline models span three architectural series — spatio-temporal recurrent (LSTM-family), graph-based, and attention-based — and are complemented by the proposed hybrid model, HMT-TSF (Hybrid Multi-scale Temporal Spatio-Feature Forecaster). Each model is evaluated using the composite Combined% metric, which aggregates three complementary error perspectives into a single scalar: `Combined% = max(0, 100 − MAPE% − MAE% − RMSE%)`, where all percentage terms are normalised by mean ridership demand, making them directly addable. Supplementary metrics (R², absolute MAE, RMSE) are reported alongside Combined% to provide a complete performance profile. The headline configuration is `base_nomco_lb14` (base architecture, MCO-excluded dataset, 14-day look-back), which serves as the project default and primary comparison benchmark.

All models share a standardised training protocol: AdamW optimiser (Loshchilov & Hutter, 2019) with decoupled weight decay, HuberLoss objective, a five-epoch linear learning rate warm-up followed by ReduceLROnPlateau scheduling, and a 70/15/15 chronological train/validation/test split. Fit diagnostics are derived from three signals recorded per training run: validation drift above the best checkpoint (threshold 25%), the gap ratio between best validation loss and minimum training loss (threshold 3×), and train/validation divergence in the final training epochs.

---

## 4.2 Baseline Model Performance — Default Configuration (`base_nomco_lb14`)

Table 4.2.1 presents the complete default-configuration ranking of all 16 baseline variants. The subsections that follow analyse the results by architectural series.

| Rank | Model | Combined% | MAPE% | MAE% | RMSE% | R² | MAE | RMSE |
|------|-------|-----------|-------|------|-------|-----|-----|------|
| 1 | Informer | 79.13 | 6.59 | 5.53 | 8.76 | 0.772 | 69,476 | 110,050 |
| 2 | BiLSTM | 79.04 | 6.52 | 5.77 | 8.67 | 0.776 | 72,460 | 109,007 |
| 3 | TPA-LSTM | 78.67 | 6.62 | 6.04 | 8.67 | 0.776 | 75,871 | 109,009 |
| 4 | CNN-BiLSTM | 78.18 | 6.81 | 6.12 | 8.89 | 0.764 | 76,884 | 111,782 |
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

*Table 4.2.1. Default-configuration (`base_nomco_lb14`) performance of all 16 baseline variants, ranked by Combined%.*

### 4.2.1 LSTM-Family Models

The LSTM-family series occupies five of the top eight positions in the default configuration, confirming the strong suitability of recurrent architectures for the sequential, periodicity-rich structure of Malaysian transit ridership data. Strigula (2026) established that LSTM achieves consistently competitive accuracy across diverse time series benchmarks as a general-purpose sequential baseline, and the results of this study corroborate that claim: the base LSTM achieves a Combined% of 78.13 with R² = 0.760, ranking fifth overall and establishing a strong performance floor for the comparison chain.

BiLSTM (79.04%, R² = 0.776) and TPA-LSTM (78.67%, R² = 0.776) are the strongest LSTM-family models, with both models narrowly ahead of the LSTM baseline on all headline metrics. Alajmi and Almutairi (2025) demonstrated that bidirectional encoding over a sequential window improves traffic congestion prediction accuracy by enabling mid-window events to be contextualised in light of subsequent observations; this advantage is reflected in BiLSTM's 0.91 Combined% gain over the unidirectional LSTM. TPA-LSTM's competitive placement validates the temporal pattern attention mechanism: Wei et al. (2023) showed that 1-D CNN-based pattern attention over LSTM hidden states extracts multi-scale periodic signals — daily and weekly ridership cycles — more effectively than positional bidirectionality alone, and TPA-LSTM's near-equal R² to BiLSTM (0.776) confirms this for Malaysian ridership.

CNN-BiLSTM (78.18%, R² = 0.765) and CNN-LSTM sequential mode (77.64%, R² = 0.752) demonstrate that hierarchical local-then-global encoding can match or approach bidirectional models. Chen et al. (2022) found that combining CNN local pattern extraction with BiLSTM bidirectional context reduces prediction error more than either component in isolation for multi-sensor traffic data, and CNN-BiLSTM's rank-4 placement provides supporting evidence for this combined inductive bias in transit ridership applications. ST-LSTM (78.01%, R² = 0.770) — which introduces an explicit parallel spatial MLP stream alongside the LSTM temporal stream — ranks sixth, confirming that decoupled spatial encoding provides modest but measurable improvement over a flat multivariate input (Cui et al., 2025).

The top-8 default-configuration rankings fall within a narrow 1.5 Combined% band (77.64–79.13), indicating a saturated accuracy regime where architectural innovations produce incremental rather than step-change gains under normal operating conditions.

### 4.2.2 Graph-Based Models

Graph-based models display a wider performance spread than the LSTM-family series, reflecting greater architectural heterogeneity in how inter-feature correlations are encoded. MTGNN (77.93%, R² = 0.775) ranks seventh overall and is the strongest graph model in the default configuration, narrowly ahead of STGCN (76.81%, R² = 0.750) and STSGCN (76.97%, R² = 0.736). Wu et al. (2025) demonstrated that end-to-end learned asymmetric graph adjacency with multi-scale dilated inception outperforms static correlation-based graph models for traffic flow prediction; MTGNN's competitive placement in the graph series is consistent with this finding. The asymmetric learned adjacency allows the model to capture causally directed feature relationships — for example, fuel price signals influencing ridership without the reverse — that the symmetric Pearson graphs of STGCN and ASTGCN cannot represent.

STGCN (76.81%) performs better than expected relative to earlier literature comparisons, benefiting from the Pearson-correlation feature graph construction applied to the 79-feature input. Deng (2025) confirmed that interleaving Chebyshev graph convolution with temporal gated convolution captures relational dependencies that purely recurrent models miss; nevertheless, STGCN trails the LSTM baseline by 1.32 Combined% (76.81 versus 78.13), indicating that explicit graph-structured message passing over feature correlations does not by itself overcome the strong sequential inductive bias of recurrent models in this configuration. STSGCN (76.97%), which uses a synchronous 3N×3N joint spatial-temporal adjacency matrix, slightly exceeds STGCN, supporting Chen et al.'s (2023) finding that simultaneous spatial and temporal graph operations better capture synchronous multi-feature regime shifts — such as the coordinated response of multiple ridership lines and external signals during public holiday periods.

The weakest graph models in default configuration are PDR-STGCN (73.91%, R² = 0.688) and STFGNN (73.94%, R² = 0.688). PDR-STGCN's relatively low base performance, despite its dynamic relational graph and periodicity-aware two-channel input, reflects limited base capacity (hidden = 160, dk = 48) relative to the demands of its dual static-dynamic graph design; tuning recovers part of this deficit (+1.65 Combined%, Section 4.3) but does not lift the model out of the lower rank tier. STFGNN's dual spatial-temporal graph fusion, while theoretically aligned with Chang et al.'s (2025) spatio-temporal fusion approach, shows limited advantage over STGCN at base capacity under normal operating conditions, with its primary differentiating characteristics emerging in the sensitivity analyses reported in Sections 4.4 and 4.5.

### 4.2.3 Attention-Based Models

Informer (79.13%, R² = 0.772) is the single strongest baseline model in the default configuration, narrowly outperforming BiLSTM (79.04%) by 0.09 Combined%. Song et al. (2024) applied a graph-augmented Informer to long-term traffic flow prediction and demonstrated that ProbSparse attention with distilling achieves competitive forecasting accuracy with reduced computational overhead; at T_in = 14, ProbSparse degenerates to near-full attention (approximately 13 of 14 queries active), ensuring that the approximation introduces no meaningful accuracy loss while the generative decoder architecture provides a structural advantage over autoregressive multi-step prediction. Informer's placement at the head of all 16 baselines, combined with its superior MCO robustness documented in Section 4.5 and perfect generalisation score in Section 4.6, positions it as the most well-rounded baseline in this study.

Autoformer (74.73%, R² = 0.708) ranks 13th overall, trailing the LSTM-family and graph-based leaders by a substantial margin. Ma and Zhang (2025) identified that decomposition-based Autoformer achieves stable performance on strongly periodic signals under appropriate sequence lengths; however, Autoformer's sensitivity to the high-dimensional feature space (79 features) and its accumulation of spectral leakage across feature channels under moderate look-back windows appears to limit performance relative to simpler recurrent baselines. Autoformer's MCO-under-distribution-shift fragility (Section 4.5) further reduces its overall utility ranking despite its theoretical alignment with ridership periodicity.

ASTGCN (75.69%, R² = 0.703) performs moderately in the default configuration. Cui et al. (2023) showed that dual spatial-temporal attention substantially improves non-stationary pattern sensitivity over static graph message-passing, and ASTGCN's base-configuration performance reflects the benefit of dynamic per-sample attention weighting. However, ASTGCN's performance is highly sensitive to hyperparameter configuration, with the largest tuning gain among all attention models (+3.13 Combined%) documented in Section 4.3.

---

## 4.3 Effect of Hyperparameter Tuning

Hyperparameter tuning produces heterogeneous effects across the 16 models: some architectures benefit substantially, while others regress. The median tuning improvement is approximately +0.5 Combined%, indicating that tuning is broadly flat on test-set accuracy but imposes consistent generalisation costs, as detailed in Section 4.6.

The largest beneficiaries of tuning are ASTGCN (+3.13, to 78.82%), CNN-LSTM-Augmented (+1.74, to 77.10%), PDR-STGCN (+1.65, to 75.56%), and STGCN (+1.58, to 78.39%). For ASTGCN, the architectural change from d_model=64 to d_model=128, n_heads=4 to 8, and n_blocks=2 to 3 substantially increases the expressiveness of the dual spatial-temporal attention mechanism, consistent with Cui et al.'s (2023) finding that deeper and wider attention configurations improve sensitivity to non-stationary input patterns. For PDR-STGCN, the tuned configuration raises hidden dimensions from 160 to 256, adds a third block (n_blocks 2 → 3), widens the dynamic attention key dimension (dk 48 → 64), and reduces the temporal kernel size (kt=3 → kt=2) to maintain temporal resolution validity at the increased depth. For STGCN, the gain is achieved by doubling the hidden dimension from 128 to 256 while strengthening regularisation (dropout 0.20 → 0.25, weight decay 2e-4 → 3e-4) and retaining the base two-block, kt=3 structure — consistent with Deng (2025), who demonstrated that wider spatio-temporal graph networks extract richer relational representations when paired with adequate regularisation.

The most severe tuning regressions occur in STFGNN (−8.47, to 65.47%), CNN-LSTM sequential (−4.28, to 73.36%), and STSGCN (−3.33, to 73.65%). STFGNN's catastrophic regression is the largest loss in the entire study and represents the most striking example of tuning-induced failure: raising hidden dimensions to 128, adding a fourth fusion block, and increasing dropout to 0.35 causes a near-collapse on the test split. This suggests that the dual-graph fusion architecture is highly sensitive to capacity increases at the base training budget, with the additional parameters memorising training patterns that do not transfer to the test period. Similarly, CNN-LSTM's sequential mode degrades under tuning because the increased CNN filter width (32 → 64) amplifies the single-stream bottleneck that is already identified as a weakness at base capacity.

Notably, MTGNN (−1.54) and CNN-BiLSTM (−2.06) also regress modestly under tuning, indicating that additional capacity harms rather than helps these models. These results confirm that the test-accuracy effect of tuning is model-specific rather than universally beneficial, and that the best configuration per model must be empirically validated rather than assumed to be the most heavily parameterised variant.

---

## 4.4 Look-Back Window Sensitivity

A systematic sweep across three look-back window lengths (lb14 = 14 days, lb28 = 28 days, lb56 = 56 days) reveals a consistent finding: **lb14 is optimal for 10 of 16 models, lb28 is optimal for 6 models, and lb56 never delivers the best performance for any model.** This result has important practical implications for transit ridership forecasting system design, as it demonstrates that longer historical context does not improve seven-day-ahead forecast accuracy and in several cases causes severe degradation.

STGCN and MTGNN are among the six models best served by lb28 — STGCN achieving 77.68% at lb28 versus 76.81% at lb14, while MTGNN is the most window-stable model across the entire study (77.93 / 78.11 / 77.76 across lb14/28/56). Wu et al. (2025) attribute MTGNN's stability to its dilated inception module with multi-scale kernels ([1, 3, 5, 7]) and mix-hop graph convolution, which together prevent any single temporal scale from dominating and thereby reduce sensitivity to the total context window length. This confirms MTGNN as the recommended choice when look-back flexibility is required or when operational data constraints prevent consistent 14-day window availability.

The most severe look-back collapse is exhibited by STFGNN, which degrades from 73.94% at lb14 to 61.30% at lb28 and catastrophically to 43.25% at lb56. At lb56, STFGNN's R² falls to −0.112 (from 0.688 at lb14) — the model performs worse than naive mean-prediction, effectively losing all predictive utility. Chang et al. (2025) identified that spatio-temporal fusion graph networks are sensitive to the spectral properties of the input temporal profiles used to construct the temporal adjacency matrix (A_tem); at lb56, the 56-timestep temporal profiles used to compute A_tem become sufficiently noisy that the temporal graph provides misleading structural priors that degrade rather than improve prediction. This makes STFGNN the most look-back-fragile model in the study and effectively disqualifies it from any deployment configuration using windows longer than 14 days.

CNN-LSTM sequential mode also degrades sharply at lb28 (73.90% versus 77.64% at lb14), a sensitivity pattern not observed in other LSTM-family models. This appears attributable to the sequential CNN→LSTM coupling: with longer input sequences, the two-block CNN at base filter width (32) compresses diverse long-range temporal patterns into a representation too narrow to preserve the additional context, while the parallel and augmented modes — which either maintain direct raw-feature access or add a skip connection — do not suffer the same degradation.

---

## 4.5 MCO Structural Break Robustness

The MCO pandemic period (2020-03-18 to 2021-12-31) represents a structural break in Malaysian transit ridership — a discontinuity in which ridership fell from normal levels of 800,000–1,200,000 daily boardings to between 50,000 and 200,000 under mobility restrictions. Including this window in the training and evaluation data (`mco` configuration) tests each model's ability to generalise across a severe distributional shift. The magnitude of performance degradation when switching from the `nomco` to `mco` configuration (Δ Combined%) serves as the primary robustness indicator.

The MCO degradation spread is dramatic: from −5.71 (HMT-TSF, documented in Section 4.7) to −32.94 (STFGNN). Among baselines, Informer sustains the smallest degradation (−5.88), retaining a Combined% of 73.25 under MCO — the highest absolute MCO performance among all baselines. BiLSTM (−6.29) is the second most MCO-robust baseline. These results align with the architectural properties of ProbSparse attention: Song et al. (2024) observed that Informer's sparse attention selectively engages the most predictive query-key pairs and thereby reduces sensitivity to the distributional anomalies associated with regime shifts, a property that partially transfers across the pre- and post-MCO boundary.

Two models fail under MCO in a technically meaningful sense — CNN-LSTM sequential (R² = −0.050) and STFGNN (R² = −0.306). A negative R² indicates performance inferior to naive mean-prediction; both models produce ridership forecasts under MCO conditions that are less accurate than simply predicting the training-set mean for every day. This failure is particularly striking for CNN-LSTM, which ranks eighth in the default configuration (77.64%), demonstrating that strong nominal performance offers no guarantee of robustness to distributional shift. Topilin et al. (2025) noted that sequential CNN-LSTM architectures can memorise characteristic local temporal patterns at the expense of adaptability to regime changes, consistent with the catastrophic MCO degradation observed here (Δ −32.07).

STGCN demonstrates unexpectedly strong MCO robustness (Δ −8.27, mco R² = 0.615), performing better under the structural break than more expressive models including MTGNN (Δ −10.61) and most of the LSTM family; among baselines, only Informer (−5.88), BiLSTM (−6.29), and TPA-LSTM (−7.61) sustain smaller degradations. Deng (2025) showed that the fixed Chebyshev graph in STGCN acts as a regulariser that constrains how far the model's predictions can drift when input distributions shift; while this rigidity limits peak accuracy, it also limits catastrophic failure. This trade-off between expressiveness and distributional stability is a consistent theme across the MCO analysis: the models with the most flexible learned representations (STFGNN with dual graphs, CNN-LSTM with CNN-stage representation) degrade most severely, while models with stronger structural inductive biases (Informer with ProbSparse attention, STGCN with fixed graph) prove more robust.

---

## 4.6 Generalisation and Overfitting Analysis

Fit diagnostics reveal a systematic pattern across model families that is largely independent of test-set accuracy: **all six LSTM-family models and both simple-convolution graph models (STGCN, PDR-STGCN) are chronic overfitters, achieving zero good-fit verdicts across all 12 evaluated configurations (6 base, 6 tuned).** This finding establishes that the strong test-set Combined% values reported for BiLSTM, TPA-LSTM, and CNN-BiLSTM in Section 4.2 are achieved via memorisation-driven generalisation rather than clean model-data fit.

The gap ratio — the ratio of best validation loss to minimum training loss — provides the clearest diagnostic signal. Gap ratios for the LSTM-family span 3.3× to 13.6×, with the upper extreme reached by CNN-BiLSTM-base under MCO at lb56 (13.51×). PDR-STGCN is the most extreme memoriser in the study, reaching gap ratios of 38.61× at tuned_nomco_lb56 — a value indicating that the training loss has collapsed to near-zero while the validation loss remains at a level consistent with moderate prediction error. This extreme memorisation is consistent with the dynamic relational graph architecture, which has sufficient capacity to overfit training-set inter-feature correlations that do not generalise to the test period.

By contrast, Informer achieves a perfect 6/6 good-fit rate at base configuration — the only baseline model with a clean fit record. Its gap ratios (2.16×–2.98×) remain consistently below the 3× threshold. This result provides independent evidence, beyond test-set accuracy, for Informer's position as the most reliable and deployment-ready baseline: it is simultaneously the highest-accuracy baseline, the most MCO-robust baseline, and the only baseline with verified generalisation quality. Song et al. (2024) attribute Informer's stable generalisation partly to ProbSparse attention's implicit regularisation: by restricting the number of active queries to a logarithmic subset, ProbSparse attention prevents the model from fitting high-frequency noise in the training sequence, maintaining a broader and more generalisable representation.

MTGNN (3/6 good-fit at base, 4/6 at tuned) and STSGCN (4/6 at base, 3/6 at tuned) are the cleanest graph models after Informer. MTGNN's tightest gap ratios in the study (1.40×–1.50× in certain tuned configurations) support Wu et al.'s (2025) observation that end-to-end learned graph adjacency with mix-hop convolution provides intrinsic regularisation through the graph structure itself. The complete fit-diagnostic ranking (based on fraction of good-fit configs) is: HMT-TSF (10/10) > STSGCN, Autoformer, and MTGNN (7/12 each) > Informer (6/12 — though the only baseline with a perfect 6/6 record at base configuration; all six of its misses occur in tuned variants) > STFGNN (4/12) > ASTGCN (3/12) > all remaining models (0/12).

The tuning effect on diagnostics is uniformly negative: the good-fit rate falls from 26% (base) to 14% (tuned) across all 14 models, confirming that expanded architectural capacity raises gap ratios and post-checkpoint validation drift without delivering equivalent accuracy gains. This finding cautions against interpreting tuned test-set improvements as evidence of genuine improvement in predictive quality.

---

## 4.7 Proposed Model Performance: HMT-TSF

### 4.7.1 Main Results Across Configurations

HMT-TSF was evaluated across ten configurations covering two MCO conditions (nomco, mco) and five look-back windows (7, 14, 28, 56, 84 days). The full results are presented in Table 4.7.1.

| Configuration | Combined% | MAPE% | MAE% | RMSE% | R² | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| nomco_lb7  | 79.90 | 6.16 | 5.42 | 8.52 | 0.784 | 68,085 | 106,956 |
| **nomco_lb14** | **81.82** | **5.45** | **4.58** | **8.16** | **0.802** | **57,553** | **102,512** |
| nomco_lb28 | 80.23 | 5.94 | 5.39 | 8.44 | 0.788 | 67,735 | 106,116 |
| nomco_lb56 | 77.75 | 6.89 | 6.33 | 9.02 | 0.755 | 79,447 | 113,291 |
| nomco_lb84 | 74.17 | 8.13 | 7.59 | 10.11 | 0.695 | 94,581 | 126,031 |
| mco_lb7    | 75.25 | 7.68 | 6.85 | 10.23 | 0.712 | 84,288 | 125,937 |
| mco_lb14   | 76.11 | 7.31 | 6.64 | 9.94  | 0.729 | 81,913 | 122,596 |
| mco_lb28   | 75.94 | 7.45 | 6.58 | 10.02 | 0.722 | 81,350 | 123,840 |
| **mco_lb56**   | **76.53** | 7.24 | 6.23 | 10.00 | 0.722 | 77,313 | 124,147 |
| mco_lb84   | 73.95 | 8.33 | 6.90 | 10.82 | 0.673 | 85,604 | 134,339 |

*Table 4.7.1. HMT-TSF results across all ten configurations. Bold rows denote the best nomco and mco configurations respectively.*

The nomco look-back profile is single-peaked at lb14 (81.82%), declining symmetrically in both directions: lb7 (79.90) < lb14 (81.82) > lb28 (80.23) > lb56 (77.75) > lb84 (74.17). This confirms that the 14-day window is decisively optimal for a seven-day forecast horizon under normal operating conditions, matching the principal recommendation for baseline models in Section 4.4. The MCO look-back profile inverts this preference: under the pandemic regime, the best window is lb56 (76.53%), with lb14 second (76.11%). Longer context windows allow the model to average through the lockdown discontinuity rather than anchoring the look-back on the disrupted recent period — a property that shorter windows cannot provide.

HMT-TSF's superiority over naive persistence is substantial in every configuration. The naive persistence baseline (last-value carry-forward) achieves a Combined% of approximately 34.8 under nomco and 25.7 under mco, reflecting ridership's strong non-persistent daily dynamics driven by weekly periodicity, holidays, and external shocks. HMT-TSF's margin over naive at the headline configuration (nomco_lb14) is +46.98 Combined% and +1.763 R², quantifying the structural pattern complexity captured by the model beyond trivial temporal autocorrelation.

### 4.7.2 Comparison Against Baselines

HMT-TSF at nomco_lb14 (Combined% 81.82, R² 0.802, MAE 57,553) leads every baseline on every headline metric. The margin over the best baseline — Informer at 79.13% — is +2.69 Combined% and −11,923 MAE (a 17.2% reduction in absolute daily ridership error). The best tuned baseline, Informer tuned at 79.99%, remains 1.83 Combined% below HMT-TSF, confirming that the performance advantage is not artefactual and cannot be recovered by hyperparameter tuning of any single-architecture baseline model.

This performance advantage stems from HMT-TSF's parallel multi-modal architecture: temporal dynamics are captured by the Multi-Scale Temporal Convolutional Network (TCN) with WaveNet-style gating and stochastic depth (DropPath) regularisation; inter-feature correlations are propagated by the Feature Graph Encoder using a Pearson-correlation adjacency; and regime-specific distributional variation is addressed by the Regime Gating Embedding with K=3 learnable vectors corresponding to the pre-MCO, MCO, and post-MCO operational periods. The Feature Group Fusion module independently embeds five semantically distinct feature sources before a learned softmax gate adaptively weights their contributions, enabling the model to selectively emphasise the most predictive feature groups for each input context — a capability that no baseline model implements.

### 4.7.3 MCO Robustness

HMT-TSF is the most MCO-robust model in the entire study, sustaining a degradation of only Δ −5.71 Combined% at lb14 (nomco 81.82 → mco 76.11), marginally ahead of Informer (Δ −5.88). At longer look-back windows, HMT-TSF's MCO degradation diminishes further: Δ −4.29 at lb28, Δ −1.22 at lb56, and Δ −0.22 at lb84 — effectively MCO-invariant at lb84. This diminishing degradation with increasing context window reflects the Regime Gating Embedding's ability to blend regime-specific representations more accurately when a longer window provides richer signals for soft regime classification.

Compared to the fragile tail of baselines — STFGNN (Δ −32.94, mco R² = −0.306) and CNN-LSTM sequential (Δ −32.06, mco R² = −0.050) — HMT-TSF's MCO degradation is over five times smaller in absolute Combined% terms. Crucially, HMT-TSF also retains the highest absolute MCO accuracy (76.11%) compared to all baselines, outperforming even the second-best MCO performer, Informer (73.25%), by 2.86 Combined%. This combination of minimal degradation and highest retained performance constitutes the central empirical argument for HMT-TSF's regime-aware architectural design: the dual mechanism of Reversible Instance Normalisation (RevIN), which removes within-batch amplitude drift, and the Regime Gating Embedding, which adjusts prediction priors for each operational regime, collectively enable the model to transfer across the extreme distributional discontinuity introduced by the MCO period.

### 4.7.4 Walk-Forward Temporal Validation

To verify that the headline test-set performance is not attributable to a single favourable temporal split, three sequential non-overlapping out-of-sample validation blocks were evaluated. Results confirm that performance is temporally stable: no block falls below 68.33 Combined% or 0.593 R², and the model produces strong predictions across all three temporal segments. The headline nomco_lb14 configuration achieves block-level Combined% of 85.18 (Block 1), 76.99 (Block 2), and 83.72 (Block 3), with Block 2 consistently representing the weakest segment — likely corresponding to a seasonal transition or behavioural shift in the middle of the evaluation period.

The nomco_lb56 configuration is the most temporally stable of all evaluated settings, achieving a block spread of only 3.4 Combined% (78.38 / 75.72 / 79.16), compared to nomco_lb28's maximum single-block performance of 86.63 at the cost of the widest spread (14.5). This stability-accuracy trade-off provides a practical decision criterion: when temporal consistency across evaluation periods is more important than peak accuracy — for example, in operational monitoring applications where forecast quality must be guaranteed across all seasons — nomco_lb56 is the preferred configuration, even at a 4.07 Combined% penalty relative to the headline lb14 result.

### 4.7.5 Generalisation and Fit Diagnosis

HMT-TSF achieves `good_fit` verdicts on all ten evaluated configurations — the only model family in the study with a perfect fit record. Gap ratios range from 1.80× to 2.84×, uniformly below the 3× memorisation threshold, and validation drift remains within 12.3% across all configurations. This clean fit profile contrasts sharply with the chronic overfitting observed in all LSTM-family models (gap ratios 3.3×–13.6×) and the extreme memorisation of PDR-STGCN (up to 38.61×). Crucially, all five MCO-included configurations also achieve `good_fit`, establishing HMT-TSF as the only architecture in the study that maintains clean generalisation under the structural break — a direct reflection of RevIN and regime gating jointly mitigating the distributional shift that causes gap ratios to escalate in every baseline family.

---

## 4.8 SHAP Feature Importance Analysis

SHAP (SHapley Additive exPlanations) values were computed for the trained nomco_lb14 model using `shap.GradientExplainer` with 100 test samples as the background set. Absolute SHAP values were averaged over samples and time steps to produce a single 79-element importance vector, one score per input feature.

### 4.8.1 Top Feature Drivers

The ten highest-importance features reveal that the model's learned drivers are closely aligned with established transit demand theory. Historical ridership on individual service lines dominates the importance ranking: `rail_komuter` (KTM Komuter, feat_12) is ranked first and appears in the top-15 features of all ten evaluated configurations (100% consistency), confirming strong autocorrelation as the dominant predictive signal. `rail_mrt_kajang` (MRT Kajang, feat_4) ranks second and `rail_lrt_ampang` (LRT Ampang, feat_3) third, reflecting the high-frequency daily co-movement patterns of the three major Klang Valley rail corridors.

Fuel price features emerge as the second-most important group: `fp_chg_ron95_budi95` (RON95 Budi95 subsidy price change, feat_42) ranks fourth and appears in the top-15 in 80% of configurations. `fp_lv_diesel_pct_chg` (diesel percentage change, feat_37) and `fp_chg_diesel` (diesel absolute price change, feat_40) also appear in the top 10. These rankings quantify the mode-substitution effect: when Malaysian fuel prices increase, private vehicle operating costs rise and transit ridership correspondingly increases, a demand-shaping mechanism that the model has learned to detect directly from historical co-movements.

`day_of_week` (feat_20) is the only temporal feature in the top 10 (rank 7, consistent in 60% of configurations), reflecting weekly periodicity as the dominant temporal signal. Selangor rainfall (`rainfall_mm__MY10`, feat_53) ranks eighth, confirming that wet-weather conditions in the Klang Valley — the geographic concentration of Malaysian transit ridership — measurably increase rail uptake by reducing walkability and cycling access to transit stations.

### 4.8.2 Cross-Configuration Feature Stability

Aggregating SHAP importance across all ten configurations reveals a stable set of universally important features and a set of zero-importance features that are consistent regardless of look-back window or MCO condition. `rail_komuter` is the only feature that appears in the top-15 of all ten configurations, establishing it as the single most reliable predictor across all operational regimes and window lengths. `fp_chg_ron95_budi95` (80% consistency) and `fp_lv_ron97_pct_chg` (70%) are the most consistent external feature drivers, while `rainfall_mm__MY03` (Kelantan, 70%) and `rainfall_mm__MY02` (Kedah, 60%) show above-expected rainfall importance — both states sit on the northeast monsoon corridor (November–January), and the model appears to have detected that east-coast precipitation events propagate to nationwide transit demand patterns via weather-induced travel behavioural shifts.

A lookback-dependent feature importance shift is also observed: at lb7, the very short context forces the model to rely more heavily on external fuel price levels and regional rainfall features, as there is insufficient temporal depth for autocorrelation to dominate. At lb14–28 (the optimal performance regime), a balanced feature profile emerges with ridership lines, rainfall, and fuel change features contributing concurrently. At lb56–84, `rail_komuter` dominates overwhelmingly, as the long autocorrelation window provides sufficient historical ridership depth for the model to anchor forecasts primarily on the recent ridership trajectory.

---

## 4.9 Feature-Reduced Variant: HMT-TSF-FR

Cross-run SHAP aggregation identified 23 features that contribute zero importance across all ten configurations (universal zeros) and three additional near-zero features (zero in 8 of 10 runs), totalling 26 removable features. The 23 universal zeros include the entire Static feature group (all 17 features: population, GTFS, OSM POI, and GADM descriptors), which contribute no discriminative signal because time-invariant structural capacity is already implicitly encoded in historical ridership levels: high-capacity corridors persistently exhibit high ridership throughout the study period, making static spatial descriptors redundant given sufficient look-back context. Three temporal features (month, is_weekend, dow_sin) and three fuel-level features (fp_lv_ron97, fp_lv_diesel, fp_lv_diesel_eastmsia) are also universally inactive, likely because their predictive information is more efficiently captured by the corresponding cyclical encodings and fuel-change-rate features retained in the reduced feature set.

HMT-TSF-FR — trained on the reduced 53-feature input — achieves Combined% of 81.77 at nomco_lb14, a difference of only −0.05 from the full 81.82 baseline, confirming that the 26 removed features carry negligible predictive content. The input tensor is reduced from (B, T_in, 79) to (B, T_in, 53), the graph adjacency is reduced from 79×79 to 53×53, and GradientExplainer memory requirements decrease by approximately 33% — practical advantages for deployment or repeated SHAP evaluation.

At longer nomco look-back windows, HMT-TSF-FR outperforms the standard model: +0.88 at nomco_lb28 (81.11 vs 80.23), +0.69 at nomco_lb56 (78.44 vs 77.75), and +1.34 at nomco_lb84 (75.51 vs 74.17). Removing zero-SHAP features appears to provide a noise-regularisation effect under long autocorrelation windows: with fewer irrelevant input dimensions, the GCN message-passing and TCN temporal encoding focus on the features that actually carry predictive signal, reducing the risk that zero-information features introduce confounding gradients at longer context lengths.

One notable exception is `mco_lb84`, where HMT-TSF-FR degrades sharply (−4.50 Combined%, R² 0.530 versus 0.673 for the standard model). The mco_lb84 regime — the longest context window under the most severe distributional shift — appears to require some of the near-zero features (likely temporal or fuel-level signals) to navigate the lockdown-spanning window; their removal causes an accuracy collapse not observed in any other configuration. The standard model should therefore be retained for `mco` conditions when long look-back windows are required. All HMT-TSF-FR configurations remain `good_fit` with gap ratios within 1.74–2.74×, confirming that feature reduction does not introduce overfitting.

---

## 4.10 Summary of Findings

The empirical findings of this study can be summarised across five thematic conclusions.

**First, HMT-TSF outperforms all 16 baselines in default configuration.** With a Combined% of 81.82, R² of 0.802, and MAE of 57,553 at nomco_lb14, HMT-TSF leads the next-best model — Informer (79.13%) — by 2.69 Combined% and reduces daily ridership MAE by 17.2%. Even against the best tuned baseline (Informer tuned at 79.99%), HMT-TSF maintains a 1.83 Combined% advantage. The performance gain is attributable to three simultaneous architectural contributions absent from any single baseline: multi-scale temporal encoding via the causal dilated TCN, feature-correlation message passing via the GCN, and distributional shift mitigation via RevIN and regime gating.

**Second, HMT-TSF is the most MCO-robust and most generalisable model.** Its MCO degradation (Δ −5.71) is the smallest among all 16 models, its MCO retained accuracy (76.11%) is the highest among all models, and its 10/10 good-fit record under both normal and MCO conditions is unique in the study. These three properties collectively establish that HMT-TSF's accuracy advantage is genuine (not memorisation-driven) and stable across regime transitions.

**Third, the LSTM-family occupies most of the top-8 rankings but exhibits systematic overfitting.** BiLSTM and TPA-LSTM are the strongest baselines by R² (0.776), but both achieve zero good-fit verdicts across all 12 evaluated configurations, with gap ratios confirming that their test-set accuracy is partly memorisation-driven. Informer is the sole baseline with a clean fit record (6/6 good-fit at base) and the most practically deployable baseline for production applications.

**Fourth, lb14 is the unambiguously optimal look-back window for a seven-day forecast horizon.** Longer windows do not improve accuracy and trigger severe collapses in architecturally fragile models (STFGNN at lb56, CNN-LSTM at lb28). The sole exception is MCO-regime operation, where longer context helps both HMT-TSF and graph-based models absorb the lockdown discontinuity.

**Fifth, SHAP analysis confirms that the model's learned feature importance hierarchy is interpretable and domain-consistent.** Historical ridership autocorrelation, fuel price change signals, and Klang Valley weather conditions are the primary predictive drivers, while the entire static feature group is confirmed as redundant given sufficient temporal context. The 53-feature HMT-TSF-FR variant matches the standard model's headline performance within 0.05 Combined% and is recommended for `nomco` operational deployments.

---

## References

Alajmi, M. S., & Almutairi, S. M. (2025). Intelligent traffic congestion forecasting using BiLSTM and adaptive secretary bird optimizer for sustainable urban transportation. *Scientific Reports*, *15*. https://doi.org/10.1038/s41598-025-02933-9

Chang, J., Yin, J., Hao, Y., & Gao, C. (2025). STFDSGCN: Spatio-temporal fusion graph neural network based on dynamic sparse graph convolution GRU for traffic flow forecast. *Sensors*, *25*(11), 3446. https://doi.org/10.3390/s25113446

Chen, L., Ren, Q., Zeng, J., Zou, F., Luo, S., Tian, J., & Xing, Y. (2023). CSFPre: Expressway key sections based on CEEMDAN-STSGCN-FCM during the holidays for traffic flow prediction. *PLOS ONE*, *18*(4), e0283898. https://doi.org/10.1371/journal.pone.0283898

Chen, W., Yang, Z., Xu, G., & Sun, Y. (2022). Short-term traffic flow prediction based on CNN-BiLSTM with multicomponent information. *Applied Sciences*, *12*(17), 8714. https://doi.org/10.3390/app12178714

Cui, H., Si, B., Chi, D., Li, Y., Li, G., & Chen, Y. (2025). Short-term passenger flow prediction for urban rail systems: A deep learning approach utilizing multi-source big data. *PLOS ONE*, *20*(1), e0333094. https://doi.org/10.1371/journal.pone.0333094

Cui, Z., Zhang, J., Noh, G., & Park, H. J. (2023). ADSTGCN: A dynamic adaptive deeper spatio-temporal graph convolutional network for multi-step traffic forecasting. *Sensors*, *23*(15), 6950. https://doi.org/10.3390/s23156950

Deng, H. (2025). Traffic-forecasting model with spatio-temporal kernel. *Electronics*, *14*(7), 1410. https://doi.org/10.3390/electronics14071410

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In *International Conference on Learning Representations*. https://openreview.net/forum?id=Bkg6RiCqY7

Ma, X., & Zhang, H. (2025). Time series forecasting method based on multi-scale feature fusion and Autoformer. *Applied Sciences*, *15*(7), 3768. https://doi.org/10.3390/app15073768

PDR-STGCN: An enhanced STGCN with multi-scale periodic fusion and a dynamic relational graph for traffic forecasting. (2026). *Systems*, *14*(1), 102. https://doi.org/10.3390/systems14010102

Song, Y., Luo, R., Zhou, T., Zhou, C., & Su, R. (2024). Graph attention Informer for long-term traffic flow prediction under the impact of sports events. *Sensors*, *24*(15), 4796. https://doi.org/10.3390/s24154796

Strigula, M. (2026). Beyond traditional forecasting methods: Evaluating LSTM performance on diverse time series. *Mathematics*, *14*(5), 838. https://doi.org/10.3390/math14050838

Topilin, I., Jiang, J., Feofilova, A., & Beskopylny, N. (2025). Traffic flow prediction via a hybrid CPO-CNN-LSTM-attention architecture. *Smart Cities*, *8*(5), 148. https://doi.org/10.3390/smartcities8050148

Wei, L., Guo, D., Chen, Z., Yang, J., & Feng, T. (2023). Forecasting short-term passenger flow of subway stations based on the temporal pattern attention mechanism and the long short-term memory network. *ISPRS International Journal of Geo-Information*, *12*(1), 25. https://doi.org/10.3390/ijgi12010025

Wu, Z., Liu, X., & Zhang, X. (2025). Multi dynamic temporal representation graph convolutional network for traffic flow prediction. *Scientific Reports*, *15*, 16734. https://doi.org/10.1038/s41598-025-01157-1
