# Chapter 4: Findings and Results

---

## 4.1 Overview of Experimental Evaluation

This chapter reports the empirical findings from a systematic comparative evaluation of 15 deep-learning architectures — 14 baseline models, evaluated as 16 ranked variants because CNN-LSTM contributes three independent fusion modes, plus the proposed hybrid HMT-TSF — for seven-day Malaysian transit ridership forecasting. The evaluation is structured across four analytical dimensions: (1) default-configuration performance under normal operating conditions (MCO-excluded, 14-day look-back); (2) the effect of hyperparameter tuning on both accuracy and generalisation; (3) look-back window sensitivity across three window lengths (14, 28, and 56 days); and (4) structural-break robustness under the Movement Control Order (MCO) pandemic period (2020-03-18 to 2021-12-31). A dedicated fit-diagnostics analysis examines the generalisation quality of all models independently of their test-set accuracy.

The 14 baseline models span three architectural series — spatio-temporal recurrent (LSTM-family), graph-based, and attention-based — and are complemented by the proposed hybrid model, HMT-TSF (Hybrid Multi-scale Temporal Spatio-Feature Forecaster). Each model is evaluated using the composite Combined% metric, which aggregates three complementary error perspectives into a single scalar: `Combined% = max(0, 100 − MAPE% − MAE% − RMSE%)`, where all percentage terms are normalised by mean ridership demand, making them directly addable. Supplementary metrics (R², absolute MAE, RMSE) are reported alongside Combined% to provide a complete performance profile. The headline configuration is `base_nomco_lb14` (base architecture, MCO-excluded dataset, 14-day look-back), which serves as the project default and primary comparison benchmark.

All models share a standardised training protocol: AdamW optimiser (Loshchilov & Hutter, 2019) with decoupled weight decay, HuberLoss objective, a five-epoch linear learning rate warm-up followed by ReduceLROnPlateau scheduling, and a 70/15/15 chronological train/validation/test split. Fit diagnostics are derived from three signals recorded per training run: validation drift above the best checkpoint (threshold 25%), the gap ratio between best validation loss and minimum training loss (threshold 3×), and train/validation divergence in the final training epochs.

---

## 4.2 Baseline Model Performance — Default Configuration (`base_nomco_lb14`)

Table 4.2.1 presents the complete default-configuration ranking of all 16 baseline variants. The subsections that follow analyse the results by architectural series.


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

The MCO degradation spread is dramatic: from −3.79 (HMT-TSF, documented in Section 4.7) to −32.94 (STFGNN). Among baselines, Informer sustains the smallest degradation (−5.88), retaining a Combined% of 73.25 under MCO — the highest absolute MCO performance among all baselines. BiLSTM (−6.29) is the second most MCO-robust baseline. These results align with the architectural properties of ProbSparse attention: Song et al. (2024) observed that Informer's sparse attention selectively engages the most predictive query-key pairs and thereby reduces sensitivity to the distributional anomalies associated with regime shifts, a property that partially transfers across the pre- and post-MCO boundary.

Two models fail under MCO in a technically meaningful sense — CNN-LSTM sequential (R² = −0.050) and STFGNN (R² = −0.306). A negative R² indicates performance inferior to naive mean-prediction; both models produce ridership forecasts under MCO conditions that are less accurate than simply predicting the training-set mean for every day. This failure is particularly striking for CNN-LSTM, which ranks eighth in the default configuration (77.64%), demonstrating that strong nominal performance offers no guarantee of robustness to distributional shift. Topilin et al. (2025) noted that sequential CNN-LSTM architectures can memorise characteristic local temporal patterns at the expense of adaptability to regime changes, consistent with the catastrophic MCO degradation observed here (Δ −32.07).

STGCN demonstrates unexpectedly strong MCO robustness (Δ −8.27, mco R² = 0.615), performing better under the structural break than more expressive models including MTGNN (Δ −10.61) and most of the LSTM family; among baselines, only Informer (−5.88), BiLSTM (−6.29), and TPA-LSTM (−7.61) sustain smaller degradations. Deng (2025) showed that the fixed Chebyshev graph in STGCN acts as a regulariser that constrains how far the model's predictions can drift when input distributions shift; while this rigidity limits peak accuracy, it also limits catastrophic failure. This trade-off between expressiveness and distributional stability is a consistent theme across the MCO analysis: the models with the most flexible learned representations (STFGNN with dual graphs, CNN-LSTM with CNN-stage representation) degrade most severely, while models with stronger structural inductive biases (Informer with ProbSparse attention, STGCN with fixed graph) prove more robust.

---

## 4.6 Generalisation and Overfitting Analysis

Fit diagnostics reveal a systematic pattern across model families that is largely independent of test-set accuracy: **all six LSTM-family models and both simple-convolution graph models (STGCN, PDR-STGCN) are chronic overfitters, achieving zero good-fit verdicts across all 12 evaluated configurations (6 base, 6 tuned).** This finding establishes that the strong test-set Combined% values reported for BiLSTM, TPA-LSTM, and CNN-BiLSTM in Section 4.2 are achieved via memorisation-driven generalisation rather than clean model-data fit.

The gap ratio — the ratio of best validation loss to minimum training loss — provides the clearest diagnostic signal. Gap ratios for the LSTM-family span 3.3× to 13.6×, with the upper extreme reached by CNN-BiLSTM-base under MCO at lb56 (13.51×). PDR-STGCN is the most extreme memoriser in the study, reaching gap ratios of 38.61× at tuned_nomco_lb56 — a value indicating that the training loss has collapsed to near-zero while the validation loss remains at a level consistent with moderate prediction error. This extreme memorisation is consistent with the dynamic relational graph architecture, which has sufficient capacity to overfit training-set inter-feature correlations that do not generalise to the test period.

By contrast, Informer achieves a perfect 6/6 good-fit rate at base configuration — the only baseline model with a clean fit record. Its gap ratios (2.16×–2.98×) remain consistently below the 3× threshold. This result provides independent evidence, beyond test-set accuracy, for Informer's position as the most reliable and deployment-ready baseline: it is simultaneously the highest-accuracy baseline, the most MCO-robust baseline, and the only baseline with verified generalisation quality. Song et al. (2024) attribute Informer's stable generalisation partly to ProbSparse attention's implicit regularisation: by restricting the number of active queries to a logarithmic subset, ProbSparse attention prevents the model from fitting high-frequency noise in the training sequence, maintaining a broader and more generalisable representation.

MTGNN (3/6 good-fit at base, 4/6 at tuned) and STSGCN (4/6 at base, 3/6 at tuned) are the cleanest graph models after Informer. MTGNN's tightest gap ratios in the study (1.40×–1.50× in certain tuned configurations) support Wu et al.'s (2025) observation that end-to-end learned graph adjacency with mix-hop convolution provides intrinsic regularisation through the graph structure itself. The complete fit-diagnostic ranking (based on fraction of good-fit configs) is: HMT-TSF (19/20 across both variants — 9/10 full, 10/10 feature-reduced) > STSGCN, Autoformer, and MTGNN (7/12 each) > Informer (6/12 — though the only baseline with a perfect 6/6 record at base configuration; all six of its misses occur in tuned variants) > STFGNN (4/12) > ASTGCN (3/12) > all remaining models (0/12).

The tuning effect on diagnostics is uniformly negative: the good-fit rate falls from 26% (base) to 14% (tuned) across all 14 models, confirming that expanded architectural capacity raises gap ratios and post-checkpoint validation drift without delivering equivalent accuracy gains. This finding cautions against interpreting tuned test-set improvements as evidence of genuine improvement in predictive quality.

---

## 4.7 Proposed Model Performance: HMT-TSF

### 4.7.1 Main Results Across Configurations

HMT-TSF was evaluated across ten configurations covering two MCO conditions (nomco, mco) and five look-back windows (7, 14, 28, 56, 84 days). The full results are presented in Table 4.7.1.


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


*Table 4.7.1. HMT-TSF (full, F=79) results across all ten configurations. Bold rows denote the best nomco and mco configurations respectively. The feature-reduced variant exceeds these values at every nomco lookback ≥14 (Section 4.9).*

The nomco look-back profile is single-peaked at lb14 (85.78%), declining in both directions: lb7 (84.88) < lb14 (85.78) > lb28 (84.23) > lb56 (82.53) > lb84 (77.68). This confirms that the 14-day window is decisively optimal for a seven-day forecast horizon under normal operating conditions, matching the principal recommendation for baseline models in Section 4.4. With the corrected known-future calendar conditioning, the lb14 peak now persists under MCO as well (81.99%), where the pre-fix runs had favoured lb56: short windows no longer require long context to absorb the lockdown discontinuity once the model receives explicit future calendar structure. The sole lookback at which the MCO condition outperforms nomco is lb84 (+0.77), reflecting nomco_lb84's overfit verdict (Section 4.7.5) rather than any virtue of the 84-day window.

HMT-TSF's superiority over naive persistence is substantial in every configuration. The naive persistence baseline (last-value carry-forward) achieves a Combined% of approximately 34.8 under nomco and 25.7 under mco, reflecting ridership's strong non-persistent daily dynamics driven by weekly periodicity, holidays, and external shocks. HMT-TSF's margin over naive at the headline configuration (nomco_lb14) is +50.94 Combined% and +1.858 R², quantifying the structural pattern complexity captured by the model beyond trivial temporal autocorrelation.

### 4.7.2 Comparison Against Baselines

HMT-TSF at nomco_lb14 (Combined% 85.78, R² 0.897, MAE 49,897) leads every baseline on every headline metric. The margin over the best baseline — Informer at 79.13% — is +6.65 Combined% and −19,579 MAE (a 28.2% reduction in absolute daily ridership error). The best tuned baseline, Informer tuned at 79.99%, remains 5.79 Combined% below HMT-TSF — and 6.60 below the feature-reduced variant (86.59%, Section 4.9) — confirming that the performance advantage is not artefactual and cannot be recovered by hyperparameter tuning of any single-architecture baseline model. Part of this margin is attributable to an input advantage rather than architecture alone: HMT-TSF is the only model in the study that receives the known-future calendar tensor (`X_future`), so the comparison should be read as architecture-plus-conditioning versus architecture-only baselines (see `REVISION.md` on comparison fairness).

This performance advantage stems from HMT-TSF's parallel multi-modal architecture: temporal dynamics are captured by the Multi-Scale Temporal Convolutional Network (TCN) with WaveNet-style gating and stochastic depth (DropPath) regularisation; inter-feature correlations are propagated by the Feature Graph Encoder using a Pearson-correlation adjacency; and regime-specific distributional variation is addressed by the Regime Gating Embedding with K=3 learnable vectors corresponding to the pre-MCO, MCO, and post-MCO operational periods. The Feature Group Fusion module independently embeds five semantically distinct feature sources before a learned softmax gate adaptively weights their contributions, enabling the model to selectively emphasise the most predictive feature groups for each input context — a capability that no baseline model implements.

### 4.7.3 MCO Robustness

HMT-TSF is the most MCO-robust model in the entire study at the headline lookback, sustaining a degradation of only Δ −3.79 Combined% at lb14 (nomco 85.78 → mco 81.99), ahead of the best baseline (Informer tuned, Δ −4.64). Degradation shrinks further at lb56 (Δ −3.06) and reverses sign at lb84 (Δ +0.77, an artefact of nomco_lb84's overfit rather than genuine MCO advantage).

Compared to the fragile tail of baselines — STFGNN (Δ −32.94, mco R² = −0.306) and CNN-LSTM sequential (Δ −32.06, mco R² = −0.050) — HMT-TSF's MCO degradation is roughly an order of magnitude smaller. Crucially, HMT-TSF also retains the highest absolute MCO accuracy in the study (81.99%) — the only result above the 75% target with meaningful headroom — outperforming the best MCO baseline (Informer tuned, 75.79%) by 6.20 Combined%. This combination of minimal degradation and highest retained performance constitutes the central empirical argument for HMT-TSF's regime-aware architectural design: the dual mechanism of Reversible Instance Normalisation (RevIN), which removes within-batch amplitude drift, and the Regime Gating Embedding, which adjusts prediction priors for each operational regime, collectively enable the model to transfer across the extreme distributional discontinuity introduced by the MCO period. Notably, the feature-reduced variant is markedly less MCO-robust (Δ −7.85 at lb14; Section 4.9), indicating that part of this robustness resides in the redundancy of the full feature set rather than the architecture alone.

### 4.7.4 Walk-Forward Temporal Validation

To verify that the headline test-set performance is not attributable to a single favourable temporal split, three sequential non-overlapping out-of-sample validation blocks were evaluated. Results confirm that performance is temporally stable: no block falls below 71.99 Combined% or 0.707 R², and the model produces strong predictions across all three temporal segments. The headline nomco_lb14 configuration achieves block-level Combined% of 90.62 (Block 1), 82.14 (Block 2), and 85.12 (Block 3), with Block 2 the weakest segment under nomco at short lookbacks — likely corresponding to a seasonal transition or behavioural shift in the middle of the evaluation period. Under mco the pattern shifts: Block 1 (which absorbs the lockdown-adjacent segment) is the weakest and Block 2 the strongest.

The mco_lb84 configuration is the most temporally stable of all evaluated settings (block spread 3.7 Combined%), followed by mco_lb56 (6.2); among nomco configurations, lb7 is the most stable (6.6) and the headline lb14 spreads 8.5. This stability–accuracy trade-off provides a practical decision criterion: when temporal consistency across evaluation periods matters more than peak accuracy, longer-context MCO-trained configurations are preferred; for peak accuracy, nomco_lb14 dominates.

### 4.7.5 Generalisation and Fit Diagnosis

HMT-TSF achieves `good_fit` verdicts on nine of ten evaluated configurations, and the feature-reduced variant on all ten — a 19/20 record that no baseline family approaches. Gap ratios range from 1.57× to 2.64×, uniformly below the 3× memorisation threshold; the headline nomco_lb14 run posts the cleanest profile in the study (gap 1.57×, validation drift 0.61%). The single exception is nomco_lb84, diagnosed `overfit` on validation drift (+36.9% above its early best at epoch 18 of 48) — the clearest evidence in the study that an 84-day window over a 7-day horizon supplies more context than the task can use. This otherwise clean fit profile contrasts sharply with the chronic overfitting observed in all LSTM-family models (gap ratios 3.3×–13.6×) and the extreme memorisation of PDR-STGCN (up to 38.61×). Crucially, all five MCO-included configurations achieve `good_fit` in both variants, establishing HMT-TSF as the only architecture in the study that maintains clean generalisation under the structural break — a direct reflection of RevIN and regime gating jointly mitigating the distributional shift that causes gap ratios to escalate in every baseline family.

---

## 4.8 SHAP Feature Importance Analysis

SHAP (SHapley Additive exPlanations) values were computed for the trained nomco_lb14 model using `shap.GradientExplainer` with 100 test samples as the background set. Absolute SHAP values were averaged over samples, time steps, and forecast horizons to produce a single 79-element importance vector, one score per input feature. Feature names are resolved from the authoritative aligned column order (`feature_metadata.json → column_order`); the full cross-run table is exported to `src/outputs/shap_crossrun_summary.csv`.

### 4.8.1 Top Feature Drivers

The highest-importance features reveal that the model's learned drivers are closely aligned with established transit demand theory, with two complementary views. By raw attribution magnitude on the headline run, the annual-cycle encodings dominate: `month_cos` (feat_46), `month_sin` (feat_45), and `month` (feat_41) carry the three largest mean |SHAP| values, indicating that the model anchors its seven-day forecasts on seasonal position — school terms, festive seasons, and the northeast monsoon all phase-lock to the calendar. The strongest non-seasonal driver is `total_ridership` (feat_12) — the system-wide aggregate that is itself the forecast target — confirming target autocorrelation as the dominant data-driven signal; it is also the only feature appearing in the top-15 of all ten evaluated configurations (100% consistency).

Holiday structure forms the second calendar tier: `days_to_next_public_hol` (feat_36) ranks in the headline top-10 and is the second-most consistent feature across configurations (80%), confirming that holiday anticipation — the pre-holiday dips and post-holiday return surges identified in the EDA — is a first-order demand driver. Among external covariates, the two *unfrozen* fuel grades lead: `fp_lv_diesel` (feat_20, 60% consistency) and `fp_lv_ron97` (feat_19, 50%) carry measurable but secondary mode-substitution signal, while every administered RON95 series is SHAP-inactive (Section 4.8.2). Regional rainfall (Perlis, Terengganu, and Penang — feat_55/57/53) completes the headline top-10, consistent with monsoon-corridor weather measurably shifting mode choice.

### 4.8.2 Cross-Configuration Feature Stability

Aggregating SHAP importance across all ten configurations reveals a stable set of universally important features and a set of zero-importance features that are consistent regardless of look-back window or MCO condition. `total_ridership` (10/10 configurations) and `days_to_next_public_hol` (8/10) head the consistency ranking, followed by `month_sin` and `rail_lrt_kj` (7/10 each), then `month_cos`, `month`, and `fp_lv_diesel` (6/10 each) — a profile in which calendar structure and aggregate autocorrelation jointly dominate, with line-level ridership and unfrozen fuel prices as secondary drivers.

At the other extreme, 23 features are zero in all ten runs (the entire 17-feature static group plus six administered or East Malaysia fuel-price columns) and three are zero in eight of ten runs (the RON95 level and its change variants — RON95 was frozen at RM2.05 throughout the post-MCO window). These 26 features constitute the removal set for the feature-reduced variant (Section 4.9); their inactivity in the regenerated, correctly-labelled runs independently re-validates the original reduction choice.

A lookback-dependent feature importance shift is clearly observed in the per-run rankings: at lb7–lb28, calendar features dominate by magnitude (holiday flags and month encodings occupy the top ranks in every short-window configuration, both nomco and mco), as the short context provides little autocorrelation depth. At lb56–84, the profile inverts and `total_ridership` (with the major Klang Valley rail lines) takes the top ranks, as the long autocorrelation window lets the model anchor forecasts primarily on the recent ridership trajectory.

---

## 4.9 Feature-Reduced Variant: HMT-TSF-FR

Cross-run SHAP aggregation identified 23 features that contribute zero importance across all ten configurations (universal zeros) and three additional near-zero features (zero in 8 of 10 runs), totalling 26 removable features: the entire Static group (all 17 features: population, GTFS, OSM POI, and GADM descriptors) plus nine fuel-price columns (`fp_lv_ron95`, `fp_lv_diesel_eastmsia`, `fp_lv_ron95_budi95`, `fp_lv_ron95_skps`, `fp_lv_ron95_pct_chg`, `fp_chg_ron95`, `fp_chg_diesel_eastmsia`, `fp_chg_ron95_budi95`, `fp_chg_ron95_skps`). The static group contributes no discriminative signal because time-invariant structural capacity is already implicitly encoded in historical ridership levels: high-capacity corridors persistently exhibit high ridership throughout the study period, making static spatial descriptors redundant given sufficient look-back context. The nine inactive fuel columns are administered prices that were frozen or near-constant across the post-MCO training window (RON95 was held at RM2.05 throughout), or East Malaysia variants irrelevant to the Peninsular ridership base — near-zero day-to-day variance leaves them nothing to contribute.

HMT-TSF-FR — trained on the reduced 53-feature input — achieves Combined% of 86.59 at nomco_lb14 (R² 0.906, MAE 46,323), **exceeding** the full model (85.78) by +0.81 and setting the study-wide headline result. The input tensor is reduced from (B, T_in, 79) to (B, T_in, 53), the graph adjacency is reduced from 79×79 to 53×53, and GradientExplainer memory requirements decrease by approximately 33% — so the best-performing configuration is also the cheapest to train and explain.

The FR advantage grows with the nomco look-back window: +0.81 at lb14, +1.84 at lb28 (86.07 vs 84.23), +1.52 at lb56 (84.04 vs 82.53), and +4.26 at lb84 (81.94 vs 77.68). Removing zero-SHAP features provides a noise-regularisation effect under long autocorrelation windows: with fewer irrelevant input dimensions, the GCN message-passing and TCN temporal encoding focus on the features that actually carry predictive signal. The effect is strong enough to repair the full model's only diagnostic failure — nomco_lb84 flips from `overfit` to a comfortable `good_fit` under feature reduction.

The trade-off emerges under the structural break: HMT-TSF-FR loses to the full model in **every** mco configuration, with degradation deepening as the window grows (−0.85 at lb7, −3.25 at lb14, −3.44 at lb28, −3.57 at lb56, −5.22 at lb84, where FR's R² falls to 0.680 versus 0.791). The redundant near-constant fuel-price columns evidently provide a stabilising anchor under the lockdown-spanning distributional shift; their removal costs MCO robustness (FR's lb14 MCO degradation is −7.85 versus the full model's −3.79). The full model should therefore be retained for `mco`/shock-prone conditions. All HMT-TSF-FR configurations remain `good_fit` with gap ratios within 1.66–2.72×, confirming that feature reduction does not introduce overfitting.

---

## 4.10 Summary of Findings

The empirical findings of this study can be summarised across five thematic conclusions.

**First, HMT-TSF outperforms all 16 baselines in default configuration.** With a Combined% of 85.78, R² of 0.897, and MAE of 49,897 at nomco_lb14 — rising to 86.59 / 0.906 / 46,323 in the feature-reduced variant — HMT-TSF leads the next-best model, Informer (79.13%), by 6.65 Combined% and reduces daily ridership MAE by 28.2% (33.3% for FR). Even against the best tuned baseline (Informer tuned at 79.99%), HMT-TSF maintains a 5.79 Combined% advantage (6.60 for FR). The performance gain is attributable to three simultaneous architectural contributions absent from any single baseline — multi-scale temporal encoding via the causal dilated TCN, feature-correlation message passing via the GCN, and distributional shift mitigation via RevIN and regime gating — plus one input advantage unique to HMT-TSF: known-future calendar conditioning via `X_future` (see the comparison-fairness note in Section 4.7.2).

**Second, HMT-TSF is the most MCO-robust and most generalisable model.** Its MCO degradation at the headline lookback (Δ −3.79) is the smallest among all 16 models, its MCO retained accuracy (81.99%) is the highest in the study by a 6.2-point margin, and its 19/20 good-fit record across both variants and both regimes is unmatched. These properties collectively establish that HMT-TSF's accuracy advantage is genuine (not memorisation-driven) and stable across regime transitions.

**Third, the LSTM-family occupies most of the top-8 rankings but exhibits systematic overfitting.** BiLSTM and TPA-LSTM are the strongest baselines by R² (0.776), but both achieve zero good-fit verdicts across all 12 evaluated configurations, with gap ratios confirming that their test-set accuracy is partly memorisation-driven. Informer is the sole baseline with a clean fit record (6/6 good-fit at base) and the most practically deployable baseline for production applications.

**Fourth, lb14 is the unambiguously optimal look-back window for a seven-day forecast horizon.** Longer windows do not improve accuracy and trigger severe collapses in architecturally fragile models (STFGNN at lb56, CNN-LSTM at lb28); the longest window even produces HMT-TSF's only overfit verdict (nomco_lb84). With corrected future-calendar conditioning, lb14 is optimal for HMT-TSF under both regimes — the earlier observation that MCO operation favoured longer context does not survive the pipeline fix, though it persists for the graph-based baselines.

**Fifth, SHAP analysis confirms that the model's learned feature importance hierarchy is interpretable and domain-consistent.** Seasonal calendar position (month encodings) and holiday anticipation dominate attribution magnitude, aggregate ridership autocorrelation is the most consistent driver across configurations, and the unfrozen fuel grades (diesel, RON97) plus monsoon-corridor rainfall contribute secondary signal — while the entire static feature group and the frozen administered RON95 series are confirmed as redundant. The 53-feature HMT-TSF-FR variant exceeds the standard model at every nomco lookback ≥14 (setting the 86.59% study headline) and is recommended for normal operations; the full model's feature redundancy buys superior MCO robustness and is retained for shock-prone regimes.

---

## References

Alajmi, M. S., & Almutairi, S. M. (2025). Intelligent traffic congestion forecasting using BiLSTM and adaptive secretary bird optimizer for sustainable urban transportation. *Scientific Reports*, *15*. [https://doi.org/10.1038/s41598-025-02933-9](https://doi.org/10.1038/s41598-025-02933-9)

Chang, J., Yin, J., Hao, Y., & Gao, C. (2025). STFDSGCN: Spatio-temporal fusion graph neural network based on dynamic sparse graph convolution GRU for traffic flow forecast. *Sensors*, *25*(11), 3446. [https://doi.org/10.3390/s25113446](https://doi.org/10.3390/s25113446)

Chen, L., Ren, Q., Zeng, J., Zou, F., Luo, S., Tian, J., & Xing, Y. (2023). CSFPre: Expressway key sections based on CEEMDAN-STSGCN-FCM during the holidays for traffic flow prediction. *PLOS ONE*, *18*(4), e0283898. [https://doi.org/10.1371/journal.pone.0283898](https://doi.org/10.1371/journal.pone.0283898)

Chen, W., Yang, Z., Xu, G., & Sun, Y. (2022). Short-term traffic flow prediction based on CNN-BiLSTM with multicomponent information. *Applied Sciences*, *12*(17), 8714. [https://doi.org/10.3390/app12178714](https://doi.org/10.3390/app12178714)

Cui, H., Si, B., Chi, D., Li, Y., Li, G., & Chen, Y. (2025). Short-term passenger flow prediction for urban rail systems: A deep learning approach utilizing multi-source big data. *PLOS ONE*, *20*(1), e0333094. [https://doi.org/10.1371/journal.pone.0333094](https://doi.org/10.1371/journal.pone.0333094)

Cui, Z., Zhang, J., Noh, G., & Park, H. J. (2023). ADSTGCN: A dynamic adaptive deeper spatio-temporal graph convolutional network for multi-step traffic forecasting. *Sensors*, *23*(15), 6950. [https://doi.org/10.3390/s23156950](https://doi.org/10.3390/s23156950)

Deng, H. (2025). Traffic-forecasting model with spatio-temporal kernel. *Electronics*, *14*(7), 1410. [https://doi.org/10.3390/electronics14071410](https://doi.org/10.3390/electronics14071410)

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In *International Conference on Learning Representations*. [https://openreview.net/forum?id=Bkg6RiCqY7](https://openreview.net/forum?id=Bkg6RiCqY7)

Ma, X., & Zhang, H. (2025). Time series forecasting method based on multi-scale feature fusion and Autoformer. *Applied Sciences*, *15*(7), 3768. [https://doi.org/10.3390/app15073768](https://doi.org/10.3390/app15073768)

PDR-STGCN: An enhanced STGCN with multi-scale periodic fusion and a dynamic relational graph for traffic forecasting. (2026). *Systems*, *14*(1), 102. [https://doi.org/10.3390/systems14010102](https://doi.org/10.3390/systems14010102)

Song, Y., Luo, R., Zhou, T., Zhou, C., & Su, R. (2024). Graph attention Informer for long-term traffic flow prediction under the impact of sports events. *Sensors*, *24*(15), 4796. [https://doi.org/10.3390/s24154796](https://doi.org/10.3390/s24154796)

Strigula, M. (2026). Beyond traditional forecasting methods: Evaluating LSTM performance on diverse time series. *Mathematics*, *14*(5), 838. [https://doi.org/10.3390/math14050838](https://doi.org/10.3390/math14050838)

Topilin, I., Jiang, J., Feofilova, A., & Beskopylny, N. (2025). Traffic flow prediction via a hybrid CPO-CNN-LSTM-attention architecture. *Smart Cities*, *8*(5), 148. [https://doi.org/10.3390/smartcities8050148](https://doi.org/10.3390/smartcities8050148)

Wei, L., Guo, D., Chen, Z., Yang, J., & Feng, T. (2023). Forecasting short-term passenger flow of subway stations based on the temporal pattern attention mechanism and the long short-term memory network. *ISPRS International Journal of Geo-Information*, *12*(1), 25. [https://doi.org/10.3390/ijgi12010025](https://doi.org/10.3390/ijgi12010025)

Wu, Z., Liu, X., & Zhang, X. (2025). Multi dynamic temporal representation graph convolutional network for traffic flow prediction. *Scientific Reports*, *15*, 16734. [https://doi.org/10.1038/s41598-025-01157-1](https://doi.org/10.1038/s41598-025-01157-1)