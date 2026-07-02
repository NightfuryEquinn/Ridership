# HMT-TSF — What Is New About It

> Last updated: 2026-07-02

This page explains what actually makes HMT-TSF new. The individual building blocks it
uses (RevIN, TCN, GCN, self-attention) are well known; the novelty is in **how they are
combined, routed, and conditioned** to forecast Malaysian transit ridership through a
major shock like COVID. Ten distinct contributions are listed, each written as
**Main Idea → Evidence → Analysis → Link** and backed by a recent journal reference.

Baseline context: fourteen published deep-learning paradigms in this study treat the 79-feature input as a flat vector, encode temporal and spatial structure through a single pathway, and lack explicit regime adaptation. HMT-TSF is the only architecture that fuses **semantic feature grouping**, **asymmetric tri-stream encoding**, **metadata-free regime gating**, and **known-future calendar conditioning** in one end-to-end model.

---

## 1. Asymmetric Tri-Stream Architecture with Selective Bypass Routing

### Main Idea

HMT-TSF runs three parallel encoders — temporal (TCN), spatial (GCN), and regime (soft embedding mixture) — but **does not feed all three the same representation**. The temporal branch receives RevIN-normalised input through Feature Group Fusion and a Temporal Transformer Block before Multi-Scale TCN; the spatial and regime branches receive **raw RevIN-normalised features directly**, bypassing group fusion and global attention so cross-feature correlations and distributional signatures are learned from the unprojected multivariate signal.

### Evidence

No baseline in the comparison chain (LSTM through Informer) implements parallel encoders with asymmetric input routing. Graph-based models (STGCN, ASTGCN, MTGNN) pass the full sequence into GCN blocks; attention models embed all features jointly. Empirically, HMT-TSF at `nomco_lb14` achieves Combined% 85.78 (FR: 86.59) versus the best baseline Informer at 79.13 — a +6.65 pp margin with MCO degradation of only −3.79 pp (Informer: −5.88 pp). Gao et al. (2023) fuse multi-scale temporal and dual-graph convolution in traffic flow prediction, but both streams share the same node representation rather than deliberately divergent preprocessing.

**Citation:** Gao, H., Fu, Z., Sun, J., Bian, G., & Li, C. (2023). MD-GCN: A multi-scale temporal dual graph convolution network for traffic flow prediction. *Sensors*, *23*(2), 841. https://doi.org/10.3390/s23020841

### Analysis

Bypass routing is architecturally motivated, not an implementation convenience. Feature Group Fusion compresses heterogeneous sources into a shared `d_model` space optimised for temporal convolution; forcing the GCN to operate on that projection would entangle cross-feature Pearson correlations with group-level mixing weights. Regime inference from `x̂.mean(T)` likewise requires access to raw amplitude-normalised feature patterns (e.g., near-zero fuel-price variance during lockdown) that group fusion would attenuate. The design constitutes a **deliberate information-path separation** absent from prior hybrid traffic forecasters.

### Link

Position this as the **top-level architectural contribution**: a tri-modal forecaster where modality-specific preprocessing is co-designed with modality-specific encoders, rather than a late fusion of identically encoded streams.

---

## 2. Semantic Feature Group Fusion with Learned Softmax Gating over Non-Contiguous Sources

### Main Idea

HMT-TSF partitions the aligned feature matrix into five **semantically labelled groups** (target context, lag, temporal/cyclical, external, static) defined by data-source role, not column adjacency. Two groups (temporal, external) span **non-contiguous index segments** matching `feature_align.py` column order. Each group is independently embedded to `d/2`, weighted by a **learned softmax gate** (`group_gate`), concatenated, and projected to `d_model` before the temporal encoder stack.

### Evidence

All fourteen baselines consume `X ∈ ℝ^{T×79}` as a flat multivariate series. SHAP analysis (Section 4.8) confirms the learned hierarchy is domain-consistent — calendar encodings and `total_ridership` dominate — yet no baseline can **adaptively reweight feature sources per timestep**. Xu et al. (2024) combine squeeze-and-excitation gating with ConvLSTM for traffic flow, but gate **channel activations within a single convolutional stream**, not heterogeneous data-source groups with disjoint column indices.

**Citation:** Xu, L., Hu, Y., Wei, X., Zhou, X., & Yu, X. (2024). SE-MAConvLSTM: A deep learning framework for short-term traffic flow prediction combining squeeze-and-excitation network and multi-attention convolutional LSTM. *PLoS ONE*, *19*(11), e0312601. https://doi.org/10.1371/journal.pone.0312601

### Analysis

Malaysian ridership inputs fuse eight independent pipelines (ridership, fuel, rainfall, holidays, GTFS, OSM, GADM, population). Treating them as exchangeable dimensions forces the model to discover source structure from scratch; FGF injects **domain ontology as an inductive bias**. The softmax gate learns context-dependent allocation — e.g., elevating temporal/cyclical weight on holiday-adjacent days — without hand-crafted rules. Non-contiguous segment handling is essential for faithful group assignment; a naive contiguous split would mislabel fuel columns as temporal and corrupt group embeddings.

### Link

Claim **novelty in multivariate transit feature engineering at the architecture level**: semantic grouping plus adaptive gating over administratively aligned, non-contiguous feature layouts — not merely feature selection.

---

## 3. Metadata-Free Regime Gating Embedding for Structural-Break Adaptation

### Main Idea

A **Regime Gating Embedding** module maintains K = 3 learnable regime vectors (pre-MCO, MCO, post-MCO). At each forward pass, mean-pooled RevIN-normalised features project to K softmax logits; the output is a soft mixture `h_r = Σ_k gate_k · E_k`. **No calendar date or regime label is supplied at inference** — the gate is inferred entirely from the observed feature vector.

### Evidence

HMT-TSF is the most MCO-robust model at the headline lookback (Δ −3.79 pp, retained accuracy 81.99%) while STFGCN collapses (Δ −32.94 pp). Hou et al. (2025) apply deep learning to urban rail origin–destination flow under varying demand conditions, but rely on explicit temporal segmentation rather than learned, input-inferred regime mixtures. Kim et al. (2022) address distribution shift via RevIN instance normalisation; HMT-TSF **complements** amplitude normalisation with a discrete regime prior learned in embedding space.

**Citations:**

- Hou, Z., Han, J., & Yang, G. (2025). Analysis of passenger flow characteristics and origin–destination passenger flow prediction in urban rail transit based on deep learning. *Applied Sciences*, *15*(5), 2853. https://doi.org/10.3390/app15052853
- Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022). Reversible instance normalization for accurate time-series forecasting against distribution shift. *International Conference on Learning Representations (ICLR 2022)*. https://openreview.net/forum?id=cGDAkQo1C0p

### Analysis

The Malaysian MCO period (March 2020 – December 2021) is not a magnitude outlier alone — it is a **regime change in which features predict demand at all** (fuel prices and network capacity became irrelevant when movement was prohibited). Static Pearson graphs computed on pre-break training data cannot adapt; regime gating provides a **continuous, differentiable regime switch** without hard date boundaries, enabling smooth blending during easing phases. Metadata-free inference is operationally critical: production systems may not know they have entered a novel shock until feature patterns diverge.

### Link

Frame as a **new mechanism for pandemic-resilient transit forecasting**: joint RevIN + soft regime embedding, inferred from features rather than timestamps — directly addressing the structural-break limitation identified across all graph baselines in Chapter 4.

---

## 4. Attention-Weighted Feature Graph Encoder (Features-as-Nodes with Temporal Pre-Pooling)

### Main Idea

The spatial branch constructs a Pearson-correlation adjacency over **feature nodes** (not geographic stations), but unlike project baselines STGCN/ASTGCN, node features are not raw sequences or simple temporal means. A **learned temporal attention** module compresses `x̂ ∈ ℝ^{T×F}` to `(B, F, 1)` node scalars before two-layer GCN propagation and global mean pooling.

### Evidence

Jiang and Luo (2022) survey GNN traffic forecasting and note that node definition (road segment vs. region vs. sensor) critically determines what spatial relations are captured; none of the surveyed architectures apply **attention-weighted temporal collapse** prior to feature-graph convolution on multivariate national ridership. Project graph baselines use the full `(T, F)` tensor or fixed temporal slices as node attributes. HMT-TSF decouples **when** (attention over T) from **what co-moves** (GCN over F).

**Citation:** Jiang, W., & Luo, J. (2022). Graph neural network for traffic forecasting: A survey. *Expert Systems with Applications*, *207*, 117921. https://doi.org/10.1016/j.eswa.2022.117921

### Analysis

Ridership spikes at recent timesteps (e.g., pre-holiday dip) and baseline weeks ago carry different predictive roles; mean-pooling treats them equally. Attention-weighted pooling lets the graph branch emphasise peak-demand or anomaly days when propagating fuel–rainfall–ridership correlations. Combined with bypass routing (Component 1), the GCN specialises in **cross-feature relational structure** without re-learning temporal dynamics already handled by the TCN stream.

### Link

Contributes a **novel features-as-nodes encoder** distinct from geographic graph methods: attention-temporalised node features + correlation adjacency, designed for national multivariate transit panels.

---

## 5. Global–Local Temporal Stack: Transformer Block before Multi-Scale Causal TCN

### Main Idea

Between Feature Group Fusion and Multi-Scale TCN, a single **Temporal Transformer Block** (learnable positional embeddings, multi-head self-attention, post-LN residual, `d → d` FFN) adds **global lookback attention** before **local dilated causal convolution** at three scales (full, half, quarter window with learned scale blending and DropPath regularisation).

### Evidence

Bouchiat et al. (2023) adapt Transformers for time-series via persistence initialisation; Wu et al. (2023) use multi-periodicity via TimesNet. Neither places a **lightweight global attention stage immediately upstream of multi-scale causal TCN with stochastic depth** in a hybrid spatial–temporal–regime forecaster. HMT-TSF's lb56 configuration activates all three TCN scales; nomco_lb14 peaks at 85.78% while Autoformer — which relies on spectral decomposition alone — ranks 13th at 74.73% and collapses at long lookbacks.

**Citations:**

- Bouchiat, K., Kosorus, H., & Prieler, V. (2023). Persistence initialization: A novel adaptation of the Transformer architecture for time series forecasting. *Applied Intelligence*, *53*, 27931–27946. https://doi.org/10.1007/s10489-023-04927-4
- Wu, H., Xu, T., Liu, H., & Tong, Y. (2023). TimesNet: Temporal 2D-variation modeling for general time series analysis. *International Conference on Learning Representations (ICLR 2023)*. https://openreview.net/forum?id=ju_Uqw384Oq

### Analysis

TCN receptive fields grow with dilation but remain locally connected; Transformers attend to any timestep pair in O(T²) — expensive but valuable at T ≤ 84. Inserting one attention block **before** convolution lets the model identify which historical days matter globally, then extract multi-scale local patterns from that context-enriched sequence. DropPath on TCN blocks prevents the deeper temporal stack from overfitting when scale count increases (lb56, lb84).

### Link

Novel **global-then-local temporal encoding policy** within a hybrid forecaster — neither pure Transformer nor pure TCN, but a sequenced complementarity explicitly motivated by transit's multi-resolution periodicity (daily, weekly, monthly).

---

## 6. SE-Bottleneck Gated Fusion of Temporal, Spatial, and Regime Streams

### Main Idea

Encoder outputs `h_t`, `h_s`, `h_r` ∈ ℝ^d are concatenated to ℝ^{3d}. A **Squeeze-and-Excitation bottleneck** (`3d → 3d/4 → 3d`, sigmoid) produces channel gates `g`, applied as `g ⊙ h` before projection `3d → d → d` with GELU, dropout, and LayerNorm — calibrating stream contribution **per sample** without a full `3d × 3d` fusion matrix.

### Evidence

SE gating in traffic contexts (Xu et al., 2024) modulates convolutional feature maps. HMT-TSF extends SE to **cross-modality fusion of three heterogeneous representation spaces** (convolutional temporal summary, graph-pooled spatial summary, regime embedding). No baseline fuses temporal, graph, and regime vectors; single-paradigm models encode only one structural type.

**Citation:** Xu, L., Hu, Y., Wei, X., Zhou, X., & Yu, X. (2024). SE-MAConvLSTM: A deep learning framework for short-term traffic flow prediction combining squeeze-and-excitation network and multi-attention convolutional LSTM. *PLoS ONE*, *19*(11), e0312601. https://doi.org/10.1371/journal.pone.0312601

### Analysis

Naïve concatenation followed by a linear layer treats all streams equally; the SE bottleneck learns **sample-conditional stream emphasis** — e.g., elevating `h_r` during MCO-like feature patterns, or `h_s` when cross-feature fuel–ridership correlations sharpen. Bottleneck factorisation keeps parameter count manageable at `d_model = 256` (max configuration ~12.4M parameters).

### Link

Publishable as the **fusion layer innovation** that makes tri-stream hybridisation trainable and interpretable: explicit, gated modality arbitration rather than implicit single-stream superposition.

---

## 7. Future Temporal Projection — Zero-Initialised Known-Calendar Conditioning

### Main Idea

For each forecast step t ∈ {1,…,7}, deterministic future calendar features (holiday flags, cyclical encodings, lead–lag counters) are pre-computed as `X_future ∈ ℝ^{T_out × n_temporal}`. A per-step MLP (`n_t → max(2n_t, 32) → 1`) learns an **additive correction in RevIN-normalised space**; output weights are **zero-initialised** so training begins as a no-op and calendar conditioning activates only when gradients justify it.

### Evidence

HMT-TSF is the **only** model in the study receiving `X_future`; this is a deliberate capability, not a protocol oversight. No compared baseline conditions on known future holidays within the forecast horizon. AlKhereibi et al. (2023) use ML for metro ridership with land-use covariates but do not architecturally inject **future-known calendar tensors** into a deep forecaster's output stage.

**Citation:** AlKhereibi, S., Wakjira, T. W., Kucukvar, M., & Onat, N. C. (2023). Predictive machine learning algorithms for metro ridership based on urban land use policies in support of transit-oriented development. *Sustainability*, *15*(2), 1718. https://doi.org/10.3390/su15021718

### Analysis

Seven-day ridership forecasts span weekends and public holidays whose calendar structure is **known at prediction time**; ignoring `X_future` forces the model to extrapolate holiday effects from history alone. Zero-initialisation prevents the calendar branch from destabilising early training (analogous to residual network identity paths). Applied before RevIN denormalisation, corrections scale with per-sample ridership level.

### Link

Novel **forecast-horizon calendar conditioning module** for multi-step transit demand — critical for fair disclosure in publication (architecture + conditioning vs. architecture alone) and for operational deployment where holiday calendars are always available.

---

## 8. Dual Forecast Heads with Suppressed Neural Boosting and Validation-Gated Post-Hoc Residual Correction

### Main Idea

Two in-graph heads predict `y_primary` (highway residual MLP) and `y_boost` (linear, scaled by `sigmoid(α)` with **α init = −2.0 ≈ 0.12 contribution**). An optional **out-of-graph** CatBoost/sklearn booster fits training residuals and applies `y_final += 0.5 × Δ_boost` only if validation Combined% improves.

### Evidence

Huber et al. (2024) hybridise linear and gradient boosting for long-term series; Liu et al. (2026) stack CatBoost for index forecasting. HMT-TSF uniquely combines **in-graph suppressed residual head** (trainable end-to-end) with **validation-gated post-hoc boosting** (applied only at extreme lookbacks in corrected runs: nomco_lb7, nomco_lb84, mco_lb84). Nine of ten full-model configurations achieve `good_fit` — unmatched generalisation among all architectures.

**Citations:**

- Huber, T., Aksan, E., & Ratsch, G. (2024). LTBoost: Boosted hybrids of ensemble linear and gradient algorithms for the long-term time series forecasting. *Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM 2024)*. https://doi.org/10.1145/3627673.3679527
- Liu, Y., Li, Q., Ma, C., & Xu, X. (2026). A CatBoost-based prediction framework for logistics industry prosperity index to support sustainable decision-making: An empirical study from China. *Sustainability*, *18*(5), 2178. https://doi.org/10.3390/su18052178

### Analysis

Single-head models cannot separate stable baseline forecasts from learnable residual corrections. Suppressed α prevents the boost head from dominating before the primary head converges. Validation gating avoids the common stacking pitfall of test-set leakage or guaranteed degradation when residuals are poorly structured.

### Link

Methodological contribution: **two-stage residual refinement policy** (neural + optional tree) with explicit activation criteria — relevant for ensembles in operational forecasting pipelines.

---

## 9. Composite Training Objective: Geometrically Weighted Huber with Temporal Smoothness Regularisation

### Main Idea

Loss = `WeightedHuber(γ=0.9 step decay) + λ · ‖y_{t+1} − y_t‖²` across the seven-day horizon. Step 1 weight is 1.0; step 7 receives 0.9^6 ≈ 0.53. Huber δ = 1.0 provides outlier robustness; smoothness λ = 0.01 penalises day-to-day oscillations in multi-step outputs.

### Evidence

Casolaro et al. (2023) identify horizon-weighted losses and temporal regularisation as open problems in deep forecasting. Baselines in this study use unweighted Huber or MSE uniformly across steps. HMT-TSF's walk-forward blocks show no nomco segment below 73.4 Combined% (full model), with smooth seven-day profiles operationally preferable to spiky multi-step trajectories.

**Citation:** Casolaro, A., Capone, V., Iannuzzo, G., & Camastra, F. (2023). Deep learning for time series forecasting: Advances and open problems. *Information*, *14*(11), 598. https://doi.org/10.3390/info14110598

### Analysis

Transit operators prioritise day-ahead accuracy over day-seven accuracy; geometric decay encodes this utility without discarding distant steps. Smoothness regularisation acts as a soft continuity prior on ridership — physically plausible for daily aggregates absent discrete service disruptions.

### Link

Supporting **training-innovation** claim: objective function co-designed with seven-day operational horizon and Malaysian ridership outlier profile (MCO spikes, holiday surges).

---

## 10. SHAP-Guided Architecture-Aware Feature Reduction (HMT-TSF-FR)

### Main Idea

After initial HMT-TSF training, SHAP attributions identify 26 zero-importance features (17 static + 9 near-constant administered fuel columns). Removal is applied **at model load time** with automatic re-mapping of `FEAT_GROUPS` segments and graph adjacency to a 53-feature space — producing HMT-TSF-FR without retraining the pipeline from raw data.

### Evidence

HMT-TSF-FR achieves **86.59% Combined% and R² 0.906** at nomco_lb14 — study-wide best — while shrinking adjacency from 79×79 to 53×53. FR wins every nomco lookback ≥ 14 (+0.81 to +4.26 pp) but loses under MCO (feature redundancy as robustness budget). Rahimi et al. (2024) note high collinearity inflates SHAP variance; this study closes the loop by **using SHAP not only for interpretation but for architecture-preserving dimensionality reduction**.

**Citation:** Rahimi, S., Mesbah, M., & Nazemi, M. (2024). Interpretable machine learning for traffic prediction: A SHAP-based analysis. *Transportation Research Part C: Emerging Technologies*, *164*, 104653. https://doi.org/10.1016/j.trc.2024.104653

### Analysis

Static features (population, GTFS counts, OSM POI) are time-invariant — zero variance implies zero correlation and zero SHAP, confirming redundancy given sufficient ridership history. The FR/full split is **regime-dependent**: FR for normal operations, full model for shock-prone regimes — a deployable policy emerging from interpretability analysis.

### Link

Novel **interpretability-to-architecture feedback loop**: SHAP-driven feature reduction integrated into the hybrid model's group definitions and GCN — not a separate preprocessing paper.

---

## Synthesis: Publication Positioning Statement

| # | Component | Novelty class | Prior-art gap |
|---|-----------|---------------|---------------|
| 1 | Asymmetric tri-stream bypass routing | **Architectural** | Hybrid models share representations across streams |
| 2 | Semantic FGF + non-contiguous group gates | **Architectural** | Flat multivariate input in all transit DL baselines |
| 3 | Metadata-free regime gating embedding | **Architectural** | Static graphs / global normalisation under COVID break |
| 4 | Attention-weighted feature-graph encoder | **Architectural** | Mean-pool or sequence GCN on features-as-nodes |
| 5 | Transformer → multi-scale TCN stack | **Architectural** | Pure Transformer or pure TCN paradigms |
| 6 | SE-bottleneck tri-stream fusion | **Architectural** | Single-modality encoders only |
| 7 | Zero-init future calendar projection | **Architectural + conditioning** | No known-future calendar injection in baselines |
| 8 | Dual-head + validation-gated boosting | **Methodological** | End-to-end single head or ungated stacking |
| 9 | Weighted Huber + smoothness loss | **Methodological** | Uniform step losses in baselines |
| 10 | SHAP-guided FR variant | **Methodological** | Post-hoc feature selection disconnected from model |

**Recommended article framing:** Propose HMT-TSF as a **purpose-built hybrid forecaster** for national multivariate transit ridership under structural break, centred on Components 1–7 as core architectural claims and Components 8–10 as training and deployment innovations. Report headline results with explicit `X_future` disclosure; include ablation studies (no regime gate, no bypass, flat input, no `X_future`) to isolate each component's marginal contribution.

**Empirical anchors for the article abstract:**

- nomco_lb14: Combined% **86.59** (FR) / **85.78** (full), R² **0.906** / **0.897**
- Best baseline gap: **+6.65 pp** vs Informer (79.13%)
- MCO robustness: Δ **−3.79 pp** (best in study), retained MCO accuracy **81.99%**
- Generalisation: **19/20** good-fit configurations (full + FR)

---

## What Is *Not* Claimed as Novel

The following are **acknowledged prior art** — including mechanisms inherited directly from the fourteen baseline models in the comparison chain — and should be cited, not positioned as contributions. HMT-TSF's value is the **novel composition** documented in Components 1–10 above, not any individual reused primitive.

### External prior art (outside this study)

| Mechanism | Source | Role in HMT-TSF |
|-----------|--------|------------------|
| RevIN (reversible instance normalisation) | Kim et al. (2022) | Per-sample input normalisation and output denormalisation |
| Causal dilated TCN | Bai (2018) | Multi-Scale TCN backbone with left-padded, exponentially dilated convolutions |
| WaveNet gated activation (`tanh ⊙ σ`) | van den Oord et al. (2016) | `TCNResBlock` information gating |
| DropPath / stochastic depth | Huang et al. (2016) | Per-block residual dropout in the TCN stack |
| SE-Net channel gating | Hu et al. (2018) | Bottleneck squeeze-and-excitation in `GatedFusion` (extended to tri-stream fusion, not claimed as original SE) |
| Highway / gated residual networks | Srivastava et al. (2015) | Inspiration for element-wise fusion gates |

### Inherited from the shared experimental protocol (all 14 baselines)

These elements are **common to every model in the comparison chain** and define the evaluation setting; HMT-TSF adopts them unchanged:

- **Input/output contract:** multivariate look-back window `T_in` (extended to {7, 14, 28, 56, 84} vs {14, 28, 56} for baselines), simultaneous multi-step forecast `T_out = 7`, 79-feature aligned dataset from eight spatio-temporal sources
- **Data pipeline:** MinMax-scaled sequences from `sequence_builder.py`, `scaler_X` / `scaler_y` fitted on the training split only, chronological 70 / 15 / 15 train / validation / test split, MCO-included and MCO-excluded conditions
- **Training optimisations (2026 project-wide upgrade):** AdamW optimiser with decoupled weight decay, HuberLoss (`δ = 1.0`) as the base objective, linear LR warm-up → `ReduceLROnPlateau`, gradient clipping (`max_norm = 1.0`), early stopping on validation loss
- **Evaluation currency:** `compute_metrics()` Combined% / MAPE% / MAE% / RMSE% / R², 16-way comparison table and plot via `comparison_table.py`
- **Canonical training-loop pattern:** output directory structure, `results.json` schema, fit-diagnostics hooks, and walk-forward block evaluation — following `stlstm.py` as the project reference implementation

### Inherited from Series 1 — Spatio-temporal / LSTM-family (LSTM, BiLSTM, TPA-LSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM)

| Baseline contribution | Adopted in HMT-TSF? | Notes |
|-----------------------|---------------------|-------|
| Flat multivariate sequence input `(B, T_in, F)` | **Yes** | Same tensor layout; FGF replaces flat treatment |
| Two-layer MLP forecast head (`d → d → T_out`) | **Partially** | `NeuralForecastHead` uses the same MLP-head pattern with highway residual + LayerNorm |
| Recurrent LSTM / BiLSTM encoding | **No** | Temporal branch uses TCN + Transformer instead |
| TPA pattern attention (1-D CNN over LSTM hidden states) | **No** | Different mechanism: global MHA in `TemporalTransformerBlock` + learned time-attn in GCN branch |
| CNN-LSTM local-then-global hierarchy (Conv1d → LSTM) | **Partially** | Local-then-global idea retained via Transformer (global) → TCN (local dilated), but without LSTM or the three CNN-LSTM fusion modes |
| ST-LSTM parallel spatial + temporal streams | **Partially** | Parallel encoding concept extended to three streams (temporal / spatial GCN / regime); spatial stream uses graph convolution, not ST-LSTM's shared-weight MLP + mean pool |

### Inherited from Series 2 — Graph-based (STGCN, ASTGCN, STSGCN, PDR-STGCN, MTGNN, STFGNN)

| Baseline contribution | Adopted in HMT-TSF? | Notes |
|-----------------------|---------------------|-------|
| **Features-as-nodes** graph formulation (N = F features, not geographic stations) | **Yes** | Core spatial representation; established by STGCN in this project |
| **Pearson-correlation adjacency** on `X_train`, threshold = 0.1, soft edge weights, symmetric normalisation | **Yes** | `build_feature_adj()` explicitly matches STGCN/ASTGCN project convention |
| 2-layer graph message passing over feature nodes | **Yes** | `FeatureGraphEncoder`: `GraphConvLayer` × 2 → global mean pool (simplified vs Chebyshev k-hop in STGCN) |
| WaveNet-style / GLU gated temporal convolution | **Partially** | Gated activation in TCN blocks; STGCN's interleaved Chebyshev ST-Conv blocks are **not** reused |
| ASTGCN scaled Chebyshev Laplacian `L̃ = −A_sym` | **No** | HMT-TSF uses standard sym-normalised adjacency without Chebyshev expansion |
| ASTGCN dual spatial-temporal attention | **No** | Only a single temporal MHA block on the fused sequence, not node×time attention |
| STSGCN synchronous 3N×3N spatial-temporal graph | **No** | Spatial and temporal structure are encoded in separate streams |
| PDR-STGCN dynamic per-sample attention adjacency | **No** | Adjacency is static Pearson, not input-adaptive |
| PDR-STGCN periodicity-aware 2-channel input (signal + weekly lag-difference) | **No** | Weekly periodicity handled via lag features, calendar `X_future`, and multi-scale TCN |
| PDR-STGCN multi-scale periodic fusion | **Partially** | Multi-scale **temporal** TCN at three window lengths, not PDR's periodicity encoding |
| MTGNN end-to-end learned asymmetric adjacency from node embeddings | **No** | Graph is fixed at training time |
| MTGNN dilated inception temporal blocks | **Partially** | Dilated causal conv in TCN shares the multi-scale temporal inductive bias, different architecture |
| STFGNN dual spatial + temporal adjacency matrices | **No** | No temporal adjacency graph |
| STFGNN learned scalar λ fusion of dual graphs | **No** | Tri-stream SE-bottleneck fusion replaces scalar graph blending |

### Inherited from Series 3 — Attention-based (TPA-LSTM, ASTGCN, Autoformer, Informer)

| Baseline contribution | Adopted in HMT-TSF? | Notes |
|-----------------------|---------------------|-------|
| Multi-head self-attention over the look-back window | **Partially** | Single `TemporalTransformerBlock` (one MHA layer + FFN), not a deep encoder stack |
| Learnable positional embeddings | **Yes** | Injected in `TemporalTransformerBlock` before MHA |
| AMP mixed precision (`GradScaler` + `autocast`) | **Yes** | Same CUDA fp16 training pattern as ASTGCN, Autoformer, and Informer |
| Autoformer series decomposition (trend / seasonal moving-average split) | **No** | No explicit decomposition branch |
| Autoformer FFT-based auto-correlation (`O(L log L)`) | **No** | Global attention is standard MHA, not frequency-domain autocorrelation |
| Informer ProbSparse attention | **No** | Full attention at all lookback lengths used |
| Informer encoder distilling (Conv1d + MaxPool halving sequence) | **No** | Full `T_in` retained through the Transformer block |
| Informer generative decoder (start-token + zero-filled future) | **No** | Direct multi-step head; `X_future` supplies calendar context separately |
| TPA-LSTM (listed in attention folder) | **No** | See Series 1 row above |

### Explicitly *not* inherited from any baseline

The following are **HMT-TSF-only** in this study (cross-reference Components 1–10):

- Semantic Feature Group Fusion with non-contiguous group segments and learned softmax `group_gate`
- Asymmetric bypass routing (GCN + regime branches skip FGF and Transformer)
- Metadata-free Regime Gating Embedding (K = 3 soft mixture, no date input)
- Learned temporal attention pre-pooling before feature-graph GCN
- Transformer → Multi-Scale TCN global-then-local stack
- SE-bottleneck gated fusion of temporal / spatial / regime streams
- Future Temporal Projection from known `X_future` calendar tensors
- Dual forecast heads with α-suppressed boosting + validation-gated CatBoost residual correction
- Geometrically weighted Huber + temporal smoothness composite loss
- SHAP-guided architecture-aware feature reduction (HMT-TSF-FR)

### Publication framing

When writing the article, cite the baseline sources for every reused row above and restrict the contribution claim to the **integrated hybrid design** and its **empirical validation** on Malaysian transit ridership. Ablation experiments should disable each novel component (Components 1–10) while leaving inherited baseline mechanisms in place, so reviewers can separate composition novelty from borrowed primitives.
