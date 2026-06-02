# Model Descriptions

> Last updated: 2026-05-29

This document describes all 15 deep-learning models used in the Malaysian transit ridership forecasting study. Models are divided into three baseline series — **Spatio-Temporal (LSTM-family)**, **Graph-Based**, and **Attention-Based** — plus one **Hybrid SOTA** model (HMT-TSF). All 14 baseline models share the same input/output dimensions (T_in ∈ {14, 28, 56} look-back via `--lookback`, T_out=7 forecast horizon), the same dataset (79 features across 8 spatio-temporal sources), and the same evaluation metrics (Combined%, MAPE%, MAE%, RMSE%, R², MAE, RMSE). HMT-TSF extends look-back support to {7, 14, 28, 56, 84} days.

**Shared training optimizations (all 14 models):** AdamW optimiser (decoupled weight decay), HuberLoss (default, selectable via `--loss {mse,huber,mae}`), and a 5-epoch linear LR warm-up before ReduceLROnPlateau (configurable via `--warmup-epochs`). Architecture and hyperparameter values are unchanged.

---

## Series 1 — Spatio-Temporal (LSTM-Family)

Located in `src/models/spatio-temporal-based/`. These models are built around recurrent cells and treat the feature vector at each timestep as a flat multivariate input. They do not use explicit graph structure.

---

### 1. LSTM — Baseline Long Short-Term Memory

**Script:** `lstm.py`

The foundational baseline for the comparison chain. A standard stacked LSTM reads the T_in-step look-back window left-to-right, compressing the entire input sequence into a single final hidden state h_T which is passed to an MLP head to produce T_out predictions.

**Architecture:**
```
X (B, T_in, F) → LSTM → h_T (B, hidden) → MLP → (B, T_out)
```

**Key design choices:**
- Single causal recurrent pass — no bidirectionality, no attention, no graph convolution.
- Huber loss (default), AdamW optimiser, linear LR warm-up then ReduceLROnPlateau.
- Dropout applied between stacked LSTM layers when `--layers > 1`.
- Default: `hidden=64`, `layers=1`, `dropout=0.1`, `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`.

**Role in the study:** Establishes the performance floor. All subsequent models are compared against it to quantify each architectural improvement.

**Key limitation:** The single hidden vector at T is a bottleneck. No mechanism to identify which timesteps or which input features are most predictive.

---

### 2. BiLSTM — Bidirectional LSTM

**Script:** `bilstm.py`

Extends LSTM by adding a reversed recurrent pass over the same look-back window. The forward and backward final hidden states are concatenated before the MLP head, doubling the representational capacity.

**Architecture:**
```
X (B, T_in, F) → BiLSTM → cat([h_fwd, h_bwd]) (B, hidden×2) → MLP → (B, T_out)
```

**Key design choices:**
- `bidirectional=True` is the only architectural change from LSTM.
- MLP head input size doubles to `hidden × 2`.
- Default: `hidden=64`, `layers=1`, `dropout=0.1`, `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`.

**Why BiLSTM is valid for forecasting:** Bidirectionality applies only over the observed look-back window, not into the future. At inference time the full window is known, so the backward pass is legitimate. This lets the model use later context in the window when encoding earlier timesteps — beneficial for capturing mid-window peaks or dips.

**Difference from LSTM:** Richer context at every encoded position; mid-window events (e.g. a holiday spike on day 7 of 14) are visible from both directions.

---

### 3. TPA-LSTM — Temporal Pattern Attention LSTM

**Script:** `tpalstm.py`

Adds the Temporal Pattern Attention (TPA) mechanism on top of a standard LSTM. Instead of scoring individual timestep hidden states (Bahdanau-style), TPA applies a 1-D CNN to the hidden-state matrix H to extract temporal pattern filters, then scores each filter against the final hidden state h_T.

**Architecture:**
```
X (B, T_in, F) → LSTM → H (B, T_in, hidden), h_T (B, hidden)
H[:, :-1, :] → permute → Conv1d → ReLU → adaptive avg pool → C (B, n_filters, *)
score = sigmoid(h_T @ W_score @ C^T)  → softmax → attn (B, n_filters)
context = attn @ C_pool  (B, hidden)
cat([h_T, context]) → MLP → (B, T_out)
```

**Key design choices:**
- TPA attention is over learned temporal pattern filters, not raw timesteps.
- The CNN detects multi-scale periodic sub-patterns (daily, weekly rhythms).
- Default: `hidden=64`, `filters=32`, `kernel_size=3`, `dropout=0.1`, `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`.

**Difference from BiLSTM:** Moves from positional context (bidirectionality) to pattern-level attention — the model learns which recurring temporal shapes are most predictive, rather than which position in the window to attend to.

---

### 4. CNN-LSTM

**Script:** `cnnlstm.py`

Combines a 1-D Convolutional Neural Network with an LSTM. Supports three fusion modes:

- **Sequential (default):** CNN extracts local temporal features → LSTM models dependencies across the CNN output sequence. The LSTM never sees raw features; it operates on higher-level CNN representations.
- **Parallel:** CNN and LSTM process raw input independently; outputs are concatenated before the MLP head. Preserves raw feature access for the LSTM.
- **Augmented Sequential:** Sequential CNN→LSTM hierarchy with an additional skip connection — the raw input is globally pooled and concatenated with the LSTM final hidden state. Prevents information loss from aggressive CNN filtering while keeping the hierarchical abstraction.

**Architecture (sequential):**
```
X (B, T_in, F) → permute → Conv1d × L → permute → LSTM → h_T → MLP → (B, T_out)
```

**Key design choices:**
- Each CNN block: `Conv1d → BatchNorm1d → ReLU` with same-padding (T_in preserved).
- Default mode: `sequential`, `cnn_filters=32`, `cnn_layers=2`, `hidden=64`, `dropout=0.1`, `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`.
- Augmented mode typically yields the best generalisation of the three.

**Difference from TPA-LSTM:** CNN acts as a feature extractor before the LSTM, hierarchically reducing the raw feature space; TPA-LSTM uses CNN only in the attention module while the LSTM still receives raw features.

---

### 5. CNN-BiLSTM

**Script:** `cnnbilstm.py`

Replaces the unidirectional LSTM encoder in CNN-LSTM (sequential mode) with a BiLSTM. Combines two complementary inductive biases:
- **CNN** — local pattern detection: short receptive field, translation-invariant, applied before recurrence.
- **BiLSTM** — global sequential context: bidirectional over the full CNN-feature sequence.

**Architecture:**
```
X (B, T_in, F) → permute → Conv1d × L → permute → BiLSTM
→ cat([h_fwd, h_bwd]) (B, hidden×2) → MLP → (B, T_out)
```

**Key design choices:**
- Each CNN block: `Conv1d → BatchNorm1d → ReLU`.
- BiLSTM head input size = `hidden × 2`.
- Default: `cnn_filters=32`, `cnn_layers=2`, `hidden=64`, `dropout=0.1`, `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`.

**Difference from CNN-LSTM:** The richer bidirectional context over CNN features means mid-window anomalies are encoded from both temporal directions, not just left-to-right. This is particularly useful when a holiday or event falls in the middle of the look-back window.

---

### 6. ST-LSTM — Spatio-Temporal LSTM

**Script:** `stlstm.py` — **canonical reference model** for all subsequent models.

Introduces an explicit spatial encoding stream to complement the temporal LSTM. Two parallel streams are fused before the MLP head:

- **Temporal stream (LSTM):** Standard LSTM over T_in. Captures how the feature sequence evolves over time — the "when" dimension.
- **Spatial stream (shared MLP per timestep + mean pooling):** A weight-shared MLP is applied to each timestep's full feature vector and the embeddings are mean-pooled over time. Collapses temporal order to summarise persistent cross-feature spatial patterns — the "which features co-activate" dimension.

**Architecture:**
```
Temporal:  X (B, T_in, F) → LSTM → h_T (B, hidden_t)
Spatial:   X.reshape(B*T, F) → Linear → ReLU → Linear → ReLU → (B*T, spatial_hidden)
           → reshape → (B, T_in, spatial_hidden) → mean(dim=1) → sp (B, spatial_hidden)
Fusion:    cat([h_T, sp]) → MLP → (B, T_out)
```

**Key design choices:**
- Shared spatial MLP weights across all T_in timesteps (time-invariant spatial encoder).
- Clean separation: LSTM learns temporal dynamics; spatial encoder learns persistent cross-feature interactions.
- Default: `hidden=64`, `spatial_hidden=32`, `dropout=0.1`, `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`.

**Difference from CNN-BiLSTM:** Decouples spatial (cross-feature) reasoning from temporal reasoning with separate dedicated streams; CNN-BiLSTM does not explicitly distinguish between these two types of structure.

---

## Series 2 — Graph-Based

Located in `src/models/graph-based/`. These models represent the N input features as graph nodes and use graph convolution to propagate information between correlated features. The spatial adjacency matrix is built from absolute Pearson correlation of feature columns in X_train (threshold=0.1) unless otherwise noted.

---

### 7. STGCN — Spatio-Temporal Graph Convolutional Network

**Script:** `stgcn.py`

The first graph-based model. Uses interleaved Temporal Gated Convolutions and Chebyshev Graph Convolutions in stacked ST-Conv blocks. The graph is a fixed, pre-computed symmetric-normalised Laplacian derived from feature correlation.

**Architecture:**
```
X (B, T_in, N) → reshape → (B, N, 1, T_in)

ST-Conv Block:
  TemporalGatedConv (GLU)  → (B, N, C_mid, T-Kt+1)
  ChebGraphConv (K-hop)    → (B*T', N, C_mid)
  TemporalGatedConv (GLU)  → (B, N, C_out, T'-Kt+1)
  BatchNorm2d

Output: TemporalGatedConv → mean(N) → flatten → MLP → (B, T_out)
```

**Key design choices:**
- K-order Chebyshev polynomial approximation of graph convolution avoids eigendecomposition.
- GLU (Gated Linear Unit) gating in temporal convolutions controls information flow.
- Fixed static graph: built once from training data, stored as a model buffer.
- Default: `hidden=128`, `cheb_k=3`, `kt=3`, `n_blocks=2`, `adj_threshold=0.1`, `dropout=0.1`, `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`, `weight_decay=1e-4`.

**Key limitation:** The graph is static — it does not adapt to different input samples or temporal contexts.

---

### 11. MTGNN — Multi-Scale Temporal Graph Neural Network

**Script:** `mtgnn.py`

Learns its graph adjacency end-to-end via two trainable node-embedding matrices (M1, M2), removing the need for a pre-built correlation graph. Uses a Dilated Inception module for multi-scale temporal modelling and Mix-hop graph convolution for neighbourhood aggregation.

**Architecture:**
```
A = softmax(ReLU(tanh(α) * (M1 @ M2.T − M2 @ M1.T)))   [asymmetric, learned]

Per block (B, hidden, N, T):
  InceptionBlock (4-branch dilated Conv1d: kernels 1,3,5,7 + GLU)
  MixHopConv: out = Σ_k A^k @ x @ W_k  (k=0…d_hop)
  skip + residual

Skip aggregation → ReLU → mean(N,T) → MLP → (B, T_out)
```

**Key design choices:**
- Directed, asymmetric graph — A ≠ A^T, allowing feature i to influence feature j without the reverse.
- Dilated Inception: four parallel branches with kernel sizes [1, 3, 5, 7] capture daily, multi-day, and weekly periodicity simultaneously.
- Mix-hop: independent linear projections at each hop order prevent over-smoothing.
- Default: `hidden=32`, `skip_ch=64`, `n_layers=3`, `d_emb=10`, `d_hop=2`, `dropout=0.1`, `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`, `weight_decay=1e-4`.

**Difference from STGCN:** MTGNN's graph is fully adaptive and task-specific (learned from data, not pre-computed from correlation); temporal modelling uses multi-scale inception rather than sequential Cheb+temporal convolution blocks.

---

### 9. STSGCN — Spatial-Temporal Synchronous Graph Convolutional Network

**Script:** `stsgcn.py`

Captures spatial and temporal correlations simultaneously in a single fused 3N×3N synchronous graph (STSG) built from 3 consecutive timesteps. Unlike STGCN which alternates spatial and temporal operations, STSGCN fuses them in one synchronous adjacency.

**Architecture:**
```
STSG (3N×3N) = [[A_spa,  I,      0    ],
                [I,      A_spa,  I    ],
                [0,      I,      A_spa]]

STSGCL (per layer):
  Unfold T into windows of 3: (B, T-2, 3N, C_in)
  ChebConv on L_stsg (3N×3N) + GLU: → (B, T-2, 3N, C_out)
  Extract centre N nodes:           → (B, T-2, N, C_out)

Mean pool over N, flatten T, MLP head → (B, T_out)
```

**Key design choices:**
- The STSG encodes temporal adjacency via identity blocks (I) linking timestep t to t±1, and spatial adjacency via A_spa within each timestep.
- Each STSGCL layer shrinks T by 2 (sliding window of 3).
- GLU gating within each conv for selective information flow.
- Default: `hidden=64`, `n_layers=2`, `cheb_k=2`, `dropout=0.1`, `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`, `weight_decay=1e-4`.

**Difference from STGCN:** Fuses spatial and temporal graph operations into a single synchronous convolution step rather than sequential interleaving; avoids the temporal shrinkage from stacked TemporalGatedConvs by using a structured sliding window.

---

### 12. STFGNN — Spatial-Temporal Fusion Graph Neural Network

**Script:** `stfgnn.py`

Fuses two complementary static graphs in each layer via a learnable gate:
- **A_spa (spatial graph):** Absolute Pearson correlation of feature values across training samples. Captures features that co-vary at the same time.
- **A_tem (temporal graph):** Absolute Pearson correlation of mean temporal profiles per node. Captures features that share the same intra-window dynamics (similar day-of-week pattern, etc.).

**Architecture:**
```
Input projection: (B, T_in, N) → (B, hidden, N, T_in)

Fusion Block × n_layers:
  GatedTCN          (same-length temporal conv + GLU)
  Spatial GCN       (sym-norm A_spa propagation)
  Temporal GCN      (sym-norm A_tem propagation)
  Learnable gate    h = α * spa_feat + (1-α) * tem_feat
  LayerNorm + residual

Global mean pool(N, T) → MLP → (B, T_out)
```

**Key design choices:**
- Two separate graphs encode two distinct notions of feature similarity.
- Per-layer scalar gate α learned by the network — no manual tuning of graph fusion.
- LayerNorm (not BatchNorm) for stability within each residual block.
- Default: `hidden=64`, `n_layers=3`, `dropout=0.1`, `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`, `weight_decay=1e-4`.

**Difference from STSGCN:** Uses two parallel complementary graphs (spatial + temporal) fused via a gate, instead of one synchronous graph; temporal graph explicitly models shared dynamics profiles rather than relying on structural adjacency alone.

---

### 10. PDR-STGCN — Periodicity-Aware Dynamic Relational STGCN

**Script:** `pdr_stgcn.py`

Novel architecture (this study). Extends the STGCN backbone with three innovations:

**1. Periodicity Encoding**
A second input channel is created by computing the periodic lag-difference:
```
x_diff[t] = x[t] - x[t - period]   (zero-padded for t < period)
```
For T_in=14 with weekly `period=7`, this makes weekly seasonal deviations explicit as a second feature channel, without any extra parameters.

**2. Dynamic Relational Graph Convolution**
Each ST block replaces the fixed Chebyshev convolution with a two-path convolution mixing a static base graph with an input-adaptive dynamic graph:
```
Static path:  A_sym @ h @ W_static         (sym-normalised Pearson correlation)
Dynamic path: softmax(Q @ K^T / sqrt(d_k)) @ V    (per-sample attention adjacency)
Mix:          out = σ(λ) · static + (1-σ(λ)) · dynamic   (λ learned scalar, init=0)
```

**3. ST Block Structure (STGCN backbone)**
```
TemporalGatedConv → DynamicRelationalGraphConv → TemporalGatedConv → BN
```

**Architecture:**
```
X (B, T_in, N) → periodic diff encoder → (B, N, 2, T_in)

PDR-ST Block:
  TemporalGatedConv → DynamicRelGraph → TemporalGatedConv → BatchNorm2d

Output: TemporalGatedConv → mean(N) → flatten → MLP → (B, T_out)
```

**Key design choices:**
- 2-channel input (original signal + weekly lag-difference).
- Learned scalar λ controls the mix between static and dynamic graph paths.
- Default: `hidden=128`, `kt=3`, `n_blocks=2`, `period=7`, `dk=32`, `dropout=0.1`, `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`, `weight_decay=1e-4`.

**Difference from STGCN:** Adds periodicity-aware input encoding and sample-adaptive dynamic graph; STGCN uses a fixed static graph and single-channel input. PDR-STGCN is the most expressive graph model in this study.

---

## Series 3 — Attention-Based

Located in `src/models/attention-based/`. These models use attention mechanisms as the primary modelling tool. They use AMP (fp16 + GradScaler) for mixed-precision training on the A100. Note: TPA-LSTM is listed in Series 1 (LSTM-family) because its primary encoder is still a recurrent LSTM; attention is used only as a decoder module.

---

### 8. ASTGCN — Attention-Based Spatio-Temporal Graph Convolutional Network

**Script:** `astgcn.py`

Augments STGCN with multi-head spatial and temporal attention. The graph is built from absolute Pearson correlation using a scaled Chebyshev Laplacian `L_tilde = -A_sym`. Within each ASTGCN block, spatial attention re-weights which nodes to attend to across the graph, and temporal attention re-weights which timesteps to focus on.

**Architecture (one ASTGCN block):**
```
X (B, T, N, d)
  → Spatial Attention   (multi-head, over N nodes)
  → ChebGCN             (K-hop diffusion on A)
  → Temporal Attention  (multi-head, over T steps)
  → Position-wise FFN

n_blocks stacked, then:
  → Mean pool over N → (B, T, d)
  → Flatten + MLP → (B, T_out)
```

**Key design choices:**
- Dual attention: spatial (which features/nodes to weight) + temporal (which timesteps to weight).
- Chebyshev graph convolution with `L_tilde = -A_sym` (scaled Laplacian).
- AMP (fp16) + GradScaler for memory efficiency.
- Default: `d_model=64`, `n_blocks=2`, `K=3`, `n_heads=4`, `dropout=0.1`, `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`.

**Difference from STGCN:** Adds learnable multi-head attention over both spatial (node) and temporal (time) dimensions; STGCN uses fixed gated convolutions. This allows ASTGCN to dynamically re-weight relevance at each layer.

---

### 13. Autoformer

**Script:** `autoformer.py`

A decomposition transformer that replaces standard dot-product attention with an Auto-Correlation mechanism based on FFT. Key innovations:

**1. Series Decomposition**
Applied throughout encoder and decoder as a learnable sub-layer:
```
Trend    = MovingAvg(X)          [low-frequency component]
Seasonal = X - MovingAvg(X)      [residual high-frequency component]
```

**2. Auto-Correlation Mechanism (O(L log L))**
Instead of pairwise query-key dot-products, discovers period-based dependencies via the time-delay autocorrelation:
```
corr(τ) = IFFT(FFT(Q) · conj(FFT(K)))
```
Selects the top-k lags, rolls V by each lag, and aggregates with softmax weights.

**3. Encoder-Decoder**
```
Encoder: L_e layers of [AutoCorr + Decomp + FFN + Decomp]
Decoder: L_d layers of [AutoCorr + CrossCorr + Decomp + FFN + Decomp]
         → accumulated trend + seasonal residual

Output = trend_accum[:, -T_out:] + seasonal_dec[:, -T_out:]
       → Linear(n_features, 1) → (B, T_out)
```

**Key design choices:**
- FFT operations wrapped in explicit `float32` cast inside `autocast` for numerical stability.
- AMP (fp16) + GradScaler.
- Default: `d_model=64`, `n_heads=4`, `e_layers=2`, `d_layers=1`, `moving_avg=5`, `dropout=0.1`, `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`.

**Difference from ASTGCN:** No graph structure; replaces attention with FFT-based autocorrelation and explicit series decomposition. Better suited to strongly periodic signals like daily/weekly ridership.

---

### 14. Informer

**Script:** `informer.py`

A transformer variant designed for efficient long-sequence forecasting via two mechanisms:

**1. ProbSparse Self-Attention O(L log L)**
Instead of computing all L×L query-key scores, selects the top-u "active" queries by a sparsity measure (KL divergence from uniform attention). Only these queries compute full attention; remaining queries use the mean of all values:
```
u = c · ⌈ln(L_K)⌉
```
For T_in=14, ProbSparse gracefully degenerates to near-full attention (u ≈ 13), eliminating approximation error on short sequences.

**2. Self-Attention Distilling**
After each encoder layer, a `Conv1d + ELU + MaxPool1d(2)` halves the sequence length. This allows deeper encoders without quadratic cost.

**3. Generative Decoder**
Initialised with the last T_label = T_in//2 steps as a "start token" plus zero-padding for T_out future steps. Generates all future steps in one forward pass — no autoregression.

**Architecture:**
```
Encoder: [ProbSparseAttn + ConvLayer(distil)] × (e_layers-1)
         + [ProbSparseAttn] (last layer, no distil)
Decoder: [FullAttn(self) + FullAttn(cross)] × d_layers

Output: last T_out rows → Linear(d_model, 1) → (B, T_out)
```

**Key design choices:**
- AMP (fp16) + GradScaler for memory efficiency.
- Decoder input: `[X[:, -7:, :], zeros(B, 7, F)]` → length=14.
- Default: `d_model=64`, `n_heads=4`, `e_layers=2`, `d_layers=1`, `dropout=0.1`, `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`.

**Difference from Autoformer:** No series decomposition or FFT autocorrelation; efficiency gain comes from query sparsification and distilling (sequence halving) rather than spectral-domain computation. Autoformer is periodic-dependency focused; Informer is efficient-attention focused.

---

## Comparison Chain

Each model auto-detects all prior model runs from `src/outputs/` and adds itself to the comparison table:

```
LSTM (2-way) → BiLSTM (3-way) → TPA-LSTM (4-way) → CNN-LSTM (5-way)
→ CNN-BiLSTM (6-way) → ST-LSTM (7-way) → STGCN (8-way)
→ ASTGCN (9-way) → STSGCN (10-way) → PDR-STGCN (11-way) → MTGNN (12-way)
→ STFGNN (13-way) → Autoformer (14-way) → Informer (15-way)
→ HMT-TSF (16-way)
```

---

## Series 4 — Hybrid SOTA

Located in `src/models/hybrid/`. A purpose-built model that combines temporal, spatial, and regime-aware representations in a single end-to-end architecture. Uses AMP (fp16 + GradScaler) on the A100. Full architecture diagram and component rationale in `src/models/hybrid/HMT-TSF.md`.

---

### 15. HMT-TSF — Hybrid Multi-scale Temporal Spatio-Feature Forecaster

**Script:** `src/models/hybrid/hmttsf.py`

The SOTA model for this study. Fuses three parallel encoders into a single gated representation, then optionally corrects residuals with a trained gradient boosting model.

**Five Feature-Group Encoders → Single Fused Embedding:**

All 79 input features are first split into five semantic groups and independently embedded via small MLPs before fusion:
- **Target context** (idx 0–12): 13 service-line ridership values → MLP → d/2
- **Temporal/cyclical** (idx 13–28): 16 holiday + cyclical features → MLP → d/2
- **External** (idx 29–58): 30 fuel + rainfall features → MLP → d/2
- **Lag** (idx 59–61): 3 autoregressive lags → MLP → d/2
- **Static** (idx 62–78): 17 population/GTFS/OSM/GADM features → MLP → d/2

The five embeddings are concatenated, gated, and projected to `d_model` via `Linear → GELU → LayerNorm`. RevIN (Reversible Instance Normalisation) is applied to the raw input before encoding to reduce within-batch distribution shift from MCO/COVID regime changes.

**Three Parallel Encoders:**

1. **Multi-Scale TCN:** Causal dilated convolutions at 1–3 scales (full T_in, T_in//2 for ≥28, T_in//4 for ≥56). WaveNet-style gated activations (tanh ⊙ σ) with residuals. Learned attention blending across scales → single `(B, d_model)` temporal summary.

2. **Feature Graph Encoder (GCN):** Mean-pools the fused embedding over T, treats the result as node features on an F×F Pearson-correlation adjacency (threshold=0.1, same strategy as STGCN/ASTGCN), runs 2-layer GCN, global mean-pools over nodes → `(B, d_model)` spatial summary.

3. **Regime Gating Embedding:** Mean-pools fused embedding over T, applies a linear to produce K=3 soft gate logits (pre-COVID / COVID / post-COVID), multiplies each gate weight by a learned regime embedding vector → soft mixture → `(B, d_model)` regime summary.

**Gated Fusion:**
```
h = cat[h_temporal, h_spatial, h_regime]   (B, 3·d_model)
g = σ(Linear(h))
out = Linear(g ⊙ h) → GELU → Dropout → LayerNorm   (B, d_model)
```

**Dual Forecast Heads:**
- **Primary head:** `h → 2d → d → T_out` with highway residual.
- **Boosting head:** `h → d → T_out`, scaled by `sigmoid(α)` (α initialised to 0.1 to suppress early boosting).
- `y_final = y_primary + y_boosting`

**Optional Post-Hoc Residual Boosting (Phase 3):**
- Collect neural predictions on the training set; compute residuals `Δ = y_true − y_neural`.
- Fit CatBoost (if installed) or sklearn MLP on flattened `(X_train, Δ)`.
- Apply correction: `y_final += 0.5 × Δ_boost` — only if Combined% improves on validation.
- Activate via `--use-catboost`.

**Walk-Forward Evaluation (Phase 4):**
- Test set split into 3 equal chronological blocks; Combined% and R² reported per block.
- Allows detection of temporal performance degradation.

**Custom Loss:**
```
L = WeightedHuber(step-decay γ=0.9) + λ · TemporalSmoothness
```
Step 1 has weight 1.0; subsequent steps decay geometrically. Temporal smoothness penalises `‖y_{t+1} − y_t‖²` across T_out to prevent oscillatory predictions.

**Optional HPO:** Optuna with MedianPruner — `--tune-trials N` runs N trials (up to 40 epochs each) before full training. Search space covers `d_model`, `n_tcn_blocks`, `graph_hidden`, `dropout`, `lr`, `smooth_weight`.

**Architecture (end-to-end):**
```
X (B, T_in, 79)  →  RevIN  →  FeatureGroupFusion  →  (B, T_in, d_model)
                                    ↙           ↓           ↘
                         Multi-Scale TCN   Feature GCN   Regime Gating
                                    ↘           ↓           ↙
                                         Gated Fusion  →  (B, d_model)
                                              ↓
                                    Primary + Boost Heads  →  (B, T_out=7)
                                              ↓  [optional]
                                    CatBoost / MLP residual correction
```

**Default hyperparameters:** `d_model=128`, `n_tcn_blocks=3`, `graph_hidden=64`, `n_regimes=3`, `dropout=0.1`, `epochs=150`, `batch_size=32`, `lr=1e-3`, `weight_decay=1e-4`, `patience=15`, `warmup_epochs=5`, `smooth_weight=0.01`, `loss_decay=0.9`.

**Lookback support:** {7, 14, 28, 56, 84} days — wider than the 15 base models ({14, 28, 56} only). Multi-scale TCN: Scale 2 (T//2) activates for T_in ≥ 28; Scale 3 (T//4) activates for T_in ≥ 56. Sequence dirs for lookback 7 and 84 must be built before use:
```bash
python src/features/sequence_builder.py --T-in 7
python src/features/sequence_builder.py --T-in 84
```

**Optimisation targets:** Combined% ≥ 75, R² ≥ 0.70.

---

## Fine-Tuned Model Series

Fourteen mirrored fine-tuned variants of the above models are located in three new folders:

| Folder | Models |
|--------|--------|
| `src/models/spatio-temporal-tuned/` | LSTM, BiLSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM |
| `src/models/graph-tuned/` | STGCN, MTGNN, STSGCN, STFGNN, PDR-STGCN |
| `src/models/attention-tuned/` | TPA-LSTM, ASTGCN, Autoformer, Informer |

**Tuning strategy (Option C):** Best-configuration selection (MCO-exclusion setting + lookback window) combined with revised architecture hyperparameters. Training hyperparameters (`epochs`, `batch_size`, `lr`, `patience`, `weight_decay`) are unchanged.

**Output directories** use the `_tuned` suffix: `src/outputs/{model_name}_tuned/`.

**Comparison:** each tuned run compares against all 14 base model results (not the tuned runs). Fine-tuned=yes tagging is applied externally by `src/utils/aggregate_results.py`.

### Architecture Changes at a Glance (16 tuned variants)

CNN-LSTM is split into three independently tuned variants — one per mode — each with its own best look-back window. All other tuning changes are architecture-only; training hyperparameters (`epochs`, `batch_size`, `lr`, `patience`, `weight_decay`) are unchanged across all 16 variants. Parameters that did not change from base are omitted from the Key Changes column. Dropout rationale reflects the variance observed across the three look-back window configurations during best-configuration selection.

| Model | Variant / lookback | Key Changes (base → tuned) | Dropout rationale |
|-------|-------------------|---------------------------|-------------------|
| LSTM | — / 14 | hidden 64→128, layers 1→2, dropout 0.10→0.15 | 2-layer regularisation |
| BiLSTM | — / 14 | hidden 64→128, layers 1→2, dropout 0.10→0.15 | expanded recurrent capacity |
| CNN-LSTM | sequential / 14 | cnn_filters 32→64, dropout 0.10→0.20 | MED-HIGH variance across lookbacks |
| CNN-LSTM | parallel / 14 | cnn_filters 32→64, dropout 0.10→0.20 | MED-HIGH variance across lookbacks |
| CNN-LSTM | augmented / 14 | cnn_filters 32→64, dropout 0.10→0.20 | MED-HIGH variance across lookbacks |
| CNN-BiLSTM | — / 14 | hidden 64→128, cnn_filters 32→64, dropout 0.10→0.25 | HIGH variance across lookbacks |
| ST-LSTM | — / 14 | hidden 64→128, spatial_hidden 32→64, dropout 0.10→0.20 | MED-HIGH variance across lookbacks |
| STGCN | — / 14 | hidden 128→256, n_blocks 2→3, kt 3→2 ¹, dropout 0.10→0.15 | LOW variance, stable with more filters |
| MTGNN | — / 14 | hidden 32→64, skip_ch 64→128, n_layers 3→4, dropout 0.10→0.15 | HIGH variance across lookbacks |
| STSGCN | — / 14 | hidden 64→128, n_layers 2→3, dropout 0.10→0.20 | LOW variance, synchronous graph benefits |
| STFGNN | — / 14 | hidden 64→128, n_layers 3→4, dropout 0.10→0.30 | MED-HIGH variance across lookbacks |
| PDR-STGCN | — / 14 | hidden 128→256, n_blocks 2→3, kt 3→2 ¹, dk 32→64, dropout 0.10→0.20 | MED variance across lookbacks |
| TPA-LSTM | — / 14 | hidden 64→128, filters 32→64, dropout 0.10→0.15 | moderate variance across lookbacks |
| ASTGCN | — / 14 | d_model 64→128, n_heads 4→8, n_blocks 2→3, dropout 0.10→0.20 | moderate variance across lookbacks |
| Autoformer | — / 14 | d_model 64→128, n_heads 4→8, e_layers 2→3, d_ff 128→256, dropout 0.10→0.20 | moderate variance across lookbacks |
| Informer | — / 14 | d_model 64→128, n_heads 4→8, e_layers 2→3, d_ff 128→256, dropout 0.10→0.15 | LOW variance, sparse attention stable |

> ¹ `kt` reduction is required for correctness: with deeper `n_blocks` stacking on T_in=14, `kt=3` shrinks the temporal dimension to zero before the output layer. `kt=2` restores validity while maintaining the increased depth.

## Summary Table

| # | Model | Series | Graph | Attention | AMP | Key Differentiator |
|---|-------|--------|-------|-----------|-----|--------------------|
| 1 | LSTM | ST | No | No | No | Causal baseline |
| 2 | BiLSTM | ST | No | No | No | Bidirectional context over look-back |
| 3 | TPA-LSTM | ST | No | Pattern-CNN | No | Temporal pattern filters via 1-D CNN attention |
| 4 | CNN-LSTM | ST | No | No | No | CNN local feature extraction + LSTM sequence modelling |
| 5 | CNN-BiLSTM | ST | No | No | No | CNN + bidirectional context over CNN features |
| 6 | ST-LSTM | ST | No | No | No | Explicit parallel spatial + temporal streams |
| 7 | STGCN | Graph | Static Pearson | No | No | ST-Conv blocks on fixed correlation graph |
| 8 | ASTGCN | Attention | Static Pearson | Spatial+Temporal | Yes | Dual multi-head attention over nodes and timesteps |
| 9 | STSGCN | Graph | Static Pearson | No | No | Synchronous 3N×3N spatio-temporal graph |
| 10 | PDR-STGCN | Graph | Static+Dynamic | Dynamic | No | Periodicity encoding + dynamic relational graph mixing |
| 11 | MTGNN | Graph | Learned | No | No | Asymmetric end-to-end learned graph + dilated inception |
| 12 | STFGNN | Graph | Static×2 (spa+tem) | No | No | Dual spatial+temporal graphs fused by learned gate |
| 13 | Autoformer | Attention | No | Auto-Corr (FFT) | Yes | Decomposition + FFT-based periodic autocorrelation |
| 14 | Informer | Attention | No | ProbSparse | Yes | Sparse attention + distilling for efficiency |
| 15 | HMT-TSF | Hybrid | Static Pearson (GCN) | — | Yes | Feature-group fusion + Multi-Scale TCN + GCN + Regime gating + optional CatBoost residual correction |

---

## Experimental Results

All 14 baseline models were trained and evaluated across four experimental conditions: MCO-excluded (no-MCO) and MCO-inclusive, at look-back windows of 14, 28, and 56 days. Each model was then fine-tuned with revised architecture hyperparameters under the same conditions. Aggregate results are stored in `src/outputs/aggregate_results.csv`. HMT-TSF results are in `src/outputs/aggregate_hmttsf.csv`.

**Metric definitions:** Combined% = max(0, 100 − MAPE − MAE% − RMSE%); all percentage terms use mean-demand normalisation. Higher Combined%, R² and lower MAE/RMSE are better. Targets: Combined% ≥ 75%, R² ≥ 0.70.

---

### Baseline Performance (No Tuning) — No-MCO, Lookback 14

Results for all 14 base models on the no-MCO condition at the default look-back of 14 days, sorted by Combined%:

| Rank | Model | Combined% | MAPE% | MAE% | RMSE% | R² | MAE | RMSE |
|------|-------|-----------|-------|------|-------|-----|-----|------|
| 1 | Informer | 79.13 | 6.58 | 5.52 | 8.76 | 0.772 | 69,415 | 110,069 |
| 2 | BiLSTM | 79.04 | 6.52 | 5.77 | 8.67 | 0.776 | 72,460 | 109,007 |
| 3 | TPA-LSTM | 78.67 | 6.62 | 6.04 | 8.67 | 0.776 | 75,871 | 109,009 |
| 4 | LSTM | 78.13 | 6.71 | 6.18 | 8.98 | 0.760 | 77,719 | 112,801 |
| 5 | CNN-BiLSTM | 78.18 | 6.81 | 6.12 | 8.89 | 0.765 | 76,884 | 111,782 |
| 6 | ST-LSTM | 78.01 | 6.93 | 6.27 | 8.79 | 0.770 | 78,779 | 110,453 |
| 7 | CNN-LSTM | 77.64 | 6.92 | 6.32 | 9.12 | 0.752 | 79,375 | 114,632 |
| 8 | MTGNN | 77.62 | 7.02 | 6.47 | 8.90 | 0.764 | 81,289 | 111,802 |
| 9 | STGCN | 77.44 | 6.99 | 6.46 | 9.11 | 0.753 | 81,203 | 114,496 |
| 10 | Autoformer | 76.71 | 7.15 | 6.82 | 9.33 | 0.741 | 85,660 | 117,208 |
| 11 | STFGNN | 75.89 | 7.58 | 6.86 | 9.66 | 0.722 | 86,240 | 121,428 |
| 12 | CNN-LSTM-Augmented | 75.36 | 7.77 | 7.19 | 9.68 | 0.721 | 90,356 | 121,639 |
| 14 | CNN-LSTM-Parallel | 74.78 | 8.01 | 7.46 | 9.75 | 0.717 | 93,780 | 122,536 |
| 15 | ASTGCN | 74.34 | 7.98 | 7.54 | 10.14 | 0.694 | 94,737 | 127,449 |
| 16 | STSGCN | 73.33 | 8.60 | 7.77 | 10.30 | 0.684 | 97,689 | 129,451 |
| 17 | PDR-STGCN | 67.75 | 10.26 | 9.98 | 12.01 | 0.571 | 125,481 | 150,877 |

**Observations:**
- LSTM-family models cluster between 75–79%, with simpler recurrent architectures (BiLSTM, TPA-LSTM) outperforming more complex CNN hybrids and graph models.
- Graph-based models show mixed results: MTGNN (77.62%) and STGCN (77.44%) are competitive via their learned/fixed adjacency, but STSGCN (73.33%) and PDR-STGCN (67.75%) lag — the features-as-nodes graph construction is harder to learn in synchronous or dynamic settings at base capacity.
- 10 of 14 models exceed the 75% Combined% target at baseline.

---

### Tuned Performance — No-MCO, Lookback 14

Fine-tuned model performance on no-MCO condition at look-back 14, sorted by Combined%. Δ shows the gain/loss versus the corresponding base configuration:

| Rank | Model | Combined% | MAPE% | MAE% | RMSE% | R² | MAE | RMSE | Δ Combined% |
|------|-------|-----------|-------|------|-------|-----|-----|------|-------------|
| 1 | LSTM | 81.04 | 5.70 | 5.02 | 8.24 | 0.798 | 63,117 | 103,516 | +2.91 |
| 2 | Informer | 79.99 | 6.16 | 5.20 | 8.64 | 0.778 | 65,360 | 108,628 | +0.86 |
| 3 | TPA-LSTM | 79.95 | 6.19 | 5.31 | 8.55 | 0.782 | 66,703 | 107,475 | +1.28 |
| 4 | BiLSTM | 79.44 | 6.39 | 5.85 | 8.33 | 0.794 | 73,516 | 104,667 | +0.40 |
| 5 | Autoformer | 79.27 | 6.37 | 5.63 | 8.74 | 0.773 | 70,733 | 109,856 | +2.55 |
| 6 | ST-LSTM | 79.13 | 6.51 | 5.63 | 8.72 | 0.774 | 70,796 | 109,604 | +1.12 |
| 7 | ASTGCN | 78.82 | 6.35 | 5.73 | 9.09 | 0.754 | 72,053 | 114,259 | +4.48 |
| 8 | CNN-LSTM-Augmented | 78.57 | 6.57 | 5.85 | 9.01 | 0.758 | 73,464 | 113,271 | +3.21 |
| 9 | CNN-BiLSTM | 78.53 | 6.78 | 6.01 | 8.68 | 0.776 | 75,520 | 109,064 | +0.35 |
| 10 | CNN-LSTM | 78.52 | 6.64 | 6.02 | 8.82 | 0.768 | 75,654 | 110,907 | +0.88 |
| 11 | CNN-LSTM-Parallel | 77.52 | 6.89 | 6.14 | 9.45 | 0.734 | 77,128 | 118,805 | +2.74 |
| 13 | PDR-STGCN | 77.78 | 6.73 | 6.22 | 9.26 | 0.745 | 78,230 | 116,379 | +10.03 |
| 14 | MTGNN | 77.44 | 7.02 | 6.46 | 9.08 | 0.755 | 81,209 | 114,079 | −0.17 |
| 15 | STSGCN | 77.14 | 7.04 | 6.33 | 9.49 | 0.732 | 79,494 | 119,266 | +3.81 |
| 16 | STGCN | 76.33 | 7.39 | 7.02 | 9.26 | 0.745 | 88,235 | 116,410 | −1.11 |
| 17 | STFGNN | 74.55 | 8.03 | 7.42 | 10.00 | 0.702 | 93,248 | 125,728 | −1.35 |

**Observations:**
- 13 of 16 configurations improved; three regressed (MTGNN −0.17%, STGCN −1.11%, STFGNN −1.35%). For STGCN and STFGNN the increased dropout and deeper stacking introduce noise into a fixed pre-computed graph that cannot adapt to the changed capacity.
- PDR-STGCN showed the largest gain (+10.03%), recovering from the worst base performance (67.75%) to mid-tier (77.78%). The expanded hidden size and deeper attention-based dynamic graph combine effectively once provided with sufficient capacity.
- ASTGCN also benefited substantially (+4.48%) from doubling d_model and n_blocks.
- Combined% range compresses from 67.75–79.13% (base) to 74.55–81.04% (tuned), indicating tuning reduces inter-model variance.
- All 16 tuned configurations now exceed the 74% floor; 14 of 16 exceed 77%.

---

### Tuned Performance — MCO-Inclusive, Lookback 14

Including the MCO pandemic period substantially degrades all models, revealing which architectures are robust to distribution shift. Δ shows the regression versus no-MCO tuned at lb14:

| Rank | Model | Combined% | MAPE% | MAE% | RMSE% | R² | MAE | RMSE | Δ vs no-MCO |
|------|-------|-----------|-------|------|-------|-----|-----|------|-------------|
| 1 | Informer | 75.35 | 7.68 | 7.21 | 9.76 | 0.738 | 88,924 | 120,400 | −4.64 |
| 2 | ST-LSTM | 75.05 | 7.93 | 7.05 | 9.98 | 0.726 | 86,892 | 123,067 | −4.08 |
| 3 | TPA-LSTM | 74.36 | 8.13 | 7.38 | 10.14 | 0.718 | 90,972 | 125,010 | −5.59 |
| 4 | CNN-LSTM-Parallel | 72.00 | 8.76 | 8.17 | 11.06 | 0.664 | 100,763 | 136,434 | −5.52 |
| 5 | LSTM | 71.84 | 8.96 | 8.55 | 10.66 | 0.688 | 105,421 | 131,404 | −9.20 |
| 6 | BiLSTM | 70.66 | 9.29 | 9.01 | 11.03 | 0.666 | 111,167 | 135,981 | −8.78 |
| 7 | Autoformer | 66.40 | 10.75 | 10.40 | 12.45 | 0.574 | 128,254 | 153,503 | −12.87 |
| 8 | CNN-BiLSTM | 67.53 | 10.22 | 10.20 | 12.05 | 0.601 | 125,837 | 148,570 | −11.00 |
| 9 | CNN-LSTM-Augmented | 67.30 | 10.26 | 10.26 | 12.18 | 0.593 | 126,532 | 150,165 | −11.27 |
| 10 | STFGNN | 63.62 | 11.53 | 11.23 | 13.62 | 0.490 | 138,496 | 167,971 | −10.93 |
| 12 | MTGNN | 63.06 | 12.30 | 11.02 | 13.62 | 0.490 | 135,907 | 167,969 | −14.38 |
| 13 | CNN-LSTM | 61.75 | 12.11 | 12.28 | 13.87 | 0.472 | 151,387 | 170,989 | −16.77 |
| 14 | ASTGCN | 60.50 | 12.63 | 12.50 | 14.37 | 0.433 | 154,102 | 177,240 | −18.32 |
| 15 | PDR-STGCN | 59.77 | 13.25 | 12.46 | 14.52 | 0.421 | 153,661 | 179,039 | −18.01 |
| 16 | STSGCN | 55.18 | 14.29 | 14.36 | 16.17 | 0.281 | 177,109 | 199,452 | −21.96 |
| 17 | STGCN | 53.79 | 14.42 | 14.96 | 16.84 | 0.221 | 184,451 | 207,637 | −22.54 |

**Observations:**
- Graph-based models are most vulnerable to MCO-induced distribution shift: STGCN (−22.54%), STSGCN (−21.96%), PDR-STGCN (−18.01%). The static Pearson-correlation adjacency is fitted on non-MCO training data; the COVID-driven ridership collapse invalidates the learned inter-feature correlations at test time.
- Informer (−4.64%) and ST-LSTM (−4.08%) are the most MCO-resilient baselines, losing less than 5%. Their attention and parallel-stream designs capture structural relationships that partially transfer across the regime boundary.
- LSTM (−9.20%) and BiLSTM (−8.78%) degrade more than TPA-LSTM (−5.59%), suggesting the temporal pattern attention mechanism provides additional stability under regime shift.
- Only Informer (75.35%), ST-LSTM (75.05%), and TPA-LSTM (74.36%) maintain Combined% near or above the 75% target under MCO conditions. All graph-based models fall below 64%.

---

### Cross-Lookback Analysis — Tuned No-MCO

Performance across look-back windows 14, 28, and 56 days for tuned models under no-MCO conditions. Best lookback highlighted:

| Model | lb14 Combined% | lb28 Combined% | lb56 Combined% | Best |
|-------|----------------|----------------|----------------|------|
| LSTM | 81.04 | 79.79 | 79.24 | **14** |
| Informer | 79.99 | **80.08** | 77.51 | **28** |
| TPA-LSTM | 79.95 | 79.41 | 79.33 | **14** |
| BiLSTM | 79.44 | 78.33 | 75.90 | **14** |
| Autoformer | 79.27 | 74.65 | 58.59 | **14** |
| ST-LSTM | 79.13 | 77.85 | 78.51 | **14** |
| ASTGCN | 78.82 | 68.22 | 73.60 | **14** |
| CNN-BiLSTM | 78.53 | 75.16 | 55.14 | **14** |
| CNN-LSTM | 78.52 | 76.60 | 77.83 | **14** |
| PDR-STGCN | 77.78 | 76.10 | 52.13 | **14** |
| MTGNN | 77.44 | 77.35 | **78.13** | **56** |
| STSGCN | 77.14 | 73.53 | 50.13 | **14** |
| STGCN | 76.33 | **77.17** | 74.52 | **28** |
| STFGNN | 74.55 | 72.59 | 42.19 | **14** |

**Observations:**
- lb14 is optimal for 11 of 14 models; longer lookbacks generally degrade performance.
- Autoformer (58.59%), STFGNN (42.19%), and STSGCN (50.13%) collapse at lb56 — FFT autocorrelation and synchronous graph operations amplify noise over very long input sequences.
- MTGNN uniquely prefers lb56 (78.13%). Its learned asymmetric adjacency and multi-scale dilated inception appear to extract additional benefit from quarterly temporal context, unlike models relying on pre-computed static graphs.
- Informer and STGCN marginally prefer lb28, consistent with ProbSparse attention capturing slightly longer-range dependencies without performance penalty.

---

### Tuning Effectiveness Summary

Absolute Combined% improvement from base→tuned at nomco lb14, sorted by impact:

| Model | Base | Tuned | Δ | Notes |
|-------|------|-------|---|-------|
| PDR-STGCN | 67.75 | 77.78 | +10.03 | Dynamic graph benefits strongly from doubled hidden + deeper dk |
| ASTGCN | 74.34 | 78.82 | +4.48 | Doubled d_model and n_blocks unlock attention capacity |
| STSGCN | 73.33 | 77.14 | +3.81 | Deeper synchronous graph benefits from larger hidden |
| CNN-LSTM-Augmented | 75.36 | 78.57 | +3.21 | Richer CNN filters improve skip-connection quality |
| CNN-LSTM-Parallel | 74.78 | 77.52 | +2.74 | Deeper CNN filters improve both branches |
| Autoformer | 76.71 | 79.27 | +2.55 | Larger d_ff and e_layers enhance decomposition quality |
| LSTM | 78.13 | 81.04 | +2.91 | Doubled hidden/layers extracts more sequential capacity |
| CNN-LSTM | 77.64 | 78.52 | +0.88 | Moderate gain from increased CNN filters |
| TPA-LSTM | 78.67 | 79.95 | +1.28 | Larger hidden + filters improve pattern coverage |
| ST-LSTM | 78.01 | 79.13 | +1.12 | Bigger spatial and temporal streams both contribute |
| Informer | 79.13 | 79.99 | +0.86 | Already near-optimal at base; sparse attention limits tuning upside |
| BiLSTM | 79.04 | 79.44 | +0.40 | Already near ceiling for bidirectional recurrence on this dataset |
| CNN-BiLSTM | 78.18 | 78.53 | +0.35 | Marginal gain; combined architecture already effective at base |
| MTGNN | 77.62 | 77.44 | −0.17 | Learned adjacency already near-optimal; extra layers add noise |
| STGCN | 77.44 | 76.33 | −1.11 | Increased dropout and `kt` reduction over-regularise fixed graph |
| STFGNN | 75.89 | 74.55 | −1.35 | High dropout (0.30) over-regularises gated dual-graph fusion |

---

### HMT-TSF vs Best Tuned Baselines

For a direct cross-model comparison at the canonical lb14 configuration, see `src/models/hybrid/HMT-TSF.md` → Achieved Results. Key summary:

| Condition | Best Baseline | Best Baseline Combined% | HMT-TSF Combined% | Δ |
|-----------|---------------|------------------------|-------------------|---|
| No-MCO, lb14 | LSTM (tuned) | 81.04 | 80.99 | −0.05 |
| MCO, lb14 | Informer (tuned) | 75.35 | 77.17 | +1.82 |
| No-MCO, lb28 | Informer (tuned) | 80.08 | 79.37 | −0.71 |
| No-MCO, lb56 | TPA-LSTM (tuned) | 79.33 | 78.93 | −0.40 |

Under no-MCO conditions HMT-TSF is effectively tied with the best baselines (within 0.05–0.71%). Under MCO conditions it leads by 1.82 percentage points at lb14, confirming the regime gating mechanism provides a meaningful advantage during COVID-disruption periods.

---

## Journal References

Scopus-indexed journal articles (2022–2027) cited in each model script's docstring, one per architecture.

### Series 1 — Spatio-Temporal (LSTM-Family)

| Model | Citation |
|-------|----------|
| LSTM | Strigula, M. (2026). Beyond traditional forecasting methods: Evaluating LSTM performance on diverse time series. *Mathematics*, 14(5), 838. https://doi.org/10.3390/math14050838 |
| BiLSTM | Alajmi, M. S., & Almutairi, S. M. (2025). Intelligent traffic congestion forecasting using BiLSTM and adaptive secretary bird optimizer for sustainable urban transportation. *Scientific Reports*, 15. https://doi.org/10.1038/s41598-025-02933-9 |
| TPA-LSTM | Wei, L., Guo, D., Chen, Z., Yang, J., & Feng, T. (2023). Forecasting short-term passenger flow of subway stations based on the temporal pattern attention mechanism and the long short-term memory network. *ISPRS International Journal of Geo-Information*, 12(1), 25. https://doi.org/10.3390/ijgi12010025 |
| CNN-LSTM | Topilin, I., Jiang, J., Feofilova, A., & Beskopylny, N. (2025). Traffic flow prediction via a hybrid CPO-CNN-LSTM-attention architecture. *Smart Cities*, 8(5), 148. https://doi.org/10.3390/smartcities8050148 |
| CNN-BiLSTM | Chen, W., Yang, Z., Xu, G., & Sun, Y. (2022). Short-term traffic flow prediction based on CNN-BiLSTM with multicomponent information. *Applied Sciences*, 12(17), 8714. https://doi.org/10.3390/app12178714 |
| ST-LSTM | Cui, H., Si, B., Chi, D., Li, Y., Li, G., & Chen, Y. (2025). Short-term passenger flow prediction for urban rail systems: A deep learning approach utilizing multi-source big data. *PLOS ONE*, 20(1), e0333094. https://doi.org/10.1371/journal.pone.0333094 |

### Series 2 — Graph-Based

| Model | Citation |
|-------|----------|
| STGCN | Deng, H. (2025). Traffic-forecasting model with spatio-temporal kernel. *Electronics*, 14(7), 1410. https://doi.org/10.3390/electronics14071410 |
| MTGNN | Wu, Z., Liu, X., & Zhang, X. (2025). Multi dynamic temporal representation graph convolutional network for traffic flow prediction. *Scientific Reports*, 15, 16734. https://doi.org/10.1038/s41598-025-01157-1 |
| STSGCN | Chen, L., Ren, Q., Zeng, J., Zou, F., Luo, S., Tian, J., & Xing, Y. (2023). CSFPre: Expressway key sections based on CEEMDAN-STSGCN-FCM during the holidays for traffic flow prediction. *PLOS ONE*, 18(4), e0283898. https://doi.org/10.1371/journal.pone.0283898 |
| STFGNN | Chang, J., Yin, J., Hao, Y., & Gao, C. (2025). STFDSGCN: Spatio-temporal fusion graph neural network based on dynamic sparse graph convolution GRU for traffic flow forecast. *Sensors*, 25(11), 3446. https://doi.org/10.3390/s25113446 |
| PDR-STGCN | (2026). PDR-STGCN: An enhanced STGCN with multi-scale periodic fusion and a dynamic relational graph for traffic forecasting. *Systems*, 14(1), 102. https://doi.org/10.3390/systems14010102 |

### Series 3 — Attention-Based

| Model | Citation |
|-------|----------|
| ASTGCN | Cui, Z., Zhang, J., Noh, G., & Park, H. J. (2023). ADSTGCN: A dynamic adaptive deeper spatio-temporal graph convolutional network for multi-step traffic forecasting. *Sensors*, 23(15), 6950. https://doi.org/10.3390/s23156950 |
| Autoformer | Ma, X., & Zhang, H. (2025). Time series forecasting method based on multi-scale feature fusion and Autoformer. *Applied Sciences*, 15(7), 3768. https://doi.org/10.3390/app15073768 |
| Informer | Song, Y., Luo, R., Zhou, T., Zhou, C., & Su, R. (2024). Graph attention Informer for long-term traffic flow prediction under the impact of sports events. *Sensors*, 24(15), 4796. https://doi.org/10.3390/s24154796 |
