# Model Descriptions

This document describes all 15 deep-learning models used in the Malaysian transit ridership forecasting study. Models are divided into three series: **Spatio-Temporal (LSTM-family)**, **Graph-Based**, and **Attention-Based**. All models share the same input/output dimensions (T_in=14/28/56 look-back via `--lookback`, T_out=7 forecast horizon), the same dataset (59 features across 8 spatio-temporal sources), and the same evaluation metrics (Combined%, MAPE%, MAE%, RMSE%, R², MAE, RMSE).

**Shared training optimizations (all 15 models):** AdamW optimiser (decoupled weight decay), HuberLoss (default, selectable via `--loss {mse,huber,mae}`), and a 5-epoch linear LR warm-up before ReduceLROnPlateau (configurable via `--warmup-epochs`). Architecture and hyperparameter values are unchanged.

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

### 8. MTGNN — Multi-Scale Temporal Graph Neural Network

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

### 10. STFGNN — Spatial-Temporal Fusion Graph Neural Network

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

### 11. PDR-STGCN — Periodicity-Aware Dynamic Relational STGCN

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

Located in `src/models/attention-based/`. These models use attention mechanisms as the primary modelling tool. They use AMP (fp16 + GradScaler) to fit within 6 GB VRAM. Note: TPA-LSTM is listed in Series 1 (LSTM-family) because its primary encoder is still a recurrent LSTM; attention is used only as a decoder module.

---

### 12. ASTGCN — Attention-Based Spatio-Temporal Graph Convolutional Network

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

### 13. TFT — Temporal Fusion Transformer

**Script:** `tft.py`

A fully transformer-style model designed for interpretable multi-horizon forecasting. All 59 features are treated as observed past inputs (no static or future covariates in this adaptation). Three core components:

- **Variable Selection Network (VSN):** Softmax-weighted selection over input features per timestep, identifying the most predictive features for the forecast.
- **LSTM Encoder:** Captures local sequential dynamics after feature selection.
- **Multi-Head Self-Attention:** Captures long-range temporal dependencies across the look-back window.

**Architecture:**
```
X (B, T_in, F)
  → VSN         (per-timestep feature selection) → (B, T_in, d_model)
  → LSTM encoder (local processing)              → (B, T_in, d_model)
  → Multi-head SA × n_attn_layers [Attn + GAN + FFN + GAN]
  → Mean pool                                    → (B, d_model)
  → MLP head                                     → (B, T_out)
```

**Key design choices:**
- Gated Residual Network (GRN) as the core building block: ELU + GLU gating throughout.
- Gated Add-and-Norm (GAN) residual connections for stable gradient flow.
- AMP (fp16) + GradScaler.
- Default: `d_model=64`, `n_heads=4`, `n_lstm_layers=1`, `n_attn_layers=2`, `dropout=0.1`, `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`.

**Difference from ASTGCN:** Pure transformer architecture with no explicit graph structure; uses VSN for learned feature selection rather than graph-based spatial modelling; combines recurrent local encoding with global self-attention.

---

### 14. Autoformer

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

**Difference from TFT:** Decomposes the series into trend and seasonal components explicitly and models periodicity via FFT-based autocorrelation rather than dot-product attention. Better suited to strongly periodic signals like daily/weekly ridership.

---

### 15. Informer

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
→ MTGNN (9-way) → STSGCN (10-way) → STFGNN (11-way) → PDR-STGCN (12-way)
→ ASTGCN (13-way) → TFT (14-way) → Autoformer (15-way) → Informer (16-way)
```

---

## Fine-Tuned Model Series

Fifteen mirrored fine-tuned variants of the above models are located in three new folders:

| Folder | Models |
|--------|--------|
| `src/models/spatio-temporal-tuned/` | LSTM, BiLSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM |
| `src/models/graph-tuned/` | STGCN, MTGNN, STSGCN, STFGNN, PDR-STGCN |
| `src/models/attention-tuned/` | TPA-LSTM, ASTGCN, TFT, Autoformer, Informer |

**Tuning strategy (Option C):** Best-configuration selection (MCO-exclusion setting + lookback window) combined with revised architecture hyperparameters. Training hyperparameters (`epochs`, `batch_size`, `lr`, `patience`, `weight_decay`) are unchanged.

**Output directories** use the `_tuned` suffix: `src/outputs/{model_name}_tuned/`.

**Comparison:** each tuned run compares against all 15 base model results (not the tuned runs). Fine-tuned=yes tagging is applied externally by `src/utils/aggregate_results.py`.

### Architecture Changes at a Glance

| Model | Key Changes |
|-------|-------------|
| LSTM | hidden 64→128, layers 1→2, dropout 0.10→0.15 |
| BiLSTM | hidden 64→128, layers 1→2, dropout 0.10→0.15 |
| CNN-LSTM | cnn_filters 32→64, dropout 0.10→0.20 |
| CNN-BiLSTM | hidden 64→128, cnn_filters 32→64, dropout 0.10→0.25 |
| ST-LSTM | hidden 64→128, spatial_hidden 32→64, dropout 0.10→0.20 |
| STGCN | hidden 128→256, n_blocks 2→3, kt 3→2, dropout 0.10→0.15 |
| MTGNN | hidden 32→64, skip_ch 64→128, n_layers 3→4, dropout 0.10→0.15 |
| STSGCN | hidden 64→128, n_layers 2→3, dropout 0.10→0.20 |
| STFGNN | hidden 64→128, n_layers 3→4, dropout 0.10→0.30 |
| PDR-STGCN | hidden 128→256, n_blocks 2→3, kt 3→2, dk 32→64, dropout 0.10→0.20 |
| TPA-LSTM | hidden 64→128, filters 32→64, dropout 0.10→0.15 |
| ASTGCN | d_model 64→128, n_heads 4→8, n_blocks 2→3, dropout 0.10→0.20 |
| TFT | d_model 64→128, n_heads 4→8, n_lstm_layers 1→2, n_attn_layers 2→3, dropout 0.10→0.25 |
| Autoformer | d_model 64→128, n_heads 4→8, e_layers 2→3, d_ff 128→256, dropout 0.10→0.20 |
| Informer | d_model 64→128, n_heads 4→8, e_layers 2→3, d_ff 128→256, dropout 0.10→0.15 |

Full per-model rationale, run commands, and constraint notes are in `src/models/TUNED-MODEL.md`.

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
| 8 | MTGNN | Graph | Learned | No | No | Asymmetric end-to-end learned graph + dilated inception |
| 9 | STSGCN | Graph | Static Pearson | No | No | Synchronous 3N×3N spatio-temporal graph |
| 10 | STFGNN | Graph | Static×2 (spa+tem) | No | No | Dual spatial+temporal graphs fused by learned gate |
| 11 | PDR-STGCN | Graph | Static+Dynamic | Dynamic | No | Periodicity encoding + dynamic relational graph mixing |
| 12 | ASTGCN | Attention | Static Pearson | Spatial+Temporal | Yes | Dual multi-head attention over nodes and timesteps |
| 13 | TFT | Attention | No | Self-Attn | Yes | Variable selection + LSTM encoder + Transformer decoder |
| 14 | Autoformer | Attention | No | Auto-Corr (FFT) | Yes | Decomposition + FFT-based periodic autocorrelation |
| 15 | Informer | Attention | No | ProbSparse | Yes | Sparse attention + distilling for efficiency |
