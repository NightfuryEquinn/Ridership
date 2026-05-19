# Tuned Model Descriptions

This document describes the 15 fine-tuned deep-learning model variants used in the Malaysian transit ridership forecasting study. Each tuned variant mirrors its corresponding baseline model but with revised architecture hyperparameters derived from the initial no-fine-tuning experiment results.

---

## Tuning Strategy (Option C)

**Option C = Best-Configuration Selection + Revised Architecture Hyperparameters.**

For each model, the tuned variant applies two changes:

1. **Best-configuration selection** — selects the MCO-exclusion setting and lookback window (14/28/56) that produced the highest Combined% in the baseline run. The tuned script defaults to this configuration.
2. **Revised architecture hyperparameters** — increases model capacity (hidden size, layers, filters, heads) and adjusts regularisation (dropout) based on two rules:
   - *Capacity:* models with lower baseline Combined% receive larger hidden dimensions and/or more layers.
   - *Regularisation:* models with higher variance across lookback windows receive higher dropout.

**Training hyperparameters are unchanged** — `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`, `weight_decay=1e-4` remain identical across all 15 tuned variants.

**Output directories** use the `_tuned` suffix (e.g. `src/outputs/lstm_tuned/`) to keep tuned runs separate from baseline runs.

**Comparison chain:** each tuned model compares against all 15 *base* model runs (not the tuned runs). This makes the comparison consistent with the original evaluation order.

**Fine-tuned tagging** is applied externally via `src/utils/aggregate_results.py` (not inside individual model scripts).

---

## Tuned Folder Structure

```
src/models/
├── spatio-temporal-tuned/   # LSTM, BiLSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM
├── graph-tuned/             # STGCN, MTGNN, STSGCN, STFGNN, PDR-STGCN
└── attention-tuned/         # TPA-LSTM, ASTGCN, TFT, Autoformer, Informer
```

Outputs are written to `src/outputs/{model_name}_tuned/`.

---

## Series 1 — Spatio-Temporal-Tuned (`src/models/spatio-temporal-tuned/`)

---

### 1. LSTM (Tuned)

**Script:** `src/models/spatio-temporal-tuned/lstm.py`

| Parameter | Base | Tuned | Rationale |
|-----------|------|-------|-----------|
| `hidden` | 64 | **128** | Capacity increase; baseline Combined% was low |
| `layers` | 1 | **2** | Deeper stack improves temporal abstraction |
| `dropout` | 0.10 | **0.15** | Moderate increase for 2-layer regularisation |

**Run command (recommended):**
```bash
python src/models/spatio-temporal-tuned/lstm.py
```

---

### 2. BiLSTM (Tuned)

**Script:** `src/models/spatio-temporal-tuned/bilstm.py`

| Parameter | Base | Tuned | Rationale |
|-----------|------|-------|-----------|
| `hidden` | 64 | **128** | Capacity increase; baseline Combined% was low |
| `layers` | 1 | **2** | Deeper bidirectional stack |
| `dropout` | 0.10 | **0.15** | Moderate increase for 2-layer regularisation |

**Run command (recommended):**
```bash
python src/models/spatio-temporal-tuned/bilstm.py
```

---

### 3. CNN-LSTM (Tuned)

**Script:** `src/models/spatio-temporal-tuned/cnnlstm.py`

| Parameter | Base | Tuned | Rationale |
|-----------|------|-------|-----------|
| `cnn_filters` | 32 | **64** | Wider feature maps; baseline Combined% was low |
| `dropout` | 0.10 | **0.20** | MED-HIGH; sequential mode had high variance across lookbacks |
| `hidden` | 64 | 64 (unchanged) | LSTM capacity adequate; CNN widening is the primary lever |
| `layers` | 1 | 1 (unchanged) | Single LSTM layer is sufficient post-CNN |

**Mode-specific best lookback (MCO excluded):**

| Mode | Best Lookback | Rationale |
|------|---------------|-----------|
| `sequential` | 14 | Lowest MAPE at lb14 with MCO excluded |
| `parallel` | 28 | Wider context improves parallel LSTM stream |
| `augmented` | 14 | Skip connection compensates for shorter window |

**Run commands (recommended — one per mode):**
```bash
# Sequential — best: exclude MCO, lb14
python src/models/spatio-temporal-tuned/cnnlstm.py --mode sequential --lookback 14

# Parallel — best: exclude MCO, lb28
python src/models/spatio-temporal-tuned/cnnlstm.py --mode parallel --lookback 28

# Augmented — best: exclude MCO, lb14
python src/models/spatio-temporal-tuned/cnnlstm.py --mode augmented --lookback 14
```

---

### 4. CNN-BiLSTM (Tuned)

**Script:** `src/models/spatio-temporal-tuned/cnnbilstm.py`

| Parameter | Base | Tuned | Rationale |
|-----------|------|-------|-----------|
| `hidden` | 64 | **128** | Capacity increase; baseline Combined% was low |
| `cnn_filters` | 32 | **64** | Wider feature maps to match larger BiLSTM |
| `dropout` | 0.10 | **0.25** | HIGH; highest variance across lookbacks of all ST models |
| `layers` | 1 | 1 (unchanged) | BiLSTM width increase is the primary lever |

**Run command (recommended):**
```bash
python src/models/spatio-temporal-tuned/cnnbilstm.py
```

---

### 5. ST-LSTM (Tuned)

**Script:** `src/models/spatio-temporal-tuned/stlstm.py`

| Parameter | Base | Tuned | Rationale |
|-----------|------|-------|-----------|
| `hidden` | 64 | **128** | Capacity increase; baseline Combined% was low |
| `spatial_hidden` | 32 | **64** | Scaled proportionally with temporal hidden |
| `dropout` | 0.10 | **0.20** | MED-HIGH; moderate variance across lookbacks |
| `layers` | 1 | 1 (unchanged) | Dual-stream design compensates for single layer |

**Run command (recommended):**
```bash
python src/models/spatio-temporal-tuned/stlstm.py
```

---

## Series 2 — Graph-Tuned (`src/models/graph-tuned/`)

---

### 6. STGCN (Tuned)

**Script:** `src/models/graph-tuned/stgcn.py`

| Parameter | Base | Tuned | Rationale |
|-----------|------|-------|-----------|
| `hidden` | 128 | **256** | Higher capacity for graph-based ST modelling |
| `n_blocks` | 2 | **3** | Deeper stack captures longer-range graph patterns |
| `kt` | 3 | **2** | Reduced to prevent temporal dimension collapse with n_blocks=3 |
| `dropout` | 0.10 | **0.15** | Moderate regularisation for deeper network |

**Constraint note:** With `T_in=14`, each temporal gated conv shrinks T by `kt-1`. Two temporal convs per ST block. With `n_blocks=3` and `kt=3`: `T_after = 14 - 2×2×3 = 2` — valid but leaves very little temporal context. With `kt=2`: `T_after = 14 - 2×1×3 = 8`, healthier. `kt` reduced to 2 to ensure adequate temporal context at the output.

**Run command (recommended):**
```bash
python src/models/graph-tuned/stgcn.py
```

---

### 7. MTGNN (Tuned)

**Script:** `src/models/graph-tuned/mtgnn.py`

| Parameter | Base | Tuned | Rationale |
|-----------|------|-------|-----------|
| `hidden` | 32 | **64** | Capacity increase; baseline was under-parameterised |
| `skip_ch` | 64 | **128** | Skip channel width scales with doubled hidden |
| `n_layers` | 3 | **4** | Additional mixing layer for multi-hop graph propagation |
| `dropout` | 0.10 | **0.15** | Moderate regularisation for deeper model |

**Run command (recommended):**
```bash
python src/models/graph-tuned/mtgnn.py
```

---

### 8. STSGCN (Tuned)

**Script:** `src/models/graph-tuned/stsgcn.py`

| Parameter | Base | Tuned | Rationale |
|-----------|------|-------|-----------|
| `hidden` | 64 | **128** | Capacity increase; baseline Combined% was low |
| `n_layers` | 2 | **3** | Deeper synchronous graph convolution |
| `dropout` | 0.10 | **0.20** | MED-HIGH variance across lookbacks |

**Constraint note:** Each STSGCL layer shrinks T by 2 (sliding window of 3). With `T_in=14` and `n_layers=3`: `T_after = 14 - 2×3 = 8`. Valid.

**Run command (recommended):**
```bash
python src/models/graph-tuned/stsgcn.py
```

---

### 9. STFGNN (Tuned)

**Script:** `src/models/graph-tuned/stfgnn.py`

| Parameter | Base | Tuned | Rationale |
|-----------|------|-------|-----------|
| `hidden` | 64 | **128** | Capacity increase; dual-graph fusion benefits from wider channels |
| `n_layers` | 3 | **4** | More fusion blocks for richer spatial-temporal interaction |
| `dropout` | 0.10 | **0.30** | HIGH; highest variance of all graph models across lookbacks |

**Run command (recommended):**
```bash
python src/models/graph-tuned/stfgnn.py
```

---

### 10. PDR-STGCN (Tuned)

**Script:** `src/models/graph-tuned/pdr_stgcn.py`

| Parameter | Base | Tuned | Rationale |
|-----------|------|-------|-----------|
| `hidden` | 128 | **256** | PDR-STGCN is the most expressive graph model; benefits most from capacity |
| `n_blocks` | 2 | **3** | Deeper periodic-dynamic stack |
| `kt` | 3 | **2** | Same T_after constraint as STGCN tuned (see STGCN note above) |
| `dk` | 32 | **64** | Dynamic attention key/query dim scaled up with hidden |
| `dropout` | 0.10 | **0.20** | MED-HIGH; moderate variance across lookbacks |

**Run command (recommended):**
```bash
python src/models/graph-tuned/pdr_stgcn.py
```

---

## Series 3 — Attention-Tuned (`src/models/attention-tuned/`)

---

### 11. TPA-LSTM (Tuned)

**Script:** `src/models/attention-tuned/tpalstm.py`

| Parameter | Base | Tuned | Rationale |
|-----------|------|-------|-----------|
| `hidden` | 64 | **128** | Capacity increase; baseline Combined% was low |
| `filters` | 32 | **64** | Wider pattern filters match larger LSTM hidden |
| `dropout` | 0.10 | **0.15** | Moderate increase; low-to-medium variance across lookbacks |

**Note:** No AMP used (same as base TPA-LSTM). Saves `attn_weights_test.npy` and `attention_heatmap.png` as in the base model.

**Run command (recommended):**
```bash
python src/models/attention-tuned/tpalstm.py
```

---

### 12. ASTGCN (Tuned)

**Script:** `src/models/attention-tuned/astgcn.py`

| Parameter | Base | Tuned | Rationale |
|-----------|------|-------|-----------|
| `d_model` | 64 | **128** | Larger embedding dimension for spatial-temporal attention |
| `n_heads` | 4 | **8** | More attention heads to diversify node/timestep weighting |
| `n_blocks` | 2 | **3** | Deeper dual-attention stack |
| `dropout` | 0.10 | **0.20** | MED-HIGH variance across lookbacks for attention models |

**Run command (recommended):**
```bash
python src/models/attention-tuned/astgcn.py
```

---

### 13. TFT (Tuned)

**Script:** `src/models/attention-tuned/tft.py`

| Parameter | Base | Tuned | Rationale |
|-----------|------|-------|-----------|
| `d_model` | 64 | **128** | Larger embedding for VSN and LSTM encoder |
| `n_heads` | 4 | **8** | More attention heads for temporal self-attention |
| `n_lstm_layers` | 1 | **2** | Deeper LSTM encoder for local sequential processing |
| `n_attn_layers` | 2 | **3** | Additional transformer layer for long-range dependencies |
| `dropout` | 0.10 | **0.25** | HIGH; TFT had highest variance of all attention-based models |

**Run command (recommended):**
```bash
python src/models/attention-tuned/tft.py
```

---

### 14. Autoformer (Tuned)

**Script:** `src/models/attention-tuned/autoformer.py`

| Parameter | Base | Tuned | Rationale |
|-----------|------|-------|-----------|
| `d_model` | 64 | **128** | Larger embedding for decomposition and autocorrelation |
| `n_heads` | 4 | **8** | More heads for FFT-based autocorrelation |
| `e_layers` | 2 | **3** | Deeper encoder improves periodic dependency capture |
| `d_ff` | 128 | **256** | Scaled up proportionally with d_model |
| `dropout` | 0.10 | **0.20** | MED-HIGH variance across lookbacks |

**Note:** FFT operations remain wrapped in explicit `float32` cast inside `autocast` for numerical stability — identical to the base model.

**Run command (recommended):**
```bash
python src/models/attention-tuned/autoformer.py
```

---

### 15. Informer (Tuned)

**Script:** `src/models/attention-tuned/informer.py`

| Parameter | Base | Tuned | Rationale |
|-----------|------|-------|-----------|
| `d_model` | 64 | **128** | Larger embedding for ProbSparse attention |
| `n_heads` | 4 | **8** | More heads for sparser attention patterns |
| `e_layers` | 2 | **3** | Deeper encoder; distilling produces 14→7→4 (valid with T_in=14) |
| `d_ff` | 128 | **256** | Scaled up proportionally with d_model |
| `dropout` | 0.10 | **0.15** | Low-to-moderate variance across lookbacks |

**Distilling note with e_layers=3 and T_in=14:** Each Conv+MaxPool(2) distilling step halves the sequence length. With e_layers=3, two distilling steps occur: 14→7→4. The final encoder layer receives sequence length 4 — small but valid; ProbSparse attention at this length degenerates gracefully to near-full attention (no sparsity approximation error).

**Run command (recommended):**
```bash
python src/models/attention-tuned/informer.py
```

---

## Complete Tuned Hyperparameter Summary

### Training Hyperparameters (Unchanged Across All 15 Tuned Models)

| Parameter | Value |
|-----------|-------|
| `epochs` | 150 |
| `batch_size` | 32 |
| `lr` | 1e-3 |
| `patience` | 15 |
| `weight_decay` | 1e-4 |
| `loss` | huber (delta=1.0) |
| `warmup_epochs` | 5 |

### Architecture Changes Summary

| # | Model | Tuned Script | Changed Parameters |
|---|-------|-------------|-------------------|
| 1 | LSTM | `spatio-temporal-tuned/lstm.py` | hidden 64→128, layers 1→2, dropout 0.10→0.15 |
| 2 | BiLSTM | `spatio-temporal-tuned/bilstm.py` | hidden 64→128, layers 1→2, dropout 0.10→0.15 |
| 3 | CNN-LSTM | `spatio-temporal-tuned/cnnlstm.py` | cnn_filters 32→64, dropout 0.10→0.20 |
| 4 | CNN-BiLSTM | `spatio-temporal-tuned/cnnbilstm.py` | hidden 64→128, cnn_filters 32→64, dropout 0.10→0.25 |
| 5 | ST-LSTM | `spatio-temporal-tuned/stlstm.py` | hidden 64→128, spatial_hidden 32→64, dropout 0.10→0.20 |
| 6 | STGCN | `graph-tuned/stgcn.py` | hidden 128→256, n_blocks 2→3, kt 3→2, dropout 0.10→0.15 |
| 7 | MTGNN | `graph-tuned/mtgnn.py` | hidden 32→64, skip_ch 64→128, n_layers 3→4, dropout 0.10→0.15 |
| 8 | STSGCN | `graph-tuned/stsgcn.py` | hidden 64→128, n_layers 2→3, dropout 0.10→0.20 |
| 9 | STFGNN | `graph-tuned/stfgnn.py` | hidden 64→128, n_layers 3→4, dropout 0.10→0.30 |
| 10 | PDR-STGCN | `graph-tuned/pdr_stgcn.py` | hidden 128→256, n_blocks 2→3, kt 3→2, dk 32→64, dropout 0.10→0.20 |
| 11 | TPA-LSTM | `attention-tuned/tpalstm.py` | hidden 64→128, filters 32→64, dropout 0.10→0.15 |
| 12 | ASTGCN | `attention-tuned/astgcn.py` | d_model 64→128, n_heads 4→8, n_blocks 2→3, dropout 0.10→0.20 |
| 13 | TFT | `attention-tuned/tft.py` | d_model 64→128, n_heads 4→8, n_lstm_layers 1→2, n_attn_layers 2→3, dropout 0.10→0.25 |
| 14 | Autoformer | `attention-tuned/autoformer.py` | d_model 64→128, n_heads 4→8, e_layers 2→3, d_ff 128→256, dropout 0.10→0.20 |
| 15 | Informer | `attention-tuned/informer.py` | d_model 64→128, n_heads 4→8, e_layers 2→3, d_ff 128→256, dropout 0.10→0.15 |

---

## CNN-LSTM Mode Run Commands (Complete Reference)

CNN-LSTM supports three fusion modes. Each mode was independently tuned and is run separately:

```bash
# Sequential mode — CNN→LSTM hierarchy (best: MCO excluded, lb14)
python src/models/spatio-temporal-tuned/cnnlstm.py --mode sequential --lookback 14

# Parallel mode — CNN‖LSTM fusion (best: MCO excluded, lb28)
python src/models/spatio-temporal-tuned/cnnlstm.py --mode parallel --lookback 28

# Augmented mode — CNN→LSTM + raw skip connection (best: MCO excluded, lb14)
python src/models/spatio-temporal-tuned/cnnlstm.py --mode augmented --lookback 14
```

To include the MCO period, add `--include-mco` to any command.

---

## Relationship to Baseline Results

All tuned model scripts point `PRIOR_MODELS` to the **base** output directories (`src/outputs/{model_name}/`), not the tuned directories. This ensures:
- The comparison table reflects the original 16-way baseline chain (LSTM through Informer).
- The tuned run appears as a 17th entry compared against all 15 base model results.
- Fine-tuned=yes tagging is applied externally by `src/utils/aggregate_results.py`.
