# HMT-TSF — Hybrid Multi-scale Temporal Spatio-Feature Forecaster

## Architecture Diagram

```
Input X: (B, T_in, F=79)   [MinMax-scaled by data pipeline]
          │
  ┌───────┴──────────────────────────────────────────────────────────┐
  │                 RevIN (input-only instance norm)                 │
  │  x̂ = (x − μ_sample) / σ_sample * γ + β   (per B per F)           │
  └───────┬──────────────────────────────────────────────────────────┘
          │
  ┌───────┴──────────────────────────────────────────────────────────┐
  │              Feature Group Fusion  →  (B, T_in, d_model)         │
  │                                                                  │
  │  ┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────┐ │
  │  │Target context│ │Temporal/Cyc. │ │ Lag enc. │ │Static enc.   │ │
  │  │  idx  0–12   │ │  idx 13–28   │ │ idx 59–61│ │  idx 62–78   │ │
  │  │  MLP→d/2     │ │  MLP→d/2     │ │  MLP→d/2 │ │  MLP→d/2     │ │
  │  └──────┬───────┘ └──────┬───────┘ └────┬─────┘ └──────┬───────┘ │
  │         │ External enc.  │              │              │         │
  │  ┌──────┴───────┐        │              │              │         │
  │  │  idx 29–58   │        │              │              │         │
  │  │  MLP→d/2     │        │              │              │         │
  │  └──────┬───────┘        │              │              │         │
  │         └────────────────┴──────────────┴──────────────┘         │
  │                   concat + learned group gates                   │
  │                   Linear→d_model, GELU, LayerNorm                │
  └───────┬──────────────────────────────────────────────────────────┘
          │  (B, T_in, d_model)
          ├───────────────────────────┬───────────────────────────────┐
          │                           │                               │
  ┌───────▼────────────┐   ┌──────────▼──────────-─-┐   ┌──────────────▼──┐
  │  Multi-Scale TCN   │   │ Feature Graph Encoder  │   │  Regime Gating  │
  │                    │   │                        │   │  Embedding      │
  │  Scale 1 (T_in)    │   │  x.mean(T) → (B,F,1)   │   │                 │
  │  CausalConv dil=1  │   │  Linear(1, g_hid)      │   │  x.mean(T) →    │
  │  CausalConv dil=2  │   │  GCN(g_hid, g_hid)     │   │  Linear→K logit │
  │  CausalConv dil=4  │   │  GCN(g_hid, d_model)   │   │  Softmax → gate │
  │  …                 │   │  mean(F) → (B, d_model)│   │  gate @ E_k     │
  │                    │   │                        │   │  → (B, d_model) │
  │  Scale 2 (T_in//2) │   │  Pearson adj (F×F):    │   │                 │
  │  (if T_in ≥ 28)    │   │  |corr|≥0.1, sym-norm  │   │  K=3 regime     │
  │                    │   │                        │   │  embeddings     │
  │  Scale 3 (T_in//4) │   │                        │   │  (pre/MCO/post) │
  │  (if T_in ≥ 56)    │   │                        │   │                 │
  │                    │   │                        │   │                 │
  │  learned scale attn│   │                        │   │                 │
  │  pool → (B,d_model)│   │                        │   │                 │
  └───────┬────────────┘   └──────────┬────────────-┘   └──────┬──────────┘
          │ h_t (B,d)                 │ h_s (B,d)              │ h_r (B,d)
          └───────────────────────────┴─────────────────────-──┘
                                      │
                          ┌───────────▼──────────────┐
                          │      Gated Fusion        │
                          │                          │
                          │  h = cat[h_t, h_s, h_r]  │
                          │  g = σ(Linear(h))        │
                          │  Linear(g⊙h) → d_model  │
                          │  GELU → Dropout → LN     │
                          └───────────┬──────────────┘
                                      │ (B, d_model)
                          ┌───────────▼────────────-──┐
                          │      Forecast Heads       │
                          │                           │
                          │  Primary:                 │
                          │   h → 2d → d → T_out      │
                          │   (highway residual)      │
                          │                           │
                          │  Boosting:                │
                          │   h → d → T_out           │
                          │   × sigmoid(α)            │
                          │   (small α init=0.1)      │
                          │                           │
                          │  y_final = y_p + y_b      │
                          └───────────┬────────────-──┘
                                      │
                          ┌───────────▼────────────--──┐
                          │ Post-hoc Residual Booster  │
                          │  (optional, out-of-graph)  │
                          │  CatBoost / sklearn MLP    │
                          │  trained on train residuals│
                          │  y_final += 0.5 * Δ_boost  │
                          └──────────────────────────-─┘

Output: (B, T_out=7)  [MinMax-scaled]
        → scaler_y.inverse_transform() → raw ridership counts
```

---

## Component Rationale

### RevIN (Reversible Instance Normalisation)
- Source: Kim et al. (2021), "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift"
- Applied to **input only** in this implementation (data pipeline uses pre-scaled MinMax data; `scaler_y.inverse_transform` handles the final conversion)
- Reduces within-batch distribution shift caused by MCO/COVID regime changes
- Learnable affine parameters γ, β per feature

### Multi-Scale TCN
- Source: WaveNet (van den Oord 2016), TCN (Bai 2018), TimesNet (Wu 2023)
- Causal, left-padded dilated convolutions — no future leakage
- WaveNet-style gated activation: `tanh(h₁) ⊙ σ(h₂)` with residual
- Three temporal scales: full T_in, T_in//2 (≥28), T_in//4 (≥56)
- Learned attention blending across scales — short-term and long-term patterns co-exist

### Feature Graph Encoder (GCN)
- Source: Kipf & Welling (2016), same adjacency strategy as project's STGCN/ASTGCN
- Features-as-nodes (not geographic locations)
- Pearson correlation adjacency, threshold=0.1, symmetric normalisation
- 2-layer GCN, global mean pooling → spatial context vector

### Regime Gating Embedding
- Source: Inspired by MoE (Mixture of Experts) and domain-adaptive normalisation
- Addresses the MCO/COVID structural break in Malaysian transit data
- Learns K=3 embeddings for pre-COVID, COVID, post-COVID regimes
- Gate computed from mean input features (no date metadata required at inference)
- Soft mixture → smooth transitions at boundary periods

### Gated Fusion
- Source: Highway networks (Srivastava 2015), Gated Linear Units
- Combines temporal, spatial, and regime information with learned gates
- Prevents any single modality from dominating

### Weighted Huber Loss
- Step 1 carries weight 1.0, decaying geometrically (default decay=0.9)
- Prioritises accurate near-term forecasts; tail steps still contribute to training
- Huber δ=1.0 for robustness to ridership outliers (confirmed effective in TFT/Autoformer)

### Temporal Smoothness Regularisation
- Penalises `||y_{t+1} - y_t||²` across the forecast horizon
- Prevents unrealistic ridership oscillations in multi-step outputs
- Default weight λ=0.01 (light regularisation)

---

## Training Strategy

### Phase 1 — Optuna HPO (optional, `--tune-trials N`)
- Run N trials, up to 40 epochs each with per-trial early stopping (patience=10)
- Median pruner also prunes underperforming trials after warmup (15 epochs)
- Search space covers architecture + regularisation simultaneously
- Transfer best hyperparameters to Phase 2

### Phase 2 — Full Training
```
Epochs:        150 (default)
Batch size:    32
Optimiser:     AdamW (lr=1e-3, weight_decay=1e-4)
LR schedule:   Linear warmup (5 ep) → ReduceLROnPlateau (factor=0.5, patience=5)
Gradient clip: max_norm=1.0
Early stopping: patience=15 (validation loss)
Mixed precision: AMP fp16 on CUDA (GradScaler)
Loss:          WeightedHuber + TemporalSmoothness
```

### Phase 3 — Post-Hoc Residual Boosting
- Collect neural predictions on the **training set**
- Compute residuals: `Δ = y_true − y_neural`
- Fit CatBoost (if installed) or sklearn MLP on flattened `(X_train, Δ)`
- Apply correction to test predictions: `y_final = y_neural + 0.5 × Δ_boost`
- Only applied if correction improves Combined% (automatic validation)

### Phase 4 — Walk-Forward Evaluation
- Split test set chronologically into 3 equal blocks
- Report Combined%, R² per block to detect temporal degradation
- Stable models show <3% Combined% variance across blocks

---

## Hyperparameter Search Space (Optuna)

| Parameter       | Distribution         | Default |
|-----------------|----------------------|---------|
| `d_model`       | Categorical [64, 128, 192, 256] | 128 |
| `n_tcn_blocks`  | Integer [2, 5]       | 3      |
| `graph_hidden`  | Categorical [32, 64, 128] | 64 |
| `dropout`       | Float [0.05, 0.35] step 0.05 | 0.1 |
| `lr`            | Log-uniform [5e-4, 5e-3] | 1e-3 |
| `smooth_weight` | Float [0.0, 0.05] step 0.005 | 0.01 |

Recommended N trials:
- Quick validation: 20 trials (~40 min on A100)
- Standard: 50 trials (~100 min on A100)
- Thorough: 100 trials (~200 min on A100)

---

## Model Scaling Recommendations

| Use Case              | d_model | n_tcn_blocks | graph_hidden | dropout | Expected Params |
|-----------------------|---------|--------------|--------------|---------|-----------------|
| Debug / fast iter     | 64      | 2            | 32           | 0.1     | ~850K           |
| Default (balanced)    | 128     | 3            | 64           | 0.1     | ~3.2M           |
| High capacity         | 192     | 4            | 128          | 0.15    | ~7.1M           |
| Max (A100 32GB)       | 256     | 5            | 128          | 0.20    | ~12.4M          |

All configurations comfortably fit on the A100 32GB with batch size 32 and T_in≤84.

---

## Lookback Window Guidance

| Lookback | Seq Dir                    | Best For                           |
|----------|----------------------------|------------------------------------|
| 7        | `data/sequences/lookback_7/`  | Weekly patterns, low latency    |
| 14       | `data/sequences/lstm/`        | Default; holiday cycle capture  |
| 28       | `data/sequences/lookback_28/` | Monthly seasonality             |
| 56       | `data/sequences/lookback_56/` | Bi-monthly; enables Scale 2+3  |
| 84       | `data/sequences/lookback_84/` | Quarterly; maximum context     |

Sequence directories for lookback 7 and 84 must be built first:
```bash
python src/features/sequence_builder.py --T-in 7
python src/features/sequence_builder.py --T-in 84
```

---

## Example Run Commands

### Baseline (default hyperparameters)
```bash
python src/models/hybrid/hmttsf.py
```

### Longer lookback with more TCN blocks
```bash
python src/models/hybrid/hmttsf.py --lookback 28 --n-tcn-blocks 4 --d-model 192
```

### With Optuna HPO (50 trials) then full training
```bash
python src/models/hybrid/hmttsf.py --tune-trials 50 --lookback 56
```

### With SHAP feature importance
```bash
python src/models/hybrid/hmttsf.py --shap --shap-samples 150
```

### With CatBoost residual boosting
```bash
python src/models/hybrid/hmttsf.py --use-catboost
```

### Maximum configuration (84-day lookback, HPO, SHAP, CatBoost)
```bash
python src/models/hybrid/hmttsf.py \
  --lookback 84 --d-model 256 --n-tcn-blocks 5 \
  --graph-hidden 128 --dropout 0.15 \
  --tune-trials 50 --use-catboost --shap \
  --epochs 150 --patience 20 --warmup-epochs 8
```

### 17-way comparison with all prior models
```bash
python src/models/hybrid/hmttsf.py \
  --lstm-results      src/outputs/lstm/.../results.json \
  --tft-results       src/outputs/tft/.../results.json \
  --informer-results  src/outputs/informer/.../results.json
  # (omit paths for auto-detection of latest run)
```

---

## Theoretical Connections

| Component          | Inspired by                                  |
|--------------------|----------------------------------------------|
| Multi-scale TCN    | WaveNet, TCN (Bai 2018), TimesNet             |
| Feature graph GCN  | STGCN, ASTGCN (existing project models)      |
| Regime embedding   | Domain adaptation, MoE, NLinear              |
| RevIN              | PatchTST, TimesNet, iTransformer             |
| Gated fusion       | TFT (Lim 2021), Highway networks             |
| Weighted Huber     | TiDE, N-BEATS loss variants                  |
| Temporal smoothness| Modern tabular DL regularisation             |
| Residual boosting  | GBM ensembles (LightGBM / CatBoost stacking) |
| SHAP explainability| Modern tabular DL (TabNet, XGBoost)          |
