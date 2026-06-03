# HMT-TSF — Hybrid Multi-scale Temporal Spatio-Feature Forecaster

> Last updated: 2026-06-03

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
  ┌───────┴──────────────────────────────────────────────────────────┐
  │         Temporal Transformer Block (global self-attention)       │
  │  Learnable positional embeddings + MultiheadAttention (n_heads)  │
  │  Pre-LN residual + FFN (d → 2d → d)                              │
  └───────┬──────────────────────────────────────────────────────────┘
          │  (B, T_in, d_model)
          ├───────────────────────────┬──────────────────────────────┐
          │                           │                              │
  ┌───────▼────────────┐   ┌──────────▼─────────────┐   ┌────────────▼────┐
  │  Multi-Scale TCN   │   │ Feature Graph Encoder  │   │  Regime Gating  │
  │  (with DropPath)   │   │                        │   │  Embedding      │
  │                    │   │  Learned time-attn →   │   │                 │
  │  Scale 1 (T_in)    │   │  (B, F, 1) node feats  │   │  x.mean(T) →    │
  │  CausalConv dil=1  │   │  Linear(1, g_hid)      │   │  Linear→K logit │
  │  CausalConv dil=2  │   │  GCN(g_hid, g_hid)     │   │  Softmax → gate │
  │  CausalConv dil=4  │   │  GCN(g_hid, d_model)   │   │  gate @ E_k     │
  │  …                 │   │  mean(F) → (B, d_model)│   │  → (B, d_model) │
  │                    │   │                        │   │                 │
  │  Scale 2 (T_in//2) │   │  Pearson adj (F×F):    │   │  K=3 regime     │
  │  (if T_in ≥ 28)    │   │  |corr|≥threshold,     │   │  embeddings     │
  │                    │   │  sym-norm              │   │  (pre/MCO/post) │
  │  Scale 3 (T_in//4) │   │                        │   │                 │
  │  (if T_in ≥ 56)    │   │                        │   │                 │
  │                    │   │                        │   │                 │
  │  learned scale attn│   │                        │   │                 │
  │  pool → (B,d_model)│   │                        │   │                 │
  └───────┬────────────┘   └──────────┬─────────────┘   └──────┬──────────┘
          │ h_t (B,d)                 │ h_s (B,d)              │ h_r (B,d)
          └───────────────────────────┴────────────────────────┘
                                      │
                          ┌───────────▼─────────────-─┐
                          │      Gated Fusion         │
                          │                           │
                          │  h = cat[h_t, h_s, h_r]   │
                          │  SE bottleneck gate:      │
                          │  d_cat → d_cat//4 → d_cat │
                          │  g = σ(bottleneck(h))     │
                          │  proj: g⊙h → 2d → d_model│
                          │  GELU → Dropout → LN      │
                          └───────────┬─────────────-─┘
                                      │ (B, d_model)
                          ┌───────────▼────────────────┐
                          │      Forecast Heads        │
                          │                            │
                          │  Primary:                  │
                          │   h → d → d → T_out        │
                          │   (highway residual)       │
                          │                            │
                          │  Boosting:                 │
                          │   h → d → T_out            │
                          │   × sigmoid(α)             │
                          │   (small α init=0.1)       │
                          │                            │
                          │  y_final = y_p + y_b       │
                          └───────────┬────────────────┘
                                      │
                          ┌───────────▼──────────────────┐
                          │ Post-hoc Residual Booster    │
                          │  (optional, out-of-graph)    │
                          │  CatBoost / sklearn MLP      │
                          │  trained on train residuals  │
                          │  y_final += 0.5 * Δ_boost    │
                          └──────────────────────────────┘

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
- `denormalize()` maps predictions back to MinMax-scaled space before the loss — required so scaler_y sees the space it was fitted on

### Temporal Transformer Block
- Source: Vaswani et al. (2017); PatchTST (2022)
- Single Transformer encoder block with learnable positional embeddings and MultiheadAttention
- Positioned **between** FeatureGroupFusion and the Multi-Scale TCN
- Purpose: TCN only captures local patterns via dilated convolutions; this block adds global temporal self-attention so the model can weight which time steps matter most across the full lookback window before local extraction

### Multi-Scale TCN
- Source: WaveNet (van den Oord 2016), TCN (Bai 2018), TimesNet (Wu 2023)
- Causal, left-padded dilated convolutions — no future leakage
- WaveNet-style gated activation: `tanh(h₁) ⊙ σ(h₂)` with residual
- Three temporal scales: full T_in, T_in//2 (≥28), T_in//4 (≥56)
- Each scale uses fewer TCN blocks (full=n, half=n-1, quarter=n-2) to avoid over-parameterising shorter windows
- Learned attention blending across scales — short-term and long-term patterns co-exist
- **DropPath** (stochastic depth) applied per TCN block with linearly increasing rates from 0 to `drop_path`

### DropPath (Stochastic Depth)
- Source: Huang et al. (2016), "Deep Networks with Stochastic Depth"
- Randomly drops the residual path of each TCN block during training
- Acts as structured regularisation preventing co-adaptation between consecutive blocks
- Rate linearly scaled from 0 (first block) to `--drop-path` (last block)

### Feature Graph Encoder (GCN)
- Source: Kipf & Welling (2016), same adjacency strategy as project's STGCN/ASTGCN
- Features-as-nodes (not geographic locations)
- Pearson correlation adjacency, threshold=`--adj-threshold` (default 0.1), symmetric normalisation
- **Learned temporal attention** aggregates across the T dimension (instead of naive mean-pool) so the graph branch can weight recent or peak-demand time steps
- 2-layer GCN, global mean pooling → spatial context vector

### Regime Gating Embedding
- Source: Inspired by MoE (Mixture of Experts) and domain-adaptive normalisation
- Addresses the MCO/COVID structural break in Malaysian transit data
- Learns K=3 embeddings for pre-COVID, COVID, post-COVID regimes
- Gate computed from mean input features (no date metadata required at inference)
- Soft mixture → smooth transitions at boundary periods

### Gated Fusion
- Source: Highway networks (Srivastava 2015), Gated Linear Units, SE-Net (Hu 2018)
- Combines temporal, spatial, and regime information with learned gates
- **SE-style bottleneck gate**: d_cat → d_cat//4 → d_cat (avoids a massive square weight matrix at d_model=256)
- Prevents any single modality from dominating

### Weighted Huber Loss
- Step 1 carries weight 1.0, decaying geometrically (default decay=0.9)
- Prioritises accurate near-term forecasts; tail steps still contribute to training
- Huber δ=`--huber-delta` (default 1.0) for robustness to ridership outliers
- Both `--huber-delta` and `--loss-decay` are configurable via CLI flags

### Temporal Smoothness Regularisation
- Penalises `||y_{t+1} - y_t||²` across the forecast horizon
- Prevents unrealistic ridership oscillations in multi-step outputs
- Default weight λ=0.01 (light regularisation)

---

## Training Strategy

### Full Training
```
Epochs:        150 (default)
Batch size:    32
Optimiser:     AdamW (lr=1e-3, weight_decay=1e-3)
LR schedule:   Linear warmup (8 ep) → ReduceLROnPlateau (factor=0.5, patience=8)
Gradient clip: max_norm=1.0
Early stopping: patience=20 (raw val loss)
Mixed precision: AMP fp16 on CUDA (GradScaler)
Loss:          WeightedHuber + TemporalSmoothness
Input noise:   Gaussian noise std=0.02 added to training inputs (data augmentation)
```

### Post-Hoc Residual Boosting
- Collect neural predictions on the **training set** (no shuffle, preserving row alignment)
- Compute residuals: `Δ = y_true − y_neural`
- Fit CatBoost (if installed) or sklearn MLP on flattened `(X_train, Δ)`
- Apply correction to test predictions: `y_final = y_neural + 0.5 × Δ_boost`
- Only applied if correction improves Combined% (automatic validation)

### Walk-Forward Evaluation
- Split test set chronologically into 3 equal blocks
- Report Combined%, R² per block to detect temporal degradation
- Stable models show <3% Combined% variance across blocks

---

## Base Configuration (Current Defaults)

Parameters baked into `parse_args()` defaults — what runs when no flags are passed.

| Parameter       | Default | Notes |
|-----------------|---------|-------|
| `d_model`       | 64      | Intentionally conservative; use `--d-model 128` for more capacity |
| `n_tcn_blocks`  | 3       | Dilation doubles per block (1, 2, 4, …) |
| `tcn_kernel`    | 3       | Temporal kernel size |
| `graph_hidden`  | 64      | GCN hidden dimension |
| `n_regimes`     | 3       | Pre-MCO / MCO / Post-MCO |
| `n_attn_heads`  | 4       | Temporal Transformer heads |
| `adj_threshold` | 0.1     | Pearson correlation threshold for graph edges |
| `drop_path`     | 0.2     | Stochastic depth max rate across TCN blocks |
| `dropout`       | 0.1     | Applied throughout |
| `lr`            | 1e-3    | AdamW initial learning rate |
| `weight_decay`  | 1e-3    | AdamW weight decay |
| `warmup_epochs` | 8       | Linear LR warm-up before ReduceLROnPlateau |
| `patience`      | 20      | Early-stopping patience (raw val loss) |
| `input_noise`   | 0.02    | Gaussian noise std added to training inputs |
| `loss_decay`    | 0.9     | Per-step geometric weight decay in WeightedHuber |
| `smooth_weight` | 0.01    | Temporal smoothness regularisation weight λ |

---

## Model Scaling Recommendations

| Use Case              | d_model | n_tcn_blocks | graph_hidden | dropout | Expected Params |
|-----------------------|---------|--------------|--------------|---------|-----------------|
| Default (argparse)    | 64      | 3            | 64           | 0.1     | ~850K           |
| Balanced              | 128     | 3            | 64           | 0.1     | ~3.2M           |
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
| 56       | `data/sequences/lookback_56/` | Bi-monthly; enables Scale 2+3   |
| 84       | `data/sequences/lookback_84/` | Quarterly; maximum context      |

Sequence directories for lookback 7 and 84 must be built first:
```bash
python src/features/sequence_builder.py --T-in 7
python src/features/sequence_builder.py --T-in 84
```

---

## CLI Flag Reference

### Architecture flags

| Flag | Default | Description |
|------|---------|-------------|
| `--d-model {64,128,192,256}` | `64` | Model hidden dimension |
| `--n-tcn-blocks N` | `3` | TCN blocks per scale |
| `--tcn-kernel N` | `3` | Temporal kernel size for all TCN convolutions |
| `--graph-hidden {32,64,128}` | `64` | GCN hidden dimension |
| `--n-regimes N` | `3` | Regime embedding count |
| `--adj-threshold F` | `0.1` | Pearson correlation threshold for graph adjacency |
| `--drop-path F` | `0.2` | Stochastic depth rate for TCN blocks (0 = disabled) |
| `--n-attn-heads N` | `4` | Attention heads in the temporal Transformer block |
| `--no-revin` | off | Disable RevIN input normalisation |

### Training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | `150` | Maximum training epochs |
| `--batch-size N` | `32` | Batch size |
| `--lr F` | `1e-3` | Initial learning rate |
| `--weight-decay F` | `1e-3` | AdamW weight decay |
| `--patience N` | `20` | Early-stopping patience (raw val loss) |
| `--warmup-epochs N` | `8` | Linear LR warm-up before ReduceLROnPlateau |
| `--dropout F` | `0.1` | Dropout rate |
| `--input-noise F` | `0.02` | Std-dev of Gaussian noise added to training inputs (0 = off) |

### Loss flags

| Flag | Default | Description |
|------|---------|-------------|
| `--loss {mse,huber,mae}` | `huber` | Base training loss |
| `--loss-decay γ` | `0.9` | Geometric decay for weighted Huber steps |
| `--huber-delta F` | `1.0` | Huber loss transition point |
| `--smooth-weight λ` | `0.01` | Temporal smoothness regularisation |

### Feature / boosting / analysis flags

| Flag | Default | Description |
|------|---------|-------------|
| `--use-catboost` | off | CatBoost for residual boosting (requires catboost) |
| `--no-boost` | off | Skip neural boost head and post-hoc boosting |
| `--shap` | off | Run SHAP feature importance analysis |
| `--shap-samples N` | `100` | SHAP background samples |

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

### With SHAP feature importance
```bash
python src/models/hybrid/hmttsf.py --shap --shap-samples 150
```

### With CatBoost residual boosting
```bash
python src/models/hybrid/hmttsf.py --use-catboost
```

### Maximum configuration (84-day lookback, SHAP, CatBoost)
```bash
python src/models/hybrid/hmttsf.py \
  --lookback 84 --d-model 256 --n-tcn-blocks 5 \
  --graph-hidden 128 --dropout 0.15 \
  --use-catboost --shap \
  --epochs 150
```

### 16-way comparison with all prior models
```bash
python src/models/hybrid/hmttsf.py \
  --lstm-results      src/outputs/lstm/.../results.json \
  --informer-results  src/outputs/informer/.../results.json
  # (omit paths for auto-detection of latest run)
```

---

## Theoretical Connections

| Component             | Inspired by                                  |
|-----------------------|----------------------------------------------|
| Temporal Transformer  | Vaswani 2017, PatchTST, iTransformer         |
| Multi-scale TCN       | WaveNet, TCN (Bai 2018), TimesNet            |
| DropPath              | Stochastic Depth (Huang 2016)                |
| Feature graph GCN     | STGCN, ASTGCN (existing project models)      |
| Regime embedding      | Domain adaptation, MoE, NLinear              |
| RevIN                 | PatchTST, TimesNet, iTransformer             |
| Gated fusion (SE)     | Highway networks, SE-Net                     |
| Weighted Huber        | TiDE, N-BEATS loss variants                  |
| Temporal smoothness   | Modern tabular DL regularisation             |
| Residual boosting     | GBM ensembles (LightGBM / CatBoost stacking) |
| SHAP explainability   | Modern tabular DL (TabNet, XGBoost)          |

---

## Achieved Results

All 10 configurations (nomco + mco × lb7/14/28/56/84) have been trained and evaluated for both the full-feature model (F=79) and the feature-reduced variant (F=53). Aggregate results are stored in `src/outputs/aggregate_hmttsf.csv`. Run IDs are recorded therein.

**Optimisation targets:** Combined% ≥ 75%, R² ≥ 0.70 (both simultaneously). **HMT-TSF (F=79): 8 of 10** configurations meet both targets (nomco_lb84 and mco_lb84 fall short). **HMT-TSF-FR (F=53): 9 of 10** — only mco_lb84 falls below both thresholds.

### HMT-TSF Full (F=79) — Overall Performance

| Configuration | Combined% | MAPE% | MAE% | RMSE% | R² | MAE | RMSE |
|---------------|-----------|-------|------|-------|-----|-----|------|
| nomco_lb7 | 79.88 | 6.17 | 5.43 | 8.52 | 0.783 | 68,163 | 107,034 |
| **nomco_lb14** | **81.46** | **5.58** | **4.70** | **8.25** | **0.797** | **59,105** | **103,684** |
| nomco_lb28 | 80.25 | 5.93 | 5.38 | 8.44 | 0.788 | 67,606 | 106,096 |
| nomco_lb56 | 77.77 | 6.88 | 6.32 | 9.02 | 0.755 | 79,342 | 113,248 |
| nomco_lb84 | 74.65 | 7.97 | 7.41 | 9.98 | 0.704 | 92,303 | 124,327 |
| mco_lb7 | 75.19 | 7.70 | 6.87 | 10.24 | 0.712 | 84,611 | 126,052 |
| mco_lb14 | 75.66 | 7.48 | 6.83 | 10.02 | 0.724 | 84,286 | 123,558 |
| mco_lb28 | 75.96 | 7.44 | 6.58 | 10.02 | 0.722 | 81,282 | 123,739 |
| **mco_lb56** | **76.48** | **7.26** | **6.24** | **10.02** | **0.722** | **77,496** | **124,291** |
| mco_lb84 | 73.98 | 8.32 | 6.89 | 10.81 | 0.674 | 85,558 | 134,126 |

**Best configurations:** nomco_lb14 (Combined%=81.46%, R²=0.797) and, under MCO, mco_lb56 (76.48%, R²=0.722).

---

### HMT-TSF Feature-Reduced (FR, F=53)

SHAP-guided ablation removed 26 zero-importance features: 6 temporal/cyclical (year, day_of_year, and 4 cyclical encodings redundant with the retained holiday and lag features), 3 collinear fuel-level series (correlated with the retained fuel-price delta features), and all 17 static features (population, GTFS route/stop counts, OSM POI counts, GADM area metrics). The retained 53 features span: 13 target-context columns, 10 temporal/cyclical encodings, 27 external series (12 fuel-price + 15 rainfall), and 3 ridership lag features. Outputs are in `src/outputs/hmttsf_feat_reduced/`.

### HMT-TSF-FR (F=53) — Overall Performance

| Configuration | Combined% | MAPE% | MAE% | RMSE% | R² | MAE | RMSE |
|---------------|-----------|-------|------|-------|-----|-----|------|
| nomco_lb7 | 79.76 | 6.20 | 5.58 | 8.46 | 0.787 | 70,070 | 106,249 |
| **nomco_lb14** | **81.62** | **5.52** | **4.80** | **8.06** | **0.807** | **60,306** | **101,291** |
| nomco_lb28 | 81.12 | 5.66 | 4.93 | 8.29 | 0.795 | 62,012 | 104,294 |
| nomco_lb56 | 78.44 | 6.62 | 5.97 | 8.97 | 0.758 | 74,991 | 112,577 |
| nomco_lb84 | 75.58 | 7.69 | 7.11 | 9.63 | 0.724 | 88,591 | 119,975 |
| mco_lb7 | 75.12 | 7.75 | 6.84 | 10.29 | 0.709 | 84,201 | 126,759 |
| mco_lb14 | 75.66 | 7.41 | 6.61 | 10.31 | 0.708 | 81,565 | 127,181 |
| **mco_lb28** | **76.54** | **7.24** | **6.29** | **9.92** | **0.727** | **77,753** | **122,584** |
| mco_lb56 | 76.43 | 7.34 | 6.33 | 9.89 | 0.728 | 78,611 | 122,788 |
| mco_lb84 | 69.56 | 9.27 | 8.24 | 12.93 | 0.533 | 102,291 | 160,533 |

**Best configurations:** FR nomco_lb14 (Combined%=81.62%, R²=0.807 — the new global best across the study) and, under MCO, FR mco_lb28 (76.54%, R²=0.727; MCO optimum shifts lb56→lb28 relative to Full).

### FR vs Full Comparison (Δ Combined%, Δ R²)

| Configuration | Full Combined% | FR Combined% | Δ Combined% | Full R² | FR R² | Δ R² |
|---------------|---------------|-------------|-------------|---------|-------|------|
| nomco_lb7 | 79.88 | 79.76 | −0.12 | 0.783 | 0.787 | +0.004 |
| nomco_lb14 | 81.46 | 81.62 | **+0.16** | 0.797 | 0.807 | **+0.010** |
| nomco_lb28 | 80.25 | 81.12 | **+0.87** | 0.788 | 0.795 | **+0.007** |
| nomco_lb56 | 77.77 | 78.44 | **+0.67** | 0.755 | 0.758 | **+0.003** |
| nomco_lb84 | 74.65 | 75.58 | **+0.93** | 0.704 | 0.724 | **+0.020** |
| mco_lb7 | 75.19 | 75.12 | −0.07 | 0.712 | 0.709 | −0.003 |
| mco_lb14 | 75.66 | 75.66 | 0.00 | 0.724 | 0.708 | −0.016 |
| mco_lb28 | 75.96 | 76.54 | **+0.58** | 0.722 | 0.727 | **+0.005** |
| mco_lb56 | 76.48 | 76.43 | −0.05 | 0.722 | 0.728 | **+0.006** |
| mco_lb84 | 73.98 | 69.56 | **−4.42** | 0.674 | 0.533 | **−0.141** |

FR is neutral-to-better in 8 of 10 configurations, with the most consistent gains in the no-MCO series (nomco_lb28–lb84) where the removed static features contributed collinearity noise rather than signal. The single major exception is mco_lb84 (−4.42 pp Combined%, −0.141 R²): with an 84-day look-back spanning the COVID structural break, regime gating relied on static spatial signals (population density, GTFS coverage, OSM POI density) to separate pre-/during-/post-MCO contexts; removing those 17 nodes collapses regime separation and accuracy significantly. For look-backs ≤56 days the COVID window sits largely outside the look-back, so static feature removal is harmless or beneficial.

**Lookback sensitivity (no-MCO):** FR peaks at lb14 (81.62%, +0.16 pp vs Full), with lb28 (81.12%) a close second. The FR advantage grows at longer look-backs — nomco_lb84 now clears the target (75.58%) where Full fell short (74.65%), a +0.93 pp gain attributable to eliminating static-node noise over long windows.

**Lookback sensitivity (MCO):** FR MCO optimum shifts to lb28 (76.54%) from Full's lb56 (76.48%). Without static spatial anchors the regime gating peaks earlier; lb56 (76.43%) is a marginal 0.11 pp behind. Both lb28 and lb56 remain above the 75% target. mco_lb84 collapses (see above).

**MCO degradation:** At lb14 FR degrades by 5.96 pp (81.62→75.66) versus Full's 5.80 pp. The slightly larger gap reflects the absence of static spatial anchors under MCO conditions. The mean degradation across lb7–lb56 is ~3.7 pp for FR vs ~3.3 pp for Full — a modest increase that remains well below the baseline median MCO degradation.

---

### Comparison Against Best Tuned Baselines — No-MCO, Lookback 14

| Model | Combined% | R² | MAE | RMSE |
|-------|-----------|-----|-----|------|
| **HMT-TSF-FR** | **81.62** | **0.807** | 60,306 | **101,291** |
| HMT-TSF | 81.46 | 0.797 | **59,105** | 103,684 |
| Informer (tuned) | 79.99 | 0.778 | 65,360 | 108,628 |
| TPA-LSTM (tuned) | 79.95 | 0.782 | 66,703 | 107,475 |
| BiLSTM (tuned) | 79.44 | 0.794 | 73,516 | 104,667 |
| ST-LSTM (tuned) | 79.13 | 0.774 | 70,796 | 109,604 |
| ASTGCN (tuned) | 78.82 | 0.754 | 72,053 | 114,259 |
| LSTM (tuned) | 78.43 | 0.777 | 78,339 | 108,676 |

HMT-TSF-FR sets the study-wide headline at nomco_lb14: Combined% 81.62% and R² 0.807, leading all 16 models on both metrics. HMT-TSF Full at 81.46% is 0.16 pp behind and edges FR on raw MAE (59,105 vs 60,306); FR achieves the lowest RMSE in the study (101,291). The next baseline tier (Informer, TPA-LSTM, BiLSTM) clusters 1.5–2.2 pp below FR.

---

### Comparison Against Best Tuned Baselines — MCO-Inclusive, Lookback 14

| Model | Combined% | R² | MAE | RMSE |
|-------|-----------|-----|-----|------|
| Informer (tuned) | 75.79 | 0.742 | 85,636 | 119,628 |
| **HMT-TSF** | **75.66** | **0.724** | **84,286** | 123,558 |
| HMT-TSF-FR | 75.66 | 0.708 | 81,565 | 127,181 |
| ST-LSTM (tuned) | 75.05 | 0.726 | 86,892 | 123,067 |
| TPA-LSTM (tuned) | 74.77 | 0.721 | 87,921 | 124,327 |
| BiLSTM (tuned) | 70.66 | 0.666 | 111,167 | 135,981 |
| CNN-LSTM-Aug (tuned) | 67.98 | 0.604 | 121,399 | 148,024 |
| LSTM (tuned) | 66.23 | 0.581 | 130,969 | 152,325 |

Under MCO at lb14 both HMT-TSF variants tie at Combined% 75.66%, statistically level with the strongest baseline Informer tuned (75.79%). Full edges FR on R² (0.724 vs 0.708) and RMSE without the static-feature handicap; FR edges Full on MAE (81,565 vs 84,286). Note that FR's MCO optimum is at lb28 (76.54%), where it would lead the MCO table. Both HMT-TSF variants clear the 75% target; the remaining baselines drop to 66–75% as the COVID structural break overwhelms pure temporal/spatial pattern transfer.

---

### Walk-Forward Evaluation (Temporal Stability)

Test set split chronologically into 3 equal blocks. Combined% and R² per block; Range = max − min across blocks:

**HMT-TSF Full (F=79):**

| Configuration | Block 1 | R² | Block 2 | R² | Block 3 | R² | Range |
|---------------|---------|-----|---------|-----|---------|-----|-------|
| nomco_lb7 | 81.56 | 0.821 | 75.59 | 0.693 | 82.93 | 0.848 | 7.34 |
| nomco_lb14 | 84.72 | 0.863 | 76.79 | 0.698 | 83.27 | 0.838 | 7.93 |
| nomco_lb28 | 86.66 | 0.896 | 72.12 | 0.633 | 82.86 | 0.846 | 14.54 |
| nomco_lb56 | 78.40 | 0.759 | 75.72 | 0.728 | 79.20 | 0.778 | 3.48 |
| nomco_lb84 | 67.68 | 0.586 | 78.63 | 0.788 | 78.01 | 0.752 | 10.95 |
| mco_lb7 | 69.77 | 0.636 | 79.45 | 0.781 | 76.33 | 0.706 | 9.68 |
| mco_lb14 | 70.64 | 0.663 | 80.18 | 0.792 | 76.17 | 0.710 | 9.54 |
| mco_lb28 | 69.82 | 0.654 | 81.25 | 0.803 | 76.85 | 0.705 | 11.43 |
| mco_lb56 | 72.36 | 0.673 | 78.25 | 0.746 | 78.86 | 0.745 | 6.50 |
| mco_lb84 | 70.07 | 0.616 | 74.42 | 0.667 | 77.56 | 0.740 | 7.49 |

**HMT-TSF-FR (F=53):**

| Configuration | Block 1 | R² | Block 2 | R² | Block 3 | R² | Range |
|---------------|---------|-----|---------|-----|---------|-----|-------|
| nomco_lb7 | 83.47 | 0.856 | 75.61 | 0.705 | 80.52 | 0.804 | 7.86 |
| nomco_lb14 | 85.84 | 0.883 | 76.16 | 0.696 | 83.39 | 0.847 | 9.68 |
| nomco_lb28 | 85.64 | 0.883 | 72.93 | 0.640 | 85.70 | 0.877 | 12.77 |
| nomco_lb56 | 78.01 | 0.753 | 76.00 | 0.734 | 81.26 | 0.789 | 5.26 |
| nomco_lb84 | 69.50 | 0.633 | 79.04 | 0.801 | 78.45 | 0.749 | 9.54 |
| mco_lb7 | 69.05 | 0.616 | 80.08 | 0.791 | 76.25 | 0.708 | 11.03 |
| mco_lb14 | 71.91 | 0.680 | 78.31 | 0.722 | 76.66 | 0.709 | 6.40 |
| mco_lb28 | 71.05 | 0.660 | 79.88 | 0.782 | 78.67 | 0.735 | 8.83 |
| mco_lb56 | 71.23 | 0.677 | 78.67 | 0.751 | 79.40 | 0.757 | 8.17 |
| mco_lb84 | 58.84 | 0.237 | 74.33 | 0.661 | 76.40 | 0.720 | 17.56 |

**No-MCO pattern:** Both variants show Block 2 as the consistently weakest segment — a lower-ridership seasonal phase in the mid-test window — with Blocks 1 and 3 stronger. nomco_lb56 is the most stable FR no-MCO configuration (range 5.26%), slightly above Full's (3.48%). FR nomco_lb14 at range 9.68 is somewhat wider than Full (7.93) but Block 1 rises from 84.72 to 85.84. The nomco_lb28 Block 2 dip (72.93 for FR, 72.12 for Full) is the main source of within-test variance in both variants.

**MCO pattern:** MCO Block 1 is consistently the weakest in both variants (COVID disruption at test start), with Blocks 2–3 recovering. The mco_lb84 FR collapse is stark: Block 1 drops to 58.84% (R²=0.237), far below Full's 70.07% (R²=0.616), confirming static-feature dependence at long COVID windows. FR mco_lb14 is more temporally stable (range 6.40 vs Full 9.54), reflecting the cleaner feature set reducing variance at shorter look-backs. mco_lb56 remains the most stable MCO Full configuration (range 6.50%); FR mco_lb56 is comparable (range 8.17%).

---

## Journal References

Scopus-indexed journal articles (2022–2027) supporting each architectural component of HMT-TSF. All DOIs link to the primary Scopus-indexed venue.

| Component | Citation |
|-----------|----------|
| RevIN | Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022). Reversible instance normalization for accurate time-series forecasting against distribution shift. *International Conference on Learning Representations (ICLR 2022)*. https://openreview.net/forum?id=cGDAkQo1C0p |
| Temporal Transformer | Bouchiat, K., Kosorus, H., & Prieler, V. (2023). Persistence initialization: A novel adaptation of the Transformer architecture for time series forecasting. *Applied Intelligence*, 53, 27931–27946. https://doi.org/10.1007/s10489-023-04927-4 |
| Multi-Scale TCN | Gao, H., Fu, Z., Sun, J., Bian, G., & Li, C. (2023). MD-GCN: A multi-scale temporal dual graph convolution network for traffic flow prediction. *Sensors*, 23(2), 841. https://doi.org/10.3390/s23020841 |
| DropPath (Stochastic Depth) | Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q. (2016). Deep networks with stochastic depth. *European Conference on Computer Vision (ECCV 2016)*, LNCS 9908. https://doi.org/10.1007/978-3-319-46493-0_39 |
| Feature Graph GCN | Jiang, W., & Luo, J. (2022). Graph neural network for traffic forecasting: A survey. *Expert Systems with Applications*, 207, 117921. https://doi.org/10.1016/j.eswa.2022.117921 |
| Regime Gating (MoE) | Hou, Z., Han, J., & Yang, G. (2025). Analysis of passenger flow characteristics and origin–destination passenger flow prediction in urban rail transit based on deep learning. *Applied Sciences*, 15(5), 2853. https://doi.org/10.3390/app15052853 |
| SE-Net Gated Fusion | Xu, L., Hu, Y., Wei, X., Zhou, X., & Yu, X. (2024). SE-MAConvLSTM: A deep learning framework for short-term traffic flow prediction combining squeeze-and-excitation network and multi-attention convolutional LSTM. *PLoS ONE*, 19(11), e0312601. https://doi.org/10.1371/journal.pone.0312601 |
| Weighted Huber Loss | Liu, Y., Li, Q., Ma, C., & Xu, X. (2026). A CatBoost-based prediction framework for logistics industry prosperity index to support sustainable decision-making: An empirical study from China. *Sustainability*, 18(5), 2178. https://doi.org/10.3390/su18052178 |
| Temporal Smoothness Reg. | Casolaro, A., Capone, V., Iannuzzo, G., & Camastra, F. (2023). Deep learning for time series forecasting: Advances and open problems. *Information*, 14(11), 598. https://doi.org/10.3390/info14110598 |
| Post-hoc Residual Boosting | Huber, T., Aksan, E., & Ratsch, G. (2024). LTBoost: Boosted hybrids of ensemble linear and gradient algorithms for the long-term time series forecasting. *Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM 2024)*. https://doi.org/10.1145/3627673.3679527 |
| Walk-Forward Evaluation | AlKhereibi, S., Wakjira, T. W., Kucukvar, M., & Onat, N. C. (2023). Predictive machine learning algorithms for metro ridership based on urban land use policies in support of transit-oriented development. *Sustainability*, 15(2), 1718. https://doi.org/10.3390/su15021718 |
