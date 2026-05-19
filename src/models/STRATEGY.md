# Hyperparameter Tuning Strategy

This document defines what hyperparameters to adjust for each of the 17 model configurations (15 models; CNN-LSTM counts as 3 because its modes are tuned independently) based on whether the most recent run performed **better** or **worse** than the prior run.

"Better" means Combined% increased. "Worse" means Combined% decreased or variance across forecast horizons increased substantially.

All tuning assumes the **shared training schedule is fixed**: `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`, `weight_decay=1e-4`, `loss=huber`. Do not change these unless a cross-cutting adjustment is warranted (see end of document).

---

## How to Read This Document

Each model entry shows:
- **Current tuned config** — the defaults from the tuned script
- **If BETTER** — push performance further in the same direction
- **If WORSE** — pull back to recover; likely overfitting or mis-sized architecture

The two most common failure modes are:
- **Overfitting** (worse val/test, good train): reduce hidden/layers, increase dropout, try shorter lookback
- **Underfitting** (poor train AND val/test): increase hidden/layers, reduce dropout, try longer lookback

---

## Series 1 — Spatio-Temporal

---

### 1. LSTM

**Current tuned config:** `hidden=128, layers=2, dropout=0.15, lookback=14`

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 128 → 256 | More temporal compression capacity |
| `--layers` | 2 → 3 | Deeper hierarchical temporal abstraction |
| `--dropout` | 0.15 → 0.10 | Ease regularisation if model has room to grow |
| `--lookback` | 14 → 28 | Longer context may capture weekly periodicity better |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 128 → 64 | Return to base capacity; 128 may overfit on small dataset |
| `--layers` | 2 → 1 | Reduce depth; single layer eliminates inter-layer dropout dependency |
| `--dropout` | 0.15 → 0.25 | Stronger regularisation if train/test gap is large |
| `--lookback` | 14 → 56 | Longer window sometimes helps LSTM generalise by seeing more cycles |

---

### 2. BiLSTM

**Current tuned config:** `hidden=128, layers=2, dropout=0.15, lookback=14`

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 128 → 256 | Head input = hidden×2 so capacity doubles |
| `--layers` | 2 → 3 | Deeper bidirectional stack |
| `--dropout` | 0.15 → 0.10 | Less aggressive if model generalises well |
| `--lookback` | 14 → 28 | Bidirectional processing benefits from wider windows |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 128 → 64 | Head is 128×2=256 units — may over-parameterise for dataset size |
| `--layers` | 2 → 1 | 4-direction encoding at 2 layers is demanding; single layer is stabler |
| `--dropout` | 0.15 → 0.25 | Increase regularisation |
| `--lookback` | 14 → 56 | More temporal context can stabilise bidirectional encoding |

---

### 3. CNN-LSTM — Sequential Mode

**Current tuned config:** `cnn_filters=64, hidden=64, layers=1, dropout=0.20, lookback=14`

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--cnn-filters` | 64 → 128 | Wider CNN feature maps before LSTM |
| `--hidden` | 64 → 128 | Increase LSTM capacity to match wider CNN output |
| `--dropout` | 0.20 → 0.10 | Ease regularisation |
| `--lookback` | 14 → 28 | LSTM operating on CNN features benefits from longer windows |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--cnn-filters` | 64 → 32 | Return to base; wider filters may introduce noise |
| `--dropout` | 0.20 → 0.35 | Stronger regularisation to curb CNN overfitting |
| `--cnn-layers` | 2 → 1 | Fewer CNN layers if deep CNN is destroying temporal structure |
| `--lookback` | 14 → 56 | More raw context before CNN compression |

---

### 4. CNN-LSTM — Parallel Mode

**Current tuned config:** `cnn_filters=64, hidden=64, layers=1, dropout=0.20, lookback=28`

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--cnn-filters` | 64 → 128 | Wider CNN branch for richer parallel features |
| `--hidden` | 64 → 128 | Match LSTM branch capacity with CNN branch |
| `--dropout` | 0.20 → 0.10 | Ease regularisation |
| `--lookback` | 28 → 56 | Parallel branches benefit most from very long context |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--cnn-filters` | 64 → 32 | Reduce redundancy in the concatenated head |
| `--dropout` | 0.20 → 0.35 | Parallel concat doubles head input; more regularisation needed |
| `--lookback` | 28 → 14 | Shorter window may reduce noise for the LSTM branch |

---

### 5. CNN-LSTM — Augmented Mode

**Current tuned config:** `cnn_filters=64, hidden=64, layers=1, dropout=0.20, lookback=14`

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--cnn-filters` | 64 → 128 | Wider CNN improves the hierarchical branch |
| `--hidden` | 64 → 128 | Larger LSTM to match wider CNN |
| `--dropout` | 0.20 → 0.10 | Skip connection already regularises; can reduce dropout |
| `--lookback` | 14 → 28 | Skip-pooled raw context is richer with more timesteps |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--cnn-filters` | 64 → 32 | Reduce CNN branch; skip connection may already carry enough information |
| `--dropout` | 0.20 → 0.35 | Concatenated skip makes the head wider — needs more regularisation |
| `--lookback` | 14 → 56 | Very long raw skip context may help the head discriminate patterns |

---

### 6. CNN-BiLSTM

**Current tuned config:** `hidden=128, cnn_filters=64, layers=1, dropout=0.25, lookback=14`

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 128 → 256 | Head receives hidden×2; doubling again gives 512 — large but worth testing |
| `--cnn-filters` | 64 → 128 | Wider CNN to match larger BiLSTM |
| `--dropout` | 0.25 → 0.15 | Ease regularisation if model generalises well |
| `--lookback` | 14 → 28 | Bidirectional over wider CNN features benefits from longer windows |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 128 → 64 | Head is 256 units — likely overfitting; return to base |
| `--cnn-filters` | 64 → 32 | Narrow CNN output to reduce head dimensionality |
| `--dropout` | 0.25 → 0.40 | Already high but dataset may require stronger regularisation |
| `--cnn-layers` | 2 → 1 | Single CNN block if deep CNN is losing temporal structure |

---

### 7. ST-LSTM

**Current tuned config:** `hidden=128, spatial_hidden=64, layers=1, dropout=0.20, lookback=14`

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 128 → 256 | Expand temporal stream capacity |
| `--spatial-hidden` | 64 → 128 | Scale spatial encoder proportionally |
| `--layers` | 1 → 2 | Stack a second LSTM layer in the temporal stream |
| `--dropout` | 0.20 → 0.10 | Ease regularisation if dual-stream helps generalisation |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 128 → 64 | Return to base; spatial stream may already carry enough information |
| `--spatial-hidden` | 64 → 32 | Reduce spatial MLP; shared weights across T_in timesteps scale quickly |
| `--dropout` | 0.20 → 0.30 | More aggressive inter-stream regularisation |
| `--lookback` | 14 → 56 | Spatial stream pools over T_in so longer windows enrich the spatial summary |

---

## Series 2 — Graph

> **T_after constraint (STGCN, PDR-STGCN):** `T_after = T_in − 2 × (kt−1) × n_blocks`.
> With T_in=14, kt=2, n_blocks=3: T_after=8. Any parameter change must keep T_after ≥ 4.
>
> **T_after constraint (STSGCN):** `T_after = T_in − 2 × n_layers`.
> With T_in=14, n_layers=3: T_after=8. Must keep T_after ≥ 4.

---

### 8. STGCN

**Current tuned config:** `hidden=256, n_blocks=3, kt=2, dropout=0.15, lookback=14`

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 256 → 512 | Larger channel width per ST-conv block |
| `--n-blocks` | 3 → 4 | One more block; T_after = 14 − 2×1×4 = 6 ✓ |
| `--dropout` | 0.15 → 0.10 | Ease regularisation |
| `--adj-threshold` | 0.1 → 0.05 | Denser graph captures weaker feature correlations |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 256 → 128 | Return to base; over-parameterised for graph size |
| `--n-blocks` | 3 → 2 | Shallower stack; also allows kt=3 without T_after collapse |
| `--kt` | 2 → 3 | Wider temporal receptive field but only viable with n_blocks≤2 |
| `--dropout` | 0.15 → 0.25 | Stronger regularisation across graph layers |
| `--adj-threshold` | 0.1 → 0.3 | Sparser graph; noise edges may be degrading graph convolution |

---

### 9. MTGNN

**Current tuned config:** `hidden=64, skip_ch=128, n_layers=4, dropout=0.15, lookback=14`

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 64 → 128 | Wider inception and graph channels |
| `--skip-ch` | 128 → 256 | Skip aggregation matches wider hidden |
| `--n-layers` | 4 → 5 | Additional mixing layer for higher-order graph propagation |
| `--d-hop` | 2 → 3 | Higher-order neighbourhood diffusion |
| `--dropout` | 0.15 → 0.10 | Ease regularisation |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--n-layers` | 4 → 3 | Return to base; over-smoothing risk in learned-graph models |
| `--hidden` | 64 → 32 | Return to base; learned graph is expressive — capacity increase may overfit |
| `--dropout` | 0.15 → 0.25 | Stronger regularisation for the adaptive adjacency |
| `--d-hop` | 2 → 1 | Reduce neighbourhood aggregation radius |

---

### 10. STSGCN

**Current tuned config:** `hidden=128, n_layers=3, dropout=0.20, lookback=14`

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 128 → 256 | Wider synchronous graph convolution channels |
| `--n-layers` | 3 → 4 | T_after = 14 − 2×4 = 6 ✓; deeper synchronous stack |
| `--dropout` | 0.20 → 0.10 | Ease regularisation |
| `--cheb-k` | 2 → 3 | Higher-order Chebyshev polynomial on the 3N×3N STSG |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 128 → 64 | Return to base; STSG is already large (3N×3N) |
| `--n-layers` | 3 → 2 | Return to base; sliding-window shrinkage limits useful depth |
| `--dropout` | 0.20 → 0.35 | Stronger regularisation inside each STSGCL layer |
| `--adj-threshold` | 0.1 → 0.3 | Sparser base adjacency if noisy edges degrade STSG construction |

---

### 11. STFGNN

**Current tuned config:** `hidden=128, n_layers=4, dropout=0.30, lookback=14`

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 128 → 256 | Wider channels in both spatial and temporal GCN branches |
| `--n-layers` | 4 → 5 | Additional dual-graph fusion block |
| `--dropout` | 0.30 → 0.20 | Try reducing if extra capacity is regularised naturally by the two-graph fusion |
| `--lookback` | 14 → 28 | Temporal GCN benefits from richer intra-window dynamics |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--n-layers` | 4 → 3 | Return to base; gate α is learned — more layers may cause gate collapse |
| `--hidden` | 128 → 64 | Return to base; dual-graph fusion at 128 may be too large for dataset |
| `--dropout` | 0.30 → 0.45 | Highest-variance graph model; may need even stronger regularisation |
| `--adj-threshold` | 0.1 → 0.2 | Tighten both adjacencies to reduce noise edge influence |

---

### 12. PDR-STGCN

**Current tuned config:** `hidden=256, n_blocks=3, kt=2, dk=64, dropout=0.20, lookback=14`

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 256 → 512 | Largest graph model; more capacity is the primary lever |
| `--n-blocks` | 3 → 4 | T_after = 14 − 2×1×4 = 6 ✓; deeper periodic-dynamic stack |
| `--dk` | 64 → 128 | Richer key/query space for the dynamic attention adjacency |
| `--dropout` | 0.20 → 0.10 | Ease regularisation |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 256 → 128 | Return to base; dynamic attention is already expressive — reduce static capacity |
| `--n-blocks` | 3 → 2 | Return to base; also re-enables kt=3 for wider temporal receptive field |
| `--kt` | 2 → 3 | Wider temporal conv only viable with n_blocks≤2 (T_after = 14−2×2×2 = 6) |
| `--dk` | 64 → 32 | Narrower dynamic attention if adaptive adjacency is overfitting |
| `--dropout` | 0.20 → 0.30 | Stronger regularisation across dynamic path |

---

## Series 3 — Attention

---

### 13. TPA-LSTM

**Current tuned config:** `hidden=128, filters=64, dropout=0.15, lookback=14`

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 128 → 256 | Richer LSTM hidden state and attention context vector |
| `--filters` | 64 → 128 | More temporal pattern filters in the CNN attention module |
| `--dropout` | 0.15 → 0.10 | Ease regularisation |
| `--lookback` | 14 → 28 | More timesteps means more patterns for TPA to detect |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--hidden` | 128 → 64 | Return to base; attention context doubles the head input |
| `--filters` | 64 → 32 | Return to base; more filters without more hidden causes imbalance |
| `--dropout` | 0.15 → 0.25 | Stronger regularisation; TPA attention weights can overfit to spurious patterns |
| `--kernel-size` | 3 → 5 | Wider CNN kernel captures longer-period patterns in the hidden matrix |

---

### 14. ASTGCN

**Current tuned config:** `d_model=128, n_heads=8, n_blocks=3, dropout=0.20, lookback=14`

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--d-model` | 128 → 256 | Larger embeddings for spatial and temporal attention heads |
| `--n-heads` | 8 → 16 | More diverse attention patterns (ensure d_model divisible by n_heads) |
| `--n-blocks` | 3 → 4 | Additional dual-attention + ChebGCN layer |
| `--dropout` | 0.20 → 0.10 | Ease regularisation |
| `--adj-threshold` | 0.1 → 0.05 | Denser graph for richer spatial attention weighting |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--d-model` | 128 → 64 | Return to base; dual attention at 128 is very expressive |
| `--n-heads` | 8 → 4 | Return to base; more heads fragment d_head too much at small feature count |
| `--n-blocks` | 3 → 2 | Return to base; spatial+temporal attention stacks can overfit |
| `--dropout` | 0.20 → 0.35 | Stronger regularisation across attention layers |
| `--adj-threshold` | 0.1 → 0.3 | Sparser graph to reduce noisy spatial attention edges |

---

### 15. TFT

**Current tuned config:** `d_model=128, n_heads=8, n_lstm_layers=2, n_attn_layers=3, dropout=0.25, lookback=14`

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--d-model` | 128 → 256 | Larger embeddings for VSN, LSTM encoder, and self-attention |
| `--n-heads` | 8 → 16 | More diverse temporal attention heads |
| `--n-attn-layers` | 3 → 4 | Additional GRN-Attention-GRN block |
| `--dropout` | 0.25 → 0.15 | Ease regularisation if VSN selection is stable |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--d-model` | 128 → 64 | Return to base; VSN + LSTM + Attn at 128 has very high parameter count |
| `--n-lstm-layers` | 2 → 1 | Return to base; deeper LSTM before Attn may cause gradient saturation |
| `--n-attn-layers` | 3 → 2 | Return to base; additional transformer blocks can overfit on short sequences |
| `--dropout` | 0.25 → 0.40 | Highest dropout of all models — may need even stronger regularisation |
| `--lookback` | 14 → 28 | VSN selects from more timesteps; longer context may improve variable selection |

---

### 16. Autoformer

**Current tuned config:** `d_model=128, n_heads=8, e_layers=3, d_ff=256, dropout=0.20, lookback=14`

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--d-model` | 128 → 256 | Larger FFT-domain representations |
| `--d-ff` | 256 → 512 | Scale point-wise FFN proportionally |
| `--e-layers` | 3 → 4 | Deeper encoder for more autocorrelation refinement passes |
| `--n-heads` | 8 → 16 | More autocorrelation heads (FFT is per-head) |
| `--moving-avg` | 5 → 7 | Wider moving average smooths longer seasonal trends |
| `--dropout` | 0.20 → 0.10 | Ease regularisation if decomposition is stable |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--d-model` | 128 → 64 | Return to base; FFT operations at 128 double memory footprint |
| `--e-layers` | 3 → 2 | Return to base; deeper autocorrelation may overfit to in-sample periodicity |
| `--d-ff` | 256 → 128 | Return to base |
| `--dropout` | 0.20 → 0.30 | Stronger regularisation in the encoder-decoder stack |
| `--lookback` | 14 → 56 | Autocorrelation detects period-based lags; longer windows reveal more lag candidates |

---

### 17. Informer

**Current tuned config:** `d_model=128, n_heads=8, e_layers=3, d_ff=256, dropout=0.15, lookback=14`

> **Distilling note with e_layers=3 and T_in=14:** 14→7→4 (two Conv+MaxPool(2) steps). Adding e_layers=4 gives 14→7→4→2 — borderline. Prefer widening d_model over adding encoder layers.

**If BETTER:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--d-model` | 128 → 256 | Wider ProbSparse attention representations (preferred over more e_layers) |
| `--d-ff` | 256 → 512 | Scale point-wise FFN proportionally |
| `--n-heads` | 8 → 16 | More query groups for sparse attention |
| `--dropout` | 0.15 → 0.10 | Ease regularisation |
| `--lookback` | 14 → 28 | Longer sequences give ProbSparse more to work with; distilling 28→14→7 |

**If WORSE:**
| Lever | Adjustment | Why |
|-------|-----------|-----|
| `--e-layers` | 3 → 2 | Return to base; avoid distilling to length 4 which limits ProbSparse |
| `--d-model` | 128 → 64 | Return to base; AMP mitigates memory but 128 may overfit |
| `--d-ff` | 256 → 128 | Return to base |
| `--dropout` | 0.15 → 0.25 | Increase regularisation inside ProbSparse and decoder layers |
| `--factor` | 5 → 3 | Reduce the top-u active queries to cut attention noise |

---

## Cross-Cutting Adjustments

These apply to all models regardless of series.

### Lookback Window (`--lookback {14, 28, 56}`)

| Situation | Action |
|-----------|--------|
| Model captures short-term patterns well but misses weekly rhythm | Try `--lookback 28` or `--lookback 56` |
| Model noise-prone; val loss unstable | Try `--lookback 14` (less input noise) |
| Horizon 6–7 errors are disproportionately high | Try `--lookback 56` (more future-relevant context) |

### MCO Inclusion (`--include-mco`)

| Situation | Action |
|-----------|--------|
| Model performs well on pre-2020 but poorly on 2022+ | Add `--include-mco`; the 2020–2021 regime shift is informative context |
| Model trained without MCO underperforms tuned with MCO | MCO adds noise; keep excluded |

### Loss Function (`--loss {huber, mse, mae}`)

| Situation | Action |
|-----------|--------|
| MAPE% is the dominant drag on Combined% (outliers in error) | Switch to `--loss mae` for L1 robustness |
| RMSE% is the dominant drag (large individual errors) | Switch to `--loss mse` for stronger quadratic penalisation |
| Both MAPE% and RMSE% drag equally | Keep default `--loss huber` |

### Weight Decay (`--weight-decay`)

| Situation | Action |
|-----------|--------|
| Model overfits (train Combined% >> test Combined%) and increasing dropout alone is insufficient | Try `--weight-decay 5e-4` or `--weight-decay 1e-3` |
| Model underfits and decreasing dropout alone is insufficient | Try `--weight-decay 1e-5` |

### LR Warm-up (`--warmup-epochs`)

| Situation | Action |
|-----------|--------|
| Early training loss spikes then recovers — gradient instability | Increase `--warmup-epochs 10` |
| Model converges very slowly; patience triggers before a good minimum is found | Reduce `--warmup-epochs 2` to reach plateau faster |

### Learning Rate (`--lr`)

> Treat this as a last resort — the shared LR ensures comparability across all 15 models.

| Situation | Action |
|-----------|--------|
| Loss does not decrease after warmup (all models on same task) | Try `--lr 5e-4` |
| Loss oscillates and never converges | Try `--lr 2e-4` |

---

## Quick-Reference Summary

| Model | Primary "if better" lever | Primary "if worse" lever |
|-------|--------------------------|--------------------------|
| LSTM | hidden 128→256, layers 2→3 | hidden 128→64, layers 2→1, dropout +0.10 |
| BiLSTM | hidden 128→256, layers 2→3 | hidden 128→64, layers 2→1, dropout +0.10 |
| CNN-LSTM (seq) | cnn_filters 64→128, hidden 64→128 | cnn_filters 64→32, dropout +0.15 |
| CNN-LSTM (par) | cnn_filters 64→128, lookback 28→56 | dropout +0.15, lookback 28→14 |
| CNN-LSTM (aug) | cnn_filters 64→128, hidden 64→128 | cnn_filters 64→32, dropout +0.15 |
| CNN-BiLSTM | hidden 128→256, cnn_filters 64→128 | hidden 128→64, dropout +0.15 |
| ST-LSTM | hidden 128→256, spatial_hidden 64→128 | hidden 128→64, spatial_hidden 64→32 |
| STGCN | hidden 256→512, n_blocks 3→4 | hidden 256→128, n_blocks 3→2 |
| MTGNN | hidden 64→128, n_layers 4→5 | n_layers 4→3, hidden 64→32 |
| STSGCN | hidden 128→256, n_layers 3→4 | hidden 128→64, n_layers 3→2 |
| STFGNN | hidden 128→256, n_layers 4→5 | n_layers 4→3, dropout +0.15 |
| PDR-STGCN | hidden 256→512, n_blocks 3→4 | hidden 256→128, n_blocks 3→2 |
| TPA-LSTM | hidden 128→256, filters 64→128 | hidden 128→64, filters 64→32, dropout +0.10 |
| ASTGCN | d_model 128→256, n_heads 8→16 | d_model 128→64, n_blocks 3→2, dropout +0.15 |
| TFT | d_model 128→256, n_attn_layers 3→4 | d_model 128→64, n_attn_layers 3→2, dropout +0.15 |
| Autoformer | d_model 128→256, e_layers 3→4 | d_model 128→64, e_layers 3→2 |
| Informer | d_model 128→256 (not e_layers) | e_layers 3→2, dropout +0.10 |
