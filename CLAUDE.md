# CLAUDE.md

> Last updated: 2026-06-11

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **2026-06-11 revision notes** (full audit in `REVISION.md`):
> - `cuda.py` was renamed `check_cuda.py` (the old name shadowed the `cuda.bindings` package that newer PyTorch builds import).
> - `sequence_builder.py` now resolves the 16 `X_future` temporal columns **by name** from feature metadata (they are non-contiguous in the aligned column order); `split_dates.json` gains `temporal_feat_indices` / `temporal_feat_names`. Sequence sets built before this fix carry incorrect `X_future` content and must be rebuilt.
> - HMT-TSF's `FEAT_GROUPS` now matches the actual `feature_align.py` column order (groups take tuples of index segments), and the residual-boost apply/skip decision is gated on **validation** metrics (was test). Existing HMT-TSF results pre-date these fixes.

## Project

Malaysian transit ridership forecasting research (Masters FYP). Compares 15 deep-learning models across three series against a shared dataset of 8 spatio-temporal feature sources.

## Hardware

- RAM: 32 GB
- GPU: NVIDIA A100 (32 GB VRAM)

## Environment

```bash
# Activate the project venv (Windows)
.venv\Scripts\activate

# Verify GPU
python check_cuda.py     # prints True if CUDA is available
```

All scripts are run from the **repository root**. Each model file manually inserts the root onto `sys.path` at the top so that `src.utils` is importable without a package install.

## Running the Feature Pipeline

Full step-by-step instructions are in `src/features/PIPELINE.md`. The short version:

```bash
# 1a–1d. Core independent cleaning (any order)
python src/features/ridership.py
python src/features/fuelprice.py
python src/features/holiday.py
python src/features/rainfall.py

# 1e. GADM
python src/features/gadm.py

# 1f. GTFS (one call per operator)
python src/features/gtfs.py --input data/raw/gtfs_rapid_rail_kl --output data/cleaned/gtfs_rapid_rail_kl --operator rapid_rail_kl
# (repeat gtfs.py for rapidbus_kl, rapidbus_penang, ktmb)

# 1g–1h. Depends on gadm + gtfs outputs
python src/features/population.py
python src/features/osm.py

# 2. Merge all 8 sources → two daily matrices (MCO included + MCO excluded)
python src/features/feature_align.py
# outputs: data/features/features_aligned.csv (MCO included)
#          data/features/features_aligned_no_mco.csv (MCO excluded)

# 3. Sliding-window tensors — two sets per lookback (MCO excluded + MCO included)
python src/features/sequence_builder.py --features-path data/features/features_aligned_no_mco.csv --T-in 14 --out-dir data/sequences/lstm
python src/features/sequence_builder.py --features-path data/features/features_aligned.csv         --T-in 14 --out-dir data/sequences/lstm_mco
python src/features/sequence_builder.py --features-path data/features/features_aligned_no_mco.csv --T-in 28 --out-dir data/sequences/lookback_28
python src/features/sequence_builder.py --features-path data/features/features_aligned.csv         --T-in 28 --out-dir data/sequences/lookback_28_mco
python src/features/sequence_builder.py --features-path data/features/features_aligned_no_mco.csv --T-in 56 --out-dir data/sequences/lookback_56
python src/features/sequence_builder.py --features-path data/features/features_aligned.csv         --T-in 56 --out-dir data/sequences/lookback_56_mco
# Or run all of the above in one shot:
# python src/features/run_pipeline.py --skip-clean --skip-align
```

## Running a Model

All 15 models are run the same way — from the repo root, passing the script path:

```bash
python src/models/spatio-temporal-based/lstm.py
python src/models/graph-based/stgcn.py
python src/models/graph-based/stsgcn.py
python src/models/graph-based/stfgnn.py
python src/models/graph-based/pdr_stgcn.py
```

Each model script accepts `--seq-dir`, `--epochs`, `--batch-size`, `--lr`, `--patience`, `--device`, `--seed`, and model-specific hyperparameter flags. See the docstring at the top of each file. `--device auto` selects CUDA → MPS → CPU automatically.

Each model script docstring includes a `References` section citing one Scopus-indexed journal article (2022–2027) that discusses the architecture. Full citations are also listed in `src/models/MODEL.md` under the **Journal References** section.

## Running the Hybrid SOTA Model

HMT-TSF is in `src/models/hybrid/`. Run from the repo root:

```bash
# Default (lookback=14, d_model=64, 3 TCN blocks)
python src/models/hybrid/hmttsf.py

# Longer lookback with larger model
python src/models/hybrid/hmttsf.py --lookback 28 --n-tcn-blocks 4 --d-model 192

# With CatBoost post-hoc residual boosting
python src/models/hybrid/hmttsf.py --use-catboost

# With SHAP feature importance
python src/models/hybrid/hmttsf.py --shap --shap-samples 150

# Maximum configuration (84-day lookback, SHAP, CatBoost)
python src/models/hybrid/hmttsf.py \
  --lookback 84 --d-model 256 --n-tcn-blocks 5 \
  --graph-hidden 128 --dropout 0.15 \
  --use-catboost --shap \
  --epochs 150
```

HMT-TSF-specific flags (in addition to the shared flags below):

| Flag | Default | Description |
|------|---------|-------------|
| `--d-model {64,128,192,256}` | `64` | Model hidden dimension |
| `--n-tcn-blocks N` | `3` | TCN blocks per scale |
| `--graph-hidden {32,64,128}` | `64` | GCN hidden dimension |
| `--n-regimes N` | `3` | Regime embedding count |
| `--use-catboost` | off | CatBoost post-hoc residual boosting |
| `--no-boost` | off | Disable neural boost head |
| `--no-revin` | off | Disable RevIN input normalisation |
| `--shap` | off | Run SHAP feature importance |
| `--shap-samples N` | `100` | SHAP background samples |
| `--prob-sparse` | off | Replace full self-attention with ProbSparse (O(L log L)); no-op for lb≤14 |
| `--sparse-factor N` | `5` | c factor: n_top = c·⌈ln(L)⌉ active queries |
| `--smooth-weight λ` | `0.01` | Temporal smoothness regularisation |
| `--loss-decay γ` | `0.9` | Geometric decay for weighted Huber steps |

Lookback choices extend to `{7, 14, 28, 56, 84}`. Sequence dirs for lookback 7 and 84 must be built first if needed:
```bash
python src/features/sequence_builder.py --T-in 7
python src/features/sequence_builder.py --T-in 84
```

Output is written to `src/outputs/hmttsf/{YYYYMMDD_HHMMSS}/`. Full architecture details and per-component Scopus-indexed journal references (2022–2027) are in `src/models/hybrid/HMT-TSF.md`.

## Running a Tuned Model

Fine-tuned variants of all 14 base models are in three mirrored folders (HMT-TSF has no separate tuned variant). Run from the repo root:

```bash
# Spatio-temporal tuned
python src/models/spatio-temporal-tuned/lstm.py
python src/models/spatio-temporal-tuned/bilstm.py
python src/models/spatio-temporal-tuned/cnnlstm.py    # --mode sequential|parallel|augmented
python src/models/spatio-temporal-tuned/cnnbilstm.py
python src/models/spatio-temporal-tuned/stlstm.py

# Graph tuned
python src/models/graph-tuned/stgcn.py
python src/models/graph-tuned/mtgnn.py
python src/models/graph-tuned/stsgcn.py
python src/models/graph-tuned/stfgnn.py
python src/models/graph-tuned/pdr_stgcn.py

# Attention tuned
python src/models/attention-tuned/tpalstm.py
python src/models/attention-tuned/astgcn.py
python src/models/attention-tuned/autoformer.py
python src/models/attention-tuned/informer.py
```

CNN-LSTM tuned mode-specific run commands (each mode has an independently selected best lookback):
```bash
python src/models/spatio-temporal-tuned/cnnlstm.py --mode sequential --lookback 14
python src/models/spatio-temporal-tuned/cnnlstm.py --mode parallel   --lookback 28
python src/models/spatio-temporal-tuned/cnnlstm.py --mode augmented  --lookback 14
```

Tuned outputs are written to `src/outputs/{model_name}_tuned/`. Full rationale and per-model details are in `src/models/MODEL.md`.

New shared flags added to all 15 models:

| Flag | Default | Description |
|------|---------|-------------|
| `--lookback {14,28,56}` | `14` | Look-back window for the 15 base models; auto-resolves `--seq-dir` when not set explicitly |
| `--loss {mse,huber,mae}` | `huber` | Training loss function |
| `--warmup-epochs N` | `5` | Linear LR warm-up epochs before ReduceLROnPlateau takes over |

Sequence directory auto-resolution (base models): `--lookback 14` → `data/sequences/lstm/`, `--lookback 28` → `data/sequences/lookback_28/`, `--lookback 56` → `data/sequences/lookback_56/`.

HMT-TSF additionally supports `--lookback 7` → `data/sequences/lookback_7/` and `--lookback 84` → `data/sequences/lookback_84/` (build both MCO conditions first — see `src/features/PIPELINE.md` Step 3).

## Architecture

### Data Flow

```
data/raw/                      8 raw sources
    └── src/features/          cleaning scripts (one per source)
data/cleaned/                  cleaned CSVs, GeoJSONs, .npy adjacency matrices
    └── feature_align.py       merges onto daily index; also derives lag + trend features
                               → features_aligned.csv        (MCO included, 2,557 days, 2019-01-01 – 2025-12-31)
                               → features_aligned_no_mco.csv (MCO excluded, 1,461 days, 2022-01-01 – 2025-12-31)
data/features/                 both aligned CSVs + feature_metadata{,_no_mco}.json (79 cols)
    └── sequence_builder.py    called twice per lookback (once per MCO condition)
data/sequences/lstm/           lookback=14, MCO excluded: X/y_train/val/test.npy, scalers, split_dates.json
data/sequences/lstm_mco/       lookback=14, MCO included: same layout
data/sequences/lookback_28/    lookback=28, MCO excluded
data/sequences/lookback_28_mco/ lookback=28, MCO included
data/sequences/lookback_56/    lookback=56, MCO excluded
data/sequences/lookback_56_mco/ lookback=56, MCO included
    └── src/models/**/*.py     model training (--lookback + --seq-dir selects the right dir)
src/outputs/{model}/           timestamped run dirs with results.json, plots, model.pt
```

**Feature count breakdown** (all 8 sources present, 79 total as of current pipeline):

| Group | ~Count | Features |
|---|---|---|
| Targets | 13 | 12 service lines + total_ridership |
| Temporal | 16 | holiday flags/lead-lag/cyclical + year + day_of_year |
| External | 30 | fuel (15) + rainfall (15) |
| Lag | 3 | ridership_lag_7, ridership_lag_14, ridership_lag_28 |
| Static | 17 | population + GTFS + OSM POI + GADM |

### The Three Model Series (Base) + Hybrid SOTA

| Series | Location | Models |
|---|---|---|
| Spatio-temporal (LSTM-family) | `src/models/spatio-temporal-based/` | LSTM, BiLSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM |
| Graph-based | `src/models/graph-based/` | STGCN, MTGNN, STSGCN, STFGNN, PDR-STGCN |
| Attention-based | `src/models/attention-based/` | TPA-LSTM, ASTGCN, Autoformer, Informer |

(TPA-LSTM is architecturally LSTM-family but its script lives only in `src/models/attention-based/tpalstm.py`.)
| Hybrid SOTA | `src/models/hybrid/` | HMT-TSF |

### The Three Tuned Series

| Series | Location | Models |
|---|---|---|
| Spatio-temporal tuned | `src/models/spatio-temporal-tuned/` | LSTM, BiLSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM |
| Graph tuned | `src/models/graph-tuned/` | STGCN, MTGNN, STSGCN, STFGNN, PDR-STGCN |
| Attention tuned | `src/models/attention-tuned/` | TPA-LSTM, ASTGCN, Autoformer, Informer |

**Canonical reference model:** `src/models/spatio-temporal-based/stlstm.py` — the training loop, output structure, and comparison pattern here should be followed when adding new models.

### Graph Approach (features-as-nodes)

All graph-based, ASTGCN, and HMT-TSF models treat the N input features as graph nodes rather than geographic locations. STGCN, ASTGCN, STSGCN, STFGNN, PDR-STGCN, and HMT-TSF build a fixed spatial adjacency from absolute Pearson correlation of feature columns in `X_train` (threshold=0.1). STFGNN additionally builds a temporal adjacency from the correlation of each node's mean temporal profile. STSGCN constructs a 3N×3N Spatial-Temporal Synchronous Graph (STSG) that captures both spatial and temporal correlations in one synchronous adjacency. PDR-STGCN combines a static sym-normalised correlation adjacency with an input-adaptive dynamic attention adjacency (mixed via a learned scalar λ), and adds a periodicity-aware 2-channel input encoding (original signal + weekly lag-difference). MTGNN learns its adjacency end-to-end from node embeddings.

ASTGCN additionally computes a scaled Chebyshev Laplacian `L_tilde = -A_sym`.

### Comparison Chain

Each model auto-detects and compares against all prior model runs. Results are loaded from `results.json` via `load_model_results()` in `src/utils/comparison_table.py` — if a path is not supplied via CLI, it globs the most recent `results.json` under that model's output directory. The chain grows as each new model is added:

```
LSTM (2-way) → BiLSTM (3-way) → TPA-LSTM (4-way) → CNN-LSTM (5-way)
→ CNN-BiLSTM (6-way) → ST-LSTM (7-way) → STGCN (8-way)
→ ASTGCN (9-way) → STSGCN (10-way) → PDR-STGCN (11-way) → MTGNN (12-way)
→ STFGNN (13-way) → Autoformer (14-way) → Informer (15-way)
→ HMT-TSF (16-way)
```

### Shared Utilities (`src/utils/`)

- **`metrics.py`** — `compute_metrics(y_true, y_pred)` returns `Combined%`, `MAPE%`, `MAE%`, `RMSE%`, `R²`, `MAE`, `RMSE`. Combined% = `max(0, 100 − MAPE − MAE% − RMSE%)`. All percentage terms use mean-demand normalisation so they are directly addable. See `src/METRICS.md` for full definitions.
- **`comparison_table.py`** — `print_comparison_table()` renders an N-way console table; `plot_comparison()` saves a 6-panel PNG (bar charts for overall metrics + per-horizon line charts). Both are called at the end of every model script.

### AMP (Mixed Precision)

The attention-based models (ASTGCN, Autoformer, Informer) and HMT-TSF use `torch.cuda.amp.GradScaler` + `autocast` for mixed-precision training on the A100. Autoformer wraps its FFT ops with an explicit `float32` cast inside `autocast` for numerical stability. LSTM-family and graph-based models do not use AMP.

### Output Structure

Each model run writes to `src/outputs/{model_name}/{YYYYMMDD_HHMMSS}/`:
- `model.pt` — saved model weights
- `results.json` — overall + per-step metrics dict (the comparison currency)
- `predictions.npy` — raw test predictions
- `comparison_{n}way.png` — multi-panel comparison plot

### Key Hyperparameters / Defaults

- `T_in=14` (look-back window, choices: 14/28/56), `T_out=7` (forecast horizon)
- Chronological split: 70% train / 15% val / 15% test
- MCO period (2020-03-18 – 2021-12-31) split at feature_align stage: `features_aligned_no_mco.csv` excludes it; `features_aligned.csv` retains it. Both sequence sets are always built.
- Sequences stored as `float16` by default; load and cast to `float32` before feeding to models
- All scalers fitted on training split only (`scaler_X.pkl`, `scaler_y.pkl`)

### Training Optimizations (all 15 models)

All models use the following improved training setup (structural changes only — no hyperparameter value changes):

| Component | Before | After | Rationale |
|-----------|--------|-------|-----------|
| Optimiser | `Adam` | `AdamW` | Decoupled weight decay (Loshchilov & Hutter 2019) |
| Loss | `MSELoss` | `HuberLoss(delta=1.0)` (default) | Robust to ridership outliers; selectable via `--loss` |
| LR schedule | `ReduceLROnPlateau` only | Linear warmup → `ReduceLROnPlateau` | Avoids unstable early updates; warmup via `--warmup-epochs` (default 5 for 15 base models, 8 for HMT-TSF) |

### Initial-Run (No Fine-Tuning) Hyperparameters

Shared baseline training schedule (argparse defaults): `epochs=150`, `batch_size=32`, `lr=1e-3`, `patience=15`, `dropout=0.1`, `weight_decay=1e-4`, `warmup_epochs=5`. Per-model deviations baked into the base scripts:

| Model | Deviations from shared values |
|-------|-------------------------------|
| STGCN | `dropout=0.20`, `weight_decay=2e-4` |
| STSGCN | `epochs=200`, `lr=5e-4`, `patience=25` |
| STFGNN | `dropout=0.20`, `weight_decay=3e-4` |
| PDR-STGCN | `dropout=0.15`, `weight_decay=2e-4` |
| ASTGCN | `dropout=0.15`, `weight_decay=2e-4` |
| Autoformer | `dropout=0.15`, `weight_decay=2e-4` |
| HMT-TSF | `patience=20`, `warmup_epochs=8`, `weight_decay=1e-3` |

All other models (LSTM, BiLSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM, TPA-LSTM, MTGNN, Informer) use the shared values exactly.

Architecture defaults (per-model, aligned with the argparse defaults):
- LSTM / BiLSTM / TPA-LSTM: `hidden=64`, `layers=1` (TPA-LSTM also `filters=32`)
- CNN-LSTM / CNN-BiLSTM: `hidden=64`, `cnn_filters=32`, `cnn_layers=2`
- ST-LSTM: `hidden=64`, `spatial_hidden=32`
- STGCN: `hidden=128`, `kt=3`, `n_blocks=2`, `cheb_k=3`
- PDR-STGCN: `hidden=160`, `kt=3`, `n_blocks=2`, `dk=48`, `period=7`
- MTGNN: `hidden=32`, `skip_ch=64`, `n_layers=3`, `d_emb=10`, `d_hop=2`
- STSGCN: `hidden=96`, `n_layers=3`, `cheb_k=3`
- STFGNN: `hidden=64`, `n_layers=3`
- ASTGCN / Autoformer / Informer: `d_model=64`, `n_heads=4`

### Tuned Architecture Defaults (Fine-Tuned Variants)

Hyperparameters changed in the tuned scripts (values are the actual argparse defaults; anything not listed is unchanged from the base script):

| Model | Tuned Defaults |
|-------|---------------|
| LSTM (tuned) | `hidden=128`, `layers=2`, `dropout=0.20`, `weight_decay=2e-4` |
| BiLSTM (tuned) | `hidden=256`, `layers=3`, `dropout=0.1`, `weight_decay=1e-4` |
| CNN-LSTM (tuned) | `cnn_filters=64`, `dropout=0.20` (hidden/layers unchanged) |
| CNN-BiLSTM (tuned) | `hidden=128`, `cnn_filters=64`, `cnn_layers=1`, `dropout=0.25` |
| ST-LSTM (tuned) | `hidden=128`, `spatial_hidden=128`, `dropout=0.30`, `weight_decay=1e-4` |
| STGCN (tuned) | `hidden=256`, `n_blocks=2`, `kt=3`, `dropout=0.25`, `weight_decay=3e-4` |
| MTGNN (tuned) | `hidden=64`, `skip_ch=64`, `n_layers=3`, `d_emb=7`, `dropout=0.30`, `weight_decay=3e-4` |
| STSGCN (tuned) | `hidden=128`, `n_layers=3`, `dropout=0.25`, `weight_decay=2e-4`, `lr=1e-3`, `epochs=200`, `patience=25` |
| STFGNN (tuned) | `hidden=128`, `n_layers=3`, `dropout=0.35`, `weight_decay=3e-4`, `adj_threshold=0.15` |
| PDR-STGCN (tuned) | `hidden=256`, `n_blocks=3`, `kt=2`, `dk=64`, `period=7`, `dropout=0.25`, `weight_decay=2e-4` |
| TPA-LSTM (tuned) | `hidden=256`, `filters=128`, `dropout=0.15`, `weight_decay=1e-4` |
| ASTGCN (tuned) | `d_model=128`, `n_heads=8`, `n_blocks=3`, `dropout=0.20`, `weight_decay=1e-4` |
| Autoformer (tuned) | `d_model=128`, `n_heads=8`, `e_layers=3`, `d_ff=256`, `dropout=0.25`, `weight_decay=2e-4` |
| Informer (tuned) | `d_model=256`, `n_heads=16`, `e_layers=3`, `d_ff=512`, `dropout=0.1`, `weight_decay=1e-4` |

Full per-model rationale is in `src/models/MODEL.md`.
