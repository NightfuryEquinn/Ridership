# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Malaysian transit ridership forecasting research (Masters FYP). Compares 15 deep-learning models across three series against a shared dataset of 8 spatio-temporal feature sources.

## Hardware

- CPU: 13th Gen Intel Core i5
- RAM: 32 GB
- GPU: NVIDIA GeForce RTX 4050 (6 GB VRAM)

## Environment

```bash
# Activate the project venv (Windows)
.venv\Scripts\activate

# Verify GPU
python cuda.py     # prints True if CUDA is available
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

# 2. Merge all 8 sources → daily matrix (also computes lag + trend features)
python src/features/feature_align.py

# 3. Sliding-window tensors → data/sequences/{lstm | lookback_28 | lookback_56}/
python src/features/sequence_builder.py               # default: --T-in 14
python src/features/sequence_builder.py --T-in 28     # → data/sequences/lookback_28/
python src/features/sequence_builder.py --T-in 56     # → data/sequences/lookback_56/
```

## Running a Model

All 15 models are run the same way — from the repo root, passing the script path:

```bash
python src/models/spatio-temporal-based/lstm.py
python src/models/graph-based/stgcn.py
python src/models/graph-based/stsgcn.py
python src/models/graph-based/stfgnn.py
python src/models/graph-based/pdr_stgcn.py
python src/models/attention-based/tft.py
```

Each model script accepts `--seq-dir`, `--epochs`, `--batch-size`, `--lr`, `--patience`, `--device`, `--seed`, and model-specific hyperparameter flags. See the docstring at the top of each file. `--device auto` selects CUDA → MPS → CPU automatically.

## Running a Tuned Model

Fine-tuned variants of all 15 models are in three mirrored folders. Run from the repo root:

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
python src/models/attention-tuned/tft.py
python src/models/attention-tuned/autoformer.py
python src/models/attention-tuned/informer.py
```

CNN-LSTM tuned mode-specific run commands (each mode has an independently selected best lookback):
```bash
python src/models/spatio-temporal-tuned/cnnlstm.py --mode sequential --lookback 14
python src/models/spatio-temporal-tuned/cnnlstm.py --mode parallel   --lookback 28
python src/models/spatio-temporal-tuned/cnnlstm.py --mode augmented  --lookback 14
```

Tuned outputs are written to `src/outputs/{model_name}_tuned/`. Full rationale and per-model details are in `src/models/TUNED-MODEL.md`.

New shared flags added to all 15 models:

| Flag | Default | Description |
|------|---------|-------------|
| `--lookback {14,28,56}` | `14` | Look-back window; auto-resolves `--seq-dir` when not set explicitly |
| `--loss {mse,huber,mae}` | `huber` | Training loss function |
| `--warmup-epochs N` | `5` | Linear LR warm-up epochs before ReduceLROnPlateau takes over |

Sequence directory auto-resolution: `--lookback 14` → `data/sequences/lstm/`, `--lookback 28` → `data/sequences/lookback_28/`, `--lookback 56` → `data/sequences/lookback_56/`.

## Architecture

### Data Flow

```
data/raw/                      8 raw sources
    └── src/features/          cleaning scripts (one per source)
data/cleaned/                  cleaned CSVs, GeoJSONs, .npy adjacency matrices
    └── feature_align.py       merges onto daily index → features_aligned.csv
                               also derives: ridership_lag_{7,14,28}, year, day_of_year
data/features/                 features_aligned.csv (~69 cols) + feature_metadata.json
    └── sequence_builder.py    sliding windows, MinMaxScaler, chronological split
data/sequences/lstm/           lookback=14 (default): X_train/val/test.npy, y_*.npy, scaler_X/y.pkl, split_dates.json
data/sequences/lookback_28/    lookback=28: same layout
data/sequences/lookback_56/    lookback=56: same layout
    └── src/models/**/*.py     model training (--lookback selects the right dir)
src/outputs/{model}/           timestamped run dirs with results.json, plots, model.pt
```

**Feature count breakdown** (all 8 sources present, ~69 total):

| Group | ~Count | Features |
|---|---|---|
| Targets | 13 | 12 service lines + total_ridership |
| Temporal | 16 | holiday flags/lead-lag/cyclical + year + day_of_year |
| External | 30 | fuel (15) + rainfall (15) |
| Lag | 3 | ridership_lag_7, ridership_lag_14, ridership_lag_28 |
| Static | 17 | population + GTFS + OSM POI + GADM |

### The Three Model Series (Base)

| Series | Location | Models |
|---|---|---|
| Spatio-temporal (LSTM-family) | `src/models/spatio-temporal-based/` | LSTM, BiLSTM, TPA-LSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM |
| Graph-based | `src/models/graph-based/` | STGCN, MTGNN, STSGCN, STFGNN, PDR-STGCN |
| Attention-based | `src/models/attention-based/` | TPA-LSTM, ASTGCN, TFT, Autoformer, Informer |

### The Three Tuned Series

| Series | Location | Models |
|---|---|---|
| Spatio-temporal tuned | `src/models/spatio-temporal-tuned/` | LSTM, BiLSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM |
| Graph tuned | `src/models/graph-tuned/` | STGCN, MTGNN, STSGCN, STFGNN, PDR-STGCN |
| Attention tuned | `src/models/attention-tuned/` | TPA-LSTM, ASTGCN, TFT, Autoformer, Informer |

**Canonical reference model:** `src/models/spatio-temporal-based/stlstm.py` — the training loop, output structure, and comparison pattern here should be followed when adding new models.

### Graph Approach (features-as-nodes)

All graph-based and ASTGCN models treat the N input features as graph nodes rather than geographic locations. STGCN, ASTGCN, STSGCN, STFGNN, and PDR-STGCN build a fixed spatial adjacency from absolute Pearson correlation of feature columns in `X_train` (threshold=0.1). STFGNN additionally builds a temporal adjacency from the correlation of each node's mean temporal profile. STSGCN constructs a 3N×3N Spatial-Temporal Synchronous Graph (STSG) that captures both spatial and temporal correlations in one synchronous adjacency. PDR-STGCN combines a static sym-normalised correlation adjacency with an input-adaptive dynamic attention adjacency (mixed via a learned scalar λ), and adds a periodicity-aware 2-channel input encoding (original signal + weekly lag-difference). MTGNN learns its adjacency end-to-end from node embeddings.

ASTGCN additionally computes a scaled Chebyshev Laplacian `L_tilde = -A_sym`.

### Comparison Chain

Each model auto-detects and compares against all prior model runs. Results are loaded from `results.json` via `load_model_results()` in `src/utils/comparison_table.py` — if a path is not supplied via CLI, it globs the most recent `results.json` under that model's output directory. The chain grows as each new model is added:

```
LSTM (2-way) → BiLSTM (3-way) → TPA-LSTM (4-way) → CNN-LSTM (5-way)
→ CNN-BiLSTM (6-way) → ST-LSTM (7-way) → STGCN (8-way)
→ MTGNN (9-way) → STSGCN (10-way) → STFGNN (11-way) → PDR-STGCN (12-way)
→ ASTGCN (13-way) → TFT (14-way) → Autoformer (15-way) → Informer (16-way)
```

### Shared Utilities (`src/utils/`)

- **`metrics.py`** — `compute_metrics(y_true, y_pred)` returns `Combined%`, `MAPE%`, `MAE%`, `RMSE%`, `R²`, `MAE`, `RMSE`. Combined% = `max(0, 100 − MAPE − MAE% − RMSE%)`. All percentage terms use mean-demand normalisation so they are directly addable. See `src/METRICS.md` for full definitions.
- **`comparison_table.py`** — `print_comparison_table()` renders an N-way console table; `plot_comparison()` saves a 6-panel PNG (bar charts for overall metrics + per-horizon line charts). Both are called at the end of every model script.

### AMP (Mixed Precision)

The four new attention-based models (ASTGCN, TFT, Autoformer, Informer) use `torch.cuda.amp.GradScaler` + `autocast` to fit within RTX 4050 6 GB VRAM. Autoformer wraps its FFT ops with an explicit `float32` cast inside `autocast` for numerical stability. LSTM-family and graph-based models do not use AMP.

### Output Structure

Each model run writes to `src/outputs/{model_name}/{YYYYMMDD_HHMMSS}/`:
- `model.pt` — saved model weights
- `results.json` — overall + per-step metrics dict (the comparison currency)
- `predictions.npy` — raw test predictions
- `comparison_{n}way.png` — multi-panel comparison plot

### Key Hyperparameters / Defaults

- `T_in=14` (look-back window, choices: 14/28/56), `T_out=7` (forecast horizon)
- Chronological split: 70% train / 15% val / 15% test
- MCO period (2020-03-18 – 2021-12-31) excluded from sequences by default
- Sequences stored as `float16` by default; load and cast to `float32` before feeding to models
- All scalers fitted on training split only (`scaler_X.pkl`, `scaler_y.pkl`)

### Training Optimizations (all 15 models)

All models use the following improved training setup (structural changes only — no hyperparameter value changes):

| Component | Before | After | Rationale |
|-----------|--------|-------|-----------|
| Optimiser | `Adam` | `AdamW` | Decoupled weight decay (Loshchilov & Hutter 2019) |
| Loss | `MSELoss` | `HuberLoss(delta=1.0)` (default) | Robust to ridership outliers; selectable via `--loss` |
| LR schedule | `ReduceLROnPlateau` only | Linear warmup (5 epochs) → `ReduceLROnPlateau` | Avoids unstable early updates; warmup via `--warmup-epochs` |

### Standardised Initial-Run (No Fine-Tuning) Hyperparameters

All 15 models share the same training schedule for the initial baseline comparison run. These values are baked in as argparse defaults and must not be changed per-model without explicit justification.

| Parameter | Value | Applies to |
|-----------|-------|------------|
| `epochs` | 150 | all models |
| `batch_size` | 32 | all models |
| `lr` | 1e-3 | all models |
| `patience` | 15 | all models |
| `dropout` | 0.1 | all models |
| `weight_decay` | 1e-4 | all models |

Architecture defaults (per-model, aligned with MODEL.md):
- LSTM / BiLSTM / TPA-LSTM: `hidden=64`, `layers=1`
- CNN-LSTM / CNN-BiLSTM: `hidden=64`, `cnn_filters=32`, `cnn_layers=2`
- ST-LSTM: `hidden=64`, `spatial_hidden=32`
- STGCN / PDR-STGCN: `hidden=128`, `kt=3`, `n_blocks=2`
- MTGNN: `hidden=32`, `skip_ch=64`, `n_layers=3`, `d_emb=10`, `d_hop=2`
- STSGCN: `hidden=64`, `n_layers=2`, `cheb_k=2`
- STFGNN: `hidden=64`, `n_layers=3`
- ASTGCN / TFT / Autoformer / Informer: `d_model=64`, `n_heads=4`

### Tuned Architecture Defaults (Fine-Tuned Variants)

Architecture hyperparameters changed in the tuned scripts (training params unchanged):

| Model | Tuned Defaults |
|-------|---------------|
| LSTM (tuned) | `hidden=128`, `layers=2`, `dropout=0.15` |
| BiLSTM (tuned) | `hidden=128`, `layers=2`, `dropout=0.15` |
| CNN-LSTM (tuned) | `cnn_filters=64`, `dropout=0.20` (hidden/layers unchanged) |
| CNN-BiLSTM (tuned) | `hidden=128`, `cnn_filters=64`, `dropout=0.25` |
| ST-LSTM (tuned) | `hidden=128`, `spatial_hidden=64`, `dropout=0.20` |
| STGCN (tuned) | `hidden=256`, `n_blocks=3`, `kt=2`, `dropout=0.15` |
| MTGNN (tuned) | `hidden=64`, `skip_ch=128`, `n_layers=4`, `dropout=0.15` |
| STSGCN (tuned) | `hidden=128`, `n_layers=3`, `dropout=0.20` |
| STFGNN (tuned) | `hidden=128`, `n_layers=4`, `dropout=0.30` |
| PDR-STGCN (tuned) | `hidden=256`, `n_blocks=3`, `kt=2`, `dk=64`, `dropout=0.20` |
| TPA-LSTM (tuned) | `hidden=128`, `filters=64`, `dropout=0.15` |
| ASTGCN (tuned) | `d_model=128`, `n_heads=8`, `n_blocks=3`, `dropout=0.20` |
| TFT (tuned) | `d_model=128`, `n_heads=8`, `n_lstm_layers=2`, `n_attn_layers=3`, `dropout=0.25` |
| Autoformer (tuned) | `d_model=128`, `n_heads=8`, `e_layers=3`, `d_ff=256`, `dropout=0.20` |
| Informer (tuned) | `d_model=128`, `n_heads=8`, `e_layers=3`, `d_ff=256`, `dropout=0.15` |

Full per-model rationale is in `src/models/TUNED-MODEL.md`.
