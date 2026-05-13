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
# 1. Independent cleaning (any order)
python src/features/ridership.py
python src/features/fuelprice.py
python src/features/holiday.py
python src/features/rainfall.py
python src/features/gadm.py
python src/features/gtfs.py --input data/raw/gtfs_rapid_rail_kl --output data/cleaned/gtfs_rapid_rail_kl --operator rapid_rail_kl
# (repeat gtfs.py for rapidbus_kl, rapidbus_penang, ktmb)

# 2. Depends on gadm + gtfs outputs
python src/features/population.py
python src/features/osm.py

# 3. Merge all sources → daily matrix
python src/features/feature_align.py

# 4. Sliding-window tensors → data/sequences/lstm/
python src/features/sequence_builder.py
```

## Running a Model

All 15 models are run the same way — from the repo root, passing the script path:

```bash
python src/models/spatio-temporal-based/lstm.py
python src/models/graph-based/stgcn.py
python src/models/graph-based/stsgcn.py
python src/models/graph-based/stfgnn.py
python src/models/graph-based/md_stgcn.py
python src/models/attention-based/tft.py
```

Each model script accepts `--seq-dir`, `--epochs`, `--batch-size`, `--lr`, `--patience`, `--device`, `--seed`, and model-specific hyperparameter flags. See the docstring at the top of each file. `--device auto` selects CUDA → MPS → CPU automatically.

## Architecture

### Data Flow

```
data/raw/                   8 raw sources
    └── src/features/       cleaning scripts (one per source)
data/cleaned/               cleaned CSVs, GeoJSONs, .npy adjacency matrices
    └── feature_align.py    merges onto daily index → features_aligned.csv
data/features/              features_aligned.csv + feature_metadata.json
    └── sequence_builder.py sliding windows, MinMaxScaler, chronological split
data/sequences/lstm/        X_train/val/test.npy, y_*.npy, scaler_X/y.pkl, split_dates.json
    └── src/models/**/*.py  model training
src/outputs/{model}/        timestamped run dirs with results.json, plots, model.pt
```

### The Three Model Series

| Series | Location | Models |
|---|---|---|
| Spatio-temporal (LSTM-family) | `src/models/spatio-temporal-based/` | LSTM, BiLSTM, TPA-LSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM |
| Graph-based | `src/models/graph-based/` | STGCN, MTGNN, STSGCN, STFGNN, MD-STGCN |
| Attention-based | `src/models/attention-based/` | TPA-LSTM, ASTGCN, TFT, Autoformer, Informer |

**Canonical reference model:** `src/models/spatio-temporal-based/stlstm.py` — the training loop, output structure, and comparison pattern here should be followed when adding new models.

### Graph Approach (features-as-nodes)

All graph-based and ASTGCN models treat the N input features as graph nodes rather than geographic locations. STGCN, ASTGCN, STSGCN, STFGNN, and MD-STGCN build a fixed spatial adjacency from absolute Pearson correlation of feature columns in `X_train` (threshold=0.1). STFGNN additionally builds a temporal adjacency from the correlation of each node's mean temporal profile. STSGCN constructs a 3N×3N Spatial-Temporal Synchronous Graph (STSG) that captures both spatial and temporal correlations in one synchronous adjacency. MD-STGCN uses bidirectional K-step diffusion convolution (forward D^{-1}A and backward D^{-1}A^T) with multi-scale temporal convolutions. MTGNN learns its adjacency end-to-end from node embeddings.

ASTGCN additionally computes a scaled Chebyshev Laplacian `L_tilde = -A_sym`.

### Comparison Chain

Each model auto-detects and compares against all prior model runs. Results are loaded from `results.json` via `load_model_results()` in `src/utils/comparison_table.py` — if a path is not supplied via CLI, it globs the most recent `results.json` under that model's output directory. The chain grows as each new model is added:

```
LSTM (2-way) → BiLSTM (3-way) → TPA-LSTM (4-way) → CNN-LSTM (5-way)
→ CNN-BiLSTM (6-way) → ST-LSTM (7-way) → STGCN (8-way)
→ MTGNN (9-way) → STSGCN (10-way) → STFGNN (11-way) → MD-STGCN (12-way)
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

- `T_in=14` (look-back window), `T_out=7` (forecast horizon)
- Chronological split: 70% train / 15% val / 15% test
- MCO period (2020-03-18 – 2021-12-31) excluded from sequences by default
- Sequences stored as `float16` by default; load and cast to `float32` before feeding to models
- All scalers fitted on training split only (`scaler_X.pkl`, `scaler_y.pkl`)
