# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

Masters FYP comparing spatio-temporal models (BiLSTM, TPA-LSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM, GRU) against a baseline LSTM for next-day ridership forecasting across 12 Malaysian public transit networks. The 8 spatio-temporal features are: ridership service levels, fuel price, holiday indicators, rainfall, population density, POI counts, GTFS-derived accessibility, and walking distance. Hardware: 13th Gen i5 + RTX 4050 (6 GB vRAM).

## Environment

All Python runs inside `.venv`. Activate first:
```bash
source .venv/Scripts/activate   # Git Bash / bash on Windows
```

Verify GPU is available:
```bash
python cuda.py
```

## Pipeline — Run Order

Each step depends on the previous output. Run from the repo root:

```bash
# 1. Clean individual data sources (run each as needed)
python src/features/ridership.py
python src/features/fuelprice.py
python src/features/holiday.py
python src/features/rainfall.py
python src/features/population.py
python src/features/osm.py
python src/features/gtfs.py
python src/features/gadm.py

# 2. Build graph adjacency matrices
python src/features/graph_builder.py

# 3. Align all cleaned sources into a single daily feature matrix
python src/features/feature_align.py
# Output: data/features/features_aligned.csv, data/features/feature_metadata.json

# 4. Build sliding-window sequences (shared by LSTM / BiLSTM / TPA-LSTM)
python src/features/sequence_builder.py
# Output: data/sequences/lstm/ and data/sequences/gcn/

# 5. Train models
python src/models/lstm.py                          # baseline
python src/models/bilstm.py
python src/models/tpalstm.py                       # auto-detects prior LSTM/BiLSTM runs for comparison
python src/models/cnnlstm.py                       # auto-detects all three prior runs for comparison
python src/models/cnnbilstm.py                     # auto-detects all four prior runs for comparison
python src/models/stlstm.py                        # auto-detects all five prior runs for comparison
python src/models/gru.py                           # auto-detects all six prior runs for comparison
```

## Model CLI Flags (common across all five models)

| Flag | Default | Notes |
|---|---|---|
| `--seq-dir` | `data/sequences/lstm` | Path to sequence .npy files |
| `--hidden` | 64 | LSTM hidden size |
| `--layers` | 1 | Stacked LSTM layers |
| `--dropout` | 0.2 | Active only when `--layers > 1` |
| `--batch-size` | 64 | |
| `--epochs` | 50 | |
| `--lr` | 1e-3 | |
| `--patience` | 10 | Early stopping patience |
| `--device` | auto | `cpu \| cuda \| mps \| auto` |
| `--seed` | 42 | |

TPA-LSTM additional flags: `--filters 32`, `--kernel-size 3`, `--lstm-results <path>`, `--bilstm-results <path>`.

CNN-LSTM additional flags: `--cnn-filters 64`, `--cnn-layers 2`, `--cnn-kernel-size 3`, `--mode sequential|parallel` (default `sequential`), `--lstm-results <path>`, `--bilstm-results <path>`, `--tpalstm-results <path>`. Sequential: `X→CNN→LSTM→head`; parallel: CNN and LSTM each read raw X independently, outputs concatenated before head.

CNN-BiLSTM additional flags: `--cnn-filters 64`, `--cnn-layers 2`, `--cnn-kernel-size 3`, `--lstm-results <path>`, `--bilstm-results <path>`, `--tpalstm-results <path>`, `--cnnlstm-results <path>`.

GRU shares the same common flags as LSTM with no additional flags. Auto-detects all six prior runs. Output: `src/outputs/gru/`.

ST-LSTM additional flags: `--spatial-hidden 32` (spatial encoder output dim), `--lstm-results <path>`, `--bilstm-results <path>`, `--tpalstm-results <path>`, `--cnnlstm-results <path>`, `--cnnbilstm-results <path>`. Two parallel streams: temporal LSTM on raw X → h_T; spatial MLP (shared per timestep, mean-pooled over T) → sp; cat([h_T, sp]) → MLP head.

## Outputs

Each training run writes a timestamped folder under `src/outputs/<model>/YYYYMMDD_HHMMSS/` containing:
- `results.json` — all metrics
- `model.pt` — saved weights (gitignored)
- `loss_curves.png`, `test_predictions.png`, `per_step_metrics.png`

TPA-LSTM auto-detects the most recent LSTM and BiLSTM `results.json` for comparison plots.

CNN-LSTM additionally produces `comparison_four_way.png` (vs all three prior models). Its `results.json` includes delta metrics against each baseline.

CNN-BiLSTM additionally produces `comparison_five_way.png` (vs all four prior models). Its `results.json` includes delta metrics against each baseline.

ST-LSTM additionally produces `comparison_N_way.png` (vs all available prior models, up to 6-way). Its `results.json` includes delta metrics against each baseline.

## Evaluation Metrics

Defined in `src/METRICS.md`. Primary ranking metric is **Combined%** (higher = better):

```
Combined% = max(0, 100 − MAPE − MAE% − RMSE%)
```

Where `MAE% = MAE / ȳ × 100` and `RMSE% = RMSE / ȳ × 100`. R² is reported as a secondary diagnostic.

## Architecture Overview

### Data Flow
```
Raw sources (GTFS, OSM, GADM, rainfall, fuel, holiday, population, ridership)
    → src/features/*.py  (per-source cleaning → data/cleaned/)
    → src/features/feature_align.py  (→ data/features/features_aligned.csv)
    → src/features/sequence_builder.py  (→ data/sequences/lstm/ and gcn/)
    → src/models/*.py  (train + evaluate → src/outputs/<model>/)
```

### Sequence Builder Design Decisions
- Scaler is fit on **train split only** to prevent leakage.
- Temporal split is chronological (70/15/15 train/val/test). No shuffle.
- Sequences straddling the MCO (Movement Control Order) gap are excluded by default (`--include-mco` to override).
- Two granularities: `lstm/` (N_samples, T_in, F) for flat models; `gcn/` (N_samples, N_nodes, T_in, F_node) for graph models.
- Default look-back `T_in=14` days, forecast horizon `T_out=7` days.

### TPA-LSTM Attention
The TPA module applies a 1-D CNN over the LSTM hidden-state matrix H (treating hidden dims as channels over time) to extract temporal patterns, then scores them against the final hidden state h_T via sigmoid + softmax. This allows the model to identify which recurring periodic patterns (daily, weekly, holiday) in the look-back window are predictive of the next step — superior to standard Bahdanau attention for transit ridership's multi-scale periodicity.

## EDA Scripts

`src/eda/` contains standalone analysis scripts organised by data source (univariate), variable pair (bivariate), and theme (multivariate/theme). These produce plots and CSVs under their local `results/` directories and are not part of the training pipeline.
