# Malaysian Transit Ridership Forecasting Research

> Last updated: 2026-05-27

Masters Final Year Project comparing 15 deep-learning models for Malaysian public transit ridership forecasting across three model series using 8 spatio-temporal feature sources.

## Key Results

Performance targets for this study are **Combined% ≥ 75%** and **R² ≥ 0.7** (both must be met simultaneously).

**Peak result — highest Combined% and R² achieved together:**

| Model | Config | Combined% | R² | Exceeds targets? |
|-------|--------|-----------|----|-----------------|
| **LSTM (tuned)** | tuned · nomco · lb14 | **81.04%** | **0.798** | Yes — both |
| Informer (tuned) | tuned · nomco · lb28 | 80.08% | 0.785 | Yes — both |
| **HMT-TSF** | nomco · lb14 | **80.04%** | **0.783** | Yes — both |
| TPA-LSTM (tuned) | tuned · nomco · lb14 | 79.95% | 0.782 | Yes — both |

**LSTM (tuned · nomco · lb14) is the single best result on both metrics simultaneously** — Combined% = 81.04% and R² = 0.798. Among the 17 baseline models, **Informer** is the most consistent (mean Combined% = 76.93%, std dev = 2.45 pp), winning 8 of 12 configuration matchups and maintaining a MCO-regime floor of 72.62%. Full analysis is in [`src/outputs/RESULTS.md`](src/outputs/RESULTS.md).

**HMT-TSF (Hybrid SOTA)** achieves the best overall consistency across both nomco and MCO conditions: 10-config mean **77.30%**, std dev **1.55 pp**, and 9/10 configurations clearing both study targets. It is the top model for MCO-inclusive deployment (best at mco·lb14 and mco·lb28, surpassing Informer tuned) and the only model with a nomco→mco degradation below 2 pp. Full analysis is in [`src/outputs/HMT-TSF-RESULTS.md`](src/outputs/HMT-TSF-RESULTS.md).

---

## Overview

This repository contains the implementation and evaluation of 15 deep learning models plus one hybrid SOTA model for forecasting Malaysian public transit ridership. The models are organized into three baseline series and one hybrid series:
- **Spatio-temporal (LSTM-family)**: LSTM, BiLSTM, TPA-LSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM
- **Graph-based**: STGCN, MTGNN, STSGCN, STFGNN, PDR-STGCN
- **Attention-based**: TPA-LSTM, ASTGCN, TFT, Autoformer, Informer
- **Hybrid (SOTA)**: HMT-TSF (Hybrid Multi-scale Temporal Spatio-Feature Forecaster)

All models are trained and evaluated on the same dataset comprising 8 spatio-temporal feature sources:
- Ridership data
- Fuel prices
- Holiday calendars
- Rainfall measurements
- Administrative boundaries (GADM)
- Public transit schedules (GTFS)
- Population statistics
- Points of interest (OpenStreetMap)

## Project Structure

```
Ridership/
├── data/                   # All data files
│   ├── raw/                # Raw data sources (8 feature types)
│   ├── cleaned/            # Preprocessed data
│   ├── features/           # Merged feature dataset
│   ├── sequences/          # Processed sequences for training
│   └── scalers/            # Additional scaler objects
├── src/                    # Source code
│   ├── features/           # Feature engineering pipeline
│   ├── models/             # Deep learning models (organized by series)
│   │   ├── spatio-temporal-based/   # Base ST models (LSTM, BiLSTM, TPA-LSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM)
│   │   ├── graph-based/             # Base graph models (STGCN, MTGNN, STSGCN, STFGNN, PDR-STGCN)
│   │   ├── attention-based/         # Base attention models (TPA-LSTM, ASTGCN, TFT, Autoformer, Informer)
│   │   ├── spatio-temporal-tuned/   # Tuned ST variants
│   │   ├── graph-tuned/             # Tuned graph variants
│   │   ├── attention-tuned/         # Tuned attention variants
│   │   ├── hybrid/                  # Hybrid SOTA model (HMT-TSF)
│   │   └── MODEL.md                 # All model descriptions + tuning rationale
│   ├── outputs/            # Model outputs (organized by model and timestamp)
│   └── utils/              # Shared utilities (metrics, comparison)
├── docs/                   # Documentation (PDFs, reports, presentations)
├── journal_articles/       # Supporting research papers
├── CLAUDE.md               # Detailed project guidance
├── cuda.py                 # GPU/CUDA verification script
├── LICENSE                 # License file
├── METRICS.md              # Detailed metric definitions
├── README.md               # This file
└── timeline_submission_research_project.pdf # Project timeline
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (check for requirements.txt or install commonly used ML packages)
- GPU recommended for faster training (CUDA-compatible)

### Environment Setup

```bash
# Activate the virtual environment (Windows)
.venv\Scripts\activate

# Verify GPU availability
python cuda.py  # Should print True if CUDA is available
```

## Running the Feature Pipeline

All data processing scripts are located in `src/features/`. The pipeline consists of four stages:

### 1. Independent Cleaning (any order)
```bash
python src/features/ridership.py
python src/features/fuelprice.py
python src/features/holiday.py
python src/features/rainfall.py
python src/features/gadm.py
python src/features/gtfs.py --input data/raw/gtfs_rapid_rail_kl --output data/cleaned/gtfs_rapid_rail_kl --operator rapid_rail_kl
# Repeat for rapidbus_kl, rapidbus_penang, ktmb
```

### 2. Dependent Processing (requires GADM + GTFS outputs)
```bash
python src/features/population.py
python src/features/osm.py
```

### 3. Feature Alignment
```bash
python src/features/feature_align.py
# Produces: data/features/features_aligned.csv and data/features/feature_metadata.json
```

### 4. Sequence Building
```bash
python src/features/sequence_builder.py               # lookback=14 → data/sequences/lstm/
python src/features/sequence_builder.py --T-in 28     # lookback=28 → data/sequences/lookback_28/
python src/features/sequence_builder.py --T-in 56     # lookback=56 → data/sequences/lookback_56/
```

Full step-by-step instructions are available in `src/features/PIPELINE.md`.

## Running Models

### Baseline Models

All 15 baseline models are executed from the repository root with a consistent interface:

```bash
# Spatio-temporal models
python src/models/spatio-temporal-based/lstm.py
python src/models/spatio-temporal-based/bilstm.py
python src/models/spatio-temporal-based/tpalstm.py
python src/models/spatio-temporal-based/cnnlstm.py
python src/models/spatio-temporal-based/cnnbilstm.py
python src/models/spatio-temporal-based/stlstm.py

# Graph-based models
python src/models/graph-based/stgcn.py
python src/models/graph-based/mtgnn.py
python src/models/graph-based/stsgcn.py
python src/models/graph-based/stfgnn.py
python src/models/graph-based/pdr_stgcn.py

# Attention-based models
python src/models/attention-based/tft.py
python src/models/attention-based/astgcn.py
python src/models/attention-based/autoformer.py
python src/models/attention-based/informer.py
```

### Hybrid SOTA Model

HMT-TSF is a purpose-built hybrid that fuses three parallel encoders (Multi-Scale TCN, Feature GCN, Regime Gating) with a gated combiner and an optional post-hoc residual boosting stage:

```bash
# Default run (lookback=14, d_model=128, 3 TCN blocks)
python src/models/hybrid/hmttsf.py

# Longer lookback with more TCN depth
python src/models/hybrid/hmttsf.py --lookback 28 --n-tcn-blocks 4 --d-model 192

# With Optuna HPO (50 trials) then full training
python src/models/hybrid/hmttsf.py --tune-trials 50 --lookback 56

# With CatBoost residual boosting
python src/models/hybrid/hmttsf.py --use-catboost

# With SHAP feature importance
python src/models/hybrid/hmttsf.py --shap --shap-samples 150

# Maximum configuration (84-day lookback, HPO, SHAP, CatBoost)
python src/models/hybrid/hmttsf.py \
  --lookback 84 --d-model 256 --n-tcn-blocks 5 \
  --graph-hidden 128 --dropout 0.15 \
  --tune-trials 50 --use-catboost --shap \
  --epochs 150
```

See `src/models/hybrid/HMT-TSF.md` for the full architecture diagram, component rationale, hyperparameter search space, and scaling recommendations.

### Fine-Tuned Models

Fifteen fine-tuned variants with revised architecture hyperparameters (training params unchanged):

```bash
# Spatio-temporal tuned
python src/models/spatio-temporal-tuned/lstm.py
python src/models/spatio-temporal-tuned/bilstm.py
python src/models/spatio-temporal-tuned/cnnlstm.py    # see CNN-LSTM modes below
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

**CNN-LSTM tuned — three modes, each with its own best lookback:**
```bash
python src/models/spatio-temporal-tuned/cnnlstm.py --mode sequential --lookback 14
python src/models/spatio-temporal-tuned/cnnlstm.py --mode parallel   --lookback 28
python src/models/spatio-temporal-tuned/cnnlstm.py --mode augmented  --lookback 14
```

Tuned outputs are written to `src/outputs/{model_name}_tuned/`. Full architecture change rationale is in `src/models/MODEL.md` (Fine-Tuned Model Series section).

### Common Arguments
All models accept these arguments:
- `--seq-dir`: Override sequence directory (if not set, resolved from `--lookback`)
- `--lookback {14,28,56}`: Look-back window for the 15 base models; auto-selects the matching `data/sequences/` directory (default: `14`). HMT-TSF additionally supports `7` and `84`.
- `--loss {mse,huber,mae}`: Training loss function (default: `huber`)
- `--warmup-epochs N`: Linear LR warm-up epochs before ReduceLROnPlateau (default: `5` for the 15 base models; `8` for HMT-TSF)
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--lr`: Learning rate
- `--patience`: Early stopping patience (default: `15` for the 15 base models; `20` for HMT-TSF)
- `--device`: Computation device (`auto`, `cuda`, `cpu`, or `mps`)
- `--seed`: Random seed for reproducibility
- Model-specific hyperparameters (see each script's docstring)

**HMT-TSF additional arguments:**
- `--d-model {64,128,192,256}`: Model dimension (default: `128`)
- `--n-tcn-blocks N`: Number of TCN blocks per scale (default: `3`)
- `--graph-hidden {32,64,128}`: GCN hidden dimension (default: `64`)
- `--n-regimes N`: Number of regime embeddings (default: `3`)
- `--tune-trials N`: Optuna HPO trials before full training (default: `0`, disabled)
- `--use-catboost`: Enable CatBoost post-hoc residual boosting stage
- `--no-boost`: Disable the neural boost head
- `--no-revin`: Disable Reversible Instance Normalisation
- `--shap`: Run SHAP feature importance after evaluation
- `--shap-samples N`: Background samples for SHAP KernelExplainer (default: `100`)
- `--smooth-weight λ`: Temporal smoothness regularisation weight (default: `0.01`)
- `--loss-decay γ`: Geometric decay factor for step-weighted Huber loss (default: `0.9`)

The `--device auto` option automatically selects CUDA → MPS → CPU.

## Output Structure

Each baseline model run creates a timestamped directory in `src/outputs/{model_name}/`; each tuned run writes to `src/outputs/{model_name}_tuned/`; the hybrid model writes to `src/outputs/hmttsf/`:

```
src/outputs/{model_name}/{YYYYMMDD_HHMMSS}/
├── model.pt          # Saved PyTorch model weights
├── results.json      # Evaluation metrics (used for comparisons)
├── predictions.npy   # Raw test predictions
└── comparison_{n}way.png  # Multi-panel comparison plot (vs previous models)
```

The `results.json` file contains:
- Overall metrics: Combined%, MAPE%, MAE%, RMSE%, R², MAE, RMSE
- Per-forecast-horizon metrics for steps 1-7
- These metrics are used by the automatic comparison system

## Model Comparison System

Models are compared in a historical chain where each new model evaluates against all previous models:
```
LSTM (2-way) → BiLSTM (3-way) → TPA-LSTM (4-way) → CNN-LSTM (5-way)
→ CNN-BiLSTM (6-way) → ST-LSTM (7-way) → STGCN (8-way)
→ MTGNN (9-way) → STSGCN (10-way) → STFGNN (11-way) → PDR-STGCN (12-way)
→ ASTGCN (13-way) → TFT (14-way) → Autoformer (15-way) → Informer (16-way)
→ HMT-TSF (17-way)
```

The comparison system automatically:
1. Loads previous model results from their `results.json` files
2. Generates console comparison tables
3. Creates 6-panel visualization plots (overall metrics + per-horizon line charts)

## Technical Details

### Training Optimizations (all models)
- **Optimiser**: AdamW (decoupled weight decay)
- **Loss**: HuberLoss (delta=1.0, default) — robust to ridership outliers; selectable via `--loss`
- **LR schedule**: Linear warm-up → ReduceLROnPlateau; configurable via `--warmup-epochs` (default 5 epochs for the 15 base models, 8 for HMT-TSF)

### HMT-TSF Specific
- **Loss**: WeightedHuber (step-decayed, γ=0.9) + TemporalSmoothness regularisation (λ=0.01)
- **AMP**: GradScaler + autocast fp16 on CUDA (same as attention-based models)
- **Gradient clipping**: max_norm=1.0
- **Optional HPO**: Optuna with MedianPruner (via `--tune-trials N`)
- **Optional residual boosting**: CatBoost or sklearn MLP trained on train-set residuals; correction applied at 0.5× weight only if it improves Combined% (via `--use-catboost`)
- **Walk-forward evaluation**: test set split into 3 equal blocks; per-block Combined% and R² reported
- **Lookback support**: 7 / 14 / 28 / 56 / 84 days (`lookback_7` and `lookback_84` sequence dirs must be built separately if needed)

### Data Configuration
- **Look-back window (T_in)**: 14 / 28 / 56 days for the 15 base models; 7 / 14 / 28 / 56 / 84 days for HMT-TSF (controlled via `--lookback` or `--T-in`)
- **Forecast horizon (T_out)**: 7 days
- **Data split**: 70% train / 15% validation / 15% test (chronological)
- **MCO exclusion**: Movement Control Order period (2020-03-18 to 2021-12-31) excluded by default
- **Precision**: Sequences stored as float16, cast to float32 before model input
- **Scaling**: MinMaxScaler fitted on training data only

### Model-Specific Notes
- **Reference Model**: `src/models/spatio-temporal-based/stlstm.py` serves as the canonical reference for training loop structure, output format, and comparison patterns
- **Graph Construction**: Graph-based models and HMT-TSF treat features as nodes; adjacency built from Pearson correlation (threshold=0.1) of training features
- **AMP Usage**: Attention-based models (ASTGCN, TFT, Autoformer, Informer) and HMT-TSF use mixed precision training on the A100
- **TFT/Autoformer Specifics**: 
  - TFT uses standard AMP implementation
  - Autoformer wraps FFT operations in explicit float32 casts within autocast for numerical stability
- **HMT-TSF Architecture**: Five feature-group encoders → parallel Multi-Scale TCN + Feature GCN + Regime Gating → Gated Fusion → dual forecast heads (primary + boost) → optional CatBoost residual correction

## Evaluation Metrics

See `src/METRICS.md` for detailed definitions. Key metrics include:
- **Combined%**: `max(0, 100 − MAPE − MAE% − RMSE%)` (higher is better)
- **MAPE%**: Mean Absolute Percentage Error
- **MAE%**: Mean Absolute Error (normalized)
- **RMSE%**: Root Mean Square Error (normalized)
- **R²**: Coefficient of Determination
- **MAE**: Mean Absolute Error (original units)
- **RMSE**: Root Mean Square Error (original units)

All percentage metrics use mean-demand normalization for direct comparability and additivity.

## Project Guidelines

As outlined in `CLAUDE.md`:
- All scripts are executed from the repository root
- Each model script manually adds the project root to `sys.path` for imports
- The feature pipeline must be run before training any models
- Model outputs are automatically versioned by timestamp to prevent overwrites

## References

Each model script docstring includes a `References` section citing one Scopus-indexed journal article (2022–2027) for its architecture. The full citation list is consolidated in `src/models/MODEL.md` under the **Journal References** section, grouped by model series.

HMT-TSF has per-component Scopus-indexed citations (one per architectural block: RevIN, Temporal Transformer, Multi-Scale TCN, DropPath, Feature GCN, Regime Gating, SE-Net Gated Fusion, Weighted Huber Loss, Temporal Smoothness Regularisation, Optuna HPO, Post-hoc Residual Boosting, Walk-Forward Evaluation). These are listed in `src/models/hybrid/HMT-TSF.md` under the **Journal References** section.

See `journal_articles/` directory for additional supporting research papers that informed this work.

## License

See `LICENSE` file for licensing information.