# Malaysian Transit Ridership Forecasting Research

Masters Final Year Project comparing 15 deep-learning models for Malaysian public transit ridership forecasting across three model series using 8 spatio-temporal feature sources.

## Overview

This repository contains the implementation and evaluation of 15 deep learning models for forecasting Malaysian public transit ridership. The models are organized into three series:
- **Spatio-temporal (LSTM-family)**: LSTM, BiLSTM, TPA-LSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM
- **Graph-based**: STGCN, MTGNN, STSGCN, STFGNN, PDR-STGCN
- **Attention-based**: TPA-LSTM, ASTGCN, TFT, Autoformer, Informer

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
│   │   ├── MODEL.md                 # Base model descriptions
│   │   └── TUNED-MODEL.md           # Tuned model descriptions and rationale
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

Tuned outputs are written to `src/outputs/{model_name}_tuned/`. Full architecture change rationale is in `src/models/TUNED-MODEL.md`.

### Common Arguments
All models accept these arguments:
- `--seq-dir`: Override sequence directory (if not set, resolved from `--lookback`)
- `--lookback {14,28,56}`: Look-back window; auto-selects the matching `data/sequences/` directory (default: `14`)
- `--loss {mse,huber,mae}`: Training loss function (default: `huber`)
- `--warmup-epochs N`: Linear LR warm-up epochs before ReduceLROnPlateau (default: `5`)
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--lr`: Learning rate
- `--patience`: Early stopping patience
- `--device`: Computation device (`auto`, `cuda`, `cpu`, or `mps`)
- `--seed`: Random seed for reproducibility
- Model-specific hyperparameters (see each script's docstring)

The `--device auto` option automatically selects CUDA → MPS → CPU.

## Output Structure

Each baseline model run creates a timestamped directory in `src/outputs/{model_name}/`; each tuned run writes to `src/outputs/{model_name}_tuned/`:

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
```

The comparison system automatically:
1. Loads previous model results from their `results.json` files
2. Generates console comparison tables
3. Creates 6-panel visualization plots (overall metrics + per-horizon line charts)

## Technical Details

### Training Optimizations (all 15 models)
- **Optimiser**: AdamW (decoupled weight decay) — same hyperparameter values as before
- **Loss**: HuberLoss (delta=1.0, default) — robust to ridership outliers; selectable via `--loss`
- **LR schedule**: 5-epoch linear warm-up → ReduceLROnPlateau; configurable via `--warmup-epochs`

### Data Configuration
- **Look-back window (T_in)**: 14 / 28 / 56 days (controlled via `--lookback` or `--T-in`)
- **Forecast horizon (T_out)**: 7 days
- **Data split**: 70% train / 15% validation / 15% test (chronological)
- **MCO exclusion**: Movement Control Order period (2020-03-18 to 2021-12-31) excluded by default
- **Precision**: Sequences stored as float16, cast to float32 before model input
- **Scaling**: MinMaxScaler fitted on training data only

### Model-Specific Notes
- **Reference Model**: `src/models/spatio-temporal-based/stlstm.py` serves as the canonical reference for training loop structure, output format, and comparison patterns
- **Graph Construction**: Graph-based models treat features as nodes; adjacency built from Pearson correlation (threshold=0.1) of training features
- **AMP Usage**: Four attention-based models (ASTGCN, TFT, Autoformer, Informer) use mixed precision training to fit within RTX 4050 6GB VRAM
- **TFT/Autoformer Specifics**: 
  - TFT uses standard AMP implementation
  - Autoformer wraps FFT operations in explicit float32 casts within autocast for numerical stability

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

See `journal_articles/` directory for supporting research papers that informed this work.

## License

See `LICENSE` file for licensing information.