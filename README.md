# Malaysian Transit Ridership Forecasting Research

> Last updated: 2026-06-05

Masters Final Year Project comparing 15 deep-learning models for Malaysian public transit ridership forecasting across three model series using 8 spatio-temporal feature sources.

---

## Key Results

Performance targets for this study are **Combined% ≥ 75%** and **R² ≥ 0.7** (both must be met simultaneously). Headline ranking uses the **no-MCO (nomco) · lb14** configuration; MCO-inclusive results are reported as a structural-break robustness check.

### Best Overall (no-MCO · lb14)

| Rank | Model | Combined% | R² | MAE | RMSE | Meets targets? |
|------|-------|-----------|----|-----|------|----------------|
| 1 | **HMT-TSF** (F=79) | **81.82%** | 0.802 | **57,553** | 102,512 | Yes — both |
| 2 | HMT-TSF-FR (F=53) | 81.77% | **0.805** | 59,120 | **101,614** | Yes — both |
| 3 | Informer (tuned) | 79.99% | 0.778 | 65,360 | 108,628 | Yes — both |
| 4 | TPA-LSTM (tuned) | 79.95% | 0.782 | 66,703 | 107,475 | Yes — both |
| 5 | BiLSTM (tuned) | 79.44% | 0.794 | 73,516 | 104,667 | Yes — both |

**HMT-TSF** sets the study-wide headline at nomco·lb14: Combined% 81.82%, R² 0.802, lowest MAE (57,553). HMT-TSF-FR is within 0.05 pp (81.77%, R² 0.805, RMSE 101,614). The next best baseline (Informer tuned) is 1.83 pp below. Full baseline and tuned results are in `src/models/MODEL.md`; HMT-TSF cross-configuration results are in `src/models/hybrid/HMT-TSF.md`.

### Tuned Models vs. HMT-TSF (no-MCO · lb14)

| Rank | Model | Combined% | R² | Δ vs Base |
|------|-------|-----------|-----|----------|
| 1 | **HMT-TSF** | **81.82** | 0.802 | — |
| 2 | HMT-TSF-FR | 81.77 | **0.805** | — |
| 3 | Informer (tuned) | 79.99 | 0.778 | +0.86 pp |
| 4 | TPA-LSTM (tuned) | 79.95 | 0.782 | +1.28 pp |
| 5 | BiLSTM (tuned) | 79.44 | 0.794 | +0.40 pp |
| 6 | ST-LSTM (tuned) | 79.13 | 0.774 | +1.12 pp |
| 7 | ASTGCN (tuned) | 78.82 | 0.754 | +4.48 pp |
| 8 | CNN-LSTM-Aug (tuned) | 78.57 | — | +3.21 pp |
| 9 | PDR-STGCN (tuned) | 77.78 | — | **+10.03 pp** |
| 10 | STSGCN (tuned) | 77.14 | — | +3.81 pp |

### MCO Robustness (Tuned · lb14)

| Rank | Model | MCO Combined% | Δ vs No-MCO |
|------|-------|--------------|------------|
| 1 | **HMT-TSF** | **76.11** | −5.71 pp |
| 2 | Informer (tuned) | 75.79 | −4.64 pp |
| 3 | HMT-TSF-FR | 75.59 | −6.18 pp |
| 4 | ST-LSTM (tuned) | 75.05 | −4.08 pp |
| 5 | TPA-LSTM (tuned) | 74.77 | −5.18 pp |
| … | … | … | … |
| Last | STGCN (tuned) | 53.79 | −22.54 pp |

Graph-based models suffer the most under MCO: their static Pearson-correlation adjacency is fitted on normal-period training data and cannot adapt to the COVID-driven ridership collapse. HMT-TSF is the most MCO-robust model (76.11%, −5.71 pp), with Informer the best-performing baseline (< −5 pp); HMT-TSF's regime-gating component is the primary driver of this resilience.

---

## 1. Data Sources

Eight independent raw sources are integrated through the feature pipeline.

| # | Source | Raw File / Directory | Temporal Coverage | Content | Why Used |
|---|--------|---------------------|-------------------|---------|----------|
| 1 | [**Ridership**](https://data.gov.my) | `data/raw/ridership_headline.csv` | 2019-01-01 – present | Daily passenger counts for 12 transit services + total | Primary forecast target; provides the 13 output variables (12 service lines + total) the models are trained to predict |
| 2 | [**Fuel Price**](https://data.gov.my) | `data/raw/fuelprice.csv` | 2017-03-30 – present | Weekly retail prices for 6 fuel types (level + weekly change) | Demand driver; EDA confirms higher fuel prices correlate with increased transit use, capturing modal substitution behaviour |
| 3 | [**Public & School Holidays**](https://www.timeanddate.com) | `data/raw/school_public_holiday.csv` | 2019–2026 | Date-range calendar of public and academic break periods | Demand modifier; holidays cause systematic ridership drops; lead/lag features capture anticipation dips and post-holiday return surges |
| 4 | [**Rainfall**](https://data.humdata.org) | `data/raw/mys_rainfall_subnat_2019_2026.csv` | 2019-01-01 – 2026 | Daily precipitation at state level; 9 metrics including accumulation and anomaly indices | Weather covariate; state-level daily precipitation adds a spatially-varying external signal that influences travel mode choice |
| 5 | [**Population Density**](https://hub.worldpop.org) | `data/raw/malaysia_population_density_2020.csv` | 2020 (static) | ~1 km² gridded raster of population density across Malaysia | Spatial context; residential density is a static proxy for transit catchment potential and latent demand at each stop |
| 6 | [**GADM Administrative Boundaries**](https://gadm.org) | `data/raw/gadm_mys_l1.json` | 2020 (static) | GeoJSON polygons for 16 Malaysian states (Level-1 admin divisions) | Spatial structure; state boundary polygons are used to spatially aggregate features by administrative region and derive inter-state adjacency weights |
| 7 | [**GTFS Transit Network**](https://data.gov.my) | `data/raw/gtfs_*/` (4 operators) | Snapshot | Standard GTFS files for Rapid Rail KL, RapidBus KL, RapidBus Penang, KTMB | Network topology; stop counts, route density, and travel-time edges quantify transit supply and accessibility per region |
| 8 | [**OSM Points of Interest**](https://overpass-api.de) | `data/raw/osm_pois.json` | Snapshot | OpenStreetMap POI extract; 7 categories (transport, food, retail, education, healthcare, leisure, other) | Land-use context; POI category counts within a 500 m catchment of each stop capture activity generators that drive trip origins and destinations |

---

## 2. Data Preprocessing

### 2.1 Per-Source Cleaning

**Ridership** (`src/features/ridership.py`)
- Parse dd/mm/yyyy dates; flag MCO period (2020-03-18 – 2021-12-31) with `is_mco=1` — rows are retained, not removed, to preserve sequence continuity
- Zero-fill structural nulls for services not yet launched (pre-launch absence ≠ random missingness)
- Compute `total_ridership` as row sum (min_count=1); add cyclical encodings `dow_sin/cos`, `month_sin/cos`
- MinMaxScaler [0, 1] fitted on train split only; serialised to `data/scalers/ridership_scaler.pkl`

**Fuel Price** (`src/features/fuelprice.py`)
- Split `series_type` into level vs. change tables; upsample weekly → daily via forward-fill
- Compute `pct_change()` on RON95, RON97, diesel for stationarity
- MinMaxScaler fitted on data ≤ 2022-01-01 (aligned with ridership train boundary)

**Holidays** (`src/features/holiday.py`)
- Expand date-range rows into one binary flag per calendar day
- Lead/lag features: `days_to_next_public_hol`, `days_since_last_public_hol` (and school equivalents) — captures anticipation behaviour

**Rainfall** (`src/features/rainfall.py`)
- Retain only `version="final"`; aggregate district rows to state level
- Pivot long → wide format; linear interpolation with `limit=7` for short gaps

**GADM** (`src/features/gadm.py`)
- Compute state centroids via Shapely; build 16×16 shared-border adjacency matrix (km-weighted)
- Export `gadm_adj_matrix.npy` (binary) and `gadm_adj_matrix_weighted.npy`

**GTFS** (`src/features/gtfs.py`)
- Per-operator presets handle non-standard columns and frequency-based vs. schedule-based timetables
- Validate stop coordinates within Malaysia bounding box; deduplicate stops and trips
- Export `gtfs_stop_nodes_<operator>.csv` and directed `gtfs_stop_edges_<operator>.csv` with travel_time_s

**Population** (`src/features/population.py`)
- Clip density at 99th percentile; apply `log1p` transform; state-level aggregation via GeoPandas spatial join
- Stop-level nearest-neighbour lookup via BallTree (Haversine)

**OSM** (`src/features/osm.py`)
- Deduplicate on OSM element id; map `amenity/shop/leisure/tourism` tags to 7 POI categories
- BallTree catchment query (500 m radius) per GTFS stop; apply `log1p` to counts

### 2.2 Feature Alignment (`src/features/feature_align.py`)

All 8 cleaned sources are merged and two output files are produced in a single run:

| Output | MCO rows | Date range | Days |
|--------|----------|-----------|------|
| `data/features/features_aligned.csv` | included | 2019-01-01 – 2025-12-31 | 2,557 |
| `data/features/features_aligned_no_mco.csv` | excluded (2020-03-18 – 2021-12-31) | 2022-01-01 – 2025-12-31 | 1,461 |

**79-column feature matrix breakdown:**

| Feature Group | Count | Examples |
|---------------|-------|---------|
| Targets | 13 | `total_ridership`, 12 service lines |
| Temporal / Cyclical | 16 | `is_public_holiday`, `days_to_next_public_hol`, `dow_sin/cos`, `month_sin/cos`, `year`, `day_of_year` |
| External | 30 | 6 fuel price levels + 6 fuel pct-changes + 15 state rainfall series |
| Lag | 3 | `ridership_lag_7`, `ridership_lag_14`, `ridership_lag_28` |
| Static | 17 | `pop_density_median`, 4 GTFS network counts, 8 OSM POI category means, 3 GADM spatial metrics |

Ridership lag features are computed on the full 2019+ history before slicing to ensure valid values at the start of the 2022 window. Static sources (population, GTFS, OSM, GADM) are broadcast to every row.

### 2.3 Sequence Building (`src/features/sequence_builder.py`)

MCO filtering is performed by `feature_align.py` — `sequence_builder.py` simply loads whichever features file it is pointed at. It is called **twice per lookback window** to produce both experimental conditions:

- **X** shape `(N, T_in, 79)`, **y** shape `(N, 7)` for T_in ∈ {14, 28, 56}
- Chronological 70 / 15 / 15 split — no shuffle
- MinMaxScaler fitted on training split only; arrays stored as `float16`
- Graph models build a Pearson correlation adjacency on-the-fly from `X_train` (threshold 0.1, symmetric normalisation)

Six sequence directories are produced (three windows × two MCO conditions):

| Dir | T_in | MCO rows |
|-----|------|----------|
| `data/sequences/lstm/` | 14 | excluded |
| `data/sequences/lstm_mco/` | 14 | included |
| `data/sequences/lookback_28/` | 28 | excluded |
| `data/sequences/lookback_28_mco/` | 28 | included |
| `data/sequences/lookback_56/` | 56 | excluded |
| `data/sequences/lookback_56_mco/` | 56 | included |

---

## 3. Notable EDA Findings

The EDA suite in `src/eda/` contains 40 scripts across univariate, bivariate, multivariate, and thematic categories.

| Finding | Script | Key Detail |
|---------|--------|-----------|
| **MCO collapse and unequal recovery** | `eda/ridership/ridership.py` | Change-point detection (Binseg, 5 breakpoints) identifies the lockdown onset and three recovery phases; per-service collapse % and post-MCO recovery rate differ substantially by mode |
| **Northeast monsoon pattern** | `eda/rainfall/rainfall.py` | Monthly wet-day ratio and extreme-event frequency (> 50 mm) peak in November–January; east-coast states record significantly more extreme days than west-coast counterparts |
| **Fuel price elasticity** | `eda/theme/demand_drivers_series.py` | Binned scatter of RON95 price vs. total ridership shows a moderate positive correlation — higher fuel prices are associated with increased transit use, consistent with modal substitution |
| **Holiday anticipation effect** | `eda/bivariate/holiday_ridership.py` | Lead/lag cross-correlation reveals a ridership dip 1–2 days before public holidays and a return surge on the first post-holiday workday; school holidays show a softer, multi-day ramp |
| **Klang Valley transit concentration** | `eda/gtfs/gtfs.py` | DBSCAN clustering on stop coordinates identifies dense clusters in Greater KL; most peninsular states outside Selangor/KL have fewer than 5% of total stops |
| **Multicollinearity in fuel features** | `eda/multivariate/full_demand_model_feature_correlation_matrix.py` | Condition number and eigenvalue analysis flags high collinearity among RON95 level, RON97 level, and diesel level — informed SHAP-guided feature reduction in HMT-TSF (79 → 53 features) |
| **Underserved population zones** | `eda/multivariate/underserved_population_identification.py` | Joint analysis of population density × transit supply identifies priority zones where high residential density coincides with low stop coverage |
| **Seasonal demand–supply mismatch** | `eda/multivariate/seasonal_demand_supply_mismatch.py` | School holiday periods show systematic over-supply relative to demand; Ramadan shifts demand composition by service type |
| **Network topology by service** | `eda/gtfs/gtfs.py` | Average node degree and edge-to-node density ratios vary substantially across operators; ETS and Intercity are sparsely connected long-haul networks while LRT Kelana Jaya is most densely meshed |
| **Feature correlation matrix** | `eda/multivariate/full_demand_model_feature_correlation_matrix.py` | Full 79×79 correlation heatmap; lag features (lag_7/14/28) are the strongest individual predictors; static features (GTFS, OSM, GADM) cluster at near-zero correlation with ridership |

---

## Overview

### Model Series

This repository implements and compares **14 deep-learning baseline models** plus one **Hybrid SOTA model (HMT-TSF)** organised into four series:

| Series | Location | Models |
|--------|----------|--------|
| Spatio-temporal (LSTM-family) | `src/models/spatio-temporal-based/` | LSTM, BiLSTM, TPA-LSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM |
| Graph-based | `src/models/graph-based/` | STGCN, MTGNN, STSGCN, STFGNN, PDR-STGCN |
| Attention-based | `src/models/attention-based/` | TPA-LSTM, ASTGCN, Autoformer, Informer |
| Hybrid SOTA | `src/models/hybrid/` | HMT-TSF |

Fine-tuned variants of all 14 baseline models are in mirrored `*-tuned/` folders.

---

## Project Structure

```
Ridership/
├── data/                   # All data files
│   ├── raw/                # Raw data sources (8 feature types)
│   ├── cleaned/            # Preprocessed data
│   ├── features/           # Merged feature datasets (MCO included + excluded)
│   ├── sequences/          # Processed sequences (6 dirs: 3 lookbacks × 2 MCO conditions)
│   └── scalers/            # Additional scaler objects
├── src/                    # Source code
│   ├── features/           # Feature engineering pipeline
│   ├── models/             # Deep learning models (organized by series)
│   │   ├── spatio-temporal-based/   # Base ST models
│   │   ├── graph-based/             # Base graph models
│   │   ├── attention-based/         # Base attention models
│   │   ├── spatio-temporal-tuned/   # Tuned ST variants
│   │   ├── graph-tuned/             # Tuned graph variants
│   │   ├── attention-tuned/         # Tuned attention variants
│   │   ├── hybrid/                  # Hybrid SOTA model (HMT-TSF)
│   │   └── MODEL.md                 # All model descriptions + tuning rationale
│   ├── outputs/            # Model outputs (organized by model and timestamp)
│   └── utils/              # Shared utilities (metrics, comparison)
├── docs/                   # Documentation (PDFs, reports, presentations)
├── journal_articles/       # Supporting research papers
├── CLAUDE.md               # Detailed project guidance for Claude Code
├── cuda.py                 # GPU/CUDA verification script
├── LICENSE
├── METRICS.md              # Detailed metric definitions
└── README.md               # This file
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- GPU recommended (CUDA-compatible; tested on NVIDIA A100 32 GB)

### Environment Setup

```bash
# Activate the virtual environment (Windows)
.venv\Scripts\activate

# Verify GPU availability
python cuda.py  # Should print True if CUDA is available
```

---

## Running the Feature Pipeline

All data processing scripts are in `src/features/`. Full step-by-step instructions are in `src/features/PIPELINE.md`. The short version:

### Step 1 — Independent Cleaning (any order)
```bash
python src/features/ridership.py
python src/features/fuelprice.py
python src/features/holiday.py
python src/features/rainfall.py
python src/features/gadm.py
python src/features/gtfs.py --input data/raw/gtfs_rapid_rail_kl  --output data/cleaned/gtfs_rapid_rail_kl  --operator rapid_rail_kl
python src/features/gtfs.py --input data/raw/gtfs_rapidbus_kl    --output data/cleaned/gtfs_rapidbus_kl    --operator rapidbus_kl
python src/features/gtfs.py --input data/raw/gtfs_rapidbus_penang --output data/cleaned/gtfs_rapidbus_penang --operator rapidbus_penang
python src/features/gtfs.py --input data/raw/gtfs_ktmb           --output data/cleaned/gtfs_ktmb           --operator ktmb
```

### Step 1g–1h — Dependent Processing (requires GADM + GTFS outputs)
```bash
python src/features/population.py
python src/features/osm.py
```

### Step 2 — Feature Alignment
```bash
python src/features/feature_align.py
# Produces:
#   data/features/features_aligned.csv            ← MCO included (2,557 days, 2019-01-01 – 2025-12-31)
#   data/features/features_aligned_no_mco.csv     ← MCO excluded (1,461 days, 2022-01-01 – 2025-12-31)
#   data/features/feature_metadata.json
#   data/features/feature_metadata_no_mco.json
```

### Step 3 — Sequence Building (both MCO conditions per lookback)
```bash
# lookback=14
python src/features/sequence_builder.py --features-path data/features/features_aligned_no_mco.csv --T-in 14 --out-dir data/sequences/lstm
python src/features/sequence_builder.py --features-path data/features/features_aligned.csv         --T-in 14 --out-dir data/sequences/lstm_mco
# lookback=28
python src/features/sequence_builder.py --features-path data/features/features_aligned_no_mco.csv --T-in 28 --out-dir data/sequences/lookback_28
python src/features/sequence_builder.py --features-path data/features/features_aligned.csv         --T-in 28 --out-dir data/sequences/lookback_28_mco
# lookback=56
python src/features/sequence_builder.py --features-path data/features/features_aligned_no_mco.csv --T-in 56 --out-dir data/sequences/lookback_56
python src/features/sequence_builder.py --features-path data/features/features_aligned.csv         --T-in 56 --out-dir data/sequences/lookback_56_mco
```

Or run everything in one shot (skipping re-cleaning if data is already clean):
```bash
python src/features/run_pipeline.py                    # full pipeline
python src/features/run_pipeline.py --skip-clean       # skip cleaning, run align + sequences
python src/features/run_pipeline.py --skip-clean --skip-align  # sequences only
```

---

## Running Models

### Baseline Models

All 14 baseline models use a consistent interface from the repository root:

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
python src/models/attention-based/astgcn.py
python src/models/attention-based/autoformer.py
python src/models/attention-based/informer.py
```

### Hybrid SOTA Model

```bash
# Default run (lookback=14, d_model=64, 3 TCN blocks)
python src/models/hybrid/hmttsf.py

# Longer lookback with more TCN depth
python src/models/hybrid/hmttsf.py --lookback 28 --n-tcn-blocks 4 --d-model 192

# With CatBoost residual boosting
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

See `src/models/hybrid/HMT-TSF.md` for the full architecture diagram, component rationale, and scaling recommendations.

### Fine-Tuned Models

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
python src/models/attention-tuned/autoformer.py
python src/models/attention-tuned/informer.py
```

CNN-LSTM tuned — three modes, each with its own best lookback:
```bash
python src/models/spatio-temporal-tuned/cnnlstm.py --mode sequential --lookback 14
python src/models/spatio-temporal-tuned/cnnlstm.py --mode parallel   --lookback 28
python src/models/spatio-temporal-tuned/cnnlstm.py --mode augmented  --lookback 14
```

### Common Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--seq-dir` | auto | Override sequence directory |
| `--lookback {14,28,56}` | `14` | Look-back window; auto-selects matching `data/sequences/` dir |
| `--loss {mse,huber,mae}` | `huber` | Training loss function |
| `--warmup-epochs N` | `5` | Linear LR warm-up epochs (HMT-TSF: `8`) |
| `--epochs N` | `150` | Training epochs |
| `--batch-size N` | `32` | Batch size |
| `--lr F` | `1e-3` | Learning rate |
| `--patience N` | `15` | Early stopping patience (HMT-TSF: `20`) |
| `--device` | `auto` | `auto` → CUDA → MPS → CPU |
| `--seed N` | — | Random seed for reproducibility |

HMT-TSF additional arguments: `--d-model {64,128,192,256}`, `--n-tcn-blocks N`, `--graph-hidden {32,64,128}`, `--n-regimes N`, `--use-catboost`, `--no-boost`, `--no-revin`, `--shap`, `--shap-samples N`, `--smooth-weight λ`, `--loss-decay γ`.

---

## 4. Model Comparison

### 4.1 Spatio-Temporal Series (LSTM-Family)

Six models in `src/models/spatio-temporal-based/`; tuned variants in `src/models/spatio-temporal-tuned/`.

| Model | Key Architecture | Strengths | Weaknesses |
|-------|-----------------|-----------|------------|
| **LSTM** | Unidirectional LSTM [hidden=64, layers=1] → MLP | Simplest baseline; establishes performance floor; low parameter count | Single hidden vector at T is an information bottleneck; no mechanism to identify which timesteps or features drive the forecast |
| **BiLSTM** | Bidirectional LSTM; concatenate forward + backward hidden states → MLP | Mid-window peaks and dips visible from both temporal directions; richer positional context | No explicit attention or pattern extraction; already near capacity ceiling at base hyperparameters |
| **TPA-LSTM** | LSTM → Conv1d on hidden-state matrix → temporal pattern attention → MLP | Discovers which recurring temporal shapes matter most; attention weights are semi-interpretable | CNN filters fixed per training session; still depends on LSTM hidden vector quality |
| **CNN-LSTM** | Conv1d (local extraction) → LSTM (sequence modelling) → MLP; three modes: sequential, parallel, augmented | Hierarchical feature extraction; augmented mode generalises best via raw skip connection | CNN flattens the spatial dimension; mode selection (3 variants) adds experimental complexity |
| **CNN-BiLSTM** | Conv1d → BiLSTM → MLP | Two complementary inductive biases (local translation-invariant features + bidirectional global context) | Higher parameter count; requires careful regularisation |
| **ST-LSTM** | Parallel temporal (LSTM) and spatial (weight-shared MLP across time) streams → concat → MLP | Clean conceptual separation of "when" from "which features co-activate"; spatial encoder is weight-shared | Spatial encoder mean-pools across time, losing temporal order; streams fused late without mid-encoding interaction |

### 4.2 Graph-Based Series

Five models in `src/models/graph-based/`; tuned variants in `src/models/graph-tuned/`. All treat the N input features as graph nodes. Adjacency is built from absolute Pearson correlation of feature columns in X_train (threshold 0.1) unless noted.

| Model | Key Architecture | Strengths | Weaknesses |
|-------|-----------------|-----------|------------|
| **STGCN** | Interleaved temporal gated convolutions (GLU) and Chebyshev graph convolutions; fixed pre-computed Laplacian; 2 ST-Conv blocks | Computationally efficient; K-hop Chebyshev avoids eigendecomposition; interpretable static adjacency | Static graph cannot adapt to changing feature correlations; most severe MCO degradation (−22.54 pp) of all models |
| **MTGNN** | End-to-end learnable asymmetric adjacency from node embeddings; multi-scale inception blocks with dilated kernels (1, 3, 5, 7) | No pre-computed correlation needed; asymmetric graph allows directional feature influence; only model to prefer lb=56 | Graph learning adds parameters; asymmetric adjacency can overfit the training-period topology |
| **STSGCN** | Block-tridiagonal 3N×3N Spatial-Temporal Synchronous Graph; ChebConv + GLU per layer | Unified single-pass encoding of spatial and temporal adjacency | Each layer shrinks the temporal dimension by 2, limiting depth; among the weakest under MCO (−21.96 pp) |
| **STFGNN** | Dual static graphs: spatial (Pearson co-variation) + temporal (mean temporal profiles); per-layer learned gate α | Two complementary graph perspectives; gate α interpretable (which graph dominates per layer) | Over-regularisation at high dropout; performance regressed −1.35 pp from base to tuned |
| **PDR-STGCN** | Two-channel input (original + weekly lag-difference); mixed adjacency: static sym-norm Pearson + dynamic attention; learned scalar λ | Only model with explicit seasonal encoding; dynamic graph adapts per input sample; largest tuning gain (+10.03 pp) | Most complex baseline; worst base performance (67.75%) before tuning; training can be unstable |

### 4.3 Attention-Based Series

Three models in `src/models/attention-based/`; tuned variants in `src/models/attention-tuned/`. All use AMP (fp16 + GradScaler) on the A100.

| Model | Key Architecture | Strengths | Weaknesses |
|-------|-----------------|-----------|------------|
| **ASTGCN** | Multi-head spatial attention (over nodes) → Chebyshev GCN → multi-head temporal attention (over timesteps), repeated across n_blocks | Dual attention re-weights both spatial node relevance and temporal step relevance; strong tuning gain (+4.48 pp) | Graph adjacency is still static; poor MCO robustness (−18.32 pp) |
| **Autoformer** | FFT-based auto-correlation (O(L log L)) discovers periodic lags; explicit trend/seasonal decomposition via moving-average sub-layer | Sub-linear complexity; trend/seasonal decomposition aligns with transit periodicity | Catastrophic performance collapse at lb=56 (58.59%); FFT amplifies noise at longer sequences |
| **Informer** | ProbSparse self-attention: top-u queries selected by KL-divergence sparsity; sequence distilling halves T after each encoder layer; generative decoder | Most robust baseline under MCO (−4.64 pp, best among all baselines); graceful degradation for short sequences | Sequence distilling discards some temporal resolution; limited tuning upside (near-optimal at base) |

### 4.4 Hybrid SOTA

`src/models/hybrid/hmttsf.py`; full architecture in `src/models/hybrid/HMT-TSF.md`.

| Model | Key Architecture | Strengths | Weaknesses |
|-------|-----------------|-----------|------------|
| **HMT-TSF** | RevIN → 5-group semantic feature encoders → Temporal Transformer block → [Multi-Scale TCN ‖ Feature GCN ‖ Regime Gating] → SE-gated fusion → dual forecast heads; optional CatBoost residual correction | Best study-wide: Combined% 81.82%, R² 0.802 (nomco_lb14, full model); HMT-TSF-FR (53 features) is within 0.05 pp; regime gating handles MCO structural break; widest lookback support (7–84 days); SHAP-guided feature reduction (79→53 features) cuts GCN memory ~33% | Most complex architecture; many interacting hyperparameters; SHAP and CatBoost add significant runtime; mco_lb84 collapses when static spatial features are removed (HMT-TSF-FR −4.50 pp) |

---

## 5. Performance Results

### Baseline Models (no-MCO · lb14)

| Rank | Model | Series | Combined% | R² |
|------|-------|--------|-----------|-----|
| 1 | Informer | Attention | 79.13 | 0.772 |
| 2 | BiLSTM | Spatio-temporal | 79.04 | 0.776 |
| 3 | TPA-LSTM | Spatio-temporal | 78.67 | 0.776 |
| 4 | CNN-BiLSTM | Spatio-temporal | 78.18 | 0.765 |
| 5 | ST-LSTM | Spatio-temporal | 78.01 | 0.770 |
| 6 | LSTM | Spatio-temporal | 78.13 | 0.760 |
| 7 | CNN-LSTM | Spatio-temporal | 77.64 | 0.752 |
| 8 | MTGNN | Graph | 77.62 | 0.764 |
| 9 | STGCN | Graph | 77.44 | 0.753 |
| 10 | Autoformer | Attention | 76.71 | 0.741 |
| 11 | STFGNN | Graph | 75.89 | 0.722 |
| 12 | ASTGCN | Attention | 74.34 | 0.694 |
| 13 | STSGCN | Graph | 73.33 | 0.684 |
| 14 | PDR-STGCN | Graph | 67.75 | 0.571 |

10 of 14 models exceed the 75% Combined% target at baseline.

### Cross-Lookback Analysis (Tuned · no-MCO)

| Model | lb14 Combined% | lb28 Combined% | lb56 Combined% | Best |
|-------|----------------|----------------|----------------|------|
| LSTM | 81.04 | 79.79 | 79.24 | **14** |
| Informer | 79.99 | **80.08** | 77.51 | **28** |
| TPA-LSTM | 79.95 | 79.41 | 79.33 | **14** |
| BiLSTM | 79.44 | 78.33 | 75.90 | **14** |
| Autoformer | 79.27 | 74.65 | 58.59 | **14** |
| ST-LSTM | 79.13 | 77.85 | 78.51 | **14** |
| ASTGCN | 78.82 | 68.22 | 73.60 | **14** |
| CNN-BiLSTM | 78.53 | 75.16 | 55.14 | **14** |
| CNN-LSTM | 78.52 | 76.60 | 77.83 | **14** |
| PDR-STGCN | 77.78 | 76.10 | 52.13 | **14** |
| MTGNN | 77.44 | 77.35 | **78.13** | **56** |
| STSGCN | 77.14 | 73.53 | 50.13 | **14** |
| STGCN | 76.33 | **77.17** | 74.52 | **28** |
| STFGNN | 74.55 | 72.59 | 42.19 | **14** |

lb14 is optimal for 11 of 14 models. MTGNN uniquely prefers lb56 (its learned asymmetric adjacency benefits from quarterly context). Autoformer, STFGNN, and STSGCN collapse at lb56 (FFT/synchronous graph operations amplify noise over long windows).

Full detailed results, including the MCO-inclusive table and tuning effectiveness summary, are in `src/models/MODEL.md` → Experimental Results section.

---

## Output Structure

Each model run creates a timestamped directory:

```
src/outputs/{model_name}/{YYYYMMDD_HHMMSS}/
├── model.pt               # Saved PyTorch model weights
├── results.json           # Evaluation metrics (used for comparisons)
├── predictions.npy        # Raw test predictions
└── comparison_{n}way.png  # Multi-panel comparison plot (vs previous models)
```

Tuned runs write to `src/outputs/{model_name}_tuned/`; HMT-TSF writes to `src/outputs/hmttsf/`.

---

## Model Comparison System

Each model auto-detects all prior runs from `src/outputs/` and adds itself to the comparison table. Results are loaded from `results.json` via `load_model_results()` in `src/utils/comparison_table.py`.

```
LSTM (2-way) → BiLSTM (3-way) → TPA-LSTM (4-way) → CNN-LSTM (5-way)
→ CNN-BiLSTM (6-way) → ST-LSTM (7-way) → STGCN (8-way)
→ ASTGCN (9-way) → STSGCN (10-way) → PDR-STGCN (11-way) → MTGNN (12-way)
→ STFGNN (13-way) → Autoformer (14-way) → Informer (15-way)
→ HMT-TSF (16-way)
```

The comparison system generates console tables and 6-panel PNG visualisations (overall metrics + per-horizon line charts).

---

## Technical Details

### Training Optimizations (all models)
- **Optimiser**: AdamW (decoupled weight decay)
- **Loss**: HuberLoss (delta=1.0, default) — robust to ridership outliers; selectable via `--loss`
- **LR schedule**: Linear warm-up → ReduceLROnPlateau; configurable via `--warmup-epochs`

### HMT-TSF Specific
- **Loss**: WeightedHuber (step-decayed, γ=0.9) + TemporalSmoothness regularisation (λ=0.01)
- **AMP**: GradScaler + autocast fp16 on CUDA
- **Gradient clipping**: max_norm=1.0
- **Optional residual boosting**: CatBoost or sklearn MLP on train-set residuals; applied at 0.5× weight only if it improves Combined%
- **Walk-forward evaluation**: test set split into 3 equal blocks; per-block Combined% and R² reported

### Data Configuration
- **Look-back window (T_in)**: 14 / 28 / 56 days for the 14 base models; 7 / 14 / 28 / 56 / 84 days for HMT-TSF
- **Forecast horizon (T_out)**: 7 days
- **Data split**: 70% train / 15% validation / 15% test (chronological)
- **MCO split**: performed in `feature_align.py`; sequences are always produced for both conditions (`data/sequences/lstm/` = MCO excluded, `data/sequences/lstm_mco/` = MCO included)
- **Precision**: Sequences stored as float16, cast to float32 before model input
- **Scaling**: MinMaxScaler fitted on training data only

### Model-Specific Notes
- **Reference model**: `src/models/spatio-temporal-based/stlstm.py` — canonical reference for training loop, output format, and comparison patterns
- **Graph construction**: Graph-based models and HMT-TSF treat features as nodes; adjacency from Pearson correlation (threshold=0.1) of training features
- **AMP usage**: ASTGCN, Autoformer, Informer, and HMT-TSF use mixed precision on the A100; Autoformer wraps FFT ops in explicit float32 casts for numerical stability
- **HMT-TSF architecture**: Five feature-group encoders → Temporal Transformer → parallel Multi-Scale TCN + Feature GCN + Regime Gating → SE-Gated Fusion → dual forecast heads → optional CatBoost residual correction

---

## Evaluation Metrics

See `src/METRICS.md` for detailed definitions.

| Metric | Formula | Better direction |
|--------|---------|-----------------|
| **Combined%** | `max(0, 100 − MAPE − MAE% − RMSE%)` | Higher |
| **MAPE%** | Mean Absolute Percentage Error | Lower |
| **MAE%** | Mean Absolute Error (normalised by mean demand) | Lower |
| **RMSE%** | Root Mean Square Error (normalised) | Lower |
| **R²** | Coefficient of Determination | Higher |
| **MAE** | Mean Absolute Error (original ridership units) | Lower |
| **RMSE** | Root Mean Square Error (original units) | Lower |

All percentage metrics use mean-demand normalisation so they are directly addable in the Combined% formula.

---

## Project Guidelines

As outlined in `CLAUDE.md`:
- All scripts are executed from the **repository root**
- Each model script manually adds the project root to `sys.path` for imports
- The feature pipeline must be run before training any models
- Model outputs are automatically versioned by timestamp to prevent overwrites
- `--device auto` selects CUDA → MPS → CPU automatically

---

## References

Each model script docstring cites one Scopus-indexed journal article (2022–2027) for its architecture. The full citation list is in `src/models/MODEL.md` → Journal References, grouped by model series.

HMT-TSF has per-component Scopus-indexed citations (one per architectural block). These are in `src/models/hybrid/HMT-TSF.md` → Journal References.

Additional supporting research papers are in `journal_articles/`.
