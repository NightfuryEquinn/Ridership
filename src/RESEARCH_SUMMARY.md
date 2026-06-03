# Research Summary

> Malaysian Transit Ridership Forecasting — Masters FYP
> Last updated: 2026-06-03

---

## 1. Data Sources

Eight independent raw sources are integrated through the feature pipeline.

| # | Source | Raw File / Directory | Temporal Coverage | Content |
|---|--------|---------------------|-------------------|---------|
| 1 | **Ridership** | `data/raw/ridership_headline.csv` | 2019-01-01 – present | Daily passenger counts for 12 transit services + total |
| 2 | **Fuel Price** | `data/raw/fuelprice.csv` | 2017-03-30 – present | Weekly retail prices for 6 fuel types (level + weekly change) |
| 3 | **Public & School Holidays** | `data/raw/school_public_holiday.csv` | 2019–2026 | Date-range calendar of public and academic break periods; state-specific coverage |
| 4 | **Rainfall** | `data/raw/mys_rainfall_subnat_2019_2026.csv` | 2019-01-01 – 2026 | Daily precipitation at state and district level; 9 metrics including accumulation and anomaly indices |
| 5 | **Population Density** | `data/raw/malaysia_population_density_2020.csv` | 2020 (static) | ~1 km² gridded raster of population density (persons/km²) across Malaysia |
| 6 | **GADM Administrative Boundaries** | `data/raw/gadm_mys_l1.json` | 2020 (static) | GeoJSON polygons for 16 Malaysian states (Level-1 admin divisions) |
| 7 | **GTFS Transit Network** | `data/raw/gtfs_*/` (4 operators) | Snapshot | Standard GTFS files (stops, routes, trips, timetables) for Rapid Rail KL, RapidBus KL, RapidBus Penang, KTMB |
| 8 | **OSM Points of Interest** | `data/raw/osm_pois.json` | Snapshot | OpenStreetMap POI extract; 7 categories (transport, food, retail, education, healthcare, leisure, other) |

---

## 2. Data Preprocessing

### 2.1 Per-Source Cleaning

**Ridership** (`src/features/ridership.py`)
- Parse dd/mm/yyyy dates; flag MCO period (2020-03-18 – 2021-12-31) with `is_mco=1` — rows are retained, not removed, to preserve sequence continuity
- Zero-fill structural nulls for services not yet launched at a given date (pre-launch absence ≠ random missingness)
- Compute `total_ridership` as row sum (min_count=1)
- Add cyclical temporal encodings: `dow_sin/cos`, `month_sin/cos`
- MinMaxScaler [0, 1] fitted on train split only; serialised to `data/scalers/ridership_scaler.pkl`

**Fuel Price** (`src/features/fuelprice.py`)
- Filter records from 2019-01-01; split `series_type` into level vs. change tables
- Upsample weekly → daily via forward-fill (semantically correct: announced price holds until the next weekly change)
- Compute `pct_change()` on RON95, RON97, diesel for stationarity
- MinMaxScaler fitted on data ≤ 2022-01-01 (aligned with ridership train boundary)

**Holidays** (`src/features/holiday.py`)
- Expand date-range rows into one binary flag per calendar day
- Compute lead/lag features: `days_to_next_public_hol`, `days_since_last_public_hol` (and school equivalents) — captures anticipation behaviour
- Handle state-specific coverage (e.g., Thaipusam: SGR, PNG, PRK, JHR, KUL only)

**Rainfall** (`src/features/rainfall.py`)
- Retain only `version="final"` records; drop `adm_level=2` (sub-state granularity)
- Aggregate district rows to state level: sum `rainfall_mm`, mean anomaly indices
- Pivot long → wide format (one column per state × metric): required for GCN tensor construction
- Linear interpolation with `limit=7` for short gaps; bfill/ffill at edges

**GADM** (`src/features/gadm.py`)
- Compute state centroids via Shapely
- Build 16 × 16 shared-border adjacency matrix: boundary intersection length in km; discard touches < 1e-6 degrees
- Export binary adjacency (`gadm_adj_matrix.npy`) and km-weighted adjacency (`gadm_adj_matrix_weighted.npy`)

**GTFS** (`src/features/gtfs.py`)
- Per-operator presets handle non-standard columns and frequency-based vs. schedule-based timetables
- Validate stop coordinates within Malaysia bounding box; deduplicate stops and trips; fix route_id ambiguities
- Export `gtfs_stop_nodes_<operator>.csv` and directed `gtfs_stop_edges_<operator>.csv` with travel_time_s

**Population** (`src/features/population.py`)
- Clip density values at 99th percentile to reduce raster outliers
- Apply `log1p` transform to correct right-skewed distribution
- State-level aggregation via GeoPandas spatial join; stop-level nearest-neighbour lookup via BallTree (Haversine)

**OSM** (`src/features/osm.py`)
- Deduplicate on OSM element id; map `amenity/shop/leisure/tourism` tags to 7 POI categories
- BallTree catchment query (500 m radius) per GTFS stop to count POIs by category
- Apply `log1p` to all POI count columns

### 2.2 Feature Alignment (`src/features/feature_align.py`)

All 8 cleaned sources are merged onto a master daily index (2022-01-01 – 2025-12-31, 1 461 days):

- Ridership lag features (`lag_7`, `lag_14`, `lag_28`) computed on the full 2019+ history first, then sliced — ensures valid values at the start of the window
- Time-varying sources (fuel prices, rainfall, holidays) reindexed to the daily master; remaining gaps filled by forward-fill or linear interpolation
- Static sources (population, GTFS, OSM, GADM) broadcast to every row
- Final output: `data/features/features_aligned.csv` — **1 461 rows × 79 columns**

| Feature Group | Count | Examples |
|---------------|-------|---------|
| Targets | 13 | `total_ridership`, 12 service lines |
| Temporal / Cyclical | 16 | `is_public_holiday`, `days_to_next_public_hol`, `dow_sin/cos`, `month_sin/cos`, `year`, `day_of_year` |
| External | 30 | 6 fuel price levels + 6 fuel pct-changes + 15 state rainfall series (MY01–MY17 excl. 3) + 3 anomaly series |
| Lag | 3 | `ridership_lag_7`, `ridership_lag_14`, `ridership_lag_28` |
| Static | 17 | `pop_density_median`, `pop_density_log_median`, 4 GTFS network counts, 8 OSM POI category means, 3 GADM spatial metrics |

### 2.3 Sequence Building (`src/features/sequence_builder.py`)

- Sliding-window sequences: **X** shape `(N, T_in, 79)`, **y** shape `(N, 7)` for look-back T_in ∈ {14, 28, 56}
- Chronological 70 / 15 / 15 split — no shuffle
- MCO rows excluded from sequence construction by default
- MinMaxScaler fitted on training split only (prevents future leakage); arrays stored as `float16`
- Graph models build a Pearson correlation adjacency on-the-fly from `X_train` (threshold 0.1, symmetric normalisation)

---

## 3. Notable Exploratory Data Analysis

The EDA suite in `src/eda/` contains 40 scripts across univariate, bivariate, multivariate, and thematic categories.

| Finding | Script | Key Detail |
|---------|--------|-----------|
| **MCO collapse and unequal recovery** | `eda/ridership/ridership.py` | Change-point detection (Binseg, 5 breakpoints) identifies the lockdown onset and three recovery phases; per-service collapse % and post-MCO recovery rate differ substantially by mode |
| **Northeast monsoon pattern** | `eda/rainfall/rainfall.py` | Monthly wet-day ratio and extreme-event frequency (> 50 mm) peak in November–January; east-coast states (Kelantan, Terengganu, Pahang) record significantly more extreme days than west-coast counterparts |
| **Fuel price elasticity** | `eda/theme/demand_drivers_series.py` | Binned scatter of RON95 price vs. total ridership shows a moderate positive correlation — higher fuel prices are associated with increased transit use, consistent with modal substitution |
| **Holiday anticipation effect** | `eda/bivariate/holiday_ridership.py` | Lead/lag cross-correlation reveals a ridership dip 1–2 days before public holidays and a return surge on the first post-holiday workday; school holidays show a softer, multi-day ramp |
| **Klang Valley transit concentration** | `eda/gtfs/gtfs.py` | DBSCAN clustering (eps = 0.05 rad) on stop coordinates identifies dense clusters in Greater KL; most peninsular states outside Selangor/KL have fewer than 5% of total stops |
| **Multicollinearity in fuel features** | `eda/multivariate/full_demand_model_feature_correlation_matrix.py` | Condition number and eigenvalue analysis flags high collinearity among RON95 level, RON97 level, and diesel level — informed SHAP-guided feature reduction in HMT-TSF (79 → 53 features) |
| **Underserved population zones** | `eda/multivariate/underserved_population_identification.py` | Joint analysis of population density × transit supply × walking friction identifies priority zones where high residential density coincides with low stop coverage and poor pedestrian access |
| **Seasonal demand–supply mismatch** | `eda/multivariate/seasonal_demand_supply_mismatch.py` | School holiday periods show systematic over-supply relative to demand; Ramadan shifts demand composition by service type, with bus ridership declining more than rail |
| **Network topology by service** | `eda/gtfs/gtfs.py` | Average node degree and edge-to-node density ratios vary substantially across operators; ETS and Intercity are sparsely connected long-haul networks while LRT Kelana Jaya is the most densely meshed |
| **Feature correlation matrix** | `eda/multivariate/full_demand_model_feature_correlation_matrix.py` | Full 79 × 79 correlation heatmap; lag features (lag_7/14/28) are the strongest individual predictors of next-day ridership; static features (GTFS, OSM, GADM) cluster at near-zero correlation with ridership |

---

## 4. Model Comparison

### 4.1 Spatio-Temporal Series (LSTM-Family)

Six models in `src/models/spatio-temporal-based/`; tuned variants in `src/models/spatio-temporal-tuned/`.

| Model | Key Architecture | Strengths | Weaknesses |
|-------|-----------------|-----------|------------|
| **LSTM** | Unidirectional LSTM [hidden=64, layers=1] → MLP | Simplest baseline; establishes performance floor; low parameter count | Single hidden vector at T is an information bottleneck; no mechanism to identify which timesteps or features drive the forecast |
| **BiLSTM** | Bidirectional LSTM [hidden=64]; concatenate forward + backward hidden states → MLP | Mid-window peaks and dips are visible from both temporal directions; richer positional context than LSTM | No explicit attention or pattern extraction; already near capacity ceiling at base hyperparameters |
| **TPA-LSTM** | LSTM → Conv1d on hidden-state matrix → temporal pattern attention → MLP | Discovers which recurring temporal shapes (daily, weekly rhythms) matter most; attention weights are semi-interpretable | CNN filters are fixed per training session; architecture still depends on LSTM hidden vector quality |
| **CNN-LSTM** | Conv1d layers (local extraction) → LSTM (sequence modelling) → MLP; three modes: sequential, parallel, augmented | Hierarchical feature extraction; augmented mode (CNN→LSTM + raw skip connection) generalises best by preserving direct access to input features | CNN flattens the spatial dimension of features; mode selection (3 variants) adds experimental complexity |
| **CNN-BiLSTM** | Conv1d → BiLSTM [hidden=64] → MLP | Two complementary inductive biases (local translation-invariant features + bidirectional global context); strong on mid-window anomalies | Higher parameter count than CNN-LSTM; requires careful regularisation to avoid overfitting |
| **ST-LSTM** | Parallel temporal stream (LSTM) and spatial stream (weight-shared MLP across time) → concat → MLP | Clean conceptual separation of "when" (LSTM) from "which features co-activate" (spatial encoder); spatial encoder is weight-shared across timesteps | Spatial encoder mean-pools across the time dimension, losing temporal order; spatial and temporal streams are fused late without interaction during encoding |

### 4.2 Graph-Based Series

Five models in `src/models/graph-based/`; tuned variants in `src/models/graph-tuned/`.
All models treat the N input features as graph nodes. Adjacency is built from absolute Pearson correlation of feature columns in X_train (threshold 0.1, symmetric normalisation) unless noted.

| Model | Key Architecture | Strengths | Weaknesses |
|-------|-----------------|-----------|------------|
| **STGCN** | Interleaved temporal gated convolutions (GLU) and Chebyshev graph convolutions; fixed pre-computed Laplacian; 2 ST-Conv blocks | Computationally efficient; K-hop Chebyshev avoids eigendecomposition; well-validated on traffic benchmarks; interpretable static adjacency | Static graph cannot adapt to changing feature correlations; most severe MCO degradation (−22.54 pp) of all models |
| **MTGNN** | End-to-end learnable asymmetric adjacency from node embeddings; multi-scale inception blocks with dilated kernels (1, 3, 5, 7); skip connections | No pre-computed correlation needed; asymmetric graph allows directional feature influence; only model to prefer lb=56 (benefits from longer context) | Graph learning adds parameters; asymmetric adjacency can overfit the training-period topology |
| **STSGCN** | Block-tridiagonal 3N × 3N Spatial-Temporal Synchronous Graph (STSG); ChebConv + GLU per layer; centre-node extraction | Unified single-pass encoding of spatial and temporal adjacency; avoids alternating ST modules; stable on long sequences | Each layer shrinks the temporal dimension by 2, limiting network depth; among the weakest under MCO (−21.96 pp) |
| **STFGNN** | Dual static graphs: spatial (Pearson co-variation) + temporal (correlation of mean temporal profiles); per-layer learned gate α blends spatial and temporal representations | Two complementary graph perspectives; gate α is interpretable (which graph dominates per layer); LayerNorm + residual stabilises deep stacks | Over-regularisation at high dropout; performance regressed by −1.35 pp from base to tuned — high dropout coefficient suppresses useful signal |
| **PDR-STGCN** | Two-channel input: original signal + weekly lag-difference (explicit periodicity); mixed adjacency: static sym-norm Pearson + per-sample dynamic attention; learned scalar λ balances static vs. dynamic | Only model with explicit seasonal encoding via lag-difference channel; dynamic relational graph adapts per input sample; largest tuning gain in the study (+10.03 pp) | Most complex baseline; highest number of hyperparameters (period, dk, λ); worst base performance (67.75%) before tuning; training can be unstable |

### 4.3 Attention-Based Series

Three models in `src/models/attention-based/`; tuned variants in `src/models/attention-tuned/`.
All use AMP (fp16 + GradScaler) on the A100.

| Model | Key Architecture | Strengths | Weaknesses |
|-------|-----------------|-----------|------------|
| **ASTGCN** | Multi-head spatial attention (over nodes) → Chebyshev GCN → multi-head temporal attention (over timesteps), repeated across n_blocks | Dual attention re-weights both spatial node relevance and temporal step relevance; transparent attention maps; strong tuning gain (+4.48 pp) from increased model capacity | Graph adjacency is still static; high parameter count per block; poor MCO robustness (−18.32 pp from no-MCO to MCO) |
| **Autoformer** | FFT-based auto-correlation (O(L log L)) discovers periodic lags; explicit trend/seasonal decomposition via moving-average sub-layer throughout encoder and decoder | Sub-linear complexity; trend/seasonal decomposition aligns naturally with transit periodicity; self-correlation selects dominant lag periods automatically | Catastrophic performance collapse at lb=56 (Combined% 58.59%); FFT amplifies noise at longer sequences; sensitive to decomposition kernel width |
| **Informer** | ProbSparse self-attention: top-u queries selected by KL-divergence sparsity measure, others receive mean(V); sequence distilling halves T after each encoder layer; generative (non-autoregressive) decoder | Most robust baseline under MCO (−4.64 pp, best among all baselines); graceful degradation to full attention for short sequences (lb=14); generative decoder avoids autoregressive exposure bias | Sequence distilling discards some temporal resolution; many interacting hyperparameters; limited upside from tuning (near-optimal at base) |

### 4.4 Hybrid SOTA

`src/models/hybrid/hmttsf.py`; full architecture details in `src/models/hybrid/HMT-TSF.md`.

| Model | Key Architecture | Strengths | Weaknesses |
|-------|-----------------|-----------|------------|
| **HMT-TSF** | RevIN instance normalisation → 5-group semantic feature encoders → Temporal Transformer block → [Multi-Scale TCN (WaveNet-style, DropPath) ‖ Feature GCN (Pearson adjacency + learned temporal attention) ‖ Regime Gating Embedding (K=3 soft mixture)] → SE-gated fusion → dual forecast heads (primary + boosting); optional post-hoc CatBoost residual correction | Best study-wide performance: Combined% 81.62%, R² 0.807 (nomco_lb14, feature-reduced variant); regime gating handles MCO structural break without date metadata; widest lookback support (7–84 days); SHAP-guided feature reduction (79→53 features) further improves accuracy | Most complex architecture; many interacting hyperparameters requiring manual search; SHAP analysis and CatBoost boosting add significant runtime; mco_lb84 collapses when static spatial features are removed (−4.42 pp vs. full-feature variant) |

---

## 5. Performance Summary (No-MCO, Lookback = 14)

Combined% = `max(0, 100 − MAPE% − MAE% − RMSE%)` where all percentage terms use mean-demand normalisation. Higher is better.

### Baseline Models

| Rank | Model | Series | Combined% | R² |
|------|-------|--------|-----------|-----|
| 1 | Informer | Attention | 79.13 | 0.778 |
| 2 | BiLSTM | Spatio-temporal | 79.04 | 0.793 |
| 3 | TPA-LSTM | Spatio-temporal | 78.67 | 0.781 |
| 4 | CNN-BiLSTM | Spatio-temporal | 78.18 | 0.768 |
| 5 | ST-LSTM | Spatio-temporal | 78.01 | 0.769 |
| 6 | LSTM | Spatio-temporal | 78.13 | 0.774 |
| 7 | CNN-LSTM | Spatio-temporal | 77.64 | 0.751 |
| 8 | MTGNN | Graph | 77.62 | 0.762 |
| 9 | STGCN | Graph | 77.44 | 0.756 |
| 10 | Autoformer | Attention | 76.71 | 0.744 |
| 11 | STFGNN | Graph | 75.89 | 0.731 |
| 12 | STSGCN | Graph | 73.33 | 0.698 |
| 13 | ASTGCN | Attention | 74.34 | 0.714 |
| 14 | PDR-STGCN | Graph | 67.75 | 0.641 |

### Tuned Models vs. HMT-TSF

| Rank | Model | Combined% | R² | Δ vs Base |
|------|-------|-----------|-----|----------|
| 1 | **HMT-TSF-FR** | **81.62** | **0.807** | — |
| 2 | HMT-TSF | 81.46 | 0.797 | — |
| 3 | Informer (tuned) | 79.99 | 0.778 | +0.86 pp |
| 4 | TPA-LSTM (tuned) | 79.95 | 0.782 | +1.28 pp |
| 5 | BiLSTM (tuned) | 79.44 | 0.794 | +0.40 pp |
| 6 | ST-LSTM (tuned) | 79.13 | 0.774 | +1.12 pp |
| 7 | ASTGCN (tuned) | 78.82 | 0.754 | +4.48 pp |
| 8 | CNN-LSTM Aug (tuned) | 78.57 | — | +3.21 pp |
| 9 | PDR-STGCN (tuned) | 77.78 | — | **+10.03 pp** |
| 10 | STSGCN (tuned) | 77.14 | — | +3.81 pp |

### MCO Robustness (Tuned, Lookback = 14)

| Rank | Model | MCO Combined% | Δ vs No-MCO |
|------|-------|--------------|------------|
| 1 | Informer (tuned) | 75.79 | −4.64 pp |
| 2 | HMT-TSF / HMT-TSF-FR | 75.66 | −5.80 / −5.96 pp |
| 3 | ST-LSTM (tuned) | 75.05 | −4.08 pp |
| 4 | TPA-LSTM (tuned) | 74.77 | −5.18 pp |
| … | … | … | … |
| Last | STGCN (tuned) | 53.79 | −22.54 pp |
