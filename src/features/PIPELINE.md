# Feature Engineering Pipeline

End-to-end guide for generating cleaned data and building model-ready sequences
from all 8 spatio-temporal feature sources. All commands are run from the
**repository root**.

---

## Pipeline Overview

```
Raw data (data/raw/)
        │
        ▼
 [Cleaning scripts]  ─── 8 sources, some with dependencies (see order below)
        │
        ▼
 feature_align.py    ─── merges all sources onto a daily date index
        │
        ▼
 sequence_builder.py ─── sliding-window tensors (X, y) for all 15 models
        │
        ▼
 data/sequences/lstm/  ─── X_train.npy, y_train.npy, …, scaler_X.pkl, …
```

---

## Prerequisites

```bash
pip install pandas numpy scikit-learn joblib geopandas shapely
```

`geopandas` and `shapely` are only needed by `gadm.py` and `population.py`.
All other scripts run on the core stack (pandas, numpy, scikit-learn).

---

## Step 1 — Run Cleaning Scripts

Run in the order shown. Scripts marked **independent** can run in any order or
in parallel. Scripts marked **depends on** must run after their prerequisites.

### 1a. Ridership (independent)

Cleans the raw daily ridership CSV, flags the MCO period, adds cyclical
temporal encoding, and exports a train/val/test split.

```bash
python src/features/ridership.py
```

Outputs:
- `data/cleaned/ridership_headline_clean.csv`
- `data/cleaned/ridership_post_mco_split.csv`
- `data/scalers/ridership_scaler.pkl`

---

### 1b. Fuel Price (independent)

Upsamples weekly fuel price data to daily by forward-fill, adds a
percent-change feature, and exports level + change tables.

```bash
python src/features/fuelprice.py
```

Outputs:
- `data/cleaned/fuelprice_level_daily.csv`
- `data/cleaned/fuelprice_change_daily.csv`
- `data/cleaned/fuelprice_cleaned.csv`
- `data/scalers/fuelprice_scaler.pkl`

---

### 1c. Holidays (independent)

Expands date-range holiday records to one row per day, adds lead/lag
features (days to/since next holiday) and cyclical DOW/month encoding.

```bash
python src/features/holiday.py
```

Outputs:
- `data/cleaned/holiday_daily_features.csv`
- `data/cleaned/school_public_holiday_clean.csv`

---

### 1d. Rainfall (independent)

Pivots long-format state-level rainfall to wide format (date × state),
interpolates interior gaps up to 7 days, and backfills/forward-fills edges.

```bash
python src/features/rainfall.py
```

Outputs:
- `data/cleaned/rainfall_wide_daily.csv`
- `data/cleaned/rainfall_combined_final.csv`

---

### 1e. GADM Boundaries (independent)

Cleans the GeoJSON admin boundaries, extracts state centroids, and builds
the shared-border adjacency matrix used by graph-based models.

```bash
python src/features/gadm.py
```

Outputs:
- `data/cleaned/gadm_mys_l1_clean.geojson`
- `data/cleaned/gadm_state_nodes.csv`
- `data/cleaned/gadm_adj_matrix.npy`
- `data/cleaned/gadm_adj_matrix_weighted.npy`
- `data/cleaned/gadm_adj_edges.csv`

---

### 1f. GTFS Static Data (independent, run for each operator)

Cleans GTFS feeds and exports stop-node and stop-edge tables consumed by
`population.py`, `osm.py`, and `feature_align.py`. Run once per operator.

```bash
python src/features/gtfs.py \
    --input  data/raw/gtfs_rapid_rail_kl \
    --output data/cleaned/gtfs_rapid_rail_kl \
    --operator rapid_rail_kl

python src/features/gtfs.py \
    --input  data/raw/gtfs_rapidbus_kl \
    --output data/cleaned/gtfs_rapidbus_kl \
    --operator rapidbus_kl

python src/features/gtfs.py \
    --input  data/raw/gtfs_rapidbus_penang \
    --output data/cleaned/gtfs_rapidbus_penang \
    --operator rapidbus_penang

python src/features/gtfs.py \
    --input  data/raw/gtfs_ktmb \
    --output data/cleaned/gtfs_ktmb \
    --operator ktmb
```

Outputs per operator (e.g. `rapid_rail_kl`):
- `data/cleaned/gtfs_rapid_rail_kl/gtfs_stop_nodes_rapid_rail_kl.csv`
- `data/cleaned/gtfs_rapid_rail_kl/gtfs_stop_edges_rapid_rail_kl.csv`
- Cleaned GTFS text files (`stops.txt`, `trips.txt`, etc.)

---

### 1g. Population Density (depends on: GADM, GTFS)

Cleans the population density raster, aggregates to state level using GADM
polygons, and samples density at each GTFS stop via nearest-grid lookup.

```bash
python src/features/population.py
```

Outputs:
- `data/cleaned/population_density_clean.csv`
- `data/cleaned/population_by_state.csv`
- `data/cleaned/population_at_stops.csv`

---

### 1h. OSM Points of Interest (depends on: GTFS)

Cleans raw OSM POI JSON, deduplicates elements, and counts POIs by
category within a 500 m catchment of each GTFS stop.

```bash
python src/features/osm.py
```

Outputs:
- `data/cleaned/osm_pois_clean.json`
- `data/cleaned/poi_counts_at_stops.csv`

---

## Step 2 — Feature Alignment

Merges all 8 cleaned sources onto a single daily date index, producing the
flat feature matrix consumed by every model. Static sources (population,
GTFS, OSM POI, GADM) are broadcast to all dates.

```bash
python src/features/feature_align.py
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--date-start` | `2022-01-01` | Start of master date range |
| `--date-end` | `2025-12-31` | End of master date range |
| `--output-dir` | `data/features` | Output directory |

Example with custom range:

```bash
python src/features/feature_align.py \
    --date-start 2022-01-01 \
    --date-end   2025-12-31
```

Outputs:
- `data/features/features_aligned.csv` — daily matrix (days × N_features)
- `data/features/feature_metadata.json` — column groups, source list, null counts

The terminal output reports the total feature count and a breakdown by group
(targets, temporal, external, static). Feature sources included are listed
explicitly; any `[SKIP]` messages indicate a cleaning script has not been run.

---

## Step 3 — Sequence Builder

Slices the aligned feature matrix into overlapping sliding windows and
produces scaled tensors for all 15 models.

```bash
python src/features/sequence_builder.py
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--features-path` | `data/features/features_aligned.csv` | Aligned features |
| `--T-in` | `14` | Look-back window (days) |
| `--T-out` | `7` | Forecast horizon (days) |
| `--target` | `total_ridership` | Target column name |
| `--train-frac` | `0.70` | Training split fraction |
| `--val-frac` | `0.15` | Validation split fraction |
| `--include-mco` | off | Include MCO-period rows |
| `--dtype` | `float16` | Storage dtype (`float16` or `float32`) |
| `--compress` | off | Save `.npz` instead of `.npy` |

Outputs (`data/sequences/lstm/`):
- `X_train.npy`, `y_train.npy`
- `X_val.npy`, `y_val.npy`
- `X_test.npy`, `y_test.npy`
- `scaler_X.pkl`, `scaler_y.pkl`
- `split_dates.json` — split boundaries, T_in, T_out, n_features, target index

Tensor shapes:
- `X`: `(N_samples, T_in=14, N_features)`
- `y`: `(N_samples, T_out=7)`

> `N_features` is whatever `features_aligned.csv` contains. Check
> `split_dates.json → n_features` for the exact count after running.

---

## Full Run (copy-paste)

```bash
# 1. Independent cleaning scripts
python src/features/ridership.py
python src/features/fuelprice.py
python src/features/holiday.py
python src/features/rainfall.py
python src/features/gadm.py

# GTFS — one call per operator
python src/features/gtfs.py --input data/raw/gtfs_rapid_rail_kl  --output data/cleaned/gtfs_rapid_rail_kl  --operator rapid_rail_kl
python src/features/gtfs.py --input data/raw/gtfs_rapidbus_kl    --output data/cleaned/gtfs_rapidbus_kl    --operator rapidbus_kl
python src/features/gtfs.py --input data/raw/gtfs_rapidbus_penang --output data/cleaned/gtfs_rapidbus_penang --operator rapidbus_penang
python src/features/gtfs.py --input data/raw/gtfs_ktmb           --output data/cleaned/gtfs_ktmb           --operator ktmb

# 2. Scripts that depend on GADM + GTFS outputs
python src/features/population.py
python src/features/osm.py

# 3. Feature alignment (all 8 sources → one daily matrix)
python src/features/feature_align.py

# 4. Sequence builder (daily matrix → model-ready tensors)
python src/features/sequence_builder.py
```

---

## Notes

- **MCO exclusion** — `sequence_builder.py` drops 2020-03-18 to 2021-12-31 by
  default. Pass `--include-mco` to keep this period (the `is_mco` flag is still
  available as a feature for models that can learn the anomaly).

- **Static feature scaling** — GTFS, OSM POI, GADM, and population features are
  broadcast to every date as constant columns. `MinMaxScaler` in
  `sequence_builder.py` scales them normally along with dynamic features; this
  is correct because the scaler fits only on the training split.

- **[SKIP] messages** — `feature_align.py` prints `[SKIP]` for any source whose
  cleaned file is not yet present. The matrix is still produced with whatever
  sources are available, but the missing features will not be in it.

- **Graph adjacency** — Graph-based models (STGCN, Graph WaveNet, DCRNN, STGAT,
  PatchTST+Graph, ASTGCN) build their feature-correlation adjacency matrix
  on-the-fly at training time from the N_features columns in `X_train.npy`.
  No separate graph file is needed.
