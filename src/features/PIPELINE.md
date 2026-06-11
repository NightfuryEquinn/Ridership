# Feature Engineering Pipeline

> Last updated: 2026-05-22

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
        │                 also computes lag features + year/day_of_year
        │                 exports TWO matrices:
        │                   features_aligned.csv         (MCO included)
        │                   features_aligned_no_mco.csv  (MCO excluded)
        ▼
 sequence_builder.py ─── called TWICE per lookback window (once per MCO condition)
        │
        ▼
 data/sequences/lstm/      ─── MCO excluded, T_in=14
 data/sequences/lstm_mco/  ─── MCO included, T_in=14
 (same pattern for lookback_28/ and lookback_56/)
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

Merges all 8 cleaned sources onto a single daily date index. Also computes
two derived feature sets inline:

- **Autoregressive lag features** (`ridership_lag_{7,14,28}`) — computed from
  the full historical ridership series so the first rows of the 2022 window
  have valid values even for the 28-day lag.
- **Secular trend features** (`year`, `day_of_year`) — capture the post-MCO
  ridership recovery trajectory and intra-year seasonality not captured by
  monthly cyclical encoding.

Static sources (population, GTFS, OSM POI, GADM) are broadcast to all dates.
Any source whose cleaned file is absent prints `[SKIP]` and is omitted from
the matrix — the pipeline still produces a valid (but partial) output.

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
- `data/features/features_aligned.csv` — MCO included, 79 columns × 2,557 days (2019-01-01 – 2025-12-31)
- `data/features/features_aligned_no_mco.csv` — MCO excluded, 79 columns × 1,461 days (2022-01-01 – 2025-12-31)
- `data/features/feature_metadata.json` — column groups, source list, null counts (MCO included)
- `data/features/feature_metadata_no_mco.json` — same metadata for the MCO-excluded window

The terminal output reports the total feature count and a breakdown by group
(targets, temporal, external, lag, static). Feature sources included are listed
explicitly; any `[SKIP]` messages indicate a cleaning script has not been run.

**Expected feature count** (all 8 sources present): **79 columns**

| Group | Count | Source |
|---|---|---|
| Targets | 13 | ridership (12 lines + total) |
| Temporal | 16 | holiday flags/lead-lag/cyclical + year + day_of_year |
| External | 30 | fuel (15) + rainfall (15) |
| Lag | 3 | ridership_lag_7/14/28 |
| Static | 17 | population + GTFS + OSM POI + GADM |

> The exact column count is written to `data/features/feature_metadata.json` and to `data/sequences/*/split_dates.json → n_features` after each run. Use these files as the authoritative source if sources change.

---

## Step 3 — Sequence Builder

Slices the aligned feature matrix into overlapping sliding windows and
produces scaled tensors for all models. MCO filtering is **not** done here —
pass the appropriate features file via `--features-path`.

`run_pipeline.py` calls this script **twice per lookback window** automatically,
pointing at `features_aligned_no_mco.csv` and `features_aligned.csv` in turn.
To run manually for a single condition:

```bash
# MCO excluded (default for model training)
python src/features/sequence_builder.py \
    --features-path data/features/features_aligned_no_mco.csv \
    --out-dir data/sequences/lstm

# MCO included
python src/features/sequence_builder.py \
    --features-path data/features/features_aligned.csv \
    --out-dir data/sequences/lstm_mco
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--features-path` | `data/features/features_aligned.csv` | Aligned features CSV to load |
| `--T-in` | `14` | Look-back window (days) |
| `--T-out` | `7` | Forecast horizon (days) |
| `--target` | `total_ridership` | Target column name |
| `--train-frac` | `0.70` | Training split fraction |
| `--val-frac` | `0.15` | Validation split fraction |
| `--out-dir` | auto | Output directory (default: `lstm/` for T_in=14, else `lookback_{N}/`) |
| `--dtype` | `float16` | Storage dtype (`float16` or `float32`) |
| `--compress` | off | Save `.npz` instead of `.npy` |

Each output directory contains:
- `X_train.npy`, `y_train.npy`
- `X_val.npy`, `y_val.npy`
- `X_test.npy`, `y_test.npy`
- `X_future_train.npy`, `X_future_val.npy`, `X_future_test.npy` ← known future temporal features
- `scaler_X.pkl`, `scaler_y.pkl`
- `split_dates.json` — split boundaries, T_in, T_out, n_features, target index

Tensor shapes:
- `X`: `(N_samples, T_in, N_features)`
- `y`: `(N_samples, T_out=7)`
- `X_future`: `(N_samples, T_out=7, 16)` — temporal features for forecast horizon

> `N_features` is whatever the input CSV contains. Check
> `split_dates.json → n_features` for the exact count after running.

> **Temporal column resolution (fixed 2026-06-11):** the 16 `X_future`
> columns are resolved **by name** from `feature_metadata*.json →
> column_groups.temporal` and selected in ascending column-index order;
> the resolved indices and names are recorded in `split_dates.json →
> temporal_feat_indices / temporal_feat_names`. The temporal group is
> non-contiguous in the aligned column order (year/day_of_year sit apart
> from the holiday/cyclical block). An earlier version used a hardcoded
> contiguous slice (13–28) that silently selected lag/trend/fuel columns
> after a column reorder — sequence sets built before this fix carry the
> wrong `X_future` content and must be rebuilt.

---

## Full Run (copy-paste)

```bash
# 1a–1d. Core independent cleaning scripts
python src/features/ridership.py
python src/features/fuelprice.py
python src/features/holiday.py
python src/features/rainfall.py

# 1e. GADM boundaries
python src/features/gadm.py

# 1f. GTFS — one call per operator
python src/features/gtfs.py --input data/raw/gtfs_rapid_rail_kl  --output data/cleaned/gtfs_rapid_rail_kl  --operator rapid_rail_kl
python src/features/gtfs.py --input data/raw/gtfs_rapidbus_kl    --output data/cleaned/gtfs_rapidbus_kl    --operator rapidbus_kl
python src/features/gtfs.py --input data/raw/gtfs_rapidbus_penang --output data/cleaned/gtfs_rapidbus_penang --operator rapidbus_penang
python src/features/gtfs.py --input data/raw/gtfs_ktmb           --output data/cleaned/gtfs_ktmb           --operator ktmb

# 1g–1h. Scripts that depend on GADM + GTFS outputs
python src/features/population.py
python src/features/osm.py

# 2. Feature alignment (all 8 sources + lag + trend → one daily matrix)
python src/features/feature_align.py

# 3. Sequence builder — run twice per lookback (MCO excluded + MCO included)
python src/features/sequence_builder.py --features-path data/features/features_aligned_no_mco.csv --T-in 14 --out-dir data/sequences/lstm
python src/features/sequence_builder.py --features-path data/features/features_aligned.csv         --T-in 14 --out-dir data/sequences/lstm_mco

python src/features/sequence_builder.py --features-path data/features/features_aligned_no_mco.csv --T-in 28 --out-dir data/sequences/lookback_28
python src/features/sequence_builder.py --features-path data/features/features_aligned.csv         --T-in 28 --out-dir data/sequences/lookback_28_mco

python src/features/sequence_builder.py --features-path data/features/features_aligned_no_mco.csv --T-in 56 --out-dir data/sequences/lookback_56
python src/features/sequence_builder.py --features-path data/features/features_aligned.csv         --T-in 56 --out-dir data/sequences/lookback_56_mco
```

---

## Notes

- **MCO split** — `feature_align.py` exports both `features_aligned.csv` (MCO
  included) and `features_aligned_no_mco.csv` (MCO excluded, 2020-03-18 –
  2021-12-31 rows dropped). `sequence_builder.py` performs no MCO filtering
  itself — select the correct input file via `--features-path`. `run_pipeline.py`
  builds both sequence sets automatically.

- **Static feature scaling** — GTFS, OSM POI, GADM, and population features are
  broadcast to every date as constant columns. `MinMaxScaler` in
  `sequence_builder.py` scales them normally along with dynamic features; this
  is correct because the scaler fits only on the training split.

- **[SKIP] messages** — `feature_align.py` prints `[SKIP]` for any source whose
  cleaned file is not yet present. The matrix is still produced with whatever
  sources are available, but the missing features will not be in it.

- **Lag features** — `ridership_lag_{7,14,28}` are computed from the full
  historical ridership series (2019-present) so that rows at 2022-01-01 have
  valid values. For lookback=14, `ridership_lag_28` provides 28-day look-back
  that the sequence window alone cannot reach.

  **MCO-bridging caveat:** in the MCO-excluded dataset the lag features at the
  start of the post-MCO window reference dates *inside* the excluded MCO
  period (e.g., `ridership_lag_28` at 2022-01-01 references 2021-12-04). This
  is deliberate — fabricating or zeroing those values would inject artificial
  signal — but it means the "MCO-excluded" condition still carries indirect
  MCO information through the lag channel for the first 28 days of the
  series. State this when describing the no-MCO condition in the methodology
  chapter.

- **Graph adjacency** — Graph-based models (STGCN, MTGNN, STSGCN, STFGNN,
  PDR-STGCN, ASTGCN) and HMT-TSF build their feature-correlation adjacency
  matrix on-the-fly at training time from the N_features columns in
  `X_train.npy`. No separate graph file is needed.

- **HMT-TSF lookback 7 and 84** — Sequence dirs for these two non-standard
  lookbacks must be built separately if needed (both MCO conditions):
  ```bash
  python src/features/sequence_builder.py --features-path data/features/features_aligned_no_mco.csv --T-in 7  --out-dir data/sequences/lookback_7
  python src/features/sequence_builder.py --features-path data/features/features_aligned.csv         --T-in 7  --out-dir data/sequences/lookback_7_mco
  python src/features/sequence_builder.py --features-path data/features/features_aligned_no_mco.csv --T-in 84 --out-dir data/sequences/lookback_84
  python src/features/sequence_builder.py --features-path data/features/features_aligned.csv         --T-in 84 --out-dir data/sequences/lookback_84_mco
  ```
