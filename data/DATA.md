# DATA.md

This document describes the layout, contents, and conventions of every directory under `data/`. The pipeline that produces these artefacts is documented in `src/features/PIPELINE.md`.

---

## Top-Level Layout

```
data/
├── raw/            Raw source files — never modified by pipeline scripts
├── cleaned/        Per-source cleaned outputs (CSVs, GeoJSONs, .npy adjacency)
├── features/       Final aligned feature matrix + metadata
├── scalers/        Intermediate scikit-learn scalers (pre-sequence-builder)
├── sequences/      Sliding-window tensors ready for model training
└── DATA.md         This file
```

---

## `data/raw/`

Original, unmodified source data. Scripts read from here but never write back.

### Root files

| File | Size | Description |
|------|------|-------------|
| `fuelprice.csv` | ~48 KB | Weekly RON 95 / RON 97 / diesel retail prices (MYR/litre) published by KPDNHEP |
| `gadm_mys_l1.json` | ~382 KB | Malaysia state/federal-territory boundaries at GADM level-1 (GeoJSON) |
| `malaysia_population_density_2020.csv` | ~23 MB | 2020 census gridded population density raster (one row per grid cell) |
| `mys_rainfall_subnat_2019_2026.csv` | ~4.5 MB | Daily sub-national rainfall totals 2019–2026 from open government data |
| `osm_pois.json` | ~11 MB | Raw OpenStreetMap points-of-interest exported from the Overpass API |
| `ridership_headline.csv` | ~179 KB | Daily ridership counts per service line published by Prasarana / KTMB |
| `school_public_holiday.csv` | ~13 KB | Malaysian public holiday and school holiday calendar |

### GTFS subdirectories

Each operator ships a standard GTFS feed. All four are stored unmodified.

| Directory | Operator | Lines covered |
|-----------|----------|---------------|
| `gtfs_ktmb/` | Keretapi Tanah Melayu Berhad | KTM Komuter, ETS |
| `gtfs_rapid_rail_kl/` | Prasarana (rail) | LRT Ampang/Kelana Jaya, MRT Putrajaya, Monorail |
| `gtfs_rapidbus_kl/` | Prasarana (bus) | RapidBus Kuala Lumpur |
| `gtfs_rapidbus_penang/` | Prasarana (bus) | RapidBus Penang |

Each GTFS directory contains the standard subset of:

| File | Description |
|------|-------------|
| `agency.txt` | Operator name, URL, timezone |
| `calendar.txt` | Service-day schedule (weekday/weekend flags) |
| `routes.txt` | Route IDs, short names, and types |
| `stops.txt` | Stop IDs, names, and WGS-84 coordinates |
| `stop_times.txt` | Arrival/departure times per trip |
| `trips.txt` | Trip-to-route mapping and headsigns |
| `shapes.txt` | (Rapid Rail, RapidBus only) Route polyline geometry |
| `frequencies.txt` | (Rapid Rail, RapidBus only) Headway-based frequency records |

---

## `data/cleaned/`

Outputs of the per-source cleaning scripts in `src/features/`. All files here are deterministically reproducible from `data/raw/` by running the pipeline.

### Ridership

| File | Description |
|------|-------------|
| `ridership_headline_clean.csv` | Daily ridership per service line, duplicates removed, date index normalised |
| `ridership_post_mco_split.csv` | Post-MCO subset (2022-01-01 onwards) used for trend analysis |

### Fuel price

| File | Description |
|------|-------------|
| `fuelprice_cleaned.csv` | Weekly prices forward-filled to a daily index |
| `fuelprice_level_daily.csv` | Daily absolute price levels for all three fuel grades (RON 95 / RON 97 / diesel) |
| `fuelprice_change_daily.csv` | Day-on-day price changes derived from the level file |

### Holiday

| File | Description |
|------|-------------|
| `school_public_holiday_clean.csv` | Normalised holiday calendar with standardised date column |
| `holiday_daily_features.csv` | Daily binary flags + lead/lag indicators for each holiday type, aligned to the full date range |

### GADM (administrative geography)

| File | Description |
|------|-------------|
| `gadm_mys_l1_clean.geojson` | Cleaned state/FT boundaries (invalid geometries fixed, CRS standardised to WGS-84) |
| `gadm_state_nodes.csv` | One row per state: node ID, state name, centroid lat/lon |
| `gadm_adj_edges.csv` | Edge list of shared-border state pairs (undirected) |
| `gadm_adj_matrix.npy` | Binary N×N adjacency matrix (NumPy float32) |
| `gadm_adj_matrix_weighted.npy` | Shared-border-length-weighted N×N adjacency matrix (NumPy float32) |

### GTFS (cleaned transit network)

Four mirrored subdirectories — one per operator — each containing the standard GTFS `.txt` files (rewritten for consistency) plus two derived graph files:

| File | Description |
|------|-------------|
| `gtfs_stop_nodes_[system].csv` | One row per stop: node ID, stop name, lat, lon, route count |
| `gtfs_stop_edges_[system].csv` | Edge list connecting consecutive stops on each route (undirected) |

Operator tags: `ktmb`, `rapid_rail_kl`, `rapidbus_kl`, `rapidbus_penang`.

### Population

| File | Description |
|------|-------------|
| `population_density_clean.csv` | Full cleaned gridded density raster (~35 MB); lat/lon/density per cell |
| `population_by_state.csv` | State-level aggregated population totals |
| `population_at_stops.csv` | Population density interpolated to each GTFS stop location (all operators) |

### OpenStreetMap POI

| File | Description |
|------|-------------|
| `osm_pois_clean.json` | Deduplicated POI list with standardised category tags (~9 MB) |
| `poi_counts_at_stops.csv` | POI category counts within a configurable radius of each GTFS stop |

### Rainfall

| File | Description |
|------|-------------|
| `rainfall_wide_daily.csv` | Wide-format daily rainfall: one column per sub-national region (~3.3 MB) |
| `rainfall_combined_final.csv` | Long-format final rainfall features used by `feature_align.py` |

---

## `data/features/`

Output of `src/features/feature_align.py`. This is the single source of truth for the ML feature set.

| File | Description |
|------|-------------|
| `features_aligned.csv` | Daily feature matrix: every source merged onto a shared date index. ~79 columns (see breakdown below). MCO period rows are retained here; exclusion happens in `sequence_builder.py`. |
| `feature_metadata.json` | JSON schema describing each feature column: source, type, value range, and derived-feature formula. |

### Feature column breakdown (~79 columns)

| Group | Count | Examples |
|-------|-------|---------|
| Target (ridership) | 13 | `lrt_ampang`, `mrt_putrajaya`, …, `total_ridership` |
| Temporal | 16 | `is_public_holiday`, `holiday_lead_1`, `day_sin`, `day_cos`, `year`, `day_of_year` |
| Fuel | 15 | `ron95_level`, `ron97_level`, `diesel_level`, `ron95_change`, … |
| Rainfall | 15 | `rainfall_[region]` (one per sub-national region) |
| Lag | 3 | `ridership_lag_7`, `ridership_lag_14`, `ridership_lag_28` |
| Static / geospatial | 17 | `population_at_stop`, `poi_count_transit`, `gtfs_stop_count`, `gadm_border_count`, … |

---

## `data/scalers/`

Intermediate scikit-learn scalers fitted during the feature-cleaning stage (before `sequence_builder.py`). Used internally by their respective cleaning scripts for exploratory normalisation; the definitive scalers for model training are in `data/sequences/*/scaler_X.pkl` and `scaler_y.pkl`.

| File | Description |
|------|-------------|
| `fuelprice_scaler.pkl` | Scaler fitted on fuel price columns during `fuelprice.py` |
| `ridership_scaler.pkl` | Scaler fitted on ridership columns during `ridership.py` |

---

## `data/sequences/`

Output of `src/features/sequence_builder.py`. Each subdirectory is a self-contained dataset for one lookback window, ready to be loaded by any model script.

### Subdirectory layout

| Directory | Lookback (`T_in`) | Built by |
|-----------|-------------------|----------|
| `lstm/` | 14 days | `sequence_builder.py` (default, no flag) |
| `lookback_28/` | 28 days | `sequence_builder.py --T-in 28` |
| `lookback_56/` | 56 days | `sequence_builder.py --T-in 56` |

Additional directories (`lookback_7/`, `lookback_84/`) can be built on demand for HMT-TSF:

```bash
python src/features/sequence_builder.py --T-in 7
python src/features/sequence_builder.py --T-in 84
```

### Files inside each subdirectory

| File | Shape | dtype | Description |
|------|-------|-------|-------------|
| `X_train.npy` | `(N_train, T_in, F)` | float16 | Training feature windows |
| `X_val.npy` | `(N_val, T_in, F)` | float16 | Validation feature windows |
| `X_test.npy` | `(N_test, T_in, F)` | float16 | Test feature windows |
| `y_train.npy` | `(N_train, T_out, C)` | float16 | Training target windows |
| `y_val.npy` | `(N_val, T_out, C)` | float16 | Validation target windows |
| `y_test.npy` | `(N_test, T_out, C)` | float16 | Test target windows |
| `scaler_X.pkl` | — | — | `MinMaxScaler` fitted on `X_train` only |
| `scaler_y.pkl` | — | — | `MinMaxScaler` fitted on `y_train` only |
| `split_dates.json` | — | — | ISO-8601 boundary dates for train / val / test splits |

**Dimension key:** `N_*` = number of windows, `T_in` = lookback steps, `T_out = 7` (forecast horizon), `F` = feature count (~79), `C = 13` (target series).

**Usage note:** Arrays are stored as `float16` to save disk space. Model scripts must cast to `float32` before feeding tensors to PyTorch:

```python
X_train = np.load("data/sequences/lstm/X_train.npy").astype(np.float32)
```

### Chronological split ratios

| Split | Ratio | Notes |
|-------|-------|-------|
| Train | 70% | Scalers fitted here only |
| Validation | 15% | Early stopping and LR scheduling |
| Test | 15% | Held-out evaluation; never seen during training |

MCO period (2020-03-18 – 2021-12-31) is excluded from all splits.

---

## File-type summary

| Extension | Count | Purpose |
|-----------|-------|---------|
| `.csv` | ~31 | Tabular timeseries, feature matrices, graph edge/node lists |
| `.txt` | ~41 | GTFS feed files (comma-separated, GTFS spec) |
| `.json` | ~3 | GeoJSON boundaries, feature metadata, split date configs |
| `.geojson` | 1 | Cleaned administrative boundaries |
| `.npy` | ~27 | Binary NumPy arrays (ML sequences and adjacency matrices) |
| `.pkl` | ~8 | Pickled scikit-learn scalers |

---

## Naming conventions

| Pattern | Meaning |
|---------|---------|
| `*_clean.csv` / `*_cleaned.csv` | Direct output of a cleaning script |
| `*_daily.*` | Resampled or aggregated to a daily cadence |
| `*_features.*` | Feature-engineered derivative (binary flags, cyclical encodings, etc.) |
| `*_aligned.*` | All sources merged onto a shared date index |
| `*_level_*` / `*_change_*` | Absolute value vs. period-on-period delta |
| `*_wide_*` | Pivoted to wide format (one column per series) |
| `gtfs_[system]/` | GTFS feed scoped to one transit operator |
| `*_nodes.csv` / `*_edges.csv` | Graph representation (node list / edge list) |
| `X_*.npy` / `y_*.npy` | Feature / target tensors for `train`, `val`, `test` |
| `scaler_*.pkl` | Fitted scikit-learn scaler for the named feature group |
