# DATA.md

> Last updated: 2026-05-22

This document describes the layout, contents, and conventions of every directory under `data/`. The pipeline that produces these artefacts is documented in `PIPELINE.md`.

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

#### GTFS Validation Findings

Reports generated with the GTFS validator are in `gtfs_reports/`. Per-operator summary:

| Operator | Total notices | Errors | Warnings | Infos |
|----------|--------------|--------|----------|-------|
| KTMB | 510 | 2 | 508 | 0 |
| Rapid Rail KL | 137 | 0 | 127 | 10 |
| RapidBus KL | 5,064 | 0 | 5,064 | 0 |
| RapidBus Penang | 6,038 | 0 | 6,038 | 0 |

**Fixed in `src/features/gtfs.py`:**

| Issue | Operator(s) | Count | Resolution |
|-------|-------------|-------|------------|
| `foreign_key_violation` — stop_id in stop_times not in stops | KTMB | 2 | `clean_stop_times()` drops rows with unknown stop_id |
| `unknown_column` — non-standard columns in stops/routes/stop_times | Rapid Rail KL | 10 (INFO) | `drop_nonstandard()` removes all 10 via per-operator preset |
| `unusable_trip` — trip has fewer than 2 usable stop_time rows | KTMB (67), RapidBus KL (11) | 78 | `clean_stop_times()` drops the rows; `run()` then prunes the trip_ids from `trips_df` |
| `unused_trip` — trip defined but has no stop_time rows at all | KTMB (67), RapidBus KL (11) | 78 | `run()` prunes any trip_id absent from the cleaned stop_times after all cleaning passes |

**Intentionally ignored (no modelling impact):**

| Issue | Operator(s) | Count | Reason ignored |
|-------|-------------|-------|----------------|
| `mixed_case_recommended_field` — stop names / headsigns in ALL CAPS | All 4 | 10,970 | Stop names are not model features; display-only |
| `expired_calendar` — service date range in the past | KTMB, RapidRail, RapidBus KL | 5 | Static historical dataset; expiry is expected and irrelevant to topology extraction |
| `fast_travel_between_consecutive_stops` / `fast_travel_between_far_stops` | KTMB (313), RapidBus KL (189) | 502 | Affects `travel_time_s` in edge table; graph models use Pearson-correlation adjacency, not travel-time weights |
| `stop_without_stop_time` — stop defined but never served | KTMB | 8 | Produces isolated node with no edges; negligible effect on stop-count static feature |
| `stop_too_far_from_shape` — stop >100 m from route polyline | RapidRail (2), RapidBus KL (27), Penang (1) | 30 | Shapes not used in feature pipeline or adjacency construction |
| `stops_match_shape_out_of_order` | RapidBus KL | 6 | Shapes not used in feature pipeline |
| `route_color_contrast` — insufficient colour contrast | Rapid Rail KL | 1 | UI/accessibility concern; not a model feature |
| `route_long_name_contains_short_name` | RapidRail (1), Penang (47) | 48 | Route names not used as model features |
| `route_short_name_too_long` | KTMB (3), RapidBus KL (2) | 5 | Route names not used as model features |
| `missing_recommended_file` — feed_info.txt absent | All 4 | 4 | Publisher metadata; not consumed by any pipeline script |
| `trip_coverage_not_active_for_next7_days` | KTMB | 1 | Live-feed validator artefact; inapplicable to static historical snapshot |

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
| `features_aligned.csv` | Daily feature matrix, **MCO period included**. 79 columns × 2,557 days (2019-01-01 – 2025-12-31). Fed to `sequence_builder.py` for MCO-inclusive experiments. |
| `features_aligned_no_mco.csv` | Same matrix with MCO rows (2020-03-18 – 2021-12-31) and pre-2022 rows dropped. 79 columns × 1,461 days (2022-01-01 – 2025-12-31). Default model training input. |
| `feature_metadata.json` | JSON schema for `features_aligned.csv`: **column_order** (authoritative index→name mapping for tensors and SHAP labels), column groups (semantic, not positional), source list, null counts. |
| `feature_metadata_no_mco.json` | Same as above but with the day count after MCO exclusion. |

> **Lag MCO-bridging note:** in `features_aligned_no_mco.csv`, the
> `ridership_lag_{7,14,28}` columns at the start of the post-MCO window
> reference dates inside the excluded MCO period (e.g., lag_28 at
> 2022-01-01 → 2021-12-04). Deliberate — see `PIPELINE.md`.

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

Each lookback window produces **two** sibling directories — one with MCO excluded (default) and one with MCO included. Both are built automatically by `run_pipeline.py` step 3.

| Directory | Lookback (`T_in`) | MCO rows |
|-----------|-------------------|----------|
| `lstm/` | 14 days | excluded |
| `lstm_mco/` | 14 days | included |
| `lookback_28/` | 28 days | excluded |
| `lookback_28_mco/` | 28 days | included |
| `lookback_56/` | 56 days | excluded |
| `lookback_56_mco/` | 56 days | included |

Additional directories for HMT-TSF can be built on demand:

```bash
python src/features/sequence_builder.py --features-path data/features/features_aligned_no_mco.csv --T-in 7  --out-dir data/sequences/lookback_7
python src/features/sequence_builder.py --features-path data/features/features_aligned.csv         --T-in 7  --out-dir data/sequences/lookback_7_mco
python src/features/sequence_builder.py --features-path data/features/features_aligned_no_mco.csv --T-in 84 --out-dir data/sequences/lookback_84
python src/features/sequence_builder.py --features-path data/features/features_aligned.csv         --T-in 84 --out-dir data/sequences/lookback_84_mco
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

MCO rows are removed in `feature_align.py` (written to `features_aligned_no_mco.csv`) before sequence building. The `_mco/` directories are built from `features_aligned.csv` and therefore contain MCO-period windows.

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
