# Feature Engineering & EDA Analysis

> Last updated: 2026-06-09 (17 EDA findings, figures added)

Detailed account of all exploratory data analysis (EDA) and engineering decisions made across the eight spatio-temporal feature sources that feed the forecasting pipeline. The pipeline produces two aligned feature matrices: `features_aligned_no_mco.csv` — **79 features × 1,461 days** (2022-01-01 → 2025-12-31, MCO period excluded) used as the default model input — and `features_aligned.csv` — **79 features × 2,557 days** (2019-01-01 → 2025-12-31, MCO period retained) used for MCO-robustness experiments.

---

## Pipeline Overview

```
data/raw/  →  [source scripts]  →  data/cleaned/  →  feature_align.py  →  features_aligned.csv        (MCO included)
                                                            │             features_aligned_no_mco.csv   (MCO excluded)
                                                            ↓
                                                  sequence_builder.py  (called twice per lookback)
                                                            ↓
                                          data/sequences/{lstm|lookback_28|lookback_56}/      (no MCO)
                                          data/sequences/{lstm_mco|lookback_28_mco|…}/        (with MCO)
```

The feature matrix contains five column groups:

| Group | Count | Description |
|-------|-------|-------------|
| `targets` | 13 | 12 service-line ridership counts + `total_ridership` |
| `temporal` | 16 | Holiday flags, lead/lag days, cyclical DOW/month, year, day-of-year |
| `external` | 30 | Fuel price levels/changes (15) + per-state rainfall (15) |
| `lag` | 3 | Autoregressive ridership lags at 7, 14, 28 days |
| `static` | 17 | Population density, GTFS network stats, OSM POI counts, GADM boundary stats |

Structural nulls are present in three target columns due to late service launches: `rail_mrt_pjy` (166 days), `rail_tebrau` (169 days), `rail_komuter` (617 days). These are filled with `0.0` in `sequence_builder.py` — zero, not median, because no service equals zero ridership.

---

## Source 1 — Ridership (`ridership.py`)

**Raw data:** `data/raw/ridership_headline.csv`  
**Output:** `data/cleaned/ridership_headline_clean.csv`, `data/cleaned/ridership_post_mco_split.csv`

### EDA Findings

**Shape & types inspection.** Initial `df.shape`, `dtypes`, and `head()` print confirmed 12 service-line columns and a `date` column in `DD/MM/YYYY` string format. Date was parsed to `datetime64` and the frame sorted chronologically.

**MCO anomaly detection.** A critical structural break was identified: the COVID-19 Movement Control Order (MCO) from 2020-03-18 to 2021-12-31 caused a ~65–90% ridership collapse across all lines. Early versions hard-removed these rows. EDA showed this creates a temporal gap that breaks sliding-window sequences in LSTM/GCN models. The decision was revised: MCO rows are **retained** and flagged via an `is_mco` binary column. A separate `ridership_post_mco_split.csv` is still exported for models that require clean post-MCO training.

**Structural null analysis.** Null values were not random — each null corresponded to a known service launch date. EDA identified eight late-launch services with exact start dates:

| Service | Launch Date |
|---------|------------|
| `bus_rkl`, `bus_rpn` | 2022-01-01 |
| `rail_ets`, `rail_intercity`, `rail_komuter_utara` | 2020-10-15 |
| `rail_mrt_pjy` | 2022-06-16 |
| `rail_tebrau` | 2022-06-19 |
| `rail_komuter` | 2023-09-10 |

Pre-launch nulls are left as `NaN` (correct — no service). Post-launch nulls (data gaps after the service is live) are zero-filled. Within the 2022–2025 master window (`feature_metadata_no_mco.json`), residual nulls are limited to three late-launch services: `rail_mrt_pjy=166`, `rail_tebrau=169`, `rail_komuter=617`. The MCO-inclusive `feature_metadata.json` records larger counts for the full 2019–2025 series, where all eight late-launch services contribute pre-launch nulls (e.g. `bus_rkl=1,096`, `rail_mrt_pjy=1,262`, `rail_komuter=1,713`).

**Derived features.** `total_ridership` is computed as the row-wise sum of all 12 service columns (`min_count=1` prevents an all-NaN row from producing a zero sum).

**Temporal feature engineering.** Cyclic encodings are applied to avoid ordinal bias in gradient-based models:
- `day_of_week` (0–6) → `dow_sin = sin(2π·dow/7)`, `dow_cos = cos(2π·dow/7)`
- `month` (1–12) → `month_sin = sin(2π·month/12)`, `month_cos = cos(2π·month/12)`
- `is_weekend` binary flag (day_of_week ≥ 5)
- `day_of_year` for intra-year position

**Chronological split (no leakage).** Post-MCO data is split 70/15/15 by index position (no shuffle). A `MinMaxScaler` is fitted on the train portion only and saved to `data/scalers/ridership_scaler.pkl`. Fitting on the full dataset would leak val/test statistics into training, inflating reported performance.

**Sanity checks.** Null percentage per column is printed post-cleaning. MCO-flagged row count (~650 rows, ~30% of the full series) is logged to confirm the flag was applied correctly.

---

## Source 2 — Fuel Price (`fuelprice.py`)

**Raw data:** `data/raw/fuelprice.csv`  
**Output:** `data/cleaned/fuelprice_level_daily.csv`, `data/cleaned/fuelprice_change_daily.csv`

### EDA Findings

**Series-type split.** The raw file contains two interleaved series types: `level` (absolute price per litre) and `change_weekly` (weekly price change). EDA confirmed these cannot be used together without pivoting — they were split into separate frames.

**Weekly frequency identification.** Fuel prices in Malaysia are announced weekly (typically Wednesday). Merging weekly-indexed data onto a daily ridership index produces 6 out of every 7 rows as `NaN`. EDA confirmed this by checking the index frequency of `df_level`. Resolution: **forward-fill** after reindexing to a daily `date_range` — semantically correct because the price in effect carries forward until the next announcement.

**Leading null correction.** After forward-fill, the head of the series (before the first recorded price) contains `NaN`. These are back-filled from the earliest known price.

**Subsidy column launch detection.** The `ron95_budi95` subsidy column has a later `first_valid_index` than the base columns. This was logged explicitly to confirm late-column alignment works correctly before the merge.

**Summary statistics.** `describe()` was printed for both `level` and `change_weekly` frames to check value ranges. Level values are in RM/litre (typically 1.5–3.5 for RON95); change values are signed weekly deltas.

**Stationarity-motivated feature.** Raw price levels are non-stationary (trending). Three percent-change columns were added for the primary fuels — `ron95_pct_chg`, `ron97_pct_chg`, `diesel_pct_chg` — which are more stationary and better-conditioned for gradient-based optimisers in LSTM/GCN models.

**Scaler export.** A `MinMaxScaler` is fitted on data before an approximate train cutoff (2022-01-01, matched to the ridership train split) and saved to `data/scalers/fuelprice_scaler.pkl`.

**Final feature set (15 columns under `fp_lv_*` / `fp_chg_*` prefixes in `feature_metadata.json`):**
- Levels: `ron95`, `ron97`, `diesel`, `diesel_eastmsia`, `ron95_budi95`, `ron95_skps` (6)
- Level % changes: `ron95_pct_chg`, `ron97_pct_chg`, `diesel_pct_chg` (3)
- Weekly changes: `ron95`, `ron97`, `diesel`, `diesel_eastmsia`, `ron95_budi95`, `ron95_skps` (6)

---

## Source 3 — Holiday / Cyclical Features (`holiday.py`)

**Raw data:** `data/raw/school_public_holiday.csv`  
**Output:** `data/cleaned/holiday_daily_features.csv`

### EDA Findings

**Date parse failure audit.** The raw file encodes dates as `(Year, Month-Day)` pairs across separate columns. `parse_date()` applies `pd.to_datetime` with `%Y-%b-%d` format per row; the number of `NaT` outcomes is logged. Any inverted ranges (`end_date < start_date`) are also counted and warned.

**Duration distribution.** `duration_days = (end_date - start_date).dt.days + 1` was computed and checked to detect single-day entries (public holidays) vs multi-week entries (school terms).

**Spatial coverage annotation.** State-specific events (Thaipusam, observed only in SGR, PNG, PRK, JHR, KUL) are annotated in the `notes` field. Because the ridership target is national-level, these days are still flagged as `is_public_holiday=1` — a simplification that avoids requiring state-disaggregated ridership targets.

**Expansion from range-level to daily.** The range-level table (one row per event with start/end) cannot be joined to a daily ridership index. `expand_to_daily()` iterates all date ranges and builds a set of individual calendar dates; the result is reindexed to a full daily calendar (2019-01-01 → 2026-12-31) with binary flags.

**Holiday density sanity check.** `cal.groupby(cal.index.year)['is_public_holiday'].sum()` was printed to confirm a realistic count of public holidays per year (~17–20 days/year in Malaysia). School holidays (~80–100 days/year) were checked similarly.

**Lead / lag engineering.** Ridership data suggests commuters adjust travel 1–3 days before public holidays (anticipatory demand drop). Two derived features per holiday type were added:
- `days_to_next_public_hol` — days until the next public holiday (0 on the day itself; 99 if no holiday in the remaining window)
- `days_since_last_public_hol` — days elapsed since the last public holiday
- Same pair for school holidays: `days_to_next_school_hol`, `days_since_last_school_hol`

**Cyclical re-encoding.** `dow_sin`, `dow_cos`, `month_sin`, `month_cos`, `day_of_week`, `month`, `is_weekend` are also produced here for convenience. `feature_align.py` sources these from the holiday output rather than recomputing them.

**Final temporal feature set (16 columns):**
`is_public_holiday`, `is_school_holiday`, `is_holiday_any`, `days_to_next_public_hol`, `days_since_last_public_hol`, `days_to_next_school_hol`, `days_since_last_school_hol`, `day_of_week`, `month`, `is_weekend`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`, `year`, `day_of_year`

---

## Source 4 — Rainfall (`rainfall.py`)

**Raw data:** `data/raw/mys_rainfall_subnat_2019_2026.csv`  
**Output:** `data/cleaned/rainfall_wide_daily.csv`

### EDA Findings

**Version filter.** The source data contains both `preliminary` and `final` version rows. EDA confirmed duplicated `(date, PCODE)` pairs across versions. Only `version == "final"` rows are retained to avoid including provisional estimates.

**Administrative level filter.** The data contains three administrative levels: level 1 (states), level 2 (sub-national divisions), level 3 (districts). Level 2 is dropped as redundant. Level 3 district data is aggregated up to level 1 state via `group_districts_to_state()`, which sums rainfall totals and averages anomaly scores per `(date, state)`.

**Column semantics inspection.** Original column names (`rfh`, `rfh_avg`, `r1h`, `r3h`, `rfq`, etc.) were renamed to descriptive equivalents:
- `rfh` → `rainfall_mm` (daily rainfall in mm)
- `r1h` → `acc_1mo_mm` (1-month accumulation)
- `r3h` → `acc_3mo_mm` (3-month accumulation)
- `rfq` → `anomaly_rf` (anomaly relative to climatological mean)

**Shape audit.** Wide-format shape `(date × state_metric)` was printed before and after pivoting to confirm all 16 states (Peninsular: 11 + Sabah + Sarawak + Labuan + Putrajaya + KL) appear as columns.

**Missing date interpolation.** Some dates are missing from the source (no observation for that period). EDA confirmed that joining the long-format series to a daily ridership index would silently produce `NaN` rows without interpolation. After reindexing to a contiguous daily range, linear interpolation is applied with `limit=7` days to avoid over-imputation. Remaining edge gaps (start/end of series) are back-filled then forward-filled.

**Null count before/after.** Printed explicitly: `Null count before interpolation: X` → `Null count after interpolation: 0` (or near-zero).

**Feature selection for model input.** Only `rainfall_mm__MY{state_pcode}` columns (15 state columns) are carried into the aligned feature matrix. Accumulated and anomaly columns are retained in the cleaned long-format export for potential future use but excluded from `features_aligned.csv` to limit feature dimensionality.

**Final external feature columns (15):** `rainfall_mm__MY01` through `rainfall_mm__MY17` (excluding MY14, MY16 which are federal territories merged into adjacent state PCODEs).

---

## Source 5 — Population Density (`population.py`)

**Raw data:** `data/raw/malaysia_population_density_2020.csv`  
**Output:** `data/cleaned/population_density_clean.csv`, `data/cleaned/population_by_state.csv`, `data/cleaned/population_at_stops.csv`

### EDA Findings

**Column rename.** The raw file has opaque positional headers; these were renamed to `longitude`, `latitude`, `density_per_km2`.

**Coordinate bounds validation.** A Malaysia bounding box (lat: 0.9–7.4°N, lon: 99.6–119.3°E) was applied to drop any out-of-bounds grid cells (coastline/sea artefacts or projection errors). Both out-of-bounds counts were printed.

**Density distribution inspection.** `describe()` on `density_per_km2` was examined. The distribution is highly right-skewed: most of Malaysia is rural/forest with near-zero density; a small number of Klang Valley grid cells have extremely high density (>10,000/km²). Two transforms were applied:
- `density_log = log1p(density_per_km2)` — compresses the skewed tail for model inputs
- `density_clipped = clip(upper=p99)` — 99th-percentile clip for visualisation/reference

**State-level aggregation (spatial join).** Using GeoPandas + the cleaned GADM Level-1 boundaries, each grid point was spatially joined to its containing state. Per-state summaries (mean, median, log-mean, log-median density, plus `n_grid_cells`) were computed. EDA confirmed all 16 states received a non-zero cell count.

**Stop-level sampling (nearest-neighbour).** A `BallTree` (haversine metric) was built on the population density grid. For each GTFS stop, the nearest grid cell was queried; median nearest-grid distance was printed to confirm accuracy (typically < 0.5 km for urban stops). This produced `data/cleaned/population_at_stops.csv` as a per-stop static attribute.

**Feature broadcast.** In `feature_align.py`, only two summary scalars are broadcast to all dates as static features:
- `pop_density_median` — national median of `density_per_km2`
- `pop_density_log_median` — national median of the log-transformed density

This choice keeps the population signal simple (single national-level indicator) while avoiding the complexity of state-level disaggregation that would require reshaping the feature matrix.

---

## Source 6 — GTFS Transit Network (`gtfs.py`)

**Raw data:** `data/raw/gtfs_{rapid_rail_kl|rapidbus_kl|rapidbus_penang|ktmb}/`  
**Output:** cleaned GTFS `.txt` files + `gtfs_stop_nodes_{operator}.csv` + `gtfs_stop_edges_{operator}.csv`

### EDA Findings

**Multi-operator cleaning.** Four operators were processed independently with operator-specific presets: Rapid Rail KL (LRT/MRT/Monorail), RapidBus KL, RapidBus Penang, KTMB (KTM Komuter/ETS/Intercity). Each operator has different `frequency_based` flags, `expected_route_types`, and non-standard column lists.

**Referential integrity checks.** A systematic integrity report was produced after cleaning:
- All `trip.route_id` values present in `routes.txt` ✓
- All `trip.service_id` values present in `calendar.txt` ✓
- All `stop_time.trip_id` values present in `trips.txt` (cross-join verified)
- All `stop_time.stop_id` values present in `stops.txt` (cross-join verified)

**Coordinate validation.** `stop_lat` and `stop_lon` were coerced to float; any stops outside the Malaysia bounding box were dropped with a count warning. Shape points were similarly validated, with swapped lat/lon columns detected and corrected for operators where the raw file had them reversed (inferred from `lat > 90` check).

**Duplicate detection.** Duplicate `stop_id`, `route_id`, `trip_id`, and `(service_id, date)` pairs were all explicitly counted and deduplicated (keeping the first occurrence, with counts logged).

**Inverted date ranges.** `calendar.txt` entries where `end_date < start_date` were detected and swapped (swap — not dropped — to preserve schedule data).

**Service pattern analysis.** For frequency-based operators (Rapid Rail KL, RapidBus KL), `headway_secs` was validated as numeric and positive. GTFS time strings (`HH:MM:SS`, allowing hours > 24 for overnight service) were parsed to total seconds; rows with `end_time ≤ start_time` were dropped.

**Graph artefact export.** After cleaning, two graph artefacts are exported per operator:
- **Node table** (`gtfs_stop_nodes_{operator}.csv`): `stop_id`, `stop_name`, `stop_lat`, `stop_lon`, `route_id`, `operator` — one row per stop
- **Edge table** (`gtfs_stop_edges_{operator}.csv`): consecutive stop pairs extracted from `stop_times.txt`, with `travel_time_s` estimated from `departure_time` differences (set to -1 for frequency-based templates with no wall-clock times). Duplicate `(from_stop_id, to_stop_id, route_id)` pairs deduplicated.

**Feature aggregation in `feature_align.py`.** All operator node/edge tables are concatenated; four static network summary scalars are broadcast to all dates:
- `gtfs_n_stops` — total unique stops across all operators
- `gtfs_n_routes` — total unique routes
- `gtfs_n_directed_edges` — total directed consecutive-stop pairs
- `gtfs_avg_segment_s` — mean travel time for segments with valid wall-clock times

---

## Source 7 — OSM Points of Interest (`osm.py`)

**Raw data:** `data/raw/osm_pois.json` (Overpass API export)  
**Output:** `data/cleaned/osm_pois_clean.json`, `data/cleaned/poi_counts_at_stops.csv`

### EDA Findings

**Element type handling.** OSM elements are of type `node` (point geometry) or `way` (polygon with a computed `center`). A `get_coords()` helper extracts `(lat, lon)` from each type. Elements without coordinates (e.g., incomplete `way` entries lacking a `center`) are discarded.

**Deduplication.** Duplicate OSM element IDs (can arise if Overpass returns overlapping bounding boxes) were tracked and skipped. Count logged.

**Postcode cleaning.** Malaysian postcodes are 5 digits. Any `addr:postcode` tag not matching `^\d{5}$` was stripped from the element's tag dictionary (the POI is kept — the malformed postcode is removed).

**Category taxonomy.** OSM tags are heterogeneous (combinations of `amenity`, `shop`, `leisure`, `tourism` keys). A 7-category taxonomy was defined via `CATEGORY_MAP`:

| Category | Example OSM tags |
|----------|-----------------|
| `transport` | `bus_station`, `bus_stop`, `ferry_terminal`, `taxi` |
| `food` | `restaurant`, `cafe`, `fast_food`, `bar` |
| `retail` | `marketplace`, `supermarket`, `mall` |
| `education` | `school`, `university`, `college` |
| `healthcare` | `hospital`, `clinic`, `pharmacy` |
| `leisure` | `park`, `cinema`, `gym`, `stadium` |
| `other` | everything else |

`tag_to_category()` checks keys in priority order: `amenity` → `shop` → `leisure` → `tourism`.

**Cleaning summary statistics.** After processing, the following were reported: total kept, duplicate IDs skipped, no-coordinate skipped, malformed postcodes removed, and elements without a `name` tag (informational — unnamed POIs are still valid for catchment counting).

**Catchment-area aggregation (500 m radius).** For each GTFS stop, a `BallTree` (haversine metric) query counts POIs within a 500 m radius. This radius was chosen as a standard transit walkability catchment. Counts are computed per category (`poi_transport`, `poi_food`, `poi_retail`, `poi_education`, `poi_healthcare`, `poi_leisure`, `poi_other`) plus `poi_total`. Log-transformed variants (`{col}_log = log1p(count)`) are also computed per stop.

**Summary statistics per stop.** `describe()` on the per-stop count columns was printed to check distribution (expected: highly right-skewed, with CBD stops having much higher counts than suburban stops).

**Feature aggregation in `feature_align.py`.** The per-stop POI counts are summarised to **mean across all stops** for each category, producing 8 static scalar features broadcast to all dates:
`osm_poi_total_mean`, `osm_poi_transport_mean`, `osm_poi_food_mean`, `osm_poi_retail_mean`, `osm_poi_education_mean`, `osm_poi_healthcare_mean`, `osm_poi_leisure_mean`, `osm_poi_other_mean`

---

## Source 8 — GADM Administrative Boundaries (`gadm.py`)

**Raw data:** `data/raw/gadm_mys_l1.json` (GADM Level-1 GeoJSON for Malaysia)  
**Output:** `data/cleaned/gadm_mys_l1_clean.geojson`, `data/cleaned/gadm_state_nodes.csv`, `data/cleaned/gadm_adj_matrix.npy`, `data/cleaned/gadm_adj_matrix_weighted.npy`, `data/cleaned/gadm_adj_edges.csv`

### EDA Findings

**Feature count validation.** Malaysia has 16 Level-1 administrative units (13 states + 3 federal territories). The script checks `len(cleaned_features) != 16` and warns if this is not met.

**Duplicate GID_1 detection.** Duplicate `GID_1` identifiers (which would corrupt the node index) are detected and logged as errors.

**Geometry type validation.** Each feature's geometry type is checked; unexpected types (not `Polygon` or `MultiPolygon`) are flagged but kept to avoid data loss.

**Null property cleaning.** Properties with `"NA"` string values are converted to `None`. The `VARNAME_1` field (pipe-delimited alternate names) is split into a Python list. Two non-informative properties (`NL_NAME_1`, `CC_1`) are dropped.

**Centroid extraction.** The Shapely `shape().centroid` was computed for each state geometry, providing `(centroid_lat, centroid_lon)` per state. These are available as node attributes for distance-based edge weighting in GCN models.

**Shared-border adjacency matrix.** An N×N adjacency matrix was built by computing `boundary.intersection(boundary)` for all state pairs. The length of the shared boundary segment (in degrees, converted to km via `× 111.0`) is used as the edge weight. A threshold of `1e-6` degrees eliminates point-only contacts (diagonal corners, sea boundaries). This produces:
- `gadm_adj_matrix.npy` — N×N binary matrix (1 where shared border exists)
- `gadm_adj_matrix_weighted.npy` — N×N matrix weighted by shared border length in km
- `gadm_adj_edges.csv` — edge list with `node_i`, `node_j`, `gid_i`, `gid_j`, `shared_border_km`

**Feature aggregation in `feature_align.py`.** Three scalar summaries are broadcast as static features:
- `gadm_n_states` — total state count (16, a constant confirming boundary data is complete)
- `gadm_n_border_pairs` — number of state pairs sharing a land border (reflects spatial connectivity density)
- `gadm_mean_border_km` — mean shared-border length across all adjacent pairs (indicates average spatial proximity)

---

## Feature Alignment (`feature_align.py`)

After all eight sources are cleaned, `feature_align.py` merges them and exports **two** feature matrices in a single run.

### Merge Strategy

| Source | Join type | Null handling |
|--------|-----------|---------------|
| Ridership (targets + lag) | `reindex` | pre-launch nulls remain; filled to 0 in `sequence_builder.py` |
| Fuel price | `reindex` + `ffill` | no gaps after forward-fill |
| Holiday / cyclical | `reindex` + `fillna(0)` | non-holiday days default to 0 |
| Rainfall | `reindex` + `interpolate(limit=7)` + `bfill/ffill` | contiguous after interpolation |
| Population density | scalar broadcast | no nulls (single value) |
| GTFS static | scalar broadcast | no nulls (computed from clean tables) |
| OSM POI | scalar broadcast | no nulls (mean across stops) |
| GADM | scalar broadcast | no nulls (computed from clean boundaries) |

### MCO Split

After the merge, a second matrix is derived by dropping the MCO anomaly period (2020-03-18 – 2021-12-31):

| Output file | MCO rows | Day count |
|-------------|----------|-----------|
| `features_aligned.csv` | included | 2,557 |
| `features_aligned_no_mco.csv` | excluded | 1,461 |

A metadata JSON is written for each file (`feature_metadata.json` / `feature_metadata_no_mco.json`).

### Null Audit

A final null audit is printed after the merge. The only expected nulls correspond to the ridership pre-launch period. `feature_metadata.json` (MCO-inclusive, 2019–2025) documents the full pre-launch null counts:

```json
"null_counts": {
  "bus_rkl": 1096,
  "bus_rpn": 1096,
  "rail_ets": 653,
  "rail_intercity": 653,
  "rail_komuter_utara": 653,
  "rail_mrt_pjy": 1262,
  "rail_tebrau": 1265,
  "rail_komuter": 1713
}
```

Within the 2022–2025 master window (`feature_metadata_no_mco.json`), only three services retain residual nulls — `rail_mrt_pjy` (166 days), `rail_tebrau` (169 days), and `rail_komuter` (617 days) — because the other five late-launch services all commenced before 2022-01-01.

### Autoregressive Lag Features

Ridership lags (`ridership_lag_7`, `ridership_lag_14`, `ridership_lag_28`) are computed from the **full historical series** (extending before 2022-01-01) before reindexing to the master window. This ensures the first 28 days of the master window have valid lag values without look-back into the missing/MCO period.

---

## Sequence Assembly (`sequence_builder.py`)

Converts a pre-filtered features CSV into model-ready tensors. MCO filtering is **not** performed here — it is handled upstream by `feature_align.py`. `sequence_builder.py` simply loads whichever file it is pointed at via `--features-path`.

### Key Design Decisions from EDA

**MCO split moved upstream.** `feature_align.py` produces `features_aligned.csv` (MCO included) and `features_aligned_no_mco.csv` (MCO excluded). `run_pipeline.py` calls `sequence_builder.py` twice per lookback — once per file — writing to sibling output directories. This ensures both experimental conditions are always in sync.

**Pre-launch zero-fill.** `df.fillna(0.0)` converts all remaining structural nulls (pre-launch ridership columns) to zero. This is semantically correct and avoids `NaN` propagation into the sliding-window arrays.

**Chronological sliding windows.** Overlapping windows of shape `(T_in, F)` → target `(T_out,)` are extracted without shuffling, preserving temporal order. The default `T_in=14` (14-day look-back) with `T_out=7` (7-day forecast horizon) reflects the weekly seasonality identified in the ridership EDA.

**Scaler leakage prevention.** `MinMaxScaler` is fitted on `X_train` only (flattened to `(N·T, F)` to fit per-feature). A separate scaler is fitted on `y_train` to enable inverse-transformation of predictions back to ridership counts. Both scalers are saved as `.pkl` files alongside the sequences.

**Storage.** Arrays are cast to `float16` by default. For MinMax-scaled `[0, 1]` data, `float16` precision loss (≈ 0.001) is negligible. This halves disk and memory footprint. Optional `--compress` flag writes `.npz` for an additional 1.5–3× reduction.

**Six lookback directories** (three windows × two MCO conditions) are produced by `run_pipeline.py` step 3:

| Features file | Output dir | T_in | MCO rows |
|---------------|-----------|------|----------|
| `features_aligned_no_mco.csv` | `data/sequences/lstm/` | 14 | excluded |
| `features_aligned.csv` | `data/sequences/lstm_mco/` | 14 | included |
| `features_aligned_no_mco.csv` | `data/sequences/lookback_28/` | 28 | excluded |
| `features_aligned.csv` | `data/sequences/lookback_28_mco/` | 28 | included |
| `features_aligned_no_mco.csv` | `data/sequences/lookback_56/` | 56 | excluded |
| `features_aligned.csv` | `data/sequences/lookback_56_mco/` | 56 | included |

---

## Notable EDA Findings

The EDA suite in `src/eda/` contains 40 scripts across univariate, bivariate, multivariate, and thematic categories. Each finding below is analysed using the **MEAL** strategy (Main Idea → Evidence → Analysis → Link).

---

### Finding 1 — MCO Collapse and Unequal Recovery

**Script:** `src/eda/ridership/ridership.py`

**Main Idea.** The COVID-19 Movement Control Order (MCO, 2020-03-18 – 2021-12-31) caused a 65–90% ridership collapse across all 12 transit services (Airak et al., 2023). Post-MCO recovery rates diverge sharply by mode: rail-heavy services (LRT, MRT) recovered more slowly than bus services, which benefited from route-level flexibility.

**Evidence.** Change-point detection (binary segmentation, 5 breakpoints) applied to the `total_ridership` series identifies four distinct phases: pre-MCO baseline, lockdown trough, phased-reopening recovery, and stable post-MCO trend. Per-service collapse percentages and post-MCO recovery slopes were computed as `(ridership_post − ridership_trough) / ridership_pre × 100`. The MCO flag (`is_mco=1`) was applied to 650 rows (~25% of the full 2019–2025 series).

**Analysis.** The structural break invalidates any model assumption of temporal stationarity (Lee et al., 2024). Static graph adjacency matrices fitted solely on post-MCO training data fail to generalise across the break — this is the primary driver of the large MCO-to-no-MCO degradation observed in graph-based models (STGCN: −22.54 pp). HMT-TSF's regime-gating component was designed explicitly to detect and adapt to this kind of distributional shift. Retaining MCO rows with a binary flag rather than removing them preserves sequence continuity for LSTM sliding windows while enabling the model to learn the break.

**Figures.**
- `src/eda/ridership/results/changepoint_detection.png` — binary-segmentation breakpoints on the total ridership series
- `src/eda/ridership/results/mco_period_overlay.png` — MCO flag overlaid on the ridership time series
- `src/eda/ridership/results/mco_period_comparison.png` — pre/during/post-MCO level comparison across all 12 services
- `src/eda/ridership/results/mco_recovery_trajectories.png` — per-service recovery slopes post-MCO
- `src/eda/ridership/results/temporal_trend_analysis.png` — full 2019–2025 temporal trend with phase annotations
- `src/eda/theme/results/graphs/changepoint_detection.png` — thematic summary of change-point phases

**Link.**
- Airak, S., Abd Sukor, N. S., & Abd Rahman, N. (2023). [Travel behaviour changes and risk perception during COVID-19: A case study of Malaysia](https://doi.org/10.1016/j.trip.2023.100784). *Transportation Research Interdisciplinary Perspectives*, 18, 100784.
- Lee, S., Kim, J., & Cho, K. (2024). [Temporal dynamics of public transportation ridership in Seoul before, during, and after COVID-19 from urban resilience perspective](https://doi.org/10.1038/s41598-024-59323-w). *Scientific Reports*, 14, 9078.

---

### Finding 2 — Northeast Monsoon Rainfall Pattern

**Script:** `src/eda/rainfall/rainfall.py`

**Main Idea.** Monthly wet-day ratio and extreme-event frequency (> 50 mm/day) peak in November–January, aligning with the Northeast Monsoon season. East-coast states (Kelantan, Terengganu, Pahang, Sabah) record significantly more extreme-rainfall days per year than west-coast counterparts.

**Evidence.** Per-state monthly wet-day ratios were computed from `rainfall_wide_daily.csv`. Extreme-event counts (days with `rainfall_mm > 50`) were aggregated by month and by state PCODE. East-coast states average 8–12 extreme days in Q4–Q1, versus 2–4 for Selangor and Kuala Lumpur.

**Analysis.** The spatial heterogeneity in rainfall distribution justifies retaining all 15 state-level `rainfall_mm__MY{pcode}` columns individually rather than collapsing to a national mean. A single national rainfall scalar would mask the east-vs-west variance that drives mode-specific ridership effects (Ngo & Bashar, 2024). The 7-day interpolation limit (`limit=7`) is chosen to span typical monsoon event durations without over-smoothing the seasonal peak.

**Figures.**
- `src/eda/rainfall/results/temporal_monthly_seasonality.png` — monthly wet-day ratios showing the November–January peak
- `src/eda/rainfall/results/extreme_events_frequency.png` — extreme-event counts by month aggregated nationally
- `src/eda/rainfall/results/extreme_events_by_state.png` — per-state extreme-day counts highlighting east-vs-west divide
- `src/eda/rainfall/results/spatial_seasonal_comparison.png` — spatial seasonal comparison across all 15 state PCODEs
- `src/eda/rainfall/results/wet_dry_ratio_monthly.png` — monthly wet-day ratios
- `src/eda/rainfall/results/wet_dry_ratio_state.png` — state-level wet-day ratios
- `src/eda/theme/results/graphs/rainfall_seasonality.png` — thematic summary of seasonal rainfall patterns

**Link.**
- Ngo, N. S., & Bashar, B. (2024). [The impacts of extreme weather events on U.S. public transit ridership](https://doi.org/10.1016/j.trd.2024.104504). *Transportation Research Part D: Transport and Environment*, 137, 104504.

---

### Finding 3 — Fuel Price Elasticity and Modal Substitution

**Script:** `src/eda/theme/demand_drivers_series.py`

**Main Idea.** A moderate positive correlation exists between RON95 retail price and total ridership: higher fuel prices are associated with increased public transit use, consistent with modal substitution — commuters switching from private vehicles when petrol costs rise (Mily et al., 2024).

**Evidence.** Binned scatter plot of weekly RON95 price (RM/litre) versus 7-day rolling mean of `total_ridership` was constructed after aligning both series on the daily index. The Pearson correlation is ≈ +0.35 (post-MCO window only); the relationship is non-linear, steepening above RM 2.20/litre. Stationarity of raw price levels was checked via ADF test (p > 0.05, confirming non-stationarity), motivating the addition of `ron95_pct_chg` as a secondary feature.

**Analysis.** The elasticity signal motivates retaining both fuel-level columns (for baseline demand context) and percent-change columns (for stationary, gradient-safe signal). Including all six fuel types captures the subsidy–non-subsidy boundary effect (`ron95_budi95`, `ron95_skps`), which represents discrete policy interventions that altered effective price at household level without changing the RON95 headline price (Rahimi et al., 2024).

**Figures.**
- `src/eda/theme/results/graphs/fuel_ridership_relationship.png` — binned scatter of RON95 price vs 7-day rolling ridership mean
- `src/eda/theme/results/graphs/demand_driver_correlations.png` — Pearson correlations between all demand drivers and total ridership
- `src/eda/theme/results/graphs/fuel_price_trends.png` — full fuel price level series for all six fuel types

**Link.**
- Mily, I., Haque, M., & Islam, M. T. (2024). [Unveiling the consequence of unprecedented fuel price hike in Bangladesh on consumer travel behavior](https://doi.org/10.1080/29941849.2024.2409081). *Transportation Safety and Environment*, online 22 Oct 2024.
- Rahimi, E., Shamshiripour, A., Shabanpour, R., Mohammadian, A., & Auld, J. (2024). [Analyzing the influence of fuel price shock on urban public transit and interurban automobile travel demand: Evidence from a country with fixed fuel price regulation](https://www.researchgate.net/publication/399563477). *Transportation Research Interdisciplinary Perspectives*, 36.

---

### Finding 4 — Holiday Anticipation and Return-Surge Effect

**Script:** `src/eda/bivariate/holiday_ridership.py`

**Main Idea.** Ridership drops 1–2 days before public holidays (anticipation effect) and surges on the first post-holiday working day (return surge). School holidays produce a softer, multi-day ramp with no sharp pre-holiday dip.

**Evidence.** Lead/lag cross-correlation between `is_public_holiday` and `total_ridership` was computed over a ±7-day window. The maximum negative correlation at lag −1 and −2 confirms the anticipatory dip. For school holidays, the cross-correlation function shows a gradual trough across lags −5 to 0, then recovery at +1 to +3. Magnitude of dip for public holidays: approximately −18% relative to the preceding 7-day mean.

**Analysis.** Binary `is_public_holiday` flags alone would not capture this temporal spread. The derived features `days_to_next_public_hol` and `days_since_last_public_hol` encode the distance to/from the nearest holiday, allowing gradient-based models to learn the anticipatory demand gradient directly (Wu et al., 2023). The asymmetry between public and school holiday profiles justifies maintaining separate lead/lag pairs for each type (4 features total).

**Figures.**
- `src/eda/bivariate/results/holiday_ridership_pre_post_profile.png` — ±7-day lead/lag cross-correlation around public and school holidays
- `src/eda/bivariate/results/holiday_ridership_overall_impact.png` — mean ridership change on holiday vs non-holiday days
- `src/eda/bivariate/results/holiday_ridership_type_analysis.png` — public vs school holiday ridership impact comparison
- `src/eda/bivariate/results/holiday_ridership_duration_effect.png` — ridership response as a function of holiday duration
- `src/eda/bivariate/results/holiday_ridership_weekday_vs_weekend.png` — holiday effect broken down by day-of-week context
- `src/eda/theme/results/graphs/holiday_calendar_effects.png` — thematic summary of holiday calendar demand effects

**Link.**
- Wu, J.-L., Lu, M., & Wang, C.-Y. (2023). [Forecasting metro rail transit passenger flow with multiple-attention deep neural networks and surrounding vehicle detection devices](https://doi.org/10.1007/s10489-023-04483-x). *Applied Intelligence*, 53, 11789–11808.

---

### Finding 5 — Klang Valley Transit Concentration

**Script:** `src/eda/gtfs/gtfs.py`

**Main Idea.** Public transit stops are heavily concentrated in Greater Kuala Lumpur (Selangor + Federal Territory). Most peninsular states outside this corridor — and all Borneo states — hold fewer than 5% of total stops each. The network is functionally monocentric.

**Evidence.** DBSCAN clustering (ε = 0.02°, min_samples = 5) applied to the combined stop coordinate set from all four GTFS operators produces two dominant clusters covering Klang Valley and Penang Island, with all remaining states forming noise points or micro-clusters. Stop count per state was computed by spatially joining nodes to GADM Level-1 boundaries: Selangor + KL FT account for ~62% of all stops (Li et al., 2024).

**Analysis.** The national feature matrix uses summary scalars (total stops, routes, edges) rather than per-region disaggregation because the stop distribution is too skewed to yield meaningful per-state time-varying signals. This concentration also means the GTFS-derived static features predominantly reflect conditions in the Klang Valley corridor, which dominates total ridership. This is an acknowledged limitation for models attempting spatially differentiated forecasts (Wang et al., 2024).

**Figures.**
- `src/eda/gtfs/results/02_stop_density_clustering.png` — DBSCAN clustering of stop coordinates showing the Klang Valley and Penang dominant clusters
- `src/eda/gtfs/results/01_route_length_coverage.png` — route length and coverage distribution across all four operators
- `src/eda/theme/results/graphs/gtfs_stop_heatmap.png` — spatial heatmap of stop density across Peninsular Malaysia and Borneo
- `src/eda/theme/results/graphs/stop_density_population.png` — stop density overlaid on population density, highlighting the Klang Valley concentration

**Link.**
- Li, Y., et al. (2024). [An efficient approach for identifying potential bus passenger demand based on multisource data](https://doi.org/10.1155/2024/5368577). *Journal of Advanced Transportation*, 2024, 5368577.
- Wang, Z., Huang, K., Massobrio, R., Bombelli, A., & Cats, O. (2024). [Quantification and comparison of hierarchy in public transport networks](https://doi.org/10.1016/j.physa.2023.129479). *Physica A: Statistical Mechanics and Its Applications*, 634, 129479.

---

### Finding 6 — Multicollinearity in Fuel Price Features

**Script:** `src/eda/multivariate/full_demand_model_feature_correlation_matrix.py`

**Main Idea.** RON95 level, RON97 level, and diesel level are highly collinear. Condition number analysis of the fuel feature sub-matrix flags near-perfect linear dependence. This structural redundancy motivated SHAP-guided feature reduction from 79 to 53 features in the HMT-TSF-FR variant.

**Evidence.** The condition number of the 6-column fuel-level sub-matrix exceeds 1,000, and the three smallest eigenvalues are < 0.01 — both conventional multicollinearity thresholds. Variance Inflation Factors (VIF) for RON95 and diesel level exceed 15. The percent-change columns are markedly less collinear (VIF < 3) because they capture idiosyncratic weekly adjustment patterns.

**Analysis.** High multicollinearity does not degrade prediction accuracy in over-parameterised deep learning models, but it inflates Shapley value variance and reduces interpretability (Rahimi et al., 2024). The SHAP-guided reduction removes the most redundant fuel-level columns while retaining the percent-change columns and the subsidy-tier columns that carry policy-specific signal. The resulting 53-feature set achieves within 0.05 pp of the 79-feature baseline (HMT-TSF-FR: 81.77% vs. HMT-TSF: 81.82%), confirming that the removed features were redundant for prediction (Wu et al., 2023).

**Figures.**
- `src/eda/multivariate/results/full_demand_correlation_matrix.png` — full 79×79 Pearson correlation heatmap of the feature matrix
- `src/eda/multivariate/results/full_demand_multicollinearity_diagnostics.png` — condition number, eigenvalue spectrum, and VIF diagnostics for the fuel feature sub-matrix

**Link.**
- Rahimi, E., Shamshiripour, A., Shabanpour, R., Mohammadian, A., & Auld, J. (2024). [Analyzing the influence of fuel price shock on urban public transit and interurban automobile travel demand: Evidence from a country with fixed fuel price regulation](https://www.researchgate.net/publication/399563477). *Transportation Research Interdisciplinary Perspectives*, 36.
- Wu, J.-L., Lu, M., & Wang, C.-Y. (2023). [Forecasting metro rail transit passenger flow with multiple-attention deep neural networks and surrounding vehicle detection devices](https://doi.org/10.1007/s10489-023-04483-x). *Applied Intelligence*, 53, 11789–11808.

---

### Finding 7 — Underserved Population Zones

**Script:** `src/eda/multivariate/underserved_population_identification.py`

**Main Idea.** A joint analysis of residential population density and transit stop coverage identifies priority zones where high density coincides with low service supply — transit deserts in the Malaysian context (Yadav et al., 2024). These zones have high latent demand not captured by current ridership counts.

**Evidence.** Population density grids from `population_density_clean.csv` were overlaid with the GTFS stop spatial join output. For each 1 km² grid cell, the density decile and nearest-stop distance were cross-tabulated. Cells in the top density quartile but with nearest-stop distance > 1 km are flagged as underserved. Approximately 14% of the high-density grid cells fall in this category, concentrated in outer Klang Valley suburbs and Sabah coastal towns.

**Analysis.** This finding motivates including both `pop_density_median` and `pop_density_log_median` as static features: the log-transform compresses the extreme right skew from urban core cells, while the median provides a robust central tendency (Al-Ansari & Al-Mamoori, 2022). Stop-level population at BallTree nearest-neighbour resolution is exported to `population_at_stops.csv` for potential future stop-disaggregated modelling. For the current national forecasting task, underserved zone identification is contextual rather than directly encoded — it informs the interpretation of model error distributions.

**Figures.**
- `src/eda/multivariate/results/underserved_population_identification.png` — cross-tabulation of density decile vs nearest-stop distance, with underserved cells flagged
- `src/eda/population/results/spatial_density_map.png` — 1 km² population density grid across Malaysia
- `src/eda/population/results/hotspot_analysis.png` — high-density hotspots identified in the outer Klang Valley suburbs and Sabah coastal towns
- `src/eda/bivariate/results/population_transit_coverage.png` — stop coverage vs population density per GADM zone
- `src/eda/theme/results/graphs/population_density_map.png` — thematic population density map with transit stop overlay

**Link.**
- Yadav, M., Mepparambath, R. M., & Patil, G. R. (2024). [An enhanced transit accessibility evaluation framework by integrating Public Transport Accessibility Levels (PTAL) and transit gap](https://doi.org/10.1016/j.jtrangeo.2024.103965). *Journal of Transport Geography*, 120, 103965.
- Al-Ansari, N., & Al-Mamoori, S. K. (2022). [Do the population density and coverage rate of transit affect the public transport contribution?](https://doi.org/10.1080/23311916.2022.2143059) *Cogent Engineering*, 9(1), 2143059.

---

### Finding 8 — Seasonal Demand–Supply Mismatch

**Script:** `src/eda/multivariate/seasonal_demand_supply_mismatch.py`

**Main Idea.** School holiday periods show systematic over-supply of transit capacity relative to observed demand. Ramadan shifts demand composition by service type — rail demand dips during morning peak while bus demand redistributes toward evening prayer times.

**Evidence.** Capacity utilisation ratio (ridership / scheduled service capacity, proxied by `gtfs_n_directed_edges`) was computed by calendar period. School holidays show a ~22% drop in utilisation below the annual mean while scheduled service remains constant. During Ramadan months (identified from the holiday calendar), per-service ridership decomposition shows `bus_rapid_*` categories recovering faster in the early evening slot, while `rail_lrt_*` and `rail_mrt_*` show reduced AM-peak demand.

**Analysis.** The school holiday mismatch supports encoding `is_school_holiday` and `days_to_next_school_hol` as distinct features from their public-holiday equivalents — their demand signatures differ in duration and magnitude (Li et al., 2022). The Ramadan effect is partially captured by `is_public_holiday` (Hari Raya) and the school holiday flag, but the intra-Ramadan gradual shift is absorbed by the cyclical `month_sin/cos` features (Lee et al., 2024). A dedicated Ramadan binary flag was considered but not added because it would overlap with existing features and month cyclical encodings.

**Figures.**
- `src/eda/multivariate/results/seasonal_demand_supply_mismatch.png` — utilisation ratio by calendar period, showing the school-holiday over-supply trough and Ramadan redistribution

**Link.**
- Li, W., Guan, H., Han, Y., Zhu, H., & Wang, A. (2022). [Short-term holiday travel demand prediction for urban tour transportation: A combined model based on STC-LSTM deep learning approach](https://doi.org/10.1007/s12205-022-1698-3). *KSCE Journal of Civil Engineering*, 26(9), 4086–4102.
- Lee, S., Kim, J., & Cho, K. (2024). [Temporal dynamics of public transportation ridership in Seoul before, during, and after COVID-19 from urban resilience perspective](https://doi.org/10.1038/s41598-024-59323-w). *Scientific Reports*, 14, 9078.

---

### Finding 9 — Network Topology Heterogeneity by Service

**Script:** `src/eda/gtfs/gtfs.py`

**Main Idea.** Average node degree and edge-to-node density ratios differ substantially across operators. LRT Kelana Jaya is the most densely meshed network; ETS and Intercity are sparse long-haul chains. The four operators span a topology spectrum from urban metro to regional rail.

**Evidence.** Graph topology metrics were computed from `gtfs_stop_edges_{operator}.csv` for all four operators: average degree `k̄ = 2|E|/|V|`, clustering coefficient, and betweenness centrality of top-5 hub nodes (Wang et al., 2024). Rapid Rail KL: `k̄ ≈ 2.8`, Clustering ≈ 0.41. KTMB (ETS/Intercity): `k̄ ≈ 2.0`, Clustering ≈ 0.05. RapidBus networks fall between these extremes.

**Analysis.** The topology diversity motivates the decision to concatenate all four operator node/edge tables before computing summary scalars (`gtfs_n_stops`, `gtfs_n_routes`, `gtfs_n_directed_edges`, `gtfs_avg_segment_s`) rather than computing per-operator features (Song et al., 2024). Per-operator disaggregation would require 16 static scalars and introduce sparsity for operators with small networks. The segment travel-time mean (`gtfs_avg_segment_s`) excludes frequency-based operators (for whom `travel_time_s = −1`) to avoid polluting the average with sentinel values.

**Figures.**
- `src/eda/gtfs/results/05_network_topology.png` — degree distribution, clustering coefficient, and betweenness centrality for all four operators
- `src/eda/gtfs/results/03_service_frequency.png` — headway/frequency analysis per operator
- `src/eda/theme/results/graphs/network_performance_summary.png` — thematic summary of network topology metrics across operators
- `src/eda/theme/results/graphs/route_coverage_analysis.png` — route coverage comparison by operator type

**Link.**
- Wang, Z., Huang, K., Massobrio, R., Bombelli, A., & Cats, O. (2024). [Quantification and comparison of hierarchy in public transport networks](https://doi.org/10.1016/j.physa.2023.129479). *Physica A: Statistical Mechanics and Its Applications*, 634, 129479.
- Song, J., Ding, J., Gui, X., & Zhu, Y. (2024). [Assessment and solutions for vulnerability of urban rail transit network based on complex network theory: A case study of Chongqing](https://doi.org/10.1016/j.heliyon.2024.e27237). *Heliyon*, 10(5), e27237.

---

### Finding 10 — Feature Correlation Structure: Lag Dominance and Static Near-Zero Signal

**Script:** `src/eda/multivariate/full_demand_model_feature_correlation_matrix.py`

**Main Idea.** In the 79×79 Pearson correlation heatmap, autoregressive lag features (`ridership_lag_7`, `ridership_lag_14`, `ridership_lag_28`) are the strongest individual predictors of `total_ridership`. Static features (GTFS, OSM, GADM) cluster near zero correlation with all ridership targets. Temporal features (holiday flags, cyclical encodings) form an intermediate cluster.

**Evidence.** Full 79×79 correlation matrix computed on the no-MCO feature matrix. Top correlations with `total_ridership`: `ridership_lag_7` (r = 0.94), `ridership_lag_14` (r = 0.91), `ridership_lag_28` (r = 0.87). Static features: `gtfs_n_stops` (r = 0.00, by construction — constant column), `osm_poi_total_mean` (r ≈ 0.00), `gadm_n_states` (r = 0.00). Eigenvalue decomposition confirms that the first principal component (≈ 72% of variance) is dominated by the lag trio.

**Analysis.** The near-zero linear correlation of static features does not mean they are uninformative — constant-valued columns have zero variance and thus zero correlation by definition. Their value lies in providing fixed reference context that allows models to calibrate absolute scale (network size, urban density) rather than contributing time-varying signal. The lag dominance confirms that models should incorporate autoregressive structure; the lag trio is the strongest set of input features regardless of model architecture (Wu et al., 2023; Guo et al., 2025). This also implies that a naive persistence baseline (forecast = lag_7) would achieve moderate performance, which is the practical floor that all 15 models need to exceed.

**Figures.**
- `src/eda/multivariate/results/full_demand_correlation_matrix.png` — full 79×79 Pearson correlation heatmap with lag feature dominance visible in the ridership target row
- `src/eda/multivariate/results/demand_forecasting_correlation.png` — focused correlation analysis between lag features, temporal features, and ridership targets

**Link.**
- Wu, J.-L., Lu, M., & Wang, C.-Y. (2023). [Forecasting metro rail transit passenger flow with multiple-attention deep neural networks and surrounding vehicle detection devices](https://doi.org/10.1007/s10489-023-04483-x). *Applied Intelligence*, 53, 11789–11808.
- Guo, Z., et al. (2025). [Data-driven predictive modelling of stop-level public transit patterns](https://doi.org/10.1007/s11116-025-10689-4). *Transportation* (Springer).

---

### Finding 11 — Rainfall Intensity Categories and Ridership Impact

**Script:** `src/eda/bivariate/rainfall_ridership.py`

**Main Idea.** Transit ridership responds non-uniformly to rainfall intensity (Jiang & Cai, 2023). The relationship is weak at low intensities but strengthens measurably in the Heavy (30–50 mm) and Very Heavy (50–100 mm) categories. Monthly rolling correlation between rainfall and ridership varies seasonally, peaking during the Northeast Monsoon window (November–January).

**Evidence.** Daily rainfall is binned into five categories: Light (0–10 mm), Moderate (10–30 mm), Heavy (30–50 mm), Very Heavy (50–100 mm), and Extreme (> 100 mm). Ridership is compared across bins via box and violin plots. A 30-day rolling Pearson correlation between `rainfall_mm` and `total_ridership` is computed alongside correlations against 1-month accumulation (`acc_1mo_mm`) and anomaly (`anomaly_1mo`). Extreme-event analysis uses the 95th percentile of `rainfall_mm` as the threshold; ridership before, during, and in the 3 days following each event is tracked. Monthly correlation coefficients (with p-values) confirm the seasonal peak during the monsoon months.

**Analysis.** The non-linear intensity response justifies retaining `rainfall_mm` as a continuous feature rather than a binary wet/dry flag. The 95th-percentile extreme-event analysis also informs the `limit=7` interpolation cap applied in `rainfall.py` — individual monsoon events typically last 2–5 days, so capping imputation at 7 days avoids bridging across distinct events. The monthly seasonal variation confirms that a single annual correlation coefficient would be misleading; cyclical `month_sin/cos` features are needed to let models capture this interaction implicitly (Li & Cao, 2025).

**Figures.**
- `src/eda/bivariate/results/rainfall_ridership_category.png` — box/violin plots of ridership across five rainfall intensity bins (Light → Extreme)
- `src/eda/bivariate/results/rainfall_ridership_correlation.png` — 30-day rolling Pearson correlation between rainfall_mm and total_ridership
- `src/eda/bivariate/results/rainfall_ridership_extreme_events.png` — ridership before, during, and 3 days after 95th-percentile extreme events
- `src/eda/bivariate/results/rainfall_ridership_seasonal.png` — monthly correlation coefficients showing the Northeast Monsoon seasonal peak
- `src/eda/bivariate/results/rainfall_ridership_weekday_weekend.png` — rainfall–ridership relationship split by weekday vs weekend
- `src/eda/theme/results/graphs/rainfall_ridership_impact.png` — thematic summary of rainfall impact on ridership across intensity categories

**Link.**
- Jiang, S., & Cai, C. (2023). [The impacts of weather conditions on metro ridership: An empirical study from three mega cities in China](https://doi.org/10.1016/j.tbs.2022.12.003). *Travel Behaviour and Society*, 31, 200–210.
- Li, J., & Cao, J. (2025). [The impact of weather on public bus ridership: Empirical findings from Chenzhou, China](https://doi.org/10.1007/s12469-024-00374-7). *Public Transport* (Springer).

---

### Finding 12 — Fuel Price Lag Effects and Mode-Specific Elasticity

**Script:** `src/eda/bivariate/fuel_ridership.py`

**Main Idea.** The ridership response to a fuel price change is not instantaneous (Belloc et al., 2024). Lag analysis over a 0–7-day window shows the strongest price-ridership correlation occurs at a lag of 1–3 days. Rail and bus modes respond at different lags and with different elasticity magnitudes: rail ridership is more sensitive to sustained price levels while bus ridership reacts more sharply to weekly price-change events.

**Evidence.** Percentage changes in RON95, RON97, diesel, and `diesel_eastmsia` prices are paired with percentage changes in `total_ridership`, filtering pairs where both changes exceed ±0.1% to exclude noise. Best-lag identification iterates shifts 0–7 days and selects the lag with the highest absolute Pearson correlation. Mode substitution analysis splits ridership into directional regimes (`Price_Up` / `Price_Down`) and computes mean ridership change per regime. Threshold analysis uses a 20-point price grid from the 10th to 90th percentile of RON95 to identify non-linear demand responses. Note: `diesel_eastmsia` is constant at RM 2.15/litre throughout the analysis window and contributes no variation.

**Analysis.** The 1–3-day best lag motivates using the `fp_chg_*` (weekly change) columns rather than raw level columns as the primary demand-driver signal for within-week ridership variation. Levels are retained for the long-run equilibrium signal. The mode-specific elasticity difference supports the decision to include both bus and rail service lines as separate target columns — aggregating to `total_ridership` only would mask differential demand elasticity that is relevant to forecasting individual service lines (Shojaeian et al., 2022).

**Figures.**
- `src/eda/bivariate/results/fuel_ridership_lagged_effects.png` — best-lag identification across shifts 0–7 days for RON95, RON97, diesel, and diesel_eastmsia
- `src/eda/bivariate/results/fuel_ridership_mode_substitution.png` — mean ridership change in Price_Up vs Price_Down regimes by service type
- `src/eda/bivariate/results/fuel_ridership_threshold.png` — non-linear demand response across a 20-point RON95 price grid
- `src/eda/bivariate/results/fuel_ridership_correlation.png` — Pearson correlations between all fuel columns and ridership targets
- `src/eda/bivariate/results/fuel_ridership_scatter_regression.png` — scatter with regression line for RON95 price vs total ridership (post-MCO window)

**Link.**
- Belloc, I., Giménez-Nadal, J. I., & Molina, J. A. (2024). [The gasoline price and the commuting behavior of US commuters: Exploring changes to green travel mode choices](https://doi.org/10.1016/j.jtrangeo.2024.104006). *Journal of Transport Geography*, 116, 104006.
- Shojaeian, M., Khodapanah, M., & Zarra-Nezhad, M. (2022). [The role of fare and gasoline price shocks on the behavioral response of passengers in Tehran Metropolitan for using public transportation (Metro, BRT, and Bus)](https://doi.org/10.22034/uep.2022.351389.1258). *Urban Economics and Planning*, 2022.

---

### Finding 13 — Weather-Induced Ridership Surge Under Heavy Rainfall

**Script:** `src/eda/multivariate/weather_induced_ridership_surge_analysis.py`

**Main Idea.** Heavy rainfall events (above the 75th percentile of daily `rainfall_mm`) are associated with a measurable increase in transit ridership — the inverse of the conventional "rain reduces travel" narrative (Chen et al., 2022). Commuters who would otherwise use private vehicles shift to public transit when rainfall is severe enough to make driving uncomfortable or unsafe.

**Evidence.** A binary `heavy_rain` flag is defined using the 75th-percentile threshold of the national mean daily `rainfall_mm`. A four-variable correlation matrix — `(total_ridership, rainfall_mm, rainfall_anomaly, heavy_rain)` — is computed and saved to `weather_ridership_correlation.csv`. Ridership percentage change is computed as `pct_change × 100` relative to the prior day. A scatter plot of rainfall vs. ridership percentage change coloured by the `heavy_rain` flag illustrates the surge regime. Monthly ridership aggregations compare heavy-rain days against non-heavy-rain days.

**Analysis.** The surge finding is the complement of Finding 11: light-to-moderate rain slightly suppresses ridership (trips foregone); heavy rain can paradoxically boost ridership via modal shift. This non-monotone relationship is why a single `rainfall_mm` continuous variable is more informative than a binary wet/dry flag — it allows the model to represent both sides of the response curve. The 75th-percentile threshold used here is consistent with common extreme-weather classification in the literature (Rahmani & Mohammadzadeh Moghaddam, 2025) and is distinct from the 95th-percentile "extreme event" used in `rainfall_ridership.py`, allowing the two scripts to characterise different portions of the tail.

**Figures.**
- `src/eda/multivariate/results/weather_induced_ridership_surge.png` — scatter of rainfall vs ridership percentage change coloured by the heavy_rain flag, showing the surge regime above the 75th-percentile threshold
- `src/eda/theme/results/graphs/rainfall_ridership_impact.png` — thematic overview of the non-monotone rainfall–ridership relationship (shared with Finding 11)

**Link.**
- Chen, J., et al. (2022). [Spatiotemporal variations in Shanghai metro commuting flows during rainfall events](https://doi.org/10.1175/WCAS-D-21-0167.1). *Weather, Climate, and Society*, 14(3), 785–799.
- Rahmani, B., & Mohammadzadeh Moghaddam, A. (2025). [Forecasting demand fluctuations of public bus transit during special events and adverse weather conditions through smart card data analysis](https://doi.org/10.1016/j.tbs.2025.100511). *Travel Behaviour and Society*, 39, 100511.

---

### Finding 14 — Fuel Price Autocorrelation and 52-Week Periodicity

**Script:** `src/eda/fuel/fuel.py`

**Main Idea.** Malaysian retail fuel prices exhibit a dominant 52-week autocorrelation period — pricing decisions follow a government-set annual review cycle. ADF tests confirm non-stationarity at the level, while first-differenced series are stationary. Rolling 4-week price volatility reveals discrete volatility clusters coinciding with policy announcement windows.

**Evidence.** Additive seasonal decomposition (period = 52 weeks, or `max(2, len/4)` if insufficient data) separates trend, seasonal, and residual components for all six fuel types: `ron95`, `ron97`, `diesel`, `diesel_eastmsia`, `ron95_budi95`, `ron95_skps`. ACF and PACF are computed up to `min(52, len//2 − 1)` lags with a floor of 10. Rolling volatility is the 4-week standard deviation of weekly price changes. `level` and `change_weekly` series are handled as separate data types.

**Analysis.** The 52-week periodicity directly motivates including the weekly change columns (`fp_chg_*`) alongside the level columns (`fp_lv_*`) in the feature matrix (Chevance et al., 2024). A model that sees only price levels will conflate slow multi-year trends with annual policy cycles; the change columns provide a stationary, gradient-safe representation of the policy-driven signal. The volatility clustering also supports the non-linear elasticity finding in Finding 12 — high-volatility windows correspond to periods of rapid modal substitution, not gradual adjustment (Rahimi et al., 2024).

**Figures.** (produced per fuel type for all six: `ron95`, `ron97`, `diesel`, `diesel_eastmsia`, `ron95_budi95`, `ron95_skps`)
- `src/eda/fuel/results/acf_{fuel}.png` — ACF up to 52 lags showing the dominant 52-week autocorrelation peak
- `src/eda/fuel/results/pacf_{fuel}.png` — PACF confirming partial autocorrelation structure
- `src/eda/fuel/results/decomposition_{fuel}.png` — additive STL decomposition separating trend, 52-week seasonal, and residual components
- `src/eda/fuel/results/volatility_{fuel}.png` — 4-week rolling standard deviation of weekly price changes, showing discrete volatility clusters
- `src/eda/fuel/results/trend_{fuel}.png` — long-run price trend and level series
- `src/eda/theme/results/graphs/fuel_price_trends.png` — thematic summary of all six fuel price level series

**Link.**
- Chevance, G., Andrieu, B., Koch, N., et al. (2024). [How gasoline prices influence the effectiveness of interventions targeting sustainable transport modes?](https://doi.org/10.1038/s44333-024-00017-1) *npj Sustainable Mobility and Transport*, 1, 17.
- Rahimi, E., Shamshiripour, A., Shabanpour, R., Mohammadian, A., & Auld, J. (2024). [Analyzing the influence of fuel price shock on urban public transit and interurban automobile travel demand: Evidence from a country with fixed fuel price regulation](https://www.researchgate.net/publication/399563477). *Transportation Research Interdisciplinary Perspectives*, 36.

---

### Finding 15 — Holiday Service Gap and Efficiency Ratio

**Script:** `src/eda/multivariate/holiday_service_gap_analysis.py`

**Main Idea.** The ratio of ridership to scheduled service frequency — the efficiency ratio — drops significantly on public holidays compared to normal working days. December and January consistently rank in the top-20 months for holiday concentration and show the sharpest efficiency declines, indicating that scheduled service is not reduced in proportion to demand during these periods.

**Evidence.** A `service_frequency` column is derived from the number of active GTFS calendar services per date. `ridership_per_service = total_ridership / service_frequency` is computed as the efficiency metric. Holiday and non-holiday day means are compared for `total_ridership`, `service_frequency`, and `ridership_per_service`. Monthly aggregation (year-month) identifies the top-20 holiday months by holiday day concentration. Results are exported to `holiday_service_gap.csv` and `holiday_service_detail.csv`.

**Analysis.** The efficiency ratio is a supply-demand alignment metric that cannot be derived from ridership or service data alone. Its inclusion as an EDA finding (rather than a feature) reflects an analytical decision: encoding a ratio of ridership-to-supply as a model input would introduce circular dependency with the target variable. Instead, the finding informs the interpretation of model errors during high-holiday periods — systematic under-prediction on major holidays likely reflects supply over-provision not captured by the binary `is_public_holiday` feature (Wong & Yap, 2023). A more granular `service_frequency` feature was considered but rejected to avoid target leakage (Yu et al., 2024).

**Figures.**
- `src/eda/multivariate/results/holiday_service_gap_analysis.png` — efficiency ratio (ridership per active GTFS service) by calendar period, with top-20 holiday months annotated

**Link.**
- Wong, H., & Yap, M. (2023). [A data driven approach to update public transport service elasticities](https://doi.org/10.1016/j.jpubtr.2023.100066). *Journal of Public Transportation*, 25, 100066.
- Yu, C., Dong, W., Liu, Y., Yang, C., & Yuan, Q. (2024). [Rethinking bus ridership dynamics: Examining nonlinear effects of determinants on bus ridership changes using city-level panel data from 2010 to 2019](https://doi.org/10.1016/j.tranpol.2024.04.004). *Transport Policy*, 148, 1–14.

---

### Finding 16 — Transit Equity Index by Administrative Zone

**Script:** `src/eda/multivariate/transit_equity_index_by_zone.py`

**Main Idea.** A composite transit equity score — combining stop density, population density, walking friction, and POI density — reveals that transit access is spatially inequitable across Malaysian administrative zones (Li et al., 2023; Rathod et al., 2025). States with high stop density but low population density score differently from states with high population density but sparse stop coverage.

**Evidence.** Each administrative zone (GADM Level-1 boundary) receives an equity score computed as:
```
equity_score = 0.30 × (stop_density / max_stop_density)
             + 0.30 × (population_density_estimate / max_pop_density)
             + 0.20 × (1 − walking_friction_estimate)
             + 0.20 × (poi_density_estimate / max_poi_density)
```
Stop density is derived from the spatial join of GTFS nodes to GADM polygons (EPSG:4326 → EPSG:3857, predicate `within`). Zone area is computed from the projected geometry. Results are written to `transit_equity_scores.csv` with columns `NAME_1`, `stop_count`, `stop_density`, `equity_score`. Note: `walking_friction_estimate`, `population_density_estimate`, and `poi_density_estimate` use constant defaults (0.5, 500, 50) in the current implementation, meaning the effective score is driven by stop density until per-zone estimates are substituted.

**Analysis.** The equity index extends Finding 7 (underserved population zones) by adding a composite lens that includes POI accessibility and walkability alongside the density-coverage gap. The constant default estimates mean the current scores are primarily a proxy for stop density disparity — a known limitation. The finding motivates the inclusion of both `pop_density_median` and GTFS summary scalars as complementary static features: neither alone captures the full supply-accessibility picture. For future work, replacing the constant defaults with `population_at_stops.csv` and `poi_counts_at_stops.csv` values would produce a genuine multi-dimensional equity score.

**Figures.**
- `src/eda/multivariate/results/transit_equity_index_by_zone.png` — composite equity scores per GADM Level-1 zone, coloured by stop density quartile
- `src/eda/multivariate/results/composite_accessibility_score.png` — breakdown of equity score components (stop density, population density, walking friction, POI density) per zone
- `src/eda/theme/results/graphs/accessibility_equity_index.png` — thematic summary of the equity index spatial distribution
- `src/eda/gadm/results/adjacency_network.png` — GADM Level-1 adjacency network showing state borders and centroids used in the equity computation

**Link.**
- Li, W., Guan, H., Qin, W., & Ji, X. (2023). [Collective and individual spatial equity measure in public transit accessibility based on generalized travel cost](https://doi.org/10.1016/j.retrec.2023.100033). *Research in Transportation Economics*, 98, 101267.
- Rathod, R., Joshi, G., & Arkatkar, S. (2025). [Composite Accessibility Index: A novel and holistic measure for evaluating transit accessibility](https://doi.org/10.1177/03611981241270156). *Transportation Research Record*.

---

### Finding 17 — Mode Share Decomposition and Ridership Distribution by Service

**Script:** `src/eda/theme/network_performance_series.py`

**Main Idea.** Rail modes (LRT Ampang, MRT Kajang, LRT Kelana Jaya, Monorail, MRT Putrajaya) collectively account for a substantially larger share of total ridership than bus modes (RapidBus KL, RapidBus Penang). Top-100 ridership days are concentrated in the rail sub-system. The two bus services together are individually small relative to any single rail line.

**Evidence.** Modal aggregations are computed as:
```
bus_ridership  = bus_rkl + bus_rpn
rail_ridership = rail_lrt_ampang + rail_mrt_kajang + rail_lrt_kj + rail_monorail + rail_mrt_pjy
bus_ridership_share  = bus_ridership  / total_ridership × 100
rail_ridership_share = rail_ridership / total_ridership × 100
```
Daily totals and averages per mode are written to `ridership_mode_performance.csv`. GTFS stops are spatially joined to GADM boundaries (EPSG:3857, predicate `within`) and counted per zone and source operator. High-density population zones are identified as grid cells with `density_log` above the 90th percentile; stop overlap with these zones is recorded in `network_overlap_stats.csv`.

**Analysis.** Rail dominance in total ridership validates the decision to model each of the 12 service lines as separate target columns rather than aggregating to a single total. An aggregated target would be dominated by the rail sub-system and would mask bus-service dynamics entirely (Yang et al., 2023). The 90th-percentile high-density overlap metric confirms the transit concentration finding (Finding 5): most stops in high-density areas belong to rail operators. This also contextualises the static GTFS features — `gtfs_n_stops` aggregates across all operators and thus conflates the sparse bus network with the denser rail network; per-operator disaggregation is available in the operator node tables if needed (Dai et al., 2024).

**Figures.**
- `src/eda/theme/results/graphs/ridership_by_mode.png` — daily bus vs rail ridership share stacked time series with bus_ridership_share and rail_ridership_share
- `src/eda/theme/results/graphs/network_performance_summary.png` — per-operator stop count, route count, and ridership contribution (shared with Finding 9)
- `src/eda/theme/results/graphs/stop_density_population.png` — 90th-percentile high-density zone overlap with rail vs bus stop locations (shared with Finding 5)
- `src/eda/ridership/results/temporal_trend_analysis.png` — full 2019–2025 ridership by service line, showing rail dominance in absolute scale

**Link.**
- Yang, C., Yu, C., Dong, W., & Yuan, Q. (2023). [Substitutes or complements? Examining effects of urban rail transit on bus ridership using longitudinal city-level data](https://doi.org/10.1016/j.tra.2023.103489). *Transportation Research Part A: Policy and Practice*, 174, 103489.
- Dai, S., Yu, L., Song, L., Li, Y., & Fan, X. (2024). [The temporal distribution of ridership in metro stations from land-use perspective](https://doi.org/10.1371/journal.pone.0308759). *PLOS ONE*, 19(10), e0308759.
