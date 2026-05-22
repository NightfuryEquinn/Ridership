# Feature Engineering & EDA Analysis

Detailed account of all exploratory data analysis (EDA) and engineering decisions made across the eight spatio-temporal feature sources that feed the forecasting pipeline. The output of this pipeline is `data/features/features_aligned.csv` — a daily matrix of **79 features × 1,461 days** (2022-01-01 → 2025-12-31) consumed by all 15 models.

---

## Pipeline Overview

```
data/raw/  →  [source scripts]  →  data/cleaned/  →  feature_align.py  →  features_aligned.csv
                                                            ↓
                                                  sequence_builder.py
                                                            ↓
                                          data/sequences/{lstm|lookback_28|lookback_56}/
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

Pre-launch nulls are left as `NaN` (correct — no service). Post-launch nulls (data gaps after the service is live) are zero-filled. The `feature_metadata.json` records the residual null counts: `rail_mrt_pjy=166`, `rail_tebrau=169`, `rail_komuter=617` — all attributable to the pre-launch period falling within the 2022–2025 master window.

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

After all eight sources are cleaned, `feature_align.py` merges them onto a master daily index (2022-01-01 → 2025-12-31, 1,461 days).

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

### Null Audit

A final null audit is printed after the merge. The only expected nulls in the aligned matrix correspond to the ridership pre-launch period. These are documented in `feature_metadata.json`:

```json
"null_counts": {
  "rail_mrt_pjy": 166,
  "rail_tebrau": 169,
  "rail_komuter": 617
}
```

### Autoregressive Lag Features

Ridership lags (`ridership_lag_7`, `ridership_lag_14`, `ridership_lag_28`) are computed from the **full historical series** (extending before 2022-01-01) before reindexing to the master window. This ensures the first 28 days of the master window have valid lag values without look-back into the missing/MCO period.

---

## Sequence Assembly (`sequence_builder.py`)

Converts `features_aligned.csv` into model-ready tensors.

### Key Design Decisions from EDA

**MCO exclusion.** MCO-period rows (2020-03-18 – 2021-12-31) are excluded from sequences by default (`--include-mco` flag to override). The 2022-01-01 master start date means the MCO period falls entirely outside the master window anyway — this flag guards against future date-range extensions.

**Pre-launch zero-fill.** `df.fillna(0.0)` converts all remaining structural nulls (pre-launch ridership columns) to zero. This is semantically correct and avoids `NaN` propagation into the sliding-window arrays.

**Chronological sliding windows.** Overlapping windows of shape `(T_in, F)` → target `(T_out,)` are extracted without shuffling, preserving temporal order. The default `T_in=14` (14-day look-back) with `T_out=7` (7-day forecast horizon) reflects the weekly seasonality identified in the ridership EDA.

**Scaler leakage prevention.** `MinMaxScaler` is fitted on `X_train` only (flattened to `(N·T, F)` to fit per-feature). A separate scaler is fitted on `y_train` to enable inverse-transformation of predictions back to ridership counts. Both scalers are saved as `.pkl` files alongside the sequences.

**Storage.** Arrays are cast to `float16` by default. For MinMax-scaled `[0, 1]` data, `float16` precision loss (≈ 0.001) is negligible. This halves disk and memory footprint. Optional `--compress` flag writes `.npz` for an additional 1.5–3× reduction.

**Three lookback variants.** Three sequence directories are produced to support the lookback ablation study:

| Flag | Output Dir | T_in |
|------|-----------|------|
| (default) | `data/sequences/lstm/` | 14 |
| `--T-in 28` | `data/sequences/lookback_28/` | 28 |
| `--T-in 56` | `data/sequences/lookback_56/` | 56 |
