"""
feature_align.py  — Temporal Feature Alignment

Merges all cleaned data sources onto a single daily date index, producing
the flat feature matrix consumed by sequence_builder.py and used by all
14 models in the stack:

  LSTM-family      : LSTM, BiLSTM, TPA-LSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM
  Graph-based      : STGCN, MTGNN, STSGCN, STFGNN, PDR-STGCN
  Attention-based  : TPA-LSTM, ASTGCN, Autoformer, Informer

All 8 feature sources are incorporated:

  Temporal / dynamic (vary by date):
    data/cleaned/ridership_headline_clean.csv       ← ridership service levels
    data/cleaned/fuelprice_level_daily.csv          ← daily fuel price levels
    data/cleaned/fuelprice_change_daily.csv         ← daily fuel price changes
    data/cleaned/holiday_daily_features.csv         ← holiday flags + cyclical encoding
    data/cleaned/rainfall_wide_daily.csv            ← daily rainfall per state

  Derived (computed from above):
    ridership_lag_{7,14,28}                         ← autoregressive lag features
    year, day_of_year                               ← secular trend + intra-year position

  Static (broadcast to all dates):
    data/cleaned/population_density_clean.csv       ← national population density
    data/cleaned/gtfs_*/gtfs_stop_nodes_*.csv       ← GTFS transit network counts
    data/cleaned/gtfs_*/gtfs_stop_edges_*.csv       ← GTFS route edge statistics
    data/cleaned/poi_counts_at_stops.csv            ← OSM POI category counts per stop
    data/cleaned/gadm_state_nodes.csv               ← GADM state count
    data/cleaned/gadm_adj_edges.csv                 ← GADM shared-border statistics

Output:
  data/features/features_aligned.csv    — daily matrix (days × N_features)
  data/features/feature_metadata.json   — column groups, dtypes, null counts

Usage:
  python src/features/feature_align.py
  python src/features/feature_align.py --date-start 2022-01-01 --date-end 2025-12-31
  python src/features/feature_align.py --output-dir data/features
"""

import os
import glob
import json
import argparse
import pandas as pd
import numpy as np


# ── Helper ────────────────────────────────────────────────────────────────────

def load_indexed(path: str, date_col: str = "date") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    return df


def main():
    p = argparse.ArgumentParser(description="Align all cleaned features onto a daily date index")
    p.add_argument("--date-start",       default="2019-01-01",
                   help="Start of master date range for MCO-included output (default: 2019-01-01)")
    p.add_argument("--date-end",         default="2025-12-31",
                   help="End of master date range (default: 2025-12-31)")
    p.add_argument("--date-start-nomco", default="2022-01-01",
                   help="Start of date range for MCO-excluded output (default: 2022-01-01)")
    p.add_argument("--output-dir",       default="data/features",
                   help="Directory for output files (default: data/features)")
    args = p.parse_args()

    DATE_START       = args.date_start
    DATE_END         = args.date_end
    DATE_START_NOMCO = args.date_start_nomco
    OUT_DIR          = args.output_dir

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Master date range ──────────────────────────────────────────────────────
    master_idx = pd.DatetimeIndex(
        pd.date_range(DATE_START, DATE_END, freq="D"),
        name="date"
    )
    print(f"Master date range: {DATE_START} → {DATE_END}  ({len(master_idx)} days)")

    aligned = pd.DataFrame(index=master_idx)

    # ── [1/8] Ridership (target + service-level features) ────────────────────────
    # Load the FULL historical series (2019-present) so lag features can be
    # computed without losing the first 28 days of the master date range.
    print("\n[1/8] Loading ridership...")
    ridership = load_indexed("data/cleaned/ridership_headline_clean.csv")

    service_cols = [c for c in ridership.columns
                    if c not in ("is_mco", "day_of_week", "month", "day_of_year",
                                  "is_weekend", "dow_sin", "dow_cos",
                                  "month_sin", "month_cos")]
    ridership_sub = ridership[service_cols].reindex(master_idx)

    for col in ridership_sub.columns:
        ridership_sub[col] = ridership_sub[col].astype(float)

    aligned = aligned.join(ridership_sub)
    print(f"  Ridership columns added: {list(service_cols)}")
    print(f"  Null rate: {aligned[list(service_cols)].isnull().mean().mean():.2%}")

    # ── Autoregressive lag features ───────────────────────────────────────────────
    # Computed from the FULL ridership series (extends before DATE_START) so that
    # the first rows of the master window have valid lag values.
    # For lookback=14: lag_28 falls outside the window — having it as an explicit
    # feature lets the model see further back without extending the sequence length.
    lag_cols = []
    if "total_ridership" in ridership.columns:
        total_full = ridership["total_ridership"].astype(float)
        for lag in [7, 14, 28]:
            lag_series          = total_full.shift(lag)
            col_name            = f"ridership_lag_{lag}"
            lag_series.name     = col_name
            aligned[col_name]   = lag_series.reindex(master_idx).bfill()
            lag_cols.append(col_name)
        print(f"  Lag features added: {lag_cols}")

    # ── Secular trend + intra-year position ──────────────────────────────────────
    aligned["year"]        = aligned.index.year
    aligned["day_of_year"] = aligned.index.dayofyear
    trend_cols = ["year", "day_of_year"]
    print(f"  Trend features added: {trend_cols}")

    # ── [2/8] Fuel price (daily forward-filled) ───────────────────────────────────
    print("\n[2/8] Loading fuel prices...")
    fuel_level  = load_indexed("data/cleaned/fuelprice_level_daily.csv")
    fuel_change = load_indexed("data/cleaned/fuelprice_change_daily.csv")

    fuel_level  = fuel_level.reindex(master_idx).ffill()
    fuel_change = fuel_change.reindex(master_idx).ffill()

    fuel_level.columns  = [f"fp_lv_{c}" for c in fuel_level.columns]
    fuel_change.columns = [f"fp_chg_{c}" for c in fuel_change.columns]

    aligned = aligned.join(fuel_level).join(fuel_change)
    fp_cols = list(fuel_level.columns) + list(fuel_change.columns)
    print(f"  Fuel price columns: {fp_cols}")
    print(f"  Null rate: {aligned[fp_cols].isnull().mean().mean():.2%}")

    # ── [3/8] Holiday / cyclical features ─────────────────────────────────────────
    print("\n[3/8] Loading holiday features...")
    holidays = load_indexed("data/cleaned/holiday_daily_features.csv")
    holidays  = holidays.reindex(master_idx).fillna(0)

    aligned = aligned.join(holidays)
    hol_cols = list(holidays.columns)
    print(f"  Holiday columns: {hol_cols[:8]}{'...' if len(hol_cols) > 8 else ''}")

    # ── [4/8] Rainfall (state-level wide format) ──────────────────────────────────
    print("\n[4/8] Loading rainfall...")
    rainfall = load_indexed("data/cleaned/rainfall_wide_daily.csv")
    rainfall  = rainfall.reindex(master_idx)
    rainfall  = rainfall.interpolate(method="linear", limit=7).bfill().ffill()

    rf_mm_cols = [c for c in rainfall.columns if "rainfall_mm__" in c]
    aligned = aligned.join(rainfall[rf_mm_cols])
    print(f"  Rainfall state columns: {len(rf_mm_cols)}")
    print(f"  Null rate: {aligned[rf_mm_cols].isnull().mean().mean():.2%}")

    # ── [5/8] Population density (static — broadcast to all dates) ───────────────
    print("\n[5/8] Loading population density summary...")
    try:
        pop = pd.read_csv("data/cleaned/population_density_clean.csv")
        pop_median = pop["density_per_km2"].median()
        pop_log_median = pop["density_log"].median()
        aligned["pop_density_median"]     = pop_median
        aligned["pop_density_log_median"] = pop_log_median
        print(f"  National median density: {pop_median:.2f} per km²  "
              f"(log: {pop_log_median:.4f})")
    except FileNotFoundError:
        print("  [SKIP] population_density_clean.csv not found")

    # ── [6/8] GTFS static network features (static — broadcast to all dates) ───────
    print("\n[6/8] Loading GTFS static network features...")
    gtfs_cols = []
    node_frames = [pd.read_csv(p) for p in glob.glob("data/cleaned/gtfs_*/gtfs_stop_nodes_*.csv")]
    edge_frames = [pd.read_csv(p) for p in glob.glob("data/cleaned/gtfs_*/gtfs_stop_edges_*.csv")]

    if node_frames:
        df_nodes_all = pd.concat(node_frames, ignore_index=True).drop_duplicates("stop_id")
        n_stops  = len(df_nodes_all)
        n_routes = int(df_nodes_all["route_id"].nunique()) if "route_id" in df_nodes_all.columns else 0

        if edge_frames:
            df_edges_all = pd.concat(edge_frames, ignore_index=True)
            n_edges  = len(df_edges_all)
            valid_tt = df_edges_all.loc[df_edges_all["travel_time_s"] > 0, "travel_time_s"]
            avg_tt_s = round(float(valid_tt.mean()), 1) if len(valid_tt) > 0 else 0.0
        else:
            n_edges, avg_tt_s = 0, 0.0

        aligned["gtfs_n_stops"]          = n_stops
        aligned["gtfs_n_routes"]         = n_routes
        aligned["gtfs_n_directed_edges"] = n_edges
        aligned["gtfs_avg_segment_s"]    = avg_tt_s
        gtfs_cols = ["gtfs_n_stops", "gtfs_n_routes", "gtfs_n_directed_edges", "gtfs_avg_segment_s"]
        print(f"  stops={n_stops}, routes={n_routes}, edges={n_edges}, "
              f"avg_segment_time={avg_tt_s:.1f}s")
    else:
        print("  [SKIP] No GTFS node tables found — run gtfs.py first")

    # ── [7/8] OSM POI network features (static — broadcast to all dates) ───────────
    print("\n[7/8] Loading OSM POI network features...")
    poi_cols = []
    try:
        df_poi = pd.read_csv("data/cleaned/poi_counts_at_stops.csv")
        poi_cat_cols = ["poi_total", "poi_transport", "poi_food", "poi_retail",
                        "poi_education", "poi_healthcare", "poi_leisure", "poi_other"]
        for col in poi_cat_cols:
            if col in df_poi.columns:
                new_col = f"osm_{col}_mean"
                aligned[new_col] = round(float(df_poi[col].mean()), 4)
                poi_cols.append(new_col)
        print(f"  OSM POI columns added: {poi_cols}")
    except FileNotFoundError:
        print("  [SKIP] poi_counts_at_stops.csv not found — run osm.py first")

    # ── [8/8] GADM boundary features (static — broadcast to all dates) ─────────────
    print("\n[8/8] Loading GADM boundary features...")
    gadm_cols = []
    try:
        df_gadm_nodes = pd.read_csv("data/cleaned/gadm_state_nodes.csv")
        n_states = len(df_gadm_nodes)

        df_gadm_edges = pd.read_csv("data/cleaned/gadm_adj_edges.csv")
        n_border_pairs  = len(df_gadm_edges)
        mean_border_km  = round(float(df_gadm_edges["shared_border_km"].mean()), 2) \
                          if not df_gadm_edges.empty else 0.0

        aligned["gadm_n_states"]       = n_states
        aligned["gadm_n_border_pairs"] = n_border_pairs
        aligned["gadm_mean_border_km"] = mean_border_km
        gadm_cols = ["gadm_n_states", "gadm_n_border_pairs", "gadm_mean_border_km"]
        print(f"  states={n_states}, border_pairs={n_border_pairs}, "
              f"mean_border={mean_border_km:.1f} km")
    except FileNotFoundError:
        print("  [SKIP] GADM files not found — run gadm.py first")

    # ── Final null audit ──────────────────────────────────────────────────────────
    print("\n=== Null Audit (aligned matrix) ===")
    null_pct = aligned.isnull().mean().mul(100).round(2)
    if null_pct.max() > 0:
        print("Columns with nulls:")
        print(null_pct[null_pct > 0].to_string())
    else:
        print("No nulls — all features fully aligned.")

    print(f"\nAligned feature matrix shape: {aligned.shape}")
    print(f"  {len(aligned)} rows (days)  ×  {aligned.shape[1]} feature columns")

    # ── Feature metadata ───────────────────────────────────────────────────────────
    target_cols   = ["total_ridership"] + [c for c in service_cols if c != "total_ridership"]
    temporal_cols = hol_cols + trend_cols
    external_cols = fp_cols + rf_mm_cols
    static_cols   = (
        ["pop_density_median", "pop_density_log_median"]
        + gtfs_cols
        + poi_cols
        + gadm_cols
    )

    sources_present = [
        "ridership", "fuel_price", "holiday_cyclical", "rainfall",
        "population_density", "gtfs_static", "osm_poi", "gadm_boundaries",
    ]

    metadata = {
        "date_range":      {"start": DATE_START, "end": DATE_END},
        "total_features":  int(aligned.shape[1]),
        "total_days":      int(len(aligned)),
        "feature_sources": sources_present,
        "column_groups": {
            "targets":    [c for c in target_cols    if c in aligned.columns],
            "temporal":   [c for c in temporal_cols  if c in aligned.columns],
            "external":   [c for c in external_cols  if c in aligned.columns],
            "lag":        [c for c in lag_cols        if c in aligned.columns],
            "static":     [c for c in static_cols    if c in aligned.columns],
        },
        "null_counts": {col: int(v) for col, v in aligned.isnull().sum().items() if v > 0},
    }

    with open(f"{OUT_DIR}/feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ── MCO-excluded variant ───────────────────────────────────────────────────────
    _mco_start = pd.Timestamp("2020-03-18")
    _mco_end   = pd.Timestamp("2021-12-31")
    _nomco_start = pd.Timestamp(DATE_START_NOMCO)
    aligned_no_mco = aligned[
        (aligned.index >= _nomco_start) &
        ((aligned.index < _mco_start) | (aligned.index > _mco_end))
    ]
    print(f"\nMCO-excluded variant: start={DATE_START_NOMCO}, "
          f"MCO rows removed ({_mco_start.date()} – {_mco_end.date()}), "
          f"{len(aligned_no_mco)} rows remaining")

    metadata_no_mco = {**metadata,
                       "date_range":  {"start": DATE_START_NOMCO, "end": DATE_END},
                       "total_days":  int(len(aligned_no_mco)),
                       "null_counts": {col: int(v) for col, v in aligned_no_mco.isnull().sum().items() if v > 0},
                       "mco_excluded": True}

    with open(f"{OUT_DIR}/feature_metadata_no_mco.json", "w") as f:
        json.dump(metadata_no_mco, f, indent=2)

    # ── Export ─────────────────────────────────────────────────────────────────────
    aligned.to_csv(f"{OUT_DIR}/features_aligned.csv")
    aligned_no_mco.to_csv(f"{OUT_DIR}/features_aligned_no_mco.csv")

    print("\nExported:")
    print(f"  {OUT_DIR}/features_aligned.csv         ← MCO included  ({len(aligned)} days)")
    print(f"  {OUT_DIR}/features_aligned_no_mco.csv  ← MCO excluded  ({len(aligned_no_mco)} days)")
    print(f"  {OUT_DIR}/feature_metadata.json")
    print(f"  {OUT_DIR}/feature_metadata_no_mco.json")
    print(f"\nFeature sources included: {sources_present}")
    print(f"Total features: {aligned.shape[1]}")
    print(f"  targets={len([c for c in target_cols if c in aligned.columns])}, "
          f"temporal={len([c for c in temporal_cols if c in aligned.columns])}, "
          f"external={len([c for c in external_cols if c in aligned.columns])}, "
          f"lag={len([c for c in lag_cols if c in aligned.columns])}, "
          f"static={len([c for c in static_cols if c in aligned.columns])}")


if __name__ == "__main__":
    main()
