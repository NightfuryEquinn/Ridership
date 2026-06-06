"""
gtfs.py  — GTFS Cleaning + Graph Export for ST Models
Changes vs original:
  - After cleaning, exports two graph artefacts needed by GCN-based models:
      (a) gtfs_stop_nodes.csv   — node table: stop_id, lat, lon, route_id, operator
      (b) gtfs_stop_edges.csv   — edge list: (stop_i, stop_j, travel_time_s, route_id)
          derived from consecutive stops in stop_times.txt
  - These are consumed by graph_builder.py to assemble the final A matrix.
  - All original cleaning logic is preserved unchanged.

Usage (unchanged):
  python src/cleaning/gtfs.py --input ./data/raw/gtfs_rapid_rail_kl
                               --output ./data/cleaned/gtfs_rapid_rail_kl
                               --operator rapid_rail_kl
"""

import os
import re
import sys
import argparse
import logging
from collections import defaultdict

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("gtfs_cleaner")

# ── Malaysia bounding box ──────────────────────────────────────────────────────
MY_LAT = (0.9, 7.4)
MY_LON = (99.6, 119.3)

MY_POSTCODE_RE = re.compile(r"^\d{5}$")

OPERATOR_PRESETS = {
    "rapid_rail_kl": {
        "label": "Rapid Rail KL",
        "expected_service_ids": {"MonFri", "Sat", "Sun", "weekday", "weekend"},
        "frequency_based": True,
        "expected_route_types": {0, 1},
        "agency_id": "rapidrail",
        "stop_prefix_route": {"KG": "KGL", "PY": "PYL"},
        "nonstandard_stop_cols": ["geometry", "category", "isOKU", "status", "search"],
        "nonstandard_route_cols": ["category", "status"],
        "nonstandard_st_cols": ["route_id", "direction_id"],
    },
    "rapidbus_kl": {
        "label": "RapidBus KL",
        "expected_service_ids": {"weekday", "weekend"},
        "frequency_based": True,
        "expected_route_types": {3},
        "agency_id": "rapidkl",
        "stop_prefix_route": {},
        "nonstandard_stop_cols": [],
        "nonstandard_route_cols": [],
        "nonstandard_st_cols": [],
    },
    "rapidbus_penang": {
        "label": "RapidBus Penang",
        "expected_service_ids": None,
        "frequency_based": False,
        "expected_route_types": {3},
        "agency_id": "rapidpg",
        "stop_prefix_route": {},
        "nonstandard_stop_cols": [],
        "nonstandard_route_cols": [],
        "nonstandard_st_cols": [],
    },
    "ktmb": {
        "label": "KTMB",
        "expected_service_ids": {"komuter_weekday", "komuter_weekend", "komuter_utara", "ets", "intercity"},
        "frequency_based": False,
        "expected_route_types": {0, 2},
        "agency_id": "ktmb",
        "stop_prefix_route": {},
        "nonstandard_stop_cols": [],
        "nonstandard_route_cols": [],
        "nonstandard_st_cols": [],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def read_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df

def write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info(f"  → wrote {os.path.basename(path)} ({len(df):,} rows)")

def check_coords(df, lat_col, lon_col, label):
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    invalid = (
        df[lat_col].isna() | df[lon_col].isna() |
        ~df[lat_col].between(*MY_LAT) |
        ~df[lon_col].between(*MY_LON)
    )
    if invalid.any():
        log.warning(f"  [{label}] {invalid.sum()} rows with invalid/out-of-bounds coordinates — dropped")
        df = df[~invalid].copy()
    return df

def drop_nonstandard(df, cols, label):
    to_drop = [c for c in cols if c in df.columns]
    if to_drop:
        log.info(f"  [{label}] dropping non-standard columns: {to_drop}")
        df = df.drop(columns=to_drop)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Per-file cleaners (unchanged — see original for full docstrings)
# ══════════════════════════════════════════════════════════════════════════════

def clean_agency(df, preset):
    log.info("[agency.txt]")
    required = ["agency_id", "agency_name", "agency_url", "agency_timezone"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.warning(f"  Missing required columns: {missing}")
    df = df.drop_duplicates(subset=["agency_id"])
    log.info(f"  {len(df)} agenc{'y' if len(df)==1 else 'ies'} — OK")
    return df

def clean_calendar(df, preset):
    log.info("[calendar.txt]")
    for col in ["start_date", "end_date"]:
        df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
        bad = df[col].isna().sum()
        if bad:
            log.warning(f"  {bad} unparseable {col} values")
    inverted = df["end_date"] < df["start_date"]
    if inverted.any():
        log.warning(f"  {inverted.sum()} rows with end_date < start_date — swapping")
        df.loc[inverted, ["start_date", "end_date"]] = df.loc[inverted, ["end_date", "start_date"]].values
    day_cols = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    bitmasks = df.set_index("service_id")[day_cols].astype(str).apply("|".join, axis=1)
    dups = bitmasks[bitmasks.duplicated(keep=False)]
    if not dups.empty:
        log.warning(f"  Identical day-pattern service_ids: {dups.index.tolist()}")
    df["start_date"] = df["start_date"].dt.strftime("%Y%m%d")
    df["end_date"]   = df["end_date"].dt.strftime("%Y%m%d")
    all_service_ids = set(df["service_id"])
    log.info(f"  {len(df)} service definitions — {all_service_ids}")
    return df, all_service_ids

def clean_calendar_dates(df):
    if df is None:
        log.info("[calendar_dates.txt] not present — skipped")
        return None
    log.info("[calendar_dates.txt]")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    bad = df["date"].isna().sum()
    if bad:
        log.warning(f"  {bad} unparseable date values — dropped")
        df = df[df["date"].notna()]
    df["date"] = df["date"].dt.strftime("%Y%m%d")
    df["exception_type"] = pd.to_numeric(df["exception_type"], errors="coerce")
    invalid_type = ~df["exception_type"].isin([1, 2])
    if invalid_type.any():
        log.warning(f"  {invalid_type.sum()} rows with invalid exception_type — dropped")
        df = df[~invalid_type]
    dups = df.duplicated(subset=["service_id", "date"])
    if dups.any():
        log.warning(f"  {dups.sum()} duplicate (service_id, date) pairs — keeping first")
        df = df[~dups]
    log.info(f"  {len(df)} exception rows")
    return df

def clean_routes(df, preset):
    log.info("[routes.txt]")
    df = drop_nonstandard(df, preset["nonstandard_route_cols"], "routes")
    df["route_type"] = pd.to_numeric(df["route_type"], errors="coerce")
    unexpected = ~df["route_type"].isin(preset["expected_route_types"])
    if unexpected.any():
        log.warning(f"  Unexpected route_type values: {df.loc[unexpected, ['route_id','route_type']].to_dict('records')}")
    for col in ["route_color", "route_text_color"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lstrip("#").str.upper().where(df[col].notna(), other=None)
    dups = df.duplicated(subset=["route_id"])
    if dups.any():
        log.warning(f"  {dups.sum()} duplicate route_id(s) — keeping first")
        df = df[~dups]
    route_ids = set(df["route_id"])
    log.info(f"  {len(df)} routes — {route_ids}")
    return df, route_ids

def clean_stops(df, preset):
    log.info("[stops.txt]")
    df = drop_nonstandard(df, preset["nonstandard_stop_cols"], "stops")
    if preset["stop_prefix_route"]:
        def fix_route(row):
            prefix = re.sub(r"\d", "", row["stop_id"])[:2].upper()
            return preset["stop_prefix_route"].get(prefix, row.get("route_id", ""))
        if "route_id" in df.columns:
            before = (df["route_id"] == "MRT").sum()
            df["route_id"] = df.apply(fix_route, axis=1)
            log.info(f"  Fixed {before} ambiguous route_id='MRT' via stop_id prefix")
    df = check_coords(df, "stop_lat", "stop_lon", "stops")
    df["stop_name"] = df["stop_name"].str.strip()
    dups = df.duplicated(subset=["stop_id"])
    if dups.any():
        log.warning(f"  {dups.sum()} duplicate stop_id(s) — keeping first")
        df = df[~dups]
    stop_ids = set(df["stop_id"])
    log.info(f"  {len(df)} stops")
    return df, stop_ids

def clean_shapes(df):
    if df is None:
        log.info("[shapes.txt] not present — trips will have no map geometry")
        return None
    log.info("[shapes.txt]")
    if "shape_pt_lon" in df.columns and "shape_pt_lat" in df.columns:
        sample_lat = pd.to_numeric(df["shape_pt_lat"].iloc[0], errors="coerce")
        if sample_lat and sample_lat > 90:
            log.warning("  Columns shape_pt_lat and shape_pt_lon appear reversed — swapping")
            df = df.rename(columns={"shape_pt_lat": "shape_pt_lon", "shape_pt_lon": "shape_pt_lat"})
        other_cols = [c for c in df.columns if c not in ["shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"]]
        df = df[["shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"] + other_cols]
    df = check_coords(df, "shape_pt_lat", "shape_pt_lon", "shapes")
    df["shape_pt_sequence"] = pd.to_numeric(df["shape_pt_sequence"], errors="coerce")
    bad_seq = df["shape_pt_sequence"].isna()
    if bad_seq.any():
        log.warning(f"  {bad_seq.sum()} non-numeric shape_pt_sequence values — dropped")
        df = df[~bad_seq]
    df = df.sort_values(["shape_id", "shape_pt_sequence"]).reset_index(drop=True)
    log.info(f"  {len(df)} shape points across {df['shape_id'].nunique()} shapes")
    return df

def clean_trips(df, route_ids, service_ids, shape_ids, preset):
    log.info("[trips.txt]")
    bad_routes = ~df["route_id"].isin(route_ids)
    if bad_routes.any():
        log.warning(f"  {bad_routes.sum()} trips reference unknown route_id(s) — dropped")
        df = df[~bad_routes]
    bad_services = ~df["service_id"].isin(service_ids)
    if bad_services.any():
        log.warning(f"  {bad_services.sum()} trips reference unknown service_id(s) — dropped")
        df = df[~bad_services]
    if shape_ids is not None and "shape_id" in df.columns:
        bad_shapes = df["shape_id"].notna() & ~df["shape_id"].isin(shape_ids)
        if bad_shapes.any():
            log.warning(f"  {bad_shapes.sum()} trips reference unknown shape_id(s) — nulling")
            df.loc[bad_shapes, "shape_id"] = None
    used_services = set(df["service_id"])
    unused = service_ids - used_services
    if unused:
        log.warning(f"  {len(unused)} service_id(s) in calendar but unused: {unused}")
    dups = df.duplicated(subset=["trip_id"])
    if dups.any():
        log.warning(f"  {dups.sum()} duplicate trip_id(s) — keeping first")
        df = df[~dups]
    trip_ids = set(df["trip_id"])
    log.info(f"  {len(df)} trips")
    return df, trip_ids

def clean_frequencies(df, trip_ids, preset):
    if df is None:
        if preset["frequency_based"]:
            log.warning("[frequencies.txt] not present — expected for this operator.")
        else:
            log.info("[frequencies.txt] not present — operator is schedule-based (expected)")
        return None
    log.info("[frequencies.txt]")
    bad = ~df["trip_id"].isin(trip_ids)
    if bad.any():
        log.warning(f"  {bad.sum()} frequency rows reference unknown trip_id(s) — dropped")
        df = df[~bad]
    df["headway_secs"] = pd.to_numeric(df["headway_secs"], errors="coerce")
    bad_hw = df["headway_secs"].isna() | (df["headway_secs"] <= 0)
    if bad_hw.any():
        log.warning(f"  {bad_hw.sum()} rows with invalid headway_secs — dropped")
        df = df[~bad_hw]
    def parse_gtfs_time(t):
        try:
            h, m, s = map(int, t.strip().split(":"))
            return h * 3600 + m * 60 + s
        except Exception:
            return None
    df["_start_secs"] = df["start_time"].apply(parse_gtfs_time)
    df["_end_secs"]   = df["end_time"].apply(parse_gtfs_time)
    bad_times = df["_start_secs"].isna() | df["_end_secs"].isna()
    if bad_times.any():
        log.warning(f"  {bad_times.sum()} rows with unparseable time strings — dropped")
        df = df[~bad_times]
    inverted = df["_end_secs"] <= df["_start_secs"]
    if inverted.any():
        log.warning(f"  {inverted.sum()} rows where end_time ≤ start_time — dropped")
        df = df[~inverted]
    df = df.drop(columns=["_start_secs", "_end_secs"])
    freq_trip_ids = set(df["trip_id"])
    log.info(f"  {len(df)} frequency bands across {len(freq_trip_ids)} trips")
    return df

def clean_stop_times(df, trip_ids, stop_ids, freq_trip_ids, preset):
    log.info("[stop_times.txt]")
    df = drop_nonstandard(df, preset["nonstandard_st_cols"], "stop_times")
    bad_trips = ~df["trip_id"].isin(trip_ids)
    if bad_trips.any():
        log.warning(f"  {bad_trips.sum()} stop_time rows reference unknown trip_id(s) — dropped")
        df = df[~bad_trips]
    bad_stops = ~df["stop_id"].isin(stop_ids)
    if bad_stops.any():
        log.warning(f"  {bad_stops.sum()} stop_time rows reference unknown stop_id(s) — dropped")
        df = df[~bad_stops]
    df["stop_sequence"] = pd.to_numeric(df["stop_sequence"], errors="coerce")
    bad_seq = df["stop_sequence"].isna() | (df["stop_sequence"] < 0)
    if bad_seq.any():
        log.warning(f"  {bad_seq.sum()} rows with invalid stop_sequence — dropped")
        df = df[~bad_seq]
    df = df.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)
    trip_stop_counts = df.groupby("trip_id")["stop_id"].count()
    unusable = trip_stop_counts[trip_stop_counts < 2].index.tolist()
    if unusable:
        log.warning(f"  {len(unusable)} trip(s) with < 2 stops — dropped: {unusable}")
        df = df[~df["trip_id"].isin(unusable)]
    log.info(f"  {len(df):,} stop_time rows")
    return df

def clean_transfers(df, stop_ids):
    if df is None:
        log.info("[transfers.txt] not present — skipped")
        return None
    log.info("[transfers.txt]")
    for col in ["from_stop_id", "to_stop_id"]:
        bad = ~df[col].isin(stop_ids)
        if bad.any():
            log.warning(f"  {bad.sum()} rows with unknown {col} — dropped")
            df = df[~bad]
    df["transfer_type"] = pd.to_numeric(df["transfer_type"], errors="coerce")
    invalid = ~df["transfer_type"].isin([0, 1, 2, 3])
    if invalid.any():
        log.warning(f"  {invalid.sum()} rows with invalid transfer_type — dropped")
        df = df[~invalid]
    if "min_transfer_time" in df.columns:
        df["min_transfer_time"] = pd.to_numeric(df["min_transfer_time"], errors="coerce")
        negative = df["min_transfer_time"] < 0
        if negative.any():
            log.warning(f"  {negative.sum()} negative min_transfer_time values — set to 0")
            df.loc[negative, "min_transfer_time"] = 0
    log.info(f"  {len(df)} transfer entries")
    return df

def integrity_report(route_ids, service_ids, stop_ids, shape_ids,
                     trip_ids, freq_trip_ids, st_trip_ids, st_stop_ids):
    log.info("")
    log.info("════ Referential Integrity Summary ════")
    checks = [
        ("All trip route_ids in routes",     True),
        ("All trip service_ids in calendar", True),
        ("All stop_time trip_ids in trips",  not (st_trip_ids - trip_ids)),
        ("All stop_time stop_ids in stops",  not (st_stop_ids - stop_ids)),
    ]
    for label, ok in checks:
        log.info(f"  {'✓' if ok else '✗'}  {label}")
    log.info("═══════════════════════════════════════")


# ══════════════════════════════════════════════════════════════════════════════
# NEW: Graph artefact export
# ══════════════════════════════════════════════════════════════════════════════

def export_graph_artefacts(
    stops_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    output_dir: str,
    operator: str,
) -> None:
    """
    Export two graph artefacts for use by graph_builder.py:

    1. gtfs_stop_nodes_{operator}.csv
       Columns: stop_id, stop_name, stop_lat, stop_lon, route_id, operator
       One row per unique stop — used as graph nodes.

    2. gtfs_stop_edges_{operator}.csv
       Columns: from_stop_id, to_stop_id, route_id, travel_time_s, operator
       One row per consecutive stop pair on any trip — used as graph edges.
       travel_time_s is estimated from departure_time differences where
       available; set to -1 for frequency-based templates (no wall-clock times).
    """
    log.info("")
    log.info("════ Exporting Graph Artefacts ════")

    # ── Node table ─────────────────────────────────────────────────────────────
    node_cols = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
    if "route_id" in stops_df.columns:
        node_cols.append("route_id")
    df_nodes = stops_df[node_cols].copy()
    df_nodes["operator"] = operator

    node_path = os.path.join(output_dir, f"gtfs_stop_nodes_{operator}.csv")
    df_nodes.to_csv(node_path, index=False)
    log.info(f"  → node table: {node_path} ({len(df_nodes)} stops)")

    # ── Edge table from stop_times ─────────────────────────────────────────────
    # Join stop_times with trips to get route_id per stop_time row.
    st = stop_times_df.copy()
    if "route_id" not in st.columns and trips_df is not None and "route_id" in trips_df.columns:
        trip_route = trips_df[["trip_id", "route_id"]].drop_duplicates()
        st = st.merge(trip_route, on="trip_id", how="left")

    # Parse departure_time to total seconds for travel time estimation.
    def gtfs_time_to_secs(t):
        try:
            h, m, s = map(int, str(t).strip().split(":"))
            return h * 3600 + m * 60 + s
        except Exception:
            return None

    if "departure_time" in st.columns:
        st["_dep_secs"] = st["departure_time"].apply(gtfs_time_to_secs)
    else:
        st["_dep_secs"] = None

    edges = []
    for trip_id, grp in st.groupby("trip_id"):
        grp = grp.sort_values("stop_sequence").reset_index(drop=True)
        route_id = grp["route_id"].iloc[0] if "route_id" in grp.columns else None
        for k in range(len(grp) - 1):
            from_stop = grp.loc[k,   "stop_id"]
            to_stop   = grp.loc[k+1, "stop_id"]
            dep_k     = grp.loc[k,   "_dep_secs"]
            dep_k1    = grp.loc[k+1, "_dep_secs"]

            if dep_k is not None and dep_k1 is not None:
                travel_time_s = int(dep_k1) - int(dep_k)
                if travel_time_s < 0:
                    travel_time_s = -1   # overnight / data error
            else:
                travel_time_s = -1       # frequency-based template, no wall-clock

            edges.append({
                "from_stop_id":  from_stop,
                "to_stop_id":    to_stop,
                "route_id":      route_id,
                "travel_time_s": travel_time_s,
                "operator":      operator,
            })

    df_edges = pd.DataFrame(edges).drop_duplicates(
        subset=["from_stop_id", "to_stop_id", "route_id"]
    )
    edge_path = os.path.join(output_dir, f"gtfs_stop_edges_{operator}.csv")
    df_edges.to_csv(edge_path, index=False)
    log.info(f"  → edge table: {edge_path} ({len(df_edges)} directed edges)")
    log.info("═══════════════════════════════════════")


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run(input_dir, output_dir, operator):
    preset = OPERATOR_PRESETS[operator]
    log.info(f"Operator : {preset['label']}")
    log.info(f"Input    : {input_dir}")
    log.info(f"Output   : {output_dir}")

    def src(fn): return os.path.join(input_dir, fn)
    def dst(fn): return os.path.join(output_dir, fn)

    agency_df         = read_csv(src("agency.txt"))
    calendar_df       = read_csv(src("calendar.txt"))
    calendar_dates_df = read_csv(src("calendar_dates.txt"))
    routes_df         = read_csv(src("routes.txt"))
    stops_df          = read_csv(src("stops.txt"))
    shapes_df         = read_csv(src("shapes.txt"))
    trips_df          = read_csv(src("trips.txt"))
    frequencies_df    = read_csv(src("frequencies.txt"))
    stop_times_df     = read_csv(src("stop_times.txt"))
    transfers_df      = read_csv(src("transfers.txt"))

    required = {
        "agency.txt": agency_df, "calendar.txt": calendar_df,
        "routes.txt": routes_df, "stops.txt": stops_df,
        "trips.txt": trips_df,   "stop_times.txt": stop_times_df,
    }
    missing_required = [k for k, v in required.items() if v is None]
    if missing_required:
        log.error(f"Missing required GTFS files: {missing_required}")
        sys.exit(1)

    agency_df                      = clean_agency(agency_df, preset)
    calendar_df, service_ids       = clean_calendar(calendar_df, preset)
    calendar_dates_df              = clean_calendar_dates(calendar_dates_df)
    routes_df, route_ids           = clean_routes(routes_df, preset)
    stops_df, stop_ids             = clean_stops(stops_df, preset)
    shapes_df                      = clean_shapes(shapes_df)
    shape_ids = set(shapes_df["shape_id"]) if shapes_df is not None else None

    trips_df, trip_ids = clean_trips(trips_df, route_ids, service_ids, shape_ids, preset)

    frequencies_df = clean_frequencies(frequencies_df, trip_ids, preset)
    freq_trip_ids  = set(frequencies_df["trip_id"]) if frequencies_df is not None else None

    stop_times_df  = clean_stop_times(stop_times_df, trip_ids, stop_ids, freq_trip_ids, preset)

    surviving_trip_ids = set(stop_times_df["trip_id"])
    dropped_trips = trip_ids - surviving_trip_ids
    if dropped_trips:
        log.warning(f"  {len(dropped_trips)} trip(s) with no surviving stop_time rows — dropped from trips")
        trips_df = trips_df[trips_df["trip_id"].isin(surviving_trip_ids)]
        trip_ids = surviving_trip_ids

    transfers_df   = clean_transfers(transfers_df, stop_ids)

    integrity_report(
        route_ids, service_ids, stop_ids, shape_ids,
        trip_ids, freq_trip_ids,
        set(stop_times_df["trip_id"]),
        set(stop_times_df["stop_id"]),
    )

    log.info("Writing clean GTFS files...")
    write_csv(agency_df,     dst("agency.txt"))
    write_csv(calendar_df,   dst("calendar.txt"))
    write_csv(routes_df,     dst("routes.txt"))
    write_csv(stops_df,      dst("stops.txt"))
    write_csv(trips_df,      dst("trips.txt"))
    write_csv(stop_times_df, dst("stop_times.txt"))
    if calendar_dates_df is not None: write_csv(calendar_dates_df, dst("calendar_dates.txt"))
    if shapes_df         is not None: write_csv(shapes_df,         dst("shapes.txt"))
    if frequencies_df    is not None: write_csv(frequencies_df,    dst("frequencies.txt"))
    if transfers_df      is not None: write_csv(transfers_df,      dst("transfers.txt"))

    # ── NEW: export graph artefacts ───────────────────────────────────────────
    export_graph_artefacts(stops_df, stop_times_df, trips_df, output_dir, operator)

    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTFS cleaner for Malaysian transit operators")
    parser.add_argument("--input",    required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--operator", required=True, choices=list(OPERATOR_PRESETS.keys()))
    args = parser.parse_args()
    run(args.input, args.output, args.operator)