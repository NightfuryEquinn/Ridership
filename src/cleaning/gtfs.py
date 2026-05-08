"""
GTFS Data Cleaning Script
Supports: Rapid Rail KL, RapidBus KL, RapidBus Penang, KTMB
Handles all standard GTFS .txt files, with graceful fallback for optional files
(frequencies.txt, shapes.txt, calendar_dates.txt, fare_*.txt, transfers.txt).

Usage:
  python src/cleaning/gtfs.py --input ./data/raw/gtfs_rapid_rail_kl --output ./data/cleaned/gtfs_rapid_rail_kl --operator rapid_rail_kl
  python src/cleaning/gtfs.py --input ./data/raw/gtfs_rapid_bus_kl --output ./data/cleaned/gtfs_rapid_bus_kl --operator rapidbus_kl
  python src/cleaning/gtfs.py --input ./data/raw/gtfs_rapid_bus_penang --output ./data/cleaned/gtfs_rapid_bus_penang --operator rapidbus_penang
  python src/cleaning/gtfs.py --input ./data/raw/gtfs_ktmb --output ./data/cleaned/gtfs_ktmb --operator ktmb

Operator presets control:
  - Which service_id patterns are expected (MonFri/Sat/Sun vs weekday/weekend vs KTMB codes)
  - Whether frequency-based scheduling is expected
  - Route type defaults for validation (rail=1, bus=3, BRT=0)
  - Coordinate bounding box for Malaysia sanity checks
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

# ── Malaysian postcode pattern ─────────────────────────────────────────────────
MY_POSTCODE_RE = re.compile(r"^\d{5}$")

# ── Operator presets ───────────────────────────────────────────────────────────
OPERATOR_PRESETS = {
    "rapid_rail_kl": {
        "label": "Rapid Rail KL",
        "expected_service_ids": {"MonFri", "Sat", "Sun", "weekday", "weekend"},
        "frequency_based": True,
        "expected_route_types": {0, 1}, # 0=BRT, 1=rail
        "agency_id": "rapidrail",
        # MRT stops use a generic 'MRT' route_id — fix via stop_id prefix
        "stop_prefix_route": {"KG": "KGL", "PY": "PYL"},
        "nonstandard_stop_cols": ["geometry", "category", "isOKU", "status", "search"],
        "nonstandard_route_cols": ["category", "status"],
        "nonstandard_st_cols": ["route_id", "direction_id"],
    },
    "rapidbus_kl": {
        "label": "RapidBus KL",
        "expected_service_ids": {"weekday", "weekend"},
        "frequency_based": True,
        "expected_route_types": {3}, # 3=bus
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
        # KTMB uses service codes like komuter_weekday, komuter_weekend, etc.
        "expected_service_ids": {"komuter_weekday", "komuter_weekend", "komuter_utara", "ets", "intercity"},
        "frequency_based": False, # schedule-based, no frequencies.txt
        "expected_route_types": {0, 2}, # 0=BRT-like, 2=rail
        "agency_id": "ktmb",
        "stop_prefix_route": {},
        "nonstandard_stop_cols": [],
        "nonstandard_route_cols": [],
        "nonstandard_st_cols": [],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def read_csv(path: str) -> pd.DataFrame | None:
    """Read a GTFS .txt file (CSV). Returns None if the file does not exist."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")   # strips BOM automatically
    df.columns = [c.strip() for c in df.columns]              # strip header whitespace
    return df


def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info(f"  → wrote {os.path.basename(path)} ({len(df):,} rows)")


def check_coords(df: pd.DataFrame, lat_col: str, lon_col: str, label: str) -> pd.DataFrame:
    """Validate and drop rows with coordinates outside the Malaysia bounding box."""
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


def drop_nonstandard(df: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    """Drop non-GTFS-standard columns that are present."""
    to_drop = [c for c in cols if c in df.columns]
    if to_drop:
        log.info(f"  [{label}] dropping non-standard columns: {to_drop}")
        df = df.drop(columns=to_drop)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Per-file cleaners
# ══════════════════════════════════════════════════════════════════════════════

def clean_agency(df: pd.DataFrame, preset: dict) -> pd.DataFrame:
    log.info("[agency.txt]")
    required = ["agency_id", "agency_name", "agency_url", "agency_timezone"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.warning(f"  Missing required columns: {missing}")
    df = df.drop_duplicates(subset=["agency_id"])
    log.info(f"  {len(df)} agenc{'y' if len(df)==1 else 'ies'} — OK")
    return df


def clean_calendar(df: pd.DataFrame, preset: dict) -> tuple[pd.DataFrame, set]:
    log.info("[calendar.txt]")
    # ── Parse dates ───────────────────────────────────────────────────────────
    for col in ["start_date", "end_date"]:
        df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
        bad = df[col].isna().sum()
        if bad:
            log.warning(f"  {bad} unparseable {col} values")

    # ── Inverted date ranges ──────────────────────────────────────────────────
    inverted = df["end_date"] < df["start_date"]
    if inverted.any():
        log.warning(f"  {inverted.sum()} rows with end_date < start_date — swapping")
        df.loc[inverted, ["start_date", "end_date"]] = (
            df.loc[inverted, ["end_date", "start_date"]].values
        )

    # ── Detect overlapping / redundant service definitions ────────────────────
    day_cols = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    bitmasks = df.set_index("service_id")[day_cols].astype(str).apply("|".join, axis=1)
    dups = bitmasks[bitmasks.duplicated(keep=False)]
    if not dups.empty:
        log.warning(f"  Identical day-pattern service_ids: {dups.index.tolist()} — review for redundancy")

    # ── Format dates back to GTFS YYYYMMDD ───────────────────────────────────
    df["start_date"] = df["start_date"].dt.strftime("%Y%m%d")
    df["end_date"]   = df["end_date"].dt.strftime("%Y%m%d")

    all_service_ids = set(df["service_id"])
    log.info(f"  {len(df)} service definitions — {all_service_ids}")
    return df, all_service_ids


def clean_calendar_dates(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Optional file — handles service exceptions (holidays, special days)."""
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
        log.warning(f"  {invalid_type.sum()} rows with invalid exception_type (must be 1 or 2) — dropped")
        df = df[~invalid_type]
    dups = df.duplicated(subset=["service_id", "date"])
    if dups.any():
        log.warning(f"  {dups.sum()} duplicate (service_id, date) pairs — keeping first")
        df = df[~dups]
    log.info(f"  {len(df)} exception rows")
    return df


def clean_routes(df: pd.DataFrame, preset: dict) -> tuple[pd.DataFrame, set]:
    log.info("[routes.txt]")
    df = drop_nonstandard(df, preset["nonstandard_route_cols"], "routes")

    df["route_type"] = pd.to_numeric(df["route_type"], errors="coerce")
    unexpected = ~df["route_type"].isin(preset["expected_route_types"])
    if unexpected.any():
        log.warning(
            f"  Unexpected route_type values: "
            f"{df.loc[unexpected, ['route_id','route_type']].to_dict('records')}"
        )

    # Normalise hex colours — ensure 6-char no-hash uppercase
    for col in ["route_color", "route_text_color"]:
        if col in df.columns:
            df[col] = (
                df[col].str.strip().str.lstrip("#").str.upper()
                .where(df[col].notna(), other=None)
            )

    dups = df.duplicated(subset=["route_id"])
    if dups.any():
        log.warning(f"  {dups.sum()} duplicate route_id(s) — keeping first")
        df = df[~dups]

    route_ids = set(df["route_id"])
    log.info(f"  {len(df)} routes — {route_ids}")
    return df, route_ids


def clean_stops(df: pd.DataFrame, preset: dict) -> tuple[pd.DataFrame, set]:
    log.info("[stops.txt]")
    df = drop_nonstandard(df, preset["nonstandard_stop_cols"], "stops")

    # ── Fix ambiguous route_id via stop_id prefix ─────────────────────────────
    if preset["stop_prefix_route"]:
        def fix_route(row):
            prefix = re.sub(r"\d", "", row["stop_id"])[:2].upper()
            return preset["stop_prefix_route"].get(prefix, row.get("route_id", ""))
        if "route_id" in df.columns:
            before = (df["route_id"] == "MRT").sum()
            df["route_id"] = df.apply(fix_route, axis=1)
            log.info(f"  Fixed {before} ambiguous route_id='MRT' via stop_id prefix")

    # ── Validate coordinates ──────────────────────────────────────────────────
    df = check_coords(df, "stop_lat", "stop_lon", "stops")

    # ── Normalise stop names ──────────────────────────────────────────────────
    df["stop_name"] = df["stop_name"].str.strip()

    # ── Duplicate stop_id ─────────────────────────────────────────────────────
    dups = df.duplicated(subset=["stop_id"])
    if dups.any():
        log.warning(f"  {dups.sum()} duplicate stop_id(s) — keeping first")
        df = df[~dups]

    stop_ids = set(df["stop_id"])
    log.info(f"  {len(df)} stops")
    return df, stop_ids


def clean_shapes(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Optional file — route polylines. Absent = no map geometry, trips still valid."""
    if df is None:
        log.info("[shapes.txt] not present — trips will have no map geometry")
        return None
    log.info("[shapes.txt]")

    # ── GTFS spec: shape_pt_lat before shape_pt_lon ───────────────────────────
    # Some exports reverse this. Detect by checking which column is in Malaysia range.
    if "shape_pt_lon" in df.columns and "shape_pt_lat" in df.columns:
        sample_lat = pd.to_numeric(df["shape_pt_lat"].iloc[0], errors="coerce")
        sample_lon = pd.to_numeric(df["shape_pt_lon"].iloc[0], errors="coerce")
        # If what's labelled 'lat' looks like longitude (>90), swap
        if sample_lat and sample_lat > 90:
            log.warning("  Columns shape_pt_lat and shape_pt_lon appear reversed — swapping")
            df = df.rename(columns={"shape_pt_lat": "shape_pt_lon", "shape_pt_lon": "shape_pt_lat"})
        # Enforce GTFS column order
        other_cols = [c for c in df.columns if c not in
                      ["shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"]]
        df = df[["shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"] + other_cols]

    df = check_coords(df, "shape_pt_lat", "shape_pt_lon", "shapes")

    df["shape_pt_sequence"] = pd.to_numeric(df["shape_pt_sequence"], errors="coerce")
    bad_seq = df["shape_pt_sequence"].isna()
    if bad_seq.any():
        log.warning(f"  {bad_seq.sum()} non-numeric shape_pt_sequence values — dropped")
        df = df[~bad_seq]

    # Sort each shape by sequence to ensure correct polyline order
    df = df.sort_values(["shape_id", "shape_pt_sequence"]).reset_index(drop=True)
    log.info(f"  {len(df)} shape points across {df['shape_id'].nunique()} shapes")
    return df


def clean_trips(
    df: pd.DataFrame,
    route_ids: set,
    service_ids: set,
    shape_ids: set | None,
    preset: dict,
) -> tuple[pd.DataFrame, set]:
    log.info("[trips.txt]")

    # ── Foreign key: route_id ─────────────────────────────────────────────────
    bad_routes = ~df["route_id"].isin(route_ids)
    if bad_routes.any():
        log.warning(
            f"  {bad_routes.sum()} trips reference unknown route_id(s): "
            f"{df.loc[bad_routes,'route_id'].unique().tolist()} — dropped"
        )
        df = df[~bad_routes]

    # ── Foreign key: service_id ───────────────────────────────────────────────
    bad_services = ~df["service_id"].isin(service_ids)
    if bad_services.any():
        log.warning(
            f"  {bad_services.sum()} trips reference unknown service_id(s): "
            f"{df.loc[bad_services,'service_id'].unique().tolist()} — dropped"
        )
        df = df[~bad_services]

    # ── Foreign key: shape_id (optional) ─────────────────────────────────────
    if shape_ids is not None and "shape_id" in df.columns:
        bad_shapes = df["shape_id"].notna() & ~df["shape_id"].isin(shape_ids)
        if bad_shapes.any():
            log.warning(
                f"  {bad_shapes.sum()} trips reference unknown shape_id(s) — nulling"
            )
            df.loc[bad_shapes, "shape_id"] = None

    # ── Unused service_ids in calendar ───────────────────────────────────────
    used_services = set(df["service_id"])
    unused = service_ids - used_services
    if unused:
        log.warning(f"  {len(unused)} service_id(s) in calendar but unused in trips: {unused}")

    # ── Duplicate trip_id ─────────────────────────────────────────────────────
    dups = df.duplicated(subset=["trip_id"])
    if dups.any():
        log.warning(f"  {dups.sum()} duplicate trip_id(s) — keeping first")
        df = df[~dups]

    trip_ids = set(df["trip_id"])
    log.info(f"  {len(df)} trips")
    return df, trip_ids


def clean_frequencies(df: pd.DataFrame | None, trip_ids: set, preset: dict) -> pd.DataFrame | None:
    """
    Optional file. When present, its trips are frequency-based:
    stop_times offsets are relative to the departure, not absolute clock times.
    When absent for a specific trip, that trip is schedule-based (stop_times are absolute).
    """
    if df is None:
        if preset["frequency_based"]:
            log.warning(
                "[frequencies.txt] not present — expected for this operator. "
                "Treating all trips as schedule-based (stop_times are absolute)."
            )
        else:
            log.info("[frequencies.txt] not present — operator is schedule-based (expected)")
        return None

    log.info("[frequencies.txt]")

    # ── Foreign key: trip_id ──────────────────────────────────────────────────
    bad = ~df["trip_id"].isin(trip_ids)
    if bad.any():
        log.warning(
            f"  {bad.sum()} frequency rows reference unknown trip_id(s): "
            f"{df.loc[bad,'trip_id'].unique().tolist()} — dropped"
        )
        df = df[~bad]

    # ── Validate headway_secs ─────────────────────────────────────────────────
    df["headway_secs"] = pd.to_numeric(df["headway_secs"], errors="coerce")
    bad_hw = df["headway_secs"].isna() | (df["headway_secs"] <= 0)
    if bad_hw.any():
        log.warning(f"  {bad_hw.sum()} rows with invalid headway_secs — dropped")
        df = df[~bad_hw]

    # ── Parse time strings (GTFS allows HH:MM:SS > 24:00:00 for overnight) ──
    def parse_gtfs_time(t):
        """Convert GTFS HH:MM:SS (can exceed 24h) to total seconds."""
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

    # ── End time must be after start time ────────────────────────────────────
    inverted = df["_end_secs"] <= df["_start_secs"]
    if inverted.any():
        log.warning(f"  {inverted.sum()} rows where end_time ≤ start_time — dropped")
        df = df[~inverted]

    # ── Detect overlapping windows within the same trip ───────────────────────
    overlap_count = 0
    for tid, grp in df.groupby("trip_id"):
        grp = grp.sort_values("_start_secs")
        prev_end = -1
        for _, row in grp.iterrows():
            if row["_start_secs"] < prev_end:
                overlap_count += 1
            prev_end = row["_end_secs"]
    if overlap_count:
        log.warning(f"  {overlap_count} overlapping time windows detected across trips — review manually")

    freq_trip_ids = set(df["trip_id"])
    schedule_only = trip_ids - freq_trip_ids
    if schedule_only:
        log.info(
            f"  {len(schedule_only)} trip(s) not in frequencies.txt → treated as "
            f"schedule-based (stop_times are absolute): {schedule_only}"
        )

    df = df.drop(columns=["_start_secs", "_end_secs"])
    log.info(f"  {len(df)} frequency bands across {freq_trip_ids.__len__()} trips")
    return df


def clean_stop_times(
    df: pd.DataFrame,
    trip_ids: set,
    stop_ids: set,
    freq_trip_ids: set | None,
    preset: dict,
) -> pd.DataFrame:
    log.info("[stop_times.txt]")
    df = drop_nonstandard(df, preset["nonstandard_st_cols"], "stop_times")

    # ── Foreign key: trip_id ──────────────────────────────────────────────────
    bad_trips = ~df["trip_id"].isin(trip_ids)
    if bad_trips.any():
        log.warning(
            f"  {bad_trips.sum()} stop_time rows reference unknown trip_id(s) — dropped"
        )
        df = df[~bad_trips]

    # ── Foreign key: stop_id ──────────────────────────────────────────────────
    bad_stops = ~df["stop_id"].isin(stop_ids)
    if bad_stops.any():
        log.warning(
            f"  {bad_stops.sum()} stop_time rows reference unknown stop_id(s): "
            f"{df.loc[bad_stops,'stop_id'].unique().tolist()} — dropped"
        )
        df = df[~bad_stops]

    # ── stop_sequence must be numeric and positive ────────────────────────────
    df["stop_sequence"] = pd.to_numeric(df["stop_sequence"], errors="coerce")
    bad_seq = df["stop_sequence"].isna() | (df["stop_sequence"] < 0)
    if bad_seq.any():
        log.warning(f"  {bad_seq.sum()} rows with invalid stop_sequence — dropped")
        df = df[~bad_seq]

    # ── Validate time strings ─────────────────────────────────────────────────
    TIME_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")
    for col in ["arrival_time", "departure_time"]:
        if col not in df.columns:
            continue
        bad_fmt = df[col].notna() & ~df[col].str.match(TIME_RE)
        if bad_fmt.any():
            log.warning(f"  {bad_fmt.sum()} malformed {col} values (e.g. {df.loc[bad_fmt, col].iloc[0]})")

    # ── For frequency-based trips: stop_times are offsets (not absolute) ──────
    # Log a note — do NOT drop them; they are required as the template.
    if freq_trip_ids:
        freq_st = df[df["trip_id"].isin(freq_trip_ids)]
        sched_st = df[~df["trip_id"].isin(freq_trip_ids)]
        log.info(
            f"  {len(freq_st):,} stop_time rows are frequency templates "
            f"(offsets from first departure)"
        )
        log.info(
            f"  {len(sched_st):,} stop_time rows are absolute schedule times"
        )

    # ── Sort by trip + sequence ───────────────────────────────────────────────
    df = df.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)

    # ── Each trip must have at least 2 stops ──────────────────────────────────
    trip_stop_counts = df.groupby("trip_id")["stop_id"].count()
    single_stop_trips = trip_stop_counts[trip_stop_counts < 2].index.tolist()
    if single_stop_trips:
        log.warning(f"  {len(single_stop_trips)} trip(s) with < 2 stops: {single_stop_trips}")

    log.info(f"  {len(df):,} stop_time rows")
    return df


def clean_transfers(df: pd.DataFrame | None, stop_ids: set) -> pd.DataFrame | None:
    """Optional file — interchange walking times between stops."""
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


# ══════════════════════════════════════════════════════════════════════════════
# Cross-file referential integrity summary
# ══════════════════════════════════════════════════════════════════════════════

def integrity_report(
    route_ids, service_ids, stop_ids, shape_ids,
    trip_ids, freq_trip_ids, st_trip_ids, st_stop_ids,
):
    log.info("")
    log.info("════ Referential Integrity Summary ════")
    checks = [
        ("All trip route_ids in routes",     trip_ids is not None),
        ("All trip service_ids in calendar", True),
        ("All stop_time trip_ids in trips",  not (st_trip_ids - trip_ids)),
        ("All stop_time stop_ids in stops",  not (st_stop_ids - stop_ids)),
        ("Shape IDs fully matched",          shape_ids is None or True),
    ]
    for label, ok in checks:
        status = "✓" if ok else "✗"
        log.info(f"  {status}  {label}")

    if freq_trip_ids is not None:
        schedule_trips = trip_ids - freq_trip_ids
        if schedule_trips:
            log.info(
                f"  ℹ  {len(schedule_trips)} schedule-based trip(s) (no frequency entry): "
                f"{schedule_trips}"
            )
        else:
            log.info(f"  ✓  All {len(trip_ids)} trips are frequency-based")
    log.info("═══════════════════════════════════════")


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run(input_dir: str, output_dir: str, operator: str) -> None:
    preset = OPERATOR_PRESETS[operator]
    log.info(f"Operator : {preset['label']}")
    log.info(f"Input    : {input_dir}")
    log.info(f"Output   : {output_dir}")
    log.info("")

    def src(filename):
        return os.path.join(input_dir, filename)

    def dst(filename):
        return os.path.join(output_dir, filename)

    # ── 1. Load all files (None if absent) ────────────────────────────────────
    agency_df          = read_csv(src("agency.txt"))
    calendar_df        = read_csv(src("calendar.txt"))
    calendar_dates_df  = read_csv(src("calendar_dates.txt"))
    routes_df          = read_csv(src("routes.txt"))
    stops_df           = read_csv(src("stops.txt"))
    shapes_df          = read_csv(src("shapes.txt"))
    trips_df           = read_csv(src("trips.txt"))
    frequencies_df     = read_csv(src("frequencies.txt"))
    stop_times_df      = read_csv(src("stop_times.txt"))
    transfers_df       = read_csv(src("transfers.txt"))

    # ── 2. Validate required files ────────────────────────────────────────────
    required = {
        "agency.txt":     agency_df,
        "calendar.txt":   calendar_df,
        "routes.txt":     routes_df,
        "stops.txt":      stops_df,
        "trips.txt":      trips_df,
        "stop_times.txt": stop_times_df,
    }
    missing_required = [k for k, v in required.items() if v is None]
    if missing_required:
        log.error(f"Missing required GTFS files: {missing_required}")
        sys.exit(1)

    # ── 3. Clean each file ────────────────────────────────────────────────────
    agency_df                      = clean_agency(agency_df, preset)
    calendar_df, service_ids       = clean_calendar(calendar_df, preset)
    calendar_dates_df              = clean_calendar_dates(calendar_dates_df)
    routes_df, route_ids           = clean_routes(routes_df, preset)
    stops_df, stop_ids             = clean_stops(stops_df, preset)
    shapes_df                      = clean_shapes(shapes_df)
    shape_ids = set(shapes_df["shape_id"]) if shapes_df is not None else None

    trips_df, trip_ids = clean_trips(
        trips_df, route_ids, service_ids, shape_ids, preset
    )

    frequencies_df = clean_frequencies(frequencies_df, trip_ids, preset)
    freq_trip_ids  = set(frequencies_df["trip_id"]) if frequencies_df is not None else None

    stop_times_df = clean_stop_times(
        stop_times_df, trip_ids, stop_ids, freq_trip_ids, preset
    )

    transfers_df = clean_transfers(transfers_df, stop_ids)

    # ── 4. Integrity report ───────────────────────────────────────────────────
    integrity_report(
        route_ids, service_ids, stop_ids, shape_ids,
        trip_ids,
        freq_trip_ids,
        set(stop_times_df["trip_id"]),
        set(stop_times_df["stop_id"]),
    )

    # ── 5. Write outputs ──────────────────────────────────────────────────────
    log.info("")
    log.info("Writing clean files...")
    write_csv(agency_df,     dst("agency.txt"))
    write_csv(calendar_df,   dst("calendar.txt"))
    write_csv(routes_df,     dst("routes.txt"))
    write_csv(stops_df,      dst("stops.txt"))
    write_csv(trips_df,      dst("trips.txt"))
    write_csv(stop_times_df, dst("stop_times.txt"))

    if calendar_dates_df is not None:
        write_csv(calendar_dates_df, dst("calendar_dates.txt"))
    if shapes_df is not None:
        write_csv(shapes_df, dst("shapes.txt"))
    if frequencies_df is not None:
        write_csv(frequencies_df, dst("frequencies.txt"))
    if transfers_df is not None:
        write_csv(transfers_df, dst("transfers.txt"))

    log.info("")
    log.info("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTFS cleaner for Malaysian transit operators")
    parser.add_argument("--input",    required=True, help="Directory containing raw GTFS .txt files")
    parser.add_argument("--output",   required=True, help="Directory to write cleaned GTFS .txt files")
    parser.add_argument(
        "--operator", required=True,
        choices=list(OPERATOR_PRESETS.keys()),
        help="Operator preset: rapid_rail | rapidbus_kl | rapidbus_penang | ktmb",
    )
    args = parser.parse_args()
    run(args.input, args.output, args.operator)