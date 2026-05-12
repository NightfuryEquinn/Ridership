"""
osm.py  — Cleaning + ST-model preparation
Changes vs original:
  - After cleaning, POIs are aggregated into catchment-area feature counts
    around each GTFS stop (radius = 500 m by default).
  - Output: data/cleaned/poi_counts_at_stops.csv
      Columns: stop_id, poi_total, poi_transport, poi_food, poi_retail,
               poi_education, poi_healthcare, poi_leisure, poi_other
  - These become static node features in the GCN pipeline
    (alongside population density), enriching the spatial embedding.

All original cleaning logic is preserved unchanged.
"""

import json, re, glob
from collections import defaultdict

import numpy as np
import pandas as pd


# ── Module-level constants ────────────────────────────────────────────────────

CATCHMENT_RADIUS_M = 500    # metres — tunable; 300–800 m is typical for transit

# OSM tag → category mapping
CATEGORY_MAP = {
    "bus_station": "transport", "bus_stop": "transport",
    "ferry_terminal": "transport", "taxi": "transport",
    "restaurant": "food",  "cafe": "food",   "fast_food": "food",
    "food_court": "food",  "bar": "food",    "pub": "food",
    "marketplace": "retail", "supermarket": "retail",
    "convenience": "retail", "mall": "retail",
    "school": "education", "university": "education",
    "college": "education", "kindergarten": "education",
    "hospital": "healthcare", "clinic": "healthcare",
    "pharmacy": "healthcare", "doctors": "healthcare",
    "park": "leisure",   "cinema": "leisure",
    "gym": "leisure",    "stadium": "leisure",
    "sports_centre": "leisure",
}


# ── Helper functions ──────────────────────────────────────────────────────────

def get_coords(elem):
    if elem["type"] == "node":
        return elem.get("lat"), elem.get("lon")
    elif elem["type"] == "way":
        center = elem.get("center", {})
        return center.get("lat"), center.get("lon")
    return None, None


MY_POSTCODE_RE = re.compile(r"^\d{5}$")

def clean_postcode(pc):
    if pc is None:
        return None
    pc = pc.strip()
    return pc if MY_POSTCODE_RE.match(pc) else None


def tag_to_category(tags: dict) -> str:
    for key in ("amenity", "shop", "leisure", "tourism"):
        val = tags.get(key, "")
        if val in CATEGORY_MAP:
            return CATEGORY_MAP[val]
    return "other"


def aggregate_poi_counts(
    cleaned_pois: list,
    stops_df:     pd.DataFrame,
    radius_m:     float = CATCHMENT_RADIUS_M,
) -> pd.DataFrame:
    """
    For each stop in stops_df, count POIs by category within radius_m metres.
    Uses a BallTree for efficient radius search (O(N log N) vs O(N²) brute force).
    """
    try:
        from sklearn.neighbors import BallTree
    except ImportError:
        print("[WARN] scikit-learn not installed — POI aggregation skipped. "
              "Run: pip install scikit-learn")
        return pd.DataFrame()

    poi_lats = np.array([e["lat"] for e in cleaned_pois], dtype=np.float64)
    poi_lons = np.array([e["lon"] for e in cleaned_pois], dtype=np.float64)
    poi_cats = [tag_to_category(e["tags"]) for e in cleaned_pois]

    poi_coords = np.radians(np.column_stack([poi_lats, poi_lons]))
    tree = BallTree(poi_coords, metric="haversine")

    EARTH_RADIUS_M = 6_371_000
    radius_rad = radius_m / EARTH_RADIUS_M

    categories = ["transport", "food", "retail", "education",
                  "healthcare", "leisure", "other"]

    records = []
    stop_coords = np.radians(stops_df[["stop_lat", "stop_lon"]].values.astype(np.float64))
    indices_list = tree.query_radius(stop_coords, r=radius_rad)

    for i, (_, stop_row) in enumerate(stops_df.iterrows()):
        idxs = indices_list[i]
        cats_in_radius = [poi_cats[j] for j in idxs]
        row = {"stop_id": stop_row["stop_id"]}
        row["poi_total"] = len(idxs)
        for cat in categories:
            row[f"poi_{cat}"] = cats_in_radius.count(cat)
        records.append(row)

    return pd.DataFrame(records)


def main():
    # ── Load ──────────────────────────────────────────────────────────────────────
    with open("data/raw/osm_pois.json") as f:
        osm = json.load(f)

    elements = osm["elements"]
    print(f"=== OSM POIs ===")
    print(f"Raw element count: {len(elements)}")

    # ── 1–4. Deduplicate and build cleaned list ───────────────────────────────────
    seen_ids = set()
    cleaned  = []
    stats    = defaultdict(int)

    for elem in elements:
        eid = elem["id"]
        if eid in seen_ids:
            stats["duplicate_id_skipped"] += 1
            continue
        seen_ids.add(eid)

        tags = elem.get("tags", {})
        lat, lon = get_coords(elem)
        if lat is None or lon is None:
            stats["no_coords_skipped"] += 1
            continue

        if "addr:postcode" in tags:
            cleaned_pc = clean_postcode(tags["addr:postcode"])
            if cleaned_pc is None:
                stats["bad_postcode_removed"] += 1
                tags = {k: v for k, v in tags.items() if k != "addr:postcode"}
            else:
                tags["addr:postcode"] = cleaned_pc

        new_elem = {"id": eid, "type": elem["type"], "lat": lat, "lon": lon, "tags": tags}
        cleaned.append(new_elem)
        stats["kept"] += 1

    no_name = sum(1 for e in cleaned if "name" not in e["tags"])
    stats["no_name_tag"] = no_name

    # ── 5. Report ─────────────────────────────────────────────────────────────────
    print(f"\nCleaning summary:")
    print(f"  Kept:                        {stats['kept']:,}")
    print(f"  Duplicate IDs skipped:       {stats['duplicate_id_skipped']:,}")
    print(f"  No-coordinate skipped:       {stats['no_coords_skipped']:,}")
    print(f"  Malformed postcodes removed: {stats['bad_postcode_removed']:,}")
    print(f"  Elements without name tag:   {stats['no_name_tag']:,} "
          f"({100*stats['no_name_tag']/max(stats['kept'],1):.1f}%)")

    # ── 6. Export cleaned JSON ────────────────────────────────────────────────────
    out = {**{k: v for k, v in osm.items() if k != "elements"}, "elements": cleaned}
    with open("data/cleaned/osm_pois_clean.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nExported: data/cleaned/osm_pois_clean.json")

    # ── POI catchment aggregation around GTFS stops ───────────────────────────────
    stop_frames = []
    for path in glob.glob("data/cleaned/gtfs_*/gtfs_stop_nodes_*.csv"):
        stop_frames.append(pd.read_csv(path))

    if stop_frames:
        df_stops = pd.concat(stop_frames, ignore_index=True).drop_duplicates("stop_id")
        df_stops["stop_lat"] = pd.to_numeric(df_stops["stop_lat"], errors="coerce")
        df_stops["stop_lon"] = pd.to_numeric(df_stops["stop_lon"], errors="coerce")
        df_stops = df_stops.dropna(subset=["stop_lat", "stop_lon"])

        print(f"\n=== POI Catchment Aggregation ({CATCHMENT_RADIUS_M}m radius) ===")
        print(f"Stops: {len(df_stops)},  POIs: {len(cleaned)}")

        df_poi_counts = aggregate_poi_counts(cleaned, df_stops, CATCHMENT_RADIUS_M)

        if not df_poi_counts.empty:
            count_cols = [c for c in df_poi_counts.columns if c.startswith("poi_")]
            for col in count_cols:
                df_poi_counts[f"{col}_log"] = np.log1p(df_poi_counts[col])

            print(f"\nPOI count summary (per stop):")
            print(df_poi_counts[count_cols].describe().round(2))

            df_poi_counts.to_csv("data/cleaned/poi_counts_at_stops.csv", index=False)
            print(f"\nExported: data/cleaned/poi_counts_at_stops.csv")
            print("  Join to stop node table on stop_id for GCN node features.")
        else:
            print("[SKIP] POI aggregation returned empty — check sklearn installation.")
    else:
        print("\n[SKIP] No GTFS stop node tables found — run gtfs.py first.")


if __name__ == "__main__":
    main()
