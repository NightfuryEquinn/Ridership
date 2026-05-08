"""
spatial_features.py
-------------------
Computes spatial features per GADM zone:
  - POI density & diversity
  - Population density (log-transformed)
  - GTFS stop density
Outputs: spatial_features.parquet
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape

warnings.filterwarnings("ignore", category=FutureWarning)

GADM_PATH  = Path("data/cleaned/gadm_mys_l1_clean.geojson")
POI_PATH   = Path("data/cleaned/osm_pois_clean.json")
POP_PATH   = Path("data/cleaned/population_density_clean.csv")
GTFS_DIRS  = [
    Path("data/cleaned/gtfs_ktmb"),
    Path("data/cleaned/gtfs_rapid_bus_penang"),
    Path("data/cleaned/gtfs_rapid_rail_kl"),
]
OUT_PATH   = Path("data/features/spatial_features.parquet")

GADM_ID_COL   = "GID_1"     # adjust if your file uses a different field
GADM_NAME_COL = "NAME_1"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_gadm(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path).to_crs("EPSG:3857")
    gdf["area_km2"] = gdf.geometry.area / 1e6
    return gdf


def load_pois(path: Path) -> gpd.GeoDataFrame:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    features = raw if isinstance(raw, list) else raw.get("features", raw.get("elements", []))
    rows = []
    for feat in features:
        geom = feat.get("geometry") or {}
        tags = feat.get("tags") or feat.get("properties") or {}
        rows.append({
            "geometry": shape(geom) if "type" in geom else None,
            "amenity": tags.get("amenity", ""),
            "shop": tags.get("shop", ""),
            "leisure": tags.get("leisure", ""),
            "tourism": tags.get("tourism", ""),
        })
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326").dropna(subset=["geometry"]).to_crs("EPSG:3857")
    gdf["poi_type"] = (
        gdf["amenity"].where(gdf["amenity"] != "")
        .fillna(gdf["shop"].where(gdf["shop"] != ""))
        .fillna(gdf["leisure"].where(gdf["leisure"] != ""))
        .fillna(gdf["tourism"].where(gdf["tourism"] != ""))
        .fillna("other")
    )
    return gdf


def load_population(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def load_gtfs_stops(gtfs_dirs: list[Path]) -> gpd.GeoDataFrame:
    frames = []
    for d in gtfs_dirs:
        stops_file = d / "stops.txt"
        if stops_file.exists():
            s = pd.read_csv(stops_file)
            s["source"] = d.name
            frames.append(s)
    if not frames:
        return gpd.GeoDataFrame(columns=["stop_id", "stop_lat", "stop_lon", "geometry"])
    stops = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    return gdf


# ---------------------------------------------------------------------------
# Feature computers
# ---------------------------------------------------------------------------

def compute_poi_features(gadm: gpd.GeoDataFrame, pois: gpd.GeoDataFrame) -> pd.DataFrame:
    joined = gpd.sjoin(pois, gadm[[GADM_ID_COL, "geometry", "area_km2"]], how="inner", predicate="within")

    # Density = count per km²
    density = joined.groupby(GADM_ID_COL).size().rename("poi_count")
    area_map = gadm.set_index(GADM_ID_COL)["area_km2"]
    density_df = density.to_frame().join(area_map)
    density_df["poi_density_per_km2"] = density_df["poi_count"] / density_df["area_km2"]

    # Diversity = number of unique POI types (Shannon entropy)
    type_counts = joined.groupby([GADM_ID_COL, "poi_type"]).size().unstack(fill_value=0)
    probs = type_counts.div(type_counts.sum(axis=1), axis=0)
    shannon = (-probs * np.log(probs + 1e-9)).sum(axis=1).rename("poi_diversity_shannon")
    diversity_df = shannon.to_frame()
    diversity_df["poi_n_types"] = (type_counts > 0).sum(axis=1)

    return density_df[["poi_count", "poi_density_per_km2"]].join(diversity_df)


def compute_population_features(gadm: gpd.GeoDataFrame, pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects pop_df to have columns matching GADM_ID_COL + a population column.
    Falls back to spatial join if lat/lon columns are present.
    If density column is provided (and no population column), uses density directly.
    """
    # Try to find a population column
    pop_cols = [c for c in pop_df.columns if "pop" in c.lower()]
    density_cols = [c for c in pop_df.columns if "density" in c.lower()]

    if pop_cols:
        pop_col = pop_cols[0]
        use_density = False
    elif density_cols:
        # Use density column directly (already per km2)
        pop_col = density_cols[0]
        use_density = True
    else:
        # Fallback: use first numeric column if lat/lon present, else error
        if {"latitude", "longitude"}.issubset(pop_df.columns):
            numeric_cols = pop_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                pop_col = numeric_cols[0]
                use_density = True  # assume it's density
            else:
                raise ValueError("Population CSV has lat/lon but no numeric columns.")
        else:
            raise ValueError("Population CSV needs either a population column, density column, or latitude/longitude.")

    if GADM_ID_COL in pop_df.columns:
        if use_density:
            merged = gadm[[GADM_ID_COL, "area_km2"]].merge(pop_df[[GADM_ID_COL, pop_col]], on=GADM_ID_COL, how="left")
            merged["pop_density_per_km2"] = merged[pop_col]
        else:
            merged = gadm[[GADM_ID_COL, "area_km2"]].merge(pop_df[[GADM_ID_COL, pop_col]], on=GADM_ID_COL, how="left")
            merged["pop_density_per_km2"] = merged[pop_col] / merged["area_km2"]
    elif {"latitude", "longitude"}.issubset(pop_df.columns):
        pop_gdf = gpd.GeoDataFrame(
            pop_df,
            geometry=gpd.points_from_xy(pop_df["longitude"], pop_df["latitude"]),
            crs="EPSG:4326",
        ).to_crs("EPSG:3857")
        joined = gpd.sjoin(pop_gdf, gadm[[GADM_ID_COL, "area_km2", "geometry"]], how="inner", predicate="within")
        if use_density:
            merged = joined.groupby(GADM_ID_COL)[pop_col].mean().reset_index()
        else:
            merged = joined.groupby(GADM_ID_COL)[pop_col].mean().reset_index()
        merged = gadm[[GADM_ID_COL, "area_km2"]].merge(merged, on=GADM_ID_COL, how="left")
        if use_density:
            merged["pop_density_per_km2"] = merged[pop_col]
        else:
            merged["pop_density_per_km2"] = merged[pop_col] / merged["area_km2"]
    else:
        raise ValueError("Population CSV needs either GID_1 or latitude/longitude columns.")

    merged = merged.set_index(GADM_ID_COL)
    merged["pop_log_density"] = np.log1p(merged["pop_density_per_km2"])
    return merged[["pop_density_per_km2", "pop_log_density"]]


def compute_gtfs_stop_density(gadm: gpd.GeoDataFrame, stops: gpd.GeoDataFrame) -> pd.DataFrame:
    if stops.empty:
        return pd.DataFrame(index=gadm[GADM_ID_COL])

    joined = gpd.sjoin(stops, gadm[[GADM_ID_COL, "area_km2", "geometry"]], how="inner", predicate="within")
    stop_count = joined.groupby(GADM_ID_COL).size().rename("gtfs_stop_count")
    area_map = gadm.set_index(GADM_ID_COL)["area_km2"]
    df = stop_count.to_frame().join(area_map)
    df["gtfs_stop_density_per_km2"] = df["gtfs_stop_count"] / df["area_km2"]
    return df[["gtfs_stop_count", "gtfs_stop_density_per_km2"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    gadm_path: Path = GADM_PATH,
    poi_path: Path  = POI_PATH,
    pop_path: Path  = POP_PATH,
    gtfs_dirs: list[Path] = GTFS_DIRS,
    out_path: Path  = OUT_PATH,
) -> pd.DataFrame:
    print("[spatial] Loading GADM ...")
    gadm = load_gadm(gadm_path)

    print("[spatial] Loading POIs ...")
    pois = load_pois(poi_path)
    poi_feats = compute_poi_features(gadm, pois)

    print("[spatial] Loading population ...")
    pop_df = load_population(pop_path)
    pop_feats = compute_population_features(gadm, pop_df)

    print("[spatial] Loading GTFS stops ...")
    stops = load_gtfs_stops(gtfs_dirs)
    gtfs_feats = compute_gtfs_stop_density(gadm, stops)

    # Join all features on GADM zone ID
    result = (
        gadm[[GADM_ID_COL, GADM_NAME_COL, "area_km2"]]
        .set_index(GADM_ID_COL)
        .join(poi_feats, how="left")
        .join(pop_feats, how="left")
        .join(gtfs_feats, how="left")
        .fillna(0)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path)
    print(f"[spatial] Saved -> {out_path}  shape={result.shape}")
    return result


if __name__ == "__main__":
    run()