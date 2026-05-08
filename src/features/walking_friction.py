"""
walking_friction.py
-------------------
Extracts walking friction impedance values for GTFS stops and GADM zones
from walking_friction_clean.tif.
Outputs: walking_friction_weights.parquet
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform_geom

warnings.filterwarnings("ignore")

FRICTION_PATH = Path("data/cleaned/walking_friction_clean.tif")
GADM_PATH     = Path("data/cleaned/gadm_mys_l1_clean.geojson")
GTFS_DIRS     = [
    Path("data/cleaned/gtfs_ktmb"),
    Path("data/cleaned/gtfs_rapid_bus_penang"),
    Path("data/cleaned/gtfs_rapid_rail_kl"),
]
OUT_PATH = Path("data/features/walking_friction_weights.parquet")

GADM_ID_COL = "GID_1"
NODATA_VAL  = -9999.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gtfs_stops(gtfs_dirs: list[Path]) -> gpd.GeoDataFrame:
    frames = []
    for d in gtfs_dirs:
        stops_file = d / "stops.txt"
        if stops_file.exists():
            s = pd.read_csv(stops_file)
            s["source"] = d.name
            frames.append(s)
    if not frames:
        return gpd.GeoDataFrame(columns=["stop_id", "stop_lat", "stop_lon", "source", "geometry"])
    stops = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["stop_id"])
    gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    )
    return gdf


def sample_raster_at_points(
    src: rasterio.DatasetReader, gdf: gpd.GeoDataFrame
) -> np.ndarray:
    """Sample raster band 1 at each point in gdf (reprojected to raster CRS)."""
    raster_crs = src.crs.to_epsg()
    gdf_proj = gdf.to_crs(epsg=raster_crs) if gdf.crs.to_epsg() != raster_crs else gdf

    values = []
    band = src.read(1)
    nodata = src.nodata if src.nodata is not None else NODATA_VAL

    for geom in gdf_proj.geometry:
        try:
            row, col = rowcol(src.transform, geom.x, geom.y)
            val = float(band[row, col])
            val = np.nan if val == nodata else val
        except (IndexError, ValueError):
            val = np.nan
        values.append(val)
    return np.array(values)


def sample_raster_zonal(
    src: rasterio.DatasetReader, gadm: gpd.GeoDataFrame
) -> pd.Series:
    """Compute mean friction within each GADM polygon using rasterio.mask."""
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import mapping

    raster_crs_str = src.crs.to_string()
    gadm_proj = gadm.to_crs(src.crs)

    means = {}
    nodata = src.nodata if src.nodata is not None else NODATA_VAL

    for _, row in gadm_proj.iterrows():
        geom = [mapping(row.geometry)]
        try:
            out_image, _ = rio_mask(src, geom, crop=True, nodata=nodata)
            data = out_image[0].astype(float)
            data[data == nodata] = np.nan
            means[row[GADM_ID_COL]] = float(np.nanmean(data)) if not np.all(np.isnan(data)) else np.nan
        except Exception:
            means[row[GADM_ID_COL]] = np.nan

    return pd.Series(means, name="friction_mean")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    friction_path: Path = FRICTION_PATH,
    gadm_path: Path     = GADM_PATH,
    gtfs_dirs: list[Path] = GTFS_DIRS,
    out_path: Path      = OUT_PATH,
) -> pd.DataFrame:

    print(f"[friction] Opening raster {friction_path}")
    src = rasterio.open(friction_path)

    # ---- 1. GTFS stop-level friction ----------------------------------------
    print("[friction] Loading GTFS stops ...")
    stops = load_gtfs_stops(gtfs_dirs)

    stop_friction = pd.DataFrame({"stop_id": stops["stop_id"].values})
    stop_friction["source"] = stops["source"].values
    stop_friction["stop_lat"] = stops["stop_lat"].values
    stop_friction["stop_lon"] = stops["stop_lon"].values
    stop_friction["friction_value"] = sample_raster_at_points(src, stops)

    # Normalize to [0, 1] as an edge-weight proxy (higher friction -> harder)
    fmin = np.nanmin(stop_friction["friction_value"])
    fmax = np.nanmax(stop_friction["friction_value"])
    stop_friction["friction_normalized"] = (stop_friction["friction_value"] - fmin) / (fmax - fmin + 1e-9)

    # Invert: use as *passability* weight for graph edges (1 = easiest)
    stop_friction["passability_weight"] = 1.0 - stop_friction["friction_normalized"]

    # ---- 2. GADM zone-level zonal statistics --------------------------------
    print("[friction] Computing zonal means per GADM zone ...")
    gadm = gpd.read_file(gadm_path)
    zone_friction = sample_raster_zonal(src, gadm).to_frame()
    zone_friction["friction_norm_zone"] = (zone_friction["friction_mean"] - fmin) / (fmax - fmin + 1e-9)
    zone_friction["passability_zone"] = 1.0 - zone_friction["friction_norm_zone"]
    zone_friction.index.name = GADM_ID_COL

    src.close()

    # ---- 3. Join stop -> GADM zone -------------------------------------------
    stops_gdf = gpd.GeoDataFrame(
        stop_friction,
        geometry=gpd.points_from_xy(stop_friction["stop_lon"], stop_friction["stop_lat"]),
        crs="EPSG:4326",
    )
    gadm_proj = gadm[[GADM_ID_COL, "geometry"]].to_crs("EPSG:4326")
    stops_with_zone = gpd.sjoin(stops_gdf, gadm_proj, how="left", predicate="within")
    stop_friction[GADM_ID_COL] = stops_with_zone[GADM_ID_COL].values

    # ---- 4. Output: multi-indexed (entity, type) ----------------------------
    # Flatten: stops table + zone table concatenated
    stop_out = stop_friction.set_index("stop_id")
    stop_out["entity_type"] = "gtfs_stop"

    zone_out = zone_friction.copy()
    zone_out["entity_type"] = "gadm_zone"
    zone_out.index.name = "entity_id"
    stop_out.index.name = "entity_id"

    result = pd.concat([stop_out.rename_axis("entity_id"), zone_out])
    result.index = result.index.astype(str)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path)
    print(f"[friction] Saved -> {out_path}  shape={result.shape}")
    return result


if __name__ == "__main__":
    run()