"""
population.py  — Cleaning + ST-model preparation
Changes vs original:
  - Added aggregation to state-level totals using GADM boundaries,
    producing a 16-row table keyed by GID_1 that can be joined to the
    state node table in the GCN pipeline.
  - Added stop-level sampling (sample the raster at each GTFS stop
    coordinate), producing a per-stop static feature used as a GCN
    node attribute in sequence_builder.py.
  - The raw grid CSV is still cleaned and exported as before.
"""

import glob
import pandas as pd
import numpy as np

# Optional spatial dependencies (needed for aggregation steps)
try:
    import geopandas as gpd
    from shapely.geometry import Point
    SPATIAL_AVAILABLE = True
except ImportError:
    SPATIAL_AVAILABLE = False
    print("[WARN] geopandas/shapely not installed — "
          "state-level and stop-level aggregation will be skipped. "
          "Run: pip install geopandas shapely")


def main():
    # ── Load ──────────────────────────────────────────────────────────────────────
    df = pd.read_csv("data/raw/malaysia_population_density_2020.csv")

    print("=== Raw shape ===")
    print(df.shape)
    print(df.dtypes)
    print(df.head())

    # ── 1. Rename opaque columns ──────────────────────────────────────────────────
    df.columns = ["longitude", "latitude", "density_per_km2"]

    # ── 2. Validate coordinate bounds ────────────────────────────────────────────
    BOUNDS = dict(lon=(99.6, 119.3), lat=(0.9, 7.4))
    out_lon = ~df["longitude"].between(*BOUNDS["lon"])
    out_lat = ~df["latitude"].between(*BOUNDS["lat"])
    print(f"\nCoordinates out of Malaysia bounding box: lon={out_lon.sum()}, lat={out_lat.sum()}")
    df = df[~(out_lon | out_lat)].copy()

    # ── 3. Validate density values ────────────────────────────────────────────────
    print("\nDensity stats:")
    print(df["density_per_km2"].describe().round(4))
    neg = (df["density_per_km2"] < 0).sum()
    print(f"Negative density values: {neg}")

    # ── 4. Log and clip transforms ────────────────────────────────────────────────
    p99 = df["density_per_km2"].quantile(0.99)
    df["density_log"]     = np.log1p(df["density_per_km2"])
    df["density_clipped"] = df["density_per_km2"].clip(upper=p99)

    print(f"\n99th-percentile clip threshold: {p99:.2f} per km²")
    print(f"Log-transform range: {df['density_log'].min():.4f} → {df['density_log'].max():.4f}")

    # ── 5. Export grid-level clean CSV (unchanged) ────────────────────────────────
    print(f"\nFinal shape: {df.shape}")
    print(f"Nulls: {df.isnull().sum().sum()}")
    df.to_csv("data/cleaned/population_density_clean.csv", index=False)
    print("Exported: data/cleaned/population_density_clean.csv")

    # ══════════════════════════════════════════════════════════════════════════════
    # State-level aggregation
    # ══════════════════════════════════════════════════════════════════════════════
    if SPATIAL_AVAILABLE:
        print("\n=== State-level aggregation ===")
        try:
            gadm = gpd.read_file("data/cleaned/gadm_mys_l1_clean.geojson")
            gadm = gadm[["GID_1", "NAME_1", "geometry"]]

            gdf_pop = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                crs="EPSG:4326"
            )
            gadm = gadm.to_crs("EPSG:4326")

            joined = gpd.sjoin(gdf_pop, gadm, how="left", predicate="within")

            state_pop = joined.groupby(["GID_1", "NAME_1"]).agg(
                density_mean_per_km2  = ("density_per_km2", "mean"),
                density_median_per_km2= ("density_per_km2", "median"),
                density_log_mean      = ("density_log",     "mean"),
                density_log_median    = ("density_log",     "median"),
                n_grid_cells          = ("density_per_km2", "count"),
            ).reset_index()

            state_pop.to_csv("data/cleaned/population_by_state.csv", index=False)
            print(f"State-level rows: {len(state_pop)}")
            print(state_pop[["GID_1", "NAME_1", "density_mean_per_km2"]].to_string(index=False))
            print("Exported: data/cleaned/population_by_state.csv")

        except Exception as e:
            print(f"[WARN] State aggregation failed: {e}")
    else:
        print("\n[SKIP] State-level aggregation requires geopandas.")

    # ══════════════════════════════════════════════════════════════════════════════
    # Stop-level sampling
    # ══════════════════════════════════════════════════════════════════════════════
    if SPATIAL_AVAILABLE:
        print("\n=== Stop-level population sampling ===")

        stop_frames = []
        for path in glob.glob("data/cleaned/gtfs_*/gtfs_stop_nodes_*.csv"):
            stop_frames.append(pd.read_csv(path))

        if stop_frames:
            df_stops = pd.concat(stop_frames, ignore_index=True).drop_duplicates("stop_id")
            df_stops["stop_lat"] = pd.to_numeric(df_stops["stop_lat"], errors="coerce")
            df_stops["stop_lon"] = pd.to_numeric(df_stops["stop_lon"], errors="coerce")
            df_stops = df_stops.dropna(subset=["stop_lat", "stop_lon"])

            try:
                from sklearn.neighbors import BallTree

                grid_coords = np.radians(df[["latitude", "longitude"]].values)
                tree = BallTree(grid_coords, metric="haversine")

                stop_coords = np.radians(df_stops[["stop_lat", "stop_lon"]].values)
                dist, idx   = tree.query(stop_coords, k=1)

                df_stops = df_stops.copy()
                df_stops["nearest_grid_dist_km"] = dist.flatten() * 6371
                df_stops["density_per_km2"]      = df["density_per_km2"].iloc[idx.flatten()].values
                df_stops["density_log"]          = df["density_log"].iloc[idx.flatten()].values

                df_stops.to_csv("data/cleaned/population_at_stops.csv", index=False)
                print(f"Stop-level sampling: {len(df_stops)} stops")
                print(f"  Median nearest-grid distance: "
                      f"{df_stops['nearest_grid_dist_km'].median():.2f} km")
                print("Exported: data/cleaned/population_at_stops.csv")

            except ImportError:
                print("[WARN] scikit-learn not installed — stop sampling skipped. "
                      "Run: pip install scikit-learn")
            except Exception as e:
                print(f"[WARN] Stop sampling failed: {e}")
        else:
            print("[SKIP] No gtfs_stop_nodes_*.csv found — run gtfs.py first.")
    else:
        print("[SKIP] Stop-level sampling requires geopandas + scikit-learn.")


if __name__ == "__main__":
    main()
