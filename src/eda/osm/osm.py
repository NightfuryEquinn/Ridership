import json
import pandas as pd
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import warnings
import os
warnings.filterwarnings('ignore')

DATA_PATH = "data/cleaned/osm_pois_clean.json"
OUTPUT_DIR = "src/eda/osm/results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_pois(data):
    pois = []
    for elem in data.get("elements", []):
        if elem.get("type") == "node":
            tags = elem.get("tags", {})
            lat = elem.get("lat")
            lon = elem.get("lon")
            if lat and lon:
                category = (
                    tags.get("amenity") or
                    tags.get("shop") or
                    tags.get("healthcare") or
                    tags.get("building") or
                    "unknown"
                )
                pois.append({
                    "id": elem.get("id"),
                    "lat": lat,
                    "lon": lon,
                    "name": tags.get("name", ""),
                    "category": category,
                    "tags": tags
                })
    return pd.DataFrame(pois)

def poi_frequency_ranking(df):
    print("=" * 60)
    print("1. POI CATEGORY FREQUENCY & RANKING")
    print("=" * 60)
    freq = df["category"].value_counts()
    print(f"\nTotal POIs: {len(df)}")
    print(f"Unique categories: {df["category"].nunique()}")
    print("\nTop 20 POI Categories:")
    print("-" * 40)
    for i, (cat, count) in enumerate(freq.head(20).items(), 1):
        pct = count / len(df) * 100
        print(f"{i:3}. {cat:<25} {count:>6} ({pct:>5.1f}%)")

    freq_df = freq.reset_index()
    freq_df.columns = ["category", "count"]
    freq_df["percentage"] = freq_df["count"] / len(df) * 100
    freq_df["rank"] = range(1, len(freq_df) + 1)
    freq_df = freq_df[["rank", "category", "count", "percentage"]]
    freq_df.to_csv(os.path.join(OUTPUT_DIR, "poi_frequency_ranking.csv"), index=False)

    return freq

def spatial_clustering(df):
    print("\n" + "=" * 60)
    print("2. SPATIAL CLUSTERING BY POI TYPE")
    print("=" * 60)

    coords = df[["lat", "lon"]].values
    categories = df["category"].unique()

    cluster_results = {}
    for cat in categories:
        cat_df = df[df["category"] == cat]
        if len(cat_df) < 3:
            cluster_results[cat] = {"n_clusters": 0, "samples": len(cat_df)}
            continue

        coords_cat = cat_df[["lat", "lon"]].values
        kms_rad = 6371.0
        eps_rad = 10 / kms_rad

        db = DBSCAN(eps=eps_rad, min_samples=2, metric="haversine")
        labels = db.fit_predict(np.radians(coords_cat))

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = list(labels).count(-1)

        cluster_results[cat] = {
            "n_clusters": n_clusters,
            "noise": noise,
            "samples": len(cat_df)
        }

    clustered = {k: v for k, v in cluster_results.items() if v["n_clusters"] > 0}
    print(f"\nCategories with spatial clusters (eps=10km, min_samples=2):")
    print("-" * 50)
    print(f"{'Category':<25} {'Clusters':>10} {'Noise':>8} {'Total':>8}")
    print("-" * 50)
    for cat, v in sorted(clustered.items(), key=lambda x: x[1]["n_clusters"], reverse=True)[:15]:
        print(f"{cat:<25} {v['n_clusters']:>10} {v['noise']:>8} {v['samples']:>8}")

    print(f"\nCategories without clusters: {len(cluster_results) - len(clustered)}")

    cluster_df = pd.DataFrame(cluster_results).T.reset_index()
    cluster_df.columns = ["category", "n_clusters", "noise", "samples"]
    cluster_df = cluster_df.sort_values("n_clusters", ascending=False)
    cluster_df.to_csv(os.path.join(OUTPUT_DIR, "spatial_clustering.csv"), index=False)

    return cluster_results

def poi_density(df):
    print("\n" + "=" * 60)
    print("3. POI DENSITY PER UNIT AREA")
    print("=" * 60)

    lats = df["lat"]
    lons = df["lon"]

    lat_range = lats.max() - lats.min()
    lon_range = lons.max() - lons.min()

    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.radians(lats.mean()))

    area_lat = lat_range * km_per_deg_lat
    area_lon = lon_range * km_per_deg_lon
    total_area = area_lat * area_lon

    print(f"\nBounding Box:")
    print(f"  Lat: {lats.min():.4f} to {lats.max():.4f} (range: {lat_range:.4f} deg = {area_lat:.2f} km)")
    print(f"  Lon: {lons.min():.4f} to {lons.max():.4f} (range: {lon_range:.4f} deg = {area_lon:.2f} km)")
    print(f"  Approximate bounding area: {total_area:.2f} km²")

    overall_density = len(df) / total_area if total_area > 0 else 0
    print(f"\nOverall POI density: {overall_density:.4f} POIs/km²")

    grid_sizes = [0.5, 1.0, 2.0, 5.0]
    print("\nGrid-based density analysis:")
    print("-" * 50)

    for grid_size in grid_sizes:
        n_lat = max(1, int(np.ceil(area_lat / grid_size)))
        n_lon = max(1, int(np.ceil(area_lon / grid_size)))

        lat_bins = np.linspace(lats.min(), lats.max(), n_lat + 1)
        lon_bins = np.linspace(lons.min(), lons.max(), n_lon + 1)

        density_grid, _, _ = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])

        cell_area = (area_lon / n_lon) * (area_lat / n_lat) if n_lon > 0 and n_lat > 0 else 0
        density_values = density_grid.flatten() / cell_area if cell_area > 0 else density_grid.flatten()

        valid_densities = density_values[density_values > 0]

        print(f"\n  Grid size: {grid_size} km x {grid_size} km ({n_lat} x {n_lon} cells)")
        if len(valid_densities) > 0:
            print(f"    Max density:  {valid_densities.max():.4f} POIs/km²")
            print(f"    Mean density: {valid_densities.mean():.4f} POIs/km²")
            print(f"    Median density: {np.median(valid_densities):.4f} POIs/km²")

    cat_areas = {}
    print("\nDensity by category:")
    print("-" * 50)
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        cat_lat_range = cat_df["lat"].max() - cat_df["lat"].min()
        cat_lon_range = cat_df["lon"].max() - cat_df["lon"].min()
        cat_area = (cat_lat_range * km_per_deg_lat) * (cat_lon_range * km_per_deg_lon)
        cat_density = len(cat_df) / cat_area if cat_area > 0 else 0
        cat_areas[cat] = {
            "count": len(cat_df),
            "span_area": cat_area,
            "density": cat_density
        }

    sorted_cats = sorted(cat_areas.items(), key=lambda x: x[1]["density"], reverse=True)
    for cat, stats in sorted_cats[:15]:
        print(f"  {cat:<25} {stats['count']:>5} POIs, density: {stats['density']:.4f} POIs/km²")

    density_summary = {
        "metric": ["total_area_km2", "overall_density_pois_per_km2", "n_pois", "n_categories"],
        "value": [total_area, overall_density, len(df), df["category"].nunique()]
    }
    pd.DataFrame(density_summary).to_csv(os.path.join(OUTPUT_DIR, "density_summary.csv"), index=False)

    cat_density_df = pd.DataFrame([
        {"category": cat, "count": v["count"], "span_area_km2": v["span_area"], "density_pois_per_km2": v["density"]}
        for cat, v in sorted_cats
    ])
    cat_density_df.to_csv(os.path.join(OUTPUT_DIR, "density_by_category.csv"), index=False)

    return {"total_area": total_area, "overall_density": overall_density, "cat_areas": cat_areas}

def main():
    print("Loading OSM POI data...")
    data = load_data(DATA_PATH)
    print(f"Loaded {len(data.get('elements', []))} elements")

    df = extract_pois(data)
    print(f"Extracted {len(df)} POIs with valid coordinates")

    freq = poi_frequency_ranking(df)
    cluster_results = spatial_clustering(df)
    density_results = poi_density(df)

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    print(f"\nExported CSVs to {OUTPUT_DIR}:")
    print("  - poi_frequency_ranking.csv")
    print("  - spatial_clustering.csv")
    print("  - density_summary.csv")
    print("  - density_by_category.csv")

if __name__ == "__main__":
    main()