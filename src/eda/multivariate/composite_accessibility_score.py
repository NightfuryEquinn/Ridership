import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import geopandas as gpd
from pathlib import Path

BASE = Path(__file__).parent.parent.parent.parent / "data" / "cleaned"
OUT = Path(__file__).parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

poi_data = json.load(open(BASE / "osm_pois_clean.json", encoding="utf-8"))
pois = []
for el in poi_data["elements"]:
    if "lat" in el and "lon" in el and "tags" in el:
        pois.append({"lat": el["lat"], "lon": el["lon"]})
pois_df = pd.DataFrame(pois)

gdf_zones = gpd.read_file(BASE / "gadm_mys_l1_clean.geojson")

stops_dfs = []
for gtfs_dir in ["gtfs_rapid_bus_kl", "gtfs_ktmb", "gtfs_rapid_rail_kl"]:
    stops_file = BASE / gtfs_dir / "stops.txt"
    if stops_file.exists():
        df = pd.read_csv(stops_file)
        df["source"] = gtfs_dir
        stops_dfs.append(df)

all_stops = pd.concat(stops_dfs, ignore_index=True)

pop = pd.read_csv(BASE / "population_density_clean.csv")

pois_np = pois_df[["lon", "lat"]].values
stops_np = all_stops[["stop_lon", "stop_lat"]].values

min_lon, max_lon = min(pois_np[:, 0].min(), stops_np[:, 0].min()) - 0.1, max(pois_np[:, 0].max(), stops_np[:, 0].max()) + 0.1
min_lat, max_lat = min(pois_np[:, 1].min(), stops_np[:, 1].min()) - 0.1, max(pois_np[:, 1].max(), stops_np[:, 1].max()) + 0.1

grid_size = 0.2
lon_edges = np.arange(min_lon, max_lon + grid_size, grid_size)
lat_edges = np.arange(min_lat, max_lat + grid_size, grid_size)

def haversine_vec(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

scores = []
for i, lat in enumerate(lat_centers):
    for j, lon in enumerate(lon_centers):
        dist_poi = haversine_vec(lat, lon, pois_np[:, 1], pois_np[:, 0])
        dist_stops = haversine_vec(lat, lon, stops_np[:, 1], stops_np[:, 0])

        pois_nearby = np.sum(dist_poi <= 1.0)
        stops_nearby = np.sum(dist_stops <= 0.5)

        pop_mask = (np.abs(pop["latitude"] - lat) < grid_size) & (np.abs(pop["longitude"] - lon) < grid_size)
        pop_nearby = pop.loc[pop_mask, "density_per_km2"].mean() if pop_mask.any() else 0

        stop_proximity_score = min(stops_nearby / 5, 1.0)
        poi_density_score = min(pois_nearby / 20, 1.0)
        friction_score = 0.7

        accessibility_score = 0.4 * stop_proximity_score + 0.35 * poi_density_score + 0.25 * friction_score

        scores.append({
            "lon": lon,
            "lat": lat,
            "stops_nearby": int(stops_nearby),
            "pois_nearby": int(pois_nearby),
            "pop_density": float(pop_nearby) if not np.isnan(pop_nearby) else 0,
            "accessibility_score": accessibility_score
        })

scores_df = pd.DataFrame(scores)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
sc1 = ax1.scatter(scores_df["lon"], scores_df["lat"], c=scores_df["stops_nearby"], cmap="YlOrRd", s=50, alpha=0.7)
plt.colorbar(sc1, ax=ax1, label="Stops within 500m")
ax1.set_title("Stop Proximity Score per Grid Cell")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")

ax2 = axes[0, 1]
sc2 = ax2.scatter(scores_df["lon"], scores_df["lat"], c=scores_df["accessibility_score"], cmap="RdYlGn", s=50, alpha=0.7)
plt.colorbar(sc2, ax=ax2, label="Accessibility Score")
ax2.set_title("Composite Accessibility Score per Grid Cell")
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")

bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ["Very Low", "Low", "Medium", "High", "Very High"]
scores_df["accessibility_category"] = pd.cut(scores_df["accessibility_score"], bins=bins, labels=labels)

ax3 = axes[1, 0]
counts = scores_df["accessibility_category"].value_counts().reindex(labels)
counts.plot(kind="bar", ax=ax3, color="teal", alpha=0.8)
ax3.set_xlabel("Accessibility Category")
ax3.set_ylabel("Number of Grid Cells")
ax3.set_title("Distribution of Accessibility Scores")
ax3.tick_params(axis="x", rotation=45)

ax4 = axes[1, 1]
ax4.scatter(scores_df["pois_nearby"], scores_df["stops_nearby"], c=scores_df["accessibility_score"], cmap="RdYlGn", s=50, alpha=0.5)
ax4.set_xlabel("POIs within 1km")
ax4.set_ylabel("Stops within 500m")
ax4.set_title("POI Density vs Stop Proximity (color = accessibility)")

plt.tight_layout()
plt.savefig(OUT / "composite_accessibility_score.png", dpi=150, bbox_inches="tight")
plt.close()

scores_df.to_csv(OUT / "composite_accessibility_scores.csv", index=False)
print("9. composite_accessibility_score.py - DONE")