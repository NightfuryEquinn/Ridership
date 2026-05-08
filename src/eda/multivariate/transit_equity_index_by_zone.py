import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import geopandas as gpd
from pathlib import Path

BASE = Path(__file__).parent.parent.parent.parent / "data" / "cleaned"
OUT = Path(__file__).parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

stops_dfs = []
for gtfs_dir in ["gtfs_rapid_bus_kl", "gtfs_ktmb", "gtfs_rapid_rail_kl"]:
    stops_file = BASE / gtfs_dir / "stops.txt"
    if stops_file.exists():
        df = pd.read_csv(stops_file)
        df["source"] = gtfs_dir
        stops_dfs.append(df)

all_stops = pd.concat(stops_dfs, ignore_index=True)
stops_gdf = gpd.GeoDataFrame(all_stops, geometry=gpd.points_from_xy(all_stops.stop_lon, all_stops.stop_lat), crs="EPSG:4326")

gdf_zones = gpd.read_file(BASE / "gadm_mys_l1_clean.geojson")
gdf_zones = gdf_zones.to_crs("EPSG:3857")
stops_gdf_proj = stops_gdf.to_crs("EPSG:3857")

stops_per_zone = gpd.sjoin(stops_gdf_proj, gdf_zones[["NAME_1", "geometry"]], how="left", predicate="within")
stop_density = stops_per_zone.groupby("NAME_1").size().reset_index(name="stop_count")

zones_merged = gdf_zones.merge(stop_density, on="NAME_1", how="left")
zones_merged["stop_count"] = zones_merged["stop_count"].fillna(0)
zones_merged["zone_area_km2"] = zones_merged.geometry.area / 1e6
zones_merged["stop_density"] = zones_merged["stop_count"] / zones_merged["zone_area_km2"]

coords = zones_merged.geometry.centroid
zones_merged["centroid_lat"] = coords.y / 111320
zones_merged["centroid_lon"] = coords.x / (111320 * np.cos(np.radians(3)))

zones_merged["walking_friction_estimate"] = 0.5
zones_merged["population_density_estimate"] = 500
zones_merged["poi_density_estimate"] = 50

stop_density_max = zones_merged["stop_density"].max()
if stop_density_max == 0 or np.isnan(stop_density_max):
    stop_density_max = 1

zones_merged["equity_score"] = (
    0.3 * (zones_merged["stop_density"] / stop_density_max) +
    0.3 * (zones_merged["population_density_estimate"] / zones_merged["population_density_estimate"].max()) +
    0.2 * (1 - np.clip(zones_merged["walking_friction_estimate"], 0, 1)) +
    0.2 * (zones_merged["poi_density_estimate"] / zones_merged["poi_density_estimate"].max())
)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
zones_merged.plot(column="stop_density", cmap="YlOrRd", legend=True, ax=ax1)
ax1.set_title("Stop Density by Admin Zone")
ax1.axis("off")

ax2 = axes[0, 1]
zones_merged.plot(column="equity_score", cmap="RdYlGn", legend=True, ax=ax2)
ax2.set_title("Transit Equity Index Score")
ax2.axis("off")

scores_sorted = zones_merged[["NAME_1", "equity_score", "stop_density", "stop_count"]].sort_values("equity_score", ascending=True)
ax3 = axes[1, 0]
ax3.barh(scores_sorted["NAME_1"], scores_sorted["equity_score"], color="teal")
ax3.set_title("Equity Score by Zone (Higher = More Equitable)")
ax3.set_xlabel("Equity Score")

ax4 = axes[1, 1]
ax4.scatter(scores_sorted["stop_density"], scores_sorted["equity_score"], c="steelblue", s=80)
for _, row in scores_sorted.iterrows():
    ax4.annotate(row["NAME_1"][:10], (row["stop_density"], row["equity_score"]), fontsize=7)
ax4.set_xlabel("Stop Density (stops/km2)")
ax4.set_ylabel("Equity Score")
ax4.set_title("Stop Density vs Equity Score")

plt.tight_layout()
plt.savefig(OUT / "transit_equity_index_by_zone.png", dpi=150, bbox_inches="tight")
plt.close()

zones_merged[["NAME_1", "stop_count", "stop_density", "equity_score"]].to_csv(OUT / "transit_equity_scores.csv", index=False)
print("2. transit_equity_index_by_zone.py - DONE")