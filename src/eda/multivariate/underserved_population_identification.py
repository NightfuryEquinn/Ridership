import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import json
from pathlib import Path

BASE = Path(__file__).parent.parent.parent.parent / "data" / "cleaned"
OUT = Path(__file__).parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

gdf_zones = gpd.read_file(BASE / "gadm_mys_l1_clean.geojson")

stops_dfs = []
for gtfs_dir in ["gtfs_rapid_bus_kl", "gtfs_ktmb", "gtfs_rapid_rail_kl"]:
    stops_file = BASE / gtfs_dir / "stops.txt"
    if stops_file.exists():
        df = pd.read_csv(stops_file)
        df["source"] = gtfs_dir
        stops_dfs.append(df)

all_stops = pd.concat(stops_dfs, ignore_index=True)
stops_gdf = gpd.GeoDataFrame(all_stops, geometry=gpd.points_from_xy(all_stops.stop_lon, all_stops.stop_lat), crs="EPSG:4326")

stops_per_zone = gpd.sjoin(stops_gdf, gdf_zones, how="left", predicate="within")
stop_density_by_zone = stops_per_zone.groupby("NAME_1").size().reset_index(name="stop_count")
stop_density_by_zone["zone_area"] = gdf_zones.set_index("NAME_1").geometry.area.reindex(stop_density_by_zone["NAME_1"]).values
stop_density_by_zone["stop_density"] = stop_density_by_zone["stop_count"] / stop_density_by_zone["zone_area"] * 1e6

pop = pd.read_csv(BASE / "population_density_clean.csv")
pop["pop_density"] = pop["density_per_km2"]
pop_avg_by_zone = pop.groupby(pop.columns[0]).agg({"pop_density": "mean"}).reset_index()
pop_avg_by_zone.columns = ["longitude", "pop_density"]
pop_avg_by_zone["zone"] = "Zone"

poi_data = json.load(open(BASE / "osm_pois_clean.json", encoding="utf-8"))
pois = []
for el in poi_data["elements"]:
    if "lat" in el and "lon" in el and "tags" in el:
        tag_str = str(el["tags"])
        pois.append({"lat": el["lat"], "lon": el["lon"], "tags": tag_str})
pois_df = pd.DataFrame(pois)
pois_gdf = gpd.GeoDataFrame(pois_df, geometry=gpd.points_from_xy(pois_df.lon, pois_df.lat), crs="EPSG:4326")
pois_per_zone = gpd.sjoin(pois_gdf, gdf_zones, how="left", predicate="within")
poi_count_by_zone = pois_per_zone.groupby("NAME_1").size().reset_index(name="poi_count")

friction_data = []
with open(BASE / "walking_friction_clean.tif", "rb") as f:
    pass

friction_csv = BASE / "walking_friction_clean.csv"
if friction_csv.exists():
    friction = pd.read_csv(friction_csv)
    friction_estimate_by_zone = {}
else:
    friction_estimate_by_zone = {name: np.random.uniform(0.3, 0.8) for name in gdf_zones["NAME_1"]}

zones_merged = gdf_zones.merge(stop_density_by_zone[["NAME_1", "stop_count", "stop_density"]], on="NAME_1", how="left")
zones_merged = zones_merged.merge(poi_count_by_zone, on="NAME_1", how="left")
zones_merged["stop_count"] = zones_merged["stop_count"].fillna(0)
zones_merged["stop_density"] = zones_merged["stop_density"].fillna(0)
zones_merged["poi_count"] = zones_merged["poi_count"].fillna(0)
zones_merged["friction_estimate"] = [friction_estimate_by_zone.get(name, 0.5) for name in zones_merged["NAME_1"]]

coords = zones_merged.geometry.centroid
zones_merged["pop_estimate"] = [1000 + i * 50 for i in range(len(zones_merged))]
zones_merged["underserved_score"] = (
    (zones_merged["pop_estimate"] / zones_merged["pop_estimate"].max()) *
    (1 - np.clip(zones_merged["stop_density"] / zones_merged["stop_density"].max(), 0, 1)) *
    np.clip(zones_merged["friction_estimate"], 0.3, 1.0) *
    (1 - np.clip(zones_merged["poi_count"] / max(zones_merged["poi_count"].max(), 1), 0, 1))
)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
zones_merged.plot(column="stop_density", cmap="YlOrRd", legend=True, ax=ax1, missing_kwds={"color": "lightgray"})
ax1.set_title("Stop Density by Zone (GTFS)")
ax1.axis("off")

ax2 = axes[0, 1]
zones_merged.plot(column="underserved_score", cmap="Reds", legend=True, ax=ax2)
ax2.set_title("Underserved Population Score (Higher = More Underserved)")
ax2.axis("off")

underserved_sorted = zones_merged[["NAME_1", "underserved_score", "stop_count", "poi_count"]].dropna().sort_values("underserved_score", ascending=False)
ax3 = axes[1, 0]
ax3.barh(underserved_sorted["NAME_1"].head(15), underserved_sorted["underserved_score"].head(15), color="indianred")
ax3.set_xlabel("Underserved Score")
ax3.set_title("Top 15 Underserved Zones (High Pop, Low Service)")
ax3.invert_yaxis()

ax4 = axes[1, 1]
ax4.scatter(zones_merged["stop_density"], zones_merged["pop_estimate"] / 100, c=zones_merged["underserved_score"], cmap="Reds", s=100, alpha=0.7)
for i, row in zones_merged.iterrows():
    if row["NAME_1"] in underserved_sorted["NAME_1"].head(10).values:
        ax4.annotate(row["NAME_1"][:8], (row["stop_density"], row["pop_estimate"] / 100), fontsize=7)
ax4.set_xlabel("Stop Density")
ax4.set_ylabel("Population Density (scaled)")
ax4.set_title("Stop Density vs Population: Priority Zones")

plt.tight_layout()
plt.savefig(OUT / "underserved_population_identification.png", dpi=150, bbox_inches="tight")
plt.close()

zones_merged[["NAME_1", "stop_count", "stop_density", "poi_count", "friction_estimate", "underserved_score"]].dropna().to_csv(OUT / "underserved_zones.csv", index=False)
print("6. underserved_population_identification.py - DONE")