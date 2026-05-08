import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

BASE = Path(__file__).parent.parent.parent.parent / "data" / "cleaned"
OUT = Path(__file__).parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

ridership = pd.read_csv(BASE / "ridership_headline_clean.csv", parse_dates=["date"])

stops_dfs = []
for gtfs_dir in ["gtfs_rapid_bus_kl", "gtfs_ktmb", "gtfs_rapid_rail_kl"]:
    stops_file = BASE / gtfs_dir / "stops.txt"
    trips_file = BASE / gtfs_dir / "trips.txt"
    stop_times_file = BASE / gtfs_dir / "stop_times.txt"
    routes_file = BASE / gtfs_dir / "routes.txt"

    if stops_file.exists():
        stops_df = pd.read_csv(stops_file)
        stops_df["source"] = gtfs_dir

        if trips_file.exists() and stop_times_file.exists():
            trips_df = pd.read_csv(trips_file)
            stop_times_df = pd.read_csv(stop_times_file)
            routes_df = pd.read_csv(routes_file) if routes_file.exists() else pd.DataFrame()

            stop_route = stop_times_df[["stop_id", "trip_id"]].drop_duplicates()
            stop_route = stop_route.merge(trips_df[["trip_id", "route_id"]], on="trip_id", how="left")
            stop_route = stop_route.merge(routes_df[["route_id", "route_short_name"]], on="route_id", how="left")
            stop_route_agg = stop_route.groupby("stop_id").agg({"route_id": "first", "route_short_name": "first"}).reset_index()

            stops_df = stops_df.merge(stop_route_agg, on="stop_id", how="left")

        stops_dfs.append(stops_df)

all_stops = pd.concat(stops_dfs, ignore_index=True)

poi_data = json.load(open(BASE / "osm_pois_clean.json", encoding="utf-8"))
pois = []
poi_types = {"mall": [], "school": [], "hospital": []}
for el in poi_data["elements"]:
    if "lat" in el and "lon" in el and "tags" in el:
        tags = el["tags"]
        ptype = None
        if "shop" in tags and tags["shop"] == "mall":
            ptype = "mall"
        elif "amenity" in tags:
            if tags["amenity"] == "school":
                ptype = "school"
            elif tags["amenity"] in ["hospital", "clinic"]:
                ptype = "hospital"
        if ptype:
            pois.append({"lat": el["lat"], "lon": el["lon"], "poi_type": ptype})

pois_df = pd.DataFrame(pois)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

route_performance = []
for route_id, route_stops in all_stops.groupby("route_id"):
    stops_coords = route_stops[["stop_lat", "stop_lon"]].drop_duplicates()
    if len(stops_coords) == 0:
        continue
    center_lat = stops_coords["stop_lat"].mean()
    center_lon = stops_coords["stop_lon"].mean()

    nearby_pois = pois_df.copy()
    nearby_pois["dist_km"] = haversine_km(nearby_pois["lat"], nearby_pois["lon"], center_lat, center_lon)
    nearby_pois_1km = nearby_pois[nearby_pois["dist_km"] <= 1.0]

    route_performance.append({
        "route_id": route_id,
        "n_stops": len(route_stops),
        "center_lat": center_lat,
        "center_lon": center_lon,
        " malls_1km": len(nearby_pois_1km[nearby_pois_1km["poi_type"] == "mall"]),
        " schools_1km": len(nearby_pois_1km[nearby_pois_1km["poi_type"] == "school"]),
        " hospitals_1km": len(nearby_pois_1km[nearby_pois_1km["poi_type"] == "hospital"]),
        "total_pois_1km": len(nearby_pois_1km)
    })

route_perf_df = pd.DataFrame(route_performance)
route_perf_df.columns = [c.strip() for c in route_perf_df.columns]

routes_df_list = []
for gtfs_dir in ["gtfs_rapid_bus_kl", "gtfs_ktmb", "gtfs_rapid_rail_kl"]:
    routes_file = BASE / gtfs_dir / "routes.txt"
    if routes_file.exists():
        routes_df_list.append(pd.read_csv(routes_file))
routes_all = pd.concat(routes_df_list, ignore_index=True) if routes_df_list else pd.DataFrame()

if len(routes_all) > 0:
    route_perf_df = route_perf_df.merge(routes_all[["route_id", "route_short_name", "route_long_name"]], on="route_id", how="left")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
poi_counts = route_perf_df.groupby("total_pois_1km").size().reset_index(name="n_routes")
poi_counts = poi_counts.sort_values("total_pois_1km")
ax1.bar(poi_counts["total_pois_1km"], poi_counts["n_routes"], color="steelblue")
ax1.set_xlabel("Total POIs within 1km of Route")
ax1.set_ylabel("Number of Routes")
ax1.set_title("Route Distribution by Nearby POI Density")

ax2 = axes[0, 1]
ax2.scatter(route_perf_df["total_pois_1km"], route_perf_df["n_stops"], c="teal", s=60, alpha=0.6)
ax2.set_xlabel("Total POIs within 1km")
ax2.set_ylabel("Number of Stops on Route")
ax2.set_title("Route Stops vs Nearby POI Density")

ax3 = axes[1, 0]
poi_type_cols = [c for c in route_perf_df.columns if "_1km" in c and "total" not in c]
for col in poi_type_cols:
    vals = route_perf_df[col].value_counts().sort_index()
    ax3.plot(vals.index, vals.values, marker="o", label=col.replace("_1km", ""), linewidth=2)
ax3.set_xlabel("Count of POI type within 1km")
ax3.set_ylabel("Number of Routes")
ax3.set_title("Routes by POI Type Proximity")
ax3.legend()

top_routes = route_perf_df.nlargest(20, "total_pois_1km")
ax4 = axes[1, 1]
labels = top_routes["route_short_name"].fillna(top_routes["route_id"]).astype(str)
ax4.barh(labels, top_routes["total_pois_1km"], color="coral")
ax4.set_xlabel("Total POIs within 1km")
ax4.set_title("Top 20 Routes by POI Accessibility")

plt.tight_layout()
plt.savefig(OUT / "destination_based_route_performance.png", dpi=150, bbox_inches="tight")
plt.close()

route_perf_df.to_csv(OUT / "route_poi_performance.csv", index=False)
print("7. destination_based_route_performance.py - DONE")