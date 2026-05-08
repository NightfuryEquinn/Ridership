"""
GTFS Exploratory Data Analysis
Covers 12 services: Rapid Bus Penang, Rapid Bus KL, Monorail, MRT Putrajaya,
MRT Kajang, LRT Kelana Jaya, LRT Ampang, ETS, Intercity, Komuter,
Komuter Utara, Komuter Terbau
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = "../../../data/cleaned"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SERVICE_CONFIG = {
    "Rapid Bus Penang": {"path": f"{BASE_PATH}/gtfs_rapid_bus_penang", "type": "bus"},
    "Rapid Bus KL": {"path": f"{BASE_PATH}/gtfs_rapid_bus_kl", "type": "bus"},
    "Monorail": {"path": f"{BASE_PATH}/gtfs_rapid_rail_kl", "type": "rail", "route_ids": ["MR"]},
    "MRT Putrajaya": {"path": f"{BASE_PATH}/gtfs_rapid_rail_kl", "type": "rail", "route_ids": ["PYL"]},
    "MRT Kajang": {"path": f"{BASE_PATH}/gtfs_rapid_rail_kl", "type": "rail", "route_ids": ["KGL"]},
    "LRT Kelana Jaya": {"path": f"{BASE_PATH}/gtfs_rapid_rail_kl", "type": "rail", "route_ids": ["KJ"]},
    "LRT Ampang": {"path": f"{BASE_PATH}/gtfs_rapid_rail_kl", "type": "rail", "route_ids": ["AG"]},
    "ETS": {"path": f"{BASE_PATH}/gtfs_ktmb", "type": "rail", "route_ids": ["ETS"]},
    "Intercity": {"path": f"{BASE_PATH}/gtfs_ktmb", "type": "rail", "route_ids": ["ERT", "ES", "SH", "ST"]},
    "Komuter": {"path": f"{BASE_PATH}/gtfs_ktmb", "type": "rail", "route_ids": ["KC05_KB18", "KA15_KD19", "100_47300", "100_9000"]},
    "Komuter Utara": {"path": f"{BASE_PATH}/gtfs_ktmb", "type": "rail", "route_ids": ["100_47300", "100_9000"]},
    "Komuter Terbau": {"path": f"{BASE_PATH}/gtfs_ktmb", "type": "rail", "route_ids": ["ST"]},
}


def load_gtfs_data(service_name, config):
    """Load GTFS data for a specific service."""
    base_path = config["path"]
    data = {}
    files = ["routes", "stops", "trips", "stop_times", "calendar", "shapes", "frequencies"]
    for f in files:
        fp = f"{base_path}/{f}.txt"
        if os.path.exists(fp):
            data[f] = pd.read_csv(fp, dtype=str)
    return data


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def calculate_route_length(stops_df, route_ids, data):
    """Calculate total route length in km."""
    if "trips" not in data or "stop_times" not in data or "routes" not in data:
        return []
    
    trips = data["trips"]
    stop_times = data["stop_times"]
    routes = data["routes"]
    
    filtered_trips = trips[trips["route_id"].isin(route_ids)]
    if len(filtered_trips) == 0:
        return []
    
    trip_ids = filtered_trips["trip_id"].unique()
    route_trips = stop_times[stop_times["trip_id"].isin(trip_ids)]
    
    if len(route_trips) == 0:
        return []
    
    route_lengths = []
    for trip_id in trip_ids[:100]:
        trip_stops = route_trips[route_trips["trip_id"] == trip_id].sort_values("stop_sequence")
        if len(trip_stops) < 2:
            continue
        merged = trip_stops.merge(stops_df[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left")
        merged = merged.dropna(subset=["stop_lat", "stop_lon"])
        if len(merged) < 2:
            continue
        total_dist = 0
        for i in range(len(merged) - 1):
            total_dist += haversine_km(
                float(merged.iloc[i]["stop_lat"]), float(merged.iloc[i]["stop_lon"]),
                float(merged.iloc[i+1]["stop_lat"]), float(merged.iloc[i+1]["stop_lon"])
            )
        route_lengths.append(total_dist)
    return route_lengths


def analyze_route_length_coverage():
    """Objective 1: Route length & coverage distribution."""
    print("Running Objective 1: Route length & coverage distribution...")
    
    results = {}
    for service_name, config in SERVICE_CONFIG.items():
        try:
            data = load_gtfs_data(service_name, config)
            if "stops" not in data:
                results[service_name] = {"num_stops": 0, "route_lengths": [], "stop_coverage": 0}
                continue
            stops = data["stops"]
            route_ids = config.get("route_ids", [])
            if route_ids and "routes" in data:
                routes = data["routes"]
                route_ids = [r for r in route_ids if r in routes["route_id"].values]
            else:
                route_ids = []
            
            num_stops = len(stops)
            route_lengths = calculate_route_length(stops, route_ids, data) if route_ids else []
            results[service_name] = {
                "num_stops": num_stops,
                "route_lengths": route_lengths,
                "stop_coverage": num_stops
            }
        except Exception as e:
            results[service_name] = {"num_stops": 0, "route_lengths": [], "stop_coverage": 0}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    services = list(results.keys())
    num_stops = [results[s]["num_stops"] for s in services]
    axes[0, 0].barh(services, num_stops, color="steelblue", alpha=0.8)
    axes[0, 0].set_xlabel("Number of Stops")
    axes[0, 0].set_title("Number of Stops by Service")
    
    all_lengths = []
    for s in services:
        all_lengths.extend(results[s]["route_lengths"])
    if all_lengths:
        axes[0, 1].hist([r for r in all_lengths if r > 0], bins=30, color="coral", edgecolor="darkred", alpha=0.7)
        axes[0, 1].set_xlabel("Route Length (km)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Route Length Distribution (All Services)")
    
    route_means = [np.mean(results[s]["route_lengths"]) if results[s]["route_lengths"] else 0 for s in services]
    axes[1, 0].barh(services, route_means, color="seagreen", alpha=0.8)
    axes[1, 0].set_xlabel("Average Route Length (km)")
    axes[1, 0].set_title("Average Route Length by Service")
    
    stop_data = pd.DataFrame({"Service": services, "Stops": num_stops})
    stop_data = stop_data.sort_values("Stops", ascending=False)
    axes[1, 1].bar(stop_data["Service"], stop_data["Stops"], color="mediumpurple", alpha=0.8)
    axes[1, 1].set_ylabel("Number of Stops")
    axes[1, 1].set_title("Stop Count by Service (Sorted)")
    axes[1, 1].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_route_length_coverage.png", dpi=150)
    plt.close()
    print("  Saved 01_route_length_coverage.png")


def analyze_stop_density_clustering():
    """Objective 2: Stop density spatial clustering."""
    print("Running Objective 2: Stop density spatial clustering...")
    
    all_stops = []
    service_labels = []
    
    for service_name, config in SERVICE_CONFIG.items():
        try:
            data = load_gtfs_data(service_name, config)
            if "stops" not in data:
                continue
            stops = data["stops"]
            if "stop_lat" in stops.columns and "stop_lon" in stops.columns:
                valid_stops = stops.dropna(subset=["stop_lat", "stop_lon"])
                for _, row in valid_stops.iterrows():
                    all_stops.append([float(row["stop_lat"]), float(row["stop_lon"])])
                    service_labels.append(service_name)
        except:
            continue
    
    if len(all_stops) < 10:
        print("  Insufficient stop data for clustering")
        return
    
    coords = np.array(all_stops)
    
    unique_services = list(set(service_labels))
    color_map = {s: plt.cm.tab20(i % 20) for i, s in enumerate(unique_services)}
    colors = [color_map[s] for s in service_labels]
    
    coords_rad = np.radians(coords)
    db = DBSCAN(eps=0.05, min_samples=5, metric='haversine')
    labels = db.fit_predict(coords_rad)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    scatter = axes[0].scatter(coords[:, 1], coords[:, 0], c=labels, cmap='tab20', 
                               s=15, alpha=0.7)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title("Stop Density Clustering (DBSCAN)")
    plt.colorbar(scatter, ax=axes[0], label="Cluster ID")
    
    cluster_sizes = {}
    for label in set(labels):
        if label != -1:
            cluster_sizes[label] = np.sum(labels == label)
    
    if cluster_sizes:
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:15]
        cluster_ids = [f"Cluster {c[0]}" for c in sorted_clusters]
        cluster_counts = [c[1] for c in sorted_clusters]
        axes[1].bar(cluster_ids, cluster_counts, color="steelblue", alpha=0.8)
        axes[1].set_ylabel("Number of Stops")
        axes[1].set_title("Top 15 Clusters by Stop Count")
        axes[1].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_stop_density_clustering.png", dpi=150)
    plt.close()
    print("  Saved 02_stop_density_clustering.png")


def analyze_service_frequency():
    """Objective 3: Service frequency by route/time."""
    print("Running Objective 3: Service frequency by route/time...")
    
    all_freq_data = []
    
    for service_name, config in SERVICE_CONFIG.items():
        try:
            data = load_gtfs_data(service_name, config)
            if "frequencies" not in data:
                continue
            freq = data["frequencies"]
            if "trip_id" in freq.columns and "headway_secs" in freq.columns:
                merged = freq.copy()
                if "trips" in data:
                    merged = merged.merge(data["trips"][["trip_id", "route_id"]], on="trip_id", how="left")
                    route_ids = config.get("route_ids", [])
                    if route_ids:
                        merged = merged[merged["route_id"].isin(route_ids)]
                merged["service"] = service_name
                merged["headway_mins"] = pd.to_numeric(merged["headway_secs"], errors="coerce") / 60
                all_freq_data.append(merged)
        except:
            continue
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    if all_freq_data:
        combined = pd.concat(all_freq_data, ignore_index=True)
        combined = combined.dropna(subset=["headway_mins"])
        
        if len(combined) > 0:
            service_headway = combined.groupby("service")["headway_mins"].mean().sort_values()
            axes[0, 0].barh(service_headway.index, service_headway.values, color="teal", alpha=0.8)
            axes[0, 0].set_xlabel("Average Headway (minutes)")
            axes[0, 0].set_title("Average Headway by Service")
            
            for service in combined["service"].unique():
                svc_data = combined[combined["service"] == service]["headway_mins"]
                if len(svc_data) > 0:
                    axes[0, 1].hist(svc_data, bins=20, alpha=0.5, label=service)
            axes[0, 1].set_xlabel("Headway (minutes)")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].set_title("Headway Distribution by Service")
            axes[0, 1].legend(fontsize=7)
            
            service_median = combined.groupby("service")["headway_mins"].median().sort_values()
            axes[1, 0].barh(service_median.index, service_median.values, color="coral", alpha=0.8)
            axes[1, 0].set_xlabel("Median Headway (minutes)")
            axes[1, 0].set_title("Median Headway by Service")
            
            headway_by_route = combined.groupby(["service", "route_id"])["headway_mins"].mean().reset_index()
            pivot_data = headway_by_route.pivot_table(index="route_id", columns="service", values="headway_mins")
            if not pivot_data.empty:
                sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[1, 1], cbar_kws={"label": "Headway (mins)"})
                axes[1, 1].set_title("Headway Heatmap (Route x Service)")
    else:
        axes[0, 0].text(0.5, 0.5, "No frequency data available", ha="center", va="center")
        axes[0, 1].text(0.5, 0.5, "No frequency data available", ha="center", va="center")
        axes[1, 0].text(0.5, 0.5, "No frequency data available", ha="center", va="center")
        axes[1, 1].text(0.5, 0.5, "No frequency data available", ha="center", va="center")
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_service_frequency.png", dpi=150)
    plt.close()
    print("  Saved 03_service_frequency.png")


def analyze_trip_count_patterns():
    """Objective 4: Trip count by day/hour pattern."""
    print("Running Objective 4: Trip count by day/hour pattern...")
    
    all_trip_data = []
    
    for service_name, config in SERVICE_CONFIG.items():
        try:
            data = load_gtfs_data(service_name, config)
            if "trips" not in data or "stop_times" not in data:
                continue
            
            trips = data["trips"]
            stop_times = data["stop_times"]
            
            route_ids = config.get("route_ids", [])
            if route_ids:
                trips = trips[trips["route_id"].isin(route_ids)]
            
            trip_ids = trips["trip_id"].unique()
            trip_st = stop_times[stop_times["trip_id"].isin(trip_ids)]
            
            trip_first = trip_st.sort_values("departure_time").groupby("trip_id").first().reset_index()
            trip_first["service"] = service_name
            
            def parse_time(t):
                try:
                    parts = str(t).split(":")
                    return int(parts[0])
                except:
                    return 0
            
            trip_first["hour"] = trip_first["departure_time"].apply(parse_time)
            all_trip_data.append(trip_first)
        except:
            continue
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    if all_trip_data:
        combined = pd.concat(all_trip_data, ignore_index=True)
        
        hour_counts = combined.groupby(["service", "hour"]).size().reset_index(name="trip_count")
        pivot_hour = hour_counts.pivot_table(index="hour", columns="service", values="trip_count", fill_value=0)
        
        pivot_hour.plot(kind="line", marker="o", ax=axes[0, 0], linewidth=1.5, markersize=3)
        axes[0, 0].set_xlabel("Hour of Day")
        axes[0, 0].set_ylabel("Number of Trips")
        axes[0, 0].set_title("Trip Count by Hour of Day")
        axes[0, 0].legend(fontsize=6, loc="upper right")
        
        service_total = combined.groupby("service").size().sort_values(ascending=False)
        axes[0, 1].barh(service_total.index, service_total.values, color="steelblue", alpha=0.8)
        axes[0, 1].set_xlabel("Total Trip Count")
        axes[0, 1].set_title("Total Trips by Service")
        
        morning = combined[(combined["hour"] >= 6) & (combined["hour"] < 12)]
        evening = combined[(combined["hour"] >= 17) & (combined["hour"] < 21)]
        
        if len(morning) > 0:
            morning_counts = morning.groupby("service").size()
            axes[1, 0].bar(morning_counts.index, morning_counts.values, color="orange", alpha=0.8)
        axes[1, 0].set_ylabel("Trip Count")
        axes[1, 0].set_title("Morning Peak Trips (6AM-12PM)")
        axes[1, 0].tick_params(axis="x", rotation=45)
        
        if len(evening) > 0:
            evening_counts = evening.groupby("service").size()
            axes[1, 1].bar(evening_counts.index, evening_counts.values, color="darkred", alpha=0.8)
        axes[1, 1].set_ylabel("Trip Count")
        axes[1, 1].set_title("Evening Peak Trips (5PM-9PM)")
        axes[1, 1].tick_params(axis="x", rotation=45)
    else:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No trip data available", ha="center", va="center")
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_trip_count_patterns.png", dpi=150)
    plt.close()
    print("  Saved 04_trip_count_patterns.png")


def analyze_network_topology():
    """Objective 5: Network graph topology analysis."""
    print("Running Objective 5: Network graph topology analysis...")
    
    graph_metrics = {}
    
    for service_name, config in SERVICE_CONFIG.items():
        try:
            data = load_gtfs_data(service_name, config)
            if "trips" not in data or "stop_times" not in data or "stops" not in data:
                graph_metrics[service_name] = {"nodes": 0, "edges": 0, "avg_degree": 0, "components": 0}
                continue
            
            trips = data["trips"]
            stop_times = data["stop_times"]
            stops = data["stops"]
            
            route_ids = config.get("route_ids", [])
            if route_ids:
                trips = trips[trips["route_id"].isin(route_ids)]
            
            trip_ids = trips["trip_id"].unique()
            route_st = stop_times[stop_times["trip_id"].isin(trip_ids)]
            
            edges = set()
            for trip_id in trip_ids[:200]:
                trip_stops = route_st[route_st["trip_id"] == trip_id].sort_values("stop_sequence")
                for i in range(len(trip_stops) - 1):
                    s1 = trip_stops.iloc[i]["stop_id"]
                    s2 = trip_stops.iloc[i+1]["stop_id"]
                    edges.add((s1, s2))
                    edges.add((s2, s1))
            
            nodes = len(stops)
            num_edges = len(edges)
            avg_degree = (2 * num_edges / nodes) if nodes > 0 else 0
            
            graph_metrics[service_name] = {
                "nodes": nodes,
                "edges": num_edges,
                "avg_degree": avg_degree,
                "components": 1
            }
        except Exception as e:
            graph_metrics[service_name] = {"nodes": 0, "edges": 0, "avg_degree": 0, "components": 0}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    services = list(graph_metrics.keys())
    nodes = [graph_metrics[s]["nodes"] for s in services]
    edges = [graph_metrics[s]["edges"] for s in services]
    degrees = [graph_metrics[s]["avg_degree"] for s in services]
    
    x = np.arange(len(services))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, nodes, width, label="Nodes (Stops)", color="steelblue", alpha=0.8)
    axes[0, 0].bar(x + width/2, [e/10 for e in edges], width, label="Edges/10", color="coral", alpha=0.8)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(services, rotation=45, ha="right")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Network Nodes and Edges by Service")
    axes[0, 0].legend()
    
    axes[0, 1].barh(services, degrees, color="seagreen", alpha=0.8)
    axes[0, 1].set_xlabel("Average Degree")
    axes[0, 1].set_title("Average Node Degree by Service")
    
    metrics_df = pd.DataFrame(graph_metrics).T
    if not metrics_df.empty:
        metrics_normalized = metrics_df.copy()
        for col in metrics_normalized.columns:
            max_val = metrics_normalized[col].max()
            if max_val > 0:
                metrics_normalized[col] = metrics_normalized[col] / max_val
        
        metrics_normalized.plot(kind="bar", ax=axes[1, 0], alpha=0.8)
        axes[1, 0].set_title("Normalized Network Metrics by Service")
        axes[1, 0].set_ylabel("Normalized Value")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].legend(loc="upper right")
    
    edge_to_node_ratio = [graph_metrics[s]["edges"] / max(graph_metrics[s]["nodes"], 1) for s in services]
    axes[1, 1].bar(services, edge_to_node_ratio, color="mediumpurple", alpha=0.8)
    axes[1, 1].set_ylabel("Edge/Node Ratio")
    axes[1, 1].set_title("Network Density (Edge/Node Ratio)")
    axes[1, 1].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_network_topology.png", dpi=150)
    plt.close()
    print("  Saved 05_network_topology.png")
    
    summary_df = pd.DataFrame(graph_metrics).T
    summary_df.to_csv(f"{OUTPUT_DIR}/05_network_metrics_summary.csv")
    print(f"  Saved 05_network_metrics_summary.csv")


def generate_summary_stats():
    """Generate summary statistics for all services."""
    print("Generating summary statistics...")
    
    summary = []
    for service_name, config in SERVICE_CONFIG.items():
        try:
            data = load_gtfs_data(service_name, config)
            row = {"Service": service_name}
            if "stops" in data:
                row["Num_Stops"] = len(data["stops"])
            if "routes" in data:
                row["Num_Routes"] = len(data["routes"])
            if "trips" in data:
                row["Num_Trips"] = len(data["trips"])
            if "calendar" in data:
                row["Num_Calendar"] = len(data["calendar"])
            summary.append(row)
        except:
            summary.append({"Service": service_name, "Num_Stops": 0, "Num_Routes": 0, "Num_Trips": 0, "Num_Calendar": 0})
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{OUTPUT_DIR}/00_summary_stats.csv", index=False)
    print(f"  Saved 00_summary_stats.csv")
    print(summary_df.to_string())


if __name__ == "__main__":
    print("=" * 60)
    print("GTFS Exploratory Data Analysis")
    print("=" * 60)
    
    generate_summary_stats()
    analyze_route_length_coverage()
    analyze_stop_density_clustering()
    analyze_service_frequency()
    analyze_trip_count_patterns()
    analyze_network_topology()
    
    print("=" * 60)
    print(f"All EDA plots saved to '{OUTPUT_DIR}' directory.")
    print("=" * 60)