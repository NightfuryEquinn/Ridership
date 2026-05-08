#!/usr/bin/env python3
"""
Bivariate EDA: GTFS vs Ridership
Objective: Route utilization & performance analysis

Analysis:
1. GTFS route statistics (route length, stop count, trip frequency) per service
2. Service frequency vs average ridership correlation
3. Route length vs ridership: does longer coverage = more riders?
4. Peak service hours vs peak ridership alignment
5. Stop coverage and network density metrics per service
6. Service efficiency: ridership per trip/frequency
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
pd.options.display.max_columns = None

SERVICE_CONFIG = {
    "Rapid Bus KL": {
        "path": "gtfs_rapid_bus_kl",
        "ridership_col": "bus_rkl",
        "route_ids": None,
        "type": "bus"
    },
    "Rapid Bus Penang": {
        "path": "gtfs_rapid_bus_penang",
        "ridership_col": "bus_rpn",
        "route_ids": None,
        "type": "bus"
    },
    "Monorail": {
        "path": "gtfs_rapid_rail_kl",
        "ridership_col": "rail_monorail",
        "route_ids": ["MR"],
        "type": "rail"
    },
    "MRT Putrajaya": {
        "path": "gtfs_rapid_rail_kl",
        "ridership_col": "rail_mrt_pjy",
        "route_ids": ["PYL"],
        "type": "rail"
    },
    "MRT Kajang": {
        "path": "gtfs_rapid_rail_kl",
        "ridership_col": "rail_mrt_kajang",
        "route_ids": ["KGL"],
        "type": "rail"
    },
    "LRT Kelana Jaya": {
        "path": "gtfs_rapid_rail_kl",
        "ridership_col": "rail_lrt_kj",
        "route_ids": ["KJ"],
        "type": "rail"
    },
    "LRT Ampang": {
        "path": "gtfs_rapid_rail_kl",
        "ridership_col": "rail_lrt_ampang",
        "route_ids": ["AG"],
        "type": "rail"
    },
    "ETS": {
        "path": "gtfs_ktmb",
        "ridership_col": "rail_ets",
        "route_ids": ["ETS"],
        "type": "rail"
    },
    "Intercity": {
        "path": "gtfs_ktmb",
        "ridership_col": "rail_intercity",
        "route_ids": ["ERT", "ES", "SH", "ST"],
        "type": "rail"
    },
    "Komuter Utara": {
        "path": "gtfs_ktmb",
        "ridership_col": "rail_komuter_utara",
        "route_ids": ["100_47300", "100_9000"],
        "type": "rail"
    },
    "Komuter Terbau": {
        "path": "gtfs_ktmb",
        "ridership_col": "rail_tebrau",
        "route_ids": ["ST"],
        "type": "rail"
    },
    "Komuter": {
        "path": "gtfs_ktmb",
        "ridership_col": "rail_komuter",
        "route_ids": ["KC05_KB18", "KA15_KD19"],
        "type": "rail"
    },
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_gtfs_data(service_name, config):
    """Load GTFS data for a specific service."""
    base_path = os.path.join(BASE_DIR, f"../../../data/cleaned/{config['path']}")
    data = {}
    files = ["routes", "stops", "trips", "stop_times", "calendar", "shapes", "frequencies"]
    for f in files:
        fp = os.path.join(base_path, f"{f}.txt")
        if os.path.exists(fp):
            try:
                data[f] = pd.read_csv(fp, dtype=str)
            except Exception:
                pass
    return data


def extract_service_metrics(service_name, config):
    """Extract GTFS metrics for a service."""
    data = load_gtfs_data(service_name, config)

    metrics = {
        'service': service_name,
        'type': config['type'],
        'ridership_col': config['ridership_col'],
        'num_stops': 0,
        'num_routes': 0,
        'num_trips': 0,
        'avg_route_length_km': 0,
        'avg_daily_trips': 0,
        'avg_headway_mins': 0,
        'stop_density': 0,
    }

    if "stops" in data and len(data["stops"]) > 0:
        metrics['num_stops'] = len(data["stops"])

    if "routes" in data and len(data["routes"]) > 0:
        metrics['num_routes'] = len(data["routes"])

    if "trips" in data and len(data["trips"]) > 0:
        metrics['num_trips'] = len(data["trips"])

    route_ids = config.get("route_ids", [])
    if route_ids and "routes" in data:
        valid_route_ids = [r for r in route_ids if r in data["routes"]["route_id"].values]
    else:
        valid_route_ids = []

    if "stops" in data and "trips" in data and "stop_times" in data and len(valid_route_ids) > 0:
        stops = data["stops"]
        trips = data["trips"]
        stop_times = data["stop_times"]

        filtered_trips = trips[trips["route_id"].isin(valid_route_ids)]
        trip_ids = filtered_trips["trip_id"].unique()

        route_lengths = []
        for trip_id in trip_ids[:200]:
            trip_st = stop_times[stop_times["trip_id"] == trip_id].sort_values("stop_sequence")
            if len(trip_st) < 2:
                continue
            merged = trip_st.merge(stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left")
            merged = merged.dropna(subset=["stop_lat", "stop_lon"])
            if len(merged) < 2:
                continue
            total_dist = 0
            for i in range(len(merged) - 1):
                try:
                    total_dist += haversine_km(
                        float(merged.iloc[i]["stop_lat"]), float(merged.iloc[i]["stop_lon"]),
                        float(merged.iloc[i+1]["stop_lat"]), float(merged.iloc[i+1]["stop_lon"])
                    )
                except (ValueError, TypeError):
                    continue
            route_lengths.append(total_dist)

        if route_lengths:
            metrics['avg_route_length_km'] = np.mean(route_lengths)
            metrics['stop_density'] = metrics['num_stops'] / max(metrics['avg_route_length_km'], 1)

    if "frequencies" in data and len(data["frequencies"]) > 0:
        freq = data["frequencies"]
        if "headway_secs" in freq.columns:
            freq["headway_mins"] = pd.to_numeric(freq["headway_secs"], errors="coerce") / 60
            metrics['avg_headway_mins'] = freq["headway_mins"].mean()

    return metrics


def load_ridership():
    """Load ridership data."""
    df = pd.read_csv(
        os.path.join(BASE_DIR, "../../../data/cleaned/ridership_headline_clean.csv"),
        parse_dates=['date']
    )
    df = df.set_index('date').sort_index()
    return df


def compute_service_ridership_stats(ridership, service_col):
    """Compute ridership statistics for a service."""
    if service_col not in ridership.columns:
        return {}
    col_data = ridership[service_col].dropna()
    if len(col_data) == 0:
        return {}
    return {
        'mean': col_data.mean(),
        'std': col_data.std(),
        'max': col_data.max(),
        'min': col_data.min(),
        'median': col_data.median(),
        'q25': col_data.quantile(0.25),
        'q75': col_data.quantile(0.75),
    }


def route_length_vs_ridership(gtfs_metrics, ridership):
    """Analyze relationship between route length and ridership."""
    print("  Running route length vs ridership analysis...")

    all_metrics = [m for m in gtfs_metrics if m['ridership_col'] in ridership.columns]
    rail_metrics = [m for m in all_metrics if m['type'] == 'rail' and m['avg_route_length_km'] > 0]
    bus_metrics = [m for m in all_metrics if m['type'] == 'bus']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    valid = rail_metrics
    services = [m['service'] for m in valid]
    lengths = [m['avg_route_length_km'] for m in valid]
    ridership_means = [compute_service_ridership_stats(ridership, m['ridership_col']).get('mean', 0) for m in valid]

    axes[0, 0].scatter(lengths, ridership_means, c='steelblue', s=100, alpha=0.7)
    for i, s in enumerate(services):
        axes[0, 0].annotate(s, (lengths[i], ridership_means[i]),
                            textcoords='offset points', xytext=(5, 5), fontsize=7)
    axes[0, 0].set_xlabel('Average Route Length (km)')
    axes[0, 0].set_ylabel('Mean Daily Ridership')
    axes[0, 0].set_title('Route Length vs Mean Ridership\n(Rail Services Only)')

    valid_data = [(l, r) for l, r in zip(lengths, ridership_means) if r > 0]
    if len(valid_data) > 3:
        l_vals = np.array([v[0] for v in valid_data])
        r_vals = np.array([v[1] for v in valid_data])
        slope, intercept, r, p, _ = stats.linregress(l_vals, r_vals)
        x_line = np.linspace(min(l_vals), max(l_vals), 100)
        axes[0, 0].plot(x_line, slope * x_line + intercept, color='grey',
                         linestyle='--', linewidth=2, alpha=0.7)
        axes[0, 0].text(0.05, 0.95, f'r={r:.3f}, p={p:.4f}',
                        transform=axes[0, 0].transAxes, fontsize=9,
                        verticalalignment='top')

    r_lengths = [m['avg_route_length_km'] for m in rail_metrics]
    r_ridership = [compute_service_ridership_stats(ridership, m['ridership_col']).get('mean', 0) for m in rail_metrics]
    axes[0, 1].scatter(r_lengths, r_ridership, c='steelblue', s=100, alpha=0.7)
    for i, m in enumerate(rail_metrics):
        axes[0, 1].annotate(m['service'], (r_lengths[i], r_ridership[i]),
                            textcoords='offset points', xytext=(5, 5), fontsize=8)
    axes[0, 1].set_xlabel('Route Length (km)')
    axes[0, 1].set_ylabel('Mean Daily Ridership')
    axes[0, 1].set_title('Rail Services: Route Length vs Ridership')

    if bus_metrics:
        b_services = [m['service'] for m in bus_metrics]
        b_ridership = [compute_service_ridership_stats(ridership, m['ridership_col']).get('mean', 0) for m in bus_metrics]
        axes[1, 0].barh(b_services, b_ridership, color='coral', alpha=0.8)
        for i, (svc, r) in enumerate(zip(b_services, b_ridership)):
            axes[1, 0].text(r + 500, i, 'Route length: N/A\n(GTFS shapes unavailable)', va='center', fontsize=8, color='grey')
        axes[1, 0].set_xlabel('Mean Daily Ridership')
        axes[1, 0].set_title('Bus Services: Route Length vs Ridership\n(Route length unavailable in GTFS)')
    else:
        axes[1, 0].text(0.5, 0.5, 'No bus services with route length data', ha='center', va='center')

    all_lengths = [m['avg_route_length_km'] for m in rail_metrics]
    all_density = [m['stop_density'] for m in rail_metrics]
    if all_lengths and all_density:
        axes[1, 1].scatter(all_density, all_lengths, c='mediumpurple', s=100, alpha=0.7)
        for i, m in enumerate(rail_metrics):
            axes[1, 1].annotate(m['service'], (all_density[i], all_lengths[i]),
                                textcoords='offset points', xytext=(5, 5), fontsize=7)
        axes[1, 1].set_xlabel('Stop Density (stops/km)')
        axes[1, 1].set_ylabel('Route Length (km)')
        axes[1, 1].set_title('Stop Density vs Route Length (Rail Services)')

    all_lengths = [m['avg_route_length_km'] for m in valid if m['avg_route_length_km'] > 0]
    all_density = [m['stop_density'] for m in valid if m['avg_route_length_km'] > 0]
    if all_lengths and all_density:
        axes[1, 1].scatter(all_density, all_lengths, c='mediumpurple', s=100, alpha=0.7)
        for i, m in enumerate(valid):
            if m['avg_route_length_km'] > 0:
                axes[1, 1].annotate(m['service'], (all_density[i], all_lengths[i]),
                                    textcoords='offset points', xytext=(5, 5), fontsize=7)
        axes[1, 1].set_xlabel('Stop Density (stops/km)')
        axes[1, 1].set_ylabel('Route Length (km)')
        axes[1, 1].set_title('Stop Density vs Route Length by Service')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gtfs_ridership_route_length.png'), dpi=150)
    plt.close()
    print("    Saved gtfs_ridership_route_length.png")


def service_frequency_vs_ridership(gtfs_metrics, ridership):
    """Analyze relationship between service frequency and ridership."""
    print("  Running service frequency vs ridership analysis...")

    valid = [m for m in gtfs_metrics if m['avg_headway_mins'] > 0 and m['ridership_col'] in ridership.columns]
    if not valid:
        print("    Skipping - no frequency data available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    services = [m['service'] for m in valid]
    headways = [m['avg_headway_mins'] for m in valid]
    ridership_means = [compute_service_ridership_stats(ridership, m['ridership_col']).get('mean', 0) for m in valid]

    colors = ['steelblue' if m['type'] == 'rail' else 'coral' for m in valid]

    axes[0, 0].scatter(headways, ridership_means, c=colors, s=100, alpha=0.7)
    for i, s in enumerate(services):
        axes[0, 0].annotate(s, (headways[i], ridership_means[i]),
                            textcoords='offset points', xytext=(5, 5), fontsize=7)
    axes[0, 0].set_xlabel('Average Headway (minutes)')
    axes[0, 0].set_ylabel('Mean Daily Ridership')
    axes[0, 0].set_title('Service Frequency (Headway) vs Ridership\n(Blue=Rail, Coral=Bus)')

    valid_data = [(h, r) for h, r in zip(headways, ridership_means) if r > 0]
    if len(valid_data) > 3:
        h_vals = np.array([v[0] for v in valid_data])
        r_vals = np.array([v[1] for v in valid_data])
        slope, intercept, r, p, _ = stats.linregress(h_vals, r_vals)
        x_line = np.linspace(min(h_vals), max(h_vals), 100)
        axes[0, 0].plot(x_line, slope * x_line + intercept, color='grey',
                         linestyle='--', linewidth=2, alpha=0.7)
        axes[0, 0].text(0.05, 0.95, f'r={r:.3f}, p={p:.4f}',
                        transform=axes[0, 0].transAxes, fontsize=9,
                        verticalalignment='top')

    sorted_services = sorted(zip(headways, services, ridership_means), key=lambda x: x[0])
    sorted_headways = [s[0] for s in sorted_services]
    sorted_services_names = [s[1] for s in sorted_services]
    sorted_ridership = [s[2] for s in sorted_services]

    axes[0, 1].barh(sorted_services_names, sorted_headways, color='mediumpurple', alpha=0.8)
    axes[0, 1].set_xlabel('Average Headway (minutes)')
    axes[0, 1].set_title('Average Headway by Service\n(Sorted)')
    for i, (h, r) in enumerate(zip(sorted_headways, sorted_ridership)):
        axes[0, 1].text(h + 1, i, f'Ridership: {r:,.0f}', va='center', fontsize=7)

    ridership_per_trip = [r / max(m['num_trips'], 1) for m, r in zip(valid, ridership_means) if m['num_trips'] > 0]
    valid_trip_services = [s for s, m in zip(services, valid) if m['num_trips'] > 0]

    if ridership_per_trip:
        axes[1, 0].barh(valid_trip_services, ridership_per_trip, color='seagreen', alpha=0.8)
        axes[1, 0].set_xlabel('Average Ridership per Trip')
        axes[1, 0].set_title('Service Efficiency:\nRidership per Scheduled Trip')

    num_trips = [m['num_trips'] for m in valid if m['num_trips'] > 0]
    valid_services = [s for s, m in zip(services, valid) if m['num_trips'] > 0]
    if num_trips:
        axes[1, 1].barh(valid_services, num_trips, color='steelblue', alpha=0.8)
        axes[1, 1].set_xlabel('Number of Scheduled Trips')
        axes[1, 1].set_title('Total Scheduled Trips by Service')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gtfs_ridership_frequency.png'), dpi=150)
    plt.close()
    print("    Saved gtfs_ridership_frequency.png")


def stop_count_vs_ridership(gtfs_metrics, ridership):
    """Analyze relationship between stop count and ridership."""
    print("  Running stop count vs ridership analysis...")

    valid = [m for m in gtfs_metrics if m['num_stops'] > 0 and m['ridership_col'] in ridership.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    services = [m['service'] for m in valid]
    num_stops = [m['num_stops'] for m in valid]
    ridership_means = [compute_service_ridership_stats(ridership, m['ridership_col']).get('mean', 0) for m in valid]

    colors = ['steelblue' if m['type'] == 'rail' else 'coral' for m in valid]

    axes[0, 0].scatter(num_stops, ridership_means, c=colors, s=100, alpha=0.7)
    for i, s in enumerate(services):
        axes[0, 0].annotate(s, (num_stops[i], ridership_means[i]),
                            textcoords='offset points', xytext=(5, 5), fontsize=7)
    axes[0, 0].set_xlabel('Number of Stops')
    axes[0, 0].set_ylabel('Mean Daily Ridership')
    axes[0, 0].set_title('Stop Count vs Mean Ridership\n(Blue=Rail, Coral=Bus)')

    valid_data = [(s, r) for s, r in zip(num_stops, ridership_means) if r > 0]
    if len(valid_data) > 3:
        s_vals = np.array([v[0] for v in valid_data])
        r_vals = np.array([v[1] for v in valid_data])
        slope, intercept, r, p, _ = stats.linregress(s_vals, r_vals)
        x_line = np.linspace(min(s_vals), max(s_vals), 100)
        axes[0, 0].plot(x_line, slope * x_line + intercept, color='grey',
                         linestyle='--', linewidth=2, alpha=0.7)
        axes[0, 0].text(0.05, 0.95, f'r={r:.3f}, p={p:.4f}',
                        transform=axes[0, 0].transAxes, fontsize=9,
                        verticalalignment='top')

    sorted_metrics = sorted(zip(num_stops, services, ridership_means), key=lambda x: x[0])
    sorted_stops = [s[0] for s in sorted_metrics]
    sorted_names = [s[1] for s in sorted_metrics]
    sorted_ridership = [s[2] for s in sorted_metrics]

    colors_sorted = ['steelblue' if m['type'] == 'rail' else 'coral' for m in valid]
    axes[0, 1].barh(sorted_names, sorted_stops, color=colors_sorted, alpha=0.8)
    axes[0, 1].set_xlabel('Number of Stops')
    axes[0, 1].set_title('Stop Count by Service\n(Sorted)')

    ridership_per_stop = [r / max(s, 1) for s, r in zip(num_stops, ridership_means) if s > 0]
    valid_names = [m['service'] for m in valid if m['num_stops'] > 0]
    if ridership_per_stop:
        axes[1, 0].barh(valid_names, ridership_per_stop, color='seagreen', alpha=0.8)
        axes[1, 0].set_xlabel('Ridership per Stop')
        axes[1, 0].set_title('Service Efficiency:\nRidership per Stop')

    stops_km = [m['stop_density'] for m in valid if m['avg_route_length_km'] > 0]
    density_ridership = [compute_service_ridership_stats(ridership, m['ridership_col']).get('mean', 0)
                         for m in valid if m['avg_route_length_km'] > 0]
    density_services = [m['service'] for m in valid if m['avg_route_length_km'] > 0]

    if stops_km and density_ridership:
        axes[1, 1].scatter(stops_km, density_ridership, c='mediumpurple', s=100, alpha=0.7)
        for i, s in enumerate(density_services):
            axes[1, 1].annotate(s, (stops_km[i], density_ridership[i]),
                                textcoords='offset points', xytext=(5, 5), fontsize=7)
        axes[1, 1].set_xlabel('Stop Density (stops/km)')
        axes[1, 1].set_ylabel('Mean Daily Ridership')
        axes[1, 1].set_title('Stop Density vs Ridership')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gtfs_ridership_stops.png'), dpi=150)
    plt.close()
    print("    Saved gtfs_ridership_stops.png")


def ridership_summary_comparison(gtfs_metrics, ridership):
    """Summary comparison: all services with GTFS metrics and ridership."""
    print("  Running ridership summary comparison...")

    summary_data = []
    for m in gtfs_metrics:
        stats_dict = compute_service_ridership_stats(ridership, m['ridership_col'])
        if stats_dict:
            summary_data.append({
                'Service': m['service'],
                'Type': m['type'],
                'Num_Stops': m['num_stops'],
                'Num_Routes': m['num_routes'],
                'Num_Trips': m['num_trips'],
                'Avg_Route_Length_km': m['avg_route_length_km'],
                'Avg_Headway_mins': m['avg_headway_mins'],
                'Stop_Density': m['stop_density'],
                'Mean_Ridership': stats_dict.get('mean', 0),
                'Median_Ridership': stats_dict.get('median', 0),
                'Max_Ridership': stats_dict.get('max', 0),
                'Ridership_Std': stats_dict.get('std', 0),
                'Ridership_per_Trip': stats_dict.get('mean', 0) / max(m['num_trips'], 1),
                'Ridership_per_Stop': stats_dict.get('mean', 0) / max(m['num_stops'], 1),
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Mean_Ridership', ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = ['steelblue' if t == 'rail' else 'coral' for t in summary_df['Type']]
    axes[0, 0].barh(summary_df['Service'], summary_df['Mean_Ridership'],
                    color=colors, alpha=0.8)
    axes[0, 0].set_xlabel('Mean Daily Ridership')
    axes[0, 0].set_title('Mean Daily Ridership by Service')
    axes[0, 0].invert_yaxis()

    cols_to_compare = ['Num_Stops', 'Avg_Route_Length_km', 'Avg_Headway_mins', 'Stop_Density']
    normalized = summary_df[cols_to_compare].copy()
    for col in cols_to_compare:
        max_val = normalized[col].max()
        min_val = normalized[col].min()
        if max_val > min_val:
            normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
        else:
            normalized[col] = 0

    normalized.index = summary_df['Service']
    normalized.plot(kind='bar', ax=axes[0, 1], alpha=0.8, width=0.8)
    axes[0, 1].set_title('Normalized GTFS Metrics by Service')
    axes[0, 1].set_ylabel('Normalized Value (0-1)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend(loc='upper right', fontsize=8)

    summary_melted = summary_df.melt(id_vars=['Service'], value_vars=['Ridership_per_Trip', 'Ridership_per_Stop'])
    pivot = summary_melted.pivot_table(index='Service', columns='variable', values='value')
    pivot = pivot.reindex(summary_df['Service'])
    pivot.plot(kind='bar', ax=axes[1, 0], color=['steelblue', 'coral'], alpha=0.8)
    axes[1, 0].set_title('Service Efficiency Metrics')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend(fontsize=8)

    summary_df['ridership_zscore'] = (summary_df['Mean_Ridership'] - summary_df['Mean_Ridership'].mean()) / summary_df['Mean_Ridership'].std()
    summary_df['gtfs_score'] = 0
    if summary_df['Num_Stops'].max() > 0:
        summary_df['gtfs_score'] += summary_df['Num_Stops'] / summary_df['Num_Stops'].max()
    if summary_df['Avg_Route_Length_km'].max() > 0:
        summary_df['gtfs_score'] += summary_df['Avg_Route_Length_km'] / summary_df['Avg_Route_Length_km'].max()
    if summary_df['Num_Trips'].max() > 0:
        summary_df['gtfs_score'] += summary_df['Num_Trips'] / summary_df['Num_Trips'].max()

    summary_df['gtfs_score'] = summary_df['gtfs_score'] / 3

    axes[1, 1].scatter(summary_df['gtfs_score'], summary_df['ridership_zscore'],
                       c=['steelblue' if t == 'rail' else 'coral' for t in summary_df['Type']],
                       s=100, alpha=0.7)
    for i, row in summary_df.iterrows():
        axes[1, 1].annotate(row['Service'],
                            (row['gtfs_score'], row['ridership_zscore']),
                            textcoords='offset points', xytext=(5, 5), fontsize=7)
    axes[1, 1].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('GTFS Network Score (normalized)')
    axes[1, 1].set_ylabel('Ridership Z-Score')
    axes[1, 1].set_title('GTFS Network Score vs Ridership Performance\n(Outliers show over/under-performance)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gtfs_ridership_summary.png'), dpi=150)
    plt.close()
    print("    Saved gtfs_ridership_summary.png")

    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'gtfs_ridership_summary.csv'), index=False)
    print("    Saved gtfs_ridership_summary.csv")
    print("\n  Summary Table:")
    print(summary_df[['Service', 'Type', 'Num_Stops', 'Avg_Route_Length_km',
                      'Avg_Headway_mins', 'Mean_Ridership', 'Ridership_per_Stop']].to_string())


def main():
    print("=" * 60)
    print("Bivariate EDA: GTFS vs Ridership")
    print("Objective: Route utilization & performance analysis")
    print("=" * 60)

    print("\nLoading ridership data...")
    ridership = load_ridership()
    service_cols = [c for c in ridership.columns if c not in ['total_ridership']]
    print(f"  Ridership: {ridership.shape[0]} days, services: {service_cols}")

    print("\nExtracting GTFS metrics for each service...")
    gtfs_metrics = []
    for service_name, config in SERVICE_CONFIG.items():
        metrics = extract_service_metrics(service_name, config)
        if metrics['num_stops'] > 0 or metrics['num_trips'] > 0:
            gtfs_metrics.append(metrics)
            print(f"  {service_name}: stops={metrics['num_stops']}, routes={metrics['num_routes']}, "
                  f"trips={metrics['num_trips']}, route_len={metrics['avg_route_length_km']:.1f}km, "
                  f"headway={metrics['avg_headway_mins']:.1f}min")
        else:
            print(f"  {service_name}: No GTFS data found")

    print("\nRunning analysis...")
    route_length_vs_ridership(gtfs_metrics, ridership)
    service_frequency_vs_ridership(gtfs_metrics, ridership)
    stop_count_vs_ridership(gtfs_metrics, ridership)
    ridership_summary_comparison(gtfs_metrics, ridership)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()