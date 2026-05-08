"""
Network Performance Theme EDA
Is the transit network performing well where demand is? GTFS routes vs ridership
vs population vs POI attractors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "cleaned"
OUTPUT_DIR = BASE_DIR / "src" / "eda" / "theme" / "results"

GRAPH_DIR = OUTPUT_DIR / "graphs"
CSV_DIR = OUTPUT_DIR / "csv"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


def load_network_data():
    """Load all data for network performance analysis."""
    ridership = pd.read_csv(DATA_DIR / "ridership_headline_clean.csv", parse_dates=['date'])
    population = pd.read_csv(DATA_DIR / "population_density_clean.csv")

    gdf_boundary = gpd.read_file(DATA_DIR / "gadm_mys_l1_clean.geojson")

    stops_list = []
    routes_list = []
    for gtfs_dir in ['gtfs_rapid_bus_kl', 'gtfs_rapid_bus_penang', 'gtfs_ktmb', 'gtfs_rapid_rail_kl']:
        gtfs_path = DATA_DIR / gtfs_dir / "stops.txt"
        routes_path = DATA_DIR / gtfs_dir / "routes.txt"
        if gtfs_path.exists():
            stops_df = pd.read_csv(gtfs_path)
            stops_df['source'] = gtfs_dir
            stops_list.append(stops_df)
        if routes_path.exists():
            routes_df = pd.read_csv(routes_path)
            routes_df['source'] = gtfs_dir
            routes_list.append(routes_df)

    stops = pd.concat(stops_list, ignore_index=True) if stops_list else pd.DataFrame()
    routes = pd.concat(routes_list, ignore_index=True) if routes_list else pd.DataFrame()

    with open(DATA_DIR / "osm_pois_clean.json", 'r', encoding='utf-8') as f:
        pois_data = json.load(f)
    pois = pd.json_normalize(pois_data['elements'])

    return ridership, population, gdf_boundary, stops, routes, pois


def analyze_route_coverage(stops, routes, population, gdf_boundary):
    """Analyze route coverage vs population and POIs."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    ax1 = axes[0, 0]
    stops_per_source = stops.groupby('source').size().sort_values(ascending=True)
    stops_per_source.plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_title('Transit Stops by GTFS Source', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Stops')

    ax2 = axes[0, 1]
    if not routes.empty:
        routes_per_source = routes.groupby('source').size().sort_values(ascending=True)
        routes_per_source.plot(kind='barh', ax=ax2, color='coral')
        ax2.set_title('Transit Routes by GTFS Source', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Routes')
    else:
        ax2.text(0.5, 0.5, 'No route data available', ha='center', va='center', transform=ax2.transAxes)

    ax3 = axes[1, 0]
    if not stops.empty and not gdf_boundary.empty:
        stops_gdf = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops['stop_lon'], stops['stop_lat']), crs="EPSG:4326")
        stops_joined = gpd.sjoin(stops_gdf, gdf_boundary, how='left', predicate='within')
        stops_per_state = stops_joined.groupby('NAME_1').size().reset_index(name='stop_count')
        stops_per_state = stops_per_state.dropna().sort_values('stop_count', ascending=True)
        if len(stops_per_state) > 0:
            ax3.barh(stops_per_state['NAME_1'], stops_per_state['stop_count'], color='seagreen')
            ax3.set_title('Transit Stops by State', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Number of Stops')
        else:
            ax3.text(0.5, 0.5, 'No stops-state overlap', ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'No spatial overlap data', ha='center', va='center', transform=ax3.transAxes)

    ax4 = axes[1, 1]
    pop_by_region = []
    if not gdf_boundary.empty:
        for idx, row in gdf_boundary.iterrows():
            mask = (population['longitude'] >= row.geometry.bounds[0]) & \
                   (population['longitude'] <= row.geometry.bounds[2]) & \
                   (population['latitude'] >= row.geometry.bounds[1]) & \
                   (population['latitude'] <= row.geometry.bounds[3])
            pop_count = population[mask].shape[0]
            pop_by_region.append({'state': row['NAME_1'], 'grid_cells': pop_count})
        pop_df = pd.DataFrame(pop_by_region).sort_values('grid_cells', ascending=True)
        ax4.barh(pop_df['state'], pop_df['grid_cells'], color='purple', alpha=0.6)
        ax4.set_title('Population Grid Cells by State', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Grid Cell Count')

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'route_coverage_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    summary_data = []
    if not stops.empty:
        summary_data.append({'metric': 'total_stops', 'value': len(stops)})
    if not routes.empty:
        summary_data.append({'metric': 'total_routes', 'value': len(routes)})
    if len(summary_data) > 0:
        pd.DataFrame(summary_data).to_csv(CSV_DIR / 'network_route_summary.csv', index=False)


def analyze_ridership_by_mode(ridership):
    """Analyze ridership performance by transit mode."""
    modes = ['bus_rkl', 'bus_rpn', 'rail_lrt_ampang', 'rail_mrt_kajang', 'rail_lrt_kj', 'rail_monorail', 'rail_mrt_pjy']
    mode_names = ['Bus RKL', 'Bus RPN', 'LRT Ampang', 'MRT Kajang', 'LRT Kelana Jaya', 'Monorail', 'MRT PJY']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    mode_stats = []
    for mode, name in zip(modes, mode_names):
        if mode in ridership.columns:
            mode_sum = ridership[mode].sum()
            mode_mean = ridership[mode].mean()
            if mode_sum > 0:
                mode_stats.append({'mode': name, 'total': mode_sum, 'daily_avg': mode_mean})

    mode_df = pd.DataFrame(mode_stats).sort_values('total', ascending=True)
    ax1.barh(mode_df['mode'], mode_df['total'] / 1e6, color='steelblue')
    ax1.set_title('Total Ridership by Transit Mode (Millions)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Total Ridership (Millions)')

    ax2 = axes[0, 1]
    ax2.barh(mode_df['mode'], mode_df['daily_avg'] / 1000, color='coral')
    ax2.set_title('Average Daily Ridership by Mode (Thousands)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Average Daily Ridership (Thousands)')

    ax3 = axes[1, 0]
    ridership_sorted = ridership.sort_values('total_ridership', ascending=False).head(100)
    ax3.plot(range(len(ridership_sorted)), ridership_sorted['total_ridership'].values / 1000, color='green', linewidth=1)
    ax3.set_title('Top 100 Days by Total Ridership', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Day Rank')
    ax3.set_ylabel('Total Ridership (Thousands)')

    ax4 = axes[1, 1]
    ridership['date'] = pd.to_datetime(ridership['date'])
    ridership['year_month'] = ridership['date'].dt.to_period('M')
    monthly_total = ridership.groupby('year_month')['total_ridership'].sum()
    monthly_total.plot(ax=ax4, marker='o', linewidth=2, color='purple')
    ax4.set_title('Monthly Total Ridership Over Time', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Total Ridership')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'ridership_by_mode.png', dpi=150, bbox_inches='tight')
    plt.close()

    mode_performance = pd.DataFrame(mode_stats)
    if not mode_performance.empty:
        mode_performance.to_csv(CSV_DIR / 'ridership_mode_performance.csv', index=False)


def analyze_population_poi_network_overlap(population, stops, pois, gdf_boundary):
    """Analyze spatial overlap of population, POIs, and transit network."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    ax1 = axes[0, 0]
    if not stops.empty:
        ax1.scatter(population['longitude'], population['latitude'],
                   c='lightgray', s=0.5, alpha=0.2, label='Population')
        ax1.scatter(stops['stop_lon'], stops['stop_lat'],
                   c='blue', s=5, alpha=0.5, label='Transit Stops')
        ax1.set_title('Population and Transit Stops', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.legend(loc='upper right')
        ax1.set_aspect('equal')
    else:
        ax1.text(0.5, 0.5, 'No stop data available', ha='center', va='center', transform=ax1.transAxes)

    ax2 = axes[0, 1]
    pois_geo = pois.dropna(subset=['lat', 'lon']).copy() if not pois.empty else pd.DataFrame()
    if len(pois_geo) > 0:
        ax2.scatter(pois_geo['lon'], pois_geo['lat'], c='green', s=1, alpha=0.3, label='POIs')
        if not stops.empty:
            ax2.scatter(stops['stop_lon'], stops['stop_lat'], c='blue', s=5, alpha=0.5, label='Stops')
        ax2.set_title('POIs and Transit Stops', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.legend(loc='upper right')
        ax2.set_aspect('equal')
    else:
        ax2.text(0.5, 0.5, 'No POI data available', ha='center', va='center', transform=ax2.transAxes)

    ax3 = axes[1, 0]
    if not gdf_boundary.empty and not stops.empty:
        gdf_boundary.plot(ax=ax3, facecolor='lightblue', edgecolor='black', linewidth=0.5, alpha=0.5)
        ax3.scatter(stops['stop_lon'], stops['stop_lat'], c='red', s=2, alpha=0.5, label='Stops')
        ax3.set_title('Transit Network Coverage by State', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax3.legend(loc='upper right')
        ax3.set_aspect('equal')
    else:
        ax3.text(0.5, 0.5, 'No boundary data', ha='center', va='center', transform=ax3.transAxes)

    ax4 = axes[1, 1]
    high_density = population[population['density_log'] > population['density_log'].quantile(0.9)]
    stops_near_high_density = stops[
        (stops['stop_lat'] >= high_density['latitude'].min()) &
        (stops['stop_lat'] <= high_density['latitude'].max()) &
        (stops['stop_lon'] >= high_density['longitude'].min()) &
        (stops['stop_lon'] <= high_density['longitude'].max())
    ] if not stops.empty and len(high_density) > 0 else pd.DataFrame()
    ax4.scatter(population['longitude'], population['latitude'],
               c=population['density_log'], cmap='YlOrRd', s=0.5, alpha=0.3)
    if not stops_near_high_density.empty:
        ax4.scatter(stops_near_high_density['stop_lon'], stops_near_high_density['stop_lat'],
                   c='blue', s=10, alpha=0.8, label=f'Stops in Top 10% Density ({len(stops_near_high_density)})')
    ax4.set_title('High-Density Areas and Transit Access', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.legend(loc='upper right')
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'population_poi_network_overlap.png', dpi=150, bbox_inches='tight')
    plt.close()

    overlap_stats = pd.DataFrame({
        'high_density_stops': [len(stops_near_high_density) if not stops_near_high_density.empty else 0],
        'total_stops': [len(stops) if not stops.empty else 0],
        'total_pop_points': [len(population)],
        'total_pois': [len(pois_geo) if len(pois_geo) > 0 else 0]
    })
    overlap_stats.to_csv(CSV_DIR / 'network_overlap_stats.csv', index=False)


def analyze_network_performance_summary(ridership, stops, population, pois):
    """Comprehensive network performance summary."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    ridership_modes = ridership[['bus_rkl', 'bus_rpn', 'rail_lrt_ampang', 'rail_mrt_kajang',
                                  'rail_lrt_kj', 'rail_monorail', 'rail_mrt_pjy']].sum()
    ridership_modes.plot(kind='pie', ax=ax1, autopct='%1.1f%%',
                         colors=plt.cm.Set3(np.linspace(0, 1, len(ridership_modes))))
    ax1.set_title('Ridership Share by Transit Mode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('')

    ax2 = axes[0, 1]
    bus_ridership = ridership['bus_rkl'].fillna(0) + ridership['bus_rpn'].fillna(0)
    rail_ridership = (ridership['rail_lrt_ampang'].fillna(0) + ridership['rail_mrt_kajang'].fillna(0) +
                      ridership['rail_lrt_kj'].fillna(0) + ridership['rail_monorail'].fillna(0) +
                      ridership['rail_mrt_pjy'].fillna(0))
    comparison = pd.DataFrame({'Bus': [bus_ridership.sum()], 'Rail': [rail_ridership.sum()]})
    comparison.plot(kind='bar', ax=ax2, color=['#3498db', '#e74c3c'])
    ax2.set_title('Total Bus vs Rail Ridership', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Ridership')
    ax2.set_xticklabels(['Ridership'], rotation=0)

    ax3 = axes[1, 0]
    pop_quantiles = population['density_per_km2'].quantile([0.25, 0.5, 0.75, 0.9])
    ax3.bar(['Q25', 'Q50', 'Q75', 'Q90'], pop_quantiles.values, color='teal')
    ax3.set_title('Population Density Quantiles', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Density (per km²)')
    for i, v in enumerate(pop_quantiles.values):
        ax3.text(i, v + 100, f'{v:.0f}', ha='center', va='bottom', fontsize=9)

    ax4 = axes[1, 1]
    pois_geo = pois.dropna(subset=['lat', 'lon']).copy() if not pois.empty else pd.DataFrame()
    if len(pois_geo) > 0:
        poi_amenity_col = 'tags.amenity' if 'tags.amenity' in pois_geo.columns else None
        if poi_amenity_col:
            poi_amenity_counts = pois_geo[poi_amenity_col].value_counts().head(10)
            poi_amenity_counts.plot(kind='barh', ax=ax4, color='seagreen')
            ax4.set_title('Top 10 POI Amenity Types', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Count')
        else:
            ax4.text(0.5, 0.5, 'No POI amenity data', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'No POI data', ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'network_performance_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    summary_metrics = pd.DataFrame({
        'metric': ['total_ridership', 'bus_ridership_share', 'rail_ridership_share',
                    'total_stops', 'total_pop_points', 'total_pois'],
        'value': [
            ridership['total_ridership'].sum(),
            bus_ridership.sum() / ridership['total_ridership'].sum() * 100,
            rail_ridership.sum() / ridership['total_ridership'].sum() * 100,
            len(stops) if not stops.empty else 0,
            len(population),
            len(pois_geo) if len(pois_geo) > 0 else 0
        ]
    }).round(2)
    summary_metrics.to_csv(CSV_DIR / 'network_performance_metrics.csv', index=False)


def run_network_performance_eda():
    """Run all network performance EDA."""
    print("Loading data for network performance analysis...")
    ridership, population, gdf_boundary, stops, routes, pois = load_network_data()
    print(f"  Ridership data: {len(ridership)} rows")
    print(f"  Population points: {len(population)}")
    print(f"  Admin boundaries: {len(gdf_boundary)}")
    print(f"  GTFS stops: {len(stops)}")
    print(f"  GTFS routes: {len(routes)}")
    print(f"  POIs: {len(pois)}")

    print("\nAnalyzing route coverage...")
    analyze_route_coverage(stops, routes, population, gdf_boundary)

    print("Analyzing ridership by mode...")
    analyze_ridership_by_mode(ridership)

    print("Analyzing population-POI-network overlap...")
    analyze_population_poi_network_overlap(population, stops, pois, gdf_boundary)

    print("Generating network performance summary...")
    analyze_network_performance_summary(ridership, stops, population, pois)

    print(f"\nNetwork Performance EDA complete. Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_network_performance_eda()