"""
Accessibility & Equity Theme EDA
Who has good access to transit and services? Combine friction, stop density, POI,
population, and zone boundaries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.stats import gaussian_kde
import json
import rasterio
from rasterio.plot import show
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


def load_accessibility_data():
    """Load all data for accessibility analysis."""
    population = pd.read_csv(DATA_DIR / "population_density_clean.csv")

    gdf_boundary = gpd.read_file(DATA_DIR / "gadm_mys_l1_clean.geojson")

    stops_list = []
    for gtfs_dir in ['gtfs_rapid_bus_kl', 'gtfs_rapid_bus_penang', 'gtfs_ktmb', 'gtfs_rapid_rail_kl']:
        gtfs_path = DATA_DIR / gtfs_dir / "stops.txt"
        if gtfs_path.exists():
            stops_df = pd.read_csv(gtfs_path)
            stops_df['source'] = gtfs_dir
            stops_list.append(stops_df)
    stops = pd.concat(stops_list, ignore_index=True) if stops_list else pd.DataFrame()

    with open(DATA_DIR / "osm_pois_clean.json", 'r', encoding='utf-8') as f:
        pois_data = json.load(f)
    pois = pd.json_normalize(pois_data['elements'])

    friction_path = DATA_DIR / "walking_friction_clean.tif"
    friction_data = None
    if friction_path.exists():
        with rasterio.open(friction_path) as src:
            friction_data = src.read(1)

    return population, gdf_boundary, stops, pois, friction_data


def calculate_stop_accessibility(stops, gdf_boundary):
    """Calculate stop accessibility metrics by admin boundary."""
    if stops.empty or gdf_boundary.empty:
        return pd.DataFrame()

    stops_gdf = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops['stop_lon'], stops['stop_lat']), crs="EPSG:4326")

    stops_per_boundary = gpd.sjoin(stops_gdf, gdf_boundary, how='left', predicate='within')

    accessibility = stops_per_boundary.groupby('NAME_1').agg({
        'stop_id': 'count',
        'stop_lat': ['mean', 'std'],
        'stop_lon': ['mean', 'std']
    }).reset_index()
    accessibility.columns = ['state', 'stop_count', 'lat_mean', 'lat_std', 'lon_mean', 'lon_std']
    return accessibility


def plot_stop_density_by_population(population, stops, gdf_boundary):
    """Stop density vs population analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    ax1 = axes[0, 0]
    if len(gdf_boundary) > 0:
        gdf_boundary.plot(ax=ax1, facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax1.scatter(population['longitude'], population['latitude'],
                c=population['density_log'], cmap='YlOrRd', s=0.5, alpha=0.3, label='Population')
    if not stops.empty:
        ax1.scatter(stops['stop_lon'], stops['stop_lat'], c='blue', s=2, alpha=0.5, label='Transit Stops')
    ax1.set_title('Population Density vs Transit Stops', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')

    ax2 = axes[0, 1]
    if not stops.empty:
        stops_per_source = stops.groupby('source').size().sort_values(ascending=True)
        stops_per_source.plot(kind='barh', ax=ax2, color='steelblue')
        ax2.set_title('Transit Stops by GTFS Source', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Stops')
    else:
        ax2.text(0.5, 0.5, 'No stop data available', ha='center', va='center', transform=ax2.transAxes)

    ax3 = axes[1, 0]
    pop_density_bins = pd.cut(population['density_log'], bins=10)
    pop_by_bin = population.groupby(pop_density_bins).size()
    ax3.bar(range(len(pop_by_bin)), pop_by_bin.values, color='coral', alpha=0.7)
    ax3.set_title('Population Distribution by Density (Log)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Density Decile')
    ax3.set_ylabel('Population Grid Cells')

    ax4 = axes[1, 1]
    accessibility = calculate_stop_accessibility(stops, gdf_boundary)
    if not accessibility.empty:
        accessibility_sorted = accessibility.sort_values('stop_count', ascending=True)
        ax4.barh(accessibility_sorted['state'], accessibility_sorted['stop_count'], color='seagreen')
        ax4.set_title('Transit Stops by State', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Number of Stops')
    else:
        ax4.text(0.5, 0.5, 'No accessibility data available', ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'stop_density_population.png', dpi=150, bbox_inches='tight')
    plt.close()

    if not stops.empty:
        stops_summary = stops.groupby('source').size().reset_index(name='stop_count')
        stops_summary.to_csv(CSV_DIR / 'stops_by_source.csv', index=False)

    return accessibility


def plot_friction_accessibility_overlay(population, friction_data):
    """Overlay walking friction with population density."""
    if friction_data is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.text(0.5, 0.5, 'Walking friction data not available', ha='center', va='center', fontsize=14)
        plt.savefig(GRAPH_DIR / 'friction_population_overlay.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax1 = axes[0]
    friction_masked = np.ma.masked_where(friction_data == 0, friction_data)
    im1 = ax1.imshow(friction_masked, cmap='viridis', aspect='auto')
    plt.colorbar(im1, ax=ax1, label='Friction Value')
    ax1.set_title('Walking Friction Surface', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Pixel X')
    ax1.set_ylabel('Pixel Y')

    ax2 = axes[1]
    pop_sorted = population.sort_values('density_log', ascending=False)
    threshold = int(len(pop_sorted) * 0.1)
    high_density = pop_sorted.head(threshold)
    ax2.scatter(population['longitude'], population['latitude'],
                c='lightgray', s=0.5, alpha=0.3, label='All Population')
    ax2.scatter(high_density['longitude'], high_density['latitude'],
                c='red', s=1, alpha=0.5, label='Top 10% Density')
    ax2.set_title('High-Density Population Locations', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.legend(loc='upper right')
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'friction_population_overlay.png', dpi=150, bbox_inches='tight')
    plt.close()

    friction_stats = pd.DataFrame({
        'statistic': ['min', 'max', 'mean', 'median', 'std', 'p10', 'p90'],
        'value': [np.nanmin(friction_data), np.nanmax(friction_data),
                  np.nanmean(friction_data), np.nanmedian(friction_data), np.nanstd(friction_data),
                  np.nanpercentile(friction_data, 10), np.nanpercentile(friction_data, 90)]
    })
    friction_stats.to_csv(CSV_DIR / 'friction_accessibility_stats.csv', index=False)


def plot_poi_service_accessibility(pois, population, gdf_boundary):
    """POI and service accessibility analysis."""
    if pois.empty:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.text(0.5, 0.5, 'No POI data available', ha='center', va='center', fontsize=14)
        plt.savefig(GRAPH_DIR / 'poi_service_accessibility.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    pois_geo = pois.dropna(subset=['lat', 'lon']).copy()
    amenity_col = 'tags.amenity' if 'tags.amenity' in pois_geo.columns else None
    pois_geo['amenity_type'] = pois_geo[amenity_col] if amenity_col else 'other'

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    ax1 = axes[0, 0]
    amenity_counts = pois_geo['amenity_type'].value_counts().head(15)
    amenity_counts.plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_title('Top 15 POI Amenity Types', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Count')

    ax2 = axes[0, 1]
    key_amenities = ['school', 'hospital', 'restaurant', 'bank', 'fuel']
    colors = {'school': 'blue', 'hospital': 'red', 'restaurant': 'green', 'bank': 'orange', 'fuel': 'purple'}
    for amenity in key_amenities:
        subset = pois_geo[pois_geo['amenity_type'] == amenity]
        if len(subset) > 0:
            ax2.scatter(subset['lon'], subset['lat'], s=5, alpha=0.3,
                       label=f'{amenity} ({len(subset)})', c=colors.get(amenity, 'gray'))
    ax2.set_title('Key Service POIs', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.legend(loc='upper right', markerscale=3)
    ax2.set_aspect('equal')

    ax3 = axes[1, 0]
    pois_per_state = []
    if not gdf_boundary.empty and len(pois_geo) > 0:
        pois_gdf = gpd.GeoDataFrame(pois_geo, geometry=gpd.points_from_xy(pois_geo['lon'], pois_geo['lat']), crs="EPSG:4326")
        pois_joined = gpd.sjoin(pois_gdf, gdf_boundary, how='left', predicate='within')
        pois_per_state = pois_joined.groupby('NAME_1').size().reset_index(name='poi_count')
        pois_per_state = pois_per_state.dropna().sort_values('poi_count', ascending=True)
        if len(pois_per_state) > 0:
            ax3.barh(pois_per_state['NAME_1'], pois_per_state['poi_count'], color='teal')
            ax3.set_title('POIs by State', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Count')
        else:
            ax3.text(0.5, 0.5, 'No POI-state overlap data', ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'No boundary data available', ha='center', va='center', transform=ax3.transAxes)

    ax4 = axes[1, 1]
    service_categories = {
        'education': pois_geo[pois_geo['amenity_type'] == 'school'],
        'healthcare': pois_geo[pois_geo['amenity_type'] == 'hospital'],
        'food': pois_geo[pois_geo['amenity_type'] == 'restaurant'],
        'finance': pois_geo[pois_geo['amenity_type'] == 'bank'],
        'transport': pois_geo[pois_geo['amenity_type'] == 'fuel']
    }
    service_counts = {k: len(v) for k, v in service_categories.items()}
    ax4.pie(service_counts.values(), labels=service_counts.keys(), autopct='%1.1f%%', colors=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
    ax4.set_title('Service POI Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'poi_service_accessibility.png', dpi=150, bbox_inches='tight')
    plt.close()

    poi_summary = pd.DataFrame({
        'category': list(service_counts.keys()),
        'count': list(service_counts.values())
    })
    poi_summary.to_csv(CSV_DIR / 'poi_service_summary.csv', index=False)


def plot_accessibility_equity_index(population, stops, pois, gdf_boundary):
    """Combined accessibility equity index by state."""
    if gdf_boundary.empty:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.text(0.5, 0.5, 'No boundary data available', ha='center', va='center', fontsize=14)
        plt.savefig(GRAPH_DIR / 'accessibility_equity_index.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax1 = axes[0]
    gdf_boundary['area_km2'] = gdf_boundary.geometry.area / 1e6

    pops_in_boundary = []
    for idx, row in gdf_boundary.iterrows():
        mask = (population['longitude'] >= row.geometry.bounds[0]) & \
               (population['longitude'] <= row.geometry.bounds[2]) & \
               (population['latitude'] >= row.geometry.bounds[1]) & \
               (population['latitude'] <= row.geometry.bounds[3])
        pops_in_boundary.append(population[mask]['density_per_km2'].sum())

    gdf_boundary['pop_total'] = pops_in_boundary
    gdf_boundary['pop_density'] = gdf_boundary['pop_total'] / gdf_boundary['area_km2']

    stops_gdf = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops['stop_lon'], stops['stop_lat']), crs="EPSG:4326") if not stops.empty else gpd.GeoDataFrame()
    stops_per_state = []
    if not stops.empty:
        stops_joined = gpd.sjoin(stops_gdf, gdf_boundary, how='left', predicate='within')
        stops_per_state = stops_joined.groupby('NAME_1').size().reset_index(name='stop_count')

    equity_data = gdf_boundary[['NAME_1', 'pop_total', 'area_km2']].copy()
    equity_data = equity_data.merge(stops_per_state, on='NAME_1', how='left')
    equity_data['stop_count'] = equity_data['stop_count'].fillna(0)
    equity_data['stops_per_100k_pop'] = (equity_data['stop_count'] / equity_data['pop_total'] * 100000).replace([np.inf, -np.inf], 0)

    equity_data_sorted = equity_data.sort_values('stops_per_100k_pop', ascending=True)
    ax1.barh(equity_data_sorted['NAME_1'], equity_data_sorted['stops_per_100k_pop'], color='seagreen')
    ax1.set_title('Transit Stops per 100K Population by State', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Stops per 100K Population')

    ax2 = axes[1]
    equity_data_sorted = equity_data.sort_values('pop_total', ascending=True)
    ax2.barh(equity_data_sorted['NAME_1'], equity_data_sorted['pop_total'] / 1e6, color='coral')
    ax2.set_title('Population by State (Millions)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Population (Millions)')

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'accessibility_equity_index.png', dpi=150, bbox_inches='tight')
    plt.close()

    equity_data.to_csv(CSV_DIR / 'accessibility_equity_by_state.csv', index=False)


def run_accessibility_equity_eda():
    """Run all accessibility/equity EDA."""
    print("Loading data for accessibility/equity analysis...")
    population, gdf_boundary, stops, pois, friction_data = load_accessibility_data()
    print(f"  Population points: {len(population)}")
    print(f"  Admin boundaries: {len(gdf_boundary)}")
    print(f"  GTFS stops: {len(stops)}")
    print(f"  POIs: {len(pois)}")
    print(f"  Friction data: {'Available' if friction_data is not None else 'Not available'}")

    print("\nGenerating stop density vs population analysis...")
    accessibility = plot_stop_density_by_population(population, stops, gdf_boundary)

    print("Generating friction-population overlay...")
    plot_friction_accessibility_overlay(population, friction_data)

    print("Generating POI service accessibility...")
    plot_poi_service_accessibility(pois, population, gdf_boundary)

    print("Generating accessibility equity index...")
    plot_accessibility_equity_index(population, stops, pois, gdf_boundary)

    print(f"\nAccessibility/Equity EDA complete. Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_accessibility_equity_eda()