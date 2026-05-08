"""
Spatial/Geographic Theme EDA
Population density maps, walking friction surfaces, GTFS stop heatmaps,
POI spatial clustering, admin-boundary overlaps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import json
import geopandas as gpd
from shapely.geometry import Point
import os
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


def load_spatial_data():
    """Load all spatial datasets."""
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

    return population, gdf_boundary, stops, pois


def plot_population_density_map(population):
    """Population density maps."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax1 = axes[0]
    scatter = ax1.scatter(population['longitude'], population['latitude'],
                          c=population['density_per_km2'], cmap='YlOrRd',
                          s=1, alpha=0.5)
    plt.colorbar(scatter, ax=ax1, label='Density (per km²)')
    ax1.set_title('Population Density', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_aspect('equal')

    ax2 = axes[1]
    scatter2 = ax2.scatter(population['longitude'], population['latitude'],
                           c=population['density_log'], cmap='viridis',
                           s=1, alpha=0.5)
    plt.colorbar(scatter2, ax=ax2, label='Log Density')
    ax2.set_title('Population Density (Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'population_density_map.png', dpi=150, bbox_inches='tight')
    plt.close()

    density_stats = population['density_per_km2'].describe().round(2)
    density_summary = pd.DataFrame({
        'statistic': density_stats.index,
        'value': density_stats.values
    })
    density_summary.to_csv(CSV_DIR / 'population_density_stats.csv', index=False)


def plot_gtfs_stop_heatmaps(stops, gdf_boundary):
    """GTFS stop heatmaps with boundary overlay."""
    if stops.empty:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.text(0.5, 0.5, 'No GTFS stop data available', ha='center', va='center', fontsize=14)
        plt.savefig(GRAPH_DIR / 'gtfs_stop_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax1 = axes[0]
    if len(gdf_boundary) > 0:
        gdf_boundary.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=1)
    ax1.scatter(stops['stop_lon'], stops['stop_lat'], c='red', s=2, alpha=0.5, label='Stops')
    ax1.set_title('All GTFS Stops', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')

    ax2 = axes[1]
    from scipy.stats import gaussian_kde
    x = stops['stop_lon'].dropna()
    y = stops['stop_lat'].dropna()
    if len(x) > 10:
        try:
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
            xmin, xmax = x.min() - 0.1, x.max() + 0.1
            ymin, ymax = y.min() - 0.1, y.max() + 0.1
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            zz = np.reshape(kde(positions).T, xx.shape)
            im = ax2.imshow(zz, cmap='hot', aspect='auto',
                           extent=[xmin, xmax, ymin, ymax], origin='lower')
            plt.colorbar(im, ax=ax2, label='Stop Density')
        except:
            ax2.scatter(x, y, s=2, alpha=0.5)
    ax2.set_title('Stop Density Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'gtfs_stop_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    stops_by_source = stops.groupby('source').size().reset_index(name='stop_count')
    stops_by_source.to_csv(CSV_DIR / 'stops_by_gtfs_source.csv', index=False)


def plot_poi_spatial_clustering(pois):
    """POI spatial clustering analysis."""
    if pois.empty:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.text(0.5, 0.5, 'No POI data available', ha='center', va='center', fontsize=14)
        plt.savefig(GRAPH_DIR / 'poi_spatial_clustering.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    pois_geo = pois.dropna(subset=['lat', 'lon']).copy()

    tag_cols = [c for c in pois_geo.columns if c.startswith('tags.')]
    all_tags = {}
    for col in tag_cols:
        value_counts = pois_geo[col].dropna().value_counts()
        for val, count in value_counts.items():
            all_tags[f"{col}.{val}"] = count

    tag_counts = pd.Series(all_tags).sort_values(ascending=False).head(20)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    ax1 = axes[0, 0]
    ax1.scatter(pois_geo['lon'], pois_geo['lat'], s=1, alpha=0.3, c='blue')
    ax1.set_title('All POI Locations', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_aspect('equal')

    ax2 = axes[0, 1]
    tag_counts.plot(kind='barh', ax=ax2, color='steelblue')
    ax2.set_title('Top 20 POI Tag Types', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Count')

    amenity_types = ['school', 'hospital', 'restaurant', 'bank', 'fuel']
    amenity_pois = pois_geo[pois_geo['tags.amenity'].isin(amenity_types)] if 'tags.amenity' in pois_geo.columns else pd.DataFrame()
    if len(amenity_pois) > 0:
        colors = {'school': 'blue', 'hospital': 'red', 'restaurant': 'green', 'bank': 'orange', 'fuel': 'purple'}
        ax_amenity = axes[1, 0]
        for amenity in amenity_types:
            subset = amenity_pois[amenity_pois['tags.amenity'] == amenity]
            if len(subset) > 0:
                ax_amenity.scatter(subset['lon'], subset['lat'], s=10, alpha=0.5,
                          label=f'{amenity} ({len(subset)})', c=colors.get(amenity, 'gray'))
        ax_amenity.legend(loc='upper right')
        ax_amenity.set_title('Key Amenities POIs', fontsize=12, fontweight='bold')
        ax_amenity.set_xlabel('Longitude')
        ax_amenity.set_ylabel('Latitude')
        ax_amenity.set_aspect('equal')
    else:
        axes[1, 0].text(0.5, 0.5, 'No key amenity POIs found', ha='center', va='center')

    ax4 = axes[1, 1]
    ax4.hist(pois_geo['lat'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.set_title('POI Latitude Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Latitude')
    ax4.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'poi_spatial_clustering.png', dpi=150, bbox_inches='tight')
    plt.close()

    poi_summary = pd.DataFrame({
        'total_pois': [len(pois_geo)],
        'lat_range': [f"{pois_geo['lat'].min():.2f} to {pois_geo['lat'].max():.2f}"],
        'lon_range': [f"{pois_geo['lon'].min():.2f} to {pois_geo['lon'].max():.2f}"]
    })
    poi_summary.to_csv(CSV_DIR / 'poi_summary_stats.csv', index=False)


def plot_admin_boundary_overlaps(gdf_boundary, stops, population):
    """Admin boundary overlaps with stops and population."""
    if gdf_boundary.empty:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.text(0.5, 0.5, 'No boundary data available', ha='center', va='center', fontsize=14)
        plt.savefig(GRAPH_DIR / 'admin_boundary_overlaps.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax1 = axes[0]
    gdf_boundary.plot(ax=ax1, facecolor='lightgreen', edgecolor='black', linewidth=1, alpha=0.5)
    if not stops.empty:
        ax1.scatter(stops['stop_lon'], stops['stop_lat'], c='red', s=1, alpha=0.3, label='GTFS Stops')
    ax1.set_title('Admin Boundaries with GTFS Stops', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')

    ax2 = axes[1]
    gdf_boundary.plot(ax=ax2, facecolor='lightblue', edgecolor='black', linewidth=1, alpha=0.5)
    ax2.scatter(population['longitude'], population['latitude'],
                c=population['density_log'], cmap='YlOrRd', s=1, alpha=0.5)
    ax2.set_title('Admin Boundaries with Population Density', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'admin_boundary_overlaps.png', dpi=150, bbox_inches='tight')
    plt.close()

    boundary_stats = gdf_boundary[['NAME_1', 'geometry']].copy()
    boundary_stats['area_km2'] = boundary_stats.geometry.area / 1e6
    boundary_stats = boundary_stats.drop(columns=['geometry'])
    boundary_stats.columns = ['state', 'area_km2']
    boundary_stats.to_csv(CSV_DIR / 'admin_boundary_stats.csv', index=False)


def plot_walking_friction_surface():
    """Walking friction surface visualization."""
    import rasterio
    from rasterio.plot import show

    friction_path = DATA_DIR / "walking_friction_clean.tif"

    if not friction_path.exists():
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.text(0.5, 0.5, 'Walking friction data not available', ha='center', va='center', fontsize=14)
        plt.savefig(GRAPH_DIR / 'walking_friction_surface.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    with rasterio.open(friction_path) as src:
        data = src.read(1)
        data_masked = np.ma.masked_where(data == src.nodata, data)
        show(data_masked, ax=ax, cmap='viridis', title='Walking Friction Surface')
    plt.savefig(GRAPH_DIR / 'walking_friction_surface.png', dpi=150, bbox_inches='tight')
    plt.close()

    friction_data = data[~np.isnan(data) & (data != src.nodata)]
    friction_stats = pd.DataFrame({
        'statistic': ['min', 'max', 'mean', 'median', 'std'],
        'value': [friction_data.min(), friction_data.max(),
                  friction_data.mean(), np.median(friction_data), friction_data.std()]
    })
    friction_stats.to_csv(CSV_DIR / 'walking_friction_stats.csv', index=False)


def run_spatial_eda():
    """Run all spatial EDA."""
    print("Loading spatial data...")
    population, gdf_boundary, stops, pois = load_spatial_data()
    print(f"  Population points: {len(population)}")
    print(f"  Admin boundaries: {len(gdf_boundary)}")
    print(f"  GTFS stops: {len(stops)}")
    print(f"  POIs: {len(pois)}")

    print("\nGenerating population density maps...")
    plot_population_density_map(population)

    print("Generating GTFS stop heatmaps...")
    plot_gtfs_stop_heatmaps(stops, gdf_boundary)

    print("Generating POI spatial clustering...")
    plot_poi_spatial_clustering(pois)

    print("Generating admin boundary overlaps...")
    plot_admin_boundary_overlaps(gdf_boundary, stops, population)

    print("Generating walking friction surface...")
    plot_walking_friction_surface()

    print(f"\nSpatial EDA complete. Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_spatial_eda()