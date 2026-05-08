import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'C:\Users\xianz\OneDrive - Universiti Malaya\master\research_project\data\cleaned'
OUTPUT_DIR = r'C:\Users\xianz\OneDrive - Universiti Malaya\master\research_project\src\eda\bivariate\results'

def load_gtfs_stops():
    gtfs_dirs = ['gtfs_ktmb', 'gtfs_rapid_bus_kl', 'gtfs_rapid_rail_kl']
    all_stops = []
    for dir_name in gtfs_dirs:
        try:
            stops_file = f'{DATA_DIR}/{dir_name}/stops.txt'
            df = pd.read_csv(stops_file)
            df['source'] = dir_name
            if 'stop_lat' in df.columns and 'stop_lon' in df.columns:
                all_stops.append(df[['stop_lat', 'stop_lon', 'source']])
        except Exception as e:
            print(f'Error loading {dir_name}: {e}')
    return pd.concat(all_stops, ignore_index=True)

def load_population():
    pop_df = pd.read_csv(f'{DATA_DIR}/population_density_clean.csv')
    return pop_df

def create_gtfs_gdf(stops_df):
    geometry = [Point(lon, lat) for lon, lat in zip(stops_df['stop_lon'], stops_df['stop_lat'])]
    return gpd.GeoDataFrame(stops_df, geometry=geometry, crs='EPSG:4326')

def create_population_gdf(pop_df):
    geometry = [Point(lon, lat) for lon, lat in zip(pop_df['longitude'], pop_df['latitude'])]
    gdf = gpd.GeoDataFrame(pop_df, geometry=geometry, crs='EPSG:4326')
    return gdf.to_crs('EPSG:3857')

def analyze_transit_coverage_gaps():
    print('Loading GTFS stops...')
    gtfs_stops = load_gtfs_stops()
    gtfs_gdf = create_gtfs_gdf(gtfs_stops)
    gtfs_utm = gtfs_gdf.to_crs('EPSG:3857')

    print('Loading population data...')
    pop_df = load_population()
    pop_gdf = create_population_gdf(pop_df)

    print('Analyzing transit coverage by population density...')
    buffer_distances = [500, 1000, 1500, 2000]
    results = []

    for dist in buffer_distances:
        gtfs_buffer = gtfs_utm.copy()
        gtfs_buffer['geometry'] = gtfs_buffer.geometry.buffer(dist)
        gtfs_buffer = gtfs_buffer[['geometry']].buffer(dist).unary_union

        covered_pop = pop_gdf[pop_gdf.geometry.within(gtfs_buffer)]
        coverage_pct = (len(covered_pop) / len(pop_gdf)) * 100

        results.append({
            'buffer_m': dist,
            'covered_pop_points': len(covered_pop),
            'total_pop_points': len(pop_gdf),
            'coverage_pct': coverage_pct
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{OUTPUT_DIR}/gtfs_population_coverage.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(results_df['buffer_m'], results_df['coverage_pct'], 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Buffer Distance (m)')
    axes[0].set_ylabel('Population Points Covered (%)')
    axes[0].set_title('GTFS Transit Coverage vs Population Points')
    axes[0].grid(True, alpha=0.3)

    pop_sorted = pop_gdf.sort_values('density_per_km2', ascending=False).reset_index(drop=True)
    pop_sorted['density_decile'] = pd.qcut(pop_sorted['density_per_km2'], 10, labels=False, duplicates='drop')
    axes[1].hist(pop_gdf['density_per_km2'], bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Population Density (per km²)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Population Density Distribution')
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/gtfs_population_coverage.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Results saved to {OUTPUT_DIR}')
    return results_df

if __name__ == '__main__':
    analyze_transit_coverage_gaps()