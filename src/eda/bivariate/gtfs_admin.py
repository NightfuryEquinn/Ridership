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
                all_stops.append(df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'source']])
        except Exception as e:
            print(f'Error loading {dir_name}: {e}')
    return pd.concat(all_stops, ignore_index=True)

def load_admin_boundaries():
    gdf = gpd.read_file(f'{DATA_DIR}/gadm_mys_l1_clean.geojson')
    return gdf

def analyze_gtfs_by_admin():
    print('Loading GTFS stops...')
    stops_df = load_gtfs_stops()
    print(f'Loaded {len(stops_df)} GTFS stops')

    print('Loading admin boundaries...')
    admin_gdf = load_admin_boundaries()
    print(f'Loaded {len(admin_gdf)} admin zones: {admin_gdf["NAME_1"].tolist()}')

    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=[Point(lon, lat) for lon, lat in zip(stops_df['stop_lon'], stops_df['stop_lat'])],
        crs='EPSG:4326'
    )

    stops_utm = stops_gdf.to_crs(admin_gdf.crs)
    admin_utm = admin_gdf.to_crs(stops_utm.crs)

    stops_in_admin = gpd.sjoin(stops_utm, admin_utm, how='left', predicate='within')

    stops_in_admin['stop_id'] = stops_in_admin['stop_id_left'] if 'stop_id_left' in stops_in_admin.columns else stops_in_admin['stop_id']

    coverage_stats = []

    for _, admin_row in admin_utm.iterrows():
        admin_name = admin_row['NAME_1']
        admin_geom = admin_row.geometry

        stops_in_zone = stops_in_admin[stops_in_admin.get('NAME_1', None) == admin_name]

        stops_by_source = stops_in_zone.groupby('source').size() if len(stops_in_zone) > 0 else pd.Series()

        coverage_stats.append({
            'admin_zone': admin_name,
            'total_stops': len(stops_in_zone),
            'ktmb_stops': stops_by_source.get('gtfs_ktmb', 0),
            'rapid_bus_kl_stops': stops_by_source.get('gtfs_rapid_bus_kl', 0),
            'rapid_rail_kl_stops': stops_by_source.get('gtfs_rapid_rail_kl', 0),
            'area_km2': admin_geom.area / 1e6
        })

    coverage_df = pd.DataFrame(coverage_stats)
    coverage_df['stops_per_km2'] = coverage_df['total_stops'] / coverage_df['area_km2']
    coverage_df = coverage_df.sort_values('total_stops', ascending=False)
    coverage_df.to_csv(f'{OUTPUT_DIR}/gtfs_admin_coverage.csv', index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    coverage_df.plot(kind='bar', x='admin_zone', y='total_stops', ax=axes[0, 0],
                     color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Admin Zone (State)')
    axes[0, 0].set_ylabel('Number of GTFS Stops')
    axes[0, 0].set_title('GTFS Stop Count by Admin Zone')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    stacked_data = coverage_df[['admin_zone', 'ktmb_stops', 'rapid_bus_kl_stops', 'rapid_rail_kl_stops']]
    stacked_data = stacked_data.set_index('admin_zone')
    stacked_data.plot(kind='bar', stacked=True, ax=axes[0, 1], edgecolor='black')
    axes[0, 1].set_xlabel('Admin Zone (State)')
    axes[0, 1].set_ylabel('Number of GTFS Stops')
    axes[0, 1].set_title('GTFS Stops by Source and Admin Zone')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend(title='GTFS Source')

    coverage_df.plot(kind='bar', x='admin_zone', y='stops_per_km2', ax=axes[1, 0],
                     color='darkgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Admin Zone (State)')
    axes[1, 0].set_ylabel('Stops per km²')
    axes[1, 0].set_title('GTFS Stop Density by Admin Zone')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    ax_map = axes[1, 1]
    admin_utm.plot(ax=ax_map, column='NAME_1', legend=True, edgecolor='black', alpha=0.7)
    stops_utm.plot(ax=ax_map, color='red', markersize=5, alpha=0.6)
    ax_map.set_title('GTFS Stops by Admin Zone')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/gtfs_admin_coverage.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Results saved to {OUTPUT_DIR}')

    return coverage_df

if __name__ == '__main__':
    analyze_gtfs_by_admin()