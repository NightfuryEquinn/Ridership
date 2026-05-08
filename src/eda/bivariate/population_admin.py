import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'C:\Users\xianz\OneDrive - Universiti Malaya\master\research_project\data\cleaned'
OUTPUT_DIR = r'C:\Users\xianz\OneDrive - Universiti Malaya\master\research_project\src\eda\bivariate\results'

def load_population():
    pop_df = pd.read_csv(f'{DATA_DIR}/population_density_clean.csv')
    return pop_df

def load_admin_boundaries():
    gdf = gpd.read_file(f'{DATA_DIR}/gadm_mys_l1_clean.geojson')
    return gdf

def analyze_population_by_admin():
    print('Loading population data...')
    pop_df = load_population()
    print(f'Loaded {len(pop_df)} population points')

    print('Loading admin boundaries...')
    admin_gdf = load_admin_boundaries()
    print(f'Loaded {len(admin_gdf)} admin zones: {admin_gdf["NAME_1"].tolist()}')

    pop_gdf = gpd.GeoDataFrame(
        pop_df,
        geometry=[Point(lon, lat) for lon, lat in zip(pop_df['longitude'], pop_df['latitude'])],
        crs='EPSG:4326'
    )

    pop_utm = pop_gdf.to_crs(admin_gdf.crs)
    admin_utm = admin_gdf.to_crs(pop_utm.crs)

    pop_in_admin = gpd.sjoin(pop_utm, admin_utm, how='left', predicate='within')

    pop_stats = []

    for _, admin_row in admin_utm.iterrows():
        admin_name = admin_row['NAME_1']
        admin_geom = admin_row.geometry

        pop_in_zone = pop_in_admin[pop_in_admin.get('NAME_1', None) == admin_name]

        if len(pop_in_zone) > 0:
            pop_stats.append({
                'admin_zone': admin_name,
                'pop_point_count': len(pop_in_zone),
                'mean_density': pop_in_zone['density_per_km2'].mean(),
                'median_density': pop_in_zone['density_per_km2'].median(),
                'std_density': pop_in_zone['density_per_km2'].std(),
                'min_density': pop_in_zone['density_per_km2'].min(),
                'max_density': pop_in_zone['density_per_km2'].max(),
                'total_density_sum': pop_in_zone['density_per_km2'].sum(),
                'area_km2': admin_geom.area / 1e6,
                'pop_density_per_km2': pop_in_zone['density_per_km2'].sum() / (admin_geom.area / 1e6)
            })
        else:
            pop_stats.append({
                'admin_zone': admin_name,
                'pop_point_count': 0,
                'mean_density': 0,
                'median_density': 0,
                'std_density': 0,
                'min_density': 0,
                'max_density': 0,
                'total_density_sum': 0,
                'area_km2': admin_geom.area / 1e6,
                'pop_density_per_km2': 0
            })

    pop_stats_df = pd.DataFrame(pop_stats)
    pop_stats_df = pop_stats_df.sort_values('pop_point_count', ascending=False)
    pop_stats_df.to_csv(f'{OUTPUT_DIR}/population_admin_stats.csv', index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pop_stats_df.plot(kind='bar', x='admin_zone', y='pop_point_count', ax=axes[0, 0],
                      color='coral', edgecolor='black')
    axes[0, 0].set_xlabel('Admin Zone (State)')
    axes[0, 0].set_ylabel('Population Grid Points')
    axes[0, 0].set_title('Population Grid Points by Admin Zone')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    pop_stats_df.plot(kind='bar', x='admin_zone', y='mean_density', ax=axes[0, 1],
                      color='purple', edgecolor='black')
    axes[0, 1].set_xlabel('Admin Zone (State)')
    axes[0, 1].set_ylabel('Mean Population Density (per km²)')
    axes[0, 1].set_title('Mean Population Density by Admin Zone')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    pop_stats_df.plot(kind='bar', x='admin_zone', y='pop_density_per_km2', ax=axes[1, 0],
                      color='teal', edgecolor='black')
    axes[1, 0].set_xlabel('Admin Zone (State)')
    axes[1, 0].set_ylabel('Population Density per km²')
    axes[1, 0].set_title('Aggregate Population Density by Admin Zone')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    admin_utm.plot(ax=axes[1, 1], column='NAME_1', legend=True, edgecolor='black', alpha=0.7)
    pop_utm_sample = pop_utm.sample(min(5000, len(pop_utm)))
    pop_utm_sample.plot(ax=axes[1, 1], column='density_per_km2', cmap='YlOrRd',
                        markersize=2, alpha=0.5, legend=True)
    axes[1, 1].set_title('Population Density by Admin Zone')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/population_admin_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Results saved to {OUTPUT_DIR}')

    return pop_stats_df

if __name__ == '__main__':
    analyze_population_by_admin()