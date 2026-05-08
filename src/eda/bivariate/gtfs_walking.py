import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import rasterio
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

def get_walking_friction_at_points(lon, lat, raster_path):
    with rasterio.open(raster_path) as src:
        try:
            row, col = src.index(lon, lat)
            if row >= 0 and row < src.height and col >= 0 and col < src.width:
                return src.read(1)[row, col]
            return np.nan
        except:
            return np.nan

def analyze_walking_accessibility():
    print('Loading GTFS stops...')
    stops_df = load_gtfs_stops()
    print(f'Loaded {len(stops_df)} stops')

    print('Sampling walking friction at transit stops...')
    raster_path = f'{DATA_DIR}/walking_friction_clean.tif'

    with rasterio.open(raster_path) as src:
        friction_data = src.read(1)
        height, width = friction_data.shape
        transform = src.transform

        friction_values = []
        for idx, row in stops_df.iterrows():
            try:
                col, row_idx = src.index(row['stop_lon'], row['stop_lat'])
                if 0 <= row_idx < height and 0 <= col < width:
                    val = friction_data[row_idx, col]
                    friction_values.append(val if val != src.nodata else np.nan)
                else:
                    friction_values.append(np.nan)
            except:
                friction_values.append(np.nan)

            if idx % 1000 == 0:
                print(f'  Processed {idx}/{len(stops_df)} stops')

    stops_df['walking_friction'] = friction_values

    stops_valid = stops_df.dropna(subset=['walking_friction'])
    stops_valid.to_csv(f'{OUTPUT_DIR}/gtfs_walking_friction.csv', index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(stops_valid['walking_friction'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Walking Friction Value')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Walking Friction at GTFS Stops')
    axes[0, 0].grid(True, alpha=0.3)

    source_stats = stops_valid.groupby('source')['walking_friction'].agg(['mean', 'median', 'std', 'count'])
    source_stats.reset_index().to_csv(f'{OUTPUT_DIR}/gtfs_walking_by_source.csv', index=False)

    source_stats['mean'].plot(kind='bar', ax=axes[0, 1], color='steelblue', edgecolor='black')
    axes[0, 1].set_xlabel('GTFS Source')
    axes[0, 1].set_ylabel('Mean Walking Friction')
    axes[0, 1].set_title('Mean Walking Friction by GTFS Source')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    scatter = axes[1, 0].scatter(stops_valid['stop_lon'], stops_valid['stop_lat'],
                                  c=stops_valid['walking_friction'], cmap='RdYlGn_r',
                                  alpha=0.6, s=10)
    plt.colorbar(scatter, ax=axes[1, 0], label='Walking Friction')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')
    axes[1, 0].set_title('Spatial Distribution of Walking Friction at Transit Stops')

    axes[1, 1].boxplot([stops_valid[stops_valid['source'] == src]['walking_friction'].dropna()
                        for src in stops_valid['source'].unique()],
                       labels=stops_valid['source'].unique())
    axes[1, 1].set_xlabel('GTFS Source')
    axes[1, 1].set_ylabel('Walking Friction')
    axes[1, 1].set_title('Walking Friction Distribution by GTFS Source')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/gtfs_walking_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Results saved to {OUTPUT_DIR}')

    return stops_valid

if __name__ == '__main__':
    analyze_walking_accessibility()