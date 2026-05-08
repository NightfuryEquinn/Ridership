import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import json
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

def load_osm_pois():
    with open(f'{DATA_DIR}/osm_pois_clean.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    pois = []
    for elem in data.get('elements', []):
        if elem.get('type') == 'node' and 'lat' in elem and 'lon' in elem:
            tags = elem.get('tags', {})
            poi_type = tags.get('amenity') or tags.get('shop') or tags.get('tourism') or 'other'
            pois.append({
                'poi_id': elem['id'],
                'lat': elem['lat'],
                'lon': elem['lon'],
                'poi_type': poi_type,
                'name': tags.get('name', '')
            })
    return pd.DataFrame(pois)

def calculate_connectivity(stops_df, pois_df, max_distance=1000):
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=[Point(lon, lat) for lon, lat in zip(stops_df['stop_lon'], stops_df['stop_lat'])],
        crs='EPSG:4326'
    ).to_crs('EPSG:3857')

    pois_gdf = gpd.GeoDataFrame(
        pois_df,
        geometry=[Point(lon, lat) for lon, lat in zip(pois_df['lon'], pois_df['lat'])],
        crs='EPSG:4326'
    ).to_crs('EPSG:3857')

    stops_coords = np.array([(geom.x, geom.y) for geom in stops_gdf.geometry])
    pois_coords = np.array([(geom.x, geom.y) for geom in pois_gdf.geometry])

    tree = cKDTree(pois_coords)

    connectivity_results = []
    for i, (stop_idx, stop_row) in enumerate(stops_gdf.iterrows()):
        stop_point = stops_coords[i]
        distances, indices = tree.query(stop_point, k=1)
        nearest_poi = pois_df.iloc[indices]

        connectivity_results.append({
            'stop_id': stop_row['stop_id'],
            'stop_name': stop_row['stop_name'],
            'stop_lon': stop_row['stop_lon'],
            'stop_lat': stop_row['stop_lat'],
            'source': stop_row['source'],
            'nearest_poi_id': nearest_poi['poi_id'],
            'nearest_poi_type': nearest_poi['poi_type'],
            'nearest_poi_name': nearest_poi['name'],
            'distance_m': distances
        })

        if i % 500 == 0:
            print(f'  Processed {i}/{len(stops_gdf)} stops')

    return pd.DataFrame(connectivity_results)

def analyze_transit_connectivity():
    print('Loading GTFS stops...')
    stops_df = load_gtfs_stops()
    print(f'Loaded {len(stops_df)} GTFS stops')

    print('Loading OSM POIs...')
    pois_df = load_osm_pois()
    print(f'Loaded {len(pois_df)} OSM POIs')

    poi_types = pois_df['poi_type'].value_counts()
    print(f'POI types: {poi_types.head(10).to_dict()}')

    print('Calculating connectivity...')
    connectivity_df = calculate_connectivity(stops_df, pois_df)
    connectivity_df.to_csv(f'{OUTPUT_DIR}/gtfs_osm_connectivity.csv', index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(connectivity_df['distance_m'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Distance to Nearest POI (m)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distance from GTFS Stops to Nearest POI')
    axes[0, 0].axvline(connectivity_df['distance_m'].median(), color='red', linestyle='--',
                        label=f'Median: {connectivity_df["distance_m"].median():.0f}m')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    poi_type_dist = connectivity_df.groupby('nearest_poi_type')['distance_m'].agg(['mean', 'count'])
    poi_type_dist = poi_type_dist[poi_type_dist['count'] >= 5].sort_values('mean')
    poi_type_dist['mean'].plot(kind='barh', ax=axes[0, 1], color='steelblue', edgecolor='black')
    axes[0, 1].set_xlabel('Mean Distance to Nearest POI (m)')
    axes[0, 1].set_ylabel('POI Type')
    axes[0, 1].set_title('Mean Distance to Nearest POI by Type')
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    source_stats = connectivity_df.groupby('source')['distance_m'].agg(['mean', 'median', 'std', 'count'])
    source_stats.reset_index().to_csv(f'{OUTPUT_DIR}/gtfs_osm_by_source.csv', index=False)
    source_stats['mean'].plot(kind='bar', ax=axes[1, 0], color='darkgreen', edgecolor='black')
    axes[1, 0].set_xlabel('GTFS Source')
    axes[1, 0].set_ylabel('Mean Distance (m)')
    axes[1, 0].set_title('Mean Distance to Nearest POI by GTFS Source')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    poi_connectivity_counts = connectivity_df['nearest_poi_type'].value_counts().head(10)
    poi_connectivity_counts.plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_ylabel('')
    axes[1, 1].set_title('Distribution of POI Types as Nearest to GTFS Stops')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/gtfs_osm_connectivity.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Results saved to {OUTPUT_DIR}')

    return connectivity_df

if __name__ == '__main__':
    analyze_transit_connectivity()