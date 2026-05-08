#!/usr/bin/env python3
"""
Bivariate EDA: Walking Friction vs OSM POI Density
Objective: Effective accessibility to services weighted by walkability

Analysis:
1. POI density vs mean walking friction scatter by admin zone
2. Effective accessibility index (POI count / mean friction per zone)
3. Spatial mismatch mapping: high-POI, high-friction zones
4. POI type accessibility by friction zone classification
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from shapely.geometry import Point
from scipy import stats
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = os.path.join(BASE_DIR, "../../../data/cleaned")
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
pd.options.display.max_columns = None


def load_osm_pois():
    print("  Loading OSM POIs...")
    with open(os.path.join(DATA_DIR, "osm_pois_clean.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)
    pois = []
    for el in data.get('elements', []):
        if el.get('type') == 'node' and 'lat' in el and 'lon' in el:
            tags = el.get('tags', {})
            pois.append({
                'poi_id': el['id'],
                'lat': el['lat'],
                'lon': el['lon'],
                'amenity': tags.get('amenity', 'other'),
                'shop': tags.get('shop', None),
                'tourism': tags.get('tourism', None),
                'name': tags.get('name', '')
            })
    pois_df = pd.DataFrame(pois)
    print(f"    Loaded {len(pois_df):,} POIs")
    return pois_df


def load_friction_surface():
    print("  Loading walking friction surface...")
    friction_path = os.path.join(DATA_DIR, "walking_friction_clean.tif")
    with rasterio.open(friction_path) as src:
        bounds = src.bounds
        crs_fric = src.crs
        res = src.res
    print(f"    Friction surface: bounds={bounds}, res={res}")
    return friction_path, crs_fric


def load_admin_boundaries():
    print("  Loading admin boundaries...")
    gdf = gpd.read_file(os.path.join(DATA_DIR, "gadm_mys_l1_clean.geojson"))
    print(f"    Loaded {len(gdf)} admin zones")
    return gdf


def sample_friction_at_pois(pois_df, friction_path, crs_fric):
    print("  Sampling friction at POI locations...")
    pois_gdf = gpd.GeoDataFrame(
        pois_df,
        geometry=[Point(lon, lat) for lon, lat in zip(pois_df['lon'], pois_df['lat'])],
        crs='EPSG:4326'
    )
    pois_proj = pois_gdf.to_crs(crs_fric)
    sample_points = [(pt.x, pt.y) for pt in pois_proj.geometry]

    with rasterio.open(friction_path) as src:
        samples = list(src.sample(sample_points))

    pois_df = pois_df.copy()
    pois_df['friction_value'] = [s[0] if len(s) > 0 and not np.isnan(s[0]) else np.nan for s in samples]

    valid = pois_df.dropna(subset=['friction_value'])
    print(f"    Valid friction samples: {len(valid):,} / {len(pois_df):,}")
    return pois_df


def aggregate_by_admin(pois_df, admin_gdf):
    print("  Aggregating POI-friction metrics by admin zone...")

    pois_gdf = gpd.GeoDataFrame(
        pois_df,
        geometry=[Point(lon, lat) for lon, lat in zip(pois_df['lon'], pois_df['lat'])],
        crs='EPSG:4326'
    )
    admin_proj = admin_gdf.to_crs(pois_gdf.crs)

    pois_in_admin = gpd.sjoin(pois_gdf, admin_proj, how='left', predicate='within')

    poi_count_by_admin = pois_in_admin.groupby('NAME_1').size().reset_index(name='total_pois')

    friction_by_admin = pois_in_admin.groupby('NAME_1')['friction_value'].agg(
        ['mean', 'median', 'std', 'min', 'max']
    ).reset_index()
    friction_by_admin.columns = ['NAME_1', 'mean_friction', 'median_friction',
                                  'std_friction', 'min_friction', 'max_friction']

    poi_type_by_admin = pois_in_admin.groupby(['NAME_1', 'amenity']).size().unstack(fill_value=0).reset_index()
    poi_type_by_admin.columns = ['NAME_1'] + [f'poi_type_{c}' for c in poi_type_by_admin.columns[1:]]

    merged = admin_gdf[['NAME_1', 'geometry']].merge(poi_count_by_admin, on='NAME_1', how='left')
    merged = merged.merge(friction_by_admin, on='NAME_1', how='left')
    merged = merged.merge(poi_type_by_admin, on='NAME_1', how='left')

    merged['poi_density'] = merged['total_pois'] / (merged.geometry.area / 1e6)
    merged['effective_accessibility'] = merged['poi_density'] / merged['mean_friction'].replace(0, np.nan)
    merged['accessibility_score'] = merged['poi_density'] * (1 / merged['mean_friction'].replace(0, np.nan))

    stats_data = []
    for _, row in merged.iterrows():
        stats_data.append({
            'admin_zone': row['NAME_1'],
            'total_pois': row['total_pois'],
            'poi_density_per_km2': row['poi_density'],
            'mean_friction': row['mean_friction'],
            'median_friction': row['median_friction'],
            'std_friction': row['std_friction'],
            'min_friction': row['min_friction'],
            'max_friction': row['max_friction'],
            'effective_accessibility': row['effective_accessibility'],
            'accessibility_score': row['accessibility_score']
        })

    stats_df = pd.DataFrame(stats_data).sort_values('effective_accessibility', ascending=False)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'walking_osm_accessibility.csv'), index=False)
    print(f"    Saved walking_osm_accessibility.csv")
    return stats_df, merged, pois_in_admin


def friction_poi_correlation(pois_df):
    print("  Analyzing friction vs POI type distribution...")
    valid = pois_df.dropna(subset=['friction_value'])

    corr_by_type = {}
    for amenity in valid['amenity'].unique():
        subset = valid[valid['amenity'] == amenity]
        if len(subset) > 10:
            corr, p = stats.spearmanr(subset['friction_value'], subset['poi_id'] * 0 + 1)
            corr_by_type[amenity] = {'count': len(subset), 'mean_friction': subset['friction_value'].mean()}

    return corr_by_type


def plot_results(stats_df, merged, pois_in_admin):
    print("  Generating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0, 0]
    stats_sorted = stats_df.sort_values('mean_friction', ascending=True)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(stats_sorted)))
    ax.barh(stats_sorted['admin_zone'], stats_sorted['mean_friction'],
            xerr=stats_sorted['std_friction'], color=colors, alpha=0.8, capsize=3)
    ax.set_xlabel('Mean Walking Friction at POI Locations')
    ax.set_title('Mean Walking Friction at POIs by Admin Zone')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[0, 1]
    ax.scatter(stats_df['mean_friction'], stats_df['poi_density_per_km2'],
               s=stats_df['total_pois'] / 5 + 20, alpha=0.7, c='teal', edgecolors='black')
    for _, row in stats_df.iterrows():
        ax.annotate(row['admin_zone'][:6], (row['mean_friction'], row['poi_density_per_km2']),
                    fontsize=7, alpha=0.7)
    ax.set_xlabel('Mean Walking Friction')
    ax.set_ylabel('POI Density (per km²)')
    ax.set_title('Walking Friction vs POI Density\n(bubble size = total POIs)')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.scatter(stats_df['mean_friction'], stats_df['effective_accessibility'],
               s=100, alpha=0.7, c='coral', edgecolors='black')
    for _, row in stats_df.iterrows():
        ax.annotate(row['admin_zone'][:6], (row['mean_friction'], row['effective_accessibility']),
                    fontsize=7, alpha=0.7)
    ax.set_xlabel('Mean Walking Friction')
    ax.set_ylabel('Effective Accessibility (POI density / friction)')
    ax.set_title('Effective Accessibility vs Walking Friction')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    ax = axes[1, 0]
    merged_sorted = stats_df.sort_values('effective_accessibility', ascending=True)
    colors_bar = ['coral' if x < 1 else 'seagreen' for x in merged_sorted['effective_accessibility']]
    ax.barh(merged_sorted['admin_zone'], merged_sorted['effective_accessibility'],
            color=colors_bar, alpha=0.8, edgecolor='black')
    ax.axvline(1, color='grey', linestyle='--', alpha=0.7, label='Balanced (1.0)')
    ax.set_xlabel('Effective Accessibility Index')
    ax.set_title('Effective POI Accessibility by Zone\n(POI density / mean friction)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[1, 1]
    top_types = pois_in_admin['amenity'].value_counts().head(8)
    pois_in_admin_top = pois_in_admin[pois_in_admin['amenity'].isin(top_types.index)]
    pivot = pois_in_admin_top.pivot_table(index='NAME_1', columns='amenity',
                                           values='friction_value', aggfunc='mean')
    if len(pivot) > 0:
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax,
                    cbar_kws={'label': 'Mean Friction'})
    ax.set_title('Mean Friction at POI Types by Admin Zone')
    ax.tick_params(axis='x', rotation=30)

    ax = axes[1, 2]
    merged_map = admin_gdf_temp = gpd.read_file(os.path.join(DATA_DIR, "gadm_mys_l1_clean.geojson"))
    merged_map = merged_map.merge(
        stats_df[['admin_zone', 'effective_accessibility']].rename(columns={'admin_zone': 'NAME_1'}),
        on='NAME_1', how='left'
    )
    merged_map.plot(column='effective_accessibility', cmap='RdYlGn', legend=True,
                   edgecolor='black', alpha=0.8, ax=ax)
    ax.set_title('Effective POI Accessibility by Admin Zone')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'walking_osm_accessibility.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved walking_osm_accessibility.png")


def main():
    print("=" * 60)
    print("BIVARIATE EDA: Walking Friction vs OSM POI Accessibility")
    print("=" * 60)

    pois_df = load_osm_pois()
    friction_path, crs_fric = load_friction_surface()
    admin_gdf = load_admin_boundaries()

    pois_df = sample_friction_at_pois(pois_df, friction_path, crs_fric)
    stats_df, merged, pois_in_admin = aggregate_by_admin(pois_df, admin_gdf)
    friction_poi_correlation(pois_df)

    plot_results(stats_df, merged, pois_in_admin)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()