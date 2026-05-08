#!/usr/bin/env python3
"""
Bivariate EDA: Population vs OSM POI Accessibility
Objective: POI accessibility relative to population density

Analysis:
1. Spatial correlation between population density and POI counts
2. Accessibility gap analysis (underserved high-pop areas, over-served low-pop)
3. POI type breakdown by population density quintiles
4. Iso- accessibility contour mapping
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
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


def load_population():
    print("  Loading population data...")
    pop_df = pd.read_csv(os.path.join(DATA_DIR, "population_density_clean.csv"))
    print(f"    Loaded {len(pop_df):,} population grid points")
    return pop_df


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
                'leisure': tags.get('leisure', None),
                'name': tags.get('name', '')
            })
    pois_df = pd.DataFrame(pois)
    print(f"    Loaded {len(pois_df):,} POIs")
    return pois_df


def load_admin_boundaries():
    print("  Loading admin boundaries...")
    gdf = gpd.read_file(os.path.join(DATA_DIR, "gadm_mys_l1_clean.geojson"))
    print(f"    Loaded {len(gdf)} admin zones: {gdf['NAME_1'].tolist()}")
    return gdf


def spatial_join_pop_poi_admin(pop_df, pois_df, admin_gdf):
    print("  Performing spatial joins...")

    pop_gdf = gpd.GeoDataFrame(
        pop_df,
        geometry=[Point(lon, lat) for lon, lat in zip(pop_df['longitude'], pop_df['latitude'])],
        crs='EPSG:4326'
    )
    pop_utm = pop_gdf.to_crs(admin_gdf.crs)

    pois_gdf = gpd.GeoDataFrame(
        pois_df,
        geometry=[Point(lon, lat) for lon, lat in zip(pois_df['lon'], pois_df['lat'])],
        crs='EPSG:4326'
    )
    pois_utm = pois_gdf.to_crs(pop_utm.crs)

    admin_utm = admin_gdf.to_crs(pop_utm.crs)

    pop_in_admin = gpd.sjoin(pop_utm, admin_utm, how='left', predicate='within')
    pois_in_admin = gpd.sjoin(pois_utm, admin_utm, how='left', predicate='within')

    return pop_in_admin, pois_in_admin, admin_utm, pop_utm, pois_utm


def compute_poi_accessibility(pop_df, pois_df, admin_gdf):
    print("  Computing POI accessibility metrics...")

    pop_gdf = gpd.GeoDataFrame(
        pop_df,
        geometry=[Point(lon, lat) for lon, lat in zip(pop_df['longitude'], pop_df['latitude'])],
        crs='EPSG:4326'
    )
    crs = admin_gdf.crs
    pop_utm = pop_gdf.to_crs(crs)
    pois_gdf = gpd.GeoDataFrame(
        pois_df,
        geometry=[Point(lon, lat) for lon, lat in zip(pois_df['lon'], pois_df['lat'])],
        crs='EPSG:4326'
    )
    pois_utm = pois_gdf.to_crs(crs)

    pop_coords = np.array(list(zip(pop_utm.geometry.x, pop_utm.geometry.y)))
    poi_coords = np.array(list(zip(pois_utm.geometry.x, pois_utm.geometry.y)))

    buffer_distances = [500, 1000, 2000]
    accessibility_results = []

    for dist in buffer_distances:
        print(f"    Building KD-tree for {dist}m radius POI count...")
        tree = cKDTree(poi_coords)
        counts = np.zeros(len(pop_coords))
        for i, pc in enumerate(pop_coords):
            idx = tree.query_ball_point(pc, r=dist)
            counts[i] = len(idx)

        accessibility_results.append({
            'dist_m': dist,
            'mean_poi_count': counts.mean(),
            'median_poi_count': np.median(counts),
            'std_poi_count': counts.std(),
            'zero_access_pct': (counts == 0).mean() * 100,
            'max_poi_count': counts.max()
        })

        if dist == 1000:
            pop_utm[f'poi_count_{dist}m'] = counts

    access_df = pd.DataFrame(accessibility_results)
    return access_df, pop_utm


def analyze_by_admin(pop_in_admin, pois_in_admin, admin_utm):
    print("  Analyzing POI accessibility by admin zone...")

    admin_poi_counts = pois_in_admin.groupby('NAME_1').size().reset_index(name='total_pois')

    poi_type_by_admin = pois_in_admin.groupby(['NAME_1', 'amenity']).size().unstack(fill_value=0)

    pop_density_by_admin = pop_in_admin.groupby('NAME_1').agg({
        'density_per_km2': ['mean', 'median', 'sum'],
        'density_log': 'mean',
        'latitude': 'count'
    }).reset_index()
    pop_density_by_admin.columns = ['NAME_1', 'mean_density', 'median_density',
                                     'sum_density', 'mean_log_density', 'pop_point_count']

    merged = admin_utm.merge(admin_poi_counts, on='NAME_1', how='left')
    merged = merged.merge(pop_density_by_admin, on='NAME_1', how='left')

    merged['poi_density'] = merged['total_pois'] / (merged.geometry.area / 1e6)

    merged['accessibility_index'] = merged['poi_density'] / merged['mean_density'].replace(0, np.nan)

    stats_data = []
    for _, row in merged.iterrows():
        stats_data.append({
            'admin_zone': row['NAME_1'],
            'total_pois': row['total_pois'],
            'poi_density_per_km2': row['poi_density'],
            'mean_pop_density': row['mean_density'],
            'median_pop_density': row['median_density'],
            'pop_point_count': row['pop_point_count'],
            'accessibility_index': row['accessibility_index']
        })

    stats_df = pd.DataFrame(stats_data).sort_values('accessibility_index', ascending=False)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'population_osm_accessibility.csv'), index=False)
    print(f"    Saved population_osm_accessibility.csv")

    return stats_df, poi_type_by_admin, merged


def analyze_population_quintiles(pop_utm, pois_utm):
    print("  Analyzing POI accessibility by population density quintiles...")

    pop_utm['density_quintile'] = pd.qcut(pop_utm['density_per_km2'].rank(method='first'),
                                           5, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)'])

    if 'poi_count_1000m' in pop_utm.columns:
        quintile_stats = pop_utm.groupby('density_quintile').agg({
            'density_per_km2': ['mean', 'count'],
            'poi_count_1000m': ['mean', 'median', 'std']
        }).reset_index()
        quintile_stats.columns = ['quintile', 'mean_density', 'pop_count',
                                   'mean_poi_count', 'median_poi_count', 'std_poi_count']
        quintile_stats.to_csv(os.path.join(OUTPUT_DIR, 'population_osm_quintile.csv'), index=False)
        print(f"    Saved population_osm_quintile.csv")
        return quintile_stats
    return None


def plot_results(stats_df, quintile_stats, pop_in_admin, pois_in_admin, admin_utm, pop_utm):
    print("  Generating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0, 0]
    stats_sorted = stats_df.sort_values('total_pois', ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(stats_sorted)))
    ax.barh(stats_sorted['admin_zone'], stats_sorted['total_pois'], color=colors)
    ax.set_xlabel('Total POI Count')
    ax.set_title('Total POI Count by Admin Zone')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[0, 1]
    stats_sorted = stats_df.sort_values('accessibility_index', ascending=True)
    colors = ['coral' if x < 1 else 'seagreen' for x in stats_sorted['accessibility_index']]
    ax.barh(stats_sorted['admin_zone'], stats_sorted['accessibility_index'], color=colors)
    ax.axvline(1, color='grey', linestyle='--', alpha=0.7, label='Balanced (1.0)')
    ax.set_xlabel('Accessibility Index (POI density / Pop density)')
    ax.set_title('POI Accessibility vs Population Density by Zone')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[0, 2]
    ax.scatter(stats_df['mean_pop_density'], stats_df['poi_density_per_km2'],
               s=stats_df['pop_point_count'] / 50, alpha=0.7, c='steelblue', edgecolors='black')
    for _, row in stats_df.iterrows():
        ax.annotate(row['admin_zone'][:6], (row['mean_pop_density'], row['poi_density_per_km2']),
                    fontsize=7, alpha=0.7)
    ax.set_xlabel('Mean Population Density (per km²)')
    ax.set_ylabel('POI Density (per km²)')
    ax.set_title('Population Density vs POI Density\n(bubble size = pop points)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if quintile_stats is not None:
        ax.bar(quintile_stats['quintile'], quintile_stats['mean_poi_count'],
               yerr=quintile_stats['std_poi_count'], capsize=4,
               color='teal', alpha=0.8)
        ax.set_xlabel('Population Density Quintile')
        ax.set_ylabel('Mean POI Count (1km radius)')
        ax.set_title('POI Accessibility by Population Density Quintile')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    ax.scatter(stats_df['mean_pop_density'], stats_df['total_pois'],
               alpha=0.7, c='coral', s=80, edgecolors='black')
    ax.set_xlabel('Mean Population Density (per km²)')
    ax.set_ylabel('Total POIs in Zone')
    ax.set_title('Population Density vs Total POIs')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    if len(stats_df) > 2:
        slope, intercept, r, p, _ = stats.linregress(
            np.log10(stats_df['mean_pop_density'].clip(lower=0.1)),
            np.log10(stats_df['total_pois'].clip(lower=1))
        )
        ax.text(0.05, 0.95, f'log-log slope: {slope:.2f}\nr={r:.3f}, p={p:.4f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax = axes[1, 2]
    merged_map = admin_utm.copy()
    merged_map = merged_map.merge(stats_df[['admin_zone', 'accessibility_index']].rename(
        columns={'admin_zone': 'NAME_1'}), on='NAME_1', how='left')
    merged_map.plot(column='accessibility_index', cmap='RdYlGn', legend=True,
                   edgecolor='black', alpha=0.8, ax=ax)
    ax.set_title('POI Accessibility Index by Admin Zone')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'population_osm_accessibility.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved population_osm_accessibility.png")


def main():
    print("=" * 60)
    print("BIVARIATE EDA: Population vs OSM POI Accessibility")
    print("=" * 60)

    pop_df = load_population()
    pois_df = load_osm_pois()
    admin_gdf = load_admin_boundaries()

    access_df, pop_utm = compute_poi_accessibility(pop_df, pois_df, admin_gdf)
    pop_in_admin, pois_in_admin, admin_utm, _, _ = spatial_join_pop_poi_admin(pop_df, pois_df, admin_gdf)

    stats_df, poi_type_by_admin, merged = analyze_by_admin(pop_in_admin, pois_in_admin, admin_utm)
    quintile_stats = analyze_population_quintiles(pop_utm, None)

    plot_results(stats_df, quintile_stats, pop_in_admin, pois_in_admin, admin_utm, pop_utm)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()