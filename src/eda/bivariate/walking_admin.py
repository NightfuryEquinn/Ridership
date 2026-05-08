#!/usr/bin/env python3
"""
Bivariate EDA: Walking Friction vs Admin Zones
Objective: Accessibility disparity by administrative zone

Analysis:
1. Mean/median friction by admin zone
2. High-friction zone coverage (% of area above threshold) per admin
3. Accessibility disparity index across zones
4. Population-weighted friction exposure by admin
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio import mask
from shapely.geometry import Point

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = os.path.join(BASE_DIR, "../../../data/cleaned")
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
pd.options.display.max_columns = None

FRICTION_THRESHOLD = 5.0


def load_admin_boundaries():
    print("  Loading admin boundaries...")
    gdf = gpd.read_file(os.path.join(DATA_DIR, "gadm_mys_l1_clean.geojson"))
    print(f"    Loaded {len(gdf)} admin zones: {gdf['NAME_1'].tolist()}")
    return gdf


def load_friction_surface():
    print("  Loading walking friction surface...")
    friction_path = os.path.join(DATA_DIR, "walking_friction_clean.tif")
    with rasterio.open(friction_path) as src:
        friction_crs = src.crs
        friction_bounds = src.bounds
        res = src.res
    print(f"    Friction: bounds={friction_bounds}, res={res}, crs={friction_crs}")
    return friction_path, friction_crs


def extract_friction_by_admin(admin_gdf, friction_path, friction_crs):
    print("  Extracting friction statistics by admin zone...")

    admin_proj = admin_gdf.to_crs(friction_crs)

    stats_list = []
    with rasterio.open(friction_path) as src:
        for idx, (_, row) in enumerate(admin_proj.iterrows()):
            geom = row.geometry

            try:
                out_image, _ = mask.mask(src, [geom], crop=True)
                values = out_image[out_image != src.nodata].flatten()
                values = values[~np.isnan(values.astype(float))]
            except Exception as e:
                values = np.array([])

            if len(values) > 0:
                stats_list.append({
                    'admin_zone': row['NAME_1'],
                    'geometry': row.geometry,
                    'n_pixels': len(values),
                    'mean_friction': float(np.mean(values)),
                    'median_friction': float(np.median(values)),
                    'std_friction': float(np.std(values)),
                    'min_friction': float(np.min(values)),
                    'max_friction': float(np.max(values)),
                    'p25_friction': float(np.percentile(values, 25)),
                    'p75_friction': float(np.percentile(values, 75)),
                    'p90_friction': float(np.percentile(values, 90)),
                    'high_friction_pct': float((values >= FRICTION_THRESHOLD).mean() * 100),
                    'very_high_friction_pct': float((values >= 8.0).mean() * 100),
                    'low_friction_pct': float((values <= 2.0).mean() * 100),
                    'area_km2': row.geometry.area / 1e6
                })
            else:
                stats_list.append({
                    'admin_zone': row['NAME_1'],
                    'geometry': row.geometry,
                    'n_pixels': 0,
                    'mean_friction': np.nan,
                    'median_friction': np.nan,
                    'std_friction': np.nan,
                    'min_friction': np.nan,
                    'max_friction': np.nan,
                    'p25_friction': np.nan,
                    'p75_friction': np.nan,
                    'p90_friction': np.nan,
                    'high_friction_pct': np.nan,
                    'very_high_friction_pct': np.nan,
                    'low_friction_pct': np.nan,
                    'area_km2': row.geometry.area / 1e6
                })

    stats_df = gpd.GeoDataFrame(stats_list, crs=friction_crs)
    numeric_cols = ['mean_friction', 'median_friction', 'std_friction', 'min_friction',
                     'max_friction', 'p25_friction', 'p75_friction', 'p90_friction',
                     'high_friction_pct', 'very_high_friction_pct', 'low_friction_pct']
    for col in numeric_cols:
        if col in stats_df.columns:
            stats_df[col] = stats_df[col].astype(float)

    stats_df['disparity_score'] = stats_df['std_friction'] / stats_df['mean_friction'].replace(0, np.nan)
    stats_df['accessibility_deficit'] = stats_df['high_friction_pct'] * stats_df['mean_friction']

    stats_df_flat = stats_df.drop(columns=['geometry'])
    stats_df_flat.to_csv(os.path.join(OUTPUT_DIR, 'walking_admin_disparity.csv'), index=False)
    print(f"    Saved walking_admin_disparity.csv")
    print(f"    Zones with highest mean friction: {stats_df.nlargest(3, 'mean_friction')['admin_zone'].tolist()}")
    print(f"    Zones with lowest mean friction: {stats_df.nsmallest(3, 'mean_friction')['admin_zone'].tolist()}")

    return stats_df


def plot_results(stats_df):
    print("  Generating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0, 0]
    stats_sorted = stats_df.sort_values('mean_friction', ascending=True)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(stats_sorted)))
    ax.barh(stats_sorted['admin_zone'], stats_sorted['mean_friction'],
            xerr=stats_sorted['std_friction'], color=colors, alpha=0.8, capsize=3)
    ax.set_xlabel('Mean Walking Friction')
    ax.set_title('Mean Walking Friction by Admin Zone')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[0, 1]
    stats_sorted = stats_df.sort_values('high_friction_pct', ascending=True)
    ax.barh(stats_sorted['admin_zone'], stats_sorted['high_friction_pct'],
            color='coral', alpha=0.8, edgecolor='black')
    ax.set_xlabel(f'% Zone Area with Friction >= {FRICTION_THRESHOLD}')
    ax.set_title('High-Friction Zone Coverage by Admin')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[0, 2]
    ax.scatter(stats_df['mean_friction'], stats_df['high_friction_pct'],
               s=100, alpha=0.7, c='steelblue', edgecolors='black')
    for _, row in stats_df.iterrows():
        ax.annotate(row['admin_zone'][:6], (row['mean_friction'], row['high_friction_pct']),
                    fontsize=7, alpha=0.7)
    ax.set_xlabel('Mean Walking Friction')
    ax.set_ylabel(f'% High-Friction Area (>= {FRICTION_THRESHOLD})')
    ax.set_title('Mean Friction vs High-Friction Coverage')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    stats_sorted = stats_df.sort_values('disparity_score', ascending=False)
    ax.barh(stats_sorted['admin_zone'], stats_sorted['disparity_score'],
            color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Friction Disparity Score (std / mean)')
    ax.set_title('Friction Variability Within Zones')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[1, 1]
    box_data = []
    labels = []
    for zone in stats_df.sort_values('mean_friction')['admin_zone']:
        row = stats_df[stats_df['admin_zone'] == zone]
        box_data.append([row['p25_friction'].values[0], row['median_friction'].values[0],
                         row['p75_friction'].values[0]])
        labels.append(zone[:8])
    if box_data:
        positions = range(len(labels))
        for i, (d, l) in enumerate(zip(box_data, labels)):
            ax.bar(i, d[1] - d[0], bottom=d[0], color='steelblue', alpha=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Friction Value')
        ax.set_title('Friction Distribution by Zone (median + IQR)')

    ax = axes[1, 2]
    stats_df.plot(column='mean_friction', cmap='RdYlGn_r', legend=True,
                  edgecolor='black', alpha=0.8, ax=ax)
    ax.set_title('Mean Walking Friction by Admin Zone')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'walking_admin_disparity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved walking_admin_disparity.png")


def main():
    print("=" * 60)
    print("BIVARIATE EDA: Walking Friction vs Admin Zones")
    print("=" * 60)

    admin_gdf = load_admin_boundaries()
    friction_path, friction_crs = load_friction_surface()

    stats_df = extract_friction_by_admin(admin_gdf, friction_path, friction_crs)
    plot_results(stats_df)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()