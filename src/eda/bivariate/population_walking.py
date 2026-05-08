#!/usr/bin/env python3
"""
Bivariate EDA: Population vs Walking Friction
Objective: Population exposure to high-friction pedestrian zones

Analysis:
1. Distribution of walking friction values at population locations
2. High-friction zone identification and population exposure calculation
3. Correlation between population density and friction exposure
4. Admin zone disparity in pedestrian accessibility
5. Effective mobility score (inverse friction weighted by population)
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.sample import sample_gen
from shapely.geometry import Point
from scipy import stats

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = os.path.join(BASE_DIR, "../../../data/cleaned")
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
pd.options.display.max_columns = None

FRICTION_THRESHOLD = 5.0


def load_population():
    print("  Loading population data...")
    pop_df = pd.read_csv(os.path.join(DATA_DIR, "population_density_clean.csv"))
    print(f"    Loaded {len(pop_df):,} population grid points")
    return pop_df


def load_friction_surface():
    print("  Loading walking friction surface...")
    friction_path = os.path.join(DATA_DIR, "walking_friction_clean.tif")
    with rasterio.open(friction_path) as src:
        bounds = src.bounds
        crs = src.crs
        res = src.res
    print(f"    Friction surface bounds: {bounds}, resolution: {res}")
    return friction_path, bounds, crs


def load_admin_boundaries():
    print("  Loading admin boundaries...")
    gdf = gpd.read_file(os.path.join(DATA_DIR, "gadm_mys_l1_clean.geojson"))
    print(f"    Loaded {len(gdf)} admin zones")
    return gdf


def sample_friction_at_points(pop_df, friction_path):
    print("  Sampling friction values at population points...")
    coords = [(lon, lat) for lon, lat in zip(pop_df['longitude'], pop_df['latitude'])]

    friction_values = []
    with rasterio.open(friction_path) as src:
        try:
            projected = gpd.GeoDataFrame(
                pop_df.copy(),
                geometry=[Point(lon, lat) for lon, lat in zip(pop_df['longitude'], pop_df['latitude'])],
                crs='EPSG:4326'
            ).to_crs(src.crs)

            sample_points = [(pt.x, pt.y) for pt in projected.geometry]

            with rasterio.open(friction_path) as src_opened:
                samples = list(src_opened.sample(sample_points))
                friction_values = [s[0] if len(s) > 0 else np.nan for s in samples]
        except Exception as e:
            print(f"    Projection failed ({e}), using lon/lat directly...")
            with rasterio.open(friction_path) as src_opened:
                samples = list(src_opened.sample(coords))
                friction_values = [s[0] if len(s) > 0 else np.nan for s in samples]

    pop_df = pop_df.copy()
    pop_df['friction_value'] = friction_values

    valid = pop_df.dropna(subset=['friction_value'])
    print(f"    Valid friction samples: {len(valid):,} / {len(pop_df):,}")
    print(f"    Friction range: {valid['friction_value'].min():.3f} - {valid['friction_value'].max():.3f}")
    print(f"    Friction mean: {valid['friction_value'].mean():.3f}, median: {valid['friction_value'].median():.3f}")

    return pop_df


def identify_high_friction_zones(pop_df, threshold):
    print(f"  Identifying high-friction zones (threshold >= {threshold})...")
    pop_df = pop_df.copy()
    pop_df['is_high_friction'] = (pop_df['friction_value'] >= threshold).astype(int)
    pop_df['friction_category'] = pd.cut(
        pop_df['friction_value'].fillna(-1),
        bins=[-np.inf, 2, 4, threshold, np.inf],
        labels=['Low (<2)', 'Medium (2-4)', 'High (4-{})'.format(threshold), 'Very High (>={})'.format(threshold)]
    )
    high_friction_pop = pop_df[pop_df['is_high_friction'] == 1]
    print(f"    Population points in high-friction zones: {len(high_friction_pop):,} "
          f"({len(high_friction_pop)/len(pop_df)*100:.1f}%)")
    return pop_df


def analyze_exposure_by_admin(pop_df, admin_gdf):
    print("  Analyzing friction exposure by admin zone...")

    pop_gdf = gpd.GeoDataFrame(
        pop_df,
        geometry=[Point(lon, lat) for lon, lat in zip(pop_df['longitude'], pop_df['latitude'])],
        crs='EPSG:4326'
    )
    pop_utm = pop_gdf.to_crs(admin_gdf.crs)
    admin_utm = admin_gdf.to_crs(pop_utm.crs)

    pop_in_admin = gpd.sjoin(pop_utm, admin_utm, how='left', predicate='within')

    friction_by_admin = pop_in_admin.groupby('NAME_1').agg({
        'friction_value': ['mean', 'median', 'std', 'min', 'max'],
        'is_high_friction': ['sum', 'mean'],
        'density_per_km2': ['mean', 'sum']
    }).reset_index()
    friction_by_admin.columns = ['admin_zone', 'mean_friction', 'median_friction',
                                  'std_friction', 'min_friction', 'max_friction',
                                  'high_friction_count', 'high_friction_pct',
                                  'mean_pop_density', 'sum_pop_density']

    pop_in_admin['pop_x_friction'] = pop_in_admin['density_per_km2'] * pop_in_admin['friction_value']
    pop_weighted = pop_in_admin.groupby('NAME_1').agg({
        'pop_x_friction': 'sum',
        'density_per_km2': 'sum'
    }).reset_index()
    pop_weighted['population_weighted_friction'] = pop_weighted['pop_x_friction'] / pop_weighted['density_per_km2'].replace(0, np.nan)
    pop_weighted = pop_weighted[['NAME_1', 'population_weighted_friction']].rename(
        columns={'NAME_1': 'admin_zone'})

    friction_by_admin = friction_by_admin.merge(pop_weighted, on='admin_zone', how='left')
    friction_by_admin = friction_by_admin.sort_values('mean_friction', ascending=False)
    friction_by_admin.to_csv(os.path.join(OUTPUT_DIR, 'population_walking_exposure.csv'), index=False)
    print(f"    Saved population_walking_exposure.csv")

    return friction_by_admin, pop_in_admin, admin_utm, pop_utm


def friction_density_correlation(pop_df):
    print("  Analyzing friction vs population density correlation...")

    valid = pop_df.dropna(subset=['friction_value', 'density_per_km2'])
    valid = valid[valid['friction_value'] >= 0]

    if len(valid) > 100:
        corr_pearson, p_pearson = stats.pearsonr(valid['density_per_km2'], valid['friction_value'])
        corr_spearman, p_spearman = stats.spearmanr(valid['density_per_km2'], valid['friction_value'])
        print(f"    Pearson correlation: r={corr_pearson:.4f}, p={p_pearson:.4f}")
        print(f"    Spearman correlation: r={corr_spearman:.4f}, p={p_spearman:.4f}")
        return corr_pearson, p_pearson, corr_spearman, p_spearman
    return np.nan, np.nan, np.nan, np.nan


def plot_results(pop_df, friction_by_admin, admin_utm, pop_utm):
    print("  Generating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0, 0]
    valid = pop_df.dropna(subset=['friction_value'])
    ax.hist(valid['friction_value'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(FRICTION_THRESHOLD, color='coral', linestyle='--', linewidth=2,
               label=f'Threshold ({FRICTION_THRESHOLD})')
    ax.set_xlabel('Walking Friction Value')
    ax.set_ylabel('Population Grid Point Count')
    ax.set_title('Distribution of Walking Friction at Population Locations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    friction_by_admin_sorted = friction_by_admin.sort_values('mean_friction', ascending=True)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(friction_by_admin_sorted)))
    ax.barh(friction_by_admin_sorted['admin_zone'], friction_by_admin_sorted['mean_friction'],
            xerr=friction_by_admin_sorted['std_friction'], color=colors, alpha=0.8, capsize=3)
    ax.set_xlabel('Mean Walking Friction Value')
    ax.set_title('Mean Walking Friction by Admin Zone')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[0, 2]
    ax.scatter(friction_by_admin['mean_pop_density'], friction_by_admin['mean_friction'],
               s=friction_by_admin['high_friction_count'] / 5 + 20,
               alpha=0.7, c='coral', edgecolors='black')
    for _, row in friction_by_admin.iterrows():
        ax.annotate(row['admin_zone'][:6], (row['mean_pop_density'], row['mean_friction']),
                    fontsize=7, alpha=0.7)
    ax.set_xlabel('Mean Population Density (per km²)')
    ax.set_ylabel('Mean Walking Friction')
    ax.set_title('Population Density vs Walking Friction by Zone\n(bubble size = high-friction points)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    category_counts = pop_df['friction_category'].value_counts().reindex(
        ['Low (<2)', 'Medium (2-4)', 'High (4-{})'.format(FRICTION_THRESHOLD), 'Very High (>={})'.format(FRICTION_THRESHOLD)])
    colors_cat = ['seagreen', 'gold', 'orange', 'coral']
    ax.bar(category_counts.index, category_counts.values, color=colors_cat, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Friction Category')
    ax.set_ylabel('Population Grid Points')
    ax.set_title('Population Distribution by Friction Category')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (cat, val) in enumerate(category_counts.items()):
        ax.text(i, val + 500, f'{val/len(pop_df)*100:.1f}%', ha='center', fontsize=9)

    ax = axes[1, 1]
    friction_by_admin_sorted = friction_by_admin.sort_values('high_friction_pct', ascending=True)
    ax.barh(friction_by_admin_sorted['admin_zone'],
            friction_by_admin_sorted['high_friction_pct'] * 100,
            color='coral', alpha=0.8, edgecolor='black')
    ax.set_xlabel('% Population in High-Friction Zones')
    ax.set_title('Population Exposure to High-Friction Zones by Zone')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[1, 2]
    merged_map = admin_utm.copy()
    merged_map = merged_map.merge(
        friction_by_admin[['admin_zone', 'mean_friction']].rename(columns={'admin_zone': 'NAME_1'}),
        on='NAME_1', how='left'
    )
    merged_map.plot(column='mean_friction', cmap='RdYlGn_r', legend=True,
                   edgecolor='black', alpha=0.8, ax=ax)
    ax.set_title('Mean Walking Friction by Admin Zone')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'population_walking_exposure.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved population_walking_exposure.png")


def main():
    print("=" * 60)
    print("BIVARIATE EDA: Population vs Walking Friction")
    print("=" * 60)

    pop_df = load_population()
    friction_path, bounds, crs = load_friction_surface()
    admin_gdf = load_admin_boundaries()

    pop_df = sample_friction_at_points(pop_df, friction_path)
    pop_df = identify_high_friction_zones(pop_df, FRICTION_THRESHOLD)
    friction_by_admin, pop_in_admin, admin_utm, pop_utm = analyze_exposure_by_admin(pop_df, admin_gdf)
    friction_density_correlation(pop_df)

    plot_results(pop_df, friction_by_admin, admin_utm, pop_utm)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()