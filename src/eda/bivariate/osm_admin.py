#!/usr/bin/env python3
"""
Bivariate EDA: OSM POI Diversity & Density vs Admin Zones
Objective: POI diversity and density by administrative zone

Analysis:
1. Total POI count and density by admin zone
2. POI type diversity (Shannon index) per zone
3. POI category breakdown (amenity, shop, tourism, leisure)
4. Dominant POI type per admin zone
5. POI density vs zone area scatter
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
from scipy.stats import entropy

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
                'leisure': tags.get('leisure', None),
                'name': tags.get('name', '')
            })
    pois_df = pd.DataFrame(pois)
    print(f"    Loaded {len(pois_df):,} POIs")
    return pois_df


def load_admin_boundaries():
    print("  Loading admin boundaries...")
    gdf = gpd.read_file(os.path.join(DATA_DIR, "gadm_mys_l1_clean.geojson"))
    print(f"    Loaded {len(gdf)} admin zones")
    return gdf


def compute_shannon_diversity(counts):
    if counts.sum() == 0:
        return 0.0
    proportions = counts / counts.sum()
    proportions = proportions[proportions > 0]
    return entropy(proportions, base=2)


def analyze_by_admin(pois_df, admin_gdf):
    print("  Analyzing POI diversity and density by admin zone...")

    pois_gdf = gpd.GeoDataFrame(
        pois_df,
        geometry=[Point(lon, lat) for lon, lat in zip(pois_df['lon'], pois_df['lat'])],
        crs='EPSG:4326'
    )
    admin_proj = admin_gdf.to_crs(pois_gdf.crs)

    pois_in_admin = gpd.sjoin(pois_gdf, admin_proj, how='left', predicate='within')

    total_by_admin = pois_in_admin.groupby('NAME_1').size().reset_index(name='total_pois')
    amenity_by_admin = pois_in_admin.groupby(['NAME_1', 'amenity']).size().unstack(fill_value=0)

    amenity_cols = [c for c in amenity_by_admin.columns if c != 'NAME_1']
    for col in amenity_cols:
        amenity_by_admin[f'pct_{col}'] = amenity_by_admin[col] / amenity_by_admin.sum(axis=1) * 100

    amenity_by_admin = amenity_by_admin.reset_index()

    stats_by_admin = []
    for _, row in admin_proj.iterrows():
        zone = row['NAME_1']
        geom = row.geometry
        zone_pois = pois_in_admin[pois_in_admin['NAME_1'] == zone]
        total = len(zone_pois)

        if total > 0:
            amenity_counts = zone_pois['amenity'].value_counts()
            shannon = compute_shannon_diversity(amenity_counts)

            shop_count = len(zone_pois[zone_pois['shop'].notna()])
            tourism_count = len(zone_pois[zone_pois['tourism'].notna()])
            leisure_count = len(zone_pois[zone_pois['leisure'].notna()])

            dominant_type = amenity_counts.index[0] if len(amenity_counts) > 0 else 'other'
            dominant_pct = amenity_counts.iloc[0] / total * 100 if len(amenity_counts) > 0 else 0

            top_types = amenity_counts.head(3).to_dict()
        else:
            shannon = 0
            shop_count = tourism_count = leisure_count = 0
            dominant_type = 'none'
            dominant_pct = 0
            top_types = {}

        stats_by_admin.append({
            'admin_zone': zone,
            'total_pois': total,
            'poi_density_per_km2': total / (geom.area / 1e6),
            'area_km2': geom.area / 1e6,
            'shannon_diversity': shannon,
            'shop_count': shop_count,
            'tourism_count': tourism_count,
            'leisure_count': leisure_count,
            'dominant_type': dominant_type,
            'dominant_pct': dominant_pct,
            'top3_types': str(top_types)
        })

    stats_df = pd.DataFrame(stats_by_admin)
    stats_df = stats_df.rename(columns={'admin_zone': 'NAME_1'})
    stats_df = stats_df.merge(amenity_by_admin, on='NAME_1', how='left')
    stats_df = stats_df.rename(columns={'NAME_1': 'admin_zone'})
    stats_df = stats_df.sort_values('total_pois', ascending=False)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'osm_admin_diversity.csv'), index=False)
    print(f"    Saved osm_admin_diversity.csv")
    print(f"    Highest POI density: {stats_df.nlargest(1, 'poi_density_per_km2')[['admin_zone', 'poi_density_per_km2']].values}")

    return stats_df, pois_in_admin


def plot_results(stats_df, pois_in_admin):
    print("  Generating plots...")

    n_zones = len(stats_df)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0, 0]
    stats_sorted = stats_df.sort_values('total_pois', ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(stats_sorted)))
    ax.barh(stats_sorted['admin_zone'], stats_sorted['total_pois'], color=colors, alpha=0.8)
    ax.set_xlabel('Total POI Count')
    ax.set_title('Total POI Count by Admin Zone')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[0, 1]
    stats_sorted = stats_df.sort_values('shannon_diversity', ascending=True)
    ax.barh(stats_sorted['admin_zone'], stats_sorted['shannon_diversity'],
            color='teal', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Shannon Diversity Index (bits)')
    ax.set_title('POI Type Diversity by Admin Zone')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[0, 2]
    ax.scatter(stats_df['area_km2'], stats_df['total_pois'],
               s=100, alpha=0.7, c='coral', edgecolors='black')
    for _, row in stats_df.iterrows():
        ax.annotate(row['admin_zone'][:6], (row['area_km2'], row['total_pois']),
                    fontsize=7, alpha=0.7)
    ax.set_xlabel('Zone Area (km²)')
    ax.set_ylabel('Total POIs')
    ax.set_title('Zone Area vs POI Count')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    top_amenities = pois_in_admin['amenity'].value_counts().head(10).index.tolist()
    pivot_data = pois_in_admin[pois_in_admin['amenity'].isin(top_amenities)]
    pivot = pivot_data.pivot_table(index='NAME_1', columns='amenity', values='poi_id', aggfunc='count')
    pivot = pivot.reindex(stats_df.sort_values('total_pois', ascending=False)['admin_zone'])
    sns.heatmap(pivot.fillna(0), annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'POI Count'})
    ax.set_title('POI Type Count by Admin Zone (Top 10 Types)')
    ax.tick_params(axis='x', rotation=30)

    ax = axes[1, 1]
    category_data = []
    for _, row in stats_df.iterrows():
        total = row['total_pois']
        if total > 0:
            shop_pct = row.get('pct_shop', 0) if not pd.isna(row.get('pct_shop', np.nan)) else 0
            tour_pct = row.get('pct_tourism', 0) if not pd.isna(row.get('pct_tourism', np.nan)) else 0
            leis_pct = row.get('pct_leisure', 0) if not pd.isna(row.get('pct_leisure', np.nan)) else 0
            amen_pct = 100 - shop_pct - tour_pct - leis_pct
            category_data.append({
                'zone': row['admin_zone'],
                'Amenity': amen_pct,
                'Shop': shop_pct,
                'Tourism': tour_pct,
                'Leisure': leis_pct
            })
        else:
            category_data.append({'zone': row['admin_zone'], 'Amenity': 0, 'Shop': 0, 'Tourism': 0, 'Leisure': 0})

    cat_df = pd.DataFrame(category_data).set_index('zone')
    cat_df = cat_df.reindex(stats_df.sort_values('total_pois', ascending=False)['admin_zone'])
    cat_df.plot(kind='bar', stacked=True, ax=ax, color=['steelblue', 'seagreen', 'coral', 'gold'], alpha=0.8)
    ax.set_xlabel('Admin Zone')
    ax.set_ylabel('POI Category Share (%)')
    ax.set_title('POI Category Breakdown by Admin Zone')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='upper right')

    ax = axes[1, 2]
    dominant_types = stats_df.groupby('dominant_type').size().sort_values(ascending=False)
    dominant_types.plot(kind='bar', ax=ax, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Dominant POI Type')
    ax.set_ylabel('Number of Admin Zones')
    ax.set_title('Most Common Dominant POI Type per Zone')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'osm_admin_diversity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved osm_admin_diversity.png")


def main():
    print("=" * 60)
    print("BIVARIATE EDA: OSM POI Diversity & Density by Admin Zone")
    print("=" * 60)

    pois_df = load_osm_pois()
    admin_gdf = load_admin_boundaries()

    stats_df, pois_in_admin = analyze_by_admin(pois_df, admin_gdf)
    plot_results(stats_df, pois_in_admin)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()