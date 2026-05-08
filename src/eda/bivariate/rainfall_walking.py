#!/usr/bin/env python3
"""
Bivariate EDA: Rainfall vs Walking Friction
Objective: Combined pedestrian mobility barrier analysis

Analysis:
1. Correlation between rainfall anomalies and walking friction
2. Identify critical barrier zones where high rainfall + high friction coincide
3. Temporal patterns: high-rainfall days and friction exposure
4. Spatial overlay of rainfall risk zones and friction barriers
5. Combined mobility risk index (rainfall anomaly * friction)
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


def load_rainfall():
    print("  Loading rainfall data...")
    rf_df = pd.read_csv(os.path.join(DATA_DIR, "rainfall_combined_final.csv"), parse_dates=['date'])
    print(f"    Loaded {len(rf_df):,} rainfall records from {rf_df['date'].min()} to {rf_df['date'].max()}")
    print(f"    Adm levels: {rf_df['adm_level'].unique()}, Adm IDs: {rf_df['adm_id'].nunique()}")
    return rf_df


def load_friction_surface():
    print("  Loading walking friction surface...")
    friction_path = os.path.join(DATA_DIR, "walking_friction_clean.tif")
    with rasterio.open(friction_path) as src:
        friction_crs = src.crs
        friction_bounds = src.bounds
        res = src.res
    print(f"    Friction: bounds={friction_bounds}, res={res}")
    return friction_path, friction_crs


def load_admin_boundaries():
    print("  Loading admin boundaries...")
    gdf = gpd.read_file(os.path.join(DATA_DIR, "gadm_mys_l1_clean.geojson"))
    print(f"    Loaded {len(gdf)} admin zones")
    return gdf


def extract_friction_by_admin(admin_gdf, friction_path, friction_crs):
    print("  Extracting friction by admin zone...")
    admin_proj = admin_gdf.to_crs(friction_crs)

    friction_stats = {}
    with rasterio.open(friction_path) as src:
        for idx, (_, row) in enumerate(admin_proj.iterrows()):
            geom = row.geometry
            try:
                out_image, _ = mask.mask(src, [geom], crop=True)
                values = out_image[out_image != src.nodata].flatten()
                values = values[~np.isnan(values.astype(float))]
            except Exception:
                values = np.array([])

            if len(values) > 0:
                friction_stats[row['NAME_1']] = {
                    'mean_friction': float(np.mean(values)),
                    'median_friction': float(np.median(values)),
                    'high_friction_pct': float((values >= 5.0).mean() * 100),
                    'very_high_friction_pct': float((values >= 8.0).mean() * 100)
                }
            else:
                friction_stats[row['NAME_1']] = {
                    'mean_friction': np.nan,
                    'median_friction': np.nan,
                    'high_friction_pct': np.nan,
                    'very_high_friction_pct': np.nan
                }
    return friction_stats


STATE_TO_PCODE = {
    'Johor': 'MY01', 'Kedah': 'MY02', 'Kelantan': 'MY03', 'Melaka': 'MY04',
    'NegeriSembilan': 'MY05', 'Pahang': 'MY06', 'Perak': 'MY07', 'Perlis': 'MY08',
    'PulauPinang': 'MY09', 'Sabah': 'MY10', 'Sarawak': 'MY11', 'Selangor': 'MY12',
    'Trengganu': 'MY13', 'KualaLumpur': 'MY14', 'Labuan': 'MY15', 'Putrajaya': 'MY16'
}

def spatial_overlay(rainfall_df, friction_stats, admin_gdf):
    print("  Performing spatial overlay of rainfall and friction...")

    rf_l2 = rainfall_df[rainfall_df['adm_level'] == 2].copy()

    rf_l2['state_pcode'] = rf_l2['PCODE'].str[:4]

    rf_by_state = rf_l2.groupby('state_pcode').agg({
        'rainfall_mm': ['mean', 'max'],
        'anomaly_rf': 'mean',
        'anomaly_1mo': 'mean',
        'anomaly_3mo': 'mean'
    }).reset_index()
    rf_by_state.columns = ['state_pcode', 'mean_rainfall', 'max_rainfall',
                           'mean_anomaly_rf', 'mean_anomaly_1mo', 'mean_anomaly_3mo']

    zone_stats = []
    for _, row in admin_gdf.iterrows():
        zone = row['NAME_1']
        pcode = STATE_TO_PCODE.get(zone, None)
        fric = friction_stats.get(zone, {})

        zone_rf = rf_by_state[rf_by_state['state_pcode'] == pcode] if pcode else pd.DataFrame()

        zone_stats.append({
            'admin_zone': zone,
            'mean_friction': fric.get('mean_friction', np.nan),
            'median_friction': fric.get('median_friction', np.nan),
            'high_friction_pct': fric.get('high_friction_pct', np.nan),
            'very_high_friction_pct': fric.get('very_high_friction_pct', np.nan),
            'mean_rainfall': zone_rf['mean_rainfall'].values[0] if len(zone_rf) > 0 else np.nan,
            'max_rainfall': zone_rf['max_rainfall'].values[0] if len(zone_rf) > 0 else np.nan,
            'mean_anomaly_rf': zone_rf['mean_anomaly_rf'].values[0] if len(zone_rf) > 0 else np.nan,
            'mean_anomaly_1mo': zone_rf['mean_anomaly_1mo'].values[0] if len(zone_rf) > 0 else np.nan,
            'mean_anomaly_3mo': zone_rf['mean_anomaly_3mo'].values[0] if len(zone_rf) > 0 else np.nan
        })

    zone_df = pd.DataFrame(zone_stats)
    zone_df['combined_mobility_barrier'] = (
        (zone_df['high_friction_pct'].fillna(0) / 100) *
        (zone_df['mean_anomaly_rf'].fillna(0).clip(lower=0) / 100 + 1)
    )
    zone_df = zone_df.sort_values('combined_mobility_barrier', ascending=False)
    zone_df.to_csv(os.path.join(OUTPUT_DIR, 'rainfall_walking_mobility_barrier.csv'), index=False)
    print(f"    Saved rainfall_walking_mobility_barrier.csv")

    return zone_df


def temporal_correlation(rainfall_df):
    print("  Analyzing temporal correlation between rainfall and friction proxy...")

    rf_l1 = rainfall_df[rainfall_df['adm_level'] == 1].copy()
    rf_l1 = rf_l1.sort_values('date')

    rf_l1['rainfall_quintile'] = pd.qcut(rf_l1['rainfall_mm'].rank(method='first'),
                                          5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    rf_summary = rf_l1.groupby('date').agg({
        'rainfall_mm': 'mean',
        'anomaly_rf': 'mean',
        'anomaly_1mo': 'mean'
    }).reset_index()

    rf_summary['rainfall_change'] = rf_summary['rainfall_mm'].pct_change() * 100
    rf_summary['anomaly_change'] = rf_summary['anomaly_rf'].pct_change() * 100

    rf_summary.to_csv(os.path.join(OUTPUT_DIR, 'rainfall_walking_temporal.csv'), index=False)
    print(f"    Saved rainfall_walking_temporal.csv")

    return rf_summary


def plot_results(zone_df, rf_summary):
    print("  Generating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0, 0]
    valid = zone_df.dropna(subset=['mean_friction', 'mean_anomaly_rf'])
    if len(valid) > 0:
        ax.scatter(valid['mean_friction'], valid['mean_anomaly_rf'],
                   s=100, alpha=0.7, c='steelblue', edgecolors='black')
        for _, row in valid.iterrows():
            ax.annotate(row['admin_zone'][:6],
                        (row['mean_friction'], row['mean_anomaly_rf']),
                        fontsize=7, alpha=0.7)
    ax.set_xlabel('Mean Walking Friction')
    ax.set_ylabel('Mean Rainfall Anomaly (%)')
    ax.set_title('Walking Friction vs Rainfall Anomaly by Zone')
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    zone_sorted = zone_df.sort_values('combined_mobility_barrier', ascending=True)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(zone_sorted)))
    ax.barh(zone_sorted['admin_zone'], zone_sorted['combined_mobility_barrier'],
            color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Combined Mobility Barrier Index')
    ax.set_title('Combined Rainfall + Friction Mobility Barrier by Zone')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[0, 2]
    zone_sorted = zone_df.sort_values('high_friction_pct', ascending=True)
    ax.barh(zone_sorted['admin_zone'], zone_sorted['high_friction_pct'],
            color='coral', alpha=0.8, edgecolor='black')
    ax.set_xlabel(f'% High-Friction Area (>= 5.0)')
    ax.set_title('High-Friction Zone Coverage by Admin')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[1, 0]
    if len(rf_summary) > 0:
        rf_sample = rf_summary.set_index('date').sort_index()
        ax.plot(rf_sample.index, rf_sample['rainfall_mm'].rolling(30).mean(),
                color='steelblue', alpha=0.8, label='30d MA')
        ax.fill_between(rf_sample.index, 0, rf_sample['rainfall_mm'],
                        alpha=0.2, color='steelblue', label='Daily')
        ax.set_xlabel('Date')
        ax.set_ylabel('Mean Daily Rainfall (mm)')
        ax.set_title('Daily Rainfall Time Series (30-day Moving Average)')
        ax.tick_params(axis='x', rotation=30)
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if len(rf_summary) > 0:
        ax.scatter(rf_summary['anomaly_rf'], rf_summary['rainfall_mm'],
                   alpha=0.3, s=10, c='coral')
        ax.set_xlabel('Rainfall Anomaly (%)')
        ax.set_ylabel('Daily Rainfall (mm)')
        ax.set_title('Rainfall Anomaly vs Daily Rainfall')
        ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    valid = zone_df.dropna(subset=['mean_friction', 'very_high_friction_pct'])
    if len(valid) > 0:
        ax.scatter(valid['mean_friction'], valid['very_high_friction_pct'],
                   s=100, alpha=0.7, c='purple', edgecolors='black')
        for _, row in valid.iterrows():
            ax.annotate(row['admin_zone'][:6],
                        (row['mean_friction'], row['very_high_friction_pct']),
                        fontsize=7, alpha=0.7)
    ax.set_xlabel('Mean Walking Friction')
    ax.set_ylabel(f'% Very High-Friction Area (>= 8.0)')
    ax.set_title('Friction vs Very High-Friction Zone Coverage')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rainfall_walking_mobility.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved rainfall_walking_mobility.png")


def main():
    print("=" * 60)
    print("BIVARIATE EDA: Rainfall vs Walking Friction")
    print("=" * 60)

    rainfall_df = load_rainfall()
    friction_path, friction_crs = load_friction_surface()
    admin_gdf = load_admin_boundaries()

    friction_stats = extract_friction_by_admin(admin_gdf, friction_path, friction_crs)
    zone_df = spatial_overlay(rainfall_df, friction_stats, admin_gdf)
    rf_summary = temporal_correlation(rainfall_df)

    plot_results(zone_df, rf_summary)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()