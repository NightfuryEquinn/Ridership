#!/usr/bin/env python3
"""
Bivariate EDA: Population Density vs Ridership
Objective: Per-capita ridership by density zone

Analysis:
1. Population density distribution across Malaysia (grid-level statistics)
2. GTFS stop density mapping: which transit stops serve high/low density zones
3. Service coverage in high-density urban vs low-density rural areas
4. Population-weighted ridership: estimate ridership per capita per zone
5. Urban vs rural transit utilization comparison
6. Density zone classification and ridership patterns by zone type
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial import ConvexHull

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
pd.options.display.max_columns = None


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_population_data():
    """Load population density grid data."""
    df = pd.read_csv(
        os.path.join(BASE_DIR, "../../../data/cleaned/population_density_clean.csv")
    )
    df = df[['longitude', 'latitude', 'density_per_km2', 'density_log', 'density_clipped']].copy()
    return df


def load_ridership():
    """Load ridership data."""
    df = pd.read_csv(
        os.path.join(BASE_DIR, "../../../data/cleaned/ridership_headline_clean.csv"),
        parse_dates=['date']
    )
    df = df.set_index('date').sort_index()
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    df['month'] = df.index.month
    df['year'] = df.index.year
    return df


def load_gtfs_all_stops():
    """Load all GTFS stops from all services."""
    gtfs_folders = [
        "gtfs_rapid_bus_kl",
        "gtfs_rapid_bus_penang",
        "gtfs_rapid_rail_kl",
        "gtfs_ktmb"
    ]

    all_stops = []
    service_labels = []

    for folder in gtfs_folders:
        path = os.path.join(BASE_DIR, f"../../../data/cleaned/{folder}")
        stops_file = os.path.join(path, "stops.txt")
        if os.path.exists(stops_file):
            try:
                stops = pd.read_csv(stops_file, dtype=str)
                if 'stop_lat' in stops.columns and 'stop_lon' in stops.columns:
                    for _, row in stops.iterrows():
                        try:
                            lat = float(row['stop_lat'])
                            lon = float(row['stop_lon'])
                            if -6 < lat < 8 and 99 < lon < 120:
                                all_stops.append([lat, lon])
                                service_labels.append(folder)
                        except (ValueError, TypeError):
                            continue
            except Exception:
                continue

    return np.array(all_stops), service_labels


def classify_density_zones(pop_df):
    """Classify grid cells into density zones using quantiles."""
    density = pop_df['density_per_km2'].dropna()
    if len(density) == 0:
        return pop_df

    q25 = density.quantile(0.25)
    q50 = density.quantile(0.50)
    q75 = density.quantile(0.75)
    q90 = density.quantile(0.90)

    def classify(val):
        if val <= q25:
            return 'Rural (Q0-Q25)'
        elif val <= q50:
            return 'Suburban (Q25-Q50)'
        elif val <= q75:
            return 'Urban (Q50-Q75)'
        elif val <= q90:
            return 'Dense Urban (Q75-Q90)'
        else:
            return 'Core Urban (Q90-Q100)'

    pop_df['density_zone'] = pop_df['density_per_km2'].apply(classify)
    return pop_df


def map_stops_to_zones(all_stops, pop_df, buffer_km=5):
    """Map GTFS stops to population density zones using BallTree for speed."""
    from sklearn.neighbors import BallTree

    pop_with_zone = pop_df.copy()

    if len(all_stops) == 0:
        pop_with_zone['num_stops_nearby'] = 0
        pop_with_zone['has_transit'] = 0
        return pop_with_zone

    stops_rad = np.radians(all_stops)
    pop_coords = np.radians(pop_with_zone[['latitude', 'longitude']].values)

    tree = BallTree(stops_rad, metric='haversine')
    counts_per_cell = tree.query_radius(pop_coords, r=buffer_km / 6371, count_only=True)
    pop_with_zone['num_stops_nearby'] = counts_per_cell
    pop_with_zone['has_transit'] = (pop_with_zone['num_stops_nearby'] > 0).astype(int)

    return pop_with_zone


def population_density_overview(pop_df):
    """Overview of population density distribution."""
    print("  Running population density overview...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    density = pop_df['density_per_km2'].dropna()
    axes[0, 0].hist(density, bins=50, color='steelblue', edgecolor='darkblue', alpha=0.7)
    axes[0, 0].set_xlabel('Population Density (per km²)')
    axes[0, 0].set_ylabel('Frequency (Grid Cells)')
    axes[0, 0].set_title('Population Density Distribution\n(All Grid Cells)')
    axes[0, 0].axvline(density.median(), color='coral', linestyle='--', label=f'Median: {density.median():.1f}')
    axes[0, 0].axvline(density.mean(), color='seagreen', linestyle='--', label=f'Mean: {density.mean():.1f}')
    axes[0, 0].legend()

    log_density = pop_df['density_log'].dropna()
    axes[0, 1].hist(log_density, bins=50, color='coral', edgecolor='darkred', alpha=0.7)
    axes[0, 1].set_xlabel('Log10(Population Density)')
    axes[0, 1].set_ylabel('Frequency (Grid Cells)')
    axes[0, 1].set_title('Log-Transformed Density Distribution')

    zone_counts = pop_df['density_zone'].value_counts()
    zone_order = ['Rural (Q0-Q25)', 'Suburban (Q25-Q50)', 'Urban (Q50-Q75)',
                  'Dense Urban (Q75-Q90)', 'Core Urban (Q90-Q100)']
    zone_counts = zone_counts.reindex([z for z in zone_order if z in zone_counts.index])
    colors = ['seagreen', 'steelblue', 'coral', 'orange', 'darkred']
    axes[1, 0].bar(zone_counts.index, zone_counts.values, color=colors, alpha=0.8)
    axes[1, 0].set_xlabel('Density Zone')
    axes[1, 0].set_ylabel('Number of Grid Cells')
    axes[1, 0].set_title('Grid Cells by Density Zone Classification')
    axes[1, 0].tick_params(axis='x', rotation=20)

    if 'latitude' in pop_df.columns and 'longitude' in pop_df.columns:
        high_density = pop_df[pop_df['density_per_km2'] > pop_df['density_per_km2'].quantile(0.75)]
        low_density = pop_df[pop_df['density_per_km2'] <= pop_df['density_per_km2'].quantile(0.25)]

        axes[1, 1].scatter(pop_df['longitude'], pop_df['latitude'],
                            c=pop_df['density_per_km2'], cmap='YlOrRd',
                            s=1, alpha=0.5)
        axes[1, 1].set_xlabel('Longitude')
        axes[1, 1].set_ylabel('Latitude')
        axes[1, 1].set_title('Geographic Distribution of Population Density\n(Color = Density)')
        plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Density per km²')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'population_density_overview.png'), dpi=150)
    plt.close()
    print("    Saved population_density_overview.png")

    print(f"    Total grid cells: {len(pop_df)}")
    print(f"    Density range: {density.min():.2f} - {density.max():.2f} per km²")
    print(f"    Median density: {density.median():.2f} per km²")
    print(f"    Mean density: {density.mean():.2f} per km²")


def transit_coverage_by_zone(pop_with_zone):
    """Analyze transit coverage across density zones."""
    print("  Running transit coverage by density zone...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    zone_stats = pop_with_zone.groupby('density_zone').agg({
        'density_per_km2': ['mean', 'median', 'count'],
        'num_stops_nearby': ['mean', 'sum'],
        'has_transit': 'sum'
    })
    zone_order = ['Rural (Q0-Q25)', 'Suburban (Q25-Q50)', 'Urban (Q50-Q75)',
                  'Dense Urban (Q75-Q90)', 'Core Urban (Q90-Q100)']
    zone_stats = zone_stats.reindex([z for z in zone_order if z in zone_stats.index])

    axes[0, 0].bar(zone_stats.index, zone_stats[('num_stops_nearby', 'mean')],
                    color='steelblue', alpha=0.8)
    axes[0, 0].set_xlabel('Density Zone')
    axes[0, 0].set_ylabel('Average Stops within 5km')
    axes[0, 0].set_title('Transit Stop Density by Population Zone')
    axes[0, 0].tick_params(axis='x', rotation=20)

    transit_pct = zone_stats[('has_transit', 'sum')] / zone_stats[('density_per_km2', 'count')] * 100
    transit_pct = transit_pct.reindex([z for z in zone_order if z in transit_pct.index])
    axes[0, 1].bar(transit_pct.index, transit_pct.values, color='coral', alpha=0.8)
    axes[0, 1].set_xlabel('Density Zone')
    axes[0, 1].set_ylabel('% Grid Cells with Transit Coverage')
    axes[0, 1].set_title('Transit Coverage Rate by Density Zone')
    axes[0, 1].tick_params(axis='x', rotation=20)
    for i, v in enumerate(transit_pct.values):
        axes[0, 1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=8)

    pop_by_zone = pop_with_zone.groupby('density_zone')['density_per_km2'].sum()
    pop_by_zone = pop_by_zone.reindex([z for z in zone_order if z in pop_by_zone.index])
    stops_by_zone = zone_stats[('num_stops_nearby', 'sum')]
    stops_by_zone = stops_by_zone.reindex([z for z in zone_order if z in stops_by_zone.index])

    stops_per_100k = []
    for zone in pop_by_zone.index:
        pop = pop_by_zone.get(zone, 0)
        stops = stops_by_zone.get(zone, 0)
        stops_per_100k.append(stops / max(pop / 100000, 1))

    axes[1, 0].bar(pop_by_zone.index, stops_per_100k, color='seagreen', alpha=0.8)
    axes[1, 0].set_xlabel('Density Zone')
    axes[1, 0].set_ylabel('Stops per 100k Population (proxy)')
    axes[1, 0].set_title('Transit Stop Accessibility by Zone\n(Stops per 100k Population Proxy)')
    axes[1, 0].tick_params(axis='x', rotation=20)

    zone_pop_totals = pop_with_zone.groupby('density_zone')['density_per_km2'].agg(['sum', 'mean'])
    zone_pop_totals = zone_pop_totals.reindex([z for z in zone_order if z in zone_pop_totals.index])
    axes[1, 1].bar(zone_pop_totals.index, zone_pop_totals['sum'], color='mediumpurple', alpha=0.8)
    axes[1, 1].set_xlabel('Density Zone')
    axes[1, 1].set_ylabel('Total Population Density Sum (proxy)')
    axes[1, 1].set_title('Total Population by Density Zone')
    axes[1, 1].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'population_transit_coverage.png'), dpi=150)
    plt.close()
    print("    Saved population_transit_coverage.png")


def urban_vs_rural_analysis(pop_with_zone):
    """Urban vs rural ridership analysis using GTFS coverage."""
    print("  Running urban vs rural analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    urban_grid = pop_with_zone[pop_with_zone['has_transit'] == 1]
    rural_grid = pop_with_zone[pop_with_zone['has_transit'] == 0]

    stats_text = f"""Transit-Served (has stop within 5km):
  Grid cells: {len(urban_grid):,}
  Mean density: {urban_grid['density_per_km2'].mean():.2f}/km²
  Median density: {urban_grid['density_per_km2'].median():.2f}/km²
  Total density sum: {urban_grid['density_per_km2'].sum():,.0f}

Unserved (no stop within 5km):
  Grid cells: {len(rural_grid):,}
  Mean density: {rural_grid['density_per_km2'].mean():.2f}/km²
  Median density: {rural_grid['density_per_km2'].median():.2f}/km²
  Total density sum: {rural_grid['density_per_km2'].sum():,.0f}

Note: Urban = served, Rural = unserved"""

    axes[0, 0].text(0.05, 0.95, stats_text, transform=axes[0, 0].transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    family='monospace')
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Urban vs Rural Coverage Summary')

    high_density = pop_with_zone[pop_with_zone['density_per_km2'] > pop_with_zone['density_per_km2'].quantile(0.75)]
    low_density = pop_with_zone[pop_with_zone['density_per_km2'] <= pop_with_zone['density_per_km2'].quantile(0.25)]

    high_served = (high_density['has_transit'] == 1).sum() / len(high_density) * 100
    low_served = (low_density['has_transit'] == 1).sum() / len(low_density) * 100

    axes[0, 1].bar(['Low Density\n(Q0-Q25)', 'High Density\n(Q75-Q100)'],
                   [low_served, high_served],
                   color=['seagreen', 'coral'], alpha=0.8)
    axes[0, 1].set_ylabel('% Grid Cells with Transit Coverage')
    axes[0, 1].set_title('Transit Coverage:\nLow vs High Density Areas')
    axes[0, 1].set_ylim(0, max(low_served, high_served) * 1.3)
    for i, v in enumerate([low_served, high_served]):
        axes[0, 1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=10)

    urban_low = low_density[low_density['has_transit'] == 1]
    urban_high = high_density[high_density['has_transit'] == 1]
    rural_low = low_density[low_density['has_transit'] == 0]
    rural_high = high_density[high_density['has_transit'] == 0]

    if len(urban_low) > 0:
        axes[1, 0].hist(urban_low['density_per_km2'], bins=20, alpha=0.6,
                        label=f'Served Low-Density ({len(urban_low):,})', color='steelblue', density=True)
    if len(rural_low) > 0:
        axes[1, 0].hist(rural_low['density_per_km2'], bins=20, alpha=0.6,
                        label=f'Unserved Low-Density ({len(rural_low):,})', color='seagreen', density=True)
    axes[1, 0].set_xlabel('Population Density (per km²)')
    axes[1, 0].set_ylabel('Density (normalized)')
    axes[1, 0].set_title('Density Distribution: Served vs Unserved\n(Low-Density Q0-Q25 Zone)')
    axes[1, 0].legend(fontsize=8)

    if len(high_density) > 0 and len(low_density) > 0:
        ks_stat, ks_p = stats.ks_2samp(high_density['density_per_km2'], low_density['density_per_km2'])
        axes[1, 1].text(0.1, 0.5,
                        f"Kolmogorov-Smirnov Test\nHigh Density (Q75-Q100) vs Low Density (Q0-Q25)\n\n"
                        f"KS Statistic: {ks_stat:.4f}\np-value: {ks_p:.4e}\n\n"
                        f"High Density: n={len(high_density):,}, mean={high_density['density_per_km2'].mean():.1f}\n"
                        f"Low Density: n={len(low_density):,}, mean={low_density['density_per_km2'].mean():.1f}",
                        transform=axes[1, 1].transAxes, fontsize=9,
                        verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Statistical Comparison of Density Distributions')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'population_urban_rural.png'), dpi=150)
    plt.close()
    print("    Saved population_urban_rural.png")

    summary_csv = pd.DataFrame({
        'Category': ['Transit-Served', 'Unserved', 'Low Density (Q0-Q25)', 'High Density (Q75-Q100)'],
        'Grid_Cells': [len(urban_grid), len(rural_grid), len(low_density), len(high_density)],
        'Mean_Density_per_km2': [
            urban_grid['density_per_km2'].mean(),
            rural_grid['density_per_km2'].mean(),
            low_density['density_per_km2'].mean(),
            high_density['density_per_km2'].mean()
        ],
        'Median_Density_per_km2': [
            urban_grid['density_per_km2'].median(),
            rural_grid['density_per_km2'].median(),
            low_density['density_per_km2'].median(),
            high_density['density_per_km2'].median()
        ],
        'Has_Transit_Count': [
            (urban_grid['has_transit'] == 1).sum(),
            (rural_grid['has_transit'] == 1).sum(),
            (low_density['has_transit'] == 1).sum(),
            (high_density['has_transit'] == 1).sum()
        ],
        'Transit_Coverage_Pct': [
            100.0, 0.0, low_served, high_served
        ]
    })
    summary_csv.to_csv(os.path.join(OUTPUT_DIR, 'population_urban_rural_stats.csv'), index=False)
    print("    Saved population_urban_rural_stats.csv")


def ridership_correlation_by_service(pop_with_zone, ridership):
    """Correlate ridership with transit coverage metrics."""
    print("  Running ridership correlation by service...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pop_summary = {
        'total_grid_cells': len(pop_with_zone),
        'transit_served_cells': (pop_with_zone['has_transit'] == 1).sum(),
        'mean_density': pop_with_zone['density_per_km2'].mean(),
        'median_density': pop_with_zone['density_per_km2'].median(),
        'mean_density_served': pop_with_zone[pop_with_zone['has_transit'] == 1]['density_per_km2'].mean() if (pop_with_zone['has_transit'] == 1).any() else 0,
        'mean_density_unserved': pop_with_zone[pop_with_zone['has_transit'] == 0]['density_per_km2'].mean() if (pop_with_zone['has_transit'] == 0).any() else 0,
        'total_density_served': pop_with_zone[pop_with_zone['has_transit'] == 1]['density_per_km2'].sum(),
        'total_density_unserved': pop_with_zone[pop_with_zone['has_transit'] == 0]['density_per_km2'].sum(),
    }

    service_cols = [c for c in ridership.columns if c.startswith('bus_') or c.startswith('rail_')]

    ridership_stats = {}
    for col in service_cols:
        if col in ridership.columns:
            valid = ridership[col].dropna()
            if len(valid) > 0:
                ridership_stats[col] = {
                    'mean': valid.mean(),
                    'std': valid.std(),
                    'q25': valid.quantile(0.25),
                    'median': valid.median(),
                    'q75': valid.quantile(0.75),
                }

    service_names = list(ridership_stats.keys())
    means = [ridership_stats[s]['mean'] for s in service_names]
    axes[0, 0].barh(service_names, means, color='steelblue', alpha=0.8)
    axes[0, 0].set_xlabel('Mean Daily Ridership')
    axes[0, 0].set_title('Mean Daily Ridership by Service')
    axes[0, 0].tick_params(axis='y', rotation=0)

    total_pop = pop_summary['total_density_served'] + pop_summary['total_density_unserved']
    served_pct = pop_summary['total_density_served'] / total_pop * 100 if total_pop > 0 else 0
    unserved_pct = pop_summary['total_density_unserved'] / total_pop * 100 if total_pop > 0 else 0

    axes[0, 1].bar(['Transit-Served\nPop. Density', 'Unserved\nPop. Density'],
                   [pop_summary['mean_density_served'], pop_summary['mean_density_unserved']],
                   color=['steelblue', 'coral'], alpha=0.8)
    axes[0, 1].set_ylabel('Mean Population Density (per km²)')
    axes[0, 1].set_title(f'Population Density:\nTransit-Served vs Unserved\n({served_pct:.1f}% vs {unserved_pct:.1f}% of total)')

    service_pop_means = []
    for col in service_cols:
        if col in ridership.columns:
            service_pop_means.append(pop_summary['mean_density_served'])
        else:
            service_pop_means.append(0)

    axes[1, 0].scatter(service_pop_means, means, c='mediumpurple', s=100, alpha=0.7)
    for i, s in enumerate(service_names):
        axes[1, 0].annotate(s, (service_pop_means[i], means[i]),
                            textcoords='offset points', xytext=(5, 5), fontsize=7)
    axes[1, 0].set_xlabel('Mean Population Density of Service Coverage Area (proxy)')
    axes[1, 0].set_ylabel('Mean Daily Ridership')
    axes[1, 0].set_title('Service Coverage Density vs Mean Ridership')

    if len(service_cols) > 1:
        correlations = []
        for col in service_cols:
            if col in ridership.columns:
                corr = ridership[col].corr(ridership.get('total_ridership', pd.Series([np.nan] * len(ridership))))
                correlations.append({'service': col, 'corr_with_total': corr})

        corr_df = pd.DataFrame(correlations).sort_values('corr_with_total')
        colors = ['coral' if x < 0 else 'seagreen' for x in corr_df['corr_with_total']]
        axes[1, 1].barh(corr_df['service'], corr_df['corr_with_total'], color=colors, alpha=0.8)
        axes[1, 1].axvline(0, color='grey', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Correlation with Total Ridership')
        axes[1, 1].set_title('Service Ridership Correlation\nwith Total Ridership')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'population_ridership_correlation.png'), dpi=150)
    plt.close()
    print("    Saved population_ridership_correlation.png")


def spatial_density_heatmap(pop_df, all_stops):
    """Create spatial heatmap of density and transit coverage."""
    print("  Running spatial density heatmap...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    scatter = axes[0].scatter(pop_df['longitude'], pop_df['latitude'],
                               c=np.log1p(pop_df['density_per_km2']),
                               cmap='YlOrRd', s=1, alpha=0.5)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Population Density Heatmap\n(Log scale)')
    plt.colorbar(scatter, ax=axes[0], label='Log(Density + 1)')

    if len(all_stops) > 0:
        axes[1].scatter(all_stops[:, 1], all_stops[:, 0],
                        c='blue', s=1, alpha=0.3, label='GTFS Stops')
        axes[1].scatter(pop_df['longitude'], pop_df['latitude'],
                        c=np.log1p(pop_df['density_per_km2']),
                        cmap='YlOrRd', s=1, alpha=0.3)
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].set_title('GTFS Stop Locations\n(overlaid on population density)')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No GTFS stop data available',
                     ha='center', va='center', transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'population_spatial_heatmap.png'), dpi=150)
    plt.close()
    print("    Saved population_spatial_heatmap.png")


def main():
    print("=" * 60)
    print("Bivariate EDA: Population Density vs Ridership")
    print("Objective: Per-capita ridership by density zone")
    print("=" * 60)

    print("\nLoading population density data...")
    pop_df = load_population_data()
    print(f"  Grid cells: {len(pop_df)}")
    print(f"  Density range: {pop_df['density_per_km2'].min():.2f} - {pop_df['density_per_km2'].max():.2f} per km²")

    print("\nLoading ridership data...")
    ridership = load_ridership()
    service_cols = [c for c in ridership.columns if c not in ['total_ridership', 'day_of_week', 'is_weekend', 'month', 'year']]
    print(f"  Ridership: {ridership.shape[0]} days, services: {service_cols}")

    print("\nLoading GTFS stop locations...")
    all_stops, service_labels = load_gtfs_all_stops()
    print(f"  Total GTFS stops loaded: {len(all_stops)}")

    print("\nClassifying density zones...")
    pop_df = classify_density_zones(pop_df)
    print(f"  Zone distribution:\n{pop_df['density_zone'].value_counts().to_dict()}")

    print("\nMapping stops to zones...")
    pop_with_zone = map_stops_to_zones(all_stops, pop_df, buffer_km=5)
    transit_served = (pop_with_zone['has_transit'] == 1).sum()
    print(f"  Grid cells with transit coverage (5km buffer): {transit_served} / {len(pop_with_zone)} ({transit_served/len(pop_with_zone)*100:.1f}%)")

    print("\nRunning analysis...")
    population_density_overview(pop_df)
    transit_coverage_by_zone(pop_with_zone)
    urban_vs_rural_analysis(pop_with_zone)
    ridership_correlation_by_service(pop_with_zone, ridership)
    spatial_density_heatmap(pop_df, all_stops)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()