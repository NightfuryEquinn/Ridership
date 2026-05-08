import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import os
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "../../../data/cleaned/population_density_clean.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nBasic statistics:")
print(df.describe())

def plot_spatial_density_map():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    scatter1 = axes[0].scatter(
        df['longitude'], df['latitude'],
        c=df['density_per_km2'], cmap='YlOrRd',
        s=20, alpha=0.8, edgecolors='none'
    )
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Population Density (per km²)')
    plt.colorbar(scatter1, ax=axes[0], label='Density')

    scatter2 = axes[1].scatter(
        df['longitude'], df['latitude'],
        c=df['density_log'], cmap='YlOrRd',
        s=20, alpha=0.8, edgecolors='none'
    )
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('Population Density (log scale)')
    plt.colorbar(scatter2, ax=axes[1], label='Log Density')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "spatial_density_map.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: spatial_density_map.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    xi = np.linspace(df['longitude'].min(), df['longitude'].max(), 100)
    yi = np.linspace(df['latitude'].min(), df['latitude'].max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata(
        (df['longitude'], df['latitude']),
        df['density_log'],
        (xi, yi),
        method='cubic'
    )

    contour = ax.contourf(xi, yi, zi, levels=20, cmap='YlOrRd', alpha=0.8)
    ax.scatter(df['longitude'], df['latitude'], c='black', s=1, alpha=0.3)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Interpolated Population Density Surface')
    plt.colorbar(contour, ax=ax, label='Log Density')
    plt.savefig(os.path.join(OUTPUT_DIR, "spatial_density_contour.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: spatial_density_contour.png")


def plot_urban_peri_urban_gradient():
    density_threshold = df['density_per_km2'].median()
    urban_mask = df['density_per_km2'] >= density_threshold
    df['zone_type'] = np.where(urban_mask, 'Urban', 'Peri-urban')

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    urban_df = df[df['zone_type'] == 'Urban']
    peri_df = df[df['zone_type'] == 'Peri-urban']

    axes[0, 0].scatter(
        urban_df['longitude'], urban_df['latitude'],
        c=urban_df['density_per_km2'], cmap='Reds',
        s=25, alpha=0.8, edgecolors='none'
    )
    axes[0, 0].set_title(f'Urban Zones (n={len(urban_df)})')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')

    axes[0, 1].scatter(
        peri_df['longitude'], peri_df['latitude'],
        c=peri_df['density_per_km2'], cmap='Blues',
        s=25, alpha=0.8, edgecolors='none'
    )
    axes[0, 1].set_title(f'Peri-urban Zones (n={len(peri_df)})')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')

    urban_center_lon = urban_df['longitude'].mean()
    urban_center_lat = urban_df['latitude'].mean()
    distance_from_center = np.sqrt(
        (df['longitude'] - urban_center_lon)**2 +
        (df['latitude'] - urban_center_lat)**2
    )
    df['distance_from_urban_center'] = distance_from_center

    bins = np.linspace(df['distance_from_urban_center'].min(), df['distance_from_urban_center'].max(), 15)
    df['distance_bin'] = pd.cut(df['distance_from_urban_center'], bins=bins)
    gradient_df = df.groupby('distance_bin')['density_per_km2'].agg(['mean', 'std', 'count']).reset_index()
    gradient_df['bin_center'] = gradient_df['distance_bin'].apply(lambda x: x.mid)

    axes[1, 0].plot(gradient_df['bin_center'], gradient_df['mean'], 'o-', color='darkgreen', linewidth=2)
    axes[1, 0].fill_between(
        gradient_df['bin_center'],
        gradient_df['mean'] - gradient_df['std'],
        gradient_df['mean'] + gradient_df['std'],
        alpha=0.3, color='green'
    )
    axes[1, 0].set_xlabel('Distance from Urban Center (degrees)')
    axes[1, 0].set_ylabel('Mean Density (per km²)')
    axes[1, 0].set_title('Density vs Distance from Urban Center')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(['Urban', 'Peri-urban'],
                   [urban_df['density_per_km2'].mean(), peri_df['density_per_km2'].mean()],
                   yerr=[urban_df['density_per_km2'].std(), peri_df['density_per_km2'].std()],
                   color=['red', 'blue'], alpha=0.7, capsize=5)
    axes[1, 1].set_ylabel('Mean Density (per km²)')
    axes[1, 1].set_title('Urban vs Peri-urban Mean Density')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "urban_peri_urban_gradient.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: urban_peri_urban_gradient.png")

    print(f"\nUrban vs Peri-urban Summary:")
    print(f"  Urban: mean={urban_df['density_per_km2'].mean():.2f}, std={urban_df['density_per_km2'].std():.2f}")
    print(f"  Peri-urban: mean={peri_df['density_per_km2'].mean():.2f}, std={peri_df['density_per_km2'].std():.2f}")


def plot_population_clustering():
    SAMPLE_SIZE = 10000
    np.random.seed(42)
    sample_idx = np.random.choice(len(df), size=min(SAMPLE_SIZE, len(df)), replace=False)
    sample_df = df.iloc[sample_idx].copy()
    coords = sample_df[['longitude', 'latitude']].values
    density = sample_df['density_log'].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    distance_matrix = pdist(coords)
    linkage_matrix = linkage(distance_matrix, method='ward')

    dendrogram(
        linkage_matrix,
        ax=axes[0],
        truncate_mode='lastp',
        p=30,
        leaf_rotation=90,
        leaf_font_size=8,
        show_contracted=True
    )
    axes[0].set_title('Hierarchical Clustering Dendrogram (Sampled)')
    axes[0].set_xlabel('Sample Index or (Cluster Size)')
    axes[0].set_ylabel('Distance')

    n_clusters = 5
    sample_clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    sample_df['cluster'] = sample_clusters

    scatter = axes[1].scatter(
        sample_df['longitude'], sample_df['latitude'],
        c=sample_df['cluster'], cmap='tab10',
        s=sample_df['density_log'] * 10, alpha=0.7, edgecolors='white', linewidths=0.5
    )
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title(f'Population Clusters (k={n_clusters}, n={len(sample_df)})')
    plt.colorbar(scatter, ax=axes[1], label='Cluster')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "population_clustering.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: population_clustering.png")

    cluster_stats = sample_df.groupby('cluster').agg({
        'longitude': 'mean',
        'latitude': 'mean',
        'density_per_km2': 'mean',
        'density_log': 'mean'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        sample_df['longitude'], sample_df['latitude'],
        c=sample_df['density_log'], cmap='hot',
        s=20, alpha=0.6, edgecolors='none'
    )
    ax.scatter(
        cluster_stats['longitude'], cluster_stats['latitude'],
        c='cyan', s=200, marker='*', edgecolors='black', linewidths=1,
        label='Cluster Centers', zorder=5
    )

    plt.colorbar(scatter, ax=ax, label='Log Density')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Hotspot Analysis - Cluster Centers (Sampled)')
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "hotspot_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: hotspot_analysis.png")

    print(f"\nCluster Summary:")
    print(cluster_stats.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster_id in sorted(sample_df['cluster'].unique()):
        cluster_data = sample_df[sample_df['cluster'] == cluster_id]
        ax.hist(cluster_data['density_per_km2'], bins=20, alpha=0.5, label=f'Cluster {cluster_id}')
    ax.set_xlabel('Density (per km²)')
    ax.set_ylabel('Frequency')
    ax.set_title('Density Distribution by Cluster')
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_density_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cluster_density_distribution.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Population Density EDA")
    print("=" * 60)

    print("\n1. Creating Spatial Density Map...")
    plot_spatial_density_map()

    print("\n2. Analyzing Urban vs Peri-urban Gradient...")
    plot_urban_peri_urban_gradient()

    print("\n3. Performing Population Clustering (Hotspot)...")
    plot_population_clustering()

    print("\n" + "=" * 60)
    print("EDA Complete! All outputs saved to:", OUTPUT_DIR)
    print("=" * 60)