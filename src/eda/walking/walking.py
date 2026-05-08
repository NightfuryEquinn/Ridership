import numpy as np
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import label, find_objects
import os
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "../../../data/cleaned/walking_friction_clean.tif"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NODATA_VAL = -9999.0

with rasterio.open(DATA_PATH) as src:
    data = src.read(1).astype(np.float32)
    transform = src.transform
    crs = src.crs
    nodata = src.nodata

h, w = data.shape
nodata_mask = data == NODATA_VAL
valid_mask = ~nodata_mask

valid_data = data[valid_mask]
accessibility_score = 1.0 / data
accessibility_score_2d = np.where(valid_mask, accessibility_score, np.nan)
print(f"\nAccessibility Score (1/friction):")
print(f"  Min: {accessibility_score.min():.4f}")
print(f"  Max: {accessibility_score.max():.4f}")
print(f"  Mean: {accessibility_score_2d[valid_mask].mean():.4f}")
print(f"  Median: {np.nanmedian(accessibility_score_2d):.4f}")

coords_y, coords_x = np.where(valid_mask)
lons = coords_x * transform.a + transform.c
lats = coords_y * transform.e + transform.f

def plot_friction_surface():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    vmin, vmax = np.percentile(valid_data, [2, 98])
    im1 = axes[0].imshow(data, cmap='RdYlGn_r', vmin=vmin, vmax=vmax,
                         extent=[lons.min(), lons.max(), lats.min(), lats.max()])
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Friction Surface (min/m)')
    plt.colorbar(im1, ax=axes[0], label='Friction (min/m)')

    im2 = axes[1].imshow(np.where(valid_mask, accessibility_score_2d, np.nan),
                          cmap='RdYlGn', vmin=0, vmax=np.percentile(accessibility_score_2d[valid_mask], 95),
                          extent=[lons.min(), lons.max(), lats.min(), lats.max()])
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('Accessibility Score (1/friction)')
    plt.colorbar(im2, ax=axes[1], label='Accessibility')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "friction_surface_spatial.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: friction_surface_spatial.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    im3 = axes[0, 0].hist(valid_data, bins=100, color='steelblue', edgecolor='none', alpha=0.7)
    axes[0, 0].axvline(valid_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {valid_data.mean():.4f}')
    axes[0, 0].axvline(np.median(valid_data), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(valid_data):.4f}')
    axes[0, 0].set_xlabel('Friction (min/m)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Friction Value Distribution')
    axes[0, 0].legend()

    log_friction = np.log10(valid_data)
    axes[0, 1].hist(log_friction, bins=100, color='purple', edgecolor='none', alpha=0.7)
    axes[0, 1].set_xlabel('Log10(Friction)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Log10 Friction Distribution')

    axes[1, 0].hist(accessibility_score_2d[valid_mask], bins=100, color='green', edgecolor='none', alpha=0.7)
    axes[1, 0].axvline(accessibility_score_2d[valid_mask].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {accessibility_score_2d[valid_mask].mean():.4f}')
    axes[1, 0].axvline(np.nanmedian(accessibility_score_2d), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.nanmedian(accessibility_score_2d):.4f}')
    axes[1, 0].set_xlabel('Accessibility Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Accessibility Score Distribution')
    axes[1, 0].legend()

    percentiles = np.percentile(valid_data, np.arange(0, 101, 10))
    axes[1, 1].boxplot(valid_data.ravel(), vert=True)
    axes[1, 1].set_ylabel('Friction (min/m)')
    axes[1, 1].set_title('Friction Value Boxplot')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "friction_distributions.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: friction_distributions.png")


def plot_accessibility_zones():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    low_thresh = np.percentile(valid_data, 25)
    high_thresh = np.percentile(valid_data, 75)

    low_acc = data < low_thresh
    mid_acc = (data >= low_thresh) & (data <= high_thresh)
    high_acc = data > high_thresh

    zone_data = np.zeros_like(data, dtype=int)
    zone_data[low_acc & valid_mask] = 1
    zone_data[mid_acc & valid_mask] = 2
    zone_data[high_acc & valid_mask] = 3

    cmap_zone = plt.cm.colors.ListedColormap(['green', 'yellow', 'red'])
    im = axes[0].imshow(zone_data, cmap=cmap_zone,
                        extent=[lons.min(), lons.max(), lats.min(), lats.max()])
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Accessibility Zones (Low/Medium/High Friction)')

    zone_labels = ['Low Friction (High Access)', 'Medium Friction', 'High Friction (Low Access)']
    colors = ['green', 'yellow', 'red']
    handles = [plt.Rectangle((0,0),1,1, color=c) for c in colors]
    axes[0].legend(handles, zone_labels, loc='lower left')

    zone_counts = [np.sum(low_acc), np.sum(mid_acc), np.sum(high_acc)]
    axes[1].bar(['Low', 'Medium', 'High'], zone_counts, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Accessibility Zone')
    axes[1].set_ylabel('Pixel Count')
    axes[1].set_title('Accessibility Zone Distribution')
    for i, v in enumerate(zone_counts):
        axes[1].text(i, v + max(zone_counts)*0.01, f'{v:,}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accessibility_zones.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: accessibility_zones.png")

    print(f"\nAccessibility Zone Summary:")
    print(f"  Low friction (high access): {zone_counts[0]:,} pixels ({100*zone_counts[0]/valid_mask.sum():.1f}%)")
    print(f"  Medium friction: {zone_counts[1]:,} pixels ({100*zone_counts[1]/valid_mask.sum():.1f}%)")
    print(f"  High friction (low access): {zone_counts[2]:,} pixels ({100*zone_counts[2]/valid_mask.sum():.1f}%)")


def plot_friction_hotspots():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    threshold_high = np.percentile(valid_data, 90)
    hotspot_mask = (data > threshold_high) & valid_mask

    axes[0, 0].imshow(np.where(hotspot_mask, data, np.nan), cmap='Reds',
                      extent=[lons.min(), lons.max(), lats.min(), lats.max()])
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    axes[0, 0].set_title(f'Friction Hotspots (>{threshold_high:.4f} min/m, top 10%)')

    axes[0, 1].hist(valid_data, bins=100, color='gray', edgecolor='none', alpha=0.5, label='All valid')
    axes[0, 1].axvline(threshold_high, color='red', linestyle='--', linewidth=2,
                       label=f'90th pctl: {threshold_high:.4f}')
    axes[0, 1].hist(data[hotspot_mask], bins=50, color='red', edgecolor='none', alpha=0.7, label='Hotspots')
    axes[0, 1].set_xlabel('Friction (min/m)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Hotspot Friction Distribution')
    axes[0, 1].legend()

    labeled_hotspots, num_hotspots = label(hotspot_mask)
    print(f"\nHotspot Identification:")
    print(f"  Hotspot threshold (90th pctl): {threshold_high:.6f} min/m")
    print(f"  Number of hotspot clusters: {num_hotspots}")

    hotspot_sizes = []
    for i in range(1, num_hotspots + 1):
        hotspot_sizes.append(np.sum(labeled_hotspots == i))
    hotspot_sizes = np.array(hotspot_sizes)

    if len(hotspot_sizes) > 0:
        axes[1, 0].hist(hotspot_sizes, bins=50, color='darkred', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Cluster Size (pixels)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Distribution of Hotspot Cluster Sizes (n={num_hotspots})')
        axes[1, 0].axvline(np.mean(hotspot_sizes), color='blue', linestyle='--',
                          label=f'Mean: {np.mean(hotspot_sizes):.1f}')
        axes[1, 0].legend()

        largest_clusters_idx = np.argsort(hotspot_sizes)[-10:][::-1]
        largest_sizes = hotspot_sizes[largest_clusters_idx]
        axes[1, 1].bar(range(1, 11), largest_sizes, color='darkred', edgecolor='black', alpha=0.8)
        axes[1, 1].set_xlabel('Rank')
        axes[1, 1].set_ylabel('Cluster Size (pixels)')
        axes[1, 1].set_title('Top 10 Largest Hotspot Clusters')
        for i, v in enumerate(largest_sizes):
            axes[1, 1].text(i + 1, v + max(largest_sizes)*0.02, f'{v:,}', ha='center', fontsize=9)
    else:
        axes[1, 0].text(0.5, 0.5, 'No hotspots detected', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, 'No hotspots detected', ha='center', va='center', transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "friction_hotspots.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: friction_hotspots.png")

    hotspot_slices = find_objects(labeled_hotspots)
    print(f"\nTop 10 Largest Hotspot Regions:")
    sorted_idx = np.argsort(hotspot_sizes)[::-1][:10]
    for rank, idx in enumerate(sorted_idx, 1):
        slc = hotspot_slices[idx]
        region = data[slc]
        center_y = (slc[0].start + slc[0].stop) // 2
        center_x = (slc[1].start + slc[1].stop) // 2
        center_lon = center_x * transform.a + transform.c
        center_lat = center_y * transform.e + transform.f
        print(f"  {rank}. Cluster at ({center_lat:.4f}, {center_lon:.4f}), "
              f"size={hotspot_sizes[idx]:,} px, max_friction={region.max():.4f}")


def plot_correlation_analysis():
    fig, ax = plt.subplots(figsize=(10, 8))

    sample_size = min(50000, valid_mask.sum())
    np.random.seed(42)
    sample_idx = np.random.choice(valid_mask.sum(), size=sample_size, replace=False)
    sample_data = valid_data[sample_idx]
    sample_lons = lons[sample_idx]
    sample_lats = lats[sample_idx]

    im = ax.scatter(sample_lons, sample_lats, c=sample_data, cmap='RdYlGn_r',
                    s=5, alpha=0.6, vmin=np.percentile(sample_data, 2),
                    vmax=np.percentile(sample_data, 98))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial Friction Pattern (Sampled)')
    plt.colorbar(im, ax=ax, label='Friction (min/m)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "spatial_friction_pattern.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: spatial_friction_pattern.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    bins_lon = np.linspace(lons.min(), lons.max(), 20)
    bins_lat = np.linspace(lats.min(), lats.max(), 20)

    lon_means = []
    lon_stds = []
    for i in range(len(bins_lon) - 1):
        mask = (lons >= bins_lon[i]) & (lons < bins_lon[i+1])
        if mask.sum() > 0:
            # Get indices of valid pixels in this longitude bin
            valid_indices = np.where(valid_mask)[0]
            lon_valid_mask = mask[valid_indices]
            if lon_valid_mask.sum() > 0:
                lon_data_in_bin = valid_data[lon_valid_mask]
                lon_means.append(lon_data_in_bin.mean())
                lon_stds.append(lon_data_in_bin.std())
            else:
                lon_means.append(np.nan)
                lon_stds.append(np.nan)
        else:
            lon_means.append(np.nan)
            lon_stds.append(np.nan)

    lat_means = []
    lat_stds = []
    for i in range(len(bins_lat) - 1):
        mask = (lats >= bins_lat[i]) & (lats < bins_lat[i+1])
        if mask.sum() > 0:
            # Get indices of valid pixels in this latitude bin
            valid_indices = np.where(valid_mask)[0]
            lat_valid_mask = mask[valid_indices]
            if lat_valid_mask.sum() > 0:
                lat_data_in_bin = valid_data[lat_valid_mask]
                lat_means.append(lat_data_in_bin.mean())
                lat_stds.append(lat_data_in_bin.std())
            else:
                lat_means.append(np.nan)
                lat_stds.append(np.nan)
        else:
            lat_means.append(np.nan)
            lat_stds.append(np.nan)

    ax.errorbar(bins_lon[:-1] + np.diff(bins_lon)/2, lon_means, yerr=lon_stds,
                fmt='o-', capsize=3, color='blue', alpha=0.7, label='Longitude')
    ax.errorbar(bins_lat[:-1] + np.diff(bins_lat)/2, lat_means, yerr=lat_stds,
                fmt='s-', capsize=3, color='red', alpha=0.7, label='Latitude')
    ax.set_xlabel('Coordinate')
    ax.set_ylabel('Mean Friction (min/m)')
    ax.set_title('Mean Friction by Geographic Region')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "friction_by_region.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: friction_by_region.png")


if __name__ == "__main__":
    print("\n1. Friction Surface Spatial Analysis...")
    plot_friction_surface()

    print("\n2. Accessibility Score Distribution...")
    plot_accessibility_zones()

    print("\n3. Friction Hotspot Identification...")
    plot_friction_hotspots()

    print("\n4. Correlation Analysis...")
    plot_correlation_analysis()

    print("\n" + "=" * 60)
    print("EDA Complete! All outputs saved to:", OUTPUT_DIR)
    print("=" * 60)