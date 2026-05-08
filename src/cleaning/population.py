import pandas as pd
import numpy as np

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/malaysia_population_density_2020.csv")

print("=== Raw shape ===")
print(df.shape)
print(df.dtypes)
print(df.head())

# ── 1. Rename opaque columns ──────────────────────────────────────────────────
df.columns = ["longitude", "latitude", "density_per_km2"]

# ── 2. Validate coordinate bounds (Peninsular + East Malaysia) ────────────────
BOUNDS = dict(lon=(99.6, 119.3), lat=(0.9, 7.4))

out_lon = ~df["longitude"].between(*BOUNDS["lon"])
out_lat = ~df["latitude"].between(*BOUNDS["lat"])
print(f"\nCoordinates out of Malaysia bounding box: lon={out_lon.sum()}, lat={out_lat.sum()}")

df = df[~(out_lon | out_lat)].copy()

# ── 3. Validate density values ────────────────────────────────────────────────
print("\nDensity stats:")
print(df["density_per_km2"].describe().round(4))

neg = (df["density_per_km2"] < 0).sum()
print(f"Negative density values: {neg}")        # should be 0

# ── 4. Outlier handling — right-skewed distribution ───────────────────────────
# The max (≈18,495 per km²) represents dense KL grid cells.
# Without transformation, all colour weight collapses to a handful of urban cells
# on any choropleth. Two options are created; pick the one that fits the use case.

p99 = df["density_per_km2"].quantile(0.99)
df["density_log"]     = np.log1p(df["density_per_km2"])   # smooth, preserves rank
df["density_clipped"] = df["density_per_km2"].clip(upper=p99)  # readable colour scale

print(f"\n99th-percentile clip threshold: {p99:.2f} per km²")
print(f"Log-transform range: {df['density_log'].min():.4f} → {df['density_log'].max():.4f}")

# ── 5. Sanity checks ──────────────────────────────────────────────────────────
print(f"\nFinal shape: {df.shape}")
print(f"Nulls: {df.isnull().sum().sum()}")
print(df.head())

# ── 6. Export ─────────────────────────────────────────────────────────────────
df.to_csv("data/cleaned/population_density_clean.csv", index=False)
print("\nExported: data/cleaned/population_density_clean.csv")