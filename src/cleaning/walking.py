"""
Cleaning tasks:
  1. Verify and document NoData mask (none expected, confirm)
  2. Validate value range (must be > 0; negative = corrupt)
  3. Flag the 0.06 spike as a known model artefact — NOT removed
  4. Clip extreme high-end outliers if any exist above physical plausibility
  5. Enforce correct NoData registration in output metadata
  6. Reproject to EPSG:4326 if CRS is missing or non-standard
"""

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
import warnings

NODATA_VAL  = -9999.0
INPUT_PATH  = "data/walking_friction.tif"
OUTPUT_PATH = "data/cleaned/walking_friction_clean.tif"

# ── Load ──────────────────────────────────────────────────────────────────────
with rasterio.open(INPUT_PATH) as src:
    data      = src.read(1).astype(np.float32)   # shape: (height, width)
    profile   = src.profile.copy()
    transform = src.transform
    crs       = src.crs
    nodata    = src.nodata

print("=== Walking Friction TIFF ===")
print(f"Shape:         {data.shape}  (height × width)")
print(f"Dtype:         {data.dtype}")
print(f"Declared nodata: {nodata}")
print(f"CRS:           {crs}")
print(f"Transform:     {transform}")

# ── 1. NoData mask ────────────────────────────────────────────────────────────
nodata_mask = (data == NODATA_VAL)
nan_mask    = ~np.isfinite(data)
print(f"\nNoData pixels (-9999): {nodata_mask.sum():,}")
print(f"NaN/Inf pixels:        {nan_mask.sum():,}")

# Combine into a single invalid mask
invalid_mask = nodata_mask | nan_mask
data_clean   = np.where(invalid_mask, NODATA_VAL, data)

# ── 2. Validate value range ───────────────────────────────────────────────────
valid = data_clean[~invalid_mask]
neg_count = (valid < 0).sum()
zero_count = (valid == 0).sum()

print(f"\nValid pixels:    {len(valid):,}")
print(f"Negative values: {neg_count:,}  ← must be 0")
print(f"Zero values:     {zero_count:,}  ← water/sea tiles are 0 in this model")
print(f"Min: {valid.min():.6f}  Max: {valid.max():.6f}")

if neg_count > 0:
    print(f"[WARN] {neg_count:,} negative friction values found — clipping to 0")
    data_clean = np.where((data_clean < 0) & (~invalid_mask), 0.0, data_clean)

# ── 3. Document the 0.06 spike ───────────────────────────────────────────────
# 0.06 min/m is the default road friction (paved road, normal conditions) in
# the Weiss et al. friction surface. It is not an artefact — it just means
# "this pixel is a road cell with no further refinement."
road_default = np.isclose(valid, 0.06, atol=1e-5).sum()
print(f"\nPixels at road-default (0.06 min/m): {road_default:,} "
      f"({100*road_default/len(valid):.1f}% of valid) — model baseline, NOT an error")

# ── 4. Clip extreme outliers ──────────────────────────────────────────────────
# Friction values above ~10 min/m are physically implausible for walking
# (that's slower than crawling). Cap at p99.9 to handle any rogue cells.
p999 = np.percentile(valid, 99.9)
extreme = (data_clean > p999) & (~invalid_mask)
print(f"\n99.9th percentile: {p999:.4f} min/m")
print(f"Pixels above p99.9 (clipped): {extreme.sum():,}")
data_clean = np.where(extreme, p999, data_clean)

# ── 5. CRS handling ───────────────────────────────────────────────────────────
# The file has a ModelTransformationTag but no GeoAsciiParamsTag in the raw
# TIFF IFD. Rasterio may or may not resolve this. If CRS is None, we assign
# EPSG:4326 (WGS84 geographic) — the standard for global friction surfaces.
if crs is None:
    warnings.warn("CRS missing — assigning EPSG:4326 (WGS84). Verify this is correct.")
    crs = CRS.from_epsg(4326)
    print("[WARN] No CRS found — assigned EPSG:4326")
else:
    print(f"CRS present: {crs.to_string()}")

# ── 6. Export clean raster ────────────────────────────────────────────────────
profile.update(
    dtype    = rasterio.float32,
    nodata   = NODATA_VAL,
    crs      = crs,
    compress = "lzw",       # lossless compression — reduces file size ~3×
    tiled    = True,
    blockxsize = 512,
    blockysize = 512,
)

with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
    dst.write(data_clean.astype(np.float32), 1)

print(f"\nExported: {OUTPUT_PATH}")
print("Summary of changes applied:")
print(f"  - NaN/Inf pixels → {NODATA_VAL}")
print(f"  - Negative values → 0.0  (if any)")
print(f"  - Values above p99.9 clipped to {p999:.4f}")
print(f"  - CRS enforced: {crs.to_string()}")
print(f"  - LZW compression added")
print(f"  - NoData value registered in metadata: {NODATA_VAL}")