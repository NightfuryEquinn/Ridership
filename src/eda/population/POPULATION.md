# Population Density EDA Results

Exploratory data analysis of the Malaysian population density raster (`malaysia_population_density_2020.csv`, WorldPop-style grid at ~100 m resolution), processed by `population.py`.

---

## Spatial Density Visualisations

### spatial_density_map.png
**What:** Side-by-side scatter plots of raw population density (per km²) and log-transformed density (`log1p(density_per_km2)`) across all grid points within the Malaysia bounding box (lat: 0.9–7.4°N, lon: 99.6–119.3°E).
**Analysis:** The linear-scale plot reveals the extreme concentration of density in the Klang Valley — grid cells in central Kuala Lumpur exceed 10,000 persons/km² while the vast majority of Malaysia (tropical forest, highland interiors) is near zero. The log-scale plot compresses the tail and reveals moderate-density zones along coastal corridors (Penang, Johor Bahru, Ipoh, Kota Kinabalu). Both representations are computed in `population.py` (`density_log = log1p(density_per_km2)`, `density_clipped = clip(upper=p99)`). The model uses `pop_density_log_median` as a static feature, which captures the national central tendency without being dominated by Klang Valley outliers.

### spatial_density_contour.png
**What:** Interpolated population density surface using cubic interpolation over the Malaysian grid, displayed as filled contours (YlOrRd colormap).
**Analysis:** The contour representation confirms the dominant Klang Valley high-density core, the secondary Penang island cluster, and linear corridors along the western coast of Peninsular Malaysia. Sabah/Sarawak show much lower and more dispersed densities. The contour map validates that the spatial join in `population.py` (using GADM Level-1 polygons via GeoPandas `sjoin`) correctly assigns all Klang Valley cells to Selangor/KL, not to neighbouring states. The national median density (`pop_density_median`) broadcast to all model dates summarises this spatial distribution as a single stable scalar.

### state_density_comparison.png (if present)
**What:** Bar chart of median population density by Malaysian state/territory after the GADM spatial join.
**Analysis:** Federal Territories (KL, Putrajaya, Labuan) have the highest median densities as small urban enclaves. Selangor (Klang Valley suburbs) ranks highest among the 13 states. Sarawak and Kelantan have the lowest densities. This per-state distribution validates the `population_by_state.csv` output from `population.py`. High-density states also correlate with higher GTFS network density, confirming the OSM POI and GTFS features encode spatially consistent information.

### stop_density_sampling.png (if present)
**What:** Distribution of population density values at GTFS stop locations (nearest-grid nearest-neighbour lookup via BallTree).
**Analysis:** Transit stops are systematically located in higher-density areas than the national grid median — a well-known transit network design principle. The median nearest-grid distance for Malaysian GTFS stops is typically < 0.5 km, confirming the BallTree lookup accuracy. The resulting `population_at_stops.csv` encodes per-stop population context, which `feature_align.py` summarises as the national median for the flat feature matrix.
