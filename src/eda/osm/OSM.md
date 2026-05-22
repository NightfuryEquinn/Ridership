# OSM EDA Results

> Last updated: 2026-05-22

Exploratory data analysis of OpenStreetMap Points of Interest (POIs) from `osm_pois.json` (Overpass API export covering Malaysia), processed by `osm.py` into per-stop catchment counts.

---

## Data Source Overview

The raw OSM export contains nodes and ways (polygons with computed centroids) tagged with `amenity`, `shop`, `leisure`, or `tourism` keys. After deduplication and coordinate validation, `osm.py` exports `data/cleaned/osm_pois_clean.json`.

POIs are aggregated into a 500 m catchment radius around each GTFS stop using a BallTree haversine query. The per-stop counts feed into `features_aligned.csv` as eight mean-per-stop static features: `osm_poi_total_mean`, `osm_poi_transport_mean`, `osm_poi_food_mean`, `osm_poi_retail_mean`, `osm_poi_education_mean`, `osm_poi_healthcare_mean`, `osm_poi_leisure_mean`, `osm_poi_other_mean`.

---

## CSV Outputs (no PNGs generated)

The OSM EDA pipeline produces tabular outputs rather than PNG visualisations. These are stored in `src/eda/osm/results/`:

### density_by_category.csv
Counts of POIs per OSM category (`transport`, `food`, `retail`, `education`, `healthcare`, `leisure`, `other`) aggregated across all stops. Confirms the category taxonomy in `CATEGORY_MAP` covers the dominant OSM tag values in the Malaysian dataset.

### density_summary.csv
Per-stop descriptive statistics (mean, median, p25, p75, p99) for `poi_total` and each category. Expected to show high right-skew: CBD stops (e.g., near KL Sentral, Bukit Bintang) have 200–500+ POIs within 500 m; suburban terminal stops may have fewer than 10. This skew motivates the `log1p` transformed variants computed per-stop in `osm.py`.

### poi_frequency_ranking.csv
Ranking of POI sub-types by raw frequency across the dataset. Top tags are expected to be `restaurant`, `cafe`, `bus_stop`, `school`, `pharmacy` — consistent with Malaysian urban environments. This confirms the category mapping is capturing the most common POI types.

### spatial_clustering.csv
Per-stop coordinates joined with `poi_total` count, allowing GIS inspection of spatial concentration. High-POI clusters align with Klang Valley commercial centres and the Penang island urban core — consistent with the GTFS stop density clusters observed in the GTFS EDA.

---

## Feature Impact Notes

The eight `osm_poi_*_mean` features are static (broadcast to all dates). As scalars, they encode the national-average stop-level POI density rather than spatial variation. Richer spatial disaggregation (per-stop or per-line POI features) would require restructuring the model input tensor but could improve accuracy for line-level forecasting in future work.
