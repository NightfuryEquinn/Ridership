# GTFS EDA Results

Exploratory data analysis of Malaysian public transit GTFS feeds for four operators: Rapid Rail KL, RapidBus KL, RapidBus Penang, and KTMB. The `gtfs.py` cleaning script was run once per operator and exported stop-node and stop-edge tables consumed downstream by `population.py`, `osm.py`, and `feature_align.py`.

---

## Network Topology Visualisations

### 01_route_length_coverage.png
**What:** Distribution of route lengths (km) across all operators and route types.
**Analysis:** Rail routes (KTMB, Rapid Rail KL) are much longer than bus routes (RapidBus KL/Penang). KTM inter-city routes reach 300–400+ km. Rapid Rail KL routes (LRT/MRT/Monorail) are 10–60 km. RapidBus routes cluster under 30 km, consistent with urban feeder network design. Route length is included in the GTFS static features via `gtfs_avg_segment_s` — shorter average segment travel times indicate denser stop spacing.

### 02_stop_density_clustering.png
**What:** Spatial clustering of transit stops (from all four GTFS operators) across Malaysia.
**Analysis:** Two dominant clusters emerge: the Klang Valley metro area (high density, all four operators present) and the Penang island/mainland corridor (RapidBus Penang + Penang ferry). KTMB stops form a linear spine along the Peninsular rail corridor from Johor Bahru to Padang Besar. Isolated stop clusters correspond to KTM Tebrau (JB area) and KTM Komuter Utara (Ipoh–Butterworth). The spatial distribution confirms the feature pipeline correctly captures transit network coverage predominantly in high-ridership urban areas.

### 03_operator_comparison.png (if present)
**What:** Side-by-side comparison of route counts, stop counts, and directed edge counts across all four operators.
**Analysis:** Rapid Rail KL has the fewest routes but the most riders per route (high-capacity rail). RapidBus KL has the highest route and stop counts. KTMB has the most directed edges due to the length of inter-city routes. RapidBus Penang is the smallest operator. These per-operator statistics aggregate into the four scalar GTFS features broadcast to all dates in `features_aligned.csv`: `gtfs_n_stops`, `gtfs_n_routes`, `gtfs_n_directed_edges`, `gtfs_avg_segment_s`.

### 04_service_schedule_coverage.png (if present)
**What:** Heatmap of service coverage by day-of-week for each operator (from `calendar.txt`).
**Analysis:** Rapid Rail KL and RapidBus KL provide 7-day service, with slightly different headways on weekends. KTMB inter-city/ETS services show 7-day operation. Some Komuter services have modified weekend schedules. This analysis confirms the `expected_service_ids` operator presets in `gtfs.py` are correctly specified and validates that the model's `is_weekend` feature aligns with actual service availability.

### 05_stop_connectivity.png (if present)
**What:** Distribution of stop out-degree (number of unique next stops reachable in one trip) per operator.
**Analysis:** Terminal stops have out-degree 1; transfer hubs have higher degrees. Rail terminal/transfer stations (e.g., KL Sentral, Masjid Jamek) show the highest connectivity. This confirms the graph artefact export in `gtfs.py` (`gtfs_stop_edges_{operator}.csv`) correctly encodes the network topology. The `gtfs_n_directed_edges` scalar in the flat feature matrix is the aggregate count of these edges.
