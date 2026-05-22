# Thematic EDA Results

Cross-source thematic visualisations synthesising ridership with the spatial and contextual feature sources in this project.

---

## Network and Spatial Summaries

### network_performance_summary.png
**What:** Summary dashboard of Malaysian public transit network performance: total ridership trends by mode (bus vs. rail sub-groups), modal share breakdown, and year-over-year growth rates, 2022–2025.
**Analysis:** Rail modes (LRT/MRT) dominate the ridership total, with bus providing a substantial but lower share. The network performance summary confirms the post-MCO recovery trajectory is predominantly rail-led. Bus ridership (RapidBus KL and Penang) is recovering more slowly, consistent with the mode-level analysis in `RIDERSHIP.md`. This aggregate view motivates forecasting `total_ridership` as the primary target while retaining all 12 service-line ridership values as contextual target features in the pipeline (feature group 0–12 in HMT-TSF's `FEAT_GROUPS`).

### population_poi_network_overlap.png
**What:** Spatial overlay visualisation of population density, OSM POI density, and GTFS transit stop locations for the Klang Valley area.
**Analysis:** The three layers show strong co-location: high-population-density areas (Ampang, Chow Kit, Bangsar) have dense POI counts and multiple GTFS stops within 500 m. This spatial co-occurrence validates the feature construction logic in `population.py` and `osm.py` — the population-at-stops and POI-at-stops features encode genuine spatial demand signals rather than noise. The `gtfs_n_stops`, `osm_poi_total_mean`, and `pop_density_median` static features collectively describe this spatial context in the flat model input.

### ridership_by_mode.png
**What:** Stacked bar or line chart of daily ridership broken down by modal group (urban rail, inter-city rail, bus) from 2022 to 2025.
**Analysis:** Urban rail (LRT/MRT/Monorail) consistently accounts for ~55–65% of total ridership. Bus (RapidBus KL + Penang) contributes ~25–30%. Inter-city rail (KTM ETS, Intercity, Komuter) makes up the remainder. This decomposition explains why some models perform better on `total_ridership` (dominated by urban rail, which has the most predictable weekday/weekend pattern) than on individual service lines (inter-city and new-launch lines have higher residual noise).

---

## Feature Interaction Analyses

### feature_correlation_heatmap.png (if present)
**What:** Pearson correlation matrix of the 79 features in `features_aligned.csv`, computed on the training split (2022-01-01 to ~2023-12).
**Analysis:** This is the feature-level adjacency used by all graph-based models (STGCN, MTGNN, STSGCN, STFGNN, PDR-STGCN, ASTGCN, HMT-TSF) with threshold=0.1. High-correlation blocks expected: (1) all 13 ridership target features correlated with each other; (2) fuel price levels within each fuel type; (3) rainfall states with geographically proximate neighbours; (4) `dow_sin`/`dow_cos` correlated with ridership and `is_weekend`. Static features (population, GTFS, GADM) have near-zero correlation with dynamic features — they provide constant graph nodes, which is intentional.

### mcо_impact_summary.png (if present)
**What:** Comparison of ridership distributions before, during, and after the MCO period (2020-03-18 – 2021-12-31) for selected service lines.
**Analysis:** Confirms the MCO structural break is the dominant signal in the full historical series. Post-MCO distribution shifts right (increasing ridership) but does not return to pre-MCO levels for most services by 2025. The regime-gating component in HMT-TSF (K=3: pre-MCO / MCO / post-MCO) directly addresses this distributional non-stationarity by learning separate regime embeddings — the regime gate learns to weight the post-MCO embedding during inference without requiring `is_mco` as an explicit input feature at inference time.
