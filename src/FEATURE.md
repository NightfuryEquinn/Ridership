# Feature Engineering Pipeline Summary

## Overview
This document describes the feature engineering pipeline used to prepare data for transportation demand forecasting models. The pipeline processes multiple data sources to create comprehensive feature sets for both temporal (LSTM-based) and graph-based (TGACN, GCN-SBULSTM, SGCNN-STEP) models.

## Pipeline Steps

### 1. Ridership Features (`ridership_temporal.parquet`)
- Source: `data/cleaned/ridership_headline_clean.csv`
- Target column: `bus_rkl` (bus ridership)
- Features: Lagged values, rolling statistics, temporal encodings
- Output: 2647 rows × 34 columns

### 2. Fuel Price Features (`fuel_temporal.parquet`)
- Source: `data/cleaned/fuelprice_cleaned.csv`
- Features: Fuel price transformations, lagged values, rolling statistics
- Output: 2647 rows × 31 columns

### 3. Holiday Features (`holiday_flags.parquet`)
- Source: `data/cleaned/school_public_holiday_clean.csv`
- Features: Public holiday indicators, school holiday indicators, combined holiday flags
- Output: 2647 rows × 18 columns

### 4. Rainfall Features (`rainfall_temporal.parquet`)
- Source: `data/cleaned/rainfall_combined_final.csv`
- Features: Rainfall measurements from multiple stations, aggregated features
- Output: 261 rows × 1617 columns (daily resolution)

### 5. Spatial Features (`spatial_features.parquet`)
- Sources: GADM boundaries, POIs, population data, GTFS stops
- Features: POI density, population density, accessibility measures
- Output: 16 rows × 10 columns (one per GADM zone)

### 6. Walking Friction Weights (`walking_friction_weights.parquet`)
- Source: `data/cleaned/walking_friction_clean.tif` (raster)
- Features: Average walking friction per GADM zone
- Output: 2299 rows × 11 columns (per stop, per zone)

### 7. Graph Construction (`graph_adjacency.pt`, `node_features.pt`)
- GADM Graph: Administrative boundaries connectivity
  - Nodes: 16 zones
  - Edges: 20 undirected edges
  - Node features: 9-dimensional (area, perimeter, etc.)
- GTFS Graph: Public transport network connectivity
  - Nodes: 2283 stops
  - Edges: 2565 undirected edges (sequential connections)
  - Node features: 5-dimensional (lat, lon, zone_id, etc.)

### 8. Feature Fusion (`feature_matrix_lstm.parquet`)
- Combines all temporal features (ridership, fuel, holiday, rainfall, spatial)
- Common date range: 2019-01-01 to 2026-03-21
- Final shape: 2637 rows × 1717 columns
- Missing data: 30.16% (primarily from rainfall data which is daily)

## Output Files
- `feature_matrix_lstm.parquet`: Main feature set for LSTM/BiLSTM/TPA-LSTM models
- `graph_adjacency.pt`: Graph adjacency matrices for GADM and GTFS graphs
- `node_features.pt`: Node feature matrices for graph models
- Individual feature files: Intermediate outputs for each processing step
- `pipeline_report.txt`: Execution log and validation results

## Model Applications
- **Temporal Models**: LSTM, BiLSTM, TPA-LSTM use `feature_matrix_lstm.parquet`
- **Graph Models**: TGACN, GCN-SBULSTM, SGCNN-STEP use `graph_adjacency.pt` and `node_features.pt`

## Validation
All pipeline steps completed successfully with:
- No critical errors in data processing
- Graph structures validated for symmetry and connectivity
- Feature matrices saved in appropriate formats (Parquet for tabular, PyTorch tensors for graphs)