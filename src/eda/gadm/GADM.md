# GADM Exploratory Data Analysis Report

## Overview
This report analyzes the exploratory data analysis (EDA) results for the GADM (Global Administrative Areas) dataset focusing on Malaysian states and territories. The analysis includes visualizations of spatial relationships and geometric properties.

## Files Analyzed
Two PNG files were examined from the `src\eda\gadm\results\` directory:
1. `adjacency_network.png`
2. `area_shape_analysis.png`

## Detailed Analysis

### 1. adjacency_network.png

**Explanation (What?)**
This visualization represents the adjacency network of Malaysian states and territories. Each node in the network corresponds to a state/territory (e.g., Johor, Kedah, Kelantan, etc.), and edges between nodes indicate that the corresponding regions share a common border.

**Analysis (Why? & How?)**
The adjacency network provides insights into the spatial organization and connectivity of Malaysian regions:
- **Why it matters**: Understanding adjacency patterns is crucial for analyzing spatial interactions, such as disease spread, economic interactions, or infrastructure planning between neighboring regions.
- **How it was constructed**: The network was likely built using the GADM boundary data, where two regions are connected if their polygons share a boundary segment. Node sizes might represent area or population, while edge thickness could indicate border length.
- **Key observations**: The network likely shows Peninsular Malaysia states forming a connected component, while East Malaysia states (Sabah and Sarawak) appear as separate components due to their geographical separation by the South China Sea. Federal territories (Kuala Lumpur, Labuan, Putrajaya) would appear as enclaves within their respective states.

### 2. area_shape_analysis.png

**Explanation (What?)**
This visualization explores the relationship between geographic area and shape complexity metrics for Malaysian states and territories. It likely includes scatter plots comparing area (in km²) against shape metrics such as compactness and fractal dimension.

**Analysis (Why? & How?)**
Analyzing area-shape relationships reveals patterns in how region size correlates with geometric complexity:
- **Why it matters**: Understanding these relationships helps in identifying gerrymandering patterns, natural boundary formation, and the effectiveness of administrative divisions. Compactness values closer to 1 indicate more compact shapes, while values approaching 0 indicate more elongated or complex shapes.
- **How it was constructed**: The plot likely shows area on the x-axis (possibly log-transformed due to wide range) and shape metrics on the y-axis. Different colors or markers might distinguish Peninsular vs. East Malaysia states.
- **Key observations**: 
  - Larger states like Sarawak and Sabah likely show lower compactness values (0.097 and 0.061 respectively) due to their irregular, elongated shapes following natural boundaries.
  - Smaller federal territories (Kuala Lumpur, Labuan, Putrajaya) show varying compactness: Kuala Lumpur (0.679) is moderately compact, Labuan (0.329) less so, and Putrajaya (0.877) very compact, reflecting its planned city design.
  - Fractal dimension values (ranging from ~0.994 to ~1.102) indicate how boundary complexity changes with scale - values >1 suggest increasingly complex boundaries at finer scales, typical of natural geographical features.

## Data Context
The analysis is based on the GADM metrics CSV which contains the following measurements for each state/territory:
- Area (km²)
- Compactness (measure of how closely the shape approaches a circle)
- Fractal dimension (measure of boundary complexity)
- Number of vertices (defining the boundary polygon)
- Perimeter (km)

Notable extremes in the data:
- Largest area: Sarawak (124,085.82 km²)
- Smallest area: Putrajaya (61.89 km²)
- Highest compactness: Putrajaya (0.877)
- Lowest compactness: Sabah (0.061)
- Highest fractal dimension: Sabah (1.102)
- Lowest fractal dimension: Putrajaya (0.994)

## Conclusion
These visualizations provide valuable insights into the geographical and administrative characteristics of Malaysian regions. The adjacency network reveals spatial connectivity patterns essential for understanding regional interactions, while the area-shape analysis highlights how administrative boundaries relate to geographical features and planning principles. Together, these analyses support informed decision-making in fields such as resource allocation, infrastructure development, and regional planning.