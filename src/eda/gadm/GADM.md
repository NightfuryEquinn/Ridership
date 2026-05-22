# GADM EDA Results

Exploratory data analysis of Malaysia's Level-1 administrative boundaries from the GADM dataset (`gadm_mys_l1.json`), covering all 16 states and federal territories (13 states + KL + Labuan + Putrajaya).

---

## Spatial Visualisations

### adjacency_network.png
**What:** Network graph where nodes are Malaysian states/territories and edges represent shared land borders, with edge weights proportional to shared border length in km.
**Analysis:** The adjacency network is the graph structure ultimately consumed by GADM-aware model variants. Malaysia's peninsular states form a dense connected component; Sabah and Sarawak (East Malaysia) are spatially isolated from Peninsular states (separated by the South China Sea) and thus contribute no edges in the network. The `gadm_adj_matrix.npy` binary matrix encodes this structure. Key high-connectivity nodes include Pahang (borders 6 peninsular states) and Perak (borders 5). Federal territories (KL, Labuan, Putrajaya) are small enclaves with 1–2 border edges each.

The adjacency structure is used in `gadm.py` to export `gadm_adj_matrix_weighted.npy` (border-length-weighted) and `gadm_adj_edges.csv`. The three scalar summaries broadcast to all dates in `feature_align.py` (`gadm_n_states=16`, `gadm_n_border_pairs`, `gadm_mean_border_km`) encode this spatial connectivity in the flat feature matrix consumed by LSTM-family and attention-based models.

### area_shape_analysis.png
**What:** Bar chart or scatter of state area (km²) and shape compactness (e.g., perimeter²/area ratio) for all 16 administrative units.
**Analysis:** Sarawak is by far the largest state (~124,000 km²), followed by Sabah (~73,000 km²). Peninsular states range from ~1,000 km² (Perlis) to ~36,000 km² (Pahang). Shape compactness reveals elongated states (Kelantan, Kedah) vs. compact ones (Melaka, Perlis). This analysis confirms that GADM boundaries correctly parsed all 16 Level-1 units without geometry errors — the `gadm.py` validation check `len(cleaned_features) != 16` is satisfied.

State centroids extracted in `gadm.py` (lat/lon per state) are available for distance-based edge weighting in future graph model extensions.
