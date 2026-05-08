"""
graph_construction.py
---------------------
Builds PyTorch Geometric graphs for spatial-temporal models:
  - GADM Graph  : nodes = admin zones, edges = shared boundaries
  - GTFS Graph  : nodes = stops, edges = routes weighted by walking friction + distance
  - Node features: POI density, population, GTFS service metrics
  - Edge features: walking friction impedance

Outputs:
  graph_adjacency.pt   dict of {graph_name: torch_geometric.data.Data}
  node_features.pt     dict of {graph_name: torch.Tensor}
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GADM_PATH        = Path("data/cleaned/gadm_mys_l1_clean.geojson")
GTFS_DIRS        = [
    Path("data/cleaned/gtfs_ktmb"),
    Path("data/cleaned/gtfs_rapid_bus_penang"),
    Path("data/cleaned/gtfs_rapid_rail_kl"),
]
SPATIAL_FEAT_PATH   = Path("data/features/spatial_features.parquet")
FRICTION_FEAT_PATH  = Path("data/features/walking_friction_weights.parquet")
RIDERSHIP_FEAT_PATH = Path("data/features/ridership_temporal.parquet")

OUT_ADJACENCY  = Path("data/features/graph_adjacency.pt")
OUT_NODE_FEATS = Path("data/features/node_features.pt")

GADM_ID_COL = "GID_1"


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _to_tensor(df: pd.DataFrame, fill: float = 0.0) -> torch.Tensor:
    arr = df.select_dtypes(include=[np.number]).fillna(fill).values.astype(np.float32)
    return torch.from_numpy(arr)


def _row_normalise(t: torch.Tensor) -> torch.Tensor:
    col_min = t.min(dim=0).values
    col_max = t.max(dim=0).values
    denom = (col_max - col_min).clamp(min=1e-9)
    return (t - col_min) / denom


# ---------------------------------------------------------------------------
# 1. GADM Graph
# ---------------------------------------------------------------------------

def build_gadm_graph(
    gadm_path: Path,
    spatial_feat_path: Path,
) -> Data:
    print("[graph] Building GADM graph ...")
    gadm = gpd.read_file(gadm_path).to_crs("EPSG:4326")
    gadm = gadm.reset_index(drop=True)
    zone_ids = gadm[GADM_ID_COL].tolist()
    n = len(zone_ids)

    # --- Adjacency: shared boundary (touches or intersects boundary) ---------
    edges_src, edges_dst = [], []
    geoms = gadm.geometry.tolist()
    for i in range(n):
        for j in range(i + 1, n):
            if geoms[i].touches(geoms[j]) or geoms[i].intersects(geoms[j].boundary):
                edges_src += [i, j]
                edges_dst += [j, i]

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    # --- Node features -------------------------------------------------------
    if spatial_feat_path.exists():
        sp = pd.read_parquet(spatial_feat_path)
        # Align to gadm ordering
        sp = sp.reindex(zone_ids)
        x = _row_normalise(_to_tensor(sp))
    else:
        print("[graph] spatial_features.parquet not found; using identity node features.")
        x = torch.eye(n, dtype=torch.float)

    # --- Edge weights: uniform (no friction at zone-zone boundary level) -----
    edge_attr = torch.ones(edge_index.shape[1], 1)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n,
        zone_ids=zone_ids,
    )
    print(f"[graph] GADM graph: {n} nodes, {edge_index.shape[1]//2} undirected edges")
    return data


# ---------------------------------------------------------------------------
# 2. GTFS Graph
# ---------------------------------------------------------------------------

def _load_gtfs_stops(gtfs_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for d in gtfs_dirs:
        f = d / "stops.txt"
        if f.exists():
            s = pd.read_csv(f)
            s["source"] = d.name
            frames.append(s)
    if not frames:
        return pd.DataFrame(columns=["stop_id", "stop_lat", "stop_lon", "source"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["stop_id"])


def _load_gtfs_stop_times(gtfs_dirs: list[Path]) -> pd.DataFrame:
    """Load stop_times to infer route edges between consecutive stops."""
    frames = []
    for d in gtfs_dirs:
        f = d / "stop_times.txt"
        if f.exists():
            st = pd.read_csv(f, usecols=["trip_id", "stop_id", "stop_sequence"])
            st["source"] = d.name
            frames.append(st)
    if not frames:
        return pd.DataFrame(columns=["trip_id", "stop_id", "stop_sequence", "source"])
    return pd.concat(frames, ignore_index=True)


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def build_gtfs_graph(
    gtfs_dirs: list[Path],
    friction_feat_path: Path,
    spatial_feat_path: Path,
) -> Data:
    print("[graph] Building GTFS graph ...")

    stops = _load_gtfs_stops(gtfs_dirs)
    if stops.empty:
        raise FileNotFoundError("No GTFS stops.txt found in any of the provided directories.")

    stop_ids = stops["stop_id"].tolist()
    n = len(stop_ids)
    sid2idx = {sid: i for i, sid in enumerate(stop_ids)}

    # --- Build edges from stop_times -----------------------------------------
    stop_times = _load_gtfs_stop_times(gtfs_dirs)

    if stop_times.empty:
        print("[graph] stop_times.txt not found; no route edges will be added.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3))
    else:
        stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])
        stop_times["next_stop_id"] = stop_times.groupby("trip_id")["stop_id"].shift(-1)
        edge_df = stop_times.dropna(subset=["next_stop_id"])[["stop_id", "next_stop_id"]].copy()
        edge_df = edge_df[edge_df["stop_id"] != edge_df["next_stop_id"]]
        edge_df["src"] = edge_df["stop_id"].map(sid2idx)
        edge_df["dst"] = edge_df["next_stop_id"].map(sid2idx)
        edge_df = edge_df.dropna(subset=["src", "dst"])
        edge_df = edge_df[["src", "dst"]].astype(int)
        edge_rev = edge_df.rename(columns={"src": "dst", "dst": "src"})
        edge_all = pd.concat([edge_df, edge_rev]).drop_duplicates()

        src_idx = edge_all["src"].values
        dst_idx = edge_all["dst"].values

        # Vectorized haversine on numpy arrays
        stop_coord = stops.set_index("stop_id")[["stop_lat", "stop_lon"]]
        lat_arr = stop_coord.loc[stop_ids, "stop_lat"].values
        lon_arr = stop_coord.loc[stop_ids, "stop_lon"].values
        dists = _haversine_m(lat_arr[src_idx], lon_arr[src_idx],
                             lat_arr[dst_idx], lon_arr[dst_idx])
        edge_index = torch.from_numpy(np.vstack([src_idx, dst_idx])).long()

        # --- Edge features: friction + distance ----------------------------------
        friction_lookup = {}
        if friction_feat_path.exists():
            fr = pd.read_parquet(friction_feat_path)
            gtfs_fr = fr[fr.get("entity_type", pd.Series(dtype=str)) == "gtfs_stop"]
            if "passability_weight" in gtfs_fr.columns:
                friction_lookup = gtfs_fr["passability_weight"].to_dict()

        dist_arr = dists.astype(np.float32)
        dist_norm = dist_arr / (dist_arr.max() + 1e-9)

        src_ids = [stop_ids[i] for i in src_idx]
        dst_ids = [stop_ids[i] for i in dst_idx]
        pass_src = np.array([friction_lookup.get(sid, 0.5) for sid in src_ids], dtype=np.float32)
        pass_dst = np.array([friction_lookup.get(sid, 0.5) for sid in dst_ids], dtype=np.float32)
        pass_mean = (pass_src + pass_dst) / 2.0

        # Combined weight: lower distance + higher passability -> stronger edge
        combined = pass_mean * (1.0 - dist_norm)
        edge_attr = torch.from_numpy(
            np.stack([dist_norm, pass_mean, combined], axis=1)
        ).float()

    # --- Node features -------------------------------------------------------
    node_feat_cols = ["poi_density_per_km2", "poi_diversity_shannon",
                      "pop_log_density", "gtfs_stop_density_per_km2"]

    if spatial_feat_path.exists():
        sp = pd.read_parquet(spatial_feat_path)
        # Spatial join: assign each stop its GADM zone features
        stops_gdf = gpd.GeoDataFrame(
            stops, geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
            crs="EPSG:4326",
        )
        gadm_tmp = gpd.read_file(GADM_PATH)[["GID_1", "geometry"]].to_crs("EPSG:4326")
        joined = gpd.sjoin(stops_gdf, gadm_tmp, how="left", predicate="within")
        stops["zone_id"] = joined["GID_1"].values

        avail_cols = [c for c in node_feat_cols if c in sp.columns]
        feat_df = sp[avail_cols].reindex(stops["zone_id"].values).fillna(0).reset_index(drop=True)
        x = _row_normalise(_to_tensor(pd.DataFrame(feat_df)))
    else:
        print("[graph] spatial_features.parquet not found; using degree as node feature.")
        deg = torch.zeros(n)
        if edge_index.numel() > 0:
            for i in range(edge_index.shape[1]):
                deg[edge_index[0, i]] += 1
        x = deg.unsqueeze(1)

    # Append friction as a node-level feature
    frict_node = torch.tensor(
        [friction_lookup.get(sid, 0.5) for sid in stop_ids], dtype=torch.float
    ).unsqueeze(1)
    x = torch.cat([x, frict_node], dim=1)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n,
        stop_ids=stop_ids,
    )
    print(f"[graph] GTFS graph: {n} nodes, {edge_index.shape[1]//2 if edge_index.numel() else 0} undirected edges")
    return data


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def run(
    gadm_path: Path        = GADM_PATH,
    gtfs_dirs: list[Path]  = GTFS_DIRS,
    spatial_feat_path: Path = SPATIAL_FEAT_PATH,
    friction_feat_path: Path = FRICTION_FEAT_PATH,
    out_adjacency: Path    = OUT_ADJACENCY,
    out_node_feats: Path   = OUT_NODE_FEATS,
) -> dict:
    out_adjacency.parent.mkdir(parents=True, exist_ok=True)

    graphs = {}

    # GADM graph
    try:
        graphs["gadm"] = build_gadm_graph(gadm_path, spatial_feat_path)
    except Exception as e:
        print(f"[graph] GADM graph failed: {e}")

    # GTFS graph
    try:
        graphs["gtfs"] = build_gtfs_graph(gtfs_dirs, friction_feat_path, spatial_feat_path)
    except Exception as e:
        print(f"[graph] GTFS graph failed: {e}")

    # Persist
    torch.save(graphs, out_adjacency)
    print(f"[graph] Saved graph_adjacency.pt -> {out_adjacency}")

    node_feats = {name: g.x for name, g in graphs.items()}
    torch.save(node_feats, out_node_feats)
    print(f"[graph] Saved node_features.pt -> {out_node_feats}")

    # Validation
    _validate(graphs)

    return graphs


def _validate(graphs: dict):
    print("\n[graph] Validation")
    for name, g in graphs.items():
        ei = g.edge_index
        # Symmetry check (undirected)
        if ei.numel() == 0:
            print(f"  [{name}] nodes={g.num_nodes}, edges=0, symmetric=True, nodes_with_edges=0, node_feat_dim={list(g.x.shape)}")
            continue
        fwd = set(zip(ei[0].tolist(), ei[1].tolist()))
        bwd = set(zip(ei[1].tolist(), ei[0].tolist()))
        symmetric = fwd == bwd
        # Connectivity: number of unique nodes that appear in edges
        connected_nodes = len(torch.unique(ei))
        print(
            f"  [{name}] nodes={g.num_nodes}, edges={ei.shape[1]//2}, "
            f"symmetric={symmetric}, nodes_with_edges={connected_nodes}, "
            f"node_feat_dim={list(g.x.shape)}"
        )


if __name__ == "__main__":
    run()
