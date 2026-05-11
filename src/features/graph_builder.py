"""
graph_builder.py  — Graph Assembly for GCN-based ST Models
NEW FILE

Combines GTFS stop-level edges and GADM state-level borders into
adjacency matrices for GCN-SBULSTM, TGACN, and STGNN-STEP.

Two graph granularities are supported:
  (A) STOP-LEVEL graph  — one node per transit stop
      Used by: GCN-SBULSTM, TGACN (fine-grained spatial)
      Edges: consecutive stops on the same route (from GTFS stop_times)
      Edge weight: inverse travel time (faster connection = stronger edge)

  (B) STATE-LEVEL graph — one node per Malaysian state (16 nodes)
      Used by: STGNN-STEP, coarser spatial models
      Edges: shared border (from GADM adjacency matrix)
      Edge weight: shared border length in km

Outputs (under data/graph/):
  stop_nodes.csv               — unified stop node table (all operators)
  stop_adj_binary.npy          — N_stop × N_stop binary adjacency
  stop_adj_weighted.npy        — N_stop × N_stop travel-time-weighted
  stop_adj_edges.csv           — edge list for stop graph
  state_nodes.csv              — state node table (copy of GADM output)
  state_adj_binary.npy         — 16 × 16 binary adjacency
  state_adj_weighted.npy       — 16 × 16 border-length-weighted
  node_id_to_idx.json          — {stop_id: row_index} lookup
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

os.makedirs("data/graph", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# A. STOP-LEVEL GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_stop_graph(gtfs_dirs: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load node and edge tables from all cleaned GTFS directories,
    merge into a unified graph, and return:
      - df_nodes: unified stop node DataFrame
      - A_binary: N×N binary adjacency matrix
      - A_weighted: N×N travel-time-weighted adjacency matrix
    """
    print("=== Building Stop-Level Graph ===")

    # ── 1. Collect node tables ────────────────────────────────────────────────
    node_frames = []
    for gtfs_dir in gtfs_dirs:
        for path in glob.glob(os.path.join(gtfs_dir, "gtfs_stop_nodes_*.csv")):
            node_frames.append(pd.read_csv(path))

    if not node_frames:
        raise FileNotFoundError(
            "No gtfs_stop_nodes_*.csv files found. Run gtfs.py for each operator first."
        )

    df_nodes = pd.concat(node_frames, ignore_index=True)
    df_nodes = df_nodes.drop_duplicates(subset="stop_id").reset_index(drop=True)
    df_nodes["node_idx"] = df_nodes.index
    print(f"Total unique stops (nodes): {len(df_nodes)}")

    # Build lookup
    stop_to_idx = {sid: i for i, sid in enumerate(df_nodes["stop_id"])}

    # ── 2. Collect edge tables ────────────────────────────────────────────────
    edge_frames = []
    for gtfs_dir in gtfs_dirs:
        for path in glob.glob(os.path.join(gtfs_dir, "gtfs_stop_edges_*.csv")):
            edge_frames.append(pd.read_csv(path))

    if not edge_frames:
        raise FileNotFoundError(
            "No gtfs_stop_edges_*.csv files found. Run gtfs.py for each operator first."
        )

    df_edges = pd.concat(edge_frames, ignore_index=True)

    # Filter to stops that appear in the node table
    df_edges = df_edges[
        df_edges["from_stop_id"].isin(stop_to_idx) &
        df_edges["to_stop_id"].isin(stop_to_idx)
    ].copy()

    print(f"Total directed edges: {len(df_edges)}")

    # ── 3. Build adjacency matrices ───────────────────────────────────────────
    N = len(df_nodes)
    A_binary   = np.zeros((N, N), dtype=np.float32)
    A_weighted = np.zeros((N, N), dtype=np.float32)

    for _, row in df_edges.iterrows():
        i = stop_to_idx[row["from_stop_id"]]
        j = stop_to_idx[row["to_stop_id"]]
        t = row["travel_time_s"]

        A_binary[i, j] = 1.0
        A_binary[j, i] = 1.0   # undirected: model can use directed if needed

        # Edge weight: inverse travel time (capped at reasonable range)
        # Unresolvable travel times (-1) use a default of 120 s (2 min).
        if isinstance(t, (int, float)) and t > 0:
            w = 1.0 / t
        else:
            w = 1.0 / 120.0
        A_weighted[i, j] = max(A_weighted[i, j], w)
        A_weighted[j, i] = max(A_weighted[j, i], w)

    # ── 4. Normalise weighted adjacency (row-normalise) ───────────────────────
    # Row-normalised adjacency is standard in GCN message passing.
    # D^{-1} A  — each row sums to 1 (or 0 for isolated nodes).
    row_sums = A_weighted.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1   # avoid division by zero for isolated nodes
    A_weighted_norm = A_weighted / row_sums

    density = A_binary.sum() / (N * N)
    print(f"Adjacency density: {density:.4f}  ({int(A_binary.sum()//2)} undirected edges)")

    # ── 5. Export ─────────────────────────────────────────────────────────────
    df_nodes.to_csv("data/graph/stop_nodes.csv", index=False)
    df_edges.to_csv("data/graph/stop_adj_edges.csv", index=False)
    np.save("data/graph/stop_adj_binary.npy",      A_binary)
    np.save("data/graph/stop_adj_weighted.npy",    A_weighted_norm)

    node_id_to_idx = {row["stop_id"]: int(row["node_idx"]) for _, row in df_nodes.iterrows()}
    with open("data/graph/stop_node_id_to_idx.json", "w") as f:
        json.dump(node_id_to_idx, f, indent=2)

    print("Exported:")
    print("  data/graph/stop_nodes.csv")
    print("  data/graph/stop_adj_binary.npy        (N×N binary)")
    print("  data/graph/stop_adj_weighted.npy       (N×N row-normalised inverse-time)")
    print("  data/graph/stop_adj_edges.csv")
    print("  data/graph/stop_node_id_to_idx.json")

    return df_nodes, A_binary, A_weighted_norm


# ══════════════════════════════════════════════════════════════════════════════
# B. STATE-LEVEL GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_state_graph(
    nodes_path:    str = "data/cleaned/gadm_state_nodes.csv",
    adj_bin_path:  str = "data/cleaned/gadm_adj_matrix.npy",
    adj_wt_path:   str = "data/cleaned/gadm_adj_matrix_weighted.npy",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load the GADM-derived adjacency matrices and normalise the weighted version.
    Returns:
      - df_nodes: state node DataFrame
      - A_binary: 16×16 binary
      - A_weighted_norm: 16×16 row-normalised
    """
    print("\n=== Building State-Level Graph ===")

    df_nodes   = pd.read_csv(nodes_path)
    A_binary   = np.load(adj_bin_path)
    A_weighted = np.load(adj_wt_path)

    N = len(df_nodes)
    print(f"State nodes: {N}")

    # Row-normalise weighted adjacency
    row_sums = A_weighted.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A_weighted_norm = A_weighted / row_sums

    # Export (copies to graph/ directory for consistency)
    import shutil
    shutil.copy(nodes_path, "data/graph/state_nodes.csv")
    np.save("data/graph/state_adj_binary.npy",   A_binary)
    np.save("data/graph/state_adj_weighted.npy", A_weighted_norm)

    state_id_to_idx = {row["gid_1"]: int(row["node_idx"]) for _, row in df_nodes.iterrows()}
    with open("data/graph/state_node_id_to_idx.json", "w") as f:
        json.dump(state_id_to_idx, f, indent=2)

    print(f"Adjacency density: {A_binary.sum() / (N*N):.4f}")
    print("Exported:")
    print("  data/graph/state_nodes.csv")
    print("  data/graph/state_adj_binary.npy")
    print("  data/graph/state_adj_weighted.npy")
    print("  data/graph/state_node_id_to_idx.json")

    return df_nodes, A_binary, A_weighted_norm


# ══════════════════════════════════════════════════════════════════════════════
# C. Spectral normalisation helper (used by GCN / TGACN internally)
# ══════════════════════════════════════════════════════════════════════════════

def symmetric_normalise(A: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """
    Computes D^{-1/2} (A + I) D^{-1/2}  — standard GCN normalisation
    (Kipf & Welling 2017). Pass add_self_loops=False if your model
    adds them internally.
    """
    if add_self_loops:
        A = A + np.eye(A.shape[0], dtype=A.dtype)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1).clip(min=1e-9)))
    return D_inv_sqrt @ A @ D_inv_sqrt


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build graph adjacency matrices for ST models")
    parser.add_argument(
        "--gtfs-dirs", nargs="+",
        default=[
            "data/cleaned/gtfs_rapid_rail_kl",
            "data/cleaned/gtfs_rapid_bus_kl",
            "data/cleaned/gtfs_rapid_bus_penang",
            "data/cleaned/gtfs_ktmb",
        ],
        help="Paths to cleaned GTFS directories (must contain gtfs_stop_nodes_*.csv)"
    )
    parser.add_argument("--skip-stops",  action="store_true", help="Skip stop-level graph")
    parser.add_argument("--skip-states", action="store_true", help="Skip state-level graph")
    args = parser.parse_args()

    if not args.skip_stops:
        build_stop_graph(args.gtfs_dirs)

    if not args.skip_states:
        build_state_graph()

    print("\n✓ Graph artefacts ready under data/graph/")
    print("  Load in your model with:")
    print("    A = np.load('data/graph/stop_adj_binary.npy')")
    print("    A_hat = symmetric_normalise(A)  # for GCN layers")