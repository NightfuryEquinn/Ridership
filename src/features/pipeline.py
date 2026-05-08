"""
pipeline.py
-----------
Orchestrates the full feature engineering pipeline:
  1. Temporal features  : ridership, fuel, holiday, rainfall
  2. Spatial features   : POIs, population, GTFS stop density
  3. Walking friction   : edge weights for graph models
  4. Graph construction : GADM + GTFS PyTorch Geometric graphs
  5. Fusion             : merge all features into model-ready outputs

Outputs
-------
  data/features/feature_matrix_lstm.parquet   -> LSTM / BiLSTM / TPA-LSTM
  data/features/graph_adjacency.pt            -> TGACN / GCN-SBULSTM / SGCNN-STEP
  data/features/node_features.pt              -> TGACN / GCN-SBULSTM / SGCNN-STEP
"""

import logging
import traceback
import warnings
from pathlib import Path
from typing import Optional
import torch_geometric.data

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Canonical output paths
# ---------------------------------------------------------------------------
FEATURES_DIR = Path("data/features")

OUT_LSTM    = FEATURES_DIR / "feature_matrix_lstm.parquet"
OUT_GRAPH   = FEATURES_DIR / "graph_adjacency.pt"
OUT_NODES   = FEATURES_DIR / "node_features.pt"
OUT_REPORT  = FEATURES_DIR / "pipeline_report.txt"

# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEPS = [
    "ridership",
    "fuel",
    "holiday",
    "rainfall",
    "spatial",
    "walking_friction",
    "graph_construction",
    "fusion",
]


# ---------------------------------------------------------------------------
# Individual step runners  (import lazily so partial runs still work)
# ---------------------------------------------------------------------------

def _run_step(name: str, results: dict) -> Optional[pd.DataFrame]:
    if name == "ridership":
        from .ridership_features import run
        return run()

    if name == "fuel":
        from .fuel_features import run
        return run()

    if name == "holiday":
        from .holiday_features import run
        return run()

    if name == "rainfall":
        from .rainfall_features import run
        return run()

    if name == "spatial":
        from .spatial_features import run
        return run()

    if name == "walking_friction":
        from .walking_friction import run
        return run()

    if name == "graph_construction":
        from .graph_construction import run
        return run()

    if name == "fusion":
        return _fuse(results)

    raise ValueError(f"Unknown step: {name}")


# ---------------------------------------------------------------------------
# Fusion logic
# ---------------------------------------------------------------------------

def _load_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_parquet(path)
    log.warning("Feature file not found, skipping: %s", path)
    return None


def _fuse(results: dict) -> pd.DataFrame:
    """
    Merge all temporal feature tables on a shared daily DatetimeIndex.
    Produces feature_matrix_lstm.parquet.
    """
    log.info("[fusion] Merging temporal feature tables ...")

    temporal_paths = {
        "ridership": FEATURES_DIR / "ridership_temporal.parquet",
        "fuel"     : FEATURES_DIR / "fuel_temporal.parquet",
        "holiday"  : FEATURES_DIR / "holiday_flags.parquet",
        "rainfall" : FEATURES_DIR / "rainfall_temporal.parquet",
    }

    frames = {}
    for key, path in temporal_paths.items():
        df = _load_if_exists(path)
        if df is not None:
            frames[key] = df

    if not frames:
        raise RuntimeError("No temporal feature files found. Run earlier pipeline steps first.")

    # ---- Align on a common daily DatetimeIndex ------------------------------
    all_indices = [f.index for f in frames.values()]
    start = max(idx.min() for idx in all_indices)
    end   = min(idx.max() for idx in all_indices)
    log.info("[fusion] Common date range: %s -> %s", start.date(), end.date())

    common_idx = pd.date_range(start, end, freq="D")

    aligned = {}
    for key, df in frames.items():
        df_num = df.select_dtypes(include=[np.number])
        df_num = df_num.reindex(common_idx)        # introduce NaN for missing days
        df_num = df_num.ffill(limit=3).bfill(limit=3)
        # Prefix columns to avoid collisions
        df_num.columns = [f"{key}__{c}" for c in df_num.columns]
        aligned[key] = df_num

    fused = pd.concat(aligned.values(), axis=1)

    # ---- Append static spatial features (broadcast over time) ---------------
    spatial_path = FEATURES_DIR / "spatial_features.parquet"
    sp = _load_if_exists(spatial_path)
    if sp is not None:
        # Spatial is zone-level; attach zone-level aggregates as scalar columns
        numeric_sp = sp.select_dtypes(include=[np.number])
        zone_agg = {
            f"spatial__{col}__mean" : numeric_sp[col].mean() for col in numeric_sp.columns
        }
        zone_agg.update({
            f"spatial__{col}__max"  : numeric_sp[col].max() for col in numeric_sp.columns
        })
        for col_name, val in zone_agg.items():
            fused[col_name] = val
        log.info("[fusion] Appended %d spatial aggregate columns.", len(zone_agg))

    # ---- Drop columns that are all-NaN --------------------------------------
    before = fused.shape[1]
    fused = fused.dropna(axis=1, how="all")
    fused = fused.dropna(axis=0, how="all")
    log.info("[fusion] Dropped %d all-NaN columns. Final shape: %s", before - fused.shape[1], fused.shape)

    # ---- Final dtype cleanup ------------------------------------------------
    fused = fused.astype(np.float32)
    fused.index.name = "date"

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    fused.to_parquet(OUT_LSTM)
    log.info("[fusion] Saved LSTM feature matrix -> %s  shape=%s", OUT_LSTM, fused.shape)
    return fused


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_lstm_matrix(df: pd.DataFrame) -> list[str]:
    issues = []
    nan_pct = df.isna().mean()
    high_nan = nan_pct[nan_pct > 0.10]
    if not high_nan.empty:
        issues.append(f"{len(high_nan)} columns have >10% NaN: {high_nan.index.tolist()[:5]} ...")
    if not df.index.is_monotonic_increasing:
        issues.append("Date index is not monotonically increasing.")
    return issues


def _validate_graphs(graph_path: Path) -> list[str]:
    issues = []
    if not graph_path.exists():
        issues.append(f"graph_adjacency.pt not found at {graph_path}")
        return issues
    graphs = torch.load(graph_path, weights_only=False)
    for name, g in graphs.items():
        ei = g.edge_index
        if ei.numel() == 0:
            issues.append(f"{name}: edge_index is empty.")
            continue
        # Symmetry
        fwd = set(map(tuple, ei.T.tolist()))
        bwd = set((b, a) for a, b in fwd)
        if fwd != bwd:
            issues.append(f"{name}: adjacency matrix is not symmetric.")
        # Self-loops
        self_loops = (ei[0] == ei[1]).sum().item()
        if self_loops > 0:
            issues.append(f"{name}: {self_loops} self-loop(s) detected.")
    return issues


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(step_status: dict, lstm_df: Optional[pd.DataFrame], graph_path: Path):
    lines = ["=" * 60, "FEATURE ENGINEERING PIPELINE REPORT", "=" * 60, ""]

    lines.append("STEP STATUS")
    lines.append("-" * 40)
    for step, status in step_status.items():
        lines.append(f"  {step:<25} {status}")
    lines.append("")

    if lstm_df is not None:
        lines.append("LSTM FEATURE MATRIX")
        lines.append("-" * 40)
        lines.append(f"  Shape            : {lstm_df.shape}")
        lines.append(f"  Date range       : {lstm_df.index.min().date()} -> {lstm_df.index.max().date()}")
        lines.append(f"  NaN %            : {lstm_df.isna().mean().mean()*100:.2f}%")
        lines.append(f"  Feature groups   :")
        for prefix in ["ridership", "fuel", "holiday", "rainfall", "spatial"]:
            cols = [c for c in lstm_df.columns if c.startswith(prefix)]
            lines.append(f"    {prefix:<20} {len(cols)} columns")
        lines.append("")

        issues = _validate_lstm_matrix(lstm_df)
        if issues:
            lines.append("  WARNINGS:")
            for iss in issues:
                lines.append(f"    [!] {iss}")
        else:
            lines.append("  [OK] No validation issues.")
        lines.append("")

    lines.append("GRAPH VALIDATION")
    lines.append("-" * 40)
    graph_issues = _validate_graphs(graph_path)
    if graph_issues:
        for iss in graph_issues:
            lines.append(f"  [!] {iss}")
    else:
            lines.append("  [OK] No graph validation issues.")

    if graph_path.exists():
        graphs = torch.load(graph_path, weights_only=False)
        for name, g in graphs.items():
            n_edges = g.edge_index.shape[1] // 2 if g.edge_index.numel() > 0 else 0
            lines.append(f"  [{name}] nodes={g.num_nodes}, undirected_edges={n_edges}, node_feat_dim={list(g.x.shape)}")

    lines += ["", "=" * 60]
    report = "\n".join(lines)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(report)
    print("\n" + report)
    log.info("Report saved -> %s", OUT_REPORT)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(steps: list[str] = STEPS, stop_on_error: bool = False) -> dict:
    """
    Run the full pipeline (or a subset of steps).

    Parameters
    ----------
    steps        : ordered list of step names to execute.
    stop_on_error: if True, abort the run on first failure.

    Returns
    -------
    dict mapping step name -> result object (DataFrame, dict, or None on failure).
    """
    results     : dict = {}
    step_status : dict = {}
    lstm_df     : Optional[pd.DataFrame] = None

    for step in steps:
        log.info("--- Running step: %s ---", step)
        try:
            result = _run_step(step, results)
            results[step] = result
            step_status[step] = "[OK] OK"
            if step == "fusion" and isinstance(result, pd.DataFrame):
                lstm_df = result
        except Exception as exc:
            step_status[step] = f"[X] FAILED: {exc}"
            log.error("Step '%s' failed: %s", step, exc)
            log.debug(traceback.format_exc())
            if stop_on_error:
                break

    _write_report(step_status, lstm_df, OUT_GRAPH)
    return results


# ---------------------------------------------------------------------------
# Model-specific dataset helpers
# ---------------------------------------------------------------------------

def load_lstm_dataset(
    path: Path = OUT_LSTM,
    seq_len: int = 30,
    horizon: int = 7,
    target_prefix: str = "ridership__",
) -> tuple:
    """
    Load feature_matrix_lstm.parquet and return (X, y) tensors shaped for
    LSTM / BiLSTM / TPA-LSTM:  X -> [N, seq_len, features],  y -> [N, horizon].

    Parameters
    ----------
    seq_len       : look-back window length (timesteps).
    horizon       : forecast horizon (timesteps ahead).
    target_prefix : column prefix used to identify target variable(s).
    """
    df = pd.read_parquet(path).astype(np.float32)
    target_cols  = [c for c in df.columns if c.startswith(target_prefix)]
    feature_cols = [c for c in df.columns if c not in target_cols]

    X_all = df[feature_cols].values
    y_all = df[target_cols].values

    Xs, ys = [], []
    for i in range(len(df) - seq_len - horizon + 1):
        Xs.append(X_all[i : i + seq_len])
        ys.append(y_all[i + seq_len : i + seq_len + horizon])

    X = torch.from_numpy(np.stack(Xs))   # [N, seq_len, features]
    y = torch.from_numpy(np.stack(ys))   # [N, horizon, targets]

    log.info("LSTM dataset: X=%s  y=%s", list(X.shape), list(y.shape))
    return X, y


def load_graph_dataset(
    graph_path: Path   = OUT_GRAPH,
    node_path: Path    = OUT_NODES,
    graph_name: str    = "gtfs",
) -> "torch_geometric.data.Data":
    """
    Load and return the PyTorch Geometric Data object for a named graph.
    Compatible with TGACN, GCN-SBULSTM, SGCNN-STEP.
    """
    graphs     = torch.load(graph_path)
    node_feats = torch.load(node_path)

    if graph_name not in graphs:
        raise KeyError(f"Graph '{graph_name}' not found. Available: {list(graphs.keys())}")

    data = graphs[graph_name]
    data.x = node_feats[graph_name]
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument(
        "--steps", nargs="+", default=STEPS,
        help=f"Steps to run. Default: all. Options: {STEPS}",
    )
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="Abort pipeline on first step failure.",
    )
    args = parser.parse_args()

    run(steps=args.steps, stop_on_error=args.stop_on_error)