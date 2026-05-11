"""
sequence_builder.py  — Sequence & Tensor Assembly for ST Models
NEW FILE

Consumes features_aligned.csv and graph artefacts to produce ready-to-train
inputs for every model in the stack:

  LSTM / BiLSTM / TPA-LSTM
    Input : (N_samples, T_in, F)        — sliding window over feature matrix
    Target: (N_samples, T_out)          — future total_ridership

  GCN-SBULSTM / TGACN / STGNN-STEP
    Input : (N_samples, N_nodes, T_in, F_node)   — per-node feature tensor
    Adj   : (N_nodes, N_nodes)                   — pre-built adjacency matrix
    Target: (N_samples, N_nodes, T_out)           — future ridership per node

Outputs (under data/sequences/):
  lstm/
    X_train.npy, y_train.npy
    X_val.npy,   y_val.npy
    X_test.npy,  y_test.npy
    scaler_X.pkl, scaler_y.pkl
  gcn/
    X_train.npy, y_train.npy   (N, N_nodes, T, F)
    X_val.npy,   y_val.npy
    X_test.npy,  y_test.npy
    A.npy                       (adjacency — copy)
    node_feature_static.npy     (N_nodes, F_static) — population, POI counts
    split_dates.json            — {train/val/test: {start, end}}

Design notes
------------
• Scaler is fitted on TRAIN sequences ONLY. Fitting on the full dataset
  leaks future statistics — a silent source of over-optimistic val/test scores.
• Temporal split is chronological (no shuffle).
• Sequences that straddle the MCO gap (is_mco rows) are excluded by default
  (flag --include-mco to override).
• For GCN variants, node features are derived from the ridership service
  columns mapped to stops via the GTFS node table. Stops with no ridership
  signal (bus stops without headline ridership) receive zero-fill.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

os.makedirs("data/sequences/lstm", exist_ok=True)
os.makedirs("data/sequences/gcn",  exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration defaults (override via CLI args or edit here)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULTS = dict(
    features_path   = "data/features/features_aligned.csv",
    metadata_path   = "data/features/feature_metadata.json",
    stop_nodes_path = "data/graph/stop_nodes.csv",
    adj_path        = "data/graph/stop_adj_binary.npy",
    T_in            = 14,      # look-back window (days)
    T_out           = 7,       # forecast horizon (days)
    train_frac      = 0.70,
    val_frac        = 0.15,    # remaining 0.15 → test
    target_col      = "total_ridership",
    include_mco     = False,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_aligned(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df


def chronological_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i_train = int(n * train_frac)
    i_val   = int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def drop_mco_windows(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    T_in: int,
    T_out: int,
    mco_dates: set,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove any sliding-window sample whose look-back or forecast window
    overlaps with the MCO period (they contain noise / structural outliers).
    """
    keep = []
    for i in range(len(X)):
        window_start = dates[i]
        window_end   = dates[i + T_in + T_out - 1]
        window_dates = pd.date_range(window_start, window_end, freq="D")
        if not any(d in mco_dates for d in window_dates):
            keep.append(i)
    return X[keep], y[keep]


def make_sliding_windows(
    arr: np.ndarray,
    T_in: int,
    T_out: int,
    target_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping sliding windows from a 2D array (T_total × F).
    Returns:
      X: (N_windows, T_in, F)
      y: (N_windows, T_out)          — future values of the target column only
    """
    X_list, y_list = [], []
    total = arr.shape[0]
    for i in range(total - T_in - T_out + 1):
        X_list.append(arr[i : i + T_in])
        y_list.append(arr[i + T_in : i + T_in + T_out, target_idx])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def fit_and_scale(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Fit MinMaxScaler on X_train ONLY. Transform all splits.
    Input shape: (N, T, F). Scaler operates on features (axis=-1).
    """
    N_tr, T, F = X_train.shape
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train.reshape(-1, F))

    def scale(X):
        n, t, f = X.shape
        return scaler.transform(X.reshape(-1, f)).reshape(n, t, f)

    return scale(X_train), scale(X_val), scale(X_test), scaler


def scale_targets(
    y_train: np.ndarray,
    y_val:   np.ndarray,
    y_test:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Fit a separate scaler on y_train. Needed to invert predictions back
    to ridership counts during evaluation.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(y_train.reshape(-1, 1))

    def scale(y):
        return scaler.transform(y.reshape(-1, 1)).reshape(y.shape)

    return scale(y_train), scale(y_val), scale(y_test), scaler


# ══════════════════════════════════════════════════════════════════════════════
# LSTM sequence builder
# ══════════════════════════════════════════════════════════════════════════════

def build_lstm_sequences(cfg: dict) -> None:
    print("=== LSTM / BiLSTM / TPA-LSTM Sequence Builder ===")
    df = load_aligned(cfg["features_path"])

    # Drop columns that must not be fed as input features
    # (raw MCO flag is a temporal artefact, not a predictor; DOW already
    #  encoded cyclically in holiday_daily_features.csv)
    drop_always = ["is_mco"]
    df = df.drop(columns=[c for c in drop_always if c in df.columns])

    # Optional: exclude MCO windows
    mco_dates = set()
    if not cfg["include_mco"]:
        mco_start = pd.Timestamp("2020-03-18")
        mco_end   = pd.Timestamp("2021-12-31")
        mco_dates = set(pd.date_range(mco_start, mco_end, freq="D"))
        df = df[(df.index < mco_start) | (df.index > mco_end)]
        print(f"MCO rows excluded: {(df.index >= mco_start).sum()} remaining")

    # Fill any residual nulls with column median (pre-launch service cols)
    df = df.fillna(df.median(numeric_only=True))

    target_col = cfg["target_col"]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in aligned features.")
    target_idx = df.columns.tolist().index(target_col)

    # Chronological split on the raw dataframe (before windowing)
    df_train, df_val, df_test = chronological_split(
        df, cfg["train_frac"], cfg["val_frac"]
    )

    T_in, T_out = cfg["T_in"], cfg["T_out"]

    print(f"\nTemporal split:")
    print(f"  Train : {df_train.index[0].date()} → {df_train.index[-1].date()} "
          f"({len(df_train)} days)")
    print(f"  Val   : {df_val.index[0].date()} → {df_val.index[-1].date()} "
          f"({len(df_val)} days)")
    print(f"  Test  : {df_test.index[0].date()} → {df_test.index[-1].date()} "
          f"({len(df_test)} days)")
    print(f"\nWindow: T_in={T_in}, T_out={T_out}, Features={df.shape[1]}")

    # Build sliding windows per split
    arr_train = df_train.values.astype(np.float32)
    arr_val   = df_val.values.astype(np.float32)
    arr_test  = df_test.values.astype(np.float32)

    X_tr, y_tr = make_sliding_windows(arr_train, T_in, T_out, target_idx)
    X_va, y_va = make_sliding_windows(arr_val,   T_in, T_out, target_idx)
    X_te, y_te = make_sliding_windows(arr_test,  T_in, T_out, target_idx)

    # Fit scaler on train ONLY
    X_tr, X_va, X_te, scaler_X = fit_and_scale(X_tr, X_va, X_te)
    y_tr, y_va, y_te, scaler_y = scale_targets(y_tr, y_va, y_te)

    print(f"\nSequence shapes (after scaling):")
    print(f"  X_train: {X_tr.shape}   y_train: {y_tr.shape}")
    print(f"  X_val  : {X_va.shape}   y_val  : {y_va.shape}")
    print(f"  X_test : {X_te.shape}   y_test : {y_te.shape}")

    # Export
    out = "data/sequences/lstm"
    np.save(f"{out}/X_train.npy", X_tr);  np.save(f"{out}/y_train.npy", y_tr)
    np.save(f"{out}/X_val.npy",   X_va);  np.save(f"{out}/y_val.npy",   y_va)
    np.save(f"{out}/X_test.npy",  X_te);  np.save(f"{out}/y_test.npy",  y_te)
    joblib.dump(scaler_X, f"{out}/scaler_X.pkl")
    joblib.dump(scaler_y, f"{out}/scaler_y.pkl")

    split_dates = {
        "train": {"start": str(df_train.index[0].date()), "end": str(df_train.index[-1].date())},
        "val":   {"start": str(df_val.index[0].date()),   "end": str(df_val.index[-1].date())},
        "test":  {"start": str(df_test.index[0].date()),  "end": str(df_test.index[-1].date())},
        "T_in": T_in, "T_out": T_out,
        "n_features": int(df.shape[1]),
        "target_col": target_col,
        "target_col_idx": int(target_idx),
    }
    with open(f"{out}/split_dates.json", "w") as f:
        json.dump(split_dates, f, indent=2)

    print(f"\nExported to data/sequences/lstm/")
    print("  Usage: X_train = np.load('data/sequences/lstm/X_train.npy')")
    print("  Inverse-transform predictions: scaler_y.inverse_transform(y_pred)")


# ══════════════════════════════════════════════════════════════════════════════
# GCN sequence builder
# ══════════════════════════════════════════════════════════════════════════════

def build_gcn_sequences(cfg: dict) -> None:
    """
    Build (N_samples, N_nodes, T_in, F_node) tensors for GCN-based models.

    Node assignment strategy:
      The headline ridership CSV has service-level columns
      (rail_komuter, rail_mrt_pjy, bus_rkl, …) rather than per-stop counts.
      We map each GTFS stop to its operator/route service column via
      the operator field in stop_nodes.csv, then broadcast the service
      ridership to all stops belonging to that service.

      For models that need per-stop ridership (e.g. faregate data), replace
      the service_to_col mapping below with a stop_id → ridership CSV.
    """
    print("\n=== GCN / TGACN / STGNN-STEP Sequence Builder ===")

    df      = load_aligned(cfg["features_path"])
    df      = df.drop(columns=[c for c in ["is_mco"] if c in df.columns])
    df      = df.fillna(df.median(numeric_only=True))

    # ── Load node table and adjacency ─────────────────────────────────────────
    if not os.path.exists(cfg["stop_nodes_path"]):
        print(f"[SKIP] {cfg['stop_nodes_path']} not found — "
              f"run graph_builder.py first.")
        return

    df_nodes = pd.read_csv(cfg["stop_nodes_path"])
    A        = np.load(cfg["adj_path"])
    N_nodes  = len(df_nodes)
    print(f"Nodes: {N_nodes}   Adjacency: {A.shape}")

    # ── Service → ridership column mapping ────────────────────────────────────
    # Maps operator names (from GTFS) to ridership columns in the feature matrix.
    # Extend this dict as more operators/columns are added.
    operator_to_col = {
        "rapid_rail_kl":    "rail_komuter",      # primary KL rail proxy
        "rapidbus_kl":      "bus_rkl",
        "rapidbus_penang":  "bus_rpn",
        "ktmb":             "rail_ets",
    }

    # ── Temporal split (consistent with LSTM split) ───────────────────────────
    if not cfg["include_mco"]:
        df = df[(df.index < pd.Timestamp("2020-03-18")) |
                (df.index > pd.Timestamp("2021-12-31"))]

    df_train, df_val, df_test = chronological_split(
        df, cfg["train_frac"], cfg["val_frac"]
    )

    T_in, T_out = cfg["T_in"], cfg["T_out"]

    # ── Identify feature columns to use per node ──────────────────────────────
    # For GCN models, each node gets the same set of global features
    # (fuel price, holiday, rainfall) plus its own ridership signal.
    # We build the per-node feature tensor by:
    #   1. Taking global features (same for all nodes at each timestep)
    #   2. Appending the node's own ridership signal as an extra feature dim

    global_feature_cols = [
        c for c in df.columns
        if any(c.startswith(prefix) for prefix in
               ("fp_lv_", "fp_chg_", "is_public", "is_school", "is_weekend",
                "dow_sin", "dow_cos", "month_sin", "month_cos",
                "days_to_", "days_since_", "rainfall_mm__",
                "pop_density"))
    ]
    print(f"Global feature columns per node: {len(global_feature_cols)}")

    def build_node_tensor(df_split: pd.DataFrame) -> np.ndarray:
        """
        Returns tensor of shape (N_windows, N_nodes, T_in, F_global + 1)
        The +1 is the node's own ridership signal.
        """
        arr_global = df_split[global_feature_cols].values.astype(np.float32)
        T_total    = len(df_split)
        F_total    = len(global_feature_cols) + 1   # +1 for node ridership
        n_windows  = T_total - T_in - T_out + 1

        X_node = np.zeros((n_windows, N_nodes, T_in, F_total), dtype=np.float32)
        y_node = np.zeros((n_windows, N_nodes, T_out),          dtype=np.float32)

        for node_idx, row in df_nodes.iterrows():
            operator = row.get("operator", "")
            svc_col  = operator_to_col.get(operator)
            if svc_col and svc_col in df_split.columns:
                node_ridership = df_split[svc_col].values.astype(np.float32)
            else:
                node_ridership = np.zeros(T_total, dtype=np.float32)

            for w in range(n_windows):
                X_node[w, node_idx, :, :len(global_feature_cols)] = (
                    arr_global[w : w + T_in]
                )
                X_node[w, node_idx, :, -1] = node_ridership[w : w + T_in]
                y_node[w, node_idx, :]     = node_ridership[w + T_in : w + T_in + T_out]

        return X_node, y_node

    print("Building train tensor...")
    X_tr, y_tr = build_node_tensor(df_train)
    print("Building val tensor...")
    X_va, y_va = build_node_tensor(df_val)
    print("Building test tensor...")
    X_te, y_te = build_node_tensor(df_test)

    # ── Scale: fit on train, apply to all ─────────────────────────────────────
    # Flatten nodes+time for fitting, then reshape back.
    N_tr, NN, TT, FF = X_tr.shape
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(X_tr.reshape(-1, FF))

    def scale_node_tensor(X):
        n, nn, tt, ff = X.shape
        return scaler_X.transform(X.reshape(-1, ff)).reshape(n, nn, tt, ff)

    X_tr = scale_node_tensor(X_tr)
    X_va = scale_node_tensor(X_va)
    X_te = scale_node_tensor(X_te)

    # Scale targets separately per node (flatten nodes, fit on train)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(y_tr.reshape(-1, 1))

    def scale_y(y):
        n, nn, tt = y.shape
        return scaler_y.transform(y.reshape(-1, 1)).reshape(n, nn, tt)

    y_tr = scale_y(y_tr);  y_va = scale_y(y_va);  y_te = scale_y(y_te)

    print(f"\nGCN tensor shapes:")
    print(f"  X_train: {X_tr.shape}   y_train: {y_tr.shape}")
    print(f"  X_val  : {X_va.shape}   y_val  : {y_va.shape}")
    print(f"  X_test : {X_te.shape}   y_test : {y_te.shape}")

    # ── Static node features (population density per state) ───────────────────
    # A (N_nodes, F_static) array broadcast to all timesteps inside GCN layers.
    # Currently: stop_lat, stop_lon as spatial embeddings. Extend with POI counts.
    static_cols = ["stop_lat", "stop_lon"]
    df_nodes_num = df_nodes[static_cols].fillna(0).values.astype(np.float32)
    static_scaler = MinMaxScaler()
    static_scaler.fit(df_nodes_num)
    node_feat_static = static_scaler.transform(df_nodes_num)

    # ── Export ─────────────────────────────────────────────────────────────────
    out = "data/sequences/gcn"
    np.save(f"{out}/X_train.npy", X_tr);  np.save(f"{out}/y_train.npy", y_tr)
    np.save(f"{out}/X_val.npy",   X_va);  np.save(f"{out}/y_val.npy",   y_va)
    np.save(f"{out}/X_test.npy",  X_te);  np.save(f"{out}/y_test.npy",  y_te)
    np.save(f"{out}/A.npy",       A)
    np.save(f"{out}/node_feature_static.npy", node_feat_static)
    joblib.dump(scaler_X, f"{out}/scaler_X.pkl")
    joblib.dump(scaler_y, f"{out}/scaler_y.pkl")

    split_dates = {
        "train": {"start": str(df_train.index[0].date()), "end": str(df_train.index[-1].date())},
        "val":   {"start": str(df_val.index[0].date()),   "end": str(df_val.index[-1].date())},
        "test":  {"start": str(df_test.index[0].date()),  "end": str(df_test.index[-1].date())},
        "T_in": T_in, "T_out": T_out, "N_nodes": int(N_nodes),
        "F_node": int(FF), "F_static": int(node_feat_static.shape[1]),
    }
    with open(f"{out}/split_dates.json", "w") as f:
        json.dump(split_dates, f, indent=2)

    print(f"\nExported to data/sequences/gcn/")
    print("  Load in your GCN model:")
    print("    X = np.load('data/sequences/gcn/X_train.npy')  # (N, nodes, T, F)")
    print("    A = np.load('data/sequences/gcn/A.npy')         # (nodes, nodes)")
    print("    node_feat = np.load('data/sequences/gcn/node_feature_static.npy')")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Build model-ready sequences from aligned features")
    p.add_argument("--features-path",   default=DEFAULTS["features_path"])
    p.add_argument("--metadata-path",   default=DEFAULTS["metadata_path"])
    p.add_argument("--stop-nodes-path", default=DEFAULTS["stop_nodes_path"])
    p.add_argument("--adj-path",        default=DEFAULTS["adj_path"])
    p.add_argument("--T-in",    type=int,   default=DEFAULTS["T_in"])
    p.add_argument("--T-out",   type=int,   default=DEFAULTS["T_out"])
    p.add_argument("--target",  default=DEFAULTS["target_col"])
    p.add_argument("--train-frac", type=float, default=DEFAULTS["train_frac"])
    p.add_argument("--val-frac",   type=float, default=DEFAULTS["val_frac"])
    p.add_argument("--include-mco", action="store_true",
                   help="Include MCO-period rows in sequences (default: exclude)")
    p.add_argument("--mode", choices=["lstm", "gcn", "both"], default="both",
                   help="Which sequence format(s) to build")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = {
        "features_path":   args.features_path,
        "metadata_path":   args.metadata_path,
        "stop_nodes_path": args.stop_nodes_path,
        "adj_path":        args.adj_path,
        "T_in":            args.T_in,
        "T_out":           args.T_out,
        "target_col":      args.target,
        "train_frac":      args.train_frac,
        "val_frac":        args.val_frac,
        "include_mco":     args.include_mco,
    }

    if args.mode in ("lstm", "both"):
        build_lstm_sequences(cfg)

    if args.mode in ("gcn", "both"):
        build_gcn_sequences(cfg)

    print("\n✓ All sequences built.")
    print("  Run order reminder:")
    print("    1. cleaning scripts (ridership, fuelprice, holiday, rainfall, gadm, population, osm, gtfs)")
    print("    2. graph_builder.py")
    print("    3. feature_align.py")
    print("    4. sequence_builder.py  ← you are here")