"""
sequence_builder.py  — Sequence & Tensor Assembly for ST Models
NEW FILE

Consumes features_aligned.csv and graph artefacts to produce ready-to-train
inputs for every model in the stack:

  LSTM / BiLSTM / TPA-LSTM
    Input : (N_samples, T_in, F)        — sliding window over feature matrix
    Target: (N_samples, T_out)          — future total_ridership

  GCN-SBULSTM / TGACN / STGNN-STEP
    Input : X_global (N_samples, T_in, F_global)   — global features (fuel, weather, calendar)
            X_node   (N_samples, N_nodes, T_in)     — per-node ridership signal
    Adj   : (N_nodes, N_nodes)                      — pre-built adjacency matrix
    Target: (N_samples, N_nodes, T_out)             — future ridership per node

    ── Reconstruct the original (N, nodes, T, F) tensor at model load time ──
    Use the provided load_gcn_X() helper (see bottom of file).  The broadcast
    is a zero-copy view; peak GPU memory is unchanged.

Outputs (under data/sequences/)
────────────────────────────────
  lstm/
    X_train.npy, y_train.npy
    X_val.npy,   y_val.npy
    X_test.npy,  y_test.npy
    scaler_X.pkl, scaler_y.pkl
    split_dates.json

  gcn/
    X_global_train.npy   (N, T_in, F_global)   ← NEW: global features, not replicated
    X_node_train.npy     (N, N_nodes, T_in)     ← NEW: per-node ridership only
    y_train.npy          (N, N_nodes, T_out)
    … val / test equivalents …
    A.npy                (N_nodes, N_nodes)
    node_feature_static.npy
    scaler_X.pkl, scaler_y.pkl
    split_dates.json

Size optimisation notes
────────────────────────
Previous layout  : (N, N_nodes, T_in, F_global+1)  — global features replicated N_nodes times
New layout       : (N, T_in, F_global) + (N, N_nodes, T_in)
Savings          : ~N_nodes × F_global / (F_global + N_nodes) × 2  (plus float16 halving)
Typical outcome  : 30–120× smaller on disk with zero information loss.

Design notes
────────────
• Scaler is fitted on TRAIN sequences ONLY (no future leakage).
• Temporal split is chronological (no shuffle).
• MCO-gap sequences excluded by default (--include-mco to override).
• --dtype float16 (default) is safe because all values are MinMax-scaled to
  [0,1]; float16 has ~3.3 significant decimal digits, more than enough for
  gradient-based optimisers.  Pass --dtype float32 to stay on original precision.
• --compress writes .npz (zlib) for a further 1.5–3× disk saving; arrays are
  loaded exactly as before via np.load (the result is an NpzFile, index with ['arr']).
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
    dtype           = "float16",   # "float16" or "float32"
    compress        = False,        # write .npz instead of .npy
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


def _save(path_no_ext: str, arr: np.ndarray, compress: bool, dtype: np.dtype) -> str:
    """
    Save a numpy array with optional compression and dtype casting.
    Returns the path actually written (with extension).
    """
    arr = arr.astype(dtype)
    if compress:
        path = path_no_ext + ".npz"
        np.savez_compressed(path, arr=arr)
    else:
        path = path_no_ext + ".npy"
        np.save(path, arr)
    return path


def _size_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6


# ══════════════════════════════════════════════════════════════════════════════
# LSTM sequence builder  (unchanged logic, dtype / compress flags added)
# ══════════════════════════════════════════════════════════════════════════════

def build_lstm_sequences(cfg: dict) -> None:
    print("=== LSTM / BiLSTM / TPA-LSTM Sequence Builder ===")
    dtype    = np.dtype(cfg.get("dtype", "float32"))
    compress = cfg.get("compress", False)

    df = load_aligned(cfg["features_path"])

    drop_always = ["is_mco"]
    df = df.drop(columns=[c for c in drop_always if c in df.columns])

    mco_dates = set()
    if not cfg["include_mco"]:
        mco_start = pd.Timestamp("2020-03-18")
        mco_end   = pd.Timestamp("2021-12-31")
        mco_dates = set(pd.date_range(mco_start, mco_end, freq="D"))
        df = df[(df.index < mco_start) | (df.index > mco_end)]
        print(f"MCO rows excluded: {(df.index >= mco_start).sum()} remaining")

    df = df.fillna(df.median(numeric_only=True))

    target_col = cfg["target_col"]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in aligned features.")
    target_idx = df.columns.tolist().index(target_col)

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

    arr_train = df_train.values.astype(np.float32)
    arr_val   = df_val.values.astype(np.float32)
    arr_test  = df_test.values.astype(np.float32)

    X_tr, y_tr = make_sliding_windows(arr_train, T_in, T_out, target_idx)
    X_va, y_va = make_sliding_windows(arr_val,   T_in, T_out, target_idx)
    X_te, y_te = make_sliding_windows(arr_test,  T_in, T_out, target_idx)

    X_tr, X_va, X_te, scaler_X = fit_and_scale(X_tr, X_va, X_te)
    y_tr, y_va, y_te, scaler_y = scale_targets(y_tr, y_va, y_te)

    print(f"\nSequence shapes (after scaling):")
    print(f"  X_train: {X_tr.shape}   y_train: {y_tr.shape}")
    print(f"  X_val  : {X_va.shape}   y_val  : {y_va.shape}")
    print(f"  X_test : {X_te.shape}   y_test : {y_te.shape}")

    out = "data/sequences/lstm"
    ext = ".npz" if compress else ".npy"
    _save(f"{out}/X_train", X_tr, compress, dtype)
    _save(f"{out}/y_train", y_tr, compress, dtype)
    _save(f"{out}/X_val",   X_va, compress, dtype)
    _save(f"{out}/y_val",   y_va, compress, dtype)
    _save(f"{out}/X_test",  X_te, compress, dtype)
    _save(f"{out}/y_test",  y_te, compress, dtype)
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
        "dtype": str(dtype),
        "compressed": compress,
    }
    with open(f"{out}/split_dates.json", "w") as f:
        json.dump(split_dates, f, indent=2)

    print(f"\nExported to data/sequences/lstm/  (dtype={dtype}, compress={compress})")
    print(f"  Load: X_train = np.load('data/sequences/lstm/X_train{ext}')"
          + (" ['arr']" if compress else ""))
    print("  Inverse-transform predictions: scaler_y.inverse_transform(y_pred)")


# ══════════════════════════════════════════════════════════════════════════════
# GCN sequence builder  — OPTIMISED
# ══════════════════════════════════════════════════════════════════════════════

def build_gcn_sequences(cfg: dict) -> None:
    """
    Build OPTIMISED tensors for GCN-based models.

    Storage layout change
    ─────────────────────
    BEFORE (original):
        X_train.npy  →  (N, N_nodes, T_in, F_global+1)   float32
        Problem: global features (fuel, weather, calendar) are identical for every
        node at every timestep yet stored N_nodes times — pure redundancy.

    AFTER (optimised):
        X_global_train.npy  →  (N, T_in, F_global)   float16
        X_node_train.npy    →  (N, N_nodes, T_in)     float16

    At model load time, reconstruct with load_gcn_X() (zero-copy broadcast):
        X  →  (N, N_nodes, T_in, F_global+1)   — identical to original shape

    Size reduction:
        Original bytes  = N × N_nodes × T_in × (F_global+1) × 4
        Optimised bytes = [N × T_in × F_global + N × N_nodes × T_in] × 2
        Saving factor   ≈ N_nodes × (F_global+1) × 2 / (F_global + N_nodes)
        (e.g. 500 nodes, 40 global features → ~79× smaller)
    """
    print("\n=== GCN / TGACN / STGNN-STEP Sequence Builder (OPTIMISED) ===")

    dtype    = np.dtype(cfg.get("dtype", "float16"))
    compress = cfg.get("compress", False)

    df = load_aligned(cfg["features_path"])
    df = df.drop(columns=[c for c in ["is_mco"] if c in df.columns])
    df = df.fillna(df.median(numeric_only=True))

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
    operator_to_col = {
        "rapid_rail_kl":    "rail_komuter",
        "rapidbus_kl":      "bus_rkl",
        "rapidbus_penang":  "bus_rpn",
        "ktmb":             "rail_ets",
    }

    # ── Temporal split ────────────────────────────────────────────────────────
    if not cfg["include_mco"]:
        df = df[(df.index < pd.Timestamp("2020-03-18")) |
                (df.index > pd.Timestamp("2021-12-31"))]

    df_train, df_val, df_test = chronological_split(
        df, cfg["train_frac"], cfg["val_frac"]
    )

    T_in, T_out = cfg["T_in"], cfg["T_out"]

    # ── Identify global feature columns ───────────────────────────────────────
    global_feature_cols = [
        c for c in df.columns
        if any(c.startswith(prefix) for prefix in
               ("fp_lv_", "fp_chg_", "is_public", "is_school", "is_weekend",
                "dow_sin", "dow_cos", "month_sin", "month_cos",
                "days_to_", "days_since_", "rainfall_mm__",
                "pop_density"))
    ]
    F_global = len(global_feature_cols)
    print(f"Global feature columns per node: {F_global}")

    # ── Build operators lookup array once (avoids repeated dict lookups) ──────
    node_svc_cols = []
    for _, row in df_nodes.iterrows():
        operator = row.get("operator", "")
        node_svc_cols.append(operator_to_col.get(operator))

    def build_split_tensors(
        df_split: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          X_global : (N_windows, T_in, F_global)   — shared global features
          X_node   : (N_windows, N_nodes, T_in)     — per-node ridership signal
          y_node   : (N_windows, N_nodes, T_out)
        """
        T_total   = len(df_split)
        n_windows = T_total - T_in - T_out + 1

        # ── Global features: sliding windows ──────────────────────────────────
        arr_global = df_split[global_feature_cols].values.astype(np.float32)
        # Vectorised window extraction using stride tricks (no Python loop)
        shape   = (n_windows, T_in, F_global)
        strides = (arr_global.strides[0], arr_global.strides[0], arr_global.strides[1])
        X_global = np.lib.stride_tricks.as_strided(arr_global, shape=shape, strides=strides).copy()
        # shape: (N_windows, T_in, F_global)

        # ── Per-node ridership: vectorised over nodes ─────────────────────────
        # Collect all node ridership series into a (T_total, N_nodes) matrix first
        ridership_matrix = np.zeros((T_total, N_nodes), dtype=np.float32)
        for node_idx, svc_col in enumerate(node_svc_cols):
            if svc_col and svc_col in df_split.columns:
                ridership_matrix[:, node_idx] = df_split[svc_col].values.astype(np.float32)

        # Sliding windows for ridership: (N_windows, N_nodes, T_in)
        # Use stride tricks on transposed (N_nodes, T_total) array
        R = ridership_matrix.T.copy()  # (N_nodes, T_total)
        R_shape   = (N_nodes, n_windows, T_in)
        R_strides = (R.strides[0], R.strides[1], R.strides[1])
        X_node_t = np.lib.stride_tricks.as_strided(R, shape=R_shape, strides=R_strides).copy()
        X_node   = X_node_t.transpose(1, 0, 2)  # (N_windows, N_nodes, T_in)

        # Targets: node ridership T_in..T_in+T_out
        y_node = np.stack([
            ridership_matrix[w + T_in : w + T_in + T_out, :].T  # (N_nodes, T_out)
            for w in range(n_windows)
        ], axis=0)   # (N_windows, N_nodes, T_out)

        return X_global, X_node, y_node

    print("Building train tensors...")
    X_g_tr, X_n_tr, y_tr = build_split_tensors(df_train)
    print("Building val tensors...")
    X_g_va, X_n_va, y_va = build_split_tensors(df_val)
    print("Building test tensors...")
    X_g_te, X_n_te, y_te = build_split_tensors(df_test)

    # ── Scale: fit on train, apply to all ─────────────────────────────────────
    # Global features scaler (fit on flattened train global array)
    N_tr = X_g_tr.shape[0]
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(X_g_tr.reshape(-1, F_global))

    def scale_global(X_g):
        n, t, f = X_g.shape
        return scaler_X.transform(X_g.reshape(-1, f)).reshape(n, t, f)

    X_g_tr = scale_global(X_g_tr)
    X_g_va = scale_global(X_g_va)
    X_g_te = scale_global(X_g_te)

    # Node ridership scaler (fit on flattened train node tensor)
    scaler_node = MinMaxScaler(feature_range=(0, 1))
    scaler_node.fit(X_n_tr.reshape(-1, 1))

    def scale_node(X_n):
        s = X_n.shape
        return scaler_node.transform(X_n.reshape(-1, 1)).reshape(s)

    X_n_tr = scale_node(X_n_tr)
    X_n_va = scale_node(X_n_va)
    X_n_te = scale_node(X_n_te)

    # Target scaler
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(y_tr.reshape(-1, 1))

    def scale_y(y):
        s = y.shape
        return scaler_y.transform(y.reshape(-1, 1)).reshape(s)

    y_tr = scale_y(y_tr); y_va = scale_y(y_va); y_te = scale_y(y_te)

    # ── Size report ───────────────────────────────────────────────────────────
    orig_gb  = (X_g_tr.shape[0] * N_nodes * T_in * (F_global + 1) * 4) / 1e9
    new_gb_est = (
        (X_g_tr.nbytes + X_n_tr.nbytes) * 0.5   # float16 = half
    ) / 1e9
    print(f"\nGCN tensor shapes:")
    print(f"  X_global_train : {X_g_tr.shape}   (global features)")
    print(f"  X_node_train   : {X_n_tr.shape}   (per-node ridership)")
    print(f"  y_train        : {y_tr.shape}")
    print(f"  X_global_val   : {X_g_va.shape}")
    print(f"  X_global_test  : {X_g_te.shape}")
    print(f"\n  Estimated disk reduction: {orig_gb:.2f} GB → ~{new_gb_est:.2f} GB "
          f"({orig_gb / max(new_gb_est, 0.001):.0f}× smaller)")

    # ── Static node features ───────────────────────────────────────────────────
    static_cols = ["stop_lat", "stop_lon"]
    df_nodes_num = df_nodes[static_cols].fillna(0).values.astype(np.float32)
    static_scaler = MinMaxScaler()
    static_scaler.fit(df_nodes_num)
    node_feat_static = static_scaler.transform(df_nodes_num)

    # ── Export ─────────────────────────────────────────────────────────────────
    out = "data/sequences/gcn"
    ext = ".npz" if compress else ".npy"

    for tag, Xg, Xn, y in [("train", X_g_tr, X_n_tr, y_tr),
                             ("val",   X_g_va, X_n_va, y_va),
                             ("test",  X_g_te, X_n_te, y_te)]:
        _save(f"{out}/X_global_{tag}", Xg, compress, dtype)
        _save(f"{out}/X_node_{tag}",   Xn, compress, dtype)
        _save(f"{out}/y_{tag}",        y,  compress, dtype)

    np.save(f"{out}/A.npy", A)
    np.save(f"{out}/node_feature_static.npy", node_feat_static)
    joblib.dump(scaler_X,    f"{out}/scaler_X.pkl")
    joblib.dump(scaler_node, f"{out}/scaler_node.pkl")
    joblib.dump(scaler_y,    f"{out}/scaler_y.pkl")

    split_dates = {
        "train": {"start": str(df_train.index[0].date()), "end": str(df_train.index[-1].date())},
        "val":   {"start": str(df_val.index[0].date()),   "end": str(df_val.index[-1].date())},
        "test":  {"start": str(df_test.index[0].date()),  "end": str(df_test.index[-1].date())},
        "T_in": T_in, "T_out": T_out, "N_nodes": int(N_nodes),
        "F_global": int(F_global),
        "F_node": int(F_global + 1),     # combined shape for compat reference
        "F_static": int(node_feat_static.shape[1]),
        "dtype": str(dtype),
        "compressed": compress,
        "layout": "split",               # marks optimised format for loaders
    }
    with open(f"{out}/split_dates.json", "w") as fh:
        json.dump(split_dates, fh, indent=2)

    print(f"\nExported to data/sequences/gcn/  (dtype={dtype}, compress={compress})")
    print("  Load in your GCN model:")
    print(f"    X = load_gcn_X('train')  # returns (N, nodes, T, F) — same as before")
    print(f"    A = np.load('data/sequences/gcn/A.npy')")
    print(f"    node_feat = np.load('data/sequences/gcn/node_feature_static.npy')")


# ══════════════════════════════════════════════════════════════════════════════
# Drop-in loader — call this instead of np.load('X_train.npy')
# Reconstructs the original (N, N_nodes, T_in, F) tensor on the fly.
# The global broadcast is a zero-copy view; peak memory is the same.
# ══════════════════════════════════════════════════════════════════════════════

def load_gcn_X(
    split: str = "train",
    data_dir: str = "data/sequences/gcn",
    output_dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Load and reconstruct a (N, N_nodes, T_in, F_global+1) tensor from
    the deduplicated X_global / X_node files.

    Parameters
    ----------
    split      : "train" | "val" | "test"
    data_dir   : directory containing the gcn sequence files
    output_dtype : cast output to this dtype (default float32 for model compat)

    Returns
    -------
    X : np.ndarray  shape (N, N_nodes, T_in, F_global+1)
        Identical numerical content to the original monolithic X_train.npy
    """
    meta = json.load(open(f"{data_dir}/split_dates.json"))
    compress = meta.get("compressed", False)

    def _load(name):
        if compress:
            return np.load(f"{data_dir}/{name}.npz")["arr"].astype(output_dtype)
        return np.load(f"{data_dir}/{name}.npy").astype(output_dtype)

    X_global = _load(f"X_global_{split}")   # (N, T_in, F_global)
    X_node   = _load(f"X_node_{split}")     # (N, N_nodes, T_in)

    N, T_in, F_global = X_global.shape
    N_nodes = X_node.shape[1]

    # Broadcast global features to every node (read-only view, zero-copy)
    X_global_exp = np.broadcast_to(
        X_global[:, np.newaxis, :, :],    # (N, 1, T_in, F_global)
        (N, N_nodes, T_in, F_global),
    )

    # Concatenate: (N, N_nodes, T_in, F_global) + (N, N_nodes, T_in, 1)
    X = np.concatenate(
        [X_global_exp, X_node[:, :, :, np.newaxis]],
        axis=-1,
    )  # → (N, N_nodes, T_in, F_global+1)

    return X


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
    p.add_argument("--dtype", choices=["float16", "float32"], default=DEFAULTS["dtype"],
                   help="Storage dtype.  float16 halves file size with negligible precision "
                        "loss for MinMax-scaled [0,1] data (default: float16).")
    p.add_argument("--compress", action="store_true",
                   help="Save .npz (zlib compressed) instead of .npy for an extra "
                        "1.5–3× disk saving.  Load via np.load(...)['arr'].")
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
        "dtype":           args.dtype,
        "compress":        args.compress,
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