"""
sequence_builder.py  — Sequence & Tensor Assembly for All Models

Consumes features_aligned.csv (all 8 spatio-temporal feature sources) and
produces ready-to-train sliding-window sequences for all 14 models in the stack.

All models load from data/sequences/lstm/:
  LSTM-family      : LSTM, BiLSTM, TPA-LSTM, CNN-LSTM, CNN-BiLSTM, ST-LSTM
  Graph-based      : STGCN, MTGNN, STSGCN, STFGNN, PDR-STGCN
  Attention-based  : TPA-LSTM, ASTGCN, Autoformer, Informer

Graph models build their adjacency matrices on-the-fly from the training
data (Pearson correlation, threshold 0.1) — no external graph files needed.

Output (data/sequences/lstm/)
──────────────────────────────
  X_train.npy, y_train.npy
  X_val.npy,   y_val.npy
  X_test.npy,  y_test.npy
  scaler_X.pkl, scaler_y.pkl
  split_dates.json

Tensor shapes
─────────────
  X : (N_samples, T_in=14, F=N_features)   — N_features from features_aligned.csv
  y : (N_samples, T_out=7)

  Actual N_features is logged at runtime and stored in split_dates.json.

Design notes
────────────
• Scaler is fitted on TRAIN sequences ONLY (no future leakage).
• Temporal split is chronological (no shuffle): 70 / 15 / 15.
• MCO-gap rows excluded by default (--include-mco to override).
• Pre-launch structural nulls (rail services not yet running) filled
  with 0.0 — not median — because no service = zero ridership.
• Static features (population, GTFS, OSM POI, GADM) are broadcast to all
  dates in feature_align.py and scale normally through MinMaxScaler.
• --dtype float16 (default) is safe for MinMax-scaled [0,1] data.
• --compress writes .npz (zlib) for an extra 1.5–3× disk saving;
  load via np.load(...)['arr'].
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

os.makedirs("data/sequences", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration defaults (override via CLI args or edit here)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULTS = dict(
    features_path = "data/features/features_aligned.csv",
    T_in          = 14,      # look-back window (days)
    T_out         = 7,       # forecast horizon (days)
    train_frac    = 0.70,
    val_frac      = 0.15,    # remaining 0.15 → test
    target_col    = "total_ridership",
    dtype         = "float16",   # "float16" or "float32"
    compress      = False,
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


_TEMPORAL_FEAT_SLICE = slice(13, 29)   # 16 temporal features, indices 13–28
                                        # (is_public_holiday … day_of_year)
                                        # Matches FEAT_GROUPS["temporal"] in hmttsf.py


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
      y: (N_windows, T_out)   — future values of the target column only
    """
    X_list, y_list = [], []
    total = arr.shape[0]
    for i in range(total - T_in - T_out + 1):
        X_list.append(arr[i : i + T_in])
        y_list.append(arr[i + T_in : i + T_in + T_out, target_idx])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def make_future_temporal(arr: np.ndarray, T_in: int, T_out: int) -> np.ndarray:
    """
    For each sliding window of size T_in, extract the T_out future rows for
    the temporal feature columns (day-of-week, weekend flag, holiday flags, etc.).
    These features are deterministic for any calendar date, so they can be
    provided to the model as known future context.

    Returns: (N_windows, T_out, 16)  — raw unscaled values, same window count
             as make_sliding_windows.
    """
    total = arr.shape[0]
    n = total - T_in - T_out + 1
    out = np.empty((n, T_out, 16), dtype=np.float32)
    for i in range(n):
        out[i] = arr[i + T_in : i + T_in + T_out, _TEMPORAL_FEAT_SLICE]
    return out


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
    """Save array with optional compression and dtype casting."""
    arr = arr.astype(dtype)
    if compress:
        path = path_no_ext + ".npz"
        np.savez_compressed(path, arr=arr)
    else:
        path = path_no_ext + ".npy"
        np.save(path, arr)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Sequence builder
# ══════════════════════════════════════════════════════════════════════════════

def build_sequences(cfg: dict) -> None:
    print("=== Sequence Builder (all models) ===")
    dtype    = np.dtype(cfg.get("dtype", "float32"))
    compress = cfg.get("compress", False)

    df = load_aligned(cfg["features_path"])
    df = df.drop(columns=[c for c in ["is_mco"] if c in df.columns])
    print(f"Loaded {len(df)} rows from {cfg['features_path']}")

    # Pre-launch structural nulls (rail lines not yet operational) → 0
    df = df.fillna(0.0)

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

    # ── Future temporal features ──────────────────────────────────────────────
    # Extract temporal-feature columns (indices 13–28) for the T_out forecast
    # steps of each window.  Day-of-week, weekend flag, and holiday flags are
    # fully deterministic for any future date, so they can be given to the
    # model as known decoder context — fixing the flat weekend prediction issue.
    Xf_tr = make_future_temporal(arr_train, T_in, T_out)
    Xf_va = make_future_temporal(arr_val,   T_in, T_out)
    Xf_te = make_future_temporal(arr_test,  T_in, T_out)

    # Scale with the same scaler as X (temporal-column slice only)
    _t_scale = scaler_X.scale_[_TEMPORAL_FEAT_SLICE]  # (16,)
    _t_min   = scaler_X.min_[_TEMPORAL_FEAT_SLICE]    # (16,)
    Xf_tr = Xf_tr * _t_scale + _t_min
    Xf_va = Xf_va * _t_scale + _t_min
    Xf_te = Xf_te * _t_scale + _t_min

    print(f"\nSequence shapes (after scaling):")
    print(f"  X_train: {X_tr.shape}   y_train: {y_tr.shape}")
    print(f"  X_val  : {X_va.shape}   y_val  : {y_va.shape}")
    print(f"  X_test : {X_te.shape}   y_test : {y_te.shape}")
    print(f"  X_future_train: {Xf_tr.shape}  (temporal cols 13–28, future T_out steps)")

    T_in_val = cfg["T_in"]
    out = cfg.get("out_dir") or (
        "data/sequences/lstm" if T_in_val == 14
        else f"data/sequences/lookback_{T_in_val}"
    )
    os.makedirs(out, exist_ok=True)
    ext = ".npz" if compress else ".npy"
    _save(f"{out}/X_train",        X_tr,  compress, dtype)
    _save(f"{out}/y_train",        y_tr,  compress, dtype)
    _save(f"{out}/X_val",          X_va,  compress, dtype)
    _save(f"{out}/y_val",          y_va,  compress, dtype)
    _save(f"{out}/X_test",         X_te,  compress, dtype)
    _save(f"{out}/y_test",         y_te,  compress, dtype)
    _save(f"{out}/X_future_train", Xf_tr, compress, dtype)
    _save(f"{out}/X_future_val",   Xf_va, compress, dtype)
    _save(f"{out}/X_future_test",  Xf_te, compress, dtype)
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
        "temporal_feat_start": 13,
        "temporal_feat_end":   29,   # exclusive; 16 temporal features (indices 13–28)
    }
    with open(f"{out}/split_dates.json", "w") as f:
        json.dump(split_dates, f, indent=2)

    print(f"\nExported to {out}/  (dtype={dtype}, compress={compress})")
    print(f"  Load: X_train = np.load('{out}/X_train{ext}')"
          + (" ['arr']" if compress else ""))
    print("  Inverse-transform predictions: scaler_y.inverse_transform(y_pred)")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Build model-ready sequences from aligned features (all 15 models)"
    )
    p.add_argument("--features-path", default=DEFAULTS["features_path"],
                   help="Path to features_aligned.csv from feature_align.py "
                        "(all 8 spatio-temporal sources)")
    p.add_argument("--T-in",    type=int,   default=DEFAULTS["T_in"],
                   choices=[7, 14, 28, 56, 84],
                   help="Look-back window in days (14 → data/sequences/lstm/, "
                        "other values → data/sequences/lookback_{N}/)")
    p.add_argument("--T-out",   type=int,   default=DEFAULTS["T_out"])
    p.add_argument("--target",  default=DEFAULTS["target_col"])
    p.add_argument("--train-frac", type=float, default=DEFAULTS["train_frac"])
    p.add_argument("--val-frac",   type=float, default=DEFAULTS["val_frac"])
    p.add_argument("--out-dir", default=None,
                   help="Override output directory (default: data/sequences/lstm for T_in=14, "
                        "data/sequences/lookback_{N} otherwise)")
    p.add_argument("--dtype", choices=["float16", "float32"], default=DEFAULTS["dtype"],
                   help="Storage dtype. float16 halves file size with negligible precision "
                        "loss for MinMax-scaled [0,1] data (default: float16).")
    p.add_argument("--compress", action="store_true",
                   help="Save .npz (zlib compressed) instead of .npy. "
                        "Load via np.load(...)['arr'].")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = {
        "features_path": args.features_path,
        "T_in":          args.T_in,
        "T_out":         args.T_out,
        "target_col":    args.target,
        "train_frac":    args.train_frac,
        "val_frac":      args.val_frac,
        "out_dir":       args.out_dir,
        "dtype":         args.dtype,
        "compress":      args.compress,
    }

    build_sequences(cfg)

    print("\n✓ Sequences built.")
    print("  Run order reminder:")
    print("    1. cleaning scripts (ridership, fuelprice, holiday, rainfall, gadm, population, osm, gtfs)")
    print("    2. feature_align.py")
    print("    3. sequence_builder.py  ← you are here")
