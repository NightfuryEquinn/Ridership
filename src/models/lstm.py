"""
Multi-Service LSTM v5 — RapidKL Ridership Forecasting
Hardware target: 13th Gen Intel Core i5-13500HX + NVIDIA RTX 4050 (6 GB VRAM)

Pipeline source: FEATURE.md
   - Input  : feature_matrix_lstm_clean.parquet
   - Targets: all 12 ridership services (bus + rail)
   - Range  : 2019-01-01 to 2026-03-21

v5 changes from v4 (single-service bus_rkl only):
  - Multi-service: shared LSTM backbone + per-service MLP heads
  - Targets auto-detected via TARGET_COL_RE (no hardcoded service name)
  - Feature engineering removed: all lag/rolling/calendar features already
    present in the cleaned parquet; duplicating them inflated input size
    and introduced near-identical columns that confused the input projection
  - Per-service StandardScaler and post-hoc linear calibration
  - Metrics reported per-service and as macro-average
  - Checkpoint saves both feat_scaler and tgt_scalers for inference
  - Loss plot y-axis corrected from "Huber Loss" to "MSE Loss"

Architecture (unchanged from v4):
  Input (B, T, n_features)
    → InputProj  Linear(n_features, proj_dim) + ReLU + LayerNorm
    → LSTM(proj_dim, hidden=128, layers=2, dropout=0.2)
    → LayerNorm + Dropout
    → n_services × [Linear(128, 64) + ReLU + Linear(64, horizon)]
  Output: (B, n_services, horizon)
"""

import math
import re
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────
CFG = dict(
    # Paths
    data_path  = "data/features/feature_matrix_lstm_clean.parquet",
    output_dir = "src/outputs/lstm",

    # Targets — raw service columns are auto-detected via TARGET_COL_RE.
    # Negative lookahead excludes derived features (roll, lag, zscore, anomaly,
    # cyclical encodings, anything with a digit like "7d").
    target_col_re = r"^ridership__(bus|rail)_(?!.*(?:roll|lag|zscore|anomaly|sin|cos|\d))[a-z_]+$",

    # Sequence
    seq_len  = 28,   # 4-week look-back captures weekly seasonality
    horizon  = 7,    # forecast horizon (days)

    # Model
    proj_dim     = 128,
    hidden_size  = 128,
    num_layers   = 2,
    dropout      = 0.2,
    bidirectional = False,

    # Training
    batch_size   = 32,
    max_epochs   = 5,
    lr           = 1e-3,
    weight_decay = 1e-5,
    patience     = 2,
    grad_clip    = 1.0,

    # LR schedule — Reduce on plateau
    lr_scheduler = "ReduceLROnPlateau",
    lr_patience  = 2,
    lr_factor    = 0.5,
    lr_min       = 1e-8,

    # Data split (chronological)
    train_ratio = 0.70,
    val_ratio   = 0.15,
    # test_ratio = 0.15 (remainder)

    # Reproducibility
    seed = 42,
)


# ─────────────────────────────────────────────
# 1. UTILITIES
# ─────────────────────────────────────────────
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev   = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        logging.info(f"GPU  : {props.name}  |  VRAM: {props.total_memory / 1e9:.1f} GB")
    else:
        dev = torch.device("cpu")
        logging.info("GPU not available — running on CPU")
    return dev


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s  %(levelname)s  %(message)s",
        handlers = [logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    logging.info(f"Logs -> {log_path}")


# ─────────────────────────────────────────────
# 2. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

# Derived ridership suffixes used to detect and normalise feature coverage.
# Listed longest-first so suffix matching is unambiguous when iterating.
_RIDERSHIP_DERIVED_SUFFIXES: tuple[str, ...] = (
    "roll_mean_14d", "roll_max_14d",
    "roll_mean_7d",  "roll_max_7d",
    "roll_mean_3d",  "roll_max_3d",
    "lag_m7d", "lag_m3d", "lag_m1d",
    "lag_7d",  "lag_3d",  "lag_1d",
    "zscore",  "anomaly",
)

# Rainfall temporal windows kept after station aggregation.
# lag_3d / lag_7d are dropped: they correlate strongly with lag_1d (r ≥ 0.77)
# and the LSTM already captures multi-step temporal patterns internally.
_RAIN_KEEP_WINDOWS: frozenset[str] = frozenset(
    {"raw", "lag_1d", "roll_3d", "roll_7d", "roll_14d"}
)
_RAIN_DROP_WINDOWS: frozenset[str] = frozenset({"lag_3d", "lag_7d"})


def _aggregate_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audit fix 2 — Reduce rainfall features from ~114 to ~11 aggregated cols.

    Issue (lstm_feature_audit.html): 14 stations × lag(1d/3d/7d) ×
    roll(3d/7d/14d) + flags expands to 114 columns (8.1× explosion). Nearly
    all station pairs share the same rainfall signal; per-station detail
    adds noise rather than useful variance.

    Strategy:
      • For each temporal window in _RAIN_KEEP_WINDOWS: collapse all station
        columns to one spatial mean and one spatial max.
      • Drop per-station lag_3d / lag_7d entirely (lag_m1d vs lag_m7d r = 0.87;
        the LSTM handles temporal context without redundant lags).
      • Merge the 14 heavy_rain_flag_* columns into a single any-station flag.
      • Remove all original per-station columns afterwards.

    Assumes the naming convention:
      rain[fall]_<station>[_lag_Nd | _roll_Nd]      (raw / lag / rolling)
      heavy_rain_flag_<station>                      (binary flags)
    """
    rain_re   = re.compile(r"^(rain(?:fall)?_|heavy_rain_flag_)", re.IGNORECASE)
    rain_cols = [c for c in df.columns if rain_re.match(c)]

    if not rain_cols:
        logging.info("[Audit] No rainfall columns detected — skipping de-bloat step.")
        return df

    # ── 1. Aggregate heavy_rain_flag_* → single any-station flag ──────────
    flag_cols = [c for c in rain_cols if "heavy_rain_flag" in c.lower()]
    if flag_cols:
        df["rain_any_heavy_flag"] = df[flag_cols].max(axis=1).astype(np.float32)
        logging.info(
            f"[Audit] {len(flag_cols)} heavy_rain_flag_* -> rain_any_heavy_flag"
        )

    # ── 2. Group non-flag rain columns by temporal window ─────────────────
    window_re = re.compile(r"_(lag_\d+d|roll_\d+d)$", re.IGNORECASE)
    non_flag  = [c for c in rain_cols if "heavy_rain_flag" not in c.lower()]

    window_groups: dict[str, list[str]] = {}
    for col in non_flag:
        m      = window_re.search(col)
        window = m.group(1).lower() if m else "raw"
        window_groups.setdefault(window, []).append(col)

    # ── 3. Aggregate KEEP windows; drop DROP windows ───────────────────────
    cols_to_drop: list[str] = flag_cols[:]

    for window, cols in window_groups.items():
        sfx = f"__{window}" if window != "raw" else ""

        if window in _RAIN_KEEP_WINDOWS:
            df[f"rain_spatial_mean{sfx}"] = (
                df[cols].mean(axis=1).astype(np.float32)
            )
            df[f"rain_spatial_max{sfx}"] = (
                df[cols].max(axis=1).astype(np.float32)
            )
            cols_to_drop.extend(cols)
            logging.info(
                f"[Audit] {len(cols):>3} rain cols ({window:<10}) "
                f"-> rain_spatial_mean{sfx}, rain_spatial_max{sfx}"
            )

        elif window in _RAIN_DROP_WINDOWS:
            cols_to_drop.extend(cols)
            logging.info(
                f"[Audit] Dropped {len(cols):>3} rain cols ({window}) "
                f"— redundant lag (r ≥ 0.77 with lag_1d)"
            )

        else:
            logging.info(
                f"[Audit] Unknown rainfall window '{window}' "
                f"({len(cols)} cols) — kept as-is"
            )

    df = df.drop(columns=cols_to_drop, errors="ignore")

    n_new = sum(
        1 for c in df.columns
        if c.startswith("rain_spatial_") or c == "rain_any_heavy_flag"
    )
    logging.info(
        f"[Audit] Rainfall de-bloat complete: {len(rain_cols)} cols -> {n_new} cols"
    )
    return df


def _enforce_ridership_coverage(df: pd.DataFrame, target_re_str: str) -> pd.DataFrame:
    """
    Audit fix 3 — Drop ridership-derived feature types that are absent for
    any transit line so every service presents a uniform feature space.

    Issue (lstm_feature_audit.html): 6 of 12 lines are missing at least one
    of roll_mean_7d, roll_max_7d, or anomaly flag. The shared LSTM backbone
    receives different feature spaces per series, which confuses the input
    projection and creates implicit data leakage through missingness patterns.

    Strategy: compute the intersection of feature-type suffixes that every
    line has; drop columns whose suffix falls outside that intersection.
    Imputing zeros is avoided because zero is a meaningful value in scaled
    ridership space.
    """
    target_re = re.compile(target_re_str)

    # Map col → (line_key, suffix) for every ridership-derived column
    line_to_suffixes: dict[str, set[str]] = {}
    col_meta: dict[str, tuple[str, str]] = {}

    for col in df.columns:
        if target_re.match(col) or not col.startswith("ridership__"):
            continue
        for suffix in _RIDERSHIP_DERIVED_SUFFIXES:
            if col.endswith(f"_{suffix}"):
                # Everything between "ridership__" and "_<suffix>" is the line key
                line_key = col[len("ridership__") : -(len(suffix) + 1)]
                col_meta[col] = (line_key, suffix)
                line_to_suffixes.setdefault(line_key, set()).add(suffix)
                break  # longest-first guarantees the correct match

    if not line_to_suffixes:
        logging.info("[Audit] No ridership-derived columns found — skipping coverage check.")
        return df

    suffix_sets  = list(line_to_suffixes.values())
    common       = set.intersection(*suffix_sets)
    all_seen     = set.union(*suffix_sets)
    inconsistent = all_seen - common

    if not inconsistent:
        logging.info("[Audit] Ridership feature coverage is uniform — no columns dropped.")
        return df

    to_drop = [col for col, (_, sfx) in col_meta.items() if sfx in inconsistent]
    logging.info(
        f"[Audit] Dropping {len(to_drop)} ridership-derived cols with "
        f"inconsistent line coverage (suffixes: {sorted(inconsistent)})"
    )
    return df.drop(columns=to_drop, errors="ignore")


def load_and_clean(cfg: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Load the pre-cleaned parquet, apply feature-audit fixes, detect service
    targets, and return (df, feature_cols, target_cols).

    Audit fixes applied (lstm_feature_audit.html):
      1. Zero-variance columns dropped (e.g. heavy_rain_flag_1902, std = 0).
      2. Rainfall de-bloat: 114 per-station cols → ~11 spatial aggregates;
         redundant lag_3d / lag_7d removed (r ≥ 0.77 with lag_1d).
      3. Ridership coverage normalised: derived feature types absent for any
         line are dropped so the LSTM sees a uniform input space per service.

    Feature engineering remains absent by design: all lag, rolling, and
    calendar features are already present in the cleaned parquet. Regenerating
    them here would duplicate columns and inflate the input projection.
    """
    path = Path(cfg["data_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at '{path}'.\n"
            "Run pretrain_clean.py first."
        )

    logging.info(f"Loading {path} …")
    df = pd.read_parquet(path)
    logging.info(f"Raw shape: {df.shape}  |  NaN%: {df.isna().mean().mean() * 100:.1f}")

    df = df.replace([np.inf, -np.inf], np.nan)

    # Ensure DatetimeIndex
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Safety impute (parquet should already be clean)
    remaining = df.isna().sum().sum()
    if remaining:
        logging.warning(f"{remaining:,} NaN found — applying ffill/bfill/zero-fill")
        df = df.ffill().bfill().fillna(0.0)

    # ── Audit fix 1: Drop zero-variance columns ──────────────────────────────
    # Catches heavy_rain_flag_1902 (std = 0) and any other constant survivors.
    # Zero-variance features add noise to the input projection without signal.
    constant = df.columns[df.nunique() <= 1].tolist()
    if constant:
        logging.info(
            f"[Audit] Dropping {len(constant)} zero-variance column(s): {constant}"
        )
        df = df.drop(columns=constant)

    # ── Audit fix 2: Rainfall feature de-bloat ───────────────────────────────
    # 114 per-station cols → ~11 spatial aggregates + 1 any-heavy-rain flag.
    df = _aggregate_rainfall(df)

    # ── Audit fix 3: Enforce consistent ridership feature coverage ───────────
    # Drop derived types missing for ≥1 line; LSTM backbone needs uniform input.
    df = _enforce_ridership_coverage(df, cfg["target_col_re"])

    # Auto-detect service targets
    target_re   = re.compile(cfg["target_col_re"])
    target_cols = [c for c in df.columns if target_re.match(c)]
    if not target_cols:
        raise RuntimeError(
            f"No target columns matched '{cfg['target_col_re']}'. "
            f"Sample columns: {list(df.columns[:8])}"
        )

    feature_cols = [c for c in df.columns if c not in target_cols]
    logging.info(
        f"Clean shape: {df.shape}  |  "
        f"features: {len(feature_cols)}  |  services: {len(target_cols)}"
    )
    logging.info(f"Services: {target_cols}")
    return df, feature_cols, target_cols


# ─────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────
class RidershipDataset(Dataset):
    """
    Sliding-window multi-service dataset.
    X : (seq_len, n_features)
    y : (n_services, horizon)
    """

    def __init__(
        self,
        X: np.ndarray,   # (T, n_features)
        y: np.ndarray,   # (T, n_services)
        seq_len: int,
        horizon: int,
    ):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self._len = max(len(self.X) - self.seq_len - self.horizon + 1, 0)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._len:
            raise IndexError("index out of range")
        x = self.X[idx : idx + self.seq_len]                               # (seq_len, n_features)
        y = self.y[idx + self.seq_len : idx + self.seq_len + self.horizon] # (horizon, n_services)
        return x, y.T                                                       # y: (n_services, horizon)


def make_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    cfg: dict,
) -> tuple[dict, StandardScaler, list[StandardScaler]]:
    """
    Chronological train/val/test split with per-service target scaling.
    Clips extreme outlier values (≥3σ) in target before scaling to improve
    training stability. Scales fit on train only to prevent data leakage.
    Returns (datasets, feat_scaler, tgt_scalers).
    tgt_scalers is indexed identically to target_cols.
    """
    T         = len(df)
    n_train   = int(T * cfg["train_ratio"])
    n_val     = int(T * cfg["val_ratio"])
    train_end = n_train
    val_end   = n_train + n_val

    X_all = df[feature_cols].values.astype(np.float32)   # (T, n_features)
    y_all = df[target_cols].values.astype(np.float32)    # (T, n_services)

    # ---- pre-scale outlier clip (per-service, fit on train) -----------------
    for i in range(y_all.shape[1]):
        tr = y_all[:train_end, i]
        mu, sg = tr.mean(), tr.std()
        lower, upper = mu - 3.0 * sg, mu + 3.0 * sg
        clipped = np.clip(y_all[:, i], lower, upper)
        y_all[:, i] = clipped

    # Fit feature scaler on train only
    feat_scaler = StandardScaler()
    feat_scaler.fit(X_all[:train_end])
    X_all = feat_scaler.transform(X_all)

    # Fit one target scaler per service on train only
    tgt_scalers: list[StandardScaler] = []
    y_scaled = np.zeros_like(y_all)
    for i in range(y_all.shape[1]):
        sc = StandardScaler()
        sc.fit(y_all[:train_end, i : i + 1])
        y_scaled[:, i] = sc.transform(y_all[:, i : i + 1]).ravel()
        tgt_scalers.append(sc)

    seq_len = cfg["seq_len"]
    horizon = cfg["horizon"]

    # Overlap by seq_len so val/test windows have full context at their start
    splits = {
        "train": (X_all[:train_end],              y_scaled[:train_end]),
        "val"  : (X_all[train_end - seq_len : val_end], y_scaled[train_end - seq_len : val_end]),
        "test" : (X_all[val_end - seq_len :],     y_scaled[val_end - seq_len :]),
    }
    datasets = {
        name: RidershipDataset(X, y, seq_len, horizon)
        for name, (X, y) in splits.items()
    }
    return datasets, feat_scaler, tgt_scalers


# ─────────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────────
class ImprovedLSTM(nn.Module):
    """
    Shared backbone + per-service heads for multi-step, multi-service forecasting.

    Architecture:
      Input (B, T, n_features)
        → InputProj  Linear(n_features, proj_dim) + ReLU + LayerNorm
        → LSTM(proj_dim, hidden_size, num_layers, dropout)
        → LayerNorm + Dropout  (on last time-step hidden state)
        → n_services × [Linear(hidden, 64) + ReLU + Linear(64, horizon)]
      Output: (B, n_services, horizon)

    The shared backbone learns cross-service temporal dynamics; each head
    specialises in one service's prediction distribution.
    """

    def __init__(
        self,
        input_size   : int,
        proj_dim     : int,
        hidden_size  : int,
        num_layers   : int,
        dropout      : float,
        horizon      : int,
        n_services   : int,
        bidirectional: bool = False,
    ):
        super().__init__()
        D = 2 if bidirectional else 1

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
        )
        self.lstm = nn.LSTM(
            input_size    = proj_dim,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
            batch_first   = True,
        )
        self.norm    = nn.LayerNorm(hidden_size * D)
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * D, 64),
                nn.ReLU(),
                nn.Linear(64, horizon),
            )
            for _ in range(n_services)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_size)
        x       = self.input_proj(x)        # (B, seq_len, proj_dim)
        out, _  = self.lstm(x)              # (B, seq_len, D*hidden)
        last    = out[:, -1, :]             # (B, D*hidden)
        last    = self.norm(last)
        last    = self.dropout(last)
        return torch.stack(                 # (B, n_services, horizon)
            [head(last) for head in self.heads], dim=1
        )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 5. TRAINING & VALIDATION
# ─────────────────────────────────────────────
def train_epoch(
    model, loader, optimizer, criterion, scaler_amp, device, grad_clip, use_amp
) -> float:
    """Single training epoch with mixed-precision support."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = criterion(model(X_batch), y_batch)
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        total_loss += criterion(model(X_batch), y_batch).item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


# ─────────────────────────────────────────────
# 6. METRICS
# ─────────────────────────────────────────────
@torch.no_grad()
def collect_predictions(
    model       : nn.Module,
    loader      : DataLoader,
    tgt_scalers : list[StandardScaler],
    device      : torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect and inverse-scale model predictions without calibration.
    Returns (preds_inv, trues_inv), each shaped (N, n_services, horizon).
    Used to fit per-service calibrators on the validation set.
    """
    model.eval()
    all_pred, all_true = [], []
    for X_batch, y_batch in loader:
        all_pred.append(model(X_batch.to(device)).cpu().numpy())
        all_true.append(y_batch.numpy())

    preds = np.concatenate(all_pred, axis=0)   # (N, n_services, horizon)
    trues = np.concatenate(all_true, axis=0)

    preds_inv, trues_inv = np.zeros_like(preds), np.zeros_like(trues)
    H = preds.shape[2]
    for i, sc in enumerate(tgt_scalers):
        preds_inv[:, i, :] = sc.inverse_transform(preds[:, i, :].reshape(-1, 1)).reshape(-1, H)
        trues_inv[:, i, :] = sc.inverse_transform(trues[:, i, :].reshape(-1, 1)).reshape(-1, H)
    return preds_inv, trues_inv


@torch.no_grad()
def evaluate(
    model        : nn.Module,
    loader       : DataLoader,
    tgt_scalers  : list[StandardScaler],
    target_cols  : list[str],
    device       : torch.device,
    calibrators  : dict[str, LinearRegression] | None = None,
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """
    Return per-service and macro-aggregate metrics in original scale.
    Combined = max(0, 100 − MAPE − MAE% − RMSE%).
    """
    preds_inv, trues_inv = collect_predictions(model, loader, tgt_scalers, device)
    H = preds_inv.shape[2]

    if calibrators is not None:
        for i, col in enumerate(target_cols):
            cal = calibrators[col]
            preds_inv[:, i, :] = (
                cal.predict(preds_inv[:, i, :].ravel().reshape(-1, 1)).reshape(-1, H)
            )

    per_service: dict[str, dict] = {}
    for i, col in enumerate(target_cols):
        p = preds_inv[:, i, :].ravel()
        t = trues_inv[:, i, :].ravel()
        mae  = mean_absolute_error(t, p)
        rmse = math.sqrt(mean_squared_error(t, p))
        r2   = r2_score(t, p)
        mape = float(np.mean(np.abs((t - p) / (np.abs(t) + 1e-8))) * 100)
        mean_t   = float(np.abs(t).mean()) + 1e-8
        mae_pct  = mae / mean_t * 100
        rmse_pct = rmse / mean_t * 100
        per_service[col] = dict(
            MAE=mae, RMSE=rmse, R2=r2, MAPE=mape,
            MAE_pct=mae_pct, RMSE_pct=rmse_pct,
            Combined=float(np.clip(100.0 - (mape + mae_pct + rmse_pct), 0, 100)),
        )

    # Macro-average across services
    agg = {k: float(np.mean([m[k] for m in per_service.values()]))
           for k in ["MAE", "RMSE", "R2", "MAPE", "MAE_pct", "RMSE_pct", "Combined"]}

    return agg, per_service, preds_inv, trues_inv


# ─────────────────────────────────────────────
# 7. PLOTTING
# ─────────────────────────────────────────────
def save_plots(
    history     : dict,
    preds_inv   : np.ndarray,
    trues_inv   : np.ndarray,
    target_cols : list[str],
    output_dir  : Path,
) -> None:
    # Loss curves
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="Train loss")
    ax.plot(history["val_loss"],   label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close(fig)
    logging.info("Saved loss_curves.png")

    # Per-service 1-day-ahead forecast vs actual (first 90 samples)
    n          = min(90, preds_inv.shape[0])
    n_services = len(target_cols)
    ncols      = 2
    nrows      = math.ceil(n_services / ncols)
    fig, axes  = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    axes       = axes.ravel()

    for i, (col, ax) in enumerate(zip(target_cols, axes)):
        ax.plot(trues_inv[:n, i, 0], label="Actual",    linewidth=1.2)
        ax.plot(preds_inv[:n, i, 0], label="Predicted", linewidth=1.2, linestyle="--")
        ax.set_title(col.replace("ridership__", ""), fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n_services, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Test Set — 1-Day-Ahead Forecast vs Actual (per service)", y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "forecast_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved forecast_vs_actual.png")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────
def main() -> None:
    cfg = CFG
    set_seed(cfg["seed"])

    output_dir = Path(cfg["output_dir"])
    setup_logging(output_dir)
    device = get_device()

    # ── Data ──────────────────────────────────────────────────────────────
    df, feature_cols, target_cols = load_and_clean(cfg)
    datasets, feat_scaler, tgt_scalers = make_splits(df, feature_cols, target_cols, cfg)

    n_features = len(feature_cols)
    n_services = len(target_cols)
    pin_memory = device.type == "cuda"

    loaders = {
        split: DataLoader(
            ds,
            batch_size         = cfg["batch_size"] if split == "train" else cfg["batch_size"] * 2,
            shuffle            = (split == "train"),
            num_workers        = 4,
            pin_memory         = pin_memory,
            persistent_workers = True,
        )
        for split, ds in datasets.items()
    }
    logging.info(
        f"Batches — train: {len(loaders['train'])}  "
        f"val: {len(loaders['val'])}  test: {len(loaders['test'])}"
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = ImprovedLSTM(
        input_size    = n_features,
        proj_dim      = cfg["proj_dim"],
        hidden_size   = cfg["hidden_size"],
        num_layers    = cfg["num_layers"],
        dropout       = cfg["dropout"],
        horizon       = cfg["horizon"],
        n_services    = n_services,
        bidirectional = cfg["bidirectional"],
    ).to(device)

    logging.info(f"Model parameters: {count_params(model):,}")
    logging.info(model)

    # ── Loss, optimiser, scheduler ────────────────────────────────────────
    criterion = nn.SmoothL1Loss()  # Huber (more robust than MSE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = cfg["lr_factor"],
        patience = cfg["lr_patience"],
        min_lr   = cfg["lr_min"],
    )

    use_amp    = device.type == "cuda"
    scaler_amp = torch.amp.GradScaler(enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────────────────
    history    = {"train_loss": [], "val_loss": []}
    best_val   = float("inf")
    best_ckpt  = output_dir / "best_model.pt"
    no_improve = 0

    logging.info("-" * 60)
    logging.info("Starting training …")
    logging.info("-" * 60)

    for epoch in range(1, cfg["max_epochs"] + 1):
        train_loss = train_epoch(
            model, loaders["train"], optimizer, criterion,
            scaler_amp, device, cfg["grad_clip"], use_amp,
        )
        val_loss = eval_epoch(model, loaders["val"], criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        logging.info(
            f"Epoch {epoch:4d}/{cfg['max_epochs']}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  lr={lr_now:.2e}"
        )

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(
                {
                    "epoch"       : epoch,
                    "model_state" : model.state_dict(),
                    "optim_state" : optimizer.state_dict(),
                    "val_loss"    : val_loss,
                    "cfg"         : cfg,
                    "feature_cols": feature_cols,
                    "target_cols" : target_cols,
                    "feat_scaler" : feat_scaler,
                    "tgt_scalers" : tgt_scalers,
                },
                best_ckpt,
            )
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                logging.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {cfg['patience']} epochs)"
                )
                break

    logging.info(f"Best val loss: {best_val:.6f} — checkpoint: {best_ckpt}")

    # ── Load best checkpoint ──────────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # ── Per-service post-hoc linear calibration on val set ───────────────
    # Fit y_cal = a·ŷ + b per service on val predictions (no leakage).
    # Corrects any residual systematic bias before test evaluation.
    val_p, val_t = collect_predictions(model, loaders["val"], tgt_scalers, device)
    calibrators: dict[str, LinearRegression] = {}
    logging.info("Calibration coefficients:")
    for i, col in enumerate(target_cols):
        cal = LinearRegression().fit(
            val_p[:, i, :].ravel().reshape(-1, 1),
            val_t[:, i, :].ravel(),
        )
        calibrators[col] = cal
        logging.info(f"  {col}: a={cal.coef_[0]:.4f}  b={cal.intercept_:.1f}")

    # ── Test evaluation ───────────────────────────────────────────────────
    agg_metrics, per_service_metrics, preds_inv, trues_inv = evaluate(
        model, loaders["test"], tgt_scalers, target_cols, device, calibrators,
    )

    logging.info("-" * 60)
    logging.info("TEST RESULTS — macro-average across all services")
    for k, v in agg_metrics.items():
        logging.info(f"  {k:10s}: {v:.4f}")
    logging.info("-" * 60)
    logging.info("TEST RESULTS — per service")
    for col, m in per_service_metrics.items():
        logging.info(f"  [{col}]")
        for k, v in m.items():
            logging.info(f"    {k:10s}: {v:.4f}")
    logging.info("-" * 60)

    # ── Plots & artefacts ─────────────────────────────────────────────────
    save_plots(history, preds_inv, trues_inv, target_cols, output_dir)

    np.save(output_dir / "test_preds.npy", preds_inv)
    np.save(output_dir / "test_trues.npy", trues_inv)

    rows = [{"service": col, **m} for col, m in per_service_metrics.items()]
    rows.append({"service": "AGGREGATE", **agg_metrics})
    pd.DataFrame(rows).to_csv(output_dir / "test_metrics.csv", index=False)

    logging.info(f"All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()