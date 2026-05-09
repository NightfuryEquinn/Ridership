"""
Improved Baseline LSTM v4 - Bus Ridership Forecasting
Hardware target: 13th Gen Intel Core i5-13500HX + NVIDIA RTX 4050 (6 GB VRAM)

Pipeline source: FEATURE.md
   - Input  : feature_matrix_lstm.parquet  (2637 rows x 1717 cols)
   - Target : ridership__bus_rkl
   - Range  : 2019-01-01 to 2026-03-21
   - Missing: ~30 % (mainly rainfall, daily resolution)

Root-cause of v3 failure (R²=0.458, MAE=26K, MAPE=12.4%, Combined=60.7):

  [BUG 1 — log1p transform caused systematic peak under-prediction]
    log1p compressed peaks (260K→12.47) and troughs (140K→11.85) into only
    0.62 log-units.  A 0.05 log-unit error appeared tiny in loss space but
    mapped to ~13K error in original scale at peak ridership.  The model
    happily converged to a solution that predicted near the conditional mean,
    systematically under-shooting every weekday peak by 30-35K.

  [BUG 2 — Huber loss ignored peaks (by design)]
    HuberLoss(delta=10000) was chosen to be "robust to outliers".  But 260K
    ridership peaks ARE NOT outliers — they are the dominant signal.  Huber
    down-weighted errors beyond delta=10K, so peak misses of 30K were penalised
    only linearly, not quadratically.  This reinforced the underprediction.

  [BUG 3 — CosineAnnealingLR T_max=500 kept LR high for 50+ epochs]
    At epoch 50, LR ≈ 4.9e-4 (barely decayed from 5e-4).  This explains the
    massive val-loss spikes seen at epochs 40-55: the model was still taking
    large gradient steps into unstable regions of the loss surface.

v4 fixes (LSTM architecture UNCHANGED):
  [FIX 1] Removed log1p / expm1.  Target is now StandardScaler-only (linear
          scale).  The loss surface is proportional to original-scale errors.
  [FIX 2] Replaced HuberLoss with MSELoss.  MSE squares every error so a 30K
          peak miss is penalised 9× more than a 10K mid-week miss.
  [FIX 3] lr_T_max 500 → 100.  LR reaches eta_min at epoch 100, removing the
          unstable high-LR phase that caused val-loss spikes at epoch 40-55.
  [FIX 4] Post-hoc linear calibration.  After training, a single OLS regression
          (y_cal = a·ŷ + b) is fit on val-set predictions to remove any
          residual systematic bias before test evaluation.  No data leakage.

  [UNCHANGED from v3]
  - Input projection layer (N_features → 128 → LSTM)
  - hidden=128, num_layers=2, dropout=0.2
  - Two-layer MLP head (128 → 64 → horizon)
  - seq_len=28, horizon=7, patience=50, AMP, batch_size=32
  - All lag/calendar feature engineering
"""

import os
import math
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────
CFG = dict(
    # Paths
    data_path       = "data/features/feature_matrix_lstm.parquet",
    output_dir      = "src/outputs/lstm",

    # Target
    target_col      = "ridership__bus_rkl",

    # Sequence
    seq_len         = 28,           # ↑ 14→28: 4-week look-back captures weekly seasonality
    horizon         = 7,            # forecast horizon (days) — unchanged

    # Model
    proj_dim        = 128,          # Increased from 128: better feature representation
    hidden_size     = 128,          # Increased from 128: greater temporal modeling capacity
    num_layers      = 2,            # ↑ 1→2
    dropout         = 0.2,          # ↓ 0.7→0.3  (0.7 was destroying 22/32 neurons)
    bidirectional   = False,

    # Training — tuned for RTX 4050 6 GB
    batch_size      = 32,
    max_epochs      = 500,          # ↑ val loss was still falling at epoch 100
    lr              = 5e-4,         # Increased from 3e-4: faster convergence
    weight_decay    = 1e-5,         # Reduced from 1e-4: less regularization with increased capacity
    patience        = 50,           # ↑ 15→30
    grad_clip       = 0.5,
    huber_delta     = 10000.0,          # UNUSED in v4 — replaced by MSELoss (see FIX 2)

    # LR schedule — cosine annealing (replaces ReduceLROnPlateau)
    lr_T_max        = 100,            # FIX 3: was 500; now decays to eta_min by epoch 100
    lr_eta_min      = 1e-7,         # minimum LR floor

    # Data split (chronological)
    train_ratio     = 0.70,
    val_ratio       = 0.15,
    # test_ratio    = 0.15 (remainder)

    # Imputation strategy for ~30 % NaN
    impute_method   = "ffill",      # "ffill" | "median" | "zero"

    # Reproducibility
    seed            = 42,
)


# ─────────────────────────────────────────────
# 1. UTILITIES
# ─────────────────────────────────────────────
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False   # flip to True to trade reproducibility for speed


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev   = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        logging.info(f"GPU  : {props.name}  |  VRAM: {props.total_memory / 1e9:.1f} GB")
    else:
        dev = torch.device("cpu")
        logging.info("GPU not available – running on CPU")
    return dev


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )
    logging.info(f"Logs -> {log_path}")


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def add_engineered_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Inject autoregressive lags, rolling statistics, and cyclical calendar
    features into the dataframe.

    Why this fixes the flat-prediction problem
    ──────────────────────────────────────────
    Without lag features the LSTM receives only exogenous signals (weather,
    economic indicators, etc.) but never sees past ridership values.  Bus
    ridership has an extremely strong weekly cycle (~140K on weekends, ~260K
    on weekdays). The model cannot learn this oscillation unless it can observe
    "yesterday was a peak day" vs "yesterday was a trough day". Lag-7 alone
    gives it the answer: whatever happened last Monday will very likely happen
    this Monday.

    Rolling stats give the model a smoothed reference level so the head can
    learn to predict *deviations* from the recent mean rather than absolute
    counts — this stabilises gradients and speeds convergence.

    All features use shift(≥1) so no future information leaks into the input.
    NaN values introduced at the start of the series are filled by the caller's
    imputation step (ffill/bfill).
    """
    # ── Autoregressive target lags ────────────────────────────────────────
    for lag in [1, 7, 14, 21]:
        df[f"_tgt_lag_{lag}"] = df[target_col].shift(lag)

    # ── 7-day rolling mean and std (shifted 1 to avoid leakage) ──────────
    shifted = df[target_col].shift(1)
    df["_tgt_roll7_mean"] = shifted.rolling(7, min_periods=7).mean()
    df["_tgt_roll7_std"]  = shifted.rolling(7, min_periods=1).std().fillna(0)

    # ── Cyclical calendar encoding ────────────────────────────────────────
    # sin/cos pairs produce continuous, periodic representations with no
    # arbitrary ordinal distance between e.g. Sunday (6) and Monday (0).
    dow = df.index.dayofweek.astype(float)           # 0=Mon … 6=Sun
    df["_dow_sin"]    = np.sin(2 * np.pi * dow / 7)
    df["_dow_cos"]    = np.cos(2 * np.pi * dow / 7)
    df["_month_sin"]  = np.sin(2 * np.pi * df.index.month / 12)
    df["_month_cos"]  = np.cos(2 * np.pi * df.index.month / 12)
    df["_is_weekend"] = (dow >= 5).astype(float)     # 1.0 = Sat/Sun

    return df


# ─────────────────────────────────────────────
# 3. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
def load_and_clean(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Load parquet, impute NaN, return cleaned df and feature columns."""
    path = Path(cfg["data_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at '{path}'.\n"
            "Run the feature engineering pipeline first."
        )

    logging.info(f"Loading {path} …")
    df = pd.read_parquet(path)
    logging.info(f"Raw shape: {df.shape}  |  NaN%: {df.isna().mean().mean()*100:.1f}")

    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # ── Parse date index ──────────────────────────────────────────────────
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # ── Inject lag / calendar features BEFORE imputation ─────────────────
    # Lag NaNs at the series start will be resolved by the ffill/bfill below
    # along with any pre-existing missing data — no special casing needed.
    df = add_engineered_features(df, cfg["target_col"])

    # ── Impute ────────────────────────────────────────────────────────────
    method = cfg["impute_method"]
    if method == "ffill":
        df = df.ffill().bfill()              # forward-fill then back-fill edges
    elif method == "median":
        df = df.fillna(df.median(numeric_only=True))
    elif method == "zero":
        df = df.fillna(0.0)
    else:
        raise ValueError(f"Unknown impute_method: {method}")

    remaining_nan = df.isna().sum().sum()
    if remaining_nan:
        logging.warning(f"{remaining_nan} NaN remain after imputation — filling with 0")
        df = df.fillna(0.0)

    # ── Drop constant / all-zero columns ─────────────────────────────────
    nunique   = df.nunique()
    drop_cols = nunique[nunique <= 1].index.tolist()
    if drop_cols:
        logging.info(f"Dropping {len(drop_cols)} constant columns")
        df = df.drop(columns=drop_cols)

    if cfg["target_col"] not in df.columns:
        raise KeyError(
            f"Target '{cfg['target_col']}' not found. "
            f"Available: {list(df.columns[:10])} …"
        )

    feature_cols = [c for c in df.columns if c != cfg["target_col"]]
    logging.info(f"Clean shape: {df.shape}  |  features: {len(feature_cols)}")
    return df, feature_cols


# ─────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────
class RidershipDataset(Dataset):
    """
    Sliding-window dataset.
    X : (seq_len, n_features)
    y : (horizon,)  — multi-step targets
    """

    def __init__(
        self,
        X: np.ndarray,   # (T, n_features)
        y: np.ndarray,   # (T,)
        seq_len: int,
        horizon: int,
    ):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.seq_len]
        y = self.y[idx + self.seq_len : idx + self.seq_len + self.horizon]
        return x, y


def make_splits(df, feature_cols, cfg):

    T = len(df)

    n_train   = int(T * cfg["train_ratio"])
    n_val     = int(T * cfg["val_ratio"])
    train_end = n_train
    val_end   = n_train + n_val

    X_all = df[feature_cols].values
    y_all = df[cfg["target_col"]].values

    # Scale on TRAIN ONLY — prevents data leakage
    feat_scaler = StandardScaler()
    feat_scaler.fit(X_all[:train_end])

    # v4 FIX 1: Remove log1p transform.
    # log1p compressed peaks (260K→12.47) and troughs (140K→11.85) into only
    # 0.62 log-units of range.  A 0.05 log-unit prediction error looked tiny in
    # loss space but mapped to ~13 K in original scale at peak ridership.
    # With Huber loss on log-scale the model found a local minimum that
    # systematically under-predicted every peak.  Using raw StandardScaler keeps
    # linear proportionality intact so the loss directly penalises large errors
    # at 260 K as much as at 140 K.
    tgt_scaler = StandardScaler()
    tgt_scaler.fit(y_all[:train_end].reshape(-1, 1))

    X_all = feat_scaler.transform(X_all)
    y_all = tgt_scaler.transform(y_all.reshape(-1, 1)).ravel()

    seq_len = cfg["seq_len"]
    horizon = cfg["horizon"]

    train_X = X_all[:train_end]
    train_y = y_all[:train_end]

    # Overlap by seq_len so the first val/test window has full context
    val_X = X_all[train_end - seq_len : val_end]
    val_y = y_all[train_end - seq_len : val_end]

    test_X = X_all[val_end - seq_len :]
    test_y = y_all[val_end - seq_len :]

    datasets = {
        "train": RidershipDataset(train_X, train_y, seq_len, horizon),
        "val"  : RidershipDataset(val_X,   val_y,   seq_len, horizon),
        "test" : RidershipDataset(test_X,  test_y,  seq_len, horizon),
    }

    return datasets, feat_scaler, tgt_scaler


# ─────────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────────
class ImprovedLSTM(nn.Module):
    """
    Input projection → Stacked LSTM → MLP head for multi-step forecasting.

    Architecture:
      Input (B, T, 1678)
        → InputProj  Linear(1678, proj_dim) + ReLU + LayerNorm      [NEW]
        → LSTM(proj_dim, hidden=128, layers=2, dropout=0.3)
        → LayerNorm(128)
        → Dropout(0.3)
        → Linear(128, 64) + ReLU                                     [NEW]
        → Linear(64, horizon)

    Why the input projection matters:
      Without it, the LSTM's input-to-hidden weight matrix is 4×1678×32 ≈ 215K
      parameters all devoted to a single, poorly-regularised embedding.  By
      projecting first to proj_dim=128, the LSTM sees a compact representation
      and can use its recurrent capacity to model temporal dynamics instead of
      compressing high-dim input each step.
    """

    def __init__(
        self,
        input_size  : int,
        proj_dim    : int,
        hidden_size : int,
        num_layers  : int,
        dropout     : float,
        horizon     : int,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        D = 2 if bidirectional else 1

        # ── Input projection ──────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
        )

        # ── Recurrent core ────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size   = proj_dim,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            # inter-layer dropout only meaningful when num_layers > 1
            dropout      = dropout if num_layers > 1 else 0.0,
            bidirectional= bidirectional,
            batch_first  = True,
        )
        self.norm    = nn.LayerNorm(hidden_size * D)
        self.dropout = nn.Dropout(dropout)

        # ── MLP head ──────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_size * D, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_size)
        x    = self.input_proj(x)          # (B, seq_len, proj_dim)
        out, _ = self.lstm(x)              # (B, seq_len, D*hidden)
        last = out[:, -1, :]               # last time-step
        last = self.norm(last)
        last = self.dropout(last)
        return self.head(last)             # (B, horizon)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 5. TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler_amp, device, grad_clip, use_amp):
    """Single training epoch with mixed-precision support."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        pred = model(X_batch)
        total_loss += criterion(pred, y_batch).item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


# ─────────────────────────────────────────────
# 7. EVALUATION METRICS
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, tgt_scaler, device, calibrator=None):
    """Return MAE / RMSE / R² in original scale (averaged across horizon steps)."""
    model.eval()
    all_pred, all_true = [], []
    for X_batch, y_batch in loader:
        pred = model(X_batch.to(device)).cpu().numpy()
        all_pred.append(pred)
        all_true.append(y_batch.numpy())

    preds = np.concatenate(all_pred, axis=0)   # (N, H)
    trues = np.concatenate(all_true, axis=0)   # (N, H)

    # Inverse-scale across the horizon
    # v4 FIX 1 (mirror): log1p removed in make_splits → expm1 also removed here.
    # Only inverse StandardScaler is needed; values are already in original scale.
    preds_inv = tgt_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_inv = tgt_scaler.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)

    # v4 FIX 4: apply linear calibration if provided
    if calibrator is not None:
        preds_inv = calibrator.predict(preds_inv.ravel().reshape(-1,1)).reshape(preds_inv.shape)

    mae  = mean_absolute_error(trues_inv.ravel(), preds_inv.ravel())
    rmse = math.sqrt(mean_squared_error(trues_inv.ravel(), preds_inv.ravel()))
    r2   = r2_score(trues_inv.ravel(), preds_inv.ravel())
    mape = np.mean(
        np.abs((trues_inv - preds_inv) / (np.abs(trues_inv) + 1e-8))
    ) * 100

    # ── MAE% and RMSE% — mean-demand-normalised percentages ──────────────
    # MAE%  = Σ|pred − actual| / Σ(actual) × 100
    #       = MAE / mean(actual) × 100          (equivalent, since both ÷ n)
    # RMSE% = RMSE / mean(actual) × 100
    #
    # Both use mean(actual) as the denominator, making them scale-free
    # percentages directly comparable to MAPE in the Combined formula.
    mean_actual = float(np.abs(trues_inv).mean()) + 1e-8
    mae_pct  = float(mae  / mean_actual * 100)
    rmse_pct = float(rmse / mean_actual * 100)

    # Combined Accuracy ∈ [0, 100]
    # ─────────────────────────────────────────────────────────────────────
    #   Combined = max(0,  100 − MAPE − MAE% − RMSE%)
    combined = float(np.clip(100.0 - (mape + mae_pct + rmse_pct), 0.0, 100.0))

    return dict(
        MAE=mae, RMSE=rmse,
        MAE_pct=mae_pct, RMSE_pct=rmse_pct,
        R2=r2, MAPE=mape, Combined=combined,
    ), preds_inv, trues_inv


# ─────────────────────────────────────────────
# 8. PLOTTING
# ─────────────────────────────────────────────
def save_plots(history: dict, preds, trues, output_dir: Path):
    # ── Loss curves ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="Train loss")
    ax.plot(history["val_loss"],   label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close(fig)
    logging.info("Saved loss_curves.png")

    # ── Forecast vs actual (first 90 days, step-1 horizon) ────────────────
    n = min(90, len(trues))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(trues[:n, 0], label="Actual",    linewidth=1.5)
    ax.plot(preds[:n, 0], label="Predicted", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Sample")
    ax.set_ylabel("bus_rkl (original scale)")
    ax.set_title("Test Set — 1-Day-Ahead Forecast vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "forecast_vs_actual.png", dpi=150)
    plt.close(fig)
    logging.info("Saved forecast_vs_actual.png")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────
def main():
    cfg = CFG
    set_seed(cfg["seed"])

    output_dir = Path(cfg["output_dir"])
    setup_logging(output_dir)
    device = get_device()

    # ── Data ──────────────────────────────────────────────────────────────
    df, feature_cols = load_and_clean(cfg)
    datasets, feat_scaler, tgt_scaler = make_splits(df, feature_cols, cfg)

    n_features = len(feature_cols)
    pin_memory = device.type == "cuda"

    loaders = {
        split: DataLoader(
            ds,
            batch_size         = cfg["batch_size"] if split == "train" else cfg["batch_size"] * 2,
            shuffle            = False,
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
        input_size   = n_features,
        proj_dim     = cfg["proj_dim"],
        hidden_size  = cfg["hidden_size"],
        num_layers   = cfg["num_layers"],
        dropout      = cfg["dropout"],
        horizon      = cfg["horizon"],
        bidirectional= cfg["bidirectional"],
    ).to(device)

    logging.info(f"Model parameters: {count_params(model):,}")
    logging.info(model)

    # ── Loss, optimiser, scheduler ────────────────────────────────────────
    # v4 FIX 2: MSE replaces HuberLoss.
    # Huber deliberately ignores large errors (its core feature for outlier
    # robustness). But ridership peaks at 260K are REAL patterns, not outliers.
    # With Huber(delta=10000) the model under-penalised ~30K peak errors and
    # found it cheaper to predict the conditional mean (flat-ish orange line).
    # MSE squares every error, so a 30K peak miss is penalised 9× more than a
    # 10K mid-week miss — forcing the model to chase peaks.
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )

    # v4 FIX 3: lr_T_max 500→100.
    # With T_max=500 the LR was still ~4.9e-4 at epoch 50 (barely decayed),
    # causing the massive val-loss spikes seen at epochs 40-55 in the chart.
    # T_max=100 makes LR reach eta_min at epoch 100 and restart, giving a
    # stable descent window in the critical first 50 epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = 100,          # FIX: was 500
        eta_min = cfg["lr_eta_min"],
    )

    # Mixed precision — free perf on Ampere / Ada GPUs
    use_amp    = device.type == "cuda"
    scaler_amp = torch.amp.GradScaler(enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────────────────
    history   = {"train_loss": [], "val_loss": []}
    best_val  = float("inf")
    best_ckpt = output_dir / "best_model.pt"
    no_improve = 0

    logging.info("-" * 60)
    logging.info("Starting training...")
    logging.info("-" * 60)

    for epoch in range(1, cfg["max_epochs"] + 1):
        train_loss = train_epoch(
            model, loaders["train"], optimizer, criterion,
            scaler_amp, device, cfg["grad_clip"], use_amp,
        )
        val_loss = eval_epoch(model, loaders["val"], criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        logging.info(
            f"Epoch {epoch:4d}/{cfg['max_epochs']}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  lr={lr_now:.2e}"
        )

        # ── Checkpoint & early stopping ───────────────────────────────────
        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(
                {
                    "epoch"      : epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_loss"   : val_loss,
                    "cfg"        : cfg,
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

    # ── Test evaluation ───────────────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # ── v4 FIX 4: Post-hoc linear calibration on val set ─────────────────
    # Even after fixes 1-3, the model may retain a small systematic bias
    # (e.g. consistently predicting 5K too low on weekday peaks).  We fit a
    # single OLS linear regression:  y_cal = a * y_pred + b  on the val set
    # (no data leakage — val is never seen during training).  The learned a, b
    # are then applied to test predictions before metrics are computed.
    # This is standard practice for baseline models and does NOT change the
    # LSTM weights or architecture.
    from sklearn.linear_model import LinearRegression as _LR

    @torch.no_grad()
    def _collect_preds(loader):
        model.eval()
        ps, ts = [], []
        for Xb, yb in loader:
            p = model(Xb.to(device)).cpu().numpy()
            ps.append(p); ts.append(yb.numpy())
        p = np.concatenate(ps); t = np.concatenate(ts)
        p_inv = tgt_scaler.inverse_transform(p.reshape(-1,1)).reshape(p.shape)
        t_inv = tgt_scaler.inverse_transform(t.reshape(-1,1)).reshape(t.shape)
        return p_inv, t_inv

    val_p, val_t = _collect_preds(loaders["val"])
    cal = _LR().fit(val_p.ravel().reshape(-1,1), val_t.ravel())
    logging.info(f"Calibration  a={cal.coef_[0]:.4f}  b={cal.intercept_:.1f}")

    test_metrics, preds_inv, trues_inv = evaluate(
        model, loaders["test"], tgt_scaler, device, calibrator=cal
    )

    logging.info("-" * 60)
    logging.info("TEST RESULTS (original scale, averaged over horizon)")
    for k, v in test_metrics.items():
        logging.info(f"  {k:8s}: {v:.4f}")
    logging.info("-" * 60)

    # ── Plots & artefacts ─────────────────────────────────────────────────
    save_plots(history, preds_inv, trues_inv, output_dir)

    np.save(output_dir / "test_preds.npy", preds_inv)
    np.save(output_dir / "test_trues.npy", trues_inv)

    metrics_df = pd.DataFrame([test_metrics])
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)
    logging.info(f"All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()