"""
Improved BiLSTM v4 - Bus Ridership Forecasting
Hardware target: 13th Gen Intel Core i5-13500HX + NVIDIA RTX 4050 (6 GB VRAM)

Key differences vs LSTM v4 (baseline):
  - bidirectional=True  → hidden state is 2× wide (forward + backward pass)
  - hidden_size=128, effective output width = 128×2 = 256
  - BiLSTM sees the full seq_len window in BOTH directions before predicting,
    capturing long-range temporal dependencies that a unidirectional LSTM misses.
    NOTE: no data leakage — both directions only see the look-back window
    (seq_len past values), never future target values.

Pipeline source: FEATURE.md
   - Input  : feature_matrix_lstm.parquet  (2637 rows x 1717 cols)
   - Target : ridership__bus_rkl
   - Range  : 2019-01-01 to 2026-03-21
   - Missing: ~30 % (mainly rainfall, daily resolution)

v4 fixes (identical to LSTM v4 — required for fair comparison):
  [FIX 1] Removed log1p / expm1.  Target is StandardScaler-only (linear scale).
          log1p compressed peaks (260K→12.47) and troughs (140K→11.85) into only
          0.62 log-units, causing systematic peak under-prediction.
  [FIX 2] Replaced HuberLoss with MSELoss.  Huber discounted 30K peak errors;
          MSE squares them so the model is forced to chase weekday peaks.
  [FIX 3] lr_T_max 500 → 100.  Prevents the large val-loss spikes caused by
          the LR being nearly unchanged for the first 50+ epochs.
  [FIX 4] Post-hoc linear calibration on val set removes residual systematic
          bias before test evaluation.  No leakage, no architecture change.
  [FIX 5] NMAE / NRMSE added — MAE and RMSE normalised by range (max−min) to
          [0, 1] so they slot directly into the Combined formula as percentages.
          Combined = max(0, 100 − MAPE − NMAE×100 − NRMSE×100).

  [COMPARISON]
  After test evaluation the script automatically loads LSTM v4 results from
  src/outputs/lstm/test_metrics.csv and prints a side-by-side leaderboard table.
  The CSV path can be overridden via CFG["baseline_metrics_path"].

  [UNCHANGED from v3]
  - bidirectional=True, hidden_size=128, num_layers=2, dropout=0.2
  - Input projection (n_features → proj_dim=128) + ReLU + LayerNorm
  - Two-layer MLP head (256 → 64 → horizon)
  - seq_len=28, horizon=7, patience=50, AMP, batch_size=32
  - All lag/calendar feature engineering
"""

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
    data_path             = "data/features/feature_matrix_lstm.parquet",
    output_dir            = "src/outputs/bilstm",
    baseline_metrics_path = "src/outputs/lstm/test_metrics.csv",  # LSTM v4 results

    # Target
    target_col      = "ridership__bus_rkl",

    # Sequence
    seq_len         = 28,           # 4-week look-back
    horizon         = 7,            # forecast horizon (days)

    # Model
    proj_dim        = 128,          # input projection dim before BiLSTM
    hidden_size     = 128,          # per-direction; effective output = 128×2 = 256
    num_layers      = 2,
    dropout         = 0.2,
    bidirectional   = True,

    # Training — tuned for RTX 4050 6 GB
    batch_size      = 32,
    max_epochs      = 500,
    lr              = 5e-4,
    weight_decay    = 1e-5,
    patience        = 50,
    grad_clip       = 0.5,
    huber_delta     = 10000.0,      # UNUSED in v4 — replaced by MSELoss (FIX 2)

    # LR schedule — cosine annealing
    lr_T_max        = 100,          # FIX 3: was 500; decays to eta_min by epoch 100
    lr_eta_min      = 1e-7,

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
    Without lag features the BiLSTM receives only exogenous signals (weather,
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
        df = df.ffill().bfill()
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
# 4. DATASET
# ─────────────────────────────────────────────
class RidershipDataset(Dataset):
    """
    Sliding-window dataset.
    X : (seq_len, n_features)
    y : (horizon,)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int, horizon: int):
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

    # v4 FIX 1: Remove log1p — use raw StandardScaler only.
    # log1p compressed the 140K–260K ridership range into 0.62 log-units,
    # making peak errors look small in loss space while being huge in reality.
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
# 5. MODEL — BiLSTM
# ─────────────────────────────────────────────
class ImprovedBiLSTM(nn.Module):
    """
    Input projection → Stacked Bidirectional LSTM → MLP head.

    Architecture:
      Input (B, T, n_features)
        → InputProj  Linear(n_features, proj_dim) + ReLU + LayerNorm
        → BiLSTM(proj_dim, hidden=64, layers=2, dropout=0.3)
             output width = 64×2 = 128  (same VRAM footprint as LSTM hidden=128)
        → LayerNorm(128)
        → Dropout(0.3)
        → Linear(128, 64) + ReLU
        → Linear(64, horizon)

    Why the input projection matters
    ─────────────────────────────────
    Without it, the BiLSTM's input-to-hidden weight matrix is
    4×n_features×hidden per direction, consuming nearly all parameters on a
    single input-embedding step.  Projecting to proj_dim=128 first lets the
    recurrent weights focus on modelling temporal dynamics.
    """

    def __init__(
        self,
        input_size  : int,
        proj_dim    : int,
        hidden_size : int,
        num_layers  : int,
        dropout     : float,
        horizon     : int,
        bidirectional: bool = True,
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
        # x: (B, seq_len, n_features)
        x      = self.input_proj(x)        # (B, seq_len, proj_dim)
        out, _ = self.lstm(x)              # (B, seq_len, D*hidden)
        last   = out[:, -1, :]             # last time-step
        last   = self.norm(last)
        last   = self.dropout(last)
        return self.head(last)             # (B, horizon)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 6. TRAINING & EVALUATION
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
    total = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        total += criterion(model(X_batch), y_batch).item() * X_batch.size(0)
    return total / len(loader.dataset)


# ─────────────────────────────────────────────
# 7. EVALUATION METRICS
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, tgt_scaler, device, calibrator=None):
    """Return metrics in original scale (averaged across horizon steps).
    See METRICS.md for detailed definitions.

    v4 changes vs v3:
      - log1p/expm1 removed (FIX 1)
      - NMAE/NRMSE added — range-normalised to [0,1] (FIX 5)
      - Combined updated to use NMAE/NRMSE instead of mean-% (FIX 5)
      - Optional post-hoc linear calibrator applied before metrics (FIX 4)
    """
    model.eval()
    all_pred, all_true = [], []
    for X_batch, y_batch in loader:
        pred = model(X_batch.to(device)).cpu().numpy()
        all_pred.append(pred)
        all_true.append(y_batch.numpy())

    preds = np.concatenate(all_pred, axis=0)   # (N, H)
    trues = np.concatenate(all_true, axis=0)   # (N, H)

    # v4 FIX 1 (mirror): log1p removed → only inverse StandardScaler needed
    preds_inv = tgt_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_inv = tgt_scaler.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)

    # v4 FIX 4: apply linear calibration if provided
    if calibrator is not None:
        preds_inv = calibrator.predict(preds_inv.ravel().reshape(-1, 1)).reshape(preds_inv.shape)

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
    ax.set_ylabel("MSE Loss")
    ax.set_title("BiLSTM — Training & Validation Loss")
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
    ax.set_title("BiLSTM Test Set — 1-Day-Ahead Forecast vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "forecast_vs_actual.png", dpi=150)
    plt.close(fig)
    logging.info("Saved forecast_vs_actual.png")


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
def compare_with_baseline(bilstm_metrics: dict, baseline_path: str):
    """
    Load LSTM v4 test_metrics.csv and print a side-by-side leaderboard table.
    Gracefully skips if the file does not exist yet.
    """
    baseline_path = Path(baseline_path)

    if not baseline_path.exists():
        logging.warning(
            f"Baseline metrics not found at '{baseline_path}'. "
            "Run lstm_v4.py first, then re-run this script to see the comparison."
        )
        return

    baseline = pd.read_csv(baseline_path).iloc[0].to_dict()

    metrics_order = ["MAE", "RMSE", "MAE_pct", "RMSE_pct", "MAPE", "R2", "Combined"]

    # Higher-is-better metrics
    higher_better = {"R2", "Combined"}

    col_w = 14
    sep = "-" * (12 + col_w * 3)

    logging.info("")
    logging.info("=" * len(sep))
    logging.info("  MODEL COMPARISON — BiLSTM v4 vs LSTM v4 (baseline)")
    logging.info("=" * len(sep))
    logging.info(
        f"  {'Metric':<12}"
        f"{'BiLSTM v4':>{col_w}}"
        f"{'LSTM v4':>{col_w}}"
        f"{'Winner':>{col_w}}"
    )
    logging.info(sep)

    for m in metrics_order:
        bi_val = bilstm_metrics.get(m, float("nan"))
        base_val = baseline.get(m, float("nan"))

        if m in higher_better:
            winner = (
                "BiLSTM (W)"
                if bi_val > base_val
                else ("LSTM (W)" if base_val > bi_val else "TIE")
            )
        else:
            winner = (
                "BiLSTM (W)"
                if bi_val < base_val
                else ("LSTM (W)" if base_val < bi_val else "TIE")
            )

        fmt = ".4f"

        logging.info(
            f"  {m:<12}"
            f"{bi_val:>{col_w}{fmt}}"
            f"{base_val:>{col_w}{fmt}}"
            f"{winner:>{col_w}}"
        )

    logging.info(sep)
    logging.info("")


# ─────────────────────────────────────────────
# 9. MAIN
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
    model = ImprovedBiLSTM(
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
    # v4 FIX 2: MSELoss replaces HuberLoss.
    # Huber discounted 30K peak errors (by design); MSE squares every error
    # so weekday-peak misses are penalised 9× more than mid-week misses.
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )

    # v4 FIX 3: T_max=100 — LR reaches eta_min at epoch 100, eliminating the
    # large val-loss spikes that occurred when LR barely decayed in first 50 epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = cfg["lr_T_max"],   # 100
        eta_min = cfg["lr_eta_min"],
    )

    use_amp    = device.type == "cuda"
    scaler_amp = torch.amp.GradScaler(enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────────────────
    history    = {"train_loss": [], "val_loss": []}
    best_val   = float("inf")
    best_ckpt  = output_dir / "best_bilstm.pt"
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

    # ── Load best checkpoint ──────────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # ── v4 FIX 4: Post-hoc linear calibration on val set ─────────────────
    # Fit y_cal = a·ŷ + b on val predictions (original scale) to remove any
    # residual systematic bias.  No data leakage — val never seen in training.
    from sklearn.linear_model import LinearRegression as _LR

    @torch.no_grad()
    def _collect_preds(loader):
        model.eval()
        ps, ts = [], []
        for Xb, yb in loader:
            p = model(Xb.to(device)).cpu().numpy()
            ps.append(p); ts.append(yb.numpy())
        p = np.concatenate(ps); t = np.concatenate(ts)
        p_inv = tgt_scaler.inverse_transform(p.reshape(-1, 1)).reshape(p.shape)
        t_inv = tgt_scaler.inverse_transform(t.reshape(-1, 1)).reshape(t.shape)
        return p_inv, t_inv

    val_p, val_t = _collect_preds(loaders["val"])
    cal = _LR().fit(val_p.ravel().reshape(-1, 1), val_t.ravel())
    logging.info(f"Calibration  a={cal.coef_[0]:.4f}  b={cal.intercept_:.1f}")

    # ── Test evaluation ───────────────────────────────────────────────────
    test_metrics, preds_inv, trues_inv = evaluate(
        model, loaders["test"], tgt_scaler, device, calibrator=cal
    )

    logging.info("-" * 60)
    logging.info("TEST RESULTS — BiLSTM v4 (original scale, averaged over horizon)")
    for k, v in test_metrics.items():
        logging.info(f"  {k:8s}: {v:.4f}")
    logging.info("-" * 60)

    # ── Baseline comparison ───────────────────────────────────────────────
    compare_with_baseline(test_metrics, cfg["baseline_metrics_path"])

    # ── Plots & artefacts ─────────────────────────────────────────────────
    save_plots(history, preds_inv, trues_inv, output_dir)

    np.save(output_dir / "test_preds.npy", preds_inv)
    np.save(output_dir / "test_trues.npy", trues_inv)
    pd.DataFrame([test_metrics]).to_csv(output_dir / "test_metrics.csv", index=False)
    logging.info(f"All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()