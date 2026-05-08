"""
TPA-LSTM v4 — Temporal Pattern Attention LSTM
Based on: Shih et al., 2019 "Temporal Pattern Attention for Multivariate Time Series Prediction"
Hardware target: 13th Gen Intel Core i5-13500HX + NVIDIA RTX 4050 (6 GB VRAM)

Architecture (UNCHANGED from v3):
  [MODEL]
  1. TPAAttention module — row-wise depthwise Conv1D over all LSTM hidden
     states H ∈ (B, T, hidden), producing H_C ∈ (B, hidden, num_filters).
     Sigmoid attention α ∈ (B, hidden) is scored via h_T.  Context vector
     c ∈ (B, num_filters) is concatenated with h_T before the MLP head.
  2. MLP head input dim: hidden + num_filters → 64 → horizon.
  3. num_filters=32, output_dir = src/outputs/tpa_lstm.

v4 fixes (identical to LSTM v4 & BiLSTM v4 — required for fair comparison):
  [FIX 1] Removed log1p / expm1.  Target is StandardScaler-only (linear scale).
          log1p compressed the 140K–260K ridership swing into 0.62 log-units,
          causing systematic peak under-prediction in every horizon step.
  [FIX 2] Replaced HuberLoss with MSELoss.  Huber discounted 30K peak errors;
          MSE squares them, forcing TPA attention to focus on high-amplitude days.
  [FIX 3] lr_T_max 500 → 100.  Removes the unstable high-LR phase (epochs 0-50)
          that caused large val-loss spikes.
  [FIX 4] Post-hoc linear calibration on val set removes residual systematic
          bias before test evaluation.  No leakage, no architecture change.
  [FIX 5] NMAE / NRMSE added — range-normalised to [0, 1].
          Combined = max(0, 100 − MAPE − NMAE×100 − NRMSE×100).

  [COMPARISON]
  After test evaluation the script automatically loads LSTM v4 results from
  src/outputs/lstm/test_metrics.csv and prints a side-by-side leaderboard.
  Path can be overridden via CFG["baseline_metrics_path"].
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
    data_path             = "data/features/feature_matrix_lstm.parquet",
    output_dir            = "src/outputs/tpa_lstm",
    baseline_metrics_path = "src/outputs/lstm/test_metrics.csv",  # LSTM v4 baseline

    # Target
    target_col      = "ridership__bus_rkl",

    # Sequence
    seq_len         = 28,
    horizon         = 7,

    # Model
    proj_dim        = 128,
    hidden_size     = 128,
    num_layers      = 2,
    dropout         = 0.2,
    bidirectional   = False,
    num_filters     = 32,           # TPA: CNN filters per hidden dim

    # Training
    batch_size      = 32,
    max_epochs      = 500,
    lr              = 5e-4,
    weight_decay    = 1e-5,
    patience        = 50,
    grad_clip       = 0.5,
    huber_delta     = 10000.0,      # UNUSED in v4 — replaced by MSELoss (FIX 2)

    # LR schedule
    lr_T_max        = 100,          # FIX 3: was 500; decays to eta_min by epoch 100
    lr_eta_min      = 1e-7,

    # Data split
    train_ratio     = 0.70,
    val_ratio       = 0.15,

    # Imputation
    impute_method   = "ffill",

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
    torch.backends.cudnn.benchmark     = False


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
    for lag in [1, 7, 14, 21]:
        df[f"_tgt_lag_{lag}"] = df[target_col].shift(lag)

    shifted = df[target_col].shift(1)
    df["_tgt_roll7_mean"] = shifted.rolling(7, min_periods=7).mean()
    df["_tgt_roll7_std"]  = shifted.rolling(7, min_periods=1).std().fillna(0)

    dow = df.index.dayofweek.astype(float)
    df["_dow_sin"]    = np.sin(2 * np.pi * dow / 7)
    df["_dow_cos"]    = np.cos(2 * np.pi * dow / 7)
    df["_month_sin"]  = np.sin(2 * np.pi * df.index.month / 12)
    df["_month_cos"]  = np.cos(2 * np.pi * df.index.month / 12)
    df["_is_weekend"] = (dow >= 5).astype(float)

    return df


# ─────────────────────────────────────────────
# 3. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
def load_and_clean(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    path = Path(cfg["data_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at '{path}'.\n"
            "Run the feature engineering pipeline first."
        )

    logging.info(f"Loading {path} …")
    df = pd.read_parquet(path)
    logging.info(f"Raw shape: {df.shape}  |  NaN%: {df.isna().mean().mean()*100:.1f}")

    df = df.replace([np.inf, -np.inf], np.nan)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df = add_engineered_features(df, cfg["target_col"])

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
    T       = len(df)
    n_train = int(T * cfg["train_ratio"])
    n_val   = int(T * cfg["val_ratio"])
    train_end = n_train
    val_end   = n_train + n_val

    X_all = df[feature_cols].values
    y_all = df[cfg["target_col"]].values

    feat_scaler = StandardScaler()
    feat_scaler.fit(X_all[:train_end])

    # v4 FIX 1: Remove log1p — raw StandardScaler only.
    # log1p compressed 140K–260K ridership into 0.62 log-units, making
    # 30K peak errors look negligible in loss space.
    tgt_scaler = StandardScaler()
    tgt_scaler.fit(y_all[:train_end].reshape(-1, 1))

    X_all = feat_scaler.transform(X_all)
    y_all = tgt_scaler.transform(y_all.reshape(-1, 1)).ravel()

    seq_len = cfg["seq_len"]
    horizon = cfg["horizon"]

    datasets = {
        "train": RidershipDataset(X_all[:train_end],             y_all[:train_end],             seq_len, horizon),
        "val"  : RidershipDataset(X_all[train_end - seq_len : val_end], y_all[train_end - seq_len : val_end], seq_len, horizon),
        "test" : RidershipDataset(X_all[val_end - seq_len :],   y_all[val_end - seq_len :],   seq_len, horizon),
    }
    return datasets, feat_scaler, tgt_scaler


# ─────────────────────────────────────────────
# 5. MODEL
# ─────────────────────────────────────────────
class TPAAttention(nn.Module):
    """
    Temporal Pattern Attention (Shih et al., 2019).

    Given all LSTM hidden states H ∈ (B, T, m) and the final state
    h_T ∈ (B, m), this module:

      1. Applies a depthwise Conv1D over H^T ∈ (B, m, T) — one set of K
         filters per hidden dimension — to capture temporal patterns.
         Result: H_C ∈ (B, m, K).

      2. Scores each hidden dimension using h_T as a query:
            score_i = Σ_k [ H_C[i,k] · (W_a h_T)[k] ]
         Sigmoid (not softmax) gives sparse selection over hidden dims:
            α ∈ (B, m)  via sigmoid

      3. Computes the context as a weighted sum of H_C rows:
            c = Σ_i α_i · H_C[i, :]  ∈ (B, K)

    The caller concatenates [h_T ‖ c] before the MLP head, giving the
    head both the recurrent summary AND which temporal patterns mattered.

    Complexity vs baseline:
      Extra params ≈ hidden * num_filters (conv weights)
                   + num_filters (W_a proj)
      For hidden=128, K=32: ~4K extra params — negligible.
    """

    def __init__(self, hidden_size: int, seq_len: int, num_filters: int = 32):
        super().__init__()
        self.num_filters = num_filters

        # Depthwise Conv1D: each of the `hidden_size` dims gets its own K filters.
        # groups=hidden_size enforces row-wise (per-dim) convolution.
        # kernel_size=seq_len → full-sequence receptive field → output width = 1.
        # No padding needed: (B, m, T) ──conv──> (B, m*K, 1).
        self.conv = nn.Conv1d(
            in_channels  = hidden_size,
            out_channels = hidden_size * num_filters,
            kernel_size  = seq_len,
            groups       = hidden_size,   # depthwise
            bias         = False,
        )

        # Project h_T from hidden → num_filters space for scoring.
        self.W_a = nn.Linear(hidden_size, num_filters, bias=False)

    def forward(self, H: torch.Tensor, h_last: torch.Tensor) -> torch.Tensor:
        """
        H      : (B, T, m)   — all LSTM hidden states
        h_last : (B, m)      — final hidden state (after LayerNorm)
        returns: (B, K)      — context vector
        """
        B, T, m = H.shape
        K = self.num_filters

        # (B, m, T) → conv → (B, m*K, 1) → (B, m, K)
        Ht = H.transpose(1, 2)                     # (B, m, T)
        Hc = torch.relu(self.conv(Ht))             # (B, m*K, 1)
        Hc = Hc.squeeze(-1).view(B, m, K)          # (B, m, K)

        # Attention scores: sigmoid over hidden dimensions
        # key: (B, K);  Hc: (B, m, K) → dot → scores: (B, m)
        key    = self.W_a(h_last)                  # (B, K)
        scores = (Hc * key.unsqueeze(1)).sum(-1)   # (B, m)
        alpha  = torch.sigmoid(scores)             # (B, m)  — sparse selection

        # Context: α-weighted sum of pattern vectors
        context = (alpha.unsqueeze(-1) * Hc).sum(1)  # (B, K)
        return context


class TPALSTM(nn.Module):
    """
    TPA-LSTM for multi-step ridership forecasting.

    Architecture:
      Input (B, T, input_size)
        → InputProj  Linear → ReLU → LayerNorm        [same as baseline]
        → LSTM(proj_dim, hidden, layers, dropout)
        → LayerNorm(hidden) on h_T
        → Dropout(dropout)
        → TPAAttention → context ∈ (B, num_filters)   [NEW]
        → concat([h_T, context])  ∈ (B, hidden+K)     [NEW]
        → Linear(hidden+K, 64) → ReLU → Linear(64, horizon)
    """

    def __init__(
        self,
        input_size  : int,
        proj_dim    : int,
        hidden_size : int,
        num_layers  : int,
        dropout     : float,
        horizon     : int,
        seq_len     : int,
        num_filters : int = 32,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
        )

        self.lstm = nn.LSTM(
            input_size  = proj_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )

        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.tpa = TPAAttention(hidden_size, seq_len, num_filters)

        # Head input = h_T (hidden) + context (num_filters)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + num_filters, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_size)
        x = self.input_proj(x)          # (B, T, proj_dim)
        H, _ = self.lstm(x)             # (B, T, hidden)

        h_last = self.norm(H[:, -1, :]) # (B, hidden)
        h_drop = self.dropout(h_last)

        context = self.tpa(H, h_last)   # (B, num_filters)

        combined = torch.cat([h_drop, context], dim=-1)  # (B, hidden+K)
        return self.head(combined)      # (B, horizon)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 6. TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler_amp, device, grad_clip, use_amp):
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
    """Return metrics in original scale (averaged across horizon steps).
    See METRICS.md for detailed definitions.

    v4 changes:
      - log1p/expm1 removed (FIX 1)
      - NMAE/NRMSE added, Combined updated (FIX 5)
      - Optional post-hoc linear calibrator (FIX 4)
    """
    model.eval()
    all_pred, all_true = [], []
    for X_batch, y_batch in loader:
        pred = model(X_batch.to(device)).cpu().numpy()
        all_pred.append(pred)
        all_true.append(y_batch.numpy())

    preds = np.concatenate(all_pred, axis=0)
    trues = np.concatenate(all_true, axis=0)

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

    # v4 FIX 5: NMAE / NRMSE — range-normalised to [0, 1]
    actual_range = float(np.abs(trues_inv).max() - np.abs(trues_inv).min()) + 1e-8
    nmae  = float(mae  / actual_range)
    nrmse = float(rmse / actual_range)

    # Combined = max(0, 100 − MAPE − NMAE×100 − NRMSE×100)
    combined = float(np.clip(100.0 - (mape + nmae * 100 + nrmse * 100), 0.0, 100.0))

    return dict(
        MAE=mae, RMSE=rmse,
        NMAE=nmae, NRMSE=nrmse,
        R2=r2, MAPE=mape, Combined=combined,
    ), preds_inv, trues_inv


# ─────────────────────────────────────────────
# 8. PLOTTING
# ─────────────────────────────────────────────
def save_plots(history: dict, preds, trues, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="Train loss")
    ax.plot(history["val_loss"],   label="Val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("TPA-LSTM — Training & Validation Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    n = min(90, len(trues))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(trues[:n, 0], label="Actual",    linewidth=1.5)
    ax.plot(preds[:n, 0], label="Predicted", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Sample"); ax.set_ylabel("bus_rkl (original scale)")
    ax.set_title("TPA-LSTM Test Set — 1-Day-Ahead Forecast vs Actual")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "forecast_vs_actual.png", dpi=150)
    plt.close(fig)
    logging.info("Saved plots.")


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
def compare_with_baseline(tpa_metrics: dict, baseline_path: str):
    """
    Load LSTM v4 test_metrics.csv and print a side-by-side leaderboard.
    Skips gracefully if file does not exist yet.
    """
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        logging.warning(
            f"Baseline metrics not found at '{baseline_path}'. "
            "Run lstm_v4.py first, then re-run to see the comparison."
        )
        return

    baseline = pd.read_csv(baseline_path).iloc[0].to_dict()

    metrics_order = ["MAE", "RMSE", "NMAE", "NRMSE", "MAPE", "R2", "Combined"]
    higher_better = {"R2", "Combined"}

    col_w = 14
    sep   = "-" * (12 + col_w * 2)

    logging.info("")
    logging.info("=" * len(sep))
    logging.info("  MODEL COMPARISON — TPA-LSTM v4 vs LSTM v4 (baseline)")
    logging.info("=" * len(sep))
    logging.info(f"  {'Metric':<12}{'TPA-LSTM v4':>{col_w}}{'LSTM v4':>{col_w}}{'Winner':>{col_w}}")
    logging.info(sep)

    for m in metrics_order:
        tpa_val  = tpa_metrics.get(m, float("nan"))
        base_val = baseline.get(m, float("nan"))

        if m in higher_better:
            winner = "TPA-LSTM (W)" if tpa_val > base_val else ("LSTM (W)" if base_val > tpa_val else "TIE")
        else:
            winner = "TPA-LSTM (W)" if tpa_val < base_val else ("LSTM (W)" if base_val < tpa_val else "TIE")

        fmt = ".4f" if m in {"NMAE", "NRMSE", "R2"} else ".2f"
        logging.info(
            f"  {m:<12}{tpa_val:>{col_w}{fmt}}{base_val:>{col_w}{fmt}}{winner:>{col_w}}"
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

    model = TPALSTM(
        input_size  = n_features,
        proj_dim    = cfg["proj_dim"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        dropout     = cfg["dropout"],
        horizon     = cfg["horizon"],
        seq_len     = cfg["seq_len"],
        num_filters = cfg["num_filters"],
    ).to(device)

    logging.info(f"Model parameters: {count_params(model):,}")
    logging.info(model)

    # v4 FIX 2: MSELoss replaces HuberLoss.
    # Huber discounted large peak errors (by design); MSE squares them so the
    # TPA attention module is trained to focus on high-amplitude weekday peaks.
    criterion  = nn.MSELoss()

    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    # v4 FIX 3: T_max=100 — LR decays to eta_min by epoch 100.
    # With T_max=500 the LR barely moved for the first 50+ epochs,
    # causing the val-loss spikes visible in the loss curves.
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["lr_T_max"], eta_min=cfg["lr_eta_min"]
    )

    use_amp    = device.type == "cuda"
    scaler_amp = torch.amp.GradScaler(enabled=use_amp)

    history    = {"train_loss": [], "val_loss": []}
    best_val   = float("inf")
    best_ckpt  = output_dir / "best_model.pt"
    no_improve = 0

    logging.info("-" * 60)
    logging.info("Starting TPA-LSTM training...")
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

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "optim_state": optimizer.state_dict(), "val_loss": val_loss, "cfg": cfg},
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
    logging.info("TEST RESULTS — TPA-LSTM v4 (original scale, averaged over horizon)")
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