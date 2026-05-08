"""
GCN-SBULSTM — Bus Ridership Forecasting
Hardware target: 13th Gen Intel Core i5-13500HX + NVIDIA RTX 4050 (6 GB VRAM)

Architecture overview
─────────────────────
  Input (B, T, F)
    → InputProj   Linear(F, N*node_dim) + ReLU + LayerNorm
    → Reshape     (B*T, N, node_dim)           ← treat each timestep as a graph
    → SBU-GCN     [AdaptiveAdj + 2×GCNLayer + skip] × seq_len   (spatial)
    → Reshape     (B, T, N*node_dim)
    → LSTM        (B, T, N*node_dim) → (B, hidden)              (temporal)
    → LayerNorm + Dropout
    → MLP head    hidden → 64 → horizon

Adaptive Adjacency (Graph WaveNet style)
─────────────────────────────────────────
  Two learnable node-embedding matrices E1, E2 ∈ R^{N×embed_dim}.
  A = softmax(ReLU(E1 @ E2ᵀ))
  No ground-truth KL transit topology needed — the graph is learned end-to-end.

Spatial-Based Unit (SBU)
─────────────────────────
  Two stacked GCN layers with a residual skip:
    H' = GCN₂(GCN₁(H, A), A)  +  H
  GCN:  H_out = ReLU(LayerNorm(A H W))
  einsum formulation avoids explicit expand/bmm overhead.
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
import torch.nn.functional as F
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
    output_dir      = "src/outputs/gcn_sbu_lstm",

    # Target
    target_col      = "ridership__bus_rkl",

    # Sequence
    seq_len         = 28,
    horizon         = 7,

    # GCN graph
    num_nodes       = 16,           # virtual spatial nodes
    node_dim        = 8,            # feature dim per node  → LSTM input = 16×8 = 128
    num_gcn_layers  = 2,            # GCN layers inside SBU
    adj_embed_dim   = 10,           # adaptive adjacency embedding dim

    # LSTM (unchanged from baseline)
    hidden_size     = 128,
    num_layers      = 2,
    dropout         = 0.2,

    # Training
    batch_size      = 32,
    max_epochs      = 500,
    lr              = 5e-4,
    weight_decay    = 1e-5,
    patience        = 50,
    grad_clip       = 0.5,
    huber_delta     = 10000.0,

    # LR schedule
    lr_T_max        = 500,
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
# 1. UTILITIES  (unchanged)
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
# 2. FEATURE ENGINEERING  (unchanged)
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
# 3. DATA LOADING & PREPROCESSING  (unchanged)
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
# 4. DATASET  (unchanged)
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

    tgt_scaler = StandardScaler()
    y_all = np.log1p(y_all)
    tgt_scaler.fit(y_all[:train_end].reshape(-1, 1))

    X_all = feat_scaler.transform(X_all)
    y_all = tgt_scaler.transform(y_all.reshape(-1, 1)).ravel()

    seq_len = cfg["seq_len"]
    horizon = cfg["horizon"]

    train_X = X_all[:train_end]
    train_y = y_all[:train_end]

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
# 5. GCN-SBULSTM MODEL  ← NEW
# ─────────────────────────────────────────────
class AdaptiveAdjacency(nn.Module):
    """
    Learnable graph adjacency via two node-embedding tables (Graph WaveNet).

        A = softmax( ReLU( E1 @ E2ᵀ ) )   shape: (N, N)

    No prior transit topology needed — the graph is discovered from data.
    Kept as a module so embeddings are part of model.parameters() and
    updated by AdamW with the rest of the network.
    """
    def __init__(self, num_nodes: int, embed_dim: int = 10):
        super().__init__()
        self.E1 = nn.Embedding(num_nodes, embed_dim)
        self.E2 = nn.Embedding(num_nodes, embed_dim)

    def forward(self, device: torch.device) -> torch.Tensor:
        idx = torch.arange(self.E1.num_embeddings, device=device)
        A   = F.softmax(F.relu(self.E1(idx) @ self.E2(idx).T), dim=-1)  # (N, N)
        return A


class GCNLayer(nn.Module):
    """
    Graph convolution:  H_out = ReLU( LayerNorm( A H W ) )

    Uses einsum for the A×H multiply — avoids expand() + bmm overhead.
    LayerNorm instead of BatchNorm: stable with variable batch sizes (last
    batch of epoch, val/test batches with batch_size*2).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W    = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # H : (BT, N, in_dim)
        # A : (N,  N)
        AH  = torch.einsum("ij,bjk->bik", A, H)   # (BT, N, in_dim)
        out = self.W(AH)                            # (BT, N, out_dim)
        return F.relu(self.norm(out))


class SpatialBasedUnit(nn.Module):
    """
    Stacked GCN layers with a residual skip connection:

        H' = GCN_K( … GCN_1(H, A) … , A )  +  H

    The skip prevents gradient vanishing and lets the model fall back to
    the identity (no spatial mixing) when spatial structure is weak.
    """
    def __init__(self, node_dim: int, num_gcn_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            GCNLayer(node_dim, node_dim) for _ in range(num_gcn_layers)
        ])

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # H: (BT, N, node_dim)
        residual = H
        for layer in self.layers:
            H = layer(H, A)
        return H + residual     # (BT, N, node_dim)


class GCNSBULSTMModel(nn.Module):
    """
    Full GCN-SBULSTM pipeline:

      (B, T, F)
        → InputProj     Linear(F, N*d) + ReLU + LayerNorm        spatial tokeniser
        → reshape       (B*T, N, d)                               one graph per step
        → SBU-GCN       AdaptiveAdj + 2×GCNLayer + skip           spatial encoding
        → reshape       (B, T, N*d)                               back to sequence
        → LSTM          (B, T, N*d) → (B, hidden)                 temporal modelling
        → LayerNorm + Dropout
        → MLP head      hidden → 64 → horizon
    """

    def __init__(
        self,
        input_size    : int,
        num_nodes     : int,
        node_dim      : int,
        hidden_size   : int,
        num_layers    : int,
        num_gcn_layers: int,
        dropout       : float,
        horizon       : int,
        adj_embed_dim : int = 10,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim  = node_dim
        lstm_input     = num_nodes * node_dim

        # ── Spatial tokeniser ─────────────────────────────────────────────
        # Projects raw high-dim feature vector to num_nodes virtual stations,
        # each described by a node_dim-dimensional embedding.
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, lstm_input),
            nn.ReLU(),
            nn.LayerNorm(lstm_input),
        )

        # ── Spatial graph ─────────────────────────────────────────────────
        self.adj = AdaptiveAdjacency(num_nodes, adj_embed_dim)
        self.sbu = SpatialBasedUnit(node_dim, num_gcn_layers)

        # ── Temporal LSTM ─────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = lstm_input,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # ── Forecast head ─────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # 1. Project to node space
        h = self.input_proj(x)                                  # (B, T, N*d)

        # 2. Spatial aggregation via GCN (process all timesteps in one shot)
        h = h.reshape(B * T, self.num_nodes, self.node_dim)     # (BT, N, d)
        A = self.adj(x.device)                                   # (N, N)
        h = self.sbu(h, A)                                       # (BT, N, d)
        h = h.reshape(B, T, self.num_nodes * self.node_dim)     # (B, T, N*d)

        # 3. Temporal modelling
        out, _ = self.lstm(h)                                    # (B, T, hidden)
        last   = out[:, -1, :]                                   # (B, hidden)
        last   = self.norm(last)
        last   = self.dropout(last)

        return self.head(last)                                   # (B, horizon)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 6. TRAINING & EVALUATION  (unchanged)
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
        total_loss += criterion(model(X_batch), y_batch).item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


# ─────────────────────────────────────────────
# 7. EVALUATION METRICS  (unchanged)
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, tgt_scaler, device):
    model.eval()
    all_pred, all_true = [], []
    for X_batch, y_batch in loader:
        pred = model(X_batch.to(device)).cpu().numpy()
        all_pred.append(pred)
        all_true.append(y_batch.numpy())

    preds = np.concatenate(all_pred, axis=0)
    trues = np.concatenate(all_true, axis=0)

    # Inverse-scale across the horizon
    # Correct order: 1. Inverse StandardScaler -> 2. expm1 (reverse log1p)
    preds_scaled_inv = tgt_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_scaled_inv = tgt_scaler.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)
    
    preds_inv = np.expm1(preds_scaled_inv)
    trues_inv = np.expm1(trues_scaled_inv)

    mae  = mean_absolute_error(trues_inv.ravel(), preds_inv.ravel())
    rmse = math.sqrt(mean_squared_error(trues_inv.ravel(), preds_inv.ravel()))
    r2   = r2_score(trues_inv.ravel(), preds_inv.ravel())
    mape = np.mean(
        np.abs((trues_inv - preds_inv) / (np.abs(trues_inv) + 1e-8))
    ) * 100

    mean_actual = np.abs(trues_inv).mean() + 1e-8
    mae_pct     = (mae  / mean_actual) * 100
    rmse_pct    = (rmse / mean_actual) * 100
    combined    = float(np.clip(100.0 - (mape + mae_pct + rmse_pct), 0.0, 100.0))

    return dict(MAE=mae, RMSE=rmse, R2=r2, MAPE=mape, Combined=combined), preds_inv, trues_inv


# ─────────────────────────────────────────────
# 8. PLOTTING  (unchanged)
# ─────────────────────────────────────────────
def save_plots(history: dict, preds, trues, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="Train loss")
    ax.plot(history["val_loss"],   label="Val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.set_title("GCN-SBULSTM — Training & Validation Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close(fig)
    logging.info("Saved loss_curves.png")

    n = min(90, len(trues))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(trues[:n, 0], label="Actual",    linewidth=1.5)
    ax.plot(preds[:n, 0], label="Predicted", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Sample"); ax.set_ylabel("bus_rkl (original scale)")
    ax.set_title("GCN-SBULSTM — Test Set 1-Day-Ahead Forecast vs Actual")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "forecast_vs_actual.png", dpi=150)
    plt.close(fig)
    logging.info("Saved forecast_vs_actual.png")


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
    model = GCNSBULSTMModel(
        input_size     = n_features,
        num_nodes      = cfg["num_nodes"],
        node_dim       = cfg["node_dim"],
        hidden_size    = cfg["hidden_size"],
        num_layers     = cfg["num_layers"],
        num_gcn_layers = cfg["num_gcn_layers"],
        dropout        = cfg["dropout"],
        horizon        = cfg["horizon"],
        adj_embed_dim  = cfg["adj_embed_dim"],
    ).to(device)

    logging.info(f"Model parameters: {count_params(model):,}")
    logging.info(model)

    # ── Loss, optimiser, scheduler ────────────────────────────────────────
    criterion = nn.HuberLoss(delta=cfg["huber_delta"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = cfg["lr_T_max"],
        eta_min = cfg["lr_eta_min"],
    )

    use_amp    = device.type == "cuda"
    scaler_amp = torch.amp.GradScaler(enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────────────────
    history    = {"train_loss": [], "val_loss": []}
    best_val   = float("inf")
    best_ckpt  = output_dir / "best_model.pt"
    no_improve = 0

    logging.info("-" * 60)
    logging.info("Starting GCN-SBULSTM training …")
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

    test_metrics, preds_inv, trues_inv = evaluate(
        model, loaders["test"], tgt_scaler, device
    )

    logging.info("-" * 60)
    logging.info("TEST RESULTS (original scale, averaged over horizon)")
    for k, v in test_metrics.items():
        logging.info(f"  {k:8s}: {v:.4f}")
    logging.info("-" * 60)

    save_plots(history, preds_inv, trues_inv, output_dir)

    np.save(output_dir / "test_preds.npy", preds_inv)
    np.save(output_dir / "test_trues.npy", trues_inv)

    metrics_df = pd.DataFrame([test_metrics])
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)
    logging.info(f"All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()