"""
TGACN v4 — Temporal Graph Attention Convolutional Network
Bus Ridership Forecasting
Hardware target: 13th Gen Intel Core i5-13500HX + NVIDIA RTX 4050 (6 GB VRAM)

Architecture
────────────
  Input (B, T, F)
    → InputProj         Linear(F, N*node_dim) + ReLU + LayerNorm
    → Reshape           (B*T, N, node_dim)        ← one graph per timestep
    → GAT Block ×K      [Multi-head GAT + residual + LayerNorm] × num_gat_layers
    → Reshape           (B, T, N*node_dim)
    → Permute           (B, N*node_dim, T)         ← Conv1d expects (B, C, L)
    → TCN Block ×L      [DilatedCausalConv1d + GELU + LayerNorm + residual] × num_tcn_layers
                        dilations = [1, 2, 4, 8, …]
                        receptive field = 1 + (kernel_size−1) × Σdilations > seq_len ✓
    → Last timestep     (B, tcn_channels)
    → MLP head          tcn_channels → 64 → horizon

Why GAT over GCN (GCN-SBULSTM):
─────────────────────────────────
  Plain GCN aggregates neighbours with weights from the adjacency matrix A — the
  same weights for every sample in the batch.  GAT computes attention coefficients
  α_ij per-sample using the node features H_i and H_j, so the routing of spatial
  information adapts to the current ridership state (e.g. a weekend trough routes
  signal differently than a weekday peak).  Multi-head attention further prevents
  over-smoothing by running independent routing functions in parallel.

  The adaptive adjacency A (Graph WaveNet style) is incorporated as a log-prior on
  the attention logits:
      e_ij = LeakyReLU( a^T [W·h_i ‖ W·h_j] ) + log(A_ij + ε)
      α_ij = softmax_j( e_ij )
  This means A shapes which edges exist while attention weights refine how much
  each existing edge contributes given the current node features.

Why TCN over LSTM (GCN-SBULSTM):
──────────────────────────────────
  TCN processes all timesteps in parallel (no sequential hidden state), so it
  trains faster on the RTX 4050.  Stacking dilations [1,2,4,8] gives a receptive
  field of 1 + 2*(1+2+4+8)=31 days with kernel_size=3 — larger than seq_len=28,
  so every output timestep sees the full look-back window.  Residual connections
  prevent gradient vanishing across deep stacks.

Adaptive Adjacency (Graph WaveNet style — same as GCN-SBULSTM):
─────────────────────────────────────────────────────────────────
  A = softmax( ReLU( E1 @ E2ᵀ ) ),  E1, E2 ∈ R^{N × embed_dim}
  No ground-truth KL transit topology needed — graph learned end-to-end.

v4 conventions (identical pipeline to LSTM v4, BiLSTM v4, TPA-LSTM v4, GCN-SBULSTM v4):
  [FIX 1] No log1p / expm1 — StandardScaler only (linear target scale).
  [FIX 2] MSELoss — forces model to chase weekday peaks (Huber discounted them).
  [FIX 3] lr_T_max=100 — LR decays to eta_min by epoch 100, removes early spikes.
  [FIX 4] Post-hoc OLS calibration on val set — removes residual systematic bias.
  [FIX 5] NMAE / NRMSE range-normalised to [0,1]; Combined uses them.
  [CMP]   Loads LSTM v4 test_metrics.csv and prints side-by-side leaderboard.
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
    data_path             = "data/features/feature_matrix_lstm.parquet",
    output_dir            = "src/outputs/tgacn",
    baseline_metrics_path = "src/outputs/lstm/test_metrics.csv",  # LSTM v4 baseline

    # Target
    target_col      = "ridership__bus_rkl",

    # Sequence
    seq_len         = 28,       # 4-week look-back
    horizon         = 7,        # 1-week forecast

    # Graph (adaptive adjacency — same as GCN-SBULSTM)
    num_nodes       = 16,       # virtual spatial nodes
    node_dim        = 8,        # feature dim per node → spatial dim = 16×8 = 128
    adj_embed_dim   = 10,       # node embedding dim for adaptive adjacency

    # GAT
    gat_heads       = 4,        # number of attention heads per GAT layer
                                # head_dim = node_dim // gat_heads = 8//4 = 2
                                # concat → 4×2 = 8 = node_dim → residual works ✓
    num_gat_layers  = 2,        # stacked GAT blocks
    gat_dropout     = 0.1,      # attention coefficient dropout

    # TCN
    tcn_channels    = 128,      # = num_nodes × node_dim; kept constant across layers
    num_tcn_layers  = 4,        # dilations: [1, 2, 4, 8]
                                # receptive field = 1 + (3-1)*(1+2+4+8) = 31 > seq_len=28 ✓
    tcn_kernel_size = 3,
    tcn_dropout     = 0.1,

    # Training
    batch_size      = 32,
    max_epochs      = 500,
    lr              = 5e-4,
    weight_decay    = 1e-5,
    patience        = 50,
    grad_clip       = 0.5,
    huber_delta     = 10000.0,  # UNUSED in v4 — replaced by MSELoss (FIX 2)

    # LR schedule
    lr_T_max        = 100,      # FIX 3: was 500; LR reaches eta_min at epoch 100
    lr_eta_min      = 1e-7,

    # Data split
    train_ratio     = 0.70,
    val_ratio       = 0.15,
    # test_ratio    = 0.15 (remainder)

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
        logging.info("GPU not available — running on CPU")
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
    # Autoregressive lags — give the model direct knowledge of the weekly cycle
    for lag in [1, 7, 14, 21]:
        df[f"_tgt_lag_{lag}"] = df[target_col].shift(lag)

    # 7-day rolling statistics (shifted by 1 to prevent leakage)
    shifted = df[target_col].shift(1)
    df["_tgt_roll7_mean"] = shifted.rolling(7, min_periods=7).mean()
    df["_tgt_roll7_std"]  = shifted.rolling(7, min_periods=1).std().fillna(0)

    # Cyclical calendar encoding — perfectly periodic, zero NaN
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
        raise ValueError(f"Unknown impute_method: {method!r}")

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
# 4. DATASET & SPLITS
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
    T         = len(df)
    n_train   = int(T * cfg["train_ratio"])
    n_val     = int(T * cfg["val_ratio"])
    train_end = n_train
    val_end   = n_train + n_val

    X_all = df[feature_cols].values
    y_all = df[cfg["target_col"]].values

    # Scale on TRAIN ONLY — prevents data leakage
    feat_scaler = StandardScaler()
    feat_scaler.fit(X_all[:train_end])

    # v4 FIX 1: No log1p — raw StandardScaler only.
    # log1p compressed 140K–260K ridership into 0.62 log-units, making 30K
    # peak errors look negligible in loss space → systematic under-prediction.
    tgt_scaler = StandardScaler()
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
# 5. TGACN MODEL
# ─────────────────────────────────────────────

class AdaptiveAdjacency(nn.Module):
    """
    Learnable graph adjacency (Graph WaveNet style).

        A = softmax( ReLU( E1 @ E2ᵀ ) )   shape: (N, N)

    All N×N entries are positive after softmax — used as a log-prior
    bias in GAT attention logits (see GATLayer below).
    """
    def __init__(self, num_nodes: int, embed_dim: int = 10):
        super().__init__()
        self.E1 = nn.Embedding(num_nodes, embed_dim)
        self.E2 = nn.Embedding(num_nodes, embed_dim)

    def forward(self, device: torch.device) -> torch.Tensor:
        idx = torch.arange(self.E1.num_embeddings, device=device)
        A   = F.softmax(F.relu(self.E1(idx) @ self.E2(idx).T), dim=-1)  # (N, N)
        return A


class GATLayer(nn.Module):
    """
    Multi-head Graph Attention layer.

    For each head h ∈ {1…H}:
      Z_h  = H @ W_h                                  (BT, N, head_dim)
      e_ij = LeakyReLU( a_h^T [Z_h_i ‖ Z_h_j] )
           + log(A_ij + ε)          ← adaptive adj as log-prior on edges
      α_ij = softmax_j( e_ij )      with optional attention dropout
      H'_h  = Σ_j α_ij Z_h_j       (BT, N, head_dim)

    Outputs are concatenated across heads → (BT, N, H*head_dim) = (BT, N, node_dim)
    then a LayerNorm residual is applied.

    Why log(A) as prior?
      A is already a (N,N) probability distribution (softmax output, all >0).
      Adding log(A_ij) to the attention logit e_ij is equivalent to
      multiplying the un-normalised attention by A_ij before softmax — it
      biases attention toward edges that the adaptive adjacency deems important,
      while still allowing the feature-based GAT scores to override weak edges.
    """
    def __init__(
        self,
        node_dim    : int,
        num_heads   : int,
        attn_dropout: float = 0.1,
        leaky_slope : float = 0.2,
    ):
        super().__init__()
        assert node_dim % num_heads == 0, (
            f"node_dim ({node_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads  = num_heads
        self.head_dim   = node_dim // num_heads
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)

        # Per-head linear projections (bias=False — LayerNorm handles shift)
        self.W = nn.Linear(node_dim, node_dim, bias=False)

        # Per-head attention vector a_h ∈ R^{2*head_dim}
        self.attn_vec = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

        self.attn_drop  = nn.Dropout(attn_dropout)
        self.norm       = nn.LayerNorm(node_dim)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H : (BT, N, node_dim)
            A : (N,  N)           adaptive adjacency (softmax output, all > 0)
        Returns:
            H': (BT, N, node_dim) with residual + LayerNorm
        """
        BT, N, D = H.shape
        residual  = H

        # Linear projection, then split into heads
        Z = self.W(H)                                              # (BT, N, D)
        Z = Z.view(BT, N, self.num_heads, self.head_dim)          # (BT, N, H, d)

        # ── Attention logits ────────────────────────────────────────────
        # e_ij = a_h^T [Z_h_i ‖ Z_h_j] for each head h
        Zi = Z.unsqueeze(2).expand(-1, -1, N, -1, -1)            # (BT, N, N, H, d)
        Zj = Z.unsqueeze(1).expand(-1, N, -1, -1, -1)            # (BT, N, N, H, d)
        ZZ = torch.cat([Zi, Zj], dim=-1)                          # (BT, N, N, H, 2d)

        # a_h dot product: a_h ∈ (H, 2d), ZZ ∈ (BT, N, N, H, 2d)
        e = (ZZ * self.attn_vec).sum(dim=-1)                      # (BT, N, N, H)
        e = self.leaky_relu(e)

        # ── Adaptive adjacency log-prior ────────────────────────────────
        # A ∈ (N, N) with all-positive entries (softmax output).
        # Adding log(A + eps) biases attention toward learned graph edges.
        log_A = torch.log(A + 1e-8)                               # (N, N)
        e = e + log_A.unsqueeze(0).unsqueeze(-1)                  # broadcast (BT, N, N, H)

        # ── Softmax + dropout ───────────────────────────────────────────
        alpha = F.softmax(e, dim=2)                                # (BT, N, N, H)
        alpha = self.attn_drop(alpha)

        # ── Aggregate neighbour features ────────────────────────────────
        # out_h_i = Σ_j alpha_ij_h * Z_j_h
        Z_perm = Z.permute(0, 2, 1, 3)                            # (BT, H, N, d)
        # alpha: (BT, N, N, H) → permute to (BT, H, N, N)
        a_perm = alpha.permute(0, 3, 1, 2)                        # (BT, H, N, N)
        out    = torch.matmul(a_perm, Z_perm)                     # (BT, H, N, d)
        out    = out.permute(0, 2, 1, 3).contiguous()             # (BT, N, H, d)
        out    = out.view(BT, N, D)                                # (BT, N, D) — concat heads

        return self.norm(out + residual)


class GATBlock(nn.Module):
    """K stacked GATLayer modules."""
    def __init__(self, node_dim: int, num_heads: int, num_layers: int, attn_dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            GATLayer(node_dim, num_heads, attn_dropout)
            for _ in range(num_layers)
        ])

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            H = layer(H, A)
        return H


class TCNLayer(nn.Module):
    """
    One dilated causal Conv1d layer with residual connection.

    Causal padding: pad (kernel_size−1)*dilation steps on the LEFT only.
    This ensures H[t] depends only on H[t'], t' ≤ t — no future leakage.

    Architecture per layer:
        x → CausalPad → Conv1d(C, C, k, dilation=d)
          → LayerNorm → GELU → Dropout
          → + residual
          → output

    GELU is used instead of ReLU because it has non-zero gradient everywhere,
    improving gradient flow in deep TCN stacks.
    """
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad_len = (kernel_size - 1) * dilation
        self.conv    = nn.Conv1d(
            channels, channels,
            kernel_size = kernel_size,
            dilation    = dilation,
            padding     = 0,      # manual causal padding — see forward()
        )
        self.norm    = nn.LayerNorm(channels)
        self.act     = nn.GELU()
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T)  — channels-first for Conv1d
        """
        residual = x
        # Left-pad only → causal
        out = F.pad(x, (self.pad_len, 0))    # (B, C, T + pad)
        out = self.conv(out)                  # (B, C, T)  — same length as input
        # LayerNorm on channel dim: (B, C, T) → (B, T, C) → norm → (B, C, T)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out)
        out = self.drop(out)
        return out + residual


class TCNBlock(nn.Module):
    """
    Stack of TCNLayer with exponentially increasing dilations [1, 2, 4, 8, …].

    Receptive field = 1 + (kernel_size − 1) × Σ_{l=0}^{L-1} 2^l
                    = 1 + (k-1) × (2^L − 1)

    With k=3, L=4:  1 + 2 × 15 = 31 > seq_len=28 ✓
    Every output timestep sees the full 28-day look-back window.
    """
    def __init__(
        self,
        channels   : int,
        num_layers : int,
        kernel_size: int,
        dropout    : float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TCNLayer(channels, kernel_size, dilation=2 ** l, dropout=dropout)
            for l in range(num_layers)
        ])
        receptive_field = 1 + (kernel_size - 1) * sum(2 ** l for l in range(num_layers))
        logging.debug(
            f"TCNBlock: {num_layers} layers, kernel={kernel_size}, "
            f"dilations={[2**l for l in range(num_layers)]}, "
            f"receptive_field={receptive_field}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, T)"""
        for layer in self.layers:
            x = layer(x)
        return x


class TGACNModel(nn.Module):
    """
    Full TGACN pipeline:

      (B, T, F)
        → InputProj     Linear(F, N*d) + ReLU + LayerNorm   [spatial tokeniser]
        → reshape       (B*T, N, d)            one graph per timestep
        → GATBlock      K × [multi-head GAT + residual + LayerNorm]
        → reshape       (B, T, N*d)
        → permute       (B, N*d, T)            Conv1d channels-first
        → TCNBlock      L × [DilatedCausalConv1d + LayerNorm + GELU + residual]
        → last step     (B, N*d)               = (B, tcn_channels)
        → MLP head      tcn_channels → 64 → horizon
    """
    def __init__(
        self,
        input_size    : int,
        num_nodes     : int,
        node_dim      : int,
        adj_embed_dim : int,
        gat_heads     : int,
        num_gat_layers: int,
        gat_dropout   : float,
        tcn_channels  : int,
        num_tcn_layers: int,
        tcn_kernel_size: int,
        tcn_dropout   : float,
        horizon       : int,
    ):
        super().__init__()
        self.num_nodes   = num_nodes
        self.node_dim    = node_dim
        spatial_dim      = num_nodes * node_dim   # = tcn_channels = 128

        assert spatial_dim == tcn_channels, (
            f"num_nodes × node_dim ({spatial_dim}) must equal tcn_channels ({tcn_channels}) "
            "for a clean skip-connection between the GAT output and TCN input."
        )

        # ── Spatial tokeniser ─────────────────────────────────────────────
        # Projects F raw features → N virtual node embeddings of size node_dim.
        # LayerNorm stabilises the distribution entering the GAT attention.
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, spatial_dim),
            nn.ReLU(),
            nn.LayerNorm(spatial_dim),
        )

        # ── Spatial graph ─────────────────────────────────────────────────
        self.adj = AdaptiveAdjacency(num_nodes, adj_embed_dim)
        self.gat = GATBlock(node_dim, gat_heads, num_gat_layers, gat_dropout)

        # ── Temporal convolutional network ────────────────────────────────
        self.tcn = TCNBlock(tcn_channels, num_tcn_layers, tcn_kernel_size, tcn_dropout)

        # ── Forecast head ─────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(tcn_channels, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, F)
        returns : (B, horizon)
        """
        B, T, _ = x.shape

        # 1. Project raw features to node space
        h = self.input_proj(x)                                     # (B, T, N*d)

        # 2. Spatial encoding via multi-head GAT (all timesteps in parallel)
        h = h.reshape(B * T, self.num_nodes, self.node_dim)        # (BT, N, d)
        A = self.adj(x.device)                                     # (N, N)
        h = self.gat(h, A)                                         # (BT, N, d)
        h = h.reshape(B, T, self.num_nodes * self.node_dim)        # (B, T, N*d)

        # 3. Temporal encoding via dilated causal TCN
        h = h.permute(0, 2, 1)                                     # (B, N*d, T)
        h = self.tcn(h)                                            # (B, N*d, T)

        # 4. Take the last (most recent) timestep
        last = h[:, :, -1]                                         # (B, N*d)

        return self.head(last)                                      # (B, horizon)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 6. TRAINING LOOP
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
# 7. EVALUATION METRICS
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, tgt_scaler, device, calibrator=None):
    """Return metrics in original scale (averaged across horizon steps).
    See METRICS.md for detailed definitions.

    v4 changes:
      - log1p / expm1 removed (FIX 1)
      - NMAE / NRMSE range-normalised to [0, 1] (FIX 5)
      - Combined = max(0, 100 − MAPE − NMAE×100 − NRMSE×100) (FIX 5)
      - Optional post-hoc OLS calibrator (FIX 4)
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
        preds_inv = calibrator.predict(
            preds_inv.ravel().reshape(-1, 1)
        ).reshape(preds_inv.shape)

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
    # Loss curves
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="Train loss")
    ax.plot(history["val_loss"],   label="Val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("TGACN — Training & Validation Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close(fig)
    logging.info("Saved loss_curves.png")

    # Forecast vs actual (first 90 test samples, horizon step 0)
    n = min(90, len(trues))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(trues[:n, 0], label="Actual",    linewidth=1.5)
    ax.plot(preds[:n, 0], label="Predicted", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Sample"); ax.set_ylabel("bus_rkl (original scale)")
    ax.set_title("TGACN — Test Set 1-Day-Ahead Forecast vs Actual")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "forecast_vs_actual.png", dpi=150)
    plt.close(fig)
    logging.info("Saved forecast_vs_actual.png")


# ─────────────────────────────────────────────
# 9. COMPARISON WITH BASELINE
# ─────────────────────────────────────────────
def compare_with_baseline(tgacn_metrics: dict, baseline_path: str):
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
    logging.info("  MODEL COMPARISON — TGACN v4 vs LSTM v4 (baseline)")
    logging.info("=" * len(sep))
    logging.info(
        f"  {'Metric':<12}"
        f"{'TGACN v4':>{col_w}}"
        f"{'LSTM v4':>{col_w}}"
        f"{'Winner':>{col_w}}"
    )
    logging.info(sep)

    for m in metrics_order:
        bi_val = tgacn_metrics.get(m, float("nan"))
        base_val = baseline.get(m, float("nan"))

        if m in higher_better:
            winner = (
                "TGACN (W)"
                if bi_val > base_val
                else ("LSTM (W)" if base_val > bi_val else "TIE")
            )
        else:
            winner = (
                "TGACN (W)"
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
# 10. MAIN
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
    model = TGACNModel(
        input_size     = n_features,
        num_nodes      = cfg["num_nodes"],
        node_dim       = cfg["node_dim"],
        adj_embed_dim  = cfg["adj_embed_dim"],
        gat_heads      = cfg["gat_heads"],
        num_gat_layers = cfg["num_gat_layers"],
        gat_dropout    = cfg["gat_dropout"],
        tcn_channels   = cfg["tcn_channels"],
        num_tcn_layers = cfg["num_tcn_layers"],
        tcn_kernel_size= cfg["tcn_kernel_size"],
        tcn_dropout    = cfg["tcn_dropout"],
        horizon        = cfg["horizon"],
    ).to(device)

    logging.info(f"Model parameters: {count_params(model):,}")
    logging.info(model)

    # Log TCN receptive field for verification
    rf = 1 + (cfg["tcn_kernel_size"] - 1) * sum(
        2 ** l for l in range(cfg["num_tcn_layers"])
    )
    logging.info(
        f"TCN receptive field: {rf} days  "
        f"(dilations {[2**l for l in range(cfg['num_tcn_layers'])]}, "
        f"kernel={cfg['tcn_kernel_size']})  "
        f"{'(OK) covers seq_len' if rf >= cfg['seq_len'] else '(BAD) WARNING: shorter than seq_len'}"
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    # v4 FIX 2: MSELoss replaces HuberLoss.
    # Huber discounted 30K peak errors by design; MSE squares them so the
    # GAT attention routing is trained to handle high-amplitude weekday peaks.
    criterion = nn.MSELoss()

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )

    # ── LR Schedule ───────────────────────────────────────────────────────
    # v4 FIX 3: T_max=100 — LR decays to eta_min by epoch 100.
    # T_max=500 kept LR ~4.9e-4 for the first 50+ epochs, causing the
    # large val-loss spikes visible in loss curves of all v3 models.
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
    best_ckpt  = output_dir / "best_tgacn.pt"
    no_improve = 0

    logging.info("-" * 60)
    logging.info("Starting TGACN training …")
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

    # ── Load best checkpoint ──────────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # ── v4 FIX 4: Post-hoc linear calibration on val set ─────────────────
    # y_cal = a·ŷ + b fitted on val predictions in original scale.
    # Removes any residual systematic bias (e.g. consistent under-prediction
    # of weekday peaks) before the final test evaluation.
    # No data leakage — val set is never seen during training.
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
    logging.info("TEST RESULTS — TGACN v4 (original scale, averaged over horizon)")
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