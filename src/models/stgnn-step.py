"""
STGNN-STEP Enhanced v4 — Spatio-Temporal Graph Neural Network
with Scalable Transferable Enhanced Pre-training
Bus Ridership Forecasting
Hardware target: 13th Gen Intel Core i5-13500HX + NVIDIA RTX 4050 (6 GB VRAM)

Based on: Shao et al., 2022 — "Pre-training Enhanced Spatial-temporal Graph
Neural Network for Multivariate Time Series Forecasting" (KDD 2022)

Architecture
────────────
  Input X ∈ (B, T, F)
    ↓
  [Stage 1 — Feature Projection]
    InputProj   Linear(F, d_model) + ReLU + LayerNorm
                d_model = N × node_dim = 16 × 8 = 128
    ↓
  [Stage 2 — Patch Encoding  ← STEP's core insight]
    PatchSplit  T → P non-overlapping patches of size S
                T=28, S=4 → P=7  (one patch ≈ one work-week)
    PatchProj   Linear(S × d_model, N × node_dim) per patch
                → (B, P, N, node_dim)
    PosEnc      learnable patch position embedding (P, N, node_dim)
    ↓
  [Stage 3 — Dual Graph Construction  ← Enhancement]
    A_adaptive  Graph WaveNet — softmax(ReLU(E1 @ E2ᵀ))  shape (N, N)
                Batch-invariant structural prior, learned end-to-end.
    A_semantic  Computed per sample from mean-pooled patch embeddings Z̄:
                Z̄ = mean over P of patch reps → (B, N, node_dim)
                A_sem = softmax( Z̄_norm @ Z̄_normᵀ )   shape (B, N, N)
                Captures which virtual nodes show similar temporal patterns
                in the current input window — adapts to weekday/weekend context.
    A_dual      gated: σ(gate) · A_adaptive + (1-σ(gate)) · A_semantic
                gate ∈ R is a single learnable scalar — lets the model balance
                the structural prior vs data-driven similarity per training step.
    ↓
  [Stage 4 — Spatio-Temporal Transformer Blocks × L  ← Enhancement]
    Each block:
      TemporalAttn    Multi-head self-attention over the P patch dimension,
                      applied independently per node.
                      Input: reshape (B, P, N, d) → (B*N, P, d)
                      MHA → add & norm → back → (B, P, N, d)
                      Captures intra-node temporal rhythms (weekly seasonality)
                      at the patch level without the sequential bottleneck of LSTM.
      SpatialGCN      A_dual ⊗ H — graph convolution with dual adjacency.
                      Applied per patch: (B, P, N, d) → (B*P, N, d)
                      H_out = LayerNorm( A_dual_expanded @ H @ W )  + H
                      Propagates spatial signal across all P patches at once.
      FFN             Two-layer MLP with GELU, applied independently per (patch, node).
                      FFN(x) = Linear(4d → d)(GELU(Linear(d → 4d)(x))) + x
    ↓
  [Stage 5 — Output]
    LastPatch   Take patch index P-1 (most recent week) → (B, N, node_dim)
    Flatten     (B, N × node_dim) = (B, d_model)
    MLPHead     d_model → 64 → horizon

Why patches beat raw timestep processing:
──────────────────────────────────────────
  Processing individual days forces the model to attend over T=28 positions.
  Grouping into P=7 weekly patches reduces the attention sequence length 4×,
  dramatically reducing O(T²) attention cost while naturally aligning with the
  dominant weekly periodicity in bus ridership.  Each patch aggregates one
  work-week of signal, so the positional encoding indexes over weekly rhythm
  rather than individual days — a much cleaner inductive bias for this dataset.

Why dual graph beats single adaptive adjacency (TGACN, GCN-SBULSTM):
───────────────────────────────────────────────────────────────────────
  A_adaptive is a fixed structural prior: it models "these virtual nodes always
  tend to be correlated".  It cannot differentiate between a Monday morning (high
  ridership, strong inter-node correlation) and a Sunday afternoon (low ridership,
  weak correlation).  A_semantic is recomputed each forward pass from the actual
  patch embeddings, so the graph topology adapts to the current context window.
  The gated combination lets the model learn how much to trust the structural prior
  vs the data-driven similarity — effectively a learned ensemble of two graphs.

Why interleaved Spatial-Temporal beats serial GAT→TCN (TGACN):
────────────────────────────────────────────────────────────────
  TGACN applies GAT to all timesteps, then TCN on the result.  If the spatial
  aggregation is imperfect, the temporal model sees corrupted representations with
  no way to correct them.  Interleaving (temporal attn → spatial GCN) in each
  block allows the spatial step to correct the representation after every temporal
  attention layer, with residuals preserving gradients throughout.

v4 conventions (identical pipeline to all other models — required for fair comparison):
  [FIX 1] No log1p / expm1 — StandardScaler only (linear target scale).
  [FIX 2] MSELoss — forces model to chase weekday peaks.
  [FIX 3] lr_T_max=100 — LR decays to eta_min by epoch 100.
  [FIX 4] Post-hoc OLS calibration on val set.
  [FIX 5] NMAE / NRMSE range-normalised to [0, 1]; Combined uses them.
  [CMP]   Loads LSTM v4 test_metrics.csv; prints side-by-side leaderboard.
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
    output_dir            = "src/outputs/stgnn_step",
    baseline_metrics_path = "src/outputs/lstm/test_metrics.csv",

    # Target
    target_col      = "ridership__bus_rkl",

    # Sequence
    seq_len         = 28,       # look-back window: 4 weeks
    horizon         = 7,        # forecast horizon: 1 week
    patch_size      = 4,        # days per patch: 28 / 4 = 7 patches
                                # patch ≈ one work-week → natural periodicity

    # Graph (same virtual node layout as GCN-SBULSTM / TGACN for fair comparison)
    num_nodes       = 16,       # virtual spatial nodes N
    node_dim        = 8,        # feature dim per node  → d_model = 16×8 = 128
    adj_embed_dim   = 10,       # embedding dim for adaptive adjacency

    # Spatio-Temporal blocks
    num_st_layers   = 3,        # number of ST-Transformer blocks
    num_heads       = 4,        # temporal attention heads (d_model // heads = 32)
    ffn_mult        = 4,        # FFN hidden = ffn_mult × d_model inside each block
    st_dropout      = 0.1,      # dropout inside ST blocks

    # Training
    batch_size      = 32,
    max_epochs      = 500,
    lr              = 5e-4,
    weight_decay    = 1e-5,
    patience        = 50,
    grad_clip       = 0.5,
    huber_delta     = 10000.0,  # UNUSED in v4 — replaced by MSELoss (FIX 2)

    # LR schedule
    lr_T_max        = 100,      # FIX 3: was 500; LR reaches eta_min by epoch 100
    lr_eta_min      = 1e-7,

    # Data split
    train_ratio     = 0.70,
    val_ratio       = 0.15,

    # Imputation
    impute_method   = "ffill",

    # Reproducibility
    seed            = 42,
)

# Derived constants — computed once here, used in assertions inside model
_D_MODEL   = CFG["num_nodes"] * CFG["node_dim"]          # 128
_N_PATCHES = CFG["seq_len"]   // CFG["patch_size"]        # 7
assert CFG["seq_len"] % CFG["patch_size"] == 0, (
    f"seq_len ({CFG['seq_len']}) must be divisible by patch_size ({CFG['patch_size']})"
)
assert _D_MODEL % CFG["num_heads"] == 0, (
    f"d_model ({_D_MODEL}) must be divisible by num_heads ({CFG['num_heads']})"
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
    # Autoregressive lags — give direct knowledge of weekly cycle
    for lag in [1, 7, 14, 21]:
        df[f"_tgt_lag_{lag}"] = df[target_col].shift(lag)

    # 7-day rolling stats (shifted by 1 to prevent leakage)
    shifted = df[target_col].shift(1)
    df["_tgt_roll7_mean"] = shifted.rolling(7, min_periods=7).mean()
    df["_tgt_roll7_std"]  = shifted.rolling(7, min_periods=1).std().fillna(0)

    # Cyclical calendar encoding
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
    # log1p compressed 140K–260K ridership into 0.62 log-units; 30K peak
    # errors looked negligible in loss space → systematic under-prediction.
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
# 5. STGNN-STEP ENHANCED MODEL
# ─────────────────────────────────────────────

class AdaptiveAdjacency(nn.Module):
    """
    Learnable batch-invariant structural graph prior (Graph WaveNet style).

        A_adaptive = softmax( ReLU( E1 @ E2ᵀ ) )   shape: (N, N)

    E1, E2 ∈ R^{N × embed_dim} are trained jointly with the rest of the model.
    This encodes a permanent topology prior: "node i always tends to correlate
    with node j", independent of the current input window.
    """
    def __init__(self, num_nodes: int, embed_dim: int = 10):
        super().__init__()
        self.E1 = nn.Embedding(num_nodes, embed_dim)
        self.E2 = nn.Embedding(num_nodes, embed_dim)

    def forward(self, device: torch.device) -> torch.Tensor:
        idx = torch.arange(self.E1.num_embeddings, device=device)
        return F.softmax(F.relu(self.E1(idx) @ self.E2(idx).T), dim=-1)  # (N, N)


class PatchEncoder(nn.Module):
    """
    STEP-style patch encoder: splits the T-step sequence into P non-overlapping
    patches and projects each patch into graph node space.

    Input  : H ∈ (B, T, d_model)          — projected raw features
    Output : Z ∈ (B, P, N, node_dim)      — patch representations in node space

    Each patch spans S consecutive timesteps (S = patch_size).  Concatenating
    the d_model features of all S steps gives a S*d_model-dim vector per patch,
    which is then projected to N*node_dim and reshaped into (N, node_dim).

    Positional encoding:
        A learnable embedding table of size (P, N, node_dim) is added after
        projection.  This gives the model ordinal awareness of which week-patch
        it is processing (patch 0 = oldest, patch P-1 = most recent).
    """
    def __init__(
        self,
        d_model   : int,    # = N * node_dim = 128
        patch_size: int,    # S = 4 days per patch
        num_patches: int,   # P = 7
        num_nodes : int,    # N = 16
        node_dim  : int,    # 8
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.num_patches = num_patches
        self.num_nodes   = num_nodes
        self.node_dim    = node_dim

        # Linear projection: S consecutive projected features → graph node space
        self.proj = nn.Sequential(
            nn.Linear(patch_size * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Learnable patch positional encoding
        self.pos_emb = nn.Parameter(
            torch.zeros(1, num_patches, num_nodes, node_dim)
        )
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        H : (B, T, d_model)
        Returns Z : (B, P, N, node_dim)
        """
        B, T, D = H.shape
        S, P    = self.patch_size, self.num_patches

        # Reshape into patches: (B, P, S, D) → (B, P, S*D)
        patches = H.reshape(B, P, S, D).reshape(B, P, S * D)

        # Project to graph node space: (B, P, d_model) → (B, P, N*node_dim)
        Z = self.proj(patches)                                        # (B, P, N*node_dim)
        Z = Z.reshape(B, P, self.num_nodes, self.node_dim)            # (B, P, N, node_dim)

        # Add positional encoding (broadcast over B)
        return Z + self.pos_emb


class DualGraphBuilder(nn.Module):
    """
    Constructs A_dual = σ(gate) · A_adaptive + (1 - σ(gate)) · A_semantic.

    A_semantic is computed per-sample from mean-pooled patch embeddings:
        Z̄ = mean_P(Z)                         → (B, N, node_dim)
        Z̄_norm = Z̄ / (‖Z̄‖₂ + ε)              (L2-normalised per node)
        A_sem  = softmax( Z̄_norm @ Z̄_normᵀ )  → (B, N, N)

    Cosine similarity is used because node_dim=8 is small; raw dot products
    would be dominated by magnitude differences between nodes, not direction.
    L2-normalising first makes the similarity purely directional.

    gate: single learnable scalar initialised at 0 → σ(0) = 0.5, equal weight.
    """
    def __init__(self):
        super().__init__()
        # gate initialised at 0 → equal split between adaptive and semantic
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        A_adaptive: torch.Tensor,   # (N, N)   batch-invariant
        Z         : torch.Tensor,   # (B, P, N, node_dim)
    ) -> torch.Tensor:
        """Returns A_dual ∈ (B, N, N)."""
        B, P, N, d = Z.shape

        # Mean-pool over patches → (B, N, d)
        Z_mean = Z.mean(dim=1)

        # L2-normalise per node for cosine similarity
        Z_norm = F.normalize(Z_mean, p=2, dim=-1)                     # (B, N, d)

        # Semantic adjacency: cosine similarity matrix, row-softmaxed
        A_sem = F.softmax(
            torch.bmm(Z_norm, Z_norm.transpose(1, 2)),                # (B, N, N)
            dim=-1
        )

        # Gated combination — broadcast A_adaptive over batch
        w = torch.sigmoid(self.gate)
        A_dual = w * A_adaptive.unsqueeze(0) + (1.0 - w) * A_sem     # (B, N, N)
        return A_dual


class SpatialGCN(nn.Module):
    """
    Single-layer GCN with dual adjacency and residual connection.

        H_out = LayerNorm( A_dual ⊗ H @ W )  +  H

    Applied per-patch by flattening the (B, P) batch and patch dims into one.
    This processes all 7 patches spatially in a single batched matmul,
    avoiding a Python loop over patches.

    Why GCN here instead of GAT (like TGACN)?
      The temporal MHA in each ST block already produces sample-adaptive
      representations — GAT's per-sample attention would be partially redundant.
      The dual graph (A_semantic) already provides the per-sample spatial
      adaptivity.  A simpler GCN keeps parameters low and training fast.
    """
    def __init__(self, node_dim: int, dropout: float):
        super().__init__()
        self.W    = nn.Linear(node_dim, node_dim, bias=False)
        self.norm = nn.LayerNorm(node_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor, A_dual: torch.Tensor) -> torch.Tensor:
        """
        H      : (B, P, N, node_dim)
        A_dual : (B, N, N)
        Returns: (B, P, N, node_dim)
        """
        B, P, N, d = H.shape
        residual    = H

        # Expand A_dual across patches: (B, N, N) → (B, P, N, N) → (B*P, N, N)
        A_exp = A_dual.unsqueeze(1).expand(-1, P, -1, -1).reshape(B * P, N, N)

        # Flatten patches into batch: (B*P, N, d)
        H_flat = H.reshape(B * P, N, d)

        # GCN: A @ H @ W
        H_agg  = torch.bmm(A_exp, H_flat)                            # (B*P, N, d)
        H_proj = self.W(H_agg)                                        # (B*P, N, d)
        H_proj = self.drop(H_proj)

        # Residual + LayerNorm, then reshape back
        H_out = self.norm(H_proj + H_flat).reshape(B, P, N, d)
        return H_out


class STBlock(nn.Module):
    """
    One Spatio-Temporal Transformer block:

        H → TemporalAttn → SpatialGCN → FFN → H'

    TemporalAttn:
        Processes each node's patch sequence independently.
        Reshape: (B, P, N, d) → (B*N, P, d)
        MHA(Q=H, K=H, V=H)  — standard scaled dot-product, causal mask NOT used
        (we attend over all P past patches; no future leakage since all patches
        come from the look-back window only).
        Add & Norm → reshape back → (B, P, N, d)

    SpatialGCN:
        See SpatialGCN above — propagates signal across N nodes for all P patches.

    FFN:
        Two-layer MLP with GELU, applied independently to each (patch, node) token.
        Hidden size = ffn_mult × d_node = 4 × 8 = 32.
        Add & Norm.
    """
    def __init__(
        self,
        num_nodes : int,
        node_dim  : int,
        num_heads : int,      # for temporal MHA; must divide node_dim
        ffn_mult  : int,
        dropout   : float,
    ):
        super().__init__()
        assert node_dim % num_heads == 0, (
            f"node_dim ({node_dim}) must be divisible by num_heads ({num_heads})"
        )

        # Temporal: MHA over P patches, per node
        self.temp_norm = nn.LayerNorm(node_dim)
        self.temp_attn = nn.MultiheadAttention(
            embed_dim    = node_dim,
            num_heads    = num_heads,
            dropout      = dropout,
            batch_first  = True,   # input: (batch, seq, embed)
        )
        self.temp_drop = nn.Dropout(dropout)

        # Spatial: GCN with dual adjacency
        self.spat_gcn  = SpatialGCN(node_dim, dropout)

        # FFN per (patch, node) token
        self.ffn_norm  = nn.LayerNorm(node_dim)
        self.ffn       = nn.Sequential(
            nn.Linear(node_dim, ffn_mult * node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * node_dim, node_dim),
            nn.Dropout(dropout),
        )

    def forward(self, H: torch.Tensor, A_dual: torch.Tensor) -> torch.Tensor:
        """
        H      : (B, P, N, node_dim)
        A_dual : (B, N, N)
        Returns: (B, P, N, node_dim)
        """
        B, P, N, d = H.shape

        # ── Temporal attention (over P patches, per node) ─────────────────
        residual = H
        H_norm   = self.temp_norm(H)
        # Reshape: treat each node as an independent sequence of P patches
        H_t      = H_norm.permute(0, 2, 1, 3).reshape(B * N, P, d)  # (B*N, P, d)
        attn_out, _ = self.temp_attn(H_t, H_t, H_t)                 # (B*N, P, d)
        attn_out    = self.temp_drop(attn_out)
        H = residual + attn_out.reshape(B, N, P, d).permute(0, 2, 1, 3)  # (B, P, N, d)

        # ── Spatial GCN (across N nodes, per patch) ───────────────────────
        H = self.spat_gcn(H, A_dual)                                 # (B, P, N, d)

        # ── FFN (per token = per patch×node) ─────────────────────────────
        residual = H
        H = residual + self.ffn(self.ffn_norm(H))                    # (B, P, N, d)

        return H


class STGNNSTEPModel(nn.Module):
    """
    Full STGNN-STEP Enhanced pipeline — see module docstring for architecture diagram.
    """
    def __init__(
        self,
        input_size   : int,
        num_nodes    : int,
        node_dim     : int,
        adj_embed_dim: int,
        patch_size   : int,
        num_patches  : int,
        num_st_layers: int,
        num_heads    : int,
        ffn_mult     : int,
        dropout      : float,
        horizon      : int,
    ):
        super().__init__()
        d_model = num_nodes * node_dim   # 128

        # ── Stage 1: Feature projection ───────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        # ── Stage 2: Patch encoder ────────────────────────────────────────
        self.patch_enc = PatchEncoder(
            d_model    = d_model,
            patch_size = patch_size,
            num_patches= num_patches,
            num_nodes  = num_nodes,
            node_dim   = node_dim,
        )

        # ── Stage 3: Dual graph construction ──────────────────────────────
        self.adapt_adj   = AdaptiveAdjacency(num_nodes, adj_embed_dim)
        self.dual_graph  = DualGraphBuilder()

        # ── Stage 4: ST-Transformer blocks ────────────────────────────────
        self.st_blocks = nn.ModuleList([
            STBlock(
                num_nodes = num_nodes,
                node_dim  = node_dim,
                num_heads = num_heads,
                ffn_mult  = ffn_mult,
                dropout   = dropout,
            )
            for _ in range(num_st_layers)
        ])

        # ── Stage 5: Output head ──────────────────────────────────────────
        # Take the most recent patch (index P-1), flatten N×node_dim → d_model
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, T, F)
        returns : (B, horizon)
        """
        # Stage 1: project features
        H = self.input_proj(x)                             # (B, T, d_model)

        # Stage 2: encode patches
        Z = self.patch_enc(H)                              # (B, P, N, node_dim)

        # Stage 3: build dual graph (computed once, shared across all ST blocks)
        A_adap = self.adapt_adj(x.device)                 # (N, N)
        A_dual = self.dual_graph(A_adap, Z)               # (B, N, N)

        # Stage 4: interleaved spatio-temporal processing
        for block in self.st_blocks:
            Z = block(Z, A_dual)                           # (B, P, N, node_dim)

        # Stage 5: take last patch, flatten, forecast
        last = Z[:, -1, :, :].reshape(Z.shape[0], -1)     # (B, N*node_dim)
        return self.head(last)                             # (B, horizon)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 6. TRAINING UTILITIES
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

    v4 fixes applied:
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

    preds = np.concatenate(all_pred, axis=0)
    trues = np.concatenate(all_true, axis=0)

    # v4 FIX 1 (mirror): no expm1 — inverse StandardScaler only
    preds_inv = tgt_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_inv = tgt_scaler.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)

    # v4 FIX 4: apply post-hoc OLS calibration if provided
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
    ax.set_title("STGNN-STEP Enhanced — Training & Validation Loss")
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
    ax.set_title("STGNN-STEP Enhanced — Test Set 1-Day-Ahead Forecast vs Actual")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "forecast_vs_actual.png", dpi=150)
    plt.close(fig)
    logging.info("Saved forecast_vs_actual.png")

    # Dual graph gate weight at training end
    logging.info(
        "NOTE: log model.dual_graph.gate after loading checkpoint to inspect "
        "final A_adaptive vs A_semantic balance."
    )


# ─────────────────────────────────────────────
# 9. BASELINE COMPARISON
# ─────────────────────────────────────────────
def compare_with_baseline(stgnn_metrics: dict, baseline_path: str):
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
    logging.info("  MODEL COMPARISON — STGNN-STEP v4 vs LSTM v4 (baseline)")
    logging.info("=" * len(sep))
    logging.info(
        f"  {'Metric':<12}"
        f"{'STGNN-STEP v4':>{col_w}}"
        f"{'LSTM v4':>{col_w}}"
        f"{'Winner':>{col_w}}"
    )
    logging.info(sep)

    for m in metrics_order:
        bi_val = stgnn_metrics.get(m, float("nan"))
        base_val = baseline.get(m, float("nan"))

        if m in higher_better:
            winner = (
                "STGNN-STEP (W)"
                if bi_val > base_val
                else ("LSTM (W)" if base_val > bi_val else "TIE")
            )
        else:
            winner = (
                "STGNN-STEP (W)"
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

    n_features  = len(feature_cols)
    num_patches = cfg["seq_len"] // cfg["patch_size"]
    d_model     = cfg["num_nodes"] * cfg["node_dim"]

    logging.info(
        f"Patch config: seq_len={cfg['seq_len']}, patch_size={cfg['patch_size']}, "
        f"num_patches={num_patches}  |  d_model={d_model}"
    )

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
    model = STGNNSTEPModel(
        input_size    = n_features,
        num_nodes     = cfg["num_nodes"],
        node_dim      = cfg["node_dim"],
        adj_embed_dim = cfg["adj_embed_dim"],
        patch_size    = cfg["patch_size"],
        num_patches   = num_patches,
        num_st_layers = cfg["num_st_layers"],
        num_heads     = cfg["num_heads"],
        ffn_mult      = cfg["ffn_mult"],
        dropout       = cfg["st_dropout"],
        horizon       = cfg["horizon"],
    ).to(device)

    logging.info(f"Model parameters: {count_params(model):,}")
    logging.info(model)

    gate_val = torch.sigmoid(model.dual_graph.gate).item()
    logging.info(
        f"Initial graph gate: sigma(gate)={gate_val:.3f}  "
        f"-> {gate_val*100:.1f}% A_adaptive + {(1-gate_val)*100:.1f}% A_semantic"
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    # v4 FIX 2: MSELoss replaces HuberLoss.
    # MSE squares peak errors so the model is penalised 9× more for a 30K
    # miss than a 10K miss — forcing it to chase weekday peaks.
    criterion = nn.MSELoss()

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )

    # ── LR schedule ───────────────────────────────────────────────────────
    # v4 FIX 3: T_max=100 — LR decays to eta_min by epoch 100.
    # T_max=500 kept LR ~4.9e-4 for the first 50+ epochs, causing the
    # large val-loss spikes visible in all v3 model loss curves.
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
    best_ckpt  = output_dir / "best_stgnn_step.pt"
    no_improve = 0

    logging.info("-" * 60)
    logging.info("Starting STGNN-STEP Enhanced training …")
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

    # Log final gate weight — diagnostic of which graph dominated
    gate_final = torch.sigmoid(model.dual_graph.gate).item()
    logging.info(
        f"Final graph gate: sigma(gate)={gate_final:.3f}  "
        f"-> {gate_final*100:.1f}% A_adaptive + {(1-gate_final)*100:.1f}% A_semantic"
    )

    # ── v4 FIX 4: Post-hoc OLS calibration on val set ────────────────────
    # y_cal = a·ŷ + b, fitted on val predictions (original scale).
    # Removes any residual systematic bias before test evaluation.
    # No leakage — val set never seen during training.
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
    logging.info("TEST RESULTS — STGNN-STEP Enhanced v4 (original scale, averaged over horizon)")
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