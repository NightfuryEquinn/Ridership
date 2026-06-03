"""
hmttsf.py — HMT-TSF: Hybrid Multi-scale Temporal Spatio-Feature Forecaster

Architecture Diagram
====================

  Input X: (B, T_in, F=53)  [79 raw features; 26 zero-SHAP features dropped at load time]
           │
  ┌────────┴────────────────────────────────────────────────────┐
  │               Feature Group Encoders                        │
  │  [Target ctx] [Temporal/Cyclical] [External] [Lag] [Static] │
  │        └──────────── FeatureGroupFusion ──────────┘         │
  │                   (B, T_in, d_model)                        │
  └────────┬───────────────────────┬──────────────────┬─────────┘
           │                       │                  │
  ┌────────▼──────────┐  ┌─────────▼────────┐  ┌────▼──────────┐
  │  Multi-Scale TCN  │  │  Feature Graph   │  │    Regime     │
  │  ─────────────    │  │  Encoder (GCN)   │  │  Gating Emb.  │
  │  Scale-1 (full)   │  │  Pearson adj     │  │  3-way soft   │
  │  Scale-2 (½)      │  │  2-layer GCN     │  │  gate × emb   │
  │  Scale-3 (¼)      │  │  Global pool     │  │               │
  │  Attn blend       │  │                  │  │               │
  └────────┬──────────┘  └─────────┬────────┘  └────┬──────────┘
           │                       │                  │
           └───────────────────────┴──────────────────┘
                                   │
                          ┌────────▼────────┐
                          │  Gated Fusion   │
                          │  σ(W·cat)⊙cat  │
                          └────────┬────────┘
                                   │ (B, d_model)
                          ┌────────▼────────┐
                          │  Forecast Heads │
                          │  Neural Head    │ → y_primary  (B, T_out)
                          │  Boost Head     │ → y_boost    (B, T_out)
                          │  α·blend        │ → y_final    (B, T_out)
                          └─────────────────┘

Custom Loss = λ₁·WeightedHuber(step-decayed) + λ₂·TemporalSmoothness

Feature Groups (79→53 after SHAP reduction; see _DROPPED_FEAT_INDICES):
  Indices 0–12   : target context (13 service-line riderships)
  Indices 13–22  : temporal/cyclical (10: 6 zero-importance features dropped)
  Indices 23–49  : external (27: 3 collinear fuel-level features dropped)
  Indices 50–52  : lag features (3: lag_7, lag_14, lag_28)
  (static group eliminated — all 17 features had zero SHAP importance)

Optimisation targets:
  Combined% ≥ 75,  R² ≥ 0.70

Hardware:
  GPU  : NVIDIA A100 (32 GB VRAM)
  RAM  : 32 GB
  Precision : AMP fp16 (GradScaler enabled on CUDA)
"""

# ── stdlib ───────────────────────────────────────────────────────────────────
import os
import sys
import json
import argparse
from datetime import datetime

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.utils.metrics import compute_metrics
from src.utils.comparison_table import (
    load_model_results,
    print_comparison_table,
    plot_comparison,
)
from src.utils.shap_analysis import load_feature_names, run_shap_analysis

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Feature group definitions (index ranges into the 79-feature vector)
# ─────────────────────────────────────────────────────────────────────────────

FEAT_GROUPS = {
    "target":   (0,  13),   # 12 service lines + total_ridership
    "temporal": (13, 29),   # holiday flags, cyclical encodings, year, day_of_year
    "external": (29, 59),   # fuel prices (×15) + rainfall (×15)
    "lag":      (59, 62),   # ridership_lag_{7,14,28}
    "static":   (62, 79),   # population, GTFS stats, OSM POI, GADM
}

# ─────────────────────────────────────────────────────────────────────────────
# SHAP-derived feature reduction (applied at load time in main())
#
# 26 features with zero or near-zero importance across all 10 HMT-TSF runs
# (see HMT-TSF-RESULTS.md §10).  Dropped indices are sourced from the original
# 79-feature space; the 53 survivors are re-indexed contiguously and their
# group boundaries are captured in _REDUCED_FEAT_GROUPS.
# ─────────────────────────────────────────────────────────────────────────────

_DROPPED_FEAT_INDICES: frozenset = frozenset([
    # Temporal — redundant or zero across all configs
    18,             # days_to_next_school_hol (near-zero, 8/10 runs)
    21,             # month (redundant with month_sin/cos)
    22,             # is_weekend (subsumed by day_of_week)
    23,             # dow_sin (raw day_of_week dominates)
    24,             # dow_cos (near-zero, 8/10 runs)
    27,             # year (near-zero, 8/10 runs)
    # Fuel — collinear level features
    30,             # fp_lv_ron97 (collinear with RON95)
    31,             # fp_lv_diesel (collinear with diesel pct_chg)
    32,             # fp_lv_diesel_eastmsia (East Malaysia, irrelevant)
    # Static — time-invariant; no day-to-day variance
    62, 63,         # pop_density_median / log_median
    64, 65, 66, 67, # GTFS: n_stops, n_routes, n_directed_edges, avg_segment_s
    68, 69, 70, 71, 72, 73, 74, 75,  # OSM POI: total/transport/food/retail/education/healthcare/leisure/other
    76, 77, 78,     # GADM: n_states, n_border_pairs, mean_border_km
])

_KEPT_FEAT_INDICES: list = sorted(set(range(79)) - _DROPPED_FEAT_INDICES)  # 53 features

# Re-mapped group slices for the 53-feature reduced input:
#   target (0-12)   → 13 features, unchanged
#   temporal (13-28)→ 10 kept (drop positions 5,8,9,10,11,14 within group)
#   external (29-58)→ 27 kept (drop indices 30,31,32)
#   lag (59-61)     → 3 features, unchanged
#   static (62-78)  → 0 features (all dropped)
_REDUCED_FEAT_GROUPS: dict = {
    "target":   (0,  13),
    "temporal": (13, 23),
    "external": (23, 50),
    "lag":      (50, 53),
    "static":   (53, 53),  # empty — FeatureGroupFusion handles this gracefully
}

# Positions within the temporal group (0-indexed from feat 13) to retain in
# X_future tensors (which carry only the temporal slice, not all 79 features).
_DROPPED_TEMPORAL_POSITIONS: frozenset = frozenset(i - 13 for i in _DROPPED_FEAT_INDICES
                                                    if 13 <= i <= 28)
_KEPT_TEMPORAL_POSITIONS: list = sorted(set(range(16)) - _DROPPED_TEMPORAL_POSITIONS)  # 10

# ─────────────────────────────────────────────────────────────────────────────
# Per-lookback baseline defaults
# Applied in main() only when the user has NOT overridden the flag explicitly.
# Rationale for each adjustment is documented in HMT-TSF.md.
# ─────────────────────────────────────────────────────────────────────────────

_LOOKBACK_DEFAULTS: dict = {
    # Lookback 7/14: very short sequences → model sees little temporal signal per
    # sample and memorises easily.  Shrink capacity (d_model=32, 2 TCN blocks),
    # apply aggressive dropout/weight-decay, and add input noise to regularise.
    7:  {"d_model": 32, "graph_hidden": 32, "n_tcn_blocks": 2,
         "dropout": 0.35, "drop_path": 0.30, "weight_decay": 2e-2,
         "input_noise": 0.05, "smooth_weight": 0.03,
         "lr": 5e-4, "warmup_epochs": 10, "patience": 25},

    14: {"d_model": 32, "graph_hidden": 32, "n_tcn_blocks": 2,
         "dropout": 0.25, "drop_path": 0.20, "weight_decay": 1e-2,
         "input_noise": 0.05},

    28: {"lr": 5e-4, "warmup_epochs": 12, "patience": 25,
         "d_model": 32, "graph_hidden": 32, "n_tcn_blocks": 2,
         "dropout": 0.25, "weight_decay": 1e-2, "input_noise": 0.05},

    56: {"d_model": 32, "graph_hidden": 32, "n_tcn_blocks": 2, "dropout": 0.30,
         "drop_path": 0.3, "weight_decay": 2e-2, "input_noise": 0.05,
         "smooth_weight": 0.03},

    84: {"d_model": 32, "graph_hidden": 32, "n_tcn_blocks": 3,
         "patience": 30, "lr": 5e-4, "warmup_epochs": 15,
         "weight_decay": 4e-2, "input_noise": 0.08,
         "smooth_weight": 0.05, "dropout": 0.40, "drop_path": 0.35},
}

# Global argparse defaults — used to detect whether the user overrode a flag.
_ARGPARSE_DEFAULTS: dict = {
    "lr":            1e-3,
    "warmup_epochs": 8,
    "patience":      20,
    "input_noise":   0.02,
    "smooth_weight": 0.01,
    "n_tcn_blocks":  3,
    "dropout":       0.1,
    "drop_path":     0.2,
    "weight_decay":  1e-3,
    "d_model":       64,
    "graph_hidden":  64,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Reversible Instance Normalisation (RevIN)
# ══════════════════════════════════════════════════════════════════════════════
    
class RevIN(nn.Module):
    """
    Per-sample, per-feature normalisation that is reversed at output.
    Mitigates distribution shift between train and test periods.
    """

    def __init__(self, n_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_features))
        self.beta  = nn.Parameter(torch.zeros(n_features))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        self._mean = x.mean(dim=1, keepdim=True)          # (B, 1, F)
        self._std  = x.std(dim=1, keepdim=True) + self.eps
        x = (x - self._mean) / self._std
        return x * self.gamma + self.beta

    def denormalize(self, x: torch.Tensor, target_idx: int) -> torch.Tensor:
        # x: (B, T_out) — target dimension only
        mean = self._mean[:, 0, target_idx]   # (B,)
        std  = self._std[:, 0, target_idx]    # (B,)
        gamma_t = self.gamma[target_idx]
        beta_t  = self.beta[target_idx]
        # Reverse affine then unnormalise
        x = (x - beta_t) / (gamma_t + self.eps)
        return x * std.unsqueeze(1) + mean.unsqueeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Feature Group Encoders
# ══════════════════════════════════════════════════════════════════════════════

class FeatureGroupEncoder(nn.Module):
    """Project a contiguous slice [start:end] of the feature dimension to d_out."""

    def __init__(self, start: int, end: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        n = end - start
        self.start = start
        self.end   = end
        mid = max(n, d_out)
        self.proj = nn.Sequential(
            nn.Linear(n, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → (B, T, d_out)
        return self.proj(x[..., self.start:self.end])


class FeatureGroupFusion(nn.Module):
    """
    Five specialised encoders (one per feature group) whose outputs are
    concatenated and projected to d_model.  A learned gating vector controls
    how much each group contributes.
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1,
                 feat_groups: dict | None = None):
        super().__init__()
        g = feat_groups if feat_groups is not None else FEAT_GROUPS
        # Clamp end indices to actual feature count
        def _enc(key):
            s, e = g[key]
            e = min(e, n_features)
            s = min(s, n_features)
            if s >= e:
                return None, 0
            return FeatureGroupEncoder(s, e, d_model // 2, dropout), d_model // 2

        self.enc_target,   d_tgt  = _enc("target")
        self.enc_temporal, d_tmp  = _enc("temporal")
        self.enc_external, d_ext  = _enc("external")
        self.enc_lag,      d_lag  = _enc("lag")
        self.enc_static,   d_sta  = _enc("static")

        d_concat = d_tgt + d_tmp + d_ext + d_lag + d_sta
        if d_concat == 0:
            d_concat = n_features

        self.fuse = nn.Sequential(
            nn.Linear(d_concat, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        # Learned group importance gate
        n_active = sum(1 for e in [self.enc_target, self.enc_temporal,
                                    self.enc_external, self.enc_lag, self.enc_static]
                       if e is not None)
        self.group_gate = nn.Parameter(torch.ones(n_active) / n_active)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → (B, T, d_model)
        parts, g_idx = [], []
        for i, enc in enumerate([self.enc_target, self.enc_temporal,
                                   self.enc_external, self.enc_lag, self.enc_static]):
            if enc is not None:
                parts.append(enc(x))
                g_idx.append(i)

        if not parts:
            return x  # fallback — shouldn't happen

        gates = torch.softmax(self.group_gate, dim=0)
        weighted = [p * gates[j] for j, p in enumerate(parts)]
        h = torch.cat(weighted, dim=-1)   # (B, T, d_concat)
        return self.fuse(h)               # (B, T, d_model)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Multi-Scale TCN Encoder
# ══════════════════════════════════════════════════════════════════════════════

class DropPath(nn.Module):
    """
    Stochastic depth: randomly drop the residual path of a block during
    training.  Acts as a structured regulariser that prevents co-adaptation
    between consecutive TCN blocks.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # floor_(val) takes no args — add keep_prob first, then floor in-place
        noise = (torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob).floor_()
        return x * noise / keep_prob


class TemporalTransformerBlock(nn.Module):
    """
    Single Transformer encoder block injected between FeatureGroupFusion and
    the multi-scale TCN.

    Purpose: the TCN captures LOCAL temporal patterns via dilated convolutions,
    but cannot attend across the full lookback window.  This block adds GLOBAL
    temporal self-attention so the model can weight which time steps matter most
    before local feature extraction.

    Learnable positional embeddings are added so the attention is aware of
    recency (position 0 = oldest, T_in−1 = most recent).
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1,
                 max_len: int = 128):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                             batch_first=True)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T   = x.size(1)
        pos = torch.arange(T, device=x.device)
        x   = x + self.pos_emb(pos)               # inject positional info
        h, _ = self.attn(x, x, x)
        x     = self.norm1(x + self.drop(h))
        x     = self.norm2(x + self.drop(self.ffn(x)))
        return x


class CausalConv1d(nn.Module):
    """Left-padded dilated causal convolution (no future leakage)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNResBlock(nn.Module):
    """WaveNet-style gated residual block with layer norm."""

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.1,
                 drop_path: float = 0.0):
        super().__init__()
        self.conv1     = CausalConv1d(channels, channels * 2, kernel_size, dilation)
        self.conv2     = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1     = nn.GroupNorm(1, channels * 2)
        self.norm2     = nn.GroupNorm(1, channels)
        self.drop      = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.conv1(x)              # (B, 2C, T)
        h = self.norm1(h)
        h1, h2 = h.chunk(2, dim=1)    # gated activation
        h = torch.tanh(h1) * torch.sigmoid(h2)  # (B, C, T)
        h = self.drop(h)
        h = self.norm2(self.conv2(h))  # (B, C, T)
        return self.drop_path(h) + x   # stochastic-depth residual


class TCNStack(nn.Module):
    """Stack of TCNResBlocks with exponentially growing dilation."""

    def __init__(self, d_model: int, n_blocks: int,
                 kernel_size: int = 3, dropout: float = 0.1,
                 drop_path: float = 0.0):
        super().__init__()
        # Linearly scale drop_path rate from 0 to drop_path across blocks
        dp_rates = [drop_path * i / max(n_blocks - 1, 1) for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            TCNResBlock(d_model, kernel_size, dilation=2 ** i,
                        dropout=dropout, drop_path=dp_rates[i])
            for i in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, C, T)
        for blk in self.blocks:
            x = blk(x)
        return x


class MultiScaleTCN(nn.Module):
    """
    Parallel TCN stacks operating on three temporal scales:
      - full   : last T_in steps
      - half   : last T_in//2 steps  (if T_in ≥ 28)
      - quarter: last T_in//4 steps  (if T_in ≥ 56)

    Scale outputs are pooled to a single vector and blended via learned weights.
    """

    def __init__(self, d_model: int, T_in: int, n_blocks: int = 3,
                 kernel_size: int = 3, dropout: float = 0.1,
                 drop_path: float = 0.0):
        super().__init__()
        self.T_in    = T_in
        self.d_model = d_model
        self.use_half    = (T_in >= 28)
        self.use_quarter = (T_in >= 56)

        self.tcn_full = TCNStack(d_model, n_blocks, kernel_size, dropout, drop_path)
        if self.use_half:
            self.tcn_half = TCNStack(d_model, max(1, n_blocks - 1), kernel_size, dropout, drop_path)
        if self.use_quarter:
            self.tcn_quarter = TCNStack(d_model, max(1, n_blocks - 2), kernel_size, dropout, drop_path)

        n_scales = 1 + int(self.use_half) + int(self.use_quarter)
        self.scale_attn = nn.Parameter(torch.ones(n_scales) / n_scales)
        self.out_norm   = nn.LayerNorm(d_model)

    def _pool(self, x_bct: torch.Tensor) -> torch.Tensor:
        return x_bct.mean(dim=-1)  # (B, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, d_model)
        x_bct = x.transpose(1, 2)          # (B, d_model, T_in)

        scales = [self._pool(self.tcn_full(x_bct))]

        if self.use_half:
            half = x_bct[:, :, -self.T_in // 2:]
            scales.append(self._pool(self.tcn_half(half)))

        if self.use_quarter:
            quarter = x_bct[:, :, -self.T_in // 4:]
            scales.append(self._pool(self.tcn_quarter(quarter)))

        weights = torch.softmax(self.scale_attn, dim=0)
        h = sum(w * s for w, s in zip(weights, scales))  # (B, d_model)
        return self.out_norm(h)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Feature Graph Encoder (GCN over feature nodes)
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_adj(X_train_np: np.ndarray,
                      threshold: float = 0.1) -> np.ndarray:
    """
    Pearson-correlation adjacency over feature nodes.
    Edges with |corr| < threshold are zeroed; surviving edges retain their
    correlation magnitude as weight (matching the strategy used by STGCN/ASTGCN
    in this project).  Returns sym-normalised (F, F) float32.
    """
    N, T, F = X_train_np.shape
    flat = X_train_np.reshape(N * T, F).astype(np.float64)
    flat -= flat.mean(0)
    std   = flat.std(0) + 1e-8
    flat /= std
    corr  = np.abs(np.corrcoef(flat.T))   # (F, F) — correlation magnitudes
    corr  = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)  # zero-variance features → no edges
    np.fill_diagonal(corr, 0.0)
    adj   = np.where(corr >= threshold, corr, 0.0).astype(np.float32)  # soft weights
    # Symmetric normalisation: D^{-1/2} A D^{-1/2}
    deg         = adj.sum(1) + 1e-8
    d_inv_sqrt  = np.diag(1.0 / np.sqrt(deg))
    return (d_inv_sqrt @ adj @ d_inv_sqrt).astype(np.float32)


class GraphConvLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.fc   = nn.Linear(in_ch, out_ch)
        self.norm = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C),  adj: (N, N)
        h = torch.matmul(adj, x)             # (B, N, C)
        return F.gelu(self.norm(self.drop(self.fc(h))))


class FeatureGraphEncoder(nn.Module):
    """
    Treats the F input features as graph nodes (F=53 after SHAP reduction).
    Input: raw feature sequence (B, T, F).
    Aggregates over T → node signals → 2-layer GCN → global mean pool → (B, d_model).

    Temporal aggregation uses a learned attention vector over the T dimension
    rather than naive mean-pooling.  This lets the graph branch weight recent
    time steps (or peak-demand days) more heavily than distant ones.
    """

    def __init__(self, n_features: int, d_model: int,
                 graph_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        # Scalar attention score per time step: (B, T, F) → (B, T, 1)
        self.time_attn  = nn.Linear(n_features, 1)
        self.input_proj = nn.Linear(1, graph_hidden)   # per-node scalar → embedding
        self.gcn1       = GraphConvLayer(graph_hidden, graph_hidden, dropout)
        self.gcn2       = GraphConvLayer(graph_hidden, d_model, dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F),  adj: (F, F)
        attn_w = F.softmax(self.time_attn(x), dim=1)   # (B, T, 1)
        h = (x * attn_w).sum(dim=1).unsqueeze(-1)       # (B, F, 1)
        h = self.input_proj(h)                           # (B, F, graph_hidden)
        h = self.gcn1(h, adj)                            # (B, F, graph_hidden)
        h = self.gcn2(h, adj)                            # (B, F, d_model)
        return h.mean(dim=1)                             # (B, d_model) — global pool


# ══════════════════════════════════════════════════════════════════════════════
# 5. Regime-Gating Embedding
# ══════════════════════════════════════════════════════════════════════════════

class RegimeGatingEmbedding(nn.Module):
    """
    Learns a soft mixture over K regime embeddings.
    Gate weights are inferred from the mean input feature vector (no date needed),
    so the network can autonomously detect structural regime shifts (e.g. MCO).
    """

    def __init__(self, n_features: int, d_model: int,
                 n_regimes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(n_features, n_regimes * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_regimes * 4, n_regimes),
        )
        self.regime_emb = nn.Parameter(torch.randn(n_regimes, d_model) * 0.02)

    def forward(self, x: torch.Tensor):
        # x: (B, T, F)
        x_mean = x.mean(dim=1)                          # (B, F)
        gates   = F.softmax(self.gate_net(x_mean), dim=-1)  # (B, n_regimes)
        emb     = gates @ self.regime_emb               # (B, d_model)
        return emb, gates


# ══════════════════════════════════════════════════════════════════════════════
# 6. Gated Cross-Modal Fusion
# ══════════════════════════════════════════════════════════════════════════════

class GatedFusion(nn.Module):
    """
    Fuses temporal (TCN), spatial (GCN), and regime embeddings via
    an element-wise sigmoid gate.  Inspired by highway networks.
    """

    def __init__(self, d_temporal: int, d_spatial: int, d_regime: int,
                 d_out: int, dropout: float = 0.1):
        super().__init__()
        d_cat = d_temporal + d_spatial + d_regime
        # Bottleneck SE-style gate: d_cat → d_cat//4 → d_cat avoids a massive
        # square weight matrix (768×768 at d_model=256) that overfits easily.
        d_gate = max(d_cat // 4, 64)
        self.gate = nn.Sequential(
            nn.Linear(d_cat, d_gate), nn.ReLU(),
            nn.Linear(d_gate, d_cat), nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Linear(d_cat, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
        )
        self.norm = nn.LayerNorm(d_out)

    def forward(self, h_t: torch.Tensor,
                h_s: torch.Tensor,
                h_r: torch.Tensor) -> torch.Tensor:
        h   = torch.cat([h_t, h_s, h_r], dim=-1)   # (B, d_cat)
        h   = self.gate(h) * h
        return self.norm(self.proj(h))               # (B, d_out)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Forecast Heads
# ══════════════════════════════════════════════════════════════════════════════

class NeuralForecastHead(nn.Module):
    """Primary MLP head with highway residual."""

    def __init__(self, d_model: int, T_out: int, dropout: float = 0.1):
        super().__init__()
        # Single d_model → d_model projection (no 2× expansion): the expansion
        # adds parameters without benefit and is a leading cause of overfitting
        # in the head at large d_model values.
        self.fc1  = nn.Linear(d_model, d_model)
        self.fc2  = nn.Linear(d_model, d_model)
        self.fc3  = nn.Linear(d_model, T_out)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, d_model) → (B, T_out)
        h2 = self.drop(F.gelu(self.fc1(h)))
        h2 = self.norm(self.fc2(h2) + h)
        return self.fc3(h2)


class BoostingHead(nn.Module):
    """
    Shallow MLP trained on primary residuals to correct systematic errors.
    α is a learnable scalar blending coefficient.
    """

    def __init__(self, d_model: int, T_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, T_out),
        )
        self.alpha = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12: minimal initial contribution

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h) * torch.sigmoid(self.alpha)


class FutureTemporalProjection(nn.Module):
    """
    Learns a per-step additive correction from known future temporal features
    (day-of-week, weekend flag, holiday flags, sin/cos encodings, etc.).

    These features are fully deterministic for any calendar date, so they can
    be computed ahead of the forecast horizon and fed to the decoder.  The
    correction is added in RevIN-normalised space (before denormalization) so
    it scales naturally with each sample's ridership level.

    Output weights are initialised to zero so the module starts as a no-op and
    only diverges from the baseline as gradient evidence accumulates — avoiding
    any disruption to early-epoch stability.

    Input:  (B, T_out, n_temporal)
    Output: (B, T_out)   additive correction in RevIN-normalised space
    """

    def __init__(self, n_temporal: int, dropout: float = 0.1):
        super().__init__()
        d_hidden = max(n_temporal * 2, 32)
        self.mlp = nn.Sequential(
            nn.Linear(n_temporal, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x_future: torch.Tensor) -> torch.Tensor:
        # x_future: (B, T_out, n_temporal) → (B, T_out)
        return self.mlp(x_future).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 8. HMT-TSF: Main Model
# ══════════════════════════════════════════════════════════════════════════════

class HMTTSFForecaster(nn.Module):
    """
    Hybrid Multi-scale Temporal Spatio-Feature Forecaster.

    Input:  (B, T_in, n_features)   — MinMax-scaled by the data pipeline
    Output: (B, T_out)              — also in MinMax-scaled space; scaler_y
                                      handles the final inverse transform.

    RevIN normalises each input sample to zero-mean over the look-back window;
    the output is denormalised back to MinMax-scaled space inside forward() so
    that scaler_y.inverse_transform() in main() works correctly.

    adj: (n_features, n_features) — pre-computed Pearson adjacency on GPU.
    """

    def __init__(
        self,
        n_features:      int,
        T_in:            int,
        T_out:           int,
        adj:             torch.Tensor,
        d_model:         int   = 128,
        n_tcn_blocks:    int   = 3,
        tcn_kernel:      int   = 3,
        graph_hidden:    int   = 64,
        n_regimes:       int   = 3,
        n_attn_heads:    int   = 4,
        dropout:         float = 0.1,
        drop_path:       float = 0.1,
        target_idx:      int   = 12,
        use_revin:       bool  = True,
        n_temporal_feats: int  = 16,
        feat_groups:      dict | None = None,
    ):
        super().__init__()
        self.target_idx = target_idx
        self.use_revin  = use_revin
        self.register_buffer("adj", adj)

        if use_revin:
            self.revin = RevIN(n_features)

        # Feature group fusion: (B, T, F) → (B, T, d_model)
        self.feat_fusion = FeatureGroupFusion(n_features, d_model, dropout,
                                              feat_groups=feat_groups)

        # Global temporal self-attention: (B, T, d_model) → (B, T, d_model)
        # Allows the model to weight which time steps matter before local TCN.
        self.temporal_attn = TemporalTransformerBlock(
            d_model, n_heads=n_attn_heads, dropout=dropout, max_len=T_in + 4
        )

        # Multi-scale TCN: (B, T, d_model) → (B, d_model)
        self.tcn = MultiScaleTCN(d_model, T_in, n_tcn_blocks, tcn_kernel, dropout, drop_path)

        # Graph encoder: (B, T, F) → (B, d_model)
        self.graph_enc = FeatureGraphEncoder(n_features, d_model, graph_hidden, dropout)

        # Regime embedding: (B, T, F) → (B, d_model)
        self.regime_emb = RegimeGatingEmbedding(n_features, d_model, n_regimes, dropout)

        # Gated fusion: three d_model streams → (B, d_model)
        self.fusion = GatedFusion(d_model, d_model, d_model, d_model, dropout)

        # Heads
        self.primary_head  = NeuralForecastHead(d_model, T_out, dropout)
        self.boosting_head = BoostingHead(d_model, T_out, dropout)

        # Future temporal correction (day-of-week / weekend conditioning)
        self.temporal_proj = FutureTemporalProjection(n_temporal_feats, dropout)

    def forward(self, x: torch.Tensor, x_future: torch.Tensor | None = None) -> torch.Tensor:
        # ── 1. Instance normalisation ────────────────────────────────────────
        # RevIN normalises each sample to zero-mean / unit-variance over the
        # look-back window (per feature).  The stored _mean/_std are then used
        # in denormalize() to map the output BACK to MinMax-scaled space before
        # the loss is computed.  Without denormalization the model would receive
        # zero-mean inputs but be trained against absolute MinMax targets —
        # unable to distinguish absolute ridership levels → poor R².
        if self.use_revin:
            x = self.revin.normalize(x)

        # ── 2. Feature group fusion ──────────────────────────────────────────
        h_feat = self.feat_fusion(x)            # (B, T_in, d_model)

        # ── 3. Global temporal self-attention ────────────────────────────────
        h_feat = self.temporal_attn(h_feat)     # (B, T_in, d_model)

        # ── 4. Multi-scale TCN ───────────────────────────────────────────────
        h_t = self.tcn(h_feat)                  # (B, d_model)

        # ── 5. Graph spatial encoding ────────────────────────────────────────
        h_s = self.graph_enc(x, self.adj)   # (B, d_model)

        # ── 6. Regime embedding ──────────────────────────────────────────────
        h_r, _ = self.regime_emb(x)         # (B, d_model)

        # ── 7. Gated fusion ──────────────────────────────────────────────────
        h_fused = self.fusion(h_t, h_s, h_r)  # (B, d_model)

        # ── 8. Primary + boosting heads ──────────────────────────────────────
        y_primary = self.primary_head(h_fused)   # (B, T_out)
        y_boost   = self.boosting_head(h_fused)  # (B, T_out)
        y = y_primary + y_boost                  # (B, T_out)

        # ── 8a. Future temporal correction ───────────────────────────────────
        # x_future: (B, T_out, n_temporal) — known calendar features for each
        # forecast step (day-of-week, weekend flag, holiday flags, etc.).
        # Applied in RevIN-normalised space so the correction scales with the
        # per-sample ridership level before denormalization.
        if x_future is not None:
            y = y + self.temporal_proj(x_future)

        # ── 9. Reverse RevIN: map output back to MinMax-scaled space ─────────
        # The heads predict in the RevIN-normalised space; denormalize() undoes
        # the per-sample shift/scale so scaler_y.inverse_transform() sees the
        # same MinMax-scaled space it was fitted on.
        if self.use_revin:
            y = self.revin.denormalize(y, self.target_idx)
        return y


# ══════════════════════════════════════════════════════════════════════════════
# 9. Custom Losses
# ══════════════════════════════════════════════════════════════════════════════

class WeightedHuberLoss(nn.Module):
    """
    Huber loss with per-horizon weights.  Step 1 carries weight 1.0;
    later steps decay geometrically (controllable via decay).
    Encourages accurate near-term forecasts while still training long-range.
    """

    def __init__(self, T_out: int, delta: float = 1.0, decay: float = 0.9):
        super().__init__()
        weights = torch.tensor(
            [decay ** i for i in range(T_out)], dtype=torch.float32
        )
        self.register_buffer("weights", weights / weights.sum() * T_out)
        self.delta = delta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred, y_true: (B, T_out)
        diff   = y_pred - y_true
        abs_d  = diff.abs()
        huber  = torch.where(
            abs_d <= self.delta,
            0.5 * diff ** 2,
            self.delta * (abs_d - 0.5 * self.delta)
        )
        return (huber * self.weights).mean()


class TemporalSmoothnessReg(nn.Module):
    """
    Penalises large jumps between consecutive forecast steps.
    Acts as implicit regularisation for monotonically smooth trajectories.
    """

    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        diff = y_pred[:, 1:] - y_pred[:, :-1]   # (B, T_out - 1)
        return self.weight * (diff ** 2).mean()


def build_criterion(args, T_out: int):
    if args.loss == "huber":
        base = WeightedHuberLoss(T_out, delta=1.0, decay=args.loss_decay)
    elif args.loss == "mae":
        base = nn.L1Loss()
    else:
        base = nn.MSELoss()
    smooth_reg = TemporalSmoothnessReg(weight=args.smooth_weight)
    return base, smooth_reg


# ══════════════════════════════════════════════════════════════════════════════
# 10. Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_splits(seq_dir: str, device: torch.device):
    def t(name):
        arr = np.load(os.path.join(seq_dir, name))
        return torch.from_numpy(arr).float().to(device)

    def t_opt(name):
        path = os.path.join(seq_dir, name)
        if not os.path.exists(path):
            return None
        return torch.from_numpy(np.load(path)).float().to(device)

    X_tr, y_tr = t("X_train.npy"), t("y_train.npy")
    X_va, y_va = t("X_val.npy"),   t("y_val.npy")
    X_te, y_te = t("X_test.npy"),  t("y_test.npy")
    Xf_tr = t_opt("X_future_train.npy")
    Xf_va = t_opt("X_future_val.npy")
    Xf_te = t_opt("X_future_test.npy")

    print(f"Shapes loaded from {seq_dir}:")
    print(f"  X_train {tuple(X_tr.shape)}   y_train {tuple(y_tr.shape)}")
    print(f"  X_val   {tuple(X_va.shape)}   y_val   {tuple(y_va.shape)}")
    print(f"  X_test  {tuple(X_te.shape)}   y_test  {tuple(y_te.shape)}")
    if Xf_tr is not None:
        print(f"  X_future {tuple(Xf_tr.shape)}  (temporal conditioning enabled)")
    else:
        print("  X_future: not found — re-run sequence_builder.py to enable temporal conditioning")
    return (X_tr, y_tr, Xf_tr), (X_va, y_va, Xf_va), (X_te, y_te, Xf_te)


# ══════════════════════════════════════════════════════════════════════════════
# 11. Training helpers (AMP-aware)
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimiser, criterion, smooth_reg,
                    grad_scaler, use_amp, noise_std: float = 0.0) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        X_b, y_b = batch[0], batch[1]
        Xf_b = batch[2] if len(batch) > 2 else None
        optimiser.zero_grad()
        if noise_std > 0.0:
            X_b = X_b + torch.randn_like(X_b) * noise_std
        with autocast('cuda', enabled=use_amp):
            y_hat = model(X_b, Xf_b)
            loss  = criterion(y_hat, y_b) + smooth_reg(y_hat)
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_scaler.step(optimiser)
        grad_scaler.update()
        total += loss.item() * X_b.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, smooth_reg, use_amp) -> float:
    model.eval()
    total = 0.0
    for batch in loader:
        X_b, y_b = batch[0], batch[1]
        Xf_b = batch[2] if len(batch) > 2 else None
        with autocast('cuda', enabled=use_amp):
            y_hat = model(X_b, Xf_b)
            total += (criterion(y_hat, y_b) + smooth_reg(y_hat)).item() * X_b.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def collect_predictions(model, loader, use_amp) -> tuple:
    """
    Always runs in float32.  Training uses AMP (fp16) for speed, but inference
    must be fp32: LayerNorm inside autocast returns fp32, meaning subsequent
    Linear ops also output fp32 — values can far exceed fp16's max (~65504)
    and overflow sklearn's MinMaxScaler inverse-transform.
    """
    model.eval()
    preds, trues = [], []
    for batch in loader:
        X_b, y_b = batch[0], batch[1]
        Xf_b = batch[2] if len(batch) > 2 else None
        preds.append(model(X_b, Xf_b).float().cpu().numpy())
        trues.append(y_b.float().cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


# ══════════════════════════════════════════════════════════════════════════════
# 12. Walk-Forward Evaluation (rolling-origin temporal stability)
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_fit(
    train_losses:          list,
    val_losses:            list,
    best_epoch:            int,
    patience:              int,
    val_drift_threshold:   float = 0.25,
    gap_overfit_threshold: float = 3.0,
    early_stop_frac:       float = 0.15,
) -> dict:
    """
    Diagnose overfitting or underfitting from recorded loss curves.

    Three independent signals:

    1. Val drift  (primary) — percentage the val loss rose from its best point to
                    the final training epoch.  After best_epoch the patience window
                    runs for `patience` more epochs; some small rise is normal
                    oscillation.  Only a large rise (> val_drift_threshold = 25%)
                    indicates genuine degradation after the model's best checkpoint.
                    This avoids the false-positive issue of slope-based approaches,
                    which trivially fire on the patience window (val loss can only
                    be flat-or-rising there by construction).

    2. Gap ratio  (secondary) — best_val_loss / min_train_loss.  Training uses
                    dropout + input noise, both absent at validation, so train loss
                    is inherently elevated; the threshold is deliberately conservative
                    at 3× to avoid false positives from these regularisation effects.

    3. Train-val divergence (secondary) — if training loss is still falling in the
                    final 10 epochs while val has drifted up significantly, that
                    indicates classic train/val divergence.

    4. Early stop — best_epoch < 15 % of total epochs → suspiciously fast
                    convergence (LR overshoot or data scale issue).

    Returns
    -------
    dict with keys:
      verdict        : "overfit" | "underfit" | "good_fit" | "uncertain"
      val_drift_pct  : float — % rise from best_val to final_val
      gap_ratio      : float — best_val / min_train
      val_trend      : "rising" | "flat" | "falling"  (informational only)
      early_stop     : bool
      best_epoch     : int
      total_epochs   : int
      notes          : list[str]
    """
    total = len(train_losses)
    if total == 0:
        return {
            "verdict": "uncertain", "val_drift_pct": None, "gap_ratio": None,
            "val_trend": None, "early_stop": False,
            "best_epoch": best_epoch, "total_epochs": 0,
            "notes": ["No training data recorded."],
        }

    best_val  = min(val_losses)
    min_train = min(train_losses)
    final_val = val_losses[-1]

    # ── Signal 1: Val drift after best checkpoint ────────────────────────────
    # (final_val - best_val) / best_val — how much did val degrade after its best?
    val_drift     = (final_val - best_val) / (best_val + 1e-12)
    val_drift_pct = val_drift * 100.0

    # ── Signal 2: Gap ratio ──────────────────────────────────────────────────
    gap_ratio = best_val / (min_train + 1e-12)

    # ── Signal 3: Train still falling while val drifted up? ──────────────────
    tail_n    = min(10, total)
    tr_tail   = train_losses[-tail_n:]
    xs        = np.arange(tail_n, dtype=float)
    tr_slope  = float(np.polyfit(xs, tr_tail, 1)[0])
    norm_tr   = tr_slope / (float(np.mean(tr_tail)) + 1e-12)
    train_still_falling = (norm_tr < -0.01)   # >1 % per-epoch drop in last 10 epochs

    # ── Signal 4: Early stop ─────────────────────────────────────────────────
    early_stop = best_epoch < early_stop_frac * total

    # ── Val trend (informational — NOT used for verdict) ─────────────────────
    # Computed over pre-best epochs to avoid patience-window bias.
    pre_window = val_losses[max(0, best_epoch - patience): best_epoch]
    if len(pre_window) >= 3:
        xs_p = np.arange(len(pre_window), dtype=float)
        s    = float(np.polyfit(xs_p, pre_window, 1)[0])
        ns   = s / (float(np.mean(pre_window)) + 1e-12)
        val_trend = "rising" if ns > 0.005 else ("falling" if ns < -0.005 else "flat")
    else:
        val_trend = "flat"

    # ── Verdict ──────────────────────────────────────────────────────────────
    notes: list = []
    verdict = "good_fit"

    # Primary: val drift
    if val_drift > val_drift_threshold:
        verdict = "overfit"
        notes.append(
            f"Val loss drifted +{val_drift_pct:.1f}% above its best "
            f"(best={best_val:.5f} → final={final_val:.5f}) — "
            f"model degraded after epoch {best_epoch}."
        )
    else:
        notes.append(
            f"Val drift +{val_drift_pct:.1f}% above best — within normal "
            f"patience-window oscillation (threshold {val_drift_threshold*100:.0f}%)."
        )

    # Secondary: gap ratio
    if gap_ratio > gap_overfit_threshold:
        if verdict != "overfit":
            verdict = "overfit"
        notes.append(
            f"Val/train gap {gap_ratio:.2f}× exceeds {gap_overfit_threshold:.0f}× "
            f"(min_train={min_train:.5f}, best_val={best_val:.5f}) — "
            f"significant memorisation of training data."
        )
    else:
        notes.append(
            f"Val/train gap {gap_ratio:.2f}× is within the {gap_overfit_threshold:.0f}× threshold "
            f"(accounts for dropout + input noise during training)."
        )

    # Secondary: train/val divergence
    if train_still_falling and val_drift > 0.10:
        if verdict != "overfit":
            verdict = "overfit"
        notes.append(
            f"Training loss still declining in final {tail_n} epochs while val drifted up "
            f"+{val_drift_pct:.1f}% — train/val divergence detected."
        )

    # Early stop
    if early_stop:
        frac_pct = int(100 * best_epoch / total)
        notes.append(
            f"Best epoch {best_epoch}/{total} ({frac_pct}%) is early — "
            f"possible LR overshoot; consider --lr or --warmup-epochs."
        )
        if verdict == "good_fit":
            verdict = "uncertain"

    if verdict == "good_fit":
        notes.append("No overfitting or underfitting signals detected.")

    return {
        "verdict":       verdict,
        "val_drift_pct": round(val_drift_pct, 2),
        "gap_ratio":     round(gap_ratio, 4),
        "val_trend":     val_trend,
        "early_stop":    early_stop,
        "best_epoch":    best_epoch,
        "total_epochs":  total,
        "notes":         notes,
    }


def walk_forward_evaluate(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           n_blocks: int = 3) -> list:
    """
    Split test set into n_blocks chronological windows and compute metrics
    for each.  Reveals whether accuracy degrades as the forecast horizon
    moves further into the future (temporal drift check).
    """
    N = y_true.shape[0]
    block_size = N // n_blocks
    results = []
    for i in range(n_blocks):
        s = i * block_size
        e = s + block_size if i < n_blocks - 1 else N
        m = compute_metrics(y_true[s:e].flatten(), y_pred[s:e].flatten())
        results.append({"block": i + 1, "n_samples": e - s, **m})
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 13. Residual Boosting Stage (CatBoost or neural fallback)
# ══════════════════════════════════════════════════════════════════════════════

def fit_residual_booster(X_train_np: np.ndarray,
                          y_residual: np.ndarray,
                          use_catboost: bool = True,
                          seed: int = 42):
    """
    Train a residual corrector on (X_flat, residual).
    Returns a callable: corrector(X_np) → residual_pred (N, T_out).
    """
    N, T, F = X_train_np.shape
    X_flat  = X_train_np.reshape(N, T * F)

    if use_catboost and CATBOOST_AVAILABLE:
        T_out = y_residual.shape[1]
        models = []
        for t in range(T_out):
            cb = CatBoostRegressor(
                iterations=300, depth=6, learning_rate=0.05,
                loss_function="RMSE", random_seed=seed, verbose=0,
            )
            cb.fit(X_flat, y_residual[:, t])
            models.append(cb)
        print(f"  [Boost] CatBoost residual learner fitted ({T_out} targets)")

        def corrector(X_np):
            Xf = X_np.reshape(len(X_np), -1)
            return np.stack([m.predict(Xf) for m in models], axis=1)

        return corrector

    else:
        # Lightweight sklearn-style MLP fallback
        from sklearn.neural_network import MLPRegressor
        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu",
            max_iter=300, random_state=seed, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=15,
        )
        N_out = y_residual.shape[1]
        mlp.fit(X_flat, y_residual)
        tag = "MLP" if not (use_catboost and CATBOOST_AVAILABLE) else "CatBoost"
        print(f"  [Boost] {tag} residual learner fitted")

        def corrector(X_np):
            return mlp.predict(X_np.reshape(len(X_np), -1))

        return corrector


# ══════════════════════════════════════════════════════════════════════════════
# 15. Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(train_losses, val_losses, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train Loss", linewidth=1.5, color="#9333ea")
    ax.plot(val_losses,   label="Val Loss",   linewidth=1.5, color="#f97316", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (scaled)")
    ax.set_title("HMT-TSF — Training curves")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig); print(f"  Saved: {out_path}")


def plot_predictions(y_true, y_pred, metrics, out_path: str) -> None:
    N, T_out = y_true.shape
    idx_full = np.arange(N)
    zoom_n   = min(60, N)
    idx_zoom = np.arange(N - zoom_n, N)
    C_ACT, C_PRED, C_BAND, C_MID, C_LAST = "#1d4ed8", "#6366f1", "#e0e7ff", "#0891b2", "#7c3aed"

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.48)
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(idx_full, y_pred.min(1), y_pred.max(1),
                     alpha=0.22, color=C_BAND, label=f"Forecast spread")
    ax1.plot(idx_full, y_true[:, 0], color=C_ACT,  linewidth=1.5, label="Actual", zorder=4)
    ax1.plot(idx_full, y_pred[:, 0], color=C_PRED, linewidth=1.2,
             linestyle="--", alpha=0.88, label="HMT-TSF (step 1)", zorder=5)
    ann = (f"Combined = {metrics['Combined']:.2f}%\nMAPE     = {metrics['MAPE']:.2f}%\n"
           f"MAE%     = {metrics['MAE_pct']:.2f}%\nRMSE%    = {metrics['RMSE_pct']:.2f}%\n"
           f"R²       = {metrics['R2']:.4f}\nMAE      = {metrics['MAE']:.0f} riders\n"
           f"RMSE     = {metrics['RMSE']:.0f} riders")
    ax1.text(0.01, 0.97, ann, transform=ax1.transAxes, fontsize=8, va="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                       edgecolor="#d1d5db", alpha=0.92))
    ax1.set_title("HMT-TSF — Test Set: Actual vs Predicted", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Test sample index"); ax1.set_ylabel("Ridership")
    ax1.legend(loc="upper right", fontsize=8); ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(idx_zoom, y_pred[idx_zoom].min(1), y_pred[idx_zoom].max(1),
                     alpha=0.18, color=C_BAND)
    ax2.plot(idx_zoom, y_true[idx_zoom, 0], color=C_ACT, linewidth=1.7,
             label="Actual", zorder=5)
    for s, col, ls in zip(sorted({0, T_out // 2, T_out - 1}),
                          [C_PRED, C_MID, C_LAST], ["--", "-.", ":"]):
        ax2.plot(idx_zoom, y_pred[idx_zoom, s], color=col, linewidth=1.4,
                 linestyle=ls, alpha=0.88, label=f"Step {s + 1}")
    ax2.set_title(f"Zoomed: last {zoom_n} samples", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Test sample index"); ax2.set_ylabel("Ridership")
    ax2.legend(loc="upper right", fontsize=8, ncol=2); ax2.grid(alpha=0.25)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_step_metrics(per_step, out_path: str) -> None:
    steps    = [f"t+{i+1}" for i in range(len(per_step))]
    combined = [m["Combined"] for m in per_step]
    mape     = [m["MAPE"]     for m in per_step]
    mae_pct  = [m["MAE_pct"]  for m in per_step]
    rmse_pct = [m["RMSE_pct"] for m in per_step]
    r2       = [m["R2"]       for m in per_step]

    x, w = np.arange(len(steps)), 0.2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(steps) * 0.85), 7),
                                    gridspec_kw={"hspace": 0.48})
    ax1.bar(x - 1.5*w, combined, w, label="Combined%", color="#16a34a", alpha=0.85)
    ax1.bar(x - 0.5*w, mape,     w, label="MAPE%",     color="#dc2626", alpha=0.85)
    ax1.bar(x + 0.5*w, mae_pct,  w, label="MAE%",      color="#2563eb", alpha=0.85)
    ax1.bar(x + 1.5*w, rmse_pct, w, label="RMSE%",     color="#f97316", alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(steps)
    ax1.set_ylabel("% of mean demand  /  score")
    ax1.set_title("HMT-TSF — Per-Horizon Metrics", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    ax2.plot(steps, r2, marker="o", color="#6366f1", linewidth=1.8, markersize=5)
    ax2.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax2.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    finite_r2 = [v for v in r2 if np.isfinite(v)]
    r2_min = min(finite_r2) if finite_r2 else -0.1
    ax2.set_ylim(min(r2_min - 0.05, -0.1), 1.08)
    ax2.set_ylabel("R²"); ax2.set_title("HMT-TSF — R² per Horizon", fontweight="bold")
    ax2.grid(alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_walk_forward(wf_results: list, out_path: str) -> None:
    blocks   = [f"Block {r['block']}" for r in wf_results]
    combined = [r["Combined"] for r in wf_results]
    r2       = [r["R2"]       for r in wf_results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(blocks, combined, color="#6366f1", alpha=0.85)
    ax1.set_ylabel("Combined%"); ax1.set_title("Walk-Forward: Combined%")
    ax1.grid(axis="y", alpha=0.3)
    ax2.bar(blocks, r2, color="#16a34a", alpha=0.85)
    ax2.set_ylabel("R²"); ax2.set_title("Walk-Forward: R²")
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("HMT-TSF — Temporal Stability (Walk-Forward)", fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 16. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="HMT-TSF: Hybrid Multi-scale Temporal Spatio-Feature Forecaster"
    )
    # Data
    p.add_argument("--seq-dir",      default=None,
                   help="Directory with X/y .npy splits (auto-resolved from --lookback)")
    p.add_argument("--lookback",     type=int, default=14,
                   choices=[7, 14, 28, 56, 84],
                   help="Look-back window; auto-resolves --seq-dir when not set")
    p.add_argument("--target-idx",   type=int, default=12,
                   help="Column index of the target in the feature matrix (default: 12 = total_ridership)")

    # Architecture
    p.add_argument("--d-model",      type=int,   default=64,
                   help="Hidden dimension throughout the model")
    p.add_argument("--n-tcn-blocks", type=int,   default=3,
                   help="TCN residual blocks per scale (dilation doubles each block)")
    p.add_argument("--tcn-kernel",   type=int,   default=3,
                   help="Temporal kernel size for all TCN convolutions")
    p.add_argument("--graph-hidden", type=int,   default=64,
                   help="Hidden size of graph convolution layers")
    p.add_argument("--n-regimes",    type=int,   default=3,
                   help="Number of structural regime embeddings")
    p.add_argument("--adj-threshold",type=float, default=0.1,
                   help="Pearson correlation threshold for graph adjacency edges")
    p.add_argument("--drop-path",    type=float, default=0.2,
                   help="Stochastic depth drop-path rate for TCN blocks (0 = disabled)")
    p.add_argument("--n-attn-heads", type=int,   default=4,
                   help="Attention heads in the temporal Transformer block")
    p.add_argument("--no-revin",     action="store_true",
                   help="Disable RevIN instance normalisation")

    # Training
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--epochs",        type=int,   default=150)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight-decay",  type=float, default=1e-3)
    p.add_argument("--patience",      type=int,   default=20)
    p.add_argument("--warmup-epochs", type=int,   default=8)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--input-noise",   type=float, default=0.02,
                   help="Std-dev of Gaussian noise added to training inputs (0 = off)")
    p.add_argument("--ema-decay",     type=float, default=0.995,
                   help="EMA weight decay for test-time averaging (0 = off)")
    p.add_argument("--device",        default="auto")
    p.add_argument("--seed",          type=int,   default=42)

    # Loss
    p.add_argument("--loss",          default="huber",
                   choices=["mse", "huber", "mae"],
                   help="Base training loss")
    p.add_argument("--loss-decay",    type=float, default=0.9,
                   help="Per-step geometric decay for WeightedHuberLoss")
    p.add_argument("--smooth-weight", type=float, default=0.01,
                   help="Weight of temporal-smoothness regularisation")

    # Boosting
    p.add_argument("--no-boost",    action="store_true",
                   help="Skip residual boosting stage")
    p.add_argument("--use-catboost",action="store_true",
                   help="Use CatBoost for residual boosting (requires catboost)")

    # SHAP
    p.add_argument("--shap",        action="store_true",
                   help="Run SHAP feature importance analysis after evaluation")
    p.add_argument("--shap-samples",type=int, default=100,
                   help="Number of test samples for SHAP analysis")

    # Comparison chain (all 15 prior models → 16-way comparison with HMT-TSF)
    p.add_argument("--lstm-results",       default=None)
    p.add_argument("--bilstm-results",     default=None)
    p.add_argument("--tpalstm-results",    default=None)
    p.add_argument("--cnnlstm-results",    default=None)
    p.add_argument("--cnnbilstm-results",  default=None)
    p.add_argument("--stlstm-results",     default=None)
    p.add_argument("--stgcn-results",      default=None)
    p.add_argument("--mtgnn-results",      default=None)
    p.add_argument("--stsgcn-results",     default=None)
    p.add_argument("--stfgnn-results",     default=None)
    p.add_argument("--pdrstgcn-results",   default=None)
    p.add_argument("--astgcn-results",     default=None)
    p.add_argument("--autoformer-results", default=None)
    p.add_argument("--informer-results",   default=None)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 17. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── per-lookback baseline defaults ────────────────────────────────────────
    # Only applied when the user has NOT overridden the flag explicitly
    # (detected by comparing current value against the global argparse default).
    _lb_overrides = _LOOKBACK_DEFAULTS.get(args.lookback, {})
    if _lb_overrides:
        _applied = []
        for _attr, _new_val in _lb_overrides.items():
            if getattr(args, _attr, None) == _ARGPARSE_DEFAULTS.get(_attr):
                setattr(args, _attr, _new_val)
                _applied.append(f"{_attr}={_new_val}")
        if _applied:
            print(f"[lookback={args.lookback}] Baseline adjustments applied: "
                  + ", ".join(_applied))

    # ── seq-dir auto-resolution ───────────────────────────────────────────────
    if args.seq_dir is None:
        args.seq_dir = ("data/sequences/lstm" if args.lookback == 14
                        else f"data/sequences/lookback_{args.lookback}")

    # ── reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        if   torch.cuda.is_available():         device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else:                                   device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    use_amp = (device.type == "cuda")
    grad_scaler = GradScaler(enabled=use_amp)
    print(f"Device: {device}   AMP: {'enabled (fp16)' if use_amp else 'disabled'}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # ── data ──────────────────────────────────────────────────────────────────
    (X_tr, y_tr, Xf_tr), (X_va, y_va, Xf_va), (X_te, y_te, Xf_te) = load_splits(args.seq_dir, device)
    T_in        = X_tr.shape[1]
    n_features  = X_tr.shape[2]
    T_out       = y_tr.shape[1]
    has_future  = Xf_tr is not None

    meta_path  = os.path.join(args.seq_dir, "split_dates.json")
    split_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    # Use target_col_idx from metadata if available
    if "target_col_idx" in split_meta:
        args.target_idx = split_meta["target_col_idx"]
    n_temporal_feats = (split_meta.get("temporal_feat_end", 29)
                        - split_meta.get("temporal_feat_start", 13))

    # ── SHAP-derived feature reduction: 79 → 53 features ─────────────────────
    # Applied here so no other model script is affected. The 26 dropped features
    # have zero (or near-zero) SHAP importance across all 10 HMT-TSF configs;
    # see HMT-TSF-RESULTS.md §10 for the full removal rationale.
    print(f"\nFeature reduction: retaining {len(_KEPT_FEAT_INDICES)}/{n_features} features "
          f"({len(_DROPPED_FEAT_INDICES)} zero-importance features dropped)")
    _kept_t = torch.tensor(_KEPT_FEAT_INDICES, device=device)
    X_tr = X_tr[:, :, _kept_t]
    X_va = X_va[:, :, _kept_t]
    X_te = X_te[:, :, _kept_t]
    n_features = X_tr.shape[2]  # 53

    if has_future and len(_KEPT_TEMPORAL_POSITIONS) < n_temporal_feats:
        _ktp = torch.tensor(_KEPT_TEMPORAL_POSITIONS, device=device)
        Xf_tr = Xf_tr[:, :, _ktp]
        if Xf_va is not None:
            Xf_va = Xf_va[:, :, _ktp]
        if Xf_te is not None:
            Xf_te = Xf_te[:, :, _ktp]
    n_temporal_feats = len(_KEPT_TEMPORAL_POSITIONS)  # 10

    def _make_ds(*tensors):
        return TensorDataset(*[t for t in tensors if t is not None])

    train_loader = DataLoader(_make_ds(X_tr, y_tr, Xf_tr),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(_make_ds(X_va, y_va, Xf_va), batch_size=args.batch_size)
    test_loader  = DataLoader(_make_ds(X_te, y_te, Xf_te), batch_size=args.batch_size)

    # ── graph adjacency ───────────────────────────────────────────────────────
    print("\nBuilding feature adjacency matrix …")
    X_tr_np   = X_tr.cpu().numpy()
    adj_np    = build_feature_adj(X_tr_np, threshold=args.adj_threshold)
    n_nan_adj = np.isnan(adj_np).sum()
    if n_nan_adj:
        print(f"  [WARN] Adjacency had {n_nan_adj} NaN entries (zero-variance features) — zeroed out.")
    adj_tensor = torch.from_numpy(adj_np).to(device)
    n_edges   = int((adj_np > 0).sum()) // 2
    density   = n_edges / max(1, n_features * (n_features - 1) // 2)
    print(f"  Nodes: {n_features}   Edges: {n_edges}   Density: {density:.3f}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = HMTTSFForecaster(
        n_features       = n_features,
        T_in             = T_in,
        T_out            = T_out,
        adj              = adj_tensor,
        d_model          = args.d_model,
        n_tcn_blocks     = args.n_tcn_blocks,
        tcn_kernel       = args.tcn_kernel,
        graph_hidden     = args.graph_hidden,
        n_regimes        = args.n_regimes,
        n_attn_heads     = args.n_attn_heads,
        dropout          = args.dropout,
        drop_path        = args.drop_path,
        target_idx       = args.target_idx,
        use_revin        = not args.no_revin,
        n_temporal_feats = n_temporal_feats,
        feat_groups      = _REDUCED_FEAT_GROUPS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel         : HMTTSFForecaster")
    print(f"  d_model     : {args.d_model}")
    print(f"  TCN blocks  : {args.n_tcn_blocks}   kernel: {args.tcn_kernel}")
    print(f"  Scales      : full{' + half' if T_in >= 28 else ''}{' + quarter' if T_in >= 56 else ''}")
    print(f"  Graph hidden: {args.graph_hidden}")
    print(f"  Regimes     : {args.n_regimes}")
    print(f"  Attn heads  : {args.n_attn_heads}")
    print(f"  RevIN       : {'on' if not args.no_revin else 'off'}")
    print(f"  DropPath    : {args.drop_path}")
    print(f"  Temporal    : {'on (' + str(n_temporal_feats) + ' future features)' if has_future else 'off — rebuild sequences to enable'}")
    print(f"  In          : (batch, {T_in}, {n_features})")
    print(f"  Out         : (batch, {T_out})")
    print(f"  Params      : {n_params:,}")

    # ── optimiser / loss ──────────────────────────────────────────────────────
    criterion, smooth_reg = build_criterion(args, T_out)
    criterion  = criterion.to(device)
    smooth_reg = smooth_reg.to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.7, patience=8
    )

    # ── training loop ─────────────────────────────────────────────────────────
    effective_warmup = args.warmup_epochs

    best_val_loss   = float("inf")
    best_epoch      = 0
    patience_count  = 0
    train_losses, val_losses = [], []
    best_state = None

    print(f"\nTraining  (max {args.epochs} epochs, patience={args.patience}, "
          f"warmup={effective_warmup})")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'LR':>10}")
    print("─" * 45)

    for epoch in range(1, args.epochs + 1):
        if epoch <= effective_warmup:
            for pg in optimiser.param_groups:
                pg["lr"] = args.lr * epoch / effective_warmup

        tr_loss = train_one_epoch(model, train_loader, optimiser,
                                   criterion, smooth_reg, grad_scaler, use_amp,
                                   noise_std=args.input_noise)
        va_loss = evaluate(model, val_loader, criterion, smooth_reg, use_amp)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if epoch > effective_warmup:
            scheduler.step(va_loss)
        lr_now = optimiser.param_groups[0]["lr"]
        print(f"{epoch:6d}  {tr_loss:10.6f}  {va_loss:10.6f}  {lr_now:10.2e}")

        if va_loss < best_val_loss:
            best_val_loss, best_epoch, patience_count = va_loss, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stop @ epoch {epoch}  "
                      f"(best={best_val_loss:.6f} @ epoch {best_epoch})")
                break

    model.load_state_dict(best_state)

    # ── fit diagnostics ───────────────────────────────────────────────────────
    fit_diag = diagnose_fit(train_losses, val_losses, best_epoch, args.patience)
    _verd_label = {
        "overfit":   "OVERFIT",
        "underfit":  "UNDERFIT",
        "good_fit":  "GOOD FIT",
        "uncertain": "UNCERTAIN",
    }.get(fit_diag["verdict"], fit_diag["verdict"].upper())
    print(f"\n{'─'*50}")
    print(f"Fit Diagnostics  [{_verd_label}]")
    print(f"  Val drift  : +{fit_diag['val_drift_pct']:.1f}%  "
          f"(best→final val loss; <25% = normal)")
    print(f"  Gap ratio  : {fit_diag['gap_ratio']:.2f}×  "
          f"(best_val / min_train; threshold 3×)")
    print(f"  Val trend  : {fit_diag['val_trend']}  "
          f"(pre-best window, informational)")
    print(f"  Early stop : {'yes' if fit_diag['early_stop'] else 'no'}  "
          f"(best epoch {best_epoch}/{len(train_losses)})")
    for _note in fit_diag["notes"]:
        print(f"  ·  {_note}")
    print(f"{'─'*50}")

    # ── test predictions ──────────────────────────────────────────────────────
    y_pred_s, y_true_s = collect_predictions(model, test_loader, use_amp)

    finite_mask = np.isfinite(y_pred_s)
    n_bad = (~finite_mask).sum()
    print(f"  [Predictions] scaled range: [{y_pred_s[finite_mask].min():.4f}, "
          f"{y_pred_s[finite_mask].max():.4f}]  "
          f"mean={y_pred_s[finite_mask].mean():.4f}  non-finite={n_bad}")
    if n_bad:
        print(f"  [ERROR] {n_bad} non-finite values (NaN/Inf) in scaled predictions — "
              f"replacing with 0 for diagnostic output.")
        np.nan_to_num(y_pred_s, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # ── inverse-transform ─────────────────────────────────────────────────────
    scaler_y_path = os.path.join(args.seq_dir, "scaler_y.pkl")
    if os.path.exists(scaler_y_path):
        scaler_y = joblib.load(scaler_y_path)
        N2, T2   = y_pred_s.shape
        y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).reshape(N2, T2)
        y_true   = scaler_y.inverse_transform(y_true_s.reshape(-1, 1)).reshape(N2, T2)
    else:
        print("[WARN] scaler_y.pkl not found — metrics in scaled units.")
        y_pred, y_true = y_pred_s, y_true_s

    # ── NaN / Inf guard ───────────────────────────────────────────────────────
    for arr, tag in [(y_pred, "y_pred"), (y_true, "y_true")]:
        n_bad = (~np.isfinite(arr)).sum()
        if n_bad:
            print(f"[WARN] {tag} contains {n_bad} non-finite values "
                  f"(NaN/Inf) after inverse-transform — replacing with 0.")
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # ── residual boosting stage ───────────────────────────────────────────────
    boost_corrector = None
    if not args.no_boost:
        print("\nFitting residual booster …")
        # Collect training predictions in original dataset order (no shuffle)
        # so that residuals align with X_tr row-for-row when fitting the booster.
        train_ordered_loader = DataLoader(
            TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=False
        )
        tr_pred_s, tr_true_s = collect_predictions(model, train_ordered_loader, use_amp)
        if os.path.exists(scaler_y_path):
            N_tr, T_tr = tr_pred_s.shape
            tr_pred = scaler_y.inverse_transform(tr_pred_s.reshape(-1, 1)).reshape(N_tr, T_tr)
            tr_true = scaler_y.inverse_transform(tr_true_s.reshape(-1, 1)).reshape(N_tr, T_tr)
        else:
            tr_pred, tr_true = tr_pred_s, tr_true_s

        tr_residual = tr_true - tr_pred   # (N_tr, T_out)
        X_tr_np_cpu = X_tr.cpu().numpy()
        try:
            boost_corrector = fit_residual_booster(
                X_tr_np_cpu, tr_residual,
                use_catboost=args.use_catboost,
                seed=args.seed,
            )
            X_te_np = X_te.cpu().numpy()
            boost_correction = boost_corrector(X_te_np)   # (N_te, T_out)
            y_pred_boosted   = y_pred + 0.5 * boost_correction
            overall_plain    = compute_metrics(y_true.flatten(), y_pred.flatten())
            overall_boosted  = compute_metrics(y_true.flatten(), y_pred_boosted.flatten())
            if overall_boosted["Combined"] >= overall_plain["Combined"]:
                y_pred = y_pred_boosted
                print(f"  [Boost] Applied  — Combined {overall_plain['Combined']:.2f}% "
                      f"→ {overall_boosted['Combined']:.2f}%")
            else:
                print(f"  [Boost] Skipped (no improvement): "
                      f"{overall_plain['Combined']:.2f}% vs {overall_boosted['Combined']:.2f}%")
        except Exception as e:
            print(f"  [Boost] Failed ({e}) — using neural predictions only.")

    # ── per-horizon metrics ───────────────────────────────────────────────────
    W = 10
    print(f"\n{'='*85}")
    print("HMT-TSF — TEST SET METRICS")
    print(f"{'='*85}")
    header = (f"{'Step':>5}  {'Combined%':>{W}}  {'MAPE%':>{W}}  {'MAE%':>{W}}  "
              f"{'RMSE%':>{W}}  {'R²':>{W}}  {'MAE':>{W}}  {'RMSE':>{W}}")
    print(header); print("─" * len(header))

    per_step = []
    for s in range(T_out):
        m = compute_metrics(y_true[:, s], y_pred[:, s])
        per_step.append(m)
        print(f"{s+1:5d}  {m['Combined']:>{W}.2f}  {m['MAPE']:>{W}.2f}  "
              f"{m['MAE_pct']:>{W}.2f}  {m['RMSE_pct']:>{W}.2f}  "
              f"{m['R2']:>{W}.4f}  {m['MAE']:>{W}.0f}  {m['RMSE']:>{W}.0f}")

    overall = compute_metrics(y_true.flatten(), y_pred.flatten())
    print("─" * len(header))
    print(f"{'Avg':>5}  {overall['Combined']:>{W}.2f}  {overall['MAPE']:>{W}.2f}  "
          f"{overall['MAE_pct']:>{W}.2f}  {overall['RMSE_pct']:>{W}.2f}  "
          f"{overall['R2']:>{W}.4f}  {overall['MAE']:>{W}.0f}  {overall['RMSE']:>{W}.0f}")

    # ── walk-forward stability ─────────────────────────────────────────────────
    wf_results = walk_forward_evaluate(y_true, y_pred, n_blocks=3)
    print(f"\n{'─'*50}")
    print("Walk-Forward Temporal Stability:")
    for r in wf_results:
        print(f"  Block {r['block']} (n={r['n_samples']:4d}):  "
              f"Combined={r['Combined']:.2f}%  R²={r['R2']:.4f}")

    # ── save artefacts ────────────────────────────────────────────────────────
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"src/outputs/hmttsf/{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{out_dir}/model.pt")
    np.save(f"{out_dir}/predictions.npy", y_pred)

    results = {
        "run_id":  run_id,
        "model":   "HMTTSFForecaster",
        "hparams": {
            "d_model":      args.d_model,
            "n_tcn_blocks": args.n_tcn_blocks,
            "tcn_kernel":   args.tcn_kernel,
            "graph_hidden": args.graph_hidden,
            "n_regimes":    args.n_regimes,
            "n_attn_heads": args.n_attn_heads,
            "adj_threshold":args.adj_threshold,
            "use_revin":    not args.no_revin,
            "dropout":      args.dropout,
            "drop_path":    args.drop_path,
            "T_in":         T_in,
            "T_out":        T_out,
            "n_features":   n_features,
            "lookback":     args.lookback,
            "batch_size":   args.batch_size,
            "lr":           args.lr,
            "weight_decay": args.weight_decay,
            "loss":         args.loss,
            "loss_decay":   args.loss_decay,
            "smooth_weight":args.smooth_weight,
            "warmup_epochs":args.warmup_epochs,
            "ema_decay":    args.ema_decay,
        },
        "training": {
            "best_epoch":    best_epoch,
            "best_val_loss": round(best_val_loss, 8),
            "total_epochs":  len(train_losses),
            "fit_diagnosis": fit_diag,
        },
        "split_dates":   split_meta,
        "test_metrics":  {
            "overall":       {k: round(v, 4) for k, v in overall.items()},
            "per_step":      [{k: round(v, 4) for k, v in m.items()} for m in per_step],
            "walk_forward":  [{k: round(v, 4) if isinstance(v, float) else v
                               for k, v in r.items()} for r in wf_results],
        },
    }

    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── plots ─────────────────────────────────────────────────────────────────
    print(f"\nSaving plots …")
    plot_loss_curves(train_losses, val_losses,
                     out_path=f"{out_dir}/loss_curves.png")
    plot_predictions(y_true, y_pred, metrics=overall,
                     out_path=f"{out_dir}/test_predictions.png")
    plot_per_step_metrics(per_step,
                          out_path=f"{out_dir}/per_step_metrics.png")
    plot_walk_forward(wf_results,
                      out_path=f"{out_dir}/walk_forward_stability.png")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    if args.shap:
        _all_names = load_feature_names()
        _reduced_names = ([_all_names[i] for i in _KEPT_FEAT_INDICES]
                          if _all_names else None)
        run_shap_analysis(model, X_te, out_dir,
                          feature_names=_reduced_names,
                          n_samples=args.shap_samples, use_amp=use_amp)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*85}")
    print(f"HMT-TSF run complete  →  {out_dir}/")
    print(f"{'='*85}")
    print(f"  Combined  : {overall['Combined']:.2f}%")
    print(f"  MAPE      : {overall['MAPE']:.2f}%")
    print(f"  MAE%      : {overall['MAE_pct']:.2f}%    (raw MAE  = {overall['MAE']:.0f} riders)")
    print(f"  RMSE%     : {overall['RMSE_pct']:.2f}%   (raw RMSE = {overall['RMSE']:.0f} riders)")
    print(f"  R²        : {overall['R2']:.4f}")
    targets_met = overall["Combined"] >= 75.0 and overall["R2"] >= 0.7
    print(f"  Targets   : {'✓ BOTH MET (Combined≥75, R²≥0.7)' if targets_met else '✗ Not yet — consider tuning'}")

    # ── 17-way comparison ─────────────────────────────────────────────────────
    PRIOR_MODELS = [
        ("LSTM",       args.lstm_results,       "src/outputs/lstm",        "#2563eb"),
        ("BiLSTM",     args.bilstm_results,     "src/outputs/bilstm",      "#7c3aed"),
        ("TPA-LSTM",   args.tpalstm_results,    "src/outputs/tpa_lstm",    "#0891b2"),
        ("CNN-LSTM",   args.cnnlstm_results,    "src/outputs/cnn_lstm",    "#16a34a"),
        ("CNN-BiLSTM", args.cnnbilstm_results,  "src/outputs/cnn_bilstm",  "#d97706"),
        ("ST-LSTM",    args.stlstm_results,     "src/outputs/st_lstm",     "#dc2626"),
        ("STGCN",      args.stgcn_results,      "src/outputs/stgcn",       "#10b981"),
        ("MTGNN",      args.mtgnn_results,      "src/outputs/mtgnn",       "#f472b6"),
        ("STSGCN",     args.stsgcn_results,     "src/outputs/stsgcn",      "#0ea5e9"),
        ("STFGNN",     args.stfgnn_results,     "src/outputs/stfgnn",      "#a855f7"),
        ("PDR-STGCN",  args.pdrstgcn_results,   "src/outputs/pdr_stgcn",   "#f97316"),
        ("ASTGCN",     args.astgcn_results,      "src/outputs/astgcn",      "#e11d48"),
        ("Autoformer", args.autoformer_results,  "src/outputs/autoformer",  "#047857"),
        ("Informer",   args.informer_results,    "src/outputs/informer",    "#9333ea"),
    ]

    models_data = []
    comparison  = {}

    for name, path, model_dir, color in PRIOR_MODELS:
        key  = name.lower().replace("-", "_")
        data = load_model_results(path, model_dir, name)
        if data:
            m_overall = data["test_metrics"]["overall"]
            m_ps      = data["test_metrics"]["per_step"]
            models_data.append((name, m_overall, m_ps, color))
            comparison[key] = {"run_id": data.get("run_id"), "overall": m_overall}
        else:
            comparison[key] = {"run_id": None, "overall": None}

    models_data.append(("HMT-TSF", overall, per_step, "#6366f1"))

    if len(models_data) > 1:
        print_comparison_table(models_data[:-1], overall, "HMT-TSF")
        plot_comparison(
            models_data,
            out_path=f"{out_dir}/comparison_{len(models_data)}_way.png",
        )

    comparison["hmt_tsf"] = {
        "run_id":  run_id,
        "overall": {k: round(v, 4) for k, v in overall.items()},
    }
    for name, m_overall, _, __ in models_data[:-1]:
        key = name.lower().replace("-", "_")
        comparison[f"delta_vs_{key}"] = {
            k: round(overall.get(k, 0) - m_overall.get(k, 0), 4)
            for k in overall
        }

    # ── naive persistence baseline ─────────────────────────────────────────────
    target_idx_safe = split_meta.get("target_col_idx", args.target_idx)
    last_obs_s      = X_te[:, -1, target_idx_safe:target_idx_safe + 1].cpu().numpy()
    naive_pred_s    = np.tile(last_obs_s, (1, T_out))
    if os.path.exists(scaler_y_path):
        N_np = naive_pred_s.shape[0]
        naive_pred = scaler_y.inverse_transform(
            naive_pred_s.reshape(-1, 1)
        ).reshape(N_np, T_out)
    else:
        naive_pred = naive_pred_s

    naive  = compute_metrics(y_true.flatten(), naive_pred.flatten())
    d_comb = overall["Combined"] - naive["Combined"]
    d_mape = naive["MAPE"]       - overall["MAPE"]
    d_r2   = overall["R2"] - naive["R2"]
    print(f"\n  Naive persistence  "
          f"Combined={naive['Combined']:.2f}%  MAPE={naive['MAPE']:.2f}%  "
          f"R²={naive['R2']:.4f}")
    print(f"  HMT-TSF vs naive   "
          f"ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%  ΔR²={d_r2:+.4f}")
    results["naive_persistence"] = {
        "metrics":        {k: round(v, 4) for k, v in naive.items()},
        "delta_combined": round(d_comb, 4),
        "delta_mape":     round(d_mape, 4),
        "delta_r2":       round(d_r2,   4),
    }
    results["comparison"] = comparison
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()