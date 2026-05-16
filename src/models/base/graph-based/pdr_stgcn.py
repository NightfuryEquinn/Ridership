"""
pdr_stgcn.py  — Periodicity-Aware Dynamic Relational STGCN (PDR-STGCN)

Novel architecture combining three ideas into the STGCN backbone:

1. Periodicity Encoding
   A second input channel is created by computing the periodic lag-difference:
     x_diff[t] = x[t] - x[t - period]   (zero-padded for t < period)
   For ridership data with T_in=14 and weekly period=7, this captures how each
   feature deviates from the same day last week, making seasonal patterns
   explicit without any extra parameters.

2. Dynamic Relational Graph Convolution
   Each ST block replaces the fixed Chebyshev convolution with a two-path
   convolution that mixes a static base graph with an input-adaptive dynamic
   graph:
     • Static path : A_sym @ h @ W_static
                     A_sym = D^{-1/2} A D^{-1/2}, built from absolute Pearson
                     correlation of training features (same as STGCN/STFGNN).
     • Dynamic path: softmax(Q @ K^T / sqrt(d_k)) @ V
                     Q, K, V are linear projections of the current node
                     features h, so the adjacency adapts per sample and per
                     time step.
     • Mixing      : out = σ(λ) · static + (1-σ(λ)) · dynamic
                     λ is a learned scalar initialised to 0 (equal mix).

3. ST Block Structure (unchanged from STGCN)
   TemporalGatedConv → DynamicRelationalGraphConv → TemporalGatedConv → BN

Architecture:
  X (B, T_in, N)
    → periodic diff encoder  → (B, N, 2, T_in)   [2 channels: orig + diff]

  PDR-ST Block k:
    TemporalGatedConv   (B, N, C_in,  T)    →  (B, N, C_mid, T-Kt+1)
    DynamicRelGraph     (B*T', N, C_mid)    →  (B*T', N, C_mid)
    TemporalGatedConv   (B, N, C_mid, T')   →  (B, N, C_out, T'-Kt+1)
    BatchNorm2d

  Output Layer:
    TemporalGatedConv → mean over N → flatten → MLP head → (B, T_out)

Comparison:
  --lstm-results, --bilstm-results, --tpalstm-results,
  --cnnlstm-results, --cnnbilstm-results, --stlstm-results,
  --stgcn-results, --mtgnn-results, --stsgcn-results, --stfgnn-results,
  --astgcn-results, --tft-results, --autoformer-results, --informer-results
  All optional; each auto-detects the most recent run if omitted.

Usage:
  python src/models/graph-based/pdr_stgcn.py
  python src/models/graph-based/pdr_stgcn.py --hidden 64 --period 7 --dk 32
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys as _sys
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from src.utils.metrics import compute_metrics
from src.utils.comparison_table import (
    load_model_results,
    print_comparison_table,
    plot_comparison,
)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="PDR-STGCN forecaster with 12-way comparison")
    p.add_argument("--seq-dir",            default=None,
                   help="Directory with X/y .npy splits")
    p.add_argument("--hidden",             type=int,   default=128,
                   help="Graph conv channel width (C_mid = C_out = hidden)")
    p.add_argument("--kt",                 type=int,   default=3,
                   help="Temporal conv kernel size")
    p.add_argument("--n-blocks",           type=int,   default=2,
                   help="Number of PDR-ST-Conv blocks")
    p.add_argument("--period",             type=int,   default=7,
                   help="Periodic lag for difference encoding (default=7 for weekly)")
    p.add_argument("--dk",                 type=int,   default=32,
                   help="Key/Query dimension for dynamic attention graph")
    p.add_argument("--adj-threshold",      type=float, default=0.1,
                   help="Min abs Pearson correlation to keep an edge")
    p.add_argument("--dropout",            type=float, default=0.1,
                   help="Dropout on graph conv output")
    p.add_argument("--weight-decay",       type=float, default=1e-4,
                   help="Adam weight decay")
    p.add_argument("--batch-size",         type=int,   default=32)
    p.add_argument("--epochs",             type=int,   default=150)
    p.add_argument("--lr",                 type=float, default=1e-3)
    p.add_argument("--patience",           type=int,   default=15)
    p.add_argument("--device",             default="auto", help="cpu | cuda | mps | auto")
    p.add_argument("--seed",               type=int,   default=42)
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
    p.add_argument("--astgcn-results",     default=None)
    p.add_argument("--tft-results",        default=None)
    p.add_argument("--autoformer-results", default=None)
    p.add_argument("--informer-results",   default=None)
    p.add_argument("--lookback",      type=int,   default=14, choices=[14, 28, 56],
                   help="Look-back window; auto-selects seq-dir when --seq-dir is not set")
    p.add_argument("--loss",          default="huber", choices=["mse", "huber", "mae"],
                   help="Training loss: mse | huber (default) | mae")
    p.add_argument("--warmup-epochs", type=int,   default=5,
                   help="Linear LR warm-up epochs before ReduceLROnPlateau kicks in")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Graph construction helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_adj(X_train: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """
    Build a raw adjacency matrix from absolute Pearson correlation of features.
    X_train : (N_samples, T_in, N_features)
    Returns  : A_raw (N, N) float32, diagonal = 0, values >= threshold.
    """
    X_np   = X_train.cpu().numpy()
    X_flat = X_np.reshape(-1, X_np.shape[-1])
    corr   = np.corrcoef(X_flat.T).astype(np.float32)
    A      = np.abs(corr)
    np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 0.0)
    return torch.from_numpy(A)


def sym_normalize(A: torch.Tensor) -> torch.Tensor:
    """
    Symmetric normalisation: A_sym = D^{-1/2} A D^{-1/2}.
    Used directly in the static convolution path (no scaled Laplacian needed).
    """
    D          = A.sum(dim=1).clamp(min=1e-9)
    D_inv_sqrt = D.pow(-0.5)
    return D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# Model components
# ══════════════════════════════════════════════════════════════════════════════

class TemporalGatedConv(nn.Module):
    """
    Gated 1-D temporal convolution (GLU gate: tanh * sigmoid).
    Input  : (B, C_in, T)
    Output : (B, C_out, T - kernel_size + 1)   [no padding — causal shrinkage]
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h      = self.conv(x)
        h1, h2 = h.chunk(2, dim=1)
        return torch.tanh(h1) * torch.sigmoid(h2)


class DynamicRelationalGraphConv(nn.Module):
    """
    Two-path graph convolution mixing a static and a dynamic adjacency.

    Static path : A_sym @ h @ W_static
    Dynamic path: softmax(Q @ K^T / sqrt(d_k)) @ V
    Output      : σ(λ) * static_out + (1 - σ(λ)) * dynamic_out

    λ is a learned scalar initialised to 0 (equal mixture at the start).
    """

    def __init__(self, in_channels: int, out_channels: int, d_k: int = 32):
        super().__init__()
        self.d_k = d_k
        self.W_s = nn.Linear(in_channels, out_channels, bias=False)
        self.W_Q = nn.Linear(in_channels, d_k, bias=False)
        self.W_K = nn.Linear(in_channels, d_k, bias=False)
        self.W_V = nn.Linear(in_channels, out_channels, bias=False)
        self.lam = nn.Parameter(torch.zeros(1))   # sigmoid(0) = 0.5

    def forward(self, h: torch.Tensor, A_sym: torch.Tensor) -> torch.Tensor:
        """
        h     : (B_T, N, C_in)
        A_sym : (N, N)   symmetric normalised adjacency (buffer)
        Returns (B_T, N, C_out)
        """
        # Static path: one-hop message passing with sym-normalised adjacency
        h_agg = torch.einsum("nm,bmc->bnc", A_sym, h)   # (B_T, N, C_in)
        h_sta = self.W_s(h_agg)                          # (B_T, N, C_out)

        # Dynamic path: self-attention adjacency
        Q     = self.W_Q(h)                              # (B_T, N, d_k)
        K     = self.W_K(h)                              # (B_T, N, d_k)
        V     = self.W_V(h)                              # (B_T, N, C_out)
        score = torch.bmm(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)
        A_dyn = torch.softmax(score, dim=-1)             # (B_T, N, N)
        h_dyn = torch.bmm(A_dyn, V)                     # (B_T, N, C_out)

        lam = torch.sigmoid(self.lam)
        return lam * h_sta + (1.0 - lam) * h_dyn


class PDRSTConvBlock(nn.Module):
    """
    One PDR-ST-Conv block:
      TemporalGated → DynamicRelationalGraph → TemporalGated → BN

    Each block reduces T by 2*(Kt - 1).
    Input/Output shape: (B, N, C_in, T) / (B, N, C_out, T - 2*(Kt-1))
    """

    def __init__(
        self,
        in_channels:  int,
        mid_channels: int,
        out_channels: int,
        Kt:           int = 3,
        d_k:          int = 32,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.temp1   = TemporalGatedConv(in_channels,  mid_channels, Kt)
        self.graph   = DynamicRelationalGraphConv(mid_channels, mid_channels, d_k)
        self.temp2   = TemporalGatedConv(mid_channels, out_channels, Kt)
        self.bn      = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_sym: torch.Tensor) -> torch.Tensor:
        """
        x     : (B, N, C_in, T)
        A_sym : (N, N)
        Returns (B, N, C_out, T - 2*(Kt-1))
        """
        B, N, C, T = x.shape

        # ── Temporal conv 1 ───────────────────────────────────────────────────
        xt = x.reshape(B * N, C, T)
        xt = self.temp1(xt)                                # (B*N, C_mid, T1)
        T1, C1 = xt.shape[-1], xt.shape[1]
        xt = xt.reshape(B, N, C1, T1)

        # ── Dynamic relational graph conv (per time step) ─────────────────────
        xg = xt.permute(0, 3, 1, 2).reshape(B * T1, N, C1)   # (B*T1, N, C_mid)
        xg = self.graph(xg, A_sym)
        xg = F.relu(xg)
        xg = self.dropout(xg)
        xg = xg.reshape(B, T1, N, C1).permute(0, 2, 3, 1)    # (B, N, C_mid, T1)

        # ── Temporal conv 2 ───────────────────────────────────────────────────
        xt2 = xg.reshape(B * N, C1, T1)
        xt2 = self.temp2(xt2)                              # (B*N, C_out, T2)
        T2, C2 = xt2.shape[-1], xt2.shape[1]
        xt2 = xt2.reshape(B, N, C2, T2)

        # ── BatchNorm (treat N as H, T2 as W) ─────────────────────────────────
        xt2 = self.bn(xt2.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        return xt2


class PDRSTGCNForecaster(nn.Module):
    """
    Periodicity-Aware Dynamic Relational STGCN forecaster.

    The symmetric-normalised adjacency A_sym is precomputed from training-data
    feature correlation and stored as a buffer (travels with .to(device)).
    Periodicity encoding is applied inline in forward() — zero-overhead at
    inference since it uses only tensor ops.

    Input  : (B, T_in, n_features)
    Output : (B, T_out)
    """

    def __init__(
        self,
        n_features: int,
        hidden:     int,
        n_blocks:   int,
        Kt:         int,
        period:     int,
        d_k:        int,
        T_in:       int,
        T_out:      int,
        A_raw:      torch.Tensor,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.period     = period

        # ── Static adjacency buffer ───────────────────────────────────────────
        A_sym = sym_normalize(A_raw)
        self.register_buffer("A_sym", A_sym)

        # ── PDR-ST-Conv blocks ────────────────────────────────────────────────
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            c_in = 2 if i == 0 else hidden   # 2 channels: original + periodic diff
            self.blocks.append(PDRSTConvBlock(c_in, hidden, hidden, Kt, d_k, dropout))

        # ── Output temporal conv ──────────────────────────────────────────────
        self.out_temp = TemporalGatedConv(hidden, hidden, Kt)

        # ── Compute head input size ───────────────────────────────────────────
        # Each ST block: T → T - 2*(Kt-1).  Output conv: T → T - (Kt-1).
        T_after = T_in - (Kt - 1) * (2 * n_blocks + 1)
        assert T_after > 0, (
            f"T_after={T_after} ≤ 0 with Kt={Kt}, n_blocks={n_blocks}, T_in={T_in}. "
            f"Reduce Kt or n_blocks."
        )

        head_in = hidden * T_after
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, T_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T_in, N)   N = n_features"""
        B, T, N = x.shape
        p = self.period

        # ── Periodicity encoding ──────────────────────────────────────────────
        # x_diff[t] = x[t] - x[t-p]   for t >= p, else 0
        x_diff = torch.zeros_like(x)
        if p < T:
            x_diff[:, p:, :] = x[:, p:, :] - x[:, :T - p, :]

        # Stack to 2 channels: (B, T, N, 2) → (B, N, 2, T)
        enc = torch.stack([x, x_diff], dim=-1).permute(0, 2, 3, 1)

        # ── PDR-ST blocks ─────────────────────────────────────────────────────
        for block in self.blocks:
            enc = block(enc, self.A_sym)    # (B, N, C_out, T')

        # ── Output temporal conv ──────────────────────────────────────────────
        B2, N2, C2, T2 = enc.shape
        enc = enc.reshape(B2 * N2, C2, T2)
        enc = self.out_temp(enc)            # (B*N, hidden, T'')
        T3  = enc.shape[-1]
        enc = enc.reshape(B2, N2, -1, T3)

        # ── Pool over nodes, flatten, MLP ─────────────────────────────────────
        enc = enc.mean(dim=1)               # (B, hidden, T'')
        enc = enc.reshape(B2, -1)
        return self.head(enc)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_splits(seq_dir: str, device: torch.device):
    def t(name):
        return torch.from_numpy(np.load(os.path.join(seq_dir, name))).float().to(device)

    X_tr, y_tr = t("X_train.npy"), t("y_train.npy")
    X_va, y_va = t("X_val.npy"),   t("y_val.npy")
    X_te, y_te = t("X_test.npy"),  t("y_test.npy")

    print(f"Shapes loaded from {seq_dir}:")
    print(f"  X_train {tuple(X_tr.shape)}   y_train {tuple(y_tr.shape)}")
    print(f"  X_val   {tuple(X_va.shape)}   y_val   {tuple(y_va.shape)}")
    print(f"  X_test  {tuple(X_te.shape)}   y_test  {tuple(y_te.shape)}")
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimiser, criterion, device) -> float:
    model.train()
    total     = 0.0
    n_skipped = 0
    for X_b, y_b in loader:
        optimiser.zero_grad()
        loss = criterion(model(X_b), y_b)
        if not torch.isfinite(loss):
            n_skipped += 1
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total += loss.item() * X_b.size(0)
    if n_skipped:
        print(f"  [WARN] {n_skipped} batch(es) skipped — non-finite loss")
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion) -> float:
    model.eval()
    total = 0.0
    for X_b, y_b in loader:
        total += criterion(model(X_b), y_b).item() * X_b.size(0)
    return total / len(loader.dataset)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(train_losses, val_losses, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train Loss", linewidth=1.5, color="#dc2626")
    ax.plot(val_losses,   label="Val Loss",   linewidth=1.5, color="#f97316",
            linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (scaled)")
    ax.set_title("PDR-STGCN — Training curves")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_predictions(y_true, y_pred, metrics, out_path: str) -> None:
    N, T_out = y_true.shape
    idx_full = np.arange(N)
    zoom_n   = min(60, N)
    idx_zoom = np.arange(N - zoom_n, N)

    actual_s1    = y_true[:, 0]
    predicted_s1 = y_pred[:, 0]
    pred_min     = y_pred.min(axis=1)
    pred_max     = y_pred.max(axis=1)

    C_ACT  = "#1d4ed8"; C_PRED = "#f97316"; C_BAND = "#fed7aa"
    C_MID  = "#0891b2"; C_LAST = "#7c3aed"

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.48)

    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(idx_full, pred_min, pred_max, alpha=0.22, color=C_BAND,
                     label=f"Forecast spread (step 1–{T_out})")
    ax1.plot(idx_full, actual_s1,    color=C_ACT,  linewidth=1.5, label="Actual", zorder=4)
    ax1.plot(idx_full, predicted_s1, color=C_PRED, linewidth=1.2, linestyle="--",
             alpha=0.88, label="PDR-STGCN predicted (step 1)", zorder=5)
    ann = (
        f"Combined = {metrics['Combined']:.2f}%\n"
        f"MAPE     = {metrics['MAPE']:.2f}%\n"
        f"MAE%     = {metrics['MAE_pct']:.2f}%\n"
        f"RMSE%    = {metrics['RMSE_pct']:.2f}%\n"
        f"R²       = {metrics['R2']:.4f}\n"
        f"MAE      = {metrics['MAE']:.0f} riders\n"
        f"RMSE     = {metrics['RMSE']:.0f} riders"
    )
    ax1.text(0.01, 0.97, ann, transform=ax1.transAxes, fontsize=8,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                       edgecolor="#d1d5db", alpha=0.92))
    ax1.set_title("PDR-STGCN — Test set: Actual vs Predicted (full period)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Test sample index"); ax1.set_ylabel("Ridership (riders)")
    ax1.legend(loc="upper right", fontsize=8); ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(idx_zoom, pred_min[idx_zoom], pred_max[idx_zoom],
                     alpha=0.18, color=C_BAND)
    ax2.plot(idx_zoom, actual_s1[idx_zoom], color=C_ACT, linewidth=1.7,
             label="Actual", zorder=5)
    steps_show = sorted({0, T_out // 2, T_out - 1})
    palette = [C_PRED, C_MID, C_LAST]; styles = ["--", "-.", ":"]
    for s, col, ls in zip(steps_show, palette, styles):
        ax2.plot(idx_zoom, y_pred[idx_zoom, s], color=col, linewidth=1.4,
                 linestyle=ls, alpha=0.88, label=f"Predicted step {s+1}")
    ax2.set_title(f"Zoomed: last {zoom_n} samples — step 1/{T_out//2+1}/{T_out}",
                  fontsize=10, fontweight="bold")
    ax2.set_xlabel("Test sample index"); ax2.set_ylabel("Ridership (riders)")
    ax2.legend(loc="upper right", fontsize=8, ncol=2); ax2.grid(alpha=0.25)

    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_step_metrics(per_step: list, out_path: str) -> None:
    steps    = [f"t+{i+1}" for i in range(len(per_step))]
    combined = [m["Combined"]  for m in per_step]
    mape     = [m["MAPE"]      for m in per_step]
    mae_pct  = [m["MAE_pct"]   for m in per_step]
    rmse_pct = [m["RMSE_pct"]  for m in per_step]
    r2       = [m["R2"]        for m in per_step]

    x, w = np.arange(len(steps)), 0.2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(steps)*0.85), 7),
                                   gridspec_kw={"hspace": 0.48})
    ax1.bar(x-1.5*w, combined, w, label="Combined%", color="#16a34a", alpha=0.85)
    ax1.bar(x-0.5*w, mape,     w, label="MAPE%",     color="#dc2626", alpha=0.85)
    ax1.bar(x+0.5*w, mae_pct,  w, label="MAE%",      color="#2563eb", alpha=0.85)
    ax1.bar(x+1.5*w, rmse_pct, w, label="RMSE%",     color="#f97316", alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(steps)
    ax1.set_ylabel("% of mean demand / score")
    ax1.set_title("PDR-STGCN — Per-horizon metrics", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    ax2.plot(steps, r2, marker="o", color="#f97316", linewidth=1.8, markersize=5)
    ax2.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax2.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax2.set_ylim(min(min(r2)-0.05, -0.1), 1.08)
    ax2.set_ylabel("R²"); ax2.set_title("PDR-STGCN — R² per horizon step", fontweight="bold")
    ax2.grid(alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.seq_dir is None:
        args.seq_dir = ("data/sequences/lstm" if args.lookback == 14
                        else f"data/sequences/lookback_{args.lookback}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        if   torch.cuda.is_available():         device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else:                                   device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # ── Data ──────────────────────────────────────────────────────────────────
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_splits(args.seq_dir, device)
    T_in       = X_tr.shape[1]
    n_features = X_tr.shape[2]
    T_out      = y_tr.shape[1]

    meta_path  = os.path.join(args.seq_dir, "split_dates.json")
    split_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=args.batch_size)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=args.batch_size)

    # ── Build adjacency ───────────────────────────────────────────────────────
    print(f"\nBuilding feature correlation adjacency (threshold={args.adj_threshold})...")
    A_raw   = build_feature_adj(X_tr, threshold=args.adj_threshold)
    n_edges = int((A_raw > 0).sum().item() // 2)
    density = float((A_raw > 0).float().mean().item())
    print(f"  Nodes: {n_features}  Edges: {n_edges}  Density: {density:.4f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PDRSTGCNForecaster(
        n_features = n_features,
        hidden     = args.hidden,
        n_blocks   = args.n_blocks,
        Kt         = args.kt,
        period     = args.period,
        d_k        = args.dk,
        T_in       = T_in,
        T_out      = T_out,
        A_raw      = A_raw,
        dropout    = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    T_after  = T_in - (args.kt - 1) * (2 * args.n_blocks + 1)
    print(f"\nModel           : PDRSTGCNForecaster")
    print(f"  Nodes (N)     : {n_features}   Period={args.period}   d_k={args.dk}")
    print(f"  ST blocks     : {args.n_blocks}   hidden={args.hidden}   Kt={args.kt}")
    print(f"  T_in→T_after  : {T_in}→{T_after}")
    print(f"  In            : (batch, {T_in}, {n_features})")
    print(f"  Out           : (batch, {T_out})")
    print(f"  Params        : {n_params:,}")

    # ── Optimiser / loss ──────────────────────────────────────────────────────
    if args.loss == "huber":
        criterion = nn.HuberLoss(delta=1.0)
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    best_epoch     = 0
    patience_count = 0
    train_losses, val_losses = [], []
    best_state = None

    print(f"\nTraining  (max {args.epochs} epochs, patience={args.patience})")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'LR':>10}")
    print("─" * 45)

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            for pg in optimiser.param_groups:
                pg["lr"] = args.lr * epoch / args.warmup_epochs
        tr_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
        va_loss = evaluate(model, val_loader, criterion)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        if epoch > args.warmup_epochs:
            scheduler.step(va_loss)
        lr_now = optimiser.param_groups[0]["lr"]
        print(f"{epoch:6d}  {tr_loss:10.6f}  {va_loss:10.6f}  {lr_now:10.2e}")

        if va_loss < best_val_loss:
            best_val_loss, best_epoch, patience_count = va_loss, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stop at epoch {epoch}  "
                      f"(best={best_val_loss:.6f} @ epoch {best_epoch})")
                break

    model.load_state_dict(best_state)

    # ── Collect predictions ───────────────────────────────────────────────────
    model.eval()
    preds_s, trues_s = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds_s.append(model(X_b).cpu().numpy())
            trues_s.append(y_b.cpu().numpy())

    y_pred_s = np.concatenate(preds_s)
    y_true_s = np.concatenate(trues_s)

    # ── Inverse-transform ─────────────────────────────────────────────────────
    scaler_y_path = os.path.join(args.seq_dir, "scaler_y.pkl")
    if os.path.exists(scaler_y_path):
        scaler_y = joblib.load(scaler_y_path)
        N2, T2   = y_pred_s.shape
        y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).reshape(N2,T2)
        y_true   = scaler_y.inverse_transform(y_true_s.reshape(-1,1)).reshape(N2,T2)
    else:
        print("[WARN] scaler_y.pkl not found — metrics in scaled units.")
        y_pred, y_true = y_pred_s, y_true_s

    # ── Per-horizon metrics ───────────────────────────────────────────────────
    W = 10
    print(f"\n{'='*85}")
    print("PDR-STGCN — TEST SET METRICS")
    print(f"{'='*85}")
    header = (f"{'Step':>5}  {'Combined%':>{W}}  {'MAPE%':>{W}}  {'MAE%':>{W}}  "
              f"{'RMSE%':>{W}}  {'R²':>{W}}  {'MAE':>{W}}  {'RMSE':>{W}}")
    print(header); print("─" * len(header))

    per_step = []
    for s in range(T_out):
        m = compute_metrics(y_true[:, s], y_pred[:, s])
        per_step.append(m)
        print(f"{s+1:5d}  "
              f"{m['Combined']:>{W}.2f}  {m['MAPE']:>{W}.2f}  {m['MAE_pct']:>{W}.2f}  "
              f"{m['RMSE_pct']:>{W}.2f}  {m['R2']:>{W}.4f}  "
              f"{m['MAE']:>{W}.0f}  {m['RMSE']:>{W}.0f}")

    overall = compute_metrics(y_true.flatten(), y_pred.flatten())
    print("─" * len(header))
    print(f"{'Avg':>5}  "
          f"{overall['Combined']:>{W}.2f}  {overall['MAPE']:>{W}.2f}  "
          f"{overall['MAE_pct']:>{W}.2f}  {overall['RMSE_pct']:>{W}.2f}  "
          f"{overall['R2']:>{W}.4f}  {overall['MAE']:>{W}.0f}  {overall['RMSE']:>{W}.0f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"src/outputs/pdr_stgcn/{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{out_dir}/model.pt")

    results = {
        "run_id": run_id,
        "model":  "PDRSTGCNForecaster",
        "hparams": {
            "hidden":         args.hidden,
            "kt":             args.kt,
            "n_blocks":       args.n_blocks,
            "period":         args.period,
            "dk":             args.dk,
            "adj_threshold":  args.adj_threshold,
            "dropout":        args.dropout,
            "weight_decay":   args.weight_decay,
            "T_in":           T_in,
            "T_out":          T_out,
            "n_features":     n_features,
            "n_graph_nodes":  n_features,
            "n_edges":        n_edges,
            "batch_size":     args.batch_size,
            "lr":             args.lr,
            "loss":           args.loss,
            "warmup_epochs":  args.warmup_epochs,
        },
        "training": {
            "best_epoch":    best_epoch,
            "best_val_loss": round(best_val_loss, 8),
            "total_epochs":  len(train_losses),
        },
        "split_dates": split_meta,
        "test_metrics": {
            "overall":  {k: round(v, 4) for k, v in overall.items()},
            "per_step": [{k: round(v, 4) for k, v in m.items()} for m in per_step],
        },
    }
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\nSaving plots...")
    plot_loss_curves(train_losses, val_losses, f"{out_dir}/loss_curves.png")
    plot_predictions(y_true, y_pred, overall,  f"{out_dir}/test_predictions.png")
    plot_per_step_metrics(per_step,            f"{out_dir}/per_step_metrics.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*85}")
    print(f"PDR-STGCN run complete  →  {out_dir}/")
    print(f"{'='*85}")
    print(f"  Combined  : {overall['Combined']:.2f}%")
    print(f"  MAPE      : {overall['MAPE']:.2f}%")
    print(f"  MAE%      : {overall['MAE_pct']:.2f}%    (raw MAE  = {overall['MAE']:.0f} riders)")
    print(f"  RMSE%     : {overall['RMSE_pct']:.2f}%   (raw RMSE = {overall['RMSE']:.0f} riders)")
    print(f"  R²        : {overall['R2']:.4f}")

    # ── Multi-way comparison ──────────────────────────────────────────────────
    PRIOR_MODELS = [
        ("LSTM",       args.lstm_results,       "src/outputs/lstm",       "#2563eb"),
        ("BiLSTM",     args.bilstm_results,     "src/outputs/bilstm",     "#7c3aed"),
        ("TPA-LSTM",   args.tpalstm_results,    "src/outputs/tpa_lstm",   "#0891b2"),
        ("CNN-LSTM",   args.cnnlstm_results,    "src/outputs/cnn_lstm",   "#16a34a"),
        ("CNN-BiLSTM", args.cnnbilstm_results,  "src/outputs/cnn_bilstm", "#d97706"),
        ("ST-LSTM",    args.stlstm_results,     "src/outputs/st_lstm",    "#dc2626"),
        ("STGCN",      args.stgcn_results,      "src/outputs/stgcn",      "#10b981"),
        ("MTGNN",      args.mtgnn_results,      "src/outputs/mtgnn",      "#f472b6"),
        ("STSGCN",     args.stsgcn_results,     "src/outputs/stsgcn",     "#0ea5e9"),
        ("STFGNN",     args.stfgnn_results,     "src/outputs/stfgnn",     "#a855f7"),
        ("ASTGCN",     args.astgcn_results,     "src/outputs/astgcn",     "#e11d48"),
        ("TFT",        args.tft_results,        "src/outputs/tft",        "#ca8a04"),
        ("Autoformer", args.autoformer_results, "src/outputs/autoformer", "#047857"),
        ("Informer",   args.informer_results,   "src/outputs/informer",   "#9333ea"),
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

    models_data.append(("PDR-STGCN", overall, per_step, "#f97316"))

    if len(models_data) > 1:
        print_comparison_table(models_data[:-1], overall, "PDR-STGCN")
        plot_comparison(models_data,
                        out_path=f"{out_dir}/comparison_{len(models_data)}_way.png")

    comparison["pdr_stgcn"] = {"run_id": run_id,
                                "overall": {k: round(v, 4) for k, v in overall.items()}}
    for name, m_overall, _, __ in models_data[:-1]:
        key = name.lower().replace("-", "_")
        comparison[f"delta_vs_{key}"] = {
            k: round(overall.get(k, 0) - m_overall.get(k, 0), 4) for k in overall
        }

    results["comparison"] = comparison
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Naive persistence baseline ────────────────────────────────────────────
    target_idx   = split_meta.get("target_col_idx", 0)
    last_obs_s   = X_te[:, -1, target_idx:target_idx+1].cpu().numpy()
    naive_pred_s = np.tile(last_obs_s, (1, T_out))
    if os.path.exists(scaler_y_path):
        N2b, T2b = naive_pred_s.shape
        naive_pred = scaler_y.inverse_transform(
            naive_pred_s.reshape(-1,1)).reshape(N2b, T2b)
    else:
        naive_pred = naive_pred_s
    naive  = compute_metrics(y_true.flatten(), naive_pred.flatten())
    d_comb = overall["Combined"] - naive["Combined"]
    d_mape = naive["MAPE"] - overall["MAPE"]
    print(f"\n  Naive persistence  "
          f"Combined={naive['Combined']:.2f}%  MAPE={naive['MAPE']:.2f}%  "
          f"R²={naive['R2']:.4f}")
    print(f"  PDR-STGCN vs naive "
          f"ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()
