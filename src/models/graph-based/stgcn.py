"""
stgcn.py  — Spatio-Temporal Graph Convolutional Network (STGCN)

Implements the ST-Conv block from:
  Yu, B., Yin, H., & Zhu, Z. (2018). "Spatio-Temporal Graph Convolutional
  Networks: A Deep Learning Framework for Traffic Forecasting."
  IJCAI 2018.  arXiv:1709.04875

Adaptation for multivariate feature-node graph:
  The n_features input features are treated as N graph nodes. The temporal
  dimension (T_in=14) provides the scalar signal for each node. The adjacency
  matrix is derived from absolute Pearson correlation between features on
  training data, symmetrically normalised into a scaled Laplacian L_tilde
  stored as a model buffer (so it travels with .to(device)).

Architecture:
  X (B, T_in, N)  →  reshape  →  (B, N, 1, T_in)

  ST-Conv Block k:
    TemporalGatedConv   (B, N, C_in,  T)    →  (B, N, C_mid, T-Kt+1)
    ChebGraphConv       (B*T', N, C_mid)    →  (B*T', N, C_mid)
    TemporalGatedConv   (B, N, C_mid, T')   →  (B, N, C_out, T'-Kt+1)
    BatchNorm2d

  Output Layer:
    TemporalGatedConv → pool over N (mean) → flatten → MLP head → (B, T_out)

Comparison:
  --lstm-results, --bilstm-results, --tpalstm-results,
  --cnnlstm-results, --cnnbilstm-results, --stlstm-results
  All optional; each auto-detects the most recent run if omitted.

Usage:
  python stgcn.py
  python stgcn.py --hidden 64 --cheb-k 2 --n-blocks 2 --kt 3
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
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
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
    p = argparse.ArgumentParser(description="STGCN forecaster with 7-way comparison")
    p.add_argument("--seq-dir",           default="data/sequences/lstm",
                   help="Directory with X/y .npy splits")
    p.add_argument("--hidden",            type=int,   default=128,
                   help="Graph conv channel width (C_mid = C_out = hidden)")
    p.add_argument("--cheb-k",            type=int,   default=3,
                   help="Chebyshev polynomial order K")
    p.add_argument("--kt",                type=int,   default=3,
                   help="Temporal conv kernel size")
    p.add_argument("--n-blocks",          type=int,   default=2,
                   help="Number of ST-Conv blocks")
    p.add_argument("--adj-threshold",     type=float, default=0.1,
                   help="Min abs Pearson correlation to keep an edge")
    p.add_argument("--dropout",           type=float, default=0.15,
                   help="Dropout on graph conv output")
    p.add_argument("--weight-decay",      type=float, default=1e-4,
                   help="Adam weight decay")
    p.add_argument("--batch-size",        type=int,   default=32)
    p.add_argument("--epochs",            type=int,   default=200)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--patience",          type=int,   default=20)
    p.add_argument("--device",            default="auto", help="cpu | cuda | mps | auto")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--lstm-results",      default=None)
    p.add_argument("--bilstm-results",    default=None)
    p.add_argument("--tpalstm-results",   default=None)
    p.add_argument("--cnnlstm-results",   default=None)
    p.add_argument("--cnnbilstm-results", default=None)
    p.add_argument("--stlstm-results",    default=None)
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
    X_flat = X_np.reshape(-1, X_np.shape[-1])   # (N*T, F)
    corr   = np.corrcoef(X_flat.T).astype(np.float32)   # (F, F)
    A      = np.abs(corr)
    np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0, copy=False)  # guard: zero-variance cols
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 0.0)
    return torch.from_numpy(A)


def compute_scaled_laplacian(A: torch.Tensor) -> torch.Tensor:
    """
    Scaled Laplacian L_tilde = 2*L_sym/lambda_max - I.
    Approximates lambda_max = 2 → L_tilde = L_sym - I = -D^{-1/2} A D^{-1/2}.

    A : (N, N) raw adjacency (no self-loops, non-negative).
    Returns L_tilde : (N, N), eigenvalues in [-1, 1].
    """
    N           = A.shape[0]
    D           = A.sum(dim=1).clamp(min=1e-9)   # (N,)
    D_inv_sqrt  = D.pow(-0.5)                    # (N,)
    A_sym       = D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)
    L_tilde     = -A_sym                         # = L_sym - I (lambda_max ≈ 2 approx)
    return L_tilde


# ══════════════════════════════════════════════════════════════════════════════
# Model components
# ══════════════════════════════════════════════════════════════════════════════

class ChebConv(nn.Module):
    """
    K-order Chebyshev spectral graph convolution.

    Input  : x (B, N, C_in),  L_tilde (N, N) scaled Laplacian
    Output : (B, N, C_out)

    Chebyshev recursion:
      T_0 = x
      T_1 = L_tilde @ x
      T_k = 2 * L_tilde @ T_{k-1} - T_{k-2}
      out = sum_k T_k @ theta_k
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 2):
        super().__init__()
        self.K     = K
        self.theta = nn.ParameterList([
            nn.Parameter(torch.empty(in_channels, out_channels).uniform_(-0.05, 0.05))
            for _ in range(K)
        ])

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        T0  = x
        out = T0 @ self.theta[0]

        if self.K >= 2:
            T1  = torch.einsum("nm,bmc->bnc", L_tilde, x)
            out = out + T1 @ self.theta[1]

        for k in range(2, self.K):
            T2  = 2.0 * torch.einsum("nm,bmc->bnc", L_tilde, T1) - T0
            out = out + T2 @ self.theta[k]
            T0, T1 = T1, T2

        return out   # (B, N, C_out)


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
        h       = self.conv(x)
        h1, h2  = h.chunk(2, dim=1)
        return torch.tanh(h1) * torch.sigmoid(h2)


class STConvBlock(nn.Module):
    """
    One ST-Conv block: TemporalGated → ChebGraph → TemporalGated → BN.

    Each block reduces T by 2*(Kt - 1).
    Input/Output shape: (B, N, C_in, T) / (B, N, C_out, T - 2*(Kt-1))
    """

    def __init__(
        self,
        in_channels:  int,
        mid_channels: int,
        out_channels: int,
        K:            int = 2,
        Kt:           int = 3,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.temp1   = TemporalGatedConv(in_channels,  mid_channels, Kt)
        self.graph   = ChebConv(mid_channels, mid_channels, K)
        self.temp2   = TemporalGatedConv(mid_channels, out_channels, Kt)
        self.bn      = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, N, C_in, T)
        L_tilde : (N, N)
        Returns : (B, N, C_out, T - 2*(Kt-1))
        """
        B, N, C, T = x.shape

        # ── Temporal conv 1 ───────────────────────────────────────────────────
        xt = x.reshape(B * N, C, T)
        xt = self.temp1(xt)                          # (B*N, C_mid, T1)
        T1, C1 = xt.shape[-1], xt.shape[1]
        xt = xt.reshape(B, N, C1, T1)

        # ── Graph conv (applied to every timestep at once) ────────────────────
        xg = xt.permute(0, 3, 1, 2).reshape(B * T1, N, C1)   # (B*T1, N, C_mid)
        xg = self.graph(xg, L_tilde)
        xg = F.relu(xg)
        xg = self.dropout(xg)
        xg = xg.reshape(B, T1, N, C1).permute(0, 2, 3, 1)    # (B, N, C_mid, T1)

        # ── Temporal conv 2 ───────────────────────────────────────────────────
        xt2 = xg.reshape(B * N, C1, T1)
        xt2 = self.temp2(xt2)                        # (B*N, C_out, T2)
        T2, C2 = xt2.shape[-1], xt2.shape[1]
        xt2 = xt2.reshape(B, N, C2, T2)

        # ── BatchNorm (treat N as H, T2 as W) ────────────────────────────────
        xt2 = self.bn(xt2.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        return xt2


class STGCNForecaster(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network forecaster.

    The scaled Laplacian L_tilde is precomputed from training-data feature
    correlation and registered as a buffer, so forward(x) only takes x.

    Input  : (B, T_in, n_features)
    Output : (B, T_out)
    """

    def __init__(
        self,
        n_features: int,
        hidden:     int,
        n_blocks:   int,
        K:          int,
        Kt:         int,
        T_in:       int,
        T_out:      int,
        A_raw:      torch.Tensor,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features

        # ── Scaled Laplacian (buffer) ─────────────────────────────────────────
        L_tilde = compute_scaled_laplacian(A_raw)
        self.register_buffer("L_tilde", L_tilde)

        # ── ST-Conv blocks ────────────────────────────────────────────────────
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            c_in = 1 if i == 0 else hidden
            self.blocks.append(STConvBlock(c_in, hidden, hidden, K, Kt, dropout))

        # ── Output temporal conv ──────────────────────────────────────────────
        self.out_temp = TemporalGatedConv(hidden, hidden, Kt)

        # ── Compute flattened head input size ─────────────────────────────────
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
        """
        x : (B, T_in, N)   N = n_features
        """
        B, T, N = x.shape
        # (B, N, 1, T)
        x = x.permute(0, 2, 1).unsqueeze(2)

        # ST blocks
        for block in self.blocks:
            x = block(x, self.L_tilde)       # (B, N, C_out, T')

        # Output temporal conv
        B2, N2, C2, T2 = x.shape
        x = x.reshape(B2 * N2, C2, T2)
        x = self.out_temp(x)                 # (B*N, C_out, T'')
        T3 = x.shape[-1]
        x = x.reshape(B2, N2, C2, T3)

        # Pool over nodes (mean), flatten, MLP
        x = x.mean(dim=1)                    # (B, C_out, T'')
        x = x.reshape(B2, -1)               # (B, C_out * T'')
        return self.head(x)



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
    total = 0.0
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
    ax.plot(train_losses, label="Train MSE", linewidth=1.5, color="#dc2626")
    ax.plot(val_losses,   label="Val MSE",   linewidth=1.5, color="#f97316",
            linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (scaled)")
    ax.set_title("STGCN — Training curves")
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

    C_ACT  = "#1d4ed8"; C_PRED = "#10b981"; C_BAND = "#a7f3d0"
    C_MID  = "#0891b2"; C_LAST = "#7c3aed"

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.48)

    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(idx_full, pred_min, pred_max, alpha=0.22, color=C_BAND,
                     label=f"Forecast spread (step 1–{T_out})")
    ax1.plot(idx_full, actual_s1,    color=C_ACT,  linewidth=1.5, label="Actual", zorder=4)
    ax1.plot(idx_full, predicted_s1, color=C_PRED, linewidth=1.2, linestyle="--",
             alpha=0.88, label="STGCN predicted (step 1)", zorder=5)
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
    ax1.set_title("STGCN — Test set: Actual vs Predicted (full period)",
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
    ax1.set_title("STGCN — Per-horizon metrics", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    ax2.plot(steps, r2, marker="o", color="#dc2626", linewidth=1.8, markersize=5)
    ax2.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax2.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax2.set_ylim(min(min(r2)-0.05, -0.1), 1.08)
    ax2.set_ylabel("R²"); ax2.set_title("STGCN — R² per horizon step", fontweight="bold")
    ax2.grid(alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Comparison table
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
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
    A_raw = build_feature_adj(X_tr, threshold=args.adj_threshold)
    n_edges = int((A_raw > 0).sum().item() // 2)
    density = float((A_raw > 0).float().mean().item())
    print(f"  Nodes: {n_features}  Edges: {n_edges}  Density: {density:.4f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = STGCNForecaster(
        n_features = n_features,
        hidden     = args.hidden,
        n_blocks   = args.n_blocks,
        K          = args.cheb_k,
        Kt         = args.kt,
        T_in       = T_in,
        T_out      = T_out,
        A_raw      = A_raw,
        dropout    = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    T_after  = T_in - (args.kt - 1) * (2 * args.n_blocks + 1)
    print(f"\nModel         : STGCNForecaster")
    print(f"  Nodes (N)   : {n_features}   Cheb-K={args.cheb_k}   Kt={args.kt}")
    print(f"  ST blocks   : {args.n_blocks}   hidden={args.hidden}")
    print(f"  T_in→T_after: {T_in}→{T_after}")
    print(f"  In          : (batch, {T_in}, {n_features})")
    print(f"  Out         : (batch, {T_out})")
    print(f"  Params      : {n_params:,}")

    # ── Optimiser / loss ──────────────────────────────────────────────────────
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr,
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
    print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>10}")
    print("─" * 45)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
        va_loss = evaluate(model, val_loader, criterion)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
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
    print("STGCN — TEST SET METRICS")
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
    out_dir = f"src/outputs/stgcn/{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{out_dir}/model.pt")

    results = {
        "run_id": run_id,
        "model":  "STGCNForecaster",
        "hparams": {
            "hidden":         args.hidden,
            "cheb_k":         args.cheb_k,
            "kt":             args.kt,
            "n_blocks":       args.n_blocks,
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
    print(f"STGCN run complete  →  {out_dir}/")
    print(f"{'='*85}")
    print(f"  Combined  : {overall['Combined']:.2f}%")
    print(f"  MAPE      : {overall['MAPE']:.2f}%")
    print(f"  MAE%      : {overall['MAE_pct']:.2f}%    (raw MAE  = {overall['MAE']:.0f} riders)")
    print(f"  RMSE%     : {overall['RMSE_pct']:.2f}%   (raw RMSE = {overall['RMSE']:.0f} riders)")
    print(f"  R²        : {overall['R2']:.4f}")

    # ── Multi-way comparison ──────────────────────────────────────────────────
    PRIOR_MODELS = [
        ("LSTM",       args.lstm_results,      "src/outputs/lstm",       "#2563eb"),
        ("BiLSTM",     args.bilstm_results,    "src/outputs/bilstm",     "#7c3aed"),
        ("TPA-LSTM",   args.tpalstm_results,   "src/outputs/tpa_lstm",   "#0891b2"),
        ("CNN-LSTM",   args.cnnlstm_results,   "src/outputs/cnn_lstm",   "#16a34a"),
        ("CNN-BiLSTM", args.cnnbilstm_results, "src/outputs/cnn_bilstm", "#d97706"),
        ("ST-LSTM",    args.stlstm_results,    "src/outputs/st_lstm",    "#dc2626"),
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

    models_data.append(("STGCN", overall, per_step, "#10b981"))

    if len(models_data) > 1:
        print_comparison_table(models_data[:-1], overall, "STGCN")
        plot_comparison(models_data,
                        out_path=f"{out_dir}/comparison_{len(models_data)}_way.png")

    comparison["stgcn"] = {"run_id": run_id,
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
    print(f"  STGCN vs naive     "
          f"ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()
