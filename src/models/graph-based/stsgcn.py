"""
stsgcn.py  — Spatial-Temporal Synchronous Graph Convolutional Network (STSGCN)

Implements the architecture from:
  Song, C., Lin, Y., Guo, S., & Wan, H. (2020).
  "Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework
  for Spatial-Temporal Network Data Forecasting."  AAAI 2020.
  arXiv:1903.02495

Adaptation for multivariate feature-node graph:
  The n_features input features are treated as N graph nodes. STSGCN builds a
  Spatial-Temporal Synchronous Graph (STSG) that captures both spatial and
  temporal correlations in a single 3N×3N adjacency matrix for each sliding
  window of 3 consecutive timesteps:

    STSG = [[A_spa,  I,      0    ],
            [I,      A_spa,  I    ],
            [0,      I,      A_spa]]

  where A_spa is the N×N feature-correlation graph (absolute Pearson,
  threshold=0.1) and I is the N×N identity (temporal self-connections).

  The STSG Convolutional Layer (STSGCL) applies a K-order Chebyshev graph
  convolution on this 3N×3N graph for every sliding window of 3 timesteps
  and extracts the centre N nodes as output, producing a synchronous
  spatio-temporal representation at each timestep.

Architecture:
  Input: X (B, T_in, N)
  Input projection: → (B, T_in, N, hidden)

  STSGCL × n_layers:
    Unfold T into windows: (B, T-2, 3N, C_in)
    ChebConv on L_stsg (3N×3N) + GLU: → (B, T-2, 3N, C_out)
    Extract centre N nodes:           → (B, T-2, N,  C_out)

  Mean pool over N, flatten T, MLP head → (B, T_out)

Comparison:
  --lstm-results, --bilstm-results, --tpalstm-results,
  --cnnlstm-results, --cnnbilstm-results, --stlstm-results,
  --stgcn-results, --mtgnn-results
  All optional; each auto-detects the most recent run if omitted.

Usage:
  python stsgcn.py
  python stsgcn.py --hidden 64 --n-layers 2 --cheb-k 2
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
    p = argparse.ArgumentParser(description="STSGCN forecaster — 10-way comparison")
    p.add_argument("--seq-dir",           default="data/sequences/lstm")
    p.add_argument("--hidden",            type=int,   default=64,
                   help="Channel width in STSGCL layers")
    p.add_argument("--n-layers",          type=int,   default=2,
                   help="Number of STSGCL layers (each shrinks T by 2)")
    p.add_argument("--cheb-k",            type=int,   default=2,
                   help="Chebyshev polynomial order K for STSG conv")
    p.add_argument("--adj-threshold",     type=float, default=0.1,
                   help="Min abs Pearson correlation to keep a spatial edge")
    p.add_argument("--dropout",           type=float, default=0.1)
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--batch-size",        type=int,   default=32)
    p.add_argument("--epochs",            type=int,   default=200)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--patience",          type=int,   default=20)
    p.add_argument("--device",            default="auto")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--lstm-results",      default=None)
    p.add_argument("--bilstm-results",    default=None)
    p.add_argument("--tpalstm-results",   default=None)
    p.add_argument("--cnnlstm-results",   default=None)
    p.add_argument("--cnnbilstm-results", default=None)
    p.add_argument("--stlstm-results",    default=None)
    p.add_argument("--stgcn-results",     default=None)
    p.add_argument("--mtgnn-results",     default=None)
    p.add_argument("--stfgnn-results",    default=None)
    p.add_argument("--pdrstgcn-results",  default=None)
    p.add_argument("--astgcn-results",    default=None)
    p.add_argument("--tft-results",       default=None)
    p.add_argument("--autoformer-results",default=None)
    p.add_argument("--informer-results",  default=None)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Graph construction
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_adj(X_train: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """Absolute Pearson correlation adjacency between features. (N, N) float32."""
    X_np   = X_train.cpu().numpy()
    X_flat = X_np.reshape(-1, X_np.shape[-1])
    corr   = np.corrcoef(X_flat.T).astype(np.float32)
    A      = np.abs(corr)
    np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 0.0)
    return torch.from_numpy(A)


def compute_scaled_laplacian(A: torch.Tensor) -> torch.Tensor:
    """L_tilde = -D^{-½} A D^{-½}  (approximates scaled Laplacian in [-1,1])."""
    D          = A.sum(dim=1).clamp(min=1e-9)
    D_inv_sqrt = D.pow(-0.5)
    A_sym      = D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)
    return -A_sym


def build_stsg(A_spa: torch.Tensor) -> torch.Tensor:
    """
    Build the 3N×3N Spatial-Temporal Synchronous Graph adjacency.

      STSG = [[A_spa,  I,      zeros],
              [I,      A_spa,  I    ],
              [zeros,  I,      A_spa]]

    Temporal I blocks encode self-connections across consecutive time steps.
    A_spa: (N, N)  →  returns STSG_raw (3N, 3N).
    """
    N = A_spa.shape[0]
    I = torch.eye(N, dtype=A_spa.dtype)
    Z = torch.zeros(N, N, dtype=A_spa.dtype)
    row0 = torch.cat([A_spa, I,     Z    ], dim=1)
    row1 = torch.cat([I,     A_spa, I    ], dim=1)
    row2 = torch.cat([Z,     I,     A_spa], dim=1)
    return torch.cat([row0, row1, row2], dim=0)   # (3N, 3N)


# ══════════════════════════════════════════════════════════════════════════════
# Model components
# ══════════════════════════════════════════════════════════════════════════════

class ChebConv(nn.Module):
    """K-order Chebyshev spectral graph convolution. (B, N, C_in) → (B, N, C_out)."""

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
        return out


class STSGCL(nn.Module):
    """
    Spatial-Temporal Synchronous Graph Convolutional Layer.

    For each sliding window of 3 consecutive timesteps the layer:
      1. Stacks node features into (B*W, 3N, C) — layout [t-1 | t | t+1]
      2. Applies ChebConv on the 3N×3N STSG → GLU activation
      3. Extracts the centre N nodes (time t) as output

    Input : (B, T', N, C_in)   T' ≥ 3
    Output: (B, T'-2, N, C_out)
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        # GLU: produce 2×C_out then split
        self.cheb    = ChebConv(in_channels, out_channels * 2, K)
        self.ln      = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, L_stsg: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        assert T >= 3, f"STSGCL: T={T} < 3, cannot form 3-step windows"

        # ── Vectorised window extraction via unfold ───────────────────────────
        # unfold(dim=1, size=3, step=1): (B, T-2, N, C, 3)
        wins = x.unfold(1, 3, 1)                          # (B, W, N, C, 3)
        wins = wins.permute(0, 1, 4, 2, 3).contiguous()  # (B, W, 3, N, C)
        W    = wins.shape[1]
        # Layout within each window: [t-1 nodes | t nodes | t+1 nodes]
        wins = wins.reshape(B * W, 3 * N, C)              # (B*W, 3N, C)

        # ── Chebyshev graph conv on STSG + GLU ───────────────────────────────
        h       = self.cheb(wins, L_stsg)                 # (B*W, 3N, C_out*2)
        h1, h2  = h.chunk(2, dim=-1)
        h       = torch.tanh(h1) * torch.sigmoid(h2)      # (B*W, 3N, C_out)
        h       = self.dropout(h)

        # ── Extract centre N nodes (time t) ──────────────────────────────────
        C_out  = h.shape[-1]
        h_mid  = h[:, N : 2 * N, :]                       # (B*W, N, C_out)
        h_out  = self.ln(h_mid.reshape(B, W, N, C_out))   # (B, W, N, C_out)
        return h_out


class STSGCNForecaster(nn.Module):
    """
    STSGCN forecaster.

    Builds and stores the 3N×3N STSG scaled Laplacian as a buffer; forward(x)
    takes only the input sequence x.

    Input : (B, T_in, N)
    Output: (B, T_out)
    """

    def __init__(self, n_features: int, hidden: int, n_layers: int, K: int,
                 T_in: int, T_out: int, A_spa: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        T_final = T_in - 2 * n_layers
        assert T_final > 0, (
            f"T_final={T_final} ≤ 0 with n_layers={n_layers}, T_in={T_in}. "
            f"Reduce n_layers."
        )

        # STSG scaled Laplacian (3N × 3N) — registered buffer
        STSG_raw = build_stsg(A_spa)
        L_stsg   = compute_scaled_laplacian(STSG_raw)
        self.register_buffer("L_stsg", L_stsg)

        # Input projection: scalar signal → hidden channels
        self.input_proj = nn.Linear(1, hidden)

        # STSGCL stack
        self.layers = nn.ModuleList([
            STSGCL(hidden, hidden, K, dropout) for _ in range(n_layers)
        ])

        # Output head
        head_in = hidden * T_final
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, T_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T_in, N)"""
        B, T, N = x.shape
        h = self.input_proj(x.unsqueeze(-1))  # (B, T, N, hidden)
        for layer in self.layers:
            h = layer(h, self.L_stsg)         # (B, T-2, N, hidden) each pass
        h = h.mean(dim=2)                     # (B, T_final, hidden)
        h = h.reshape(B, -1)                  # (B, T_final * hidden)
        return self.head(h)


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
    total, n_skip = 0.0, 0
    for X_b, y_b in loader:
        optimiser.zero_grad()
        loss = criterion(model(X_b), y_b)
        if not torch.isfinite(loss):
            n_skip += 1; continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total += loss.item() * X_b.size(0)
    if n_skip:
        print(f"  [WARN] {n_skip} batch(es) skipped — non-finite loss")
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
    ax.plot(val_losses,   label="Val MSE",   linewidth=1.5, color="#f97316", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (scaled)")
    ax.set_title("STSGCN — Training curves")
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
    pred_min = y_pred.min(axis=1); pred_max = y_pred.max(axis=1)
    C_ACT = "#1d4ed8"; C_PRED = "#0ea5e9"; C_BAND = "#e0f2fe"
    C_MID = "#0891b2"; C_LAST = "#7c3aed"
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.48)
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(idx_full, pred_min, pred_max, alpha=0.22, color=C_BAND,
                     label=f"Forecast spread (step 1–{T_out})")
    ax1.plot(idx_full, actual_s1,    color=C_ACT,  linewidth=1.5, label="Actual", zorder=4)
    ax1.plot(idx_full, predicted_s1, color=C_PRED, linewidth=1.2, linestyle="--",
             alpha=0.88, label="STSGCN predicted (step 1)", zorder=5)
    ann = (f"Combined = {metrics['Combined']:.2f}%\nMAPE     = {metrics['MAPE']:.2f}%\n"
           f"MAE%     = {metrics['MAE_pct']:.2f}%\nRMSE%    = {metrics['RMSE_pct']:.2f}%\n"
           f"R²       = {metrics['R2']:.4f}\nMAE      = {metrics['MAE']:.0f} riders\n"
           f"RMSE     = {metrics['RMSE']:.0f} riders")
    ax1.text(0.01, 0.97, ann, transform=ax1.transAxes, fontsize=8,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                       edgecolor="#d1d5db", alpha=0.92))
    ax1.set_title("STSGCN — Test set: Actual vs Predicted (full period)",
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
    combined = [m["Combined"] for m in per_step]
    mape     = [m["MAPE"]     for m in per_step]
    mae_pct  = [m["MAE_pct"]  for m in per_step]
    rmse_pct = [m["RMSE_pct"] for m in per_step]
    r2       = [m["R2"]       for m in per_step]
    x, w = np.arange(len(steps)), 0.2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(steps)*0.85), 7),
                                   gridspec_kw={"hspace": 0.48})
    ax1.bar(x-1.5*w, combined, w, label="Combined%", color="#16a34a", alpha=0.85)
    ax1.bar(x-0.5*w, mape,     w, label="MAPE%",     color="#dc2626", alpha=0.85)
    ax1.bar(x+0.5*w, mae_pct,  w, label="MAE%",      color="#2563eb", alpha=0.85)
    ax1.bar(x+1.5*w, rmse_pct, w, label="RMSE%",     color="#f97316", alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(steps)
    ax1.set_ylabel("% of mean demand / score")
    ax1.set_title("STSGCN — Per-horizon metrics", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)
    ax2.plot(steps, r2, marker="o", color="#dc2626", linewidth=1.8, markersize=5)
    ax2.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax2.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax2.set_ylim(min(min(r2)-0.05, -0.1), 1.08)
    ax2.set_ylabel("R²"); ax2.set_title("STSGCN — R² per horizon step", fontweight="bold")
    ax2.grid(alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        if   torch.cuda.is_available():         device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else:                                   device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

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

    print(f"\nBuilding spatial feature-correlation adjacency "
          f"(threshold={args.adj_threshold})...")
    A_spa  = build_feature_adj(X_tr, threshold=args.adj_threshold)
    n_edges = int((A_spa > 0).sum().item() // 2)
    density = float((A_spa > 0).float().mean().item())
    print(f"  Nodes: {n_features}  Edges: {n_edges}  Density: {density:.4f}")

    T_final = T_in - 2 * args.n_layers
    model = STSGCNForecaster(
        n_features = n_features,
        hidden     = args.hidden,
        n_layers   = args.n_layers,
        K          = args.cheb_k,
        T_in       = T_in,
        T_out      = T_out,
        A_spa      = A_spa,
        dropout    = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel         : STSGCNForecaster")
    print(f"  Nodes (N)   : {n_features}   Cheb-K={args.cheb_k}   n_layers={args.n_layers}")
    print(f"  T_in→T_final: {T_in}→{T_final}   hidden={args.hidden}")
    print(f"  STSG size   : {3*n_features}×{3*n_features}")
    print(f"  In          : (batch, {T_in}, {n_features})")
    print(f"  Out         : (batch, {T_out})")
    print(f"  Params      : {n_params:,}")

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5)

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
        train_losses.append(tr_loss); val_losses.append(va_loss)
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
    model.eval()
    preds_s, trues_s = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds_s.append(model(X_b).cpu().numpy())
            trues_s.append(y_b.cpu().numpy())
    y_pred_s = np.concatenate(preds_s)
    y_true_s = np.concatenate(trues_s)

    scaler_y_path = os.path.join(args.seq_dir, "scaler_y.pkl")
    if os.path.exists(scaler_y_path):
        scaler_y = joblib.load(scaler_y_path)
        N2, T2   = y_pred_s.shape
        y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).reshape(N2,T2)
        y_true   = scaler_y.inverse_transform(y_true_s.reshape(-1,1)).reshape(N2,T2)
    else:
        print("[WARN] scaler_y.pkl not found — metrics in scaled units.")
        y_pred, y_true = y_pred_s, y_true_s

    W = 10
    print(f"\n{'='*85}")
    print("STSGCN — TEST SET METRICS")
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

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"src/outputs/stsgcn/{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{out_dir}/model.pt")

    results = {
        "run_id": run_id,
        "model":  "STSGCNForecaster",
        "hparams": {
            "hidden":        args.hidden,
            "n_layers":      args.n_layers,
            "cheb_k":        args.cheb_k,
            "adj_threshold": args.adj_threshold,
            "dropout":       args.dropout,
            "weight_decay":  args.weight_decay,
            "T_in":          T_in,
            "T_out":         T_out,
            "n_features":    n_features,
            "n_edges":       n_edges,
            "batch_size":    args.batch_size,
            "lr":            args.lr,
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

    print(f"\nSaving plots...")
    plot_loss_curves(train_losses, val_losses, f"{out_dir}/loss_curves.png")
    plot_predictions(y_true, y_pred, overall,  f"{out_dir}/test_predictions.png")
    plot_per_step_metrics(per_step,            f"{out_dir}/per_step_metrics.png")

    print(f"\n{'='*85}")
    print(f"STSGCN run complete  →  {out_dir}/")
    print(f"{'='*85}")
    print(f"  Combined  : {overall['Combined']:.2f}%")
    print(f"  MAPE      : {overall['MAPE']:.2f}%")
    print(f"  MAE%      : {overall['MAE_pct']:.2f}%    (raw MAE  = {overall['MAE']:.0f} riders)")
    print(f"  RMSE%     : {overall['RMSE_pct']:.2f}%   (raw RMSE = {overall['RMSE']:.0f} riders)")
    print(f"  R²        : {overall['R2']:.4f}")

    PRIOR_MODELS = [
        ("LSTM",       args.lstm_results,       "src/outputs/lstm",      "#2563eb"),
        ("BiLSTM",     args.bilstm_results,     "src/outputs/bilstm",    "#7c3aed"),
        ("TPA-LSTM",   args.tpalstm_results,    "src/outputs/tpa_lstm",  "#0891b2"),
        ("CNN-LSTM",   args.cnnlstm_results,    "src/outputs/cnn_lstm",  "#16a34a"),
        ("CNN-BiLSTM", args.cnnbilstm_results,  "src/outputs/cnn_bilstm","#d97706"),
        ("ST-LSTM",    args.stlstm_results,     "src/outputs/st_lstm",   "#dc2626"),
        ("STGCN",      args.stgcn_results,      "src/outputs/stgcn",     "#10b981"),
        ("MTGNN",      args.mtgnn_results,      "src/outputs/mtgnn",     "#f472b6"),
        ("STFGNN",     args.stfgnn_results,     "src/outputs/stfgnn",    "#a855f7"),
        ("PDR-STGCN",   args.pdrstgcn_results,    "src/outputs/pdr_stgcn",  "#f97316"),
        ("ASTGCN",     args.astgcn_results,     "src/outputs/astgcn",    "#e11d48"),
        ("TFT",        args.tft_results,        "src/outputs/tft",       "#ca8a04"),
        ("Autoformer", args.autoformer_results, "src/outputs/autoformer","#047857"),
        ("Informer",   args.informer_results,   "src/outputs/informer",  "#9333ea"),
    ]

    models_data = []; comparison = {}
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

    models_data.append(("STSGCN", overall, per_step, "#0ea5e9"))

    if len(models_data) > 1:
        print_comparison_table(models_data[:-1], overall, "STSGCN")
        plot_comparison(models_data,
                        out_path=f"{out_dir}/comparison_{len(models_data)}_way.png")

    comparison["stsgcn"] = {"run_id": run_id,
                             "overall": {k: round(v, 4) for k, v in overall.items()}}
    for name, m_overall, _, __ in models_data[:-1]:
        key = name.lower().replace("-", "_")
        comparison[f"delta_vs_{key}"] = {
            k: round(overall.get(k, 0) - m_overall.get(k, 0), 4) for k in overall
        }
    results["comparison"] = comparison
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

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
    print(f"  STSGCN vs naive    "
          f"ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()
