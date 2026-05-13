"""
graphwavenet.py  — Graph WaveNet for Transit Ridership Forecasting

Implements the architecture from:
  Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Yu, P. S. (2019).
  "Graph WaveNet for Deep Spatial-Temporal Graph Modeling."
  IJCAI 2019.  arXiv:1906.00121

Adaptation for multivariate feature-node graph:
  The n_features input features are treated as N graph nodes. An adaptive
  adjacency matrix (self-adaptive graph) is learned via two trainable node
  embedding matrices E1, E2 ∈ R^{N×d_emb}:
    A_adapt = softmax(ReLU(E1 @ E2.T))
  No pre-computed graph is required — the spatial structure is learned
  end-to-end from data.

Architecture:
  X (B, T_in, N) → embed channels → stack of WaveNet blocks
  Each WaveNet block (dilation d):
    Gated dilated causal conv (temporal):  (B, N, C, T) → (B, N, C, T)
    Adaptive graph diffusion (spatial):    (B, N, C, T) → (B, N, C, T)
    Residual + skip connection
  Sum skip connections → ReLU → mean over T,N → MLP → (B, T_out)

Comparison:
  --lstm-results, --bilstm-results, --tpalstm-results,
  --cnnlstm-results, --cnnbilstm-results, --stlstm-results,
  --stgcn-results
  All optional; each auto-detects the most recent run if omitted.

Usage:
  python graphwavenet.py
  python graphwavenet.py --hidden 64 --n-layers 8 --d-emb 10
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
    p = argparse.ArgumentParser(description="Graph WaveNet forecaster with 8-way comparison")
    p.add_argument("--seq-dir",           default="data/sequences/lstm")
    p.add_argument("--hidden",            type=int,   default=64,
                   help="Residual and skip channel width")
    p.add_argument("--n-layers",          type=int,   default=8,
                   help="Number of WaveNet blocks")
    p.add_argument("--d-emb",             type=int,   default=10,
                   help="Node embedding dimension for adaptive adjacency")
    p.add_argument("--kernel-size",       type=int,   default=2,
                   help="Dilated temporal conv kernel size")
    p.add_argument("--dropout",           type=float, default=0.3)
    p.add_argument("--batch-size",        type=int,   default=16)
    p.add_argument("--epochs",            type=int,   default=50)
    p.add_argument("--lr",                type=float, default=2e-3)
    p.add_argument("--patience",          type=int,   default=10)
    p.add_argument("--device",            default="auto")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--lstm-results",      default=None)
    p.add_argument("--bilstm-results",    default=None)
    p.add_argument("--tpalstm-results",   default=None)
    p.add_argument("--cnnlstm-results",   default=None)
    p.add_argument("--cnnbilstm-results", default=None)
    p.add_argument("--stlstm-results",    default=None)
    p.add_argument("--stgcn-results",     default=None)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Model components
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveGraphConv(nn.Module):
    """
    Graph convolution with an adaptive self-learned adjacency.

    The adjacency A_adapt = softmax(ReLU(E1 @ E2.T)) is computed from
    two learnable node-embedding matrices and normalised row-wise.

    Input  : x (B, C, N, T),  A_adapt (N, N)
    Output : (B, C_out, N, T)
    """

    def __init__(self, in_channels: int, out_channels: int, support_len: int = 1):
        super().__init__()
        # One linear per support matrix (we use 1: adaptive only)
        self.linears = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for _ in range(support_len + 1)          # +1 for self-loop term
        ])

    def forward(self, x: torch.Tensor, A_adapt: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, C, N, T)
        A_adapt : (N, N)
        """
        # Self-loop term
        out = self.linears[0](x)                    # (B, C_out, N, T)

        # Adaptive diffusion term: A_adapt @ x (over N dim)
        # x: (B, C, N, T) → (B, C, T, N) → matmul with A^T → (B, C, T, N) → (B, C, N, T)
        x_perm = x.permute(0, 1, 3, 2)             # (B, C, T, N)
        ax     = torch.matmul(x_perm, A_adapt.t()).permute(0, 1, 3, 2)  # (B, C, N, T)
        out    = out + self.linears[1](ax)

        return F.relu(out)


class WaveNetBlock(nn.Module):
    """
    One WaveNet-style block:
      gated dilated causal conv (temporal) → adaptive graph conv (spatial)
      → residual → skip

    Input/Output (residual): (B, C_res, N, T)
    Skip output            : (B, C_skip, N, T)
    """

    def __init__(
        self,
        in_channels:   int,
        residual_ch:   int,
        skip_ch:       int,
        dilation:      int,
        kernel_size:   int = 2,
        dropout:       float = 0.3,
    ):
        super().__init__()
        self.dilation = dilation
        pad            = (kernel_size - 1) * dilation   # causal left-padding

        # Gated temporal conv: outputs 2*residual_ch → split into tanh + sigmoid
        self.tcn = nn.Conv2d(
            in_channels, residual_ch * 2,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            padding=(0, pad),
        )
        self.graph_conv = AdaptiveGraphConv(residual_ch, residual_ch)
        self.res_conv   = nn.Conv2d(residual_ch, in_channels, kernel_size=1)
        self.skip_conv  = nn.Conv2d(residual_ch, skip_ch, kernel_size=1)
        self.dropout    = nn.Dropout(dropout)
        self.bn         = nn.BatchNorm2d(residual_ch)

    def forward(
        self,
        x:       torch.Tensor,
        A_adapt: torch.Tensor,
    ):
        """
        x       : (B, C_res, N, T)
        A_adapt : (N, N)
        Returns : (residual, skip)  both (B, *, N, T)
        """
        residual = x
        T        = x.shape[-1]

        # ── Temporal gated conv ───────────────────────────────────────────────
        h = self.tcn(x)                              # (B, 2*C, N, T+pad) — causal left pad
        h = h[..., :T]                               # trim to original T
        h1, h2 = h.chunk(2, dim=1)
        h = torch.tanh(h1) * torch.sigmoid(h2)       # (B, C, N, T)
        h = self.dropout(h)

        # ── Adaptive graph conv ───────────────────────────────────────────────
        h = self.graph_conv(h, A_adapt)              # (B, C, N, T)
        h = self.bn(h)

        # ── Skip connection ───────────────────────────────────────────────────
        skip = self.skip_conv(h)                     # (B, C_skip, N, T)

        # ── Residual connection ───────────────────────────────────────────────
        h_res = self.res_conv(h)                     # (B, C_in, N, T)
        return h_res + residual, skip


class GraphWaveNetForecaster(nn.Module):
    """
    Graph WaveNet forecaster.

    Input  : (B, T_in, n_features)
    Output : (B, T_out)

    Adaptive adjacency is built from two learnable node embedding matrices
    E1, E2 ∈ R^{N×d_emb}: A_adapt = softmax(ReLU(E1 @ E2.T)).
    """

    def __init__(
        self,
        n_features: int,
        hidden:     int,
        skip_ch:    int,
        n_layers:   int,
        d_emb:      int,
        kernel_size: int,
        T_out:      int,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features

        # ── Adaptive adjacency embeddings (learnable parameters) ──────────────
        self.E1 = nn.Embedding(n_features, d_emb)
        self.E2 = nn.Embedding(n_features, d_emb)
        self.node_ids = None   # set after init; registered as buffer

        # ── Input projection ──────────────────────────────────────────────────
        # Treat T dimension as signal length, N as spatial
        # Input x: (B, T, N) → permute to (B, 1, N, T) → embed to hidden
        self.in_proj = nn.Conv2d(1, hidden, kernel_size=1)

        # ── WaveNet blocks with repeating dilations ───────────────────────────
        # Pattern: dilation = [1, 2, 4, 8, 1, 2, 4, 8, ...]
        dilations  = [2 ** (i % 8) for i in range(n_layers)]
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(WaveNetBlock(hidden, hidden, skip_ch, d, kernel_size, dropout))

        # ── Output MLP ────────────────────────────────────────────────────────
        self.out_relu1 = nn.ReLU()
        self.out_proj1 = nn.Conv2d(skip_ch, skip_ch, kernel_size=1)
        self.out_relu2 = nn.ReLU()
        self.out_proj2 = nn.Conv2d(skip_ch, skip_ch, kernel_size=1)

        self.head = nn.Sequential(
            nn.Linear(skip_ch, skip_ch // 2),
            nn.ReLU(),
            nn.Linear(skip_ch // 2, T_out),
        )

    def _adaptive_adj(self, device: torch.device) -> torch.Tensor:
        idx     = torch.arange(self.n_features, device=device)
        e1      = self.E1(idx)                      # (N, d_emb)
        e2      = self.E2(idx)                      # (N, d_emb)
        A_adapt = F.softmax(F.relu(e1 @ e2.t()), dim=-1)   # (N, N)
        return A_adapt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T_in, N)
        """
        B, T, N = x.shape
        device  = x.device

        # (B, 1, N, T)
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.in_proj(x)                         # (B, hidden, N, T)

        # Adaptive adjacency (computed once per forward, shared across all blocks)
        A_adapt = self._adaptive_adj(device)        # (N, N)

        # WaveNet blocks
        skip_sum = torch.zeros(B, self.out_proj1.in_channels, N, T,
                               device=device, dtype=x.dtype)
        for block in self.blocks:
            x, skip = block(x, A_adapt)
            skip_sum = skip_sum + skip

        # Aggregate
        h = self.out_relu1(skip_sum)
        h = self.out_proj1(h)
        h = self.out_relu2(h)
        h = self.out_proj2(h)                       # (B, skip_ch, N, T)

        # Mean pool over T and N
        h = h.mean(dim=-1).mean(dim=-1)             # (B, skip_ch)
        return self.head(h)                         # (B, T_out)



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
    for X_b, y_b in loader:
        optimiser.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total += loss.item() * X_b.size(0)
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

def plot_loss_curves(train_losses, val_losses, out_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train MSE", linewidth=1.5, color="#dc2626")
    ax.plot(val_losses,   label="Val MSE",   linewidth=1.5, color="#f97316", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (scaled)")
    ax.set_title("Graph WaveNet — Training curves")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_predictions(y_true, y_pred, metrics, out_path):
    N, T_out = y_true.shape
    idx_full = np.arange(N); zoom_n = min(60, N)
    idx_zoom = np.arange(N - zoom_n, N)
    actual_s1 = y_true[:, 0]; predicted_s1 = y_pred[:, 0]
    pred_min = y_pred.min(axis=1); pred_max = y_pred.max(axis=1)
    C_ACT = "#1d4ed8"; C_PRED = "#f472b6"; C_BAND = "#fce7f3"
    C_MID = "#0891b2"; C_LAST = "#7c3aed"
    fig = plt.figure(figsize=(14, 8)); gs = gridspec.GridSpec(2, 1, hspace=0.48)
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(idx_full, pred_min, pred_max, alpha=0.22, color=C_BAND,
                     label=f"Forecast spread (step 1–{T_out})")
    ax1.plot(idx_full, actual_s1,    color=C_ACT,  linewidth=1.5, label="Actual", zorder=4)
    ax1.plot(idx_full, predicted_s1, color=C_PRED, linewidth=1.2, linestyle="--",
             alpha=0.88, label="WaveNet predicted (step 1)", zorder=5)
    ann = (f"Combined = {metrics['Combined']:.2f}%\nMAPE     = {metrics['MAPE']:.2f}%\n"
           f"MAE%     = {metrics['MAE_pct']:.2f}%\nRMSE%    = {metrics['RMSE_pct']:.2f}%\n"
           f"R²       = {metrics['R2']:.4f}\nMAE      = {metrics['MAE']:.0f} riders\n"
           f"RMSE     = {metrics['RMSE']:.0f} riders")
    ax1.text(0.01, 0.97, ann, transform=ax1.transAxes, fontsize=8, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#d1d5db", alpha=0.92))
    ax1.set_title("Graph WaveNet — Test set: Actual vs Predicted (full period)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Test sample index"); ax1.set_ylabel("Ridership (riders)")
    ax1.legend(loc="upper right", fontsize=8); ax1.grid(alpha=0.25)
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(idx_zoom, pred_min[idx_zoom], pred_max[idx_zoom], alpha=0.18, color=C_BAND)
    ax2.plot(idx_zoom, actual_s1[idx_zoom], color=C_ACT, linewidth=1.7, label="Actual", zorder=5)
    steps_show = sorted({0, T_out//2, T_out-1}); palette=[C_PRED,C_MID,C_LAST]; styles=["--","-.",":" ]
    for s, col, ls in zip(steps_show, palette, styles):
        ax2.plot(idx_zoom, y_pred[idx_zoom, s], color=col, linewidth=1.4,
                 linestyle=ls, alpha=0.88, label=f"Predicted step {s+1}")
    ax2.set_title(f"Zoomed: last {zoom_n} samples", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Test sample index"); ax2.set_ylabel("Ridership (riders)")
    ax2.legend(loc="upper right", fontsize=8, ncol=2); ax2.grid(alpha=0.25)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_step_metrics(per_step, out_path):
    steps = [f"t+{i+1}" for i in range(len(per_step))]
    combined=[m["Combined"] for m in per_step]; mape=[m["MAPE"] for m in per_step]
    mae_pct=[m["MAE_pct"] for m in per_step]; rmse_pct=[m["RMSE_pct"] for m in per_step]
    r2=[m["R2"] for m in per_step]
    x, w = np.arange(len(steps)), 0.2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(steps)*0.85), 7),
                                   gridspec_kw={"hspace": 0.48})
    ax1.bar(x-1.5*w,combined,w,label="Combined%",color="#16a34a",alpha=0.85)
    ax1.bar(x-0.5*w,mape,    w,label="MAPE%",    color="#dc2626",alpha=0.85)
    ax1.bar(x+0.5*w,mae_pct, w,label="MAE%",     color="#2563eb",alpha=0.85)
    ax1.bar(x+1.5*w,rmse_pct,w,label="RMSE%",    color="#f97316",alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(steps)
    ax1.set_ylabel("% of mean demand / score")
    ax1.set_title("Graph WaveNet — Per-horizon metrics", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)
    ax2.plot(steps, r2, marker="o", color="#dc2626", linewidth=1.8, markersize=5)
    ax2.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax2.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax2.set_ylim(min(min(r2)-0.05, -0.1), 1.08)
    ax2.set_ylabel("R²"); ax2.set_title("Graph WaveNet — R² per horizon", fontweight="bold")
    ax2.grid(alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

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

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=args.batch_size)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=args.batch_size)

    skip_ch = args.hidden * 2
    model = GraphWaveNetForecaster(
        n_features  = n_features,
        hidden      = args.hidden,
        skip_ch     = skip_ch,
        n_layers    = args.n_layers,
        d_emb       = args.d_emb,
        kernel_size = args.kernel_size,
        T_out       = T_out,
        dropout     = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel         : GraphWaveNetForecaster")
    print(f"  Nodes (N)   : {n_features}   d_emb={args.d_emb}   n_layers={args.n_layers}")
    print(f"  hidden={args.hidden}   skip_ch={skip_ch}   kernel={args.kernel_size}")
    print(f"  In          : (batch, {T_in}, {n_features})")
    print(f"  Out         : (batch, {T_out})")
    print(f"  Params      : {n_params:,}")

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=5)

    best_val_loss=float("inf"); best_epoch=0; patience_count=0
    train_losses, val_losses = [], []; best_state=None

    print(f"\nTraining  (max {args.epochs} epochs, patience={args.patience})")
    print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>10}"); print("─"*45)

    for epoch in range(1, args.epochs+1):
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
                print(f"\nEarly stop at epoch {epoch}  (best={best_val_loss:.6f} @ epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    model.eval()
    preds_s, trues_s = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds_s.append(model(X_b).cpu().numpy()); trues_s.append(y_b.cpu().numpy())
    y_pred_s = np.concatenate(preds_s); y_true_s = np.concatenate(trues_s)

    scaler_y_path = os.path.join(args.seq_dir, "scaler_y.pkl")
    if os.path.exists(scaler_y_path):
        scaler_y = joblib.load(scaler_y_path)
        N2, T2 = y_pred_s.shape
        y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).reshape(N2,T2)
        y_true = scaler_y.inverse_transform(y_true_s.reshape(-1,1)).reshape(N2,T2)
    else:
        print("[WARN] scaler_y.pkl not found — metrics in scaled units."); y_pred, y_true = y_pred_s, y_true_s

    W=10; print(f"\n{'='*85}"); print("Graph WaveNet — TEST SET METRICS"); print(f"{'='*85}")
    header=(f"{'Step':>5}  {'Combined%':>{W}}  {'MAPE%':>{W}}  {'MAE%':>{W}}  "
            f"{'RMSE%':>{W}}  {'R²':>{W}}  {'MAE':>{W}}  {'RMSE':>{W}}")
    print(header); print("─"*len(header))
    per_step=[]
    for s in range(T_out):
        m=compute_metrics(y_true[:,s],y_pred[:,s]); per_step.append(m)
        print(f"{s+1:5d}  {m['Combined']:>{W}.2f}  {m['MAPE']:>{W}.2f}  {m['MAE_pct']:>{W}.2f}  "
              f"{m['RMSE_pct']:>{W}.2f}  {m['R2']:>{W}.4f}  {m['MAE']:>{W}.0f}  {m['RMSE']:>{W}.0f}")
    overall=compute_metrics(y_true.flatten(),y_pred.flatten())
    print("─"*len(header))
    print(f"{'Avg':>5}  {overall['Combined']:>{W}.2f}  {overall['MAPE']:>{W}.2f}  "
          f"{overall['MAE_pct']:>{W}.2f}  {overall['RMSE_pct']:>{W}.2f}  "
          f"{overall['R2']:>{W}.4f}  {overall['MAE']:>{W}.0f}  {overall['RMSE']:>{W}.0f}")

    run_id=datetime.now().strftime("%Y%m%d_%H%M%S"); out_dir=f"src/outputs/graph_wavenet/{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{out_dir}/model.pt")
    results={"run_id":run_id,"model":"GraphWaveNetForecaster",
             "hparams":{"hidden":args.hidden,"skip_ch":skip_ch,"n_layers":args.n_layers,
                        "d_emb":args.d_emb,"kernel_size":args.kernel_size,"dropout":args.dropout,
                        "T_in":T_in,"T_out":T_out,"n_features":n_features,"batch_size":args.batch_size,"lr":args.lr},
             "training":{"best_epoch":best_epoch,"best_val_loss":round(best_val_loss,8),"total_epochs":len(train_losses)},
             "split_dates":split_meta,
             "test_metrics":{"overall":{k:round(v,4) for k,v in overall.items()},
                             "per_step":[{k:round(v,4) for k,v in m.items()} for m in per_step]}}
    with open(f"{out_dir}/results.json","w") as f: json.dump(results,f,indent=2)

    print(f"\nSaving plots...")
    plot_loss_curves(train_losses, val_losses, f"{out_dir}/loss_curves.png")
    plot_predictions(y_true, y_pred, overall,  f"{out_dir}/test_predictions.png")
    plot_per_step_metrics(per_step,            f"{out_dir}/per_step_metrics.png")

    print(f"\n{'='*85}"); print(f"Graph WaveNet run complete  →  {out_dir}/"); print(f"{'='*85}")
    print(f"  Combined  : {overall['Combined']:.2f}%"); print(f"  MAPE      : {overall['MAPE']:.2f}%")
    print(f"  MAE%      : {overall['MAE_pct']:.2f}%    (raw MAE  = {overall['MAE']:.0f} riders)")
    print(f"  RMSE%     : {overall['RMSE_pct']:.2f}%   (raw RMSE = {overall['RMSE']:.0f} riders)")
    print(f"  R²        : {overall['R2']:.4f}")

    PRIOR_MODELS=[
        ("LSTM",       args.lstm_results,      "src/outputs/lstm",          "#2563eb"),
        ("BiLSTM",     args.bilstm_results,    "src/outputs/bilstm",        "#7c3aed"),
        ("TPA-LSTM",   args.tpalstm_results,   "src/outputs/tpa_lstm",      "#0891b2"),
        ("CNN-LSTM",   args.cnnlstm_results,   "src/outputs/cnn_lstm",      "#16a34a"),
        ("CNN-BiLSTM", args.cnnbilstm_results, "src/outputs/cnn_bilstm",    "#d97706"),
        ("ST-LSTM",    args.stlstm_results,    "src/outputs/st_lstm",       "#dc2626"),
        ("STGCN",      args.stgcn_results,     "src/outputs/stgcn",         "#10b981"),
    ]
    models_data=[]; comparison={}
    for name,path,model_dir,color in PRIOR_MODELS:
        key=name.lower().replace("-","_"); data=load_model_results(path,model_dir,name)
        if data:
            m_overall=data["test_metrics"]["overall"]; m_ps=data["test_metrics"]["per_step"]
            models_data.append((name,m_overall,m_ps,color))
            comparison[key]={"run_id":data.get("run_id"),"overall":m_overall}
        else: comparison[key]={"run_id":None,"overall":None}
    models_data.append(("WaveNet", overall, per_step, "#f472b6"))
    if len(models_data)>1:
        print_comparison_table(models_data[:-1],overall,"WaveNet")
        plot_comparison(models_data,out_path=f"{out_dir}/comparison_{len(models_data)}_way.png")
    comparison["wavenet"]={"run_id":run_id,"overall":{k:round(v,4) for k,v in overall.items()}}
    for name,m_overall,_,__ in models_data[:-1]:
        key=name.lower().replace("-","_")
        comparison[f"delta_vs_{key}"]={k:round(overall.get(k,0)-m_overall.get(k,0),4) for k in overall}
    results["comparison"]=comparison
    with open(f"{out_dir}/results.json","w") as f: json.dump(results,f,indent=2)

    target_idx=split_meta.get("target_col_idx",0)
    last_obs_s=X_te[:,-1,target_idx:target_idx+1].cpu().numpy()
    naive_pred_s=np.tile(last_obs_s,(1,T_out))
    if os.path.exists(scaler_y_path):
        N2b,T2b=naive_pred_s.shape
        naive_pred=scaler_y.inverse_transform(naive_pred_s.reshape(-1,1)).reshape(N2b,T2b)
    else: naive_pred=naive_pred_s
    naive=compute_metrics(y_true.flatten(),naive_pred.flatten())
    d_comb=overall["Combined"]-naive["Combined"]; d_mape=naive["MAPE"]-overall["MAPE"]
    print(f"\n  Naive persistence  Combined={naive['Combined']:.2f}%  MAPE={naive['MAPE']:.2f}%  R²={naive['R2']:.4f}")
    print(f"  WaveNet vs naive   ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()
