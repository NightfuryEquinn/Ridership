"""
astgcn.py  — Attention-Based Spatial-Temporal Graph Convolutional Network (ASTGCN) — Fine-Tuned

Key Features:
  • Spatial attention over N feature nodes and temporal attention over T timesteps at each block
  • Chebyshev GCN (K-hop diffusion) applied after spatial attention for graph-aware propagation
  • n_blocks stacked for progressively deeper spatial-temporal interaction

Tuned Hyperparameters:
  d_model  : 64   → 128   low Combined% → more embedding capacity
  n_heads  : 4    → 8     proportional to d_model; richer attention subspaces
  n_blocks : 2    → 3     deeper stacking for more ST interaction
  dropout  : 0.10 → 0.20  moderate variance across lookbacks → regularisation

Architecture (one block):
  X (B, T, N, d)
  ──► Spatial Attention → ChebGCN → Temporal Attention → Position-wise FFN
  Stack n_blocks, then:
  ──► Mean pool over N → (B, T, d) → Flatten + MLP → (B, T_out)

Hardware:
  GPU  : NVIDIA A100 (32 GB VRAM)
  RAM  : 32 GB
  Precision : AMP fp16 (GradScaler enabled)

References
----------
Cui, Z., Zhang, J., Noh, G., & Park, H. J. (2023). ADSTGCN: A dynamic adaptive
deeper spatio-temporal graph convolutional network for multi-step traffic forecasting.
Sensors, 23(15), 6950.
DOI: https://doi.org/10.3390/s23156950
"""

import os
import json
import argparse
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
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
    p = argparse.ArgumentParser(description="ASTGCN (tuned) forecaster")
    p.add_argument("--seq-dir",              default=None)
    p.add_argument("--d-model",              type=int,   default=128,
                   help="Node embedding dimension  [tuned: 128, base: 64]")
    p.add_argument("--n-blocks",             type=int,   default=3,
                   help="Stacked ASTGCN blocks  [tuned: 3, base: 2]")
    p.add_argument("--K",                    type=int,   default=3,
                   help="Chebyshev polynomial order")
    p.add_argument("--n-heads",              type=int,   default=8,
                   help="Attention heads  [tuned: 8, base: 4]")
    p.add_argument("--adj-threshold",        type=float, default=0.1,
                   help="Min abs Pearson correlation to keep graph edge")
    p.add_argument("--dropout",              type=float, default=0.20,
                   help="Dropout  [tuned: 0.20, base: 0.10]")
    p.add_argument("--batch-size",           type=int,   default=32)
    p.add_argument("--epochs",               type=int,   default=150)
    p.add_argument("--lr",                   type=float, default=1e-3)
    p.add_argument("--weight-decay",         type=float, default=1e-4)
    p.add_argument("--patience",             type=int,   default=15)
    p.add_argument("--device",               default="auto")
    p.add_argument("--seed",                 type=int,   default=42)
    p.add_argument("--lstm-results",         default=None)
    p.add_argument("--bilstm-results",       default=None)
    p.add_argument("--tpalstm-results",      default=None)
    p.add_argument("--cnnlstm-results",      default=None)
    p.add_argument("--cnnbilstm-results",    default=None)
    p.add_argument("--stlstm-results",       default=None)
    p.add_argument("--stgcn-results",        default=None)
    p.add_argument("--mtgnn-results",        default=None)
    p.add_argument("--stsgcn-results",       default=None)
    p.add_argument("--stfgnn-results",       default=None)
    p.add_argument("--pdrstgcn-results",     default=None)
    p.add_argument("--tft-results",          default=None)
    p.add_argument("--autoformer-results",   default=None)
    p.add_argument("--informer-results",     default=None)
    p.add_argument("--lookback",      type=int,   default=14, choices=[14, 28, 56],
                   help="Look-back window; auto-selects seq-dir when --seq-dir is not set")
    p.add_argument("--loss",          default="huber", choices=["mse", "huber", "mae"],
                   help="Training loss: mse | huber (default) | mae")
    p.add_argument("--warmup-epochs", type=int,   default=5,
                   help="Linear LR warm-up epochs before ReduceLROnPlateau kicks in")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Graph construction
# ══════════════════════════════════════════════════════════════════════════════

def build_adj_and_laplacian(X_train: torch.Tensor, threshold: float = 0.1):
    """
    Build:
      A_norm  — symmetrically normalised adjacency (with self-loops)
      L_tilde — scaled graph Laplacian for Chebyshev expansion, ∈ [-1, 1]
    """
    X_np   = X_train.cpu().numpy()
    X_flat = X_np.reshape(-1, X_np.shape[-1])
    corr   = np.corrcoef(X_flat.T).astype(np.float32)
    corr   = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    A      = np.abs(corr)
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 0.0)

    A_hat    = A + np.eye(A.shape[0], dtype=np.float32)
    D_hat    = A_hat.sum(axis=1)
    D_inv_sq = 1.0 / np.sqrt(D_hat + 1e-9)
    A_norm   = D_inv_sq[:, None] * A_hat * D_inv_sq[None, :]

    D        = A.sum(axis=1)
    D_inv_sq2 = 1.0 / np.sqrt(D + 1e-9)
    A_sym    = D_inv_sq2[:, None] * A * D_inv_sq2[None, :]
    L_norm   = np.eye(A.shape[0], dtype=np.float32) - A_sym
    L_tilde  = L_norm - np.eye(L_norm.shape[0], dtype=np.float32)

    return torch.from_numpy(A_norm), torch.from_numpy(L_tilde)


# ══════════════════════════════════════════════════════════════════════════════
# Model components
# ══════════════════════════════════════════════════════════════════════════════

class ChebConv(nn.Module):
    """Chebyshev graph convolution."""

    def __init__(self, in_ch: int, out_ch: int, K: int):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(in_ch * K, out_ch)

    def forward(self, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        polys = [X]
        if self.K > 1:
            polys.append(torch.einsum("nm,bmc->bnc", L, X))
        for _ in range(2, self.K):
            polys.append(
                2.0 * torch.einsum("nm,bmc->bnc", L, polys[-1]) - polys[-2]
            )
        return self.linear(torch.cat(polys, dim=-1))


class ASTGCNBlock(nn.Module):
    """
    One ASTGCN attention block.

    Processing pipeline (X: B×T×N×d):
      1. Spatial Attention  — nodes attend to each other at every timestep
      2. ChebGCN            — K-hop graph diffusion adds structural bias
      3. Temporal Attention — time steps attend to each other per node
      4. Position-wise FFN
    """

    def __init__(self, d_model: int, K: int, n_heads: int, dropout: float):
        super().__init__()
        self.s_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.s_norm  = nn.LayerNorm(d_model)
        self.cheb     = ChebConv(d_model, d_model, K)
        self.cheb_act = nn.GELU()
        self.cheb_norm= nn.LayerNorm(d_model)
        self.t_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.t_norm  = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        B, T, N, D = X.shape

        Xs        = X.reshape(B * T, N, D)
        sa_out, _ = self.s_attn(Xs, Xs, Xs)
        Xs        = self.s_norm(Xs + self.drop(sa_out))

        gcn_out = self.cheb_act(self.cheb(Xs, L))
        Xs      = self.cheb_norm(Xs + self.drop(gcn_out))
        X       = Xs.reshape(B, T, N, D)

        Xt        = X.permute(0, 2, 1, 3).reshape(B * N, T, D)
        ta_out, _ = self.t_attn(Xt, Xt, Xt)
        Xt        = self.t_norm(Xt + self.drop(ta_out))

        Xt = self.ffn_norm(Xt + self.drop(self.ffn(Xt)))
        X  = Xt.reshape(B, N, T, D).permute(0, 2, 1, 3)
        return X


class ASTGCNForecaster(nn.Module):
    """
    ASTGCN forecaster.

    Input:  (B, T_in, N)   where N = n_features treated as graph nodes
    Output: (B, T_out)
    """

    def __init__(
        self,
        n_features: int,
        T_in:       int,
        T_out:      int,
        A_norm:     torch.Tensor,
        L_tilde:    torch.Tensor,
        d_model:    int   = 128,
        n_blocks:   int   = 3,
        K:          int   = 3,
        n_heads:    int   = 8,
        dropout:    float = 0.20,
    ):
        super().__init__()
        self.register_buffer("A", A_norm)
        self.register_buffer("L", L_tilde)

        self.input_proj = nn.Linear(1, d_model)
        self.t_embed    = nn.Embedding(T_in, d_model)

        self.blocks = nn.ModuleList([
            ASTGCNBlock(d_model, K, n_heads, dropout)
            for _ in range(n_blocks)
        ])

        self.head = nn.Sequential(
            nn.Linear(T_in * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, T_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape
        X = self.input_proj(x.unsqueeze(-1))              # (B, T, N, d_model)
        t_idx = torch.arange(T, device=x.device)
        t_pos = self.t_embed(t_idx)
        X = X + t_pos.unsqueeze(0).unsqueeze(2)
        for block in self.blocks:
            X = block(X, self.L)
        X   = X.mean(dim=2)                               # (B, T, d_model)
        X   = X.reshape(B, -1)                            # (B, T*d_model)
        return self.head(X)


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
# Training helpers  (AMP-aware)
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimiser, criterion, scaler, use_amp) -> float:
    model.train(); total = 0.0
    for X_b, y_b in loader:
        optimiser.zero_grad()
        with autocast(enabled=use_amp):
            loss = criterion(model(X_b), y_b)
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()
        total += loss.item() * X_b.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, use_amp) -> float:
    model.eval(); total = 0.0
    for X_b, y_b in loader:
        with autocast(enabled=use_amp):
            total += criterion(model(X_b), y_b).item() * X_b.size(0)
    return total / len(loader.dataset)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(train_losses, val_losses, out_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train Loss", linewidth=1.5, color="#e11d48")
    ax.plot(val_losses,   label="Val Loss",  linewidth=1.5, color="#f97316", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (scaled)")
    ax.set_title("ASTGCN (tuned) — Training curves")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig); print(f"  Saved: {out_path}")


def plot_predictions(y_true, y_pred, metrics, out_path):
    N, T_out = y_true.shape; idx_full = np.arange(N)
    zoom_n = min(60, N); idx_zoom = np.arange(N - zoom_n, N)
    actual_s1 = y_true[:, 0]; predicted_s1 = y_pred[:, 0]
    pred_min = y_pred.min(axis=1); pred_max = y_pred.max(axis=1)
    C_ACT = "#1d4ed8"; C_PRED = "#e11d48"; C_BAND = "#fecdd3"; C_MID = "#0891b2"; C_LAST = "#7c3aed"
    fig = plt.figure(figsize=(14, 8)); gs = gridspec.GridSpec(2, 1, hspace=0.48)
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(idx_full, pred_min, pred_max, alpha=0.22, color=C_BAND,
                     label=f"Forecast spread (step 1–{T_out})")
    ax1.plot(idx_full, actual_s1,    color=C_ACT, linewidth=1.5, label="Actual", zorder=4)
    ax1.plot(idx_full, predicted_s1, color=C_PRED, linewidth=1.2, linestyle="--", alpha=0.88,
             label="ASTGCN (tuned) predicted (step 1)", zorder=5)
    ann = (f"Combined = {metrics['Combined']:.2f}%\nMAPE     = {metrics['MAPE']:.2f}%\n"
           f"MAE%     = {metrics['MAE_pct']:.2f}%\nRMSE%    = {metrics['RMSE_pct']:.2f}%\n"
           f"R²       = {metrics['R2']:.4f}\nMAE      = {metrics['MAE']:.0f} riders\n"
           f"RMSE     = {metrics['RMSE']:.0f} riders")
    ax1.text(0.01, 0.97, ann, transform=ax1.transAxes, fontsize=8, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#d1d5db", alpha=0.92))
    ax1.set_title("ASTGCN (tuned) — Test set: Actual vs Predicted (full period)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Test sample index"); ax1.set_ylabel("Ridership (riders)")
    ax1.legend(loc="upper right", fontsize=8); ax1.grid(alpha=0.25)
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(idx_zoom, pred_min[idx_zoom], pred_max[idx_zoom], alpha=0.18, color=C_BAND)
    ax2.plot(idx_zoom, actual_s1[idx_zoom], color=C_ACT, linewidth=1.7, label="Actual", zorder=5)
    steps_show = sorted({0, T_out // 2, T_out - 1})
    palette = [C_PRED, C_MID, C_LAST]; styles = ["--", "-.", ":"]
    for s, col, ls in zip(steps_show, palette, styles):
        ax2.plot(idx_zoom, y_pred[idx_zoom, s], color=col, linewidth=1.4,
                 linestyle=ls, alpha=0.88, label=f"Predicted step {s + 1}")
    ax2.set_title(f"Zoomed: last {zoom_n} samples", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Test sample index"); ax2.set_ylabel("Ridership (riders)")
    ax2.legend(loc="upper right", fontsize=8, ncol=2); ax2.grid(alpha=0.25)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  Saved: {out_path}")


def plot_per_step_metrics(per_step, out_path):
    steps = [f"t+{i+1}" for i in range(len(per_step))]
    combined = [m["Combined"] for m in per_step]; mape = [m["MAPE"] for m in per_step]
    mae_pct = [m["MAE_pct"] for m in per_step]; rmse_pct = [m["RMSE_pct"] for m in per_step]
    r2 = [m["R2"] for m in per_step]
    x, w = np.arange(len(steps)), 0.2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(steps) * 0.85), 7),
                                    gridspec_kw={"hspace": 0.48})
    ax1.bar(x - 1.5*w, combined, w, label="Combined%", color="#16a34a", alpha=0.85)
    ax1.bar(x - 0.5*w, mape,     w, label="MAPE%",     color="#dc2626", alpha=0.85)
    ax1.bar(x + 0.5*w, mae_pct,  w, label="MAE%",      color="#2563eb", alpha=0.85)
    ax1.bar(x + 1.5*w, rmse_pct, w, label="RMSE%",     color="#f97316", alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(steps); ax1.set_ylabel("% of mean demand / score")
    ax1.set_title("ASTGCN (tuned) — Per-horizon metrics", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)
    ax2.plot(steps, r2, marker="o", color="#e11d48", linewidth=1.8, markersize=5)
    ax2.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax2.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax2.set_ylim(min(min(r2) - 0.05, -0.1), 1.08)
    ax2.set_ylabel("R²"); ax2.set_title("ASTGCN (tuned) — R² per horizon", fontweight="bold")
    ax2.grid(alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Fit Diagnostics
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
    """Return a fit-quality verdict and supporting statistics."""
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
    val_drift     = (final_val - best_val) / (best_val + 1e-12)
    val_drift_pct = val_drift * 100.0
    gap_ratio = best_val / (min_train + 1e-12)
    tail_n    = min(10, total)
    tr_tail   = train_losses[-tail_n:]
    xs        = np.arange(tail_n, dtype=float)
    tr_slope  = float(np.polyfit(xs, tr_tail, 1)[0])
    norm_tr   = tr_slope / (float(np.mean(tr_tail)) + 1e-12)
    train_still_falling = (norm_tr < -0.01)
    early_stop = best_epoch < early_stop_frac * total
    pre_window = val_losses[max(0, best_epoch - patience): best_epoch]
    if len(pre_window) >= 3:
        xs_p = np.arange(len(pre_window), dtype=float)
        s    = float(np.polyfit(xs_p, pre_window, 1)[0])
        ns   = s / (float(np.mean(pre_window)) + 1e-12)
        val_trend = "rising" if ns > 0.005 else ("falling" if ns < -0.005 else "flat")
    else:
        val_trend = "flat"
    notes: list = []
    verdict = "good_fit"
    if val_drift > val_drift_threshold:
        verdict = "overfit"
        notes.append(f"Val loss drifted +{val_drift_pct:.1f}% above its best ...")
    else:
        notes.append(f"Val drift +{val_drift_pct:.1f}% above best — within normal ...")
    if gap_ratio > gap_overfit_threshold:
        if verdict != "overfit": verdict = "overfit"
        notes.append(f"Val/train gap {gap_ratio:.2f}× exceeds ...")
    else:
        notes.append(f"Val/train gap {gap_ratio:.2f}× is within the {gap_overfit_threshold:.0f}× threshold ...")
    if train_still_falling and val_drift > 0.10:
        if verdict != "overfit": verdict = "overfit"
        notes.append(f"Training loss still declining in final {tail_n} epochs ...")
    if early_stop:
        frac_pct = int(100 * best_epoch / total)
        notes.append(f"Best epoch {best_epoch}/{total} ({frac_pct}%) is early ...")
        if verdict == "good_fit": verdict = "uncertain"
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


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.seq_dir is None:
        args.seq_dir = ("data/sequences/lstm" if args.lookback == 14
                        else f"data/sequences/lookback_{args.lookback}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.device == "auto":
        if   torch.cuda.is_available():         device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else:                                   device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    use_amp = (device.type == "cuda")
    scaler  = GradScaler(enabled=use_amp)
    print(f"Device: {device}   AMP: {'enabled (fp16)' if use_amp else 'disabled'}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_splits(args.seq_dir, device)
    T_in = X_tr.shape[1]; n_features = X_tr.shape[2]; T_out = y_tr.shape[1]
    meta_path  = os.path.join(args.seq_dir, "split_dates.json")
    split_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=args.batch_size)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=args.batch_size)

    print(f"\nBuilding feature graph (threshold={args.adj_threshold})...")
    A_norm, L_tilde = build_adj_and_laplacian(X_tr, threshold=args.adj_threshold)
    n_edges = int(((A_norm > 0).float().sum().item() - n_features) // 2)
    print(f"  Nodes: {n_features}   Approx edges (excl. self-loops): {n_edges}")

    model = ASTGCNForecaster(
        n_features = n_features,
        T_in       = T_in,
        T_out      = T_out,
        A_norm     = A_norm,
        L_tilde    = L_tilde,
        d_model    = args.d_model,
        n_blocks   = args.n_blocks,
        K          = args.K,
        n_heads    = args.n_heads,
        dropout    = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel     : ASTGCNForecaster (tuned)")
    print(f"  d_model : {args.d_model}   n_blocks: {args.n_blocks}   K: {args.K}   n_heads: {args.n_heads}")
    print(f"  In      : (batch, {T_in}, {n_features})")
    print(f"  Out     : (batch, {T_out})")
    print(f"  Params  : {n_params:,}")

    if args.loss == "huber":
        criterion = nn.HuberLoss(delta=1.0)
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf"); best_epoch = 0; patience_count = 0
    train_losses, val_losses = [], []; best_state = None
    print(f"\nTraining  (max {args.epochs} epochs, patience={args.patience}, warmup={args.warmup_epochs})")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'LR':>10}"); print("─" * 45)

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * epoch / args.warmup_epochs
            for pg in optimiser.param_groups:
                pg["lr"] = warmup_lr
        tr_loss = train_one_epoch(model, train_loader, optimiser, criterion, scaler, use_amp)
        va_loss = evaluate(model, val_loader, criterion, use_amp)
        train_losses.append(tr_loss); val_losses.append(va_loss)
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
                print(f"\nEarly stop @ epoch {epoch}  (best={best_val_loss:.6f} @ epoch {best_epoch})")
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

    model.eval(); preds_s, trues_s = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds_s.append(model(X_b).cpu().numpy())
            trues_s.append(y_b.cpu().numpy())
    y_pred_s = np.concatenate(preds_s); y_true_s = np.concatenate(trues_s)

    scaler_y_path = os.path.join(args.seq_dir, "scaler_y.pkl")
    if os.path.exists(scaler_y_path):
        scaler_y = joblib.load(scaler_y_path); N2, T2 = y_pred_s.shape
        y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).reshape(N2, T2)
        y_true = scaler_y.inverse_transform(y_true_s.reshape(-1, 1)).reshape(N2, T2)
    else:
        print("[WARN] scaler_y.pkl not found — metrics in scaled units.")
        y_pred, y_true = y_pred_s, y_true_s

    W = 10; print(f"\n{'='*85}"); print("ASTGCN (tuned) — TEST SET METRICS"); print(f"{'='*85}")
    header = (f"{'Step':>5}  {'Combined%':>{W}}  {'MAPE%':>{W}}  {'MAE%':>{W}}  "
              f"{'RMSE%':>{W}}  {'R²':>{W}}  {'MAE':>{W}}  {'RMSE':>{W}}")
    print(header); print("─" * len(header))
    per_step = []
    for s in range(T_out):
        m = compute_metrics(y_true[:, s], y_pred[:, s]); per_step.append(m)
        print(f"{s+1:5d}  {m['Combined']:>{W}.2f}  {m['MAPE']:>{W}.2f}  {m['MAE_pct']:>{W}.2f}  "
              f"{m['RMSE_pct']:>{W}.2f}  {m['R2']:>{W}.4f}  {m['MAE']:>{W}.0f}  {m['RMSE']:>{W}.0f}")
    overall = compute_metrics(y_true.flatten(), y_pred.flatten())
    print("─" * len(header))
    print(f"{'Avg':>5}  {overall['Combined']:>{W}.2f}  {overall['MAPE']:>{W}.2f}  "
          f"{overall['MAE_pct']:>{W}.2f}  {overall['RMSE_pct']:>{W}.2f}  "
          f"{overall['R2']:>{W}.4f}  {overall['MAE']:>{W}.0f}  {overall['RMSE']:>{W}.0f}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S"); out_dir = f"src/outputs/astgcn_tuned/{run_id}"
    os.makedirs(out_dir, exist_ok=True); torch.save(model.state_dict(), f"{out_dir}/model.pt")
    results = {
        "run_id": run_id, "model": "ASTGCNForecaster",
        "hparams": {
            "d_model": args.d_model, "n_blocks": args.n_blocks, "K": args.K,
            "n_heads": args.n_heads, "adj_threshold": args.adj_threshold,
            "dropout": args.dropout, "T_in": T_in, "T_out": T_out,
            "n_features": n_features, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "loss": args.loss,
            "warmup_epochs": args.warmup_epochs,
        },
        "training": {
            "best_epoch": best_epoch, "best_val_loss": round(best_val_loss, 8),
            "total_epochs": len(train_losses),
        },
        "fit_diagnosis": fit_diag,
        "split_dates": split_meta,
        "test_metrics": {
            "overall":  {k: round(v, 4) for k, v in overall.items()},
            "per_step": [{k: round(v, 4) for k, v in m.items()} for m in per_step],
        },
    }
    with open(f"{out_dir}/results.json", "w") as f: json.dump(results, f, indent=2)

    print(f"\nSaving plots...")
    plot_loss_curves(train_losses, val_losses, f"{out_dir}/loss_curves.png")
    plot_predictions(y_true, y_pred, overall,  f"{out_dir}/test_predictions.png")
    plot_per_step_metrics(per_step,            f"{out_dir}/per_step_metrics.png")

    print(f"\n{'='*85}"); print(f"ASTGCN (tuned) run complete  →  {out_dir}/"); print(f"{'='*85}")
    print(f"  Combined : {overall['Combined']:.2f}%")
    print(f"  MAPE     : {overall['MAPE']:.2f}%")
    print(f"  MAE%     : {overall['MAE_pct']:.2f}%    (raw MAE  = {overall['MAE']:.0f} riders)")
    print(f"  RMSE%    : {overall['RMSE_pct']:.2f}%   (raw RMSE = {overall['RMSE']:.0f} riders)")
    print(f"  R²       : {overall['R2']:.4f}")

    PRIOR_MODELS = [
        ("LSTM",          args.lstm_results,          "src/outputs/lstm",           "#2563eb"),
        ("BiLSTM",        args.bilstm_results,        "src/outputs/bilstm",         "#7c3aed"),
        ("TPA-LSTM",      args.tpalstm_results,       "src/outputs/tpa_lstm",       "#0891b2"),
        ("CNN-LSTM",      args.cnnlstm_results,       "src/outputs/cnn_lstm",       "#16a34a"),
        ("CNN-BiLSTM",    args.cnnbilstm_results,     "src/outputs/cnn_bilstm",     "#d97706"),
        ("ST-LSTM",       args.stlstm_results,        "src/outputs/st_lstm",        "#dc2626"),
        ("STGCN",         args.stgcn_results,         "src/outputs/stgcn",          "#10b981"),
        ("MTGNN",         args.mtgnn_results,         "src/outputs/mtgnn",          "#f472b6"),
        ("STSGCN",        args.stsgcn_results,        "src/outputs/stsgcn",         "#0ea5e9"),
        ("STFGNN",        args.stfgnn_results,        "src/outputs/stfgnn",         "#a855f7"),
        ("PDR-STGCN",     args.pdrstgcn_results,      "src/outputs/pdr_stgcn",      "#f97316"),
        ("TFT",           args.tft_results,           "src/outputs/tft",            "#ca8a04"),
        ("Autoformer",    args.autoformer_results,    "src/outputs/autoformer",     "#047857"),
        ("Informer",      args.informer_results,      "src/outputs/informer",       "#9333ea"),
    ]
    models_data = []; comparison = {}
    for name, path, model_dir, color in PRIOR_MODELS:
        data = load_model_results(path, model_dir, name)
        key  = name.lower().replace("-", "_").replace(" ", "_")
        if data:
            m_overall = data["test_metrics"]["overall"]
            m_ps      = data["test_metrics"]["per_step"]
            models_data.append((name, m_overall, m_ps, color))
            comparison[key] = {"run_id": data.get("run_id"), "overall": m_overall}
        else:
            comparison[key] = {"run_id": None, "overall": None}
    models_data.append(("ASTGCN", overall, per_step, "#e11d48"))

    if len(models_data) > 1:
        print_comparison_table(models_data[:-1], overall, "ASTGCN (tuned)")
        plot_comparison(models_data, out_path=f"{out_dir}/comparison_{len(models_data)}_way.png")

    comparison["astgcn"] = {"run_id": run_id, "overall": {k: round(v, 4) for k, v in overall.items()}}
    for name, m_overall, _, __ in models_data[:-1]:
        key = name.lower().replace("-", "_").replace(" ", "_")
        comparison[f"delta_vs_{key}"] = {k: round(overall.get(k, 0) - m_overall.get(k, 0), 4) for k in overall}
    results["comparison"] = comparison
    with open(f"{out_dir}/results.json", "w") as f: json.dump(results, f, indent=2)

    target_idx = split_meta.get("target_col_idx", 0)
    last_obs_s = X_te[:, -1, target_idx:target_idx+1].cpu().numpy()
    naive_pred_s = np.tile(last_obs_s, (1, T_out))
    if os.path.exists(scaler_y_path):
        N2b, T2b = naive_pred_s.shape
        naive_pred = scaler_y.inverse_transform(naive_pred_s.reshape(-1, 1)).reshape(N2b, T2b)
    else:
        naive_pred = naive_pred_s
    naive = compute_metrics(y_true.flatten(), naive_pred.flatten())
    d_comb = overall["Combined"] - naive["Combined"]; d_mape = naive["MAPE"] - overall["MAPE"]
    print(f"\n  Naive persistence   Combined={naive['Combined']:.2f}%  MAPE={naive['MAPE']:.2f}%  R²={naive['R2']:.4f}")
    print(f"  ASTGCN (tuned) vs naive     ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()
