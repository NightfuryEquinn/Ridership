"""
tft.py  — Temporal Fusion Transformer for Transit Ridership Forecasting

Adaptation:
  The original TFT uses static metadata, known future inputs, and
  observed past inputs. Since all 59 features are observed past inputs
  (no static or future covariates), this implementation focuses on:

  • Gated Residual Network (GRN) — core building block with ELU + GLU gate
  • Variable Selection Network (VSN) — softmax-weighted feature selection
  • LSTM encoder — captures local temporal dynamics
  • Multi-head Self-Attention — captures long-range dependencies
  • Gated Add-and-Norm — residual connections with gating throughout
  • MLP output head — projects to T_out scalar predictions

Architecture (forward pass):
  X (B, T_in, F)
  ──► VSN            : per-timestep feature selection → (B, T_in, d_model)
  ──► LSTM encoder   : local processing → (B, T_in, d_model)
  ──► Multi-head SA  : n_attn_layers × [Self-Attn + GAN + FFN + GAN]
  ──► Mean pool      : (B, d_model)
  ──► MLP head       : (B, T_out)

Hardware optimisations (RTX 4050 6 GB, 32 GB RAM, i5):
  • AMP (fp16) halves VRAM for activations and optimizer state.
  • GradScaler prevents gradient underflow in fp16.
  • Conservative defaults: d_model=64, n_lstm_layers=1, n_attn_layers=2.

Comparison (13-way):
  All 11 prior + ASTGCN, auto-detected from output directories.

Usage:
  python tft.py
  python tft.py --d-model 128 --n-heads 8 --n-attn-layers 2
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
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
    p = argparse.ArgumentParser(description="TFT forecaster — 13-way comparison")
    p.add_argument("--seq-dir",              default=None)
    p.add_argument("--d-model",              type=int,   default=64,
                   help="Hidden dimension throughout TFT")
    p.add_argument("--n-heads",              type=int,   default=4,
                   help="Number of attention heads")
    p.add_argument("--n-lstm-layers",        type=int,   default=1,
                   help="Stacked LSTM layers in encoder")
    p.add_argument("--n-attn-layers",        type=int,   default=2,
                   help="Transformer self-attention layers")
    p.add_argument("--dropout",              type=float, default=0.1)
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
    p.add_argument("--astgcn-results",       default=None)
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
# Model components
# ══════════════════════════════════════════════════════════════════════════════

class GRN(nn.Module):
    """
    Gated Residual Network (GRN) — core TFT building block.

    GRN(x) = LayerNorm(x + GLU(Linear2(ELU(Linear1(x)))))

    The GLU gate (Gated Linear Unit) allows the network to suppress
    irrelevant information:
        GLU([a, b]) = a ⊙ σ(b)
    """

    def __init__(self, d_model: int, d_hidden: int = None, dropout: float = 0.1):
        super().__init__()
        d_hidden = d_hidden or d_model
        self.fc1     = nn.Linear(d_model, d_hidden)
        self.fc2     = nn.Linear(d_hidden, d_model * 2)   # output for GLU split
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Skip-connection projection if d_model differs from input (not used here)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        h        = F.elu(self.fc1(x))                    # (..., d_hidden)
        h        = self.fc2(h)                            # (..., d_model * 2)
        h1, h2   = h.chunk(2, dim=-1)                    # each (..., d_model)
        glu_out  = h1 * torch.sigmoid(h2)                # GLU
        return self.norm(x + self.dropout(glu_out))


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN).

    At each timestep, learns softmax weights over all F features and
    produces a single d_model-dimensional representation of the selected
    feature mix.

    Steps:
      1. Project each feature individually: F × Linear(1, d_model)
      2. Flatten → compute selection weights via GRN → Softmax over F
      3. Weighted sum of individual projections → GRN → output
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        # Individual feature projections (shared weight for memory efficiency:
        #   one Linear maps ALL features simultaneously via split)
        self.feat_proj = nn.Linear(n_features, n_features * d_model)

        # Selection weights: GRN on flattened projected features → softmax
        self.select_grn = GRN(n_features * d_model, n_features * d_model // 2, dropout)
        self.select_fc  = nn.Linear(n_features * d_model, n_features)

        # Output GRN
        self.output_grn = GRN(d_model, dropout=dropout)
        self.d_model    = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        B, T, F = x.shape
        # Project all features jointly then reshape to per-feature embeddings
        feat_flat  = self.feat_proj(x)                         # (B, T, F*d_model)
        feat_stack = feat_flat.reshape(B, T, F, self.d_model)  # (B, T, F, d_model)

        # Selection weights
        sel_in      = self.select_grn(feat_flat)               # (B, T, F*d_model)
        sel_weights = torch.softmax(self.select_fc(sel_in), dim=-1)  # (B, T, F)

        # Weighted sum over features
        out = (sel_weights.unsqueeze(-1) * feat_stack).sum(dim=2)    # (B, T, d_model)
        return self.output_grn(out)


class GatedAddNorm(nn.Module):
    """
    Gated Add-and-Norm — skip connection with learned gate.

    output = LayerNorm(x + GLU([skip, x]))
    where skip is the pre-layer input (as in TFT's residual connections).
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gate    = nn.Linear(d_model, d_model * 2)   # GLU gate
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x, skip: (..., d_model)
        g  = self.gate(x)
        g1, g2 = g.chunk(2, dim=-1)
        gated = g1 * torch.sigmoid(g2)
        return self.norm(skip + self.dropout(gated))


class TFTForecaster(nn.Module):
    """
    Temporal Fusion Transformer forecaster (adapted for observed-inputs only).

    Input:  (B, T_in, n_features)
    Output: (B, T_out)
    """

    def __init__(
        self,
        n_features:    int,
        T_in:          int,
        T_out:         int,
        d_model:       int   = 64,
        n_heads:       int   = 4,
        n_lstm_layers: int   = 1,
        n_attn_layers: int   = 2,
        dropout:       float = 0.1,
    ):
        super().__init__()

        # 1. Variable Selection Network
        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)
        self.vsn_gan = GatedAddNorm(d_model, dropout)

        # 2. LSTM encoder for local processing
        self.lstm = nn.LSTM(
            input_size  = d_model,
            hidden_size = d_model,
            num_layers  = n_lstm_layers,
            batch_first = True,
            dropout     = dropout if n_lstm_layers > 1 else 0.0,
        )
        self.lstm_gan = GatedAddNorm(d_model, dropout)

        # 3. Multi-head self-attention (interpretable — single head per TFT paper,
        #    but we use multi-head for stronger performance)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_attn_layers)
        ])
        self.attn_gans = nn.ModuleList([
            GatedAddNorm(d_model, dropout) for _ in range(n_attn_layers)
        ])

        # 4. Position-wise FFN per attention layer
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(n_attn_layers)
        ])
        self.ffn_gans = nn.ModuleList([
            GatedAddNorm(d_model, dropout) for _ in range(n_attn_layers)
        ])

        # 5. Output: mean-pool over T_in → head → T_out
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, T_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, F)

        # 1. Variable selection
        vsn_in = self.vsn(x)                              # (B, T, d_model)
        h      = self.vsn_gan(vsn_in, vsn_in)             # (B, T, d_model)

        # 2. LSTM encoder
        lstm_in  = h
        lstm_out, _ = self.lstm(h)                        # (B, T, d_model)
        h        = self.lstm_gan(lstm_out, lstm_in)       # gated skip

        # 3. Self-attention layers
        for attn, attn_gan, ffn, ffn_gan in zip(
            self.attn_layers, self.attn_gans, self.ffns, self.ffn_gans
        ):
            attn_in     = h
            attn_out, _ = attn(h, h, h)                  # (B, T, d_model)
            h           = attn_gan(attn_out, attn_in)     # gated skip

            ffn_in = h
            h      = ffn_gan(ffn(h), ffn_in)              # gated skip

        # 4. Pool and project
        h   = h.mean(dim=1)                               # (B, d_model)
        return self.head(h)                               # (B, T_out)



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
# Comparison helpers
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(train_losses, val_losses, out_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train Loss", linewidth=1.5, color="#ca8a04")
    ax.plot(val_losses,   label="Val Loss",   linewidth=1.5, color="#f97316", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (scaled)")
    ax.set_title("TFT — Training curves")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig); print(f"  Saved: {out_path}")


def plot_predictions(y_true, y_pred, metrics, out_path):
    N, T_out = y_true.shape; idx_full = np.arange(N)
    zoom_n = min(60, N); idx_zoom = np.arange(N - zoom_n, N)
    actual_s1 = y_true[:, 0]; predicted_s1 = y_pred[:, 0]
    pred_min = y_pred.min(axis=1); pred_max = y_pred.max(axis=1)
    C_ACT = "#1d4ed8"; C_PRED = "#ca8a04"; C_BAND = "#fef9c3"; C_MID = "#0891b2"; C_LAST = "#7c3aed"
    fig = plt.figure(figsize=(14, 8)); gs = gridspec.GridSpec(2, 1, hspace=0.48)
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(idx_full, pred_min, pred_max, alpha=0.22, color=C_BAND,
                     label=f"Forecast spread (step 1–{T_out})")
    ax1.plot(idx_full, actual_s1,    color=C_ACT, linewidth=1.5, label="Actual", zorder=4)
    ax1.plot(idx_full, predicted_s1, color=C_PRED, linewidth=1.2, linestyle="--", alpha=0.88,
             label="TFT predicted (step 1)", zorder=5)
    ann = (f"Combined = {metrics['Combined']:.2f}%\nMAPE     = {metrics['MAPE']:.2f}%\n"
           f"MAE%     = {metrics['MAE_pct']:.2f}%\nRMSE%    = {metrics['RMSE_pct']:.2f}%\n"
           f"R²       = {metrics['R2']:.4f}\nMAE      = {metrics['MAE']:.0f} riders\n"
           f"RMSE     = {metrics['RMSE']:.0f} riders")
    ax1.text(0.01, 0.97, ann, transform=ax1.transAxes, fontsize=8, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#d1d5db", alpha=0.92))
    ax1.set_title("TFT — Test set: Actual vs Predicted (full period)", fontsize=11, fontweight="bold")
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
    ax1.set_title("TFT — Per-horizon metrics", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)
    ax2.plot(steps, r2, marker="o", color="#ca8a04", linewidth=1.8, markersize=5)
    ax2.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax2.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax2.set_ylim(min(min(r2) - 0.05, -0.1), 1.08)
    ax2.set_ylabel("R²"); ax2.set_title("TFT — R² per horizon", fontweight="bold")
    ax2.grid(alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  Saved: {out_path}")


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

    model = TFTForecaster(
        n_features    = n_features,
        T_in          = T_in,
        T_out         = T_out,
        d_model       = args.d_model,
        n_heads       = args.n_heads,
        n_lstm_layers = args.n_lstm_layers,
        n_attn_layers = args.n_attn_layers,
        dropout       = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel          : TFTForecaster")
    print(f"  d_model      : {args.d_model}   n_heads: {args.n_heads}")
    print(f"  LSTM layers  : {args.n_lstm_layers}   Attn layers: {args.n_attn_layers}")
    print(f"  In           : (batch, {T_in}, {n_features})")
    print(f"  Out          : (batch, {T_out})")
    print(f"  Params       : {n_params:,}")

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
    model.eval(); preds_s, trues_s = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds_s.append(model(X_b).cpu().numpy()); trues_s.append(y_b.cpu().numpy())
    y_pred_s = np.concatenate(preds_s); y_true_s = np.concatenate(trues_s)

    scaler_y_path = os.path.join(args.seq_dir, "scaler_y.pkl")
    if os.path.exists(scaler_y_path):
        scaler_y = joblib.load(scaler_y_path); N2, T2 = y_pred_s.shape
        y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).reshape(N2, T2)
        y_true = scaler_y.inverse_transform(y_true_s.reshape(-1, 1)).reshape(N2, T2)
    else:
        print("[WARN] scaler_y.pkl not found — metrics in scaled units.")
        y_pred, y_true = y_pred_s, y_true_s

    W = 10; print(f"\n{'='*85}"); print("TFT — TEST SET METRICS"); print(f"{'='*85}")
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

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S"); out_dir = f"src/outputs/tft/{run_id}"
    os.makedirs(out_dir, exist_ok=True); torch.save(model.state_dict(), f"{out_dir}/model.pt")
    results = {
        "run_id": run_id, "model": "TFTForecaster",
        "hparams": {
            "d_model": args.d_model, "n_heads": args.n_heads,
            "n_lstm_layers": args.n_lstm_layers, "n_attn_layers": args.n_attn_layers,
            "dropout": args.dropout, "T_in": T_in, "T_out": T_out,
            "n_features": n_features, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "loss": args.loss,
            "warmup_epochs": args.warmup_epochs,
        },
        "training": {
            "best_epoch": best_epoch, "best_val_loss": round(best_val_loss, 8),
            "total_epochs": len(train_losses),
        },
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

    print(f"\n{'='*85}"); print(f"TFT run complete  →  {out_dir}/"); print(f"{'='*85}")
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
        ("PDR-STGCN",      args.pdrstgcn_results,       "src/outputs/pdr_stgcn",       "#f97316"),
        ("ASTGCN",        args.astgcn_results,        "src/outputs/astgcn",         "#e11d48"),
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
    models_data.append(("TFT", overall, per_step, "#ca8a04"))

    if len(models_data) > 1:
        print_comparison_table(models_data[:-1], overall, "TFT")
        plot_comparison(models_data, out_path=f"{out_dir}/comparison_{len(models_data)}_way.png")

    comparison["tft"] = {"run_id": run_id, "overall": {k: round(v, 4) for k, v in overall.items()}}
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
    print(f"  TFT vs naive        ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()
