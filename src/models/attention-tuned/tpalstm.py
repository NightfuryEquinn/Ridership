"""
tpalstm.py  — TPA-LSTM (Fine-Tuned) for Transit Ridership Forecasting

Tuned vs base (src/models/attention-based/tpalstm.py):
  | Parameter    | Base | Tuned | Rationale                                      |
  |--------------|------|-------|------------------------------------------------|
  | hidden       |  64  |  128  | Low Combined% → more capacity                  |
  | filters      |  32  |   64  | Proportional to hidden; richer pattern space   |
  | dropout      | 0.10 |  0.15 | Moderate variance across lookbacks → mild reg  |

Training hyperparameters (epochs, batch_size, lr, patience, weight_decay)
are intentionally unchanged — only architecture parameters differ.

Output directory: src/outputs/tpa_lstm_tuned/

Usage:
  python src/models/attention-tuned/tpalstm.py
  python src/models/attention-tuned/tpalstm.py --hidden 128 --filters 64 --dropout 0.15
  python src/models/attention-tuned/tpalstm.py --lookback 28
  python src/models/attention-tuned/tpalstm.py --include-mco
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
    p = argparse.ArgumentParser(
        description="TPA-LSTM (tuned) forecaster"
    )
    p.add_argument("--seq-dir",        default=None,
                   help="Directory with X/y .npy splits")
    p.add_argument("--hidden",         type=int,   default=128,
                   help="LSTM hidden size  [tuned: 128, base: 64]")
    p.add_argument("--layers",         type=int,   default=1,
                   help="LSTM stacked layers")
    p.add_argument("--dropout",        type=float, default=0.15,
                   help="Inter-layer dropout  [tuned: 0.15, base: 0.10]")
    p.add_argument("--filters",        type=int,   default=64,
                   help="CNN filters in TPA module  [tuned: 64, base: 32]")
    p.add_argument("--kernel-size",    type=int,   default=3,
                   help="1-D CNN kernel size in TPA module")
    p.add_argument("--batch-size",     type=int,   default=32)
    p.add_argument("--epochs",         type=int,   default=150)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--patience",       type=int,   default=15)
    p.add_argument("--device",         default="auto", help="cpu | cuda | mps | auto")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--lstm-results",        default=None)
    p.add_argument("--bilstm-results",      default=None)
    p.add_argument("--cnnlstm-results",     default=None)
    p.add_argument("--cnnbilstm-results",   default=None)
    p.add_argument("--stlstm-results",      default=None)
    p.add_argument("--stgcn-results",       default=None)
    p.add_argument("--mtgnn-results",       default=None)
    p.add_argument("--stsgcn-results",      default=None)
    p.add_argument("--stfgnn-results",      default=None)
    p.add_argument("--pdrstgcn-results",    default=None)
    p.add_argument("--astgcn-results",      default=None)
    p.add_argument("--tft-results",         default=None)
    p.add_argument("--autoformer-results",  default=None)
    p.add_argument("--informer-results",    default=None)
    p.add_argument("--lookback",      type=int,   default=14, choices=[14, 28, 56],
                   help="Look-back window; auto-selects seq-dir when --seq-dir is not set")
    p.add_argument("--loss",          default="huber", choices=["mse", "huber", "mae"],
                   help="Training loss: mse | huber (default) | mae")
    p.add_argument("--warmup-epochs", type=int,   default=5,
                   help="Linear LR warm-up epochs before ReduceLROnPlateau kicks in")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class TemporalPatternAttention(nn.Module):
    """
    TPA attention module.

    Takes the full LSTM hidden-state sequence H and the final hidden state h_T
    and returns a context vector of size `hidden_size`.

    Steps:
      1. Slice H into context states H_ctx = H[:, :-1, :]  (exclude last step)
      2. Transpose to (B, hidden, T-1) — hidden dims become "channels" over time
      3. Apply Conv1d with `n_filters` filters of width `kernel_size`
         to extract recurring temporal patterns
      4. Max-pool across the time axis → pattern matrix C: (B, n_filters, hidden)
         (achieved by treating Conv1d output as (B, n_filters, *) then pooling)
      5. Score each filter against h_T via a learnt scoring vector:
            score = sigmoid(C @ W_score @ h_T^T)   →  (B, n_filters)
      6. Softmax → attention weights α: (B, n_filters)
      7. Context = α^T @ C_mean  →  (B, hidden)
    """

    def __init__(self, hidden_size: int, n_filters: int, kernel_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_filters   = n_filters
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            in_channels  = hidden_size,
            out_channels = n_filters,
            kernel_size  = kernel_size,
            padding      = kernel_size // 2,
        )

        self.W_score   = nn.Linear(hidden_size, n_filters, bias=False)
        self.W_context = nn.Linear(n_filters, hidden_size, bias=False)

    def forward(
        self,
        H:   torch.Tensor,   # (B, T_in, hidden_size)
        h_T: torch.Tensor,   # (B, hidden_size)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H_dim = H.shape

        H_ctx = H[:, :-1, :]
        H_ctx = H_ctx.permute(0, 2, 1).contiguous()   # (B, hidden, T-1)

        C = F.relu(self.conv(H_ctx))                   # (B, n_filters, T-1)
        C_pool = C.mean(dim=-1)                        # (B, n_filters)

        score        = torch.sigmoid(self.W_score(h_T) * C_pool)   # (B, n_filters)
        attn_weights = torch.softmax(score, dim=-1)                 # (B, n_filters)

        context = self.W_context(attn_weights)                      # (B, hidden_size)

        return context, attn_weights


class TPALSTMForecaster(nn.Module):
    """
    TPA-LSTM: LSTM encoder + Temporal Pattern Attention → MLP head.

    Input:  (batch, T_in, n_features)
    Output: (batch, T_out)

    Head input = cat([h_T, context]) = hidden_size * 2
    """

    def __init__(
        self,
        n_features:  int,
        hidden_size: int,
        n_layers:    int,
        T_out:       int,
        n_filters:   int   = 64,
        kernel_size: int   = 3,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )

        self.tpa = TemporalPatternAttention(
            hidden_size = hidden_size,
            n_filters   = n_filters,
            kernel_size = kernel_size,
        )

        head_in = hidden_size * 2
        self.head = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.ReLU(),
            nn.Linear(head_in // 2, T_out),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        H, (h_n, _) = self.lstm(x)
        h_T = h_n[-1]

        context, attn = self.tpa(H, h_T)

        h_cat = torch.cat([h_T, context], dim=-1)
        out   = self.head(h_cat)

        return (out, attn) if return_attn else out


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_splits(seq_dir: str, device: torch.device):
    def t(name):
        return torch.from_numpy(
            np.load(os.path.join(seq_dir, name))
        ).float().to(device)

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


@torch.no_grad()
def collect_predictions_and_attention(
    model:  TPALSTMForecaster,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds_s, trues_s, attns = [], [], []
    for X_b, y_b in loader:
        out, attn = model(X_b, return_attn=True)
        preds_s.append(out.cpu().numpy())
        trues_s.append(y_b.cpu().numpy())
        attns.append(attn.cpu().numpy())
    return (
        np.concatenate(preds_s),
        np.concatenate(trues_s),
        np.concatenate(attns),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(train_losses, val_losses, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train Loss", linewidth=1.5, color="#0891b2")
    ax.plot(val_losses,   label="Val Loss",   linewidth=1.5, color="#f97316", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (scaled)")
    ax.set_title("TPA-LSTM (tuned) — Training curves")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig); print(f"  Saved: {out_path}")


def plot_predictions(y_true, y_pred, metrics, out_path: str) -> None:
    N, T_out = y_true.shape
    idx_full = np.arange(N)
    zoom_n   = min(60, N)
    idx_zoom = np.arange(N - zoom_n, N)
    actual_s1    = y_true[:, 0]
    predicted_s1 = y_pred[:, 0]
    pred_min     = y_pred.min(axis=1)
    pred_max     = y_pred.max(axis=1)
    C_ACTUAL = "#1d4ed8"; C_PRED = "#0891b2"; C_BAND = "#a5f3fc"; C_MID = "#ea580c"; C_LAST = "#7c3aed"
    fig = plt.figure(figsize=(14, 8)); gs = gridspec.GridSpec(2, 1, hspace=0.48)
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(idx_full, pred_min, pred_max, alpha=0.22, color=C_BAND,
                     label=f"Forecast spread (step 1–{T_out})")
    ax1.plot(idx_full, actual_s1,    color=C_ACTUAL, linewidth=1.5, label="Actual", zorder=4)
    ax1.plot(idx_full, predicted_s1, color=C_PRED, linewidth=1.2, linestyle="--", alpha=0.88,
             label="TPA-LSTM (tuned) predicted (step 1)", zorder=5)
    annotation = (f"Combined = {metrics['Combined']:.2f}%\nMAPE     = {metrics['MAPE']:.2f}%\n"
                  f"MAE%     = {metrics['MAE_pct']:.2f}%\nRMSE%    = {metrics['RMSE_pct']:.2f}%\n"
                  f"R²       = {metrics['R2']:.4f}\nMAE      = {metrics['MAE']:.0f} riders\n"
                  f"RMSE     = {metrics['RMSE']:.0f} riders")
    ax1.text(0.01, 0.97, annotation, transform=ax1.transAxes, fontsize=8, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#d1d5db", alpha=0.92))
    ax1.set_title("TPA-LSTM (tuned) — Test set: Actual vs Predicted (full period)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Test sample index"); ax1.set_ylabel("Ridership (riders)")
    ax1.legend(loc="upper right", fontsize=8); ax1.grid(alpha=0.25)
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(idx_zoom, pred_min[idx_zoom], pred_max[idx_zoom], alpha=0.18, color=C_BAND)
    ax2.plot(idx_zoom, actual_s1[idx_zoom], color=C_ACTUAL, linewidth=1.7, label="Actual", zorder=5)
    steps_to_show = sorted({0, T_out // 2, T_out - 1})
    palette = [C_PRED, C_MID, C_LAST]; styles = ["--", "-.", ":"]
    for s, col, ls in zip(steps_to_show, palette, styles):
        ax2.plot(idx_zoom, y_pred[idx_zoom, s], color=col, linewidth=1.4,
                 linestyle=ls, alpha=0.88, label=f"Predicted step {s + 1}")
    ax2.set_title(f"Zoomed: last {zoom_n} samples — step 1 / {T_out // 2 + 1} / {T_out} horizon comparison",
                  fontsize=10, fontweight="bold")
    ax2.set_xlabel("Test sample index"); ax2.set_ylabel("Ridership (riders)")
    ax2.legend(loc="upper right", fontsize=8, ncol=2); ax2.grid(alpha=0.25)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  Saved: {out_path}")


def plot_per_step_metrics(per_step: list, out_path: str) -> None:
    steps    = [f"t+{i+1}" for i in range(len(per_step))]
    combined = [m["Combined"] for m in per_step]; mape = [m["MAPE"] for m in per_step]
    mae_pct  = [m["MAE_pct"]  for m in per_step]; rmse_pct = [m["RMSE_pct"] for m in per_step]
    r2       = [m["R2"]       for m in per_step]
    x, w = np.arange(len(steps)), 0.2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(steps) * 0.85), 7),
                                    gridspec_kw={"hspace": 0.48})
    ax1.bar(x - 1.5*w, combined, w, label="Combined%", color="#16a34a", alpha=0.85)
    ax1.bar(x - 0.5*w, mape,     w, label="MAPE%",     color="#dc2626", alpha=0.85)
    ax1.bar(x + 0.5*w, mae_pct,  w, label="MAE%",      color="#2563eb", alpha=0.85)
    ax1.bar(x + 1.5*w, rmse_pct, w, label="RMSE%",     color="#f97316", alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(steps); ax1.set_ylabel("% of mean demand  /  score")
    ax1.set_title("TPA-LSTM (tuned) — Per-horizon metrics", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)
    ax2.plot(steps, r2, marker="o", color="#0891b2", linewidth=1.8, markersize=5)
    ax2.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax2.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax2.set_ylim(min(min(r2) - 0.05, -0.1), 1.08)
    ax2.set_ylabel("R²"); ax2.set_title("TPA-LSTM (tuned) — R² per horizon step", fontweight="bold")
    ax2.grid(alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  Saved: {out_path}")


def plot_attention_heatmap(
    attn_weights: np.ndarray,
    out_path:     str,
    n_samples:    int = 200,
) -> None:
    W = attn_weights[:n_samples].T    # (n_filters, n_samples)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"hspace": 0.45})
    ax = axes[0]
    im = ax.imshow(W, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Attention weight")
    ax.set_xlabel(f"Test sample index (first {n_samples})")
    ax.set_ylabel("CNN filter (temporal pattern)")
    ax.set_title("TPA Attention Weights — which filters activate per test window", fontweight="bold")
    ax.set_yticks(np.arange(W.shape[0]))
    ax.set_yticklabels([f"Filter {i}" for i in range(W.shape[0])], fontsize=7)
    ax2 = axes[1]
    mean_attn = attn_weights.mean(axis=0); std_attn = attn_weights.std(axis=0)
    x = np.arange(len(mean_attn))
    ax2.bar(x, mean_attn, color="#0891b2", alpha=0.82,
            yerr=std_attn, capsize=3, error_kw={"linewidth": 0.8})
    ax2.set_xlabel("CNN filter index"); ax2.set_ylabel("Mean attention weight ± std")
    ax2.set_title("Average filter importance across all test windows", fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels([f"F{i}" for i in range(len(mean_attn))], fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
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
    print(f"Device: {device}")
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

    model = TPALSTMForecaster(
        n_features  = n_features,
        hidden_size = args.hidden,
        n_layers    = args.layers,
        T_out       = T_out,
        n_filters   = args.filters,
        kernel_size = args.kernel_size,
        dropout     = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel   : TPALSTMForecaster (tuned)")
    print(f"  LSTM  : hidden={args.hidden}  layers={args.layers}")
    print(f"  TPA   : filters={args.filters}  kernel={args.kernel_size}")
    print(f"  Head  : hidden×2={args.hidden*2} → {T_out}")
    print(f"  In    : (batch, {T_in}, {n_features})")
    print(f"  Out   : (batch, {T_out})")
    print(f"  Params: {n_params:,}")

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
            for pg in optimiser.param_groups:
                pg["lr"] = args.lr * epoch / args.warmup_epochs
        tr_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
        va_loss = evaluate(model, val_loader, criterion)
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
                print(f"\nEarly stop at epoch {epoch}  (best={best_val_loss:.6f} @ epoch {best_epoch})")
                break

    model.load_state_dict(best_state)

    y_pred_s, y_true_s, attn_weights = collect_predictions_and_attention(model, test_loader)

    scaler_y_path = os.path.join(args.seq_dir, "scaler_y.pkl")
    if os.path.exists(scaler_y_path):
        scaler_y = joblib.load(scaler_y_path)
        N, T     = y_pred_s.shape
        y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).reshape(N, T)
        y_true   = scaler_y.inverse_transform(y_true_s.reshape(-1, 1)).reshape(N, T)
    else:
        print("[WARN] scaler_y.pkl not found — metrics in scaled units.")
        y_pred, y_true = y_pred_s, y_true_s

    W = 10
    print(f"\n{'='*85}"); print("TPA-LSTM (tuned) — TEST SET METRICS"); print(f"{'='*85}")
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

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"src/outputs/tpa_lstm_tuned/{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{out_dir}/model.pt")
    np.save(f"{out_dir}/attn_weights_test.npy", attn_weights)

    results = {
        "run_id": run_id, "model": "TPALSTMForecaster",
        "hparams": {
            "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
            "n_filters": args.filters, "kernel_size": args.kernel_size,
            "T_in": T_in, "T_out": T_out, "n_features": n_features,
            "batch_size": args.batch_size, "lr": args.lr,
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
    plot_attention_heatmap(attn_weights,       f"{out_dir}/attention_heatmap.png")

    print(f"\n{'='*85}"); print(f"TPA-LSTM (tuned) run complete  →  {out_dir}/"); print(f"{'='*85}")
    print(f"  Combined  : {overall['Combined']:.2f}%")
    print(f"  MAPE      : {overall['MAPE']:.2f}%")
    print(f"  MAE%      : {overall['MAE_pct']:.2f}%    (raw MAE  = {overall['MAE']:.0f} riders)")
    print(f"  RMSE%     : {overall['RMSE_pct']:.2f}%   (raw RMSE = {overall['RMSE']:.0f} riders)")
    print(f"  R²        : {overall['R2']:.4f}")
    print(f"\n  Attention: mean filter entropy = "
          f"{float(-(attn_weights * np.log(attn_weights + 1e-9)).sum(axis=1).mean()):.4f}"
          f"  (higher = more distributed; lower = more selective)")

    PRIOR_MODELS = [
        ("LSTM",       args.lstm_results,       "src/outputs/lstm",       "#2563eb"),
        ("BiLSTM",     None,                    "src/outputs/bilstm",     "#7c3aed"),
        ("CNN-LSTM",   args.cnnlstm_results,    "src/outputs/cnn_lstm",   "#16a34a"),
        ("CNN-BiLSTM", args.cnnbilstm_results,  "src/outputs/cnn_bilstm", "#d97706"),
        ("ST-LSTM",    args.stlstm_results,     "src/outputs/st_lstm",    "#dc2626"),
        ("STGCN",      args.stgcn_results,      "src/outputs/stgcn",      "#10b981"),
        ("MTGNN",      args.mtgnn_results,      "src/outputs/mtgnn",      "#f472b6"),
        ("STSGCN",     args.stsgcn_results,     "src/outputs/stsgcn",     "#0ea5e9"),
        ("STFGNN",     args.stfgnn_results,     "src/outputs/stfgnn",     "#a855f7"),
        ("PDR-STGCN",  args.pdrstgcn_results,   "src/outputs/pdr_stgcn",  "#f97316"),
        ("ASTGCN",     args.astgcn_results,     "src/outputs/astgcn",     "#e11d48"),
        ("TFT",        args.tft_results,        "src/outputs/tft",        "#ca8a04"),
        ("Autoformer", args.autoformer_results, "src/outputs/autoformer", "#047857"),
        ("Informer",   args.informer_results,   "src/outputs/informer",   "#9333ea"),
    ]

    models_data = []; comparison = {}
    for name, path, model_dir, color in PRIOR_MODELS:
        key  = name.lower().replace("-", "_").replace(" ", "_")
        data = load_model_results(path, model_dir, name)
        if data:
            m_overall = data["test_metrics"]["overall"]
            m_ps      = data["test_metrics"]["per_step"]
            models_data.append((name, m_overall, m_ps, color))
            comparison[key] = {"run_id": data.get("run_id"), "overall": m_overall}
        else:
            comparison[key] = {"run_id": None, "overall": None}

    models_data.append(("TPA-LSTM", overall, per_step, "#0891b2"))

    if len(models_data) > 1:
        print_comparison_table(models_data[:-1], overall, "TPA-LSTM (tuned)")
        plot_comparison(models_data, out_path=f"{out_dir}/comparison_{len(models_data)}_way.png")

    comparison["tpa_lstm"] = {"run_id": run_id, "overall": {k: round(v, 4) for k, v in overall.items()}}
    for name, m_overall, _, __ in models_data[:-1]:
        key = name.lower().replace("-", "_").replace(" ", "_")
        comparison[f"delta_vs_{key}"] = {k: round(overall.get(k, 0) - m_overall.get(k, 0), 4) for k in overall}
    results["comparison"] = comparison
    with open(f"{out_dir}/results.json", "w") as f: json.dump(results, f, indent=2)

    target_idx   = split_meta.get("target_col_idx", 0)
    last_obs_s   = X_te[:, -1, target_idx:target_idx+1].cpu().numpy()
    naive_pred_s = np.tile(last_obs_s, (1, T_out))
    if os.path.exists(scaler_y_path):
        N2, T2 = naive_pred_s.shape
        naive_pred = scaler_y.inverse_transform(naive_pred_s.reshape(-1, 1)).reshape(N2, T2)
    else:
        naive_pred = naive_pred_s
    naive  = compute_metrics(y_true.flatten(), naive_pred.flatten())
    d_comb = overall["Combined"] - naive["Combined"]; d_mape = naive["MAPE"] - overall["MAPE"]
    print(f"\n  Naive persistence  Combined={naive['Combined']:.2f}%  MAPE={naive['MAPE']:.2f}%  R²={naive['R2']:.4f}")
    print(f"  TPA-LSTM (tuned) vs naive  ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()
