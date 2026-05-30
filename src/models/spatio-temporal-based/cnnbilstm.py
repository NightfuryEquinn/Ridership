"""
cnnbilstm.py  — Convolutional Neural Network Bidirectional LSTM (CNN-BiLSTM)

Key Features:
  • Combines CNN local pattern extraction with bidirectional LSTM sequential context
  • CNN applied channels-first with same-padding, preserving the time dimension
  • BiLSTM replaces LSTM so the encoder sees the full look-back window from both directions
  • Mid-window anomalies encoded with past and future context rather than only past
  • Each CNN block: Conv1d → BatchNorm1d → ReLU

Architecture:
  X             : (B, T_in, F)
  permute        → (B, F, T_in)              # channels-first for Conv1d
  Conv1d × L    → (B, cnn_filters, T_in)    # same-padding preserves T_in
  permute        → (B, T_in, cnn_filters)
  BiLSTM         → h_n : (2×layers, B, hidden)
  h_fwd          = h_n[-2] : (B, hidden)    # last layer, forward direction
  h_bwd          = h_n[-1] : (B, hidden)    # last layer, backward direction
  cat            → (B, hidden × 2)
  MLP head      → (B, T_out)

Hardware:
  GPU  : NVIDIA A100 (32 GB VRAM)
  RAM  : 32 GB
  Precision : float32

References
----------
Chen, W., Yang, Z., Xu, G., & Sun, Y. (2022). Short-term traffic flow prediction
based on CNN-BiLSTM with multicomponent information. Applied Sciences, 12(17), 8714.
DOI: https://doi.org/10.3390/app12178714
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
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
        description="CNN-BiLSTM forecaster with LSTM / BiLSTM / TPA-LSTM / CNN-LSTM comparison"
    )
    p.add_argument("--seq-dir",          default=None,
                   help="Directory with X/y .npy splits")
    p.add_argument("--hidden",           type=int,   default=64,
                   help="BiLSTM hidden size (per direction)")
    p.add_argument("--layers",           type=int,   default=1,
                   help="Stacked BiLSTM layers")
    p.add_argument("--dropout",          type=float, default=0.1,
                   help="Inter-layer BiLSTM dropout (active only when --layers > 1)")
    p.add_argument("--cnn-filters",      type=int,   default=32,
                   help="Number of CNN filters per convolutional layer")
    p.add_argument("--cnn-layers",       type=int,   default=2,
                   help="Number of stacked 1-D CNN blocks")
    p.add_argument("--cnn-kernel-size",  type=int,   default=3,
                   help="1-D CNN kernel size (same-padding applied)")
    p.add_argument("--batch-size",       type=int,   default=32)
    p.add_argument("--epochs",           type=int,   default=150)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--weight-decay",     type=float, default=1e-4)
    p.add_argument("--patience",         type=int,   default=15)
    p.add_argument("--device",           default="auto", help="cpu | cuda | mps | auto")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--lstm-results",     default=None,
                   help="Path to lstm results.json (auto-detected if omitted)")
    p.add_argument("--bilstm-results",   default=None,
                   help="Path to bilstm results.json (auto-detected if omitted)")
    p.add_argument("--tpalstm-results",  default=None,
                   help="Path to tpa_lstm results.json (auto-detected if omitted)")
    p.add_argument("--cnnlstm-results",   default=None,
                   help="Path to cnn_lstm results.json (auto-detected if omitted)")
    p.add_argument("--stlstm-results",    default=None)
    p.add_argument("--stgcn-results",     default=None)
    p.add_argument("--mtgnn-results",     default=None)
    p.add_argument("--stsgcn-results",    default=None)
    p.add_argument("--stfgnn-results",    default=None)
    p.add_argument("--pdrstgcn-results",  default=None)
    p.add_argument("--astgcn-results",    default=None)
    p.add_argument("--tft-results",       default=None)
    p.add_argument("--autoformer-results",default=None)
    p.add_argument("--informer-results",  default=None)
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

class CNNBiLSTMForecaster(nn.Module):
    """
    CNN-BiLSTM: 1-D CNN feature extractor → BiLSTM encoder → MLP head.

    Input:  (batch, T_in, n_features)
    Output: (batch, T_out)

    The CNN maps (B, F, T_in) → (B, cnn_filters, T_in) using same-padding so
    temporal length is preserved for the BiLSTM.  BatchNorm after each Conv1d
    stabilises training on heterogeneous features.

    The BiLSTM final hidden state is the concatenation of the forward direction
    (h_n[-2]) and the backward direction (h_n[-1]) from the top layer, giving
    a vector of size hidden_size × 2 that feeds the MLP head.
    """

    def __init__(
        self,
        n_features:   int,
        cnn_filters:  int,
        cnn_layers:   int,
        kernel_size:  int,
        hidden_size:  int,
        lstm_layers:  int,
        T_out:        int,
        dropout:      float = 0.0,
    ):
        super().__init__()

        # ── CNN front-end ─────────────────────────────────────────────────────
        pad = kernel_size // 2
        cnn_blocks = []
        in_ch = n_features
        for _ in range(cnn_layers):
            cnn_blocks.extend([
                nn.Conv1d(in_ch, cnn_filters, kernel_size=kernel_size, padding=pad),
                nn.BatchNorm1d(cnn_filters),
                nn.ReLU(),
            ])
            in_ch = cnn_filters
        self.cnn = nn.Sequential(*cnn_blocks)

        # ── BiLSTM encoder ────────────────────────────────────────────────────
        self.bilstm = nn.LSTM(
            input_size    = cnn_filters,
            hidden_size   = hidden_size,
            num_layers    = lstm_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if lstm_layers > 1 else 0.0,
        )

        # ── MLP head ──────────────────────────────────────────────────────────
        # Input = hidden_size × 2  (forward ‖ backward concat)
        head_in = hidden_size * 2
        self.head = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.ReLU(),
            nn.Linear(head_in // 2, T_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T_in, F)

        Returns
        -------
        out : (B, T_out)
        """
        # CNN: permute to channels-first, apply, permute back
        x_cnn = x.permute(0, 2, 1)             # (B, F, T_in)
        x_cnn = self.cnn(x_cnn)                # (B, cnn_filters, T_in)
        x_cnn = x_cnn.permute(0, 2, 1)         # (B, T_in, cnn_filters)

        # BiLSTM: h_n shape = (2 × lstm_layers, B, hidden)
        _, (h_n, _) = self.bilstm(x_cnn)
        h_fwd = h_n[-2]                         # last layer forward:  (B, hidden)
        h_bwd = h_n[-1]                         # last layer backward: (B, hidden)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, hidden × 2)

        return self.head(h_cat)                 # (B, T_out)



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



# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(train_losses, val_losses, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train Loss", linewidth=1.5, color="#d97706")
    ax.plot(val_losses,   label="Val Loss",   linewidth=1.5, color="#f97316",
            linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (scaled)")
    ax.set_title("CNN-BiLSTM — Training curves")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_predictions(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    metrics:  dict,
    out_path: str,
) -> None:
    """
    Two-panel overlay — identical layout to all other model scripts.
    Panel 1: full test period actual vs predicted (step 1) + spread band.
    Panel 2: zoomed last 60 samples, step 1 / mid / last horizon lines.
    """
    N, T_out = y_true.shape
    idx_full = np.arange(N)
    zoom_n   = min(60, N)
    idx_zoom = np.arange(N - zoom_n, N)

    actual_s1    = y_true[:, 0]
    predicted_s1 = y_pred[:, 0]
    pred_min     = y_pred.min(axis=1)
    pred_max     = y_pred.max(axis=1)

    C_ACTUAL = "#1d4ed8"
    C_PRED   = "#d97706"   # amber — distinct from all prior models
    C_BAND   = "#fde68a"
    C_MID    = "#0891b2"
    C_LAST   = "#7c3aed"

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.48)

    # ── Panel 1: full period ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(idx_full, pred_min, pred_max,
                     alpha=0.22, color=C_BAND,
                     label=f"Forecast spread (step 1–{T_out})")
    ax1.plot(idx_full, actual_s1,    color=C_ACTUAL, linewidth=1.5,
             label="Actual", zorder=4)
    ax1.plot(idx_full, predicted_s1, color=C_PRED, linewidth=1.2,
             linestyle="--", alpha=0.88,
             label="CNN-BiLSTM predicted (step 1)", zorder=5)

    annotation = (
        f"Combined = {metrics['Combined']:.2f}%\n"
        f"MAPE     = {metrics['MAPE']:.2f}%\n"
        f"MAE%     = {metrics['MAE_pct']:.2f}%\n"
        f"RMSE%    = {metrics['RMSE_pct']:.2f}%\n"
        f"R²       = {metrics['R2']:.4f}\n"
        f"MAE      = {metrics['MAE']:.0f} riders\n"
        f"RMSE     = {metrics['RMSE']:.0f} riders"
    )
    ax1.text(
        0.01, 0.97, annotation,
        transform=ax1.transAxes, fontsize=8, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="#d1d5db", alpha=0.92),
    )
    ax1.set_title("CNN-BiLSTM — Test set: Actual vs Predicted (full period)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Test sample index")
    ax1.set_ylabel("Ridership (riders)")
    ax1.legend(loc="upper right", fontsize=8); ax1.grid(alpha=0.25)

    # ── Panel 2: zoomed ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(idx_zoom, pred_min[idx_zoom], pred_max[idx_zoom],
                     alpha=0.18, color=C_BAND)
    ax2.plot(idx_zoom, actual_s1[idx_zoom], color=C_ACTUAL,
             linewidth=1.7, label="Actual", zorder=5)

    steps_to_show = sorted({0, T_out // 2, T_out - 1})
    palette = [C_PRED, C_MID, C_LAST]
    styles  = ["--", "-.", ":"]
    for s, col, ls in zip(steps_to_show, palette, styles):
        ax2.plot(idx_zoom, y_pred[idx_zoom, s], color=col, linewidth=1.4,
                 linestyle=ls, alpha=0.88, label=f"Predicted step {s + 1}")

    ax2.set_title(
        f"Zoomed: last {zoom_n} samples — "
        f"step 1 / {T_out // 2 + 1} / {T_out} horizon comparison",
        fontsize=10, fontweight="bold",
    )
    ax2.set_xlabel("Test sample index")
    ax2.set_ylabel("Ridership (riders)")
    ax2.legend(loc="upper right", fontsize=8, ncol=2); ax2.grid(alpha=0.25)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_step_metrics(per_step: list, out_path: str) -> None:
    steps    = [f"t+{i+1}" for i in range(len(per_step))]
    combined = [m["Combined"]  for m in per_step]
    mape     = [m["MAPE"]      for m in per_step]
    mae_pct  = [m["MAE_pct"]   for m in per_step]
    rmse_pct = [m["RMSE_pct"]  for m in per_step]
    r2       = [m["R2"]        for m in per_step]

    x, w = np.arange(len(steps)), 0.2
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(8, len(steps) * 0.85), 7),
        gridspec_kw={"hspace": 0.48},
    )

    ax1.bar(x - 1.5*w, combined, w, label="Combined%", color="#16a34a", alpha=0.85)
    ax1.bar(x - 0.5*w, mape,     w, label="MAPE%",     color="#dc2626", alpha=0.85)
    ax1.bar(x + 0.5*w, mae_pct,  w, label="MAE%",      color="#2563eb", alpha=0.85)
    ax1.bar(x + 1.5*w, rmse_pct, w, label="RMSE%",     color="#f97316", alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(steps)
    ax1.set_ylabel("% of mean demand  /  score")
    ax1.set_title("CNN-BiLSTM — Per-horizon metrics", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    ax2.plot(steps, r2, marker="o", color="#d97706", linewidth=1.8, markersize=5)
    ax2.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax2.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax2.set_ylim(min(min(r2) - 0.05, -0.1), 1.08)
    ax2.set_ylabel("R²")
    ax2.set_title("CNN-BiLSTM — R² per horizon step", fontweight="bold")
    ax2.grid(alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


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
    train_still_falling = (norm_tr < -0.01)

    # ── Signal 4: Early stop ─────────────────────────────────────────────────
    early_stop = best_epoch < early_stop_frac * total

    # ── Val trend (informational — NOT used for verdict) ─────────────────────
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

    if train_still_falling and val_drift > 0.10:
        if verdict != "overfit":
            verdict = "overfit"
        notes.append(
            f"Training loss still declining in final {tail_n} epochs while val drifted up "
            f"+{val_drift_pct:.1f}% — train/val divergence detected."
        )

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

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CNNBiLSTMForecaster(
        n_features   = n_features,
        cnn_filters  = args.cnn_filters,
        cnn_layers   = args.cnn_layers,
        kernel_size  = args.cnn_kernel_size,
        hidden_size  = args.hidden,
        lstm_layers  = args.layers,
        T_out        = T_out,
        dropout      = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel      : CNNBiLSTMForecaster")
    print(f"  CNN      : filters={args.cnn_filters}  layers={args.cnn_layers}"
          f"  kernel={args.cnn_kernel_size}")
    print(f"  BiLSTM   : hidden={args.hidden} (per dir)  layers={args.layers}"
          f"  → head_in={args.hidden * 2}")
    print(f"  Head     : hidden×2={args.hidden * 2} → {T_out}")
    print(f"  In       : (batch, {T_in}, {n_features})")
    print(f"  Out      : (batch, {T_out})")
    print(f"  Params   : {n_params:,}")

    # ── Optimiser / loss ──────────────────────────────────────────────────────
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
        N, T     = y_pred_s.shape
        y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).reshape(N, T)
        y_true   = scaler_y.inverse_transform(y_true_s.reshape(-1, 1)).reshape(N, T)
    else:
        print("[WARN] scaler_y.pkl not found — metrics in scaled units.")
        y_pred, y_true = y_pred_s, y_true_s

    # ── Per-horizon metrics ───────────────────────────────────────────────────
    W = 10
    print(f"\n{'='*85}")
    print("CNN-BiLSTM — TEST SET METRICS")
    print(f"{'='*85}")
    header = (
        f"{'Step':>5}  "
        f"{'Combined%':>{W}}  {'MAPE%':>{W}}  {'MAE%':>{W}}  "
        f"{'RMSE%':>{W}}  {'R²':>{W}}  {'MAE':>{W}}  {'RMSE':>{W}}"
    )
    print(header)
    print("─" * len(header))

    per_step = []
    for s in range(T_out):
        m = compute_metrics(y_true[:, s], y_pred[:, s])
        per_step.append(m)
        print(
            f"{s+1:5d}  "
            f"{m['Combined']:>{W}.2f}  {m['MAPE']:>{W}.2f}  {m['MAE_pct']:>{W}.2f}  "
            f"{m['RMSE_pct']:>{W}.2f}  {m['R2']:>{W}.4f}  "
            f"{m['MAE']:>{W}.0f}  {m['RMSE']:>{W}.0f}"
        )

    overall = compute_metrics(y_true.flatten(), y_pred.flatten())
    print("─" * len(header))
    print(
        f"{'Avg':>5}  "
        f"{overall['Combined']:>{W}.2f}  {overall['MAPE']:>{W}.2f}  "
        f"{overall['MAE_pct']:>{W}.2f}  {overall['RMSE_pct']:>{W}.2f}  "
        f"{overall['R2']:>{W}.4f}  {overall['MAE']:>{W}.0f}  {overall['RMSE']:>{W}.0f}"
    )

    # ── Save artefacts ────────────────────────────────────────────────────────
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"src/outputs/cnn_bilstm/{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{out_dir}/model.pt")

    results = {
        "run_id": run_id,
        "model":  "CNNBiLSTMForecaster",
        "hparams": {
            "cnn_filters":     args.cnn_filters,
            "cnn_layers":      args.cnn_layers,
            "cnn_kernel_size": args.cnn_kernel_size,
            "hidden":          args.hidden,
            "lstm_layers":     args.layers,
            "dropout":         args.dropout,
            "bidirectional":   True,
            "T_in":            T_in,
            "T_out":           T_out,
            "n_features":      n_features,
            "batch_size":      args.batch_size,
            "lr":              args.lr,
            "weight_decay":    args.weight_decay,
            "loss":            args.loss,
            "warmup_epochs":   args.warmup_epochs,
        },
        "training": {
            "best_epoch":    best_epoch,
            "best_val_loss": round(best_val_loss, 8),
            "total_epochs":  len(train_losses),
        },
        "fit_diagnosis": fit_diag,
        "split_dates": split_meta,
        "test_metrics": {
            "overall":  {k: round(v, 4) for k, v in overall.items()},
            "per_step": [{k: round(v, 4) for k, v in m.items()} for m in per_step],
        },
    }

    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── CNN-BiLSTM plots ──────────────────────────────────────────────────────
    print(f"\nSaving plots...")
    plot_loss_curves(train_losses, val_losses,
                     out_path=f"{out_dir}/loss_curves.png")
    plot_predictions(y_true, y_pred, metrics=overall,
                     out_path=f"{out_dir}/test_predictions.png")
    plot_per_step_metrics(per_step,
                          out_path=f"{out_dir}/per_step_metrics.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*85}")
    print(f"CNN-BiLSTM run complete  →  {out_dir}/")
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
        ("ST-LSTM",    args.stlstm_results,     "src/outputs/st_lstm",    "#dc2626"),
        ("STGCN",      args.stgcn_results,      "src/outputs/stgcn",      "#10b981"),
        ("MTGNN",      args.mtgnn_results,      "src/outputs/mtgnn",      "#f472b6"),
        ("STSGCN",     args.stsgcn_results,     "src/outputs/stsgcn",     "#0ea5e9"),
        ("STFGNN",     args.stfgnn_results,     "src/outputs/stfgnn",     "#a855f7"),
        ("PDR-STGCN",   args.pdrstgcn_results,    "src/outputs/pdr_stgcn",   "#f97316"),
        ("ASTGCN",     args.astgcn_results,     "src/outputs/astgcn",     "#e11d48"),
        ("TFT",        args.tft_results,        "src/outputs/tft",        "#ca8a04"),
        ("Autoformer", args.autoformer_results, "src/outputs/autoformer", "#047857"),
        ("Informer",   args.informer_results,   "src/outputs/informer",   "#9333ea"),
    ]

    models_data = []
    comparison  = {}
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

    models_data.append(("CNN-BiLSTM", overall, per_step, "#d97706"))

    if len(models_data) > 1:
        print_comparison_table(models_data[:-1], overall, "CNN-BiLSTM")
        plot_comparison(models_data, out_path=f"{out_dir}/comparison_{len(models_data)}_way.png")

    comparison["cnn_bilstm"] = {"run_id": run_id, "overall": {k: round(v, 4) for k, v in overall.items()}}
    for name, m_overall, _, __ in models_data[:-1]:
        key = name.lower().replace("-", "_").replace(" ", "_")
        comparison[f"delta_vs_{key}"] = {k: round(overall.get(k, 0) - m_overall.get(k, 0), 4) for k in overall}
    results["comparison"] = comparison
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Naive persistence baseline ────────────────────────────────────────────
    target_idx   = split_meta.get("target_col_idx", 0)
    last_obs_s   = X_te[:, -1, target_idx : target_idx + 1].cpu().numpy()
    naive_pred_s = np.tile(last_obs_s, (1, T_out))

    if os.path.exists(scaler_y_path):
        N2, T2     = naive_pred_s.shape
        naive_pred = scaler_y.inverse_transform(
            naive_pred_s.reshape(-1, 1)
        ).reshape(N2, T2)
    else:
        naive_pred = naive_pred_s

    naive  = compute_metrics(y_true.flatten(), naive_pred.flatten())
    d_comb = overall["Combined"] - naive["Combined"]
    d_mape = naive["MAPE"] - overall["MAPE"]
    print(f"\n  Naive persistence    "
          f"Combined={naive['Combined']:.2f}%  MAPE={naive['MAPE']:.2f}%  "
          f"R²={naive['R2']:.4f}")
    print(f"  CNN-BiLSTM vs naive  "
          f"ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()
