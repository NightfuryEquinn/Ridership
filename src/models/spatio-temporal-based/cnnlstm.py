"""
cnnlstm.py  — CNN-LSTM for Transit Ridership Forecasting

Mirrors bilstm.py and tpalstm.py in training loop, metrics, and plot style.
Supports two fusion modes via --mode:

  sequential (default)
    CNN extracts local temporal features → LSTM models dependencies across
    the CNN output sequence → MLP head on the final LSTM hidden state.
    The LSTM never sees raw features; it operates entirely on higher-level
    CNN representations.

    X → CNN → LSTM → MLP
    head_in = hidden

  parallel
    CNN and LSTM process the raw input independently and simultaneously.
    The CNN output is globally pooled to a fixed-size vector; the LSTM
    produces its final hidden state from the original feature sequence.
    Both vectors are concatenated before the MLP head, letting the model
    exploit local CNN patterns and global LSTM context without one branch
    constraining the other.

    X → CNN → global avg pool ─┐
                                ├─ cat → MLP
    X → LSTM → h_T            ─┘
    head_in = cnn_filters + hidden

  augmented
    Augmented Sequential CNN-LSTM.
    Retains the sequential CNN→LSTM hierarchy but adds a skip connection
    that globally pools the raw input and concatenates it with the LSTM
    final hidden state before the MLP head.  The LSTM still only sees
    higher-level CNN representations (not raw features), preserving the
    hierarchical abstraction of sequential, while the skip path restores
    direct access to low-level temporal signals that aggressive CNN
    filtering may suppress (e.g. absolute ridership level, rare spikes).

    X → CNN → LSTM → h_T ─────────────────┐
                                            ├─ cat → MLP
    X → global avg pool → skip_vec ────────┘
    head_in = hidden + n_features

Why three modes?
  Sequential is the classic stacked design — strong when CNN features are
  a better input to the LSTM than raw features (noisy, high-dimensional).
  Parallel preserves the original feature sequence for the LSTM branch,
  which can matter when raw temporal correlations (e.g. absolute ridership
  level at t−1) carry information that the CNN discards through its
  filters.  Augmented Sequential bridges both: the LSTM still benefits from
  CNN abstraction while the skip connection prevents information loss,
  typically yielding stronger generalisation than either alone.

Architecture — sequential:
  X             : (B, T_in, F)
  permute        → (B, F, T_in)
  Conv1d × L    → (B, cnn_filters, T_in)   # same-padding
  permute        → (B, T_in, cnn_filters)
  LSTM           → h_n[-1] : (B, hidden)
  MLP            → (B, T_out)

Architecture — parallel:
  X             : (B, T_in, F)
  permute        → (B, F, T_in)
  Conv1d × L    → (B, cnn_filters, T_in)   # same-padding
  mean(dim=-1)  → cnn_out : (B, cnn_filters)   # global avg pool
  LSTM(X)        → h_n[-1] : (B, hidden)
  cat            → (B, cnn_filters + hidden)
  MLP            → (B, T_out)

Architecture — augmented:
  X             : (B, T_in, F)
  permute        → (B, F, T_in)
  Conv1d × L    → (B, cnn_filters, T_in)   # same-padding
  permute        → (B, T_in, cnn_filters)
  LSTM           → h_n[-1] : (B, hidden)
  mean(dim=1, X) → skip    : (B, n_features) # global avg pool of raw input
  cat            → (B, hidden + n_features)
  MLP            → (B, T_out)

Each CNN block: Conv1d → BatchNorm1d → ReLU.

Comparison:
  --lstm-results     path/to/lstm/results.json
  --bilstm-results   path/to/bilstm/results.json
  --tpalstm-results  path/to/tpa_lstm/results.json
  All three are optional; each auto-detects the most recent run if omitted.

Usage:
  python cnnlstm.py                                    # sequential, auto-compare
  python cnnlstm.py --mode parallel                    # parallel mode
  python cnnlstm.py --mode augmented                   # augmented sequential mode
  python cnnlstm.py --cnn-filters 64 --cnn-layers 2 --cnn-kernel-size 3
  python cnnlstm.py --lstm-results src/outputs/lstm/<id>/results.json \\
                    --bilstm-results src/outputs/bilstm/<id>/results.json \\
                    --tpalstm-results src/outputs/tpa_lstm/<id>/results.json
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
    p = argparse.ArgumentParser(
        description="CNN-LSTM forecaster with LSTM / BiLSTM / TPA-LSTM comparison"
    )
    p.add_argument("--seq-dir",          default=None,
                   help="Directory with X/y .npy splits (shared with LSTM/BiLSTM/TPA-LSTM)")
    p.add_argument("--hidden",           type=int,   default=64,
                   help="LSTM hidden size")
    p.add_argument("--layers",           type=int,   default=1,
                   help="Stacked LSTM layers")
    p.add_argument("--dropout",          type=float, default=0.1,
                   help="Inter-layer LSTM dropout (active only when --layers > 1)")
    p.add_argument("--cnn-filters",      type=int,   default=32,
                   help="Number of CNN filters per convolutional layer")
    p.add_argument("--cnn-layers",       type=int,   default=2,
                   help="Number of stacked 1-D CNN blocks")
    p.add_argument("--cnn-kernel-size",  type=int,   default=3,
                   help="1-D CNN kernel size (same-padding applied)")
    p.add_argument("--mode",             default="sequential",
                   choices=["sequential", "parallel", "augmented"],
                   help=("sequential: CNN→LSTM→head; "
                         "parallel: CNN‖LSTM→head; "
                         "augmented: CNN→LSTM→head + raw skip connection"))
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
    p.add_argument("--tpalstm-results",   default=None,
                   help="Path to tpa_lstm results.json (auto-detected if omitted)")
    p.add_argument("--cnnbilstm-results", default=None)
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

class CNNLSTMForecaster(nn.Module):
    """
    CNN-LSTM with three fusion modes.

    Input:  (batch, T_in, n_features)
    Output: (batch, T_out)

    mode='sequential'  (default)
      CNN output (same-padded, preserves T_in) feeds directly into LSTM.
      LSTM input_size = cnn_filters.  head_in = hidden_size.

    mode='parallel'
      CNN and LSTM both receive the raw input X independently.
      CNN output is globally average-pooled → (B, cnn_filters).
      LSTM receives X directly → h_T: (B, hidden_size).
      Both are concatenated before the MLP head.
      LSTM input_size = n_features.  head_in = cnn_filters + hidden_size.

    mode='augmented'
      Augmented Sequential.
      CNN output feeds LSTM exactly as in sequential (LSTM never sees raw X).
      Additionally, raw X is globally average-pooled into a skip vector and
      concatenated with the LSTM's final hidden state before the MLP head.
      LSTM input_size = cnn_filters.  head_in = hidden_size + n_features.
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
        mode:         str   = "sequential",
    ):
        super().__init__()
        assert mode in ("sequential", "parallel", "augmented"), f"Unknown mode: {mode}"
        self.mode        = mode
        self.n_features  = n_features

        # ── CNN front-end (shared by both modes) ──────────────────────────────
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

        # ── LSTM encoder ──────────────────────────────────────────────────────
        # sequential/augmented: LSTM reads CNN features → input_size = cnn_filters
        # parallel:             LSTM reads raw features → input_size = n_features
        lstm_input = n_features if mode == "parallel" else cnn_filters
        self.lstm = nn.LSTM(
            input_size  = lstm_input,
            hidden_size = hidden_size,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )

        # ── MLP head ──────────────────────────────────────────────────────────
        # sequential: head_in = hidden_size
        # parallel:   head_in = cnn_filters + hidden_size
        # augmented:  head_in = hidden_size + n_features  (LSTM h_T + raw skip)
        if mode == "sequential":
            head_in = hidden_size
        elif mode == "parallel":
            head_in = cnn_filters + hidden_size
        else:  # augmented
            head_in = hidden_size + n_features
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
        # CNN branch — always runs
        x_cnn = x.permute(0, 2, 1)        # (B, F, T_in)
        x_cnn = self.cnn(x_cnn)           # (B, cnn_filters, T_in)

        if self.mode == "sequential":
            # Feed CNN output sequence into LSTM
            x_cnn = x_cnn.permute(0, 2, 1)        # (B, T_in, cnn_filters)
            _, (h_n, _) = self.lstm(x_cnn)
            h_T = h_n[-1]                          # (B, hidden)
            return self.head(h_T)

        elif self.mode == "augmented":
            # Same CNN→LSTM hierarchy as sequential …
            x_cnn = x_cnn.permute(0, 2, 1)        # (B, T_in, cnn_filters)
            _, (h_n, _) = self.lstm(x_cnn)
            h_T = h_n[-1]                          # (B, hidden)
            # … plus a skip connection: global avg pool of raw input
            skip = x.mean(dim=1)                   # (B, n_features)
            h_cat = torch.cat([h_T, skip], dim=-1) # (B, hidden + n_features)
            return self.head(h_cat)

        else:  # parallel
            # CNN branch: global average pool over time
            cnn_out = x_cnn.mean(dim=-1)           # (B, cnn_filters)

            # LSTM branch: reads raw input directly
            _, (h_n, _) = self.lstm(x)
            h_T = h_n[-1]                          # (B, hidden)

            # Fuse and predict
            h_cat = torch.cat([cnn_out, h_T], dim=-1)  # (B, cnn_filters + hidden)
            return self.head(h_cat)



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
    ax.plot(train_losses, label="Train Loss", linewidth=1.5, color="#16a34a")
    ax.plot(val_losses,   label="Val Loss",   linewidth=1.5, color="#f97316",
            linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (scaled)")
    ax.set_title("CNN-LSTM — Training curves")
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
    Two-panel overlay — identical layout to lstm.py, bilstm.py, tpalstm.py.
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
    C_PRED   = "#16a34a"   # green — distinct from LSTM blue, BiLSTM purple, TPA teal
    C_BAND   = "#bbf7d0"
    C_MID    = "#ea580c"
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
             label="CNN-LSTM predicted (step 1)", zorder=5)

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
    ax1.set_title("CNN-LSTM — Test set: Actual vs Predicted (full period)",
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
    ax1.set_title("CNN-LSTM — Per-horizon metrics", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    ax2.plot(steps, r2, marker="o", color="#16a34a", linewidth=1.8, markersize=5)
    ax2.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax2.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax2.set_ylim(min(min(r2) - 0.05, -0.1), 1.08)
    ax2.set_ylabel("R²")
    ax2.set_title("CNN-LSTM — R² per horizon step", fontweight="bold")
    ax2.grid(alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
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

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CNNLSTMForecaster(
        n_features   = n_features,
        cnn_filters  = args.cnn_filters,
        cnn_layers   = args.cnn_layers,
        kernel_size  = args.cnn_kernel_size,
        hidden_size  = args.hidden,
        lstm_layers  = args.layers,
        T_out        = T_out,
        dropout      = args.dropout,
        mode         = args.mode,
    ).to(device)

    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.mode == "sequential":
        head_in = args.hidden
        lstm_in = args.cnn_filters
    elif args.mode == "parallel":
        head_in = args.cnn_filters + args.hidden
        lstm_in = n_features
    else:  # augmented
        head_in = args.hidden + n_features
        lstm_in = args.cnn_filters
    print(f"\nModel     : CNNLSTMForecaster  [{args.mode}]")
    print(f"  CNN     : filters={args.cnn_filters}  layers={args.cnn_layers}"
          f"  kernel={args.cnn_kernel_size}")
    if args.mode == "sequential":
        print(f"  LSTM    : in={lstm_in}  hidden={args.hidden}  layers={args.layers}")
        print(f"  Head    : {args.hidden} → {T_out}")
    elif args.mode == "parallel":
        print(f"  LSTM    : in={lstm_in} (raw)  hidden={args.hidden}  layers={args.layers}")
        print(f"  Head    : cnn_pool({args.cnn_filters}) ‖ h_T({args.hidden})"
              f" = {head_in} → {T_out}")
    else:  # augmented
        print(f"  LSTM    : in={lstm_in} (CNN feats)  hidden={args.hidden}  layers={args.layers}")
        print(f"  Skip    : raw X global avg pool → ({n_features},)")
        print(f"  Head    : h_T({args.hidden}) ‖ skip({n_features})"
              f" = {head_in} → {T_out}")
    print(f"  In      : (batch, {T_in}, {n_features})")
    print(f"  Out     : (batch, {T_out})")
    print(f"  Params  : {n_params:,}")

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
    mode_label = f"CNN-LSTM [{args.mode}]"
    print(f"\n{'='*85}")
    print(f"{mode_label} — TEST SET METRICS")
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
    out_dir = f"src/outputs/cnn_lstm/{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{out_dir}/model.pt")

    results = {
        "run_id": run_id,
        "model":  "CNNLSTMForecaster",
        "hparams": {
            "mode":            args.mode,
            "cnn_filters":     args.cnn_filters,
            "cnn_layers":      args.cnn_layers,
            "cnn_kernel_size": args.cnn_kernel_size,
            "hidden":          args.hidden,
            "lstm_layers":     args.layers,
            "dropout":         args.dropout,
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
        "split_dates": split_meta,
        "test_metrics": {
            "overall":  {k: round(v, 4) for k, v in overall.items()},
            "per_step": [{k: round(v, 4) for k, v in m.items()} for m in per_step],
        },
    }

    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── CNN-LSTM plots ────────────────────────────────────────────────────────
    print(f"\nSaving plots...")
    plot_loss_curves(train_losses, val_losses,
                     out_path=f"{out_dir}/loss_curves.png")
    plot_predictions(y_true, y_pred, metrics=overall,
                     out_path=f"{out_dir}/test_predictions.png")
    plot_per_step_metrics(per_step,
                          out_path=f"{out_dir}/per_step_metrics.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*85}")
    print(f"{mode_label} run complete  →  {out_dir}/")
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
        ("CNN-BiLSTM", args.cnnbilstm_results,  "src/outputs/cnn_bilstm", "#d97706"),
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

    models_data.append(("CNN-LSTM", overall, per_step, "#16a34a"))

    if len(models_data) > 1:
        print_comparison_table(models_data[:-1], overall, "CNN-LSTM")
        plot_comparison(models_data, out_path=f"{out_dir}/comparison_{len(models_data)}_way.png")

    comparison["cnn_lstm"] = {"run_id": run_id, "overall": {k: round(v, 4) for k, v in overall.items()}}
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
    print(f"\n  Naive persistence  "
          f"Combined={naive['Combined']:.2f}%  MAPE={naive['MAPE']:.2f}%  "
          f"R²={naive['R2']:.4f}")
    print(f"  CNN-LSTM vs naive  "
          f"ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()