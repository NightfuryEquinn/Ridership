"""
bilstm.py  — Bidirectional LSTM for Transit Ridership Forecasting

Mirrors lstm_baseline.py exactly in training loop, metrics, and plot style.
The only architectural change is bidirectional=True in the LSTM cell, which
concatenates the forward and backward hidden states before the MLP head,
doubling the representational capacity at each layer.

Why BiLSTM over LSTM for this task?
  A standard LSTM at timestep t only sees tokens 0…t (causal).
  A BiLSTM also sees tokens t…T_in via a reversed pass, letting the model
  use later context in the look-back window when encoding timestep t.
  In practice this helps capture mid-window peaks/dips that a unidirectional
  model may underweight because it hasn't "seen ahead" yet.

  NOTE: BiLSTM is still valid for forecasting — the look-back window is
  already fully observed at inference time; bidirectionality applies only
  over the INPUT sequence, not into the future.

Architectural difference vs LSTM:
  LSTM   head input size = hidden_size
  BiLSTM head input size = hidden_size * 2   (forward ‖ backward concat)

Comparison:
  Pass --lstm-results src/outputs/lstm/<run_id>/results.json to print a
  side-by-side metric table and save a comparison bar chart.

Usage:
  python bilstm.py                                         # defaults
  python bilstm.py --hidden 64 --layers 2 --dropout 0.2
  python bilstm.py --lstm-results src/outputs/lstm/<id>/results.json
"""

import os
import json
import argparse
import glob
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


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="BiLSTM forecaster with LSTM comparison")
    p.add_argument("--seq-dir",       default="data/sequences/lstm",
                   help="Directory containing X/y .npy splits (shared with LSTM)")
    p.add_argument("--hidden",        type=int,   default=64)
    p.add_argument("--layers",        type=int,   default=1)
    p.add_argument("--dropout",       type=float, default=0.2,
                   help="Inter-layer dropout (only active when --layers > 1)")
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--device",        default="auto", help="cpu | cuda | mps | auto")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument(
        "--lstm-results",
        default=None,
        help=(
            "Path to a lstm_baseline results.json for side-by-side comparison. "
            "If omitted, the script searches src/outputs/lstm/**/results.json and "
            "uses the most recent run automatically."
        ),
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class BiLSTMForecaster(nn.Module):
    """
    Bidirectional LSTM → MLP head.

    Input  : (batch, T_in, n_features)
    Output : (batch, T_out)

    The final hidden state from the top layer is the concatenation of the
    forward direction (h_n[-2]) and the backward direction (h_n[-1]),
    giving a vector of size hidden_size * 2 that feeds the MLP head.

    For a 1-layer BiLSTM:
      h_n shape : (2, batch, hidden_size)   — indices 0=fwd, 1=bwd
    For L-layer BiLSTM:
      h_n shape : (2*L, batch, hidden_size) — last fwd = [-2], last bwd = [-1]
    """

    def __init__(
        self,
        n_features:  int,
        hidden_size: int,
        n_layers:    int,
        T_out:       int,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.bilstm = nn.LSTM(
            input_size   = n_features,
            hidden_size  = hidden_size,
            num_layers   = n_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if n_layers > 1 else 0.0,
        )

        # Input to head is hidden_size * 2  (forward ‖ backward)
        head_in = hidden_size * 2
        self.head = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.ReLU(),
            nn.Linear(head_in // 2, T_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.bilstm(x)
        # h_n : (num_layers * 2, batch, hidden_size)
        # Concatenate last forward and last backward hidden states
        h_fwd = h_n[-2]                        # (batch, hidden_size)
        h_bwd = h_n[-1]                        # (batch, hidden_size)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)   # (batch, hidden_size*2)
        return self.head(h_cat)                # (batch, T_out)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics  (identical to lstm_baseline.py — definitions from METRICS.md)
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Combined  = max(0, 100 − MAPE − MAE% − RMSE%)   higher is better, [0, 100]
    MAPE      = mean(|ŷ − y| / |y|) × 100
    MAE_pct   = (MAE / ȳ) × 100
    RMSE_pct  = (RMSE / ȳ) × 100
    R2        = 1 − SSR / SST
    MAE, RMSE : raw ridership counts (diagnostic)
    """
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    y_mean   = np.mean(y_true)
    abs_err  = np.abs(y_true - y_pred)
    sq_err   = (y_true - y_pred) ** 2

    mae_raw  = float(np.mean(abs_err))
    rmse_raw = float(np.sqrt(np.mean(sq_err)))
    mape     = float(np.mean(abs_err / (np.abs(y_true) + 1.0)) * 100)

    denom    = y_mean if y_mean > 0 else 1.0
    mae_pct  = float(mae_raw  / denom * 100)
    rmse_pct = float(rmse_raw / denom * 100)
    combined = float(max(0.0, 100.0 - mape - mae_pct - rmse_pct))

    ss_res = float(np.sum(sq_err))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "Combined":  combined,
        "MAPE":      mape,
        "MAE_pct":   mae_pct,
        "RMSE_pct":  rmse_pct,
        "R2":        r2,
        "MAE":       mae_raw,
        "RMSE":      rmse_raw,
    }


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
# Training helpers  (identical to lstm_baseline.py)
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
    ax.plot(train_losses, label="Train MSE", linewidth=1.5, color="#7c3aed")
    ax.plot(val_losses,   label="Val MSE",   linewidth=1.5, color="#f97316", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (scaled)")
    ax.set_title("BiLSTM — Training curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_predictions(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    metrics: dict,
    out_path: str,
) -> None:
    """
    Two-panel overlay — identical layout to lstm_baseline.py.
    Panel 1: full test period actual vs predicted (step 1) + spread band.
    Panel 2: zoomed last 60 samples with step 1 / mid / last horizon lines.
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
    C_PRED   = "#7c3aed"   # purple — visually distinct from LSTM red
    C_BAND   = "#ddd6fe"
    C_MID    = "#ea580c"
    C_LAST   = "#dc2626"

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.48)

    # ── Panel 1: full test period ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(idx_full, pred_min, pred_max,
                     alpha=0.22, color=C_BAND,
                     label=f"Forecast spread (step 1–{T_out})")
    ax1.plot(idx_full, actual_s1,    color=C_ACTUAL, linewidth=1.5,
             label="Actual", zorder=4)
    ax1.plot(idx_full, predicted_s1, color=C_PRED, linewidth=1.2,
             linestyle="--", alpha=0.88, label="BiLSTM predicted (step 1)", zorder=5)

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
    ax1.set_title("BiLSTM — Test set: Actual vs Predicted (full period)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Test sample index")
    ax1.set_ylabel("Ridership (riders)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.25)

    # ── Panel 2: zoomed ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(idx_zoom, pred_min[idx_zoom], pred_max[idx_zoom],
                     alpha=0.18, color=C_BAND)
    ax2.plot(idx_zoom, actual_s1[idx_zoom], color=C_ACTUAL, linewidth=1.7,
             label="Actual", zorder=5)

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
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    ax2.grid(alpha=0.25)

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
    ax1.set_title("BiLSTM — Per-horizon metrics", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    ax2.plot(steps, r2, marker="o", color="#7c3aed", linewidth=1.8, markersize=5)
    ax2.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax2.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax2.set_ylim(min(min(r2) - 0.05, -0.1), 1.08)
    ax2.set_ylabel("R²"); ax2.set_title("BiLSTM — R² per horizon step", fontweight="bold")
    ax2.grid(alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Comparison helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_lstm_results(lstm_results_path: str | None) -> dict | None:
    """
    Load LSTM results.json.  If path is None, auto-detect the most recent run
    under src/outputs/lstm/**/results.json.
    """
    if lstm_results_path:
        if not os.path.exists(lstm_results_path):
            print(f"[WARN] --lstm-results path not found: {lstm_results_path}")
            return None
        with open(lstm_results_path) as f:
            return json.load(f)

    # Auto-detect
    candidates = sorted(glob.glob("src/outputs/lstm/**/results.json", recursive=True))
    if not candidates:
        print("[INFO] No LSTM results.json found — skipping comparison.")
        return None

    path = candidates[-1]   # most recent alphabetically (run_id is a timestamp)
    print(f"[INFO] Auto-detected LSTM results: {path}")
    with open(path) as f:
        return json.load(f)


def print_comparison_table(lstm_m: dict, bi_m: dict) -> None:
    """
    Print a side-by-side table of overall test metrics for LSTM vs BiLSTM,
    with a Δ column (BiLSTM − LSTM, sign-adjusted so positive always = better).
    """
    METRICS_CFG = [
        # (key,        label,       higher_is_better)
        ("Combined",   "Combined%", True),
        ("MAPE",       "MAPE%",     False),
        ("MAE_pct",    "MAE%",      False),
        ("RMSE_pct",   "RMSE%",     False),
        ("R2",         "R²",        True),
        ("MAE",        "MAE",       False),
        ("RMSE",       "RMSE",      False),
    ]

    W = 12
    sep = "─" * (6 + W * 3 + 14)

    print(f"\n{'='*len(sep)}")
    print("LSTM  vs  BiLSTM — Overall Test Metrics Comparison")
    print(f"{'='*len(sep)}")
    print(f"{'Metric':<14} {'LSTM':>{W}} {'BiLSTM':>{W}} {'Δ (BiLSTM−LSTM)':>{W}}  Better")
    print(sep)

    for key, label, higher_better in METRICS_CFG:
        lstm_v  = lstm_m.get(key, float("nan"))
        bi_v    = bi_m.get(key, float("nan"))
        delta   = bi_v - lstm_v

        # For lower-is-better metrics, positive delta means BiLSTM is worse
        if higher_better:
            improved = delta > 0
        else:
            improved = delta < 0

        winner = "BiLSTM ✓" if improved else ("LSTM ✓" if delta != 0 else "Tie")

        # Format: Combined/MAPE/MAE%/RMSE% → 2 dp; R2 → 4 dp; MAE/RMSE → 0 dp
        if key in ("MAE", "RMSE"):
            fmt = ".0f"
        elif key == "R2":
            fmt = ".4f"
        else:
            fmt = ".2f"

        delta_sign = "+" if delta >= 0 else ""
        print(
            f"{label:<14} "
            f"{lstm_v:{W}{fmt}} "
            f"{bi_v:{W}{fmt}} "
            f"{delta_sign}{delta:{W}{fmt}}  "
            f"{winner}"
        )

    print(sep)


def plot_comparison(
    lstm_m:    dict,
    bi_m:      dict,
    lstm_ps:   list,   # per-step metrics for LSTM
    bi_ps:     list,   # per-step metrics for BiLSTM
    out_path:  str,
) -> None:
    """
    Four-panel comparison figure:
      [0] Overall percentage metrics — grouped bars (LSTM vs BiLSTM)
      [1] Overall R² — side-by-side bars
      [2] Combined% per horizon step — two lines
      [3] R² per horizon step — two lines
    """
    PCT_KEYS   = ["Combined", "MAPE", "MAE_pct", "RMSE_pct"]
    PCT_LABELS = ["Combined%", "MAPE%", "MAE%", "RMSE%"]
    C_LSTM  = "#2563eb"
    C_BI    = "#7c3aed"

    n_steps = len(bi_ps)
    steps   = [f"t+{i+1}" for i in range(n_steps)]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.48, wspace=0.35)

    # ── [0] Overall percentage metrics ───────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    x   = np.arange(len(PCT_KEYS))
    w   = 0.35
    lstm_vals = [lstm_m[k] for k in PCT_KEYS]
    bi_vals   = [bi_m[k]   for k in PCT_KEYS]

    bars_l = ax0.bar(x - w/2, lstm_vals, w, label="LSTM",   color=C_LSTM,  alpha=0.82)
    bars_b = ax0.bar(x + w/2, bi_vals,   w, label="BiLSTM", color=C_BI,    alpha=0.82)

    # Label each bar with its value
    for bar in list(bars_l) + list(bars_b):
        h = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                 f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    ax0.set_xticks(x); ax0.set_xticklabels(PCT_LABELS, fontsize=9)
    ax0.set_ylabel("% of mean demand  /  score")
    ax0.set_title("Overall — Percentage Metrics", fontweight="bold")
    ax0.legend(fontsize=8); ax0.grid(axis="y", alpha=0.3)

    # ── [1] Overall R² ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    r2_vals  = [lstm_m["R2"], bi_m["R2"]]
    r2_bars  = ax1.bar(["LSTM", "BiLSTM"], r2_vals,
                        color=[C_LSTM, C_BI], alpha=0.82, width=0.4)
    for bar in r2_bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                 f"{h:.4f}", ha="center", va="bottom", fontsize=9)
    ax1.set_ylim(0, min(1.12, max(r2_vals) * 1.15 + 0.05))
    ax1.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":", label="R²=1")
    ax1.set_ylabel("R²")
    ax1.set_title("Overall — R²", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    # ── [2] Combined% per horizon step ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    lstm_comb = [m["Combined"] for m in lstm_ps]
    bi_comb   = [m["Combined"] for m in bi_ps]
    ax2.plot(steps, lstm_comb, marker="o", color=C_LSTM,  linewidth=1.8,
             markersize=5, label="LSTM")
    ax2.plot(steps, bi_comb,  marker="s", color=C_BI,    linewidth=1.8,
             markersize=5, label="BiLSTM", linestyle="--")
    ax2.fill_between(steps, lstm_comb, bi_comb,
                     alpha=0.12, color="#a78bfa",
                     label="BiLSTM − LSTM gap")
    ax2.set_ylabel("Combined%")
    ax2.set_title("Combined% per Horizon Step", fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # ── [3] R² per horizon step ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    lstm_r2 = [m["R2"] for m in lstm_ps]
    bi_r2   = [m["R2"] for m in bi_ps]
    ax3.plot(steps, lstm_r2, marker="o", color=C_LSTM, linewidth=1.8,
             markersize=5, label="LSTM")
    ax3.plot(steps, bi_r2,  marker="s", color=C_BI,   linewidth=1.8,
             markersize=5, label="BiLSTM", linestyle="--")
    ax3.fill_between(steps, lstm_r2, bi_r2,
                     alpha=0.12, color="#a78bfa")
    ax3.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax3.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax3.set_ylabel("R²")
    ax3.set_title("R² per Horizon Step", fontweight="bold")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    fig.suptitle("LSTM vs BiLSTM — Test Set Comparison", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


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

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=args.batch_size)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=args.batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BiLSTMForecaster(
        n_features=n_features, hidden_size=args.hidden,
        n_layers=args.layers,  T_out=T_out, dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel  : BiLSTMForecaster  hidden={args.hidden}  layers={args.layers}")
    print(f"  In   : (batch, {T_in}, {n_features})")
    print(f"  Out  : (batch, {T_out})")
    print(f"  Head : hidden_size × 2 = {args.hidden * 2}  →  {T_out}")
    print(f"  Params : {n_params:,}")

    # ── Optimiser / loss ──────────────────────────────────────────────────────
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
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
        N, T     = y_pred_s.shape
        y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).reshape(N, T)
        y_true   = scaler_y.inverse_transform(y_true_s.reshape(-1, 1)).reshape(N, T)
    else:
        print("[WARN] scaler_y.pkl not found — metrics in scaled units.")
        y_pred, y_true = y_pred_s, y_true_s

    # ── Per-horizon metrics ───────────────────────────────────────────────────
    W = 10
    print(f"\n{'='*85}")
    print("BiLSTM — TEST SET METRICS")
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
    out_dir = f"src/outputs/bilstm/{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{out_dir}/model.pt")

    results = {
        "run_id": run_id,
        "model":  "BiLSTMForecaster",
        "hparams": {
            "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
            "bidirectional": True,
            "T_in": T_in, "T_out": T_out, "n_features": n_features,
            "batch_size": args.batch_size, "lr": args.lr,
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

    # ── BiLSTM-only plots ─────────────────────────────────────────────────────
    print(f"\nSaving plots...")
    plot_loss_curves(train_losses, val_losses,
                     out_path=f"{out_dir}/loss_curves.png")
    plot_predictions(y_true, y_pred, metrics=overall,
                     out_path=f"{out_dir}/test_predictions.png")
    plot_per_step_metrics(per_step,
                          out_path=f"{out_dir}/per_step_metrics.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*85}")
    print(f"BiLSTM run complete  →  {out_dir}/")
    print(f"{'='*85}")
    print(f"  Combined  : {overall['Combined']:.2f}%")
    print(f"  MAPE      : {overall['MAPE']:.2f}%")
    print(f"  MAE%      : {overall['MAE_pct']:.2f}%    (raw MAE  = {overall['MAE']:.0f} riders)")
    print(f"  RMSE%     : {overall['RMSE_pct']:.2f}%   (raw RMSE = {overall['RMSE']:.0f} riders)")
    print(f"  R²        : {overall['R2']:.4f}")

    # ── LSTM comparison ───────────────────────────────────────────────────────
    lstm_data = load_lstm_results(args.lstm_results)

    if lstm_data:
        lstm_overall  = lstm_data["test_metrics"]["overall"]
        lstm_per_step = lstm_data["test_metrics"]["per_step"]

        # Ensure per-step lists are the same length (trim to shorter if needed)
        min_steps  = min(len(lstm_per_step), len(per_step))
        lstm_ps_tr = lstm_per_step[:min_steps]
        bi_ps_tr   = per_step[:min_steps]

        print_comparison_table(lstm_overall, overall)

        plot_comparison(
            lstm_m   = lstm_overall,
            bi_m     = overall,
            lstm_ps  = lstm_ps_tr,
            bi_ps    = bi_ps_tr,
            out_path = f"{out_dir}/comparison_lstm_vs_bilstm.png",
        )

        # Embed comparison in results.json for reference
        results["comparison_vs_lstm"] = {
            "lstm_run_id":    lstm_data.get("run_id", "unknown"),
            "lstm_overall":   lstm_overall,
            "bilstm_overall": {k: round(v, 4) for k, v in overall.items()},
            "delta": {
                k: round(overall.get(k, 0) - lstm_overall.get(k, 0), 4)
                for k in overall
            },
        }
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(
            "\n[INFO] No LSTM results loaded — skipping comparison.\n"
            "       Re-run with: --lstm-results src/outputs/lstm/<run_id>/results.json"
        )

    # ── Naive persistence baseline ────────────────────────────────────────────
    target_idx   = split_meta.get("target_col_idx", 0)
    last_obs_s   = X_te[:, -1, target_idx : target_idx + 1].cpu().numpy()
    naive_pred_s = np.tile(last_obs_s, (1, T_out))

    if os.path.exists(scaler_y_path):
        N2, T2     = naive_pred_s.shape
        naive_pred = scaler_y.inverse_transform(naive_pred_s.reshape(-1, 1)).reshape(N2, T2)
    else:
        naive_pred = naive_pred_s

    naive  = compute_metrics(y_true.flatten(), naive_pred.flatten())
    d_comb = overall["Combined"] - naive["Combined"]
    d_mape = naive["MAPE"] - overall["MAPE"]
    print(f"\n  Naive persistence  "
          f"Combined={naive['Combined']:.2f}%  MAPE={naive['MAPE']:.2f}%  "
          f"R²={naive['R2']:.4f}")
    print(f"  BiLSTM vs naive    "
          f"ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()