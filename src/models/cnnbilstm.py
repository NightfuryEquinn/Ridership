"""
cnnbilstm.py  — CNN-BiLSTM for Transit Ridership Forecasting

Mirrors cnnlstm.py in training loop, metrics, and plot style.
Replaces the unidirectional LSTM encoder with a BiLSTM, giving the model
full bidirectional context over the CNN-extracted feature sequence.

Why CNN-BiLSTM over CNN-LSTM for this task?
  CNN-LSTM feeds CNN features into a unidirectional LSTM that only sees
  earlier timesteps when encoding each position.  Replacing it with a BiLSTM
  adds a reversed pass so the encoder sees the entire look-back window from
  both directions at every step.  For transit ridership the look-back window
  is fully observed at inference time, so bidirectionality is valid and
  beneficial: mid-window anomalies (e.g. a holiday spike on day 7 of a 14-day
  window) are encoded with both past and future context rather than only past.

  CNN-BiLSTM thus combines two complementary inductive biases:
    • CNN  — local pattern detection (short receptive field, translation-
             invariant, applied before recurrence)
    • BiLSTM — global sequential context (bidirectional over the full
               CNN-feature sequence)

  The head receives a richer representation than CNN-LSTM's h_T alone:
  cat([h_fwd, h_bwd]) of size hidden_size × 2 from the top BiLSTM layer.

Architecture (forward pass):
  X             : (B, T_in, F)
  permute        → (B, F, T_in)              # channels-first for Conv1d
  Conv1d × L    → (B, cnn_filters, T_in)    # same-padding preserves T_in
  permute        → (B, T_in, cnn_filters)    # back to sequence format
  BiLSTM         → h_n: (2×lstm_layers, B, hidden)
  h_fwd = h_n[-2]: (B, hidden)              # last layer, forward direction
  h_bwd = h_n[-1]: (B, hidden)              # last layer, backward direction
  h_cat          : (B, hidden × 2)
  MLP head      → (B, T_out)

Each CNN block: Conv1d → BatchNorm1d → ReLU.

Comparison:
  --lstm-results     path/to/lstm/results.json
  --bilstm-results   path/to/bilstm/results.json
  --tpalstm-results  path/to/tpa_lstm/results.json
  --cnnlstm-results  path/to/cnn_lstm/results.json
  All four optional; each auto-detects the most recent run if omitted.

Usage:
  python cnnbilstm.py                              # defaults, auto-compare
  python cnnbilstm.py --cnn-filters 64 --cnn-layers 2 --cnn-kernel-size 3
  python cnnbilstm.py --lstm-results src/outputs/lstm/<id>/results.json \\
                      --bilstm-results src/outputs/bilstm/<id>/results.json \\
                      --tpalstm-results src/outputs/tpa_lstm/<id>/results.json \\
                      --cnnlstm-results src/outputs/cnn_lstm/<id>/results.json
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
    p = argparse.ArgumentParser(
        description="CNN-BiLSTM forecaster with LSTM / BiLSTM / TPA-LSTM / CNN-LSTM comparison"
    )
    p.add_argument("--seq-dir",          default="data/sequences/lstm",
                   help="Directory with X/y .npy splits")
    p.add_argument("--hidden",           type=int,   default=512,
                   help="BiLSTM hidden size (per direction)")
    p.add_argument("--layers",           type=int,   default=1,
                   help="Stacked BiLSTM layers")
    p.add_argument("--dropout",          type=float, default=0.2,
                   help="Inter-layer BiLSTM dropout (active only when --layers > 1)")
    p.add_argument("--cnn-filters",      type=int,   default=128,
                   help="Number of CNN filters per convolutional layer")
    p.add_argument("--cnn-layers",       type=int,   default=3,
                   help="Number of stacked 1-D CNN blocks")
    p.add_argument("--cnn-kernel-size",  type=int,   default=3,
                   help="1-D CNN kernel size (same-padding applied)")
    p.add_argument("--batch-size",       type=int,   default=32)
    p.add_argument("--epochs",           type=int,   default=50)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--patience",         type=int,   default=10)
    p.add_argument("--device",           default="auto", help="cpu | cuda | mps | auto")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--lstm-results",     default=None,
                   help="Path to lstm results.json (auto-detected if omitted)")
    p.add_argument("--bilstm-results",   default=None,
                   help="Path to bilstm results.json (auto-detected if omitted)")
    p.add_argument("--tpalstm-results",  default=None,
                   help="Path to tpa_lstm results.json (auto-detected if omitted)")
    p.add_argument("--cnnlstm-results",  default=None,
                   help="Path to cnn_lstm results.json (auto-detected if omitted)")
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
# Metrics  (identical across all models — definitions from METRICS.md)
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
# Comparison loader helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_model_results(path: str | None, model_dir: str, label: str) -> dict | None:
    """
    Load a results.json from `path`.
    If path is None, auto-detect the most recent run under `model_dir`.
    """
    if path:
        if not os.path.exists(path):
            print(f"[WARN] {label} results not found at: {path}")
            return None
        with open(path) as f:
            return json.load(f)

    candidates = sorted(glob.glob(f"{model_dir}/**/results.json", recursive=True))
    if not candidates:
        print(f"[INFO] No {label} results.json found under {model_dir} — skipping.")
        return None

    detected = candidates[-1]
    print(f"[INFO] Auto-detected {label} results: {detected}")
    with open(detected) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(train_losses, val_losses, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train MSE", linewidth=1.5, color="#d97706")
    ax.plot(val_losses,   label="Val MSE",   linewidth=1.5, color="#f97316",
            linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (scaled)")
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


def plot_five_way_comparison(
    lstm_m:   dict | None,
    bi_m:     dict | None,
    tpa_m:    dict | None,
    cnnl_m:   dict | None,
    cnnbi_m:  dict,
    lstm_ps:  list | None,
    bi_ps:    list | None,
    tpa_ps:   list | None,
    cnnl_ps:  list | None,
    cnnbi_ps: list,
    out_path: str,
) -> None:
    """
    Six-panel 5-way comparison: LSTM / BiLSTM / TPA-LSTM / CNN-LSTM / CNN-BiLSTM.

    [0,0] Overall percentage metrics — grouped bars (up to 5 models)
    [0,1] Overall R²                 — side-by-side bars
    [1,0] Combined% per horizon step — lines per model
    [1,1] R² per horizon step        — lines per model
    [2,0] Raw MAE per step           — lines per model
    [2,1] Raw RMSE per step          — lines per model
    """
    C_LSTM  = "#2563eb"   # blue
    C_BI    = "#7c3aed"   # purple
    C_TPA   = "#0891b2"   # teal
    C_CNNL  = "#16a34a"   # green
    C_CNNBI = "#d97706"   # amber

    # Build model list dynamically (skip None prior models)
    models = []
    if lstm_m  is not None: models.append(("LSTM",       lstm_m,  lstm_ps,  C_LSTM))
    if bi_m    is not None: models.append(("BiLSTM",     bi_m,    bi_ps,    C_BI))
    if tpa_m   is not None: models.append(("TPA-LSTM",   tpa_m,   tpa_ps,   C_TPA))
    if cnnl_m  is not None: models.append(("CNN-LSTM",   cnnl_m,  cnnl_ps,  C_CNNL))
    models.append(                        ("CNN-BiLSTM", cnnbi_m, cnnbi_ps, C_CNNBI))

    PCT_KEYS   = ["Combined", "MAPE", "MAE_pct", "RMSE_pct"]
    PCT_LABELS = ["Combined%", "MAPE%", "MAE%", "RMSE%"]

    n_steps  = len(cnnbi_ps)
    steps    = [f"t+{i+1}" for i in range(n_steps)]

    n_models = len(models)
    w_bar    = 0.15
    offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * w_bar

    fig = plt.figure(figsize=(16, 13))
    gs  = gridspec.GridSpec(3, 2, hspace=0.52, wspace=0.32)

    # ── [0,0] Overall % metrics ───────────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    x    = np.arange(len(PCT_KEYS))
    for (name, m, _, col), off in zip(models, offsets):
        vals = [m[k] for k in PCT_KEYS]
        bars = ax00.bar(x + off, vals, w_bar, label=name, color=col, alpha=0.82)
        for bar in bars:
            h = bar.get_height()
            ax00.text(bar.get_x() + bar.get_width() / 2, h + 0.25,
                      f"{h:.1f}", ha="center", va="bottom", fontsize=5)
    ax00.set_xticks(x); ax00.set_xticklabels(PCT_LABELS, fontsize=9)
    ax00.set_ylabel("% of mean demand  /  score")
    ax00.set_title("Overall — Percentage Metrics", fontweight="bold")
    ax00.legend(fontsize=7); ax00.grid(axis="y", alpha=0.3)

    # ── [0,1] Overall R² ──────────────────────────────────────────────────────
    ax01  = fig.add_subplot(gs[0, 1])
    names = [m[0] for m in models]
    r2s   = [m[1]["R2"] for m in models]
    cols  = [m[3] for m in models]
    bars  = ax01.bar(names, r2s, color=cols, alpha=0.82, width=0.4)
    for bar in bars:
        h = bar.get_height()
        ax01.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                  f"{h:.4f}", ha="center", va="bottom", fontsize=8)
    ax01.set_ylim(0, min(1.12, max(r2s) * 1.15 + 0.05))
    ax01.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":", label="R²=1")
    ax01.tick_params(axis="x", labelsize=8)
    ax01.set_ylabel("R²")
    ax01.set_title("Overall — R²", fontweight="bold")
    ax01.legend(fontsize=8); ax01.grid(axis="y", alpha=0.3)

    # ── [1,0] Combined% per step ──────────────────────────────────────────────
    ax10    = fig.add_subplot(gs[1, 0])
    markers = ["o", "s", "^", "D", "v"]
    styles  = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for (name, _, ps, col), mk, ls in zip(models, markers, styles):
        if ps is None:
            continue
        vals = [m["Combined"] for m in ps[:n_steps]]
        ax10.plot(steps, vals, marker=mk, color=col, linewidth=1.8,
                  markersize=5, label=name, linestyle=ls)
    ax10.set_ylabel("Combined%")
    ax10.set_title("Combined% per Horizon Step", fontweight="bold")
    ax10.legend(fontsize=8); ax10.grid(alpha=0.3)

    # ── [1,1] R² per step ─────────────────────────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    for (name, _, ps, col), mk, ls in zip(models, markers, styles):
        if ps is None:
            continue
        vals = [m["R2"] for m in ps[:n_steps]]
        ax11.plot(steps, vals, marker=mk, color=col, linewidth=1.8,
                  markersize=5, label=name, linestyle=ls)
    ax11.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax11.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax11.set_ylabel("R²")
    ax11.set_title("R² per Horizon Step", fontweight="bold")
    ax11.legend(fontsize=8); ax11.grid(alpha=0.3)

    # ── [2,0] Raw MAE per step ────────────────────────────────────────────────
    ax20 = fig.add_subplot(gs[2, 0])
    for (name, _, ps, col), mk, ls in zip(models, markers, styles):
        if ps is None:
            continue
        vals = [m["MAE"] for m in ps[:n_steps]]
        ax20.plot(steps, vals, marker=mk, color=col, linewidth=1.8,
                  markersize=5, label=name, linestyle=ls)
    ax20.set_ylabel("MAE (riders)")
    ax20.set_title("Raw MAE per Horizon Step", fontweight="bold")
    ax20.legend(fontsize=8); ax20.grid(alpha=0.3)

    # ── [2,1] Raw RMSE per step ───────────────────────────────────────────────
    ax21 = fig.add_subplot(gs[2, 1])
    for (name, _, ps, col), mk, ls in zip(models, markers, styles):
        if ps is None:
            continue
        vals = [m["RMSE"] for m in ps[:n_steps]]
        ax21.plot(steps, vals, marker=mk, color=col, linewidth=1.8,
                  markersize=5, label=name, linestyle=ls)
    ax21.set_ylabel("RMSE (riders)")
    ax21.set_title("Raw RMSE per Horizon Step", fontweight="bold")
    ax21.legend(fontsize=8); ax21.grid(alpha=0.3)

    model_names = " vs ".join(m[0] for m in models)
    fig.suptitle(f"{model_names} — Test Set Comparison",
                 fontsize=12, fontweight="bold", y=1.01)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Comparison table
# ══════════════════════════════════════════════════════════════════════════════

def print_five_way_table(
    lstm_m:  dict | None,
    bi_m:    dict | None,
    tpa_m:   dict | None,
    cnnl_m:  dict | None,
    cnnbi_m: dict,
) -> None:
    """
    Side-by-side overall metric table for all available models.
    Δ columns show CNN-BiLSTM minus each baseline.
    Winner column names the best model for each metric.
    """
    METRICS_CFG = [
        ("Combined",  "Combined%", True),
        ("MAPE",      "MAPE%",     False),
        ("MAE_pct",   "MAE%",      False),
        ("RMSE_pct",  "RMSE%",     False),
        ("R2",        "R²",        True),
        ("MAE",       "MAE",       False),
        ("RMSE",      "RMSE",      False),
    ]

    has_lstm = lstm_m  is not None
    has_bi   = bi_m    is not None
    has_tpa  = tpa_m   is not None
    has_cnnl = cnnl_m  is not None

    W       = 10
    n_prior = has_lstm + has_bi + has_tpa + has_cnnl
    sep_len = 16 + W * (1 + n_prior) + 18 * n_prior + 12
    sep     = "─" * sep_len

    print(f"\n{'='*sep_len}")
    print("5-Way Comparison — Overall Test Metrics")
    print(f"{'='*sep_len}")

    header = f"{'Metric':<14}"
    if has_lstm: header += f" {'LSTM':>{W}}"
    if has_bi:   header += f" {'BiLSTM':>{W}}"
    if has_tpa:  header += f" {'TPA-LSTM':>{W}}"
    if has_cnnl: header += f" {'CNN-LSTM':>{W}}"
    header += f" {'CNN-BiLSTM':>{W}}"
    if has_lstm: header += f"  {'Δ vs LSTM':>{W}}"
    if has_bi:   header += f"  {'Δ vs BiLSTM':>{W}}"
    if has_tpa:  header += f"  {'Δ vs TPA':>{W}}"
    if has_cnnl: header += f"  {'Δ vs CNNL':>{W}}"
    header += "  Best"
    print(header)
    print(sep)

    for key, label, higher_better in METRICS_CFG:
        cnnbi_v = cnnbi_m.get(key, float("nan"))
        fmt     = ".0f" if key in ("MAE", "RMSE") else (".4f" if key == "R2" else ".2f")

        row        = f"{label:<14}"
        candidates = {"CNN-BiLSTM": cnnbi_v}

        if has_lstm:
            lv = lstm_m.get(key, float("nan"))
            candidates["LSTM"] = lv
            row += f" {lv:{W}{fmt}}"
        if has_bi:
            bv = bi_m.get(key, float("nan"))
            candidates["BiLSTM"] = bv
            row += f" {bv:{W}{fmt}}"
        if has_tpa:
            tv = tpa_m.get(key, float("nan"))
            candidates["TPA-LSTM"] = tv
            row += f" {tv:{W}{fmt}}"
        if has_cnnl:
            cv = cnnl_m.get(key, float("nan"))
            candidates["CNN-LSTM"] = cv
            row += f" {cv:{W}{fmt}}"

        row += f" {cnnbi_v:{W}{fmt}}"

        if has_lstm:
            d = cnnbi_v - lstm_m.get(key, 0)
            row += f"  {'+' if d>=0 else ''}{d:{W}{fmt}}"
        if has_bi:
            d = cnnbi_v - bi_m.get(key, 0)
            row += f"  {'+' if d>=0 else ''}{d:{W}{fmt}}"
        if has_tpa:
            d = cnnbi_v - tpa_m.get(key, 0)
            row += f"  {'+' if d>=0 else ''}{d:{W}{fmt}}"
        if has_cnnl:
            d = cnnbi_v - cnnl_m.get(key, 0)
            row += f"  {'+' if d>=0 else ''}{d:{W}{fmt}}"

        best = max(candidates, key=lambda k: candidates[k] if higher_better
                   else -candidates[k])
        row += f"  {best}"
        print(row)

    print(sep)


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

    # ── 5-way comparison ──────────────────────────────────────────────────────
    lstm_data = load_model_results(
        args.lstm_results, "src/outputs/lstm", "LSTM"
    )
    bi_data = load_model_results(
        args.bilstm_results, "src/outputs/bilstm", "BiLSTM"
    )
    tpa_data = load_model_results(
        args.tpalstm_results, "src/outputs/tpa_lstm", "TPA-LSTM"
    )
    cnnl_data = load_model_results(
        args.cnnlstm_results, "src/outputs/cnn_lstm", "CNN-LSTM"
    )

    lstm_m,  lstm_ps  = None, None
    bi_m,    bi_ps    = None, None
    tpa_m,   tpa_ps   = None, None
    cnnl_m,  cnnl_ps  = None, None

    if lstm_data:
        lstm_m  = lstm_data["test_metrics"]["overall"]
        lstm_ps = lstm_data["test_metrics"]["per_step"]
    if bi_data:
        bi_m  = bi_data["test_metrics"]["overall"]
        bi_ps = bi_data["test_metrics"]["per_step"]
    if tpa_data:
        tpa_m  = tpa_data["test_metrics"]["overall"]
        tpa_ps = tpa_data["test_metrics"]["per_step"]
    if cnnl_data:
        cnnl_m  = cnnl_data["test_metrics"]["overall"]
        cnnl_ps = cnnl_data["test_metrics"]["per_step"]

    print_five_way_table(lstm_m, bi_m, tpa_m, cnnl_m, overall)

    # Align per-step lengths across available models
    min_steps = len(per_step)
    if lstm_ps:  min_steps = min(min_steps, len(lstm_ps))
    if bi_ps:    min_steps = min(min_steps, len(bi_ps))
    if tpa_ps:   min_steps = min(min_steps, len(tpa_ps))
    if cnnl_ps:  min_steps = min(min_steps, len(cnnl_ps))

    plot_five_way_comparison(
        lstm_m   = lstm_m,
        bi_m     = bi_m,
        tpa_m    = tpa_m,
        cnnl_m   = cnnl_m,
        cnnbi_m  = overall,
        lstm_ps  = lstm_ps[:min_steps]  if lstm_ps  else None,
        bi_ps    = bi_ps[:min_steps]    if bi_ps    else None,
        tpa_ps   = tpa_ps[:min_steps]   if tpa_ps   else None,
        cnnl_ps  = cnnl_ps[:min_steps]  if cnnl_ps  else None,
        cnnbi_ps = per_step[:min_steps],
        out_path = f"{out_dir}/comparison_five_way.png",
    )

    # Embed comparison in results.json
    results["comparison"] = {
        "lstm":       {"run_id": lstm_data.get("run_id")  if lstm_data  else None,
                       "overall": lstm_m},
        "bilstm":     {"run_id": bi_data.get("run_id")    if bi_data    else None,
                       "overall": bi_m},
        "tpa_lstm":   {"run_id": tpa_data.get("run_id")   if tpa_data   else None,
                       "overall": tpa_m},
        "cnn_lstm":   {"run_id": cnnl_data.get("run_id")  if cnnl_data  else None,
                       "overall": cnnl_m},
        "cnn_bilstm": {"run_id": run_id,
                       "overall": {k: round(v, 4) for k, v in overall.items()}},
        "delta_vs_lstm": (
            {k: round(overall.get(k, 0) - lstm_m.get(k, 0), 4) for k in overall}
            if lstm_m else None
        ),
        "delta_vs_bilstm": (
            {k: round(overall.get(k, 0) - bi_m.get(k, 0), 4) for k in overall}
            if bi_m else None
        ),
        "delta_vs_tpalstm": (
            {k: round(overall.get(k, 0) - tpa_m.get(k, 0), 4) for k in overall}
            if tpa_m else None
        ),
        "delta_vs_cnnlstm": (
            {k: round(overall.get(k, 0) - cnnl_m.get(k, 0), 4) for k in overall}
            if cnnl_m else None
        ),
    }
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
