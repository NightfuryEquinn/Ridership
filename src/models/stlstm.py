"""
stlstm.py  — Spatio-Temporal LSTM for Transit Ridership Forecasting

Mirrors all other model scripts in training loop, metrics, and plot style.

Why ST-LSTM?
  Plain LSTM, BiLSTM, TPA-LSTM, and the CNN-LSTM variants all treat the
  feature vector at each timestep as an opaque vector fed into a recurrent
  cell.  None of them explicitly model cross-feature interactions — i.e.
  how combinations of spatial attributes (population density, POI counts,
  GTFS accessibility, walking distance) interact with each other independent
  of the temporal ordering.

  ST-LSTM decouples this with two parallel streams:

  Temporal stream  (LSTM)
    Standard LSTM over the T_in look-back window.  Captures how the
    feature sequence evolves over time — the "when" dimension.
    Output: h_T  (B, hidden_t)

  Spatial stream  (shared MLP per timestep, then temporal pooling)
    A small MLP with shared weights is applied to each timestep's full
    feature vector, projecting it to a compact spatial embedding.
    The embeddings are mean-pooled over the time axis, collapsing
    temporal order and summarising "which cross-feature spatial patterns
    persist across the look-back window".
    Output: sp  (B, spatial_hidden)

  Fusion
    cat([h_T, sp])  →  MLP head  →  (B, T_out)

  The spatial stream is time-invariant (shared weights) and pooled, making
  it insensitive to the ordering of timesteps — that sensitivity is left
  entirely to the LSTM.  This clean separation lets each branch specialise:
  the LSTM learns temporal dynamics, the spatial encoder learns persistent
  cross-feature patterns.

Architecture (forward pass):
  X                 : (B, T_in, F)

  Temporal:
    LSTM(X)          → h_T : (B, hidden_t)

  Spatial:
    X.reshape(B*T, F) → Linear → ReLU → Linear → ReLU
                      → (B*T, spatial_hidden)
    reshape            → (B, T_in, spatial_hidden)
    mean(dim=1)        → sp : (B, spatial_hidden)

  Fusion:
    cat([h_T, sp])   → (B, hidden_t + spatial_hidden)
    MLP head         → (B, T_out)

Comparison:
  --lstm-results      path/to/lstm/results.json
  --bilstm-results    path/to/bilstm/results.json
  --tpalstm-results   path/to/tpa_lstm/results.json
  --cnnlstm-results   path/to/cnn_lstm/results.json
  --cnnbilstm-results path/to/cnn_bilstm/results.json
  All five optional; each auto-detects the most recent run if omitted.

Usage:
  python stlstm.py                               # defaults, auto-compare
  python stlstm.py --hidden 64 --spatial-hidden 32
  python stlstm.py --lstm-results src/outputs/lstm/<id>/results.json
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
        description="ST-LSTM forecaster with 6-way comparison"
    )
    p.add_argument("--seq-dir",           default="data/sequences/lstm",
                   help="Directory with X/y .npy splits")
    p.add_argument("--hidden",            type=int,   default=512,
                   help="LSTM hidden size (temporal stream)")
    p.add_argument("--layers",            type=int,   default=1,
                   help="Stacked LSTM layers")
    p.add_argument("--dropout",           type=float, default=0.2,
                   help="Inter-layer LSTM dropout (active only when --layers > 1)")
    p.add_argument("--spatial-hidden",    type=int,   default=256,
                   help="Spatial encoder hidden size (per-timestep MLP output dim)")
    p.add_argument("--batch-size",        type=int,   default=16)
    p.add_argument("--epochs",            type=int,   default=50)
    p.add_argument("--lr",                type=float, default=1e-3)
    p.add_argument("--patience",          type=int,   default=10)
    p.add_argument("--device",            default="auto", help="cpu | cuda | mps | auto")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--lstm-results",      default=None,
                   help="Path to lstm results.json (auto-detected if omitted)")
    p.add_argument("--bilstm-results",    default=None,
                   help="Path to bilstm results.json (auto-detected if omitted)")
    p.add_argument("--tpalstm-results",   default=None,
                   help="Path to tpa_lstm results.json (auto-detected if omitted)")
    p.add_argument("--cnnlstm-results",   default=None,
                   help="Path to cnn_lstm results.json (auto-detected if omitted)")
    p.add_argument("--cnnbilstm-results", default=None,
                   help="Path to cnn_bilstm results.json (auto-detected if omitted)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class SpatialEncoder(nn.Module):
    """
    Shared-weight MLP applied to each timestep's feature vector, then
    mean-pooled over the time axis.

    This stream is explicitly time-invariant: the same projection is applied
    at every t independently, so the result represents "what persistent
    cross-feature spatial patterns exist in this look-back window" rather
    than "when they occurred".

    Input:  (B, T_in, F)
    Output: (B, spatial_hidden)
    """

    def __init__(self, n_features: int, spatial_hidden: int):
        super().__init__()
        mid = max(spatial_hidden, n_features // 2)
        self.mlp = nn.Sequential(
            nn.Linear(n_features, mid),
            nn.ReLU(),
            nn.Linear(mid, spatial_hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        x_flat  = x.reshape(B * T, F)           # (B*T, F)
        sp      = self.mlp(x_flat)              # (B*T, spatial_hidden)
        sp      = sp.reshape(B, T, -1)          # (B, T, spatial_hidden)
        return sp.mean(dim=1)                   # (B, spatial_hidden) — pool over T


class STLSTMForecaster(nn.Module):
    """
    Spatio-Temporal LSTM: parallel temporal LSTM + spatial MLP encoder → MLP head.

    Input:  (batch, T_in, n_features)
    Output: (batch, T_out)

    head_in = hidden_size + spatial_hidden
    """

    def __init__(
        self,
        n_features:     int,
        hidden_size:    int,
        lstm_layers:    int,
        spatial_hidden: int,
        T_out:          int,
        dropout:        float = 0.0,
    ):
        super().__init__()

        # ── Temporal stream ───────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )

        # ── Spatial stream ────────────────────────────────────────────────────
        self.spatial_enc = SpatialEncoder(n_features, spatial_hidden)

        # ── MLP head ──────────────────────────────────────────────────────────
        head_in = hidden_size + spatial_hidden
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
        # Temporal branch
        _, (h_n, _) = self.lstm(x)
        h_T = h_n[-1]                           # (B, hidden)

        # Spatial branch
        sp = self.spatial_enc(x)                # (B, spatial_hidden)

        # Fuse and predict
        h_cat = torch.cat([h_T, sp], dim=-1)    # (B, hidden + spatial_hidden)
        return self.head(h_cat)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
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
    ax.plot(train_losses, label="Train MSE", linewidth=1.5, color="#dc2626")
    ax.plot(val_losses,   label="Val MSE",   linewidth=1.5, color="#f97316",
            linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (scaled)")
    ax.set_title("ST-LSTM — Training curves")
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
    N, T_out = y_true.shape
    idx_full = np.arange(N)
    zoom_n   = min(60, N)
    idx_zoom = np.arange(N - zoom_n, N)

    actual_s1    = y_true[:, 0]
    predicted_s1 = y_pred[:, 0]
    pred_min     = y_pred.min(axis=1)
    pred_max     = y_pred.max(axis=1)

    C_ACTUAL = "#1d4ed8"
    C_PRED   = "#dc2626"   # red — distinct from all prior models
    C_BAND   = "#fecaca"
    C_MID    = "#0891b2"
    C_LAST   = "#7c3aed"

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.48)

    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(idx_full, pred_min, pred_max,
                     alpha=0.22, color=C_BAND,
                     label=f"Forecast spread (step 1–{T_out})")
    ax1.plot(idx_full, actual_s1,    color=C_ACTUAL, linewidth=1.5,
             label="Actual", zorder=4)
    ax1.plot(idx_full, predicted_s1, color=C_PRED, linewidth=1.2,
             linestyle="--", alpha=0.88,
             label="ST-LSTM predicted (step 1)", zorder=5)

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
    ax1.set_title("ST-LSTM — Test set: Actual vs Predicted (full period)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Test sample index")
    ax1.set_ylabel("Ridership (riders)")
    ax1.legend(loc="upper right", fontsize=8); ax1.grid(alpha=0.25)

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
    ax1.set_title("ST-LSTM — Per-horizon metrics", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    ax2.plot(steps, r2, marker="o", color="#dc2626", linewidth=1.8, markersize=5)
    ax2.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax2.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax2.set_ylim(min(min(r2) - 0.05, -0.1), 1.08)
    ax2.set_ylabel("R²")
    ax2.set_title("ST-LSTM — R² per horizon step", fontweight="bold")
    ax2.grid(alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_six_way_comparison(models_data: list, out_path: str) -> None:
    """
    Six-panel comparison across all available models.

    models_data: list of (name, overall_metrics, per_step_list, color)
                 — None entries already filtered out by caller.

    [0,0] Overall % metrics   [0,1] Overall R²
    [1,0] Combined% per step  [1,1] R² per step
    [2,0] Raw MAE per step    [2,1] Raw RMSE per step
    """
    PCT_KEYS   = ["Combined", "MAPE", "MAE_pct", "RMSE_pct"]
    PCT_LABELS = ["Combined%", "MAPE%", "MAE%", "RMSE%"]

    n_steps  = min(len(d[2]) for d in models_data)
    steps    = [f"t+{i+1}" for i in range(n_steps)]
    n_models = len(models_data)
    w_bar    = max(0.10, 0.80 / n_models)
    offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * w_bar

    markers = ["o", "s", "^", "D", "v", "P"]
    styles  = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    fig = plt.figure(figsize=(16, 13))
    gs  = gridspec.GridSpec(3, 2, hspace=0.52, wspace=0.32)

    # ── [0,0] Overall % metrics ───────────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    x    = np.arange(len(PCT_KEYS))
    for (name, m, _, col), off in zip(models_data, offsets):
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
    names = [d[0] for d in models_data]
    r2s   = [d[1]["R2"] for d in models_data]
    cols  = [d[3] for d in models_data]
    bars  = ax01.bar(names, r2s, color=cols, alpha=0.82, width=0.4)
    for bar in bars:
        h = bar.get_height()
        ax01.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                  f"{h:.4f}", ha="center", va="bottom", fontsize=8)
    ax01.set_ylim(0, min(1.12, max(r2s) * 1.15 + 0.05))
    ax01.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":", label="R²=1")
    ax01.tick_params(axis="x", labelsize=7, rotation=15)
    ax01.set_ylabel("R²")
    ax01.set_title("Overall — R²", fontweight="bold")
    ax01.legend(fontsize=8); ax01.grid(axis="y", alpha=0.3)

    # ── [1,0] Combined% per step ──────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    for (name, _, ps, col), mk, ls in zip(models_data, markers, styles):
        vals = [m["Combined"] for m in ps[:n_steps]]
        ax10.plot(steps, vals, marker=mk, color=col, linewidth=1.8,
                  markersize=5, label=name, linestyle=ls)
    ax10.set_ylabel("Combined%")
    ax10.set_title("Combined% per Horizon Step", fontweight="bold")
    ax10.legend(fontsize=7); ax10.grid(alpha=0.3)

    # ── [1,1] R² per step ─────────────────────────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    for (name, _, ps, col), mk, ls in zip(models_data, markers, styles):
        vals = [m["R2"] for m in ps[:n_steps]]
        ax11.plot(steps, vals, marker=mk, color=col, linewidth=1.8,
                  markersize=5, label=name, linestyle=ls)
    ax11.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax11.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax11.set_ylabel("R²")
    ax11.set_title("R² per Horizon Step", fontweight="bold")
    ax11.legend(fontsize=7); ax11.grid(alpha=0.3)

    # ── [2,0] Raw MAE per step ────────────────────────────────────────────────
    ax20 = fig.add_subplot(gs[2, 0])
    for (name, _, ps, col), mk, ls in zip(models_data, markers, styles):
        vals = [m["MAE"] for m in ps[:n_steps]]
        ax20.plot(steps, vals, marker=mk, color=col, linewidth=1.8,
                  markersize=5, label=name, linestyle=ls)
    ax20.set_ylabel("MAE (riders)")
    ax20.set_title("Raw MAE per Horizon Step", fontweight="bold")
    ax20.legend(fontsize=7); ax20.grid(alpha=0.3)

    # ── [2,1] Raw RMSE per step ───────────────────────────────────────────────
    ax21 = fig.add_subplot(gs[2, 1])
    for (name, _, ps, col), mk, ls in zip(models_data, markers, styles):
        vals = [m["RMSE"] for m in ps[:n_steps]]
        ax21.plot(steps, vals, marker=mk, color=col, linewidth=1.8,
                  markersize=5, label=name, linestyle=ls)
    ax21.set_ylabel("RMSE (riders)")
    ax21.set_title("Raw RMSE per Horizon Step", fontweight="bold")
    ax21.legend(fontsize=7); ax21.grid(alpha=0.3)

    title = " vs ".join(d[0] for d in models_data)
    fig.suptitle(f"{title} — Test Set Comparison",
                 fontsize=11, fontweight="bold", y=1.01)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Comparison table
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(models_data: list, st_overall: dict) -> None:
    """
    models_data: list of (name, overall_metrics, per_step, color) for prior models.
    Prints a table with Δ columns showing ST-LSTM minus each baseline.
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

    n_prior = len(models_data)
    W       = 10
    sep_len = 16 + W * (1 + n_prior) + 18 * n_prior + 12
    sep     = "─" * sep_len

    label = f"{n_prior + 1}-Way"
    print(f"\n{'='*sep_len}")
    print(f"{label} Comparison — Overall Test Metrics")
    print(f"{'='*sep_len}")

    header = f"{'Metric':<14}"
    for name, _, __, ___ in models_data:
        header += f" {name:>{W}}"
    header += f" {'ST-LSTM':>{W}}"
    for name, _, __, ___ in models_data:
        header += f"  {f'Δ vs {name}'[:W]:>{W}}"
    header += "  Best"
    print(header)
    print(sep)

    for key, label, higher_better in METRICS_CFG:
        st_v = st_overall.get(key, float("nan"))
        fmt  = ".0f" if key in ("MAE", "RMSE") else (".4f" if key == "R2" else ".2f")

        row        = f"{label:<14}"
        candidates = {"ST-LSTM": st_v}

        for name, m, _, __ in models_data:
            v = m.get(key, float("nan"))
            candidates[name] = v
            row += f" {v:{W}{fmt}}"

        row += f" {st_v:{W}{fmt}}"

        for name, m, _, __ in models_data:
            d = st_v - m.get(key, 0)
            row += f"  {'+' if d >= 0 else ''}{d:{W}{fmt}}"

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
    model = STLSTMForecaster(
        n_features     = n_features,
        hidden_size    = args.hidden,
        lstm_layers    = args.layers,
        spatial_hidden = args.spatial_hidden,
        T_out          = T_out,
        dropout        = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_in  = args.hidden + args.spatial_hidden
    print(f"\nModel         : STLSTMForecaster")
    print(f"  Temporal    : LSTM  hidden={args.hidden}  layers={args.layers}")
    print(f"  Spatial     : MLP   spatial_hidden={args.spatial_hidden}"
          f"  mid={max(args.spatial_hidden, n_features // 2)}")
    print(f"  Head        : {head_in} → {T_out}")
    print(f"  In          : (batch, {T_in}, {n_features})")
    print(f"  Out         : (batch, {T_out})")
    print(f"  Params      : {n_params:,}")

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
    print("ST-LSTM — TEST SET METRICS")
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
    out_dir = f"src/outputs/st_lstm/{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{out_dir}/model.pt")

    results = {
        "run_id": run_id,
        "model":  "STLSTMForecaster",
        "hparams": {
            "hidden":          args.hidden,
            "lstm_layers":     args.layers,
            "dropout":         args.dropout,
            "spatial_hidden":  args.spatial_hidden,
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

    # ── ST-LSTM plots ─────────────────────────────────────────────────────────
    print(f"\nSaving plots...")
    plot_loss_curves(train_losses, val_losses,
                     out_path=f"{out_dir}/loss_curves.png")
    plot_predictions(y_true, y_pred, metrics=overall,
                     out_path=f"{out_dir}/test_predictions.png")
    plot_per_step_metrics(per_step,
                          out_path=f"{out_dir}/per_step_metrics.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*85}")
    print(f"ST-LSTM run complete  →  {out_dir}/")
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
    ]

    models_data = []   # (name, overall, per_step, color)
    comparison  = {}

    for name, path, model_dir, color in PRIOR_MODELS:
        key  = name.lower().replace("-", "_")
        data = load_model_results(path, model_dir, name)
        if data:
            m_overall = data["test_metrics"]["overall"]
            m_ps      = data["test_metrics"]["per_step"]
            models_data.append((name, m_overall, m_ps, color))
            comparison[key] = {
                "run_id":  data.get("run_id"),
                "overall": m_overall,
            }
        else:
            comparison[key] = {"run_id": None, "overall": None}

    # Always include ST-LSTM last
    models_data.append(("ST-LSTM", overall, per_step, "#dc2626"))

    if len(models_data) > 1:
        print_comparison_table(models_data[:-1], overall)

        plot_six_way_comparison(
            models_data,
            out_path=f"{out_dir}/comparison_{len(models_data)}_way.png",
        )

    # Embed deltas and comparison in results.json
    comparison["st_lstm"] = {
        "run_id":  run_id,
        "overall": {k: round(v, 4) for k, v in overall.items()},
    }
    for name, m_overall, _, __ in models_data[:-1]:
        key = name.lower().replace("-", "_")
        delta_key = f"delta_vs_{key}"
        comparison[delta_key] = {
            k: round(overall.get(k, 0) - m_overall.get(k, 0), 4)
            for k in overall
        }

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
    print(f"  ST-LSTM vs naive   "
          f"ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()
