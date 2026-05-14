"""
lstm_baseline.py  — Baseline LSTM for Transit Ridership Forecasting

Loads the sequences produced by sequence_builder.py and trains a single-layer
LSTM, then evaluates on the held-out test set.

This is the baseline. It intentionally stays simple:
  - No attention, no graph convolution, no bidirectional layers
  - Standard MSE loss, Adam optimiser, ReduceLROnPlateau scheduler
  - Metrics: Combined%, MAE%, RMSE%, MAPE, R², raw MAE, raw RMSE

Metric definitions (from METRICS.md):
  MAPE      = mean(|ŷ - y| / |y|) × 100
  MAE%      = (MAE / ȳ) × 100           — MAE as % of mean demand
  RMSE%     = (RMSE / ȳ) × 100          — RMSE as % of mean demand
  Combined  = max(0, 100 − MAPE − MAE% − RMSE%)   [higher is better]
  R²        = 1 − SSR/SST

Use it to establish a performance floor before swapping in BiLSTM,
TPA-LSTM, or GCN-based variants.

Usage:
  python lstm_baseline.py                          # defaults
  python lstm_baseline.py --hidden 128 --layers 2  # tune
  python lstm_baseline.py --seq-dir data/sequences/lstm --epochs 100
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
matplotlib.use("Agg")   # headless — no display required
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
    p = argparse.ArgumentParser(description="Baseline LSTM for ridership forecasting")
    p.add_argument("--seq-dir",    default="data/sequences/lstm")
    p.add_argument("--hidden",     type=int,   default=64,  help="LSTM hidden size")
    p.add_argument("--layers",     type=int,   default=1,   help="LSTM stacked layers")
    p.add_argument("--dropout",    type=float, default=0.2, help="Dropout (needs --layers > 1)")
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int,   default=10,  help="Early-stopping patience")
    p.add_argument("--device",     default="auto",          help="cpu | cuda | mps | auto")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--bilstm-results",    default=None)
    p.add_argument("--tpalstm-results",   default=None)
    p.add_argument("--cnnlstm-results",   default=None)
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
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class LSTMForecaster(nn.Module):
    """
    Single or stacked LSTM → MLP head.

    Input:  (batch, T_in, n_features)
    Output: (batch, T_out)
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
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, T_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])   # top layer's final hidden → (batch, T_out)



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

def plot_loss_curves(train_losses: list, val_losses: list, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train MSE", linewidth=1.5, color="#2563eb")
    ax.plot(val_losses,   label="Val MSE",   linewidth=1.5, color="#f97316", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (scaled)")
    ax.set_title("LSTM Baseline — Training curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_predictions(
    y_true:   np.ndarray,   # (N_test, T_out) — original ridership scale
    y_pred:   np.ndarray,   # (N_test, T_out)
    metrics:  dict,
    out_path: str,
) -> None:
    """
    Two-panel overlay of actual vs predicted ridership on the test set.

    Panel 1 — Full test period
        Solid blue  : actual ridership (step-1 per window)
        Dashed red  : predicted ridership (step-1 per window)
        Shaded band : min–max of predicted values across all T_out steps,
                      giving a visual sense of the forecast spread

    Panel 2 — Zoomed: last 60 samples
        Draws step 1, the middle step, and the last step separately so
        horizon degradation is visible at close range.
    """
    N, T_out = y_true.shape
    idx_full = np.arange(N)
    zoom_n   = min(60, N)
    idx_zoom = np.arange(N - zoom_n, N)

    # Step-1 series (primary signal for the overlay)
    actual_s1    = y_true[:, 0]
    predicted_s1 = y_pred[:, 0]

    # Prediction spread band across all forecast steps
    pred_min = y_pred.min(axis=1)
    pred_max = y_pred.max(axis=1)

    # Colours
    C_ACTUAL = "#1d4ed8"
    C_PRED   = "#dc2626"
    C_BAND   = "#fca5a5"
    C_MID    = "#ea580c"
    C_LAST   = "#7c3aed"

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.48)

    # ── Panel 1: full test period ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    ax1.fill_between(
        idx_full, pred_min, pred_max,
        alpha=0.22, color=C_BAND,
        label=f"Forecast spread (step 1–{T_out})",
    )
    ax1.plot(idx_full, actual_s1,    color=C_ACTUAL, linewidth=1.5,
             label="Actual", zorder=4)
    ax1.plot(idx_full, predicted_s1, color=C_PRED,   linewidth=1.2,
             linestyle="--", alpha=0.88, label="Predicted (step 1)", zorder=5)

    # Metric annotation box (top-left)
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

    ax1.set_title(
        "LSTM Baseline — Test set: Actual vs Predicted (full period)",
        fontsize=11, fontweight="bold",
    )
    ax1.set_xlabel("Test sample index")
    ax1.set_ylabel("Ridership (riders)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.25)

    # ── Panel 2: zoomed — last 60 samples, multi-step overlay ────────────────
    ax2 = fig.add_subplot(gs[1])

    ax2.fill_between(
        idx_zoom, pred_min[idx_zoom], pred_max[idx_zoom],
        alpha=0.18, color=C_BAND,
    )
    ax2.plot(idx_zoom, actual_s1[idx_zoom],
             color=C_ACTUAL, linewidth=1.7, label="Actual", zorder=5)

    # Show step 1, middle, and last step to illustrate horizon degradation
    steps_to_show = sorted({0, T_out // 2, T_out - 1})
    palette       = [C_PRED, C_MID, C_LAST]
    styles        = ["--", "-.", ":"]

    for s, col, ls in zip(steps_to_show, palette, styles):
        ax2.plot(
            idx_zoom, y_pred[idx_zoom, s],
            color=col, linewidth=1.4, linestyle=ls, alpha=0.88,
            label=f"Predicted step {s + 1}",
        )

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
    """
    Bar chart of Combined%, MAPE, MAE%, RMSE% and a line plot of R²
    for each forecast horizon step.
    """
    steps    = [f"t+{i+1}" for i in range(len(per_step))]
    combined = [m["Combined"]  for m in per_step]
    mape     = [m["MAPE"]      for m in per_step]
    mae_pct  = [m["MAE_pct"]   for m in per_step]
    rmse_pct = [m["RMSE_pct"]  for m in per_step]
    r2       = [m["R2"]        for m in per_step]

    x  = np.arange(len(steps))
    w  = 0.2

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(8, len(steps) * 0.85), 7),
        gridspec_kw={"hspace": 0.48},
    )

    # Percentage metrics — grouped bars
    ax1.bar(x - 1.5*w, combined, w, label="Combined%", color="#16a34a", alpha=0.85)
    ax1.bar(x - 0.5*w, mape,     w, label="MAPE%",     color="#dc2626", alpha=0.85)
    ax1.bar(x + 0.5*w, mae_pct,  w, label="MAE%",      color="#2563eb", alpha=0.85)
    ax1.bar(x + 1.5*w, rmse_pct, w, label="RMSE%",     color="#f97316", alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(steps)
    ax1.set_ylabel("% of mean demand  /  score")
    ax1.set_title("Per-horizon metrics — percentage breakdown", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # R² per step — line plot
    ax2.plot(steps, r2, marker="o", color="#7c3aed", linewidth=1.8, markersize=5,
             label="R²")
    ax2.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax2.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":", label="R²=1 (perfect)")
    ax2.set_ylim(min(min(r2) - 0.05, -0.1), 1.08)
    ax2.set_ylabel("R²")
    ax2.set_title("R² per forecast horizon step", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

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
    model = LSTMForecaster(
        n_features=n_features, hidden_size=args.hidden,
        n_layers=args.layers, T_out=T_out, dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel : LSTMForecaster  hidden={args.hidden}  layers={args.layers}")
    print(f"  In  : (batch, {T_in}, {n_features})")
    print(f"  Out : (batch, {T_out})")
    print(f"  Params: {n_params:,}")

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

    y_pred_s = np.concatenate(preds_s)   # (N, T_out) — scaled
    y_true_s = np.concatenate(trues_s)

    # ── Inverse-transform to original ridership scale ─────────────────────────
    scaler_y_path = os.path.join(args.seq_dir, "scaler_y.pkl")
    if os.path.exists(scaler_y_path):
        scaler_y = joblib.load(scaler_y_path)
        N, T     = y_pred_s.shape
        y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).reshape(N, T)
        y_true   = scaler_y.inverse_transform(y_true_s.reshape(-1, 1)).reshape(N, T)
    else:
        print("[WARN] scaler_y.pkl not found — metrics reported in scaled units.")
        y_pred, y_true = y_pred_s, y_true_s

    # ── Per-horizon metrics ───────────────────────────────────────────────────
    W = 10   # column width
    print(f"\n{'='*85}")
    print("TEST SET METRICS")
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
    out_dir = f"src/outputs/lstm/{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{out_dir}/model.pt")

    results = {
        "run_id": run_id,
        "model":  "LSTMForecaster",
        "hparams": {
            "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
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

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\nSaving plots...")
    plot_loss_curves(
        train_losses, val_losses,
        out_path=f"{out_dir}/loss_curves.png",
    )
    plot_predictions(
        y_true, y_pred,
        metrics=overall,
        out_path=f"{out_dir}/test_predictions.png",
    )
    plot_per_step_metrics(
        per_step,
        out_path=f"{out_dir}/per_step_metrics.png",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*85}")
    print(f"Run complete  →  {out_dir}/")
    print(f"{'='*85}")
    print(f"  Combined  : {overall['Combined']:.2f}%   (higher is better, max 100)")
    print(f"  MAPE      : {overall['MAPE']:.2f}%")
    print(f"  MAE%      : {overall['MAE_pct']:.2f}%    (raw MAE  = {overall['MAE']:.0f} riders)")
    print(f"  RMSE%     : {overall['RMSE_pct']:.2f}%   (raw RMSE = {overall['RMSE']:.0f} riders)")
    print(f"  R²        : {overall['R2']:.4f}")

    # ── Multi-way comparison ──────────────────────────────────────────────────
    PRIOR_MODELS = [
        ("BiLSTM",     args.bilstm_results,     "src/outputs/bilstm",     "#7c3aed"),
        ("TPA-LSTM",   args.tpalstm_results,    "src/outputs/tpa_lstm",   "#0891b2"),
        ("CNN-LSTM",   args.cnnlstm_results,    "src/outputs/cnn_lstm",   "#16a34a"),
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

    models_data_cmp = []
    comparison      = {}
    for name, path, model_dir, color in PRIOR_MODELS:
        key  = name.lower().replace("-", "_").replace(" ", "_")
        data = load_model_results(path, model_dir, name)
        if data:
            m_overall = data["test_metrics"]["overall"]
            m_ps      = data["test_metrics"]["per_step"]
            models_data_cmp.append((name, m_overall, m_ps, color))
            comparison[key] = {"run_id": data.get("run_id"), "overall": m_overall}
        else:
            comparison[key] = {"run_id": None, "overall": None}

    models_data_cmp.insert(0, ("LSTM", overall, per_step, "#2563eb"))

    if len(models_data_cmp) > 1:
        print_comparison_table(models_data_cmp[1:], overall, "LSTM")
        plot_comparison(models_data_cmp,
                        out_path=f"{out_dir}/comparison_{len(models_data_cmp)}_way.png")

    comparison["lstm"] = {"run_id": run_id, "overall": {k: round(v, 4) for k, v in overall.items()}}
    for name, m_overall, _, __ in models_data_cmp[1:]:
        key = name.lower().replace("-", "_").replace(" ", "_")
        comparison[f"delta_vs_{key}"] = {k: round(overall.get(k, 0) - m_overall.get(k, 0), 4) for k in overall}
    results["comparison"] = comparison
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Naive persistence baseline comparison ─────────────────────────────────
    # "Predict tomorrow = last observed value" — the LSTM must beat this
    # to demonstrate it has actually learnt temporal patterns.
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
    print(f"  LSTM vs naive      "
          f"ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()