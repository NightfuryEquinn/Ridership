"""
informer.py  — Informer — Fine-Tuned

Key Features:
  • ProbSparse self-attention in O(L log L): selects top-u queries by KL-divergence sparsity measure
  • Self-attention distilling halves sequence length after each encoder layer via Conv1d + MaxPool1d
  • Generative decoder generates all T_out steps in one forward pass with start token + zero-padding
  • With e_layers=3 and T_in=14: distilling produces 14 → 7 → 4; decoder input remains length 14

Tuned Hyperparameters:
  d_model  : 64   → 128   low Combined% → more capacity
  n_heads  : 4    → 8     proportional to d_model
  e_layers : 2    → 3     deeper encoder; more ProbSparse + distil steps
  d_ff     : 128  → 256   proportional to d_model; wider FFN
  dropout  : 0.10 → 0.15  moderate variance across lookbacks → mild regularisation

Architecture:
  Encoder:  [ProbSparseAttn + ConvLayer(distil)] × (e_layers-1) + [ProbSparseAttn]
  Decoder:  [FullAttn(self) + FullAttn(cross)] × d_layers
  Output:   last T_out rows → Linear(d_model, 1) → (B, T_out)

Hardware:
  GPU  : NVIDIA A100 (32 GB VRAM)
  RAM  : 32 GB
  Precision : AMP fp16 (GradScaler enabled)

References
----------
Song, Y., Luo, R., Zhou, T., Zhou, C., & Su, R. (2024). Graph attention Informer
for long-term traffic flow prediction under the impact of sports events. Sensors,
24(15), 4796.
DOI: https://doi.org/10.3390/s24154796
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
    p = argparse.ArgumentParser(description="Informer (tuned) forecaster")
    p.add_argument("--seq-dir",              default=None)
    p.add_argument("--d-model",              type=int,   default=256,
                   help="Transformer model dimension  [tuned: 128, base: 64]")
    p.add_argument("--n-heads",              type=int,   default=16,
                   help="Number of attention heads  [tuned: 8, base: 4]")
    p.add_argument("--e-layers",             type=int,   default=3,
                   help="Encoder layers  [tuned: 3, base: 2]")
    p.add_argument("--d-layers",             type=int,   default=1,
                   help="Decoder layers")
    p.add_argument("--d-ff",                 type=int,   default=512,
                   help="FFN inner dimension  [tuned: 256, base: 128]")
    p.add_argument("--factor",               type=int,   default=7,
                   help="ProbSparse top-k factor (k = factor * ceil(ln(L_K)))")
    p.add_argument("--dropout",              type=float, default=0.1,
                   help="Dropout  [tuned: 0.15, base: 0.10]")
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

class ProbSparseAttention(nn.Module):
    """
    ProbSparse Self-Attention (Zhou et al., 2021).

    For short sequences (L ≤ 14), u ≈ L so this gracefully degenerates to
    full attention without any approximation error.
    """

    def __init__(self, d_model: int, n_heads: int, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.factor  = factor
        self.scale   = 1.0 / math.sqrt(self.d_head)
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj= nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _prob_qk(self, Q, K, sample_k, n_top):
        B, H, L_Q, D = Q.shape
        L_K = K.shape[2]
        idx_sample = torch.randint(L_K, (B, H, sample_k), device=Q.device)
        K_sample   = K.gather(2, idx_sample.unsqueeze(-1).expand(-1, -1, -1, D))
        Q_K_sample = torch.einsum("bhld,bhsd->bhls", Q, K_sample)
        M          = Q_K_sample.max(-1).values - Q_K_sample.mean(-1)
        M_top_idx  = M.topk(n_top, dim=-1, sorted=False).indices
        Q_reduce   = Q.gather(2, M_top_idx.unsqueeze(-1).expand(-1, -1, -1, D))
        Q_K        = torch.einsum("bhld,bhsd->bhls", Q_reduce, K) * self.scale
        return Q_K, M_top_idx

    def forward(self, Q, K, V):
        B, L_Q, _ = Q.shape
        L_K       = K.shape[1]
        Q = self.q_proj(Q).reshape(B, L_Q, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K = self.k_proj(K).reshape(B, L_K, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        V = self.v_proj(V).reshape(B, L_K, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        sample_k = max(1, int(math.log(max(L_K, 2))))
        n_top    = min(L_Q, self.factor * max(1, int(math.ceil(math.log(max(L_K, 2))))))
        V_mean = V.mean(dim=2, keepdim=True).expand(-1, -1, L_Q, -1)
        out    = V_mean.clone()
        if n_top < L_Q:
            scores, top_idx = self._prob_qk(Q, K, sample_k, n_top)
            attn = self.dropout(torch.softmax(scores, dim=-1))
            v_top = torch.einsum("bhls,bhsd->bhld", attn, V)
            out.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head), v_top)
        else:
            scores = torch.einsum("bhld,bhsd->bhls", Q, K) * self.scale
            attn   = self.dropout(torch.softmax(scores, dim=-1))
            out    = torch.einsum("bhls,bhsd->bhld", attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, L_Q, -1)
        return self.dropout(self.out_proj(out))


class ConvLayer(nn.Module):
    """Informer distilling layer: Conv1d → ELU → MaxPool1d(2). Halves sequence length."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode="circular")
        self.norm    = nn.LayerNorm(d_model)
        self.act     = nn.ELU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv(x.permute(0, 2, 1)))
        x = self.maxpool(x)
        x = self.norm(x.permute(0, 2, 1))
        return x


class InformerEncoderLayer(nn.Module):
    """One Informer encoder layer: ProbSparseAttn + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, factor: int, dropout: float):
        super().__init__()
        self.attn  = ProbSparseAttention(d_model, n_heads, factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x, x, x)
        x        = self.norm1(x + self.drop(attn_out))
        ffn_out  = self.ffn(x)
        x        = self.norm2(x + self.drop(ffn_out))
        return x


class InformerDecoderLayer(nn.Module):
    """One Informer decoder layer: FullAttn(self) + FullAttn(cross) + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        sa_out, _ = self.self_attn(x, x, x)
        x         = self.norm1(x + self.drop(sa_out))
        ca_out, _ = self.cross_attn(x, enc_out, enc_out)
        x         = self.norm2(x + self.drop(ca_out))
        ffn_out   = self.ffn(x)
        x         = self.norm3(x + self.drop(ffn_out))
        return x


class InformerForecaster(nn.Module):
    """
    Informer forecaster.

    Input:  (B, T_in, n_features)
    Output: (B, T_out)

    Decoder start token = [X[:, -T_label:, :], zeros(B, T_out, F)]
    where T_label = T_in // 2.
    """

    def __init__(
        self,
        n_features: int,
        T_in:       int,
        T_out:      int,
        d_model:    int   = 128,
        n_heads:    int   = 8,
        e_layers:   int   = 3,
        d_layers:   int   = 1,
        d_ff:       int   = 256,
        factor:     int   = 5,
        dropout:    float = 0.15,
    ):
        super().__init__()
        self.T_out   = T_out
        self.T_label = T_in // 2

        self.enc_embed = nn.Linear(n_features, d_model)
        self.dec_embed = nn.Linear(n_features, d_model)

        self.register_buffer("enc_pe", self._sin_pe(T_in, d_model))
        T_dec = self.T_label + T_out
        self.register_buffer("dec_pe", self._sin_pe(T_dec, d_model))

        self.enc_layers  = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, factor, dropout)
            for _ in range(e_layers)
        ])
        self.conv_layers = nn.ModuleList([
            ConvLayer(d_model) for _ in range(e_layers - 1)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        self.dec_layers = nn.ModuleList([
            InformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(d_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)

        self.out_proj = nn.Linear(d_model, 1)

    @staticmethod
    def _sin_pe(length: int, d_model: int) -> torch.Tensor:
        pe  = torch.zeros(length, d_model)
        pos = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_in, F = x.shape
        T_out  = self.T_out
        T_label= self.T_label
        zeros   = torch.zeros(B, T_out, F, device=x.device, dtype=x.dtype)
        dec_in  = torch.cat([x[:, -T_label:, :], zeros], dim=1)
        enc = self.enc_embed(x) + self.enc_pe[:, :T_in, :]
        for i, enc_layer in enumerate(self.enc_layers):
            enc = enc_layer(enc)
            if i < len(self.conv_layers):
                enc = self.conv_layers[i](enc)
        enc = self.enc_norm(enc)
        T_dec = T_label + T_out
        dec   = self.dec_embed(dec_in) + self.dec_pe[:, :T_dec, :]
        for dec_layer in self.dec_layers:
            dec = dec_layer(dec, enc)
        dec = self.dec_norm(dec)
        out = self.out_proj(dec[:, -T_out:, :])
        return out.squeeze(-1)


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
    ax.plot(train_losses, label="Train Loss", linewidth=1.5, color="#9333ea")
    ax.plot(val_losses,   label="Val Loss",   linewidth=1.5, color="#f97316", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (scaled)")
    ax.set_title("Informer (tuned) — Training curves")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig); print(f"  Saved: {out_path}")


def plot_predictions(y_true, y_pred, metrics, out_path):
    N, T_out = y_true.shape; idx_full = np.arange(N)
    zoom_n = min(60, N); idx_zoom = np.arange(N - zoom_n, N)
    actual_s1 = y_true[:, 0]; predicted_s1 = y_pred[:, 0]
    pred_min = y_pred.min(axis=1); pred_max = y_pred.max(axis=1)
    C_ACT = "#1d4ed8"; C_PRED = "#9333ea"; C_BAND = "#f3e8ff"; C_MID = "#0891b2"; C_LAST = "#e11d48"
    fig = plt.figure(figsize=(14, 8)); gs = gridspec.GridSpec(2, 1, hspace=0.48)
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(idx_full, pred_min, pred_max, alpha=0.22, color=C_BAND,
                     label=f"Forecast spread (step 1–{T_out})")
    ax1.plot(idx_full, actual_s1,    color=C_ACT, linewidth=1.5, label="Actual", zorder=4)
    ax1.plot(idx_full, predicted_s1, color=C_PRED, linewidth=1.2, linestyle="--", alpha=0.88,
             label="Informer (tuned) predicted (step 1)", zorder=5)
    ann = (f"Combined = {metrics['Combined']:.2f}%\nMAPE     = {metrics['MAPE']:.2f}%\n"
           f"MAE%     = {metrics['MAE_pct']:.2f}%\nRMSE%    = {metrics['RMSE_pct']:.2f}%\n"
           f"R²       = {metrics['R2']:.4f}\nMAE      = {metrics['MAE']:.0f} riders\n"
           f"RMSE     = {metrics['RMSE']:.0f} riders")
    ax1.text(0.01, 0.97, ann, transform=ax1.transAxes, fontsize=8, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#d1d5db", alpha=0.92))
    ax1.set_title("Informer (tuned) — Test set: Actual vs Predicted (full period)", fontsize=11, fontweight="bold")
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
    ax1.set_title("Informer (tuned) — Per-horizon metrics", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)
    ax2.plot(steps, r2, marker="o", color="#9333ea", linewidth=1.8, markersize=5)
    ax2.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax2.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax2.set_ylim(min(min(r2) - 0.05, -0.1), 1.08)
    ax2.set_ylabel("R²"); ax2.set_title("Informer (tuned) — R² per horizon", fontweight="bold")
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

    model = InformerForecaster(
        n_features = n_features,
        T_in       = T_in,
        T_out      = T_out,
        d_model    = args.d_model,
        n_heads    = args.n_heads,
        e_layers   = args.e_layers,
        d_layers   = args.d_layers,
        d_ff       = args.d_ff,
        factor     = args.factor,
        dropout    = args.dropout,
    ).to(device)

    T_label = T_in // 2
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel       : InformerForecaster (tuned)")
    print(f"  d_model   : {args.d_model}   n_heads: {args.n_heads}   d_ff: {args.d_ff}")
    print(f"  e_layers  : {args.e_layers}   d_layers: {args.d_layers}   factor: {args.factor}")
    print(f"  T_label   : {T_label}   T_dec: {T_label + T_out}")
    print(f"  In        : (batch, {T_in}, {n_features})")
    print(f"  Dec input : (batch, {T_label + T_out}, {n_features})   Out: (batch, {T_out})")
    print(f"  Params    : {n_params:,}")

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

    W = 10; print(f"\n{'='*85}"); print("Informer (tuned) — TEST SET METRICS"); print(f"{'='*85}")
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

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S"); out_dir = f"src/outputs/informer_tuned/{run_id}"
    os.makedirs(out_dir, exist_ok=True); torch.save(model.state_dict(), f"{out_dir}/model.pt")
    results = {
        "run_id": run_id, "model": "InformerForecaster",
        "hparams": {
            "d_model": args.d_model, "n_heads": args.n_heads, "d_ff": args.d_ff,
            "e_layers": args.e_layers, "d_layers": args.d_layers, "factor": args.factor,
            "T_label": T_label, "dropout": args.dropout, "T_in": T_in, "T_out": T_out,
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

    print(f"\n{'='*85}"); print(f"Informer (tuned) run complete  →  {out_dir}/"); print(f"{'='*85}")
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
        ("ASTGCN",        args.astgcn_results,        "src/outputs/astgcn",         "#e11d48"),
        ("Autoformer",    args.autoformer_results,    "src/outputs/autoformer",     "#047857"),
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
    models_data.append(("Informer", overall, per_step, "#9333ea"))

    if len(models_data) > 1:
        print_comparison_table(models_data[:-1], overall, "Informer (tuned)")
        plot_comparison(models_data, out_path=f"{out_dir}/comparison_{len(models_data)}_way.png")

    comparison["informer"] = {"run_id": run_id, "overall": {k: round(v, 4) for k, v in overall.items()}}
    for name, m_overall, _, __ in models_data[:-1]:
        key = name.lower().replace("-", "_").replace(" ", "_")
        comparison[f"delta_vs_{key}"] = {k: round(overall.get(k, 0) - m_overall.get(k, 0), 4) for k in overall}
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
    d_r2   = overall["R2"] - naive["R2"]
    print(f"\n  Naive persistence   Combined={naive['Combined']:.2f}%  MAPE={naive['MAPE']:.2f}%  R²={naive['R2']:.4f}")
    print(f"  Informer (tuned) vs naive   ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%  ΔR²={d_r2:+.4f}")
    results["naive_persistence"] = {
        "metrics":        {k: round(v, 4) for k, v in naive.items()},
        "delta_combined": round(d_comb, 4),
        "delta_mape":     round(d_mape, 4),
        "delta_r2":       round(d_r2,   4),
    }
    results["comparison"] = comparison
    with open(f"{out_dir}/results.json", "w") as f: json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
