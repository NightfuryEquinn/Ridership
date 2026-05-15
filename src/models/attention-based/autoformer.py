"""
autoformer.py  — Autoformer for Transit Ridership Forecasting

Key ideas:
  1. Series Decomposition: X = MovingAvg(X) [trend] + (X − MovingAvg(X)) [seasonal]
     Applied as a learnable building block throughout encoder and decoder.

  2. Auto-Correlation Mechanism (replaces standard softmax attention):
     Instead of pairwise query-key dot-products, compute the time-delayed
     autocorrelation of queries with keys via FFT in O(L log L):
       corr(τ) = IFFT(FFT(Q) · conj(FFT(K)))
     Select the top-k lags, roll V by each lag, and aggregate with
     softmax weights proportional to corr(τ).
     This discovers periodic sub-series dependencies rather than
     point-wise token similarity.

  3. Encoder-Decoder:
     Encoder: L_e layers of [AutoCorr + Decomp + FFN + Decomp]
     Decoder: L_d layers of [AutoCorr + CrossCorr + Decomp + FFN + Decomp]
              with accumulated trend from each decomposition.

Adaptation for T_in=14, T_out=7:
  Decoder start token = last T_out steps of encoder input (seasonal) +
  zeros(T_out) for the unknown future → length = 2 * T_out = T_dec.
  Trend init = last T_out steps of encoder input (trend component) +
  zeros(T_out). The decoder accumulates trend across layers; seasonal
  is predicted as the residual.

  Output = trend_accum[:, -T_out:, :] + seasonal_dec[:, -T_out:, :]
         → Linear(n_features, 1) per timestep → (B, T_out)

Hardware optimisations (RTX 4050 6 GB, 32 GB RAM, i5):
  • AMP (fp16) for forward pass and gradient computation.
  • GradScaler for numerically stable fp16 training.
  • Conservative defaults: d_model=64, e_layers=2, d_layers=1.
  • FFT operations are AMP-compatible on CUDA.

Comparison (14-way):
  All 11 prior + ASTGCN + TFT, auto-detected from output directories.

Usage:
  python autoformer.py
  python autoformer.py --d-model 128 --n-heads 8 --e-layers 2 --d-layers 1
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
    p = argparse.ArgumentParser(description="Autoformer forecaster — 14-way comparison")
    p.add_argument("--seq-dir",              default="data/sequences/lstm")
    p.add_argument("--d-model",              type=int,   default=64,
                   help="Transformer model dimension")
    p.add_argument("--n-heads",              type=int,   default=4,
                   help="Attention heads")
    p.add_argument("--e-layers",             type=int,   default=2,
                   help="Encoder layers")
    p.add_argument("--d-layers",             type=int,   default=1,
                   help="Decoder layers")
    p.add_argument("--d-ff",                 type=int,   default=128,
                   help="FFN inner dimension")
    p.add_argument("--moving-avg",           type=int,   default=5,
                   help="Moving average kernel size for series decomposition")
    p.add_argument("--factor",               type=int,   default=3,
                   help="Auto-correlation top-k factor (k = factor * log(L))")
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
    p.add_argument("--tft-results",          default=None)
    p.add_argument("--informer-results",     default=None)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Model components
# ══════════════════════════════════════════════════════════════════════════════

class MovingAvg(nn.Module):
    """
    Learnable (but structurally fixed) moving average for trend extraction.

    Uses AvgPool1d with padding to ensure the output has the same length
    as the input. The padding repeats the first/last values.
    """

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        # Pad front and back by (kernel_size-1)//2 and kernel_size-1-(kernel_size-1)//2
        front = x[:, :1, :].expand(-1, (self.kernel_size - 1) // 2, -1)
        end   = x[:, -1:, :].expand(-1, self.kernel_size - 1 - (self.kernel_size - 1) // 2, -1)
        x_pad = torch.cat([front, x, end], dim=1)   # (B, L + kernel_size - 1, C)
        # Pool over time axis: AvgPool1d expects (B, C, L)
        trend = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)   # (B, L, C)
        return trend


class SeriesDecomp(nn.Module):
    """
    Series decomposition block.

    Returns (seasonal, trend) where:
        trend    = MovingAvg(x)
        seasonal = x - trend
    """

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        trend    = self.moving_avg(x)   # (B, L, C)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelation(nn.Module):
    """
    Auto-Correlation attention mechanism (Wu et al., 2021).

    Replaces standard dot-product attention with time-delay correlation:
      1. Project Q, K, V
      2. FFT Q and K → frequency domain
      3. Correlation: S(τ) = IFFT(FFT(Q) · conj(FFT(K)))
      4. Select top-k delays by mean correlation magnitude
      5. Roll V by each selected delay and aggregate with softmax weights

    Complexity: O(L log L) instead of O(L²).
    """

    def __init__(self, d_model: int, n_heads: int, factor: int = 3, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.factor  = factor

        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj= nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _time_delay_agg(self, V: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        Select top-k delays from correlation and aggregate V.

        V    : (B, H, L, D)
        corr : (B, H, L, D) — time-domain correlation
        """
        B, H, L, D = V.shape
        top_k = max(1, int(self.factor * math.log(max(L, 2))))
        top_k = min(top_k, L)

        # Mean correlation over heads and d_head dimensions → (B, L)
        mean_corr = corr.abs().mean(dim=(1, 3))           # (B, L)
        weights, delays = torch.topk(mean_corr, top_k, dim=-1)  # (B, k)
        weights = torch.softmax(weights, dim=-1)          # (B, k)

        out = torch.zeros_like(V)                         # (B, H, L, D)
        for k_idx in range(top_k):
            delay_k = delays[:, k_idx]                    # (B,)
            w_k     = weights[:, k_idx].reshape(B, 1, 1, 1)  # (B,1,1,1)
            # Batch-wise roll: shift each sample by its delay
            rolled = torch.stack([
                torch.roll(V[b], shifts=-int(delay_k[b].item()), dims=1)
                for b in range(B)
            ], dim=0)                                     # (B, H, L, D)
            out = out + w_k * rolled
        return out

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        # Q: (B, L_q, d_model)   K, V: (B, L_kv, d_model)
        # L_q == L_kv for self-attention; may differ for cross-attention.
        B, L_q,  _ = Q.shape
        _, L_kv, _ = K.shape

        Q = self.q_proj(Q).reshape(B, L_q,  self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K = self.k_proj(K).reshape(B, L_kv, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        V = self.v_proj(V).reshape(B, L_kv, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        # Q: (B, H, L_q, D)   K, V: (B, H, L_kv, D)

        # FFT-based cross-correlation
        # Cast to float32 for FFT stability under AMP.
        # Use nfft = max(L_q, L_kv) so both sequences fit, then trim to L_q.
        Q_f = Q.float(); K_f = K.float(); V_f = V.float()
        nfft  = max(L_q, L_kv)
        Q_fft = torch.fft.rfft(Q_f, n=nfft, dim=2, norm="ortho")   # (B, H, nfft//2+1, D)
        K_fft = torch.fft.rfft(K_f, n=nfft, dim=2, norm="ortho")
        corr_fft = Q_fft * torch.conj(K_fft)
        corr = torch.fft.irfft(corr_fft, n=nfft, dim=2, norm="ortho")  # (B, H, nfft, D)
        corr = corr[:, :, :L_q, :]                                  # (B, H, L_q, D)

        # Resample V to L_q when lengths differ (cross-attention case)
        if L_kv != L_q:
            V_f = F.adaptive_avg_pool1d(
                V_f.reshape(B * self.n_heads, L_kv, self.d_head).permute(0, 2, 1),
                L_q,
            ).permute(0, 2, 1).reshape(B, self.n_heads, L_q, self.d_head)

        # Time-delay aggregation
        out = self._time_delay_agg(V_f, corr)              # (B, H, L_q, D)
        out = out.to(Q.dtype)                               # back to original dtype

        # Reshape
        out = out.permute(0, 2, 1, 3).reshape(B, L_q, -1)  # (B, L_q, d_model)
        return self.dropout(self.out_proj(out))


class AutoformerEncoderLayer(nn.Module):
    """
    One Autoformer encoder layer:
        h → AutoCorr → Decomp(h + auto_out) → FFN → Decomp(h + ffn_out)
    Only the seasonal part (residual) is passed forward; trend is discarded
    within the encoder (no accumulation there).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, factor: int,
                 moving_avg: int, dropout: float):
        super().__init__()
        self.auto_corr  = AutoCorrelation(d_model, n_heads, factor, dropout)
        self.decomp1    = SeriesDecomp(moving_avg)
        self.ffn        = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.decomp2    = SeriesDecomp(moving_avg)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.drop       = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: (B, L, d_model)
        auto_out         = self.auto_corr(x, x, x)
        seasonal1, _     = self.decomp1(self.norm1(x + self.drop(auto_out)))
        ffn_out          = self.ffn(seasonal1)
        seasonal2, _     = self.decomp2(self.norm2(seasonal1 + self.drop(ffn_out)))
        return seasonal2


class AutoformerDecoderLayer(nn.Module):
    """
    One Autoformer decoder layer:
        x_dec, x_enc →
          AutoCorr(self)  → Decomp → trend_1
          CrossCorr(enc)  → Decomp → trend_2
          FFN             → Decomp → trend_3
        Returns (updated seasonal, accumulated_trend_delta)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, factor: int,
                 moving_avg: int, n_features: int, dropout: float):
        super().__init__()
        self.self_attn   = AutoCorrelation(d_model, n_heads, factor, dropout)
        self.cross_attn  = AutoCorrelation(d_model, n_heads, factor, dropout)
        self.decomp1     = SeriesDecomp(moving_avg)
        self.decomp2     = SeriesDecomp(moving_avg)
        self.decomp3     = SeriesDecomp(moving_avg)
        self.ffn         = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.norm3  = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)
        # Project accumulated trend residuals to feature space for output addition
        self.trend_proj = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor):
        # x       : (B, L_dec, d_model)
        # enc_out : (B, L_enc, d_model)
        sa_out         = self.self_attn(x, x, x)
        x, trend1      = self.decomp1(self.norm1(x + self.drop(sa_out)))
        ca_out         = self.cross_attn(x, enc_out, enc_out)
        x, trend2      = self.decomp2(self.norm2(x + self.drop(ca_out)))
        ffn_out        = self.ffn(x)
        x, trend3      = self.decomp3(self.norm3(x + self.drop(ffn_out)))
        # Combine trend components
        trend_delta    = self.trend_proj(trend1 + trend2 + trend3)  # (B, L_dec, n_features)
        return x, trend_delta


class AutoformerForecaster(nn.Module):
    """
    Autoformer forecaster.

    Input:  (B, T_in, n_features)
    Output: (B, T_out)

    The encoder processes X_enc = X.
    The decoder start token is:
        X_dec_seasonal = [X_seasonal[:, -T_out:, :], zeros(B, T_out, F)]   length=T_dec
        X_dec_trend    = [X_trend[:, -T_out:, :],    zeros(B, T_out, F)]   length=T_dec
    The decoder outputs seasonal + accumulated trend, both of shape (B, T_dec, F).
    We take the last T_out timesteps and project to a scalar per step.
    """

    def __init__(
        self,
        n_features: int,
        T_in:       int,
        T_out:      int,
        d_model:    int   = 64,
        n_heads:    int   = 4,
        e_layers:   int   = 2,
        d_layers:   int   = 1,
        d_ff:       int   = 128,
        moving_avg: int   = 5,
        factor:     int   = 3,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.T_out      = T_out
        self.n_features = n_features

        # Series decomposition for initial decoder token construction
        self.decomp_init = SeriesDecomp(moving_avg)

        # Input projections
        self.enc_proj = nn.Linear(n_features, d_model)
        self.dec_proj = nn.Linear(n_features, d_model)

        # Sinusoidal positional encoding (fixed)
        self.register_buffer(
            "enc_pe", self._sinusoidal_pe(T_in, d_model)
        )
        T_dec = 2 * T_out
        self.register_buffer(
            "dec_pe", self._sinusoidal_pe(T_dec, d_model)
        )

        # Encoder
        self.enc_layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model, n_heads, d_ff, factor, moving_avg, dropout)
            for _ in range(e_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # Decoder
        self.dec_layers = nn.ModuleList([
            AutoformerDecoderLayer(d_model, n_heads, d_ff, factor, moving_avg, n_features, dropout)
            for _ in range(d_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)

        # Output projections: seasonal (d_model → 1) and trend (n_features → 1)
        self.out_proj       = nn.Linear(d_model, 1)
        self.trend_proj_out = nn.Linear(n_features, 1)

    @staticmethod
    def _sinusoidal_pe(length: int, d_model: int) -> torch.Tensor:
        pe  = torch.zeros(length, d_model)
        pos = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        return pe.unsqueeze(0)   # (1, length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, F)
        B, T_in, F = x.shape
        T_out = self.T_out

        # Decompose encoder input for decoder initialisation
        x_seasonal, x_trend = self.decomp_init(x)   # (B, T_in, F) each

        # Build decoder start token (length = 2*T_out)
        zeros     = torch.zeros(B, T_out, F, device=x.device, dtype=x.dtype)
        dec_seas  = torch.cat([x_seasonal[:, -T_out:, :], zeros], dim=1)  # (B, 2*T_out, F)
        dec_trend = torch.cat([x_trend[:, -T_out:, :],    zeros], dim=1)  # (B, 2*T_out, F)

        # Encoder
        enc = self.enc_proj(x) + self.enc_pe[:, :T_in, :]     # (B, T_in, d_model)
        for layer in self.enc_layers:
            enc = layer(enc)
        enc = self.enc_norm(enc)

        # Decoder
        dec  = self.dec_proj(dec_seas) + self.dec_pe[:, :2*T_out, :]  # (B, 2*T_out, d_model)
        trend_accum = dec_trend                                # (B, 2*T_out, F)
        for layer in self.dec_layers:
            dec, trend_delta = layer(dec, enc)
            trend_accum = trend_accum + trend_delta            # (B, 2*T_out, F)
        dec = self.dec_norm(dec)

        # Take last T_out steps from seasonal and trend
        dec_out   = dec[:, -T_out:, :]             # (B, T_out, d_model)
        trend_out = trend_accum[:, -T_out:, :]     # (B, T_out, F)

        # Seasonal part: d_model → 1 per step (learned)
        seasonal_pred = self.out_proj(
            dec_out.reshape(B * T_out, -1)
        ).reshape(B, T_out)                        # (B, T_out)

        # Trend part: F → 1 per step (learned, not a raw mean)
        trend_pred = self.trend_proj_out(
            trend_out.reshape(B * T_out, -1)
        ).reshape(B, T_out)                        # (B, T_out)

        return seasonal_pred + trend_pred



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
    ax.plot(train_losses, label="Train MSE", linewidth=1.5, color="#047857")
    ax.plot(val_losses,   label="Val MSE",   linewidth=1.5, color="#f97316", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (scaled)")
    ax.set_title("Autoformer — Training curves")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig); print(f"  Saved: {out_path}")


def plot_predictions(y_true, y_pred, metrics, out_path):
    N, T_out = y_true.shape; idx_full = np.arange(N)
    zoom_n = min(60, N); idx_zoom = np.arange(N - zoom_n, N)
    actual_s1 = y_true[:, 0]; predicted_s1 = y_pred[:, 0]
    pred_min = y_pred.min(axis=1); pred_max = y_pred.max(axis=1)
    C_ACT = "#1d4ed8"; C_PRED = "#047857"; C_BAND = "#d1fae5"; C_MID = "#0891b2"; C_LAST = "#7c3aed"
    fig = plt.figure(figsize=(14, 8)); gs = gridspec.GridSpec(2, 1, hspace=0.48)
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(idx_full, pred_min, pred_max, alpha=0.22, color=C_BAND,
                     label=f"Forecast spread (step 1–{T_out})")
    ax1.plot(idx_full, actual_s1,    color=C_ACT, linewidth=1.5, label="Actual", zorder=4)
    ax1.plot(idx_full, predicted_s1, color=C_PRED, linewidth=1.2, linestyle="--", alpha=0.88,
             label="Autoformer predicted (step 1)", zorder=5)
    ann = (f"Combined = {metrics['Combined']:.2f}%\nMAPE     = {metrics['MAPE']:.2f}%\n"
           f"MAE%     = {metrics['MAE_pct']:.2f}%\nRMSE%    = {metrics['RMSE_pct']:.2f}%\n"
           f"R²       = {metrics['R2']:.4f}\nMAE      = {metrics['MAE']:.0f} riders\n"
           f"RMSE     = {metrics['RMSE']:.0f} riders")
    ax1.text(0.01, 0.97, ann, transform=ax1.transAxes, fontsize=8, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#d1d5db", alpha=0.92))
    ax1.set_title("Autoformer — Test set: Actual vs Predicted (full period)", fontsize=11, fontweight="bold")
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
    ax1.set_title("Autoformer — Per-horizon metrics", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)
    ax2.plot(steps, r2, marker="o", color="#047857", linewidth=1.8, markersize=5)
    ax2.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax2.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax2.set_ylim(min(min(r2) - 0.05, -0.1), 1.08)
    ax2.set_ylabel("R²"); ax2.set_title("Autoformer — R² per horizon", fontweight="bold")
    ax2.grid(alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
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

    # Ensure moving_avg kernel ≤ T_in and T_out is achievable
    moving_avg = min(args.moving_avg, T_in)

    model = AutoformerForecaster(
        n_features = n_features,
        T_in       = T_in,
        T_out      = T_out,
        d_model    = args.d_model,
        n_heads    = args.n_heads,
        e_layers   = args.e_layers,
        d_layers   = args.d_layers,
        d_ff       = args.d_ff,
        moving_avg = moving_avg,
        factor     = args.factor,
        dropout    = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel       : AutoformerForecaster")
    print(f"  d_model   : {args.d_model}   n_heads: {args.n_heads}   d_ff: {args.d_ff}")
    print(f"  e_layers  : {args.e_layers}   d_layers: {args.d_layers}")
    print(f"  moving_avg: {moving_avg}   factor: {args.factor}")
    print(f"  In        : (batch, {T_in}, {n_features})")
    print(f"  Dec input : (batch, {2*T_out}, {n_features})   Out: (batch, {T_out})")
    print(f"  Params    : {n_params:,}")

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf"); best_epoch = 0; patience_count = 0
    train_losses, val_losses = [], []; best_state = None
    print(f"\nTraining  (max {args.epochs} epochs, patience={args.patience})")
    print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>10}"); print("─" * 45)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimiser, criterion, scaler, use_amp)
        va_loss = evaluate(model, val_loader, criterion, use_amp)
        train_losses.append(tr_loss); val_losses.append(va_loss)
        scheduler.step(va_loss); lr_now = optimiser.param_groups[0]["lr"]
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

    W = 10; print(f"\n{'='*85}"); print("Autoformer — TEST SET METRICS"); print(f"{'='*85}")
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

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S"); out_dir = f"src/outputs/autoformer/{run_id}"
    os.makedirs(out_dir, exist_ok=True); torch.save(model.state_dict(), f"{out_dir}/model.pt")
    results = {
        "run_id": run_id, "model": "AutoformerForecaster",
        "hparams": {
            "d_model": args.d_model, "n_heads": args.n_heads, "d_ff": args.d_ff,
            "e_layers": args.e_layers, "d_layers": args.d_layers,
            "moving_avg": moving_avg, "factor": args.factor,
            "dropout": args.dropout, "T_in": T_in, "T_out": T_out,
            "n_features": n_features, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay,
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

    print(f"\n{'='*85}"); print(f"Autoformer run complete  →  {out_dir}/"); print(f"{'='*85}")
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
        ("TFT",           args.tft_results,           "src/outputs/tft",            "#ca8a04"),
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
    models_data.append(("Autoformer", overall, per_step, "#047857"))

    if len(models_data) > 1:
        print_comparison_table(models_data[:-1], overall, "Autoformer")
        plot_comparison(models_data, out_path=f"{out_dir}/comparison_{len(models_data)}_way.png")

    comparison["autoformer"] = {"run_id": run_id, "overall": {k: round(v, 4) for k, v in overall.items()}}
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
    print(f"  Autoformer vs naive ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()
