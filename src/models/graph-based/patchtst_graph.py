"""
patchtst_graph.py  — PatchTST + Graph Hybrid for Transit Ridership Forecasting

Combines:
  (1) PatchTST (Nie et al., 2023)  — Transformer on time patches per channel
        Nie, Y., Nguyen, N. H., Seyfi, P., & Kolar, M. (2023).
        "A Time Series is Worth 64 Words: Long-term Forecasting with
        Transformers."  ICLR 2023.  arXiv:2211.14730

  (2) Graph Convolution (Kipf & Welling, 2017) — cross-feature message passing
        Kipf, T. N., & Welling, M. (2017).
        "Semi-Supervised Classification with Graph Convolutional Networks."
        ICLR 2017.  arXiv:1609.02907

Architecture:
  The n_features input features are treated as N independent channels (nodes).

  PatchTST branch (channel-independence):
    X (B, T_in, N) → per-node sliding patches (B, N, n_patches, patch_len)
    → Patch embed → Positional encoding → Transformer encoder (shared weights)
    → Mean pool patches → node embeddings (B, N, d_model)

  Graph branch:
    adjacency A (N, N) from abs Pearson correlation of training features
    → L GCN layers on node embeddings (B, N, d_model)

  Fusion:
    Mean pool over N → (B, d_model) → MLP head → (B, T_out)

Comparison:
  --lstm-results, --bilstm-results, --tpalstm-results,
  --cnnlstm-results, --cnnbilstm-results, --stlstm-results,
  --stgcn-results, --wavenet-results, --dcrnn-results, --stgat-results
  All optional; each auto-detects the most recent run if omitted.

Usage:
  python patchtst_graph.py
  python patchtst_graph.py --patch-len 2 --d-model 64 --n-heads 4 --n-layers 3
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
        description="PatchTST+Graph hybrid forecaster with 11-way comparison"
    )
    p.add_argument("--seq-dir",           default="data/sequences/lstm")
    p.add_argument("--patch-len",         type=int,   default=4,
                   help="Patch length (number of timesteps per patch)")
    p.add_argument("--patch-stride",      type=int,   default=2,
                   help="Stride between patches")
    p.add_argument("--d-model",           type=int,   default=128,
                   help="Transformer model dimension")
    p.add_argument("--n-heads",           type=int,   default=4,
                   help="Number of attention heads")
    p.add_argument("--n-layers",          type=int,   default=3,
                   help="Number of Transformer encoder layers")
    p.add_argument("--ffn-dim",           type=int,   default=256,
                   help="Transformer FFN inner dimension")
    p.add_argument("--n-gcn-layers",      type=int,   default=2,
                   help="Number of GCN layers after Transformer")
    p.add_argument("--adj-threshold",     type=float, default=0.1,
                   help="Min abs correlation to keep graph edge")
    p.add_argument("--dropout",           type=float, default=0.15)
    p.add_argument("--weight-decay",      type=float, default=1e-4,
                   help="Adam weight decay")
    p.add_argument("--batch-size",        type=int,   default=32)
    p.add_argument("--epochs",            type=int,   default=150)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--patience",          type=int,   default=20)
    p.add_argument("--device",            default="auto")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--lstm-results",      default=None)
    p.add_argument("--bilstm-results",    default=None)
    p.add_argument("--tpalstm-results",   default=None)
    p.add_argument("--cnnlstm-results",   default=None)
    p.add_argument("--cnnbilstm-results", default=None)
    p.add_argument("--stlstm-results",    default=None)
    p.add_argument("--stgcn-results",     default=None)
    p.add_argument("--wavenet-results",   default=None)
    p.add_argument("--dcrnn-results",     default=None)
    p.add_argument("--stgat-results",     default=None)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Graph construction helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_adj_norm(X_train: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """
    Build symmetrically normalised adjacency A_hat = D^{-1/2}(A+I)D^{-1/2}
    from absolute Pearson correlation between features.
    X_train : (N_samples, T_in, N_features)
    Returns  : A_norm (N, N) float32.
    """
    X_np   = X_train.cpu().numpy()
    X_flat = X_np.reshape(-1, X_np.shape[-1])
    corr   = np.corrcoef(X_flat.T).astype(np.float32)
    A      = np.abs(corr)
    np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0, copy=False)  # guard: zero-variance cols
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 0.0)

    # Add self-loops and symmetric normalise
    A_hat = A + np.eye(A.shape[0], dtype=np.float32)
    D     = A_hat.sum(axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(D + 1e-9)
    A_norm = D_inv_sqrt[:, None] * A_hat * D_inv_sqrt[None, :]
    return torch.from_numpy(A_norm)


# ══════════════════════════════════════════════════════════════════════════════
# Model components
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformer over patches.
    Input / Output: (B, seq_len, d_model)
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe      = torch.zeros(max_len, d_model)
        pos     = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GCNLayer(nn.Module):
    """
    Single-layer Graph Convolutional Network (Kipf & Welling 2017).
    A_norm @ X @ W + b, then ReLU.

    Input  : x (B, N, in_features),  A_norm (N, N)
    Output : (B, N, out_features)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # A_norm @ x: (N, N) × (B, N, F) via einsum
        x_agg = torch.einsum("nm,bmc->bnc", A_norm, x)   # (B, N, in_features)
        return F.relu(self.W(x_agg))


class PatchTSTGraphForecaster(nn.Module):
    """
    PatchTST + Graph Hybrid forecaster.

    PatchTST processes temporal dynamics per feature (channel-independence).
    GCN captures cross-feature spatial relationships.
    Mean pool over N → MLP → T_out.

    The normalised adjacency A_norm is stored as a model buffer.
    forward(x) only takes x.

    Input  : (B, T_in, n_features)
    Output : (B, T_out)
    """

    def __init__(
        self,
        n_features:   int,
        patch_len:    int,
        patch_stride: int,
        T_in:         int,
        d_model:      int,
        n_heads:      int,
        n_layers:     int,
        ffn_dim:      int,
        n_gcn_layers: int,
        T_out:        int,
        A_norm:       torch.Tensor,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.n_features  = n_features
        self.patch_len   = patch_len
        self.patch_stride = patch_stride

        # ── Normalised adjacency (buffer) ─────────────────────────────────────
        self.register_buffer("A_norm", A_norm)

        # ── Compute number of patches ─────────────────────────────────────────
        n_patches = (T_in - patch_len) // patch_stride + 1
        assert n_patches >= 1, (
            f"n_patches={n_patches}≤0 with T_in={T_in}, patch_len={patch_len}, "
            f"patch_stride={patch_stride}"
        )
        self.n_patches = n_patches

        # ── Patch embedding: patch_len → d_model ──────────────────────────────
        self.patch_embed = nn.Linear(patch_len, d_model)

        # ── Positional encoding ───────────────────────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, max_len=n_patches + 4, dropout=dropout)

        # ── Transformer encoder (shared across all features) ──────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model       = d_model,
            nhead         = n_heads,
            dim_feedforward = ffn_dim,
            dropout       = dropout,
            batch_first   = True,
            norm_first    = True,    # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.transformer_norm = nn.LayerNorm(d_model)

        # ── GCN layers ────────────────────────────────────────────────────────
        self.gcn_layers = nn.ModuleList()
        for _ in range(n_gcn_layers):
            self.gcn_layers.append(GCNLayer(d_model, d_model))

        self.gcn_dropout = nn.Dropout(dropout)

        # ── MLP head ──────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, T_out),
        )

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T_in, N)
        Returns: patches (B, N, n_patches, patch_len)
        """
        B, T, N = x.shape
        patches = []
        for i in range(self.n_patches):
            start = i * self.patch_stride
            end   = start + self.patch_len
            patches.append(x[:, start:end, :])         # (B, patch_len, N)
        patches = torch.stack(patches, dim=1)           # (B, n_patches, patch_len, N)
        patches = patches.permute(0, 3, 1, 2)          # (B, N, n_patches, patch_len)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T_in, N)
        """
        B, T, N = x.shape

        # ── PatchTST branch ───────────────────────────────────────────────────
        # Extract patches: (B, N, n_patches, patch_len)
        patches = self._extract_patches(x)
        P       = self.n_patches

        # Embed patches: (B, N, P, d_model)
        h = self.patch_embed(patches)

        # Channel-independence: process each node's P patches through Transformer
        # Reshape: (B*N, P, d_model)
        h = h.reshape(B * N, P, h.shape[-1])
        h = self.pos_enc(h)

        # Transformer encoder: (B*N, P, d_model) → (B*N, P, d_model)
        h = self.transformer(h)
        h = self.transformer_norm(h)

        # Mean-pool over patches → node embedding
        h = h.mean(dim=1)                               # (B*N, d_model)
        h = h.reshape(B, N, -1)                         # (B, N, d_model)

        # ── GCN branch ────────────────────────────────────────────────────────
        for gcn in self.gcn_layers:
            h = gcn(h, self.A_norm)                     # (B, N, d_model)
            h = self.gcn_dropout(h)

        # ── Pool over N, MLP ──────────────────────────────────────────────────
        h = h.mean(dim=1)                               # (B, d_model)
        return self.head(h)                             # (B, T_out)



# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_splits(seq_dir, device):
    def t(name): return torch.from_numpy(np.load(os.path.join(seq_dir,name))).float().to(device)
    X_tr,y_tr=t("X_train.npy"),t("y_train.npy"); X_va,y_va=t("X_val.npy"),t("y_val.npy"); X_te,y_te=t("X_test.npy"),t("y_test.npy")
    print(f"Shapes loaded from {seq_dir}:")
    print(f"  X_train {tuple(X_tr.shape)}   y_train {tuple(y_tr.shape)}")
    print(f"  X_val   {tuple(X_va.shape)}   y_val   {tuple(y_va.shape)}")
    print(f"  X_test  {tuple(X_te.shape)}   y_test  {tuple(y_te.shape)}")
    return (X_tr,y_tr),(X_va,y_va),(X_te,y_te)


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimiser, criterion, device) -> float:
    model.train(); total=0.0; n_skipped=0
    for X_b,y_b in loader:
        optimiser.zero_grad(); loss=criterion(model(X_b),y_b)
        if not torch.isfinite(loss): n_skipped+=1; continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0); optimiser.step()
        total+=loss.item()*X_b.size(0)
    if n_skipped: print(f"  [WARN] {n_skipped} batch(es) skipped — non-finite loss")
    return total/len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion) -> float:
    model.eval(); total=0.0
    for X_b,y_b in loader: total+=criterion(model(X_b),y_b).item()*X_b.size(0)
    return total/len(loader.dataset)



# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(train_losses, val_losses, out_path):
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(train_losses,label="Train MSE",linewidth=1.5,color="#dc2626")
    ax.plot(val_losses,label="Val MSE",linewidth=1.5,color="#f97316",linestyle="--")
    ax.set_xlabel("Epoch");ax.set_ylabel("MSE loss (scaled)");ax.set_title("PatchTST-Graph — Training curves")
    ax.legend();ax.grid(alpha=0.3);fig.tight_layout();fig.savefig(out_path,dpi=150);plt.close(fig)
    print(f"  Saved: {out_path}")

def plot_predictions(y_true, y_pred, metrics, out_path):
    N,T_out=y_true.shape; idx_full=np.arange(N); zoom_n=min(60,N); idx_zoom=np.arange(N-zoom_n,N)
    actual_s1=y_true[:,0]; predicted_s1=y_pred[:,0]; pred_min=y_pred.min(axis=1); pred_max=y_pred.max(axis=1)
    C_ACT="#1d4ed8"; C_PRED="#818cf8"; C_BAND="#e0e7ff"; C_MID="#0891b2"; C_LAST="#7c3aed"
    fig=plt.figure(figsize=(14,8)); gs=gridspec.GridSpec(2,1,hspace=0.48)
    ax1=fig.add_subplot(gs[0])
    ax1.fill_between(idx_full,pred_min,pred_max,alpha=0.22,color=C_BAND,label=f"Forecast spread (step 1–{T_out})")
    ax1.plot(idx_full,actual_s1,color=C_ACT,linewidth=1.5,label="Actual",zorder=4)
    ax1.plot(idx_full,predicted_s1,color=C_PRED,linewidth=1.2,linestyle="--",alpha=0.88,label="PatchTST-Graph predicted (step 1)",zorder=5)
    ann=(f"Combined = {metrics['Combined']:.2f}%\nMAPE     = {metrics['MAPE']:.2f}%\n"
         f"MAE%     = {metrics['MAE_pct']:.2f}%\nRMSE%    = {metrics['RMSE_pct']:.2f}%\n"
         f"R²       = {metrics['R2']:.4f}\nMAE      = {metrics['MAE']:.0f} riders\nRMSE     = {metrics['RMSE']:.0f} riders")
    ax1.text(0.01,0.97,ann,transform=ax1.transAxes,fontsize=8,verticalalignment="top",fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.45",facecolor="white",edgecolor="#d1d5db",alpha=0.92))
    ax1.set_title("PatchTST-Graph — Test set: Actual vs Predicted (full period)",fontsize=11,fontweight="bold")
    ax1.set_xlabel("Test sample index");ax1.set_ylabel("Ridership (riders)");ax1.legend(loc="upper right",fontsize=8);ax1.grid(alpha=0.25)
    ax2=fig.add_subplot(gs[1])
    ax2.fill_between(idx_zoom,pred_min[idx_zoom],pred_max[idx_zoom],alpha=0.18,color=C_BAND)
    ax2.plot(idx_zoom,actual_s1[idx_zoom],color=C_ACT,linewidth=1.7,label="Actual",zorder=5)
    steps_show=sorted({0,T_out//2,T_out-1}); palette=[C_PRED,C_MID,C_LAST]; styles=["--","-.",":" ]
    for s,col,ls in zip(steps_show,palette,styles):
        ax2.plot(idx_zoom,y_pred[idx_zoom,s],color=col,linewidth=1.4,linestyle=ls,alpha=0.88,label=f"Predicted step {s+1}")
    ax2.set_title(f"Zoomed: last {zoom_n} samples",fontsize=10,fontweight="bold")
    ax2.set_xlabel("Test sample index");ax2.set_ylabel("Ridership (riders)");ax2.legend(loc="upper right",fontsize=8,ncol=2);ax2.grid(alpha=0.25)
    fig.savefig(out_path,dpi=150,bbox_inches="tight");plt.close(fig);print(f"  Saved: {out_path}")

def plot_per_step_metrics(per_step, out_path):
    steps=[f"t+{i+1}" for i in range(len(per_step))]
    combined=[m["Combined"] for m in per_step]; mape=[m["MAPE"] for m in per_step]
    mae_pct=[m["MAE_pct"] for m in per_step]; rmse_pct=[m["RMSE_pct"] for m in per_step]; r2=[m["R2"] for m in per_step]
    x,w=np.arange(len(steps)),0.2
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(max(8,len(steps)*0.85),7),gridspec_kw={"hspace":0.48})
    ax1.bar(x-1.5*w,combined,w,label="Combined%",color="#16a34a",alpha=0.85)
    ax1.bar(x-0.5*w,mape,w,label="MAPE%",color="#dc2626",alpha=0.85)
    ax1.bar(x+0.5*w,mae_pct,w,label="MAE%",color="#2563eb",alpha=0.85)
    ax1.bar(x+1.5*w,rmse_pct,w,label="RMSE%",color="#f97316",alpha=0.85)
    ax1.set_xticks(x);ax1.set_xticklabels(steps);ax1.set_ylabel("% of mean demand / score")
    ax1.set_title("PatchTST-Graph — Per-horizon metrics",fontweight="bold");ax1.legend(fontsize=8);ax1.grid(axis="y",alpha=0.3)
    ax2.plot(steps,r2,marker="o",color="#dc2626",linewidth=1.8,markersize=5)
    ax2.axhline(0,color="#9ca3af",linewidth=0.8,linestyle="--");ax2.axhline(1,color="#16a34a",linewidth=0.8,linestyle=":")
    ax2.set_ylim(min(min(r2)-0.05,-0.1),1.08);ax2.set_ylabel("R²");ax2.set_title("PatchTST-Graph — R² per horizon",fontweight="bold");ax2.grid(alpha=0.3)
    fig.savefig(out_path,dpi=150,bbox_inches="tight");plt.close(fig);print(f"  Saved: {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args=parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.device=="auto":
        if   torch.cuda.is_available():         device=torch.device("cuda")
        elif torch.backends.mps.is_available(): device=torch.device("mps")
        else:                                   device=torch.device("cpu")
    else: device=torch.device(args.device)
    print(f"Device: {device}")

    (X_tr,y_tr),(X_va,y_va),(X_te,y_te)=load_splits(args.seq_dir,device)
    T_in=X_tr.shape[1]; n_features=X_tr.shape[2]; T_out=y_tr.shape[1]
    meta_path=os.path.join(args.seq_dir,"split_dates.json")
    split_meta=json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    train_loader=DataLoader(TensorDataset(X_tr,y_tr),batch_size=args.batch_size,shuffle=True)
    val_loader=DataLoader(TensorDataset(X_va,y_va),batch_size=args.batch_size)
    test_loader=DataLoader(TensorDataset(X_te,y_te),batch_size=args.batch_size)

    print(f"\nBuilding feature correlation adjacency (threshold={args.adj_threshold})...")
    A_norm=build_feature_adj_norm(X_tr,threshold=args.adj_threshold)
    n_edges=int(((A_norm>0).float().sum().item()-n_features)//2)  # exclude self-loops
    print(f"  Nodes: {n_features}  Approx edges (no self-loops): {n_edges}")

    n_patches=(T_in-args.patch_len)//args.patch_stride+1
    print(f"  Patches per feature: {n_patches} (patch_len={args.patch_len}, stride={args.patch_stride})")

    model=PatchTSTGraphForecaster(
        n_features=n_features, patch_len=args.patch_len, patch_stride=args.patch_stride,
        T_in=T_in, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        ffn_dim=args.ffn_dim, n_gcn_layers=args.n_gcn_layers, T_out=T_out,
        A_norm=A_norm, dropout=args.dropout,
    ).to(device)

    n_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel         : PatchTSTGraphForecaster")
    print(f"  Nodes (N)   : {n_features}   n_patches={n_patches}   patch_len={args.patch_len}")
    print(f"  d_model={args.d_model}   n_heads={args.n_heads}   n_layers={args.n_layers}   ffn={args.ffn_dim}")
    print(f"  GCN layers  : {args.n_gcn_layers}")
    print(f"  In          : (batch, {T_in}, {n_features})")
    print(f"  Out         : (batch, {T_out})")
    print(f"  Params      : {n_params:,}")

    criterion=nn.MSELoss()
    optimiser=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,mode="min",factor=0.5,patience=5)

    best_val_loss=float("inf"); best_epoch=0; patience_count=0
    train_losses,val_losses=[],[]; best_state=None
    print(f"\nTraining  (max {args.epochs} epochs, patience={args.patience})")
    print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>10}"); print("─"*45)

    for epoch in range(1,args.epochs+1):
        tr_loss=train_one_epoch(model,train_loader,optimiser,criterion,device)
        va_loss=evaluate(model,val_loader,criterion)
        train_losses.append(tr_loss); val_losses.append(va_loss)
        scheduler.step(va_loss); lr_now=optimiser.param_groups[0]["lr"]
        print(f"{epoch:6d}  {tr_loss:10.6f}  {va_loss:10.6f}  {lr_now:10.2e}")
        if va_loss<best_val_loss:
            best_val_loss,best_epoch,patience_count=va_loss,epoch,0
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            patience_count+=1
            if patience_count>=args.patience:
                print(f"\nEarly stop at epoch {epoch}  (best={best_val_loss:.6f} @ epoch {best_epoch})"); break

    model.load_state_dict(best_state)
    model.eval(); preds_s,trues_s=[],[]
    with torch.no_grad():
        for X_b,y_b in test_loader: preds_s.append(model(X_b).cpu().numpy()); trues_s.append(y_b.cpu().numpy())
    y_pred_s=np.concatenate(preds_s); y_true_s=np.concatenate(trues_s)

    scaler_y_path=os.path.join(args.seq_dir,"scaler_y.pkl")
    if os.path.exists(scaler_y_path):
        scaler_y=joblib.load(scaler_y_path); N2,T2=y_pred_s.shape
        y_pred=scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).reshape(N2,T2)
        y_true=scaler_y.inverse_transform(y_true_s.reshape(-1,1)).reshape(N2,T2)
    else:
        print("[WARN] scaler_y.pkl not found — metrics in scaled units."); y_pred,y_true=y_pred_s,y_true_s

    W=10; print(f"\n{'='*85}"); print("PatchTST-Graph — TEST SET METRICS"); print(f"{'='*85}")
    header=(f"{'Step':>5}  {'Combined%':>{W}}  {'MAPE%':>{W}}  {'MAE%':>{W}}  {'RMSE%':>{W}}  {'R²':>{W}}  {'MAE':>{W}}  {'RMSE':>{W}}")
    print(header); print("─"*len(header))
    per_step=[]
    for s in range(T_out):
        m=compute_metrics(y_true[:,s],y_pred[:,s]); per_step.append(m)
        print(f"{s+1:5d}  {m['Combined']:>{W}.2f}  {m['MAPE']:>{W}.2f}  {m['MAE_pct']:>{W}.2f}  {m['RMSE_pct']:>{W}.2f}  {m['R2']:>{W}.4f}  {m['MAE']:>{W}.0f}  {m['RMSE']:>{W}.0f}")
    overall=compute_metrics(y_true.flatten(),y_pred.flatten())
    print("─"*len(header))
    print(f"{'Avg':>5}  {overall['Combined']:>{W}.2f}  {overall['MAPE']:>{W}.2f}  {overall['MAE_pct']:>{W}.2f}  {overall['RMSE_pct']:>{W}.2f}  {overall['R2']:>{W}.4f}  {overall['MAE']:>{W}.0f}  {overall['RMSE']:>{W}.0f}")

    run_id=datetime.now().strftime("%Y%m%d_%H%M%S"); out_dir=f"src/outputs/patchtst_graph/{run_id}"
    os.makedirs(out_dir,exist_ok=True); torch.save(model.state_dict(),f"{out_dir}/model.pt")
    results={"run_id":run_id,"model":"PatchTSTGraphForecaster",
             "hparams":{"patch_len":args.patch_len,"patch_stride":args.patch_stride,"n_patches":n_patches,
                        "d_model":args.d_model,"n_heads":args.n_heads,"n_layers":args.n_layers,"ffn_dim":args.ffn_dim,
                        "n_gcn_layers":args.n_gcn_layers,"adj_threshold":args.adj_threshold,"dropout":args.dropout,
                        "weight_decay":args.weight_decay,
                        "T_in":T_in,"T_out":T_out,"n_features":n_features,"batch_size":args.batch_size,"lr":args.lr},
             "training":{"best_epoch":best_epoch,"best_val_loss":round(best_val_loss,8),"total_epochs":len(train_losses)},
             "split_dates":split_meta,
             "test_metrics":{"overall":{k:round(v,4) for k,v in overall.items()},
                             "per_step":[{k:round(v,4) for k,v in m.items()} for m in per_step]}}
    with open(f"{out_dir}/results.json","w") as f: json.dump(results,f,indent=2)

    print(f"\nSaving plots...")
    plot_loss_curves(train_losses,val_losses,f"{out_dir}/loss_curves.png")
    plot_predictions(y_true,y_pred,overall,f"{out_dir}/test_predictions.png")
    plot_per_step_metrics(per_step,f"{out_dir}/per_step_metrics.png")

    print(f"\n{'='*85}"); print(f"PatchTST-Graph run complete  →  {out_dir}/"); print(f"{'='*85}")
    print(f"  Combined  : {overall['Combined']:.2f}%"); print(f"  MAPE      : {overall['MAPE']:.2f}%")
    print(f"  MAE%      : {overall['MAE_pct']:.2f}%    (raw MAE  = {overall['MAE']:.0f} riders)")
    print(f"  RMSE%     : {overall['RMSE_pct']:.2f}%   (raw RMSE = {overall['RMSE']:.0f} riders)")
    print(f"  R²        : {overall['R2']:.4f}")

    PRIOR_MODELS=[
        ("LSTM",          args.lstm_results,      "src/outputs/lstm",          "#2563eb"),
        ("BiLSTM",        args.bilstm_results,    "src/outputs/bilstm",        "#7c3aed"),
        ("TPA-LSTM",      args.tpalstm_results,   "src/outputs/tpa_lstm",      "#0891b2"),
        ("CNN-LSTM",      args.cnnlstm_results,   "src/outputs/cnn_lstm",      "#16a34a"),
        ("CNN-BiLSTM",    args.cnnbilstm_results, "src/outputs/cnn_bilstm",    "#d97706"),
        ("ST-LSTM",       args.stlstm_results,    "src/outputs/st_lstm",       "#dc2626"),
        ("STGCN",         args.stgcn_results,     "src/outputs/stgcn",         "#10b981"),
        ("WaveNet",       args.wavenet_results,   "src/outputs/graph_wavenet", "#f472b6"),
        ("DCRNN",         args.dcrnn_results,     "src/outputs/dcrnn",         "#38bdf8"),
        ("STGAT",         args.stgat_results,     "src/outputs/stgat",         "#fb923c"),
    ]
    models_data=[]; comparison={}
    for name,path,model_dir,color in PRIOR_MODELS:
        key=name.lower().replace("-","_"); data=load_model_results(path,model_dir,name)
        if data:
            m_overall=data["test_metrics"]["overall"]; m_ps=data["test_metrics"]["per_step"]
            models_data.append((name,m_overall,m_ps,color)); comparison[key]={"run_id":data.get("run_id"),"overall":m_overall}
        else: comparison[key]={"run_id":None,"overall":None}
    models_data.append(("PatchTST-Graph",overall,per_step,"#818cf8"))
    if len(models_data)>1:
        print_comparison_table(models_data[:-1],overall,"PatchTST-Graph")
        plot_comparison(models_data,out_path=f"{out_dir}/comparison_{len(models_data)}_way.png")
    comparison["patchtst_graph"]={"run_id":run_id,"overall":{k:round(v,4) for k,v in overall.items()}}
    for name,m_overall,_,__ in models_data[:-1]:
        key=name.lower().replace("-","_")
        comparison[f"delta_vs_{key}"]={k:round(overall.get(k,0)-m_overall.get(k,0),4) for k in overall}
    results["comparison"]=comparison
    with open(f"{out_dir}/results.json","w") as f: json.dump(results,f,indent=2)

    target_idx=split_meta.get("target_col_idx",0)
    last_obs_s=X_te[:,-1,target_idx:target_idx+1].cpu().numpy()
    naive_pred_s=np.tile(last_obs_s,(1,T_out))
    if os.path.exists(scaler_y_path):
        N2b,T2b=naive_pred_s.shape; naive_pred=scaler_y.inverse_transform(naive_pred_s.reshape(-1,1)).reshape(N2b,T2b)
    else: naive_pred=naive_pred_s
    naive=compute_metrics(y_true.flatten(),naive_pred.flatten())
    d_comb=overall["Combined"]-naive["Combined"]; d_mape=naive["MAPE"]-overall["MAPE"]
    print(f"\n  Naive persistence  Combined={naive['Combined']:.2f}%  MAPE={naive['MAPE']:.2f}%  R²={naive['R2']:.4f}")
    print(f"  PatchTST-Graph vs naive   ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()
