"""
dcrnn.py  — Diffusion Convolutional Recurrent Neural Network (DCRNN)

Implements the architecture from:
  Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018).
  "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
  Forecasting."  ICLR 2018.  arXiv:1707.01926

Adaptation for multivariate feature-node graph:
  The n_features input features are treated as N graph nodes. At each
  timestep t, node i has a scalar signal x[b, t, i]. The graph adjacency
  is built from absolute Pearson correlation between features on training
  data. Forward and backward K-step diffusion matrices are precomputed and
  stored as model buffers.

  The standard GRU cell's linear transformations are replaced by diffusion
  convolution (DiffConv) on the feature graph:

    r  = sigmoid( DiffConv([h, x]) )          ← reset gate
    u  = sigmoid( DiffConv([h, x]) )          ← update gate
    c  = tanh(    DiffConv([r*h, x]) )        ← candidate state
    h' = (1-u)*h + u*c

  Encoder: stack of DCGRUCells over T_in timesteps.
  Decoder head: mean over N nodes → MLP → (B, T_out).

Comparison:
  --lstm-results, --bilstm-results, --tpalstm-results,
  --cnnlstm-results, --cnnbilstm-results, --stlstm-results,
  --stgcn-results, --wavenet-results
  All optional; each auto-detects the most recent run if omitted.

Usage:
  python dcrnn.py
  python dcrnn.py --hidden 64 --n-layers 2 --k-diffusion 2
"""

import os
import json
import argparse
import glob
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


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DCRNN forecaster with 9-way comparison")
    p.add_argument("--seq-dir",           default="data/sequences/lstm")
    p.add_argument("--hidden",            type=int,   default=64,
                   help="DCGRU hidden size per node")
    p.add_argument("--n-layers",          type=int,   default=2,
                   help="Number of stacked DCGRUCells")
    p.add_argument("--k-diffusion",       type=int,   default=2,
                   help="Diffusion steps K (K+1 hops including identity)")
    p.add_argument("--adj-threshold",     type=float, default=0.1,
                   help="Min abs Pearson correlation to keep an edge")
    p.add_argument("--dropout",           type=float, default=0.1)
    p.add_argument("--batch-size",        type=int,   default=16)
    p.add_argument("--epochs",            type=int,   default=50)
    p.add_argument("--lr",                type=float, default=1e-3)
    p.add_argument("--patience",          type=int,   default=10)
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
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Graph construction helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_adj(X_train: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    X_np   = X_train.cpu().numpy()
    X_flat = X_np.reshape(-1, X_np.shape[-1])
    corr   = np.corrcoef(X_flat.T).astype(np.float32)
    A      = np.abs(corr)
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 0.0)
    return torch.from_numpy(A)


def compute_diffusion_matrices(A: torch.Tensor, K: int) -> tuple:
    """
    Precompute K+1 forward and backward random-walk diffusion matrices.

    Forward  random walk: A_fwd  = D^{-1} A
    Backward random walk: A_bwd  = (D^{-1} A)^T

    Returns:
      A_fwd_powers : tensor (K+1, N, N)  [A_fwd^0, A_fwd^1, ..., A_fwd^K]
      A_bwd_powers : tensor (K+1, N, N)
    """
    N     = A.shape[0]
    D     = A.sum(dim=1).clamp(min=1e-9)
    A_fwd = A / D.unsqueeze(1)      # row-normalised forward walk
    A_bwd = A_fwd.t()               # backward walk = transpose of forward

    eye   = torch.eye(N, device=A.device, dtype=A.dtype)
    fwd_list = [eye]
    bwd_list = [eye]
    Af_k = eye.clone()
    Ab_k = eye.clone()
    for _ in range(K):
        Af_k = Af_k @ A_fwd
        Ab_k = Ab_k @ A_bwd
        fwd_list.append(Af_k)
        bwd_list.append(Ab_k)

    return torch.stack(fwd_list, dim=0), torch.stack(bwd_list, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# Model components
# ══════════════════════════════════════════════════════════════════════════════

class DiffusionConv(nn.Module):
    """
    Diffusion convolution operator.

    Computes:
      out = sum_{k=0}^{K} [ A_fwd^k @ x @ theta_fwd_k
                           + A_bwd^k @ x @ theta_bwd_k ] + bias

    Input  : x (B, N, C_in)
    Output : (B, N, C_out)
    """

    def __init__(self, in_channels: int, out_channels: int, K: int):
        super().__init__()
        self.K = K
        self.theta_fwd = nn.ParameterList([
            nn.Parameter(torch.empty(in_channels, out_channels).uniform_(-0.02, 0.02))
            for _ in range(K + 1)
        ])
        self.theta_bwd = nn.ParameterList([
            nn.Parameter(torch.empty(in_channels, out_channels).uniform_(-0.02, 0.02))
            for _ in range(K + 1)
        ])
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self,
        x:          torch.Tensor,
        A_fwd_pows: torch.Tensor,
        A_bwd_pows: torch.Tensor,
    ) -> torch.Tensor:
        """
        x          : (B, N, C_in)
        A_fwd_pows : (K+1, N, N)
        A_bwd_pows : (K+1, N, N)
        """
        out = torch.zeros(
            x.shape[0], x.shape[1], self.theta_fwd[0].shape[1],
            device=x.device, dtype=x.dtype
        )
        for k in range(self.K + 1):
            # (B, N, N) × (B, N, C) via einsum: bnm,bmc→bnc
            x_fwd = torch.einsum("nm,bmc->bnc", A_fwd_pows[k], x)
            x_bwd = torch.einsum("nm,bmc->bnc", A_bwd_pows[k], x)
            out   = out + x_fwd @ self.theta_fwd[k] + x_bwd @ self.theta_bwd[k]
        return out + self.bias


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU Cell.

    Replaces the standard GRU's linear layers with DiffusionConv.
    Hidden state: h (B, N, hidden_size).

    Input  : x  (B, N, input_size),  h (B, N, hidden_size)
    Output : h' (B, N, hidden_size)
    """

    def __init__(self, input_size: int, hidden_size: int, K: int):
        super().__init__()
        in_ch = input_size + hidden_size
        self.diff_r = DiffusionConv(in_ch, hidden_size, K)   # reset gate
        self.diff_u = DiffusionConv(in_ch, hidden_size, K)   # update gate
        self.diff_c = DiffusionConv(in_ch, hidden_size, K)   # candidate

    def forward(
        self,
        x:          torch.Tensor,
        h:          torch.Tensor,
        A_fwd_pows: torch.Tensor,
        A_bwd_pows: torch.Tensor,
    ) -> torch.Tensor:
        xh  = torch.cat([x, h], dim=-1)               # (B, N, in_ch)
        r   = torch.sigmoid(self.diff_r(xh, A_fwd_pows, A_bwd_pows))
        u   = torch.sigmoid(self.diff_u(xh, A_fwd_pows, A_bwd_pows))
        xrh = torch.cat([x, r * h], dim=-1)
        c   = torch.tanh(self.diff_c(xrh, A_fwd_pows, A_bwd_pows))
        return (1 - u) * h + u * c


class DCRNNForecaster(nn.Module):
    """
    Diffusion Convolutional Recurrent Neural Network forecaster.

    Diffusion matrices are precomputed and stored as buffers.
    forward(x) only takes x, compatible with the shared training loop.

    Input  : (B, T_in, n_features)
    Output : (B, T_out)
    """

    def __init__(
        self,
        n_features:   int,
        hidden_size:  int,
        n_layers:     int,
        K:            int,
        T_out:        int,
        A_raw:        torch.Tensor,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.K           = K

        # ── Diffusion matrices (buffers) ──────────────────────────────────────
        A_fwd_pows, A_bwd_pows = compute_diffusion_matrices(A_raw, K)
        self.register_buffer("A_fwd_pows", A_fwd_pows)   # (K+1, N, N)
        self.register_buffer("A_bwd_pows", A_bwd_pows)   # (K+1, N, N)

        # ── Stacked DCGRU cells ───────────────────────────────────────────────
        self.cells = nn.ModuleList()
        for i in range(n_layers):
            in_sz = 1 if i == 0 else hidden_size   # scalar input per node at t=0
            self.cells.append(DCGRUCell(in_sz, hidden_size, K))

        self.dropout = nn.Dropout(dropout)

        # ── MLP head ──────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, T_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T_in, N)
        """
        B, T, N = x.shape
        device  = x.device

        # Initialise hidden states per layer
        h_list = [
            torch.zeros(B, N, self.hidden_size, device=device, dtype=x.dtype)
            for _ in range(self.n_layers)
        ]

        # Process each timestep
        for t in range(T):
            x_t = x[:, t, :].unsqueeze(-1)          # (B, N, 1)
            for i, cell in enumerate(self.cells):
                x_t    = cell(x_t, h_list[i], self.A_fwd_pows, self.A_bwd_pows)
                h_list[i] = x_t
                if i < self.n_layers - 1:
                    x_t = self.dropout(x_t)

        # Final hidden state of last layer
        h_final  = h_list[-1]                       # (B, N, hidden_size)
        h_pooled = h_final.mean(dim=1)              # (B, hidden_size)
        return self.head(h_pooled)                  # (B, T_out)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true=y_true.astype(np.float64); y_pred=y_pred.astype(np.float64)
    y_mean=np.mean(y_true); abs_err=np.abs(y_true-y_pred); sq_err=(y_true-y_pred)**2
    mae_raw=float(np.mean(abs_err)); rmse_raw=float(np.sqrt(np.mean(sq_err)))
    mape=float(np.mean(abs_err/(np.abs(y_true)+1.0))*100); denom=y_mean if y_mean>0 else 1.0
    mae_pct=float(mae_raw/denom*100); rmse_pct=float(rmse_raw/denom*100)
    combined=float(max(0.0,100.0-mape-mae_pct-rmse_pct))
    ss_res=float(np.sum(sq_err)); ss_tot=float(np.sum((y_true-y_mean)**2))
    r2=float(1.0-ss_res/ss_tot) if ss_tot>0 else 0.0
    return {"Combined":combined,"MAPE":mape,"MAE_pct":mae_pct,"RMSE_pct":rmse_pct,"R2":r2,"MAE":mae_raw,"RMSE":rmse_raw}


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
    model.train(); total=0.0
    for X_b,y_b in loader:
        optimiser.zero_grad(); loss=criterion(model(X_b),y_b); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0); optimiser.step()
        total+=loss.item()*X_b.size(0)
    return total/len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion) -> float:
    model.eval(); total=0.0
    for X_b,y_b in loader: total+=criterion(model(X_b),y_b).item()*X_b.size(0)
    return total/len(loader.dataset)


# ══════════════════════════════════════════════════════════════════════════════
# Comparison loader helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_model_results(path, model_dir, label):
    if path:
        if not os.path.exists(path): print(f"[WARN] {label} results not found at: {path}"); return None
        with open(path) as f: return json.load(f)
    candidates=sorted(glob.glob(f"{model_dir}/**/results.json",recursive=True))
    if not candidates: print(f"[INFO] No {label} results.json found under {model_dir} — skipping."); return None
    detected=candidates[-1]; print(f"[INFO] Auto-detected {label} results: {detected}")
    with open(detected) as f: return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(train_losses, val_losses, out_path):
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(train_losses,label="Train MSE",linewidth=1.5,color="#dc2626")
    ax.plot(val_losses,label="Val MSE",linewidth=1.5,color="#f97316",linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (scaled)"); ax.set_title("DCRNN — Training curves")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out_path,dpi=150); plt.close(fig)
    print(f"  Saved: {out_path}")

def plot_predictions(y_true, y_pred, metrics, out_path):
    N,T_out=y_true.shape; idx_full=np.arange(N); zoom_n=min(60,N); idx_zoom=np.arange(N-zoom_n,N)
    actual_s1=y_true[:,0]; predicted_s1=y_pred[:,0]; pred_min=y_pred.min(axis=1); pred_max=y_pred.max(axis=1)
    C_ACT="#1d4ed8"; C_PRED="#38bdf8"; C_BAND="#e0f2fe"; C_MID="#0891b2"; C_LAST="#7c3aed"
    fig=plt.figure(figsize=(14,8)); gs=gridspec.GridSpec(2,1,hspace=0.48)
    ax1=fig.add_subplot(gs[0])
    ax1.fill_between(idx_full,pred_min,pred_max,alpha=0.22,color=C_BAND,label=f"Forecast spread (step 1–{T_out})")
    ax1.plot(idx_full,actual_s1,color=C_ACT,linewidth=1.5,label="Actual",zorder=4)
    ax1.plot(idx_full,predicted_s1,color=C_PRED,linewidth=1.2,linestyle="--",alpha=0.88,label="DCRNN predicted (step 1)",zorder=5)
    ann=(f"Combined = {metrics['Combined']:.2f}%\nMAPE     = {metrics['MAPE']:.2f}%\n"
         f"MAE%     = {metrics['MAE_pct']:.2f}%\nRMSE%    = {metrics['RMSE_pct']:.2f}%\n"
         f"R²       = {metrics['R2']:.4f}\nMAE      = {metrics['MAE']:.0f} riders\nRMSE     = {metrics['RMSE']:.0f} riders")
    ax1.text(0.01,0.97,ann,transform=ax1.transAxes,fontsize=8,verticalalignment="top",fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.45",facecolor="white",edgecolor="#d1d5db",alpha=0.92))
    ax1.set_title("DCRNN — Test set: Actual vs Predicted (full period)",fontsize=11,fontweight="bold")
    ax1.set_xlabel("Test sample index"); ax1.set_ylabel("Ridership (riders)"); ax1.legend(loc="upper right",fontsize=8); ax1.grid(alpha=0.25)
    ax2=fig.add_subplot(gs[1])
    ax2.fill_between(idx_zoom,pred_min[idx_zoom],pred_max[idx_zoom],alpha=0.18,color=C_BAND)
    ax2.plot(idx_zoom,actual_s1[idx_zoom],color=C_ACT,linewidth=1.7,label="Actual",zorder=5)
    steps_show=sorted({0,T_out//2,T_out-1}); palette=[C_PRED,C_MID,C_LAST]; styles=["--","-.",":" ]
    for s,col,ls in zip(steps_show,palette,styles):
        ax2.plot(idx_zoom,y_pred[idx_zoom,s],color=col,linewidth=1.4,linestyle=ls,alpha=0.88,label=f"Predicted step {s+1}")
    ax2.set_title(f"Zoomed: last {zoom_n} samples",fontsize=10,fontweight="bold")
    ax2.set_xlabel("Test sample index"); ax2.set_ylabel("Ridership (riders)"); ax2.legend(loc="upper right",fontsize=8,ncol=2); ax2.grid(alpha=0.25)
    fig.savefig(out_path,dpi=150,bbox_inches="tight"); plt.close(fig); print(f"  Saved: {out_path}")

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
    ax1.set_xticks(x); ax1.set_xticklabels(steps); ax1.set_ylabel("% of mean demand / score")
    ax1.set_title("DCRNN — Per-horizon metrics",fontweight="bold"); ax1.legend(fontsize=8); ax1.grid(axis="y",alpha=0.3)
    ax2.plot(steps,r2,marker="o",color="#dc2626",linewidth=1.8,markersize=5)
    ax2.axhline(0,color="#9ca3af",linewidth=0.8,linestyle="--"); ax2.axhline(1,color="#16a34a",linewidth=0.8,linestyle=":")
    ax2.set_ylim(min(min(r2)-0.05,-0.1),1.08); ax2.set_ylabel("R²"); ax2.set_title("DCRNN — R² per horizon",fontweight="bold"); ax2.grid(alpha=0.3)
    fig.savefig(out_path,dpi=150,bbox_inches="tight"); plt.close(fig); print(f"  Saved: {out_path}")

def plot_comparison(models_data, out_path):
    PCT_KEYS=["Combined","MAPE","MAE_pct","RMSE_pct"]; PCT_LABELS=["Combined%","MAPE%","MAE%","RMSE%"]
    n_steps=min(len(d[2]) for d in models_data); steps=[f"t+{i+1}" for i in range(n_steps)]
    n_models=len(models_data); w_bar=max(0.07,0.80/n_models)
    offsets=np.linspace(-(n_models-1)/2,(n_models-1)/2,n_models)*w_bar
    markers=["o","s","^","D","v","P","*","X","h","8","p"]
    styles=["-","--","-.",(0,(3,1,1,1)),(0,(5,1)),":",( 0,(1,1)),(0,(3,5,1,5)),"--","-.","-"]
    fig=plt.figure(figsize=(16,13)); gs=gridspec.GridSpec(3,2,hspace=0.52,wspace=0.32)
    ax00=fig.add_subplot(gs[0,0]); x=np.arange(len(PCT_KEYS))
    for (name,m,_,col),off in zip(models_data,offsets):
        vals=[m[k] for k in PCT_KEYS]; bars=ax00.bar(x+off,vals,w_bar,label=name,color=col,alpha=0.82)
        for bar in bars:
            h=bar.get_height(); ax00.text(bar.get_x()+bar.get_width()/2,h+0.25,f"{h:.1f}",ha="center",va="bottom",fontsize=5)
    ax00.set_xticks(x);ax00.set_xticklabels(PCT_LABELS,fontsize=9);ax00.set_ylabel("% / score")
    ax00.set_title("Overall — Percentage Metrics",fontweight="bold");ax00.legend(fontsize=6);ax00.grid(axis="y",alpha=0.3)
    ax01=fig.add_subplot(gs[0,1]); names=[d[0] for d in models_data]; r2s=[d[1]["R2"] for d in models_data]; cols=[d[3] for d in models_data]
    bars=ax01.bar(names,r2s,color=cols,alpha=0.82,width=0.4)
    for bar in bars:
        h=bar.get_height(); ax01.text(bar.get_x()+bar.get_width()/2,h+0.004,f"{h:.4f}",ha="center",va="bottom",fontsize=7)
    ax01.set_ylim(0,min(1.12,max(r2s)*1.15+0.05)); ax01.axhline(1,color="#16a34a",linewidth=0.8,linestyle=":")
    ax01.tick_params(axis="x",labelsize=7,rotation=20);ax01.set_ylabel("R²");ax01.set_title("Overall — R²",fontweight="bold");ax01.grid(axis="y",alpha=0.3)
    for ax,(key,lbl) in zip([fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1]),
                               fig.add_subplot(gs[2,0]),fig.add_subplot(gs[2,1])],
                              [("Combined","Combined% per Horizon"),("R2","R² per Horizon"),
                               ("MAE","Raw MAE per Horizon (riders)"),("RMSE","Raw RMSE per Horizon (riders)")]):
        for (name,_,ps,col),mk,ls in zip(models_data,markers,styles):
            ax.plot(steps,[m[key] for m in ps[:n_steps]],marker=mk,color=col,linewidth=1.8,markersize=5,label=name,linestyle=ls)
        if key=="R2": ax.axhline(0,color="#9ca3af",linewidth=0.8,linestyle="--"); ax.axhline(1,color="#16a34a",linewidth=0.8,linestyle=":")
        ax.set_title(lbl,fontweight="bold"); ax.legend(fontsize=6); ax.grid(alpha=0.3)
    fig.suptitle(" vs ".join(d[0] for d in models_data)+" — Test Set Comparison",fontsize=10,fontweight="bold",y=1.01)
    fig.savefig(out_path,dpi=150,bbox_inches="tight"); plt.close(fig); print(f"  Saved: {out_path}")

def print_comparison_table(models_data, new_overall, new_name):
    METRICS_CFG=[("Combined","Combined%",True),("MAPE","MAPE%",False),("MAE_pct","MAE%",False),
                 ("RMSE_pct","RMSE%",False),("R2","R²",True),("MAE","MAE",False),("RMSE","RMSE",False)]
    n_prior=len(models_data); W=10; sep_len=16+W*(1+n_prior)+18*n_prior+12; sep="─"*sep_len
    print(f"\n{'='*sep_len}"); print(f"{n_prior+1}-Way Comparison — Overall Test Metrics"); print(f"{'='*sep_len}")
    header=f"{'Metric':<14}"
    for name,_,__,___ in models_data: header+=f" {name:>{W}}"
    header+=f" {new_name:>{W}}"
    for name,_,__,___ in models_data: header+=f"  {f'Δ vs {name}'[:W]:>{W}}"
    header+="  Best"; print(header); print(sep)
    for key,label,higher_better in METRICS_CFG:
        nv=new_overall.get(key,float("nan")); fmt=".0f" if key in ("MAE","RMSE") else (".4f" if key=="R2" else ".2f")
        row=f"{label:<14}"; cand={new_name:nv}
        for name,m,_,__ in models_data: v=m.get(key,float("nan")); cand[name]=v; row+=f" {v:{W}{fmt}}"
        row+=f" {nv:{W}{fmt}}"
        for name,m,_,__ in models_data: d=nv-m.get(key,0); row+=f"  {'+' if d>=0 else ''}{d:{W}{fmt}}"
        best=max(cand,key=lambda k:cand[k] if higher_better else -cand[k]); row+=f"  {best}"; print(row)
    print(sep)


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
    A_raw=build_feature_adj(X_tr,threshold=args.adj_threshold)
    n_edges=int((A_raw>0).sum().item()//2); density=float((A_raw>0).float().mean().item())
    print(f"  Nodes: {n_features}  Edges: {n_edges}  Density: {density:.4f}")

    model=DCRNNForecaster(
        n_features=n_features, hidden_size=args.hidden, n_layers=args.n_layers,
        K=args.k_diffusion, T_out=T_out, A_raw=A_raw, dropout=args.dropout,
    ).to(device)

    n_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel         : DCRNNForecaster")
    print(f"  Nodes (N)   : {n_features}   K={args.k_diffusion}   layers={args.n_layers}   hidden={args.hidden}")
    print(f"  Diffusion   : K+1={args.k_diffusion+1} hops forward + backward")
    print(f"  In          : (batch, {T_in}, {n_features})")
    print(f"  Out         : (batch, {T_out})")
    print(f"  Params      : {n_params:,}")

    criterion=nn.MSELoss()
    optimiser=torch.optim.Adam(model.parameters(),lr=args.lr)
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

    W=10; print(f"\n{'='*85}"); print("DCRNN — TEST SET METRICS"); print(f"{'='*85}")
    header=(f"{'Step':>5}  {'Combined%':>{W}}  {'MAPE%':>{W}}  {'MAE%':>{W}}  {'RMSE%':>{W}}  {'R²':>{W}}  {'MAE':>{W}}  {'RMSE':>{W}}")
    print(header); print("─"*len(header))
    per_step=[]
    for s in range(T_out):
        m=compute_metrics(y_true[:,s],y_pred[:,s]); per_step.append(m)
        print(f"{s+1:5d}  {m['Combined']:>{W}.2f}  {m['MAPE']:>{W}.2f}  {m['MAE_pct']:>{W}.2f}  {m['RMSE_pct']:>{W}.2f}  {m['R2']:>{W}.4f}  {m['MAE']:>{W}.0f}  {m['RMSE']:>{W}.0f}")
    overall=compute_metrics(y_true.flatten(),y_pred.flatten())
    print("─"*len(header))
    print(f"{'Avg':>5}  {overall['Combined']:>{W}.2f}  {overall['MAPE']:>{W}.2f}  {overall['MAE_pct']:>{W}.2f}  {overall['RMSE_pct']:>{W}.2f}  {overall['R2']:>{W}.4f}  {overall['MAE']:>{W}.0f}  {overall['RMSE']:>{W}.0f}")

    run_id=datetime.now().strftime("%Y%m%d_%H%M%S"); out_dir=f"src/outputs/dcrnn/{run_id}"
    os.makedirs(out_dir,exist_ok=True); torch.save(model.state_dict(),f"{out_dir}/model.pt")
    results={"run_id":run_id,"model":"DCRNNForecaster",
             "hparams":{"hidden":args.hidden,"n_layers":args.n_layers,"k_diffusion":args.k_diffusion,
                        "adj_threshold":args.adj_threshold,"dropout":args.dropout,"T_in":T_in,"T_out":T_out,
                        "n_features":n_features,"n_edges":n_edges,"batch_size":args.batch_size,"lr":args.lr},
             "training":{"best_epoch":best_epoch,"best_val_loss":round(best_val_loss,8),"total_epochs":len(train_losses)},
             "split_dates":split_meta,
             "test_metrics":{"overall":{k:round(v,4) for k,v in overall.items()},
                             "per_step":[{k:round(v,4) for k,v in m.items()} for m in per_step]}}
    with open(f"{out_dir}/results.json","w") as f: json.dump(results,f,indent=2)

    print(f"\nSaving plots...")
    plot_loss_curves(train_losses,val_losses,f"{out_dir}/loss_curves.png")
    plot_predictions(y_true,y_pred,overall,f"{out_dir}/test_predictions.png")
    plot_per_step_metrics(per_step,f"{out_dir}/per_step_metrics.png")

    print(f"\n{'='*85}"); print(f"DCRNN run complete  →  {out_dir}/"); print(f"{'='*85}")
    print(f"  Combined  : {overall['Combined']:.2f}%"); print(f"  MAPE      : {overall['MAPE']:.2f}%")
    print(f"  MAE%      : {overall['MAE_pct']:.2f}%    (raw MAE  = {overall['MAE']:.0f} riders)")
    print(f"  RMSE%     : {overall['RMSE_pct']:.2f}%   (raw RMSE = {overall['RMSE']:.0f} riders)")
    print(f"  R²        : {overall['R2']:.4f}")

    PRIOR_MODELS=[
        ("LSTM",       args.lstm_results,      "src/outputs/lstm",          "#2563eb"),
        ("BiLSTM",     args.bilstm_results,    "src/outputs/bilstm",        "#7c3aed"),
        ("TPA-LSTM",   args.tpalstm_results,   "src/outputs/tpa_lstm",      "#0891b2"),
        ("CNN-LSTM",   args.cnnlstm_results,   "src/outputs/cnn_lstm",      "#16a34a"),
        ("CNN-BiLSTM", args.cnnbilstm_results, "src/outputs/cnn_bilstm",    "#d97706"),
        ("ST-LSTM",    args.stlstm_results,    "src/outputs/st_lstm",       "#dc2626"),
        ("STGCN",      args.stgcn_results,     "src/outputs/stgcn",         "#10b981"),
        ("WaveNet",    args.wavenet_results,   "src/outputs/graph_wavenet", "#f472b6"),
    ]
    models_data=[]; comparison={}
    for name,path,model_dir,color in PRIOR_MODELS:
        key=name.lower().replace("-","_"); data=load_model_results(path,model_dir,name)
        if data:
            m_overall=data["test_metrics"]["overall"]; m_ps=data["test_metrics"]["per_step"]
            models_data.append((name,m_overall,m_ps,color)); comparison[key]={"run_id":data.get("run_id"),"overall":m_overall}
        else: comparison[key]={"run_id":None,"overall":None}
    models_data.append(("DCRNN",overall,per_step,"#38bdf8"))
    if len(models_data)>1:
        print_comparison_table(models_data[:-1],overall,"DCRNN")
        plot_comparison(models_data,out_path=f"{out_dir}/comparison_{len(models_data)}_way.png")
    comparison["dcrnn"]={"run_id":run_id,"overall":{k:round(v,4) for k,v in overall.items()}}
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
    print(f"  DCRNN vs naive     ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()
