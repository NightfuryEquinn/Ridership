"""
stgat.py  — Spatio-Temporal Graph Attention Network (STGAT)

Adapts the Graph Attention Network (Veličković et al., 2018) for
spatio-temporal ridership forecasting:

  Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., &
  Bengio, Y. (2018). "Graph Attention Networks." ICLR 2018.
  arXiv:1710.10903

Architecture (feature-as-node graph):
  The n_features input features are treated as N graph nodes. At each
  timestep t, node i has a scalar signal x[b, t, i].

  Step 1 — Node embedding: linear project each node's scalar value to
    a d_emb-dimensional vector (shared across time).
    (B, T, N) → (B, T, N, d_emb)

  Step 2 — Spatial attention (applied to every timestep simultaneously):
    For each time step t, run multi-head GAT over N nodes.
    Fully connected: each node attends to every other node.
    (B*T, N, d_emb) → GAT → (B*T, N, d_gat)  → reshape → (B, T, N, d_gat)

  Step 3 — Temporal encoding (per node):
    Flatten spatial+feature: (B, T, N*d_gat) → LSTM → (B, lstm_hidden)

  Step 4 — MLP head: (B, lstm_hidden) → (B, T_out)

Comparison:
  --lstm-results, --bilstm-results, --tpalstm-results,
  --cnnlstm-results, --cnnbilstm-results, --stlstm-results,
  --stgcn-results, --wavenet-results, --dcrnn-results
  All optional; each auto-detects the most recent run if omitted.

Usage:
  python stgat.py
  python stgat.py --d-emb 16 --n-heads 4 --n-gat-layers 2 --lstm-hidden 128
"""

import os
import json
import argparse
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
    p = argparse.ArgumentParser(description="STGAT forecaster with 10-way comparison")
    p.add_argument("--seq-dir",           default="data/sequences/lstm")
    p.add_argument("--d-emb",             type=int,   default=16,
                   help="Per-node initial embedding dimension")
    p.add_argument("--n-heads",           type=int,   default=4,
                   help="Number of attention heads in each GAT layer")
    p.add_argument("--n-gat-layers",      type=int,   default=2,
                   help="Number of stacked GAT layers")
    p.add_argument("--lstm-hidden",       type=int,   default=128,
                   help="LSTM hidden size for temporal encoding")
    p.add_argument("--lstm-layers",       type=int,   default=1,
                   help="Number of LSTM layers")
    p.add_argument("--dropout",           type=float, default=0.1)
    p.add_argument("--batch-size",        type=int,   default=8)
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
    p.add_argument("--dcrnn-results",     default=None)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Model components
# ══════════════════════════════════════════════════════════════════════════════

class GATLayer(nn.Module):
    """
    Multi-head Graph Attention Layer (Veličković et al., 2018).

    Fully connected attention — no predefined adjacency required; attention
    weights are learned from node features.

    Input  : x (B, N, C_in)
    Output : (B, N, n_heads * d_k)   [concatenated heads]
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,   # total output channels = n_heads * d_k
        n_heads:      int = 4,
        dropout:      float = 0.1,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        assert out_channels % n_heads == 0, \
            f"out_channels ({out_channels}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.d_k     = out_channels // n_heads

        # Shared linear projection applied to every node
        self.W = nn.Linear(in_channels, n_heads * self.d_k, bias=False)

        # Attention parameter: a ∈ R^{n_heads × 2*d_k}
        # e_ij = LeakyReLU( a_h^T [Wh_i || Wh_j] )  for each head h
        self.a = nn.Parameter(
            torch.empty(n_heads, 2 * self.d_k).uniform_(-0.01, 0.01)
        )

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, C_in)
        """
        B, N, _ = x.shape
        H, d_k  = self.n_heads, self.d_k

        # Linear transform: (B, N, H*d_k)
        Wh = self.W(x).view(B, N, H, d_k)       # (B, N, H, d_k)

        # Attention: e_ij for all pairs (i, j) using broadcasting
        # Wh_i: (B, N_i, 1, H, d_k)  Wh_j: (B, 1, N_j, H, d_k)
        Wh_i = Wh.unsqueeze(2)                  # (B, N, 1, H, d_k)
        Wh_j = Wh.unsqueeze(1)                  # (B, 1, N, H, d_k)

        # Concatenate along last dim → (B, N, N, H, 2*d_k)
        cat_ij = torch.cat([
            Wh_i.expand(B, N, N, H, d_k),
            Wh_j.expand(B, N, N, H, d_k),
        ], dim=-1)

        # Attention score: dot with a → (B, N, N, H)
        # self.a: (H, 2*d_k) → unsqueeze for broadcast
        e = (cat_ij * self.a[None, None, None, :, :]).sum(-1)
        e = self.leaky_relu(e)

        # Softmax over source nodes j (dim=2)
        alpha = F.softmax(e, dim=2)              # (B, N, N, H)
        alpha = self.dropout(alpha)

        # Aggregate: each node i receives messages from all j
        # alpha: (B, N_i, N_j, H)  Wh_j: (B, N_i, N_j, H, d_k)
        Wh_j_exp = Wh.unsqueeze(1).expand(B, N, N, H, d_k)
        out = (alpha.unsqueeze(-1) * Wh_j_exp).sum(dim=2)  # (B, N, H, d_k)
        out = out.reshape(B, N, H * d_k)                    # (B, N, H*d_k)

        return F.elu(out)


class STGATForecaster(nn.Module):
    """
    Spatio-Temporal Graph Attention Network forecaster.

    Steps:
      1. Per-node linear embedding (shared across time)
      2. Multi-layer GAT at every timestep simultaneously
      3. LSTM over time (treating flattened node-features as input)
      4. MLP head → T_out

    forward(x) only takes x — no external adjacency required (fully learned).

    Input  : (B, T_in, n_features)
    Output : (B, T_out)
    """

    def __init__(
        self,
        n_features:  int,
        d_emb:       int,
        n_heads:     int,
        n_gat_layers: int,
        lstm_hidden: int,
        lstm_layers: int,
        T_out:       int,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features

        # ── Node embedding (1 → d_emb per node) ──────────────────────────────
        self.node_embed = nn.Linear(1, d_emb)

        # ── GAT layers ────────────────────────────────────────────────────────
        # Each GAT layer: d_gat → d_gat, total out = n_heads * (d_emb//n_heads or d_emb)
        # We keep d_gat = n_heads * d_emb for all GAT layers
        d_gat = n_heads * d_emb
        self.gat_layers = nn.ModuleList()
        for i in range(n_gat_layers):
            c_in = d_emb if i == 0 else d_gat
            self.gat_layers.append(GATLayer(c_in, d_gat, n_heads, dropout))

        self.gat_dropout = nn.Dropout(dropout)

        # ── Temporal LSTM (input = flattened N * d_gat per timestep) ─────────
        lstm_in = n_features * d_gat
        self.lstm = nn.LSTM(
            input_size  = lstm_in,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )

        # ── MLP head ──────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden // 2, T_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T_in, N)
        """
        B, T, N = x.shape

        # ── Node embedding ────────────────────────────────────────────────────
        # x: (B, T, N) → (B*T, N, 1) → embed → (B*T, N, d_emb)
        x_flat = x.reshape(B * T, N, 1)
        h = self.node_embed(x_flat)              # (B*T, N, d_emb)

        # ── GAT layers ────────────────────────────────────────────────────────
        for gat in self.gat_layers:
            h = gat(h)                           # (B*T, N, d_gat)
            h = self.gat_dropout(h)

        # ── Reshape for LSTM ──────────────────────────────────────────────────
        d_gat = h.shape[-1]
        h = h.reshape(B, T, N * d_gat)          # (B, T, N*d_gat)

        # ── LSTM over temporal dim ────────────────────────────────────────────
        _, (h_n, _) = self.lstm(h)
        h_final = h_n[-1]                        # (B, lstm_hidden)

        return self.head(h_final)                # (B, T_out)



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
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(train_losses, val_losses, out_path):
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(train_losses,label="Train MSE",linewidth=1.5,color="#dc2626")
    ax.plot(val_losses,label="Val MSE",linewidth=1.5,color="#f97316",linestyle="--")
    ax.set_xlabel("Epoch");ax.set_ylabel("MSE loss (scaled)");ax.set_title("STGAT — Training curves")
    ax.legend();ax.grid(alpha=0.3);fig.tight_layout();fig.savefig(out_path,dpi=150);plt.close(fig)
    print(f"  Saved: {out_path}")

def plot_predictions(y_true, y_pred, metrics, out_path):
    N,T_out=y_true.shape; idx_full=np.arange(N); zoom_n=min(60,N); idx_zoom=np.arange(N-zoom_n,N)
    actual_s1=y_true[:,0]; predicted_s1=y_pred[:,0]; pred_min=y_pred.min(axis=1); pred_max=y_pred.max(axis=1)
    C_ACT="#1d4ed8"; C_PRED="#fb923c"; C_BAND="#ffedd5"; C_MID="#0891b2"; C_LAST="#7c3aed"
    fig=plt.figure(figsize=(14,8)); gs=gridspec.GridSpec(2,1,hspace=0.48)
    ax1=fig.add_subplot(gs[0])
    ax1.fill_between(idx_full,pred_min,pred_max,alpha=0.22,color=C_BAND,label=f"Forecast spread (step 1–{T_out})")
    ax1.plot(idx_full,actual_s1,color=C_ACT,linewidth=1.5,label="Actual",zorder=4)
    ax1.plot(idx_full,predicted_s1,color=C_PRED,linewidth=1.2,linestyle="--",alpha=0.88,label="STGAT predicted (step 1)",zorder=5)
    ann=(f"Combined = {metrics['Combined']:.2f}%\nMAPE     = {metrics['MAPE']:.2f}%\n"
         f"MAE%     = {metrics['MAE_pct']:.2f}%\nRMSE%    = {metrics['RMSE_pct']:.2f}%\n"
         f"R²       = {metrics['R2']:.4f}\nMAE      = {metrics['MAE']:.0f} riders\nRMSE     = {metrics['RMSE']:.0f} riders")
    ax1.text(0.01,0.97,ann,transform=ax1.transAxes,fontsize=8,verticalalignment="top",fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.45",facecolor="white",edgecolor="#d1d5db",alpha=0.92))
    ax1.set_title("STGAT — Test set: Actual vs Predicted (full period)",fontsize=11,fontweight="bold")
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
    ax1.set_title("STGAT — Per-horizon metrics",fontweight="bold");ax1.legend(fontsize=8);ax1.grid(axis="y",alpha=0.3)
    ax2.plot(steps,r2,marker="o",color="#dc2626",linewidth=1.8,markersize=5)
    ax2.axhline(0,color="#9ca3af",linewidth=0.8,linestyle="--");ax2.axhline(1,color="#16a34a",linewidth=0.8,linestyle=":")
    ax2.set_ylim(min(min(r2)-0.05,-0.1),1.08);ax2.set_ylabel("R²");ax2.set_title("STGAT — R² per horizon",fontweight="bold");ax2.grid(alpha=0.3)
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

    d_gat=args.n_heads*args.d_emb
    model=STGATForecaster(
        n_features=n_features, d_emb=args.d_emb, n_heads=args.n_heads,
        n_gat_layers=args.n_gat_layers, lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers, T_out=T_out, dropout=args.dropout,
    ).to(device)

    n_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel         : STGATForecaster")
    print(f"  Nodes (N)   : {n_features}   d_emb={args.d_emb}   n_heads={args.n_heads}   d_gat={d_gat}")
    print(f"  GAT layers  : {args.n_gat_layers}   LSTM hidden={args.lstm_hidden}   LSTM layers={args.lstm_layers}")
    print(f"  In          : (batch, {T_in}, {n_features})")
    print(f"  Out         : (batch, {T_out})")
    print(f"  Params      : {n_params:,}")

    criterion=nn.MSELoss()
    optimiser=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-5)
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

    W=10; print(f"\n{'='*85}"); print("STGAT — TEST SET METRICS"); print(f"{'='*85}")
    header=(f"{'Step':>5}  {'Combined%':>{W}}  {'MAPE%':>{W}}  {'MAE%':>{W}}  {'RMSE%':>{W}}  {'R²':>{W}}  {'MAE':>{W}}  {'RMSE':>{W}}")
    print(header); print("─"*len(header))
    per_step=[]
    for s in range(T_out):
        m=compute_metrics(y_true[:,s],y_pred[:,s]); per_step.append(m)
        print(f"{s+1:5d}  {m['Combined']:>{W}.2f}  {m['MAPE']:>{W}.2f}  {m['MAE_pct']:>{W}.2f}  {m['RMSE_pct']:>{W}.2f}  {m['R2']:>{W}.4f}  {m['MAE']:>{W}.0f}  {m['RMSE']:>{W}.0f}")
    overall=compute_metrics(y_true.flatten(),y_pred.flatten())
    print("─"*len(header))
    print(f"{'Avg':>5}  {overall['Combined']:>{W}.2f}  {overall['MAPE']:>{W}.2f}  {overall['MAE_pct']:>{W}.2f}  {overall['RMSE_pct']:>{W}.2f}  {overall['R2']:>{W}.4f}  {overall['MAE']:>{W}.0f}  {overall['RMSE']:>{W}.0f}")

    run_id=datetime.now().strftime("%Y%m%d_%H%M%S"); out_dir=f"src/outputs/stgat/{run_id}"
    os.makedirs(out_dir,exist_ok=True); torch.save(model.state_dict(),f"{out_dir}/model.pt")
    results={"run_id":run_id,"model":"STGATForecaster",
             "hparams":{"d_emb":args.d_emb,"n_heads":args.n_heads,"n_gat_layers":args.n_gat_layers,
                        "lstm_hidden":args.lstm_hidden,"lstm_layers":args.lstm_layers,"dropout":args.dropout,
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

    print(f"\n{'='*85}"); print(f"STGAT run complete  →  {out_dir}/"); print(f"{'='*85}")
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
        ("DCRNN",      args.dcrnn_results,     "src/outputs/dcrnn",         "#38bdf8"),
    ]
    models_data=[]; comparison={}
    for name,path,model_dir,color in PRIOR_MODELS:
        key=name.lower().replace("-","_"); data=load_model_results(path,model_dir,name)
        if data:
            m_overall=data["test_metrics"]["overall"]; m_ps=data["test_metrics"]["per_step"]
            models_data.append((name,m_overall,m_ps,color)); comparison[key]={"run_id":data.get("run_id"),"overall":m_overall}
        else: comparison[key]={"run_id":None,"overall":None}
    models_data.append(("STGAT",overall,per_step,"#fb923c"))
    if len(models_data)>1:
        print_comparison_table(models_data[:-1],overall,"STGAT")
        plot_comparison(models_data,out_path=f"{out_dir}/comparison_{len(models_data)}_way.png")
    comparison["stgat"]={"run_id":run_id,"overall":{k:round(v,4) for k,v in overall.items()}}
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
    print(f"  STGAT vs naive     ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()
