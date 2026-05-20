"""
stminet.py  —  STMI-Net: Sequential Hybrid ST-LSTM + MTGNN + Informer

STMI-Net is a three-stage sequential hybrid architecture for multivariate
spatio-temporal transit ridership forecasting.  Each stage addresses a
distinct modelling challenge that the component models solve individually but
cannot address together:

  Stage 1 — ST-LSTM Encoder
    Dual-stream architecture that decouples "when" from "what":
      • Temporal stream: multi-layer LSTM applied to the d_model-projected
        look-back sequence; returns ALL T_in hidden states (not just the final
        hidden) so the full temporal trajectory is preserved for downstream
        graph and attention stages.
      • Spatial stream: a shared-weight MLP applied independently to each
        timestep's raw feature vector F; captures persistent cross-feature
        interactions (e.g. how GTFS accessibility and population combine) that
        are time-invariant within the window.
    The two streams are fused via a learned per-timestep gate:
        gate  = sigmoid( W · (h_proj + sp_proj) )
        fused = gate · h_proj + (1 − gate) · sp_proj
    A residual from the linear input projection prevents gradient vanishing
    through deep stacks.  LayerNorm is applied after fusion.
    Output: (B, T_in, d_model)

  Stage 2 — MTGNN Graph Refinement
    Treats the d_model latent channels from Stage 1 as N_graph = d_model
    graph nodes, learning inter-representation dependencies adaptively:
      • Asymmetric adjacency A = softmax(ReLU(tanh(α)·(M1·M2ᵀ − M2·M1ᵀ)))
        learned end-to-end from two trainable embeddings M1, M2 ∈ ℝ^{D×d_emb}.
      • n_graph_layers MTGNN blocks, each containing:
          - 4-branch dilated-inception temporal conv (kernels 1,3,5,7, GLU)
          - Mix-hop graph conv (0- to d_hop-hop neighbourhood aggregation)
          - Skip connection + BatchNorm residual
      • Skip accumulations are summed, mean-pooled over the node axis (keeping
        T_in intact), then projected back to d_model.
      • Residual from Stage 1 output prevents over-smoothing.
    d_hop ≤ 1 is recommended to avoid representation collapse on the small
    N_graph = d_model graph.
    Output: (B, T_in, d_model)

  Stage 3 — Informer Global Temporal Reasoning
    ProbSparse Informer encoder-decoder for O(L log L) long-range attention:
      • Input already in d_model space — no feature re-projection needed.
      • Sinusoidal positional embeddings added before the encoder.
      • Encoder: e_layers ProbSparse attention layers with distilling
        (ConvLayer halves sequence length between each pair of layers).
      • Decoder: generative start token = last T_label steps of Stage-2 output
        concatenated with T_out zero-padding, all in d_model space.
        d_layers full cross-attention decoder layers.
      • Output projection: Linear(d_model → 1) on the last T_out decoder steps.
    This stage captures long-range periodic dependencies — weekly ridership
    cycles, school holidays, public events — invisible to local LSTM windows.
    Output: (B, T_out)

Sequential data-flow:
  X (B, T_in, F)
    │ input_proj  Linear(F → d_model)
  ┌─┴──────────────────────────────────────────────────────────┐
  │  Stage 1: ST-LSTM Encoder                                  │
  │    LSTM(d_model → hidden_st, all T steps)                  │
  │  + shared spatial MLP(F → sp_hidden) per timestep          │
  │    gated fusion + residual + LayerNorm                     │
  └────────────────────────────────────────────────────────────┘
    │  (B, T_in, d_model)
  ┌─┴──────────────────────────────────────────────────────────┐
  │  Stage 2: MTGNN Graph Module                               │
  │    d_model latent dims treated as N_graph nodes            │
  │    Adaptive asymmetric adjacency A (N_graph × N_graph)     │
  │    n_graph_layers × (Inception conv + MixHop graph conv)   │
  │    mean-pool over N_graph + skip_proj + residual + LN      │
  └────────────────────────────────────────────────────────────┘
    │  (B, T_in, d_model)
  ┌─┴──────────────────────────────────────────────────────────┐
  │  Stage 3: Informer Global Temporal Reasoning               │
  │    + sinusoidal positional encoding                        │
  │    ProbSparse encoder × e_layers (with distilling)         │
  │    generative decoder × d_layers (start token from Stage 2)│
  │    out_proj Linear(d_model → 1)                            │
  └────────────────────────────────────────────────────────────┘
    │  (B, T_out)

Key design choices vs the component models:
  • ST-LSTM returns a full T_in sequence (vs just h_T in the standalone model)
    so that both the graph module and the Informer can exploit temporal order.
  • MTGNN operates in latent space (N_graph = d_model) rather than raw-feature
    space; the graph learns dependencies between learned representations.
  • Informer decoder start token drawn from Stage-2 output — already in
    d_model space — eliminating the feature re-projection step.
  • AMP (fp16 + GradScaler) enabled on CUDA to fit within RTX 4050 6 GB VRAM.
  • All residuals and LayerNorms guard against gradient issues across the
    three-stage depth.

Comparison:
  Auto-compares against all 15 base models.  Prior results are auto-detected
  from src/outputs/<model_name>/ if not provided via CLI.

Usage:
  python src/models/SOTA/stminet.py
  python src/models/SOTA/stminet.py --d-model 64 --hidden-st 64 --n-heads 4
  python src/models/SOTA/stminet.py --lookback 28 --epochs 200 --patience 20
  python src/models/SOTA/stminet.py --n-graph-layers 1 --d-hop 1  # lighter graph
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
    p = argparse.ArgumentParser(
        description="STMI-Net — ST-LSTM + MTGNN + Informer hybrid (16-way comparison)"
    )
    # Data
    p.add_argument("--seq-dir",            default=None,
                   help="Directory with X/y .npy splits (auto-resolved from --lookback)")
    p.add_argument("--lookback",           type=int,   default=14, choices=[14, 28, 56],
                   help="Look-back window; auto-selects seq-dir when --seq-dir not set")
    # ── Architecture ──────────────────────────────────────────────────────────
    p.add_argument("--d-model",            type=int,   default=128,
                   help="Unified model dimension shared across all three stages")
    # Stage 1 — ST-LSTM
    p.add_argument("--hidden-st",          type=int,   default=128,
                   help="LSTM hidden size in ST-LSTM encoder (Stage 1)")
    p.add_argument("--sp-hidden",          type=int,   default=64,
                   help="Spatial MLP output dim in ST-LSTM encoder (Stage 1)")
    p.add_argument("--lstm-layers",        type=int,   default=2,
                   help="Stacked LSTM layers in Stage 1")
    # Stage 2 — MTGNN
    p.add_argument("--hidden-g",           type=int,   default=64,
                   help="MTGNN internal channel width (Stage 2)")
    p.add_argument("--n-graph-layers",     type=int,   default=2,
                   help="Number of MTGNN blocks (Stage 2); 1–2 to avoid over-smoothing")
    p.add_argument("--d-emb",              type=int,   default=16,
                   help="Node embedding dim for MTGNN adaptive adjacency (Stage 2)")
    p.add_argument("--d-hop",              type=int,   default=1,
                   help="Mix-hop propagation order (Stage 2); keep ≤2 to avoid collapse")
    # Stage 3 — Informer
    p.add_argument("--n-heads",            type=int,   default=4,
                   help="Informer attention heads (Stage 3)")
    p.add_argument("--e-layers",           type=int,   default=2,
                   help="Informer encoder layers (Stage 3)")
    p.add_argument("--d-layers",           type=int,   default=1,
                   help="Informer decoder layers (Stage 3)")
    p.add_argument("--d-ff",               type=int,   default=256,
                   help="Informer FFN inner dimension (Stage 3)")
    p.add_argument("--factor",             type=int,   default=5,
                   help="ProbSparse top-k factor c: k = c·⌈ln(L_K)⌉ (Stage 3)")
    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--dropout",            type=float, default=0.1)
    p.add_argument("--batch-size",         type=int,   default=32)
    p.add_argument("--epochs",             type=int,   default=150)
    p.add_argument("--lr",                 type=float, default=1e-3)
    p.add_argument("--weight-decay",       type=float, default=1e-4)
    p.add_argument("--patience",           type=int,   default=15)
    p.add_argument("--warmup-epochs",      type=int,   default=5,
                   help="Linear LR warm-up epochs before ReduceLROnPlateau kicks in")
    p.add_argument("--loss",               default="huber", choices=["mse", "huber", "mae"])
    p.add_argument("--device",             default="auto", help="cpu | cuda | mps | auto")
    p.add_argument("--seed",               type=int,   default=42)
    # ── Prior model result paths (all optional; auto-detected if omitted) ─────
    p.add_argument("--lstm-results",       default=None)
    p.add_argument("--bilstm-results",     default=None)
    p.add_argument("--tpalstm-results",    default=None)
    p.add_argument("--cnnlstm-results",    default=None)
    p.add_argument("--cnnbilstm-results",  default=None)
    p.add_argument("--stlstm-results",     default=None)
    p.add_argument("--stgcn-results",      default=None)
    p.add_argument("--mtgnn-results",      default=None)
    p.add_argument("--stsgcn-results",     default=None)
    p.add_argument("--stfgnn-results",     default=None)
    p.add_argument("--pdrstgcn-results",   default=None)
    p.add_argument("--astgcn-results",     default=None)
    p.add_argument("--tft-results",        default=None)
    p.add_argument("--autoformer-results", default=None)
    p.add_argument("--informer-results",   default=None)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Model building blocks
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 2 helpers (MTGNN components) ───────────────────────────────────────

class InceptionBlock(nn.Module):
    """
    4-branch dilated-inception temporal conv with GLU gating.

    Kernel sizes [1, 3, 5, 7] — odd so symmetric padding preserves T.
    Each branch produces out_ch×2 channels; tanh(h1)·sigmoid(h2) gives out_ch.
    The four branch outputs are summed.

    Input  : (B, C, N, T)
    Output : (B, out_ch, N, T)
    """
    KERNELS = [1, 3, 5, 7]

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch * 2, (1, k), padding=(0, (k - 1) // 2))
            for k in self.KERNELS
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        h = sum(conv(x)[..., :T] for conv in self.convs)   # (B, 2*out_ch, N, T)
        h1, h2 = h.chunk(2, dim=1)
        return torch.tanh(h1) * torch.sigmoid(h2)           # (B, out_ch, N, T)


class MixHopConv(nn.Module):
    """
    Mix-hop graph convolution: aggregates 0- to d_hop-hop neighbourhoods.

    out = Σ_{k=0}^{d_hop}  A^k @ x @ W_k    (per-hop independent linear)

    Input  : (B, C, N, T)  +  A (N, N)
    Output : (B, out_ch, N, T)
    """

    def __init__(self, in_ch: int, out_ch: int, d_hop: int = 1):
        super().__init__()
        self.d_hop     = d_hop
        self.hop_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, (1, 1)) for _ in range(d_hop + 1)
        ])
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = x; out = None
        for k, conv in enumerate(self.hop_convs):
            if k > 0:
                h = torch.einsum("nm,bcmt->bcnt", A, h)
            feat = conv(h)
            out  = feat if out is None else out + feat
        return self.bn(F.relu(out))


class MTGNNBlock(nn.Module):
    """One MTGNN block: InceptionBlock → MixHopConv → skip + residual."""

    def __init__(self, hidden: int, skip_ch: int, d_hop: int, dropout: float):
        super().__init__()
        self.inception  = InceptionBlock(hidden, hidden)
        self.graph_conv = MixHopConv(hidden, hidden, d_hop)
        self.skip_conv  = nn.Conv2d(hidden, skip_ch, (1, 1))
        self.res_conv   = nn.Conv2d(hidden, hidden,  (1, 1))
        self.bn         = nn.BatchNorm2d(hidden)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        """
        x    : (B, hidden, N, T)
        Returns: (h: (B, hidden, N, T), skip: (B, skip_ch, N, T))
        """
        residual = x
        h    = self.inception(x)
        h    = self.graph_conv(h, A)
        h    = self.dropout(h)
        skip = self.skip_conv(h)
        h    = self.bn(h + self.res_conv(residual))
        return h, skip


# ── Stage 3 helpers (Informer components) ────────────────────────────────────

class ProbSparseAttention(nn.Module):
    """
    ProbSparse self-attention (Zhou et al., 2021).  O(L log L).

    Selects the top-u queries (by max-minus-mean sparsity score over a sample
    of keys) and computes full attention only for those queries.  All other
    queries receive the mean of V.

    For short sequences (L ≤ 14) this gracefully degenerates to full attention.
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

    def _prob_qk(self, Q: torch.Tensor, K: torch.Tensor, sample_k: int, n_top: int):
        B, H, L_Q, D = Q.shape
        L_K = K.shape[2]
        idx    = torch.randint(L_K, (B, H, sample_k), device=Q.device)
        K_s    = K.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, D))
        Q_K_s  = torch.einsum("bhld,bhsd->bhls", Q, K_s)
        M      = Q_K_s.max(-1).values - Q_K_s.mean(-1)
        top    = M.topk(n_top, dim=-1, sorted=False).indices
        Q_r    = Q.gather(2, top.unsqueeze(-1).expand(-1, -1, -1, D))
        Q_K    = torch.einsum("bhld,bhsd->bhls", Q_r, K) * self.scale
        return Q_K, top

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, L_Q, _ = Q.shape
        L_K = K.shape[1]
        Q = self.q_proj(Q).reshape(B, L_Q, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K = self.k_proj(K).reshape(B, L_K, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        V = self.v_proj(V).reshape(B, L_K, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        sample_k = max(1, int(math.log(max(L_K, 2))))
        n_top    = min(L_Q, self.factor * max(1, int(math.ceil(math.log(max(L_K, 2))))))
        V_mean   = V.mean(dim=2, keepdim=True).expand(-1, -1, L_Q, -1)
        out      = V_mean.clone()
        if n_top < L_Q:
            scores, top_idx = self._prob_qk(Q, K, sample_k, n_top)
            attn = self.dropout(torch.softmax(scores, dim=-1))
            v_top = torch.einsum("bhls,bhsd->bhld", attn, V)
            out.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head), v_top)
        else:
            scores = torch.einsum("bhld,bhsd->bhls", Q, K) * self.scale
            out    = self.dropout(
                torch.einsum("bhls,bhsd->bhld", torch.softmax(scores, dim=-1), V)
            )
        out = out.permute(0, 2, 1, 3).reshape(B, L_Q, -1)
        return self.dropout(self.out_proj(out))


class ConvLayer(nn.Module):
    """Informer distilling: Conv1d + ELU + MaxPool1d(stride=2) — halves sequence length."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1, padding_mode="circular")
        self.norm = nn.LayerNorm(d_model)
        self.act  = nn.ELU()
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv(x.permute(0, 2, 1)))   # (B, d_model, L) → conv → ELU
        x = self.pool(x)                               # (B, d_model, L//2)
        return self.norm(x.permute(0, 2, 1))           # (B, L//2, d_model)


class InformerEncoderLayer(nn.Module):
    """ProbSparseAttn + FFN with pre-norm residuals."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, factor: int, dropout: float):
        super().__init__()
        self.attn  = ProbSparseAttention(d_model, n_heads, factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.attn(x, x, x)))
        return self.norm2(x + self.drop(self.ffn(x)))


class InformerDecoderLayer(nn.Module):
    """Full MHA self-attn + cross-attn + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        sa, _ = self.self_attn(x, x, x)
        x     = self.norm1(x + self.drop(sa))
        ca, _ = self.cross_attn(x, enc_out, enc_out)
        x     = self.norm2(x + self.drop(ca))
        return self.norm3(x + self.drop(self.ffn(x)))


# ══════════════════════════════════════════════════════════════════════════════
# The three stage modules
# ══════════════════════════════════════════════════════════════════════════════

class STLSTMEncoder(nn.Module):
    """
    Stage 1: Dual-stream (temporal LSTM + spatial MLP) encoder.

    Returns the full sequence of fused representations (B, T_in, d_model),
    not just the final hidden state, so that downstream stages can exploit
    the temporal trajectory.

    Input:  (B, T_in, n_features)
    Output: (B, T_in, d_model)
    """

    def __init__(
        self,
        n_features: int,
        d_model:    int,
        hidden_st:  int,
        sp_hidden:  int,
        lstm_layers:int,
        dropout:    float,
    ):
        super().__init__()
        # Project raw features into model dimension
        self.input_proj = nn.Linear(n_features, d_model)

        # ── Temporal stream (LSTM) ────────────────────────────────────────────
        self.lstm      = nn.LSTM(
            d_model, hidden_st, lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )
        self.temp_proj = nn.Linear(hidden_st, d_model)

        # ── Spatial stream (shared-weight MLP per timestep) ───────────────────
        mid           = max(sp_hidden, n_features // 2)
        self.sp_mlp   = nn.Sequential(
            nn.Linear(n_features, mid), nn.ReLU(),
            nn.Linear(mid, sp_hidden), nn.ReLU(),
        )
        self.sp_proj  = nn.Linear(sp_hidden, d_model)

        # ── Gated fusion ──────────────────────────────────────────────────────
        self.gate_fc  = nn.Linear(d_model, d_model)

        # ── Output normalisation ──────────────────────────────────────────────
        self.norm     = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, F)
        B, T, F = x.shape
        x_proj  = self.input_proj(x)                  # (B, T, d_model)

        # Temporal: all T hidden states
        h_seq, _ = self.lstm(x_proj)                  # (B, T, hidden_st)
        h_proj   = self.temp_proj(h_seq)               # (B, T, d_model)

        # Spatial: shared MLP applied per (B*T) token of raw features
        sp_proj  = self.sp_proj(
            self.sp_mlp(x.reshape(B * T, F)).reshape(B, T, -1)
        )                                              # (B, T, d_model)

        # Gate: how much to weight temporal vs spatial at each step
        gate  = torch.sigmoid(self.gate_fc(h_proj + sp_proj))  # (B, T, d_model)
        fused = gate * h_proj + (1 - gate) * sp_proj           # (B, T, d_model)

        # Residual connection from input projection + normalisation
        return self.norm(self.drop(fused) + x_proj)   # (B, T, d_model)


class MTGNNGraphModule(nn.Module):
    """
    Stage 2: Adaptive graph convolution over d_model latent-feature nodes.

    Treats the d_model channels from Stage 1 as N_graph = d_model graph nodes.
    An asymmetric directed adjacency is learned from M1, M2 embeddings.
    The temporal dimension T_in is preserved (only N_graph is pooled), so the
    output sequence can feed directly into Stage 3.

    Input:  (B, T_in, d_model)   — Stage 1 output
    Output: (B, T_in, d_model)   — spatially refined, residual added
    """

    def __init__(
        self,
        d_model:  int,
        hidden_g: int,
        n_layers: int,
        d_emb:    int,
        d_hop:    int,
        dropout:  float,
    ):
        super().__init__()
        # Adaptive asymmetric adjacency embeddings  (N_graph = d_model)
        self.M1    = nn.Parameter(torch.empty(d_model, d_emb))
        self.M2    = nn.Parameter(torch.empty(d_model, d_emb))
        self.alpha = nn.Parameter(torch.tensor(3.0))
        nn.init.xavier_uniform_(self.M1)
        nn.init.xavier_uniform_(self.M2)

        # Input projection: 1 scalar channel → hidden_g
        self.start_conv = nn.Conv2d(1, hidden_g, (1, 1))

        # MTGNN blocks (skip_ch = hidden_g to keep dimensions consistent)
        self.blocks = nn.ModuleList([
            MTGNNBlock(hidden_g, hidden_g, d_hop, dropout) for _ in range(n_layers)
        ])

        # Project accumulated skip from hidden_g → d_model (for residual match)
        self.skip_proj = nn.Conv1d(hidden_g, d_model, 1)

        # Residual + norm
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, d_model)
        residual = x

        # Adaptive asymmetric adjacency  A: (d_model, d_model)
        A = torch.softmax(
            torch.relu(
                torch.tanh(self.alpha) * (self.M1 @ self.M2.t() - self.M2 @ self.M1.t())
            ),
            dim=-1,
        )

        # Reshape: (B, T, D) → (B, D, T) → (B, 1, D, T) → start_conv → (B, hg, D, T)
        h = self.start_conv(x.permute(0, 2, 1).unsqueeze(1))

        skip_acc = 0
        for block in self.blocks:
            h, skip  = block(h, A)           # h: (B, hg, D, T), skip: (B, hg, D, T)
            skip_acc = skip_acc + skip

        # Pool over node dim (D = d_model), keep T
        g = F.relu(skip_acc).mean(dim=2)     # (B, hg, T)
        g = self.skip_proj(g)                # (B, d_model, T)
        g = g.permute(0, 2, 1)              # (B, T, d_model)

        # Residual + normalisation
        return self.norm(self.drop(g) + residual)


class InformerModule(nn.Module):
    """
    Stage 3: ProbSparse Informer encoder-decoder for global temporal reasoning.

    The input sequence is already in d_model space — no feature projection
    is needed.  The generative decoder start token is built from the last
    T_label = T_in // 2 steps of the Stage-2 output concatenated with T_out
    zero-padded future slots (all in d_model space).

    Input:  (B, T_in, d_model)   — Stage 2 output
    Output: (B, T_out)
    """

    def __init__(
        self,
        d_model:  int,
        T_in:     int,
        T_out:    int,
        n_heads:  int,
        e_layers: int,
        d_layers: int,
        d_ff:     int,
        factor:   int,
        dropout:  float,
    ):
        super().__init__()
        self.T_out   = T_out
        self.T_label = T_in // 2

        # Fixed sinusoidal PE (cached as buffers)
        self.register_buffer("enc_pe", self._sin_pe(T_in, d_model))
        T_dec = self.T_label + T_out
        self.register_buffer("dec_pe", self._sin_pe(T_dec, d_model))

        # Encoder: ProbSparse layers + distilling (e_layers - 1 ConvLayers)
        self.enc_layers  = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, factor, dropout)
            for _ in range(e_layers)
        ])
        self.conv_layers = nn.ModuleList([
            ConvLayer(d_model) for _ in range(e_layers - 1)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # Decoder: full self-attn + cross-attn layers
        self.dec_layers = nn.ModuleList([
            InformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(d_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)

        # Output: last T_out decoder steps → scalar per step
        self.out_proj = nn.Linear(d_model, 1)

    @staticmethod
    def _sin_pe(length: int, d_model: int) -> torch.Tensor:
        pe  = torch.zeros(length, d_model)
        pos = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        return pe.unsqueeze(0)   # (1, length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, d_model)
        B, T_in, D = x.shape

        # Generative decoder start token — reuse Stage-2 representations
        zeros  = torch.zeros(B, self.T_out, D, device=x.device, dtype=x.dtype)
        dec_in = torch.cat([x[:, -self.T_label:, :], zeros], dim=1)  # (B, T_label+T_out, D)

        # Encoder: PE + ProbSparse attn + distilling
        enc = x + self.enc_pe[:, :T_in, :]
        for i, enc_layer in enumerate(self.enc_layers):
            enc = enc_layer(enc)
            if i < len(self.conv_layers):
                enc = self.conv_layers[i](enc)           # halve sequence length
        enc = self.enc_norm(enc)

        # Decoder: PE + cross-attn over (possibly shortened) encoder memory
        T_dec = self.T_label + self.T_out
        dec   = dec_in + self.dec_pe[:, :T_dec, :]
        for dec_layer in self.dec_layers:
            dec = dec_layer(dec, enc)
        dec = self.dec_norm(dec)

        # Project last T_out decoder steps to scalar forecasts
        out = self.out_proj(dec[:, -self.T_out:, :])    # (B, T_out, 1)
        return out.squeeze(-1)                           # (B, T_out)


# ══════════════════════════════════════════════════════════════════════════════
# STMI-Net: full sequential hybrid
# ══════════════════════════════════════════════════════════════════════════════

class STMINet(nn.Module):
    """
    STMI-Net: Sequential hybrid of ST-LSTM → MTGNN → Informer.

    Input:  (B, T_in, n_features)
    Output: (B, T_out)

    Stage 1 (local encoding)    : STLSTMEncoder
    Stage 2 (spatial refinement): MTGNNGraphModule
    Stage 3 (global reasoning)  : InformerModule
    """

    def __init__(
        self,
        n_features:     int,
        T_in:           int,
        T_out:          int,
        d_model:        int   = 128,
        hidden_st:      int   = 128,
        sp_hidden:      int   = 64,
        lstm_layers:    int   = 2,
        hidden_g:       int   = 64,
        n_graph_layers: int   = 2,
        d_emb:          int   = 16,
        d_hop:          int   = 1,
        n_heads:        int   = 4,
        e_layers:       int   = 2,
        d_layers:       int   = 1,
        d_ff:           int   = 256,
        factor:         int   = 5,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.stage1 = STLSTMEncoder(
            n_features, d_model, hidden_st, sp_hidden, lstm_layers, dropout
        )
        self.stage2 = MTGNNGraphModule(
            d_model, hidden_g, n_graph_layers, d_emb, d_hop, dropout
        )
        self.stage3 = InformerModule(
            d_model, T_in, T_out, n_heads, e_layers, d_layers, d_ff, factor, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: local spatio-temporal encoding  →  (B, T_in, d_model)
        x = self.stage1(x)
        # Stage 2: adaptive graph spatial refinement  →  (B, T_in, d_model)
        x = self.stage2(x)
        # Stage 3: global long-range temporal reasoning  →  (B, T_out)
        return self.stage3(x)


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
# Training helpers (AMP-aware)
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimiser, criterion, scaler, use_amp) -> float:
    model.train()
    total, n_skip = 0.0, 0
    for X_b, y_b in loader:
        optimiser.zero_grad()
        with autocast(enabled=use_amp):
            loss = criterion(model(X_b), y_b)
        if not torch.isfinite(loss):
            n_skip += 1
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()
        total += loss.item() * X_b.size(0)
    if n_skip:
        print(f"  [WARN] {n_skip} batch(es) skipped — non-finite loss")
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, use_amp) -> float:
    model.eval()
    total = 0.0
    for X_b, y_b in loader:
        with autocast(enabled=use_amp):
            total += criterion(model(X_b), y_b).item() * X_b.size(0)
    return total / len(loader.dataset)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

C_STMI  = "#14b8a6"   # teal-500 — unique in the comparison palette
C_STMI2 = "#0d9488"   # teal-600 (darker variant for zoomed lines)

def plot_loss_curves(train_losses, val_losses, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train Loss", linewidth=1.5, color=C_STMI)
    ax.plot(val_losses,   label="Val Loss",   linewidth=1.5, color="#f97316",
            linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (scaled)")
    ax.set_title("STMI-Net — Training curves")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_predictions(y_true, y_pred, metrics, out_path: str) -> None:
    N, T_out = y_true.shape
    idx_full = np.arange(N)
    zoom_n   = min(60, N)
    idx_zoom = np.arange(N - zoom_n, N)
    actual_s1    = y_true[:, 0]
    predicted_s1 = y_pred[:, 0]
    pred_min = y_pred.min(axis=1); pred_max = y_pred.max(axis=1)

    C_ACT  = "#1d4ed8"
    C_BAND = "#ccfbf1"   # teal-100
    C_MID  = "#0891b2"
    C_LAST = "#7c3aed"

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.48)

    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(idx_full, pred_min, pred_max,
                     alpha=0.22, color=C_BAND,
                     label=f"Forecast spread (step 1–{T_out})")
    ax1.plot(idx_full, actual_s1,    color=C_ACT,  linewidth=1.5, label="Actual", zorder=4)
    ax1.plot(idx_full, predicted_s1, color=C_STMI, linewidth=1.2, linestyle="--",
             alpha=0.88, label="STMI-Net predicted (step 1)", zorder=5)
    ann = (
        f"Combined = {metrics['Combined']:.2f}%\n"
        f"MAPE     = {metrics['MAPE']:.2f}%\n"
        f"MAE%     = {metrics['MAE_pct']:.2f}%\n"
        f"RMSE%    = {metrics['RMSE_pct']:.2f}%\n"
        f"R²       = {metrics['R2']:.4f}\n"
        f"MAE      = {metrics['MAE']:.0f} riders\n"
        f"RMSE     = {metrics['RMSE']:.0f} riders"
    )
    ax1.text(0.01, 0.97, ann, transform=ax1.transAxes, fontsize=8,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                       edgecolor="#d1d5db", alpha=0.92))
    ax1.set_title("STMI-Net — Test set: Actual vs Predicted (full period)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Test sample index"); ax1.set_ylabel("Ridership (riders)")
    ax1.legend(loc="upper right", fontsize=8); ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(idx_zoom, pred_min[idx_zoom], pred_max[idx_zoom],
                     alpha=0.18, color=C_BAND)
    ax2.plot(idx_zoom, actual_s1[idx_zoom], color=C_ACT, linewidth=1.7,
             label="Actual", zorder=5)
    steps_show = sorted({0, T_out // 2, T_out - 1})
    palette = [C_STMI, C_MID, C_LAST]; styles = ["--", "-.", ":"]
    for s, col, ls in zip(steps_show, palette, styles):
        ax2.plot(idx_zoom, y_pred[idx_zoom, s], color=col, linewidth=1.4,
                 linestyle=ls, alpha=0.88, label=f"Predicted step {s + 1}")
    ax2.set_title(
        f"Zoomed: last {zoom_n} samples — "
        f"step 1 / {T_out // 2 + 1} / {T_out} horizon comparison",
        fontsize=10, fontweight="bold",
    )
    ax2.set_xlabel("Test sample index"); ax2.set_ylabel("Ridership (riders)")
    ax2.legend(loc="upper right", fontsize=8, ncol=2); ax2.grid(alpha=0.25)

    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_step_metrics(per_step: list, out_path: str) -> None:
    steps    = [f"t+{i+1}" for i in range(len(per_step))]
    combined = [m["Combined"] for m in per_step]
    mape     = [m["MAPE"]     for m in per_step]
    mae_pct  = [m["MAE_pct"]  for m in per_step]
    rmse_pct = [m["RMSE_pct"] for m in per_step]
    r2       = [m["R2"]       for m in per_step]

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
    ax1.set_ylabel("% of mean demand / score")
    ax1.set_title("STMI-Net — Per-horizon metrics", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    ax2.plot(steps, r2, marker="o", color=C_STMI, linewidth=1.8, markersize=5)
    ax2.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax2.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax2.set_ylim(min(min(r2) - 0.05, -0.1), 1.08)
    ax2.set_ylabel("R²")
    ax2.set_title("STMI-Net — R² per horizon step", fontweight="bold")
    ax2.grid(alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.seq_dir is None:
        args.seq_dir = (
            "data/sequences/lstm" if args.lookback == 14
            else f"data/sequences/lookback_{args.lookback}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────────
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

    # ── Data ──────────────────────────────────────────────────────────────────
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_splits(args.seq_dir, device)
    T_in       = X_tr.shape[1]
    n_features = X_tr.shape[2]
    T_out      = y_tr.shape[1]

    meta_path  = os.path.join(args.seq_dir, "split_dates.json")
    split_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True
    )
    val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=args.batch_size)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=args.batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = STMINet(
        n_features     = n_features,
        T_in           = T_in,
        T_out          = T_out,
        d_model        = args.d_model,
        hidden_st      = args.hidden_st,
        sp_hidden      = args.sp_hidden,
        lstm_layers    = args.lstm_layers,
        hidden_g       = args.hidden_g,
        n_graph_layers = args.n_graph_layers,
        d_emb          = args.d_emb,
        d_hop          = args.d_hop,
        n_heads        = args.n_heads,
        e_layers       = args.e_layers,
        d_layers       = args.d_layers,
        d_ff           = args.d_ff,
        factor         = args.factor,
        dropout        = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    T_label  = T_in // 2
    print(f"\nModel          : STMINet")
    print(f"  Stage 1 (ST-LSTM): d_model={args.d_model}  hidden_st={args.hidden_st}"
          f"  sp_hidden={args.sp_hidden}  lstm_layers={args.lstm_layers}")
    print(f"  Stage 2 (MTGNN) : N_graph={args.d_model}  hidden_g={args.hidden_g}"
          f"  n_layers={args.n_graph_layers}  d_emb={args.d_emb}  d_hop={args.d_hop}")
    print(f"  Stage 3 (Informer): n_heads={args.n_heads}  e_layers={args.e_layers}"
          f"  d_layers={args.d_layers}  d_ff={args.d_ff}  factor={args.factor}")
    print(f"  T_label={T_label}  T_dec={T_label + T_out}")
    print(f"  Input : (batch, {T_in}, {n_features})")
    print(f"  Output: (batch, {T_out})")
    print(f"  Params: {n_params:,}")

    # ── Optimiser / loss ──────────────────────────────────────────────────────
    if args.loss == "huber":
        criterion = nn.HuberLoss(delta=1.0)
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    best_epoch     = 0
    patience_count = 0
    train_losses, val_losses = [], []
    best_state = None

    print(f"\nTraining  (max {args.epochs} epochs, patience={args.patience},"
          f" warmup={args.warmup_epochs})")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'LR':>10}")
    print("─" * 45)

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
                print(f"\nEarly stop at epoch {epoch}"
                      f"  (best={best_val_loss:.6f} @ epoch {best_epoch})")
                break

    model.load_state_dict(best_state)

    # ── Collect predictions ───────────────────────────────────────────────────
    model.eval()
    preds_s, trues_s = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            with autocast(enabled=use_amp):
                preds_s.append(model(X_b).cpu().numpy())
            trues_s.append(y_b.cpu().numpy())

    y_pred_s = np.concatenate(preds_s)
    y_true_s = np.concatenate(trues_s)

    # ── Inverse-transform ─────────────────────────────────────────────────────
    scaler_y_path = os.path.join(args.seq_dir, "scaler_y.pkl")
    if os.path.exists(scaler_y_path):
        scaler_y = joblib.load(scaler_y_path)
        N2, T2   = y_pred_s.shape
        y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).reshape(N2, T2)
        y_true   = scaler_y.inverse_transform(y_true_s.reshape(-1, 1)).reshape(N2, T2)
    else:
        print("[WARN] scaler_y.pkl not found — metrics in scaled units.")
        y_pred, y_true = y_pred_s, y_true_s

    # ── Per-horizon metrics ───────────────────────────────────────────────────
    W = 10
    print(f"\n{'='*85}")
    print("STMI-Net — TEST SET METRICS")
    print(f"{'='*85}")
    header = (
        f"{'Step':>5}  "
        f"{'Combined%':>{W}}  {'MAPE%':>{W}}  {'MAE%':>{W}}  "
        f"{'RMSE%':>{W}}  {'R²':>{W}}  {'MAE':>{W}}  {'RMSE':>{W}}"
    )
    print(header); print("─" * len(header))

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
    out_dir = f"src/outputs/stmi_net/{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{out_dir}/model.pt")

    results = {
        "run_id": run_id,
        "model":  "STMINet",
        "hparams": {
            "d_model":        args.d_model,
            "hidden_st":      args.hidden_st,
            "sp_hidden":      args.sp_hidden,
            "lstm_layers":    args.lstm_layers,
            "hidden_g":       args.hidden_g,
            "n_graph_layers": args.n_graph_layers,
            "d_emb":          args.d_emb,
            "d_hop":          args.d_hop,
            "n_heads":        args.n_heads,
            "e_layers":       args.e_layers,
            "d_layers":       args.d_layers,
            "d_ff":           args.d_ff,
            "factor":         args.factor,
            "T_label":        T_label,
            "dropout":        args.dropout,
            "T_in":           T_in,
            "T_out":          T_out,
            "n_features":     n_features,
            "batch_size":     args.batch_size,
            "lr":             args.lr,
            "weight_decay":   args.weight_decay,
            "loss":           args.loss,
            "warmup_epochs":  args.warmup_epochs,
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

    # ── STMI-Net specific plots ───────────────────────────────────────────────
    print(f"\nSaving plots...")
    plot_loss_curves(train_losses, val_losses,
                     out_path=f"{out_dir}/loss_curves.png")
    plot_predictions(y_true, y_pred, metrics=overall,
                     out_path=f"{out_dir}/test_predictions.png")
    plot_per_step_metrics(per_step,
                          out_path=f"{out_dir}/per_step_metrics.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*85}")
    print(f"STMI-Net run complete  →  {out_dir}/")
    print(f"{'='*85}")
    print(f"  Combined  : {overall['Combined']:.2f}%")
    print(f"  MAPE      : {overall['MAPE']:.2f}%")
    print(f"  MAE%      : {overall['MAE_pct']:.2f}%    (raw MAE  = {overall['MAE']:.0f} riders)")
    print(f"  RMSE%     : {overall['RMSE_pct']:.2f}%   (raw RMSE = {overall['RMSE']:.0f} riders)")
    print(f"  R²        : {overall['R2']:.4f}")

    # ── 16-way comparison against all 15 base models ──────────────────────────
    PRIOR_MODELS = [
        ("LSTM",        args.lstm_results,       "src/outputs/lstm",        "#2563eb"),
        ("BiLSTM",      args.bilstm_results,     "src/outputs/bilstm",      "#7c3aed"),
        ("TPA-LSTM",    args.tpalstm_results,    "src/outputs/tpa_lstm",    "#0891b2"),
        ("CNN-LSTM",    args.cnnlstm_results,    "src/outputs/cnn_lstm",    "#16a34a"),
        ("CNN-BiLSTM",  args.cnnbilstm_results,  "src/outputs/cnn_bilstm",  "#d97706"),
        ("ST-LSTM",     args.stlstm_results,     "src/outputs/st_lstm",     "#dc2626"),
        ("STGCN",       args.stgcn_results,      "src/outputs/stgcn",       "#10b981"),
        ("MTGNN",       args.mtgnn_results,      "src/outputs/mtgnn",       "#f472b6"),
        ("STSGCN",      args.stsgcn_results,     "src/outputs/stsgcn",      "#0ea5e9"),
        ("STFGNN",      args.stfgnn_results,     "src/outputs/stfgnn",      "#a855f7"),
        ("PDR-STGCN",   args.pdrstgcn_results,   "src/outputs/pdr_stgcn",   "#f97316"),
        ("ASTGCN",      args.astgcn_results,     "src/outputs/astgcn",      "#e11d48"),
        ("TFT",         args.tft_results,        "src/outputs/tft",         "#ca8a04"),
        ("Autoformer",  args.autoformer_results, "src/outputs/autoformer",  "#047857"),
        ("Informer",    args.informer_results,   "src/outputs/informer",    "#9333ea"),
    ]

    models_data = []
    comparison  = {}

    for name, path, model_dir, color in PRIOR_MODELS:
        key  = name.lower().replace("-", "_")
        data = load_model_results(path, model_dir, name)
        if data:
            m_overall = data["test_metrics"]["overall"]
            m_ps      = data["test_metrics"]["per_step"]
            models_data.append((name, m_overall, m_ps, color))
            comparison[key] = {"run_id": data.get("run_id"), "overall": m_overall}
        else:
            comparison[key] = {"run_id": None, "overall": None}

    # STMI-Net is always the last entry
    models_data.append(("STMI-Net", overall, per_step, C_STMI))

    if len(models_data) > 1:
        print_comparison_table(models_data[:-1], overall, "STMI-Net")
        plot_comparison(
            models_data,
            out_path=f"{out_dir}/comparison_{len(models_data)}_way.png",
        )

    # Embed deltas and comparison in results.json
    comparison["stmi_net"] = {
        "run_id":  run_id,
        "overall": {k: round(v, 4) for k, v in overall.items()},
    }
    for name, m_overall, _, __ in models_data[:-1]:
        key = name.lower().replace("-", "_")
        comparison[f"delta_vs_{key}"] = {
            k: round(overall.get(k, 0) - m_overall.get(k, 0), 4) for k in overall
        }

    results["comparison"] = comparison
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Naive persistence baseline ────────────────────────────────────────────
    target_idx   = split_meta.get("target_col_idx", 0)
    last_obs_s   = X_te[:, -1, target_idx : target_idx + 1].cpu().numpy()
    naive_pred_s = np.tile(last_obs_s, (1, T_out))
    if os.path.exists(scaler_y_path):
        N2b, T2b   = naive_pred_s.shape
        naive_pred = scaler_y.inverse_transform(
            naive_pred_s.reshape(-1, 1)
        ).reshape(N2b, T2b)
    else:
        naive_pred = naive_pred_s

    naive  = compute_metrics(y_true.flatten(), naive_pred.flatten())
    d_comb = overall["Combined"] - naive["Combined"]
    d_mape = naive["MAPE"] - overall["MAPE"]
    print(f"\n  Naive persistence   "
          f"Combined={naive['Combined']:.2f}%  MAPE={naive['MAPE']:.2f}%  "
          f"R²={naive['R2']:.4f}")
    print(f"  STMI-Net vs naive   "
          f"ΔCombined={d_comb:+.2f}%  ΔMAPE={d_mape:+.2f}%")


if __name__ == "__main__":
    main()
