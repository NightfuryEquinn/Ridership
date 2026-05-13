"""
graph_builder.py  — Graph Utility Functions for Graph-Based ST Models

Provides shared adjacency normalisation helpers used by the six graph-based
models in the stack:

  STGCN          — Chebyshev GCN + gated temporal conv
  Graph WaveNet  — adaptive adjacency + WaveNet dilated conv
  DCRNN          — diffusion convolution GRU
  STGAT          — multi-head graph attention + LSTM
  PatchTST+Graph — PatchTST Transformer + GCN
  ASTGCN         — spatial + temporal attention + Chebyshev GCN

Graph approach (features-as-nodes)
────────────────────────────────────
All models treat the N input features as graph nodes (N = total features
from features_aligned.csv — all 8 spatio-temporal sources combined).
Adjacency is built at training time from the absolute Pearson correlation
of feature columns in the training split (threshold=0.1). No external
node/edge files are needed — models call their own build_feature_adj()
at initialisation and store the result as a model buffer.

  GraphWaveNet uses a fully learnable adaptive adjacency instead.
  STGAT uses multi-head attention; no fixed adjacency is needed.
  ASTGCN also builds a scaled Chebyshev Laplacian L_tilde = -A_sym.

This module provides:
  symmetric_normalise(A)   — D^{-1/2}(A+I)D^{-1/2}, used by PatchTST+Graph
                             and ASTGCN
  row_normalise(A)         — D^{-1}A, used in diffusion random walks (DCRNN)
"""

import numpy as np


def symmetric_normalise(A: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """
    Compute D^{-1/2} (A + I) D^{-1/2}  — standard GCN normalisation
    (Kipf & Welling 2017).

    Parameters
    ----------
    A              : (N, N) adjacency matrix (binary or weighted)
    add_self_loops : add identity before normalising (default True)

    Returns
    -------
    A_hat : (N, N) symmetrically normalised adjacency
    """
    if add_self_loops:
        A = A + np.eye(A.shape[0], dtype=A.dtype)
    d = A.sum(axis=1).clip(min=1e-9)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    return D_inv_sqrt @ A @ D_inv_sqrt


def row_normalise(A: np.ndarray) -> np.ndarray:
    """
    Compute D^{-1} A  — row-stochastic normalisation used in diffusion
    random walks (forward/backward transition matrices in DCRNN).

    Parameters
    ----------
    A : (N, N) adjacency matrix

    Returns
    -------
    A_rw : (N, N) row-normalised adjacency (each row sums to 1)
    """
    d = A.sum(axis=1, keepdims=True).clip(min=1e-9)
    return A / d


if __name__ == "__main__":
    # Quick sanity check
    import torch
    A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32)
    A_hat = symmetric_normalise(A)
    A_rw  = row_normalise(A)
    print("symmetric_normalise:")
    print(A_hat.round(4))
    print("\nrow_normalise:")
    print(A_rw.round(4))
    print("\n✓ graph_builder utilities OK")
