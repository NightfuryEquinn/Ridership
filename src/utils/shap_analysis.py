"""
shap_analysis.py — Gradient-based SHAP feature importance for PyTorch models.

Usage:
    from src.utils.shap_analysis import load_feature_names, run_shap_analysis
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    import torch.nn as nn
    from torch.amp import autocast
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

_METADATA_PATH = os.path.join("data", "features", "feature_metadata.json")
_GROUP_ORDER = ("targets", "temporal", "external", "lag", "static")


def load_feature_names(metadata_path: str = _METADATA_PATH) -> list:
    """
    Return the ordered feature name list from feature_metadata.json.
    Matches the column order of sequence tensors (targets → temporal →
    external → lag → static). Returns None if the file is not found.
    """
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)
    groups = meta.get("column_groups", {})
    names = []
    for key in _GROUP_ORDER:
        names.extend(groups.get(key, []))
    return names or None


def run_shap_analysis(
    model,
    X_test,
    out_dir: str,
    feature_names: list = None,
    n_samples: int = 100,
    use_amp: bool = False,
    top_k: int = 20,
) -> np.ndarray:
    """
    Run gradient-based SHAP analysis on a trained PyTorch model.

    Parameters
    ----------
    model : nn.Module
        Trained model (will be set to eval mode).
    X_test : torch.Tensor
        Test tensor, shape (N, T_in, F).
    out_dir : str
        Directory to write shap_importance.png and shap_values.npy.
    feature_names : list of str, optional
        F-length human-readable labels. Defaults to 'feat_N' if None or wrong length.
    n_samples : int
        Background/explanation sample count (default 100).
    use_amp : bool
        Wrap forward passes in autocast for AMP-trained models.
    top_k : int
        Features shown in the bar chart (default 20).

    Returns
    -------
    mean_importance : np.ndarray, shape (F,)
        Mean absolute SHAP per feature, averaged over samples, time, and horizons.
    """
    if not SHAP_AVAILABLE:
        print("  [SHAP] shap not installed — skipping.")
        return None

    print("  [SHAP] Computing gradient-based feature importances …")
    model.eval()
    subset = X_test[:n_samples]

    class _AMPWrapper(nn.Module):
        def __init__(self, m, amp):
            super().__init__()
            self.m = m
            self.amp = amp

        def forward(self, x):
            with autocast("cuda", enabled=self.amp):
                return self.m(x)

    wrapped = _AMPWrapper(model, use_amp)
    explainer = shap.GradientExplainer(wrapped, subset)
    shap_vals = explainer.shap_values(subset)

    if isinstance(shap_vals, list):
        shap_arr = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
    else:
        shap_arr = np.abs(shap_vals)

    # Reduce all axes except the feature axis (axis=2).
    # Handles both (N, T_in, F) and (N, T_in, F, T_out) outputs.
    axes = tuple(i for i in range(shap_arr.ndim) if i != 2)
    mean_importance = shap_arr.mean(axis=axes)  # (F,)

    F = len(mean_importance)
    top_k = min(top_k, F)
    top_idx = np.argsort(mean_importance)[-top_k:][::-1]

    labels = (
        feature_names
        if (feature_names and len(feature_names) == F)
        else [f"feat_{i}" for i in range(F)]
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(top_k), mean_importance[top_idx[::-1]], color="#2563eb", alpha=0.85)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([labels[i] for i in top_idx[::-1]], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top-{top_k} Feature Importances (SHAP)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    png_path = os.path.join(out_dir, "shap_importance.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.save(os.path.join(out_dir, "shap_values.npy"), shap_arr)
    print(f"  [SHAP] Saved: {png_path}")

    return mean_importance
