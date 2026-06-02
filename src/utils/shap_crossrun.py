"""
shap_crossrun.py — Cross-run SHAP importance aggregation.

Loads every shap_values.npy under a model output directory and reports:
  - Universal zero features (zero in every run)
  - Near-universal zeros (zero in >= `near_universal_frac` of runs)
  - Consensus top features (ranked by how many runs include them in their top-K)

Run standalone:
    python src/utils/shap_crossrun.py
    python src/utils/shap_crossrun.py --model-dir src/outputs/hmttsf --top-k 15
"""

import os
import glob
import argparse

import numpy as np

from src.utils.shap_analysis import load_feature_names


def compute_importance(npy_path: str) -> np.ndarray:
    """Load shap_values.npy and return a 1-D mean-absolute importance vector (F,)."""
    arr = np.abs(np.load(npy_path, allow_pickle=True).astype(np.float64))
    axes = tuple(i for i in range(arr.ndim) if i != 2)
    return arr.mean(axis=axes)


def load_all_importances(model_out_dir: str) -> dict:
    """
    Discover every shap_values.npy one level under model_out_dir.
    Returns {run_id: importance_array (F,)}.
    """
    pattern = os.path.join(model_out_dir, "*", "shap_values.npy")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No shap_values.npy found under {model_out_dir}")
    return {
        os.path.basename(os.path.dirname(p)): compute_importance(p)
        for p in paths
    }


def cross_run_summary(
    importances: dict,
    feature_names: list = None,
    zero_threshold: float = 1e-10,
    consensus_top_k: int = 15,
    near_universal_frac: float = 0.8,
) -> dict:
    """
    Compute and print cross-run SHAP statistics.

    Parameters
    ----------
    importances : dict
        {run_id: importance_array} from load_all_importances().
    feature_names : list of str, optional
        Human-readable feature labels in column order.
    zero_threshold : float
        Importance values at or below this are treated as zero.
    consensus_top_k : int
        Top-K cutoff used when counting how often each feature ranks highly.
    near_universal_frac : float
        Fraction of runs a feature must be zero in to qualify as near-universal.

    Returns
    -------
    dict with keys: universal_zeros, near_universal_zeros, consensus_top,
                    zero_counts, n_runs, feature_names.
    """
    run_ids = list(importances.keys())
    n_runs = len(run_ids)
    F = len(next(iter(importances.values())))
    labels = (
        feature_names
        if (feature_names and len(feature_names) == F)
        else [f"feat_{i}" for i in range(F)]
    )

    zero_counts = np.zeros(F, dtype=int)
    top_k_counts = np.zeros(F, dtype=int)
    for imp in importances.values():
        zero_counts += (imp <= zero_threshold).astype(int)
        for i in np.argsort(imp)[-consensus_top_k:]:
            top_k_counts[i] += 1

    universal = sorted(i for i in range(F) if zero_counts[i] == n_runs)
    near_univ = sorted(
        i for i in range(F)
        if zero_counts[i] >= n_runs * near_universal_frac and i not in universal
    )
    consensus = sorted(enumerate(top_k_counts), key=lambda x: -x[1])

    W = 70
    print(f"\n{'=' * W}")
    print(f"Cross-Run SHAP Summary  ({n_runs} runs, {F} features)")
    print(f"{'=' * W}")

    print(f"\n── Universal Zeros — {len(universal)} features, zero in all {n_runs} runs ──")
    for i in universal:
        print(f"  feat_{i:02d}  {labels[i]}")

    print(f"\n── Near-Universal Zeros — {len(near_univ)} features, "
          f"zero in ≥{near_universal_frac * 100:.0f}% of runs ──")
    for i in near_univ:
        print(f"  feat_{i:02d}  {labels[i]:<40}  ({zero_counts[i]}/{n_runs} runs)")

    print(f"\n── Consensus Top-{consensus_top_k} — ranked by run-count ──")
    print(f"  {'Index':>8}  {'Feature':<40}  Runs in top-{consensus_top_k}")
    for idx, count in consensus[:20]:
        if count == 0:
            break
        print(f"  feat_{idx:02d}  {labels[idx]:<40}  {count}/{n_runs} "
              f"({count / n_runs * 100:.0f}%)")

    return {
        "universal_zeros": universal,
        "near_universal_zeros": near_univ,
        "consensus_top": consensus,
        "zero_counts": zero_counts.tolist(),
        "n_runs": n_runs,
        "feature_names": labels,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-run SHAP summary")
    parser.add_argument("--model-dir", default="src/outputs/hmttsf",
                        help="Root output directory containing run subdirs")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Top-K cutoff for consensus ranking (default 15)")
    parser.add_argument("--metadata", default=None,
                        help="Path to feature_metadata.json (auto-detected if omitted)")
    args = parser.parse_args()

    feat_names = load_feature_names(args.metadata) if args.metadata else load_feature_names()
    imps = load_all_importances(args.model_dir)
    cross_run_summary(imps, feature_names=feat_names, consensus_top_k=args.top_k)
