"""
comparison_table.py — Shared comparison utilities for all forecasting models.

Usage:
    from src.utils.comparison_table import (
        load_model_results,
        print_comparison_table,
        plot_comparison,
    )
"""

import os
import json
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ══════════════════════════════════════════════════════════════════════════════
# Results loader
# ══════════════════════════════════════════════════════════════════════════════

def load_model_results(path, model_dir, label):
    """Load results.json from an explicit path or the most recent run dir."""
    if path:
        if not os.path.exists(path):
            print(f"[WARN] {label} results not found at: {path}")
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    candidates = sorted(glob.glob(f"{model_dir}/**/results.json", recursive=True))
    if not candidates:
        print(f"[INFO] No {label} results.json found under {model_dir} — skipping.")
        return None
    detected = candidates[-1]
    print(f"[INFO] Auto-detected {label} results: {detected}")
    with open(detected, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Console table
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(models_data, new_overall, new_name):
    """
    Print an N-way comparison table.

    Parameters
    ----------
    models_data : list of (name, overall_dict, per_step_list, color_hex)
        Prior model results — does NOT include the new model.
    new_overall : dict
        Overall metrics dict for the model currently being evaluated.
    new_name : str
        Display name of the model currently being evaluated.
    """
    METRICS_CFG = [
        ("Combined",  "Combined%", True),
        ("MAPE",      "MAPE%",     False),
        ("MAE_pct",   "MAE%",      False),
        ("RMSE_pct",  "RMSE%",     False),
        ("R2",        "R²",        True),
        ("MAE",       "MAE",       False),
        ("RMSE",      "RMSE",      False),
    ]
    n_prior = len(models_data)
    W = 10
    sep_len = 16 + W * (1 + n_prior) + 18 * n_prior + 12
    sep = "─" * sep_len
    print(f"\n{'=' * sep_len}")
    print(f"{n_prior + 1}-Way Comparison — Overall Test Metrics")
    print(f"{'=' * sep_len}")
    header = f"{'Metric':<14}"
    for name, _, __, ___ in models_data:
        header += f" {name:>{W}}"
    header += f" {new_name:>{W}}"
    for name, _, __, ___ in models_data:
        header += f"  {f'Δ vs {name}'[:W]:>{W}}"
    header += "  Best"
    print(header)
    print(sep)
    for key, label, higher_better in METRICS_CFG:
        nv = new_overall.get(key, float("nan"))
        fmt = ".0f" if key in ("MAE", "RMSE") else (".4f" if key == "R2" else ".2f")
        row = f"{label:<14}"
        cand = {new_name: nv}
        for name, m, _, __ in models_data:
            v = m.get(key, float("nan"))
            cand[name] = v
            row += f" {v:{W}{fmt}}"
        row += f" {nv:{W}{fmt}}"
        for name, m, _, __ in models_data:
            d = nv - m.get(key, 0)
            row += f"  {'+' if d >= 0 else ''}{d:{W}{fmt}}"
        best = max(cand, key=lambda k: cand[k] if higher_better else -cand[k])
        row += f"  {best}"
        print(row)
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# Comparison plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(models_data, out_path):
    """
    Save a multi-panel comparison figure for all models in models_data.

    Parameters
    ----------
    models_data : list of (name, overall_dict, per_step_list, color_hex)
        Includes ALL models (prior + new).
    out_path : str
        File path for the saved PNG.
    """
    PCT_KEYS   = ["Combined", "MAPE", "MAE_pct", "RMSE_pct"]
    PCT_LABELS = ["Combined%", "MAPE%", "MAE%", "RMSE%"]
    n_steps = min(len(d[2]) for d in models_data)
    steps   = [f"t+{i + 1}" for i in range(n_steps)]
    n_models = len(models_data)
    w_bar = max(0.06, 0.80 / n_models)
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * w_bar
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "8", "p", "H",
               "<", ">", "1", "2"]
    styles  = ["-", "--", "-.", (0, (3, 1, 1, 1)), (0, (5, 1)), ":",
               (0, (1, 1)), (0, (3, 5, 1, 5)), "--", "-.", "-", "--",
               "-.", ":", "--", "-"]

    # Scale figure and fonts with the number of models
    fig_w      = max(16, 12 + n_models * 0.35)
    fig_h      = max(13, 10 + n_models * 0.25)
    lbl_fs     = max(4,  9  - n_models // 3)   # axis tick labels
    annot_fs   = max(3,  6  - n_models // 4)   # bar value annotations
    legend_fs  = max(4,  7  - n_models // 4)   # legend entries
    title_fs   = max(7,  9  - n_models // 6)   # suptitle
    r2_bar_w   = max(0.2, min(0.6, 4.0 / n_models))  # R² bar width

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(3, 2, hspace=0.52, wspace=0.32)

    # ── Overall percentage metrics bar chart ──────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(PCT_KEYS))
    for (name, m, _, col), off in zip(models_data, offsets):
        vals = [m[k] for k in PCT_KEYS]
        bars = ax00.bar(x + off, vals, w_bar, label=name, color=col, alpha=0.82)
        for bar in bars:
            h = bar.get_height()
            ax00.text(bar.get_x() + bar.get_width() / 2, h + 0.25,
                      f"{h:.1f}", ha="center", va="bottom", fontsize=annot_fs)
    ax00.set_xticks(x)
    ax00.set_xticklabels(PCT_LABELS, fontsize=lbl_fs)
    ax00.set_ylabel("% / score")
    ax00.set_title("Overall — Percentage Metrics", fontweight="bold")
    ax00.legend(fontsize=legend_fs)
    ax00.grid(axis="y", alpha=0.3)

    # ── Overall R² bar chart ──────────────────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    names = [d[0] for d in models_data]
    r2s   = [d[1]["R2"] for d in models_data]
    cols  = [d[3] for d in models_data]
    bars  = ax01.bar(names, r2s, color=cols, alpha=0.82, width=r2_bar_w)
    for bar in bars:
        h = bar.get_height()
        ax01.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                  f"{h:.4f}", ha="center", va="bottom", fontsize=annot_fs)
    ax01.set_ylim(0, min(1.12, max(r2s) * 1.15 + 0.05))
    ax01.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
    ax01.tick_params(axis="x", labelsize=lbl_fs, rotation=30)
    ax01.set_ylabel("R²")
    ax01.set_title("Overall — R²", fontweight="bold")
    ax01.grid(axis="y", alpha=0.3)

    # ── Per-horizon line charts ───────────────────────────────────────────────
    panel_cfg = [
        (gs[1, 0], "Combined", "Combined% per Horizon"),
        (gs[1, 1], "R2",       "R² per Horizon"),
        (gs[2, 0], "MAE",      "Raw MAE per Horizon (riders)"),
        (gs[2, 1], "RMSE",     "Raw RMSE per Horizon (riders)"),
    ]
    for gs_pos, key, title in panel_cfg:
        ax = fig.add_subplot(gs_pos)
        for (name, _, ps, col), mk, ls in zip(models_data, markers, styles):
            ax.plot(steps, [m[key] for m in ps[:n_steps]],
                    marker=mk, color=col, linewidth=1.8, markersize=5,
                    label=name, linestyle=ls)
        if key == "R2":
            ax.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
            ax.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=legend_fs)
        ax.grid(alpha=0.3)

    fig.suptitle(
        " vs ".join(d[0] for d in models_data) + " — Test Set Comparison",
        fontsize=title_fs, fontweight="bold", y=1.01,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
