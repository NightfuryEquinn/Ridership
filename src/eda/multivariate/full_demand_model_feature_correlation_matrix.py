import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).parent.parent.parent.parent / "data" / "features"
OUT = Path(__file__).parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "features_aligned.csv", parse_dates=["date"])
features = df.drop(columns=["date"])

corr = features.corr()
n = len(corr.columns)

fig, ax = plt.subplots(figsize=(32, 28))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr, annot=False, cmap="coolwarm", center=0, ax=ax,
    mask=mask, vmin=-1, vmax=1, linewidths=0.3,
    xticklabels=True, yticklabels=True
)
ax.set_title(f"Full Demand Model Feature Correlation Matrix ({n}×{n})\n(Targets, Temporal, External, Lag, Static)", fontsize=16)
ax.tick_params(axis="x", labelsize=6, rotation=90)
ax.tick_params(axis="y", labelsize=6, rotation=0)

plt.tight_layout()
plt.savefig(OUT / "full_demand_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

high_corr_pairs = []
for i in range(n):
    for j in range(i + 1, n):
        val = corr.iloc[i, j]
        if abs(val) > 0.7:
            high_corr_pairs.append({
                "feature_1": corr.columns[i],
                "feature_2": corr.columns[j],
                "correlation": val
            })

high_corr_df = pd.DataFrame(high_corr_pairs)
if len(high_corr_df) > 0:
    high_corr_df = high_corr_df.sort_values("correlation", key=abs, ascending=False)

fig2, axes = plt.subplots(1, 2, figsize=(20, max(6, len(high_corr_df) * 0.35 + 2)))

if len(high_corr_df) > 0:
    ax1 = axes[0]
    colors = ["indianred" if v > 0 else "steelblue" for v in high_corr_df["correlation"]]
    ax1.barh(
        high_corr_df["feature_1"] + " vs " + high_corr_df["feature_2"],
        high_corr_df["correlation"],
        color=colors
    )
    ax1.axvline(x=0, color="black", linewidth=0.5)
    ax1.axvline(x=0.7, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.axvline(x=-0.7, color="blue", linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.set_xlabel("Correlation")
    ax1.set_title(f"High Correlation Pairs (|r| > 0.7) — {len(high_corr_df)} pairs")
    ax1.tick_params(axis="y", labelsize=7)
else:
    axes[0].text(0.5, 0.5, "No pairs with |r| > 0.7", ha="center", va="center",
                 transform=axes[0].transAxes, fontsize=12)
    axes[0].set_title("High Correlation Pairs (|r| > 0.7)")
    axes[0].axis("off")

corr_filled = np.nan_to_num(corr.values, nan=0.0)
np.fill_diagonal(corr_filled, 1.0)
eigenvalues = np.linalg.eigvalsh(corr_filled)
eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]
condition_number = np.sqrt(eigenvalues.max() / eigenvalues_pos.min()) if len(eigenvalues_pos) > 0 else np.nan

top5 = "\n".join([f"  {e:.4f}" for e in sorted(eigenvalues, reverse=True)[:5]])
diag_text = (
    f"Matrix size: {n}×{n}\n\n"
    f"Condition Number: {condition_number:.2f}\n\n"
    f"Eigenvalues (top 5):\n{top5}\n\n"
    f"High-corr pairs (|r|>0.7): {len(high_corr_df)}"
)
axes[1].text(0.05, 0.85, diag_text, fontsize=11, transform=axes[1].transAxes,
             verticalalignment="top", fontfamily="monospace")
axes[1].set_title("Multicollinearity Diagnostics")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(OUT / "full_demand_multicollinearity_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()

corr.to_csv(OUT / "full_demand_correlation_matrix.csv")
if len(high_corr_df) > 0:
    high_corr_df.to_csv(OUT / "full_demand_high_correlations.csv", index=False)

print(f"10. full_demand_model_feature_correlation_matrix.py - DONE ({n}×{n} matrix, {len(high_corr_df)} high-corr pairs)")
