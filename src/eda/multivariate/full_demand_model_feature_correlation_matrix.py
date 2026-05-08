import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).parent.parent.parent.parent / "data" / "cleaned"
OUT = Path(__file__).parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

ridership = pd.read_csv(BASE / "ridership_headline_clean.csv", parse_dates=["date"])
rainfall = pd.read_csv(BASE / "rainfall_combined_final.csv", parse_dates=["date"])
fuel = pd.read_csv(BASE / "fuelprice_cleaned.csv", parse_dates=["date"])
holidays = pd.read_csv(BASE / "school_public_holiday_clean.csv", parse_dates=["start_date", "end_date"])

rainfall_avg = rainfall.groupby("date").agg({"rainfall_mm": "mean", "anomaly_rf": "mean"}).reset_index()
rainfall_avg.columns = ["date", "rainfall_mm", "rainfall_anomaly"]

fuel_level = fuel[fuel["series_type"] == "level"][["date", "ron95", "diesel"]].copy()
fuel_level.columns = ["date", "fuel_ron95", "fuel_diesel"]

holiday_dates = set()
for _, row in holidays.iterrows():
    daterange = pd.date_range(row["start_date"], row["end_date"])
    holiday_dates.update(daterange)

merged = ridership.copy()
merged["day_of_week"] = merged["date"].dt.dayofweek
merged["is_holiday"] = merged["date"].isin(holiday_dates).astype(int)
merged["month"] = merged["date"].dt.month
merged["year"] = merged["date"].dt.year
merged = merged.merge(rainfall_avg, on="date", how="left")
merged = merged.merge(fuel_level, on="date", how="left")
merged["fuel_ron95"] = merged["fuel_ron95"].ffill().bfill()
merged["fuel_diesel"] = merged["fuel_diesel"].ffill().bfill()
merged["rainfall_mm"] = merged["rainfall_mm"].ffill().bfill()
merged["rainfall_anomaly"] = merged["rainfall_anomaly"].ffill().bfill()

merged["week_of_year"] = merged["date"].dt.isocalendar().week
merged["is_weekend"] = (merged["day_of_week"] >= 5).astype(int)

ridership_bus = ridership[["date", "bus_rkl", "bus_rpn"]].copy()
ridership_rail = ridership[["date", "rail_lrt_ampang", "rail_mrt_kajang", "rail_lrt_kj", "rail_monorail", "rail_mrt_pjy", "rail_ets", "rail_intercity", "rail_komuter_utara", "rail_tebrau", "rail_komuter"]].copy()

feature_cols = ["total_ridership", "rainfall_mm", "rainfall_anomaly", "fuel_ron95", "fuel_diesel", "is_holiday", "day_of_week", "month", "year", "week_of_year", "is_weekend"]

for col in ["bus_rkl", "bus_rpn", "rail_lrt_ampang", "rail_mrt_kajang", "rail_lrt_kj", "rail_monorail"]:
    if col in ridership.columns:
        merged[col] = ridership[col]
        feature_cols.append(col)

features_for_corr = merged[feature_cols].copy()
corr = features_for_corr.corr()

fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            mask=mask, vmin=-1, vmax=1, annot_kws={"size": 8}, linewidths=0.5)
ax.set_title("Full Demand Model Feature Correlation Matrix\n(Fuel, Rainfall, Ridership, Holidays, Time)", fontsize=14)

plt.tight_layout()
plt.savefig(OUT / "full_demand_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.7:
            high_corr_pairs.append({
                "feature_1": corr.columns[i],
                "feature_2": corr.columns[j],
                "correlation": corr.iloc[i, j]
            })

high_corr_df = pd.DataFrame(high_corr_pairs)
if len(high_corr_df) > 0:
    high_corr_df = high_corr_df.sort_values("correlation", key=abs, ascending=False)

fig2, axes = plt.subplots(1, 2, figsize=(16, 6))

if len(high_corr_df) > 0:
    ax1 = axes[0]
    ax1.barh(high_corr_df["feature_1"] + " vs " + high_corr_df["feature_2"], high_corr_df["correlation"], color="indianred")
    ax1.axvline(x=0, color="black", linewidth=0.5)
    ax1.set_xlabel("Correlation")
    ax1.set_title("High Correlation Pairs (|r| > 0.7) - Multicollinearity Alert")
    ax1.tick_params(axis="y", labelsize=8)

eigenvalues = np.linalg.eigvalsh(corr.values)
eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]
if len(eigenvalues_pos) > 0 and eigenvalues_pos.min() > 0:
    condition_number = np.sqrt(eigenvalues.max() / eigenvalues_pos.min())
else:
    condition_number = np.nan
axes[1].text(0.1, 0.7, f"Condition Number: {condition_number:.2f}\n\nEigenvalues (top 5):\n" + "\n".join([f"{e:.4f}" for e in sorted(eigenvalues, reverse=True)[:5]]), fontsize=12, transform=axes[1].transAxes)
axes[1].set_title("Multicollinearity Diagnostics")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(OUT / "full_demand_multicollinearity_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()

corr.to_csv(OUT / "full_demand_correlation_matrix.csv")
high_corr_df.to_csv(OUT / "full_demand_high_correlations.csv", index=False)
print("10. full_demand_model_feature_correlation_matrix.py - DONE")