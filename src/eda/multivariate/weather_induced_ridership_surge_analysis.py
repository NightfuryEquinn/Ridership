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

rainfall_avg = rainfall.groupby("date").agg({"rainfall_mm": "mean", "anomaly_rf": "mean"}).reset_index()
rainfall_avg.columns = ["date", "rainfall_mm", "rainfall_anomaly"]

merged = ridership.merge(rainfall_avg, on="date", how="left")
merged["rainfall_mm"] = merged["rainfall_mm"].ffill().bfill()
merged["rainfall_anomaly"] = merged["rainfall_anomaly"].ffill().bfill()
merged["heavy_rain"] = (merged["rainfall_mm"] > merged["rainfall_mm"].quantile(0.75)).astype(int)
merged["date_only"] = merged["date"].dt.date
merged["year"] = merged["date"].dt.year
merged["month"] = merged["date"].dt.month

merged["day_of_week"] = merged["date"].dt.dayofweek
merged["is_weekend"] = (merged["day_of_week"] >= 5).astype(int)

rainy_days = merged[merged["heavy_rain"] == 1]
dry_days = merged[merged["heavy_rain"] == 0]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

ax1 = axes[0, 0]
ax1.scatter(rainy_days["rainfall_mm"], rainy_days["total_ridership"] / 1e6, c="steelblue", alpha=0.5, s=30, label="Heavy Rain")
ax1.scatter(dry_days["rainfall_mm"], dry_days["total_ridership"] / 1e6, c="orange", alpha=0.3, s=20, label="Normal/Dry")
ax1.set_xlabel("Rainfall (mm)")
ax1.set_ylabel("Ridership (Millions)")
ax1.set_title("Rainfall vs Ridership: Weather-Driven Modal Shift")
ax1.legend()

rainy_monthly = rainy_days.groupby("month")["total_ridership"].mean()
dry_monthly = dry_days.groupby("month")["total_ridership"].mean()

ax2 = axes[0, 1]
x = range(1, 13)
ax2.plot(x, [rainy_monthly.get(i, 0) / 1e6 for i in x], marker="o", color="steelblue", label="Heavy Rain Days")
ax2.plot(x, [dry_monthly.get(i, 0) / 1e6 for i in x], marker="s", color="orange", label="Normal/Dry Days")
ax2.set_xlabel("Month")
ax2.set_ylabel("Average Ridership (Millions)")
ax2.set_title("Monthly Ridership: Heavy Rain vs Normal")
ax2.legend()
ax2.set_xticks(x)

corr = merged[["total_ridership", "rainfall_mm", "rainfall_anomaly", "heavy_rain"]].corr()
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=axes[1, 0])
axes[1, 0].set_title("Correlation: Weather-Induced Ridership")

surge = merged.groupby("date").agg({"total_ridership": "mean", "rainfall_mm": "mean", "heavy_rain": "max"}).reset_index()
surge["date"] = pd.to_datetime(surge["date"])
surge = surge.sort_values("date")
surge["ridership_pct_change"] = surge["total_ridership"].pct_change() * 100
surge["heavy_rain_event"] = surge["heavy_rain"] == 1

ax4 = axes[1, 1]
ax4.scatter(surge["rainfall_mm"], surge["ridership_pct_change"], c=surge["heavy_rain"].map({1: "red", 0: "gray"}), alpha=0.5, s=30)
ax4.axhline(y=0, color="black", linewidth=0.5)
ax4.set_xlabel("Rainfall (mm)")
ax4.set_ylabel("Ridership % Change (vs Previous Day)")
ax4.set_title("Weather-Induced Ridership Surge: % Change Analysis")

plt.tight_layout()
plt.savefig(OUT / "weather_induced_ridership_surge.png", dpi=150, bbox_inches="tight")
plt.close()

corr.to_csv(OUT / "weather_ridership_correlation.csv")
surge.to_csv(OUT / "weather_surge_events.csv", index=False)
print("4. weather_induced_ridership_surge_analysis.py - DONE")