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
holidays = pd.read_csv(BASE / "school_public_holiday_clean.csv", parse_dates=["start_date", "end_date"])

rainfall_avg = rainfall.groupby("date").agg({"rainfall_mm": "mean", "anomaly_rf": "mean", "anomaly_1mo": "mean"}).reset_index()
rainfall_avg.columns = ["date", "rainfall_mm", "rainfall_anomaly", "rainfall_anomaly_1mo"]

holiday_dates = set()
holiday_types = {}
for _, row in holidays.iterrows():
    daterange = pd.date_range(row["start_date"], row["end_date"])
    for d in daterange:
        holiday_dates.add(d)
        holiday_types[d] = row["type"]

merged = ridership.copy()
merged["month"] = merged["date"].dt.month
merged["year"] = merged["date"].dt.year
merged["day_of_week"] = merged["date"].dt.dayofweek
merged["is_holiday"] = merged["date"].isin(holiday_dates).astype(int)
merged["holiday_type"] = merged["date"].map(holiday_types).fillna("None")
merged = merged.merge(rainfall_avg, on="date", how="left")
merged["rainfall_mm"] = merged["rainfall_mm"].ffill().bfill()
merged["rainfall_anomaly"] = merged["rainfall_anomaly"].ffill().bfill()
merged["rainfall_anomaly_1mo"] = merged["rainfall_anomaly_1mo"].ffill().bfill()

seasonal_rain = merged.groupby("month")["rainfall_mm"].mean().to_dict()
seasonal_ridership = merged.groupby("month")["total_ridership"].mean().to_dict()

monthly = merged.groupby(["year", "month"]).agg({
    "total_ridership": "mean",
    "rainfall_mm": "mean",
    "rainfall_anomaly": "mean",
    "is_holiday": "sum"
}).reset_index()
monthly["date"] = pd.to_datetime(monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-01")
monthly["expected_ridership"] = monthly["month"].map(seasonal_ridership)
monthly["mismatch"] = monthly["total_ridership"] - monthly["expected_ridership"]
monthly = monthly.sort_values("date")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

ax1 = axes[0, 0]
ax1_twin = ax1.twinx()
ln1 = ax1.plot(monthly["date"], monthly["total_ridership"] / 1e6, color="steelblue", linewidth=2, label="Ridership")
ln2 = ax1_twin.plot(monthly["date"], monthly["rainfall_mm"], color="teal", linewidth=2, linestyle="--", label="Rainfall")
ax1.set_xlabel("Date")
ax1.set_ylabel("Ridership (Millions)", color="steelblue")
ax1_twin.set_ylabel("Rainfall (mm)", color="teal")
ax1.set_title("Monthly Ridership vs Rainfall Seasonality")
lines = ln1 + ln2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")

ax2 = axes[0, 1]
months = range(1, 13)
ax2.bar([m - 0.2 for m in months], [seasonal_ridership.get(m, 0) / 1e6 for m in months], width=0.4, label="Ridership", color="steelblue", alpha=0.8)
ax2.bar([m + 0.2 for m in months], [seasonal_rain.get(m, 0) for m in months], width=0.4, label="Rainfall", color="teal", alpha=0.8)
ax2.set_xlabel("Month")
ax2.set_ylabel("Value")
ax2.set_title("Seasonal Pattern: Ridership vs Rainfall by Month")
ax2.set_xticks(months)
ax2.legend()

mismatch_sorted = monthly.sort_values("mismatch")
colors = ["red" if m < 0 else "green" for m in mismatch_sorted["mismatch"]]
ax3 = axes[1, 0]
ax3.barh(range(len(mismatch_sorted)), mismatch_sorted["mismatch"] / 1e6, color=colors, alpha=0.7)
ax3.set_yticks(range(len(mismatch_sorted)))
ax3.set_yticklabels([f"{int(r['year'])}-{int(r['month']):02d}" for _, r in mismatch_sorted.iterrows()], fontsize=8)
ax3.axvline(x=0, color="black", linewidth=0.5)
ax3.set_xlabel("Mismatch (Millions)")
ax3.set_ylabel("Year-Month")
ax3.set_title("Demand-Supply Mismatch: Over/Under-Provision")

ax4 = axes[1, 1]
sc = ax4.scatter(monthly["rainfall_mm"], monthly["total_ridership"] / 1e6, c=monthly["is_holiday"], cmap="RdYlGn_r", s=100, alpha=0.7)
ax4.set_xlabel("Average Rainfall (mm)")
ax4.set_ylabel("Average Ridership (Millions)")
ax4.set_title("Rainfall vs Ridership (color = n_holidays)")
plt.colorbar(sc, ax=ax4, label="N Holidays")

plt.tight_layout()
plt.savefig(OUT / "seasonal_demand_supply_mismatch.png", dpi=150, bbox_inches="tight")
plt.close()

monthly.to_csv(OUT / "seasonal_mismatch_analysis.csv", index=False)
print("8. seasonal_demand_supply_mismatch.py - DONE")