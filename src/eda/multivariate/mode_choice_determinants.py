import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).parent.parent.parent.parent / "data" / "cleaned"
OUT = Path(__file__).parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

ridership = pd.read_csv(BASE / "ridership_headline_clean.csv", parse_dates=["date"])
fuel = pd.read_csv(BASE / "fuelprice_cleaned.csv", parse_dates=["date"])
rainfall = pd.read_csv(BASE / "rainfall_combined_final.csv", parse_dates=["date"])
holidays = pd.read_csv(BASE / "school_public_holiday_clean.csv", parse_dates=["start_date", "end_date"])

friction = pd.read_csv(BASE / "population_density_clean.csv")
friction_avg = friction["density_per_km2"].mean() if "density_per_km2" in friction.columns else np.nan

fuel_level = fuel[fuel["series_type"] == "level"][["date", "ron95", "diesel"]].copy()
fuel_level.columns = ["date", "fuel_ron95", "fuel_diesel"]

holiday_dates = set()
for _, row in holidays.iterrows():
    daterange = pd.date_range(row["start_date"], row["end_date"])
    holiday_dates.update(daterange)

rainfall_avg = rainfall.groupby("date")["rainfall_mm"].mean().reset_index()
rainfall_avg.columns = ["date", "rainfall_mm"]

merged = ridership.copy()
merged["day_of_week"] = merged["date"].dt.dayofweek
merged["is_holiday"] = merged["date"].isin(holiday_dates).astype(int)
merged = merged.merge(rainfall_avg, on="date", how="left")
merged = merged.merge(fuel_level, on="date", how="left")
merged["fuel_ron95"] = merged["fuel_ron95"].ffill().bfill()
merged["fuel_diesel"] = merged["fuel_diesel"].ffill().bfill()
merged["rainfall_mm"] = merged["rainfall_mm"].ffill().bfill()
merged["high_fuel"] = (merged["fuel_ron95"] > merged["fuel_ron95"].median()).astype(int)
merged["high_friction"] = (merged["rainfall_mm"] > merged["rainfall_mm"].median()).astype(int)
merged["low_poi"] = 0
merged["modal_shift_signal"] = merged["high_fuel"] * merged["high_friction"] * (1 - merged["low_poi"])

analysis = merged.groupby(merged["date"].dt.to_period("M")).agg({
    "total_ridership": "mean",
    "fuel_ron95": "mean",
    "rainfall_mm": "mean",
    "modal_shift_signal": "sum"
}).reset_index()
analysis["date"] = analysis["date"].dt.to_timestamp()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

ax1 = axes[0, 0]
ax1.scatter(merged["fuel_ron95"], merged["total_ridership"] / 1e6, c=merged["rainfall_mm"], cmap="Blues", alpha=0.6, s=30)
ax1.set_xlabel("Fuel Price (RON95)")
ax1.set_ylabel("Avg Daily Ridership (Millions)")
ax1.set_title("Fuel Price vs Ridership (color = rainfall)")

ax2 = axes[0, 1]
shift_periods = merged[merged["modal_shift_signal"] > 0]
normal_periods = merged[merged["modal_shift_signal"] == 0]
ax2.hist(normal_periods["total_ridership"] / 1e6, bins=30, alpha=0.5, label="Normal", color="gray")
ax2.hist(shift_periods["total_ridership"] / 1e6, bins=30, alpha=0.5, label="Modal Shift Signal", color="orange")
ax2.set_xlabel("Daily Ridership (Millions)")
ax2.set_ylabel("Frequency")
ax2.set_title("Ridership Distribution: Modal Shift Signal Periods")
ax2.legend()

corr_check = merged[["total_ridership", "fuel_ron95", "rainfall_mm", "modal_shift_signal"]].corr()
sns.heatmap(corr_check, annot=True, fmt=".3f", cmap="coolwarm", ax=axes[1, 0], center=0)
axes[1, 0].set_title("Correlation: Mode Choice Determinants")

monthly = merged.copy()
monthly["year_month"] = monthly["date"].dt.to_period("M")
monthly_grp = monthly.groupby("year_month").agg({
    "total_ridership": "mean",
    "fuel_ron95": "mean",
    "modal_shift_signal": "mean"
}).reset_index()
monthly_grp["date"] = monthly_grp["year_month"].dt.to_timestamp()

ax4 = axes[1, 1]
ax4_twin = ax4.twinx()
ln1 = ax4.plot(monthly_grp["date"], monthly_grp["total_ridership"] / 1e6, color="steelblue", label="Ridership")
ln2 = ax4_twin.plot(monthly_grp["date"], monthly_grp["fuel_ron95"], color="red", linestyle="--", label="Fuel Price")
ax4.set_xlabel("Date")
ax4.set_ylabel("Ridership (Millions)", color="steelblue")
ax4_twin.set_ylabel("Fuel Price (RON95)", color="red")
ax4.set_title("Monthly Ridership vs Fuel Price")
ax4.legend(loc="upper left")
ax4_twin.legend(loc="upper right")

plt.tight_layout()
plt.savefig(OUT / "mode_choice_determinants.png", dpi=150, bbox_inches="tight")
plt.close()

corr_check.to_csv(OUT / "mode_choice_correlation.csv")
merged[["date", "total_ridership", "fuel_ron95", "rainfall_mm", "modal_shift_signal"]].to_csv(OUT / "mode_choice_monthly.csv", index=False)
print("3. mode_choice_determinants.py - DONE")