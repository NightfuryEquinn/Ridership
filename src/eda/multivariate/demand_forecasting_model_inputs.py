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

rainfall_avg = rainfall.groupby("date")["rainfall_mm"].mean().reset_index()
rainfall_avg.columns = ["date", "rainfall_mm"]

fuel_level = fuel[fuel["series_type"] == "level"].copy()
fuel_level = fuel_level[["date", "ron95", "diesel"]].copy()
fuel_level.columns = ["date", "fuel_ron95", "fuel_diesel"]

holidays["is_holiday"] = 1
holiday_dates = set()
for _, row in holidays.iterrows():
    daterange = pd.date_range(row["start_date"], row["end_date"])
    holiday_dates.update(daterange)

merged = ridership.copy()
merged["day_of_week"] = merged["date"].dt.dayofweek
merged["is_holiday"] = merged["date"].isin(holiday_dates).astype(int)
merged = merged.merge(rainfall_avg, on="date", how="left")
merged = merged.merge(fuel_level, on="date", how="left")
merged["fuel_ron95"] = merged["fuel_ron95"].ffill().bfill()
merged["fuel_diesel"] = merged["fuel_diesel"].ffill().bfill()
merged["rainfall_mm"] = merged["rainfall_mm"].ffill().bfill()

features = merged[["total_ridership", "rainfall_mm", "fuel_ron95", "fuel_diesel", "is_holiday", "day_of_week"]].dropna()

corr = features.corr()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=axes[0], vmin=-1, vmax=1)
axes[0].set_title("Correlation Matrix: Demand Forecasting Inputs")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

X = features[["rainfall_mm", "fuel_ron95", "fuel_diesel", "is_holiday", "day_of_week"]]
y = features["total_ridership"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lr = LinearRegression()
lr.fit(X_scaled, y)

coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": lr.coef_})
coef_df = coef_df.sort_values("Coefficient", key=abs, ascending=False)

bars = axes[1].barh(coef_df["Feature"], coef_df["Coefficient"], color="steelblue")
axes[1].axvline(x=0, color="black", linewidth=0.5)
axes[1].set_title("Standardized Coefficients: Dominant Predictors of Ridership")
axes[1].set_xlabel("Coefficient (std. units)")

plt.tight_layout()
plt.savefig(OUT / "demand_forecasting_correlation.png", dpi=150, bbox_inches="tight")
plt.close()

summary = corr.round(4)
summary.to_csv(OUT / "demand_forecasting_correlation.csv")
coef_df.to_csv(OUT / "demand_forecasting_coefficients.csv", index=False)
print("1. demand_forecasting_model_inputs.py - DONE")