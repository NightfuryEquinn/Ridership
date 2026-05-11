"""
fuelprice.py  — Cleaning + ST-model preparation
Changes vs original:
  - Weekly level data is upsampled to DAILY by forward-fill so it aligns
    with the daily ridership index (weekly gaps caused NaN bleed-through
    in merged feature matrices).
  - Added percent-change feature (more stationary than raw levels).
  - Scaler exported (fitted approach consistent with ridership.py).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib, os

os.makedirs("data/cleaned", exist_ok=True)
os.makedirs("data/scalers", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/fuelprice.csv")

print("=== Raw shape ===")
print(df.shape)
print(df.dtypes)
print(df.head())

# ── Parse date ────────────────────────────────────────────────────────────────
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()
df = df[df.index >= "2019-01-01"]

# ── Split by series_type ──────────────────────────────────────────────────────
df_level  = df[df["series_type"] == "level"].drop(columns="series_type")
df_change = df[df["series_type"] == "change_weekly"].drop(columns="series_type")

launch_date = df_level[["ron95_budi95"]].first_valid_index()
print(f"\nSubsidy columns first appear: {launch_date}")

# ── Summary statistics ────────────────────────────────────────────────────────
for name, frame in [("level", df_level), ("change_weekly", df_change)]:
    nulls = frame.isnull().sum().sum()
    print(f"\n{name}: {frame.shape}, nulls = {nulls}")
    print(frame.describe().round(4))

# ── Upsample to daily (CRITICAL for temporal alignment) ──────────────────────
# Fuel prices are announced weekly. Merging weekly data onto a daily ridership
# index produces 6/7 NaN rows per week unless we resample first.
# Forward-fill is correct: the price in effect carries forward until
# the next announcement.
full_daily_idx = pd.date_range(
    start=df_level.index.min(),
    end=df_level.index.max(),
    freq="D"
)

df_level_daily  = df_level.reindex(full_daily_idx).ffill()
df_change_daily = df_change.reindex(full_daily_idx).ffill()

print(f"\nLevel  — weekly rows: {len(df_level)}, after daily resample: {len(df_level_daily)}")
print(f"Change — weekly rows: {len(df_change)}, after daily resample: {len(df_change_daily)}")

# Verify no trailing nulls remain (can happen at the head before first observation)
head_nulls = df_level_daily.isnull().sum()
if head_nulls.any():
    # Back-fill for the very start of the series (before first price is recorded)
    df_level_daily  = df_level_daily.bfill()
    df_change_daily = df_change_daily.bfill()
    print(f"[INFO] Back-filled {head_nulls.max()} leading nulls at series start")

# ── Add percent-change feature ────────────────────────────────────────────────
# Percent change is more stationary than raw price levels, which helps
# gradient-based optimisers in LSTM/GCN models.
# Use the primary fuel columns (ron95, ron97, diesel).
for col in ["ron95", "ron97", "diesel"]:
    if col in df_level_daily.columns:
        df_level_daily[f"{col}_pct_chg"] = (
            df_level_daily[col].pct_change().fillna(0)
        )

# ── Export raw cleaned (un-scaled) ───────────────────────────────────────────
df_level_daily.index.name  = "date"
df_change_daily.index.name = "date"

df_level_daily.to_csv("data/cleaned/fuelprice_level_daily.csv")
df_change_daily.to_csv("data/cleaned/fuelprice_change_daily.csv")

# ── Fit scaler on data up to 2022-01-01 (approximate train cutoff) ───────────
# The exact cutoff should match ridership_scaler.pkl's train window.
# Pass TRAIN_END as an env var or hard-code to match ridership.py.
TRAIN_END = "2022-01-01"  # adjust to match ridership.py train split end
train_mask = df_level_daily.index < TRAIN_END

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_level_daily[train_mask].values)
joblib.dump(scaler, "data/scalers/fuelprice_scaler.pkl")
print(f"\nScaler fitted on data before {TRAIN_END} and saved.")

print("\nExported:")
print("  data/cleaned/fuelprice_level_daily.csv   (forward-filled daily levels)")
print("  data/cleaned/fuelprice_change_daily.csv  (forward-filled daily changes)")
print("  data/scalers/fuelprice_scaler.pkl")

# Original combined export retained for compatibility
df.to_csv("data/cleaned/fuelprice_cleaned.csv")
print("  data/cleaned/fuelprice_cleaned.csv       (original weekly, unchanged)")