"""
ridership.py  — Cleaning + ST-model preparation
Changes vs original:
  - MCO period is NO LONGER hard-removed; instead flagged with is_mco=1
    so temporal continuity is preserved for sequence models.
    A separate df_no_mco export is still provided for models that
    need to train on clean post-MCO data only.
  - Added temporal train / val / test split helper (no data leakage).
  - Added scaler export (fitted on train split only — CRITICAL: fitting
    the scaler on the full dataset leaks future statistics into training).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib, os

os.makedirs("data/cleaned", exist_ok=True)
os.makedirs("data/scalers", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/ridership_headline.csv")

print("=== Raw shape ===")
print(df.shape)
print(df.dtypes)
print(df.head())

# ── 1. Parse date ─────────────────────────────────────────────────────────────
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df = df.sort_values("date").reset_index(drop=True)

# ── 2. Flag MCO period — do NOT remove ───────────────────────────────────────
# Hard-removing MCO rows creates a temporal gap that breaks sliding-window
# sequences in LSTM/GCN-based models. Instead, flag the period so the model
# can learn the anomaly OR sequences can be split at the boundary.
#
# MCO phases in Malaysia that severely distorted ridership:
#   MCO 1.0:  18 Mar 2020 – 03 May 2020
#   CMCO/RMCO: 04 May 2020 – ~31 Dec 2021 (continued restrictions)
MCO_START = pd.Timestamp("2020-03-18")
MCO_END   = pd.Timestamp("2021-12-31")

df["is_mco"] = (
    (df["date"] >= MCO_START) & (df["date"] <= MCO_END)
).astype("Int64")

print(f"\nMCO-flagged rows: {df['is_mco'].sum()} "
      f"({df['is_mco'].mean()*100:.1f}% of dataset)")

# ── 3. Zero-fill structural nulls (service launch dates) ──────────────────────
# Nulls are NOT random — each service simply didn't exist before its launch.
# Zero-fill ONLY for dates on or after the launch date.
launch_dates = {
    "bus_rkl":            "2022-01-01",
    "bus_rpn":            "2022-01-01",
    "rail_ets":           "2020-10-15",
    "rail_intercity":     "2020-10-15",
    "rail_komuter_utara": "2020-10-15",
    "rail_mrt_pjy":       "2022-06-16",
    "rail_tebrau":        "2022-06-19",
    "rail_komuter":       "2023-09-10",
}

for col, launch in launch_dates.items():
    launch_ts = pd.Timestamp(launch)
    post_launch_null = (df["date"] >= launch_ts) & df[col].isna()
    count = post_launch_null.sum()
    if count:
        print(f"  {col}: zero-filling {count} null(s) after {launch}")
    df.loc[post_launch_null, col] = 0

# ── 4. Cast to nullable integer ───────────────────────────────────────────────
service_cols = [c for c in df.columns if c not in ("date", "is_mco")]
df[service_cols] = df[service_cols].astype("Int64")

# ── 5. Compute total ridership ────────────────────────────────────────────────
df["total_ridership"] = df[service_cols].sum(axis=1, min_count=1)

# ── 6. Add cyclical temporal features ────────────────────────────────────────
# LSTM/ST models can use raw date features, but sine/cosine encoding captures
# the cyclic nature of day-of-week and month without ordinal bias.
df["day_of_week"]     = df["date"].dt.dayofweek          # 0=Mon, 6=Sun
df["month"]           = df["date"].dt.month
df["day_of_year"]     = df["date"].dt.dayofyear
df["is_weekend"]      = (df["day_of_week"] >= 5).astype(int)

# Cyclical encoding
df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# ── 7. Temporal train / val / test split ─────────────────────────────────────
# CRITICAL: split MUST be chronological — no shuffling.
# Using ~70 / 15 / 15 cut points on post-MCO data to avoid leakage.
# Pre-MCO data is kept but excluded from model fitting here; include it
# in df_full if your model can handle the discontinuity.
df_post_mco = df[df["is_mco"] == 0].copy()

n = len(df_post_mco)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

df_train = df_post_mco.iloc[:train_end].copy()
df_val   = df_post_mco.iloc[train_end:val_end].copy()
df_test  = df_post_mco.iloc[val_end:].copy()

print(f"\nTemporal split (post-MCO only):")
print(f"  Train: {df_train['date'].min().date()} → {df_train['date'].max().date()} "
      f"({len(df_train)} rows)")
print(f"  Val  : {df_val['date'].min().date()} → {df_val['date'].max().date()} "
      f"({len(df_val)} rows)")
print(f"  Test : {df_test['date'].min().date()} → {df_test['date'].max().date()} "
      f"({len(df_test)} rows)")

# ── 8. Fit scaler on TRAIN only ───────────────────────────────────────────────
# Fitting on the full dataset leaks val/test statistics into training —
# a common source of overly-optimistic results.
scale_cols = service_cols + ["total_ridership"]
scaler = MinMaxScaler(feature_range=(0, 1))

# Fill NaN with 0 for scaling (pre-launch nulls)
train_vals = df_train[scale_cols].fillna(0).astype(float)
scaler.fit(train_vals)

joblib.dump(scaler, "data/scalers/ridership_scaler.pkl")
print(f"\nScaler fitted on train set and saved: data/scalers/ridership_scaler.pkl")
print(f"  Columns scaled: {scale_cols}")

# ── 9. Sanity checks ──────────────────────────────────────────────────────────
print(f"\nFull dataset shape : {df.shape}")
print(f"Date range         : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"MCO rows flagged   : {df['is_mco'].sum()}")
remaining_nulls = df[service_cols].isnull().mean().mul(100).round(1)
print(f"Null % per col (pre-launch):\n{remaining_nulls[remaining_nulls > 0].to_string()}")

# ── 10. Export ────────────────────────────────────────────────────────────────
# Full dataset with MCO flag (use for models that can learn the anomaly)
df.to_csv("data/cleaned/ridership_headline_clean.csv", index=False)

# Post-MCO only with split labels (use for clean chronological training)
df_post_mco["split"] = "train"
df_post_mco.loc[df_val.index, "split"]  = "val"
df_post_mco.loc[df_test.index, "split"] = "test"
df_post_mco.to_csv("data/cleaned/ridership_post_mco_split.csv", index=False)

print("\nExported:")
print("  data/cleaned/ridership_headline_clean.csv       (full, MCO flagged)")
print("  data/cleaned/ridership_post_mco_split.csv       (post-MCO, split labelled)")
print("  data/scalers/ridership_scaler.pkl               (MinMaxScaler, train-fit)")