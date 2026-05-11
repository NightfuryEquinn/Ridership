"""
feature_align.py  — Temporal Feature Alignment
NEW FILE

Merges all cleaned data sources onto a single daily date index, producing
the flat feature matrix that sequence_builder.py consumes.

Input files (all produced by the cleaning scripts):
  data/cleaned/ridership_headline_clean.csv
  data/cleaned/fuelprice_level_daily.csv
  data/cleaned/fuelprice_change_daily.csv
  data/cleaned/holiday_daily_features.csv
  data/cleaned/rainfall_wide_daily.csv
  data/cleaned/population_density_clean.csv   (static — broadcast to all dates)

Output:
  data/features/features_aligned.csv    — daily feature matrix, date × features
  data/features/feature_metadata.json   — column groups, dtypes, null counts
"""

import os
import json
import pandas as pd
import numpy as np

os.makedirs("data/features", exist_ok=True)

# ── Helper ────────────────────────────────────────────────────────────────────
def load_indexed(path: str, date_col: str = "date") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    return df


# ── 1. Master date range ──────────────────────────────────────────────────────
# Anchor to post-MCO ridership data (training-usable dates).
# Adjust DATE_START / DATE_END as needed.
DATE_START = "2022-01-01"
DATE_END   = "2025-12-31"   # update as new ridership data arrives

master_idx = pd.DatetimeIndex(
    pd.date_range(DATE_START, DATE_END, freq="D"),
    name="date"
)
print(f"Master date range: {DATE_START} → {DATE_END}  ({len(master_idx)} days)")

aligned = pd.DataFrame(index=master_idx)


# ── 2. Ridership (target + service-level features) ───────────────────────────
print("\n[1/5] Loading ridership...")
ridership = load_indexed("data/cleaned/ridership_headline_clean.csv")

# Keep service columns + derived features; drop raw date helpers
service_cols = [c for c in ridership.columns
                if c not in ("is_mco", "day_of_week", "month", "day_of_year",
                              "is_weekend", "dow_sin", "dow_cos",
                              "month_sin", "month_cos")]
ridership_sub = ridership[service_cols].reindex(master_idx)

# Replace nullable Int64 with float for downstream numpy operations
for col in ridership_sub.columns:
    ridership_sub[col] = ridership_sub[col].astype(float)

aligned = aligned.join(ridership_sub)
print(f"  Ridership columns added: {list(service_cols)}")
print(f"  Null rate: {aligned[list(service_cols)].isnull().mean().mean():.2%}")


# ── 3. Fuel price (daily forward-filled) ──────────────────────────────────────
print("\n[2/5] Loading fuel prices...")
fuel_level  = load_indexed("data/cleaned/fuelprice_level_daily.csv")
fuel_change = load_indexed("data/cleaned/fuelprice_change_daily.csv")

fuel_level  = fuel_level.reindex(master_idx).ffill()
fuel_change = fuel_change.reindex(master_idx).ffill()

# Prefix columns to avoid collisions
fuel_level.columns  = [f"fp_lv_{c}" for c in fuel_level.columns]
fuel_change.columns = [f"fp_chg_{c}" for c in fuel_change.columns]

aligned = aligned.join(fuel_level).join(fuel_change)
fp_cols = list(fuel_level.columns) + list(fuel_change.columns)
print(f"  Fuel price columns: {fp_cols}")
print(f"  Null rate: {aligned[fp_cols].isnull().mean().mean():.2%}")


# ── 4. Holiday / cyclical features ────────────────────────────────────────────
print("\n[3/5] Loading holiday features...")
holidays = load_indexed("data/cleaned/holiday_daily_features.csv")
holidays  = holidays.reindex(master_idx).fillna(0)   # no holiday = 0

aligned = aligned.join(holidays)
hol_cols = list(holidays.columns)
print(f"  Holiday columns: {hol_cols[:8]}{'...' if len(hol_cols) > 8 else ''}")


# ── 5. Rainfall (state-level wide format) ─────────────────────────────────────
print("\n[4/5] Loading rainfall...")
rainfall = load_indexed("data/cleaned/rainfall_wide_daily.csv")
rainfall  = rainfall.reindex(master_idx)
rainfall  = rainfall.interpolate(method="linear", limit=7).bfill().ffill()

# Limit to rainfall_mm columns for the primary signal; include anomaly if desired
rf_mm_cols = [c for c in rainfall.columns if "rainfall_mm__" in c]
aligned = aligned.join(rainfall[rf_mm_cols])
print(f"  Rainfall state columns: {len(rf_mm_cols)}")
print(f"  Null rate: {aligned[rf_mm_cols].isnull().mean().mean():.2%}")


# ── 6. Population density (static — broadcast to all dates) ──────────────────
# Population is a 2020 snapshot; it doesn't vary by date.
# We compute a single national summary scalar (median density) and broadcast it.
# For node-level models, population is injected as a static node attribute
# in sequence_builder.py rather than repeated in every timestep row here.
print("\n[5/5] Loading population density summary...")
try:
    pop = pd.read_csv("data/cleaned/population_density_clean.csv")
    pop_median = pop["density_per_km2"].median()
    pop_log_median = pop["density_log"].median()
    aligned["pop_density_median"]     = pop_median
    aligned["pop_density_log_median"] = pop_log_median
    print(f"  National median density: {pop_median:.2f} per km²  "
          f"(log: {pop_log_median:.4f})")
except FileNotFoundError:
    print("  [SKIP] population_density_clean.csv not found")


# ── 7. Final null audit ───────────────────────────────────────────────────────
print("\n=== Null Audit (aligned matrix) ===")
null_pct = aligned.isnull().mean().mul(100).round(2)
if null_pct.max() > 0:
    print("Columns with nulls:")
    print(null_pct[null_pct > 0].to_string())
else:
    print("No nulls — all features fully aligned.")

print(f"\nAligned feature matrix shape: {aligned.shape}")
print(f"  {len(aligned)} rows (days)  ×  {aligned.shape[1]} feature columns")


# ── 8. Feature metadata ───────────────────────────────────────────────────────
target_cols   = ["total_ridership"] + [c for c in service_cols if c != "total_ridership"]
temporal_cols = hol_cols
external_cols = fp_cols + rf_mm_cols
static_cols   = ["pop_density_median", "pop_density_log_median"]

metadata = {
    "date_range":      {"start": DATE_START, "end": DATE_END},
    "total_features":  int(aligned.shape[1]),
    "total_days":      int(len(aligned)),
    "column_groups": {
        "targets":    [c for c in target_cols    if c in aligned.columns],
        "temporal":   [c for c in temporal_cols  if c in aligned.columns],
        "external":   [c for c in external_cols  if c in aligned.columns],
        "static":     [c for c in static_cols    if c in aligned.columns],
    },
    "null_counts": {col: int(v) for col, v in aligned.isnull().sum().items() if v > 0},
}

with open("data/features/feature_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)


# ── 9. Export ─────────────────────────────────────────────────────────────────
aligned.to_csv("data/features/features_aligned.csv")

print("\nExported:")
print("  data/features/features_aligned.csv    ← feed into sequence_builder.py")
print("  data/features/feature_metadata.json")