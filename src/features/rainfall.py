"""
rainfall.py  — Cleaning + ST-model preparation
Changes vs original:
  - After cleaning, data is pivoted to WIDE format (date × state PCODE)
    so each state becomes a column — i.e., one feature vector per date
    that can be joined to the ridership/holiday feature matrix.
  - Missing dates (gaps in the source data) are interpolated linearly
    to ensure the daily index is contiguous. Without this, joining to
    a daily ridership index silently introduces NaN rows.
  - Anomaly columns (anomaly_rf etc.) are retained as separate wide
    tables because they have different semantics to raw mm values.
"""

import pandas as pd

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/mys_rainfall_subnat_2019_2026.csv")

print("=== Raw shape ===")
print(df.shape)
print(df.dtypes)
print(df.head())

# ── 1. Parse date ─────────────────────────────────────────────────────────────
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df = df.sort_values("date").reset_index(drop=True)

# ── 2. Rename columns ────────────────────────────────────────────────────────
rename_map = {
    "rfh": "rainfall_mm",      "rfh_avg": "rainfall_avg_mm",
    "r1h": "acc_1mo_mm",       "r1h_avg": "acc_1mo_avg_mm",
    "r3h": "acc_3mo_mm",       "r3h_avg": "acc_3mo_avg_mm",
    "rfq": "anomaly_rf",       "r1q":     "anomaly_1mo",
    "r3q": "anomaly_3mo",
}
df.rename(columns=rename_map, inplace=True)

# ── 3. Keep 'final' version only ─────────────────────────────────────────────
print("\nVersion counts:")
print(df["version"].value_counts())
df_final = df[df["version"] == "final"].drop(columns="version").copy()
print(f"Rows after keeping 'final': {len(df_final)}")

# ── 4. Drop adm_level == 2 (raw sub-national, use adm_level 1 states) ─────────
df_filtered = df_final[df_final["adm_level"] != 2].copy()

# ── 5. Aggregate districts → state ───────────────────────────────────────────
def group_districts_to_state(df):
    df["state_pcode"] = df["PCODE"].str[:4]
    districts = df[df["adm_level"] == 3]
    if not districts.empty:
        state_agg = districts.groupby(["date", "state_pcode"]).agg({
            "rainfall_mm":    "sum",  "rainfall_avg_mm": "mean",
            "acc_1mo_mm":     "sum",  "acc_1mo_avg_mm":  "mean",
            "acc_3mo_mm":     "sum",  "acc_3mo_avg_mm":  "mean",
            "anomaly_rf":     "mean", "anomaly_1mo":     "mean",
            "anomaly_3mo":    "mean",
        }).reset_index()
        state_agg = state_agg.rename(columns={"state_pcode": "PCODE"})
        state_agg["adm_level"] = 1
        df = df[df["adm_level"] != 3]
        df = pd.concat([df, state_agg], ignore_index=True)
    return df

df_combined = group_districts_to_state(df_filtered.copy())
df_combined = df_combined.sort_values(["date", "PCODE"]).reset_index(drop=True)

print(f"\nCombined long-format rows: {len(df_combined)}")

# ── 6. Pivot to WIDE format (CRITICAL for ST model feature matrix) ────────────
# Long format (one row per date × state) cannot be directly stacked into the
# (N_nodes × T_timesteps × F_features) tensor that GCN-based models require.
# Pivot produces: index=date, columns=PCODE, values=metric.

metric_cols = [
    "rainfall_mm", "rainfall_avg_mm",
    "acc_1mo_mm",  "acc_1mo_avg_mm",
    "anomaly_rf",  "anomaly_1mo",     "anomaly_3mo",
]

wide_frames = {}
for metric in metric_cols:
    if metric not in df_combined.columns:
        continue
    wide = df_combined.pivot_table(
        index="date", columns="PCODE", values=metric, aggfunc="mean"
    )
    wide.columns = [f"{metric}__{c}" for c in wide.columns]
    wide_frames[metric] = wide

df_wide = pd.concat(wide_frames.values(), axis=1).sort_index()
print(f"\nWide-format shape (date × state_metric): {df_wide.shape}")
print(f"Date range: {df_wide.index.min()} → {df_wide.index.max()}")
print(f"Null count before interpolation: {df_wide.isnull().sum().sum()}")

# ── 7. Fill missing dates (contiguous daily index) ───────────────────────────
# Source data may skip dates (no observation for a given period).
# Joining to a daily ridership index will silently introduce NaN rows
# unless we make the index contiguous first.
full_idx = pd.date_range(df_wide.index.min(), df_wide.index.max(), freq="D")
df_wide  = df_wide.reindex(full_idx)

# Linear interpolation for interior gaps; limit=7 days to avoid over-imputation.
df_wide = df_wide.interpolate(method="linear", limit=7, limit_direction="forward")

# If edge gaps remain (start/end of series), backfill / forward-fill.
df_wide = df_wide.bfill().ffill()

df_wide.index.name = "date"
print(f"Null count after interpolation: {df_wide.isnull().sum().sum()}")
print(f"Final wide shape: {df_wide.shape}")

# ── 8. Sanity checks ──────────────────────────────────────────────────────────
print(f"\ncombined long: shape={df_combined.shape}, nulls={df_combined.isnull().sum().sum()}")
print(df_combined[["date", "rainfall_mm", "anomaly_rf"]].describe().round(3))

# ── 9. Export ─────────────────────────────────────────────────────────────────
df_combined.to_csv("data/cleaned/rainfall_combined_final.csv", index=False)
df_wide.to_csv("data/cleaned/rainfall_wide_daily.csv")

print("\nExported:")
print("  data/cleaned/rainfall_combined_final.csv  (long format, original)")
print("  data/cleaned/rainfall_wide_daily.csv      ← USE THIS for model features")