import pandas as pd

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/mys_rainfall_subnat_2019_2026.csv")

print("=== Raw shape ===")
print(df.shape)
print(df.dtypes)
print(df.head())

# ── 1. Parse date ─────────────────────────────────────────────────────────────
# Format is D/M/YYYY (day-first). Must be explicit — pandas defaults to M/D/YYYY.
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df = df.sort_values("date").reset_index(drop=True)

# ── 2. Rename cryptic metric columns ─────────────────────────────────────────
rename_map = {
    "rfh":     "rainfall_mm",
    "rfh_avg": "rainfall_avg_mm",
    "r1h":     "acc_1mo_mm",
    "r1h_avg": "acc_1mo_avg_mm",
    "r3h":     "acc_3mo_mm",
    "r3h_avg": "acc_3mo_avg_mm",
    "rfq":     "anomaly_rf",
    "r1q":     "anomaly_1mo",
    "r3q":     "anomaly_3mo",
}
df.rename(columns=rename_map, inplace=True)

# ── 3. Filter by version — CRITICAL ───────────────────────────────────────────
# 'prelim' and 'forecast' rows must be excluded before any historical analysis.
# Mixing them with 'final' inflates recent totals and skews averages.
print("\nVersion counts:")
print(df["version"].value_counts())

df_final = df[df["version"] == "final"].drop(columns="version").copy()
print(f"\nRows after keeping 'final' only: {len(df_final)} (dropped {len(df) - len(df_final)})")

# ── 4. Combine all administrative levels ─────────────────────────────────────
# Level 1 = national aggregate. Level 2 = state-level rows.
# adm_level is kept as a column so rows remain distinguishable.
print("\nadm_level counts (final only):")
print(df_final["adm_level"].value_counts())

df_combined = df_final.copy()

print(f"Combined rows: {len(df_combined)}")

# ── 5. Sanity checks ──────────────────────────────────────────────────────────
print(f"\ncombined: shape={df_combined.shape}, nulls={df_combined.isnull().sum().sum()}")
print(df_combined[["date", "rainfall_mm", "anomaly_rf"]].describe().round(3))

# ── 6. Export ─────────────────────────────────────────────────────────────────
df_combined.to_csv("data/cleaned/rainfall_combined_final.csv", index=False)
print("\nExported: data/cleaned/rainfall_combined_final.csv")