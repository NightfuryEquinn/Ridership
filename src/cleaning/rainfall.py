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

# ── 4. Filter out adm_level == 2 ─────────────────────────────────────────────────
print("\nadm_level counts (final only):")
print(df_final["adm_level"].value_counts())

df_filtered = df_final[df_final["adm_level"] != 2].copy()
print(f"\nRows after dropping adm_level == 2: {len(df_filtered)} (dropped {len(df_final) - len(df_filtered)})")

# ── 5. Group by administrative boundaries ───────────────────────────────────
# Group districts (MY0101, MY0102, ...) under their parent state (MY01).
print("\nadm_level counts (filtered):")
print(df_filtered["adm_level"].value_counts())

def group_districts_to_state(df):
    # Districts have adm_level == 3 and PCODE length > 4 (e.g., "MY0101")
    # States have adm_level == 2 and PCODE length == 4 (e.g., "MY01")
    df['state_pcode'] = df['PCODE'].str[:4]  # Extract state-level PCODE
    
    # Group districts (adm_level 3) by state_pcode and aggregate metrics
    districts = df[df['adm_level'] == 3]
    if not districts.empty:
        state_aggregated = districts.groupby(['date', 'state_pcode', 'adm_level']).agg({
            'rainfall_mm': 'sum',
            'rainfall_avg_mm': 'mean',
            'acc_1mo_mm': 'sum',
            'acc_1mo_avg_mm': 'mean',
            'acc_3mo_mm': 'sum',
            'acc_3mo_avg_mm': 'mean',
            'anomaly_rf': 'mean',
            'anomaly_1mo': 'mean',
            'anomaly_3mo': 'mean'
        }).reset_index()
        state_aggregated['adm_level'] = 2  # Update adm_level to state
        
        # Remove districts and append aggregated states
        df = df[df['adm_level'] != 3]
        df = pd.concat([df, state_aggregated], ignore_index=True)
    return df

df_combined = group_districts_to_state(df_filtered.copy())

# Order by date and PCODE for consistent output
df_combined = df_combined.sort_values(['date', 'PCODE']).reset_index(drop=True)

print(f"Combined rows after grouping: {len(df_combined)}")

# ── 6. Sanity checks ──────────────────────────────────────────────────────────
print(f"\ncombined: shape={df_combined.shape}, nulls={df_combined.isnull().sum().sum()}")
print(df_combined[["date", "rainfall_mm", "anomaly_rf"]].describe().round(3))

# ── 7. Export ─────────────────────────────────────────────────────────────────
df_combined.to_csv("data/cleaned/rainfall_combined_final.csv", index=False)
print("\nExported: data/cleaned/rainfall_combined_final.csv")