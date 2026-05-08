import pandas as pd

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/ridership_headline.csv")

print("=== Raw shape ===")
print(df.shape)
print(df.dtypes)
print(df.head())

# ── 1. Parse date ─────────────────────────────────────────────────────────────
# Format is D/M/YYYY — must use dayfirst=True.
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

# Remove ridership data between Mar 18, 2020 – Dec 31, 2021 (MCO) - Noise data
start_date = pd.Timestamp("2020-03-18")
end_date = pd.Timestamp("2021-12-31")
df = df[(df["date"] < start_date) | (df["date"] > end_date)]

df = df.sort_values("date").reset_index(drop=True)

# ── 2. Document and zero-fill structural nulls ────────────────────────────────
# Nulls are NOT random missingness. Each transit service has a known launch date
# and simply wasn't reporting before then. Imputing mean/median would be wrong.
# Zero-fill only for rows on or after the service launch date.
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

# Pre-launch rows remain NaN — they represent "service did not exist".

# ── 3. Cast service columns to nullable integer ───────────────────────────────
# Ridership counts are whole numbers. float64 (caused by NaNs) is misleading.
service_cols = [c for c in df.columns if c != "date"]
df[service_cols] = df[service_cols].astype("Int64")   # pandas nullable int, keeps NaN

# ── 4. Compute total ridership safely ────────────────────────────────────────
# min_count=1 ensures that rows where ALL columns are NaN stay NaN rather than
# silently becoming 0 (which would understate totals for pre-2020 dates that
# lack several services).
df["total_ridership"] = df[service_cols].sum(axis=1, min_count=1)

print("\nTotal ridership sample:")
print(df[["date", "total_ridership"]].head(10).to_string(index=False))

# ── 5. Null summary after cleaning ───────────────────────────────────────────
print("\nRemaining nulls per column (pre-launch only):")
null_pct = df[service_cols].isnull().mean().mul(100).round(1)
print(null_pct[null_pct > 0].to_string())

# ── 6. Sanity checks ─────────────────────────────────────────────────────────
print(f"\nFinal shape: {df.shape}")
print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(df.describe(include="all").T[["count", "mean", "min", "max"]].round(2))

# ── 7. Export ─────────────────────────────────────────────────────────────────
df.to_csv("data/cleaned/ridership_headline_clean.csv", index=False)
print("\nExported: data/cleaned/ridership_headline_clean.csv")