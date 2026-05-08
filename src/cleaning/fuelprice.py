import pandas as pd

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

# ── Subsidy-scheme columns ────────────────────────────────────────────────────
launch_date = df_level[["ron95_budi95"]].first_valid_index()
print(f"\nSubsidy columns first appear: {launch_date}")

# ── Summary statistics ────────────────────────────────────────────────────────
for name, frame in [
    ("level",         df_level),
    ("change_weekly", df_change),
]:
    nulls = frame.isnull().sum().sum()
    print(f"\n{name}: {frame.shape}, nulls = {nulls}")
    print(frame.describe().round(4))

# ── Export ────────────────────────────────────────────────────────────────────
df.to_csv("data/cleaned/fuelprice_cleaned.csv")
print("\nExported: data/cleaned/fuelprice_cleaned.csv")