import pandas as pd

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/school_public_holiday.csv")

print("=== School Public Holiday ===")
print(f"Raw shape: {df.shape}")
print(df.dtypes)
print()

# ── 1. Parse dates ────────────────────────────────────────────────────────────
# Dates are stored as "Mmm-DD" (e.g. "Mar-22") without a year.
# The Year column must be combined to reconstruct full dates.
# Format string: %Y-%b-%d  (e.g. "2019-Mar-22")

def parse_date(year, date_str):
    try:
        return pd.to_datetime(f"{year}-{date_str}", format="%Y-%b-%d")
    except Exception:
        return pd.NaT

df["start_date"] = df.apply(lambda r: parse_date(r["Year"], r["Start Date"]), axis=1)
df["end_date"]   = df.apply(lambda r: parse_date(r["Year"], r["End Date"]),   axis=1)

parse_failures = df["start_date"].isna().sum() + df["end_date"].isna().sum()
print(f"Date parse failures: {parse_failures}")

# ── 2. Validate date ranges (end must not precede start) ──────────────────────
inverted = df[df["end_date"] < df["start_date"]]
if len(inverted):
    print(f"[WARN] Inverted date ranges (end < start): {len(inverted)}")
    print(inverted[["Year", "Event Name", "start_date", "end_date"]])
else:
    print("Date range check: OK — no inverted ranges")

# ── 3. Compute duration in days ───────────────────────────────────────────────
df["duration_days"] = (df["end_date"] - df["start_date"]).dt.days + 1

# ── 4. Drop original raw date string columns (now redundant) ──────────────────
df = df.drop(columns=["Start Date", "End Date"])

# ── 5. Rename columns to snake_case ───────────────────────────────────────────
df = df.rename(columns={
    "Year":                       "year",
    "Type":                       "type",
    "Event Name":                 "event_name",
    "Spatial Coverage":           "spatial_coverage",
    "Notes / Replacement Logic":  "notes",
})

# ── 6. Flag 'State-specific' spatial coverage ─────────────────────────────────
# 6 rows (all Thaipusam entries) have spatial_coverage = 'State-specific'
# without naming which states. These are not wrong data — Thaipusam is observed
# only in states with significant Tamil populations (SGR, PNG, PRK, JHR, KUL).
# We annotate rather than guess-expand.
THAIPUSAM_STATES = "SGR, PNG, PRK, JHR, KUL"
mask_state_specific = df["spatial_coverage"] == "State-specific"
print(f"\nRows with 'State-specific' coverage: {mask_state_specific.sum()} "
      f"(all Thaipusam — annotated)")
df.loc[mask_state_specific, "notes"] = (
    df.loc[mask_state_specific, "notes"].fillna("") +
    f" [State-specific: typically {THAIPUSAM_STATES}]"
).str.strip()

# ── 7. Separate Academic vs Public holiday tables ─────────────────────────────
# The two types serve different analytical purposes — school break calendars
# vs national holidays — and should rarely be mixed in the same aggregation.
df_academic = df[df["type"] == "Academic"].drop(columns="type").reset_index(drop=True)
df_public   = df[df["type"] == "Public"].drop(columns="type").reset_index(drop=True)

print(f"\nAcademic break rows: {len(df_academic)}")
print(f"Public holiday rows: {len(df_public)}")
print(f"Notes column null rate: {df['notes'].isna().mean():.1%} "
      f"(expected — most rows need no annotation)")

# ── 8. Sanity checks ──────────────────────────────────────────────────────────
print(f"\nYear range: {df['year'].min()} – {df['year'].max()}")
print(f"Duplicates: {df.duplicated().sum()}")
print(f"Nulls:\n{df.isnull().sum()}")

# Group A: Kedah, Kelantan, Terengganu
# Group B: All other states
print(f"\nSample output:")
print(df.head(10).to_string(index=False))

# ── 9. Export ─────────────────────────────────────────────────────────────────
df.to_csv("data/cleaned/school_public_holiday_clean.csv", index=False)
print("\nExported: data/cleaned/school_public_holiday_combined_clean.csv (combined)")