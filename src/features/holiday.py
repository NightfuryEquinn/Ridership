"""
holiday.py  — Cleaning + ST-model preparation
Changes vs original:
  - Date ranges expanded to one row per day with binary flags
    (is_public_holiday, is_school_holiday). The original range-level
    format cannot be joined to a daily feature matrix.
  - Added temporal lead features: days_to_next_holiday,
    days_since_last_holiday — these are informative signals for ridership
    models because commuters reduce travel in anticipation of holidays.
  - Cyclical DOW/month encoding added here for convenience (consistent
    with ridership.py).
"""

import pandas as pd
import numpy as np


# ── Helper functions ──────────────────────────────────────────────────────────

def parse_date(year, date_str):
    try:
        return pd.to_datetime(f"{year}-{date_str}", format="%Y-%b-%d")
    except Exception:
        return pd.NaT


def expand_to_daily(frame: pd.DataFrame, flag_col: str) -> pd.Series:
    """
    Expand a table of (start_date, end_date) ranges into a daily boolean Series.
    Returns a Series indexed by date with value 1 on holiday/break days.
    """
    dates = set()
    for _, row in frame.iterrows():
        if pd.isna(row["start_date"]) or pd.isna(row["end_date"]):
            continue
        rng = pd.date_range(row["start_date"], row["end_date"], freq="D")
        dates.update(rng)
    idx = pd.DatetimeIndex(sorted(dates))
    s = pd.Series(1, index=idx, name=flag_col)
    return s


def days_to_next(flag_series: pd.Series) -> pd.Series:
    """How many days until the next 1 in flag_series (0 on holiday itself)."""
    result = pd.Series(np.nan, index=flag_series.index)
    last_idx = None
    for i in range(len(flag_series) - 1, -1, -1):
        if flag_series.iloc[i] == 1:
            last_idx = i
        if last_idx is not None:
            result.iloc[i] = last_idx - i
    return result.fillna(99).astype(int)   # 99 = no upcoming holiday in window


def days_since_last(flag_series: pd.Series) -> pd.Series:
    """How many days since the last 1 in flag_series (0 on holiday itself)."""
    result = pd.Series(np.nan, index=flag_series.index)
    last_idx = None
    for i in range(len(flag_series)):
        if flag_series.iloc[i] == 1:
            last_idx = i
        if last_idx is not None:
            result.iloc[i] = i - last_idx
    return result.fillna(99).astype(int)


def main():
    # ── Load ──────────────────────────────────────────────────────────────────────
    df = pd.read_csv("data/raw/school_public_holiday.csv")

    print("=== School Public Holiday ===")
    print(f"Raw shape: {df.shape}")
    print(df.dtypes)

    # ── 1. Parse dates ────────────────────────────────────────────────────────────
    df["start_date"] = df.apply(lambda r: parse_date(r["Year"], r["Start Date"]), axis=1)
    df["end_date"]   = df.apply(lambda r: parse_date(r["Year"], r["End Date"]),   axis=1)

    parse_failures = df["start_date"].isna().sum() + df["end_date"].isna().sum()
    print(f"Date parse failures: {parse_failures}")

    inverted = df[df["end_date"] < df["start_date"]]
    if len(inverted):
        print(f"[WARN] Inverted date ranges: {len(inverted)}")

    df["duration_days"] = (df["end_date"] - df["start_date"]).dt.days + 1

    df = df.drop(columns=["Start Date", "End Date"])
    df = df.rename(columns={
        "Year": "year", "Type": "type",
        "Event Name": "event_name",
        "Spatial Coverage": "spatial_coverage",
        "Notes / Replacement Logic": "notes",
    })

    # State-specific Thaipusam annotation
    THAIPUSAM_STATES = "SGR, PNG, PRK, JHR, KUL"
    mask_ss = df["spatial_coverage"] == "State-specific"
    df.loc[mask_ss, "notes"] = (
        df.loc[mask_ss, "notes"].fillna("") +
        f" [State-specific: typically {THAIPUSAM_STATES}]"
    ).str.strip()

    df_academic = df[df["type"] == "Academic"].drop(columns="type").reset_index(drop=True)
    df_public   = df[df["type"] == "Public"].drop(columns="type").reset_index(drop=True)

    # ── 2. Expand to one row per day (CRITICAL for daily feature alignment) ────────
    # A date-range table cannot be joined to a daily ridership index.
    # Expanding here produces a flat lookup that can be left-joined on 'date'.
    s_public   = expand_to_daily(df_public,   "is_public_holiday")
    s_academic = expand_to_daily(df_academic, "is_school_holiday")

    # ── 3. Build daily calendar covering the full model date range ────────────────
    DATE_START = "2019-01-01"
    DATE_END   = "2026-12-31"   # extend as needed

    daily_idx = pd.date_range(DATE_START, DATE_END, freq="D")
    cal = pd.DataFrame(index=daily_idx)
    cal.index.name = "date"

    cal["is_public_holiday"] = s_public.reindex(daily_idx).fillna(0).astype(int)
    cal["is_school_holiday"] = s_academic.reindex(daily_idx).fillna(0).astype(int)
    cal["is_holiday_any"]    = ((cal["is_public_holiday"] + cal["is_school_holiday"]) > 0).astype(int)

    # ── 4. Lead / lag features ─────────────────────────────────────────────────────
    cal["days_to_next_public_hol"]    = days_to_next(cal["is_public_holiday"])
    cal["days_since_last_public_hol"] = days_since_last(cal["is_public_holiday"])
    cal["days_to_next_school_hol"]    = days_to_next(cal["is_school_holiday"])
    cal["days_since_last_school_hol"] = days_since_last(cal["is_school_holiday"])

    # ── 5. Cyclical temporal features ────────────────────────────────────────────
    cal["day_of_week"]  = cal.index.dayofweek
    cal["month"]        = cal.index.month
    cal["is_weekend"]   = (cal["day_of_week"] >= 5).astype(int)
    cal["dow_sin"]      = np.sin(2 * np.pi * cal["day_of_week"] / 7)
    cal["dow_cos"]      = np.cos(2 * np.pi * cal["day_of_week"] / 7)
    cal["month_sin"]    = np.sin(2 * np.pi * cal["month"] / 12)
    cal["month_cos"]    = np.cos(2 * np.pi * cal["month"] / 12)

    # ── 6. Sanity checks ──────────────────────────────────────────────────────────
    print(f"\nDaily calendar shape: {cal.shape}")
    print(f"Public holidays per year:\n{cal.groupby(cal.index.year)['is_public_holiday'].sum()}")
    print(f"School holidays per year:\n{cal.groupby(cal.index.year)['is_school_holiday'].sum()}")
    print(f"Nulls: {cal.isnull().sum().sum()}")
    print(f"\nSample:\n{cal.head(10).to_string()}")

    # ── 7. Export ─────────────────────────────────────────────────────────────────
    cal.to_csv("data/cleaned/holiday_daily_features.csv")

    # Keep original range-level tables for reference
    df.to_csv("data/cleaned/school_public_holiday_clean.csv", index=False)

    print("\nExported:")
    print("  data/cleaned/holiday_daily_features.csv    ← USE THIS for model features")
    print("  data/cleaned/school_public_holiday_clean.csv (original range format)")


if __name__ == "__main__":
    main()
