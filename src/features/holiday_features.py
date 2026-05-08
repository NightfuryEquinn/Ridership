"""
holiday_features.py
-------------------
Engineers binary holiday/school-break flags from school_public_holiday_clean.csv.
Outputs: holiday_flags.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/cleaned/school_public_holiday_clean.csv")
OUT_PATH = Path("data/features/holiday_flags.parquet")

# Mapping from raw 'type' column values to our internal holiday categories
TYPE_MAP = {
    "academic": "school",
    "public": "public",
}


def load_holidays(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    df = df.sort_values("start_date")
    return df


def explode_date_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Expand rows with start_date/end_date into one row per date."""
    rows = []
    for _, row in df.iterrows():
        start = row["start_date"]
        end = row["end_date"]
        if pd.isna(start) or pd.isna(end):
            continue
        dr = pd.date_range(start=start, end=end, freq="D")
        for d in dr:
            rows.append({"date": d, "type": row["type"]})
    return pd.DataFrame(rows)


def build_daily_index(df: pd.DataFrame, ridership_path: Path) -> pd.DatetimeIndex:
    """Use ridership date range if available; else span of holiday file."""
    if ridership_path.exists():
        idx = pd.read_parquet(ridership_path).index
        return pd.date_range(idx.min(), idx.max(), freq="D")
    start = df["date"].min()
    end = df["date"].max()
    return pd.date_range(start, end, freq="D")


def engineer_flags(df_raw: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=date_index)
    out.index.name = "date"

    df_raw["type_lower"] = df_raw["type"].str.lower()

    # --- Binary flags per holiday type ---
    def dates_for_type(htype: str):
        mask = df_raw["type_lower"].str.contains(htype, na=False)
        return set(df_raw.loc[mask, "date"].dt.normalize())

    out["is_public_holiday"] = out.index.normalize().isin(dates_for_type("public")).astype(int)
    out["is_school_holiday"] = out.index.normalize().isin(dates_for_type("academic")).astype(int)

    # Any holiday (public OR school)
    out["is_any_holiday"] = ((out["is_public_holiday"] + out["is_school_holiday"]) > 0).astype(int)

    # --- Lead/lag windows (±1 day) ---
    for col in ["is_public_holiday", "is_school_holiday", "is_any_holiday"]:
        if col not in out.columns:
            continue
        out[f"{col}_lag_1d"]  = out[col].shift(1).fillna(0).astype(int)
        out[f"{col}_lead_1d"] = out[col].shift(-1).fillna(0).astype(int)
        out[f"{col}_lag_3d"]  = out[col].shift(3).fillna(0).astype(int)
        out[f"{col}_lead_3d"] = out[col].shift(-3).fillna(0).astype(int)

    # --- Holiday "window" flag: 1 if within ±1 day of any holiday ---
    out["holiday_window_1d"] = (
        (out["is_any_holiday"] + out["is_any_holiday_lag_1d"] + out["is_any_holiday_lead_1d"]) > 0
    ).astype(int)

    # --- Weekend flag (Saturday=5, Sunday=6) ---
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)

    # --- Long-weekend flag: holiday adjacent to weekend ---
    out["is_long_weekend"] = ((out["is_any_holiday"] == 1) & (out["is_weekend"] == 1)).astype(int)

    return out


def run(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH) -> pd.DataFrame:
    ridership_features_path = Path("data/features/ridership_temporal.parquet")

    print(f"[holiday] Loading {raw_path}")
    df_raw = load_holidays(raw_path)
    df_dates = explode_date_ranges(df_raw)
    date_index = build_daily_index(df_dates, ridership_features_path)

    out = engineer_flags(df_dates, date_index)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"[holiday] Saved -> {out_path}  shape={out.shape}")
    return out


if __name__ == "__main__":
    run()
