"""
fuel_features.py
----------------
Engineers temporal features from fuelprice_cleaned.csv.
Outputs: fuel_temporal.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/cleaned/fuelprice_cleaned.csv")
OUT_PATH = Path("data/features/fuel_temporal.parquet")

# Fuel type columns expected in the CSV (adjust if column names differ)
FUEL_COLS = ["ron95", "ron97", "diesel"]


def load_fuel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    return df


def resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Fuel prices are typically weekly — forward-fill to daily."""
    df = df.resample("D").last()   # last known price for that day
    df = df.ffill()                # carry forward until next change
    return df


def engineer_price_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    present_cols = [c for c in cols if c in df.columns]
    if not present_cols:
        # Fall back to all numeric columns
        present_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"[fuel] FUEL_COLS not found; using: {present_cols}")

    for col in present_cols:
        # Week-over-week change & % change
        df[f"{col}_chg_7d"] = df[col].diff(7)
        df[f"{col}_pct_chg_7d"] = df[col].pct_change(7)

        # Lags
        for lag in [1, 3, 7, 14]:
            df[f"{col}_lag_{lag}d"] = df[col].shift(lag)

        # Rolling mean (smoothed trend)
        df[f"{col}_roll_mean_30d"] = df[col].rolling(30).mean()

        # Price level relative to 30-day mean (deviation signal)
        df[f"{col}_dev_30d"] = df[col] - df[f"{col}_roll_mean_30d"]

    return df


def align_with_ridership(df: pd.DataFrame, ridership_path: Path) -> pd.DataFrame:
    """Trim fuel features to the ridership date range if the file exists."""
    if not ridership_path.exists():
        return df
    ridership_idx = pd.read_parquet(ridership_path).index
    start, end = ridership_idx.min(), ridership_idx.max()
    return df.loc[start:end]


def run(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH) -> pd.DataFrame:
    ridership_features_path = Path("data/features/ridership_temporal.parquet")

    print(f"[fuel] Loading {raw_path}")
    df = load_fuel(raw_path)
    df = resample_daily(df)
    df = engineer_price_features(df, FUEL_COLS)
    df = align_with_ridership(df, ridership_features_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"[fuel] Saved -> {out_path}  shape={df.shape}")
    return df


if __name__ == "__main__":
    run()