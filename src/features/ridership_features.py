"""
ridership_features.py
---------------------
Engineers temporal features from ridership_headline_clean.csv.
Outputs: ridership_temporal.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/cleaned/ridership_headline_clean.csv")
OUT_PATH = Path("data/features/ridership_temporal.parquet")


def load_ridership(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    return df


def engineer_rolling_windows(df: pd.DataFrame, col: str) -> pd.DataFrame:
    for window in [7, 14, 30]:
        df[f"{col}_roll_mean_{window}d"] = df[col].rolling(window).mean()
        df[f"{col}_roll_std_{window}d"] = df[col].rolling(window).std()
        df[f"{col}_roll_max_{window}d"] = df[col].rolling(window).max()
    return df


def engineer_lags(df: pd.DataFrame, col: str) -> pd.DataFrame:
    for lag in [-7, -3, -1, 1, 3, 7]:
        label = f"lag_p{abs(lag)}d" if lag > 0 else f"lag_m{abs(lag)}d"
        df[f"{col}_{label}"] = df[col].shift(-lag)
    return df


def detect_anomalies(df: pd.DataFrame, col: str, z_thresh: float = 3.0) -> pd.DataFrame:
    """Flag anomalies using rolling z-score (30-day window)."""
    roll_mean = df[col].rolling(30, min_periods=7).mean()
    roll_std = df[col].rolling(30, min_periods=7).std()
    z_score = (df[col] - roll_mean) / roll_std.replace(0, np.nan)
    df[f"{col}_anomaly"] = (z_score.abs() > z_thresh).astype(int)
    df[f"{col}_zscore"] = z_score
    return df


def align_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to daily, forward-fill short gaps (≤3 days)."""
    df = df.resample("D").mean()
    df = df.ffill(limit=3)
    return df


def run(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH) -> pd.DataFrame:
    print(f"[ridership] Loading {raw_path}")
    df = load_ridership(raw_path)

    # Identify all ridership columns (numeric columns except total_ridership to avoid duplication)
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove total_ridership if present to avoid duplicate features
    ridership_cols = [col for col in all_numeric_cols if col != 'total_ridership']
    print(f"[ridership] Target columns: {ridership_cols}")

    df = align_timestamps(df)
    
    # Process each ridership column
    for col in ridership_cols:
        df = engineer_rolling_windows(df, col)
        df = engineer_lags(df, col)
        df = detect_anomalies(df, col)

    # Day-of-week / month cyclical encoding
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"[ridership] Saved -> {out_path}  shape={df.shape}")
    return df


if __name__ == "__main__":
    run()