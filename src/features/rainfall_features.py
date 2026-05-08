"""
rainfall_features.py
--------------------
Engineers temporal rainfall features from rainfall_combined_final.csv.
Outputs: rainfall_temporal.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_RAINFALL_PATH = Path("data/cleaned/rainfall_combined_final.csv")
OUT_PATH          = Path("data/features/rainfall_temporal.parquet")

DATE_COL     = "date"
STATION_COL  = "adm_id"
RAIN_COL     = "rainfall_mm"


def load_rainfall(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.sort_values([DATE_COL, STATION_COL])
    return df


def resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sub-daily readings to daily totals per station."""
    return (
        df.groupby([pd.Grouper(key=DATE_COL, freq="D"), STATION_COL])
        [RAIN_COL]
        .sum()
        .reset_index()
    )


def engineer_rain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([STATION_COL, DATE_COL])

    for window in [3, 7, 14]:
        df[f"rain_roll_sum_{window}d"] = (
            df.groupby(STATION_COL)[RAIN_COL]
            .transform(lambda x: x.rolling(window, min_periods=1).sum())
        )
        df[f"rain_roll_mean_{window}d"] = (
            df.groupby(STATION_COL)[RAIN_COL]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    # Lag features
    for lag in [1, 3, 7]:
        df[f"rain_lag_{lag}d"] = (
            df.groupby(STATION_COL)[RAIN_COL]
            .transform(lambda x: x.shift(lag))
        )

    # Binary heavy-rain flag (>10 mm/day is a common threshold for Malaysia)
    df["heavy_rain_flag"] = (df[RAIN_COL] > 10).astype(int)

    return df


def pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot so index = date, columns = (feature, adm_id)."""
    feature_cols = [c for c in df.columns if c not in [DATE_COL, STATION_COL]]
    wide = df.pivot_table(index=DATE_COL, columns=STATION_COL, values=feature_cols)
    wide.columns = ["_".join(map(str, c)) for c in wide.columns]
    wide.index = pd.to_datetime(wide.index)
    return wide


def run(
    rain_path: Path = RAW_RAINFALL_PATH,
    out_path: Path = OUT_PATH,
) -> pd.DataFrame:
    print(f"[rainfall] Loading {rain_path}")
    df = load_rainfall(rain_path)
    df = resample_daily(df)
    df = engineer_rain_features(df)

    wide = pivot_to_wide(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(out_path)
    print(f"[rainfall] Saved -> {out_path}  shape={wide.shape}")
    return wide


if __name__ == "__main__":
    run()
