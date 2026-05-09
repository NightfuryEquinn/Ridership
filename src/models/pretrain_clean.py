"""
fix_feature_matrix.py
Dynamically cleans feature_matrix_lstm.parquet for LSTM training.

Automatically detects and handles:
  1. Zero-variance  → nunique() <= 1 (handles floats, ints, categoricals)
  2. Redundancy     → Pairwise correlation > threshold (greedy drop)
  3. Leakage        → Target correlation heuristic + forward-looking name patterns
  4. NaN handling   → ffill → bfill → safe zero-fill fallback

Usage:
  - Set TARGET_COL in config to enable correlation-based leakage detection.
  - Adjust thresholds in the CONFIG block as needed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Optional, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT  = Path("feature_matrix_lstm.parquet")
OUTPUT = Path("feature_matrix_lstm_clean.parquet")

TARGET_COL: Optional[str] = "ridership"   # e.g., "ridership". If None, uses name-heuristic only.
CORR_THRESHOLD = 0.95                     # Drop one of any pair with |r| > this
LEAKAGE_CORR_THRESHOLD = 0.98             # Drop features with |corr(target)| > this
LEAKAGE_NAME_REGEX = re.compile(r"(?i)(lead|future|next|ahead|_p\d+|t\+\d+|forward)")

# ============================================================================
# HELPERS
# ============================================================================
def audit(df: pd.DataFrame, label: str) -> None:
    total_null = df.isnull().sum().sum()
    null_pct   = total_null / max(df.size, 1)
    print(f"[{label}]  shape={df.shape}  nulls={total_null:,} ({null_pct:.2%})")


def drop_zero_variance(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """Drop columns with exactly 1 unique value (constants carry no signal)."""
    constant_mask = df.nunique(dropna=False) <= 1
    dropped = df.columns[constant_mask].tolist()
    return df.drop(columns=dropped), dropped


def drop_redundant(df: pd.DataFrame, threshold: float = CORR_THRESHOLD) -> Tuple[pd.DataFrame, list[str]]:
    """Drop one column from any pair with absolute correlation > threshold."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return df, []

    corr_matrix = num_df.corr().abs()
    # Mask upper triangle (excluding diagonal)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop), to_drop


def detect_leakage(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Dynamic leakage detection:
      a) If TARGET_COL is set: drops features with |corr| > LEAKAGE_CORR_THRESHOLD
      b) Always: scans column names for forward-looking temporal keywords
    """
    dropped: list[str] = []
    num_df = df.select_dtypes(include=[np.number])

    # 1. Correlation-based (if target exists)
    if TARGET_COL and TARGET_COL in num_df.columns:
        corr_with_target = num_df.corr()[TARGET_COL].abs()
        corr_with_target = corr_with_target.drop(TARGET_COL, errors="ignore")
        corr_leak = corr_with_target[corr_with_target > LEAKAGE_CORR_THRESHOLD].index.tolist()
        dropped.extend(corr_leak)

    # 2. Name-based heuristic (forward-looking indicators)
    name_leak = [c for c in df.columns if LEAKAGE_NAME_REGEX.search(c) and c not in dropped]
    dropped.extend(name_leak)

    # Deduplicate while preserving order
    dropped = list(dict.fromkeys(dropped))
    return df.drop(columns=dropped), dropped


def impute_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill → Back-fill → Zero-fill fallback. Preserves dtypes where possible."""
    df = df.ffill().bfill()
    remaining = df.isnull().sum().sum()
    if remaining:
        print(f"  ⚠️  WARNING: {remaining:,} NaNs remain after ffill/bfill — zero-filling.")
        # Only zero-fill numeric columns to avoid dtype corruption
        num_cols = df.select_dtypes(include=[np.number]).columns
        df.loc[:, num_cols] = df[num_cols].fillna(0.0)
    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main() -> None:
    print(f"📂 Loading: {INPUT}")
    df = pd.read_parquet(INPUT)
    audit(df, "original")

    # 1. Zero-variance
    df, zv_cols = drop_zero_variance(df)
    print(f"🗑️  Zero-variance : {len(zv_cols):>3} cols dropped")

    # 2. Redundancy
    df, red_cols = drop_redundant(df)
    print(f"🔗 Redundancy   : {len(red_cols):>3} cols dropped (corr > {CORR_THRESHOLD})")

    # 3. Leakage
    df, lk_cols = detect_leakage(df)
    reason = f"corr > {LEAKAGE_CORR_THRESHOLD}" if TARGET_COL else "name-heuristic"
    print(f"🔒 Leakage       : {len(lk_cols):>3} cols dropped ({reason})")

    total_dropped = len(zv_cols) + len(red_cols) + len(lk_cols)
    print(f"\n✅ Total dropped: {total_dropped} → {df.shape[1]} features remain")

    # 4. Imputation
    audit(df, "pre-impute")
    df = impute_time_series(df)
    audit(df, "post-impute")

    # 5. Sanity Checks
    assert df.isnull().sum().sum() == 0, "NaNs remain after imputation!"
    assert not df.duplicated().any(), "Duplicate rows detected!"
    
    # Check if any zero-variance columns were reintroduced by imputation
    remaining_zv = (df.nunique(dropna=False) <= 1).sum()
    if remaining_zv:
        print(f"  ℹ️  NOTE: {remaining_zv} cols became constant after imputation (safe to ignore).")

    # 6. Save
    df.to_parquet(OUTPUT, index=True, compression="snappy")
    print(f"💾 Saved → {OUTPUT}")

    # 7. Summary Report
    print("\n" + "="*40)
    print("📊 CLEANING SUMMARY")
    print("="*40)
    for reason, cols in [("zero-variance", zv_cols), ("redundancy", red_cols), ("leakage", lk_cols)]:
        print(f"  {reason:15s}: {len(cols):>3} cols")
        if cols:
            print(f"                 {', '.join(cols[:4])}{'...' if len(cols)>4 else ''}")
    print(f"  {'remaining':15s}: {df.shape[1]} cols × {df.shape[0]} rows")
    print(f"  {'null cells':15s}: {df.isnull().sum().sum()}")
    print(f"  {'dtypes':15s}: {df.dtypes.value_counts().to_dict()}")


if __name__ == "__main__":
    main()