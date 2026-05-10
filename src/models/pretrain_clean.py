"""
pretrain_clean.py
Cleans feature_matrix_lstm.parquet for multi-service LSTM training.

Handles:
  1. Zero-variance  → nunique() <= 1
  2. Redundancy     → Pairwise |correlation| > threshold (greedy drop,
                       raw service target columns are protected)
  3. Leakage        → Forward-looking name patterns (lead/future/next/ahead/t+N)
  4. NaN            → ffill → bfill → zero-fill fallback

Multi-service notes:
  - Target columns (raw ridership per service, no derived suffixes) are
    auto-detected and excluded from redundancy dropping — they can be
    correlated with each other but all are required as prediction targets.
  - Correlation-based leakage is omitted: it is not meaningful for
    multi-target setups (a lagged feature with high corr to its own target
    is a legitimate predictor, not leakage).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT  = Path("data/features/feature_matrix_lstm.parquet")
OUTPUT = Path("data/features/feature_matrix_lstm_clean.parquet")

CORR_THRESHOLD = 0.95

# Matches raw service-level ridership columns only.
# Negative lookahead excludes all derived features (roll, lag, zscore, anomaly,
# cyclical encodings, and anything containing a digit like "7d").
TARGET_COL_RE = re.compile(
    r"^ridership__(bus|rail)_(?!.*(?:roll|lag|zscore|anomaly|sin|cos|\d))[a-z_]+$"
)

# Forward-looking temporal keywords that signal data leakage.
LEAKAGE_NAME_RE = re.compile(r"(?i)(lead|future|next|ahead|_p\d+|t\+\d+|forward)")


# ============================================================================
# HELPERS
# ============================================================================
def audit(df: pd.DataFrame, label: str) -> None:
    total_null = df.isnull().sum().sum()
    null_pct   = total_null / max(df.size, 1)
    print(f"[{label}]  shape={df.shape}  nulls={total_null:,} ({null_pct:.2%})")


def detect_target_cols(df: pd.DataFrame) -> list[str]:
    """Auto-detect raw service ridership columns (multi-service targets)."""
    return [c for c in df.columns if TARGET_COL_RE.match(c)]


def drop_zero_variance(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns with exactly 1 unique value (carry no signal)."""
    mask    = df.nunique(dropna=False) <= 1
    dropped = df.columns[mask].tolist()
    return df.drop(columns=dropped), dropped


def drop_redundant(
    df: pd.DataFrame,
    protect_cols: list[str],
    threshold: float = CORR_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Greedy redundancy pruning: drop one column from each pair with
    |correlation| > threshold. Columns in protect_cols are never dropped.

    The greedy rule: build the upper-triangle correlation matrix; for each
    column, if any value in that column exceeds the threshold, mark it for
    removal. This naturally keeps the 'earlier' column in each correlated pair
    and removes the later one — deterministic and reproducible.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return df, []

    corr   = num_df.corr().abs()
    upper  = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = [
        col for col in upper.columns
        if col not in protect_cols and upper[col].gt(threshold).any()
    ]
    return df.drop(columns=to_drop), to_drop


def detect_leakage(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop columns whose names contain forward-looking temporal keywords.
    Correlation-based leakage detection is intentionally omitted for
    multi-service: high correlation between a lagged feature and its own
    target is legitimate signal, not leakage.
    """
    dropped = [c for c in df.columns if LEAKAGE_NAME_RE.search(c)]
    return df.drop(columns=dropped), dropped


def impute_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill → back-fill → zero-fill fallback. Preserves dtypes."""
    df = df.ffill().bfill()
    remaining = df.isnull().sum().sum()
    if remaining:
        print(f"  ⚠️  {remaining:,} NaNs remain after ffill/bfill — zero-filling.")
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

    target_cols = detect_target_cols(df)
    print(f"🎯 Targets detected: {len(target_cols)} services")
    for c in target_cols:
        print(f"     {c}")

    # 1. Zero-variance
    df, zv_cols = drop_zero_variance(df)
    print(f"\n🗑️  Zero-variance : {len(zv_cols):>3} cols dropped")

    # 2. Redundancy — protect all service target columns
    df, red_cols = drop_redundant(df, protect_cols=target_cols)
    print(f"🔗 Redundancy    : {len(red_cols):>3} cols dropped (|corr| > {CORR_THRESHOLD})")

    # 3. Leakage — name-heuristic only
    df, lk_cols = detect_leakage(df)
    print(f"🔒 Leakage       : {len(lk_cols):>3} cols dropped (forward-looking names)")

    total_dropped = len(zv_cols) + len(red_cols) + len(lk_cols)
    print(f"\n✅ Total dropped: {total_dropped} → {df.shape[1]} cols remain")

    # 4. Imputation
    audit(df, "pre-impute")
    df = impute_time_series(df)
    audit(df, "post-impute")

    # 5. Sanity checks
    assert df.isnull().sum().sum() == 0, "NaNs remain after imputation!"
    assert not df.duplicated().any(), "Duplicate rows detected!"

    remaining_zv = (df.nunique(dropna=False) <= 1).sum()
    if remaining_zv:
        print(f"  ℹ️  {remaining_zv} col(s) became constant after imputation.")

    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_targets:
        raise RuntimeError(f"Target columns lost during cleaning: {missing_targets}")

    # 6. Save
    df.to_parquet(OUTPUT, index=True, compression="snappy")
    print(f"💾 Saved → {OUTPUT}")

    # 7. Summary
    print("\n" + "=" * 40)
    print("📊 CLEANING SUMMARY")
    print("=" * 40)
    for reason, cols in [
        ("zero-variance", zv_cols),
        ("redundancy",    red_cols),
        ("leakage",       lk_cols),
    ]:
        print(f"  {reason:15s}: {len(cols):>3} cols")
        if cols:
            preview = ", ".join(cols[:4]) + ("..." if len(cols) > 4 else "")
            print(f"                 {preview}")
    print(f"  {'targets':15s}: {len(target_cols)} services")
    print(f"  {'remaining':15s}: {df.shape[1]} cols × {df.shape[0]} rows")
    print(f"  {'null cells':15s}: {df.isnull().sum().sum()}")
    print(f"  {'dtypes':15s}: {dict(df.dtypes.value_counts())}")


if __name__ == "__main__":
    main()