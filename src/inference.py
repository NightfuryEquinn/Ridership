"""
inference.py — HMT-TSF Next-7-Day Ridership Inference

Finds the best-performing HMT-TSF run for a chosen lookback window
(ranked by test Combined%), runs a forward pass on the most recent
feature window, then distributes predicted total ridership across the
12 service lines using their historical percentage shares.

Usage
-----
    python src/inference.py                                  # all defaults
    python src/inference.py --lookback 56
    python src/inference.py --start-date 2025-06-30
    python src/inference.py --lookback 28 --start-date 2025-09-15 --out forecast.csv
    python src/inference.py --list-runs                      # show all available runs

Requirements
------------
    features_aligned.csv must cover at least T_in days before --start-date.
    A trained model.pt must exist in src/outputs/hmttsf/ for the chosen lookback.
    Sequence scalers must exist in the matching data/sequences/ directory.
"""

import os
import sys
import json
import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.hybrid.hmttsf import HMTTSFForecaster

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  ← change these constants to adjust defaults project-wide
# ═════════════════════════════════════════════════════════════════════════════

HMTTSF_OUTPUT_DIR = "src/outputs/hmttsf"           # trained run directories
FEATURES_CSV      = "data/features/features_aligned.csv"

HIST_YEAR         = 2024    # Calendar year for historical service-% shares.
                            # 2024 = first full year all 12 services have data.

T_OUT             = 7       # Forecast horizon (days) — fixed by the dataset.
ADJ_THRESHOLD     = 0.1     # Pearson edge threshold for adjacency (must match training).

# Sequence directory lookup: lookback window (days) → data/sequences/{dir}
SEQ_DIR_MAP: dict[int, str] = {
    7:  "data/sequences/lookback_7",
    14: "data/sequences/lstm",          # original default directory
    28: "data/sequences/lookback_28",
    56: "data/sequences/lookback_56",
    84: "data/sequences/lookback_84",
}

# 12 service-line columns (must match features_aligned.csv column order)
SERVICE_COLS: list[str] = [
    "bus_rkl",
    "bus_rpn",
    "rail_lrt_ampang",
    "rail_mrt_kajang",
    "rail_lrt_kj",
    "rail_monorail",
    "rail_mrt_pjy",
    "rail_ets",
    "rail_intercity",
    "rail_komuter_utara",
    "rail_tebrau",
    "rail_komuter",
]

# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_results(run_dir: str) -> dict | None:
    """Return parsed results.json for a run directory, or None on failure."""
    rp = os.path.join(run_dir, "results.json")
    if not os.path.isfile(rp):
        return None
    with open(rp) as f:
        return json.load(f)


def available_runs(hmttsf_dir: str) -> list[dict]:
    """
    Return a list of dicts (one per run) sorted by lookback then Combined%.
    Each dict has: run_id, run_dir, lookback, combined, r2, hparams, split_dates.
    """
    runs = []
    for name in sorted(os.listdir(hmttsf_dir)):
        run_dir = os.path.join(hmttsf_dir, name)
        if not os.path.isdir(run_dir):
            continue
        r = _load_results(run_dir)
        if r is None:
            continue
        hp = r.get("hparams", {})
        runs.append(dict(
            run_id      = name,
            run_dir     = run_dir,
            lookback    = hp.get("lookback"),
            combined    = r["test_metrics"]["overall"]["Combined"],
            r2          = r["test_metrics"]["overall"]["R2"],
            hparams     = hp,
            split_dates = r.get("split_dates", {}),
        ))
    runs.sort(key=lambda x: (x["lookback"] or 0, -x["combined"]))
    return runs


def find_best_run(hmttsf_dir: str, lookback: int) -> dict:
    """
    Return the run dict with the highest Combined% for the given lookback.
    Raises FileNotFoundError with a helpful message if none found.
    """
    candidates = [r for r in available_runs(hmttsf_dir) if r["lookback"] == lookback]
    if not candidates:
        lbs = sorted({r["lookback"] for r in available_runs(hmttsf_dir) if r["lookback"]})
        raise FileNotFoundError(
            f"No HMT-TSF run found for lookback={lookback} in {hmttsf_dir}.\n"
            f"Available lookbacks: {lbs}\n"
            f"Train a model first:  python src/models/hybrid/hmttsf.py --lookback {lookback}"
        )
    return max(candidates, key=lambda r: r["combined"])


def print_available_runs(hmttsf_dir: str) -> None:
    """Print a summary table of all available runs."""
    runs = available_runs(hmttsf_dir)
    if not runs:
        print(f"No runs found in {hmttsf_dir}")
        return

    print(f"\n{'Run ID':<22} {'Lookback':>9} {'Combined%':>11} {'R2':>8}")
    print("-" * 55)
    prev_lb = None
    for r in runs:
        if r["lookback"] != prev_lb:
            if prev_lb is not None:
                print()
            prev_lb = r["lookback"]
        best_in_lb = max(
            [x for x in runs if x["lookback"] == r["lookback"]],
            key=lambda x: x["combined"],
        )
        marker = " <- best" if r["run_id"] == best_in_lb["run_id"] else ""
        print(
            f"  {r['run_id']:<20} {r['lookback']:>9}d "
            f"{r['combined']:>10.3f}% {r['r2']:>8.4f}{marker}"
        )
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(run: dict, device: torch.device) -> HMTTSFForecaster:
    """
    Reconstruct HMTTSFForecaster from saved weights.
    The Pearson adjacency matrix is a registered buffer stored inside model.pt,
    so no recomputation is needed — it is extracted directly from the state dict.
    """
    model_path = os.path.join(run["run_dir"], "model.pt")
    state = torch.load(model_path, map_location=device, weights_only=False)

    adj = state["adj"]   # registered buffer saved alongside learnable parameters

    hp = run["hparams"]
    sd = run["split_dates"]
    n_temporal_feats = (sd.get("temporal_feat_end", 29)
                        - sd.get("temporal_feat_start", 13))

    model = HMTTSFForecaster(
        n_features       = hp["n_features"],
        T_in             = hp["T_in"],
        T_out            = T_OUT,
        adj              = adj,
        d_model          = hp["d_model"],
        n_tcn_blocks     = hp["n_tcn_blocks"],
        tcn_kernel       = hp.get("tcn_kernel", 3),
        graph_hidden     = hp["graph_hidden"],
        n_regimes        = hp["n_regimes"],
        n_attn_heads     = hp.get("n_attn_heads", 4),
        dropout          = hp["dropout"],
        drop_path        = hp.get("drop_path", 0.1),
        target_idx       = sd.get("target_col_idx", 12),
        use_revin        = hp.get("use_revin", True),
        n_temporal_feats = n_temporal_feats,
    ).to(device)

    model.load_state_dict(state)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Feature window extraction
# ══════════════════════════════════════════════════════════════════════════════

def load_feature_csv(features_csv: str) -> pd.DataFrame:
    """
    Load features_aligned.csv with date index.
    Drops 'is_mco' if present and fills nulls with 0 (pre-launch services).
    Column order is preserved — must match what sequence_builder produced.
    """
    df = (
        pd.read_csv(features_csv, parse_dates=["date"])
        .set_index("date")
        .sort_index()
    )
    df = df.drop(columns=[c for c in ["is_mco"] if c in df.columns])
    df = df.fillna(0.0)
    return df


# Deterministic temporal columns that can be recomputed for any date.
# Keyed by column name → lambda(pd.Timestamp) → float.
_TEMPORAL_UPDATERS: dict[str, object] = {
    "year":        lambda dt: float(dt.year),
    "day_of_year": lambda dt: float(dt.day_of_year),
    "day_of_week": lambda dt: float(dt.dayofweek),          # 0=Mon, 6=Sun
    "month":       lambda dt: float(dt.month),
    "is_weekend":  lambda dt: float(dt.dayofweek >= 5),
    "dow_sin":     lambda dt: float(np.sin(2 * np.pi * dt.dayofweek / 7)),
    "dow_cos":     lambda dt: float(np.cos(2 * np.pi * dt.dayofweek / 7)),
    "month_sin":   lambda dt: float(np.sin(2 * np.pi * (dt.month - 1) / 12)),
    "month_cos":   lambda dt: float(np.cos(2 * np.pi * (dt.month - 1) / 12)),
}


def _extend_df(df: pd.DataFrame, up_to: pd.Timestamp) -> pd.DataFrame:
    """
    Extend df beyond its last date up to `up_to` by:
      - Filling every column with the mean of the last 14 known days.
      - Overwriting deterministic temporal columns with correct values.

    Returns a new DataFrame covering df.index.min() through `up_to`.
    """
    last_csv = df.index.max()
    if up_to <= last_csv:
        return df  # nothing to extend

    extra_dates = pd.date_range(last_csv + timedelta(days=1), up_to, freq="D")
    base_row    = df.iloc[-14:].mean()

    extra_rows = []
    for dt in extra_dates:
        row = base_row.copy()
        for col, fn in _TEMPORAL_UPDATERS.items():
            if col in row.index:
                row[col] = fn(dt)
        extra_rows.append(row.rename(dt))

    extra_df = pd.DataFrame(extra_rows)
    extra_df.index.name = df.index.name
    return pd.concat([df, extra_df])


def get_feature_window(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    T_in: int,
    scaler_X,
) -> tuple[torch.Tensor, int]:
    """
    Build the T_in-row input window ending at start_date (inclusive).

    If start_date is within features_aligned.csv, actual data is used.
    If start_date is beyond the CSV, the window is extended by forward-filling
    the last known row and updating deterministic temporal features for each
    new date.  All other features (ridership, lag, fuel, rainfall, static)
    are held at their last known values — callers receive a warning.

    Returns:
        tensor  : float32 tensor of shape (1, T_in, n_features)
        n_synth : number of rows that were synthetically extended (0 = all real)
    """
    csv_end = df.index.max()
    csv_start = df.index.min()

    # How many rows must be synthesised (beyond CSV end)
    n_synth = max(0, (start_date - csv_end).days)

    # Earliest allowed start_date (need T_in real+synthetic rows)
    window_start = start_date - timedelta(days=T_in - 1)
    if window_start < csv_start:
        raise ValueError(
            f"Not enough history before {start_date.date()} for lookback={T_in}d.\n"
            f"Earliest valid start_date: {(csv_start + timedelta(days=T_in - 1)).date()}"
        )

    # Extend df if needed
    df_ext = _extend_df(df, start_date)

    # Locate the window by date
    end_pos   = df_ext.index.get_loc(start_date) + 1
    start_pos = end_pos - T_in

    window = df_ext.iloc[start_pos:end_pos].values.astype(np.float64)   # (T_in, F)
    scaled = scaler_X.transform(window).astype(np.float32)               # (T_in, F)
    return torch.from_numpy(scaled).unsqueeze(0), n_synth                 # (1, T_in, F)


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def build_future_temporal(
    df: pd.DataFrame,
    forecast_dates: list,
    scaler_X,
    temporal_start: int = 13,
    temporal_end: int = 29,
) -> torch.Tensor:
    """
    Build the (1, T_out, n_temporal) future temporal conditioning tensor for the
    forecast dates, applying the same MinMaxScaler used for training.

    For dates within features_aligned.csv the actual calendar values are used.
    For dates beyond the CSV end, _extend_df recomputes deterministic features
    (day-of-week, sin/cos, weekend, month). Holiday flags are forward-filled
    from the last known row (they are unknown for truly future dates).

    Passing this to the model activates FutureTemporalProjection so that the
    7-step output varies by weekday/weekend rather than being flat.
    """
    last_forecast = max(forecast_dates)
    df_ext = _extend_df(df, last_forecast)

    raw = np.array(
        [df_ext.loc[dt].values[temporal_start:temporal_end] for dt in forecast_dates],
        dtype=np.float32,
    )   # (T_out, n_temporal)

    # Apply the same MinMaxScaler transform: X_scaled = X * scale_ + min_
    scale = scaler_X.scale_[temporal_start:temporal_end].astype(np.float32)
    min_  = scaler_X.min_[temporal_start:temporal_end].astype(np.float32)
    scaled = raw * scale + min_   # (T_out, n_temporal)

    return torch.from_numpy(scaled).unsqueeze(0)   # (1, T_out, n_temporal)


def run_inference(
    model: HMTTSFForecaster,
    X: torch.Tensor,
    scaler_y,
    device: torch.device,
    x_future: torch.Tensor | None = None,
) -> np.ndarray:
    """
    Forward pass → inverse-scale → (T_out,) array of total ridership forecasts.
    Negative predictions are clipped to 0 (possible at RevIN distribution tails).

    x_future: (1, T_out, n_temporal) future calendar features built by
    build_future_temporal(). Activates FutureTemporalProjection so the model
    produces weekday/weekend-varying predictions rather than a flat sequence.
    """
    X = X.to(device)
    if x_future is not None:
        x_future = x_future.to(device)
    with torch.no_grad():
        y_scaled = model(X, x_future)               # (1, T_out)  MinMax space

    y_np   = y_scaled.cpu().numpy().flatten()       # (T_out,)
    y_pred = scaler_y.inverse_transform(            # inverse MinMax → ridership
        y_np.reshape(-1, 1)
    ).flatten()                                     # (T_out,)
    return np.maximum(y_pred, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Historical service-line share computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_service_shares(df: pd.DataFrame, hist_year: int) -> pd.Series:
    """
    For each of the 12 service lines, compute its average daily share of
    total_ridership across hist_year.

    Steps
    -----
    1. Filter to hist_year rows where total_ridership > 0 and all 12 services
       are non-null (excludes pre-launch periods automatically).
    2. Compute daily fraction = service_i / total_ridership.
    3. Average across the year then re-normalise to sum exactly to 1.0.

    Returns a pd.Series indexed by SERVICE_COLS.
    """
    yr = df[df.index.year == hist_year].copy()

    missing = [c for c in SERVICE_COLS + ["total_ridership"] if c not in yr.columns]
    if missing:
        raise ValueError(f"Missing columns in features_aligned.csv: {missing}")

    if yr.empty:
        raise ValueError(f"No rows found for year {hist_year} in features_aligned.csv.")

    # Remove days where a service was not yet operational or total is zero
    yr = yr[yr["total_ridership"] > 0].dropna(subset=SERVICE_COLS)
    if yr.empty:
        raise ValueError(
            f"No complete rows (all 12 services non-null, total > 0) for {hist_year}."
        )

    fractions = yr[SERVICE_COLS].div(yr["total_ridership"], axis=0)
    shares    = fractions.mean()
    shares    = shares / shares.sum()   # re-normalise
    return shares


# ══════════════════════════════════════════════════════════════════════════════
# Actual ridership lookup
# ══════════════════════════════════════════════════════════════════════════════

ACTUAL_DATE_MIN = pd.Timestamp("2019-01-01")
ACTUAL_DATE_MAX = pd.Timestamp("2025-12-31")


def get_actuals(
    df: pd.DataFrame,
    forecast_dates: list,
) -> pd.DataFrame | None:
    """
    Return a DataFrame (indexed 0..T_OUT-1) with actual total_ridership and
    per-service values for each forecast date, or None if no forecast date
    falls within the actual-data window (ACTUAL_DATE_MIN – ACTUAL_DATE_MAX).

    Dates outside the CSV index get NaN rows (no actual available).
    """
    in_window = [
        d for d in forecast_dates
        if ACTUAL_DATE_MIN <= d <= ACTUAL_DATE_MAX
    ]
    if not in_window:
        return None

    cols = ["total_ridership"] + SERVICE_COLS
    rows = []
    for dt in forecast_dates:
        if ACTUAL_DATE_MIN <= dt <= ACTUAL_DATE_MAX and dt in df.index:
            rows.append(df.loc[dt, cols].values.tolist())
        else:
            rows.append([np.nan] * len(cols))

    return pd.DataFrame(rows, columns=cols)


# ══════════════════════════════════════════════════════════════════════════════
# Output formatting
# ══════════════════════════════════════════════════════════════════════════════

_SERVICE_LABELS: dict[str, str] = {
    "bus_rkl":            "Bus KL",
    "bus_rpn":            "Bus Penang",
    "rail_lrt_ampang":    "LRT Ampang",
    "rail_mrt_kajang":    "MRT Kajang",
    "rail_lrt_kj":        "LRT Kelana Jaya",
    "rail_monorail":      "Monorail KL",
    "rail_mrt_pjy":       "MRT Putrajaya",
    "rail_ets":           "ETS",
    "rail_intercity":     "Intercity Rail",
    "rail_komuter_utara": "Komuter Utara",
    "rail_tebrau":        "Tebrau",
    "rail_komuter":       "Komuter",
}


def _fmt(val: float, col_w: int) -> str:
    """Format a numeric value; show '--' when NaN (no actual available)."""
    if np.isnan(val):
        return f"{'--':>{col_w}}"
    return f"{val:>{col_w},.0f}"


def _fmt_diff(pred: float, actual: float, col_w: int) -> str:
    """Format predicted-minus-actual difference with sign; '--' when actual is NaN."""
    if np.isnan(actual):
        return f"{'--':>{col_w}}"
    diff = pred - actual
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:>{col_w - 1},.0f}"


def print_forecast_table(
    forecast_dates: list,
    total_pred: np.ndarray,
    service_preds: pd.DataFrame,
    run: dict,
    hist_year: int,
    shares: pd.Series,
    actuals: "pd.DataFrame | None" = None,
) -> None:
    has_actuals = actuals is not None

    W = 88
    print(f"\n{'=' * W}")
    print(
        f"  HMT-TSF Forecast  |  lookback={run['lookback']}d  "
        f"|  run={run['run_id']}  |  test Combined={run['combined']:.2f}%"
    )
    if has_actuals:
        print(f"  Actual data available for dates within {ACTUAL_DATE_MIN.date()} to {ACTUAL_DATE_MAX.date()}")
    print(f"{'=' * W}")

    col_w = 13
    date_w = 12
    tag_w  = 6   # "Actual", "Pred", "Diff"

    # ── header ────────────────────────────────────────────────────────────
    if has_actuals:
        header  = f"\n  {'Date':<{date_w}}  {'':>{tag_w}}  {'Total':>{col_w}}"
    else:
        header  = f"\n  {'Date':<{date_w}}  {'Total':>{col_w}}"
    header += "".join(f"  {c[:col_w]:>{col_w}}" for c in SERVICE_COLS)

    if has_actuals:
        sep = f"  {'-' * date_w}  {'-' * tag_w}  {'-' * col_w}"
    else:
        sep = f"  {'-' * date_w}  {'-' * col_w}"
    sep += "".join(f"  {'-' * col_w}" for _ in SERVICE_COLS)

    print(header)
    print(sep)

    # ── per-day rows ──────────────────────────────────────────────────────
    for i, dt in enumerate(forecast_dates):
        date_str = str(dt.date())

        if has_actuals:
            act_total = actuals.loc[i, "total_ridership"]
            act_has   = not np.isnan(act_total)

            # Actual row (or placeholder when no actual)
            act_row  = f"  {date_str:<{date_w}}  {'Actual':>{tag_w}}  {_fmt(act_total, col_w)}"
            act_row += "".join(
                f"  {_fmt(actuals.loc[i, c], col_w)}" for c in SERVICE_COLS
            )
            print(act_row)

            # Predicted row
            pred_row  = f"  {'':<{date_w}}  {'Pred':>{tag_w}}  {total_pred[i]:>{col_w},.0f}"
            pred_row += "".join(
                f"  {service_preds.loc[i, c]:>{col_w},.0f}" for c in SERVICE_COLS
            )
            print(pred_row)

            # Diff row (Pred - Actual) — only when actual exists
            if act_has:
                diff_row  = f"  {'':<{date_w}}  {'Diff':>{tag_w}}  "
                diff_row += _fmt_diff(total_pred[i], act_total, col_w)
                diff_row += "".join(
                    f"  {_fmt_diff(service_preds.loc[i, c], actuals.loc[i, c], col_w)}"
                    for c in SERVICE_COLS
                )
                print(diff_row)

            print(f"  {'-' * date_w}  {'-' * tag_w}  {'-' * col_w}"
                  + "".join(f"  {'-' * col_w}" for _ in SERVICE_COLS))

        else:
            row  = f"  {date_str:<{date_w}}  {total_pred[i]:>{col_w},.0f}"
            row += "".join(f"  {service_preds.loc[i, c]:>{col_w},.0f}" for c in SERVICE_COLS)
            print(row)

    # ── weekly summary ────────────────────────────────────────────────────
    if not has_actuals:
        print(sep)
        week_row  = f"  {'7-day total':<{date_w}}  {total_pred.sum():>{col_w},.0f}"
        week_row += "".join(f"  {service_preds[c].sum():>{col_w},.0f}" for c in SERVICE_COLS)
        print(week_row)
    else:
        # Weekly totals for Actual (only dates with data) and Pred
        act_mask  = ~actuals["total_ridership"].isna()
        act_total_sum = actuals.loc[act_mask, "total_ridership"].sum()

        sum_row_a  = f"  {'7-day total':<{date_w}}  {'Actual':>{tag_w}}  "
        sum_row_a += f"{act_total_sum:>{col_w},.0f}" if act_mask.any() else f"{'--':>{col_w}}"
        sum_row_a += "".join(
            f"  {actuals.loc[act_mask, c].sum():>{col_w},.0f}" if act_mask.any() else f"  {'--':>{col_w}}"
            for c in SERVICE_COLS
        )

        sum_row_p  = f"  {'':<{date_w}}  {'Pred':>{tag_w}}  {total_pred.sum():>{col_w},.0f}"
        sum_row_p += "".join(f"  {service_preds[c].sum():>{col_w},.0f}" for c in SERVICE_COLS)

        print(sum_row_a)
        print(sum_row_p)

    # ── share legend ──────────────────────────────────────────────────────
    print(f"\n{'-' * W}")
    print(f"  {hist_year} historical service shares (used for Pred distribution):\n")
    for col in SERVICE_COLS:
        label   = _SERVICE_LABELS.get(col, col)
        bar     = "#" * int(shares[col] * 200)
        print(f"    {label:<18}  {shares[col] * 100:5.2f}%  {bar}")
    print(f"\n{'=' * W}\n")


def build_output_df(
    forecast_dates: list,
    total_pred: np.ndarray,
    service_preds: pd.DataFrame,
    actuals: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    out = pd.DataFrame({
        "date":                [d.date() for d in forecast_dates],
        "total_pred":          total_pred.round().astype(int),
    })
    for col in SERVICE_COLS:
        out[f"{col}_pred"] = service_preds[col].round().astype(int)

    if actuals is not None:
        out.insert(2, "total_actual",
                   actuals["total_ridership"].round().where(
                       ~actuals["total_ridership"].isna(), other=pd.NA
                   ))
        for col in SERVICE_COLS:
            out[f"{col}_actual"] = actuals[col].round().where(
                ~actuals[col].isna(), other=pd.NA
            )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="HMT-TSF: predict the next 7 days of Malaysian transit ridership.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--lookback", type=int, default=14, choices=list(SEQ_DIR_MAP),
        metavar="{" + ",".join(str(k) for k in SEQ_DIR_MAP) + "}",
        help=f"Look-back window in days. A trained model must exist for this value. "
             f"(default: 14 — the best-performing HMT-TSF config, nomco Combined%%=81.46)",
    )
    p.add_argument(
        "--start-date", type=str, default=None, metavar="YYYY-MM-DD",
        help="Last day of the input window (Day 0). Forecast covers Day+1 … Day+7. "
             "Defaults to the last date in features_aligned.csv.",
    )
    p.add_argument(
        "--hist-year", type=int, default=HIST_YEAR, metavar="YEAR",
        help=f"Calendar year used to compute per-service historical shares "
             f"(default: {HIST_YEAR} — first year all 12 services have full data).",
    )
    p.add_argument(
        "--out", type=str, default=None, metavar="PATH",
        help="Optional: save the 7-day forecast to this CSV file.",
    )
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Inference device (default: auto = CUDA if available, else CPU).",
    )
    p.add_argument(
        "--list-runs", action="store_true",
        help="List all trained HMT-TSF runs with their metrics and exit.",
    )
    args = p.parse_args()

    # ── --list-runs shortcut ──────────────────────────────────────────────
    if args.list_runs:
        print_available_runs(HMTTSF_OUTPUT_DIR)
        return

    # ── device ────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n[inference]  device={device}  lookback={args.lookback}d")

    # ── load features CSV ─────────────────────────────────────────────────
    print(f"[inference]  loading {FEATURES_CSV} …")
    df = load_feature_csv(FEATURES_CSV)

    # ── resolve start_date ────────────────────────────────────────────────
    if args.start_date is None:
        start_date = df.index.max()
        print(f"[inference]  --start-date not set; using last CSV date: {start_date.date()}")
    else:
        start_date = pd.Timestamp(args.start_date)

    forecast_dates = [start_date + timedelta(days=i + 1) for i in range(T_OUT)]
    print(f"[inference]  forecast window: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")

    # ── find best run ─────────────────────────────────────────────────────
    print(f"[inference]  scanning {HMTTSF_OUTPUT_DIR} for lookback={args.lookback}d …")
    run = find_best_run(HMTTSF_OUTPUT_DIR, args.lookback)
    print(f"[inference]  best run : {run['run_id']}")
    print(f"[inference]  metrics  : Combined={run['combined']:.3f}%  R2={run['r2']:.4f}")

    # ── load sequence scalers ─────────────────────────────────────────────
    seq_dir = SEQ_DIR_MAP.get(args.lookback)
    if not seq_dir or not os.path.isdir(seq_dir):
        raise FileNotFoundError(
            f"Sequence directory for lookback={args.lookback}d not found: {seq_dir!r}\n"
            f"Build it first:  python src/features/sequence_builder.py --T-in {args.lookback}"
        )
    scaler_X = joblib.load(os.path.join(seq_dir, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(seq_dir, "scaler_y.pkl"))
    print(f"[inference]  loaded scalers from {seq_dir}/")

    # ── extract input window ──────────────────────────────────────────────
    csv_end = df.index.max()
    print(f"[inference]  extracting {args.lookback}-day window ending {start_date.date()} …")
    X, n_synth = get_feature_window(df, start_date, args.lookback, scaler_X)
    if n_synth > 0:
        print(
            f"[inference]  WARNING: {n_synth} day(s) beyond CSV end ({csv_end.date()}) "
            f"were synthesised using the mean of the last 14 known days.\n"
            f"             Temporal features (year/month/day_of_week/sin-cos/is_weekend) "
            f"are exact.\n"
            f"             Ridership, lag, fuel, rainfall, and static features "
            f"are averaged over the 14 days ending {csv_end.date()}."
        )

    # ── build future temporal conditioning ───────────────────────────────
    sd = run["split_dates"]
    t_start = sd.get("temporal_feat_start", 13)
    t_end   = sd.get("temporal_feat_end", 29)
    x_future = build_future_temporal(df, forecast_dates, scaler_X, t_start, t_end)
    print(f"[inference]  future temporal conditioning: {tuple(x_future.shape)}  "
          f"(cols {t_start}–{t_end - 1}, day-of-week/weekend/holiday/sin-cos)")

    # ── load model ────────────────────────────────────────────────────────
    print(f"[inference]  loading model from {run['run_dir']} …")
    model = load_model(run, device)

    # ── forward pass ──────────────────────────────────────────────────────
    print(f"[inference]  running inference …")
    total_pred = run_inference(model, X, scaler_y, device, x_future=x_future)

    # ── historical shares ─────────────────────────────────────────────────
    print(f"[inference]  computing {args.hist_year} service shares …")
    shares = compute_service_shares(df, args.hist_year)

    # ── distribute total ridership across 12 services ─────────────────────
    service_preds = pd.DataFrame(
        {col: total_pred * shares[col] for col in SERVICE_COLS},
        index=range(T_OUT),
    )

    # ── fetch actuals for any forecast dates within the historical window ──
    actuals = get_actuals(df, forecast_dates)
    if actuals is not None:
        n_with_actual = (~actuals["total_ridership"].isna()).sum()
        print(f"[inference]  actual ridership available for {n_with_actual}/{T_OUT} forecast day(s)")

    # ── print forecast table ──────────────────────────────────────────────
    print_forecast_table(
        forecast_dates, total_pred, service_preds,
        run=run, hist_year=args.hist_year, shares=shares,
        actuals=actuals,
    )

    # ── optional CSV export ───────────────────────────────────────────────
    if args.out:
        out_df = build_output_df(forecast_dates, total_pred, service_preds, actuals=actuals)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"[inference]  forecast saved -> {args.out}")


if __name__ == "__main__":
    main()
