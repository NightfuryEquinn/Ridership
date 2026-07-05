"""
inference.py — HMT-TSF Next-7-Day Ridership Inference

Finds the best-performing HMT-TSF run for a chosen lookback window
(ranked by test Combined%), runs a forward pass on the most recent
feature window, then distributes predicted total ridership across the
12 service lines using calendar-aware historical shares.

Usage
-----
    python src/inference.py                                  # all defaults
    python src/inference.py --lookback 56
    python src/inference.py --start-date 2026-01-01   # forecast 2026-01-01 … 2026-01-07
    python src/inference.py --lookback 28 --start-date 2025-09-15 --out forecast.csv
    python src/inference.py --list-runs                      # show all available runs
    python src/inference.py --run-id 20260611_144949 --output-dir hmttsf_feat_reduced
    python src/inference.py --share-mode dow_recent --regime nomco

Requirements
------------
    features_aligned*.csv must cover at least T_in days before the day before --start-date.
    Actual ridership for comparison is read from data/raw/ridership_headline.csv.
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

from src.models.hybrid.hmttsf import HMTTSFForecaster, _KEPT_FEAT_INDICES

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  ← change these constants to adjust defaults project-wide
# ═════════════════════════════════════════════════════════════════════════════

HMTTSF_OUTPUT_DIR_FR = "src/outputs/hmttsf_feat_reduced"
HMTTSF_OUTPUT_DIR    = "src/outputs/hmttsf"
RIDERSHIP_HEADLINE_CSV = "data/raw/ridership_headline.csv"
HOLIDAY_CALENDAR_CSV = "data/cleaned/holiday_daily_features.csv"

FEATURES_CSV_MAP: dict[str, str] = {
    "nomco": "data/features/features_aligned_no_mco.csv",
    "mco":   "data/features/features_aligned.csv",
}

HIST_YEAR         = 2024    # Calendar year for annual share mode.
T_OUT             = 7       # Forecast horizon (days) — fixed by the dataset.

# Sequence directory lookup: (regime, lookback) → data/sequences/{dir}
SEQ_DIR_MAP: dict[tuple[str, int], str] = {
    ("nomco", 7):  "data/sequences/lookback_7",
    ("nomco", 14): "data/sequences/lstm",
    ("nomco", 28): "data/sequences/lookback_28",
    ("nomco", 56): "data/sequences/lookback_56",
    ("nomco", 84): "data/sequences/lookback_84",
    ("mco", 7):    "data/sequences/lookback_7_mco",
    ("mco", 14):   "data/sequences/lstm_mco",
    ("mco", 28):   "data/sequences/lookback_28_mco",
    ("mco", 56):   "data/sequences/lookback_56_mco",
    ("mco", 84):   "data/sequences/lookback_84_mco",
}

LOOKBACK_CHOICES = sorted({lb for _, lb in SEQ_DIR_MAP})

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

LAG_COLS = ["ridership_lag_7", "ridership_lag_14", "ridership_lag_28"]

HOLIDAY_CALENDAR_COLS = [
    "is_public_holiday", "is_school_holiday", "is_holiday_any",
    "days_to_next_public_hol", "days_since_last_public_hol",
    "days_to_next_school_hol", "days_since_last_school_hol",
    "day_of_week", "month", "is_weekend",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
]

SHARE_MODES = ("annual", "dow_recent", "calendar", "window", "blend")

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


def infer_regime(split_dates: dict) -> str:
    """Return 'nomco' or 'mco' from the training split start date."""
    start = split_dates.get("train", {}).get("start", "2019-01-01")
    return "nomco" if start >= "2022-01-01" else "mco"


def is_feat_reduced(run: dict) -> bool:
    return run["hparams"].get("n_features", 79) == len(_KEPT_FEAT_INDICES)


def _split_fingerprint(split_dates: dict) -> tuple:
    """Hashable key for matching a run to its sequence directory."""
    return (
        split_dates.get("train", {}).get("start"),
        split_dates.get("val", {}).get("start"),
        split_dates.get("test", {}).get("start"),
        split_dates.get("T_in"),
        split_dates.get("n_features", 79),
    )


def resolve_seq_dir(lookback: int, split_dates: dict) -> str:
    """
    Resolve the sequence directory whose split_dates.json matches the run.
    Falls back to regime-based lookup when no exact match is found.
    """
    fp = _split_fingerprint(split_dates)
    regime = infer_regime(split_dates)
    candidates = {SEQ_DIR_MAP[(regime, lookback)]}
    # Also scan sibling dirs for exact split match
    for key, path in SEQ_DIR_MAP.items():
        if key[1] == lookback:
            candidates.add(path)

    for path in sorted(candidates):
        meta_path = os.path.join(path, "split_dates.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if _split_fingerprint(meta) == fp:
            return path

    fallback = SEQ_DIR_MAP.get((regime, lookback))
    if fallback and os.path.isdir(fallback):
        return fallback
    raise FileNotFoundError(
        f"No sequence directory found for lookback={lookback}, regime={regime}.\n"
        f"Build sequences first:  python src/features/sequence_builder.py --T-in {lookback}"
    )


def available_runs(hmttsf_dir: str) -> list[dict]:
    """
    Return a list of dicts (one per run) sorted by lookback then Combined%.
    Each dict has: run_id, run_dir, lookback, combined, r2, hparams, split_dates.
    """
    if not os.path.isdir(hmttsf_dir):
        return []
    runs = []
    for name in sorted(os.listdir(hmttsf_dir)):
        run_dir = os.path.join(hmttsf_dir, name)
        if not os.path.isdir(run_dir):
            continue
        r = _load_results(run_dir)
        if r is None:
            continue
        hp = r.get("hparams", {})
        sd = r.get("split_dates", {})
        runs.append(dict(
            run_id      = name,
            run_dir     = run_dir,
            lookback    = hp.get("lookback"),
            combined    = r["test_metrics"]["overall"]["Combined"],
            r2          = r["test_metrics"]["overall"]["R2"],
            hparams     = hp,
            split_dates = sd,
            regime      = infer_regime(sd),
            feat_reduced= hp.get("n_features", 79) == len(_KEPT_FEAT_INDICES),
        ))
    runs.sort(key=lambda x: (x["lookback"] or 0, -x["combined"]))
    return runs


def resolve_run(
    hmttsf_dir: str,
    lookback: int,
    regime: str = "nomco",
    run_id: str | None = None,
) -> dict:
    """
    Return the run dict for a pinned run_id or the best Combined% for lookback/regime.
    regime='auto' skips regime filtering.
    """
    runs = available_runs(hmttsf_dir)
    if run_id:
        match = [r for r in runs if r["run_id"] == run_id]
        if not match:
            raise FileNotFoundError(
                f"Run {run_id!r} not found in {hmttsf_dir}.\n"
                f"Use --list-runs to see available runs."
            )
        run = match[0]
        if run["lookback"] != lookback:
            print(
                f"[inference]  WARNING: run {run_id} was trained with "
                f"lookback={run['lookback']}d (requested {lookback}d)."
            )
        return run

    candidates = [r for r in runs if r["lookback"] == lookback]
    if regime != "auto":
        candidates = [r for r in candidates if r["regime"] == regime]
    if not candidates:
        lbs = sorted({r["lookback"] for r in runs if r["lookback"]})
        raise FileNotFoundError(
            f"No HMT-TSF run found for lookback={lookback}, regime={regime} "
            f"in {hmttsf_dir}.\n"
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

    print(f"\n{'Run ID':<22} {'LB':>3} {'Regime':>6} {'Feats':>5} "
          f"{'Combined%':>11} {'R2':>8}")
    print("-" * 65)
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
        nf = r["hparams"].get("n_features", 79)
        print(
            f"  {r['run_id']:<20} {r['lookback']:>3}d "
            f"{r['regime']:>6} {nf:>5} "
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

    adj = state["adj"]

    hp = run["hparams"]
    sd = run["split_dates"]
    temporal_indices = sd.get("temporal_feat_indices")
    if temporal_indices:
        n_temporal_feats = len(temporal_indices)
    else:
        n_temporal_feats = (sd.get("temporal_feat_end", 29)
                            - sd.get("temporal_feat_start", 13))

    target_idx = sd.get("target_col_idx", 12)
    if is_feat_reduced(run):
        # total_ridership stays at index 12 in the 53-feature contiguous space
        target_idx = _KEPT_FEAT_INDICES.index(target_idx) if target_idx in _KEPT_FEAT_INDICES else 12

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
        target_idx       = target_idx,
        use_revin        = hp.get("use_revin", True),
        n_temporal_feats = n_temporal_feats,
    ).to(device)

    model.load_state_dict(state)
    model.eval()
    model.float()
    return model


def load_boost_corrector(run: dict) -> dict | None:
    """Load saved residual booster if the run applied it during training."""
    if not run["hparams"].get("boost_applied"):
        return None
    path = os.path.join(run["run_dir"], "boost_corrector.pkl")
    if not os.path.isfile(path):
        print(f"[inference]  WARNING: boost_applied=true but {path} not found.")
        return None
    return joblib.load(path)


# ══════════════════════════════════════════════════════════════════════════════
# Feature window extraction
# ══════════════════════════════════════════════════════════════════════════════

def load_feature_csv(features_csv: str) -> pd.DataFrame:
    """
    Load features_aligned CSV with date index.
    Drops non-numeric columns and fills nulls with 0 (pre-launch services).
    """
    df = (
        pd.read_csv(features_csv, parse_dates=["date"])
        .set_index("date")
        .sort_index()
    )
    df = df.drop(columns=[c for c in ["is_mco"] if c in df.columns])
    df = df.fillna(0.0)
    return df


_holiday_cache: pd.DataFrame | None = None


def _load_holiday_calendar(path: str = HOLIDAY_CALENDAR_CSV) -> pd.DataFrame:
    global _holiday_cache
    if _holiday_cache is None:
        hol = (
            pd.read_csv(path, parse_dates=["date"])
            .set_index("date")
            .sort_index()
        )
        _holiday_cache = hol
    return _holiday_cache


def _merge_holiday_calendar(df: pd.DataFrame, up_to: pd.Timestamp) -> pd.DataFrame:
    """Overlay known holiday/calendar columns from the cleaned holiday CSV."""
    hol = _load_holiday_calendar()
    cols = [c for c in HOLIDAY_CALENDAR_COLS if c in hol.columns and c in df.columns]
    if not cols:
        return df
    out = df.copy()
    hol_slice = hol.loc[hol.index.min():up_to, cols]
    for col in cols:
        out[col] = hol_slice[col].reindex(out.index).combine_first(out[col])
    return out


def _recompute_ridership_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute autoregressive lag columns from total_ridership."""
    if "total_ridership" not in df.columns:
        return df
    out = df.copy()
    total = out["total_ridership"]
    for lag, col in zip([7, 14, 28], LAG_COLS):
        if col in out.columns:
            out[col] = total.shift(lag)
    return out.fillna(0.0)


# Deterministic temporal columns recomputed for any date.
_TEMPORAL_UPDATERS: dict[str, object] = {
    "year":        lambda dt: float(dt.year),
    "day_of_year": lambda dt: float(dt.day_of_year),
    "day_of_week": lambda dt: float(dt.dayofweek),
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
      - Filling non-temporal columns with the mean of the last 14 known days.
      - Overwriting deterministic temporal columns with correct values.
      - Merging holiday calendar columns from holiday_daily_features.csv.
      - Recomputing ridership_lag_* from total_ridership.
    """
    df = _merge_holiday_calendar(df, max(df.index.max(), up_to))
    last_csv = df.index.max()
    if up_to <= last_csv:
        return _recompute_ridership_lags(df)

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
    out = pd.concat([df, extra_df])
    out = _merge_holiday_calendar(out, up_to)
    return _recompute_ridership_lags(out)



def get_feature_window(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    T_in: int,
    scaler_X,
    feat_reduced: bool = False,
) -> tuple[torch.Tensor, int]:
    """
    Build the T_in-row input window ending at start_date (inclusive).
    Returns float32 tensor (1, T_in, n_features) and count of synthetic rows.
    """
    csv_end = df.index.max()
    csv_start = df.index.min()
    n_synth = max(0, (start_date - csv_end).days)

    window_start = start_date - timedelta(days=T_in - 1)
    if window_start < csv_start:
        raise ValueError(
            f"Not enough history before {start_date.date()} for lookback={T_in}d.\n"
            f"Earliest valid start_date: {(csv_start + timedelta(days=T_in - 1)).date()}"
        )

    df_ext = _extend_df(df, start_date)
    end_pos   = df_ext.index.get_loc(start_date) + 1
    start_pos = end_pos - T_in

    window = df_ext.iloc[start_pos:end_pos].values.astype(np.float64)
    scaled = scaler_X.transform(window).astype(np.float32)
    if feat_reduced:
        scaled = scaled[:, _KEPT_FEAT_INDICES]
    return torch.from_numpy(scaled).unsqueeze(0), n_synth


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def build_future_temporal(
    df: pd.DataFrame,
    forecast_dates: list,
    scaler_X,
    temporal_indices: list[int],
) -> torch.Tensor:
    """
    Build (1, T_out, n_temporal) future calendar conditioning for forecast dates.
    Uses non-contiguous temporal_feat_indices from split_dates.json.
    """
    last_forecast = max(forecast_dates)
    df_ext = _extend_df(df, last_forecast)

    raw = np.array(
        [df_ext.loc[dt].values[temporal_indices] for dt in forecast_dates],
        dtype=np.float32,
    )

    scale = scaler_X.scale_[temporal_indices].astype(np.float32)
    min_  = scaler_X.min_[temporal_indices].astype(np.float32)
    scaled = raw * scale + min_

    return torch.from_numpy(scaled).unsqueeze(0)


def run_inference(
    model: HMTTSFForecaster,
    X: torch.Tensor,
    scaler_y,
    device: torch.device,
    x_future: torch.Tensor | None = None,
    boost_pack: dict | None = None,
) -> np.ndarray:
    """
    Forward pass → optional CatBoost residual boost → inverse-scale → (T_out,) total forecast.
    """
    X = X.to(device)
    if x_future is not None:
        x_future = x_future.to(device)
    with torch.no_grad():
        y_scaled = model(X, x_future)

    y_np   = y_scaled.cpu().numpy().flatten()
    y_pred = scaler_y.inverse_transform(y_np.reshape(-1, 1)).flatten()

    if boost_pack is not None:
        corrector   = boost_pack["corrector"]
        boost_scale = boost_pack.get("boost_scale", 0.5)
        x_np = X.cpu().numpy()
        correction = corrector(x_np)
        if correction.ndim == 1:
            correction = correction.reshape(1, -1)
        y_pred = y_pred + boost_scale * correction.flatten()

    return np.maximum(y_pred, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Service-line share computation
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_active_services(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp | None,
) -> tuple[list[str], list[str]]:
    if cutoff_date is not None:
        df_before = df[df.index <= cutoff_date]
        active = [
            c for c in SERVICE_COLS
            if c in df_before.columns
            and (df_before[c].notna() & (df_before[c] > 0)).any()
        ]
        excluded = [c for c in SERVICE_COLS if c not in active]
    else:
        active = list(SERVICE_COLS)
        excluded = []
    return active, excluded


def _daily_fractions(sub: pd.DataFrame, active_services: list[str]) -> pd.DataFrame:
    sub = sub.dropna(subset=active_services)
    totals = sub[active_services].sum(axis=1)
    sub = sub[totals > 0]
    if sub.empty:
        return pd.DataFrame(columns=active_services)
    return sub[active_services].div(sub[active_services].sum(axis=1), axis=0)


def _normalize_shares_row(
    shares: pd.Series,
    active_services: list[str],
) -> pd.Series:
    out = pd.Series(0.0, index=SERVICE_COLS)
    active = shares[active_services]
    if active.sum() > 0:
        active = active / active.sum()
    out[active_services] = active
    return out


def _annual_shares(
    df: pd.DataFrame,
    hist_year: int,
    active_services: list[str],
) -> pd.Series:
    yr = df[df.index.year == hist_year].copy()
    fr = _daily_fractions(yr, active_services)
    if fr.empty:
        raise ValueError(f"No complete rows for {hist_year}.")
    return _normalize_shares_row(fr.mean(), active_services)


def _dow_recent_shares(
    df: pd.DataFrame,
    forecast_date: pd.Timestamp,
    active_services: list[str],
    cutoff_date: pd.Timestamp,
    dow_weeks: int,
) -> pd.Series:
    lookback_start = cutoff_date - timedelta(days=dow_weeks * 7)
    sub = df[(df.index > lookback_start) & (df.index <= cutoff_date)]
    sub = sub[sub.index.dayofweek == forecast_date.dayofweek]
    fr = _daily_fractions(sub, active_services)
    if fr.empty:
        return _annual_shares(df, HIST_YEAR, active_services)
    return _normalize_shares_row(fr.mean(), active_services)


def _calendar_shares(
    df: pd.DataFrame,
    forecast_date: pd.Timestamp,
    active_services: list[str],
    cutoff_date: pd.Timestamp,
    holiday_calendar: pd.DataFrame,
) -> pd.Series:
    lookback_start = cutoff_date - timedelta(days=365)
    sub = df[(df.index > lookback_start) & (df.index <= cutoff_date)]
    is_hol = False
    if forecast_date in holiday_calendar.index:
        is_hol = bool(holiday_calendar.loc[forecast_date, "is_holiday_any"])
    elif "is_holiday_any" in sub.columns:
        is_hol = False

    if "is_holiday_any" in sub.columns:
        sub = sub[sub["is_holiday_any"] == (1 if is_hol else 0)]
    fr = _daily_fractions(sub, active_services)
    if fr.empty:
        return _dow_recent_shares(df, forecast_date, active_services, cutoff_date, 8)
    return _normalize_shares_row(fr.mean(), active_services)


def _window_shares(
    window_df: pd.DataFrame,
    active_services: list[str],
) -> pd.Series:
    fr = _daily_fractions(window_df, active_services)
    if fr.empty:
        return pd.Series(1.0 / len(active_services), index=SERVICE_COLS)
    return _normalize_shares_row(fr.mean(), active_services)


def compute_shares_by_day(
    df: pd.DataFrame,
    forecast_dates: list,
    active_services: list[str],
    mode: str,
    hist_year: int,
    cutoff_date: pd.Timestamp,
    window_df: pd.DataFrame | None = None,
    dow_weeks: int = 8,
    share_alpha: float = 0.3,
) -> pd.DataFrame:
    """
    Return a (T_out, len(SERVICE_COLS)) DataFrame of daily service shares.
    """
    holiday_cal = _load_holiday_calendar()
    rows = []
    for dt in forecast_dates:
        if mode == "annual":
            s = _annual_shares(df, hist_year, active_services)
        elif mode == "dow_recent":
            s = _dow_recent_shares(df, dt, active_services, cutoff_date, dow_weeks)
        elif mode == "calendar":
            s = _calendar_shares(df, dt, active_services, cutoff_date, holiday_cal)
        elif mode == "window":
            if window_df is None:
                raise ValueError("window share mode requires input window DataFrame.")
            s = _window_shares(window_df, active_services)
        elif mode == "blend":
            if window_df is None:
                s = _dow_recent_shares(df, dt, active_services, cutoff_date, dow_weeks)
            else:
                sw = _window_shares(window_df, active_services)
                sd = _dow_recent_shares(df, dt, active_services, cutoff_date, dow_weeks)
                blended = share_alpha * sw + (1.0 - share_alpha) * sd
                s = _normalize_shares_row(blended, active_services)
        else:
            raise ValueError(f"Unknown share mode: {mode!r}")
        rows.append(s)
    return pd.DataFrame(rows, columns=SERVICE_COLS)


def allocate_service_preds(
    total_pred: np.ndarray,
    shares_by_day: pd.DataFrame,
) -> pd.DataFrame:
    """Multiply daily totals by shares, then apply largest-remainder integer rounding."""
    out = pd.DataFrame(index=range(len(total_pred)), columns=SERVICE_COLS, dtype=float)
    for i in range(len(total_pred)):
        total_int = int(round(total_pred[i]))
        frac = shares_by_day.loc[i, SERVICE_COLS].values.astype(float)
        raw = frac * total_int
        floored = np.floor(raw).astype(int)
        deficit = total_int - floored.sum()
        if deficit > 0:
            order = np.argsort(-(raw - floored))
            floored[order[:deficit]] += 1
        elif deficit < 0:
            order = np.argsort(raw - floored)
            for j in order:
                if deficit == 0:
                    break
                if floored[j] > 0:
                    floored[j] -= 1
                    deficit += 1
        out.loc[i] = floored
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Actual ridership lookup (ridership_headline.csv)
# ══════════════════════════════════════════════════════════════════════════════

def load_ridership_headline(path: str) -> pd.DataFrame:
    """Load raw headline ridership with date index and computed total_ridership."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df = df.set_index("date").sort_index()

    for col in SERVICE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    present = [c for c in SERVICE_COLS if c in df.columns]
    df["total_ridership"] = df[present].sum(axis=1, min_count=1)
    return df


def get_actuals(
    headline_df: pd.DataFrame,
    forecast_dates: list,
) -> pd.DataFrame | None:
    """Return actual total and per-service values for each forecast date."""
    if not any(dt in headline_df.index for dt in forecast_dates):
        return None

    cols = ["total_ridership"] + SERVICE_COLS
    rows = []
    for dt in forecast_dates:
        if dt in headline_df.index:
            rows.append(headline_df.loc[dt, cols].values.tolist())
        else:
            rows.append([np.nan] * len(cols))

    return pd.DataFrame(rows, columns=cols)


def _compute_line_metrics(
    service_preds: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """Per-service MAE and MAPE over forecast days with available actuals."""
    rows = []
    mask = ~actuals["total_ridership"].isna()
    for col in SERVICE_COLS:
        act = actuals.loc[mask, col].values.astype(float)
        pred = service_preds.loc[mask, col].values.astype(float)
        valid = ~np.isnan(act) & (act > 0)
        if valid.sum() == 0:
            rows.append({"service": col, "MAE": np.nan, "MAPE%": np.nan, "n": 0})
            continue
        err = np.abs(pred[valid] - act[valid])
        rows.append({
            "service": col,
            "MAE": float(err.mean()),
            "MAPE%": float((err / act[valid]).mean() * 100),
            "n": int(valid.sum()),
        })
    return pd.DataFrame(rows)


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
    if np.isnan(val):
        return f"{'--':>{col_w}}"
    return f"{val:>{col_w},.0f}"


def _fmt_diff(pred: float, actual: float, col_w: int) -> str:
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
    shares_by_day: pd.DataFrame,
    share_mode: str,
    actuals: pd.DataFrame | None = None,
    active_services: list[str] | None = None,
    excluded_services: list[str] | None = None,
) -> None:
    has_actuals = actuals is not None
    active_services   = active_services   or list(SERVICE_COLS)
    excluded_services = excluded_services or []
    avg_shares = shares_by_day[SERVICE_COLS].mean()

    W = 88
    nf = run["hparams"].get("n_features", 79)
    boost = run["hparams"].get("boost_applied", False)
    print(f"\n{'=' * W}")
    print(
        f"  HMT-TSF Forecast  |  lookback={run['lookback']}d  "
        f"|  run={run['run_id']}  |  test Combined={run['combined']:.2f}%"
    )
    print(
        f"  regime={run.get('regime', infer_regime(run['split_dates']))}  "
        f"features={nf}  boost={boost}  share_mode={share_mode}"
    )
    if has_actuals:
        print(f"  Actual data from {RIDERSHIP_HEADLINE_CSV} (where available)")
    print(f"{'=' * W}")

    col_w = 13
    date_w = 12
    tag_w  = 6

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

    for i, dt in enumerate(forecast_dates):
        date_str = str(dt.date())

        if has_actuals:
            act_total = actuals.loc[i, "total_ridership"]
            act_has   = not np.isnan(act_total)

            act_row  = f"  {date_str:<{date_w}}  {'Actual':>{tag_w}}  {_fmt(act_total, col_w)}"
            act_row += "".join(
                f"  {_fmt(actuals.loc[i, c], col_w)}" for c in SERVICE_COLS
            )
            print(act_row)

            pred_row  = f"  {'':<{date_w}}  {'Pred':>{tag_w}}  {total_pred[i]:>{col_w},.0f}"
            pred_row += "".join(
                f"  {service_preds.loc[i, c]:>{col_w},.0f}" for c in SERVICE_COLS
            )
            print(pred_row)

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

    if not has_actuals:
        print(sep)
        week_row  = f"  {'7-day total':<{date_w}}  {total_pred.sum():>{col_w},.0f}"
        week_row += "".join(f"  {service_preds[c].sum():>{col_w},.0f}" for c in SERVICE_COLS)
        print(week_row)
    else:
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

        # Per-line error summary
        line_metrics = _compute_line_metrics(service_preds, actuals)
        valid_lines = line_metrics[line_metrics["n"] > 0]
        if not valid_lines.empty:
            print(f"\n{'-' * W}")
            print("  Per-service error (days with actuals):\n")
            print(f"  {'Service':<18}  {'MAE':>10}  {'MAPE%':>8}  {'n':>4}")
            print(f"  {'-' * 18}  {'-' * 10}  {'-' * 8}  {'-' * 4}")
            for _, row in valid_lines.iterrows():
                label = _SERVICE_LABELS.get(row["service"], row["service"])
                print(
                    f"  {label:<18}  {row['MAE']:>10,.0f}  "
                    f"{row['MAPE%']:>8.2f}  {int(row['n']):>4}"
                )

            # Weekly share drift
            if act_mask.any():
                act_mix = actuals.loc[act_mask, SERVICE_COLS].sum()
                act_mix = act_mix / act_mix.sum()
                pred_mix = service_preds.loc[act_mask, SERVICE_COLS].sum()
                pred_mix = pred_mix / pred_mix.sum()
                print(f"\n  7-day ridership mix (Pred vs Actual):")
                for col in active_services:
                    label = _SERVICE_LABELS.get(col, col)
                    print(
                        f"    {label:<18}  pred {pred_mix[col]*100:5.2f}%  "
                        f"actual {act_mix[col]*100:5.2f}%  "
                        f"diff {(pred_mix[col]-act_mix[col])*100:+5.2f}pp"
                    )

    print(f"\n{'-' * W}")
    label = f"{share_mode} avg shares" if share_mode != "annual" else f"{hist_year} annual shares"
    print(f"  {label} (used for Pred distribution):\n")
    for col in active_services:
        svc_label = _SERVICE_LABELS.get(col, col)
        bar = "#" * int(avg_shares[col] * 200)
        print(f"    {svc_label:<18}  {avg_shares[col] * 100:5.2f}%  {bar}")
    if excluded_services:
        print(f"\n  Services excluded (no data before forecast window):")
        for col in excluded_services:
            svc_label = _SERVICE_LABELS.get(col, col)
            print(f"    {svc_label:<18}   0.00%  (not yet launched)")
    print(f"\n{'=' * W}\n")


def build_output_df(
    forecast_dates: list,
    total_pred: np.ndarray,
    service_preds: pd.DataFrame,
    actuals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame({
        "date":       [d.date() for d in forecast_dates],
        "total_pred": total_pred.round().astype(int),
    })
    for col in SERVICE_COLS:
        out[f"{col}_pred"] = service_preds[col].astype(int)

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
        "--lookback", type=int, default=14, choices=LOOKBACK_CHOICES,
        help="Look-back window in days (default: 14).",
    )
    p.add_argument(
        "--start-date", type=str, default=None, metavar="YYYY-MM-DD",
        help="First day of the 7-day forecast window (inclusive). "
             "The input lookback ends the day before. "
             "Defaults to the day after the last date in the features CSV.",
    )
    p.add_argument(
        "--output-dir", type=str, default=HMTTSF_OUTPUT_DIR_FR,
        choices=[HMTTSF_OUTPUT_DIR, HMTTSF_OUTPUT_DIR_FR],
        help=f"Model output directory (default: {HMTTSF_OUTPUT_DIR_FR}).",
    )
    p.add_argument(
        "--regime", type=str, default="nomco", choices=["nomco", "mco", "auto"],
        help="Training regime filter for run selection (default: nomco).",
    )
    p.add_argument(
        "--run-id", type=str, default=None, metavar="ID",
        help="Pin a specific trained run directory name.",
    )
    p.add_argument(
        "--hist-year", type=int, default=HIST_YEAR, metavar="YEAR",
        help=f"Calendar year for annual share mode (default: {HIST_YEAR}).",
    )
    p.add_argument(
        "--share-mode", type=str, default="dow_recent", choices=SHARE_MODES,
        help="Strategy for distributing total ridership across service lines "
             "(default: dow_recent).",
    )
    p.add_argument(
        "--share-alpha", type=float, default=0.3, metavar="A",
        help="Window weight for blend share mode (default: 0.3).",
    )
    p.add_argument(
        "--dow-weeks", type=int, default=8, metavar="N",
        help="Weeks of same-DOW history for dow_recent/calendar fallback (default: 8).",
    )
    p.add_argument(
        "--out", type=str, default=None, metavar="PATH",
        help="Optional: save the 7-day forecast to this CSV file.",
    )
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Inference device (default: auto).",
    )
    p.add_argument(
        "--list-runs", action="store_true",
        help="List all trained HMT-TSF runs with their metrics and exit.",
    )
    args = p.parse_args()

    if args.list_runs:
        print_available_runs(args.output_dir)
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n[inference]  device={device}  lookback={args.lookback}d  "
          f"output_dir={args.output_dir}")

    # ── resolve run (before features CSV so we know the regime) ───────────
    print(f"[inference]  scanning {args.output_dir} for lookback={args.lookback}d …")
    run = resolve_run(args.output_dir, args.lookback, args.regime, args.run_id)
    regime = run.get("regime", infer_regime(run["split_dates"]))
    feat_reduced = is_feat_reduced(run)
    print(f"[inference]  run      : {run['run_id']}")
    print(f"[inference]  metrics  : Combined={run['combined']:.3f}%  R2={run['r2']:.4f}")
    print(f"[inference]  regime   : {regime}  features={run['hparams']['n_features']}  "
          f"feat_reduced={feat_reduced}  boost={run['hparams'].get('boost_applied', False)}")

    features_csv = FEATURES_CSV_MAP[regime]
    print(f"[inference]  loading {features_csv} …")
    df = load_feature_csv(features_csv)

    # ── resolve forecast window ───────────────────────────────────────────
    if args.start_date is None:
        forecast_start = df.index.max() + timedelta(days=1)
        print(
            f"[inference]  --start-date not set; first forecast day: "
            f"{forecast_start.date()} (day after last features CSV date)"
        )
    else:
        forecast_start = pd.Timestamp(args.start_date)

    input_window_end = forecast_start - timedelta(days=1)
    forecast_dates = [forecast_start + timedelta(days=i) for i in range(T_OUT)]
    print(
        f"[inference]  input window ends : {input_window_end.date()}  "
        f"(lookback={args.lookback}d)"
    )
    print(f"[inference]  forecast window   : {forecast_dates[0].date()} to {forecast_dates[-1].date()}")

    # ── load sequence scalers matched to the run ──────────────────────────
    seq_dir = resolve_seq_dir(args.lookback, run["split_dates"])
    scaler_X = joblib.load(os.path.join(seq_dir, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(seq_dir, "scaler_y.pkl"))
    print(f"[inference]  loaded scalers from {seq_dir}/")

    sd = run["split_dates"]
    temporal_indices = sd.get("temporal_feat_indices")
    if not temporal_indices:
        temporal_indices = list(range(
            sd.get("temporal_feat_start", 13),
            sd.get("temporal_feat_end", 29),
        ))
        print("[inference]  WARNING: no temporal_feat_indices in run metadata; "
              "using legacy slice.")

    # ── extract input window ──────────────────────────────────────────────
    csv_end = df.index.max()
    print(f"[inference]  extracting {args.lookback}-day window ending {input_window_end.date()} …")
    X, n_synth = get_feature_window(
        df, input_window_end, args.lookback, scaler_X, feat_reduced=feat_reduced,
    )
    if n_synth > 0:
        print(
            f"[inference]  WARNING: {n_synth} day(s) beyond CSV end ({csv_end.date()}) "
            f"were synthesised.\n"
            f"             Holiday calendar + deterministic temporal features are exact.\n"
            f"             Ridership lags recomputed; fuel/rainfall/static averaged over "
            f"last 14 days."
        )

    # Input window for window/blend share modes (unscaled service columns)
    df_ext = _extend_df(df, input_window_end)
    end_pos = df_ext.index.get_loc(input_window_end) + 1
    window_df = df_ext.iloc[end_pos - args.lookback:end_pos]

    # ── build future temporal conditioning ───────────────────────────────
    x_future = build_future_temporal(df, forecast_dates, scaler_X, temporal_indices)
    print(f"[inference]  future temporal conditioning: {tuple(x_future.shape)}  "
          f"({len(temporal_indices)} temporal cols)")

    # ── load model and optional boost ─────────────────────────────────────
    print(f"[inference]  loading model from {run['run_dir']} …")
    model = load_model(run, device)
    boost_pack = load_boost_corrector(run)
    if boost_pack:
        print(f"[inference]  loaded CatBoost residual corrector (scale={boost_pack.get('boost_scale', 0.5)})")

    # ── forward pass ──────────────────────────────────────────────────────
    print(f"[inference]  running inference …")
    total_pred = run_inference(model, X, scaler_y, device, x_future=x_future, boost_pack=boost_pack)

    # ── service-line shares ───────────────────────────────────────────────
    active_services, excluded_services = _resolve_active_services(df, input_window_end)
    print(f"[inference]  computing service shares (mode={args.share_mode}) …")
    if excluded_services:
        excl_labels = [_SERVICE_LABELS.get(c, c) for c in excluded_services]
        print(
            f"[inference]  WARNING: {len(excluded_services)} service(s) have no data "
            f"before {input_window_end.date()} and are excluded from distribution:\n"
            + "".join(f"             - {lbl}\n" for lbl in excl_labels)
        )

    shares_by_day = compute_shares_by_day(
        df, forecast_dates, active_services, args.share_mode,
        args.hist_year, input_window_end,
        window_df=window_df if args.share_mode in ("window", "blend") else None,
        dow_weeks=args.dow_weeks,
        share_alpha=args.share_alpha,
    )

    service_preds = allocate_service_preds(total_pred, shares_by_day)

    # ── cross-check against ridership_headline.csv ────────────────────────
    print(f"[inference]  loading actuals from {RIDERSHIP_HEADLINE_CSV} …")
    headline_df = load_ridership_headline(RIDERSHIP_HEADLINE_CSV)
    actuals = get_actuals(headline_df, forecast_dates)
    if actuals is not None:
        n_with_actual = (~actuals["total_ridership"].isna()).sum()
        print(
            f"[inference]  headline actuals available for "
            f"{n_with_actual}/{T_OUT} forecast day(s)"
        )
    else:
        print(f"[inference]  no headline actuals found for the forecast window")

    print_forecast_table(
        forecast_dates, total_pred, service_preds,
        run=run, hist_year=args.hist_year, shares_by_day=shares_by_day,
        share_mode=args.share_mode,
        actuals=actuals,
        active_services=active_services,
        excluded_services=excluded_services,
    )

    if args.out:
        out_df = build_output_df(forecast_dates, total_pred, service_preds, actuals=actuals)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"[inference]  forecast saved -> {args.out}")


if __name__ == "__main__":
    main()
