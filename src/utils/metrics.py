"""
metrics.py — Shared evaluation metric for all forecasting models.

Usage:
    from src.utils.metrics import compute_metrics
"""

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return Combined%, MAPE%, MAE%, RMSE%, R², MAE, RMSE for a forecast."""
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    y_mean = np.mean(y_true)
    abs_err = np.abs(y_true - y_pred)
    sq_err = (y_true - y_pred) ** 2
    mae_raw = float(np.mean(abs_err))
    rmse_raw = float(np.sqrt(np.mean(sq_err)))
    mape = float(np.mean(abs_err / (np.abs(y_true) + 1.0)) * 100)
    denom = y_mean if y_mean > 0 else 1.0
    mae_pct = float(mae_raw / denom * 100)
    rmse_pct = float(rmse_raw / denom * 100)
    combined = float(max(0.0, 100.0 - mape - mae_pct - rmse_pct))
    ss_res = float(np.sum(sq_err))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "Combined": combined,
        "MAPE": mape,
        "MAE_pct": mae_pct,
        "RMSE_pct": rmse_pct,
        "R2": r2,
        "MAE": mae_raw,
        "RMSE": rmse_raw,
    }
