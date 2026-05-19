"""
aggregate_results.py — Collect all model run results into a summary table.

Scans src/outputs/{model}/**/ for results.json files and aggregates them
across three dimensions:

  Model      : 17 entries — 15 base models + CNN-LSTM parallel + CNN-LSTM augmented
  MCO        : "include" (train start < 2022-01-01 → 2019-2025 range)
             | "exclude" (train start >= 2022-01-01 → 2022-2025 range)
  Lookback   : 14 | 28 | 56 days  (from split_dates.T_in)
  Fine-tuned : "no"  → oldest run in the (model, MCO, lookback) group
             | "yes" → newest run in the (model, MCO, lookback) group
               (if only one run exists, it appears once as fine-tuned="no")

Output
------
  - Formatted console table grouped by (MCO, Lookback)
  - src/outputs/aggregate_results.csv

Usage
-----
  python src/utils/aggregate_results.py
  python src/utils/aggregate_results.py --outputs-root src/outputs
  python src/utils/aggregate_results.py --csv-out my_summary.csv --no-table
"""

import os
import json
import glob
import csv
import argparse
from datetime import date
from collections import defaultdict

# ── Model registry ────────────────────────────────────────────────────────────
# Each entry: (display_name, output_subdir, mode_filter_or_None)
# mode_filter is matched against hparams["mode"] in results.json (CNN-LSTM only).

MODEL_REGISTRY = [
    ("LSTM",                 "lstm",        None),
    ("BiLSTM",               "bilstm",      None),
    ("TPA-LSTM",             "tpa_lstm",    None),
    ("CNN-LSTM",             "cnn_lstm",    "sequential"),
    ("CNN-LSTM-Parallel",    "cnn_lstm",    "parallel"),
    ("CNN-LSTM-Augmented",   "cnn_lstm",    "augmented"),
    ("CNN-BiLSTM",           "cnn_bilstm",  None),
    ("ST-LSTM",              "st_lstm",     None),
    ("STGCN",                "stgcn",       None),
    ("MTGNN",                "mtgnn",       None),
    ("STSGCN",               "stsgcn",      None),
    ("STFGNN",               "stfgnn",      None),
    ("PDR-STGCN",            "pdr_stgcn",   None),
    ("ASTGCN",               "astgcn",      None),
    ("TFT",                  "tft",         None),
    ("Autoformer",           "autoformer",  None),
    ("Informer",             "informer",    None),
]

METRIC_KEYS   = ["Combined", "MAPE", "MAE_pct", "RMSE_pct", "R2", "MAE", "RMSE"]
METRIC_LABELS = ["Combined%", "MAPE%", "MAE%", "RMSE%", "R²", "MAE", "RMSE"]

MCO_CUTOFF = date(2022, 1, 1)   # train_start before this → "include"; on/after → "exclude"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Cannot read {path}: {exc}")
        return None


def _infer_mco(split_dates):
    """Return 'include' or 'exclude' based on training-split start date."""
    try:
        train_start_str = split_dates["train"]["start"]
        train_start = date.fromisoformat(train_start_str)
        return "exclude" if train_start >= MCO_CUTOFF else "include"
    except (KeyError, TypeError, ValueError):
        return "unknown"


def _get_lookback(data):
    """Extract T_in from hparams or split_dates."""
    lb = (data.get("hparams") or {}).get("T_in")
    if lb is None:
        lb = (data.get("split_dates") or {}).get("T_in")
    return int(lb) if lb is not None else None


def _get_mode(data):
    """Return hparams.mode or None (only set for CNN-LSTM variants)."""
    return (data.get("hparams") or {}).get("mode")


def _get_overall(data):
    """Return the overall metrics dict (keyed by Combined, MAPE, etc.)."""
    return (data.get("test_metrics") or {}).get("overall") or {}


def _timestamp_from_run_id(run_id):
    """Parse YYYYMMDD_HHMMSS run_id string to a comparable tuple."""
    try:
        parts = run_id.replace("_", "").replace("-", "")
        return parts   # lexicographic sort works for YYYYMMDDHHMMSS
    except Exception:
        return run_id


# ── Core scan ─────────────────────────────────────────────────────────────────

def scan_all_results(outputs_root):
    """
    Walk every results.json under outputs_root and return a list of record dicts.

    Each record has:
      model_name, model_dir, mco, lookback, mode, run_id, timestamp, metrics, path
    """
    records = []

    for display_name, subdir, mode_filter in MODEL_REGISTRY:
        model_dir = os.path.join(outputs_root, subdir)
        if not os.path.isdir(model_dir):
            continue

        pattern = os.path.join(model_dir, "**", "results.json")
        for path in sorted(glob.glob(pattern, recursive=True)):
            data = _load_json(path)
            if data is None:
                continue

            # For CNN-LSTM variants, skip files that don't match the mode filter
            if mode_filter is not None:
                file_mode = _get_mode(data)
                if file_mode != mode_filter:
                    continue

            run_id   = data.get("run_id", os.path.basename(os.path.dirname(path)))
            lb       = _get_lookback(data)
            mco      = _infer_mco(data.get("split_dates") or {})
            overall  = _get_overall(data)
            ts       = _timestamp_from_run_id(run_id)

            records.append({
                "model":     display_name,
                "subdir":    subdir,
                "mco":       mco,
                "lookback":  lb,
                "mode":      _get_mode(data),
                "run_id":    run_id,
                "timestamp": ts,
                "metrics":   overall,
                "path":      path,
            })

    return records


# ── Selection: oldest (no fine-tune) and newest (fine-tuned) ─────────────────

def select_runs(records):
    """
    Group records by (model, mco, lookback) and pick:
      oldest  → fine_tuned = "no"
      newest  → fine_tuned = "yes"  (only emitted when a 2nd distinct run exists)

    Returns a list of row dicts ready for tabulation/CSV.
    """
    groups = defaultdict(list)
    for r in records:
        key = (r["model"], r["mco"], r["lookback"])
        groups[key].append(r)

    rows = []
    for (model, mco, lookback), group in sorted(groups.items()):
        group_sorted = sorted(group, key=lambda r: r["timestamp"])
        oldest = group_sorted[0]
        newest = group_sorted[-1]

        def _make_row(rec, fine_tuned):
            m = rec["metrics"]
            return {
                "Model":       model,
                "MCO":         mco,
                "Lookback":    lookback,
                "Fine-tuned":  fine_tuned,
                "Combined%":   m.get("Combined"),
                "MAPE%":       m.get("MAPE"),
                "MAE%":        m.get("MAE_pct"),
                "RMSE%":       m.get("RMSE_pct"),
                "R²":          m.get("R2"),
                "MAE":         m.get("MAE"),
                "RMSE":        m.get("RMSE"),
                "run_id":      rec["run_id"],
                "path":        rec["path"],
            }

        rows.append(_make_row(oldest, "no"))
        if newest["run_id"] != oldest["run_id"]:
            rows.append(_make_row(newest, "yes"))

    return rows


# ── Console table ─────────────────────────────────────────────────────────────

def _fmt(v, key):
    if v is None:
        return "—"
    if key in ("MAE", "RMSE"):
        return f"{v:,.0f}"
    if key == "R²":
        return f"{v:.4f}"
    return f"{v:.2f}"


def print_table(rows):
    """Print a grouped, human-readable console table."""
    if not rows:
        print("[INFO] No results found.")
        return

    col_widths = {
        "Model":      max(len("Model"),      max(len(r["Model"])      for r in rows)),
        "MCO":        max(len("MCO"),        max(len(r["MCO"])        for r in rows)),
        "Lookback":   max(len("Lookback"),   8),
        "Fine-tuned": max(len("Fine-tuned"), 10),
        "Combined%":  max(len("Combined%"),  9),
        "MAPE%":      max(len("MAPE%"),      7),
        "MAE%":       max(len("MAE%"),       7),
        "RMSE%":      max(len("RMSE%"),      7),
        "R²":         max(len("R²"),         8),
        "MAE":        max(len("MAE"),        10),
        "RMSE":       max(len("RMSE"),       10),
    }

    display_cols = ["Model", "MCO", "Lookback", "Fine-tuned",
                    "Combined%", "MAPE%", "MAE%", "RMSE%", "R²", "MAE", "RMSE"]

    def _row_str(r_dict):
        parts = []
        for col in display_cols:
            v = r_dict.get(col)
            if col in ("Combined%", "MAPE%", "MAE%", "RMSE%", "R²", "MAE", "RMSE"):
                key_map = {"Combined%": "Combined%", "MAPE%": "MAPE%", "MAE%": "MAE%",
                           "RMSE%": "RMSE%", "R²": "R²", "MAE": "MAE", "RMSE": "RMSE"}
                s = _fmt(v, col)
                parts.append(s.rjust(col_widths[col]))
            else:
                s = str(v) if v is not None else "—"
                parts.append(s.ljust(col_widths[col]))
        return "  ".join(parts)

    def _header():
        parts = [col.ljust(col_widths[col]) if col not in
                 ("Combined%", "MAPE%", "MAE%", "RMSE%", "R²", "MAE", "RMSE")
                 else col.rjust(col_widths[col])
                 for col in display_cols]
        return "  ".join(parts)

    sep_len = sum(col_widths[c] for c in display_cols) + 2 * (len(display_cols) - 1)
    sep = "─" * sep_len

    # Group by (MCO, Lookback) for visual separation
    from itertools import groupby
    grouped = sorted(rows, key=lambda r: (r["MCO"], r["Lookback"] or 0, r["Model"], r["Fine-tuned"]))

    print(f"\n{'=' * sep_len}")
    print("Aggregate Results — All Models")
    print(f"{'=' * sep_len}")
    print(_header())
    print(sep)

    prev_group = None
    for r in grouped:
        cur_group = (r["MCO"], r["Lookback"])
        if prev_group is not None and cur_group != prev_group:
            print(sep)
        print(_row_str(r))
        prev_group = cur_group

    print(f"{'=' * sep_len}")
    print(f"Total rows: {len(rows)}")


# ── CSV export ────────────────────────────────────────────────────────────────

CSV_COLS = ["Model", "MCO", "Lookback", "Fine-tuned",
            "Combined%", "MAPE%", "MAE%", "RMSE%", "R²", "MAE", "RMSE",
            "run_id", "path"]


def save_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[INFO] CSV saved → {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Aggregate all model results into a table")
    p.add_argument("--outputs-root", default="src/outputs",
                   help="Root directory containing per-model output folders (default: src/outputs)")
    p.add_argument("--csv-out", default=None,
                   help="CSV output path (default: <outputs-root>/aggregate_results.csv)")
    p.add_argument("--no-table", action="store_true",
                   help="Skip printing the console table")
    p.add_argument("--no-csv", action="store_true",
                   help="Skip saving the CSV")
    return p.parse_args()


def main():
    args = parse_args()
    outputs_root = args.outputs_root
    csv_out = args.csv_out or os.path.join(outputs_root, "aggregate_results.csv")

    print(f"[INFO] Scanning: {os.path.abspath(outputs_root)}")
    records = scan_all_results(outputs_root)
    print(f"[INFO] Found {len(records)} result file(s) across all models.")

    if not records:
        print("[INFO] Nothing to aggregate — no results.json files found.")
        return

    rows = select_runs(records)
    print(f"[INFO] Aggregated {len(rows)} row(s) after oldest/newest selection.")

    if not args.no_table:
        print_table(rows)

    if not args.no_csv:
        save_csv(rows, csv_out)


if __name__ == "__main__":
    main()
