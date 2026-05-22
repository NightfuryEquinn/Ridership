"""
aggregate_hmttsf.py — Collect all HMT-TSF run results into a summary table.

Scans src/outputs/hmttsf/ for results.json files and aggregates across:

  MCO      : "include" (train start < 2022-01-01)
           | "exclude" (train start >= 2022-01-01)
  Lookback : 7 | 14 | 28 | 56 | 84 days  (from hparams.T_in or split_dates.T_in)

6 primary configurations (matching the base-model comparison dimensions):
  No MCO  × [lb14, lb28, lb56]
  With MCO × [lb14, lb28, lb56]

2 HMT-TSF-extended configurations:
  No MCO  × [lb7, lb84]
  With MCO × [lb7, lb84]

Additional walk-forward temporal stability section per configuration.

Output
------
  - Console: pivot table per metric (overall metrics + walk-forward blocks)
  - src/outputs/aggregate_hmttsf.csv  (wide-format)

Usage
-----
  python src/utils/aggregate_hmttsf.py
  python src/utils/aggregate_hmttsf.py --outputs-root src/outputs
  python src/utils/aggregate_hmttsf.py --csv-out my_hmttsf.csv --no-table
"""

import os
import json
import glob
import csv
import argparse
from datetime import date
from collections import defaultdict

# ── Constants ─────────────────────────────────────────────────────────────────

HMTTSF_SUBDIR = "hmttsf"
MCO_CUTOFF    = date(2022, 1, 1)   # train_start before this → "include"

METRIC_KEYS   = ["Combined", "MAPE", "MAE_pct", "RMSE_pct", "R2", "MAE", "RMSE"]
METRIC_LABELS = ["Combined%", "MAPE%", "MAE%", "RMSE%", "R²", "MAE", "RMSE"]

# 6 primary + 4 extended = 10 total configurations (mco, lookback)
_PRIMARY_LBS   = [14, 28, 56]
_EXTENDED_LBS  = [7, 84]
_ALL_LBS       = [7, 14, 28, 56, 84]
_MCO_VALS      = ["exclude", "include"]

# All (mco, lookback) combos ordered for the pivot table:
#   No MCO  [7, 14, 28, 56, 84] | With MCO [7, 14, 28, 56, 84]
_CONFIGS = [
    (mco, lb)
    for mco in _MCO_VALS
    for lb  in _ALL_LBS
]  # 10 columns

# (display_key_in_row, min_cell_width)
_PIVOT_METRICS = [
    ("Combined%", 9),
    ("MAPE%",     9),
    ("MAE%",      9),
    ("RMSE%",     9),
    ("R²",        9),
    ("MAE",      11),
    ("RMSE",     11),
]

_METRIC_HIGHER_BETTER = {
    "Combined%": True,
    "MAPE%":     False,
    "MAE%":      False,
    "RMSE%":     False,
    "R²":        True,
    "MAE":       False,
    "RMSE":      False,
}

WF_BLOCKS = 3   # HMT-TSF always uses 3 walk-forward blocks


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
        train_start = date.fromisoformat(split_dates["train"]["start"])
        return "exclude" if train_start >= MCO_CUTOFF else "include"
    except (KeyError, TypeError, ValueError):
        return "unknown"


def _get_lookback(data):
    """Extract T_in from hparams (preferred) or split_dates."""
    lb = (data.get("hparams") or {}).get("T_in")
    if lb is None:
        lb = (data.get("split_dates") or {}).get("T_in")
    return int(lb) if lb is not None else None


def _get_overall(data):
    return (data.get("test_metrics") or {}).get("overall") or {}


def _get_walk_forward(data):
    """Return list of walk-forward block dicts, or []."""
    return (data.get("test_metrics") or {}).get("walk_forward") or []


def _timestamp_from_run_id(run_id):
    try:
        return run_id.replace("_", "").replace("-", "")
    except Exception:
        return run_id


def _fmt_cell(v, metric):
    if v is None:
        return "—"
    if metric in ("MAE", "RMSE"):
        return f"{v:,.0f}"
    if metric == "R²":
        return f"{v:.4f}"
    return f"{v:.2f}"


# ── Core scan ─────────────────────────────────────────────────────────────────

def scan_hmttsf_results(outputs_root):
    """
    Walk src/outputs/hmttsf/ for all results.json files.

    Returns a list of record dicts:
      mco, lookback, run_id, timestamp, metrics (overall), walk_forward, path
    """
    records = []
    model_dir = os.path.join(outputs_root, HMTTSF_SUBDIR)
    if not os.path.isdir(model_dir):
        print(f"[WARN] Directory not found: {model_dir}")
        return records

    pattern = os.path.join(model_dir, "**", "results.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        data = _load_json(path)
        if data is None:
            continue

        run_id  = data.get("run_id", os.path.basename(os.path.dirname(path)))
        lb      = _get_lookback(data)
        mco     = _infer_mco(data.get("split_dates") or {})
        overall = _get_overall(data)
        wf      = _get_walk_forward(data)
        ts      = _timestamp_from_run_id(run_id)

        records.append({
            "mco":          mco,
            "lookback":     lb,
            "run_id":       run_id,
            "timestamp":    ts,
            "metrics":      overall,
            "walk_forward": wf,
            "path":         path,
        })

    return records


# ── Selection: newest run per (mco, lookback) ─────────────────────────────────

def select_runs(records):
    """
    Group records by (mco, lookback) and keep the newest run in each group.

    Returns a dict keyed by (mco, lookback) → row dict.
    """
    groups = defaultdict(list)
    for r in records:
        key = (r["mco"], r["lookback"])
        groups[key].append(r)

    selected = {}
    for key, group in sorted(groups.items()):
        newest = max(group, key=lambda r: r["timestamp"])
        m = newest["metrics"]
        selected[key] = {
            "MCO":          newest["mco"],
            "Lookback":     newest["lookback"],
            "Combined%":    m.get("Combined"),
            "MAPE%":        m.get("MAPE"),
            "MAE%":         m.get("MAE_pct"),
            "RMSE%":        m.get("RMSE_pct"),
            "R²":           m.get("R2"),
            "MAE":          m.get("MAE"),
            "RMSE":         m.get("RMSE"),
            "walk_forward": newest["walk_forward"],
            "run_id":       newest["run_id"],
            "path":         newest["path"],
        }
    return selected


# ── Console pivot table ────────────────────────────────────────────────────────

def print_results_table(pivot):
    """
    Print one pivot table per metric.

    Layout (10 columns, grouped):
      No MCO  [7, 14, 28, 56, 84] | With MCO [7, 14, 28, 56, 84]

    Requires a ~130-char-wide terminal.
    Primary lookbacks (14/28/56) are the base-model comparison dimensions.
    Extended lookbacks (7/84) are unique to HMT-TSF.
    """
    if not pivot:
        print("[INFO] No HMT-TSF results found.")
        return

    cell_w    = 9
    gap       = "  "
    label_w   = 10   # "HMT-TSF  "

    n_configs  = len(_CONFIGS)      # 10
    sub_span   = (cell_w + len(gap)) * len(_ALL_LBS)   # 5 lookbacks per MCO group
    total_w    = label_w + len(gap) + (cell_w + len(gap)) * n_configs - len(gap)

    eq  = "═" * total_w
    sep = "─" * total_w

    lb_labels = [str(lb) for lb in _ALL_LBS] * 2   # [7,14,28,56,84, 7,14,28,56,84]

    def _sub_hdr():
        return (
            f"{''.ljust(label_w)}{gap}"
            f"{'No MCO (exclude)'.center(sub_span)}"
            f"{'With MCO (include)'.center(sub_span)}"
        )

    def _col_hdr():
        cells = gap.join(lbl.rjust(cell_w) for lbl in lb_labels)
        return f"{'HMT-TSF'.ljust(label_w)}{gap}{cells}"

    def _ext_note():
        return (
            f"{''.ljust(label_w)}{gap}"
            f"{'↑ lb7/lb84 = HMT-TSF extended lookbacks'.ljust(sub_span)}"
            f"{'↑ lb7/lb84 = HMT-TSF extended lookbacks'.ljust(sub_span)}"
        )

    def _data_row(metric):
        cells = []
        for mco, lb in _CONFIGS:
            row = pivot.get((mco, lb))
            v   = row.get(metric) if row else None
            cells.append(_fmt_cell(v, metric).rjust(cell_w))
        return f"{'HMT-TSF'.ljust(label_w)}{gap}" + gap.join(cells)

    for metric, _ in _PIVOT_METRICS:
        print(f"\n{eq}")
        print(f"  {metric}  ·  HMT-TSF  ×  [No MCO / With MCO  ·  Lookback]")
        print(eq)
        print(_sub_hdr())
        print(_col_hdr())
        print(_ext_note())
        print(sep)
        print(_data_row(metric))
        print(eq)

    # Legend for run coverage
    found = sorted(pivot.keys())
    print(f"\n  Configurations with results ({len(found)}/10):")
    for mco, lb in found:
        row = pivot[(mco, lb)]
        tag = "primary" if lb in _PRIMARY_LBS else "extended"
        print(f"    MCO={mco:<8}  lb={lb:>2}  [{tag}]  run_id={row['run_id']}")
    missing = [cfg for cfg in _CONFIGS if cfg not in pivot]
    if missing:
        print(f"\n  Missing configurations ({len(missing)}/10):")
        for mco, lb in missing:
            tag = "primary" if lb in _PRIMARY_LBS else "extended"
            print(f"    MCO={mco:<8}  lb={lb:>2}  [{tag}]  — no results.json found")
    print(f"\n  — = no data for that configuration")


# ── Walk-forward stability section ────────────────────────────────────────────

def print_walk_forward_table(pivot):
    """
    Print a walk-forward temporal stability table for every configuration
    that has walk_forward data.

    Shows Block 1 / 2 / 3 Combined% and R² for each (mco, lookback) combo.
    """
    # Collect configs that have walk_forward results
    wf_configs = [
        (mco, lb) for (mco, lb) in _CONFIGS
        if (mco, lb) in pivot and pivot[(mco, lb)].get("walk_forward")
    ]

    if not wf_configs:
        return

    cell_w = 9
    gap    = "  "
    lbl_w  = 22   # "MCO=exclude / lb=14  "

    print(f"\n{'═' * 70}")
    print("  Walk-Forward Temporal Stability  (3 equal test blocks, newest last)")
    print(f"{'═' * 70}")

    for metric_label, metric_key in [("Combined%", "Combined"), ("R²", "R2")]:
        print(f"\n  {metric_label}")
        header = f"  {'Config'.ljust(lbl_w)}" + gap.join(f"Block {b}".rjust(cell_w) for b in range(1, WF_BLOCKS + 1))
        print(header)
        print(f"  {'─' * (lbl_w + (cell_w + len(gap)) * WF_BLOCKS - len(gap))}")

        for mco, lb in wf_configs:
            row = pivot[(mco, lb)]
            tag = "primary" if lb in _PRIMARY_LBS else "extended"
            cfg_label = f"MCO={mco} / lb={lb} [{tag}]"
            blocks = row["walk_forward"]
            cells = []
            for b in range(1, WF_BLOCKS + 1):
                b_data = next((x for x in blocks if x.get("block") == b), None)
                v = b_data.get(metric_key) if b_data else None
                cells.append(_fmt_cell(v, metric_label).rjust(cell_w))
            print(f"  {cfg_label.ljust(lbl_w)}" + gap.join(cells))

    print(f"\n{'═' * 70}")
    print("  Note: temporal stability improves if block 3 Combined% ≥ block 1 Combined%")


# ── Wide-format CSV export ────────────────────────────────────────────────────

def _csv_col(mco, lb, metric):
    """Build a flat column name like nomco_lb14_Combined%."""
    mco_tag = "nomco" if mco == "exclude" else "mco"
    return f"{mco_tag}_lb{lb}_{metric}"


def save_comparison_csv(pivot, csv_path):
    """
    Save a wide-format CSV with one row (HMT-TSF) and columns for each
    (mco × lookback) combination.

    Also includes walk-forward block columns where available.
    """
    metric_keys = [m for m, _ in _PIVOT_METRICS]

    # Base metric columns
    fieldnames = ["Model"]
    for mco, lb in _CONFIGS:
        for metric in metric_keys:
            fieldnames.append(_csv_col(mco, lb, metric))
    for mco, lb in _CONFIGS:
        fieldnames.append(_csv_col(mco, lb, "run_id"))

    # Walk-forward block columns (Combined% and R² per block)
    for mco, lb in _CONFIGS:
        for b in range(1, WF_BLOCKS + 1):
            fieldnames.append(_csv_col(mco, lb, f"wf_block{b}_Combined"))
            fieldnames.append(_csv_col(mco, lb, f"wf_block{b}_R2"))

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        rec = {"Model": "HMT-TSF"}

        for mco, lb in _CONFIGS:
            row = pivot.get((mco, lb))
            for metric in metric_keys:
                rec[_csv_col(mco, lb, metric)] = row.get(metric) if row else None
            rec[_csv_col(mco, lb, "run_id")] = row.get("run_id") if row else None

            # Walk-forward blocks
            blocks = row.get("walk_forward", []) if row else []
            for b in range(1, WF_BLOCKS + 1):
                b_data = next((x for x in blocks if x.get("block") == b), None)
                rec[_csv_col(mco, lb, f"wf_block{b}_Combined")] = b_data.get("Combined") if b_data else None
                rec[_csv_col(mco, lb, f"wf_block{b}_R2")]       = b_data.get("R2")       if b_data else None

        writer.writerow(rec)

    print(f"\n[INFO] CSV saved → {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate HMT-TSF results into a summary table"
    )
    p.add_argument("--outputs-root", default="src/outputs",
                   help="Root directory containing output folders (default: src/outputs)")
    p.add_argument("--csv-out", default=None,
                   help="CSV output path (default: <outputs-root>/aggregate_hmttsf.csv)")
    p.add_argument("--no-table",    action="store_true", help="Skip the console metric table")
    p.add_argument("--no-wf",       action="store_true", help="Skip the walk-forward section")
    p.add_argument("--no-csv",      action="store_true", help="Skip saving the CSV")
    return p.parse_args()


def main():
    args = parse_args()
    outputs_root = args.outputs_root
    csv_out = args.csv_out or os.path.join(outputs_root, "aggregate_hmttsf.csv")

    print(f"[INFO] Scanning: {os.path.abspath(os.path.join(outputs_root, HMTTSF_SUBDIR))}")
    records = scan_hmttsf_results(outputs_root)
    print(f"[INFO] Found {len(records)} HMT-TSF result file(s).")

    if not records:
        print("[INFO] Nothing to aggregate — no results.json files found in hmttsf/.")
        return

    pivot = select_runs(records)
    print(f"[INFO] Selected {len(pivot)} unique (MCO × lookback) configuration(s).")

    if not args.no_table:
        print_results_table(pivot)

    if not args.no_wf:
        print_walk_forward_table(pivot)

    if not args.no_csv:
        save_comparison_csv(pivot, csv_out)


if __name__ == "__main__":
    main()
