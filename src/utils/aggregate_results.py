"""
aggregate_results.py — Collect all model run results into a summary table.

Scans two directory tiers under outputs_root:
  Base runs  : src/outputs/{model}/           → fine_tuned = "no"
  Tuned runs : src/outputs/{model}_tuned/     → fine_tuned = "yes"

Aggregates across three dimensions:

  Model      : 17 entries — 15 base models + CNN-LSTM parallel + CNN-LSTM augmented
  MCO        : "include" (train start < 2022-01-01 → 2019-2025 range)
             | "exclude" (train start >= 2022-01-01 → 2022-2025 range)
  Lookback   : 14 | 28 | 56 days  (from split_dates.T_in)
  Fine-tuned : "no"  → newest run found in {model}/
             | "yes" → newest run found in {model}_tuned/

Output
------
  - Console: one pivot table per metric, grouped as:
      Base · No MCO  [lb14, lb28, lb56]
      Base · MCO     [lb14, lb28, lb56]
      Tuned · No MCO [lb14, lb28, lb56]
      Tuned · MCO    [lb14, lb28, lb56]
    Requires a ~140-char-wide terminal.
  - src/outputs/aggregate_results.csv  (wide-format pivot)

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

def _scan_dir(outputs_root, subdir, fine_tuned_tag, mode_filter, display_name):
    """
    Scan one directory (base or tuned) for results.json files.

    Returns a list of record dicts, each tagged with fine_tuned=fine_tuned_tag.
    """
    records = []
    model_dir = os.path.join(outputs_root, subdir)
    if not os.path.isdir(model_dir):
        return records

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

        run_id  = data.get("run_id", os.path.basename(os.path.dirname(path)))
        lb      = _get_lookback(data)
        mco     = _infer_mco(data.get("split_dates") or {})
        overall = _get_overall(data)
        ts      = _timestamp_from_run_id(run_id)

        records.append({
            "model":      display_name,
            "subdir":     subdir,
            "fine_tuned": fine_tuned_tag,
            "mco":        mco,
            "lookback":   lb,
            "mode":       _get_mode(data),
            "run_id":     run_id,
            "timestamp":  ts,
            "metrics":    overall,
            "path":       path,
        })

    return records


def scan_all_results(outputs_root):
    """
    Walk base ({model}/) and tuned ({model}_tuned/) directories for every
    entry in MODEL_REGISTRY and return a list of record dicts.

    Each record has:
      model, subdir, fine_tuned, mco, lookback, mode, run_id, timestamp,
      metrics, path
    """
    records = []

    for display_name, subdir, mode_filter in MODEL_REGISTRY:
        # Base runs: src/outputs/{subdir}/
        records.extend(_scan_dir(outputs_root, subdir, "no", mode_filter, display_name))
        # Tuned runs: src/outputs/{subdir}_tuned/
        records.extend(_scan_dir(outputs_root, subdir + "_tuned", "yes", mode_filter, display_name))

    return records


# ── Selection: newest run per (model, fine_tuned, mco, lookback) ─────────────

def select_runs(records):
    """
    Group records by (model, fine_tuned, mco, lookback) and keep the newest
    run in each group (latest timestamp).

    Returns a list of row dicts ready for tabulation/CSV.
    """
    groups = defaultdict(list)
    for r in records:
        key = (r["model"], r["fine_tuned"], r["mco"], r["lookback"])
        groups[key].append(r)

    rows = []
    for (model, fine_tuned, mco, lookback), group in sorted(groups.items()):
        newest = max(group, key=lambda r: r["timestamp"])
        m = newest["metrics"]
        rows.append({
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
            "run_id":      newest["run_id"],
            "path":        newest["path"],
        })

    return rows


# ── Side-by-side pivot table ──────────────────────────────────────────────────

# Column order: (fine_tuned, mco, lookback)
_CONFIGS = [
    ("no",  "exclude", 14), ("no",  "exclude", 28), ("no",  "exclude", 56),
    ("no",  "include", 14), ("no",  "include", 28), ("no",  "include", 56),
    ("yes", "exclude", 14), ("yes", "exclude", 28), ("yes", "exclude", 56),
    ("yes", "include", 14), ("yes", "include", 28), ("yes", "include", 56),
]

# (display_key_in_row, min_cell_width)
_PIVOT_METRICS = [
    ("Combined%", 8),
    ("MAPE%",     8),
    ("MAE%",      8),
    ("RMSE%",     8),
    ("R²",        8),
    ("MAE",      10),
    ("RMSE",     10),
]

# ── Tuned-vs-base comparison ───────────────────────────────────────────────────

# Column order for the delta table: (mco, lookback)
_COMPARE_CONFIGS = [
    ("exclude", 14), ("exclude", 28), ("exclude", 56),
    ("include", 14), ("include", 28), ("include", 56),
]

# True = higher is better; False = lower is better
_METRIC_HIGHER_BETTER = {
    "Combined%": True,
    "MAPE%":     False,
    "MAE%":      False,
    "RMSE%":     False,
    "R²":        True,
    "MAE":       False,
    "RMSE":      False,
}


def _delta_direction(delta, metric):
    """Return direction symbol: ↑ (improved), ↓ (degraded), = (unchanged)."""
    if delta is None:
        return "—"
    if abs(delta) < 1e-9:
        return "="
    higher_better = _METRIC_HIGHER_BETTER.get(metric, False)
    return "↑" if ((higher_better and delta > 0) or (not higher_better and delta < 0)) else "↓"


def _fmt_delta_cell(delta, metric, direction, cell_w):
    """Format a delta+direction value, e.g. '+2.34↑', right-justified to cell_w."""
    if delta is None:
        return "—".rjust(cell_w)
    if metric in ("MAE", "RMSE"):
        s = f"{delta:+,.0f}{direction}"
    elif metric == "R²":
        s = f"{delta:+.4f}{direction}"
    else:
        s = f"{delta:+.2f}{direction}"
    return s.rjust(cell_w)


def _delta_csv_col(mco, lb, metric, kind):
    """Build a flat column name, e.g. delta_nomco_lb14_Combined%  or  delta_nomco_lb14_Combined%_dir."""
    mco_tag = "nomco" if mco == "exclude" else "mco"
    suffix  = "" if kind == "delta" else "_dir"
    return f"delta_{mco_tag}_lb{lb}_{metric}{suffix}"


def _build_pivot(rows):
    """Return (model_order, pivot) where pivot[model][(ft, mco, lb)] = row."""
    pivot = defaultdict(dict)
    model_order = []
    for r in rows:
        m = r["Model"]
        if m not in model_order:
            model_order.append(m)
        key = (r["Fine-tuned"], r["MCO"], r["Lookback"])
        pivot[m][key] = r
    return model_order, pivot


def _fmt_cell(v, metric):
    if v is None:
        return "—"
    if metric in ("MAE", "RMSE"):
        return f"{v:,.0f}"
    if metric == "R²":
        return f"{v:.4f}"
    return f"{v:.2f}"


def print_comparison_table(rows):
    """
    Print one pivot table per metric.

    Layout (12 columns, grouped):
      Base · No MCO  [14, 28, 56] | Base · With MCO  [14, 28, 56]
      Tuned · No MCO [14, 28, 56] | Tuned · With MCO [14, 28, 56]

    Requires a ~140-char-wide terminal.
    """
    if not rows:
        print("[INFO] No results found.")
        return

    model_order, pivot = _build_pivot(rows)

    model_w = max(len("Model"), max(len(m) for m in model_order))
    cell_w  = 8          # fits "100.00", "0.9999", "1,234,567"
    gap     = "  "

    n_configs = len(_CONFIGS)                    # 12
    sub_span  = (cell_w + len(gap)) * 3          # 3 lookbacks per sub-group (No MCO / With MCO)
    grp_span  = sub_span * 2                     # Base or Tuned block

    # Total line width
    total_w = model_w + len(gap) + (cell_w + len(gap)) * n_configs - len(gap)
    eq  = "═" * total_w
    sep = "─" * total_w

    lb_labels = (["14", "28", "56"] * 4)

    def _grp_hdr():
        base_lbl  = "── Base (No Fine-Tune) ──"
        tuned_lbl = "── Fine-Tuned ──"
        return (
            f"{''.ljust(model_w)}{gap}"
            f"{base_lbl.ljust(grp_span)}"
            f"{tuned_lbl.ljust(grp_span)}"
        )

    def _sub_hdr():
        return (
            f"{''.ljust(model_w)}{gap}"
            f"{'No MCO'.center(sub_span)}"
            f"{'With MCO'.center(sub_span)}"
            f"{'No MCO'.center(sub_span)}"
            f"{'With MCO'.center(sub_span)}"
        )

    def _col_hdr():
        cells = gap.join(lbl.rjust(cell_w) for lbl in lb_labels)
        return f"{'Model'.ljust(model_w)}{gap}{cells}"

    def _data_row(model, metric):
        m_data = pivot[model]
        cells = []
        for cfg in _CONFIGS:
            r = m_data.get(cfg)
            v = r.get(metric) if r else None
            cells.append(_fmt_cell(v, metric).rjust(cell_w))
        return f"{model.ljust(model_w)}{gap}" + gap.join(cells)

    for metric, _ in _PIVOT_METRICS:
        print(f"\n{eq}")
        print(f"  {metric}  ·  Model × [Base/Tuned  ·  No MCO/With MCO  ·  Lookback]")
        print(eq)
        print(_grp_hdr())
        print(_sub_hdr())
        print(_col_hdr())
        print(sep)
        for model in model_order:
            print(_data_row(model, metric))
        print(eq)

    print(f"\nModels: {len(model_order)}  |  — = no data for that configuration")


def print_tuned_vs_base_table(rows):
    """
    Print per-metric delta tables: tuned minus base for each (MCO, lookback) combo.

    Layout (6 columns):
      No MCO [14, 28, 56] | With MCO [14, 28, 56]

    ↑ = improved, ↓ = degraded, = = unchanged, — = base or tuned absent.
    """
    if not rows:
        return

    model_order, pivot = _build_pivot(rows)

    model_w = max(len("Model"), max(len(m) for m in model_order))
    cell_w  = 12          # fits "+1,234,567↑" for large MAE/RMSE deltas
    gap     = "  "

    n_compare = len(_COMPARE_CONFIGS)              # 6
    sub_span  = (cell_w + len(gap)) * 3            # 3 lookbacks per MCO group
    total_w   = model_w + len(gap) + (cell_w + len(gap)) * n_compare - len(gap)

    eq  = "═" * total_w
    sep = "─" * total_w
    lb_labels = ["14", "28", "56", "14", "28", "56"]

    def _sub_hdr():
        return (
            f"{''.ljust(model_w)}{gap}"
            f"{'No MCO'.center(sub_span)}"
            f"{'With MCO'.center(sub_span)}"
        )

    def _col_hdr():
        cells = gap.join(lbl.rjust(cell_w) for lbl in lb_labels)
        return f"{'Model'.ljust(model_w)}{gap}{cells}"

    def _data_row(model, metric):
        m_data = pivot[model]
        cells = []
        for mco, lb in _COMPARE_CONFIGS:
            base_row  = m_data.get(("no",  mco, lb))
            tuned_row = m_data.get(("yes", mco, lb))
            if base_row is None or tuned_row is None:
                cells.append("—".rjust(cell_w))
                continue
            bv = base_row.get(metric)
            tv = tuned_row.get(metric)
            if bv is None or tv is None:
                cells.append("—".rjust(cell_w))
                continue
            delta = tv - bv
            direction = _delta_direction(delta, metric)
            cells.append(_fmt_delta_cell(delta, metric, direction, cell_w))
        return f"{model.ljust(model_w)}{gap}" + gap.join(cells)

    for metric, _ in _PIVOT_METRICS:
        print(f"\n{eq}")
        print(f"  TUNED vs BASE Δ  ·  {metric}  (↑ improved  ↓ degraded  = unchanged  — absent)")
        print(eq)
        print(_sub_hdr())
        print(_col_hdr())
        print(sep)
        for model in model_order:
            print(_data_row(model, metric))
        print(eq)

    print(f"\n  — = base or tuned run absent for that configuration")


# ── Wide-format CSV export ────────────────────────────────────────────────────

def _csv_col(ft, mco, lb, metric):
    """Build a flat column name like base_nomco_lb14_Combined%."""
    ft_tag  = "base"  if ft  == "no"      else "tuned"
    mco_tag = "nomco" if mco == "exclude"  else "mco"
    return f"{ft_tag}_{mco_tag}_lb{lb}_{metric}"


def save_comparison_csv(rows, csv_path):
    """
    Save a wide-format pivot CSV.

    Columns: Model, then for each of the 12 (fine_tuned × MCO × lookback)
    configurations: one column per metric, plus a run_id column.
    """
    model_order, pivot = _build_pivot(rows)
    metric_keys = [m for m, _ in _PIVOT_METRICS]

    fieldnames = ["Model"]
    for ft, mco, lb in _CONFIGS:
        for metric in metric_keys:
            fieldnames.append(_csv_col(ft, mco, lb, metric))
    for ft, mco, lb in _CONFIGS:
        fieldnames.append(_csv_col(ft, mco, lb, "run_id"))
    # Tuned-vs-base delta columns
    for mco, lb in _COMPARE_CONFIGS:
        for metric in metric_keys:
            fieldnames.append(_delta_csv_col(mco, lb, metric, "delta"))
            fieldnames.append(_delta_csv_col(mco, lb, metric, "dir"))

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for model in model_order:
            m_data = pivot[model]
            rec = {"Model": model}
            for ft, mco, lb in _CONFIGS:
                r = m_data.get((ft, mco, lb))
                for metric in metric_keys:
                    rec[_csv_col(ft, mco, lb, metric)] = r.get(metric) if r else None
                rec[_csv_col(ft, mco, lb, "run_id")] = r.get("run_id") if r else None
            # Compute deltas
            for mco, lb in _COMPARE_CONFIGS:
                base_r  = m_data.get(("no",  mco, lb))
                tuned_r = m_data.get(("yes", mco, lb))
                for metric in metric_keys:
                    bv = base_r.get(metric)  if base_r  else None
                    tv = tuned_r.get(metric) if tuned_r else None
                    if bv is not None and tv is not None:
                        delta     = tv - bv
                        direction = _delta_direction(delta, metric)
                        rec[_delta_csv_col(mco, lb, metric, "delta")] = round(delta, 6)
                        rec[_delta_csv_col(mco, lb, metric, "dir")]   = direction
                    else:
                        rec[_delta_csv_col(mco, lb, metric, "delta")] = None
                        rec[_delta_csv_col(mco, lb, metric, "dir")]   = None
            writer.writerow(rec)

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
        print_comparison_table(rows)
        print_tuned_vs_base_table(rows)

    if not args.no_csv:
        save_comparison_csv(rows, csv_out)


if __name__ == "__main__":
    main()
