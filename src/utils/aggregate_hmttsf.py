"""
aggregate_hmttsf.py — Collect all HMT-TSF run results into summary tables.

Scans src/outputs/hmttsf/ and src/outputs/hmttsf_feat_reduced/ for results.json
files and aggregates across two model variants:

  HMT-TSF    : full 79-feature model  (src/outputs/hmttsf/)
  HMT-TSF-FR : feature-reduced 53-feat (src/outputs/hmttsf_feat_reduced/)

Each variant covers the same MCO × lookback configurations:

  MCO      : "include" (train start < 2022-01-01)
           | "exclude" (train start >= 2022-01-01)
  Lookback : 7 | 14 | 28 | 56 | 84 days  (from hparams.T_in or split_dates.T_in)

Runs are partitioned by known-future calendar conditioning:

  use_x_future=True  : default headline runs (legacy results.json omit the field)
  use_x_future=False : --no-x-future ablation runs

6 primary configurations (matching the base-model comparison dimensions):
  No MCO  × [lb14, lb28, lb56]
  With MCO × [lb14, lb28, lb56]

2 HMT-TSF-extended configurations:
  No MCO  × [lb7, lb84]
  With MCO × [lb7, lb84]

Total: 2 variants × 10 configs = 20 cells per metric per X_future slice.

Output sections
---------------
  1. Overall metric pivot tables (per metric)
  2. Walk-forward temporal stability section
  3. Model vs Naive Persistence Baseline
  4. Quick Fit Overview  — compact coloured verdict matrix
  5. Detailed Fit Diagnosis pivot tables
  6. Verdict Summary      — counts with % bar charts
  7. Flagged Notes        — severity-sorted annotations
  8. X_future ablation delta (when --x-future both)

  CSV (default --x-future both):
    src/outputs/aggregate_hmttsf.csv             (use_x_future=True)
    src/outputs/aggregate_hmttsf_no_x_future.csv (use_x_future=False)

Usage
-----
  python src/utils/aggregate_hmttsf.py
  python src/utils/aggregate_hmttsf.py --x-future with
  python src/utils/aggregate_hmttsf.py --x-future without
  python src/utils/aggregate_hmttsf.py --x-future both
  python src/utils/aggregate_hmttsf.py --outputs-root src/outputs
  python src/utils/aggregate_hmttsf.py --csv-out my_hmttsf.csv --no-table
  python src/utils/aggregate_hmttsf.py --no-color     # plain ASCII output
  python src/utils/aggregate_hmttsf.py --overview-only
"""

import os
import sys
import json
import glob
import csv
import argparse
from datetime import date
from collections import defaultdict

# ── ANSI colour support ────────────────────────────────────────────────────────

_USE_COLOR = True   # set to False via --no-color or non-TTY stdout


def _color_on():
    return _USE_COLOR and sys.stdout.isatty()


class _C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"


def _c(text, *codes):
    if not _color_on():
        return str(text)
    return "".join(codes) + str(text) + _C.RESET


_VERDICT_CODES = {
    "good_fit": (_C.GREEN,),
    "suspect":  (_C.YELLOW,),
    "overfit":  (_C.RED, _C.BOLD),
    "underfit": (_C.MAGENTA, _C.BOLD),
}


def _color_verdict(text, verdict):
    codes = _VERDICT_CODES.get(str(verdict).lower(), (_C.DIM,))
    return _c(text, *codes)


# ── Constants ─────────────────────────────────────────────────────────────────

# Maps output subdir → display label (order defines row order in tables)
_VARIANT_SUBDIRS = {
    "hmttsf":              "HMT-TSF",
    "hmttsf_feat_reduced": "HMT-TSF-FR",
}

MCO_CUTOFF    = date(2022, 1, 1)   # train_start before this → "include"

METRIC_KEYS   = ["Combined", "MAPE", "MAE_pct", "RMSE_pct", "R2", "MAE", "RMSE"]
METRIC_LABELS = ["Combined%", "MAPE%", "MAE%", "RMSE%", "R²", "MAE", "RMSE"]

# Naive persistence metrics
NAIVE_KEYS   = ["Combined", "MAPE", "MAE_pct", "RMSE_pct", "R2", "MAE", "RMSE"]
NAIVE_LABELS = ["Naive Combined%", "Naive MAPE%", "Naive MAE%", "Naive RMSE%", "Naive R²", "Naive MAE", "Naive RMSE"]

# Delta metrics
DELTA_KEYS   = ["Combined", "MAPE", "MAE_pct", "RMSE_pct", "R2", "MAE", "RMSE"]
DELTA_LABELS = ["Delta Combined%", "Delta MAPE%", "Delta MAE%", "Delta RMSE%", "Delta R²", "Delta MAE", "Delta RMSE"]

# 6 primary + 4 extended = 10 total configurations (mco, lookback)
_PRIMARY_LBS   = [14, 28, 56]
_EXTENDED_LBS  = [7, 84]
_ALL_LBS       = [7, 14, 28, 56, 84]
_MCO_VALS      = ["exclude", "include"]
_MCO_LABELS    = {
    "exclude": "No MCO (exclude)",
    "include": "With MCO (include)",
}

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
    ("Naive Combined%", 9),
    ("Naive MAPE%",     9),
    ("Naive MAE%",      9),
    ("Naive RMSE%",     9),
    ("Naive R²",        9),
    ("Naive MAE",      11),
    ("Naive RMSE",     11),
    ("Delta Combined%", 9),
    ("Delta MAPE%",     9),
    ("Delta MAE%",      9),
    ("Delta RMSE%",     9),
    ("Delta R²",        9),
    ("Delta MAE",      11),
    ("Delta RMSE",     11),
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

# Fit diagnosis constants
_PIVOT_FIELDS = [
    ("verdict",       "Verdict",   8),
    ("val_drift_pct", "Drift%",    7),
    ("gap_ratio",     "GapRatio",  8),
    ("val_trend",     "ValTrend",  8),
    ("early_stop",    "EStop",     6),
    ("epochs",        "Best/Tot",  7),
]

_VERDICT_SHORT = {
    "good_fit": "good",
    "suspect":  "susp",
    "overfit":  "OVER",
    "underfit": "UNDR",
}

_VERDICT_SEVERITY = {
    "good_fit": 0,
    "suspect":  1,
    "overfit":  2,
    "underfit": 2,
}
_VERDICT_MARKER = {0: "  ", 1: " ?", 2: "!!"}

_DIAG_CSV_KEYS = [
    "verdict", "val_drift_pct", "gap_ratio",
    "val_trend", "early_stop", "best_epoch", "total_epochs",
]

_BAR_FULL  = "█"
_BAR_EMPTY = "░"
_BAR_WIDTH = 20


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


def _get_use_x_future(data):
    """True when known-future calendar conditioning was enabled (legacy default)."""
    hparams = data.get("hparams") or {}
    if "use_x_future" not in hparams:
        return True
    return bool(hparams["use_x_future"])


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


def _severity(verdict):
    return _VERDICT_SEVERITY.get(str(verdict).lower(), 1)


def _bar(n, total, width=_BAR_WIDTH):
    if total == 0:
        return _BAR_EMPTY * width
    filled = round(n / total * width)
    return _BAR_FULL * filled + _BAR_EMPTY * (width - filled)


def _ordered_variants(pivot):
    """Return variant labels in the canonical order defined by _VARIANT_SUBDIRS."""
    present = {k[0] for k in pivot}
    return [v for v in _VARIANT_SUBDIRS.values() if v in present]


# ── Core scan ─────────────────────────────────────────────────────────────────

def scan_hmttsf_results(outputs_root):
    """
    Walk src/outputs/hmttsf/ and src/outputs/hmttsf_feat_reduced/ for all
    results.json files.

    Returns a list of record dicts:
      variant, mco, lookback, run_id, timestamp, metrics (overall), walk_forward,
      naive_persistence, fit_diagnosis, notes, path
    """
    records = []
    for subdir, variant_label in _VARIANT_SUBDIRS.items():
        model_dir = os.path.join(outputs_root, subdir)
        if not os.path.isdir(model_dir):
            continue

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
            diag    = (data.get("training") or {}).get("fit_diagnosis") or data.get("fit_diagnosis") or {}

            records.append({
                "variant":           variant_label,
                "mco":               mco,
                "lookback":          lb,
                "use_x_future":      _get_use_x_future(data),
                "run_id":            run_id,
                "timestamp":         ts,
                "metrics":           overall,
                "walk_forward":      wf,
                "naive_persistence": data.get("naive_persistence") or {},
                "fit_diagnosis":     {
                    "verdict":       diag.get("verdict"),
                    "val_drift_pct": diag.get("val_drift_pct"),
                    "gap_ratio":     diag.get("gap_ratio"),
                    "val_trend":     diag.get("val_trend"),
                    "early_stop":    diag.get("early_stop"),
                    "best_epoch":    diag.get("best_epoch"),
                    "total_epochs":  diag.get("total_epochs"),
                },
                "notes": diag.get("notes") or [],
                "path":  path,
            })

    return records


# ── Selection: newest run per (variant, mco, lookback) ────────────────────────

def select_runs(records, use_x_future=None):
    """
    Group records by (variant, mco, lookback) and keep the newest run in each
    group.  When use_x_future is True/False, restrict to that conditioning slice.

    Returns a dict keyed by (variant, mco, lookback) → row dict.
    """
    groups = defaultdict(list)
    for r in records:
        if use_x_future is not None and r["use_x_future"] != use_x_future:
            continue
        key = (r["variant"], r["mco"], r["lookback"])
        groups[key].append(r)

    selected = {}
    for key, group in sorted(groups.items()):
        newest = max(group, key=lambda r: r["timestamp"])
        m          = newest["metrics"]
        naive_raw  = newest.get("naive_persistence") or {}
        naive_data = naive_raw.get("metrics") or {}

        def _delta(a, b):
            return round(a - b, 6) if (a is not None and b is not None) else None

        delta_combined = naive_raw.get("delta_combined")
        delta_mape     = naive_raw.get("delta_mape")
        delta_r2       = naive_raw.get("delta_r2")
        delta_mae_pct  = naive_raw.get("delta_mae_pct")  if naive_raw.get("delta_mae_pct")  is not None else _delta(m.get("MAE_pct"),  naive_data.get("MAE_pct"))
        delta_rmse_pct = naive_raw.get("delta_rmse_pct") if naive_raw.get("delta_rmse_pct") is not None else _delta(m.get("RMSE_pct"), naive_data.get("RMSE_pct"))
        delta_mae      = _delta(m.get("MAE"),  naive_data.get("MAE"))
        delta_rmse     = _delta(m.get("RMSE"), naive_data.get("RMSE"))

        selected[key] = {
            "Variant":      newest["variant"],
            "MCO":          newest["mco"],
            "Lookback":     newest["lookback"],
            "Combined%":    m.get("Combined"),
            "MAPE%":        m.get("MAPE"),
            "MAE%":         m.get("MAE_pct"),
            "RMSE%":        m.get("RMSE_pct"),
            "R²":           m.get("R2"),
            "MAE":          m.get("MAE"),
            "RMSE":         m.get("RMSE"),
            "Naive Combined%": naive_data.get("Combined"),
            "Naive MAPE%":     naive_data.get("MAPE"),
            "Naive MAE%":      naive_data.get("MAE_pct"),
            "Naive RMSE%":     naive_data.get("RMSE_pct"),
            "Naive R²":        naive_data.get("R2"),
            "Naive MAE":       naive_data.get("MAE"),
            "Naive RMSE":      naive_data.get("RMSE"),
            "Delta Combined%": delta_combined,
            "Delta MAPE%":     delta_mape,
            "Delta MAE%":      delta_mae_pct,
            "Delta RMSE%":     delta_rmse_pct,
            "Delta R²":        delta_r2,
            "Delta MAE":       delta_mae,
            "Delta RMSE":      delta_rmse,
            "walk_forward":    newest["walk_forward"],
            "fit_diagnosis":   newest["fit_diagnosis"],
            "notes":           newest["notes"],
            "run_id":          newest["run_id"],
            "use_x_future":    newest["use_x_future"],
            "path":            newest["path"],
        }
    return selected


# ── X_future ablation delta ───────────────────────────────────────────────────

def print_x_future_ablation_table(pivot_with, pivot_without):
    """Print Combined% / R² deltas (with − without) for paired configurations."""
    pairs = []
    for key in sorted(set(pivot_with) | set(pivot_without)):
        row_w = pivot_with.get(key)
        row_n = pivot_without.get(key)
        if row_w is None or row_n is None:
            continue
        c_w, c_n = row_w.get("Combined%"), row_n.get("Combined%")
        r_w, r_n = row_w.get("R²"), row_n.get("R²")
        if c_w is None or c_n is None:
            continue
        pairs.append({
            "variant": key[0],
            "mco":     key[1],
            "lookback": key[2],
            "with_c":  c_w,
            "without_c": c_n,
            "delta_c": round(c_w - c_n, 4),
            "with_r2": r_w,
            "without_r2": r_n,
            "delta_r2": round(r_w - r_n, 4) if (r_w is not None and r_n is not None) else None,
            "run_with": row_w.get("run_id"),
            "run_without": row_n.get("run_id"),
        })

    if not pairs:
        print("[INFO] No paired with/without X_future configurations for ablation table.")
        return

    total_w = 96
    eq = _c("═" * total_w, _C.CYAN)
    print(f"\n{eq}")
    print(_c("  X_future Ablation  (with − without)", _C.CYAN, _C.BOLD))
    print(eq)
    hdr = (
        f"  {'Variant':<12} {'MCO':<8} {'lb':>2}  "
        f"{'With%':>7} {'NoFut%':>7} {'ΔComb':>7}  "
        f"{'With R²':>8} {'NoFut R²':>8} {'ΔR²':>7}"
    )
    print(_c(hdr, _C.BOLD))
    print(f"  {'─' * (total_w - 2)}")

    for p in pairs:
        mco_tag = "nomco" if p["mco"] == "exclude" else "mco"
        delta_c = p["delta_c"]
        delta_r = p["delta_r2"]
        dc_str = f"{delta_c:+.2f}"
        dr_str = f"{delta_r:+.4f}" if delta_r is not None else "—"
        if _color_on() and delta_c > 0:
            dc_str = _c(dc_str, _C.GREEN)
        line = (
            f"  {p['variant']:<12} {mco_tag:<8} {p['lookback']:>2}  "
            f"{p['with_c']:7.2f} {p['without_c']:7.2f} {dc_str:>7}  "
            f"{p['with_r2']:8.4f} {p['without_r2']:8.4f} {dr_str:>7}"
        )
        print(line)

    avg_delta = sum(p["delta_c"] for p in pairs) / len(pairs)
    print(f"  {'─' * (total_w - 2)}")
    print(f"  Mean Δ Combined% (with − without): {avg_delta:+.2f}  "
          f"({len(pairs)} paired configs)")
    print(eq)


# ── Console pivot table ────────────────────────────────────────────────────────

def print_results_table(pivot, title_suffix=""):
    """
    Print one pivot table per metric.

    Layout (10 columns, grouped):
      No MCO  [7, 14, 28, 56, 84] | With MCO [7, 14, 28, 56, 84]

    One row per variant (HMT-TSF and HMT-TSF-FR).
    Primary lookbacks (14/28/56) are the base-model comparison dimensions.
    Extended lookbacks (7/84) are unique to HMT-TSF.
    """
    if not pivot:
        print("[INFO] No HMT-TSF results found.")
        return

    variants  = _ordered_variants(pivot)
    cell_w    = 9
    gap       = "  "
    label_w   = 12   # fits "HMT-TSF-FR"

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
        return f"{'Model'.ljust(label_w)}{gap}{cells}"

    def _ext_note():
        return (
            f"{''.ljust(label_w)}{gap}"
            f"{'↑ lb7/lb84 = HMT-TSF extended lookbacks'.ljust(sub_span)}"
            f"{'↑ lb7/lb84 = HMT-TSF extended lookbacks'.ljust(sub_span)}"
        )

    def _data_row(metric, variant):
        cells = []
        for mco, lb in _CONFIGS:
            row = pivot.get((variant, mco, lb))
            v   = row.get(metric) if row else None
            cells.append(_fmt_cell(v, metric).rjust(cell_w))
        return f"{variant.ljust(label_w)}{gap}" + gap.join(cells)

    for metric, _ in _PIVOT_METRICS:
        print(f"\n{eq}")
        suffix = f"  ·  {title_suffix}" if title_suffix else ""
        print(f"  {metric}  ·  HMT-TSF Variants  ×  [No MCO / With MCO  ·  Lookback]{suffix}")
        print(eq)
        print(_sub_hdr())
        print(_col_hdr())
        print(_ext_note())
        print(sep)
        for variant in variants:
            print(_data_row(metric, variant))
        print(eq)

    # Legend for run coverage
    found = sorted(pivot.keys())
    print(f"\n  Configurations with results ({len(found)}/{len(variants) * 10}):")
    for variant, mco, lb in found:
        row = pivot[(variant, mco, lb)]
        tag = "primary" if lb in _PRIMARY_LBS else "extended"
        print(f"    Variant={variant:<12}  MCO={mco:<8}  lb={lb:>2}  [{tag}]  run_id={row['run_id']}")
    missing = [
        (variant, mco, lb)
        for variant in variants
        for mco, lb in _CONFIGS
        if (variant, mco, lb) not in pivot
    ]
    if missing:
        print(f"\n  Missing configurations ({len(missing)}):")
        for variant, mco, lb in missing:
            tag = "primary" if lb in _PRIMARY_LBS else "extended"
            print(f"    Variant={variant:<12}  MCO={mco:<8}  lb={lb:>2}  [{tag}]  — no results.json found")
    print(f"\n  — = no data for that configuration")


# ── Walk-forward stability section ────────────────────────────────────────────

def print_walk_forward_table(pivot, title_suffix=""):
    """
    Print a walk-forward temporal stability table for every configuration
    that has walk_forward data.

    Shows Block 1 / 2 / 3 Combined% and R² for each (variant, mco, lookback).
    """
    wf_configs = [
        (variant, mco, lb) for (variant, mco, lb) in sorted(pivot)
        if pivot[(variant, mco, lb)].get("walk_forward")
    ]

    if not wf_configs:
        return

    cell_w = 9
    gap    = "  "
    lbl_w  = 30   # "HMT-TSF-FR  MCO=exclude / lb=14 [pri]"

    print(f"\n{'═' * 80}")
    suffix = f"  ·  {title_suffix}" if title_suffix else ""
    print(f"  Walk-Forward Temporal Stability  (3 equal test blocks, newest last){suffix}")
    print(f"{'═' * 80}")

    for metric_label, metric_key in [("Combined%", "Combined"), ("R²", "R2")]:
        print(f"\n  {metric_label}")
        header = f"  {'Config'.ljust(lbl_w)}" + gap.join(f"Block {b}".rjust(cell_w) for b in range(1, WF_BLOCKS + 1))
        print(header)
        print(f"  {'─' * (lbl_w + (cell_w + len(gap)) * WF_BLOCKS - len(gap))}")

        for variant, mco, lb in wf_configs:
            row = pivot[(variant, mco, lb)]
            tag = "pri" if lb in _PRIMARY_LBS else "ext"
            cfg_label = f"{variant}  MCO={mco}/lb={lb} [{tag}]"
            blocks = row["walk_forward"]
            cells = []
            for b in range(1, WF_BLOCKS + 1):
                b_data = next((x for x in blocks if x.get("block") == b), None)
                v = b_data.get(metric_key) if b_data else None
                cells.append(_fmt_cell(v, metric_label).rjust(cell_w))
            print(f"  {cfg_label.ljust(lbl_w)}" + gap.join(cells))

    print(f"\n{'═' * 80}")
    print("  Note: temporal stability improves if block 3 Combined% ≥ block 1 Combined%")


def print_model_vs_naive_table(pivot, title_suffix=""):
    """
    Print a table showing model performance vs naive persistence baseline.

    Shows delta metrics (model - naive) for each configuration, one row per variant.
    """
    if not pivot:
        return

    has_naive_data = any(
        row.get("Delta Combined%") is not None
        for row in pivot.values()
    )

    if not has_naive_data:
        print("[INFO] No naive persistence data found in results.")
        return

    variants = _ordered_variants(pivot)
    cell_w   = 9
    gap      = "  "
    label_w  = 12   # fits "HMT-TSF-FR"

    print(f"\n{'═' * 80}")
    suffix = f"  ·  {title_suffix}" if title_suffix else ""
    print(f"  Model vs Naive Persistence Baseline  (delta: model - naive){suffix}")
    print(f"{'═' * 80}")
    print("  ↑ = improvement over naive, ↓ = degradation vs naive, = = no change")

    metrics_info = [
        ("Delta Combined%", "Combined%", True),
        ("Delta MAPE%", "MAPE%", False),
        ("Delta MAE%", "MAE%", False),
        ("Delta RMSE%", "RMSE%", False),
        ("Delta R²", "R²", True),
    ]

    sub_span = (cell_w + len(gap)) * len(_ALL_LBS)

    for metric_label, metric_key, higher_better in metrics_info:
        print(f"\n  {metric_label}")
        header = f"  {''.ljust(label_w)}{gap}"
        header += f"{'No MCO (exclude)'.center(sub_span)}"
        header += f"{'With MCO (include)'.center(sub_span)}"
        print(header)

        lb_labels = [str(lb) for lb in _ALL_LBS] * 2
        cells_hdr = gap.join(lbl.rjust(cell_w) for lbl in lb_labels)
        print(f"{''.ljust(label_w)}{gap}{cells_hdr}")

        print(f"{''.ljust(label_w)}{gap}"
              f"{'↑ lb7/lb84 = HMT-TSF extended lookbacks'.ljust(sub_span)}"
              f"{'↑ lb7/lb84 = HMT-TSF extended lookbacks'.ljust(sub_span)}")

        print(f"  {'─' * (label_w + len(gap) + (cell_w + len(gap)) * len(_CONFIGS) - len(gap))}")

        for variant in variants:
            cells = []
            for mco, lb in _CONFIGS:
                row = pivot.get((variant, mco, lb))
                v   = row.get(metric_label) if row else None
                if v is None:
                    formatted = "—"
                else:
                    if abs(v) < 1e-9:
                        direction = "="
                    elif (higher_better and v > 0) or (not higher_better and v < 0):
                        direction = "↑"
                    else:
                        direction = "↓"

                    if "MAE" in metric_key or "RMSE" in metric_key:
                        formatted = f"{v:+,.0f}{direction}"
                    elif metric_key == "R²":
                        formatted = f"{v:+.4f}{direction}"
                    else:
                        formatted = f"{v:+.2f}{direction}"

                cells.append(formatted.rjust(cell_w))

            print(f"{variant.ljust(label_w)}{gap}" + gap.join(cells))

    print(f"\n{'═' * 80}")
    print("  Note: delta = model metric - naive metric")
    print("        Positive delta = improvement for Combined% and R²")
    print("        Positive delta = degradation for MAPE%, MAE%, RMSE%")


# ── Fit diagnosis helpers ─────────────────────────────────────────────────────

def _pivot_diag_cell(field, diag, colorize=True):
    if diag is None:
        return _c("—", _C.DIM) if colorize else "—"
    if field == "verdict":
        v = str(diag.get("verdict") or "").lower()
        if not v:
            return _c("—", _C.DIM) if colorize else "—"
        if v == "overfit":
            short = "OVER!!"
        elif v == "underfit":
            short = "UNDR!!"
        elif v == "suspect":
            short = "susp?"
        else:
            short = "good"
        return _color_verdict(short, v) if colorize else short
    if field == "val_drift_pct":
        v = diag.get("val_drift_pct")
        if v is None:
            return _c("—", _C.DIM) if colorize else "—"
        text = f"{v:.2f}"
        if colorize and v >= 25:
            return _c(text, _C.RED)
        return text
    if field == "gap_ratio":
        v = diag.get("gap_ratio")
        if v is None:
            return _c("—", _C.DIM) if colorize else "—"
        text = f"{v:.4f}"
        if colorize and v >= 3:
            return _c(text, _C.RED)
        return text
    if field == "val_trend":
        v = diag.get("val_trend")
        if v is None:
            return _c("—", _C.DIM) if colorize else "—"
        text = str(v)
        if colorize and v == "rising":
            return _c(text, _C.YELLOW)
        return text
    if field == "early_stop":
        v = diag.get("early_stop")
        if v is None:
            return _c("—", _C.DIM) if colorize else "—"
        text = "Yes" if v else "No"
        return _c(text, _C.CYAN) if (colorize and v) else text
    if field == "epochs":
        b = diag.get("best_epoch")
        t = diag.get("total_epochs")
        if b is None or t is None:
            return _c("—", _C.DIM) if colorize else "—"
        return f"{int(b)}/{int(t)}"
    return "—"


# ── 1. Quick Fit Overview ─────────────────────────────────────────────────────

def print_fit_overview(pivot, title_suffix=""):
    """
    Compact coloured verdict matrix — one row per variant, one cell per
    (MCO, lookback) configuration.

    Legend:
      good  = good_fit (green)
      susp? = suspect  (yellow)
      OVER  = overfit  (red bold)
      UNDR  = underfit (magenta bold)
      ····  = no data  (dim)
    """
    has_diag = any(
        pivot[(variant, mco, lb)]["fit_diagnosis"].get("verdict") is not None
        for (variant, mco, lb) in pivot
    )
    if not has_diag:
        return

    variants = _ordered_variants(pivot)
    cell_w   = 6
    gap      = "  "
    label_w  = 12   # fits "HMT-TSF-FR"

    n_cols_per_mco = len(_ALL_LBS)
    mco_grp_w = n_cols_per_mco * (cell_w + len(gap)) - len(gap)

    total_w = label_w + len(gap) + len(_MCO_VALS) * (mco_grp_w + len(gap))
    total_w = max(total_w, 72)
    eq  = _c("═" * total_w, _C.CYAN)
    sep = "─" * total_w

    print(f"\n{eq}")
    suffix = f"  ·  {title_suffix}" if title_suffix else ""
    print(_c(f"  Quick Fit Overview  —  HMT-TSF Variants × [MCO × Lookback]{suffix}",
             _C.CYAN, _C.BOLD))
    print(eq)

    mco_hdrs = gap.join(_MCO_LABELS[m].center(mco_grp_w) for m in _MCO_VALS)
    print(f"{''.ljust(label_w)}{gap}{mco_hdrs}")

    lb_cells = gap.join(f"lb{lb}".rjust(cell_w) for lb in _ALL_LBS)
    col_hdr  = gap.join(lb_cells for _ in _MCO_VALS)
    print(_c(f"{'Model'.ljust(label_w)}{gap}{col_hdr}", _C.BOLD))
    print(sep)

    for variant in variants:
        cells = []
        for mco in _MCO_VALS:
            for lb in _ALL_LBS:
                row = pivot.get((variant, mco, lb))
                if row is None:
                    cells.append(_c("····".rjust(cell_w), _C.DIM))
                    continue
                verdict = row["fit_diagnosis"].get("verdict") or ""
                if verdict.lower() == "overfit":
                    short = "OVER"
                elif verdict.lower() == "underfit":
                    short = "UNDR"
                elif verdict.lower() == "suspect":
                    short = "susp?"
                elif verdict.lower() == "good_fit":
                    short = "good"
                else:
                    short = "····"
                cells.append(_color_verdict(short.rjust(cell_w), verdict))
        print(f"{variant.ljust(label_w)}{gap}{gap.join(cells)}")

    print(eq)

    legend_parts = [
        _color_verdict("good", "good_fit") + "=good_fit",
        _color_verdict("susp?", "suspect") + "=suspect",
        _color_verdict("OVER", "overfit") + "=overfit",
        _color_verdict("UNDR", "underfit") + "=underfit",
        _c("····", _C.DIM) + "=no data",
    ]
    print("  Legend: " + "   ".join(legend_parts))


# ── 2. Detailed Fit Diagnosis Pivot Tables ────────────────────────────────────

def print_fit_pivot_tables(pivot, title_suffix=""):
    has_diag = any(
        pivot[(variant, mco, lb)]["fit_diagnosis"].get("verdict") is not None
        for (variant, mco, lb) in pivot
    )
    if not has_diag:
        return

    variants = _ordered_variants(pivot)
    cell_w   = 9
    gap      = "  "
    label_w  = 12   # fits "HMT-TSF-FR"

    n_lbs    = len(_ALL_LBS)
    grp_span = (cell_w + len(gap)) * n_lbs

    def _sub_hdr():
        return (
            f"{''.ljust(label_w)}{gap}"
            + gap.join(_MCO_LABELS[m].center(grp_span) for m in _MCO_VALS)
        )

    def _col_hdr():
        lb_cells = gap.join(f"lb{lb}".rjust(cell_w) for lb in _ALL_LBS)
        return (
            _c(f"{'Model'.ljust(label_w)}{gap}", _C.BOLD)
            + (gap.join(lb_cells for _ in _MCO_VALS))
        )

    total_w = label_w + len(gap) + grp_span * len(_MCO_VALS)
    eq  = _c("═" * total_w, _C.CYAN)
    sep = "─" * total_w

    for field, label, _ in _PIVOT_FIELDS:
        print(f"\n{eq}")
        suffix = f"  ·  {title_suffix}" if title_suffix else ""
        print(_c(f"  {label}  ·  HMT-TSF Variants × [MCO  ·  Lookback]{suffix}",
                   _C.CYAN, _C.BOLD))
        print(eq)
        print(_sub_hdr())
        print(_col_hdr())
        print(sep)

        for variant in variants:
            cells = []
            for mco in _MCO_VALS:
                for lb in _ALL_LBS:
                    row  = pivot.get((variant, mco, lb))
                    diag = row["fit_diagnosis"] if row else None
                    raw  = _pivot_diag_cell(field, diag, colorize=True)
                    visible_len = len(_pivot_diag_cell(field, diag, colorize=False))
                    padding     = " " * max(0, cell_w - visible_len)
                    cells.append(padding + raw)
            print(f"{variant.ljust(label_w)}{gap}" + gap.join(cells))

        print(eq)


# ── 3. Verdict Summary with bar charts ────────────────────────────────────────

def print_verdict_summary(pivot, title_suffix=""):
    diag_rows = [
        {"variant": variant, "mco": mco, "lookback": lb, "fit_diagnosis": row["fit_diagnosis"]}
        for (variant, mco, lb), row in pivot.items()
        if row["fit_diagnosis"].get("verdict") is not None
    ]
    if not diag_rows:
        return

    overall    = defaultdict(int)
    by_variant = defaultdict(lambda: defaultdict(int))
    by_mco     = defaultdict(lambda: defaultdict(int))
    by_lb      = defaultdict(lambda: defaultdict(int))

    for row in diag_rows:
        v = str(row["fit_diagnosis"].get("verdict") or "unknown").lower()
        overall[v] += 1
        by_variant[row["variant"]][v] += 1
        by_mco[row["mco"]][v] += 1
        by_lb[row["lookback"]][v] += 1

    total_w = 72
    eq = _c("═" * total_w, _C.CYAN)

    print(f"\n{eq}")
    suffix = f"  ·  {title_suffix}" if title_suffix else ""
    print(_c(f"  Verdict Summary  ({len(diag_rows)} configuration(s) total){suffix}",
             _C.CYAN, _C.BOLD))
    print(eq)

    sev_sort = lambda x: (-_VERDICT_SEVERITY.get(x[0], 1), x[0])

    def _print_counts(counter, indent="  "):
        total = sum(counter.values())
        for verdict, n in sorted(counter.items(), key=sev_sort):
            pct        = n / total * 100 if total else 0
            bar        = _bar(n, total)
            sev        = _VERDICT_SEVERITY.get(verdict, 1)
            marker     = _VERDICT_MARKER[sev]
            label      = f"{verdict}{marker}"
            bar_colored   = _color_verdict(bar, verdict)
            count_str     = _color_verdict(f"{n:>3}  ({pct:5.1f}%)", verdict)
            print(f"{indent}{label:<14}  {bar_colored}  {count_str}")

    print(_c("  Overall:", _C.BOLD))
    _print_counts(overall)

    print(_c("\n  By variant:", _C.BOLD))
    for variant in _ordered_variants(pivot):
        if variant not in by_variant:
            continue
        print(f"    {variant}:")
        _print_counts(by_variant[variant], indent="      ")

    print(_c("\n  By MCO:", _C.BOLD))
    for mco in _MCO_VALS:
        if mco not in by_mco:
            continue
        print(f"    {_MCO_LABELS[mco]}:")
        _print_counts(by_mco[mco], indent="      ")

    print(_c("\n  By lookback:", _C.BOLD))
    for lb in sorted(by_lb.keys()):
        print(f"    lb={lb}:")
        _print_counts(by_lb[lb], indent="      ")

    print(eq)


# ── 4. Flagged Notes ──────────────────────────────────────────────────────────

def print_flagged_notes(pivot, title_suffix=""):
    """
    Notes for non-clean configurations, severity-sorted (overfit/underfit first).
    """
    flagged = [
        {"variant": variant, "mco": mco, "lookback": lb, **row}
        for (variant, mco, lb), row in pivot.items()
        if _severity(row["fit_diagnosis"].get("verdict")) > 0
    ]
    if not flagged:
        print(_c("\n[INFO] All configurations: good_fit — no notes to display.", _C.GREEN))
        return

    total_w = 72
    eq = _c("═" * total_w, _C.CYAN)

    print(f"\n{eq}")
    suffix = f"  ·  {title_suffix}" if title_suffix else ""
    print(_c(f"  Fit Diagnosis Notes  (severity-sorted, then by variant · MCO · lookback){suffix}",
             _C.CYAN, _C.BOLD))
    print(eq)

    flagged_sorted = sorted(
        flagged,
        key=lambda r: (
            -_severity(r["fit_diagnosis"].get("verdict")),
            r["variant"],
            r["mco"],
            r["lookback"],
        )
    )

    print(_c("  Ranked by severity:", _C.BOLD))
    for row in flagged_sorted:
        diag    = row["fit_diagnosis"]
        verdict = diag.get("verdict") or "unknown"
        sev     = _severity(verdict)
        marker  = _VERDICT_MARKER[sev]
        drift   = diag.get("val_drift_pct")
        gap_r   = diag.get("gap_ratio")
        drift_s = f"  drift={drift:.1f}%" if drift is not None else ""
        gap_s   = f"  gap={gap_r:.2f}x"  if gap_r  is not None else ""
        tag     = "primary" if row["lookback"] in _PRIMARY_LBS else "extended"
        label   = f"{row['variant']}  MCO={row['mco']}  lb={row['lookback']} [{tag}]"
        header  = _color_verdict(
            f"  {label:<46}  [{verdict}{marker}]{drift_s}{gap_s}",
            verdict
        )
        print(header)
        for note in row.get("notes", []):
            print(f"      • {note}")

    print(f"\n{_c('  Grouped by variant · MCO · lookback:', _C.BOLD)}")
    for variant in _ordered_variants(pivot):
        variant_rows = [r for r in flagged if r["variant"] == variant]
        if not variant_rows:
            continue
        print(f"\n  ── {variant}")
        for mco in _MCO_VALS:
            mco_rows = [r for r in variant_rows if r["mco"] == mco]
            if not mco_rows:
                continue
            print(f"\n    {_MCO_LABELS[mco]}")
            for lb in sorted({r["lookback"] for r in mco_rows}):
                lb_rows = [r for r in mco_rows if r["lookback"] == lb]
                if not lb_rows:
                    continue
                print(f"\n      lb={lb}")
                for row in sorted(lb_rows, key=lambda r: -_severity(r["fit_diagnosis"].get("verdict"))):
                    diag    = row["fit_diagnosis"]
                    verdict = diag.get("verdict") or "unknown"
                    sev     = _severity(verdict)
                    marker  = _VERDICT_MARKER[sev]
                    line    = f"        {row['variant']}  ·  run_id={row['run_id']}  ·  verdict={verdict}{marker}"
                    print(_color_verdict(line, verdict))
                    for note in row.get("notes", []):
                        print(f"          • {note}")

    print(f"\n{eq}")


# ── Wide-format CSV export ────────────────────────────────────────────────────

def _csv_col(mco, lb, metric):
    """Build a flat column name like nomco_lb14_Combined%."""
    mco_tag = "nomco" if mco == "exclude" else "mco"
    return f"{mco_tag}_lb{lb}_{metric}"


def save_comparison_csv(pivot, csv_path):
    """
    Save a wide-format CSV with one row per variant (HMT-TSF and HMT-TSF-FR)
    and columns for each (mco × lookback) combination.

    Also includes walk-forward block columns and fit_diagnosis columns.
    """
    variants    = _ordered_variants(pivot)
    metric_keys = [m for m, _ in _PIVOT_METRICS]

    fieldnames = ["Variant"]
    for mco, lb in _CONFIGS:
        for metric in metric_keys:
            fieldnames.append(_csv_col(mco, lb, metric))
    for mco, lb in _CONFIGS:
        fieldnames.append(_csv_col(mco, lb, "run_id"))

    # Walk-forward block columns
    for mco, lb in _CONFIGS:
        for b in range(1, WF_BLOCKS + 1):
            fieldnames.append(_csv_col(mco, lb, f"wf_block{b}_Combined"))
            fieldnames.append(_csv_col(mco, lb, f"wf_block{b}_R2"))

    # Fit diagnosis columns
    for mco, lb in _CONFIGS:
        for key in _DIAG_CSV_KEYS:
            fieldnames.append(_csv_col(mco, lb, f"diag_{key}"))
        fieldnames.append(_csv_col(mco, lb, "diag_notes"))

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for variant in variants:
            rec = {"Variant": variant}

            for mco, lb in _CONFIGS:
                row = pivot.get((variant, mco, lb))
                for metric in metric_keys:
                    rec[_csv_col(mco, lb, metric)] = row.get(metric) if row else None
                rec[_csv_col(mco, lb, "run_id")] = row.get("run_id") if row else None

                # Walk-forward blocks
                blocks = row.get("walk_forward", []) if row else []
                for b in range(1, WF_BLOCKS + 1):
                    b_data = next((x for x in blocks if x.get("block") == b), None)
                    rec[_csv_col(mco, lb, f"wf_block{b}_Combined")] = b_data.get("Combined") if b_data else None
                    rec[_csv_col(mco, lb, f"wf_block{b}_R2")]       = b_data.get("R2")       if b_data else None

                # Fit diagnosis
                diag = row.get("fit_diagnosis") or {} if row else {}
                for key in _DIAG_CSV_KEYS:
                    rec[_csv_col(mco, lb, f"diag_{key}")] = diag.get(key)
                notes = row.get("notes") or [] if row else []
                rec[_csv_col(mco, lb, "diag_notes")] = " | ".join(notes)

            writer.writerow(rec)

    print(f"\n[INFO] CSV saved → {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def _print_report_bundle(pivot, title_suffix, args):
    """Print all console sections for one X_future slice."""
    if not pivot:
        print(f"[INFO] No results for: {title_suffix}")
        return

    variants = _ordered_variants(pivot)
    print(f"\n{'#' * 80}")
    print(f"  {title_suffix}")
    print(f"  {len(pivot)} configs across {len(variants)} variant(s): {', '.join(variants)}")
    print(f"{'#' * 80}")

    print_fit_overview(pivot, title_suffix=title_suffix)

    if args.overview_only:
        return

    if not args.no_table:
        print_results_table(pivot, title_suffix=title_suffix)

    if not args.no_wf:
        print_walk_forward_table(pivot, title_suffix=title_suffix)

    if not args.no_table:
        print_model_vs_naive_table(pivot, title_suffix=title_suffix)

    if not args.no_table:
        print_fit_pivot_tables(pivot, title_suffix=title_suffix)

    print_verdict_summary(pivot, title_suffix=title_suffix)

    if not args.no_notes:
        print_flagged_notes(pivot, title_suffix=title_suffix)


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate HMT-TSF results into summary table(s)"
    )
    p.add_argument("--outputs-root", default="src/outputs",
                   help="Root directory containing output folders (default: src/outputs)")
    p.add_argument("--x-future", choices=["with", "without", "both"], default="both",
                   help="Which X_future slice to aggregate (default: both)")
    p.add_argument("--csv-out", default=None,
                   help="CSV path for --x-future with/without (default: auto by slice)")
    p.add_argument("--no-table",      action="store_true", help="Skip the console metric table")
    p.add_argument("--no-wf",         action="store_true", help="Skip the walk-forward section")
    p.add_argument("--no-csv",        action="store_true", help="Skip saving the CSV")
    p.add_argument("--no-color",      action="store_true", help="Disable ANSI colour output")
    p.add_argument("--no-notes",      action="store_true", help="Skip flagged-notes section")
    p.add_argument("--overview-only", action="store_true",
                   help="Print only the Quick Fit Overview matrix(es), then exit")
    return p.parse_args()


def main():
    global _USE_COLOR

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    outputs_root = args.outputs_root

    if args.no_color:
        _USE_COLOR = False

    for subdir, label in _VARIANT_SUBDIRS.items():
        scan_path = os.path.abspath(os.path.join(outputs_root, subdir))
        print(f"[INFO] Scanning [{label}]: {scan_path}")

    records = scan_hmttsf_results(outputs_root)
    n_with    = sum(1 for r in records if r["use_x_future"])
    n_without = sum(1 for r in records if not r["use_x_future"])
    print(f"[INFO] Found {len(records)} result file(s): "
          f"{n_with} with X_future, {n_without} without X_future.")

    if not records:
        print("[INFO] Nothing to aggregate — no results.json files found.")
        return

    mode = args.x_future
    slices = []
    if mode in ("with", "both"):
        csv_w = (args.csv_out if mode == "with" and args.csv_out
                 else os.path.join(outputs_root, "aggregate_hmttsf.csv"))
        slices.append((True,  "use_x_future=True (headline)", csv_w))
    if mode in ("without", "both"):
        csv_n = (args.csv_out if mode == "without" and args.csv_out
                 else os.path.join(outputs_root, "aggregate_hmttsf_no_x_future.csv"))
        slices.append((False, "use_x_future=False (--no-x-future ablation)", csv_n))

    pivots = {}
    for use_xf, label, csv_path in slices:
        pivot = select_runs(records, use_x_future=use_xf)
        pivots[use_xf] = pivot
        _print_report_bundle(pivot, label, args)
        if not args.no_csv and not args.overview_only:
            save_comparison_csv(pivot, csv_path)

    if args.x_future == "both" and not args.overview_only:
        print_x_future_ablation_table(pivots.get(True, {}), pivots.get(False, {}))


if __name__ == "__main__":
    main()
