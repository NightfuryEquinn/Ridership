"""
aggregate_diagnosis.py — Collect fit_diagnosis from all model results.json files.

Scans src/outputs/ for results.json files across all model subdirectories and
aggregates fit_diagnosis, categorised by MCO × lookback window:

  MCO      : "include" (train start < 2022-01-01)
           | "exclude" (train start >= 2022-01-01)
  Lookback : any T_in value found in hparams / split_dates

Output sections
---------------
  1. Quick Fit Overview  — compact coloured verdict matrix (all models at a glance)
  2. Detailed Pivot Tables — one table per diagnosis field
  3. Verdict Summary      — counts with % bar charts, broken down by MCO and lookback
  4. Flagged Notes        — severity-sorted annotations for non-clean runs

  CSV: src/outputs/aggregate_diagnosis.csv

Usage
-----
  python src/utils/aggregate_diagnosis.py
  python src/utils/aggregate_diagnosis.py --outputs-root src/outputs
  python src/utils/aggregate_diagnosis.py --csv-out my_diag.csv --no-table
  python src/utils/aggregate_diagnosis.py --no-color     # plain ASCII output
  python src/utils/aggregate_diagnosis.py --overview-only
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

MCO_CUTOFF = date(2022, 1, 1)

_MCO_VALS = ["exclude", "include"]
_MCO_LABELS = {
    "exclude": "No MCO (exclude)",
    "include": "With MCO (include)",
}

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
    try:
        train_start = date.fromisoformat(split_dates["train"]["start"])
        return "exclude" if train_start >= MCO_CUTOFF else "include"
    except (KeyError, TypeError, ValueError):
        return "exclude"


def _get_lookback(data):
    lb = (data.get("hparams") or {}).get("T_in")
    if lb is None:
        lb = (data.get("split_dates") or {}).get("T_in")
    return int(lb) if lb is not None else 0


def _model_key_from_path(path, outputs_root):
    rel   = os.path.relpath(path, outputs_root)
    parts = rel.replace("\\", "/").split("/")
    return parts[0].lower() if parts else "unknown"


def _timestamp_from_run_id(run_id):
    try:
        return run_id.replace("_", "").replace("-", "")
    except Exception:
        return run_id or ""


def _severity(verdict):
    return _VERDICT_SEVERITY.get(str(verdict).lower(), 1)


def _bar(n, total, width=_BAR_WIDTH):
    if total == 0:
        return _BAR_EMPTY * width
    filled = round(n / total * width)
    bar = _BAR_FULL * filled + _BAR_EMPTY * (width - filled)
    return bar


# ── Pivot cell ────────────────────────────────────────────────────────────────

def _pivot_cell(field, diag, colorize=True):
    if diag is None:
        return _c("—", _C.DIM) if colorize else "—"
    if field == "verdict":
        v = str(diag.get("verdict") or "").lower()
        short = _VERDICT_SHORT.get(v, v[:4]) if v else "—"
        if v == "overfit":
            short = "OVER!!"
        elif v == "underfit":
            short = "UNDR!!"
        elif v == "suspect":
            short = "susp?"
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


# ── Core scan ─────────────────────────────────────────────────────────────────

def scan_all_results(outputs_root):
    records = []
    pattern = os.path.join(outputs_root, "**", "results.json")

    for path in sorted(glob.glob(pattern, recursive=True)):
        data = _load_json(path)
        if data is None:
            continue

        diag = data.get("fit_diagnosis")
        if not diag:
            continue

        split_dates  = data.get("split_dates") or {}
        model_key    = _model_key_from_path(path, outputs_root)
        display_name = data.get("model") or model_key
        if model_key.endswith("_tuned"):
            display_name = display_name.rstrip(")") + " (tuned)"
        run_id       = data.get("run_id", os.path.basename(os.path.dirname(path)))

        records.append({
            "model_key":    model_key,
            "display_name": display_name,
            "mco":          _infer_mco(split_dates),
            "lookback":     _get_lookback(data),
            "run_id":       run_id,
            "timestamp":    _timestamp_from_run_id(run_id),
            "fit_diagnosis": {
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


# ── Selection: newest run per (model_key, mco, lookback) ─────────────────────

def select_runs(records):
    groups = defaultdict(list)
    for r in records:
        key = (r["model_key"], r["mco"], r["lookback"])
        groups[key].append(r)

    rows = []
    for key in sorted(groups):
        newest = max(groups[key], key=lambda r: r["timestamp"])
        rows.append(newest)
    return rows


# ── Pivot helpers ─────────────────────────────────────────────────────────────

def _build_pivot_structures(rows):
    all_lbs   = sorted({r["lookback"] for r in rows})
    model_map = {}
    for r in sorted(rows, key=lambda r: r["model_key"]):
        if r["model_key"] not in model_map:
            model_map[r["model_key"]] = r["display_name"]

    pivot = {
        (r["model_key"], r["mco"], r["lookback"]): r
        for r in rows
    }
    return all_lbs, model_map, pivot


# ── 1. Quick Fit Overview ─────────────────────────────────────────────────────

def print_fit_overview(rows):
    """
    Compact coloured verdict matrix — one row per model, one cell per (MCO, lb).

    Legend:
      good  = good_fit (green)
      susp? = suspect  (yellow)
      OVER  = overfit  (red bold)
      UNDR  = underfit (magenta bold)
      ····  = no data  (dim)
    """
    if not rows:
        print("[INFO] No fit_diagnosis data found.")
        return

    all_lbs, model_map, pivot = _build_pivot_structures(rows)
    active_mco = [m for m in _MCO_VALS if any(r["mco"] == m for r in rows)]

    cell_w  = 6
    gap     = "  "
    name_w  = max((len(dn) for dn in model_map.values()), default=10)
    name_w  = max(name_w, 12)

    n_cols_per_mco = len(all_lbs)
    mco_grp_w = n_cols_per_mco * (cell_w + len(gap)) - len(gap)

    total_w = name_w + len(gap) + len(active_mco) * (mco_grp_w + len(gap))
    total_w = max(total_w, 72)
    eq  = _c("═" * total_w, _C.CYAN)
    sep = "─" * total_w

    print(f"\n{eq}")
    title = "  Quick Fit Overview  —  Model × [MCO × Lookback]"
    print(_c(title, _C.CYAN, _C.BOLD))
    print(eq)

    # MCO group header
    mco_hdrs = gap.join(
        _MCO_LABELS[m].center(mco_grp_w) for m in active_mco
    )
    print(f"{''.ljust(name_w)}{gap}{mco_hdrs}")

    # Lookback sub-header
    lb_cells = gap.join(f"lb{lb}".rjust(cell_w) for lb in all_lbs)
    col_hdr  = gap.join(lb_cells for _ in active_mco)
    print(_c(f"{'Model'.ljust(name_w)}{gap}{col_hdr}", _C.BOLD))
    print(sep)

    for model_key, display_name in model_map.items():
        cells = []
        for mco in active_mco:
            for lb in all_lbs:
                row = pivot.get((model_key, mco, lb))
                if row is None:
                    cells.append(_c("····".rjust(cell_w), _C.DIM))
                    continue
                verdict = row["fit_diagnosis"].get("verdict") or ""
                short   = _VERDICT_SHORT.get(verdict.lower(), verdict[:4])
                if verdict.lower() == "overfit":
                    short = "OVER"
                elif verdict.lower() == "underfit":
                    short = "UNDR"
                elif verdict.lower() == "suspect":
                    short = "susp?"
                elif verdict.lower() == "good_fit":
                    short = "good"
                cells.append(_color_verdict(short.rjust(cell_w), verdict))
        print(f"{display_name.ljust(name_w)}{gap}{gap.join(cells)}")

    print(eq)

    # Inline legend
    legend_parts = [
        _color_verdict("good", "good_fit") + "=good_fit",
        _color_verdict("susp?", "suspect") + "=suspect",
        _color_verdict("OVER", "overfit") + "=overfit",
        _color_verdict("UNDR", "underfit") + "=underfit",
        _c("····", _C.DIM) + "=no data",
    ]
    print("  Legend: " + "   ".join(legend_parts))


# ── 2. Detailed Pivot Tables ──────────────────────────────────────────────────

def print_pivot_tables(rows):
    if not rows:
        print("[INFO] No fit_diagnosis data found.")
        return

    all_lbs, model_map, pivot = _build_pivot_structures(rows)

    cell_w = 9
    gap    = "  "
    name_w = max((len(dn) for dn in model_map.values()), default=10)
    name_w = max(name_w, 10)

    n_lbs    = len(all_lbs)
    grp_span = (cell_w + len(gap)) * n_lbs
    active_mco = [m for m in _MCO_VALS if any(r["mco"] == m for r in rows)]

    def _sub_hdr():
        return (
            f"{''.ljust(name_w)}{gap}"
            + gap.join(
                _MCO_LABELS[m].center(grp_span)
                for m in active_mco
            )
        )

    def _col_hdr(row_label):
        lb_cells = gap.join(f"lb{lb}".rjust(cell_w) for lb in all_lbs)
        return (
            _c(f"{row_label.ljust(name_w)}{gap}", _C.BOLD)
            + (gap.join(lb_cells for _ in active_mco))
        )

    total_w = name_w + len(gap) + grp_span * len(active_mco)
    eq  = _c("═" * total_w, _C.CYAN)
    sep = "─" * total_w

    for field, label, _ in _PIVOT_FIELDS:
        print(f"\n{eq}")
        print(_c(f"  {label}  ·  Models × [MCO  ·  Lookback]", _C.CYAN, _C.BOLD))
        print(eq)
        print(_sub_hdr())
        print(_col_hdr("Model"))
        print(sep)

        for model_key, display_name in model_map.items():
            cells = []
            for mco in active_mco:
                for lb in all_lbs:
                    row  = pivot.get((model_key, mco, lb))
                    diag = row["fit_diagnosis"] if row else None
                    raw  = _pivot_cell(field, diag, colorize=True)
                    # pad to cell_w (ANSI codes don't count toward visible width)
                    visible_len = len(_pivot_cell(field, diag, colorize=False))
                    padding     = " " * max(0, cell_w - visible_len)
                    cells.append(padding + raw)
            print(f"{display_name.ljust(name_w)}{gap}" + gap.join(cells))

        print(eq)

    # Coverage
    found = sorted(pivot.keys(), key=lambda k: (k[0], 0 if k[1] == "exclude" else 1, k[2]))
    missing = [
        (mk, mco, lb)
        for mk in model_map
        for mco in active_mco
        for lb  in all_lbs
        if (mk, mco, lb) not in pivot
    ]

    print(f"\n  Configurations with data ({len(found)}):")
    for mk, mco, lb in found:
        row = pivot[(mk, mco, lb)]
        print(f"    model={mk:<20}  mco={mco:<8}  lb={lb:>2}  run_id={row['run_id']}")

    if missing:
        print(f"\n  Missing configurations ({len(missing)}):")
        for mk, mco, lb in missing:
            print(f"    model={mk:<20}  mco={mco:<8}  lb={lb:>2}  — no results.json found")


# ── 3. Verdict Summary with bar charts ────────────────────────────────────────

def print_verdict_summary(rows):
    overall = defaultdict(int)
    by_mco  = defaultdict(lambda: defaultdict(int))
    by_lb   = defaultdict(lambda: defaultdict(int))

    for row in rows:
        v = str(row["fit_diagnosis"].get("verdict") or "unknown").lower()
        overall[v] += 1
        by_mco[row["mco"]][v] += 1
        by_lb[row["lookback"]][v] += 1

    total_w = 72
    eq = _c("═" * total_w, _C.CYAN)

    print(f"\n{eq}")
    print(_c(f"  Verdict Summary  ({len(rows)} run(s) total)", _C.CYAN, _C.BOLD))
    print(eq)

    sev_sort = lambda x: (-_VERDICT_SEVERITY.get(x[0], 1), x[0])

    def _print_counts(counter, indent="  "):
        total = sum(counter.values())
        for verdict, n in sorted(counter.items(), key=sev_sort):
            pct    = n / total * 100 if total else 0
            bar    = _bar(n, total)
            sev    = _VERDICT_SEVERITY.get(verdict, 1)
            marker = _VERDICT_MARKER[sev]
            label  = f"{verdict}{marker}"
            bar_colored = _color_verdict(bar, verdict)
            count_str = _color_verdict(f"{n:>3}  ({pct:5.1f}%)", verdict)
            print(f"{indent}{label:<14}  {bar_colored}  {count_str}")

    print(_c("  Overall:", _C.BOLD))
    _print_counts(overall)

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

def print_flagged_notes(rows):
    """
    Notes for non-clean runs, severity-sorted (overfit/underfit first),
    then grouped by MCO and lookback.
    """
    flagged = [
        r for r in rows
        if _severity(r["fit_diagnosis"].get("verdict")) > 0
    ]
    if not flagged:
        print(_c("\n[INFO] All runs: good_fit — no notes to display.", _C.GREEN))
        return

    total_w = 72
    eq = _c("═" * total_w, _C.CYAN)

    print(f"\n{eq}")
    print(_c("  Fit Diagnosis Notes  (severity-sorted, then by MCO · lookback)", _C.CYAN, _C.BOLD))
    print(eq)

    # Sort by severity desc, then model name
    flagged_sorted = sorted(
        flagged,
        key=lambda r: (
            -_severity(r["fit_diagnosis"].get("verdict")),
            r["model_key"],
        )
    )

    # Print flat severity-ranked list first
    print(_c("  Ranked by severity:", _C.BOLD))
    for row in flagged_sorted:
        diag   = row["fit_diagnosis"]
        verdict = diag.get("verdict") or "unknown"
        sev     = _severity(verdict)
        marker  = _VERDICT_MARKER[sev]
        drift   = diag.get("val_drift_pct")
        gap     = diag.get("gap_ratio")
        drift_s = f"  drift={drift:.1f}%" if drift is not None else ""
        gap_s   = f"  gap={gap:.2f}x" if gap is not None else ""
        header  = _color_verdict(
            f"  {row['display_name']:<20}  [{verdict}{marker}]  lb={row['lookback']:<3}  mco={row['mco']}{drift_s}{gap_s}",
            verdict
        )
        print(header)
        for note in row["notes"]:
            print(f"      • {note}")

    # Then group by MCO / lookback for structured view
    print(f"\n{_c('  Grouped by MCO · lookback:', _C.BOLD)}")
    groups = defaultdict(list)
    for r in flagged:
        groups[(r["mco"], r["lookback"])].append(r)

    for mco in _MCO_VALS:
        lbs_in_group = sorted({lb for (m, lb) in groups if m == mco})
        if not lbs_in_group:
            continue
        print(f"\n  ── {_MCO_LABELS[mco]}")
        for lb in lbs_in_group:
            runs = groups.get((mco, lb), [])
            if not runs:
                continue
            print(f"\n    lb={lb}")
            for row in sorted(runs, key=lambda r: -_severity(r["fit_diagnosis"].get("verdict"))):
                diag    = row["fit_diagnosis"]
                verdict = diag.get("verdict") or "unknown"
                sev     = _severity(verdict)
                marker  = _VERDICT_MARKER[sev]
                line = f"      {row['display_name']}  ·  run_id={row['run_id']}  ·  verdict={verdict}{marker}"
                print(_color_verdict(line, verdict))
                for note in row["notes"]:
                    print(f"        • {note}")

    print(f"\n{eq}")


# ── CSV export ────────────────────────────────────────────────────────────────

def save_diagnosis_csv(rows, csv_path):
    fieldnames = [
        "model_key", "display_name", "mco", "lookback", "run_id",
        *_DIAG_CSV_KEYS, "notes",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            diag = row["fit_diagnosis"]
            writer.writerow({
                "model_key":    row["model_key"],
                "display_name": row["display_name"],
                "mco":          row["mco"],
                "lookback":     row["lookback"],
                "run_id":       row["run_id"],
                "notes":        " | ".join(row["notes"]),
                **{k: diag.get(k) for k in _DIAG_CSV_KEYS},
            })

    print(f"\n[INFO] CSV saved → {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate fit_diagnosis from all model results.json files"
    )
    p.add_argument("--outputs-root", default="src/outputs",
                   help="Root directory containing output folders (default: src/outputs)")
    p.add_argument("--csv-out",  default=None,
                   help="CSV output path (default: <outputs-root>/aggregate_diagnosis.csv)")
    p.add_argument("--no-table",    action="store_true", help="Skip detailed pivot tables")
    p.add_argument("--no-notes",    action="store_true", help="Skip flagged-notes section")
    p.add_argument("--no-csv",      action="store_true", help="Skip saving the CSV")
    p.add_argument("--no-color",    action="store_true", help="Disable ANSI colour output")
    p.add_argument("--overview-only", action="store_true",
                   help="Print only the Quick Fit Overview matrix, then exit")
    return p.parse_args()


def main():
    global _USE_COLOR

    # Ensure UTF-8 output on Windows consoles (box-drawing characters)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args         = parse_args()
    outputs_root = args.outputs_root
    csv_out      = args.csv_out or os.path.join(outputs_root, "aggregate_diagnosis.csv")

    if args.no_color:
        _USE_COLOR = False

    print(f"[INFO] Scanning: {os.path.abspath(outputs_root)}")
    records = scan_all_results(outputs_root)
    print(f"[INFO] Found {len(records)} result file(s) with fit_diagnosis.")

    if not records:
        print("[INFO] Nothing to aggregate — no results.json with fit_diagnosis found.")
        return

    rows = select_runs(records)
    print(f"[INFO] Selected {len(rows)} unique (model × MCO × lookback) run(s).")

    # Always print the overview matrix
    print_fit_overview(rows)

    if args.overview_only:
        return

    if not args.no_table:
        print_pivot_tables(rows)

    print_verdict_summary(rows)

    if not args.no_notes:
        print_flagged_notes(rows)

    if not args.no_csv:
        save_diagnosis_csv(rows, csv_out)


if __name__ == "__main__":
    main()
