"""
Master pipeline runner — Ridership feature engineering.

Run from the repository root:

    python run_pipeline.py [options]

Steps
-----
  1a  Ridership          (independent)
  1b  Fuel Price         (independent)
  1c  Holidays           (independent)
  1d  Rainfall           (independent)
  1e  GADM Boundaries    (independent)
  1f  GTFS x4 operators  (independent)
  1g  Population Density (depends on GADM + GTFS)
  1h  OSM POIs           (depends on GTFS)
  2   Feature Alignment
  3   Sequence Builder   (once per look-back window: 14, 28, 56)

Flags
-----
  --skip-clean        Skip cleaning steps 1a–1h
  --skip-align        Skip feature alignment (step 2)
  --skip-sequences    Skip sequence building (step 3)
  --date-start DATE   feature_align --date-start  (default: 2022-01-01)
  --date-end   DATE   feature_align --date-end    (default: 2025-12-31)
  --output-dir DIR    feature_align --output-dir  (default: data/features)
  --T-out N           Forecast horizon in days    (default: 7)
  --lookbacks L …     Look-back windows           (default: 14 28 56)
  --include-mco       Pass --include-mco to sequence_builder
  --dtype DTYPE       Storage dtype: float16 or float32 (default: float16)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ── helpers ─────────────────────────────────────────────────────────────────

def _banner(label: str, cmd: list[str]) -> None:
    line = "─" * 60
    print(f"\n{line}")
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    print(line)


def run(cmd: list[str], label: str) -> None:
    """Run *cmd* as a subprocess; print timing; exit on non-zero return."""
    _banner(label, cmd)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"\n[FAIL] {label}  (exit {result.returncode}, {elapsed:.1f}s)")
        sys.exit(result.returncode)
    print(f"\n[OK]   {label}  ({elapsed:.1f}s)")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Ridership feature engineering pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # skip flags
    parser.add_argument("--skip-clean",     action="store_true", help="Skip cleaning steps 1a–1h")
    parser.add_argument("--skip-align",     action="store_true", help="Skip feature alignment (step 2)")
    parser.add_argument("--skip-sequences", action="store_true", help="Skip sequence building (step 3)")

    # feature_align arguments
    parser.add_argument("--date-start", default="2022-01-01", metavar="DATE",
                        help="feature_align start date (default: %(default)s)")
    parser.add_argument("--date-end",   default="2025-12-31", metavar="DATE",
                        help="feature_align end date (default: %(default)s)")
    parser.add_argument("--output-dir", default="data/features", metavar="DIR",
                        help="feature_align output directory (default: %(default)s)")

    # sequence_builder arguments
    parser.add_argument("--T-out", type=int, default=7, metavar="N",
                        help="Forecast horizon in days (default: %(default)s)")
    parser.add_argument("--lookbacks", type=int, nargs="+", default=[14, 28, 56],
                        metavar="L", help="Look-back windows (default: 14 28 56)")
    parser.add_argument("--include-mco", action="store_true",
                        help="Pass --include-mco to sequence_builder")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                        help="Storage dtype (default: %(default)s)")

    args = parser.parse_args()

    py = sys.executable  # same interpreter that launched this script

    total_start = time.perf_counter()

    # ── Step 1: Cleaning ──────────────────────────────────────────────────
    if not args.skip_clean:

        # 1a–1e: independent — no argument variants needed
        independent = [
            ("1a  Ridership",       "src/features/ridership.py"),
            ("1b  Fuel Price",      "src/features/fuelprice.py"),
            ("1c  Holidays",        "src/features/holiday.py"),
            ("1d  Rainfall",        "src/features/rainfall.py"),
            ("1e  GADM Boundaries", "src/features/gadm.py"),
        ]
        for label, script in independent:
            run([py, script], label)

        # 1f: GTFS — one call per operator
        gtfs_operators = [
            ("rapid_rail_kl",   "data/raw/gtfs_rapid_rail_kl",    "data/cleaned/gtfs_rapid_rail_kl"),
            ("rapidbus_kl",     "data/raw/gtfs_rapidbus_kl",      "data/cleaned/gtfs_rapidbus_kl"),
            ("rapidbus_penang", "data/raw/gtfs_rapidbus_penang",  "data/cleaned/gtfs_rapidbus_penang"),
            ("ktmb",            "data/raw/gtfs_ktmb",             "data/cleaned/gtfs_ktmb"),
        ]
        for operator, inp, out in gtfs_operators:
            run(
                [py, "src/features/gtfs.py",
                 "--input",    inp,
                 "--output",   out,
                 "--operator", operator],
                f"1f  GTFS — {operator}",
            )

        # 1g–1h: depend on GADM + GTFS
        run([py, "src/features/population.py"], "1g  Population Density")
        run([py, "src/features/osm.py"],        "1h  OSM POIs")

    # ── Step 2: Feature Alignment ─────────────────────────────────────────
    if not args.skip_align:
        run(
            [py, "src/features/feature_align.py",
             "--date-start", args.date_start,
             "--date-end",   args.date_end,
             "--output-dir", args.output_dir],
            "2   Feature Alignment",
        )

    # ── Step 3: Sequence Builder (one run per look-back window) ───────────
    if not args.skip_sequences:
        for T_in in args.lookbacks:
            seq_cmd = [
                py, "src/features/sequence_builder.py",
                "--T-in",  str(T_in),
                "--T-out", str(args.T_out),
                "--dtype", args.dtype,
            ]
            if args.include_mco:
                seq_cmd.append("--include-mco")
            run(seq_cmd, f"3   Sequence Builder  (T-in={T_in})")

    # ── Summary ───────────────────────────────────────────────────────────
    total = time.perf_counter() - total_start
    print(f"\n{'═' * 60}")
    print(f"  Pipeline complete — total time: {total:.1f}s")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
