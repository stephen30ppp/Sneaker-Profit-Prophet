"""
Member B — One-Click Pipeline Runner

Runs the full Member B sentiment pipeline end-to-end, in dependency order:

    1. VADER  scoring               (sentiment/vader_scorer.py)
    2. FinBERT scoring              (sentiment/finbert_scorer.py)
    3. Daily aggregation            (sentiment/aggregate_sentiment.py)
    4. Price × sentiment merge      (sentiment/merge_sentiment_price.py)
    5. VADER vs FinBERT comparison  (sentiment/compare_vader_finbert.py)

Usage (from project root):
    python sentiment/run_all.py                  # run everything
    python sentiment/run_all.py --skip-finbert   # skip the slow FinBERT step
    python sentiment/run_all.py --skip-compare   # skip step 5
    python sentiment/run_all.py --only 3 4       # run only steps 3 and 4

Prerequisites (created upstream by Member A):
    data/processed/social_text_cleaned.csv
    data/processed/stockx_price_cleaned_member_a.csv

Existing output files are overwritten on each run; this script is idempotent.
"""

import argparse
import sys
import time
from pathlib import Path

# Make 'config' importable when running as: python sentiment/run_all.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PROCESSED_DIR

# Importing each module loads it but does not run main()
# (each script guards entry behind `if __name__ == "__main__":`).
from sentiment import (
    vader_scorer,
    finbert_scorer,
    aggregate_sentiment,
    merge_sentiment_price,
    compare_vader_finbert,
)

SOCIAL_FILE = PROCESSED_DIR / "social_text_cleaned.csv"
PRICE_FILE  = PROCESSED_DIR / "stockx_price_cleaned_member_a.csv"

# Step registry: (id, label, callable, requires_price_file)
STEPS = [
    (1, "VADER scoring",              vader_scorer.main,           False),
    (2, "FinBERT scoring",            finbert_scorer.main,         False),
    (3, "Daily aggregation",          aggregate_sentiment.main,    False),
    (4, "Price × sentiment merge",    merge_sentiment_price.main,  True),
    (5, "VADER vs FinBERT compare",   compare_vader_finbert.main,  False),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Member B sentiment pipeline")
    p.add_argument("--skip-finbert", action="store_true",
                   help="Skip FinBERT scoring (step 2). Comparison step is then also skipped.")
    p.add_argument("--skip-compare", action="store_true",
                   help="Skip the VADER vs FinBERT comparison (step 5).")
    p.add_argument("--only", type=int, nargs="+", metavar="N",
                   help="Run only the listed step IDs, e.g. --only 3 4")
    return p.parse_args()


def check_prerequisites(need_price: bool) -> None:
    if not SOCIAL_FILE.exists():
        sys.exit(
            f"ERROR: missing upstream file {SOCIAL_FILE}\n"
            "Run Member A's social text preprocessing first:\n"
            "    python scraping/member_a_social_text_preprocessing.py"
        )
    if need_price and not PRICE_FILE.exists():
        sys.exit(
            f"ERROR: missing upstream file {PRICE_FILE}\n"
            "Run Member A's StockX cleaning first:\n"
            "    1. Download Kaggle dataset to data/raw/StockX-Data-Contest-2019-3.csv\n"
            "    2. python scraping/member_a_stockx_cleaning.py"
        )


def select_steps(args: argparse.Namespace):
    if args.only:
        wanted = set(args.only)
        return [s for s in STEPS if s[0] in wanted]

    chosen = list(STEPS)
    if args.skip_finbert:
        # Without FinBERT scores the comparison is meaningless
        chosen = [s for s in chosen if s[0] not in (2, 5)]
    if args.skip_compare:
        chosen = [s for s in chosen if s[0] != 5]
    return chosen


def banner(text: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n  {text}\n{line}", flush=True)


def run_step(step_id: int, label: str, fn) -> float:
    banner(f"STEP {step_id}/5 — {label}")
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main() -> None:
    args = parse_args()
    chosen = select_steps(args)
    if not chosen:
        sys.exit("Nothing selected to run. Check your --only / --skip flags.")

    need_price = any(req_price for _, _, _, req_price in chosen)
    check_prerequisites(need_price=need_price)

    print(f"Member B pipeline — running {len(chosen)} step(s):")
    for sid, label, _, _ in chosen:
        print(f"  [{sid}] {label}")

    durations: list[tuple[int, str, float]] = []
    overall = time.perf_counter()
    for sid, label, fn, _ in chosen:
        try:
            elapsed = run_step(sid, label, fn)
        except SystemExit as exc:
            print(f"\nStep {sid} ({label}) aborted: {exc}", file=sys.stderr)
            sys.exit(exc.code if isinstance(exc.code, int) else 1)
        except Exception as exc:
            print(f"\nStep {sid} ({label}) crashed: {exc}", file=sys.stderr)
            raise
        durations.append((sid, label, elapsed))

    # Summary
    banner("PIPELINE SUMMARY")
    for sid, label, elapsed in durations:
        print(f"  Step {sid}: {label:35s}  {elapsed:7.2f} s")
    print(f"  {'TOTAL':41s} {time.perf_counter() - overall:7.2f} s")
    print("\nAll done. See data/processed/ and data/sentiment/ for outputs.")


if __name__ == "__main__":
    main()
