"""
Member B — Daily Sentiment Aggregation

Rolls up tweet-level sentiment scores into one row per day.

Run from project root:
    python sentiment/aggregate_sentiment.py

Input:
    data/sentiment/vader_scored_posts.csv
    data/sentiment/finbert_scored_posts.csv

Output:
    data/sentiment/vader_daily_sentiment.csv
        Columns: date, sentiment_mean, sentiment_std, tweet_count

    data/sentiment/finbert_daily_sentiment.csv
        Columns: date, sentiment_mean, sentiment_std, tweet_count

Notes:
    - VADER: sentiment_mean is the mean of vader_compound (-1 to +1).
    - FinBERT: sentiment_mean is the mean of finbert_score (mapped as
      positive=+1, neutral=0, negative=-1).
    - Days with no tweets have no row in the output (no zero-filling).
    - sentiment_std will be NaN for days with only one tweet (can't compute
      standard deviation from a single value — this is normal statistics).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import SENTIMENT_DIR

VADER_POSTS   = SENTIMENT_DIR / "vader_scored_posts.csv"
FINBERT_POSTS = SENTIMENT_DIR / "finbert_scored_posts.csv"

VADER_OUT   = SENTIMENT_DIR / "vader_daily_sentiment.csv"
FINBERT_OUT = SENTIMENT_DIR / "finbert_daily_sentiment.csv"


def aggregate_vader(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate VADER compound scores to daily level."""
    agg = (
        df.groupby("date")["vader_compound"]
        .agg(
            sentiment_mean="mean",
            sentiment_std="std",
            tweet_count="count",
        )
        .reset_index()
    )
    agg["date"] = pd.to_datetime(agg["date"]).dt.strftime("%Y-%m-%d")
    return agg.sort_values("date").reset_index(drop=True)


def aggregate_finbert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate FinBERT scores to daily level.

    finbert_score is already mapped to numeric:
        positive = +1, neutral = 0, negative = -1, unavailable = NaN

    NaN rows are excluded from mean/std/count automatically by pandas.
    """
    agg = (
        df.groupby("date")["finbert_score"]
        .agg(
            sentiment_mean="mean",
            sentiment_std="std",
            tweet_count="count",
        )
        .reset_index()
    )
    agg["date"] = pd.to_datetime(agg["date"]).dt.strftime("%Y-%m-%d")
    return agg.sort_values("date").reset_index(drop=True)


def main():
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # --- VADER ---
    if not VADER_POSTS.exists():
        print(f"ERROR: {VADER_POSTS} not found.")
        print("Run vader_scorer.py first:  python sentiment/vader_scorer.py")
        sys.exit(1)

    vader_df = pd.read_csv(VADER_POSTS)
    print(f"[OK] Loaded VADER posts: {len(vader_df):,} rows")

    vader_daily = aggregate_vader(vader_df)
    vader_daily.to_csv(VADER_OUT, index=False)
    print(f"[OK] VADER daily sentiment: {len(vader_daily)} unique dates → {VADER_OUT}")
    print(vader_daily.head(5).to_string(index=False))

    # --- FinBERT ---
    if not FINBERT_POSTS.exists():
        print(f"\nERROR: {FINBERT_POSTS} not found.")
        print("Run finbert_scorer.py first:  python sentiment/finbert_scorer.py")
        sys.exit(1)

    finbert_df = pd.read_csv(FINBERT_POSTS)
    print(f"\n[OK] Loaded FinBERT posts: {len(finbert_df):,} rows")

    real_rows = finbert_df["finbert_score"].notna().sum()
    if real_rows == 0:
        print("WARNING: All finbert_score values are NaN (FinBERT not installed).")
        print("FinBERT daily sentiment will have NaN means — this is expected.")

    finbert_daily = aggregate_finbert(finbert_df)
    finbert_daily.to_csv(FINBERT_OUT, index=False)
    print(f"[OK] FinBERT daily sentiment: {len(finbert_daily)} unique dates → {FINBERT_OUT}")
    print(finbert_daily.head(5).to_string(index=False))

    print("\n--- Summary ---")
    print(f"VADER  daily rows : {len(vader_daily)}")
    print(f"FinBERT daily rows: {len(finbert_daily)}")


if __name__ == "__main__":
    main()
