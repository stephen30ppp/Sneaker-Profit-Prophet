"""
Member B — Price × Sentiment Merge

Joins daily sentiment scores with the per-transaction price data.
The result is a multivariate DataFrame that Member C (modelling) can
use directly for LSTM/GRU training.

Run from project root:
    python sentiment/merge_sentiment_price.py

Input:
    data/processed/stockx_price_cleaned_member_a.csv
    data/sentiment/vader_daily_sentiment.csv
    data/sentiment/finbert_daily_sentiment.csv

Output:
    data/processed/price_sentiment_merged.csv
    Columns:
        date, shoe_name, sale_price, size, colorway,
        vader_sentiment_mean, vader_sentiment_std, vader_tweet_count,
        finbert_sentiment_mean, finbert_sentiment_std, finbert_tweet_count

Merge strategy:
    Left join from price data onto daily sentiment by date.
    Price rows on days with no sentiment data keep NaN in sentiment columns.
    This preserves ALL price rows (no data is dropped).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import PROCESSED_DIR, SENTIMENT_DIR

PRICE_FILE      = PROCESSED_DIR / "stockx_price_cleaned_member_a.csv"
VADER_DAILY     = SENTIMENT_DIR / "vader_daily_sentiment.csv"
FINBERT_DAILY   = SENTIMENT_DIR / "finbert_daily_sentiment.csv"
OUTPUT_FILE     = PROCESSED_DIR / "price_sentiment_merged.csv"


def load_and_check(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: {label} not found at {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"[OK] {label}: {len(df):,} rows, columns={list(df.columns)}")
    return df


def main():
    # Load all inputs
    price_df   = load_and_check(PRICE_FILE,    "Price data")
    vader_df   = load_and_check(VADER_DAILY,   "VADER daily sentiment")
    finbert_df = load_and_check(FINBERT_DAILY, "FinBERT daily sentiment")

    # Normalise date columns to the same string format before joining
    for df in [price_df, vader_df, finbert_df]:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Rename sentiment columns to avoid clashes after both merges
    vader_df = vader_df.rename(columns={
        "sentiment_mean": "vader_sentiment_mean",
        "sentiment_std":  "vader_sentiment_std",
        "tweet_count":    "vader_tweet_count",
    })
    finbert_df = finbert_df.rename(columns={
        "sentiment_mean": "finbert_sentiment_mean",
        "sentiment_std":  "finbert_sentiment_std",
        "tweet_count":    "finbert_tweet_count",
    })

    # Left join: keep all price rows; attach sentiment where available
    merged = price_df.merge(vader_df,   on="date", how="left")
    merged = merged.merge(finbert_df,   on="date", how="left")

    # Enforce final column order
    final_cols = [
        "date", "shoe_name", "sale_price", "size", "colorway",
        "vader_sentiment_mean", "vader_sentiment_std", "vader_tweet_count",
        "finbert_sentiment_mean", "finbert_sentiment_std", "finbert_tweet_count",
    ]
    merged = merged[final_cols]

    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Merged file saved: {len(merged):,} rows → {OUTPUT_FILE}")

    # Quality report
    price_days       = price_df["date"].nunique()
    vader_days       = vader_df["date"].nunique()
    finbert_days     = finbert_df["date"].nunique()
    matched_vader    = merged["vader_sentiment_mean"].notna().sum()
    matched_finbert  = merged["finbert_sentiment_mean"].notna().sum()

    print("\n--- Merge Quality Report ---")
    print(f"Unique dates in price data        : {price_days}")
    print(f"Unique dates in VADER daily       : {vader_days}")
    print(f"Unique dates in FinBERT daily     : {finbert_days}")
    print(f"Price rows WITH vader sentiment   : {matched_vader:,} / {len(merged):,}")
    print(f"Price rows WITH finbert sentiment : {matched_finbert:,} / {len(merged):,}")
    print(f"Price rows WITHOUT any sentiment  : {(merged['vader_sentiment_mean'].isna()).sum():,}")
    print("\nSample merged rows:")
    print(merged.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
