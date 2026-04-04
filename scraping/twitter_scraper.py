"""
Twitter / X Sentiment Data Collector

Pulls tweets surrounding sneaker release dates
to quantify initial hype and ongoing sentiment.
"""
import pandas as pd
from pathlib import Path

from config import RAW_DIR, TWITTER_BEARER_TOKEN


def fetch_tweets(query: str, start_date: str, end_date: str, max_results: int = 1000) -> pd.DataFrame:
    """
    Fetch tweets matching a query within a date range.

    Args:
        query: Search query (e.g., "Jordan 1 Retro High OG")
        start_date: ISO format start date
        end_date: ISO format end date
        max_results: Maximum tweets to retrieve

    Returns:
        DataFrame with columns: [date, text, likes, retweets]
    """
    # TODO: Implement with tweepy or snscrape
    raise NotImplementedError


def save_tweets(df: pd.DataFrame, shoe_name: str) -> Path:
    """Save tweet data to raw directory."""
    output_path = RAW_DIR / f"{shoe_name}_tweets.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    print("Twitter Scraper - run with search query and date range")
