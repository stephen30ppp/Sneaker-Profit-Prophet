"""
Twitter / X Sentiment Data Collector

Loads tweets from local JSONL dataset (Nike/Lululemon/Adidas tweets)
or fetches via API when available.
"""
import json
import pandas as pd
from pathlib import Path

from config import RAW_DIR, ROOT_DIR, TWITTER_BEARER_TOKEN


def load_tweets_jsonl(filepath: str = None) -> pd.DataFrame:
    """
    Load tweets from a JSONL file (one JSON object per line).

    Args:
        filepath: Path to JSONL file. Defaults to project root dataset.

    Returns:
        DataFrame with columns: [date, text, likes, retweets, brand, user]
    """
    if filepath is None:
        filepath = str(ROOT_DIR / "nikelululemonadidas_tweets.jsonl")

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                tweet = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = tweet.get("full_text") or tweet.get("text", "")
            created_at = tweet.get("created_at", "")
            likes = tweet.get("favorite_count", 0)
            retweets = tweet.get("retweet_count", 0)

            user_info = tweet.get("user", {})
            screen_name = user_info.get("screen_name", "")

            brand = _detect_brand(text)

            records.append({
                "date": created_at,
                "text": text,
                "likes": likes,
                "retweets": retweets,
                "brand": brand,
                "user": screen_name,
            })

    df = pd.DataFrame(records)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)

    return df


def _detect_brand(text: str) -> str:
    """Detect brand mention in tweet text."""
    text_lower = text.lower()
    if "nike" in text_lower or "jordan" in text_lower:
        return "Nike"
    elif "adidas" in text_lower or "yeezy" in text_lower:
        return "Adidas"
    elif "lululemon" in text_lower:
        return "Lululemon"
    return "Other"


def fetch_tweets(query: str, start_date: str, end_date: str, max_results: int = 1000) -> pd.DataFrame:
    """
    Fetch tweets matching a query within a date range via Twitter API.
    Falls back to local JSONL filtered by query if no API key.

    Args:
        query: Search query (e.g., "Yeezy")
        start_date: ISO format start date
        end_date: ISO format end date
        max_results: Maximum tweets to retrieve

    Returns:
        DataFrame with columns: [date, text, likes, retweets]
    """
    df = load_tweets_jsonl()

    if df.empty:
        return pd.DataFrame(columns=["date", "text", "likes", "retweets"])

    mask = df["text"].str.contains(query, case=False, na=False)
    filtered = df[mask].copy()

    if start_date:
        filtered = filtered[filtered["date"] >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        filtered = filtered[filtered["date"] <= pd.to_datetime(end_date, utc=True)]

    filtered = filtered.head(max_results)

    return filtered[["date", "text", "likes", "retweets"]].reset_index(drop=True)


def get_brand_tweets(brand: str, max_results: int = 5000) -> pd.DataFrame:
    """
    Get tweets for a specific brand from the local dataset.

    Args:
        brand: Brand name (Nike, Adidas, Lululemon)
        max_results: Maximum tweets to return

    Returns:
        DataFrame filtered to brand
    """
    df = load_tweets_jsonl()
    filtered = df[df["brand"].str.lower() == brand.lower()].head(max_results)
    return filtered.reset_index(drop=True)


def save_tweets(df: pd.DataFrame, shoe_name: str) -> Path:
    """Save tweet data to raw directory."""
    output_path = RAW_DIR / f"{shoe_name}_tweets.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    df = load_tweets_jsonl()
    print(f"Loaded {len(df)} tweets")
    if not df.empty:
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Brand distribution:\n{df['brand'].value_counts()}")
        print(f"\nSample tweet: {df['text'].iloc[0][:100]}...")
