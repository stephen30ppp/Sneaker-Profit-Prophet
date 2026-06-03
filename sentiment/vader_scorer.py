"""
VADER Sentiment Scorer

Rule-based sentiment analysis optimized for social media text.
Fast, no GPU required — good baseline.
"""
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import SENTIMENT_DIR


_analyzer = SentimentIntensityAnalyzer()


def score_texts(texts: pd.Series) -> pd.Series:
    """
    Compute VADER compound sentiment scores for a series of texts.

    Args:
        texts: Series of tweet/text strings

    Returns:
        Series of compound scores in [-1, 1]
    """
    scores = texts.fillna("").apply(lambda t: _analyzer.polarity_scores(str(t))["compound"])
    return scores


def aggregate_daily_sentiment(df: pd.DataFrame, date_col: str = "date", text_col: str = "text") -> pd.DataFrame:
    """
    Aggregate tweet-level scores into daily sentiment time series.

    Returns:
        DataFrame with columns: [date, sentiment_mean, sentiment_std, tweet_count]
    """
    df = df.copy()
    df["sentiment"] = score_texts(df[text_col])
    df["date_only"] = pd.to_datetime(df[date_col]).dt.date

    daily = (
        df.groupby("date_only")
        .agg(
            sentiment_mean=("sentiment", "mean"),
            sentiment_std=("sentiment", "std"),
            tweet_count=("sentiment", "count"),
        )
        .reset_index()
        .rename(columns={"date_only": "date"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)

    return daily.sort_values("date").reset_index(drop=True)


def save_sentiment(df: pd.DataFrame, name: str) -> None:
    """Save daily sentiment data to sentiment directory."""
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SENTIMENT_DIR / f"{name}_vader.csv", index=False)


if __name__ == "__main__":
    sample_texts = pd.Series([
        "These Yeezys are fire! Must cop 🔥",
        "Trash shoe, total brick. Don't buy.",
        "Just got my pair, pretty decent quality.",
        "Nike dropped the ball on this release smh",
        "BEST sneaker of 2021 hands down!!! 💯",
    ])
    scores = score_texts(sample_texts)
    for text, score in zip(sample_texts, scores):
        print(f"{score:+.4f} | {text}")
