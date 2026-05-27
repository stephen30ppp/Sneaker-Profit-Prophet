"""
VADER Sentiment Scorer

Rule-based sentiment analysis optimized for social media text.
Fast, no GPU required — good baseline.
"""
import pandas as pd

from config import SENTIMENT_DIR


def score_texts(texts: pd.Series) -> pd.Series:
    """
    Compute VADER compound sentiment scores for a series of texts.

    Args:
        texts: Series of tweet/text strings

    Returns:
        Series of compound scores in [-1, 1]
    """
    # TODO: Implement with vaderSentiment
    raise NotImplementedError


def aggregate_daily_sentiment(df: pd.DataFrame, date_col: str = "date", text_col: str = "text") -> pd.DataFrame:
    """
    Aggregate tweet-level scores into daily sentiment time series.

    Returns:
        DataFrame with columns: [date, sentiment_mean, sentiment_std, tweet_count]
    """
    # TODO: Implement aggregation
    raise NotImplementedError
