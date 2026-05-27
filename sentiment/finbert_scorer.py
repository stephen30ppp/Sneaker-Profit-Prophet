"""
FinBERT Sentiment Scorer

Transformer-based sentiment analysis fine-tuned on financial text.
Higher accuracy than VADER, but requires GPU for efficient inference.
"""
import pandas as pd

from config import SENTIMENT_DIR


def score_texts(texts: pd.Series, batch_size: int = 32) -> pd.Series:
    """
    Compute FinBERT sentiment scores for a series of texts.

    Args:
        texts: Series of text strings
        batch_size: Inference batch size

    Returns:
        Series of sentiment scores in [-1, 1]
    """
    # TODO: Implement with transformers pipeline
    raise NotImplementedError


def aggregate_daily_sentiment(df: pd.DataFrame, date_col: str = "date", text_col: str = "text") -> pd.DataFrame:
    """
    Aggregate tweet-level scores into daily sentiment time series.

    Returns:
        DataFrame with columns: [date, sentiment_mean, sentiment_std, tweet_count]
    """
    # TODO: Implement aggregation
    raise NotImplementedError
