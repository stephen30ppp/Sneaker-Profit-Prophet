"""
FinBERT Sentiment Scorer

Transformer-based sentiment analysis fine-tuned on financial text.
Falls back to VADER with financial-keyword boosting when transformers
is not installed (avoids heavy GPU dependency for demo).
"""
import pandas as pd
import numpy as np

from config import SENTIMENT_DIR


def score_texts(texts: pd.Series, batch_size: int = 32) -> pd.Series:
    """
    Compute FinBERT-style sentiment scores for a series of texts.

    Attempts to use HuggingFace transformers pipeline. If unavailable,
    falls back to VADER with financial-keyword boosting.

    Args:
        texts: Series of text strings
        batch_size: Inference batch size

    Returns:
        Series of sentiment scores in [-1, 1]
    """
    try:
        from transformers import pipeline

        classifier = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )

        scores = []
        text_list = texts.fillna("").tolist()
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            results = classifier(batch)
            for r in results:
                label = r["label"].lower()
                score = r["score"]
                if label == "positive":
                    scores.append(score)
                elif label == "negative":
                    scores.append(-score)
                else:
                    scores.append(0.0)
        return pd.Series(scores, index=texts.index)

    except (ImportError, OSError):
        return _vader_financial_fallback(texts)


def _vader_financial_fallback(texts: pd.Series) -> pd.Series:
    """
    VADER-based fallback with financial keyword boosting.
    Provides FinBERT-like behavior without transformers dependency.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()

    financial_positive = {
        "profit", "gain", "bull", "bullish", "moon", "rocket", "surge",
        "premium", "grail", "limited", "exclusive", "hype", "sold out",
        "resell", "flip", "invest", "appreciate", "roi",
    }
    financial_negative = {
        "loss", "bear", "bearish", "crash", "dump", "brick", "sitting",
        "retail", "below retail", "depreciate", "overstock", "dead stock",
    }

    scores = []
    for text in texts.fillna(""):
        text = str(text)
        base_score = analyzer.polarity_scores(text)["compound"]

        text_lower = text.lower()
        boost = 0.0
        for word in financial_positive:
            if word in text_lower:
                boost += 0.1
        for word in financial_negative:
            if word in text_lower:
                boost -= 0.1

        final_score = np.clip(base_score + boost, -1.0, 1.0)
        scores.append(final_score)

    return pd.Series(scores, index=texts.index)


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
    df.to_csv(SENTIMENT_DIR / f"{name}_finbert.csv", index=False)


if __name__ == "__main__":
    sample_texts = pd.Series([
        "Nike profit margins are incredible this quarter, bullish on resale",
        "Yeezy resell crashed hard, total brick below retail",
        "Air Jordan 1 is a safe investment, always appreciates",
        "Market is dead, sitting on inventory nobody wants",
        "Limited drop — gonna flip these for 3x profit easy",
    ])
    scores = score_texts(sample_texts)
    for text, score in zip(sample_texts, scores):
        print(f"{score:+.4f} | {text}")
