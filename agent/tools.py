"""
Agent Tools — wraps LSTM/GRU/Sentiment as callable tools for the ReAct agent.

Each tool returns a structured dict that the LLM agent can reason over.
Tools gracefully handle missing model checkpoints (uses untrained model for demo).
"""
from __future__ import annotations

import numpy as np
import torch
from config import SEQUENCE_LENGTH, FORECAST_HORIZON, MODEL_DIR


def get_lstm_forecast(price_sequence: list[float], sentiment_sequence: list[float]) -> dict:
    """
    Run LSTM model on recent price+sentiment data, return 30-day forecast.
    """
    from models.lstm_model import SneakerLSTM

    prices = price_sequence[-SEQUENCE_LENGTH:]
    sents = sentiment_sequence[-SEQUENCE_LENGTH:]

    if len(prices) < SEQUENCE_LENGTH:
        pad_len = SEQUENCE_LENGTH - len(prices)
        prices = [prices[0]] * pad_len + prices
        sents = [0.0] * pad_len + sents

    seq = np.column_stack([prices, sents])
    x = torch.FloatTensor(seq).unsqueeze(0)

    model = SneakerLSTM(input_size=2)
    checkpoint_path = MODEL_DIR / "lstm_best.pt"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

    model.eval()
    with torch.no_grad():
        prediction = model(x).squeeze().numpy()

    if prediction.ndim == 0:
        prediction = np.array([float(prediction)])

    current_price = price_sequence[-1]
    predicted_final = float(prediction[-1])

    return {
        "model": "LSTM",
        "forecast_days": FORECAST_HORIZON,
        "predicted_prices": [round(p, 2) for p in prediction.tolist()],
        "predicted_final_price": round(predicted_final, 2),
        "predicted_change_pct": round((predicted_final - current_price) / current_price * 100, 2),
        "trend_direction": "UP" if predicted_final > current_price else "DOWN",
    }


def get_gru_forecast(price_sequence: list[float], sentiment_sequence: list[float]) -> dict:
    """
    Run GRU model on recent price+sentiment data, return 30-day forecast.
    """
    from models.gru_model import SneakerGRU

    prices = price_sequence[-SEQUENCE_LENGTH:]
    sents = sentiment_sequence[-SEQUENCE_LENGTH:]

    if len(prices) < SEQUENCE_LENGTH:
        pad_len = SEQUENCE_LENGTH - len(prices)
        prices = [prices[0]] * pad_len + prices
        sents = [0.0] * pad_len + sents

    seq = np.column_stack([prices, sents])
    x = torch.FloatTensor(seq).unsqueeze(0)

    model = SneakerGRU(input_size=2)
    checkpoint_path = MODEL_DIR / "gru_best.pt"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

    model.eval()
    with torch.no_grad():
        prediction = model(x).squeeze().numpy()

    if prediction.ndim == 0:
        prediction = np.array([float(prediction)])

    current_price = price_sequence[-1]
    predicted_final = float(prediction[-1])

    return {
        "model": "GRU",
        "forecast_days": FORECAST_HORIZON,
        "predicted_prices": [round(p, 2) for p in prediction.tolist()],
        "predicted_final_price": round(predicted_final, 2),
        "predicted_change_pct": round((predicted_final - current_price) / current_price * 100, 2),
        "trend_direction": "UP" if predicted_final > current_price else "DOWN",
    }


def get_sentiment_score(texts: list[str]) -> dict:
    """
    Compute average sentiment from recent social media texts using VADER.
    Falls back to simple keyword heuristic if vaderSentiment not installed.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(t)["compound"] for t in texts]
    except ImportError:
        positive_words = {"fire", "cop", "heat", "grail", "bullish", "up", "buy", "love", "amazing", "profit"}
        negative_words = {"dead", "brick", "sell", "drop", "loss", "overrated", "trash", "bearish", "down"}
        scores = []
        for t in texts:
            words = set(t.lower().split())
            pos = len(words & positive_words)
            neg = len(words & negative_words)
            total = pos + neg
            scores.append((pos - neg) / total if total > 0 else 0.0)

    if not scores:
        return {
            "average_sentiment": 0.0,
            "num_texts_analyzed": 0,
            "sentiment_label": "NEUTRAL",
            "score_range": [0.0, 0.0],
        }

    avg_score = float(np.mean(scores))
    return {
        "average_sentiment": round(avg_score, 4),
        "num_texts_analyzed": len(scores),
        "sentiment_label": "POSITIVE" if avg_score > 0.05 else "NEGATIVE" if avg_score < -0.05 else "NEUTRAL",
        "score_range": [round(float(min(scores)), 4), round(float(max(scores)), 4)],
    }


def get_price_history(prices: list[float]) -> dict:
    """
    Compute technical indicators from recent price history.
    """
    arr = np.array(prices)
    returns = np.diff(arr) / arr[:-1] if len(arr) > 1 else np.array([0.0])

    sma_7 = float(arr[-7:].mean()) if len(arr) >= 7 else float(arr.mean())
    sma_30 = float(arr[-30:].mean()) if len(arr) >= 30 else float(arr.mean())

    return {
        "current_price": round(float(arr[-1]), 2),
        "price_7d_ago": round(float(arr[-7]), 2) if len(arr) >= 7 else round(float(arr[0]), 2),
        "price_30d_ago": round(float(arr[-30]), 2) if len(arr) >= 30 else round(float(arr[0]), 2),
        "sma_7": round(sma_7, 2),
        "sma_30": round(sma_30, 2),
        "volatility_30d": round(float(np.std(returns[-30:])), 4) if len(returns) >= 30 else round(float(np.std(returns)), 4),
        "trend": "UP" if arr[-1] > sma_7 else "DOWN" if arr[-1] < sma_7 else "FLAT",
        "momentum": "BULLISH" if sma_7 > sma_30 else "BEARISH",
        "max_price_30d": round(float(arr[-30:].max()), 2),
        "min_price_30d": round(float(arr[-30:].min()), 2),
        "daily_return_avg": round(float(np.mean(returns[-30:])), 4) if len(returns) >= 30 else round(float(np.mean(returns)), 4),
    }


TOOL_REGISTRY = {
    "get_lstm_forecast": {
        "name": "get_lstm_forecast",
        "description": "Run LSTM deep learning model to predict sneaker prices for the next 30 days. Returns predicted final price, percent change, and trend direction.",
        "parameters": ["price_sequence", "sentiment_sequence"],
        "function": get_lstm_forecast,
    },
    "get_gru_forecast": {
        "name": "get_gru_forecast",
        "description": "Run GRU deep learning model to predict sneaker prices for the next 30 days. Faster than LSTM, often similar accuracy. Returns predicted final price, percent change, and trend direction.",
        "parameters": ["price_sequence", "sentiment_sequence"],
        "function": get_gru_forecast,
    },
    "get_sentiment_score": {
        "name": "get_sentiment_score",
        "description": "Analyze sentiment of recent social media posts about this sneaker. Returns average score from -1 (very negative) to +1 (very positive), label, and score range.",
        "parameters": ["texts"],
        "function": get_sentiment_score,
    },
    "get_price_history": {
        "name": "get_price_history",
        "description": "Get technical analysis of recent price data including trend, momentum, volatility, moving averages (SMA-7, SMA-30), and key price levels.",
        "parameters": ["prices"],
        "function": get_price_history,
    },
}
