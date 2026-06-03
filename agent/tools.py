"""
Agent Tools — wraps LSTM/GRU/Sentiment as callable tools for the ReAct agent.

Each tool returns a structured dict that the LLM agent can reason over.
Tools gracefully handle missing model checkpoints (uses untrained model for demo).
"""
from __future__ import annotations

import numpy as np
import torch
from config import SEQUENCE_LENGTH, FORECAST_HORIZON, MODEL_DIR


def _build_feature_tensor(
    price_sequence: list[float],
    sentiment_sequence: list[float] | None,
) -> tuple[torch.Tensor, float, float]:
    """
    Build a (1, SEQUENCE_LENGTH, 4) tensor with feature order:
    [price_norm, sentiment_mean, return_1d, volatility_7d].
    """
    prices_arr = np.asarray(price_sequence, dtype=np.float32)
    prices_arr = np.nan_to_num(prices_arr, nan=0.0, posinf=0.0, neginf=0.0)
    if prices_arr.size == 0:
        raise ValueError("price_sequence is empty")

    sentiments_arr = np.asarray(sentiment_sequence or [], dtype=np.float32)
    sentiments_arr = np.nan_to_num(sentiments_arr, nan=0.0, posinf=0.0, neginf=0.0)

    prices = prices_arr[-SEQUENCE_LENGTH:]
    if prices.size < SEQUENCE_LENGTH:
        pad_len = SEQUENCE_LENGTH - prices.size
        prices = np.concatenate([np.full(pad_len, float(prices[0]), dtype=np.float32), prices])

    sents = sentiments_arr[-SEQUENCE_LENGTH:]
    if sents.size < SEQUENCE_LENGTH:
        pad_len = SEQUENCE_LENGTH - sents.size
        sents = np.concatenate([np.zeros(pad_len, dtype=np.float32), sents])

    price_mean = float(prices.mean())
    price_std = float(prices.std())
    if price_std > 1e-8:
        price_norm = (prices - price_mean) / price_std
    else:
        price_norm = np.zeros_like(prices)

    return_1d = np.zeros_like(price_norm)
    prev = price_norm[:-1]
    delta = np.diff(price_norm)
    valid = np.abs(prev) > 1e-8
    return_1d[1:] = np.where(valid, delta / prev, 0.0)

    volatility_7d = np.zeros_like(return_1d)
    for i in range(return_1d.size):
        start = max(0, i - 6)
        window = return_1d[start : i + 1]
        volatility_7d[i] = float(window.std(ddof=1)) if window.size > 1 else 0.0

    seq = np.column_stack([price_norm, sents, return_1d, volatility_7d]).astype(np.float32)
    x = torch.from_numpy(seq).unsqueeze(0)
    return x, price_mean, price_std


def get_lstm_forecast(price_sequence: list[float], sentiment_sequence: list[float]) -> dict:
    """
    Run LSTM model on recent price+sentiment data, return 30-day forecast.
    """
    from models.lstm_model import SneakerLSTM

    x, price_mean, price_std = _build_feature_tensor(price_sequence, sentiment_sequence)

    model = SneakerLSTM(input_size=4)
    checkpoint_path = MODEL_DIR / "lstm_best.pt"
    model_source = "checkpoint"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    else:
        model_source = "untrained"

    model.eval()
    with torch.no_grad():
        prediction_norm = model(x).squeeze().numpy()

    if prediction_norm.ndim == 0:
        prediction_norm = np.array([float(prediction_norm)])

    if price_std > 1e-8:
        prediction_price = prediction_norm * price_std + price_mean
    else:
        prediction_price = np.full_like(prediction_norm, price_mean)

    current_price = float(np.asarray(price_sequence, dtype=np.float32)[-1])
    predicted_final = float(prediction_price[-1])

    return {
        "model": "LSTM",
        "model_source": model_source,
        "forecast_days": FORECAST_HORIZON,
        "predicted_prices": [round(float(p), 2) for p in prediction_price.tolist()],
        "predicted_norm": [round(float(p), 4) for p in prediction_norm.tolist()],
        "predicted_final_price": round(predicted_final, 2),
        "predicted_change_pct": round((predicted_final - current_price) / current_price * 100, 2),
        "trend_direction": "UP" if predicted_final > current_price else "DOWN",
        "explanation": "Forecast generated from 4-feature input [price_norm, sentiment_mean, return_1d, volatility_7d].",
    }


def get_gru_forecast(price_sequence: list[float], sentiment_sequence: list[float]) -> dict:
    """
    Run GRU model on recent price+sentiment data, return 30-day forecast.
    """
    from models.gru_model import SneakerGRU

    x, price_mean, price_std = _build_feature_tensor(price_sequence, sentiment_sequence)

    model = SneakerGRU(input_size=4)
    checkpoint_path = MODEL_DIR / "gru_best.pt"
    model_source = "checkpoint"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    else:
        model_source = "untrained"

    model.eval()
    with torch.no_grad():
        prediction_norm = model(x).squeeze().numpy()

    if prediction_norm.ndim == 0:
        prediction_norm = np.array([float(prediction_norm)])

    if price_std > 1e-8:
        prediction_price = prediction_norm * price_std + price_mean
    else:
        prediction_price = np.full_like(prediction_norm, price_mean)

    current_price = float(np.asarray(price_sequence, dtype=np.float32)[-1])
    predicted_final = float(prediction_price[-1])

    return {
        "model": "GRU",
        "model_source": model_source,
        "forecast_days": FORECAST_HORIZON,
        "predicted_prices": [round(float(p), 2) for p in prediction_price.tolist()],
        "predicted_norm": [round(float(p), 4) for p in prediction_norm.tolist()],
        "predicted_final_price": round(predicted_final, 2),
        "predicted_change_pct": round((predicted_final - current_price) / current_price * 100, 2),
        "trend_direction": "UP" if predicted_final > current_price else "DOWN",
        "explanation": "Forecast generated from 4-feature input [price_norm, sentiment_mean, return_1d, volatility_7d].",
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
