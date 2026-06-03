"""
Naive Random Walk Baseline

Implements the random walk (naive) forecast as a benchmark:
the best prediction for tomorrow's price is today's price.
"""
import numpy as np
import pandas as pd


def random_walk_forecast(prices: np.ndarray, horizon: int) -> np.ndarray:
    """
    Generate naive random walk predictions.
    Prediction = last known price repeated for the entire horizon.

    Args:
        prices: Historical price array
        horizon: Number of days to forecast

    Returns:
        Array of shape (num_windows, horizon) with naive predictions
    """
    prices = np.asarray(prices, dtype=np.float64)
    num_windows = len(prices) - horizon
    if num_windows <= 0:
        return np.array([prices[-1]] * horizon).reshape(1, horizon)

    forecasts = np.zeros((num_windows, horizon))
    for i in range(num_windows):
        forecasts[i, :] = prices[i]

    return forecasts


def random_walk_mse(prices: np.ndarray, horizon: int) -> float:
    """
    Compute MSE of the random walk forecast against actuals.

    Args:
        prices: Full price array (used for both forecast origins and actuals)
        horizon: Forecast horizon

    Returns:
        Mean Squared Error of the naive forecast
    """
    prices = np.asarray(prices, dtype=np.float64)
    num_windows = len(prices) - horizon
    if num_windows <= 0:
        return 0.0

    total_se = 0.0
    count = 0
    for i in range(num_windows):
        actual = prices[i + 1 : i + 1 + horizon]
        predicted = np.full(len(actual), prices[i])
        total_se += np.sum((actual - predicted) ** 2)
        count += len(actual)

    return total_se / count if count > 0 else 0.0


if __name__ == "__main__":
    np.random.seed(42)
    fake_prices = 200 + np.cumsum(np.random.randn(100) * 5)

    horizon = 30
    forecasts = random_walk_forecast(fake_prices, horizon)
    mse = random_walk_mse(fake_prices, horizon)

    print(f"Prices: {len(fake_prices)} days")
    print(f"Forecast windows: {forecasts.shape[0]}")
    print(f"Random Walk MSE (horizon={horizon}): {mse:.4f}")
