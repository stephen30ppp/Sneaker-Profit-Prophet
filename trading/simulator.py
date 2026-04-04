"""
Algorithmic Trading Simulator (Bonus)

Simulates portfolio performance using model confidence
to trigger Buy/Hold/Sell decisions.
"""
import pandas as pd
import numpy as np

from config import BUY_THRESHOLD, SELL_THRESHOLD, INITIAL_CAPITAL


class TradingSimulator:
    """
    Mock trading engine that tracks portfolio value over time.
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings = 0  # number of units held
        self.trade_log: list[dict] = []

    def decide(self, confidence: float) -> str:
        """
        Generate Buy/Hold/Sell signal based on model confidence.

        Args:
            confidence: Model confidence score in [0, 1]

        Returns:
            "BUY", "SELL", or "HOLD"
        """
        if confidence >= BUY_THRESHOLD and self.cash > 0:
            return "BUY"
        elif confidence <= SELL_THRESHOLD and self.holdings > 0:
            return "SELL"
        return "HOLD"

    def execute(self, signal: str, price: float, date: str) -> None:
        """Execute a trade signal at the given price."""
        if signal == "BUY":
            units = int(self.cash // price)
            if units > 0:
                self.cash -= units * price
                self.holdings += units
                self.trade_log.append({"date": date, "action": "BUY", "price": price, "units": units})
        elif signal == "SELL":
            self.cash += self.holdings * price
            self.trade_log.append({"date": date, "action": "SELL", "price": price, "units": self.holdings})
            self.holdings = 0

    def portfolio_value(self, current_price: float) -> float:
        """Current total portfolio value."""
        return self.cash + self.holdings * current_price

    def get_trade_log(self) -> pd.DataFrame:
        """Return trade history as DataFrame."""
        return pd.DataFrame(self.trade_log)

    def total_return(self, current_price: float) -> float:
        """Calculate total return percentage."""
        final_value = self.portfolio_value(current_price)
        return (final_value - self.initial_capital) / self.initial_capital * 100
