"""
Algorithmic Trading Simulator

Simulates portfolio performance using agent decisions.
Includes transaction fees (StockX ~9.5%), position tracking, and performance metrics.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from config import BUY_THRESHOLD, SELL_THRESHOLD, INITIAL_CAPITAL


class TradingSimulator:
    """
    Mock trading engine that tracks portfolio value over time.
    Supports transaction fees to simulate real StockX marketplace costs.
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL, transaction_fee: float = 0.095):
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.cash = initial_capital
        self.holdings = 0
        self.trade_log: list[dict] = []
        self.portfolio_history: list[dict] = []

    def decide(self, confidence: float) -> str:
        if confidence >= BUY_THRESHOLD and self.cash > 0:
            return "BUY"
        elif confidence <= SELL_THRESHOLD and self.holdings > 0:
            return "SELL"
        return "HOLD"

    def execute(self, signal: str, price: float, date: str) -> None:
        if signal == "BUY":
            effective_price = price * (1 + self.transaction_fee)
            units = int(self.cash // effective_price)
            if units > 0:
                cost = units * effective_price
                self.cash -= cost
                self.holdings += units
                self.trade_log.append({
                    "date": date, "action": "BUY", "price": price,
                    "effective_price": effective_price, "units": units, "fee_paid": units * price * self.transaction_fee,
                })
        elif signal == "SELL":
            if self.holdings > 0:
                effective_price = price * (1 - self.transaction_fee)
                revenue = self.holdings * effective_price
                self.cash += revenue
                self.trade_log.append({
                    "date": date, "action": "SELL", "price": price,
                    "effective_price": effective_price, "units": self.holdings, "fee_paid": self.holdings * price * self.transaction_fee,
                })
                self.holdings = 0

        self.portfolio_history.append({
            "date": date, "cash": self.cash, "holdings": self.holdings,
            "portfolio_value": self.portfolio_value(price),
        })

    def portfolio_value(self, current_price: float) -> float:
        return self.cash + self.holdings * current_price

    def get_trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log)

    def get_portfolio_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.portfolio_history)

    def total_return(self, current_price: float) -> float:
        final_value = self.portfolio_value(current_price)
        return (final_value - self.initial_capital) / self.initial_capital * 100

    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        if len(self.portfolio_history) < 2:
            return 0.0
        values = [h["portfolio_value"] for h in self.portfolio_history]
        returns = np.diff(values) / values[:-1]
        if np.std(returns) == 0:
            return 0.0
        daily_rf = risk_free_rate / 252
        return float((np.mean(returns) - daily_rf) / np.std(returns) * np.sqrt(252))

    def max_drawdown(self) -> float:
        if not self.portfolio_history:
            return 0.0
        values = [h["portfolio_value"] for h in self.portfolio_history]
        peak = values[0]
        max_dd = 0.0
        for v in values:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)
        return float(max_dd)

    def reset(self) -> None:
        self.cash = self.initial_capital
        self.holdings = 0
        self.trade_log = []
        self.portfolio_history = []


class BuyAndHoldBaseline:
    """
    Buy-and-hold benchmark: buy on day 1, hold until the end.
    Used to compare against active trading strategies.
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL, transaction_fee: float = 0.095):
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee

    def run(self, prices: list[float], dates: list[str]) -> pd.DataFrame:
        buy_price = prices[0] * (1 + self.transaction_fee)
        units = int(self.initial_capital // buy_price)
        cash_remaining = self.initial_capital - units * buy_price

        history = []
        for i, (price, date) in enumerate(zip(prices, dates)):
            history.append({
                "date": date,
                "portfolio_value": cash_remaining + units * price,
            })

        return pd.DataFrame(history)

    def total_return(self, start_price: float, end_price: float) -> float:
        buy_price = start_price * (1 + self.transaction_fee)
        units = int(self.initial_capital // buy_price)
        cash_remaining = self.initial_capital - units * buy_price
        final_value = cash_remaining + units * end_price
        return (final_value - self.initial_capital) / self.initial_capital * 100
