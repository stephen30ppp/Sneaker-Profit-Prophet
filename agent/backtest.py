"""
Agent Backtesting Module

Runs the trading agent over historical data and compares its
performance against a buy-and-hold baseline.

Supports both single-agent and multi-agent evaluation.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import INITIAL_CAPITAL, OPENROUTER_API_KEY
from agent.react_agent import TradingAgent, AgentDecision
from agent.multi_agent import MultiAgentSimulation
from trading.simulator import TradingSimulator, BuyAndHoldBaseline


class AgentBacktester:
    """
    Backtest a single agent over a price/sentiment time series.
    Compares agent decisions against buy-and-hold.
    """

    def __init__(self, api_key: str = "", profile: str = "balanced", decision_interval: int = 7):
        """
        Args:
            api_key: OpenRouter API key. If empty, uses rule-based fallback.
            profile: Agent risk profile (aggressive/conservative/balanced)
            decision_interval: Days between agent decisions (to limit API calls)
        """
        self.api_key = api_key or OPENROUTER_API_KEY
        self.profile = profile
        self.decision_interval = decision_interval
        self.use_llm = bool(self.api_key)

    def _fallback_decision(self, market_state: dict) -> AgentDecision:
        """
        Rule-based fallback when no API key is available.
        Uses the same tools the LLM would, but with deterministic logic.
        """
        from agent.tools import get_price_history, get_sentiment_score, get_lstm_forecast

        price_info = get_price_history(market_state["prices"])
        sentiment_info = get_sentiment_score(market_state.get("texts", []))

        score = 0.0
        if price_info["trend"] == "UP":
            score += 0.3
        elif price_info["trend"] == "DOWN":
            score -= 0.3

        if price_info["momentum"] == "BULLISH":
            score += 0.2
        else:
            score -= 0.2

        sent = sentiment_info["average_sentiment"]
        score += sent * 0.3

        try:
            lstm_info = get_lstm_forecast(market_state["prices"], market_state["sentiments"])
            if lstm_info["predicted_change_pct"] > 5:
                score += 0.2
            elif lstm_info["predicted_change_pct"] < -5:
                score -= 0.2
        except Exception:
            pass

        confidence = (score + 1) / 2  # normalize to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        if confidence > 0.6:
            action = "BUY"
        elif confidence < 0.4:
            action = "SELL"
        else:
            action = "HOLD"

        return AgentDecision(
            action=action,
            confidence=confidence,
            reasoning=f"Rule-based: trend={price_info['trend']}, momentum={price_info['momentum']}, sentiment={sent:.2f}",
            tool_calls=[],
            full_trace="Fallback rule-based agent (no API key)",
        )

    def run(
        self,
        prices: list[float],
        sentiments: list[float],
        texts_by_day: list[list[str]],
        dates: list[str],
        window: int = 30,
    ) -> dict:
        """
        Run backtest over historical data.

        Returns dict with:
            - agent_history: DataFrame of agent portfolio over time
            - baseline_history: DataFrame of buy-and-hold portfolio
            - decisions: list of agent decisions with dates
            - metrics: performance comparison
        """
        simulator = TradingSimulator(initial_capital=INITIAL_CAPITAL)
        baseline = BuyAndHoldBaseline(initial_capital=INITIAL_CAPITAL)

        agent = None
        if self.use_llm:
            agent = TradingAgent(api_key=self.api_key, profile=self.profile)

        decisions_log = []

        for i in range(window, len(prices)):
            price = prices[i]
            date = dates[i]

            if (i - window) % self.decision_interval == 0:
                market_state = {
                    "prices": prices[max(0, i - window): i],
                    "sentiments": sentiments[max(0, i - window): i],
                    "texts": texts_by_day[i] if i < len(texts_by_day) else [],
                    "shoe_name": "Backtest Asset",
                }

                if agent and self.use_llm:
                    try:
                        decision = agent.decide(market_state)
                    except Exception:
                        decision = self._fallback_decision(market_state)
                else:
                    decision = self._fallback_decision(market_state)

                simulator.execute(decision.action, price, date)
                decisions_log.append({
                    "date": date,
                    "price": price,
                    "action": decision.action,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning,
                })
            else:
                simulator.portfolio_history.append({
                    "date": date,
                    "cash": simulator.cash,
                    "holdings": simulator.holdings,
                    "portfolio_value": simulator.portfolio_value(price),
                })

        baseline_df = baseline.run(prices[window:], dates[window:])
        agent_df = simulator.get_portfolio_history()

        final_price = prices[-1]
        metrics = {
            "agent_total_return_pct": simulator.total_return(final_price),
            "baseline_total_return_pct": baseline.total_return(prices[window], final_price),
            "agent_sharpe_ratio": simulator.sharpe_ratio(),
            "agent_max_drawdown": simulator.max_drawdown(),
            "num_trades": len(simulator.trade_log),
            "num_decisions": len(decisions_log),
            "agent_final_value": simulator.portfolio_value(final_price),
            "baseline_final_value": float(baseline_df["portfolio_value"].iloc[-1]) if not baseline_df.empty else INITIAL_CAPITAL,
        }

        return {
            "agent_history": agent_df,
            "baseline_history": baseline_df,
            "decisions": pd.DataFrame(decisions_log),
            "metrics": metrics,
            "trade_log": simulator.get_trade_log(),
        }


def generate_synthetic_data(days: int = 180, seed: int = 42) -> dict:
    """
    Generate realistic synthetic sneaker market data for backtesting demo.
    Simulates price with trend + mean reversion + noise, and correlated sentiment.
    """
    np.random.seed(seed)

    base_price = 250.0
    prices = [base_price]
    for _ in range(days - 1):
        mean_reversion = 0.01 * (base_price - prices[-1])
        trend = 0.15 / 252
        noise = np.random.normal(0, 3.0)
        new_price = prices[-1] * (1 + trend + mean_reversion / prices[-1]) + noise
        prices.append(max(new_price, 50.0))

    sentiments = []
    for i in range(days):
        price_momentum = (prices[i] - prices[max(0, i - 7)]) / prices[max(0, i - 7)] if i > 0 else 0
        sent = np.clip(price_momentum * 5 + np.random.normal(0, 0.2), -1, 1)
        sentiments.append(float(sent))

    sample_texts = [
        "These are fire, definitely copping",
        "Price going up, resale is profitable",
        "Overrated, not worth retail",
        "StockX prices climbing, bullish",
        "Market cooling down, might sell",
        "Everyone wants these, impossible to get",
        "Dead stock at this point",
        "Resale value holding strong",
        "Just sold mine for a nice profit",
        "Price dropping, glad I sold early",
    ]
    texts_by_day = []
    for i in range(days):
        n_texts = np.random.randint(2, 6)
        if sentiments[i] > 0.1:
            pool = sample_texts[:6]
        elif sentiments[i] < -0.1:
            pool = sample_texts[4:]
        else:
            pool = sample_texts
        texts_by_day.append(list(np.random.choice(pool, size=n_texts)))

    start_date = datetime(2024, 1, 1)
    dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

    return {
        "prices": prices,
        "sentiments": sentiments,
        "texts_by_day": texts_by_day,
        "dates": dates,
    }


def run_full_backtest(api_key: str = "", days: int = 180) -> dict:
    """
    Convenience function: generate data + run all 3 agent profiles + baseline.
    Returns combined results for comparison.
    """
    data = generate_synthetic_data(days=days)
    results = {}

    for profile in ["aggressive", "conservative", "balanced"]:
        bt = AgentBacktester(api_key=api_key, profile=profile, decision_interval=7)
        results[profile] = bt.run(
            prices=data["prices"],
            sentiments=data["sentiments"],
            texts_by_day=data["texts_by_day"],
            dates=data["dates"],
        )

    return {"data": data, "results": results}


if __name__ == "__main__":
    print("=" * 60)
    print("AGENT BACKTESTING")
    print("=" * 60)

    result = run_full_backtest(api_key="", days=180)

    print(f"\nBacktest Period: {result['data']['dates'][0]} to {result['data']['dates'][-1]}")
    print(f"Start Price: ${result['data']['prices'][0]:.2f}")
    print(f"End Price: ${result['data']['prices'][-1]:.2f}")
    print(f"\n{'Agent':<15} {'Return%':<10} {'Sharpe':<10} {'MaxDD':<10} {'Trades':<8}")
    print("-" * 53)

    for profile in ["aggressive", "conservative", "balanced"]:
        m = result["results"][profile]["metrics"]
        print(f"{profile:<15} {m['agent_total_return_pct']:<10.2f} {m['agent_sharpe_ratio']:<10.2f} {m['agent_max_drawdown']:<10.2%} {m['num_trades']:<8}")

    baseline_ret = result["results"]["balanced"]["metrics"]["baseline_total_return_pct"]
    print(f"{'buy-and-hold':<15} {baseline_ret:<10.2f} {'N/A':<10} {'N/A':<10} {'1':<8}")
