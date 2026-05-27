"""
Multi-Agent Trading Simulation (Bonus Component)

Multiple agents with different risk profiles compete as market forces.
Each agent independently analyzes the same market data and produces
Buy/Hold/Sell decisions. A market aggregator simulates how competing
forces would impact execution.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass

from agent.react_agent import TradingAgent, AgentDecision
from trading.simulator import TradingSimulator


@dataclass
class SimulationStep:
    date: str
    price: float
    decisions: dict[str, AgentDecision]
    market_consensus: str
    portfolio_values: dict[str, float]


class MultiAgentSimulation:
    """
    Runs multiple trading agents with different strategies on the same data.
    Compares their performance as competing market forces.
    """

    def __init__(self, api_key: str, profiles: list[str] = None):
        if profiles is None:
            profiles = ["aggressive", "conservative", "balanced"]

        self.agents: dict[str, TradingAgent] = {}
        self.simulators: dict[str, TradingSimulator] = {}

        for profile in profiles:
            self.agents[profile] = TradingAgent(api_key=api_key, profile=profile)
            self.simulators[profile] = TradingSimulator()

        self.history: list[SimulationStep] = []

    def _aggregate_decisions(self, decisions: dict[str, AgentDecision]) -> str:
        """
        Weighted voting: each agent's vote is weighted by confidence.
        Simulates how competing market forces reach equilibrium.
        """
        buy_weight = 0.0
        sell_weight = 0.0
        hold_weight = 0.0

        for name, decision in decisions.items():
            if decision.action == "BUY":
                buy_weight += decision.confidence
            elif decision.action == "SELL":
                sell_weight += decision.confidence
            else:
                hold_weight += decision.confidence

        if buy_weight > sell_weight and buy_weight > hold_weight:
            return "BUY"
        elif sell_weight > buy_weight and sell_weight > hold_weight:
            return "SELL"
        return "HOLD"

    def run_step(self, market_state: dict, date: str) -> SimulationStep:
        """
        Run one simulation step: all agents analyze and decide independently.
        """
        decisions = {}
        for name, agent in self.agents.items():
            try:
                decision = agent.decide(market_state)
            except Exception as e:
                decision = AgentDecision(
                    action="HOLD", confidence=0.0,
                    reasoning=f"Error: {str(e)}", tool_calls=[], full_trace="",
                )
            decisions[name] = decision

            simulator = self.simulators[name]
            price = market_state["prices"][-1]
            simulator.execute(decision.action, price, date)

        consensus = self._aggregate_decisions(decisions)
        portfolio_values = {
            name: sim.portfolio_value(market_state["prices"][-1])
            for name, sim in self.simulators.items()
        }

        step = SimulationStep(
            date=date,
            price=market_state["prices"][-1],
            decisions=decisions,
            market_consensus=consensus,
            portfolio_values=portfolio_values,
        )
        self.history.append(step)
        return step

    def run_backtest(
        self,
        prices: list[float],
        sentiments: list[float],
        texts_by_day: list[list[str]],
        dates: list[str],
        window: int = 30,
        decision_interval: int = 7,
    ) -> pd.DataFrame:
        """
        Run full backtest over historical data.
        """
        results = []

        for i in range(window, len(prices)):
            if (i - window) % decision_interval != 0:
                price = prices[i]
                for name, sim in self.simulators.items():
                    sim.portfolio_history.append({
                        "date": dates[i], "cash": sim.cash,
                        "holdings": sim.holdings, "portfolio_value": sim.portfolio_value(price),
                    })
                continue

            market_state = {
                "prices": prices[max(0, i - window): i],
                "sentiments": sentiments[max(0, i - window): i],
                "texts": texts_by_day[i] if i < len(texts_by_day) else [],
                "shoe_name": "Backtest Asset",
            }

            step = self.run_step(market_state, dates[i])
            results.append({
                "date": dates[i],
                "price": prices[i],
                "consensus": step.market_consensus,
                **{f"{name}_value": val for name, val in step.portfolio_values.items()},
                **{f"{name}_action": step.decisions[name].action for name in self.agents},
                **{f"{name}_confidence": step.decisions[name].confidence for name in self.agents},
            })

        return pd.DataFrame(results)

    def get_performance_summary(self, final_price: float) -> pd.DataFrame:
        rows = []
        for name, sim in self.simulators.items():
            trade_log = sim.get_trade_log()
            rows.append({
                "agent": name,
                "final_value": round(sim.portfolio_value(final_price), 2),
                "total_return_pct": round(sim.total_return(final_price), 2),
                "sharpe_ratio": round(sim.sharpe_ratio(), 3),
                "max_drawdown": round(sim.max_drawdown(), 4),
                "num_trades": len(trade_log),
                "num_buys": len(trade_log[trade_log["action"] == "BUY"]) if not trade_log.empty else 0,
                "num_sells": len(trade_log[trade_log["action"] == "SELL"]) if not trade_log.empty else 0,
            })
        return pd.DataFrame(rows)
