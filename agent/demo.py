"""
Demo: Run the complete agent system with synthetic data.
Works without API key (uses rule-based fallback) or with OpenRouter free LLM.

Usage:
    python -m agent.demo
    python -m agent.demo --with-llm   # requires OPENROUTER_API_KEY in .env
"""
import sys
import numpy as np
from config import OPENROUTER_API_KEY
from agent.react_agent import TradingAgent
from agent.backtest import AgentBacktester, generate_synthetic_data, run_full_backtest


def demo_single_decision():
    """Demonstrate a single agent making one trading decision."""
    print("=" * 60)
    print("  DEMO 1: Single Agent Decision")
    print("=" * 60)

    data = generate_synthetic_data(days=60)
    market_state = {
        "prices": data["prices"],
        "sentiments": data["sentiments"],
        "texts": data["texts_by_day"][-1],
        "shoe_name": "Jordan 1 Retro High OG Chicago",
    }

    print(f"\n  Shoe: {market_state['shoe_name']}")
    print(f"  Current Price: ${data['prices'][-1]:.2f}")
    print(f"  Price 30d ago: ${data['prices'][-30]:.2f}")
    change = (data['prices'][-1] - data['prices'][-30]) / data['prices'][-30] * 100
    print(f"  30d Change: {change:+.1f}%")

    if OPENROUTER_API_KEY:
        print(f"\n  [Using LLM agent via OpenRouter]")
        agent = TradingAgent(api_key=OPENROUTER_API_KEY, profile="balanced")
        decision = agent.decide(market_state)
        print(f"\n  DECISION: {decision.action}")
        print(f"  CONFIDENCE: {decision.confidence:.2f}")
        print(f"  REASONING: {decision.reasoning}")
        print(f"  Tools called: {len(decision.tool_calls)}")
        for tc in decision.tool_calls:
            print(f"    - {tc['tool']}")
    else:
        print(f"\n  [Using rule-based fallback — set OPENROUTER_API_KEY for LLM agent]")
        bt = AgentBacktester(api_key="", profile="balanced")
        decision = bt._fallback_decision(market_state)
        print(f"\n  DECISION: {decision.action}")
        print(f"  CONFIDENCE: {decision.confidence:.2f}")
        print(f"  REASONING: {decision.reasoning}")


def demo_backtest():
    """Run full backtest with 3 agent profiles."""
    print("\n" + "=" * 60)
    print("  DEMO 2: Multi-Profile Backtest (180 days)")
    print("=" * 60)

    result = run_full_backtest(api_key="", days=180)
    data = result["data"]

    print(f"\n  Period: {data['dates'][0]} → {data['dates'][-1]}")
    print(f"  Start Price: ${data['prices'][0]:.2f}")
    print(f"  End Price:   ${data['prices'][-1]:.2f}")
    market_return = (data['prices'][-1] - data['prices'][0]) / data['prices'][0] * 100
    print(f"  Market Move:  {market_return:+.1f}%")

    print(f"\n  {'Agent':<15} {'Return%':<10} {'Sharpe':<10} {'MaxDD':<10} {'Trades':<8}")
    print(f"  {'-' * 53}")

    for profile in ["aggressive", "conservative", "balanced"]:
        m = result["results"][profile]["metrics"]
        print(f"  {profile:<15} {m['agent_total_return_pct']:<+10.2f} {m['agent_sharpe_ratio']:<10.3f} {m['agent_max_drawdown']:<10.2%} {m['num_trades']:<8}")

    baseline_ret = result["results"]["balanced"]["metrics"]["baseline_total_return_pct"]
    print(f"  {'buy-and-hold':<15} {baseline_ret:<+10.2f} {'—':<10} {'—':<10} {'1':<8}")

    winner = max(result["results"].items(), key=lambda x: x[1]["metrics"]["agent_total_return_pct"])
    print(f"\n  Best performer: {winner[0]} ({winner[1]['metrics']['agent_total_return_pct']:+.2f}%)")

    if winner[1]["metrics"]["agent_total_return_pct"] > baseline_ret:
        print(f"  Agent BEATS buy-and-hold by {winner[1]['metrics']['agent_total_return_pct'] - baseline_ret:.2f}%")
    else:
        print(f"  Buy-and-hold wins by {baseline_ret - winner[1]['metrics']['agent_total_return_pct']:.2f}%")


def demo_decisions_trace():
    """Show the decisions timeline."""
    print("\n" + "=" * 60)
    print("  DEMO 3: Agent Decision Timeline")
    print("=" * 60)

    bt = AgentBacktester(api_key="", profile="balanced", decision_interval=14)
    data = generate_synthetic_data(days=120)
    result = bt.run(
        prices=data["prices"],
        sentiments=data["sentiments"],
        texts_by_day=data["texts_by_day"],
        dates=data["dates"],
    )

    decisions_df = result["decisions"]
    if not decisions_df.empty:
        print(f"\n  {'Date':<12} {'Price':<10} {'Action':<6} {'Conf':<6} {'Reasoning'}")
        print(f"  {'-' * 70}")
        for _, row in decisions_df.iterrows():
            reasoning_short = row['reasoning'][:40] + "..." if len(row['reasoning']) > 40 else row['reasoning']
            print(f"  {row['date']:<12} ${row['price']:<9.2f} {row['action']:<6} {row['confidence']:<6.2f} {reasoning_short}")

    print(f"\n  Final Portfolio: ${result['metrics']['agent_final_value']:.2f}")
    print(f"  Agent Return: {result['metrics']['agent_total_return_pct']:+.2f}%")
    print(f"  Baseline Return: {result['metrics']['baseline_total_return_pct']:+.2f}%")


if __name__ == "__main__":
    use_llm = "--with-llm" in sys.argv

    if use_llm and not OPENROUTER_API_KEY:
        print("ERROR: --with-llm requires OPENROUTER_API_KEY in .env")
        print("Get a free key at: https://openrouter.ai/")
        sys.exit(1)

    demo_single_decision()
    demo_backtest()
    demo_decisions_trace()

    print("\n" + "=" * 60)
    print("  All demos complete.")
    print("=" * 60)
