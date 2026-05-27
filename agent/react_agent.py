"""
ReAct Trading Agent — Reason + Act loop powered by OpenRouter free LLMs.

The agent:
1. Observes market state
2. Thinks about which tool to call
3. Calls tools (LSTM, GRU, Sentiment, Price History)
4. Reasons over results
5. Outputs a Buy/Hold/Sell decision with confidence
"""
from __future__ import annotations

import json
import re
import requests
from dataclasses import dataclass
from typing import Optional, Tuple

from config import INITIAL_CAPITAL
from agent.tools import TOOL_REGISTRY
from agent.prompts import AGENT_PROFILES, BASE_SYSTEM_PROMPT


@dataclass
class AgentDecision:
    action: str  # "BUY", "HOLD", "SELL"
    confidence: float
    reasoning: str
    tool_calls: list[dict]
    full_trace: str  # complete reasoning trace for debugging


class TradingAgent:
    """
    Autonomous trading agent using ReAct pattern.
    Calls OpenRouter free LLMs to reason over tool outputs.
    """

    def __init__(
        self,
        api_key: str,
        profile: str = "balanced",
        model: str = "meta-llama/llama-3.1-8b-instruct:free",
        max_steps: int = 5,
    ):
        self.api_key = api_key
        self.model = model
        self.max_steps = max_steps
        self.profile = AGENT_PROFILES.get(profile, AGENT_PROFILES["balanced"])
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def _build_system_prompt(self) -> str:
        tool_desc = "\n".join(
            f"- {t['name']}: {t['description']} (params: {', '.join(t['parameters'])})"
            for t in TOOL_REGISTRY.values()
        )
        return self.profile["prompt"].format(tool_descriptions=tool_desc)

    def _call_llm(self, messages: list[dict]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _parse_action(self, text: str) -> Optional[Tuple[str, dict]]:
        action_match = re.search(r"ACTION:\s*(\w+)", text)
        input_match = re.search(r"ACTION_INPUT:\s*(.+?)(?:\n|$)", text, re.DOTALL)
        if action_match and input_match:
            tool_name = action_match.group(1)
            try:
                args = json.loads(input_match.group(1).strip())
            except json.JSONDecodeError:
                args = {}
            return tool_name, args
        return None

    def _parse_decision(self, text: str) -> Optional[AgentDecision]:
        decision_match = re.search(r"DECISION:\s*(BUY|HOLD|SELL)", text)
        confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", text)
        if decision_match:
            return AgentDecision(
                action=decision_match.group(1),
                confidence=float(confidence_match.group(1)) if confidence_match else 0.5,
                reasoning=reasoning_match.group(1).strip() if reasoning_match else "",
                tool_calls=[],
                full_trace=text,
            )
        return None

    def _execute_tool(self, tool_name: str, args: dict, market_state: dict) -> str:
        if tool_name not in TOOL_REGISTRY:
            return f"Error: Unknown tool '{tool_name}'. Available: {list(TOOL_REGISTRY.keys())}"

        tool = TOOL_REGISTRY[tool_name]
        fn = tool["function"]

        try:
            if tool_name == "get_lstm_forecast":
                result = fn(market_state["prices"], market_state["sentiments"])
            elif tool_name == "get_gru_forecast":
                result = fn(market_state["prices"], market_state["sentiments"])
            elif tool_name == "get_sentiment_score":
                result = fn(market_state.get("texts", []))
            elif tool_name == "get_price_history":
                result = fn(market_state["prices"])
            else:
                result = {"error": "Tool not mapped"}
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def decide(self, market_state: dict) -> AgentDecision:
        """
        Run the full ReAct loop to produce a trading decision.

        Args:
            market_state: dict with keys:
                - prices: list[float] — recent daily prices (at least 30)
                - sentiments: list[float] — daily sentiment scores
                - texts: list[str] — recent social media texts
                - shoe_name: str
        """
        system_prompt = self._build_system_prompt()
        user_msg = (
            f"Analyze the sneaker '{market_state.get('shoe_name', 'Unknown')}' and decide: Buy, Hold, or Sell.\n"
            f"Current price: ${market_state['prices'][-1]:.2f}\n"
            f"You have access to {len(market_state['prices'])} days of price history and "
            f"{len(market_state.get('texts', []))} recent social media posts.\n"
            f"Make your decision using the available tools."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        tool_calls = []
        full_trace = []

        for step in range(self.max_steps):
            response = self._call_llm(messages)
            full_trace.append(f"--- Step {step + 1} ---\n{response}")

            decision = self._parse_decision(response)
            if decision:
                decision.tool_calls = tool_calls
                decision.full_trace = "\n".join(full_trace)
                return decision

            action = self._parse_action(response)
            if action:
                tool_name, args = action
                tool_result = self._execute_tool(tool_name, args, market_state)
                tool_calls.append({"tool": tool_name, "args": args, "result": tool_result})

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"OBSERVATION:\n{tool_result}\n\nContinue your analysis."})
            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Please either call a tool (ACTION/ACTION_INPUT) or make a final DECISION."})

        return AgentDecision(
            action="HOLD",
            confidence=0.0,
            reasoning="Max steps reached without clear decision",
            tool_calls=tool_calls,
            full_trace="\n".join(full_trace),
        )
