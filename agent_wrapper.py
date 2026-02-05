from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from langchain_openai import ChatOpenAI

from mp_deep_research.tools import ALL_TOOLS


class AgentWrapper:
    def __init__(self, api_key: str | None = None):
        try:
            from langchain.agents import create_agent
        except ImportError as exc:
            raise ImportError("langchain.agents.create_agent is unavailable in this environment.") from exc
        if api_key:
            import os
            os.environ.setdefault("OPENAI_API_KEY", api_key)

        model_name = "gpt-4o-mini"
        llm = ChatOpenAI(model=model_name, temperature=0.0)

        self.agent = create_agent(
            llm,
            tools=ALL_TOOLS,
            system_prompt="You are a materials science assistant. Use the tools to answer user queries.",
        )

    def run(self, query: str) -> Dict[str, Any]:
        """Run agent and return standardized output for evaluation."""
        result = self.agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })

        tool_calls = self._extract_tool_calls(result)
        materials_found = self._extract_materials(result)

        return {
            "final_answer": result["messages"][-1].content,
            "success": True,
            "tool_calls": tool_calls,
            "materials_found": materials_found,
            "duration_ms": 0,
            "token_count": 0,
        }

    def _extract_tool_calls(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls:
                    calls.append({
                        "name": tc["name"],
                        "arguments": tc["args"],
                    })
        return calls

    def _extract_materials(self, result: Dict[str, Any]) -> List[str]:
        """Extract MP IDs from tool outputs and messages."""
        materials: Set[str] = set()
        pattern = re.compile(r"mp-\d+")

        for msg in result.get("messages", []):
            content = getattr(msg, "content", None)
            if content:
                materials.update(pattern.findall(content))

        return sorted(materials)
