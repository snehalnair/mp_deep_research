from __future__ import annotations

from typing import Any, Dict, List

from mp_deep_research.research_agent_scope import create_research_agent


class AgentWrapper:
    def __init__(self, api_key: str | None = None):
        self.agent = create_research_agent(api_key=api_key)

    def run(self, query: str) -> Dict[str, Any]:
        """Run agent and return standardized output for evaluation."""
        result = self.agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })

        return {
            "final_answer": result["messages"][-1].content,
            "success": True,
            "tool_calls": self._extract_tool_calls(result),
            "materials_found": result.get("materials", []),
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
