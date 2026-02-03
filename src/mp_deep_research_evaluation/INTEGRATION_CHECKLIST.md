# Integration Checklist

## Quick Start (5 minutes)

### 1. Copy evaluation folder to your project
```bash
cp -r evaluation/ /path/to/mp_deep_research/src/mp_deep_research/
```

### 2. Create agent wrapper (adapt to your agent)
```python
# In your project: agent_wrapper.py

from mp_deep_research.research_agent_scope import create_research_agent

class AgentWrapper:
    def __init__(self):
        self.agent = create_research_agent()
    
    def run(self, query: str) -> dict:
        """Run agent and return standardized output."""
        result = self.agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        
        return {
            "final_answer": result["messages"][-1].content,
            "success": True,
            "tool_calls": self._extract_tool_calls(result),
            "materials_found": result.get("materials", []),
            "duration_ms": 0,  # Add timing if needed
            "token_count": 0,  # Add token counting if needed
        }
    
    def _extract_tool_calls(self, result):
        # Extract tool calls from your agent's output format
        calls = []
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls:
                    calls.append({
                        "name": tc["name"],
                        "arguments": tc["args"]
                    })
        return calls
```

### 3. Run evaluation
```python
from mp_deep_research.evaluation import (
    MaterialsResearchEvaluator,
    EvaluationConfig,
    DiscoveryBenchmark,
)
from agent_wrapper import AgentWrapper

# Initialize
agent = AgentWrapper()

# Quick test
benchmark = DiscoveryBenchmark()
for task in benchmark.get_tasks(difficulty="easy", limit=3):
    output = agent.run(task.user_query)
    print(f"{task.task_id}: {'✓' if output['success'] else '✗'}")

# Full evaluation
config = EvaluationConfig(
    experiment_name="my_evaluation",
    run_discovery_benchmark=True,
    use_llm_judge=False,  # Set True for paper
)
evaluator = MaterialsResearchEvaluator(config)
report = evaluator.run_full_evaluation(agent_runner=agent.run)

print(report.to_markdown())
```

---

## Key Integration Points

### Agent Output Format
Your agent wrapper must return this dict:
```python
{
    "final_answer": str,      # Agent's response text
    "success": bool,          # Task completed?
    "tool_calls": [           # Tools called
        {"name": "screen_materials", "arguments": {...}},
    ],
    "materials_found": [],    # MP IDs found
    "duration_ms": float,     # Execution time
    "token_count": int,       # Tokens used
}
```

### Success Checker
Define what "success" means:
```python
def check_success(output: dict, task: dict) -> bool:
    if not output["success"]:
        return False
    
    # Check if expected materials found
    expected = set(task.get("expected_materials", []))
    found = set(output.get("materials_found", []))
    return bool(expected & found) or not expected
```

### Tool Schemas (for validation)
Define your tools for Glass Box evaluation:
```python
from mp_deep_research.evaluation import ToolSchema

tools = {
    "screen_materials": ToolSchema(
        name="screen_materials",
        description="Screen materials from MP database...",
        parameters={
            "elements": {"type": "array"},
            "e_above_hull_max": {"type": "number"},
        },
        required_params=["elements"],
    ),
}
```

---

## Evaluation Modes

### 1. Quick Development Test
```python
# 3 easy tasks, no LLM judge
results = run_quick_evaluation(agent.run)
```

### 2. Full Paper Evaluation
```python
config = EvaluationConfig(
    experiment_name="paper_v1",
    run_discovery_benchmark=True,
    run_innovation_benchmark=True,
    use_llm_judge=True,
    generate_latex=True,
)
report = evaluator.run_full_evaluation(agent.run)
print(report.to_latex())  # For paper
```

### 3. Tool Validation (Booking.com Glass Box)
```python
from mp_deep_research.evaluation import validate_tool_call, check_tool_reliability

# Validate tool calls
for tc in output["tool_calls"]:
    result = validate_tool_call(tc, tool_schemas)
    print(f"{tc['name']}: {'valid' if result.is_valid else result.error_type}")

# Check tool quality
for name, schema in tools.items():
    result = check_tool_reliability(name, schema.description)
    print(f"{name}: {result.overall_score:.2f}")
```

### 4. Consistency (τ-bench)
```python
from mp_deep_research.evaluation import evaluate_consistency

metrics = evaluate_consistency(
    queries=["Find Li cathodes", "Design thermoelectrics"],
    agent_runner=agent.run,
    success_checker=lambda o: o["success"],
    k=5,  # 5 paraphrased trials
)
print(f"pass@5: {metrics.avg_pass_at_k:.3f}")
print(f"pass^5: {metrics.avg_pass_hat_k:.3f}")  # Consistency!
```

### 5. Baseline Comparison
```python
from mp_deep_research.evaluation import evaluate_system, compare_systems, BaselineType

agent_eval = evaluate_system("Agent", BaselineType.AGENT, agent.run, tasks, checker)
baseline_eval = evaluate_system("Zero-shot", BaselineType.ZERO_SHOT_LLM, baseline.run, tasks, checker)

comparison = compare_systems(agent_eval, [baseline_eval])
print(comparison.recommendation)  # DEPLOY_AGENT or USE_BASELINE
```

---

## Command Line Usage
```bash
# Quick test
python integration_example.py --mode quick

# Full evaluation
python integration_example.py --mode full --output-dir ./results

# Tool validation
python integration_example.py --mode tools

# Consistency test
python integration_example.py --mode consistency
```
